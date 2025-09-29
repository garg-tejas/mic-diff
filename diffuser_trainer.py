from typing import Optional
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'  # Changed from '6' to '0' for single GPU setup
import warnings
# Suppress torchaudio backend warning
warnings.filterwarnings("ignore", message="No audio backend is available.")
# Suppress sklearn metrics warnings during early training
warnings.filterwarnings("ignore", message="y_pred contains classes not in y_true")
warnings.filterwarnings("ignore", message="Precision is ill-defined and being set to 0.0")
warnings.filterwarnings("ignore", message="Recall is ill-defined and being set to 0.0")
# Suppress PyTorch Lightning DataLoader worker warning (we've optimized for RTX 4050)
warnings.filterwarnings("ignore", message=".*does not have many workers.*")
import numpy as np
from pathlib import Path
import copy
from pytorch_lightning.utilities.types import EVAL_DATALOADERS, STEP_OUTPUT
import torch
from torch import nn
import torch.nn.functional as F
from torchvision import datasets, transforms
from tqdm import tqdm

import pytorch_lightning as pl
import yaml
from easydict import EasyDict
import random
from pytorch_lightning import callbacks
from pytorch_lightning.accelerators import accelerator
from pytorch_lightning.core.hooks import CheckpointHooks
from pytorch_lightning.callbacks import ModelCheckpoint,DeviceStatsMonitor,EarlyStopping,LearningRateMonitor
from pytorch_lightning.strategies import DDPStrategy
from pytorch_lightning.loggers import TensorBoardLogger
from argparse import Namespace

from torch.utils.data import DataLoader
import pipeline

from torchvision.utils import save_image
from torchvision.models import vgg16
output_dir = 'logs'
version_name='Baseline'
logger = TensorBoardLogger(name='aptos',save_dir = output_dir )
import matplotlib.pyplot as plt
# import tent
import math
from pretraining.dcg import DCG as AuxCls
from model import *
from utils import *


class CoolSystem(pl.LightningModule):
    
    def __init__(self, hparams):
        super(CoolSystem, self).__init__()

        self.params = hparams
        self.epochs = self.params.training.n_epochs
        self.initlr = self.params.optim.lr

        
        config_path = r'option/diff_DDIM.yaml'
        with open(config_path, 'r') as f:
            params = yaml.safe_load(f)
        config = EasyDict(params)
        self.diff_opt = config

        self.model = ConditionalModel(self.params, guidance=self.params.diffusion.include_guidance)
        self.aux_model = AuxCls(self.params)
        
        # Use dataset-specific auxiliary model if available
        dataset_name = self.params.data.dataset.lower()
        dataset_specific_models = {
            'aptos': 'pretraining/ckpt/aptos_aux_model.pth',
            'isic': 'pretraining/ckpt/isic_aux_model.pth',
            'placental': 'pretraining/ckpt/placental_aux_model.pth'
        }
        
        # Try dataset-specific model first, fallback to placental
        aux_model_path = dataset_specific_models.get(dataset_name, 'pretraining/ckpt/placental_aux_model.pth')
        
        if not os.path.exists(aux_model_path):
            print(f"Warning: dataset-specific model not found: {aux_model_path}")
            # Try fallback models
            for fallback_name, fallback_path in dataset_specific_models.items():
                if os.path.exists(fallback_path):
                    print(f"Using fallback model: {fallback_path}")
                    aux_model_path = fallback_path
                    break
            else:
                print("No auxiliary model found! Run: python get_specific_pretrained_models.py")
                raise FileNotFoundError(f"No auxiliary model available")
        else:
            print(f"Using dataset-specific model: {aux_model_path}")
        
        self.init_weight(ckpt_path=aux_model_path)
        self.aux_model.eval()

        self.save_hyperparameters()
        
        self.gts = []
        self.preds = []
        self.test_gts = []
        self.test_preds = []

        self.DiffSampler = pipeline.SR3Sampler(
            model=self.model,
            scheduler = pipeline.create_SR3scheduler(self.diff_opt['scheduler'], 'train'),
        )
        self.DiffSampler.scheduler.set_timesteps(self.diff_opt['scheduler']['num_test_timesteps'])
        self.DiffSampler.scheduler.diff_chns = self.params.data.num_classes

    def configure_optimizers(self):
        # REQUIRED
        optimizer = get_optimizer(self.params.optim, filter(lambda p: p.requires_grad, self.model.parameters()))
        # optimizer = Lion(filter(lambda p: p.requires_grad, self.model.parameters()), lr=self.initlr,betas=[0.9,0.99],weight_decay=0)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.epochs, eta_min=self.initlr * 0.01)

        return [optimizer], [scheduler]


    def init_weight(self,ckpt_path=None):
        
        if ckpt_path:
            checkpoint = torch.load(ckpt_path,map_location=self.device)[0]
            checkpoint_model = checkpoint
            state_dict = self.aux_model.state_dict()
            # # 1. filter out unnecessary keys
            checkpoint_model = {k: v for k, v in checkpoint_model.items() if k in state_dict.keys()}
            print(checkpoint_model.keys())
            # 2. overwrite entries in the existing state dict
            state_dict.update(checkpoint_model)
            
            self.aux_model.load_state_dict(state_dict) 

    def diffusion_focal_loss(self, prior, targets, noise, noise_gt, gamma=1, alpha=10):
        probs = F.softmax(prior, dim=1)
        probs = (probs * targets).sum(dim=1)
        weights = 1+alpha*(1 - probs) ** gamma
        weights = weights.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
        loss = weights*(noise-noise_gt).square()
        return loss.mean()



    def guided_prob_map(self, y0_g, y0_l, bz, nc, np):
    
        distance_to_diag = torch.tensor([[abs(i-j)  for j in range(np)] for i in range(np)]).to(self.device)

        weight_g = 1 - distance_to_diag / (np-1)
        weight_l = distance_to_diag / (np-1)
        interpolated_value = weight_l.unsqueeze(0).unsqueeze(0) * y0_l.unsqueeze(-1).unsqueeze(-1) + weight_g.unsqueeze(0).unsqueeze(0) * y0_g.unsqueeze(-1).unsqueeze(-1)
        diag_indices = torch.arange(np)
        map = interpolated_value.clone()
        for i in range(bz):
            for j in range(nc):
                map[i,j,diag_indices,diag_indices] = y0_g[i,j].to(map.dtype)
                map[i,j, np-1, 0] = y0_l[i,j].to(map.dtype)
                map[i,j, 0, np-1] = y0_l[i,j].to(map.dtype)
        return map

    def training_step(self, batch, batch_idx):
        self.model.train()
        self.aux_model.eval()
        
        x_batch, y_batch = batch
        y_batch, _ = cast_label_to_one_hot_and_prototype(y_batch, self.params)
        y_batch = y_batch.cuda()
        #bicubic = bicubic.cuda()
        x_batch = x_batch.cuda()
        
        # Memory optimization: clear cache
        torch.cuda.empty_cache()
        
        with torch.no_grad():
            y0_aux, y0_aux_global, y0_aux_local, patches, attns, attn_map = self.aux_model(x_batch)
            # y0_aux_global,y0_aux_local = y0_aux_global.softmax(1),y0_aux_local.softmax(1)
        # loss_aux = self.aux_cost_function(y0_aux,y_batch)
        # loss_aux.backward()
        
        bz, nc, H, W = attn_map.size()
        bz, np = attns.size()
        
        y_map = y_batch.unsqueeze(1).expand(-1,np*np,-1).reshape(bz*np*np,nc)
        noise = torch.randn_like(y_map).to(self.device)
        timesteps = torch.randint(0, self.DiffSampler.scheduler.config.num_train_timesteps, (bz*np*np,), device=self.device).long()

        noisy_y = self.DiffSampler.scheduler.add_noise(y_map, timesteps=timesteps, noise=noise)
        noisy_y = noisy_y.view(bz,np*np,-1).permute(0,2,1).reshape(bz,nc,np,np)
        
        y0_cond = self.guided_prob_map(y0_aux_global,y0_aux_local,bz,nc,np)
        y_fusion = torch.cat([y0_cond, noisy_y],dim=1)

        attns = attns.unsqueeze(-1)
        attns = (attns*attns.transpose(1,2)).unsqueeze(1)
        
        # Use gradient checkpointing for memory efficiency
        if hasattr(self.model, 'gradient_checkpointing') and self.model.gradient_checkpointing:
            noise_pred = torch.utils.checkpoint.checkpoint(self.model, x_batch, y_fusion, timesteps, patches, attns)
        else:
            noise_pred = self.model(x_batch, y_fusion, timesteps, patches, attns)

        noise = noise.view(bz,np*np,-1).permute(0,2,1).reshape(bz,nc,np,np)
        loss = self.diffusion_focal_loss(y0_aux,y_batch,noise_pred,noise)

        # Scale loss by accumulation steps for gradient accumulation
        accumulation_steps = getattr(self.params.training, 'gradient_accumulation_steps', 1)
        loss = loss / accumulation_steps

        self.log("train_loss",loss,prog_bar=True)
        return {"loss":loss}

    # def validation_step_end(self,step_output):
    #     model_state_dict = self.model.state_dict()
    #     torch.save(model_state_dict, os.path.join(self.save_path,'ckp.pth'))
    #     print('checkpoint save!')
    #     ema_model_state_dict = self.ema_model.state_dict()
    #     for key in model_state_dict:
    #         ema_model_state_dict[key] = 0.999*ema_model_state_dict[key] + 0.001*model_state_dict[key]
    #     self.ema_model.load_state_dict(ema_model_state_dict)
    def on_validation_epoch_end(self):
        if len(self.gts) == 0 or len(self.preds) == 0:
            return
        gt = torch.cat(self.gts)
        pred = torch.cat(self.preds)
        ACC, BACC, Prec, Rec, F1, AUC_ovo, kappa = compute_isic_metrics(gt, pred)

        self.log('accuracy',ACC)
        self.log('f1',F1)
        self.log('Precision',Prec)        
        self.log('Recall',Rec)
        self.log('AUC',AUC_ovo)
        self.log('kappa',kappa)   

        self.gts = []
        self.preds = []
        print("Val: Accuracy {0}, F1 score {1}, Precision {2}, Recall {3}, AUROC {4}, Cohen Kappa {5}".format(ACC,F1,Prec,Rec,AUC_ovo,kappa))

    def on_test_epoch_end(self):
        if len(self.test_gts) == 0 or len(self.test_preds) == 0:
            return
        gt = torch.cat(self.test_gts)
        pred = torch.cat(self.test_preds)
        ACC, BACC, Prec, Rec, F1, AUC_ovo, kappa = compute_isic_metrics(gt, pred)

        self.log('test_accuracy',ACC)
        self.log('test_f1',F1)
        self.log('test_precision',Prec)
        self.log('test_recall',Rec)
        self.log('test_auc',AUC_ovo)
        self.log('test_kappa',kappa)

        self.test_gts = []
        self.test_preds = []
        print("Test: Accuracy {0}, F1 score {1}, Precision {2}, Recall {3}, AUROC {4}, Cohen Kappa {5}".format(ACC,F1,Prec,Rec,AUC_ovo,kappa))


    def validation_step(self,batch,batch_idx):
        self.model.eval()
        self.aux_model.eval()


        x_batch, y_batch = batch
        y_batch, _ = cast_label_to_one_hot_and_prototype(y_batch, self.params)
        y_batch = y_batch.cuda()
        x_batch = x_batch.cuda()
        y0_aux, y0_aux_global, y0_aux_local, patches, attns, attn_map = self.aux_model(x_batch)

        bz, nc, H, W = attn_map.size()
        bz, np = attns.size()


        y0_cond = self.guided_prob_map(y0_aux_global,y0_aux_local,bz,nc,np)
        yT = self.guided_prob_map(torch.rand_like(y0_aux_global),torch.rand_like(y0_aux_local),bz,nc,np)
        attns = attns.unsqueeze(-1)
        attns = (attns*attns.transpose(1,2)).unsqueeze(1)
        y_pred = self.DiffSampler.sample_high_res(x_batch,yT,conditions=[y0_cond, patches, attns])
        y_pred = y_pred.reshape(bz, nc, np*np)
        y_pred = y_pred.mean(2)
        self.preds.append(y_pred)
        self.gts.append(y_batch)

    def test_step(self, batch, batch_idx):
        self.model.eval()
        self.aux_model.eval()

        x_batch, y_batch = batch
        y_batch, _ = cast_label_to_one_hot_and_prototype(y_batch, self.params)
        y_batch = y_batch.cuda()
        x_batch = x_batch.cuda()
        y0_aux, y0_aux_global, y0_aux_local, patches, attns, attn_map = self.aux_model(x_batch)

        bz, nc, H, W = attn_map.size()
        bz, np = attns.size()

        y0_cond = self.guided_prob_map(y0_aux_global,y0_aux_local,bz,nc,np)
        yT = self.guided_prob_map(torch.rand_like(y0_aux_global),torch.rand_like(y0_aux_local),bz,nc,np)
        attns = attns.unsqueeze(-1)
        attns = (attns*attns.transpose(1,2)).unsqueeze(1)
        y_pred = self.DiffSampler.sample_high_res(x_batch,yT,conditions=[y0_cond, patches, attns])
        y_pred = y_pred.reshape(bz, nc, np*np)
        y_pred = y_pred.mean(2)
        self.test_preds.append(y_pred)
        self.test_gts.append(y_batch)

    def train_dataloader(self):
        data_object, train_dataset, val_dataset, test_dataset = get_dataset(self.params)
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.params.training.batch_size,
            shuffle=True,
            num_workers=self.params.data.num_workers,
            pin_memory=True,
            persistent_workers=True,
        )
        return train_loader

    def val_dataloader(self):
        data_object, train_dataset, val_dataset, test_dataset = get_dataset(self.params)
        if val_dataset is None:
            dataset_for_val = test_dataset
        else:
            dataset_for_val = val_dataset
        val_loader = DataLoader(
            dataset_for_val,
            batch_size=self.params.testing.batch_size,
            shuffle=False,
            num_workers=self.params.data.num_workers,
            pin_memory=True,
            persistent_workers=True,
        )
        return val_loader

    def test_dataloader(self):
        data_object, train_dataset, val_dataset, test_dataset = get_dataset(self.params)
        test_loader = DataLoader(
            test_dataset,
            batch_size=self.params.testing.batch_size,
            shuffle=False,
            num_workers=self.params.data.num_workers,
            pin_memory=True,
            persistent_workers=True,
        )
        return test_loader


def main(custom_config=None):
    import argparse
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Train DiffMICv2 on medical datasets')
    parser.add_argument('--config', type=str, default='configs/aptos.yml', 
                       help='Path to config file (default: configs/aptos.yml)')
    parser.add_argument('--resume', action='store_true', 
                       help='Resume training from checkpoint')
    parser.add_argument('--checkpoint', type=str, default=None,
                       help='Specific checkpoint path to resume from')
    args = parser.parse_args()
    
    # Use custom config if provided (for programmatic calls)
    if custom_config is not None:
        config = custom_config
        config_path = "custom_config"
    else:
        config_path = args.config
        print(f"Loading config from: {config_path}")
        
        # Check if config file exists
        if not os.path.exists(config_path):
            print(f"Config file not found: {config_path}")
            print("Available configs:")
            config_dir = Path('configs')
            if config_dir.exists():
                for conf_file in config_dir.glob('*.yml'):
                    print(f"   - {conf_file}")
            return
        
        with open(config_path, 'r') as f:
            params = yaml.safe_load(f)
        config = EasyDict(params)
    
    # Setup resume checkpoint
    RESUME = args.resume
    if args.checkpoint:
        resume_checkpoint_path = args.checkpoint
    else:
        # Auto-detect checkpoint based on dataset
        dataset_name = config.data.dataset.lower()
        resume_checkpoint_path = f'logs/{dataset_name}/version_0/checkpoints/last.ckpt'
    
    if not RESUME:
        resume_checkpoint_path = None

    # Set random seeds
    seed = 10
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True
    
    # Set optimal matmul precision for RTX 3050 Tensor Cores
    torch.set_float32_matmul_precision('medium')
    
    # Print dataset info
    print(f"Training DiffMICv2")
    print(f"Dataset: {config.data.dataset}")
    print(f"Classes: {config.data.num_classes}")
    print(f"Epochs: {config.training.n_epochs}")
    print(f"Batch size: {config.training.batch_size}")
    print(f"GPU: RTX 3050 4GB optimized")


    # hparams = Namespace(**args)

    model = CoolSystem(config)

    checkpoint_callback = ModelCheckpoint(
        monitor='f1',
        filename='aptos-epoch{epoch:02d}-accuracy-{accuracy:.4f}-f1-{f1:.4f}',
        auto_insert_metric_name=False,   
        every_n_epochs=1,
        save_top_k=1,
        mode = "max",
        save_last=True
    )
    lr_monitor_callback = LearningRateMonitor(logging_interval='step')
    
    # Get accumulation steps from config
    accumulation_steps = getattr(config.training, 'gradient_accumulation_steps', 1)
    
    trainer = pl.Trainer(
        check_val_every_n_epoch=5,
        max_epochs=config.training.n_epochs,
        accelerator='gpu',
        devices=1,
        precision=16,  # Mixed precision for memory efficiency
        logger=logger,
        strategy="auto",
        enable_progress_bar=True,
        log_every_n_steps=5,
        accumulate_grad_batches=accumulation_steps,  # Gradient accumulation
        gradient_clip_val=1.0,  # Gradient clipping for stability
        callbacks = [checkpoint_callback,lr_monitor_callback]
    ) 

    #train
    trainer.fit(model,ckpt_path=resume_checkpoint_path)
    
    #validate
    # val_path=r'DiffMIC/logs/placental/version_4/checkpoints/placental-epoch924-accuracy-0.9350-f1-0.9327.ckpt'
    # trainer.validate(model,ckpt_path=val_path)
    
if __name__ == '__main__':
	#your code
    main()
