from typing import Optional
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
os.environ['NCCL_P2P_DISABLE'] = '1'
os.environ['NCCL_IB_DISABLE'] = '1'
os.environ['NCCL_SHM_DISABLE'] = '1'
os.environ['NCCL_DEBUG'] = 'WARN'
os.environ['NCCL_TIMEOUT'] = '1800'
os.environ['TORCH_NCCL_BLOCKING_WAIT'] = '1'
os.environ['TORCH_NCCL_ASYNC_ERROR_HANDLING'] = '1'
os.environ['TORCH_DISTRIBUTED_DEBUG'] = 'OFF'
os.environ['TORCH_SHOW_CPP_STACKTRACES'] = '1'
import warnings
warnings.filterwarnings("ignore", message="No audio backend is available.")
warnings.filterwarnings("ignore", message="y_pred contains classes not in y_true")
warnings.filterwarnings("ignore", message="Precision is ill-defined and being set to 0.0")
warnings.filterwarnings("ignore", message="Recall is ill-defined and being set to 0.0")
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
from pytorch_lightning.loggers import TensorBoardLogger
from argparse import Namespace
from pytorch_lightning.strategies import DDPStrategy
from datetime import timedelta
from torch.utils.data import DataLoader
import pipeline

from torchvision.utils import save_image
from torchvision.models import vgg16
output_dir = 'logs'
version_name='Baseline'
logger = TensorBoardLogger(name='aptos',save_dir = output_dir )
import matplotlib.pyplot as plt
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
        self.model.gradient_checkpointing = False
        self.aux_model = AuxCls(self.params)
        
        dataset_name = self.params.data.dataset.lower()
        
        # Priority order: trained DCG > synthetic DCG > fallback
        trained_dcg_path = 'pretraining/ckpt/aptos_aux_model_trained.pth'
        synthetic_dcg_path = 'pretraining/ckpt/aptos_aux_model.pth'
        
        if os.path.exists(trained_dcg_path):
            aux_model_path = trained_dcg_path
            print(f"Using trained DCG model: {aux_model_path}")
            print("This model was trained for 100 epochs as specified in the paper.")
        elif os.path.exists(synthetic_dcg_path):
            aux_model_path = synthetic_dcg_path
            print(f"Using synthetic DCG model: {aux_model_path}")
            print("Warning: This is a synthetic checkpoint, not trained DCG.")
            print("For better performance, run: python dcg_pretraining.py")
        else:
            print("No DCG model found!")
            print("Please run one of the following:")
            print("1. python dcg_pretraining.py  # Train DCG for 100 epochs (recommended)")
            print("2. python pretrained_models.py  # Create synthetic DCG checkpoint")
            raise FileNotFoundError(f"No DCG model available")
        
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
        self.DiffSampler.scheduler.set_timesteps(self.diff_opt['scheduler']['ddim_steps'])
        self.DiffSampler.scheduler.diff_chns = self.params.data.num_classes

    def configure_optimizers(self):
        optimizer = get_optimizer(self.params.optim, filter(lambda p: p.requires_grad, self.model.parameters()))
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.epochs, eta_min=self.initlr * 0.01)

        return [optimizer], [scheduler]


    def init_weight(self,ckpt_path=None):
        
        if ckpt_path:
            checkpoint = torch.load(ckpt_path,map_location=self.device)[0]
            checkpoint_model = checkpoint
            state_dict = self.aux_model.state_dict()
            checkpoint_model = {k: v for k, v in checkpoint_model.items() if k in state_dict.keys()}
            print(checkpoint_model.keys())
            state_dict.update(checkpoint_model)
            
            self.aux_model.load_state_dict(state_dict) 

    def diffusion_mse_loss(self, noise_pred, noise_gt):
        return F.mse_loss(noise_pred, noise_gt)



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
        # Ensure aux_model is in eval mode and synchronized across GPUs
        self.aux_model.eval()
        for param in self.aux_model.parameters():
            param.requires_grad = False
        
        x_batch, y_batch = batch
        y_batch, _ = cast_label_to_one_hot_and_prototype(y_batch, self.params)
        y_batch = y_batch.cuda()
        x_batch = x_batch.cuda()
        
        if batch_idx % 5 == 0:
            torch.cuda.empty_cache()
            if torch.cuda.is_available():
                torch.cuda.synchronize()
        
        with torch.no_grad():
            y0_aux, y0_aux_global, y0_aux_local, patches, attns, attn_map = self.aux_model(x_batch)
        
        bz, nc, H, W = attn_map.size()
        bz, np = attns.size()
        
        y_map = y_batch.unsqueeze(1).expand(-1,np*np,-1).reshape(bz*np*np,nc)
        noise = torch.randn_like(y_map).to(self.device)
        # Heterologous timestep sampling: each noise point gets independent random timestep
        timesteps = torch.randint(0, self.DiffSampler.scheduler.config.num_train_timesteps, (bz*np*np,), device=self.device).long()

        noisy_y = self.DiffSampler.scheduler.add_noise(y_map, timesteps=timesteps, noise=noise)
        noisy_y = noisy_y.view(bz,np*np,-1).permute(0,2,1).reshape(bz,nc,np,np)
        
        y0_cond = self.guided_prob_map(y0_aux_global,y0_aux_local,bz,nc,np)
        y_fusion = torch.cat([y0_cond, noisy_y],dim=1)

        attns = attns.unsqueeze(-1)
        attns = (attns*attns.transpose(1,2)).unsqueeze(1)
        
        if hasattr(self.model, 'gradient_checkpointing') and self.model.gradient_checkpointing:
            noise_pred = torch.utils.checkpoint.checkpoint(self.model, x_batch, y_fusion, timesteps, patches, attns)
        else:
            noise_pred = self.model(x_batch, y_fusion, timesteps, patches, attns)

        noise = noise.view(bz,np*np,-1).permute(0,2,1).reshape(bz,nc,np,np)
        loss = self.diffusion_mse_loss(noise_pred,noise)

        del noise_pred, noisy_y, y_fusion, y_map, y0_cond
        
        accumulation_steps = getattr(self.params.training, 'gradient_accumulation_steps', 1)
        loss = loss / accumulation_steps

        # Additional cleanup to prevent DDP hangs
        if batch_idx % 10 == 0:
            torch.cuda.empty_cache()
            if torch.cuda.is_available():
                torch.cuda.synchronize()

        self.log("train_loss",loss,prog_bar=True)
        
        # Add barrier for DDP synchronization
        if self.trainer.world_size > 1:
            torch.distributed.barrier()
            
        return {"loss":loss}

    def on_validation_epoch_end(self):
        if len(self.gts) == 0 or len(self.preds) == 0:
            return
        
        # Synchronize across GPUs for distributed training
        if self.trainer.world_size > 1:
            # Gather all predictions and ground truths from all GPUs
            all_gts = self.all_gather(torch.cat(self.gts))
            all_preds = self.all_gather(torch.cat(self.preds))
            
            # Flatten the gathered tensors
            gt = all_gts.view(-1, all_gts.shape[-1])
            pred = all_preds.view(-1, all_preds.shape[-1])
        else:
            gt = torch.cat(self.gts)
            pred = torch.cat(self.preds)
            
        ACC, BACC, Prec, Rec, F1, AUC_ovo, kappa = compute_metric(gt, pred)

        self.log('accuracy',ACC)
        self.log('f1',F1)
        self.log('Precision',Prec)        
        self.log('Recall',Rec)
        self.log('AUC',AUC_ovo)
        self.log('kappa',kappa)   

        self.gts = []
        self.preds = []
        if self.trainer.is_global_zero:  # Only print on rank 0
            print("Val: Accuracy {0}, F1 score {1}, Precision {2}, Recall {3}, AUROC {4}, Cohen Kappa {5}".format(ACC,F1,Prec,Rec,AUC_ovo,kappa))

    def on_test_epoch_end(self):
        if len(self.test_gts) == 0 or len(self.test_preds) == 0:
            return
            
        # Synchronize across GPUs for distributed training
        if self.trainer.world_size > 1:
            # Gather all predictions and ground truths from all GPUs
            all_gts = self.all_gather(torch.cat(self.test_gts))
            all_preds = self.all_gather(torch.cat(self.test_preds))
            
            # Flatten the gathered tensors
            gt = all_gts.view(-1, all_gts.shape[-1])
            pred = all_preds.view(-1, all_preds.shape[-1])
        else:
            gt = torch.cat(self.test_gts)
            pred = torch.cat(self.test_preds)
            
        ACC, BACC, Prec, Rec, F1, AUC_ovo, kappa = compute_metric(gt, pred)

        self.log('test_accuracy',ACC)
        self.log('test_f1',F1)
        self.log('test_precision',Prec)
        self.log('test_recall',Rec)
        self.log('test_auc',AUC_ovo)
        self.log('test_kappa',kappa)

        self.test_gts = []
        self.test_preds = []
        if self.trainer.is_global_zero:  # Only print on rank 0
            print("Test: Accuracy {0}, F1 score {1}, Precision {2}, Recall {3}, AUROC {4}, Cohen Kappa {5}".format(ACC,F1,Prec,Rec,AUC_ovo,kappa))


    def validation_step(self,batch,batch_idx):
        self.model.eval()
        self.aux_model.eval()
        # Ensure aux_model parameters don't require gradients during validation
        for param in self.aux_model.parameters():
            param.requires_grad = False


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
        # Ensure aux_model parameters don't require gradients during testing
        for param in self.aux_model.parameters():
            param.requires_grad = False

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
            drop_last=True,
            num_workers=self.params.data.num_workers,
            pin_memory=True,
            prefetch_factor=2,
            persistent_workers=False,
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
            drop_last=True,
            num_workers=self.params.data.num_workers,
            pin_memory=True,
            prefetch_factor=2,
            persistent_workers=False,
        )
        return val_loader

    def test_dataloader(self):
        data_object, train_dataset, val_dataset, test_dataset = get_dataset(self.params)
        test_loader = DataLoader(
            test_dataset,
            batch_size=self.params.testing.batch_size,
            shuffle=False,
            drop_last=True,
            num_workers=self.params.data.num_workers,
            pin_memory=True,
            prefetch_factor=2,
            persistent_workers=False,
        )
        return test_loader


def main(custom_config=None):
    import argparse
    
    parser = argparse.ArgumentParser(description='Train DiffMICv2 on medical datasets')
    parser.add_argument('--config', type=str, default='configs/aptos.yml', 
                       help='Path to config file (default: configs/aptos.yml)')
    parser.add_argument('--resume', action='store_true', 
                       help='Resume training from checkpoint')
    parser.add_argument('--checkpoint', type=str, default=None,
                       help='Specific checkpoint path to resume from')
    args = parser.parse_args()
    
    if custom_config is not None:
        config = custom_config
        config_path = "custom_config"
    else:
        config_path = args.config
        print(f"Loading config from: {config_path}")
        
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
    
    RESUME = args.resume
    if args.checkpoint:
        resume_checkpoint_path = args.checkpoint
    else:
        dataset_name = config.data.dataset.lower()
        resume_checkpoint_path = f'logs/{dataset_name}/version_0/checkpoints/last.ckpt'
    
    if not RESUME:
        resume_checkpoint_path = None

    seed = 10
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True
    
    torch.set_float32_matmul_precision('medium')
    
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    
    print(f"Training DiffMICv2")
    print(f"Dataset: {config.data.dataset}")
    print(f"Classes: {config.data.num_classes}")
    print(f"Epochs: {config.training.n_epochs}")
    print(f"Batch size: {config.training.batch_size}")
    print(f"Gradient accumulation: {getattr(config.training, 'gradient_accumulation_steps', 1)}")
    print(f"Effective batch size: {config.training.batch_size * getattr(config.training, 'gradient_accumulation_steps', 1)}")


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
    
    accumulation_steps = getattr(config.training, 'gradient_accumulation_steps', 1)
    
    num_devices = 2
    strategy = DDPStrategy(
        find_unused_parameters=False,
        gradient_as_bucket_view=True,
        timeout=timedelta(minutes=30),
        static_graph=True,
        ddp_comm_hook=None,
        model_averaging_period=1000,
        process_group_backend='nccl'
    ) if num_devices > 1 else "auto"
    
    trainer = pl.Trainer(
        check_val_every_n_epoch=10,
        max_epochs=config.training.n_epochs,
        accelerator='gpu',
        devices=num_devices,
        precision='16-mixed',
        logger=logger,
        strategy=strategy,
        enable_progress_bar=True,
        log_every_n_steps=10,
        accumulate_grad_batches=accumulation_steps,
        gradient_clip_val=config.optim.grad_clip,
        callbacks = [checkpoint_callback,lr_monitor_callback],
        enable_model_summary=True,
        deterministic=False,
    ) 

    trainer.fit(model,ckpt_path=resume_checkpoint_path)
if __name__ == '__main__':
    main()
    