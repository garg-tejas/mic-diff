import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping, LearningRateMonitor
from pytorch_lightning.loggers import TensorBoardLogger
import yaml
from easydict import EasyDict
import numpy as np
import random
from pathlib import Path

from pretraining.dcg import DCG as AuxCls
from utils import get_dataset, cast_label_to_one_hot_and_prototype, compute_metric, get_optimizer

class DCGPretrainingSystem(pl.LightningModule):
    """
    DCG Pre-training system following the paper specification:
    - 100 epochs
    - Cross-entropy loss for both global and local streams
    - Adam optimizer with 2e-4 learning rate
    - Batch size 64 (scaled down for T4 GPU)
    """
    
    def __init__(self, hparams):
        super(DCGPretrainingSystem, self).__init__()
        
        self.params = hparams
        self.epochs = self.params.dcg_pretraining.n_epochs
        self.initlr = self.params.dcg_pretraining.lr
        
        # Initialize DCG model
        self.aux_model = AuxCls(self.params)
        
        # Metrics storage
        self.train_gts = []
        self.train_preds = []
        self.val_gts = []
        self.val_preds = []
        
        self.save_hyperparameters()
    
    def configure_optimizers(self):
        """Configure optimizer as specified in paper"""
        optimizer = get_optimizer(self.params.dcg_pretraining, self.aux_model.parameters())
        
        # No scheduler for DCG pre-training (constant LR as per paper)
        return optimizer
    
    def training_step(self, batch, batch_idx):
        """DCG training step with cross-entropy loss"""
        self.aux_model.train()
        
        x_batch, y_batch = batch
        y_batch, _ = cast_label_to_one_hot_and_prototype(y_batch, self.params)
        y_batch = y_batch.cuda()
        x_batch = x_batch.cuda()
        
        # Forward pass through DCG
        y0_aux, y0_aux_global, y0_aux_local, patches, attns, attn_map = self.aux_model(x_batch)
        
        # Convert one-hot labels to class indices for cross-entropy
        y_true = torch.argmax(y_batch, dim=1)
        
        # Global stream loss
        loss_g = F.cross_entropy(y0_aux_global, y_true)
        
        # Local stream loss  
        loss_l = F.cross_entropy(y0_aux_local, y_true)
        
        # Total DCG loss (L_DCG = L_g + L_l)
        total_loss = loss_g + loss_l
        
        # Scale loss for gradient accumulation
        accumulation_steps = getattr(self.params.dcg_pretraining, 'gradient_accumulation_steps', 1)
        total_loss = total_loss / accumulation_steps
        
        # Store predictions for metrics
        self.train_preds.append(y0_aux)
        self.train_gts.append(y_batch)
        
        self.log("train_loss", total_loss, prog_bar=True)
        self.log("train_loss_g", loss_g)
        self.log("train_loss_l", loss_l)
        
        return {"loss": total_loss}
    
    def validation_step(self, batch, batch_idx):
        """DCG validation step"""
        self.aux_model.eval()
        
        x_batch, y_batch = batch
        y_batch, _ = cast_label_to_one_hot_and_prototype(y_batch, self.params)
        y_batch = y_batch.cuda()
        x_batch = x_batch.cuda()
        
        with torch.no_grad():
            y0_aux, y0_aux_global, y0_aux_local, patches, attns, attn_map = self.aux_model(x_batch)
            
            # Convert one-hot labels to class indices for cross-entropy
            y_true = torch.argmax(y_batch, dim=1)
            
            # Global stream loss
            loss_g = F.cross_entropy(y0_aux_global, y_true)
            
            # Local stream loss
            loss_l = F.cross_entropy(y0_aux_local, y_true)
            
            # Total DCG loss
            total_loss = loss_g + loss_l
        
        # Store predictions for metrics
        self.val_preds.append(y0_aux)
        self.val_gts.append(y_batch)
        
        self.log("val_loss", total_loss, prog_bar=True)
        self.log("val_loss_g", loss_g)
        self.log("val_loss_l", loss_l)
        
        return {"val_loss": total_loss}
    
    def on_train_epoch_end(self):
        """Compute training metrics at end of epoch"""
        if len(self.train_gts) == 0 or len(self.train_preds) == 0:
            return
            
        # Compute metrics
        gt = torch.cat(self.train_gts)
        pred = torch.cat(self.train_preds)
        
        ACC, BACC, Prec, Rec, F1, AUC_ovo, kappa = compute_metric(gt, pred)
        
        self.log('train_accuracy', ACC)
        self.log('train_f1', F1)
        self.log('train_precision', Prec)
        self.log('train_recall', Rec)
        self.log('train_auc', AUC_ovo)
        self.log('train_kappa', kappa)
        
        # Clear for next epoch
        self.train_gts = []
        self.train_preds = []
        
        if self.trainer.is_global_zero:
            print(f"Train Epoch {self.current_epoch}: Acc={ACC:.4f}, F1={F1:.4f}, Prec={Prec:.4f}, Rec={Rec:.4f}")
    
    def on_validation_epoch_end(self):
        """Compute validation metrics at end of epoch"""
        if len(self.val_gts) == 0 or len(self.val_preds) == 0:
            return
            
        # Compute metrics
        gt = torch.cat(self.val_gts)
        pred = torch.cat(self.val_preds)
        
        ACC, BACC, Prec, Rec, F1, AUC_ovo, kappa = compute_metric(gt, pred)
        
        self.log('val_accuracy', ACC)
        self.log('val_f1', F1)
        self.log('val_precision', Prec)
        self.log('val_recall', Rec)
        self.log('val_auc', AUC_ovo)
        self.log('val_kappa', kappa)
        
        # Clear for next epoch
        self.val_gts = []
        self.val_preds = []
        
        if self.trainer.is_global_zero:
            print(f"Val Epoch {self.current_epoch}: Acc={ACC:.4f}, F1={F1:.4f}, Prec={Prec:.4f}, Rec={Rec:.4f}")
    
    def train_dataloader(self):
        """Training data loader"""
        data_object, train_dataset, val_dataset, test_dataset = get_dataset(self.params)
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.params.dcg_pretraining.batch_size,
            shuffle=True,
            drop_last=True,
            num_workers=self.params.data.num_workers,
            pin_memory=True,
            prefetch_factor=2,
            persistent_workers=False,
        )
        return train_loader
    
    def val_dataloader(self):
        """Validation data loader"""
        data_object, train_dataset, val_dataset, test_dataset = get_dataset(self.params)
        if val_dataset is None:
            dataset_for_val = test_dataset
        else:
            dataset_for_val = val_dataset
        val_loader = DataLoader(
            dataset_for_val,
            batch_size=self.params.dcg_pretraining.batch_size,
            shuffle=False,
            drop_last=True,
            num_workers=self.params.data.num_workers,
            pin_memory=True,
            prefetch_factor=2,
            persistent_workers=False,
        )
        return val_loader


def main():
    """Main DCG pre-training function"""
    import argparse
    
    parser = argparse.ArgumentParser(description='DCG Pre-training for DiffMIC-v2')
    parser.add_argument('--config', type=str, default='configs/aptos.yml', 
                       help='Path to config file')
    parser.add_argument('--resume', action='store_true', 
                       help='Resume DCG pre-training from checkpoint')
    parser.add_argument('--checkpoint', type=str, default=None,
                       help='Specific checkpoint path to resume from')
    args = parser.parse_args()
    
    # Load config
    config_path = args.config
    print(f"Loading config from: {config_path}")
    
    if not os.path.exists(config_path):
        print(f"Config file not found: {config_path}")
        return
    
    with open(config_path, 'r') as f:
        params = yaml.safe_load(f)
    config = EasyDict(params)
    
    # Set random seeds
    seed = 10
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True
    
    # Set precision
    torch.set_float32_matmul_precision('medium')
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    
    print(f"Starting DCG Pre-training")
    print(f"Dataset: {config.data.dataset}")
    print(f"Classes: {config.data.num_classes}")
    print(f"DCG Epochs: {config.dcg_pretraining.n_epochs}")
    print(f"DCG Batch size: {config.dcg_pretraining.batch_size}")
    print(f"Gradient accumulation: {getattr(config.dcg_pretraining, 'gradient_accumulation_steps', 1)}")
    print(f"Effective batch size: {config.dcg_pretraining.batch_size * getattr(config.dcg_pretraining, 'gradient_accumulation_steps', 1)}")
    print(f"DCG Learning rate: {config.dcg_pretraining.lr}")
    
    # Create model
    model = DCGPretrainingSystem(config)
    
    # Setup logging
    output_dir = 'logs'
    logger = TensorBoardLogger(name='dcg_pretraining', save_dir=output_dir)
    
    # Setup callbacks
    checkpoint_callback = ModelCheckpoint(
        monitor='val_f1',
        filename='dcg-epoch{epoch:02d}-accuracy-{val_accuracy:.4f}-f1-{val_f1:.4f}',
        auto_insert_metric_name=False,   
        every_n_epochs=10,
        save_top_k=3,
        mode="max",
        save_last=True
    )
    
    lr_monitor_callback = LearningRateMonitor(logging_interval='step')
    
    early_stopping_callback = EarlyStopping(
        monitor='val_f1',
        patience=20,
        mode='max',
        verbose=True
    )
    
    # Setup trainer
    accumulation_steps = getattr(config.dcg_pretraining, 'gradient_accumulation_steps', 1)
    trainer = pl.Trainer(
        check_val_every_n_epoch=10,
        max_epochs=config.dcg_pretraining.n_epochs,
        accelerator='gpu',
        devices=1,  # Single GPU for DCG pre-training
        precision='16-mixed',
        logger=logger,
        enable_progress_bar=True,
        log_every_n_steps=10,
        accumulate_grad_batches=accumulation_steps,
        gradient_clip_val=1.0,
        callbacks=[checkpoint_callback, lr_monitor_callback, early_stopping_callback],
        enable_model_summary=True,
        deterministic=False,
    )
    
    # Resume from checkpoint if specified
    resume_checkpoint_path = None
    if args.resume:
        if args.checkpoint:
            resume_checkpoint_path = args.checkpoint
        else:
            # Look for latest DCG checkpoint
            checkpoint_dir = Path('logs/dcg_pretraining')
            if checkpoint_dir.exists():
                checkpoints = list(checkpoint_dir.glob('**/checkpoints/*.ckpt'))
                if checkpoints:
                    resume_checkpoint_path = str(max(checkpoints, key=os.path.getctime))
                    print(f"Resuming from: {resume_checkpoint_path}")
    
    # Start training
    trainer.fit(model, ckpt_path=resume_checkpoint_path)
    
    # Save final DCG model for diffusion training
    final_checkpoint_path = 'pretraining/ckpt/aptos_aux_model_trained.pth'
    os.makedirs('pretraining/ckpt', exist_ok=True)
    
    # Extract DCG state dict and save
    dcg_state_dict = model.aux_model.state_dict()
    torch.save([dcg_state_dict], final_checkpoint_path)
    
    print(f"\n{'='*70}")
    print("DCG Pre-training Complete!")
    print(f"{'='*70}")
    print(f"Trained DCG model saved to: {final_checkpoint_path}")
    print(f"This checkpoint will be used for diffusion training.")
    print(f"{'='*70}")


if __name__ == '__main__':
    main()
