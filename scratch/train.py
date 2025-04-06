import os
import argparse
from pathlib import Path

import torch
import lightning as pl
from lightning.pytorch.loggers import TensorBoardLogger
from lightning.pytorch.callbacks import ModelCheckpoint, LearningRateMonitor, EarlyStopping

from lista_gan import LISTA_GAN

def parse_args():
    parser = argparse.ArgumentParser(description='Train LISTA-VQGAN on CIFAR-10')
    parser.add_argument('--data_dir', type=str, default='./data', help='Path to CIFAR-10 dataset')
    parser.add_argument('--output_dir', type=str, default='./output', help='Output directory')
    parser.add_argument('--batch_size', type=int, default=128, help='Batch size')
    parser.add_argument('--epochs', type=int, default=100, help='Number of epochs')
    parser.add_argument('--dict_size', type=int, default=512, help='Dictionary size')
    parser.add_argument('--latent_dim', type=int, default=128, help='Latent dimension')
    parser.add_argument('--sparse_dim', type=int, default=256, help='Sparse code dimension')
    parser.add_argument('--lista_steps', type=int, default=5, help='Number of LISTA steps')
    parser.add_argument('--learning_rate', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--dict_lr', type=float, default=1e-3, help='Dictionary learning rate')
    parser.add_argument('--sparsity_weight', type=float, default=0.1, help='Sparsity loss weight')
    parser.add_argument('--adv_weight', type=float, default=0.5, help='Adversarial loss weight')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--gpus', type=int, default=4, help='Number of GPUs to use')
    parser.add_argument('--precision', type=int, default=16, help='Training precision (16 or 32)')
    
    return parser.parse_args()

def main():
    # Parse arguments
    args = parse_args()
    
    # Set random seed
    pl.seed_everything(args.seed)
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Create model
    model = LISTA_GAN(
        in_channels=3,
        hidden_dims=[64, 128, 256],
        dict_size=args.dict_size,
        latent_dim=args.latent_dim,
        sparse_dim=args.sparse_dim,
        num_lista_steps=args.lista_steps,
        lista_thresh=0.1,
        dict_lr=args.dict_lr,
        learning_rate=args.learning_rate,
        sparsity_weight=args.sparsity_weight,
        adv_weight=args.adv_weight,
        batch_size=args.batch_size,
        data_dir=args.data_dir
    )
    
    # Set up logging and checkpointing
    logger = TensorBoardLogger(
        save_dir=str(output_dir),
        name='logs'
    )
    
    checkpoint_callback = ModelCheckpoint(
        dirpath=str(output_dir / 'checkpoints'),
        filename='{epoch:02d}-{val_loss:.4f}',
        save_top_k=3,
        monitor='val_loss',
        mode='min'
    )
    
    lr_monitor = LearningRateMonitor(logging_interval='step')
    
    early_stopping = EarlyStopping(
        monitor='val_loss',
        patience=10,
        mode='min'
    )
    
    # Create trainer
    trainer = pl.Trainer(
        max_epochs=args.epochs,
        precision=args.precision,
        logger=logger,
        callbacks=[checkpoint_callback, lr_monitor, early_stopping],
        log_every_n_steps=10,
        deterministic=True
    )
    
    # Train model
    trainer.fit(model)
    
    # Save final model
    torch.save(model.state_dict(), output_dir / 'final_model.pt')
    
    print(f"Training completed. Final model saved to {output_dir / 'final_model.pt'}")

if __name__ == "__main__":
    main()
