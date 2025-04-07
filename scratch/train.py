import os
import argparse
from pathlib import Path

import torch
import lightning as pl
from lightning.pytorch.loggers import WandbLogger, TensorBoardLogger
from lightning.pytorch.callbacks import ModelCheckpoint, LearningRateMonitor, EarlyStopping
from lightning.pytorch.strategies import DDPStrategy

def parse_args():
    parser = argparse.ArgumentParser(description='Train LISTA-GAN on CIFAR-10')
    parser.add_argument('--data_dir', type=str, default='./data', help='Path to CIFAR-10 dataset')
    parser.add_argument('--output_dir', type=str, default='./output', help='Output directory')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size')
    parser.add_argument('--epochs', type=int, default=10, help='Number of epochs')
    parser.add_argument('--dict_size', type=int, default=512, help='Dictionary size')
    parser.add_argument('--latent_dim', type=int, default=256, help='Latent dimension')
    parser.add_argument('--sparse_dim', type=int, default=512, help='Sparse code dimension')
    parser.add_argument('--lista_steps', type=int, default=5, help='Number of LISTA steps')
    parser.add_argument('--learning_rate', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--dict_lr', type=float, default=1e-3, help='Dictionary learning rate')
    parser.add_argument('--sparsity_weight', type=float, default=0.1, help='Sparsity loss weight')
    parser.add_argument('--adv_weight', type=float, default=0.5, help='Adversarial loss weight')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--gpus', type=int, default=4, help='Number of GPUs to use')
    parser.add_argument('--precision', type=str, default='32', help='Training precision (32 or 16-mixed)')
    parser.add_argument('--logger', type=str, default='wandb', help='Logger type (tensorboard or wandb)')
    parser.add_argument('--wandb_project', type=str, default='lista-gan', help='Wandb project name')
    parser.add_argument('--wandb_entity', type=str, default=None, help='Wandb entity name')
    parser.add_argument('--wandb_name', type=str, default=None, help='Wandb run name')
    parser.add_argument('--offline', action='store_true', help='Run in offline mode (for Wandb)')
    
    return parser.parse_args()

def main():
    # Parse arguments
    args = parse_args()
    
    # Set random seed
    pl.seed_everything(args.seed)
    
    # Make sure latent_dim and sparse_dim are the same
    if args.latent_dim != args.sparse_dim:
        print("Warning: latent_dim and sparse_dim must be equal. Setting both to", args.latent_dim)
        args.sparse_dim = args.latent_dim
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Import dependencies
    from lista_gan import LISTA_GAN
    
    # Check if wandb is installed if using wandb logger
    if args.logger == 'wandb':
        try:
            import wandb
        except ImportError:
            print("Wandb not installed. Installing...")
            import subprocess
            subprocess.check_call(["pip", "install", "wandb"])
            import wandb
            
    # Create model with DeepSpeed mode enabled
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
        data_dir=args.data_dir,
        deepspeed=True  # Enable single optimizer mode
    )
    
    # Set up logging
    if args.logger == 'wandb':
        logger = WandbLogger(
            project=args.wandb_project,
            name=args.wandb_name or f"lista-gan-{args.dict_size}-{args.latent_dim}",
            entity=args.wandb_entity,
            offline=args.offline,
            log_model=True,
            save_dir=str(output_dir),
            config={
                "dict_size": args.dict_size,
                "latent_dim": args.latent_dim,
                "sparse_dim": args.sparse_dim,
                "lista_steps": args.lista_steps,
                "learning_rate": args.learning_rate,
                "dict_lr": args.dict_lr,
                "sparsity_weight": args.sparsity_weight,
                "adv_weight": args.adv_weight,
                "batch_size": args.batch_size,
                "total_batch_size": args.batch_size * args.gpus,
                "precision": args.precision,
                "model_params": sum(p.numel() for p in model.parameters()),
            }
        )
    else:
        logger = TensorBoardLogger(
            save_dir=str(output_dir),
            name='logs'
        )
    
    # Set up callbacks
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
    
    # Create DDP strategy with find_unused_parameters=True
    ddp_strategy = DDPStrategy(
        find_unused_parameters=True,  # Critical for GAN training
        static_graph=False
    )
    
    # Create trainer with DDP strategy
    trainer = pl.Trainer(
        max_epochs=args.epochs,
        devices=args.gpus,
        accelerator="gpu",
        strategy=ddp_strategy,
        precision=args.precision,
        logger=logger,
        callbacks=[checkpoint_callback, lr_monitor, early_stopping],
        log_every_n_steps=10,
        enable_model_summary=True
    )
    
    # Print information
    print(f"Training with DDP on {args.gpus} GPUs")
    print(f"Batch size per GPU: {args.batch_size}")
    print(f"Total batch size: {args.batch_size * args.gpus}")
    print(f"Precision: {args.precision}")
    print(f"Logger: {args.logger}")
    if args.logger == 'wandb':
        print(f"Wandb project: {args.wandb_project}")
        print(f"Wandb run name: {args.wandb_name or f'lista-gan-{args.dict_size}-{args.latent_dim}'}")
    print(f"Starting training...")
    
    # Train model
    torch.set_float32_matmul_precision('medium')
    trainer.fit(model)
    
    # Save final model
    torch.save(model.state_dict(), output_dir / 'final_model.pt')
    
    # Finish wandb run if using wandb
    if args.logger == 'wandb':
        try:
            import wandb
            wandb.finish()
        except:
            pass
    
    print(f"Training completed. Output saved to {output_dir}")

if __name__ == "__main__":
    main()
