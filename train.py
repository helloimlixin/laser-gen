import os
import torch
import hydra
import logging
from omegaconf import DictConfig
import lightning as pl
from lightning.pytorch.callbacks import ModelCheckpoint, LearningRateMonitor, EarlyStopping, RichProgressBar
from lightning.pytorch.callbacks.progress.rich_progress import RichProgressBarTheme
from lightning.pytorch.loggers import WandbLogger
from datetime import datetime
import signal
import sys

from src.models.vqvae import VQVAE
from src.models.dlvae import DLVAE
from src.data.cifar10 import CIFAR10DataModule
from src.data.config import DataConfig
from src.data.imagenette2 import Imagenette2DataModule

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Create a unique directory for each run
timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
checkpoint_dir = os.path.join('outputs', 'checkpoints', f'run_{timestamp}')
os.makedirs(checkpoint_dir, exist_ok=True)

# Configure progress bar theme
progress_bar = RichProgressBar(
    theme=RichProgressBarTheme(
        description="green_yellow",
        progress_bar="green1",
        progress_bar_finished="green1",
        progress_bar_pulse="green1",
        batch_progress="green_yellow",
        time="grey82",
        processing_speed="grey82",
        metrics="grey82"
    ),
    leave=True
)

# Define signal handler for graceful shutdown
def signal_handler(sig, frame):
    logger.info("Received termination signal. Cleaning up...")
    if torch.distributed.is_initialized():
        try:
            torch.distributed.destroy_process_group()
            logger.info("Distributed process group destroyed")
        except Exception as e:
            logger.warning(f"Error during cleanup: {e}")
    sys.exit(0)

# Register signal handlers
signal.signal(signal.SIGINT, signal_handler)
signal.signal(signal.SIGTERM, signal_handler)

@hydra.main(config_path="configs", config_name="config", version_base="1.2")
def train(cfg: DictConfig):
    """
    Main training function using Hydra for configuration.
    
    Args:
        cfg: Hydra configuration object containing model and training parameters
    """
    # Only print configuration on rank 0
    if not torch.distributed.is_initialized() or torch.distributed.get_rank() == 0:
        print("\n" + "="*50)
        print("EXPERIMENT CONFIGURATION")
        print("="*50)
        
        print("\nGeneral Settings:")
        print(f"Random Seed: {cfg.seed}")
        print(f"Output Directory: {cfg.output_dir}")
        
        print("\nDataset Configuration:")
        print(f"Dataset: {cfg.data.dataset}")
        print(f"Data Directory: {cfg.data.data_dir}")
        print(f"Batch Size: {cfg.data.batch_size}")
        print(f"Number of Workers: {cfg.data.num_workers}")
        print(f"Image Size: {cfg.data.image_size}")
        print(f"Mean: {cfg.data.mean}")
        print(f"Std: {cfg.data.std}")
        
        print("\nModel Configuration:")
        print(f"Model Type: {cfg.model.type}")
        print(f"Input Channels: {cfg.model.in_channels}")
        print(f"Hidden Dimensions: {cfg.model.num_hiddens}")
        print(f"Embedding Dimensions: {cfg.model.embedding_dim}")
        print(f"Number of Residual Blocks: {cfg.model.num_residual_blocks}")
        print(f"Residual Hidden Dimensions: {cfg.model.num_residual_hiddens}")
        if cfg.model.type == "vqvae":
            print(f"Number of Embeddings: {cfg.model.num_embeddings}")
        elif cfg.model.type == "dlvae":
            print(f"Dictionary Size: {cfg.model.num_embeddings}")
            print(f"Sparsity: {cfg.model.sparsity_level}")
        
        print("\nTraining Configuration:")
        print(f"Learning Rate: {cfg.train.learning_rate}")
        print(f"Beta: {cfg.train.beta}")
        print(f"Max Epochs: {cfg.train.max_epochs}")
        print(f"Accelerator: {cfg.train.accelerator}")
        print(f"Devices: {cfg.train.devices}")
        print(f"Precision: {cfg.train.precision}")
        print(f"Gradient Clip Value: {cfg.train.gradient_clip_val}")
        
        print("\nWandB Configuration:")
        print(f"Project: {cfg.wandb.project}")
        print(f"Run Name: {cfg.wandb.name}")
        print(f"Save Directory: {cfg.wandb.save_dir}")
        
        print("\nCheckpoint Configuration:")
        print(f"Save Directory: {checkpoint_dir}")
        print(f"Filename Template: {cfg.model.type}-{{epoch}}-{{val_loss:.2f}}")
        print(f"Save Top K: {cfg.checkpoint.save_top_k}")
        print("="*50 + "\n")

    # Set random seed for reproducibility
    pl.seed_everything(cfg.seed)
    
    # Print GPU information only on rank 0 or if not distributed
    if not torch.distributed.is_initialized() or torch.distributed.get_rank() == 0:
        print(f"GPU available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"GPU device: {torch.cuda.get_device_name(0)}")
    
    # Initialize data module
    if not torch.distributed.is_initialized() or torch.distributed.get_rank() == 0:
        print(f"Initializing data module for dataset: {cfg.data.dataset}")
    
    if cfg.data.dataset == 'cifar10':
        datamodule = CIFAR10DataModule(DataConfig.from_dict(cfg.data))
    elif cfg.data.dataset == 'imagenette2':
        datamodule = Imagenette2DataModule(DataConfig.from_dict(cfg.data))
    else:
        raise ValueError(f"Unsupported dataset: {cfg.data.dataset}")

    # Print dataset info for debugging only on rank 0
    if not torch.distributed.is_initialized() or torch.distributed.get_rank() == 0:
        print(f"Using dataset: {cfg.data.dataset}")
        print(f"Data module type: {type(datamodule).__name__}")
    
    # Initialize model based on type
    model_params = {
        'in_channels': cfg.model.in_channels,
        'num_hiddens': cfg.model.num_hiddens,
        'num_embeddings': cfg.model.num_embeddings,
        'embedding_dim': cfg.model.embedding_dim,
        'num_residual_blocks': cfg.model.num_residual_blocks,
        'num_residual_hiddens': cfg.model.num_residual_hiddens,
        'commitment_cost': cfg.model.commitment_cost,
        'decay': cfg.model.decay,
        'perceptual_weight': cfg.model.perceptual_weight,
        'learning_rate': cfg.train.learning_rate,
        'beta': cfg.train.beta,
        'compute_fid': cfg.model.compute_fid
    }

    if cfg.model.type == "vqvae":
        model = VQVAE(**model_params)
    elif cfg.model.type == "dlvae":
        model_params['sparsity_level'] = cfg.model.sparsity_level
        model = DLVAE(**model_params)
    else:
        raise ValueError(f"Unknown model type: {cfg.model.type}")

    # Initialize wandb logger only on rank 0
    run_name = f"{cfg.wandb.name}_{cfg.model.type}_{timestamp}"
    wandb_logger = None
    if not torch.distributed.is_initialized() or torch.distributed.get_rank() == 0:
        wandb_logger = WandbLogger(
            project=cfg.wandb.project,
            name=run_name,
            save_dir=cfg.wandb.save_dir
        )

    # Initialize callbacks
    checkpoint_callback = ModelCheckpoint(
        dirpath=os.path.join(checkpoint_dir, cfg.model.type),
        filename=f"{cfg.model.type}-{{epoch}}-{{val_loss:.2f}}",
        save_top_k=cfg.checkpoint.save_top_k,
        monitor='val/loss',
        mode='min',
        save_last=True
    )
    
    callbacks = [
        checkpoint_callback,
        EarlyStopping(
            monitor="val/loss",
            patience=cfg.train.early_stopping_patience,
            mode="min"
        ),
        LearningRateMonitor(logging_interval='step'),
        progress_bar
    ]

    # Configure DeepSpeed strategy
    strategy = "deepspeed_stage_2" if cfg.train.devices > 1 else "auto"
    
    # Initialize trainer for training
    trainer = pl.Trainer(
        max_epochs=cfg.train.max_epochs,
        accelerator=cfg.train.accelerator,
        devices=cfg.train.devices,
        logger=wandb_logger,
        callbacks=callbacks,
        precision=cfg.train.precision,
        gradient_clip_val=cfg.train.gradient_clip_val,
        log_every_n_steps=cfg.train.log_every_n_steps,
        deterministic=True,
        enable_progress_bar=True,
        enable_model_summary=True,
        strategy=strategy
    )

    # Train model
    torch.set_float32_matmul_precision('medium')
    trainer.fit(model, datamodule=datamodule)

    # Test model - only run on rank 0 to avoid duplicate outputs
    if not torch.distributed.is_initialized() or torch.distributed.get_rank() == 0:
        # Create a separate trainer for testing with devices=1
        test_trainer = pl.Trainer(
            accelerator=cfg.train.accelerator,
            devices=1,
            logger=wandb_logger,
            callbacks=[progress_bar],
            precision=cfg.train.precision,
            deterministic=True,
            enable_progress_bar=True,
            strategy="auto"  # Use auto strategy for testing
        )
        test_trainer.test(model, datamodule=datamodule)

    # Clean up distributed environment
    if torch.distributed.is_initialized():
        try:
            logger.info("Cleaning up distributed environment...")
            torch.distributed.destroy_process_group()
            logger.info("Distributed environment cleaned up successfully")
        except Exception as e:
            logger.error(f"Error cleaning up distributed environment: {e}")

    print("Training and testing completed successfully!")

if __name__ == "__main__":
    train()