# LASER: Learning Adaptive Sparse Representations for Image Compression

This project implements two different types of autoencoder architectures:
1. **Vector Quantized Variational Autoencoder (VQVAE)** - A discrete latent variable model that uses vector quantization

2. **Dictionary Learning Variational Autoencoder (DLVAE)** - An autoencoder with dictionary learning bottleneck for sparse representations


The project includes training and evaluation pipelines with configurable hyperparameters through Hydra.

## Features

- 🚀 Two complementary compression approaches:
  - VQ-VAE with EMA codebook updates (referenced from src/models/bottleneck.py, lines 9-68)
  - DL-VAE with adaptive sparse coding (referenced from src/models/bottleneck.py, lines 257-291)
- ⚡ Efficient implementation:
  - Vectorized batch OMP for fast sparse coding
  - Direct gradient updates for dictionary learning
  - GPU-optimized matrix operations
- 📊 Comprehensive evaluation metrics:
  - PSNR & SSIM for reconstruction quality
  - LPIPS for perceptual quality
  - Optional FID score computation
- 🔧 Clean, modular architecture:
  - PyTorch Lightning for organized training
  - Hydra for configuration management
  - Weights & Biases logging

## Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/vae-models.git
cd vae-models

# Create and activate a virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## Project Structure

```
├── configs/                # Hydra configuration files
│   ├── checkpoint/         # Checkpoint configurations
│   ├── data/               # Dataset configurations
│   ├── model/              # Model configurations
│   ├── train/              # Training configurations
│   ├── wandb/              # W&B logging configurations
│   └── config.yaml         # Main configuration
├── src/
│   ├── data/               # Data modules
│   │   ├── cifar10.py      # CIFAR10 data module
│   │   ├── imagenette2.py  # Imagenette2 data module
│   │   └── config.py       # Data configuration
│   ├── models/             # Model implementations
│   │   ├── bottleneck.py   # Bottleneck implementations
│   │   ├── decoder.py      # Decoder architecture
│   │   ├── dlvae.py        # DLVAE implementation
│   │   ├── encoder.py      # Encoder architecture
│   │   └── vqvae.py        # VQVAE implementation
│   └── lpips.py            # LPIPS perceptual loss
└── train.py                # Main training script
```

## Usage

### Training a Model


To train a model, use the `train.py` script with Hydra configuration:

```bash
# Train VQVAE on CIFAR10
python train.py model.type=vqvae data=cifar10

# Train DLVAE on CIFAR10
python train.py model.type=dlvae data=cifar10

# Train on Imagenette2 dataset
python train.py model.type=vqvae data=imagenette2
```

## Configuration

Configuration is managed using Hydra. The configuration files are located in the `configs` directory.

## Model Architecture

### VQ-VAE
- Encoder network with residual blocks
- Vector quantization bottleneck with EMA codebook updates
- Decoder network with skip connections

### DL-VAE
- Similar encoder-decoder architecture
- Dictionary learning bottleneck with:
  - Adaptive sparse coding via batch OMP
  - Direct gradient updates for dictionary learning
  - L1 regularization for sparsity control
- Commitment loss for training stability

## License

MIT


