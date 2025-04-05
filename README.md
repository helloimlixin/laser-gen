# VQ-GAN with Hydra and PyTorch Lightning

This repository contains an implementation of Vector Quantized Generative Adversarial Network (VQ-GAN) using Hydra for configuration management and PyTorch Lightning for distributed training.

## Project Structure

```
vqgan_project/
├── configs/
│   ├── config.yaml                 # Main configuration
│   ├── dataset/                    # Dataset configurations
│   │   ├── coco.yaml
│   │   ├── lsun.yaml
│   │   ├── div2k.yaml
│   │   ├── celeba_hq.yaml
│   │   ├── ffhq.yaml               # New FFHQ dataset config
│   │   ├── oxford_pets.yaml
│   │   ├── stanford_cars.yaml
│   │   └── dtd.yaml
│   ├── model/                      # Model configurations
│   │   ├── vqgan.yaml              # Standard model
│   │   └── vqgan_large.yaml        # Larger capacity model
│   └── trainer/                    # Trainer configurations
│       ├── default.yaml            # Single GPU training
│       ├── gpu.yaml                # Multi-GPU training
│       └── gpu_16bit.yaml          # Multi-GPU with mixed precision
├── src/
│   ├── __init__.py
│   ├── models/
│   │   ├── __init__.py
│   │   ├── components.py           # Individual model components
│   │   └── vqgan.py                # Lightning module
│   ├── data/
│   │   ├── __init__.py
│   │   └── datamodule.py           # Lightning data module with FFHQ support
│   └── utils/
│       ├── __init__.py
│       └── visualization.py        # Visualization utilities
├── data/
│   ├── coco/                       # COCO dataset
│   ├── lsun/                       # LSUN dataset
│   ├── DIV2K/                      # DIV2K dataset
│   ├── celeba_hq/                  # CelebA-HQ dataset
│   └── ffhq/                       # FFHQ dataset
├── outputs/                        # Training outputs
│   └── [experiment_name]/          # Each experiment in its own directory
│       ├── checkpoints/            # Model checkpoints
│       ├── lightning_logs/         # Training logs
│       ├── samples/                # Generated samples
│       ├── test_results/           # Test results
│       └── codebook_viz/           # Codebook visualizations
├── main.py                         # Entry point
├── download_datasets.py            # Dataset download utility with FFHQ support
├── run_experiment.py               # Experiment runner with FFHQ option
├── train_high_quality.sh           # High-quality training script
├── Dockerfile                      # Docker configuration
├── docker-compose.yml              # Docker Compose configuration
├── requirements.txt                # Python dependencies
└── README.md                       # Project documentation with FFHQ info
```

## Installation

1. Clone the repository

```bash
git clone https://github.com/username/vqgan-hydra-lightning.git
cd vqgan-hydra-lightning
```

2. Create a virtual environment and install dependencies

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

## Usage

### Basic Training

To train the model with default settings:

```bash
python main.py
```

This will train the VQ-GAN on the Oxford Flowers dataset using a single GPU.

### Training with Different Configurations

Hydra makes it easy to override configuration parameters:

```bash
# Train with a different dataset
python main.py dataset=oxford_pets

# Change batch size
python main.py dataset.batch_size=32

# Use multiple GPUs
python main.py trainer=gpu

# Change learning rate
python main.py model.lr=0.0002
```

### Using Pre-defined Training Scripts

The repository includes several bash scripts for common training scenarios:

```bash
# Train on Oxford Flowers dataset with a single GPU
./train_flowers_single_gpu.sh

# Train on Oxford Flowers dataset with multiple GPUs
./train_flowers_multi_gpu.sh

# Train on CelebA dataset with a single GPU
./train_celeba_single_gpu.sh

# Train on CelebA dataset with multiple GPUs
./train_celeba_multi_gpu.sh
```

## Configuration

### Datasets

The following high-quality datasets are supported:

- `coco`: Microsoft COCO dataset (Common Objects in Context) - large-scale object detection, segmentation, and captioning dataset with 330K images
- `lsun`: Large-Scale Scene Understanding dataset - millions of labeled images for scene recognition
- `div2k`: DIV2K dataset - high-quality 2K resolution images specifically designed for super-resolution tasks
- `celeba_hq`: CelebA-HQ dataset - high-quality version of CelebA with 30,000 images at 1024×1024 resolution
- `ffhq`: Flickr-Faces-HQ dataset - 70,000 high-quality face images at 1024×1024 resolution with great variety in terms of age, ethnicity, and background
- `oxford_pets`: Oxford-IIIT Pet dataset (for comparison)
- `stanford_cars`: Stanford Cars dataset (for comparison)
- `dtd`: Describable Textures dataset (for comparison)

#### Dataset Preparation

Some datasets require manual download:

**COCO Dataset:**
1. Download from https://cocodataset.org/#download
2. Extract to `data/coco/`

**LSUN Dataset:**
1. Follow instructions at https://github.com/fyu/lsun
2. Extract to `data/lsun/`

**DIV2K Dataset:**
1. Download from https://data.vision.ee.ethz.ch/cvl/DIV2K/
2. Extract to `data/DIV2K/`

**CelebA-HQ Dataset:**
1. Download as directed in repository https://github.com/tkarras/progressive_growing_of_gans
2. Place processed images in `data/celeba_hq/`

**FFHQ Dataset:**
1. Download from https://github.com/NVlabs/ffhq-dataset
2. Place images in `data/ffhq/`

### Training Settings

The default `trainer` configuration uses a single GPU. For multi-GPU training, use the `gpu` configuration:

```bash
python main.py trainer=gpu
```

### Model Hyperparameters

Common hyperparameters can be easily adjusted:

```bash
# Change codebook size
python main.py model.num_embeddings=1024

# Change commitment cost
python main.py model.commitment_cost=0.5

# Change when to start training the discriminator
python main.py model.disc_start=5000
```

## Results

During training, results are saved in the output directory (specified by Hydra):

- **Checkpoints**: Saved model states
- **Lightning Logs**: TensorBoard logs
- **Samples**: Image reconstructions during training
- **Test Results**: Results from testing the model
- **Codebook Visualization**: Visualization of the learned codebook
- **Generated Samples**: Samples generated from random latent codes

To view TensorBoard logs:

```bash
tensorboard --logdir outputs/YYYY-MM-DD/HH-MM-SS/lightning_logs
```

## Cite

If you use this code in your research, please cite the original VQ-GAN paper:

```
@inproceedings{esser2021taming,
  title={Taming Transformers for High-Resolution Image Synthesis},
  author={Esser, Patrick and Rombach, Robin and Ommer, Bjorn},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  pages={12873--12883},
  year={2021}
}
```

## License

This project is licensed under the MIT License - see the LICENSE file for details.
