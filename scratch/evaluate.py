#!/usr/bin/env python
"""
Simplified script to evaluate image metrics from a trained LISTA-VQGAN model for CIFAR-10
"""

import os
import argparse
from pathlib import Path
import numpy as np
import torch
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader

# Import necessary local modules
from lista_gan import LISTA_GAN
from image_metrics import ImageMetrics

def parse_args():
    parser = argparse.ArgumentParser(description='Evaluate image metrics of LISTA-VQGAN model')
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to checkpoint file')
    parser.add_argument('--data_dir', type=str, default='./data', help='Path to CIFAR-10 dataset')
    parser.add_argument('--output_dir', type=str, default='./metrics', help='Output directory for plots')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--num_samples', type=int, default=100, help='Number of samples to evaluate')
    parser.add_argument('--save_images', action='store_true', help='Save comparison images')
    return parser.parse_args()

def main():
    args = parse_args()
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load the model
    print(f"Loading model from {args.checkpoint}...")
    try:
        # Get checkpoint info
        checkpoint = torch.load(args.checkpoint, map_location='cpu')
        
        # Find key hyperparameters in checkpoint
        if isinstance(checkpoint, dict) and 'hyper_parameters' in checkpoint:
            hparams = checkpoint['hyper_parameters']
            dict_size = hparams.get('dict_size', 512)
            latent_dim = hparams.get('latent_dim', 256)
            sparse_dim = hparams.get('sparse_dim', 256)
            lista_steps = hparams.get('num_lista_steps', 5)
        else:
            # Default values if not found
            dict_size = 512
            latent_dim = 256
            sparse_dim = 256
            lista_steps = 5
        
        # Create the model
        model = LISTA_GAN(
            in_channels=3,
            hidden_dims=[64, 128, 256],
            dict_size=dict_size,
            latent_dim=latent_dim,
            sparse_dim=sparse_dim,
            num_lista_steps=lista_steps,
            data_dir=args.data_dir
        )
        
        # Load the weights
        if isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
            model.load_state_dict(checkpoint['state_dict'])
        else:
            model.load_state_dict(checkpoint)
        
        print(f"Model loaded with dict_size={dict_size}, latent_dim={latent_dim}, lista_steps={lista_steps}")
    except Exception as e:
        print(f"Error loading model: {str(e)}")
        return
    
    # Set model to evaluation mode
    model.eval()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    
    # Initialize metrics calculator
    metrics_calculator = ImageMetrics(device)
    
    # Load CIFAR-10 test dataset
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    
    test_dataset = torchvision.datasets.CIFAR10(
        root=args.data_dir, train=False, download=True, transform=transform
    )
    
    # Limit number of samples if requested
    if args.num_samples and args.num_samples < len(test_dataset):
        # Create a subset with random samples
        indices = torch.randperm(len(test_dataset))[:args.num_samples]
        from torch.utils.data import Subset
        test_dataset = Subset(test_dataset, indices)
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=4
    )
    
    print(f"Evaluating on {len(test_dataset)} images...")
    
    # Initialize metrics storage
    all_metrics = {
        'mse': [],
        'psnr': [],
        'ssim': []
    }
    
    # Evaluate model
    with torch.no_grad():
        for batch_idx, (x, y) in enumerate(test_loader):
            x = x.to(device)
            
            # Forward pass
            x_recon, z, _, _ = model(x)
            
            # Calculate metrics
            metrics = metrics_calculator.calculate_metrics(x, x_recon)
            
            # Store metrics
            for name, value in metrics.items():
                all_metrics[name].append(value)
            
            # Save sample images
            if batch_idx == 0 and args.save_images:
                save_comparison_images(x, x_recon, output_dir / 'sample_comparisons.png')
                
            # Print progress
            print(f"Processed batch {batch_idx+1}/{len(test_loader)}")
    
    # Calculate statistics
    stats = {}
    for name, values in all_metrics.items():
        if values:  # Check if we have any values
            stats[name] = {
                'mean': np.mean(values),
                'std': np.std(values),
                'min': np.min(values),
                'max': np.max(values)
            }
    
    # Print results
    print("\n===== Image Quality Metrics =====")
    for name, stat in stats.items():
        print(f"{name.upper()}: mean={stat['mean']:.4f}, std={stat['std']:.4f}, min={stat['min']:.4f}, max={stat['max']:.4f}")
    
    # Save results to file
    result_file = output_dir / 'metrics_results.txt'
    with open(result_file, 'w') as f:
        f.write("===== Image Quality Metrics =====\n")
        for name, stat in stats.items():
            f.write(f"{name.upper()}: mean={stat['mean']:.4f}, std={stat['std']:.4f}, min={stat['min']:.4f}, max={stat['max']:.4f}\n")
    
    # Create histogram plots
    create_metric_plots(all_metrics, output_dir)
    
    print(f"Results saved to {output_dir}")

def save_comparison_images(original, reconstructed, save_path, samples=10):
    """
    Save a grid of original and reconstructed images
    """
    # Convert tensors to numpy arrays
    original = (original.cpu() * 0.5 + 0.5).clamp(0, 1)
    reconstructed = (reconstructed.cpu() * 0.5 + 0.5).clamp(0, 1)
    
    # Limit number of samples
    num_samples = min(samples, original.size(0))
    original = original[:num_samples]
    reconstructed = reconstructed[:num_samples]
    
    # Create figure
    fig, axes = plt.subplots(2, num_samples, figsize=(num_samples * 2, 4))
    
    for i in range(num_samples):
        # Original image
        orig_img = original[i].permute(1, 2, 0).numpy()
        axes[0, i].imshow(orig_img)
        axes[0, i].set_title('Original')
        axes[0, i].axis('off')
        
        # Reconstructed image
        recon_img = reconstructed[i].permute(1, 2, 0).numpy()
        axes[1, i].imshow(recon_img)
        axes[1, i].set_title('Reconstructed')
        axes[1, i].axis('off')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close(fig)

def create_metric_plots(metrics, output_dir):
    """
    Create histogram plots for each metric
    """
    for name, values in metrics.items():
        if not values:  # Skip empty lists
            continue
            
        plt.figure(figsize=(8, 5))
        plt.hist(values, bins=20, alpha=0.7)
        plt.title(f'{name.upper()} Distribution')
        plt.xlabel(name.upper())
        plt.ylabel('Count')
        plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig(output_dir / f'{name}_histogram.png', dpi=150)
        plt.close()

if __name__ == "__main__":
    main()
