#!/usr/bin/env python
"""
Script to visualize dictionary atom usage in the LISTA-GAN model
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
import seaborn as sns

# Import necessary modules (assuming the same structure as in the evaluation script)
from lista_gan import LISTA_GAN


def parse_args():
    parser = argparse.ArgumentParser(description='Analyze dictionary atom usage in LISTA-GAN model')
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to checkpoint file')
    parser.add_argument('--data_dir', type=str, default='./data', help='Path to CIFAR-10 dataset')
    parser.add_argument('--output_dir', type=str, default='./dict_analysis', help='Output directory for plots')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--num_samples', type=int, default=1000, help='Number of samples to evaluate')
    parser.add_argument('--activation_threshold', type=float, default=0.01, 
                       help='Threshold to consider an atom activated')
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
        indices = torch.randperm(len(test_dataset))[:args.num_samples]
        from torch.utils.data import Subset
        test_dataset = Subset(test_dataset, indices)
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=True,  # Shuffle to get a random distribution
        num_workers=4
    )
    
    print(f"Analyzing dictionary usage on {len(test_dataset)} images...")
    
    # Initialize tracking variables
    atom_usage_count = torch.zeros(sparse_dim, device=device)
    atom_activation_magnitude = torch.zeros(sparse_dim, device=device)
    atom_activations_per_class = torch.zeros((10, sparse_dim), device=device)
    
    # Track co-activation patterns (which atoms activate together)
    coactivation_matrix = torch.zeros((sparse_dim, sparse_dim), device=device)
    
    # Keep track of some sample sparse codes for visualization
    sample_sparse_codes = []
    sample_images = []
    sample_classes = []
    max_samples = 10  # Number of sample images to keep
    
    # Process batches
    samples_processed = 0
    with torch.no_grad():
        for batch_idx, (x, y) in enumerate(test_loader):
            x = x.to(device)
            y = y.to(device)
            
            # Encode with LISTA
            h = model.encoder(x)
            z, _ = model.lista_encoder(h)  # z = sparse codes
            
            # Calculate atom usage statistics
            activated = (torch.abs(z) > args.activation_threshold)
            
            # For each batch element
            for i in range(x.size(0)):
                class_idx = y[i].item()
                
                # Reshape activated from [batch, channels, height, width] to [batch, channels, height*width]
                z_flat = z[i].reshape(z.size(1), -1)  # [sparse_dim, H*W]
                activated_flat = activated[i].reshape(activated.size(1), -1)  # [sparse_dim, H*W]
                
                # Count activations per atom
                atom_activations = activated_flat.sum(dim=1)  # [sparse_dim]
                atom_usage_count += (atom_activations > 0).float()
                
                # Record per-class activations
                atom_activations_per_class[class_idx] += (atom_activations > 0).float()
                
                # Calculate mean activation magnitude when atoms are active
                for j in range(sparse_dim):
                    if activated_flat[j].sum() > 0:
                        atom_activation_magnitude[j] += z_flat[j, activated_flat[j]].abs().mean()
                
                # Update co-activation matrix (which atoms tend to activate together)
                # Outer product of binary activations
                active_atoms = (atom_activations > 0).float()
                coactivation_matrix += torch.outer(active_atoms, active_atoms)
                
                # Store sample sparse codes
                if len(sample_sparse_codes) < max_samples:
                    # Store the first few samples for visualization
                    sample_sparse_codes.append(z[i].cpu())
                    sample_images.append(x[i].cpu())
                    sample_classes.append(class_idx)
            
            samples_processed += x.size(0)
            print(f"Processed batch {batch_idx+1}/{len(test_loader)} - {samples_processed}/{len(test_dataset)} samples")
            
            if samples_processed >= args.num_samples:
                break
    
    # Normalize the statistics
    atom_activation_magnitude = atom_activation_magnitude / atom_usage_count.clamp(min=1)
    atom_usage_count = atom_usage_count / samples_processed
    for i in range(10):  # 10 classes in CIFAR-10
        class_count = (torch.tensor(sample_classes) == i).sum().item()
        if class_count > 0:
            atom_activations_per_class[i] = atom_activations_per_class[i] / class_count
    
    # Normalize co-activation matrix by diagonal to get conditional probabilities
    for i in range(sparse_dim):
        if coactivation_matrix[i, i] > 0:
            coactivation_matrix[i] = coactivation_matrix[i] / coactivation_matrix[i, i]
    
    # 1. Overall Atom Usage Distribution
    plt.figure(figsize=(14, 6))
    plt.bar(range(sparse_dim), atom_usage_count.cpu().numpy())
    plt.xlabel('Dictionary Atom Index')
    plt.ylabel('Fraction of Images Using Atom')
    plt.title('Dictionary Atom Usage Distribution')
    plt.axhline(y=atom_usage_count.mean().cpu().numpy(), color='r', linestyle='--', 
                label=f'Mean Usage: {atom_usage_count.mean().cpu().numpy():.4f}')
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / 'atom_usage_distribution.png', dpi=150)
    plt.close()
    
    # 2. Atom Activation Magnitude
    plt.figure(figsize=(14, 6))
    plt.bar(range(sparse_dim), atom_activation_magnitude.cpu().numpy())
    plt.xlabel('Dictionary Atom Index')
    plt.ylabel('Mean Activation Magnitude When Used')
    plt.title('Dictionary Atom Activation Magnitude')
    plt.axhline(y=atom_activation_magnitude.mean().cpu().numpy(), color='r', linestyle='--',
                label=f'Mean Magnitude: {atom_activation_magnitude.mean().cpu().numpy():.4f}')
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / 'atom_activation_magnitude.png', dpi=150)
    plt.close()
    
    # 3. Class-specific Atom Usage
    plt.figure(figsize=(16, 12))
    
    # Define CIFAR-10 class names
    class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 
                  'dog', 'frog', 'horse', 'ship', 'truck']
    
    for i in range(10):
        plt.subplot(5, 2, i+1)
        plt.bar(range(sparse_dim), atom_activations_per_class[i].cpu().numpy())
        plt.xlabel('Dictionary Atom Index')
        plt.ylabel('Usage Fraction')
        plt.title(f'Atom Usage for Class: {class_names[i]}')
        plt.grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'class_specific_atom_usage.png', dpi=150)
    plt.close()
    
    # 4. Co-activation Heatmap (which atoms activate together)
    plt.figure(figsize=(12, 10))
    sns.heatmap(coactivation_matrix.cpu().numpy(), cmap='viridis')
    plt.xlabel('Dictionary Atom Index')
    plt.ylabel('Dictionary Atom Index')
    plt.title('Atom Co-activation Patterns (P(atom j active | atom i active))')
    plt.tight_layout()
    plt.savefig(output_dir / 'atom_coactivation_heatmap.png', dpi=150)
    plt.close()
    
    # 5. Visualize Sample Sparse Codes
    if sample_sparse_codes:
        plt.figure(figsize=(16, 10))
        
        for i, (code, image, class_idx) in enumerate(zip(sample_sparse_codes, sample_images, sample_classes)):
            # Visualize original image
            plt.subplot(2, max_samples, i+1)
            plt.imshow((image * 0.5 + 0.5).permute(1, 2, 0).clamp(0, 1).numpy())
            plt.title(f'Class: {class_names[class_idx]}')
            plt.axis('off')
            
            # Visualize sparse code magnitude pattern as a spatial heatmap
            plt.subplot(2, max_samples, i+1+max_samples)
            
            # Reshape from [C, H, W] to [H, W, C] and take mean across channels or first channel
            code_spatial = code.abs().mean(dim=0).numpy()
            plt.imshow(code_spatial, cmap='hot')
            plt.title(f'Sparse Code Activation')
            plt.axis('off')
        
        plt.tight_layout()
        plt.savefig(output_dir / 'sample_sparse_codes.png', dpi=150)
        plt.close()
    
    # 6. Visualize Top and Bottom Used Dictionary Atoms
    # Get indices of top and bottom used atoms
    usage_np = atom_usage_count.cpu().numpy()
    top_indices = np.argsort(usage_np)[-20:]  # Top 20 most used atoms
    bottom_indices = np.argsort(usage_np)[:20]  # Bottom 20 least used atoms
    
    # Dictionary atoms are stored in the lista_encoder's dictionary parameter
    dictionary = model.lista_encoder.dictionary.detach().cpu()
    
    # Since dictionary atoms don't have a direct visual representation in latent space,
    # we can visualize them based on their usage statistics
    plt.figure(figsize=(10, 6))
    plt.bar(top_indices.tolist(), usage_np[top_indices])
    plt.xlabel('Dictionary Atom Index')
    plt.ylabel('Usage Fraction')
    plt.title('Top 20 Most Used Dictionary Atoms')
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_dir / 'top_used_atoms.png', dpi=150)
    plt.close()
    
    plt.figure(figsize=(10, 6))
    plt.bar(bottom_indices.tolist(), usage_np[bottom_indices])
    plt.xlabel('Dictionary Atom Index')
    plt.ylabel('Usage Fraction')
    plt.title('Bottom 20 Least Used Dictionary Atoms')
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_dir / 'bottom_used_atoms.png', dpi=150)
    plt.close()
    
    # 7. Save numerical results to file
    result_file = output_dir / 'dictionary_usage_stats.txt'
    with open(result_file, 'w') as f:
        f.write("===== Dictionary Atom Usage Statistics =====\n\n")
        f.write(f"Total samples analyzed: {samples_processed}\n")
        f.write(f"Dictionary size: {dict_size}\n")
        f.write(f"Sparse dimension: {sparse_dim}\n")
        f.write(f"Activation threshold: {args.activation_threshold}\n\n")
        
        f.write("Overall usage statistics:\n")
        f.write(f"Mean atom usage fraction: {atom_usage_count.mean().item():.4f}\n")
        f.write(f"Std dev of atom usage: {atom_usage_count.std().item():.4f}\n")
        f.write(f"Max atom usage fraction: {atom_usage_count.max().item():.4f} (Atom #{atom_usage_count.argmax().item()})\n")
        f.write(f"Min atom usage fraction: {atom_usage_count.min().item():.4f} (Atom #{atom_usage_count.argmin().item()})\n\n")
        
        f.write("Mean activation magnitude statistics:\n")
        f.write(f"Mean activation magnitude: {atom_activation_magnitude.mean().item():.4f}\n")
        f.write(f"Std dev of activation magnitudes: {atom_activation_magnitude.std().item():.4f}\n")
        f.write(f"Max activation magnitude: {atom_activation_magnitude.max().item():.4f} (Atom #{atom_activation_magnitude.argmax().item()})\n")
        f.write(f"Min activation magnitude: {atom_activation_magnitude.min().item():.4f} (Atom #{atom_activation_magnitude.argmin().item()})\n\n")
        
        f.write("Class-specific usage statistics:\n")
        for i in range(10):
            f.write(f"Class {i} ({class_names[i]}):\n")
            f.write(f"  Mean atom usage: {atom_activations_per_class[i].mean().item():.4f}\n")
            f.write(f"  Top 5 atoms: {torch.argsort(atom_activations_per_class[i], descending=True)[:5].tolist()}\n")
            f.write(f"  Bottom 5 atoms: {torch.argsort(atom_activations_per_class[i])[:5].tolist()}\n\n")
        
        f.write("===== End of Statistics =====\n")
    
    print(f"Dictionary atom usage analysis complete. Results saved to {output_dir}")


if __name__ == "__main__":
    main()
