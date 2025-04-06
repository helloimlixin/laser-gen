import torch
import torch.nn as nn
import torch.nn.functional as F
import lightning as pl
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR
import numpy as np
import torchvision
from torchvision import transforms
from torch.utils.data import DataLoader

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        self.norm1 = nn.GroupNorm(8, out_channels)  # Smaller group size for CIFAR-10
        self.norm2 = nn.GroupNorm(8, out_channels)
        
        if in_channels != out_channels:
            self.shortcut = nn.Conv2d(in_channels, out_channels, 1)
        else:
            self.shortcut = nn.Identity()
            
    def forward(self, x):
        residual = x
        
        x = self.conv1(x)
        x = self.norm1(x)
        x = F.silu(x)
        
        x = self.conv2(x)
        x = self.norm2(x)
        
        return F.silu(x + self.shortcut(residual))

class Encoder(nn.Module):
    def __init__(self, in_channels, hidden_dims=[64, 128, 256], latent_dim=128):
        super().__init__()
        
        # Smaller architecture for CIFAR-10
        modules = [nn.Conv2d(in_channels, hidden_dims[0], 3, stride=1, padding=1)]
        
        # Downsampling
        in_channels = hidden_dims[0]
        for h_dim in hidden_dims[1:]:
            modules.append(ResidualBlock(in_channels, in_channels))
            modules.append(nn.Conv2d(in_channels, h_dim, 4, stride=2, padding=1))
            in_channels = h_dim
            
        modules.append(ResidualBlock(in_channels, in_channels))
        modules.append(nn.Conv2d(in_channels, latent_dim, 3, stride=1, padding=1))
        
        self.encoder = nn.Sequential(*modules)
        
    def forward(self, x):
        return self.encoder(x)

class Decoder(nn.Module):
    def __init__(self, out_channels, hidden_dims=[256, 128, 64], latent_dim=128):
        super().__init__()
        
        hidden_dims = hidden_dims[::-1]  # Reverse for decoder
        
        modules = [nn.Conv2d(latent_dim, hidden_dims[0], 3, stride=1, padding=1)]
        
        # Upsampling
        in_channels = hidden_dims[0]
        for h_dim in hidden_dims[1:]:
            modules.append(ResidualBlock(in_channels, in_channels))
            modules.append(nn.ConvTranspose2d(in_channels, h_dim, 4, stride=2, padding=1))
            in_channels = h_dim
            
        modules.append(ResidualBlock(in_channels, in_channels))
        modules.append(nn.Conv2d(in_channels, out_channels, 3, stride=1, padding=1))
        
        self.decoder = nn.Sequential(*modules)
        
    def forward(self, x):
        return self.decoder(x)

class LISTASparseEncoder(nn.Module):
    def __init__(self, feature_dim, dict_size, sparse_dim, num_steps=5, thresh=0.1, learn_thresh=True):
        """
        LISTA-based sparse coding layer with online dictionary learning
        
        Args:
            feature_dim: Dimensionality of input features
            dict_size: Number of dictionary atoms
            sparse_dim: Dimensionality of sparse codes
            num_steps: Number of unfolded iterations for LISTA
            thresh: Initial threshold value for soft thresholding
            learn_thresh: Whether to learn the threshold parameter
        """
        super().__init__()
        
        # Initialize dictionary
        self.dictionary = nn.Parameter(torch.randn(dict_size, feature_dim))
        
        # LISTA parameters
        # Note: W shape is (sparse_dim, feature_dim)
        self.W = nn.Parameter(torch.randn(sparse_dim, feature_dim))
        # Note: S shape is (sparse_dim, sparse_dim)
        self.S = nn.Parameter(torch.randn(sparse_dim, sparse_dim))
        
        # Threshold parameter (can be learned)
        if learn_thresh:
            self.thresh = nn.Parameter(torch.tensor(thresh))
        else:
            self.register_buffer('thresh', torch.tensor(thresh))
            
        self.num_steps = num_steps
        
        # Save dimensions for clarity
        self.feature_dim = feature_dim
        self.dict_size = dict_size
        self.sparse_dim = sparse_dim
        
        # Initialize parameters
        self._init_params()
        
    def _init_params(self):
        # Normalize dictionary atoms
        dict_norm = F.normalize(self.dictionary, p=2, dim=1)
        self.dictionary.data = dict_norm
        
        # Initialize LISTA parameters based on dictionary
        D = self.dictionary  # Shape: [dict_size, feature_dim]
        
        # Since sparse_dim might be different from dict_size,
        # we need to adjust the initialization
        if self.sparse_dim == self.dict_size:
            # Simple case: sparse_dim equals dict_size
            L = 1.1 * torch.linalg.norm(D @ D.t(), ord=2)
            self.W.data = D / L
            self.S.data = torch.eye(self.sparse_dim) - (D @ D.t()) / L
        else:
            # Complex case: sparse_dim differs from dict_size
            # Option 1: Use a subset of dictionary atoms
            if self.sparse_dim < self.dict_size:
                # Take first sparse_dim atoms
                D_subset = D[:self.sparse_dim]
                L = 1.1 * torch.linalg.norm(D_subset @ D_subset.t(), ord=2)
                self.W.data = D_subset / L
                self.S.data = torch.eye(self.sparse_dim) - (D_subset @ D_subset.t()) / L
            
            # Option 2: Projection for when sparse_dim > dict_size
            else:
                # Create a random projection matrix
                proj = torch.randn(self.sparse_dim, self.dict_size)
                proj = F.normalize(proj, p=2, dim=1)
                
                # Project the dictionary
                D_proj = proj @ D  # Shape: [sparse_dim, feature_dim]
                L = 1.1 * torch.linalg.norm(D_proj @ D_proj.t(), ord=2)
                self.W.data = D_proj / L
                self.S.data = torch.eye(self.sparse_dim) - (D_proj @ D_proj.t()) / L
        
    def forward(self, x):
        """
        x: input features [B, C, H, W]
        """
        # Reshape input for sparse coding
        batch_size, channels, height, width = x.shape
        x_flat = x.permute(0, 2, 3, 1).reshape(-1, channels)
        
        # Normalize dictionary during forward pass
        dict_norm = F.normalize(self.dictionary, p=2, dim=1)
        
        # LISTA unfolded iterations
        z = torch.zeros(x_flat.shape[0], self.sparse_dim, device=x.device)
        
        for _ in range(self.num_steps):
            z = z + self.W @ x_flat.t()  # Residual update
            z = z - self.S @ z  # Gradient step
            z = self._soft_threshold(z, self.thresh)  # Soft thresholding
            
        # Reshape sparse codes back to original spatial dimensions
        z = z.t().reshape(batch_size, height, width, -1).permute(0, 3, 1, 2)
        
        # Return sparse codes and dictionary
        return z, dict_norm
    
    def _soft_threshold(self, x, thresh):
        """Soft thresholding function"""
        return torch.sign(x) * F.relu(torch.abs(x) - thresh)
    
    def update_dictionary(self, x, z, learning_rate=0.01):
        """
        Online dictionary update using sparse codes
        
        Args:
            x: input features [B, C, H, W]
            z: sparse codes [B, K, H, W]
            learning_rate: Dictionary learning rate
        """
        # Reshape for dictionary update
        batch_size, channels, height, width = x.shape
        x_flat = x.permute(0, 2, 3, 1).reshape(-1, channels)
        z_flat = z.permute(0, 2, 3, 1).reshape(-1, z.shape[1])
        
        # Handle the case where sparse_dim != dict_size
        if self.sparse_dim != self.dict_size:
            # Option 1: Only update a subset of dictionary elements
            if self.sparse_dim < self.dict_size:
                subset_indices = range(self.sparse_dim)
                D_subset = self.dictionary[subset_indices]
                
                # Dictionary update for subset
                grad = -torch.mm(z_flat.t(), x_flat) + torch.mm(z_flat.t(), torch.mm(z_flat, D_subset))
                
                # Update dictionary
                with torch.no_grad():
                    self.dictionary.data[subset_indices] -= learning_rate * grad
            
            # Option 2: Project sparse codes for when sparse_dim > dict_size
            else:
                # Create a projection matrix (can be learned or fixed)
                proj = torch.randn(self.dict_size, self.sparse_dim, device=z_flat.device)
                proj = F.normalize(proj, p=2, dim=1)
                
                # Project sparse codes to dictionary size
                z_proj = z_flat @ proj
                
                # Dictionary update with projected codes
                grad = -torch.mm(z_proj.t(), x_flat) + torch.mm(z_proj.t(), torch.mm(z_proj, self.dictionary))
                
                # Update dictionary
                with torch.no_grad():
                    self.dictionary.data -= learning_rate * grad
        else:
            # Standard case: sparse_dim = dict_size
            grad = -torch.mm(z_flat.t(), x_flat) + torch.mm(z_flat.t(), torch.mm(z_flat, self.dictionary))
            
            # Update dictionary
            with torch.no_grad():
                self.dictionary.data -= learning_rate * grad
        
        # Normalize dictionary atoms
        with torch.no_grad():
            dict_norm = F.normalize(self.dictionary.data, p=2, dim=1)
            self.dictionary.data = dict_norm

class Discriminator(nn.Module):
    def __init__(self, in_channels, hidden_dims=[64, 128, 256]):
        super().__init__()
        
        modules = []
        
        # Initial convolution
        modules.append(nn.Conv2d(in_channels, hidden_dims[0], 4, stride=2, padding=1))
        modules.append(nn.LeakyReLU(0.2))
        
        # Downsampling
        in_channels = hidden_dims[0]
        for h_dim in hidden_dims[1:]:
            modules.append(nn.Conv2d(in_channels, h_dim, 4, stride=2, padding=1))
            modules.append(nn.BatchNorm2d(h_dim))
            modules.append(nn.LeakyReLU(0.2))
            in_channels = h_dim
            
        # Output layer
        modules.append(nn.Conv2d(in_channels, 1, 4, stride=1, padding=0))
        
        self.model = nn.Sequential(*modules)
        
    def forward(self, x):
        return self.model(x)

class LISTA_GAN(pl.LightningModule):
    def __init__(
        self,
        in_channels=3,
        hidden_dims=[64, 128, 256],  # Smaller architecture for CIFAR-10
        dict_size=512,               # Smaller dictionary for CIFAR-10
        latent_dim=128,              # Smaller latent dimension for CIFAR-10
        sparse_dim=256,              # Smaller sparse dimension for CIFAR-10
        num_lista_steps=5,           # Fewer LISTA steps for faster training
        lista_thresh=0.1,
        dict_lr=0.001,
        learning_rate=1e-4,
        beta=0.25,
        adv_weight=0.5,
        perceptual_weight=1.0,
        sparsity_weight=0.1,
        batch_size=128,              # Larger batch size for CIFAR-10
        data_dir='./data'            # Directory for CIFAR-10 data
    ):
        super().__init__()
        
        self.encoder = Encoder(in_channels, hidden_dims, latent_dim)
        self.decoder = Decoder(in_channels, hidden_dims, latent_dim)
        
        self.lista_encoder = LISTASparseEncoder(
            feature_dim=latent_dim,
            dict_size=dict_size,
            sparse_dim=sparse_dim,
            num_steps=num_lista_steps,
            thresh=lista_thresh
        )
        
        self.discriminator = Discriminator(in_channels, hidden_dims)
        
        # Hyperparameters
        self.learning_rate = learning_rate
        self.dict_lr = dict_lr
        self.beta = beta
        self.adv_weight = adv_weight
        self.perceptual_weight = perceptual_weight
        self.sparsity_weight = sparsity_weight
        self.batch_size = batch_size
        self.data_dir = data_dir
        
        # Save hyperparameters for logging
        self.save_hyperparameters()
        
    def encode(self, x):
        h = self.encoder(x)
        z, _ = self.lista_encoder(h)
        return z
    
    def decode(self, z):
        return self.decoder(z)
    
    def forward(self, x):
        h = self.encoder(x)
        z, dictionary = self.lista_encoder(h)
        x_recon = self.decoder(z)
        return x_recon, z, dictionary, h
    
    def training_step(self, batch, batch_idx, optimizer_idx):
        x, _ = batch  # CIFAR-10 returns (image, label) pairs
        
        # Encode-decode
        x_recon, z, dictionary, h = self(x)
        
        # Dictionary update (online learning)
        if batch_idx % 10 == 0:  # Update periodically
            self.lista_encoder.update_dictionary(h, z, learning_rate=self.dict_lr)
        
        # Generator loss (optimizer_idx = 0)
        if optimizer_idx == 0:
            # Adversarial loss
            disc_fake = self.discriminator(x_recon)
            adv_loss = -torch.mean(disc_fake)
            
            # Reconstruction loss (L1)
            recon_loss = F.l1_loss(x_recon, x)
            
            # Perceptual loss (simplified for CIFAR-10 - using MSE)
            perc_loss = F.mse_loss(x_recon, x)
            
            # Sparsity loss
            sparsity_loss = torch.mean(torch.abs(z))
            
            # Combined loss
            g_loss = recon_loss + self.adv_weight * adv_loss + self.perceptual_weight * perc_loss + self.sparsity_weight * sparsity_loss
            
            # Log losses
            self.log('train_g_loss', g_loss)
            self.log('train_recon_loss', recon_loss)
            self.log('train_adv_loss', adv_loss)
            self.log('train_perc_loss', perc_loss)
            self.log('train_sparsity', sparsity_loss)
            
            return g_loss
        
        # Discriminator loss (optimizer_idx = 1)
        if optimizer_idx == 1:
            # Real images
            disc_real = self.discriminator(x)
            
            # Fake images
            disc_fake = self.discriminator(x_recon.detach())
            
            # Compute WGAN loss with gradient penalty
            d_loss_real = -torch.mean(disc_real)
            d_loss_fake = torch.mean(disc_fake)
            d_loss = d_loss_real + d_loss_fake
            
            # Log discriminator loss
            self.log('train_d_loss', d_loss)
            
            return d_loss
    
    def configure_optimizers(self):
        # Generator optimizer (encoder, decoder, LISTA)
        opt_g = Adam(list(self.encoder.parameters()) + 
                    list(self.decoder.parameters()) + 
                    list(self.lista_encoder.parameters()),
                    lr=self.learning_rate, betas=(0.5, 0.9))
        
        # Discriminator optimizer
        opt_d = Adam(self.discriminator.parameters(), 
                    lr=self.learning_rate, betas=(0.5, 0.9))
        
        # Learning rate schedulers
        scheduler_g = CosineAnnealingLR(opt_g, T_max=100)
        scheduler_d = CosineAnnealingLR(opt_d, T_max=100)
        
        return [opt_g, opt_d], [scheduler_g, scheduler_d]
    
    def validation_step(self, batch, batch_idx):
        x, _ = batch  # CIFAR-10 returns (image, label) pairs
        x_recon, z, _, _ = self(x)
        
        # Calculate validation loss
        val_loss = F.mse_loss(x_recon, x)
        self.log('val_loss', val_loss)
        
        # Log sparsity metrics
        non_zero_elements = torch.mean((torch.abs(z) > 1e-5).float())
        self.log('val_sparsity', non_zero_elements)
        
        # Log sample images periodically
        if batch_idx == 0:
            self.logger.experiment.add_images('val_input', x[:4], self.current_epoch)
            self.logger.experiment.add_images('val_recon', x_recon[:4], self.current_epoch)
            
        return val_loss
    
    def prepare_data(self):
        # Download CIFAR-10 dataset
        torchvision.datasets.CIFAR10(root=self.data_dir, train=True, download=True)
        torchvision.datasets.CIFAR10(root=self.data_dir, train=False, download=True)
    
    def setup(self, stage=None):
        # Define transforms
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
        
        # Load CIFAR-10 dataset
        if stage == 'fit' or stage is None:
            self.cifar_train = torchvision.datasets.CIFAR10(
                root=self.data_dir, train=True, transform=transform
            )
            self.cifar_val = torchvision.datasets.CIFAR10(
                root=self.data_dir, train=False, transform=transform
            )
    
    def train_dataloader(self):
        return DataLoader(
            self.cifar_train, 
            batch_size=self.batch_size, 
            shuffle=True, 
            num_workers=4,
            pin_memory=True
        )
    
    def val_dataloader(self):
        return DataLoader(
            self.cifar_val, 
            batch_size=self.batch_size, 
            shuffle=False, 
            num_workers=4,
            pin_memory=True
        )