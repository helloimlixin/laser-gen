import torch
import numpy as np

class ImageMetrics:
    """
    Simple class for computing image quality metrics for CIFAR-10 images.
    
    Metrics implemented:
    - MSE (Mean Squared Error)
    - PSNR (Peak Signal-to-Noise Ratio)
    - Custom Structural Similarity
    """
    
    def __init__(self, device):
        """
        Initialize the metrics calculator
        
        Args:
            device: torch device to use for computations
        """
        self.device = device
    
    def calculate_metrics(self, original, reconstructed):
        """
        Calculate all image metrics
        
        Args:
            original: Original images tensor [B, C, H, W], values in range [-1, 1]
            reconstructed: Reconstructed images tensor [B, C, H, W], values in range [-1, 1]
            
        Returns:
            Dictionary of metrics
        """
        # Convert to [0, 1] range for some metrics
        original_0_1 = (original + 1) / 2
        reconstructed_0_1 = (reconstructed + 1) / 2
        
        # Calculate metrics
        mse = self._calculate_mse(original, reconstructed)
        psnr_value = self._calculate_psnr(mse)
        ssim_value = self._calculate_simple_ssim(original_0_1, reconstructed_0_1)
        
        # Return all metrics
        metrics = {
            'mse': mse,
            'psnr': psnr_value,
            'ssim': ssim_value
        }
        
        return metrics
    
    def _calculate_mse(self, original, reconstructed):
        """Calculate Mean Squared Error"""
        return torch.nn.functional.mse_loss(original, reconstructed).item()
    
    def _calculate_psnr(self, mse):
        """
        Calculate Peak Signal-to-Noise Ratio from MSE
        
        Args:
            mse: Mean Squared Error value
            
        Returns:
            PSNR value
        """
        # Range of pixels is [-1, 1], so max value is 2
        return 10 * np.log10(4.0 / mse) if mse > 0 else 100.0
    
    def _calculate_simple_ssim(self, original, reconstructed):
        """
        Calculate a simplified version of Structural Similarity
        
        Args:
            original: Original images tensor [B, C, H, W], values in range [0, 1]
            reconstructed: Reconstructed images tensor [B, C, H, W], values in range [0, 1]
            
        Returns:
            Average simplified SSIM value across the batch
        """
        # Constants to stabilize division
        C1 = 0.01 ** 2
        C2 = 0.03 ** 2
        
        # Convert to NumPy for easier calculation
        orig = original.detach().cpu().numpy()
        recon = reconstructed.detach().cpu().numpy()
        
        # Calculate means per batch and channel
        mu_orig = np.mean(orig, axis=(2, 3), keepdims=True)
        mu_recon = np.mean(recon, axis=(2, 3), keepdims=True)
        
        # Calculate variances and covariance
        var_orig = np.mean((orig - mu_orig) ** 2, axis=(2, 3), keepdims=True)
        var_recon = np.mean((recon - mu_recon) ** 2, axis=(2, 3), keepdims=True)
        cov = np.mean((orig - mu_orig) * (recon - mu_recon), axis=(2, 3), keepdims=True)
        
        # Calculate SSIM
        numerator = (2 * mu_orig * mu_recon + C1) * (2 * cov + C2)
        denominator = (mu_orig ** 2 + mu_recon ** 2 + C1) * (var_orig + var_recon + C2)
        ssim = numerator / denominator
        
        # Average over batch and channels
        return float(np.mean(ssim))
    
    def calculate_batch_metrics(self, original, reconstructed):
        """
        Calculate metrics for a batch of images
        
        Args:
            original: Original images tensor [B, C, H, W], values in range [-1, 1]
            reconstructed: Reconstructed images tensor [B, C, H, W], values in range [-1, 1]
            
        Returns:
            Dictionary of metrics with average values
        """
        return self.calculate_metrics(original, reconstructed)
