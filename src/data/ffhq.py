import os
from typing import Dict, Optional, List, Tuple
import torch
from torch.utils.data import Dataset, Subset
import numpy as np
from PIL import Image

from base_dataset import BaseImageDataset, register_dataset, TransformSubset


@register_dataset('ffhq')
class FFHQDataset(BaseImageDataset):
    """
    Flickr-Faces-HQ (FFHQ) dataset implementation for VQ-GAN training.
    
    FFHQ consists of 70,000 high-quality PNG images at 1024×1024 resolution
    containing diverse faces with variations in age, ethnicity and background.
    """
    
    def __init__(
        self, 
        root_dir: str,
        transform=None,
        split: str = 'train',
        image_size: int = 256
    ):
        """
        Initialize the FFHQ dataset.
        
        Args:
            root_dir: Root directory containing the dataset
            transform: Optional transform to be applied on a sample
            split: Data split to use (train, val, test) - note: FFHQ doesn't have official splits
            image_size: Target image size (original FFHQ is 1024x1024)
        """
        super().__init__(root_dir, transform, split)
        
        self.ffhq_dir = os.path.join(root_dir, 'ffhq')
        self.image_size = image_size
        self._load_dataset()
    
    def _load_dataset(self) -> None:
        """
        Load the FFHQ dataset.
        
        FFHQ can be organized in two ways:
        1. A single directory with all images
        2. Multiple subdirectories (00000-09999, 10000-19999, etc.)
        
        This method handles both cases.
        """
        try:
            # Verify dataset directory exists
            if not os.path.exists(self.ffhq_dir):
                raise FileNotFoundError(f"Dataset directory not found: {self.ffhq_dir}")
            
            # Check if there are PNG or JPEG files directly in the root directory
            direct_images = [f for f in os.listdir(self.ffhq_dir) 
                            if f.endswith(('.png', '.jpg', '.jpeg'))]
            
            if direct_images:
                # Case 1: All images in the root directory
                self.image_files = direct_images
            else:
                # Case 2: Images in subdirectories
                self.image_files = []
                for subdir in sorted(os.listdir(self.ffhq_dir)):
                    subdir_path = os.path.join(self.ffhq_dir, subdir)
                    if os.path.isdir(subdir_path):
                        for f in os.listdir(subdir_path):
                            if f.endswith(('.png', '.jpg', '.jpeg')):
                                # Store the relative path
                                self.image_files.append(os.path.join(subdir, f))
            
            if not self.image_files:
                raise FileNotFoundError(f"No images found in {self.ffhq_dir}")
            
            # Apply split filtering if this was already split
            if hasattr(self, 'split_indices') and self.split_indices is not None:
                if self.split == 'train':
                    start_idx = 0
                    end_idx = int(len(self.image_files) * 0.8)
                elif self.split == 'val':
                    start_idx = int(len(self.image_files) * 0.8)
                    end_idx = int(len(self.image_files) * 0.9)
                elif self.split == 'test':
                    start_idx = int(len(self.image_files) * 0.9)
                    end_idx = len(self.image_files)
                else:
                    start_idx = 0
                    end_idx = len(self.image_files)
                
                self.image_files = self.image_files[start_idx:end_idx]
            
            print(f"Loaded FFHQ dataset: {len(self.image_files)} images")
            
        except Exception as e:
            print(f"Error loading FFHQ dataset: {e}")
            print("Please download the dataset from: https://github.com/NVlabs/ffhq-dataset")
            raise
    
    def _get_image_path(self, idx: int) -> str:
        """
        Get the full path to an image file.
        
        Args:
            idx: Index of the image
            
        Returns:
            str: Full path to the image file
        """
        return os.path.join(self.ffhq_dir, self.image_files[idx])
    
    def create_split_indices(self, seed: int = 42) -> Dict[str, List[int]]:
        """
        Create indices for train/val/test splits.
        
        Args:
            seed: Random seed for reproducibility
            
        Returns:
            Dictionary with indices for each split
        """
        np.random.seed(seed)
        
        # Get indices for all images
        all_indices = list(range(len(self.image_files)))
        np.random.shuffle(all_indices)
        
        # Create splits (80% train, 10% val, 10% test)
        train_size = int(0.8 * len(all_indices))
        val_size = int(0.1 * len(all_indices))
        
        train_indices = all_indices[:train_size]
        val_indices = all_indices[train_size:train_size+val_size]
        test_indices = all_indices[train_size+val_size:]
        
        return {
            'train': train_indices,
            'val': val_indices,
            'test': test_indices
        }
    
    def get_splits(self, 
                  train_transform=None, 
                  val_transform=None,
                  seed: int = 42) -> Dict[str, Dataset]:
        """
        Split the dataset into train, validation, and test sets.
        
        Args:
            train_transform: Transform to apply to training data
            val_transform: Transform to apply to validation/test data
            seed: Random seed for reproducibility
            
        Returns:
            Dict containing 'train', 'val', and 'test' datasets
        """
        # Create split indices if not already created
        split_indices = self.create_split_indices(seed)
        
        # Create a fresh dataset instance without filtered data
        full_dataset = FFHQDataset(
            root_dir=self.root_dir,
            transform=None,  # Will apply transforms after creating subsets
            split='all'  # Use all data
        )
        
        # Create subsets using the indices
        train_subset = Subset(full_dataset, split_indices['train'])
        val_subset = Subset(full_dataset, split_indices['val'])
        test_subset = Subset(full_dataset, split_indices['test'])
        
        # Apply transforms if provided
        if train_transform is not None or val_transform is not None:
            train_subset = TransformSubset(train_subset, train_transform)
            val_subset = TransformSubset(val_subset, val_transform)
            test_subset = TransformSubset(test_subset, val_transform)
        
        return {
            'train': train_subset,
            'val': val_subset,
            'test': test_subset
        }


# Usage example:
if __name__ == "__main__":
    from torchvision import transforms
    
    # Create transforms
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])
    
    # Create dataset
    dataset = FFHQDataset(
        root_dir='/path/to/data',
        transform=transform,
        split='train'
    )
    
    # Get train/val/test splits
    splits = dataset.get_splits(transform, transform)
    
    print(f"Train split: {len(splits['train'])} images")
    print(f"Val split: {len(splits['val'])} images")
    print(f"Test split: {len(splits['test'])} images")
    
    # Get a sample
    img, _ = dataset[0]
    print(f"Image shape: {img.shape}")  # Should be [3, 256, 256]
