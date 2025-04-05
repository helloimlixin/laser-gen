import os
from typing import Dict, Optional, List, Tuple
import torch
from torch.utils.data import Dataset, Subset
import numpy as np
from PIL import Image

from base_dataset import BaseImageDataset, register_dataset, TransformSubset


@register_dataset('div2k')
class DIV2KDataset(BaseImageDataset):
    """
    DIV2K dataset implementation for VQ-GAN training.
    
    DIV2K (DIVerse 2K resolution high quality images) is a dataset
    containing 1,000 2K resolution RGB images with diverse contents.
    It was designed for super-resolution tasks but is excellent for
    high-quality image generation.
    """
    
    def __init__(
        self, 
        root_dir: str,
        transform=None,
        split: str = 'train',
        scale: str = 'HR'  # HR (High Resolution) or LR (Low Resolution)
    ):
        """
        Initialize the DIV2K dataset.
        
        Args:
            root_dir: Root directory containing the dataset
            transform: Optional transform to be applied on a sample
            split: Data split to use ('train' or 'valid')
            scale: Whether to use high resolution (HR) or low resolution (LR) images
        """
        super().__init__(root_dir, transform, split)
        
        # Mapping for split names
        self.split_map = {
            'train': 'train',
            'val': 'valid',
            'valid': 'valid',
            'test': 'valid'  # DIV2K doesn't have an official test set, use valid instead
        }
        
        # Convert split name
        if split in self.split_map:
            self.actual_split = self.split_map[split]
        else:
            raise ValueError(f"Invalid split: {split}. Expected 'train', 'val', or 'test'")
        
        self.scale = scale
        self.div2k_dir = os.path.join(root_dir, 'DIV2K')
        
        # Construct path to images
        self.images_dir = os.path.join(
            self.div2k_dir, 
            f'DIV2K_{self.actual_split}_{self.scale}'
        )
        
        self._load_dataset()
    
    def _load_dataset(self) -> None:
        """
        Load the DIV2K dataset.
        """
        try:
            # Verify dataset directory exists
            if not os.path.exists(self.images_dir):
                raise FileNotFoundError(f"Dataset directory not found: {self.images_dir}")
            
            # Get all image files (typically .png)
            self.image_files = [
                f for f in os.listdir(self.images_dir)
                if f.endswith(('.png', '.jpg', '.jpeg', '.bmp'))
            ]
            
            if not self.image_files:
                raise FileNotFoundError(f"No images found in {self.images_dir}")
            
            print(f"Loaded DIV2K {self.actual_split} {self.scale} dataset: {len(self.image_files)} images")
            
        except Exception as e:
            print(f"Error loading DIV2K dataset: {e}")
            print("Please download the dataset from: https://data.vision.ee.ethz.ch/cvl/DIV2K/")
            print("Expected directory structure:")
            print(f"  {self.div2k_dir}/")
            print(f"  ├── DIV2K_train_HR/")
            print(f"  ├── DIV2K_train_LR_bicubic/")
            print(f"  ├── DIV2K_valid_HR/")
            print(f"  └── DIV2K_valid_LR_bicubic/")
            raise
    
    def _get_image_path(self, idx: int) -> str:
        """
        Get the full path to an image file.
        
        Args:
            idx: Index of the image
            
        Returns:
            str: Full path to the image file
        """
        return os.path.join(self.images_dir, self.image_files[idx])
    
    def get_splits(self, 
                  train_transform=None, 
                  val_transform=None) -> Dict[str, Dataset]:
        """
        Create train/val/test splits from DIV2K train and val sets.
        
        For DIV2K:
        - 'train' split comes directly from DIV2K train set (800 images)
        - 'val' and 'test' splits are created by dividing DIV2K val set (100 images)
        
        Args:
            train_transform: Transform to apply to training data
            val_transform: Transform to apply to validation/test data
            
        Returns:
            Dict containing 'train', 'val', and 'test' datasets
        """
        # Get the train dataset from DIV2K train split
        train_dataset = DIV2KDataset(
            root_dir=self.root_dir,
            transform=train_transform,
            split='train',
            scale=self.scale
        )
        
        # Create a dataset for the validation set
        valid_dataset = DIV2KDataset(
            root_dir=self.root_dir,
            transform=None,  # We'll apply transforms after splitting
            split='valid',
            scale=self.scale
        )
        
        # Split validation into our val/test
        valid_indices = list(range(len(valid_dataset)))
        np.random.shuffle(valid_indices)
        split = len(valid_indices) // 2
        
        val_indices = valid_indices[:split]
        test_indices = valid_indices[split:]
        
        val_dataset = Subset(valid_dataset, val_indices)
        test_dataset = Subset(valid_dataset, test_indices)
        
        # Apply transforms if provided
        if val_transform is not None:
            val_dataset = TransformSubset(val_dataset, val_transform)
            test_dataset = TransformSubset(test_dataset, val_transform)
        
        return {
            'train': train_dataset,
            'val': val_dataset,
            'test': test_dataset
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
    dataset = DIV2KDataset(
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
    print(f"Image shape: {img.shape}")

    