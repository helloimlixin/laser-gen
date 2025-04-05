import os
from typing import Dict, Optional, List, Tuple, Union
import torch
from torch.utils.data import Dataset, Subset
import numpy as np
from PIL import Image

from base_dataset import BaseImageDataset, register_dataset, TransformSubset


class LSUNWrapper(Dataset):
    """
    A wrapper for LSUN dataset to handle transformations.
    """
    
    def __init__(self, lsun_dataset, transform=None):
        """
        Initialize the LSUN wrapper.
        
        Args:
            lsun_dataset: An LSUN dataset instance
            transform: Optional transform to be applied on a sample
        """
        self.lsun_dataset = lsun_dataset
        self.transform = transform
        
    def __len__(self) -> int:
        """Returns the number of images in the dataset."""
        return len(self.lsun_dataset)
        
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        """
        Returns a sample from the dataset at the given index.
        
        Args:
            idx: Index of the sample to return
            
        Returns:
            tuple: (image, 0) where 0 is a dummy label
        """
        img, target = self.lsun_dataset[idx]
        
        if self.transform:
            img = self.transform(img)
            
        return img, 0  # Return dummy label


@register_dataset('lsun')
class LSUNDataset(BaseImageDataset):
    """
    LSUN dataset implementation for VQ-GAN training.
    
    The LSUN (Large-Scale Scene Understanding) dataset contains around 
    10 million labeled images in 10 scene categories and 20 object categories.
    
    This implementation focuses on specific categories (e.g., bedroom, church, etc.)
    as each category can have millions of images.
    """
    
    def __init__(
        self, 
        root_dir: str,
        transform=None,
        split: str = 'train',
        category: str = 'bedroom'
    ):
        """
        Initialize the LSUN dataset.
        
        Args:
            root_dir: Root directory containing the dataset
            transform: Optional transform to be applied on a sample
            split: Data split to use ('train' or 'val')
            category: LSUN category to use (e.g., 'bedroom', 'church_outdoor')
        """
        super().__init__(root_dir, transform, split)
        
        self.category = category
        self.lsun_dir = os.path.join(root_dir, 'lsun')
        
        # Convert split name (LSUN uses train/val)
        self.actual_split = split
        if split == 'test':
            self.actual_split = 'val'  # LSUN doesn't have a test set
        
        self._load_dataset()
    
    def _load_dataset(self) -> None:
        """
        Load the LSUN dataset.
        """
        try:
            # Import LSUN dataset class
            from torchvision.datasets import LSUN
            import lmdb  # required for LSUN
            
            # Define the LSUN category and split
            lsun_class = f"{self.category}_{self.actual_split}"
            
            # Verify that the LSUN directory exists
            if not os.path.exists(self.lsun_dir):
                raise FileNotFoundError(f"LSUN directory not found: {self.lsun_dir}")
            
            # Create LSUN dataset (no transform yet)
            self.dataset = LSUN(
                root=self.lsun_dir,
                classes=[lsun_class],
                transform=None
            )
            
            print(f"Loaded LSUN {lsun_class} dataset: {len(self.dataset)} images")
            
        except Exception as e:
            print(f"Error loading LSUN dataset: {e}")
            print("Please follow the instructions at https://github.com/fyu/lsun to download and prepare the LSUN dataset.")
            raise
    
    def __len__(self) -> int:
        """Returns the number of images in the dataset."""
        return len(self.dataset)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        """
        Returns a sample from the dataset at the given index.
        
        Args:
            idx: Index of the sample to return
            
        Returns:
            tuple: (image, 0) where 0 is a dummy label
        """
        img, _ = self.dataset[idx]
        
        if self.transform:
            img = self.transform(img)
            
        return img, 0  # Return dummy label
    
    def get_splits(self, 
                  train_transform=None, 
                  val_transform=None,
                  val_size: int = 5000,
                  test_size: int = 5000) -> Dict[str, Dataset]:
        """
        Create train/val/test splits from LSUN train and val sets.
        
        Args:
            train_transform: Transform to apply to training data
            val_transform: Transform to apply to validation/test data
            val_size: Number of images to use for validation (from LSUN val set)
            test_size: Number of images to use for testing (from LSUN val set)
            
        Returns:
            Dict containing 'train', 'val', and 'test' datasets
        """
        # Get the train dataset
        train_dataset = LSUNDataset(
            root_dir=self.root_dir,
            transform=train_transform,
            split='train',
            category=self.category
        )
        
        # Get the val dataset (which we'll split into val/test)
        val_full_dataset = LSUNDataset(
            root_dir=self.root_dir,
            transform=None,  # Will apply transforms after splitting
            split='val',
            category=self.category
        )
        
        # Check if we have enough validation data
        if len(val_full_dataset) < val_size + test_size:
            # If not enough val data, we'll need to use some train data
            total_needed = val_size + test_size
            available_val = len(val_full_dataset)
            
            print(f"Warning: Not enough validation data ({available_val} available, {total_needed} needed).")
            print(f"Using {available_val} for validation and taking {total_needed - available_val} from training data for testing.")
            
            # Use all validation data for val set
            val_dataset = LSUNWrapper(val_full_dataset, val_transform)
            
            # Take some training data for test set
            train_indices = list(range(len(train_dataset)))
            np.random.shuffle(train_indices)
            test_from_train = test_size
            
            test_indices = train_indices[:test_from_train]
            remaining_train_indices = train_indices[test_from_train:]
            
            # Create the datasets
            test_dataset = Subset(train_dataset, test_indices)
            train_dataset = Subset(train_dataset, remaining_train_indices)
            
            # Apply transforms for test set
            if val_transform is not None:
                test_dataset = TransformSubset(test_dataset, val_transform)
        else:
            # We have enough validation data, so split it 50/50 for val/test
            val_indices = list(range(len(val_full_dataset)))
            np.random.shuffle(val_indices)
            
            # Limit to the requested sizes
            val_indices = val_indices[:val_size]
            test_indices = val_indices[val_size:val_size + test_size]
            
            # Create the datasets
            val_dataset = Subset(val_full_dataset, val_indices)
            test_dataset = Subset(val_full_dataset, test_indices)
            
            # Apply transforms
            if val_transform is not None:
                val_dataset = TransformSubset(val_dataset, val_transform)
                test_dataset = TransformSubset(test_dataset, val_transform)
        
        # If train_transform is provided, apply it to train dataset
        if train_transform is not None and not isinstance(train_dataset, TransformSubset):
            train_dataset = LSUNWrapper(train_dataset, train_transform)
        
        return {
            'train': train_dataset,
            'val': val_dataset,
            'test': test_dataset
        }
    
    @staticmethod
    def get_available_categories() -> List[str]:
        """
        Returns a list of available LSUN categories.
        
        Returns:
            List of category names
        """
        return [
            'bedroom', 'bridge', 'church_outdoor', 'classroom',
            'conference_room', 'dining_room', 'kitchen',
            'living_room', 'restaurant', 'tower'
        ]


# Usage example:
if __name__ == "__main__":
    from torchvision import transforms
    
    # Create transforms
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])
    
    # Print available categories
    print(f"Available LSUN categories: {LSUNDataset.get_available_categories()}")
    
    # Create dataset
    dataset = LSUNDataset(
        root_dir='/path/to/data',
        transform=transform,
        split='train',
        category='bedroom'
    )
    
    # Get train/val/test splits
    splits = dataset.get_splits(transform, transform)
    
    print(f"Train split: {len(splits['train'])} images")
    print(f"Val split: {len(splits['val'])} images")
    print(f"Test split: {len(splits['test'])} images")
    
    # Get a sample
    img, _ = dataset[0]
    print(f"Image shape: {img.shape}")  # Should be [3, 256, 256]