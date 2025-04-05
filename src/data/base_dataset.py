import os
from abc import ABC, abstractmethod
from typing import Tuple, Dict, Any, List, Optional
import torch
from torch.utils.data import Dataset, Subset
import numpy as np
from PIL import Image


class BaseImageDataset(Dataset, ABC):
    """
    Abstract base class for all image datasets in the VQ-GAN project.
    
    This standardizes dataset behavior and ensures all datasets implement
    the required functionality.
    """
    
    def __init__(
        self, 
        root_dir: str,
        transform=None,
        split: str = 'train'
    ):
        """
        Initialize the base image dataset.
        
        Args:
            root_dir: Root directory containing the dataset
            transform: Optional transform to be applied on a sample
            split: Data split to use (train, val, test)
        """
        self.root_dir = root_dir
        self.transform = transform
        self.split = split
        self.image_files = []
    
    @abstractmethod
    def _load_dataset(self) -> None:
        """
        Load the dataset. This should populate self.image_files.
        Must be implemented by subclasses.
        """
        pass
    
    def __len__(self) -> int:
        """Returns the number of images in the dataset."""
        return len(self.image_files)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        """
        Returns a sample from the dataset at the given index.
        
        Args:
            idx: Index of the sample to return
            
        Returns:
            tuple: (image, 0) where 0 is a dummy label
        """
        # Default implementation for image loading
        img_path = self._get_image_path(idx)
        image = Image.open(img_path).convert('RGB')
        
        if self.transform:
            image = self.transform(image)
            
        return image, 0  # Return dummy label
    
    def _get_image_path(self, idx: int) -> str:
        """
        Get the full path to an image file.
        
        Args:
            idx: Index of the image
            
        Returns:
            str: Full path to the image file
        """
        return os.path.join(self.root_dir, self.image_files[idx])
    
    def get_splits(self, 
                   train_transform=None, 
                   val_transform=None,
                   train_ratio: float = 0.8, 
                   val_ratio: float = 0.1) -> Dict[str, Dataset]:
        """
        Split the dataset into train, validation, and test sets.
        
        Args:
            train_transform: Transform to apply to training data
            val_transform: Transform to apply to validation and test data
            train_ratio: Proportion of data to use for training
            val_ratio: Proportion of data to use for validation
            
        Returns:
            Dict containing 'train', 'val', and 'test' datasets
        """
        total_size = len(self)
        train_size = int(train_ratio * total_size)
        val_size = int(val_ratio * total_size)
        test_size = total_size - train_size - val_size
        
        indices = list(range(total_size))
        np.random.shuffle(indices)
        
        train_indices = indices[:train_size]
        val_indices = indices[train_size:train_size+val_size]
        test_indices = indices[train_size+val_size:]
        
        # Create the base subsets
        train_subset = Subset(self, train_indices)
        val_subset = Subset(self, val_indices)
        test_subset = Subset(self, test_indices)
        
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


class TransformSubset(Dataset):
    """
    A wrapper for applying transforms to a subset of a dataset.
    """
    
    def __init__(self, subset: Subset, transform=None):
        """
        Initialize the transform subset.
        
        Args:
            subset: The subset to wrap
            transform: Transform to apply to samples
        """
        self.subset = subset
        self.transform = transform
        
    def __len__(self) -> int:
        """Returns the number of samples in the subset."""
        return len(self.subset)
        
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        """
        Returns a transformed sample from the subset.
        
        Args:
            idx: Index of the sample to return
            
        Returns:
            tuple: (transformed_image, label)
        """
        img, label = self.subset[idx]
        if self.transform:
            img = self.transform(img)
        return img, label


class DatasetFactory:
    """
    Factory class for creating dataset instances.
    
    This provides a centralized registry for all available dataset types
    and handles creation with proper configuration.
    """
    
    _datasets = {}
    
    @classmethod
    def register_dataset(cls, name: str, dataset_class):
        """
        Register a dataset class with a name.
        
        Args:
            name: The name to register the dataset under
            dataset_class: The dataset class to register
        """
        cls._datasets[name.lower()] = dataset_class
    
    @classmethod
    def create(cls, name: str, **kwargs) -> BaseImageDataset:
        """
        Create a dataset instance by name.
        
        Args:
            name: The registered name of the dataset
            **kwargs: Arguments to pass to the dataset constructor
            
        Returns:
            An instance of the requested dataset
            
        Raises:
            ValueError: If the dataset name is not registered
        """
        name = name.lower()
        if name not in cls._datasets:
            raise ValueError(f"Dataset '{name}' not registered. Available datasets: {list(cls._datasets.keys())}")
        
        return cls._datasets[name](**kwargs)
    
    @classmethod
    def list_available_datasets(cls) -> List[str]:
        """
        List all registered dataset names.
        
        Returns:
            List of available dataset names
        """
        return list(cls._datasets.keys())


# Decorator for dataset registration
def register_dataset(name: str):
    """
    Decorator to register a dataset class with the DatasetFactory.
    
    Args:
        name: The name to register the dataset under
        
    Returns:
        Decorator function
    """
    def decorator(cls):
        DatasetFactory.register_dataset(name, cls)
        return cls
    return decorator
