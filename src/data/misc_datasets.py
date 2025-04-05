import os
from typing import Dict, Optional, List, Tuple
import torch
from torch.utils.data import Dataset, Subset
import torchvision.datasets as tvds
import numpy as np
from PIL import Image

from base_dataset import BaseImageDataset, register_dataset, TransformSubset


@register_dataset('oxford_pets')
class OxfordPetsDataset(BaseImageDataset):
    """
    Oxford-IIIT Pet Dataset implementation for VQ-GAN training.
    
    The dataset contains images of 37 pet breeds with around 200 images for each class.
    """
    
    def __init__(
        self, 
        root_dir: str,
        transform=None,
        split: str = 'train',
        download: bool = True
    ):
        """
        Initialize the Oxford Pets dataset.
        
        Args:
            root_dir: Root directory containing the dataset
            transform: Optional transform to be applied on a sample
            split: Data split to use ('train' or 'test')
            download: Whether to download the dataset if not found
        """
        super().__init__(root_dir, transform, split)
        
        # Map our split names to torchvision's split names
        split_map = {
            'train': 'trainval',
            'val': 'test',
            'test': 'test'
        }
        
        if split not in split_map:
            raise ValueError(f"Invalid split: {split}. Expected 'train', 'val', or 'test'")
        
        self.tv_split = split_map[split]
        self.download = download
        self._load_dataset()
    
    def _load_dataset(self) -> None:
        """
        Load the Oxford Pets dataset.
        """
        try:
            # Create the dataset
            self.dataset = tvds.OxfordIIITPet(
                root=self.root_dir,
                split=self.tv_split,
                download=self.download,
                transform=None  # Will apply transforms later
            )
            
            print(f"Loaded Oxford Pets {self.tv_split} split: {len(self.dataset)} images")
            
        except Exception as e:
            print(f"Error loading Oxford Pets dataset: {e}")
            if not self.download:
                print("Try setting download=True to download the dataset.")
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
            tuple: (image, 0) where 0 is a dummy label (ignoring breed labels)
        """
        img, _ = self.dataset[idx]
        
        if self.transform:
            img = self.transform(img)
            
        return img, 0  # Return dummy label, as we don't care about pet breeds
    
    def get_splits(self, 
                  train_transform=None, 
                  val_transform=None) -> Dict[str, Dataset]:
        """
        Create train/val/test splits.
        
        For Oxford Pets:
        - Train split comes from the 'trainval' split
        - Val and test splits are created by dividing the 'test' split
        
        Args:
            train_transform: Transform to apply to training data
            val_transform: Transform to apply to validation/test data
            
        Returns:
            Dict containing 'train', 'val', and 'test' datasets
        """
        # Get train dataset
        train_dataset = OxfordPetsDataset(
            root_dir=self.root_dir,
            transform=train_transform,
            split='train',
            download=self.download
        )
        
        # Get test dataset and split into val/test
        test_full_dataset = OxfordPetsDataset(
            root_dir=self.root_dir,
            transform=None,  # Will apply transforms after splitting
            split='test',
            download=self.download
        )
        
        # Split test dataset into val/test
        test_indices = list(range(len(test_full_dataset)))
        np.random.shuffle(test_indices)
        split = len(test_indices) // 2
        
        val_indices = test_indices[:split]
        test_indices = test_indices[split:]
        
        val_dataset = Subset(test_full_dataset, val_indices)
        test_dataset = Subset(test_full_dataset, test_indices)
        
        # Apply transforms
        if val_transform is not None:
            val_dataset = TransformSubset(val_dataset, val_transform)
            test_dataset = TransformSubset(test_dataset, val_transform)
        
        return {
            'train': train_dataset,
            'val': val_dataset,
            'test': test_dataset
        }


@register_dataset('stanford_cars')
class StanfordCarsDataset(BaseImageDataset):
    """
    Stanford Cars Dataset implementation for VQ-GAN training.
    
    The dataset contains 16,185 images of 196 classes of cars.
    """
    
    def __init__(
        self, 
        root_dir: str,
        transform=None,
        split: str = 'train',
        download: bool = True
    ):
        """
        Initialize the Stanford Cars dataset.
        
        Args:
            root_dir: Root directory containing the dataset
            transform: Optional transform to be applied on a sample
            split: Data split to use ('train' or 'test')
            download: Whether to download the dataset if not found
        """
        super().__init__(root_dir, transform, split)
        
        # Map our split names to torchvision's split names
        split_map = {
            'train': 'train',
            'val': 'test',  # Stanford Cars doesn't have a val split
            'test': 'test'
        }
        
        if split not in split_map:
            raise ValueError(f"Invalid split: {split}. Expected 'train', 'val', or 'test'")
        
        self.tv_split = split_map[split]
        self.download = download
        self._load_dataset()
    
    def _load_dataset(self) -> None:
        """
        Load the Stanford Cars dataset.
        """
        try:
            # Create the dataset
            self.dataset = tvds.StanfordCars(
                root=self.root_dir,
                split=self.tv_split,
                download=self.download,
                transform=None  # Will apply transforms later
            )
            
            print(f"Loaded Stanford Cars {self.tv_split} split: {len(self.dataset)} images")
            
        except Exception as e:
            print(f"Error loading Stanford Cars dataset: {e}")
            if not self.download:
                print("Try setting download=True to download the dataset.")
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
            tuple: (image, 0) where 0 is a dummy label (ignoring car model labels)
        """
        img, _ = self.dataset[idx]
        
        if self.transform:
            img = self.transform(img)
            
        return img, 0  # Return dummy label
    
    def get_splits(self, 
                  train_transform=None, 
                  val_transform=None) -> Dict[str, Dataset]:
        """
        Create train/val/test splits.
        
        For Stanford Cars (which has no official val split):
        - Train split comes from the 'train' split
        - Val and test splits are created by dividing the 'test' split
        
        Args:
            train_transform: Transform to apply to training data
            val_transform: Transform to apply to validation/test data
            
        Returns:
            Dict containing 'train', 'val', and 'test' datasets
        """
        # Get train dataset
        train_dataset = StanfordCarsDataset(
            root_dir=self.root_dir,
            transform=train_transform,
            split='train',
            download=self.download
        )
        
        # Get test dataset and split into val/test
        test_full_dataset = StanfordCarsDataset(
            root_dir=self.root_dir,
            transform=None,  # Will apply transforms after splitting
            split='test',
            download=self.download
        )
        
        # Split test dataset into val/test
        test_indices = list(range(len(test_full_dataset)))
        np.random.shuffle(test_indices)
        split = len(test_indices) // 2
        
        val_indices = test_indices[:split]
        test_indices = test_indices[split:]
        
        val_dataset = Subset(test_full_dataset, val_indices)
        test_dataset = Subset(test_full_dataset, test_indices)
        
        # Apply transforms
        if val_transform is not None:
            val_dataset = TransformSubset(val_dataset, val_transform)
            test_dataset = TransformSubset(test_dataset, val_transform)
        
        return {
            'train': train_dataset,
            'val': val_dataset,
            'test': test_dataset
        }


@register_dataset('dtd')
class DTDDataset(BaseImageDataset):
    """
    Describable Textures Dataset (DTD) implementation for VQ-GAN training.
    
    The dataset contains 5,640 images of 47 texture categories with 120 images per category.
    """
    
    def __init__(
        self, 
        root_dir: str,
        transform=None,
        split: str = 'train',
        download: bool = True
    ):
        """
        Initialize the DTD dataset.
        
        Args:
            root_dir: Root directory containing the dataset
            transform: Optional transform to be applied on a sample
            split: Data split to use ('train', 'val', or 'test')
            download: Whether to download the dataset if not found
        """
        super().__init__(root_dir, transform, split)
        
        # DTD uses the same split names as we do
        self.tv_split = split
        self.download = download
        self._load_dataset()
    
    def _load_dataset(self) -> None:
        """
        Load the DTD dataset.
        """
        try:
            # Create the dataset
            self.dataset = tvds.DTD(
                root=self.root_dir,
                split=self.tv_split,
                download=self.download,
                transform=None  # Will apply transforms later
            )
            
            print(f"Loaded DTD {self.tv_split} split: {len(self.dataset)} images")
            
        except Exception as e:
            print(f"Error loading DTD dataset: {e}")
            if not self.download:
                print("Try setting download=True to download the dataset.")
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
            tuple: (image, 0) where 0 is a dummy label (ignoring texture category labels)
        """
        img, _ = self.dataset[idx]
        
        if self.transform:
            img = self.transform(img)
            
        return img, 0  # Return dummy label
    
    def get_splits(self, 
                  train_transform=None, 
                  val_transform=None) -> Dict[str, Dataset]:
        """
        Create train/val/test splits.
        
        DTD already has train/val/test splits, so we just use those.
        
        Args:
            train_transform: Transform to apply to training data
            val_transform: Transform to apply to validation/test data
            
        Returns:
            Dict containing 'train', 'val', and 'test' datasets
        """
        # DTD already has train/val/test splits
        train_dataset = DTDDataset(
            root_dir=self.root_dir,
            transform=train_transform,
            split='train',
            download=self.download
        )
        
        val_dataset = DTDDataset(
            root_dir=self.root_dir,
            transform=val_transform,
            split='val',
            download=self.download
        )
        
        test_dataset = DTDDataset(
            root_dir=self.root_dir,
            transform=val_transform,
            split='test',
            download=self.download
        )
        
        return {
            'train': train_dataset,
            'val': val_dataset,
            'test': test_dataset
        }


@register_dataset('cifar10')
class CIFAR10Dataset(BaseImageDataset):
    """
    CIFAR-10 dataset implementation for VQ-GAN training.
    
    The dataset contains 60,000 32x32 color images in 10 classes.
    Primarily used for testing and debugging VQ-GAN due to its small size.
    """
    
    def __init__(
        self, 
        root_dir: str,
        transform=None,
        split: str = 'train',
        download: bool = True
    ):
        """
        Initialize the CIFAR-10 dataset.
        
        Args:
            root_dir: Root directory containing the dataset
            transform: Optional transform to be applied on a sample
            split: Data split to use ('train' or 'test')
            download: Whether to download the dataset if not found
        """
        super().__init__(root_dir, transform, split)
        
        # Map our split names to torchvision's split names
        split_map = {
            'train': True,
            'val': False,  # CIFAR10 doesn't have a val split
            'test': False
        }
        
        if split not in split_map:
            raise ValueError(f"Invalid split: {split}. Expected 'train', 'val', or 'test'")
        
        self.is_train = split_map[split]
        self.download = download
        self._load_dataset()
    
    def _load_dataset(self) -> None:
        """
        Load the CIFAR-10 dataset.
        """
        try:
            # Create the dataset
            self.dataset = tvds.CIFAR10(
                root=self.root_dir,
                train=self.is_train,
                download=self.download,
                transform=None  # Will apply transforms later
            )
            
            split_name = 'train' if self.is_train else 'test'
            print(f"Loaded CIFAR-10 {split_name} split: {len(self.dataset)} images")
            
        except Exception as e:
            print(f"Error loading CIFAR-10 dataset: {e}")
            if not self.download:
                print("Try setting download=True to download the dataset.")
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
            tuple: (image, 0) where 0 is a dummy label (ignoring class labels)
        """
        img, _ = self.dataset[idx]
        
        if self.transform:
            img = self.transform(img)
            
        return img, 0  # Return dummy label
    
    def get_splits(self, 
                  train_transform=None, 
                  val_transform=None) -> Dict[str, Dataset]:
        """
        Create train/val/test splits.
        
        For CIFAR-10 (which has no official val split):
        - We use 80% of the training data for actual training
        - We use 20% of the training data for validation
        - We use the test set as is for testing
        
        Args:
            train_transform: Transform to apply to training data
            val_transform: Transform to apply to validation/test data
            
        Returns:
            Dict containing 'train', 'val', and 'test' datasets
        """
        # Load the full training set
        full_train_dataset = CIFAR10Dataset(
            root_dir=self.root_dir,
            transform=None,  # Will apply transforms after splitting
            split='train',
            download=self.download
        )
        
        # Load the test set
        test_dataset = CIFAR10Dataset(
            root_dir=self.root_dir,
            transform=val_transform,
            split='test',
            download=self.download
        )
        
        # Split training set into train/val
        train_indices = list(range(len(full_train_dataset)))
        np.random.shuffle(train_indices)
        
        split = int(0.8 * len(train_indices))
        
        actual_train_indices = train_indices[:split]
        val_indices = train_indices[split:]
        
        train_dataset = Subset(full_train_dataset, actual_train_indices)
        val_dataset = Subset(full_train_dataset, val_indices)
        
        # Apply transforms
        if train_transform is not None:
            train_dataset = TransformSubset(train_dataset, train_transform)
        if val_transform is not None:
            val_dataset = TransformSubset(val_dataset, val_transform)
        
        return {
            'train': train_dataset,
            'val': val_dataset,
            'test': test_dataset
        }
