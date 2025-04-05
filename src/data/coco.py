import os
from typing import Dict, Optional, List, Tuple
import torch
from torch.utils.data import Dataset, Subset
import numpy as np
from PIL import Image
from torchvision.datasets import CocoDetection

from base_dataset import BaseImageDataset, register_dataset, TransformSubset


class COCOImageOnlyWrapper(Dataset):
    """
    A wrapper for COCO dataset that returns only images (no annotations).
    """
    
    def __init__(self, coco_dataset):
        """
        Initialize the COCO wrapper.
        
        Args:
            coco_dataset: A CocoDetection dataset instance
        """
        self.coco_dataset = coco_dataset
        
    def __len__(self) -> int:
        """Returns the number of images in the dataset."""
        return len(self.coco_dataset)
        
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        """
        Returns an image from the dataset at the given index.
        
        Args:
            idx: Index of the image to return
            
        Returns:
            tuple: (image, 0) where 0 is a dummy label
        """
        img, _ = self.coco_dataset[idx]  # Ignore annotations
        return img, 0  # Return dummy label


@register_dataset('coco')
class COCODataset(BaseImageDataset):
    """
    COCO dataset implementation for VQ-GAN training.
    
    The Common Objects in Context (COCO) dataset is a large-scale object detection,
    segmentation, and captioning dataset with over 330,000 images.
    """
    
    def __init__(
        self, 
        root_dir: str,
        transform=None, 
        split: str = 'train',
        year: str = '2017',
        require_annotations: bool = False
    ):
        """
        Initialize the COCO dataset.
        
        Args:
            root_dir: Root directory containing the COCO dataset
            transform: Optional transform to be applied on images
            split: Data split to use ('train' or 'val')
            year: COCO dataset year ('2017' by default)
            require_annotations: Whether to include annotations in the output
        """
        super().__init__(root_dir, transform, split)
        
        self.year = year
        self.require_annotations = require_annotations
        self.coco_root = os.path.join(root_dir, 'coco')
        
        # Determine the appropriate data split
        if split == 'train':
            self.img_dir = os.path.join(self.coco_root, f'train{year}')
            self.ann_file = os.path.join(self.coco_root, 'annotations', f'instances_train{year}.json')
        elif split in ['val', 'test']:  # COCO only has train/val, we treat val and test the same
            self.img_dir = os.path.join(self.coco_root, f'val{year}')
            self.ann_file = os.path.join(self.coco_root, 'annotations', f'instances_val{year}.json')
        else:
            raise ValueError(f"Invalid split: {split}. Expected 'train', 'val', or 'test'")
        
        self._load_dataset()
        
    def _load_dataset(self) -> None:
        """
        Load the COCO dataset.
        """
        try:
            # Verify that the directories exist
            if not os.path.exists(self.img_dir):
                raise FileNotFoundError(f"Image directory not found: {self.img_dir}")
            
            if not os.path.exists(self.ann_file):
                raise FileNotFoundError(f"Annotation file not found: {self.ann_file}")
            
            # Load COCO dataset
            self.coco_dataset = CocoDetection(
                root=self.img_dir,
                annFile=self.ann_file,
                transform=None  # We'll apply transforms later
            )
            
            # If no annotations are required, wrap the dataset
            if not self.require_annotations:
                self.coco_dataset = COCOImageOnlyWrapper(self.coco_dataset)
            
            print(f"Loaded COCO {self.year} {self.split} split: {len(self.coco_dataset)} images")
            
        except Exception as e:
            print(f"Error loading COCO dataset: {e}")
            print("Please ensure the COCO dataset is properly downloaded and organized.")
            print("See: https://cocodataset.org/#download")
            raise
    
    def __len__(self) -> int:
        """Returns the number of images in the dataset."""
        return len(self.coco_dataset)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        """
        Returns a sample from the dataset at the given index.
        
        Args:
            idx: Index of the sample to return
            
        Returns:
            tuple: (image, label) where label is 0 (dummy) or annotations
        """
        img, label = self.coco_dataset[idx]
        
        if self.transform:
            img = self.transform(img)
            
        return img, label
    
    def get_splits(self, 
                  train_transform=None, 
                  val_transform=None) -> Dict[str, Dataset]:
        """
        Create train/val/test splits from COCO train and val sets.
        
        Args:
            train_transform: Transform to apply to training data
            val_transform: Transform to apply to validation/test data
            
        Returns:
            Dict containing 'train', 'val', and 'test' datasets
        """
        # For COCO, we need a different approach:
        # 1. Train split comes from COCO's train set
        # 2. Val and test splits come from COCO's val set (divided 50/50)
        
        # Get the train dataset from COCO train split
        train_dataset = COCODataset(
            root_dir=self.root_dir,
            transform=train_transform,
            split='train',
            year=self.year,
            require_annotations=self.require_annotations
        )
        
        # Create val/test datasets from COCO val split
        val_full_dataset = COCODataset(
            root_dir=self.root_dir,
            transform=None,  # We'll apply transforms after splitting
            split='val',
            year=self.year,
            require_annotations=self.require_annotations
        )
        
        # Split COCO val into our val/test
        val_size = len(val_full_dataset)
        half_val_size = val_size // 2
        
        indices = list(range(val_size))
        np.random.shuffle(indices)
        
        val_indices = indices[:half_val_size]
        test_indices = indices[half_val_size:]
        
        val_dataset = Subset(val_full_dataset, val_indices)
        test_dataset = Subset(val_full_dataset, test_indices)
        
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
    dataset = COCODataset(
        root_dir='/path/to/data',
        transform=transform,
        split='train'
    )
    
    # Get train/val/test splits
    splits = dataset.get_splits(transform, transform)
    
    print(f"Train split: {len(splits['train'])} images")
    print(f"Val split: {len(splits['val'])} images")
    print(f"Test split: {len(splits['test'])} images")