# src/data/__init__.py

from base_dataset import BaseImageDataset, DatasetFactory, register_dataset
from transforms import TransformFactory
from datamodule import VQGANDataModule

# Import all datasets to register them
from datasets import *

# src/data/datasets/__init__.py

# Import all dataset modules to register them with DatasetFactory
from .coco import COCODataset
from .lsun import LSUNDataset
from .div2k import DIV2KDataset
from .ffhq import FFHQDataset
from .misc_datasets import (
    OxfordPetsDataset,
    StanfordCarsDataset,
    DTDDataset,
    CIFAR10Dataset
)

# Export the dataset classes
__all__ = [
    'COCODataset',
    'LSUNDataset', 
    'DIV2KDataset',
    'FFHQDataset',
    'OxfordPetsDataset',
    'StanfordCarsDataset',
    'DTDDataset',
    'CIFAR10Dataset'
]