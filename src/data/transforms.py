from typing import Tuple, Dict, Union, List, Optional
import torch
import torchvision.transforms as T


class TransformFactory:
    """
    Factory class to create different transform configurations for image datasets.
    """
    
    @staticmethod
    def get_transform_config(
        image_size: int = 256,
        normalize: bool = True,
        mean: Tuple[float, float, float] = (0.5, 0.5, 0.5),
        std: Tuple[float, float, float] = (0.5, 0.5, 0.5)
    ) -> Dict[str, T.Compose]:
        """
        Create transform configurations for train and validation/test.
        
        Args:
            image_size: Size to resize images to
            normalize: Whether to normalize the images
            mean: Mean for normalization
            std: Standard deviation for normalization
            
        Returns:
            Dictionary with 'train' and 'val' transform compositions
        """
        # Common transforms for both train and val
        common_transforms = []
        if normalize:
            common_transforms.append(T.Normalize(mean=mean, std=std))
        
        # Train specific transforms (with augmentation)
        train_transforms = [
            T.Resize((image_size, image_size)),
            T.RandomHorizontalFlip(),
            T.ToTensor(),
        ]
        train_transforms.extend(common_transforms)
        
        # Validation/test specific transforms (no augmentation)
        val_transforms = [
            T.Resize((image_size, image_size)),
            T.CenterCrop(image_size),
            T.ToTensor(),
        ]
        val_transforms.extend(common_transforms)
        
        return {
            'train': T.Compose(train_transforms),
            'val': T.Compose(val_transforms)
        }
    
    @staticmethod
    def get_simple_transform(
        image_size: int = 256,
        normalize: bool = True,
        train: bool = True
    ) -> T.Compose:
        """
        Create a simple transform for a specific split.
        
        Args:
            image_size: Size to resize images to
            normalize: Whether to normalize the images
            train: Whether to use training transforms (with augmentation)
            
        Returns:
            Composed transform
        """
        transforms = []
        
        # Resize operation
        transforms.append(T.Resize((image_size, image_size)))
        
        # Add augmentation for training
        if train:
            transforms.append(T.RandomHorizontalFlip())
        else:
            transforms.append(T.CenterCrop(image_size))
        
        # Convert to tensor
        transforms.append(T.ToTensor())
        
        # Add normalization
        if normalize:
            transforms.append(T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]))
        
        return T.Compose(transforms)
    
    @staticmethod
    def get_custom_transform(
        config: Dict[str, Union[int, float, bool, List]],
        train: bool = True
    ) -> T.Compose:
        """
        Create a custom transform based on a configuration dictionary.
        
        Args:
            config: Configuration dictionary specifying transforms
            train: Whether to use training transforms
            
        Returns:
            Composed transform
        """
        transforms = []
        
        # Process resize
        if 'image_size' in config:
            transforms.append(T.Resize((config['image_size'], config['image_size'])))
        
        # Training augmentations
        if train:
            if config.get('random_horizontal_flip', True):
                flip_prob = config.get('flip_probability', 0.5)
                transforms.append(T.RandomHorizontalFlip(p=flip_prob))
                
            if config.get('random_rotation', False):
                angle = config.get('rotation_degrees', 10)
                transforms.append(T.RandomRotation(angle))
                
            if config.get('color_jitter', False):
                brightness = config.get('brightness', 0.1)
                contrast = config.get('contrast', 0.1)
                saturation = config.get('saturation', 0.1)
                hue = config.get('hue', 0.05)
                transforms.append(
                    T.ColorJitter(brightness=brightness, contrast=contrast, 
                                  saturation=saturation, hue=hue)
                )
        else:
            # Validation/test transforms
            if config.get('center_crop', True):
                transforms.append(T.CenterCrop(config['image_size']))
        
        # Convert to tensor
        transforms.append(T.ToTensor())
        
        # Normalization
        if config.get('normalize', True):
            mean = config.get('mean', [0.5, 0.5, 0.5])
            std = config.get('std', [0.5, 0.5, 0.5])
            transforms.append(T.Normalize(mean=mean, std=std))
            
        return T.Compose(transforms)
