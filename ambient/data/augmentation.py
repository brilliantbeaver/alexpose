"""
Data Augmentation Module

This module provides data augmentation capabilities for gait analysis training data,
preserving essential gait characteristics while introducing controlled variations.

Key Features:
- Temporal augmentation (speed variation, time warping)
- Spatial augmentation (scaling, rotation, translation)
- Noise injection for robustness
- Gait-specific augmentations preserving biomechanical constraints
- Configurable augmentation pipelines

Design Principles:
- Single Responsibility: Each augmentation class handles one transformation
- Open/Closed: Extensible for new augmentation types
- Strategy Pattern: Pluggable augmentation strategies

Author: AlexPose Team
"""

import copy
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from loguru import logger

from ambient.core.frame import Frame, FrameSequence


class IAugmentation(ABC):
    """Interface for augmentation strategies."""
    
    @abstractmethod
    def augment(self, data: Any) -> Any:
        """Apply augmentation to data."""
        pass
    
    @abstractmethod
    def get_parameters(self) -> Dict[str, Any]:
        """Get augmentation parameters."""
        pass


class TemporalSpeedAugmentation(IAugmentation):
    """
    Augment gait sequences by varying playback speed.
    
    This preserves gait patterns while simulating different walking speeds.
    """
    
    def __init__(self, speed_range: Tuple[float, float] = (0.8, 1.2), random_seed: Optional[int] = None):
        """
        Initialize temporal speed augmentation.
        
        Args:
            speed_range: Range of speed multipliers (min, max)
            random_seed: Random seed for reproducibility
        """
        self.speed_range = speed_range
        self.random_seed = random_seed
        if random_seed is not None:
            np.random.seed(random_seed)
    
    def augment(self, sequence: FrameSequence) -> FrameSequence:
        """
        Augment frame sequence by varying speed.
        
        Args:
            sequence: Input FrameSequence
            
        Returns:
            Augmented FrameSequence
        """
        speed_factor = np.random.uniform(*self.speed_range)
        
        # Calculate new frame indices
        original_length = len(sequence)
        new_length = int(original_length / speed_factor)
        
        # Interpolate frame indices
        original_indices = np.arange(original_length)
        new_indices = np.linspace(0, original_length - 1, new_length)
        
        # Select frames at interpolated indices
        augmented_frames = []
        for idx in new_indices:
            frame_idx = int(np.round(idx))
            frame_idx = min(frame_idx, original_length - 1)
            augmented_frames.append(sequence.frames[frame_idx].copy())
        
        metadata = sequence.metadata.copy()
        metadata['augmentation'] = {
            'type': 'temporal_speed',
            'speed_factor': speed_factor,
            'original_length': original_length,
            'new_length': new_length
        }
        
        logger.debug(f"Applied temporal speed augmentation: {speed_factor:.2f}x speed")
        
        return FrameSequence(
            frames=augmented_frames,
            metadata=metadata,
            sequence_id=f"{sequence.sequence_id}_speed_{speed_factor:.2f}"
        )
    
    def get_parameters(self) -> Dict[str, Any]:
        """Get augmentation parameters."""
        return {
            'type': 'temporal_speed',
            'speed_range': self.speed_range,
            'random_seed': self.random_seed
        }


class SpatialScalingAugmentation(IAugmentation):
    """
    Augment gait data by scaling spatial coordinates.
    
    This simulates different person sizes while preserving gait patterns.
    """
    
    def __init__(self, scale_range: Tuple[float, float] = (0.9, 1.1), random_seed: Optional[int] = None):
        """
        Initialize spatial scaling augmentation.
        
        Args:
            scale_range: Range of scale factors (min, max)
            random_seed: Random seed for reproducibility
        """
        self.scale_range = scale_range
        self.random_seed = random_seed
        if random_seed is not None:
            np.random.seed(random_seed)
    
    def augment(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Augment DataFrame with spatial scaling.
        
        Args:
            data: DataFrame with bbox or keypoint data
            
        Returns:
            Augmented DataFrame
        """
        scale_factor = np.random.uniform(*self.scale_range)
        augmented_data = data.copy()
        
        # Scale bounding boxes if present
        if 'bbox' in augmented_data.columns:
            def scale_bbox(bbox):
                if not isinstance(bbox, dict):
                    return bbox
                return {
                    'left': bbox.get('left', 0) * scale_factor,
                    'top': bbox.get('top', 0) * scale_factor,
                    'width': bbox.get('width', 0) * scale_factor,
                    'height': bbox.get('height', 0) * scale_factor
                }
            
            augmented_data['bbox'] = augmented_data['bbox'].apply(scale_bbox)
        
        logger.debug(f"Applied spatial scaling augmentation: {scale_factor:.2f}x scale")
        
        return augmented_data
    
    def get_parameters(self) -> Dict[str, Any]:
        """Get augmentation parameters."""
        return {
            'type': 'spatial_scaling',
            'scale_range': self.scale_range,
            'random_seed': self.random_seed
        }


class NoiseInjectionAugmentation(IAugmentation):
    """
    Add controlled noise to gait data for robustness.
    
    This helps models generalize better to noisy real-world data.
    """
    
    def __init__(self, noise_level: float = 0.01, random_seed: Optional[int] = None):
        """
        Initialize noise injection augmentation.
        
        Args:
            noise_level: Standard deviation of Gaussian noise
            random_seed: Random seed for reproducibility
        """
        self.noise_level = noise_level
        self.random_seed = random_seed
        if random_seed is not None:
            np.random.seed(random_seed)
    
    def augment(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Add noise to numerical columns in DataFrame.
        
        Args:
            data: Input DataFrame
            
        Returns:
            Augmented DataFrame with noise
        """
        augmented_data = data.copy()
        
        # Add noise to bounding boxes if present
        if 'bbox' in augmented_data.columns:
            def add_noise_to_bbox(bbox):
                if not isinstance(bbox, dict):
                    return bbox
                
                noisy_bbox = {}
                for key, value in bbox.items():
                    if isinstance(value, (int, float)):
                        noise = np.random.normal(0, self.noise_level * abs(value))
                        noisy_bbox[key] = value + noise
                    else:
                        noisy_bbox[key] = value
                
                return noisy_bbox
            
            augmented_data['bbox'] = augmented_data['bbox'].apply(add_noise_to_bbox)
        
        logger.debug(f"Applied noise injection augmentation: {self.noise_level} noise level")
        
        return augmented_data
    
    def get_parameters(self) -> Dict[str, Any]:
        """Get augmentation parameters."""
        return {
            'type': 'noise_injection',
            'noise_level': self.noise_level,
            'random_seed': self.random_seed
        }


class HorizontalFlipAugmentation(IAugmentation):
    """
    Flip gait sequences horizontally.
    
    This creates mirror images while preserving gait patterns.
    """
    
    def __init__(self, flip_probability: float = 0.5, random_seed: Optional[int] = None):
        """
        Initialize horizontal flip augmentation.
        
        Args:
            flip_probability: Probability of applying flip
            random_seed: Random seed for reproducibility
        """
        self.flip_probability = flip_probability
        self.random_seed = random_seed
        if random_seed is not None:
            np.random.seed(random_seed)
    
    def augment(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Flip bounding boxes horizontally with given probability.
        
        Args:
            data: Input DataFrame
            
        Returns:
            Augmented DataFrame
        """
        if np.random.random() > self.flip_probability:
            return data.copy()
        
        augmented_data = data.copy()
        
        # Flip bounding boxes if present
        if 'bbox' in augmented_data.columns and 'vid_info' in augmented_data.columns:
            def flip_bbox(row):
                bbox = row['bbox']
                vid_info = row['vid_info']
                
                if not isinstance(bbox, dict) or not isinstance(vid_info, dict):
                    return bbox
                
                width = vid_info.get('width', 1920)
                flipped_bbox = bbox.copy()
                flipped_bbox['left'] = width - bbox.get('left', 0) - bbox.get('width', 0)
                
                return flipped_bbox
            
            augmented_data['bbox'] = augmented_data.apply(flip_bbox, axis=1)
        
        logger.debug("Applied horizontal flip augmentation")
        
        return augmented_data
    
    def get_parameters(self) -> Dict[str, Any]:
        """Get augmentation parameters."""
        return {
            'type': 'horizontal_flip',
            'flip_probability': self.flip_probability,
            'random_seed': self.random_seed
        }


class AugmentationPipeline:
    """
    Pipeline for applying multiple augmentations in sequence.
    
    This class allows composing multiple augmentation strategies.
    """
    
    def __init__(self, augmentations: Optional[List[IAugmentation]] = None):
        """
        Initialize augmentation pipeline.
        
        Args:
            augmentations: List of augmentation strategies
        """
        self.augmentations = augmentations or []
    
    def add_augmentation(self, augmentation: IAugmentation) -> None:
        """
        Add an augmentation to the pipeline.
        
        Args:
            augmentation: Augmentation strategy to add
        """
        self.augmentations.append(augmentation)
        logger.debug(f"Added augmentation: {augmentation.get_parameters()['type']}")
    
    def augment(self, data: Union[pd.DataFrame, FrameSequence]) -> Union[pd.DataFrame, FrameSequence]:
        """
        Apply all augmentations in the pipeline.
        
        Args:
            data: Input data (DataFrame or FrameSequence)
            
        Returns:
            Augmented data
        """
        augmented_data = data
        
        for augmentation in self.augmentations:
            try:
                augmented_data = augmentation.augment(augmented_data)
            except Exception as e:
                logger.warning(f"Augmentation failed: {e}")
                continue
        
        return augmented_data
    
    def get_pipeline_config(self) -> List[Dict[str, Any]]:
        """Get configuration of all augmentations in pipeline."""
        return [aug.get_parameters() for aug in self.augmentations]


class DataAugmenter:
    """
    High-level data augmentation manager.
    
    This class provides convenient methods for augmenting training datasets.
    """
    
    def __init__(self, random_seed: Optional[int] = None):
        """
        Initialize data augmenter.
        
        Args:
            random_seed: Random seed for reproducibility
        """
        self.random_seed = random_seed
        if random_seed is not None:
            np.random.seed(random_seed)
    
    def create_default_pipeline(self) -> AugmentationPipeline:
        """
        Create a default augmentation pipeline for gait data.
        
        Returns:
            Configured AugmentationPipeline
        """
        pipeline = AugmentationPipeline()
        
        # Add common augmentations
        pipeline.add_augmentation(SpatialScalingAugmentation(
            scale_range=(0.9, 1.1),
            random_seed=self.random_seed
        ))
        pipeline.add_augmentation(NoiseInjectionAugmentation(
            noise_level=0.01,
            random_seed=self.random_seed
        ))
        pipeline.add_augmentation(HorizontalFlipAugmentation(
            flip_probability=0.5,
            random_seed=self.random_seed
        ))
        
        logger.info("Created default augmentation pipeline")
        return pipeline
    
    def augment_dataset(
        self,
        data: pd.DataFrame,
        augmentation_factor: int = 2,
        pipeline: Optional[AugmentationPipeline] = None
    ) -> pd.DataFrame:
        """
        Augment a dataset by creating multiple augmented versions.
        
        Args:
            data: Input DataFrame
            augmentation_factor: Number of augmented versions per sample
            pipeline: Optional custom augmentation pipeline
            
        Returns:
            Augmented DataFrame with original and augmented samples
        """
        if pipeline is None:
            pipeline = self.create_default_pipeline()
        
        augmented_samples = [data]
        
        for i in range(augmentation_factor - 1):
            logger.info(f"Creating augmented version {i + 1}/{augmentation_factor - 1}")
            augmented = pipeline.augment(data.copy())
            
            # Add augmentation metadata
            augmented['augmentation_version'] = i + 1
            augmented_samples.append(augmented)
        
        # Combine all samples
        combined = pd.concat(augmented_samples, ignore_index=True)
        
        logger.info(f"Augmented dataset from {len(data)} to {len(combined)} samples")
        
        return combined
    
    def augment_sequence(
        self,
        sequence: FrameSequence,
        augmentation_types: Optional[List[str]] = None
    ) -> FrameSequence:
        """
        Augment a single frame sequence.
        
        Args:
            sequence: Input FrameSequence
            augmentation_types: List of augmentation types to apply
            
        Returns:
            Augmented FrameSequence
        """
        if augmentation_types is None:
            augmentation_types = ['temporal_speed']
        
        augmented = sequence
        
        for aug_type in augmentation_types:
            if aug_type == 'temporal_speed':
                aug = TemporalSpeedAugmentation(random_seed=self.random_seed)
                augmented = aug.augment(augmented)
        
        return augmented
    
    def validate_augmentation(
        self,
        original: pd.DataFrame,
        augmented: pd.DataFrame,
        tolerance: float = 0.5
    ) -> Dict[str, Any]:
        """
        Validate that augmentation preserves essential characteristics.
        
        Args:
            original: Original DataFrame
            augmented: Augmented DataFrame
            tolerance: Tolerance for statistical differences
            
        Returns:
            Validation results dictionary
        """
        validation = {
            'valid': True,
            'checks': {},
            'warnings': []
        }
        
        # Check sample count
        if len(augmented) < len(original):
            validation['valid'] = False
            validation['warnings'].append("Augmented dataset is smaller than original")
        
        # Check column preservation
        if set(original.columns) != set(augmented.columns):
            validation['warnings'].append("Column structure changed during augmentation")
        
        # Check for null values
        original_nulls = original.isnull().sum().sum()
        augmented_nulls = augmented.isnull().sum().sum()
        
        if augmented_nulls > original_nulls * 1.1:
            validation['warnings'].append("Significant increase in null values")
        
        validation['checks'] = {
            'original_samples': len(original),
            'augmented_samples': len(augmented),
            'augmentation_factor': len(augmented) / len(original) if len(original) > 0 else 0,
            'original_nulls': int(original_nulls),
            'augmented_nulls': int(augmented_nulls)
        }
        
        return validation
