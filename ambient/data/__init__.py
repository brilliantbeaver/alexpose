"""
Data Management Package

This package provides comprehensive data management capabilities for the AlexPose system,
including training data management, augmentation, and versioning.

Modules:
- training_manager: Training dataset management and organization
- augmentation: Data augmentation for model training
- versioning: Dataset versioning and provenance tracking
"""

from ambient.data.augmentation import (
    AugmentationPipeline,
    DataAugmenter,
    HorizontalFlipAugmentation,
    IAugmentation,
    NoiseInjectionAugmentation,
    SpatialScalingAugmentation,
    TemporalSpeedAugmentation,
)
from ambient.data.training_manager import (
    GAVDDatasetLoader,
    IDatasetLoader,
    TrainingDataManager,
)
from ambient.data.versioning import (
    DatasetVersion,
    DatasetVersionManager,
    ProvenanceRecord,
)

__all__ = [
    # Training Manager
    "TrainingDataManager",
    "IDatasetLoader",
    "GAVDDatasetLoader",
    # Augmentation
    "IAugmentation",
    "TemporalSpeedAugmentation",
    "SpatialScalingAugmentation",
    "NoiseInjectionAugmentation",
    "HorizontalFlipAugmentation",
    "AugmentationPipeline",
    "DataAugmenter",
    # Versioning
    "DatasetVersion",
    "ProvenanceRecord",
    "DatasetVersionManager",
]
