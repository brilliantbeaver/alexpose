"""
Pose Processing Package

This package contains pose estimation data processing utilities for
our "ambient" fall risk detection system.

@Theodore Mui
Monday, July 28, 2025 12:30:00 AM
"""

from .gait_processor import GaitDataProcessor, GaitSequenceAnalyzer
from .gavd_processor import GAVDDataLoader, GAVDProcessor, PoseDataConverter
from .keypoints import (
    BoundingBoxProcessor,
    KeypointGenerator,
    PoseKeypointExtractor,
)

__all__ = [
    "GaitDataProcessor",
    "GaitSequenceAnalyzer",
    "BoundingBoxProcessor",
    "KeypointGenerator",
    "PoseKeypointExtractor",
    "GAVDDataLoader",
    "PoseDataConverter",
    "GAVDProcessor",
]

__version__ = "1.0.0"
__author__ = "Theodore Mui"
