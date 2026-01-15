"""
Pose Estimator Module - Unified Interface for Pose Estimation

This module provides the main interface for pose estimation in the AlexPose system.
It consolidates base classes, implementations, and factory functions.

Author: AlexPose Team
"""

from typing import Optional
from pathlib import Path
from loguru import logger

# Import base classes
from ambient.pose.base_estimator import PoseEstimator, Keypoint

# Import implementations
from ambient.pose.mediapipe_estimator import MediaPipeEstimator, MEDIAPIPE_AVAILABLE
from ambient.pose.openpose_estimator import OpenPoseEstimator

# Import factory function
from ambient.pose.factory import (
    get_pose_estimator,
    create_pose_estimator,
    get_pose_estimator_factory,
    PoseEstimatorFactory
)

# Try to import optional estimators
try:
    from ambient.pose.ultralytics_estimator import UltralyticsEstimator
    ULTRALYTICS_AVAILABLE = True
except ImportError:
    UltralyticsEstimator = None
    ULTRALYTICS_AVAILABLE = False

try:
    from ambient.pose.alphapose_estimator import AlphaPoseEstimator
    ALPHAPOSE_AVAILABLE = True
except ImportError:
    AlphaPoseEstimator = None
    ALPHAPOSE_AVAILABLE = False


# Export all public interfaces
__all__ = [
    # Base classes
    "PoseEstimator",
    "Keypoint",
    
    # Implementations
    "MediaPipeEstimator",
    "OpenPoseEstimator",
    "UltralyticsEstimator",
    "AlphaPoseEstimator",
    
    # Factory
    "get_pose_estimator",
    "create_pose_estimator",
    "get_pose_estimator_factory",
    "PoseEstimatorFactory",
    
    # Availability flags
    "MEDIAPIPE_AVAILABLE",
    "ULTRALYTICS_AVAILABLE",
    "ALPHAPOSE_AVAILABLE",
]


def get_available_estimators() -> list:
    """
    Get list of available pose estimators.
    
    Returns:
        List of available estimator names
    """
    available = []
    
    if MEDIAPIPE_AVAILABLE:
        available.append("mediapipe")
    
    if ULTRALYTICS_AVAILABLE:
        available.append("ultralytics")
    
    if ALPHAPOSE_AVAILABLE:
        available.append("alphapose")
    
    # OpenPose is always listed but may not be functional
    available.append("openpose")
    
    return available


def print_available_estimators():
    """Print information about available pose estimators."""
    print("Available Pose Estimators:")
    print("-" * 50)
    
    estimators = {
        "MediaPipe": MEDIAPIPE_AVAILABLE,
        "OpenPose": False,  # Not yet implemented
        "Ultralytics YOLO": ULTRALYTICS_AVAILABLE,
        "AlphaPose": ALPHAPOSE_AVAILABLE,
    }
    
    for name, available in estimators.items():
        status = "✓ Available" if available else "✗ Not available"
        print(f"{name:20} {status}")
    
    print("-" * 50)
    print(f"Total available: {sum(estimators.values())}/{len(estimators)}")
