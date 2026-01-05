"""
Pose estimation package for the Ambient system.

This package provides a unified interface for multiple pose estimation
frameworks including OpenPose, MediaPipe, Ultralytics YOLO, and AlphaPose.

Author: AlexPose Team
"""

from ambient.pose.factory import (
    PoseEstimatorFactory,
    get_pose_estimator_factory,
    create_pose_estimator,
    get_available_pose_estimators,
    create_best_pose_estimator
)

try:
    from ambient.pose.ultralytics_estimator import UltralyticsEstimator
except ImportError:
    UltralyticsEstimator = None

try:
    from ambient.pose.alphapose_estimator import AlphaPoseEstimator
except ImportError:
    AlphaPoseEstimator = None

__all__ = [
    "PoseEstimatorFactory",
    "get_pose_estimator_factory",
    "create_pose_estimator",
    "get_available_pose_estimators",
    "create_best_pose_estimator",
    "UltralyticsEstimator",
    "AlphaPoseEstimator"
]