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
    create_best_pose_estimator,
    get_pose_estimator
)

from ambient.pose.joint_angles import (
    JointAngle,
    FrameJointAngles,
    JointAngleSequence,
    JointAngleCalculator,
    get_joint_angles
)

from ambient.pose.keypoints import (
    BoundingBoxProcessor,
    KeypointGenerator,
    PoseKeypointExtractor
)

from ambient.pose.base_estimator import (
    PoseEstimator,
    Keypoint
)

try:
    from ambient.pose.mediapipe_estimator import MediaPipeEstimator, MEDIAPIPE_AVAILABLE
except ImportError:
    MediaPipeEstimator = None
    MEDIAPIPE_AVAILABLE = False

try:
    from ambient.pose.openpose_estimator import OpenPoseEstimator
except ImportError:
    OpenPoseEstimator = None

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
    "get_pose_estimator",
    "PoseEstimator",
    "Keypoint",
    "MediaPipeEstimator",
    "MEDIAPIPE_AVAILABLE",
    "OpenPoseEstimator",
    "UltralyticsEstimator",
    "AlphaPoseEstimator",
    "JointAngle",
    "FrameJointAngles",
    "JointAngleSequence",
    "JointAngleCalculator",
    "get_joint_angles",
    "BoundingBoxProcessor",
    "KeypointGenerator",
    "PoseKeypointExtractor"
]