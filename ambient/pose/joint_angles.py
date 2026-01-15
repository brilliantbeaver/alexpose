"""
Joint angle calculation module for pose estimation data.

This module provides comprehensive joint angle calculation capabilities for gait analysis,
supporting multiple keypoint formats (MediaPipe, COCO, BODY_25) with robust error handling
and validation.

The joint angle calculations use the vector dot-product formula to compute angles between
three points, where the middle point is the vertex of the angle. This method agrees well
with marker-based systems (mean absolute error < 5° for hip/knee/ankle angles).

Author: AlexPose Team
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple
from loguru import logger


@dataclass
class JointAngle:
    """
    Represents a single joint angle measurement.
    
    Attributes:
        joint_name: Name of the joint (e.g., "left_knee", "right_hip")
        angle_degrees: Angle measurement in degrees
        confidence: Confidence score based on keypoint confidences
        frame_index: Frame number in the sequence
        landmark_indices: Tuple of (p1_idx, vertex_idx, p3_idx) used for calculation
    """
    joint_name: str
    angle_degrees: float
    confidence: float
    frame_index: int
    landmark_indices: Tuple[int, int, int] = field(default=(0, 0, 0))
    
    def __post_init__(self):
        """Validate angle and confidence values."""
        if not 0 <= self.angle_degrees <= 180:
            logger.warning(f"Unusual angle value: {self.angle_degrees}° for {self.joint_name}")
        if not 0 <= self.confidence <= 1:
            raise ValueError(f"Confidence must be between 0 and 1, got {self.confidence}")


@dataclass
class FrameJointAngles:
    """
    Joint angles for a single frame.
    
    Attributes:
        frame_index: Frame number in the sequence
        angles: Dictionary mapping joint names to JointAngle objects
        keypoint_format: Format of keypoints used (e.g., "BLAZEPOSE_33", "COCO_17")
        timestamp: Optional timestamp in seconds
    """
    frame_index: int
    angles: Dict[str, JointAngle] = field(default_factory=dict)
    keypoint_format: str = "BLAZEPOSE_33"
    timestamp: Optional[float] = None
    
    def get_angle(self, joint_name: str) -> Optional[float]:
        """Get angle value for a specific joint."""
        angle_obj = self.angles.get(joint_name)
        return angle_obj.angle_degrees if angle_obj else None
    
    def get_confidence(self, joint_name: str) -> Optional[float]:
        """Get confidence for a specific joint angle."""
        angle_obj = self.angles.get(joint_name)
        return angle_obj.confidence if angle_obj else None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary format."""
        return {
            "frame_index": self.frame_index,
            "keypoint_format": self.keypoint_format,
            "timestamp": self.timestamp,
            "angles": {
                name: {
                    "angle_degrees": angle.angle_degrees,
                    "confidence": angle.confidence,
                    "landmark_indices": angle.landmark_indices
                }
                for name, angle in self.angles.items()
            }
        }


@dataclass
class JointAngleSequence:
    """
    Joint angles for an entire sequence of frames.
    
    Attributes:
        frames: List of FrameJointAngles objects
        keypoint_format: Format of keypoints used
        fps: Frames per second of the video
        sequence_id: Optional identifier for the sequence
    """
    frames: List[FrameJointAngles] = field(default_factory=list)
    keypoint_format: str = "BLAZEPOSE_33"
    fps: float = 30.0
    sequence_id: Optional[str] = None
    
    def get_joint_angle_series(self, joint_name: str) -> np.ndarray:
        """
        Get time series of angles for a specific joint.
        
        Args:
            joint_name: Name of the joint
            
        Returns:
            NumPy array of angle values (NaN for missing values)
        """
        angles = []
        for frame in self.frames:
            angle = frame.get_angle(joint_name)
            angles.append(angle if angle is not None else np.nan)
        return np.array(angles)
    
    def get_joint_confidence_series(self, joint_name: str) -> np.ndarray:
        """Get time series of confidence scores for a specific joint."""
        confidences = []
        for frame in self.frames:
            conf = frame.get_confidence(joint_name)
            confidences.append(conf if conf is not None else 0.0)
        return np.array(confidences)
    
    def get_statistics(self, joint_name: str) -> Dict[str, float]:
        """
        Calculate statistics for a joint angle across the sequence.
        
        Args:
            joint_name: Name of the joint
            
        Returns:
            Dictionary with mean, std, min, max, range statistics
        """
        angles = self.get_joint_angle_series(joint_name)
        valid_angles = angles[~np.isnan(angles)]
        
        if len(valid_angles) == 0:
            return {
                "mean": np.nan,
                "std": np.nan,
                "min": np.nan,
                "max": np.nan,
                "range": np.nan,
                "valid_count": 0
            }
        
        return {
            "mean": float(np.mean(valid_angles)),
            "std": float(np.std(valid_angles)),
            "min": float(np.min(valid_angles)),
            "max": float(np.max(valid_angles)),
            "range": float(np.max(valid_angles) - np.min(valid_angles)),
            "valid_count": len(valid_angles)
        }
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary format."""
        return {
            "sequence_id": self.sequence_id,
            "keypoint_format": self.keypoint_format,
            "fps": self.fps,
            "num_frames": len(self.frames),
            "frames": [frame.to_dict() for frame in self.frames]
        }


class JointAngleCalculator:
    """
    Calculates joint angles from pose keypoints using vector dot-product formula.
    
    This calculator supports multiple keypoint formats and provides robust angle
    calculation with confidence weighting based on keypoint detection confidence.
    
    Joint Definitions:
    - Hip angle: shoulder -> hip -> knee
    - Knee angle: hip -> knee -> ankle
    - Ankle angle: knee -> ankle -> foot_index (or vertical reference)
    """
    
    # Keypoint format mappings
    KEYPOINT_MAPPINGS = {
        "BLAZEPOSE_33": {
            "left_shoulder": 11, "right_shoulder": 12,
            "left_hip": 23, "right_hip": 24,
            "left_knee": 25, "right_knee": 26,
            "left_ankle": 27, "right_ankle": 28,
            "left_foot_index": 31, "right_foot_index": 32
        },
        "COCO_17": {
            "left_shoulder": 5, "right_shoulder": 6,
            "left_hip": 11, "right_hip": 12,
            "left_knee": 13, "right_knee": 14,
            "left_ankle": 15, "right_ankle": 16
        },
        "BODY_25": {
            "left_shoulder": 5, "right_shoulder": 2,
            "left_hip": 12, "right_hip": 9,
            "left_knee": 13, "right_knee": 10,
            "left_ankle": 14, "right_ankle": 11,
            "left_big_toe": 19, "right_big_toe": 22
        }
    }
    
    def __init__(self, keypoint_format: str = "BLAZEPOSE_33", confidence_threshold: float = 0.3):
        """
        Initialize joint angle calculator.
        
        Args:
            keypoint_format: Format of keypoints ("BLAZEPOSE_33", "COCO_17", "BODY_25")
            confidence_threshold: Minimum confidence for valid angle calculation
        """
        self.keypoint_format = keypoint_format
        self.confidence_threshold = confidence_threshold
        
        if keypoint_format not in self.KEYPOINT_MAPPINGS:
            raise ValueError(
                f"Unsupported keypoint format: {keypoint_format}. "
                f"Supported formats: {list(self.KEYPOINT_MAPPINGS.keys())}"
            )
        
        self.mapping = self.KEYPOINT_MAPPINGS[keypoint_format]
        logger.info(f"JointAngleCalculator initialized for {keypoint_format} format")
    
    def calculate_angle(
        self,
        p1: np.ndarray,
        p2: np.ndarray,
        p3: np.ndarray,
        conf1: float,
        conf2: float,
        conf3: float
    ) -> Tuple[float, float]:
        """
        Calculate angle between three points using dot product formula.
        
        Args:
            p1: First point coordinates [x, y]
            p2: Vertex point coordinates [x, y]
            p3: Third point coordinates [x, y]
            conf1, conf2, conf3: Confidence scores for each point
            
        Returns:
            Tuple of (angle_degrees, combined_confidence)
        """
        # Calculate vectors
        v1 = p1 - p2
        v2 = p3 - p2
        
        # Calculate vector magnitudes
        mag1 = np.linalg.norm(v1)
        mag2 = np.linalg.norm(v2)
        
        # Handle degenerate cases
        if mag1 < 1e-6 or mag2 < 1e-6:
            return np.nan, 0.0
        
        # Calculate angle using dot product
        cos_angle = np.dot(v1, v2) / (mag1 * mag2)
        cos_angle = np.clip(cos_angle, -1.0, 1.0)  # Ensure valid range
        angle_rad = np.arccos(cos_angle)
        angle_deg = np.degrees(angle_rad)
        
        # Combined confidence (geometric mean)
        combined_conf = (conf1 * conf2 * conf3) ** (1/3)
        
        return float(angle_deg), float(combined_conf)
    
    def calculate_frame_angles(
        self,
        keypoints: List[Dict[str, Any]],
        frame_index: int = 0,
        timestamp: Optional[float] = None
    ) -> FrameJointAngles:
        """
        Calculate all joint angles for a single frame.
        
        Args:
            keypoints: List of keypoint dictionaries with 'x', 'y', 'confidence'
            frame_index: Frame number in sequence
            timestamp: Optional timestamp in seconds
            
        Returns:
            FrameJointAngles object containing all calculated angles
        """
        frame_angles = FrameJointAngles(
            frame_index=frame_index,
            keypoint_format=self.keypoint_format,
            timestamp=timestamp
        )
        
        # Define joint calculations
        joint_definitions = self._get_joint_definitions()
        
        for joint_name, (p1_name, p2_name, p3_name) in joint_definitions.items():
            try:
                # Get landmark indices
                p1_idx = self.mapping.get(p1_name)
                p2_idx = self.mapping.get(p2_name)
                p3_idx = self.mapping.get(p3_name)
                
                if p1_idx is None or p2_idx is None or p3_idx is None:
                    continue
                
                # Check if indices are valid for this keypoint array
                if max(p1_idx, p2_idx, p3_idx) >= len(keypoints):
                    continue
                
                # Extract keypoint data
                kp1 = keypoints[p1_idx]
                kp2 = keypoints[p2_idx]
                kp3 = keypoints[p3_idx]
                
                p1 = np.array([kp1['x'], kp1['y']])
                p2 = np.array([kp2['x'], kp2['y']])
                p3 = np.array([kp3['x'], kp3['y']])
                
                conf1 = kp1.get('confidence', 0.0)
                conf2 = kp2.get('confidence', 0.0)
                conf3 = kp3.get('confidence', 0.0)
                
                # Calculate angle
                angle_deg, combined_conf = self.calculate_angle(p1, p2, p3, conf1, conf2, conf3)
                
                # Only add if confidence meets threshold
                if combined_conf >= self.confidence_threshold and not np.isnan(angle_deg):
                    frame_angles.angles[joint_name] = JointAngle(
                        joint_name=joint_name,
                        angle_degrees=angle_deg,
                        confidence=combined_conf,
                        frame_index=frame_index,
                        landmark_indices=(p1_idx, p2_idx, p3_idx)
                    )
            
            except Exception as e:
                logger.debug(f"Failed to calculate {joint_name} angle: {e}")
                continue
        
        return frame_angles
    
    def calculate_sequence_angles(
        self,
        keypoints_array: List[List[Dict[str, Any]]],
        fps: float = 30.0,
        sequence_id: Optional[str] = None
    ) -> JointAngleSequence:
        """
        Calculate joint angles for an entire sequence of frames.
        
        Args:
            keypoints_array: List of frames, each containing list of keypoint dicts
            fps: Frames per second of the video
            sequence_id: Optional identifier for the sequence
            
        Returns:
            JointAngleSequence object containing angles for all frames
        """
        sequence = JointAngleSequence(
            keypoint_format=self.keypoint_format,
            fps=fps,
            sequence_id=sequence_id
        )
        
        for frame_idx, keypoints in enumerate(keypoints_array):
            timestamp = frame_idx / fps if fps > 0 else None
            frame_angles = self.calculate_frame_angles(keypoints, frame_idx, timestamp)
            sequence.frames.append(frame_angles)
        
        logger.info(
            f"Calculated joint angles for {len(keypoints_array)} frames "
            f"({self.keypoint_format} format)"
        )
        
        return sequence
    
    def _get_joint_definitions(self) -> Dict[str, Tuple[str, str, str]]:
        """
        Get joint angle definitions for the current keypoint format.
        
        Returns:
            Dictionary mapping joint names to (point1, vertex, point3) tuples
        """
        # Common definitions for all formats
        definitions = {
            # Left side
            "left_hip": ("left_shoulder", "left_hip", "left_knee"),
            "left_knee": ("left_hip", "left_knee", "left_ankle"),
            # Right side
            "right_hip": ("right_shoulder", "right_hip", "right_knee"),
            "right_knee": ("right_hip", "right_knee", "right_ankle"),
        }
        
        # Format-specific ankle definitions
        if self.keypoint_format == "BLAZEPOSE_33":
            definitions["left_ankle"] = ("left_knee", "left_ankle", "left_foot_index")
            definitions["right_ankle"] = ("right_knee", "right_ankle", "right_foot_index")
        elif self.keypoint_format == "BODY_25":
            definitions["left_ankle"] = ("left_knee", "left_ankle", "left_big_toe")
            definitions["right_ankle"] = ("right_knee", "right_ankle", "right_big_toe")
        # COCO_17 doesn't have foot landmarks, so ankle angles are omitted
        
        return definitions


def get_joint_angles(
    keypoints_array: List[List[Dict[str, Any]]],
    keypoint_format: str = "BLAZEPOSE_33",
    fps: float = 30.0,
    confidence_threshold: float = 0.3,
    sequence_id: Optional[str] = None
) -> JointAngleSequence:
    """
    Compute joint angles for each frame in a keypoint sequence.
    
    This is the main entry point for joint angle calculation. It computes hip, knee,
    and ankle angles using the vector dot-product formula:
    - Hip angle: shoulder -> hip -> knee
    - Knee angle: hip -> knee -> ankle  
    - Ankle angle: knee -> ankle -> foot
    
    MediaPipe coordinates can be converted from normalized positions to pixels;
    joint angles measured this way agree well with marker-based systems
    (mean absolute error < 5° for hip/knee/ankle angles).
    
    Args:
        keypoints_array: List of frames, each containing list of keypoint dictionaries
                        with 'x', 'y', 'confidence' fields
        keypoint_format: Format of keypoints ("BLAZEPOSE_33", "COCO_17", "BODY_25")
        fps: Frames per second of the video
        confidence_threshold: Minimum confidence for valid angle calculation
        sequence_id: Optional identifier for the sequence
        
    Returns:
        JointAngleSequence object containing angles for all frames with statistics
        
    Example:
        >>> keypoints = [
        ...     [{"x": 100, "y": 200, "confidence": 0.9}, ...],  # Frame 0
        ...     [{"x": 105, "y": 205, "confidence": 0.85}, ...],  # Frame 1
        ... ]
        >>> angles = get_joint_angles(keypoints, keypoint_format="BLAZEPOSE_33")
        >>> left_knee_angles = angles.get_joint_angle_series("left_knee")
        >>> stats = angles.get_statistics("left_knee")
        >>> print(f"Mean knee angle: {stats['mean']:.1f}°")
    """
    calculator = JointAngleCalculator(
        keypoint_format=keypoint_format,
        confidence_threshold=confidence_threshold
    )
    
    return calculator.calculate_sequence_angles(
        keypoints_array=keypoints_array,
        fps=fps,
        sequence_id=sequence_id
    )
