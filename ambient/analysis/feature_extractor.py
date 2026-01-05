"""
Feature extraction module for gait analysis.

This module provides comprehensive feature extraction capabilities for gait analysis,
including joint angles, velocities, accelerations, and temporal features.

Author: AlexPose Team
"""

import numpy as np
from typing import Any, Dict, List, Optional, Tuple
from loguru import logger

from ambient.core.frame import FrameSequence


class FeatureExtractor:
    """
    Comprehensive feature extractor for gait analysis.
    
    This class extracts various features from pose sequences including:
    - Joint angles and angular velocities
    - Joint positions and velocities
    - Stride characteristics
    - Temporal features
    - Symmetry measures
    """
    
    def __init__(
        self,
        keypoint_format: str = "COCO_17",
        fps: float = 30.0,
        smoothing_window: int = 5
    ):
        """
        Initialize feature extractor.
        
        Args:
            keypoint_format: Format of keypoints (COCO_17, BODY_25, etc.)
            fps: Frames per second of the video
            smoothing_window: Window size for smoothing calculations
        """
        self.keypoint_format = keypoint_format
        self.fps = fps
        self.smoothing_window = smoothing_window
        
        # Define keypoint mappings for different formats
        self.keypoint_mappings = self._get_keypoint_mappings()
        
        logger.info(f"Feature extractor initialized for {keypoint_format} format")
    
    def _get_keypoint_mappings(self) -> Dict[str, Dict[str, int]]:
        """Get keypoint mappings for different formats."""
        mappings = {
            "COCO_17": {
                "nose": 0, "left_eye": 1, "right_eye": 2, "left_ear": 3, "right_ear": 4,
                "left_shoulder": 5, "right_shoulder": 6, "left_elbow": 7, "right_elbow": 8,
                "left_wrist": 9, "right_wrist": 10, "left_hip": 11, "right_hip": 12,
                "left_knee": 13, "right_knee": 14, "left_ankle": 15, "right_ankle": 16
            },
            "BODY_25": {
                "nose": 0, "neck": 1, "right_shoulder": 2, "right_elbow": 3, "right_wrist": 4,
                "left_shoulder": 5, "left_elbow": 6, "left_wrist": 7, "mid_hip": 8,
                "right_hip": 9, "right_knee": 10, "right_ankle": 11, "left_hip": 12,
                "left_knee": 13, "left_ankle": 14, "right_eye": 15, "left_eye": 16,
                "right_ear": 17, "left_ear": 18, "left_big_toe": 19, "left_small_toe": 20,
                "left_heel": 21, "right_big_toe": 22, "right_small_toe": 23, "right_heel": 24
            },
            "BLAZEPOSE_33": {
                "nose": 0, "left_eye_inner": 1, "left_eye": 2, "left_eye_outer": 3,
                "right_eye_inner": 4, "right_eye": 5, "right_eye_outer": 6, "left_ear": 7,
                "right_ear": 8, "mouth_left": 9, "mouth_right": 10, "left_shoulder": 11,
                "right_shoulder": 12, "left_elbow": 13, "right_elbow": 14, "left_wrist": 15,
                "right_wrist": 16, "left_pinky": 17, "right_pinky": 18, "left_index": 19,
                "right_index": 20, "left_thumb": 21, "right_thumb": 22, "left_hip": 23,
                "right_hip": 24, "left_knee": 25, "right_knee": 26, "left_ankle": 27,
                "right_ankle": 28, "left_heel": 29, "right_heel": 30, "left_foot_index": 31,
                "right_foot_index": 32
            }
        }
        
        return mappings
    
    def extract_features(self, pose_sequence: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Extract comprehensive features from pose sequence.
        
        Args:
            pose_sequence: List of pose estimation results
            
        Returns:
            Dictionary containing extracted features
        """
        if not pose_sequence:
            return {}
        
        # Convert pose sequence to numpy arrays
        keypoints_array = self._poses_to_array(pose_sequence)
        
        if keypoints_array is None or keypoints_array.size == 0:
            return {}
        
        features = {}
        
        try:
            # Extract basic kinematic features
            features.update(self._extract_kinematic_features(keypoints_array))
            
            # Extract joint angle features
            features.update(self._extract_joint_angle_features(keypoints_array))
            
            # Extract temporal features
            features.update(self._extract_temporal_features(keypoints_array))
            
            # Extract stride features
            features.update(self._extract_stride_features(keypoints_array))
            
            # Extract symmetry features
            features.update(self._extract_symmetry_features(keypoints_array))
            
            # Extract stability features
            features.update(self._extract_stability_features(keypoints_array))
            
        except Exception as e:
            logger.error(f"Feature extraction failed: {e}")
            features["extraction_error"] = str(e)
        
        return features
    
    def _poses_to_array(self, pose_sequence: List[Dict[str, Any]]) -> Optional[np.ndarray]:
        """Convert pose sequence to numpy array."""
        if not pose_sequence:
            return None
        
        # Get keypoints from first valid pose to determine structure
        keypoints_data = None
        for pose in pose_sequence:
            if pose.get("keypoints"):
                keypoints_data = pose["keypoints"]
                break
        
        if not keypoints_data:
            return None
        
        num_keypoints = len(keypoints_data)
        num_frames = len(pose_sequence)
        
        # Create array: [frames, keypoints, (x, y, confidence)]
        keypoints_array = np.zeros((num_frames, num_keypoints, 3))
        
        for frame_idx, pose in enumerate(pose_sequence):
            keypoints = pose.get("keypoints", [])
            for kp_idx, kp in enumerate(keypoints):
                if kp_idx < num_keypoints:
                    keypoints_array[frame_idx, kp_idx, 0] = kp.get("x", 0)
                    keypoints_array[frame_idx, kp_idx, 1] = kp.get("y", 0)
                    keypoints_array[frame_idx, kp_idx, 2] = kp.get("confidence", 0)
        
        return keypoints_array
    
    def _extract_kinematic_features(self, keypoints: np.ndarray) -> Dict[str, Any]:
        """Extract kinematic features (positions, velocities, accelerations)."""
        features = {}
        
        # Calculate velocities (first derivative)
        velocities = np.diff(keypoints[:, :, :2], axis=0)  # Only x, y coordinates
        
        # Calculate accelerations (second derivative)
        accelerations = np.diff(velocities, axis=0)
        
        # Calculate speeds (magnitude of velocity)
        speeds = np.linalg.norm(velocities, axis=2)
        
        # Statistical features for velocities
        features["velocity_mean"] = np.mean(speeds)
        features["velocity_std"] = np.std(speeds)
        features["velocity_max"] = np.max(speeds)
        features["velocity_min"] = np.min(speeds)
        
        # Statistical features for accelerations
        accel_magnitudes = np.linalg.norm(accelerations, axis=2)
        features["acceleration_mean"] = np.mean(accel_magnitudes)
        features["acceleration_std"] = np.std(accel_magnitudes)
        features["acceleration_max"] = np.max(accel_magnitudes)
        
        # Movement smoothness (jerk)
        if len(accelerations) > 1:
            jerk = np.diff(accelerations, axis=0)
            jerk_magnitudes = np.linalg.norm(jerk, axis=2)
            features["jerk_mean"] = np.mean(jerk_magnitudes)
            features["jerk_std"] = np.std(jerk_magnitudes)
        
        return features
    
    def _extract_joint_angle_features(self, keypoints: np.ndarray) -> Dict[str, Any]:
        """Extract joint angle features."""
        features = {}
        mapping = self.keypoint_mappings.get(self.keypoint_format, {})
        
        if not mapping:
            return features
        
        try:
            # Define joint angle calculations based on keypoint format
            if self.keypoint_format == "COCO_17":
                angles = self._calculate_coco_joint_angles(keypoints, mapping)
            elif self.keypoint_format == "BODY_25":
                angles = self._calculate_body25_joint_angles(keypoints, mapping)
            else:
                # Generic angle calculation
                angles = self._calculate_generic_joint_angles(keypoints, mapping)
            
            # Statistical features for each angle
            for angle_name, angle_values in angles.items():
                if len(angle_values) > 0:
                    features[f"{angle_name}_mean"] = np.mean(angle_values)
                    features[f"{angle_name}_std"] = np.std(angle_values)
                    features[f"{angle_name}_range"] = np.max(angle_values) - np.min(angle_values)
                    features[f"{angle_name}_max"] = np.max(angle_values)
                    features[f"{angle_name}_min"] = np.min(angle_values)
        
        except Exception as e:
            logger.warning(f"Joint angle extraction failed: {e}")
        
        return features
    
    def _calculate_coco_joint_angles(self, keypoints: np.ndarray, mapping: Dict[str, int]) -> Dict[str, np.ndarray]:
        """Calculate joint angles for COCO format."""
        angles = {}
        
        # Knee angles
        if all(k in mapping for k in ["left_hip", "left_knee", "left_ankle"]):
            angles["left_knee"] = self._calculate_angle_sequence(
                keypoints, mapping["left_hip"], mapping["left_knee"], mapping["left_ankle"]
            )
        
        if all(k in mapping for k in ["right_hip", "right_knee", "right_ankle"]):
            angles["right_knee"] = self._calculate_angle_sequence(
                keypoints, mapping["right_hip"], mapping["right_knee"], mapping["right_ankle"]
            )
        
        # Hip angles (using shoulder as reference)
        if all(k in mapping for k in ["left_shoulder", "left_hip", "left_knee"]):
            angles["left_hip"] = self._calculate_angle_sequence(
                keypoints, mapping["left_shoulder"], mapping["left_hip"], mapping["left_knee"]
            )
        
        if all(k in mapping for k in ["right_shoulder", "right_hip", "right_knee"]):
            angles["right_hip"] = self._calculate_angle_sequence(
                keypoints, mapping["right_shoulder"], mapping["right_hip"], mapping["right_knee"]
            )
        
        # Ankle angles (using knee and toe if available)
        if all(k in mapping for k in ["left_knee", "left_ankle"]):
            # Approximate ankle angle using vertical reference
            angles["left_ankle"] = self._calculate_ankle_angle_sequence(
                keypoints, mapping["left_knee"], mapping["left_ankle"]
            )
        
        if all(k in mapping for k in ["right_knee", "right_ankle"]):
            angles["right_ankle"] = self._calculate_ankle_angle_sequence(
                keypoints, mapping["right_knee"], mapping["right_ankle"]
            )
        
        return angles
    
    def _calculate_body25_joint_angles(self, keypoints: np.ndarray, mapping: Dict[str, int]) -> Dict[str, np.ndarray]:
        """Calculate joint angles for BODY_25 format."""
        angles = {}
        
        # Similar to COCO but with more detailed foot information
        angles.update(self._calculate_coco_joint_angles(keypoints, mapping))
        
        # Additional foot angles if available
        if all(k in mapping for k in ["left_ankle", "left_heel", "left_big_toe"]):
            angles["left_foot"] = self._calculate_angle_sequence(
                keypoints, mapping["left_ankle"], mapping["left_heel"], mapping["left_big_toe"]
            )
        
        if all(k in mapping for k in ["right_ankle", "right_heel", "right_big_toe"]):
            angles["right_foot"] = self._calculate_angle_sequence(
                keypoints, mapping["right_ankle"], mapping["right_heel"], mapping["right_big_toe"]
            )
        
        return angles
    
    def _calculate_generic_joint_angles(self, keypoints: np.ndarray, mapping: Dict[str, int]) -> Dict[str, np.ndarray]:
        """Calculate generic joint angles."""
        # Fallback to basic angle calculations
        return {}
    
    def _calculate_angle_sequence(self, keypoints: np.ndarray, p1_idx: int, p2_idx: int, p3_idx: int) -> np.ndarray:
        """Calculate angle sequence between three points."""
        angles = []
        
        for frame in keypoints:
            p1 = frame[p1_idx, :2]  # x, y coordinates
            p2 = frame[p2_idx, :2]
            p3 = frame[p3_idx, :2]
            
            # Check if points are valid (confidence > 0)
            if (frame[p1_idx, 2] > 0 and frame[p2_idx, 2] > 0 and frame[p3_idx, 2] > 0):
                angle = self._calculate_angle(p1, p2, p3)
                angles.append(angle)
        
        return np.array(angles)
    
    def _calculate_ankle_angle_sequence(self, keypoints: np.ndarray, knee_idx: int, ankle_idx: int) -> np.ndarray:
        """Calculate ankle angle sequence using vertical reference."""
        angles = []
        
        for frame in keypoints:
            knee = frame[knee_idx, :2]
            ankle = frame[ankle_idx, :2]
            
            if frame[knee_idx, 2] > 0 and frame[ankle_idx, 2] > 0:
                # Create vertical reference point below ankle
                vertical_ref = ankle + np.array([0, 50])  # 50 pixels below
                angle = self._calculate_angle(knee, ankle, vertical_ref)
                angles.append(angle)
        
        return np.array(angles)
    
    def _calculate_angle(self, p1: np.ndarray, p2: np.ndarray, p3: np.ndarray) -> float:
        """Calculate angle between three points (p2 is the vertex)."""
        v1 = p1 - p2
        v2 = p3 - p2
        
        # Calculate angle using dot product
        cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
        cos_angle = np.clip(cos_angle, -1, 1)  # Ensure valid range
        angle = np.arccos(cos_angle)
        
        return np.degrees(angle)
    
    def _extract_temporal_features(self, keypoints: np.ndarray) -> Dict[str, Any]:
        """Extract temporal features."""
        features = {}
        
        num_frames = keypoints.shape[0]
        duration = num_frames / self.fps
        
        features["sequence_length"] = num_frames
        features["duration_seconds"] = duration
        features["fps"] = self.fps
        
        # Calculate movement frequency using FFT
        try:
            # Use center of mass movement for frequency analysis
            center_of_mass = np.mean(keypoints[:, :, :2], axis=1)  # Average across keypoints
            com_movement = np.linalg.norm(np.diff(center_of_mass, axis=0), axis=1)
            
            if len(com_movement) > 10:  # Need sufficient data for FFT
                fft = np.fft.fft(com_movement)
                freqs = np.fft.fftfreq(len(com_movement), 1/self.fps)
                
                # Find dominant frequency
                dominant_freq_idx = np.argmax(np.abs(fft[1:len(fft)//2])) + 1
                features["dominant_frequency"] = abs(freqs[dominant_freq_idx])
                
                # Calculate cadence (steps per minute) - approximate
                features["estimated_cadence"] = features["dominant_frequency"] * 60 * 2  # *2 for both legs
        
        except Exception as e:
            logger.warning(f"Temporal feature extraction failed: {e}")
        
        return features
    
    def _extract_stride_features(self, keypoints: np.ndarray) -> Dict[str, Any]:
        """Extract stride-related features."""
        features = {}
        mapping = self.keypoint_mappings.get(self.keypoint_format, {})
        
        if not mapping:
            return features
        
        try:
            # Extract ankle positions for stride analysis
            left_ankle_idx = mapping.get("left_ankle")
            right_ankle_idx = mapping.get("right_ankle")
            
            if left_ankle_idx is not None and right_ankle_idx is not None:
                left_ankle_pos = keypoints[:, left_ankle_idx, :2]
                right_ankle_pos = keypoints[:, right_ankle_idx, :2]
                
                # Calculate stride length (approximate)
                left_ankle_movement = np.diff(left_ankle_pos, axis=0)
                right_ankle_movement = np.diff(right_ankle_pos, axis=0)
                
                left_distances = np.linalg.norm(left_ankle_movement, axis=1)
                right_distances = np.linalg.norm(right_ankle_movement, axis=1)
                
                features["left_ankle_total_distance"] = np.sum(left_distances)
                features["right_ankle_total_distance"] = np.sum(right_distances)
                features["ankle_distance_asymmetry"] = abs(
                    features["left_ankle_total_distance"] - features["right_ankle_total_distance"]
                )
                
                # Step width (distance between ankles)
                ankle_distances = np.linalg.norm(left_ankle_pos - right_ankle_pos, axis=1)
                features["step_width_mean"] = np.mean(ankle_distances)
                features["step_width_std"] = np.std(ankle_distances)
                features["step_width_range"] = np.max(ankle_distances) - np.min(ankle_distances)
        
        except Exception as e:
            logger.warning(f"Stride feature extraction failed: {e}")
        
        return features
    
    def _extract_symmetry_features(self, keypoints: np.ndarray) -> Dict[str, Any]:
        """Extract left-right symmetry features."""
        features = {}
        mapping = self.keypoint_mappings.get(self.keypoint_format, {})
        
        if not mapping:
            return features
        
        # Define left-right pairs
        pairs = [
            ("left_shoulder", "right_shoulder"),
            ("left_elbow", "right_elbow"),
            ("left_wrist", "right_wrist"),
            ("left_hip", "right_hip"),
            ("left_knee", "right_knee"),
            ("left_ankle", "right_ankle")
        ]
        
        try:
            for left_name, right_name in pairs:
                left_idx = mapping.get(left_name)
                right_idx = mapping.get(right_name)
                
                if left_idx is not None and right_idx is not None:
                    left_pos = keypoints[:, left_idx, :2]
                    right_pos = keypoints[:, right_idx, :2]
                    
                    # Calculate movement symmetry
                    left_movement = np.diff(left_pos, axis=0)
                    right_movement = np.diff(right_pos, axis=0)
                    
                    left_speeds = np.linalg.norm(left_movement, axis=1)
                    right_speeds = np.linalg.norm(right_movement, axis=1)
                    
                    # Symmetry index (0 = perfect symmetry, higher = more asymmetric)
                    symmetry_index = np.mean(np.abs(left_speeds - right_speeds) / (left_speeds + right_speeds + 1e-8))
                    features[f"{left_name.replace('left_', '')}_symmetry_index"] = symmetry_index
        
        except Exception as e:
            logger.warning(f"Symmetry feature extraction failed: {e}")
        
        return features
    
    def _extract_stability_features(self, keypoints: np.ndarray) -> Dict[str, Any]:
        """Extract stability and balance features."""
        features = {}
        
        try:
            # Calculate center of mass
            center_of_mass = np.mean(keypoints[:, :, :2], axis=1)
            
            # Center of mass movement
            com_movement = np.diff(center_of_mass, axis=0)
            com_speeds = np.linalg.norm(com_movement, axis=1)
            
            features["com_movement_mean"] = np.mean(com_speeds)
            features["com_movement_std"] = np.std(com_speeds)
            features["com_stability_index"] = np.std(com_speeds) / (np.mean(com_speeds) + 1e-8)
            
            # Postural sway (if relatively stationary)
            if np.mean(com_speeds) < 5.0:  # Threshold for stationary
                sway_area = self._calculate_sway_area(center_of_mass)
                features["postural_sway_area"] = sway_area
        
        except Exception as e:
            logger.warning(f"Stability feature extraction failed: {e}")
        
        return features
    
    def _calculate_sway_area(self, center_of_mass: np.ndarray) -> float:
        """Calculate postural sway area using convex hull."""
        try:
            from scipy.spatial import ConvexHull
            hull = ConvexHull(center_of_mass)
            return hull.volume  # In 2D, volume is actually area
        except ImportError:
            # Fallback: use bounding box area
            x_range = np.max(center_of_mass[:, 0]) - np.min(center_of_mass[:, 0])
            y_range = np.max(center_of_mass[:, 1]) - np.min(center_of_mass[:, 1])
            return x_range * y_range
        except Exception:
            return 0.0