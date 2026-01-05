"""
Symmetry analysis module for gait analysis.

This module provides comprehensive left-right symmetry analysis
for identifying gait asymmetries and abnormalities.

Author: AlexPose Team
"""

import numpy as np
from typing import Any, Dict, List, Optional, Tuple
from loguru import logger

from ambient.core.frame import FrameSequence


class SymmetryAnalyzer:
    """
    Comprehensive symmetry analyzer for gait analysis.
    
    This class analyzes left-right symmetry in gait patterns using
    various metrics and statistical measures to identify asymmetries
    that may indicate pathological conditions.
    """
    
    def __init__(
        self,
        keypoint_format: str = "COCO_17",
        symmetry_threshold: float = 0.1,  # 10% asymmetry threshold
        confidence_threshold: float = 0.5
    ):
        """
        Initialize symmetry analyzer.
        
        Args:
            keypoint_format: Format of keypoints (COCO_17, BODY_25, etc.)
            symmetry_threshold: Threshold for considering asymmetry significant
            confidence_threshold: Minimum confidence for keypoint inclusion
        """
        self.keypoint_format = keypoint_format
        self.symmetry_threshold = symmetry_threshold
        self.confidence_threshold = confidence_threshold
        
        # Define keypoint mappings and symmetry pairs
        self.keypoint_mappings = self._get_keypoint_mappings()
        self.symmetry_pairs = self._get_symmetry_pairs()
        
        logger.info(f"Symmetry analyzer initialized for {keypoint_format} format")
    
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
                "left_shoulder": 11, "right_shoulder": 12, "left_elbow": 13, "right_elbow": 14,
                "left_wrist": 15, "right_wrist": 16, "left_hip": 23, "right_hip": 24,
                "left_knee": 25, "right_knee": 26, "left_ankle": 27, "right_ankle": 28,
                "left_heel": 29, "right_heel": 30, "left_foot_index": 31, "right_foot_index": 32
            }
        }
        
        return mappings
    
    def _get_symmetry_pairs(self) -> List[Tuple[str, str]]:
        """Get left-right symmetry pairs for the current keypoint format."""
        pairs = [
            ("left_shoulder", "right_shoulder"),
            ("left_elbow", "right_elbow"),
            ("left_wrist", "right_wrist"),
            ("left_hip", "right_hip"),
            ("left_knee", "right_knee"),
            ("left_ankle", "right_ankle")
        ]
        
        # Add format-specific pairs
        if self.keypoint_format == "BODY_25":
            pairs.extend([
                ("left_heel", "right_heel"),
                ("left_big_toe", "right_big_toe")
            ])
        elif self.keypoint_format == "BLAZEPOSE_33":
            pairs.extend([
                ("left_heel", "right_heel"),
                ("left_foot_index", "right_foot_index")
            ])
        
        return pairs
    
    def analyze_symmetry(self, pose_sequence: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Analyze left-right symmetry in pose sequence.
        
        Args:
            pose_sequence: List of pose estimation results
            
        Returns:
            Dictionary containing symmetry analysis results
        """
        if not pose_sequence:
            return {}
        
        # Convert poses to array format
        keypoints_array = self._poses_to_array(pose_sequence)
        if keypoints_array is None:
            return {}
        
        symmetry_results = {}
        
        try:
            # Analyze positional symmetry
            symmetry_results.update(self._analyze_positional_symmetry(keypoints_array))
            
            # Analyze movement symmetry
            symmetry_results.update(self._analyze_movement_symmetry(keypoints_array))
            
            # Analyze temporal symmetry
            symmetry_results.update(self._analyze_temporal_symmetry(keypoints_array))
            
            # Analyze angular symmetry
            symmetry_results.update(self._analyze_angular_symmetry(keypoints_array))
            
            # Calculate overall symmetry score
            symmetry_results.update(self._calculate_overall_symmetry(symmetry_results))
            
        except Exception as e:
            logger.error(f"Symmetry analysis failed: {e}")
            symmetry_results["analysis_error"] = str(e)
        
        return symmetry_results
    
    def _poses_to_array(self, pose_sequence: List[Dict[str, Any]]) -> Optional[np.ndarray]:
        """Convert pose sequence to numpy array."""
        if not pose_sequence:
            return None
        
        # Get keypoints from first valid pose
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
    
    def _analyze_positional_symmetry(self, keypoints: np.ndarray) -> Dict[str, Any]:
        """Analyze positional symmetry between left and right body parts."""
        results = {}
        mapping = self.keypoint_mappings.get(self.keypoint_format, {})
        
        if not mapping:
            return results
        
        # Calculate body center line for reference
        center_line = self._calculate_body_center_line(keypoints, mapping)
        
        for left_name, right_name in self.symmetry_pairs:
            left_idx = mapping.get(left_name)
            right_idx = mapping.get(right_name)
            
            if left_idx is not None and right_idx is not None:
                # Extract positions for valid frames
                left_positions = []
                right_positions = []
                
                for frame in keypoints:
                    if (frame[left_idx, 2] >= self.confidence_threshold and 
                        frame[right_idx, 2] >= self.confidence_threshold):
                        left_positions.append(frame[left_idx, :2])
                        right_positions.append(frame[right_idx, :2])
                
                if left_positions and right_positions:
                    left_positions = np.array(left_positions)
                    right_positions = np.array(right_positions)
                    
                    # Calculate symmetry metrics
                    symmetry_metrics = self._calculate_positional_symmetry_metrics(
                        left_positions, right_positions, center_line
                    )
                    
                    # Store results
                    joint_name = left_name.replace("left_", "")
                    for metric_name, value in symmetry_metrics.items():
                        results[f"{joint_name}_{metric_name}"] = value
        
        return results
    
    def _calculate_body_center_line(self, keypoints: np.ndarray, mapping: Dict[str, int]) -> np.ndarray:
        """Calculate the body center line for symmetry reference."""
        # Use midpoint between shoulders and hips if available
        center_points = []
        
        # Shoulder midpoint
        left_shoulder_idx = mapping.get("left_shoulder")
        right_shoulder_idx = mapping.get("right_shoulder")
        if left_shoulder_idx is not None and right_shoulder_idx is not None:
            for frame in keypoints:
                if (frame[left_shoulder_idx, 2] >= self.confidence_threshold and 
                    frame[right_shoulder_idx, 2] >= self.confidence_threshold):
                    shoulder_midpoint = (frame[left_shoulder_idx, :2] + frame[right_shoulder_idx, :2]) / 2
                    center_points.append(shoulder_midpoint)
        
        # Hip midpoint
        left_hip_idx = mapping.get("left_hip")
        right_hip_idx = mapping.get("right_hip")
        if left_hip_idx is not None and right_hip_idx is not None:
            for frame in keypoints:
                if (frame[left_hip_idx, 2] >= self.confidence_threshold and 
                    frame[right_hip_idx, 2] >= self.confidence_threshold):
                    hip_midpoint = (frame[left_hip_idx, :2] + frame[right_hip_idx, :2]) / 2
                    center_points.append(hip_midpoint)
        
        if center_points:
            return np.mean(center_points, axis=0)
        else:
            # Fallback to image center
            return np.array([320, 240])  # Assume 640x480 image
    
    def _calculate_positional_symmetry_metrics(
        self, 
        left_positions: np.ndarray, 
        right_positions: np.ndarray,
        center_line: np.ndarray
    ) -> Dict[str, float]:
        """Calculate positional symmetry metrics."""
        metrics = {}
        
        # Distance from center line
        left_distances = np.abs(left_positions[:, 0] - center_line[0])
        right_distances = np.abs(right_positions[:, 0] - center_line[0])
        
        # Symmetry index based on distance from center
        distance_symmetry = np.mean(np.abs(left_distances - right_distances) / (left_distances + right_distances + 1e-8))
        metrics["distance_symmetry_index"] = distance_symmetry
        
        # Position variance symmetry
        left_variance = np.var(left_positions, axis=0)
        right_variance = np.var(right_positions, axis=0)
        variance_symmetry = np.mean(np.abs(left_variance - right_variance) / (left_variance + right_variance + 1e-8))
        metrics["variance_symmetry_index"] = variance_symmetry
        
        # Range of motion symmetry
        left_range = np.max(left_positions, axis=0) - np.min(left_positions, axis=0)
        right_range = np.max(right_positions, axis=0) - np.min(right_positions, axis=0)
        range_symmetry = np.mean(np.abs(left_range - right_range) / (left_range + right_range + 1e-8))
        metrics["range_symmetry_index"] = range_symmetry
        
        return metrics
    
    def _analyze_movement_symmetry(self, keypoints: np.ndarray) -> Dict[str, Any]:
        """Analyze movement symmetry between left and right body parts."""
        results = {}
        mapping = self.keypoint_mappings.get(self.keypoint_format, {})
        
        if not mapping:
            return results
        
        for left_name, right_name in self.symmetry_pairs:
            left_idx = mapping.get(left_name)
            right_idx = mapping.get(right_name)
            
            if left_idx is not None and right_idx is not None:
                # Calculate velocities
                left_velocities = []
                right_velocities = []
                
                for i in range(1, len(keypoints)):
                    if (keypoints[i-1, left_idx, 2] >= self.confidence_threshold and
                        keypoints[i, left_idx, 2] >= self.confidence_threshold and
                        keypoints[i-1, right_idx, 2] >= self.confidence_threshold and
                        keypoints[i, right_idx, 2] >= self.confidence_threshold):
                        
                        left_vel = keypoints[i, left_idx, :2] - keypoints[i-1, left_idx, :2]
                        right_vel = keypoints[i, right_idx, :2] - keypoints[i-1, right_idx, :2]
                        
                        left_velocities.append(np.linalg.norm(left_vel))
                        right_velocities.append(np.linalg.norm(right_vel))
                
                if left_velocities and right_velocities:
                    left_velocities = np.array(left_velocities)
                    right_velocities = np.array(right_velocities)
                    
                    # Calculate movement symmetry metrics
                    joint_name = left_name.replace("left_", "")
                    
                    # Velocity symmetry
                    velocity_symmetry = np.mean(np.abs(left_velocities - right_velocities) / 
                                              (left_velocities + right_velocities + 1e-8))
                    results[f"{joint_name}_velocity_symmetry_index"] = velocity_symmetry
                    
                    # Movement correlation
                    if len(left_velocities) > 1:
                        correlation = np.corrcoef(left_velocities, right_velocities)[0, 1]
                        if not np.isnan(correlation):
                            results[f"{joint_name}_movement_correlation"] = correlation
                    
                    # Phase difference (simplified)
                    phase_diff = self._calculate_phase_difference(left_velocities, right_velocities)
                    results[f"{joint_name}_phase_difference"] = phase_diff
        
        return results
    
    def _calculate_phase_difference(self, left_signal: np.ndarray, right_signal: np.ndarray) -> float:
        """Calculate phase difference between left and right signals."""
        if len(left_signal) < 10 or len(right_signal) < 10:
            return 0.0
        
        try:
            # Use cross-correlation to find phase difference
            correlation = np.correlate(left_signal, right_signal, mode='full')
            max_corr_idx = np.argmax(correlation)
            phase_shift = max_corr_idx - (len(right_signal) - 1)
            
            # Normalize to [0, 1] range
            max_shift = min(len(left_signal), len(right_signal)) // 2
            normalized_phase = abs(phase_shift) / max_shift if max_shift > 0 else 0.0
            
            return min(normalized_phase, 1.0)
        
        except Exception:
            return 0.0
    
    def _analyze_temporal_symmetry(self, keypoints: np.ndarray) -> Dict[str, Any]:
        """Analyze temporal symmetry in gait patterns."""
        results = {}
        mapping = self.keypoint_mappings.get(self.keypoint_format, {})
        
        if not mapping:
            return results
        
        # Focus on ankle movements for temporal analysis
        left_ankle_idx = mapping.get("left_ankle")
        right_ankle_idx = mapping.get("right_ankle")
        
        if left_ankle_idx is not None and right_ankle_idx is not None:
            # Extract ankle positions
            left_ankle_y = []
            right_ankle_y = []
            
            for frame in keypoints:
                if (frame[left_ankle_idx, 2] >= self.confidence_threshold and
                    frame[right_ankle_idx, 2] >= self.confidence_threshold):
                    left_ankle_y.append(frame[left_ankle_idx, 1])
                    right_ankle_y.append(frame[right_ankle_idx, 1])
            
            if len(left_ankle_y) > 20:  # Need sufficient data
                left_ankle_y = np.array(left_ankle_y)
                right_ankle_y = np.array(right_ankle_y)
                
                # Detect step cycles for each foot
                left_cycles = self._detect_simple_cycles(left_ankle_y)
                right_cycles = self._detect_simple_cycles(right_ankle_y)
                
                if left_cycles and right_cycles:
                    # Calculate cycle duration symmetry
                    left_durations = [cycle[1] - cycle[0] for cycle in left_cycles]
                    right_durations = [cycle[1] - cycle[0] for cycle in right_cycles]
                    
                    if left_durations and right_durations:
                        left_mean_duration = np.mean(left_durations)
                        right_mean_duration = np.mean(right_durations)
                        
                        duration_symmetry = abs(left_mean_duration - right_mean_duration) / \
                                          ((left_mean_duration + right_mean_duration) / 2)
                        results["cycle_duration_symmetry_index"] = duration_symmetry
                        
                        # Step frequency symmetry
                        left_frequency = len(left_cycles) / len(left_ankle_y)
                        right_frequency = len(right_cycles) / len(right_ankle_y)
                        
                        frequency_symmetry = abs(left_frequency - right_frequency) / \
                                           ((left_frequency + right_frequency) / 2)
                        results["step_frequency_symmetry_index"] = frequency_symmetry
        
        return results
    
    def _detect_simple_cycles(self, signal: np.ndarray) -> List[Tuple[int, int]]:
        """Detect simple cycles in a signal using local minima."""
        if len(signal) < 10:
            return []
        
        # Find local minima
        minima = []
        for i in range(1, len(signal) - 1):
            if signal[i] < signal[i-1] and signal[i] < signal[i+1]:
                minima.append(i)
        
        # Create cycles between consecutive minima
        cycles = []
        for i in range(len(minima) - 1):
            cycles.append((minima[i], minima[i+1]))
        
        return cycles
    
    def _analyze_angular_symmetry(self, keypoints: np.ndarray) -> Dict[str, Any]:
        """Analyze angular symmetry between left and right joints."""
        results = {}
        mapping = self.keypoint_mappings.get(self.keypoint_format, {})
        
        if not mapping:
            return results
        
        # Define joint angle triplets for left and right sides
        angle_triplets = [
            (("left_hip", "left_knee", "left_ankle"), ("right_hip", "right_knee", "right_ankle"), "knee"),
            (("left_shoulder", "left_hip", "left_knee"), ("right_shoulder", "right_hip", "right_knee"), "hip")
        ]
        
        for left_triplet, right_triplet, joint_name in angle_triplets:
            # Check if all keypoints exist
            left_indices = [mapping.get(name) for name in left_triplet]
            right_indices = [mapping.get(name) for name in right_triplet]
            
            if all(idx is not None for idx in left_indices + right_indices):
                left_angles = self._calculate_angle_sequence(keypoints, left_indices)
                right_angles = self._calculate_angle_sequence(keypoints, right_indices)
                
                if len(left_angles) > 0 and len(right_angles) > 0:
                    # Calculate angular symmetry
                    min_length = min(len(left_angles), len(right_angles))
                    left_angles = left_angles[:min_length]
                    right_angles = right_angles[:min_length]
                    
                    angle_symmetry = np.mean(np.abs(left_angles - right_angles) / 
                                           (np.abs(left_angles) + np.abs(right_angles) + 1e-8))
                    results[f"{joint_name}_angle_symmetry_index"] = angle_symmetry
                    
                    # Angular range symmetry
                    left_range = np.max(left_angles) - np.min(left_angles)
                    right_range = np.max(right_angles) - np.min(right_angles)
                    
                    range_symmetry = abs(left_range - right_range) / ((left_range + right_range) / 2)
                    results[f"{joint_name}_angle_range_symmetry_index"] = range_symmetry
        
        return results
    
    def _calculate_angle_sequence(self, keypoints: np.ndarray, indices: List[int]) -> np.ndarray:
        """Calculate angle sequence for three keypoints."""
        angles = []
        
        for frame in keypoints:
            # Check if all keypoints have sufficient confidence
            if all(frame[idx, 2] >= self.confidence_threshold for idx in indices):
                p1 = frame[indices[0], :2]
                p2 = frame[indices[1], :2]
                p3 = frame[indices[2], :2]
                
                angle = self._calculate_angle(p1, p2, p3)
                angles.append(angle)
        
        return np.array(angles)
    
    def _calculate_angle(self, p1: np.ndarray, p2: np.ndarray, p3: np.ndarray) -> float:
        """Calculate angle between three points (p2 is the vertex)."""
        v1 = p1 - p2
        v2 = p3 - p2
        
        # Calculate angle using dot product
        cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-8)
        cos_angle = np.clip(cos_angle, -1, 1)  # Ensure valid range
        angle = np.arccos(cos_angle)
        
        return np.degrees(angle)
    
    def _calculate_overall_symmetry(self, symmetry_results: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate overall symmetry scores."""
        overall_results = {}
        
        # Collect all symmetry indices
        symmetry_indices = []
        for key, value in symmetry_results.items():
            if "symmetry_index" in key and isinstance(value, (int, float)) and not np.isnan(value):
                symmetry_indices.append(value)
        
        if symmetry_indices:
            overall_results["overall_symmetry_index"] = np.mean(symmetry_indices)
            overall_results["symmetry_variability"] = np.std(symmetry_indices)
            
            # Classify symmetry level
            mean_symmetry = overall_results["overall_symmetry_index"]
            if mean_symmetry < self.symmetry_threshold:
                overall_results["symmetry_classification"] = "symmetric"
            elif mean_symmetry < self.symmetry_threshold * 2:
                overall_results["symmetry_classification"] = "mildly_asymmetric"
            elif mean_symmetry < self.symmetry_threshold * 3:
                overall_results["symmetry_classification"] = "moderately_asymmetric"
            else:
                overall_results["symmetry_classification"] = "severely_asymmetric"
            
            # Count asymmetric joints
            asymmetric_count = sum(1 for idx in symmetry_indices if idx > self.symmetry_threshold)
            overall_results["asymmetric_joint_count"] = asymmetric_count
            overall_results["asymmetric_joint_percentage"] = asymmetric_count / len(symmetry_indices) * 100
        
        return overall_results
    
    def generate_symmetry_report(self, symmetry_results: Dict[str, Any]) -> str:
        """Generate a human-readable symmetry analysis report."""
        if not symmetry_results:
            return "No symmetry analysis results available."
        
        report = ["Gait Symmetry Analysis Report", "=" * 35, ""]
        
        # Overall symmetry
        if "overall_symmetry_index" in symmetry_results:
            overall_score = symmetry_results["overall_symmetry_index"]
            classification = symmetry_results.get("symmetry_classification", "unknown")
            
            report.append(f"Overall Symmetry Score: {overall_score:.3f}")
            report.append(f"Classification: {classification.replace('_', ' ').title()}")
            report.append("")
        
        # Joint-specific results
        joint_results = {}
        for key, value in symmetry_results.items():
            if "_symmetry_index" in key and not key.startswith("overall"):
                joint_name = key.replace("_symmetry_index", "").replace("_", " ").title()
                joint_results[joint_name] = value
        
        if joint_results:
            report.append("Joint-Specific Symmetry:")
            for joint, score in sorted(joint_results.items()):
                status = "Symmetric" if score < self.symmetry_threshold else "Asymmetric"
                report.append(f"  {joint}: {score:.3f} ({status})")
            report.append("")
        
        # Asymmetric joints summary
        if "asymmetric_joint_count" in symmetry_results:
            count = symmetry_results["asymmetric_joint_count"]
            percentage = symmetry_results.get("asymmetric_joint_percentage", 0)
            report.append(f"Asymmetric Joints: {count} ({percentage:.1f}%)")
            report.append("")
        
        # Recommendations
        if "overall_symmetry_index" in symmetry_results:
            overall_score = symmetry_results["overall_symmetry_index"]
            report.append("Recommendations:")
            
            if overall_score < self.symmetry_threshold:
                report.append("  - Gait appears symmetric within normal limits")
            elif overall_score < self.symmetry_threshold * 2:
                report.append("  - Mild asymmetry detected - monitor for changes")
                report.append("  - Consider gait training or physical therapy")
            else:
                report.append("  - Significant asymmetry detected")
                report.append("  - Recommend clinical evaluation")
                report.append("  - Consider underlying pathological conditions")
        
        return "\n".join(report)