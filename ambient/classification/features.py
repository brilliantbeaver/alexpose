"""
Gait Feature Extraction and Representation

This module provides feature vector representations for gait analysis.
The GaitFeatureVector class encapsulates the key features extracted from
gait sequences for use in machine learning classification.

Features include:
- Mean joint angles (hip, knee, ankle) for both legs
- Left-right asymmetry measures
- Range of motion features
- Metadata for tracking and labeling

The feature vector is designed to be:
- Consistent across different pose estimation backends
- Robust to missing or invalid data
- Compatible with scikit-learn classifiers
- Interpretable for clinical analysis

Author: AlexPose Team
"""

import numpy as np
from dataclasses import dataclass
from typing import List, Optional
from loguru import logger


@dataclass
class GaitFeatureVector:
    """
    Feature vector for gait classification.

    This represents the features extracted from a gait sequence that will
    be used for classification. Features include mean joint angles and
    asymmetry measures.
    """

    # Mean joint angles (degrees)
    left_hip_mean: float = 0.0
    left_knee_mean: float = 0.0
    left_ankle_mean: float = 0.0
    right_hip_mean: float = 0.0
    right_knee_mean: float = 0.0
    right_ankle_mean: float = 0.0

    # Left-right asymmetry (absolute differences)
    hip_asymmetry: float = 0.0
    knee_asymmetry: float = 0.0
    ankle_asymmetry: float = 0.0

    # Range of motion features
    left_hip_range: float = 0.0
    left_knee_range: float = 0.0
    left_ankle_range: float = 0.0
    right_hip_range: float = 0.0
    right_knee_range: float = 0.0
    right_ankle_range: float = 0.0

    # Metadata
    sample_id: str = ""
    condition_label: str = ""

    def __post_init__(self):
        """Calculate asymmetry features after initialization."""
        if self.hip_asymmetry == 0.0 and (self.left_hip_mean != 0.0 or self.right_hip_mean != 0.0):
            self.hip_asymmetry = abs(self.left_hip_mean - self.right_hip_mean)
        if self.knee_asymmetry == 0.0 and (self.left_knee_mean != 0.0 or self.right_knee_mean != 0.0):
            self.knee_asymmetry = abs(self.left_knee_mean - self.right_knee_mean)
        if self.ankle_asymmetry == 0.0 and (self.left_ankle_mean != 0.0 or self.right_ankle_mean != 0.0):
            self.ankle_asymmetry = abs(self.left_ankle_mean - self.right_ankle_mean)

    def to_array(self) -> np.ndarray:
        """Convert to numpy array for sklearn."""
        return np.array(
            [
                self.left_hip_mean,
                self.left_knee_mean,
                self.left_ankle_mean,
                self.right_hip_mean,
                self.right_knee_mean,
                self.right_ankle_mean,
                self.hip_asymmetry,
                self.knee_asymmetry,
                self.ankle_asymmetry,
                self.left_hip_range,
                self.left_knee_range,
                self.left_ankle_range,
                self.right_hip_range,
                self.right_knee_range,
                self.right_ankle_range,
            ]
        )

    @classmethod
    def get_feature_names(cls) -> List[str]:
        """Get ordered list of feature names."""
        return [
            "left_hip_mean",
            "left_knee_mean",
            "left_ankle_mean",
            "right_hip_mean",
            "right_knee_mean",
            "right_ankle_mean",
            "hip_asymmetry",
            "knee_asymmetry",
            "ankle_asymmetry",
            "left_hip_range",
            "left_knee_range",
            "left_ankle_range",
            "right_hip_range",
            "right_knee_range",
            "right_ankle_range",
        ]

    @classmethod
    def from_joint_angles(
        cls, joint_angle_sequence, sample_id: str = "", condition_label: str = ""
    ) -> Optional["GaitFeatureVector"]:
        """
        Create feature vector from JointAngleSequence.

        Args:
            joint_angle_sequence: JointAngleSequence object from joint_angles module
            sample_id: Identifier for this sample
            condition_label: Ground truth condition label

        Returns:
            GaitFeatureVector with computed features, or None if sequence has no valid data
            
        Note:
            Returns None if the sequence has no valid angle data (all NaN).
            If a joint has no valid angles, the corresponding features will be 0.
        """
        # Validate sequence has any valid data
        if hasattr(joint_angle_sequence, 'has_valid_data'):
            if not joint_angle_sequence.has_valid_data():
                logger.warning(
                    f"Sample '{sample_id}': No valid angle data in sequence. "
                    f"Total frames: {len(joint_angle_sequence.frames)}, "
                    f"Valid frames: {joint_angle_sequence.get_valid_frame_count()}"
                )
                return None
        
        # Extract statistics for each joint
        left_hip_stats = joint_angle_sequence.get_statistics("left_hip")
        left_knee_stats = joint_angle_sequence.get_statistics("left_knee")
        left_ankle_stats = joint_angle_sequence.get_statistics("left_ankle")
        right_hip_stats = joint_angle_sequence.get_statistics("right_hip")
        right_knee_stats = joint_angle_sequence.get_statistics("right_knee")
        right_ankle_stats = joint_angle_sequence.get_statistics("right_ankle")

        # Helper function to safely get value, replacing NaN with 0
        def safe_get(stats_dict, key, default=0.0):
            value = stats_dict.get(key, default)
            return default if np.isnan(value) else value

        # Extract mean values, replacing NaN with 0
        left_hip_mean = safe_get(left_hip_stats, "mean")
        left_knee_mean = safe_get(left_knee_stats, "mean")
        left_ankle_mean = safe_get(left_ankle_stats, "mean")
        right_hip_mean = safe_get(right_hip_stats, "mean")
        right_knee_mean = safe_get(right_knee_stats, "mean")
        right_ankle_mean = safe_get(right_ankle_stats, "mean")

        # Compute asymmetry features
        hip_asymmetry = abs(left_hip_mean - right_hip_mean)
        knee_asymmetry = abs(left_knee_mean - right_knee_mean)
        ankle_asymmetry = abs(left_ankle_mean - right_ankle_mean)

        return cls(
            left_hip_mean=left_hip_mean,
            left_knee_mean=left_knee_mean,
            left_ankle_mean=left_ankle_mean,
            right_hip_mean=right_hip_mean,
            right_knee_mean=right_knee_mean,
            right_ankle_mean=right_ankle_mean,
            hip_asymmetry=hip_asymmetry,
            knee_asymmetry=knee_asymmetry,
            ankle_asymmetry=ankle_asymmetry,
            left_hip_range=safe_get(left_hip_stats, "range"),
            left_knee_range=safe_get(left_knee_stats, "range"),
            left_ankle_range=safe_get(right_ankle_stats, "range"),
            right_hip_range=safe_get(right_hip_stats, "range"),
            right_knee_range=safe_get(right_knee_stats, "range"),
            right_ankle_range=safe_get(right_ankle_stats, "range"),
            sample_id=sample_id,
            condition_label=condition_label,
        )

    def get_feature_summary(self) -> str:
        """
        Get a human-readable summary of the feature vector.
        
        Returns:
            Formatted string with feature values and interpretations
        """
        lines = [
            f"Gait Feature Summary - {self.sample_id or 'Unknown Sample'}",
            f"Condition: {self.condition_label or 'Unknown'}",
            "",
            "Mean Joint Angles (degrees):",
            f"  Left Hip:   {self.left_hip_mean:6.1f}°    Right Hip:   {self.right_hip_mean:6.1f}°",
            f"  Left Knee:  {self.left_knee_mean:6.1f}°    Right Knee:  {self.right_knee_mean:6.1f}°",
            f"  Left Ankle: {self.left_ankle_mean:6.1f}°    Right Ankle: {self.right_ankle_mean:6.1f}°",
            "",
            "Range of Motion (degrees):",
            f"  Left Hip:   {self.left_hip_range:6.1f}°    Right Hip:   {self.right_hip_range:6.1f}°",
            f"  Left Knee:  {self.left_knee_range:6.1f}°    Right Knee:  {self.right_knee_range:6.1f}°",
            f"  Left Ankle: {self.left_ankle_range:6.1f}°    Right Ankle: {self.right_ankle_range:6.1f}°",
            "",
            "Asymmetry Measures (degrees):",
            f"  Hip:   {self.hip_asymmetry:6.1f}°",
            f"  Knee:  {self.knee_asymmetry:6.1f}°",
            f"  Ankle: {self.ankle_asymmetry:6.1f}°",
        ]
        
        return "\n".join(lines)

    def validate(self) -> tuple[bool, List[str]]:
        """
        Validate the feature vector for common issues.
        
        Returns:
            Tuple of (is_valid, list_of_issues)
        """
        issues = []
        
        # Check for NaN values
        feature_array = self.to_array()
        if np.any(np.isnan(feature_array)):
            nan_indices = np.where(np.isnan(feature_array))[0]
            nan_features = [self.get_feature_names()[i] for i in nan_indices]
            issues.append(f"NaN values in features: {nan_features}")
        
        # Check for infinite values
        if np.any(np.isinf(feature_array)):
            inf_indices = np.where(np.isinf(feature_array))[0]
            inf_features = [self.get_feature_names()[i] for i in inf_indices]
            issues.append(f"Infinite values in features: {inf_features}")
        
        # Check for unrealistic joint angles (basic sanity check)
        angle_features = [
            self.left_hip_mean, self.left_knee_mean, self.left_ankle_mean,
            self.right_hip_mean, self.right_knee_mean, self.right_ankle_mean
        ]
        
        for i, angle in enumerate(angle_features):
            if abs(angle) > 180:
                feature_name = self.get_feature_names()[i]
                issues.append(f"Unrealistic angle in {feature_name}: {angle}°")
        
        # Check for negative range values
        range_features = [
            self.left_hip_range, self.left_knee_range, self.left_ankle_range,
            self.right_hip_range, self.right_knee_range, self.right_ankle_range
        ]
        
        for i, range_val in enumerate(range_features):
            if range_val < 0:
                feature_name = self.get_feature_names()[i + 9]  # Range features start at index 9
                issues.append(f"Negative range in {feature_name}: {range_val}°")
        
        return len(issues) == 0, issues