"""
Tests for GaitFeatureVector

Tests the feature vector representation used across all classifiers.
Covers feature extraction, validation, and utility methods.
"""

import pytest
import numpy as np
from unittest.mock import Mock

from ambient.classification.features import GaitFeatureVector


class TestGaitFeatureVector:
    """Test GaitFeatureVector functionality."""

    def test_initialization_defaults(self):
        """Test default initialization."""
        fv = GaitFeatureVector()
        
        assert fv.left_hip_mean == 0.0
        assert fv.left_knee_mean == 0.0
        assert fv.left_ankle_mean == 0.0
        assert fv.right_hip_mean == 0.0
        assert fv.right_knee_mean == 0.0
        assert fv.right_ankle_mean == 0.0
        assert fv.hip_asymmetry == 0.0
        assert fv.knee_asymmetry == 0.0
        assert fv.ankle_asymmetry == 0.0
        assert fv.left_hip_range == 0.0
        assert fv.left_knee_range == 0.0
        assert fv.left_ankle_range == 0.0
        assert fv.right_hip_range == 0.0
        assert fv.right_knee_range == 0.0
        assert fv.right_ankle_range == 0.0
        assert fv.sample_id == ""
        assert fv.condition_label == ""

    def test_initialization_with_values(self):
        """Test initialization with specific values."""
        fv = GaitFeatureVector(
            left_hip_mean=45.0,
            left_knee_mean=60.0,
            left_ankle_mean=20.0,
            right_hip_mean=45.0,
            right_knee_mean=60.0,
            right_ankle_mean=20.0,
            hip_asymmetry=0.0,
            knee_asymmetry=0.0,
            ankle_asymmetry=0.0,
            left_hip_range=30.0,
            left_knee_range=40.0,
            left_ankle_range=25.0,
            right_hip_range=30.0,
            right_knee_range=40.0,
            right_ankle_range=25.0,
            sample_id="test_001",
            condition_label="normal"
        )
        
        assert fv.left_hip_mean == 45.0
        assert fv.left_knee_mean == 60.0
        assert fv.left_ankle_mean == 20.0
        assert fv.right_hip_mean == 45.0
        assert fv.right_knee_mean == 60.0
        assert fv.right_ankle_mean == 20.0
        assert fv.hip_asymmetry == 0.0
        assert fv.knee_asymmetry == 0.0
        assert fv.ankle_asymmetry == 0.0
        assert fv.left_hip_range == 30.0
        assert fv.left_knee_range == 40.0
        assert fv.left_ankle_range == 25.0
        assert fv.right_hip_range == 30.0
        assert fv.right_knee_range == 40.0
        assert fv.right_ankle_range == 25.0
        assert fv.sample_id == "test_001"
        assert fv.condition_label == "normal"

    def test_to_array(self):
        """Test conversion to numpy array."""
        fv = GaitFeatureVector(
            left_hip_mean=45.0,
            left_knee_mean=60.0,
            left_ankle_mean=20.0,
            right_hip_mean=45.0,
            right_knee_mean=60.0,
            right_ankle_mean=20.0,
            hip_asymmetry=0.0,
            knee_asymmetry=0.0,
            ankle_asymmetry=0.0,
            left_hip_range=30.0,
            left_knee_range=40.0,
            left_ankle_range=25.0,
            right_hip_range=30.0,
            right_knee_range=40.0,
            right_ankle_range=25.0,
        )
        
        array = fv.to_array()
        expected = np.array([
            45.0, 60.0, 20.0,  # Left mean angles
            45.0, 60.0, 20.0,  # Right mean angles
            0.0, 0.0, 0.0,     # Asymmetries
            30.0, 40.0, 25.0,  # Left ranges
            30.0, 40.0, 25.0   # Right ranges
        ])
        
        np.testing.assert_array_equal(array, expected)
        assert array.shape == (15,)

    def test_get_feature_names(self):
        """Test feature names retrieval."""
        names = GaitFeatureVector.get_feature_names()
        
        expected_names = [
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
        
        assert names == expected_names
        assert len(names) == 15

    def test_from_joint_angles_valid_data(self):
        """Test creation from valid joint angle sequence."""
        # Mock joint angle sequence
        mock_sequence = Mock()
        mock_sequence.has_valid_data.return_value = True
        mock_sequence.frames = [1, 2, 3]  # Mock frames
        mock_sequence.get_valid_frame_count.return_value = 3
        
        # Mock statistics for each joint
        mock_sequence.get_statistics.side_effect = [
            {"mean": 45.0, "range": 30.0},  # left_hip
            {"mean": 60.0, "range": 40.0},  # left_knee
            {"mean": 20.0, "range": 25.0},  # left_ankle
            {"mean": 45.0, "range": 30.0},  # right_hip
            {"mean": 60.0, "range": 40.0},  # right_knee
            {"mean": 20.0, "range": 25.0},  # right_ankle
        ]
        
        fv = GaitFeatureVector.from_joint_angles(
            mock_sequence,
            sample_id="test_001",
            condition_label="normal"
        )
        
        assert fv is not None
        assert fv.left_hip_mean == 45.0
        assert fv.left_knee_mean == 60.0
        assert fv.left_ankle_mean == 20.0
        assert fv.right_hip_mean == 45.0
        assert fv.right_knee_mean == 60.0
        assert fv.right_ankle_mean == 20.0
        assert fv.hip_asymmetry == 0.0  # |45 - 45|
        assert fv.knee_asymmetry == 0.0  # |60 - 60|
        assert fv.ankle_asymmetry == 0.0  # |20 - 20|
        assert fv.left_hip_range == 30.0
        assert fv.left_knee_range == 40.0
        assert fv.left_ankle_range == 25.0
        assert fv.right_hip_range == 30.0
        assert fv.right_knee_range == 40.0
        assert fv.right_ankle_range == 25.0
        assert fv.sample_id == "test_001"
        assert fv.condition_label == "normal"

    def test_from_joint_angles_invalid_data(self):
        """Test creation from invalid joint angle sequence."""
        # Mock joint angle sequence with no valid data
        mock_sequence = Mock()
        mock_sequence.has_valid_data.return_value = False
        mock_sequence.frames = []
        mock_sequence.get_valid_frame_count.return_value = 0
        
        fv = GaitFeatureVector.from_joint_angles(
            mock_sequence,
            sample_id="test_invalid",
            condition_label="unknown"
        )
        
        assert fv is None

    def test_from_joint_angles_with_nan_values(self):
        """Test creation with NaN values in statistics."""
        # Mock joint angle sequence
        mock_sequence = Mock()
        mock_sequence.has_valid_data.return_value = True
        mock_sequence.frames = [1, 2, 3]
        mock_sequence.get_valid_frame_count.return_value = 3
        
        # Mock statistics with some NaN values
        mock_sequence.get_statistics.side_effect = [
            {"mean": np.nan, "range": 30.0},  # left_hip - NaN mean
            {"mean": 60.0, "range": np.nan},  # left_knee - NaN range
            {"mean": 20.0, "range": 25.0},    # left_ankle - valid
            {"mean": 45.0, "range": 30.0},    # right_hip - valid
            {"mean": 60.0, "range": 40.0},    # right_knee - valid
            {"mean": 20.0, "range": 25.0},    # right_ankle - valid
        ]
        
        fv = GaitFeatureVector.from_joint_angles(
            mock_sequence,
            sample_id="test_nan",
            condition_label="test"
        )
        
        assert fv is not None
        assert fv.left_hip_mean == 0.0  # NaN replaced with 0
        assert fv.left_knee_mean == 60.0
        assert fv.left_knee_range == 0.0  # NaN replaced with 0
        assert fv.left_ankle_mean == 20.0
        assert fv.right_hip_mean == 45.0
        # Asymmetry calculation with NaN -> 0
        assert fv.hip_asymmetry == abs(0.0 - 45.0)  # |0 - 45| = 45

    def test_get_feature_summary(self):
        """Test feature summary generation."""
        fv = GaitFeatureVector(
            left_hip_mean=45.0,
            left_knee_mean=60.0,
            left_ankle_mean=20.0,
            right_hip_mean=47.0,
            right_knee_mean=62.0,
            right_ankle_mean=22.0,
            hip_asymmetry=2.0,
            knee_asymmetry=2.0,
            ankle_asymmetry=2.0,
            left_hip_range=30.0,
            left_knee_range=40.0,
            left_ankle_range=25.0,
            right_hip_range=32.0,
            right_knee_range=42.0,
            right_ankle_range=27.0,
            sample_id="test_001",
            condition_label="normal"
        )
        
        summary = fv.get_feature_summary()
        
        assert "test_001" in summary
        assert "normal" in summary
        assert "45.0°" in summary
        assert "60.0°" in summary
        assert "20.0°" in summary
        assert "47.0°" in summary
        assert "62.0°" in summary
        assert "22.0°" in summary
        assert "2.0°" in summary  # Asymmetry values

    def test_validate_valid_features(self):
        """Test validation of valid feature vector."""
        fv = GaitFeatureVector(
            left_hip_mean=45.0,
            left_knee_mean=60.0,
            left_ankle_mean=20.0,
            right_hip_mean=45.0,
            right_knee_mean=60.0,
            right_ankle_mean=20.0,
            hip_asymmetry=0.0,
            knee_asymmetry=0.0,
            ankle_asymmetry=0.0,
            left_hip_range=30.0,
            left_knee_range=40.0,
            left_ankle_range=25.0,
            right_hip_range=30.0,
            right_knee_range=40.0,
            right_ankle_range=25.0,
        )
        
        is_valid, issues = fv.validate()
        
        assert is_valid is True
        assert len(issues) == 0

    def test_validate_nan_values(self):
        """Test validation with NaN values."""
        fv = GaitFeatureVector(
            left_hip_mean=np.nan,
            left_knee_mean=60.0,
            left_ankle_mean=20.0,
            right_hip_mean=45.0,
            right_knee_mean=60.0,
            right_ankle_mean=20.0,
        )
        
        is_valid, issues = fv.validate()
        
        assert is_valid is False
        assert len(issues) > 0
        assert "NaN values" in issues[0]
        assert "left_hip_mean" in issues[0]

    def test_validate_infinite_values(self):
        """Test validation with infinite values."""
        fv = GaitFeatureVector(
            left_hip_mean=np.inf,
            left_knee_mean=60.0,
            left_ankle_mean=20.0,
            right_hip_mean=45.0,
            right_knee_mean=60.0,
            right_ankle_mean=20.0,
        )
        
        is_valid, issues = fv.validate()
        
        assert is_valid is False
        assert len(issues) > 0
        assert "Infinite values" in issues[0]
        assert "left_hip_mean" in issues[0]

    def test_validate_unrealistic_angles(self):
        """Test validation with unrealistic joint angles."""
        fv = GaitFeatureVector(
            left_hip_mean=200.0,  # Unrealistic angle > 180
            left_knee_mean=60.0,
            left_ankle_mean=20.0,
            right_hip_mean=45.0,
            right_knee_mean=60.0,
            right_ankle_mean=20.0,
        )
        
        is_valid, issues = fv.validate()
        
        assert is_valid is False
        assert len(issues) > 0
        assert "Unrealistic angle" in issues[0]
        assert "left_hip_mean" in issues[0]
        assert "200.0°" in issues[0]

    def test_validate_negative_ranges(self):
        """Test validation with negative range values."""
        fv = GaitFeatureVector(
            left_hip_mean=45.0,
            left_knee_mean=60.0,
            left_ankle_mean=20.0,
            right_hip_mean=45.0,
            right_knee_mean=60.0,
            right_ankle_mean=20.0,
            left_hip_range=-10.0,  # Negative range
            left_knee_range=40.0,
            left_ankle_range=25.0,
            right_hip_range=30.0,
            right_knee_range=40.0,
            right_ankle_range=25.0,
        )
        
        is_valid, issues = fv.validate()
        
        assert is_valid is False
        assert len(issues) > 0
        assert "Negative range" in issues[0]
        assert "left_hip_range" in issues[0]
        assert "-10.0°" in issues[0]

    def test_asymmetry_calculation(self):
        """Test asymmetry calculation in from_joint_angles."""
        # Mock joint angle sequence with asymmetric values
        mock_sequence = Mock()
        mock_sequence.has_valid_data.return_value = True
        mock_sequence.frames = [1, 2, 3]
        mock_sequence.get_valid_frame_count.return_value = 3
        
        # Mock statistics with asymmetric values
        mock_sequence.get_statistics.side_effect = [
            {"mean": 40.0, "range": 30.0},  # left_hip
            {"mean": 55.0, "range": 35.0},  # left_knee
            {"mean": 15.0, "range": 20.0},  # left_ankle
            {"mean": 50.0, "range": 35.0},  # right_hip
            {"mean": 65.0, "range": 45.0},  # right_knee
            {"mean": 25.0, "range": 30.0},  # right_ankle
        ]
        
        fv = GaitFeatureVector.from_joint_angles(mock_sequence)
        
        assert fv is not None
        assert fv.hip_asymmetry == abs(40.0 - 50.0)    # |40 - 50| = 10
        assert fv.knee_asymmetry == abs(55.0 - 65.0)   # |55 - 65| = 10
        assert fv.ankle_asymmetry == abs(15.0 - 25.0)  # |15 - 25| = 10