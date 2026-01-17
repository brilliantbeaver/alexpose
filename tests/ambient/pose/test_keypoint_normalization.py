"""
Tests for keypoint coordinate normalization fixes.

This module tests the fix for MediaPipe coordinate normalization issues
where floating-point precision can cause coordinates to be slightly
outside the expected [0.0, 1.0] range.
"""

import pytest
from ambient.pose.keypoint_data import Keypoint, KeypointSet, KeypointFormat


class TestKeypointNormalizationFix:
    """Test coordinate normalization clamping fixes."""

    def test_keypoint_from_dict_clamps_negative_coordinates(self):
        """Test that Keypoint.from_dict clamps negative normalized coordinates."""
        keypoint_data = {
            'id': 0,
            'name': 'test_keypoint',
            'x': 100.0,
            'y': 200.0,
            'x_normalized': -0.0019250214099884033,  # The actual error value
            'y_normalized': 0.5
        }
        
        keypoint = Keypoint.from_dict(keypoint_data)
        
        assert keypoint.x_normalized == 0.0
        assert keypoint.y_normalized == 0.5

    def test_keypoint_from_dict_clamps_large_coordinates(self):
        """Test that Keypoint.from_dict clamps coordinates > 1.0."""
        keypoint_data = {
            'id': 0,
            'name': 'test_keypoint',
            'x': 100.0,
            'y': 200.0,
            'x_normalized': 0.5,
            'y_normalized': 1.0001
        }
        
        keypoint = Keypoint.from_dict(keypoint_data)
        
        assert keypoint.x_normalized == 0.5
        assert keypoint.y_normalized == 1.0

    def test_keypoint_from_dict_preserves_valid_coordinates(self):
        """Test that valid coordinates are preserved."""
        keypoint_data = {
            'id': 0,
            'name': 'test_keypoint',
            'x': 100.0,
            'y': 200.0,
            'x_normalized': 0.3,
            'y_normalized': 0.7
        }
        
        keypoint = Keypoint.from_dict(keypoint_data)
        
        assert keypoint.x_normalized == 0.3
        assert keypoint.y_normalized == 0.7

    def test_mediapipe_integration_clamps_coordinates(self):
        """Test that KeypointSet.from_mediapipe clamps problematic coordinates."""
        
        class MockLandmark:
            def __init__(self, x, y, z=0.0, visibility=0.9, presence=0.9):
                self.x = x
                self.y = y
                self.z = z
                self.visibility = visibility
                self.presence = presence
        
        # Create landmarks with problematic coordinates
        landmarks = [
            MockLandmark(-0.0019250214099884033, 0.5),  # Negative x
            MockLandmark(0.5, 1.0001),                  # Large y
            MockLandmark(-0.1, 1.5),                    # Both problematic
        ]
        
        # Add normal landmarks to reach 33 total
        for i in range(30):
            landmarks.append(MockLandmark(0.5, 0.5))
        
        landmark_names = [f'LANDMARK_{i}' for i in range(33)]
        
        keypoint_set = KeypointSet.from_mediapipe(
            landmarks=landmarks,
            frame_width=640,
            frame_height=480,
            landmark_names=landmark_names
        )
        
        # Check that problematic coordinates were clamped
        assert keypoint_set.keypoints[0].x_normalized == 0.0  # Was negative
        assert keypoint_set.keypoints[0].y_normalized == 0.5  # Was valid
        
        assert keypoint_set.keypoints[1].x_normalized == 0.5  # Was valid
        assert keypoint_set.keypoints[1].y_normalized == 1.0  # Was > 1.0
        
        assert keypoint_set.keypoints[2].x_normalized == 0.0  # Was negative
        assert keypoint_set.keypoints[2].y_normalized == 1.0  # Was > 1.0

    def test_edge_case_coordinates(self):
        """Test edge cases for coordinate clamping."""
        test_cases = [
            (0.0, 0.0),      # Exact boundaries
            (1.0, 1.0),      # Exact boundaries
            (-1e-10, 1e-10), # Very small values
            (1.0 + 1e-10, 1.0 - 1e-10),  # Just outside boundaries
        ]
        
        for i, (x_norm, y_norm) in enumerate(test_cases):
            keypoint_data = {
                'id': i,
                'name': f'edge_case_{i}',
                'x': 100.0,
                'y': 200.0,
                'x_normalized': x_norm,
                'y_normalized': y_norm
            }
            
            keypoint = Keypoint.from_dict(keypoint_data)
            
            # Verify coordinates are within valid range
            assert 0.0 <= keypoint.x_normalized <= 1.0
            assert 0.0 <= keypoint.y_normalized <= 1.0

    def test_extreme_values_are_clamped(self):
        """Test that extreme values (inf, -inf) are properly clamped."""
        keypoint_data = {
            'id': 0,
            'name': 'extreme_test',
            'x': 100.0,
            'y': 200.0,
            'x_normalized': float('-inf'),
            'y_normalized': float('inf')
        }
        
        keypoint = Keypoint.from_dict(keypoint_data)
        
        assert keypoint.x_normalized == 0.0
        assert keypoint.y_normalized == 1.0

    def test_original_error_scenario_fixed(self):
        """Test the exact scenario that caused the original error."""
        
        class MockLandmark:
            def __init__(self, x, y, z=0.0, visibility=0.9, presence=0.9):
                self.x = x
                self.y = y
                self.z = z
                self.visibility = visibility
                self.presence = presence
        
        # Create 33 landmarks with the exact problematic coordinate
        landmarks = []
        for i in range(33):
            if i == 0:
                # Use the exact coordinate from the error message
                landmarks.append(MockLandmark(0.5, -0.0019250214099884033))
            else:
                landmarks.append(MockLandmark(0.5, 0.5))
        
        landmark_names = [f'LANDMARK_{i}' for i in range(33)]
        
        # This should not raise the original ValueError
        keypoint_set = KeypointSet.from_mediapipe(
            landmarks=landmarks,
            frame_width=640,
            frame_height=480,
            landmark_names=landmark_names
        )
        
        # Verify the problematic coordinate was clamped
        assert keypoint_set.keypoints[0].y_normalized == 0.0
        assert len(keypoint_set.keypoints) == 33
        assert keypoint_set.format == KeypointFormat.MEDIAPIPE_33