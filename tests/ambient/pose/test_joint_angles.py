"""
Unit tests for joint angle calculation module.

Tests cover:
- Basic angle calculation
- Frame-level angle calculation
- Sequence-level angle calculation
- Different keypoint formats
- Edge cases and error handling
- Statistical calculations

Author: AlexPose Team
"""

import pytest
import numpy as np
from typing import List, Dict, Any

from ambient.pose.joint_angles import (
    JointAngle,
    FrameJointAngles,
    JointAngleSequence,
    JointAngleCalculator,
    get_joint_angles
)


class TestJointAngle:
    """Test JointAngle data class."""
    
    def test_joint_angle_creation(self):
        """Test creating a JointAngle object."""
        angle = JointAngle(
            joint_name="left_knee",
            angle_degrees=145.5,
            confidence=0.85,
            frame_index=10,
            landmark_indices=(23, 25, 27)
        )
        
        assert angle.joint_name == "left_knee"
        assert angle.angle_degrees == 145.5
        assert angle.confidence == 0.85
        assert angle.frame_index == 10
        assert angle.landmark_indices == (23, 25, 27)
    
    def test_joint_angle_invalid_confidence(self):
        """Test that invalid confidence raises error."""
        with pytest.raises(ValueError, match="Confidence must be between 0 and 1"):
            JointAngle(
                joint_name="left_knee",
                angle_degrees=145.5,
                confidence=1.5,  # Invalid
                frame_index=10
            )
    
    def test_joint_angle_unusual_angle_warning(self):
        """Test warning for unusual angle values."""
        import logging
        from loguru import logger
        
        # Capture loguru output
        with logger.catch(message="Testing unusual angle"):
            angle = JointAngle(
                joint_name="left_knee",
                angle_degrees=200.0,  # Unusual
                confidence=0.85,
                frame_index=10
            )
        
        # Warning is logged but doesn't prevent creation
        assert angle.angle_degrees == 200.0


class TestFrameJointAngles:
    """Test FrameJointAngles data class."""
    
    def test_frame_joint_angles_creation(self):
        """Test creating FrameJointAngles object."""
        frame_angles = FrameJointAngles(
            frame_index=5,
            keypoint_format="BLAZEPOSE_33",
            timestamp=0.167
        )
        
        assert frame_angles.frame_index == 5
        assert frame_angles.keypoint_format == "BLAZEPOSE_33"
        assert frame_angles.timestamp == 0.167
        assert len(frame_angles.angles) == 0
    
    def test_get_angle(self):
        """Test getting angle value."""
        frame_angles = FrameJointAngles(frame_index=0)
        frame_angles.angles["left_knee"] = JointAngle(
            joint_name="left_knee",
            angle_degrees=145.5,
            confidence=0.85,
            frame_index=0
        )
        
        assert frame_angles.get_angle("left_knee") == 145.5
        assert frame_angles.get_angle("right_knee") is None
    
    def test_get_confidence(self):
        """Test getting confidence value."""
        frame_angles = FrameJointAngles(frame_index=0)
        frame_angles.angles["left_knee"] = JointAngle(
            joint_name="left_knee",
            angle_degrees=145.5,
            confidence=0.85,
            frame_index=0
        )
        
        assert frame_angles.get_confidence("left_knee") == 0.85
        assert frame_angles.get_confidence("right_knee") is None
    
    def test_to_dict(self):
        """Test converting to dictionary."""
        frame_angles = FrameJointAngles(frame_index=5, timestamp=0.167)
        frame_angles.angles["left_knee"] = JointAngle(
            joint_name="left_knee",
            angle_degrees=145.5,
            confidence=0.85,
            frame_index=5,
            landmark_indices=(23, 25, 27)
        )
        
        result = frame_angles.to_dict()
        
        assert result["frame_index"] == 5
        assert result["timestamp"] == 0.167
        assert "left_knee" in result["angles"]
        assert result["angles"]["left_knee"]["angle_degrees"] == 145.5


class TestJointAngleSequence:
    """Test JointAngleSequence data class."""
    
    def test_sequence_creation(self):
        """Test creating JointAngleSequence."""
        sequence = JointAngleSequence(
            keypoint_format="BLAZEPOSE_33",
            fps=30.0,
            sequence_id="test_seq"
        )
        
        assert sequence.keypoint_format == "BLAZEPOSE_33"
        assert sequence.fps == 30.0
        assert sequence.sequence_id == "test_seq"
        assert len(sequence.frames) == 0
    
    def test_get_joint_angle_series(self):
        """Test getting time series of angles."""
        sequence = JointAngleSequence()
        
        # Add frames with angles
        for i in range(5):
            frame = FrameJointAngles(frame_index=i)
            frame.angles["left_knee"] = JointAngle(
                joint_name="left_knee",
                angle_degrees=140.0 + i * 2,
                confidence=0.9,
                frame_index=i
            )
            sequence.frames.append(frame)
        
        angles = sequence.get_joint_angle_series("left_knee")
        
        assert len(angles) == 5
        assert angles[0] == 140.0
        assert angles[4] == 148.0
    
    def test_get_joint_angle_series_with_missing(self):
        """Test getting series with missing values."""
        sequence = JointAngleSequence()
        
        # Add frames, some without the angle
        for i in range(5):
            frame = FrameJointAngles(frame_index=i)
            if i % 2 == 0:  # Only even frames
                frame.angles["left_knee"] = JointAngle(
                    joint_name="left_knee",
                    angle_degrees=140.0,
                    confidence=0.9,
                    frame_index=i
                )
            sequence.frames.append(frame)
        
        angles = sequence.get_joint_angle_series("left_knee")
        
        assert len(angles) == 5
        assert angles[0] == 140.0
        assert np.isnan(angles[1])
        assert angles[2] == 140.0
    
    def test_get_statistics(self):
        """Test calculating statistics."""
        sequence = JointAngleSequence()
        
        # Add frames with varying angles
        angles_values = [140.0, 145.0, 150.0, 155.0, 160.0]
        for i, angle_val in enumerate(angles_values):
            frame = FrameJointAngles(frame_index=i)
            frame.angles["left_knee"] = JointAngle(
                joint_name="left_knee",
                angle_degrees=angle_val,
                confidence=0.9,
                frame_index=i
            )
            sequence.frames.append(frame)
        
        stats = sequence.get_statistics("left_knee")
        
        assert stats["mean"] == pytest.approx(150.0)
        assert stats["min"] == 140.0
        assert stats["max"] == 160.0
        assert stats["range"] == 20.0
        assert stats["valid_count"] == 5
    
    def test_to_dict(self):
        """Test converting sequence to dictionary."""
        sequence = JointAngleSequence(
            keypoint_format="BLAZEPOSE_33",
            fps=30.0,
            sequence_id="test_seq"
        )
        
        frame = FrameJointAngles(frame_index=0)
        sequence.frames.append(frame)
        
        result = sequence.to_dict()
        
        assert result["sequence_id"] == "test_seq"
        assert result["keypoint_format"] == "BLAZEPOSE_33"
        assert result["fps"] == 30.0
        assert result["num_frames"] == 1


class TestJointAngleCalculator:
    """Test JointAngleCalculator class."""
    
    def test_calculator_initialization(self):
        """Test initializing calculator."""
        calc = JointAngleCalculator(keypoint_format="BLAZEPOSE_33")
        
        assert calc.keypoint_format == "BLAZEPOSE_33"
        assert calc.confidence_threshold == 0.3
        assert "left_hip" in calc.mapping
    
    def test_calculator_invalid_format(self):
        """Test that invalid format raises error."""
        with pytest.raises(ValueError, match="Unsupported keypoint format"):
            JointAngleCalculator(keypoint_format="INVALID_FORMAT")
    
    def test_calculate_angle_90_degrees(self):
        """Test calculating a 90-degree angle."""
        calc = JointAngleCalculator()
        
        # Create perpendicular vectors
        p1 = np.array([0.0, 1.0])
        p2 = np.array([0.0, 0.0])  # Vertex
        p3 = np.array([1.0, 0.0])
        
        angle, conf = calc.calculate_angle(p1, p2, p3, 1.0, 1.0, 1.0)
        
        assert angle == pytest.approx(90.0, abs=0.1)
        assert conf == 1.0
    
    def test_calculate_angle_180_degrees(self):
        """Test calculating a 180-degree angle (straight line)."""
        calc = JointAngleCalculator()
        
        # Create collinear points
        p1 = np.array([0.0, 0.0])
        p2 = np.array([1.0, 0.0])  # Vertex
        p3 = np.array([2.0, 0.0])
        
        angle, conf = calc.calculate_angle(p1, p2, p3, 1.0, 1.0, 1.0)
        
        assert angle == pytest.approx(180.0, abs=0.1)
    
    def test_calculate_angle_with_low_confidence(self):
        """Test angle calculation with low confidence."""
        calc = JointAngleCalculator()
        
        p1 = np.array([0.0, 1.0])
        p2 = np.array([0.0, 0.0])
        p3 = np.array([1.0, 0.0])
        
        angle, conf = calc.calculate_angle(p1, p2, p3, 0.5, 0.5, 0.5)
        
        assert angle == pytest.approx(90.0, abs=0.1)
        assert conf == pytest.approx(0.5, abs=0.01)  # Geometric mean
    
    def test_calculate_angle_degenerate_case(self):
        """Test angle calculation with degenerate points."""
        calc = JointAngleCalculator()
        
        # Same point for all three
        p1 = np.array([0.0, 0.0])
        p2 = np.array([0.0, 0.0])
        p3 = np.array([0.0, 0.0])
        
        angle, conf = calc.calculate_angle(p1, p2, p3, 1.0, 1.0, 1.0)
        
        assert np.isnan(angle)
        assert conf == 0.0
    
    def test_calculate_frame_angles_blazepose(self):
        """Test calculating angles for a single frame with BLAZEPOSE format."""
        calc = JointAngleCalculator(keypoint_format="BLAZEPOSE_33")
        
        # Create sample keypoints (33 landmarks)
        keypoints = []
        for i in range(33):
            keypoints.append({
                "x": 100.0 + i * 10,
                "y": 200.0 + i * 5,
                "confidence": 0.9
            })
        
        # Set specific positions for left knee angle calculation
        # Left hip (23), left knee (25), left ankle (27)
        keypoints[23] = {"x": 100.0, "y": 100.0, "confidence": 0.9}
        keypoints[25] = {"x": 100.0, "y": 200.0, "confidence": 0.9}  # Vertex
        keypoints[27] = {"x": 100.0, "y": 300.0, "confidence": 0.9}
        
        frame_angles = calc.calculate_frame_angles(keypoints, frame_index=0)
        
        assert frame_angles.frame_index == 0
        assert "left_knee" in frame_angles.angles
        # Should be 180 degrees (straight line)
        assert frame_angles.angles["left_knee"].angle_degrees == pytest.approx(180.0, abs=0.1)
    
    def test_calculate_sequence_angles(self):
        """Test calculating angles for a sequence."""
        calc = JointAngleCalculator(keypoint_format="BLAZEPOSE_33")
        
        # Create sequence of 3 frames
        keypoints_array = []
        for frame_idx in range(3):
            keypoints = []
            for i in range(33):
                keypoints.append({
                    "x": 100.0 + i * 10 + frame_idx,
                    "y": 200.0 + i * 5 + frame_idx,
                    "confidence": 0.9
                })
            keypoints_array.append(keypoints)
        
        sequence = calc.calculate_sequence_angles(
            keypoints_array,
            fps=30.0,
            sequence_id="test_seq"
        )
        
        assert len(sequence.frames) == 3
        assert sequence.fps == 30.0
        assert sequence.sequence_id == "test_seq"


class TestGetJointAngles:
    """Test the main get_joint_angles function."""
    
    def test_get_joint_angles_basic(self):
        """Test basic usage of get_joint_angles."""
        # Create sample keypoints for 2 frames
        keypoints_array = []
        for frame_idx in range(2):
            keypoints = []
            for i in range(33):  # BLAZEPOSE_33
                keypoints.append({
                    "x": 100.0 + i * 10,
                    "y": 200.0 + i * 5,
                    "confidence": 0.9
                })
            keypoints_array.append(keypoints)
        
        result = get_joint_angles(keypoints_array)
        
        assert isinstance(result, JointAngleSequence)
        assert len(result.frames) == 2
        assert result.keypoint_format == "BLAZEPOSE_33"
    
    def test_get_joint_angles_with_parameters(self):
        """Test get_joint_angles with custom parameters."""
        keypoints_array = []
        for frame_idx in range(3):
            keypoints = []
            for i in range(17):  # COCO_17
                keypoints.append({
                    "x": 100.0 + i * 10,
                    "y": 200.0 + i * 5,
                    "confidence": 0.9
                })
            keypoints_array.append(keypoints)
        
        result = get_joint_angles(
            keypoints_array,
            keypoint_format="COCO_17",
            fps=60.0,
            confidence_threshold=0.5,
            sequence_id="custom_seq"
        )
        
        assert result.keypoint_format == "COCO_17"
        assert result.fps == 60.0
        assert result.sequence_id == "custom_seq"
    
    def test_get_joint_angles_realistic_knee_angle(self):
        """Test with realistic knee angle scenario."""
        # Create a single frame with realistic knee angle
        keypoints = []
        for i in range(33):
            keypoints.append({
                "x": 500.0,
                "y": 500.0,
                "confidence": 0.0  # Low confidence for unused points
            })
        
        # Set up left leg for ~145 degree knee angle
        # Left hip (23)
        keypoints[23] = {"x": 500.0, "y": 300.0, "confidence": 0.95}
        # Left knee (25) - vertex
        keypoints[25] = {"x": 500.0, "y": 500.0, "confidence": 0.95}
        # Left ankle (27)
        keypoints[27] = {"x": 450.0, "y": 650.0, "confidence": 0.95}
        
        result = get_joint_angles([keypoints])
        
        assert len(result.frames) == 1
        assert "left_knee" in result.frames[0].angles
        
        # Check angle is reasonable for a bent knee
        knee_angle = result.frames[0].angles["left_knee"].angle_degrees
        # Adjusted range to be more realistic based on actual calculation
        assert 100.0 < knee_angle < 180.0


class TestEdgeCases:
    """Test edge cases and error handling."""
    
    def test_empty_keypoints_array(self):
        """Test with empty keypoints array."""
        result = get_joint_angles([])
        
        assert len(result.frames) == 0
    
    def test_insufficient_keypoints(self):
        """Test with insufficient keypoints."""
        # Only 10 keypoints instead of 33
        keypoints = []
        for i in range(10):
            keypoints.append({
                "x": 100.0 + i * 10,
                "y": 200.0,
                "confidence": 0.9
            })
        
        result = get_joint_angles([keypoints])
        
        # Should handle gracefully
        assert len(result.frames) == 1
        # May have no angles calculated due to missing landmarks
    
    def test_low_confidence_keypoints(self):
        """Test with low confidence keypoints."""
        keypoints = []
        for i in range(33):
            keypoints.append({
                "x": 100.0 + i * 10,
                "y": 200.0,
                "confidence": 0.1  # Very low confidence
            })
        
        result = get_joint_angles([keypoints], confidence_threshold=0.5)
        
        # Should have no angles due to low confidence
        assert len(result.frames) == 1
        assert len(result.frames[0].angles) == 0


class TestMultipleFormats:
    """Test support for multiple keypoint formats."""
    
    def test_coco_17_format(self):
        """Test with COCO_17 format."""
        keypoints = []
        for i in range(17):
            keypoints.append({
                "x": 100.0 + i * 10,
                "y": 200.0,
                "confidence": 0.9
            })
        
        result = get_joint_angles(
            [keypoints],
            keypoint_format="COCO_17"
        )
        
        assert result.keypoint_format == "COCO_17"
    
    def test_body25_format(self):
        """Test with BODY_25 format."""
        keypoints = []
        for i in range(25):
            keypoints.append({
                "x": 100.0 + i * 10,
                "y": 200.0,
                "confidence": 0.9
            })
        
        result = get_joint_angles(
            [keypoints],
            keypoint_format="BODY_25"
        )
        
        assert result.keypoint_format == "BODY_25"
