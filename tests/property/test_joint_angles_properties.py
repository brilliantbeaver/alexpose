"""
Property-based tests for joint angle calculation.

Uses Hypothesis to test mathematical properties and invariants of joint angle calculations.

Author: AlexPose Team
"""

import pytest
import numpy as np
from hypothesis import given, strategies as st, settings, assume
from typing import List, Dict, Any

from ambient.pose.joint_angles import (
    JointAngleCalculator,
    get_joint_angles,
    JointAngleSequence
)


# Strategies for generating test data
@st.composite
def keypoint_strategy(draw):
    """Generate a single keypoint."""
    return {
        "x": draw(st.floats(min_value=0, max_value=1920, allow_nan=False, allow_infinity=False)),
        "y": draw(st.floats(min_value=0, max_value=1080, allow_nan=False, allow_infinity=False)),
        "confidence": draw(st.floats(min_value=0.0, max_value=1.0))
    }


@st.composite
def keypoints_frame_strategy(draw, num_keypoints=33):
    """Generate a frame of keypoints."""
    return [draw(keypoint_strategy()) for _ in range(num_keypoints)]


@st.composite
def keypoints_sequence_strategy(draw, min_frames=1, max_frames=10, num_keypoints=33):
    """Generate a sequence of keypoint frames."""
    num_frames = draw(st.integers(min_value=min_frames, max_value=max_frames))
    return [draw(keypoints_frame_strategy(num_keypoints)) for _ in range(num_frames)]


class TestJointAngleProperties:
    """Property-based tests for joint angle calculations."""
    
    @given(
        p1_x=st.floats(min_value=-1000, max_value=1000, allow_nan=False, allow_infinity=False),
        p1_y=st.floats(min_value=-1000, max_value=1000, allow_nan=False, allow_infinity=False),
        p2_x=st.floats(min_value=-1000, max_value=1000, allow_nan=False, allow_infinity=False),
        p2_y=st.floats(min_value=-1000, max_value=1000, allow_nan=False, allow_infinity=False),
        p3_x=st.floats(min_value=-1000, max_value=1000, allow_nan=False, allow_infinity=False),
        p3_y=st.floats(min_value=-1000, max_value=1000, allow_nan=False, allow_infinity=False),
    )
    @settings(max_examples=100, deadline=None)
    def test_angle_range_property(self, p1_x, p1_y, p2_x, p2_y, p3_x, p3_y):
        """
        Property: All calculated angles must be between 0 and 180 degrees.
        
        This is a fundamental geometric property - the angle between two vectors
        is always in the range [0, 180] degrees.
        """
        calc = JointAngleCalculator()
        
        p1 = np.array([p1_x, p1_y])
        p2 = np.array([p2_x, p2_y])
        p3 = np.array([p3_x, p3_y])
        
        # Skip degenerate cases
        if np.linalg.norm(p1 - p2) < 1e-6 or np.linalg.norm(p3 - p2) < 1e-6:
            return
        
        angle, conf = calc.calculate_angle(p1, p2, p3, 1.0, 1.0, 1.0)
        
        if not np.isnan(angle):
            assert 0.0 <= angle <= 180.0, f"Angle {angle} outside valid range"
    
    @given(
        x=st.floats(min_value=0, max_value=1000, allow_nan=False, allow_infinity=False),
        y=st.floats(min_value=0, max_value=1000, allow_nan=False, allow_infinity=False),
        offset=st.floats(min_value=1, max_value=100, allow_nan=False, allow_infinity=False)
    )
    @settings(max_examples=50, deadline=None)
    def test_straight_line_180_degrees_property(self, x, y, offset):
        """
        Property: Three collinear points should produce a 180-degree angle.
        
        When three points lie on a straight line, the angle at the middle point
        should be exactly 180 degrees.
        """
        calc = JointAngleCalculator()
        
        # Create three collinear points
        p1 = np.array([x, y])
        p2 = np.array([x + offset, y])
        p3 = np.array([x + 2 * offset, y])
        
        angle, conf = calc.calculate_angle(p1, p2, p3, 1.0, 1.0, 1.0)
        
        assert angle == pytest.approx(180.0, abs=0.1), f"Collinear points should give 180°, got {angle}°"
    
    @given(
        x=st.floats(min_value=0, max_value=1000, allow_nan=False, allow_infinity=False),
        y=st.floats(min_value=0, max_value=1000, allow_nan=False, allow_infinity=False),
        offset=st.floats(min_value=1, max_value=100, allow_nan=False, allow_infinity=False)
    )
    @settings(max_examples=50, deadline=None)
    def test_perpendicular_90_degrees_property(self, x, y, offset):
        """
        Property: Perpendicular vectors should produce a 90-degree angle.
        
        When two vectors are perpendicular, the angle between them should be
        exactly 90 degrees.
        """
        calc = JointAngleCalculator()
        
        # Create perpendicular vectors
        p1 = np.array([x, y + offset])
        p2 = np.array([x, y])  # Vertex
        p3 = np.array([x + offset, y])
        
        angle, conf = calc.calculate_angle(p1, p2, p3, 1.0, 1.0, 1.0)
        
        assert angle == pytest.approx(90.0, abs=0.1), f"Perpendicular vectors should give 90°, got {angle}°"
    
    @given(
        conf1=st.floats(min_value=0.0, max_value=1.0),
        conf2=st.floats(min_value=0.0, max_value=1.0),
        conf3=st.floats(min_value=0.0, max_value=1.0)
    )
    @settings(max_examples=50, deadline=None)
    def test_confidence_combination_property(self, conf1, conf2, conf3):
        """
        Property: Combined confidence should be geometric mean of individual confidences.
        
        The combined confidence is the geometric mean of all three confidences.
        It will be between the minimum and maximum individual confidence.
        """
        calc = JointAngleCalculator()
        
        # Use fixed perpendicular points
        p1 = np.array([0.0, 1.0])
        p2 = np.array([0.0, 0.0])
        p3 = np.array([1.0, 0.0])
        
        angle, combined_conf = calc.calculate_angle(p1, p2, p3, conf1, conf2, conf3)
        
        # Combined confidence should be geometric mean
        expected_conf = (conf1 * conf2 * conf3) ** (1/3)
        assert combined_conf == pytest.approx(expected_conf, abs=0.01)
        
        # Combined confidence should be between min and max
        min_conf = min(conf1, conf2, conf3)
        max_conf = max(conf1, conf2, conf3)
        assert min_conf - 0.01 <= combined_conf <= max_conf + 0.01
    
    @given(keypoints_sequence=keypoints_sequence_strategy(min_frames=1, max_frames=5))
    @settings(max_examples=20, deadline=None)
    def test_sequence_length_preservation_property(self, keypoints_sequence):
        """
        Property: Output sequence length should match input sequence length.
        
        For any valid input sequence, the number of output frames should equal
        the number of input frames.
        """
        result = get_joint_angles(keypoints_sequence)
        
        assert len(result.frames) == len(keypoints_sequence)
    
    @given(
        keypoints_frame=keypoints_frame_strategy(num_keypoints=33),
        frame_index=st.integers(min_value=0, max_value=100)
    )
    @settings(max_examples=30, deadline=None)
    def test_frame_index_consistency_property(self, keypoints_frame, frame_index):
        """
        Property: Frame indices should be preserved correctly.
        
        The frame index in the output should match the frame index in the input.
        """
        calc = JointAngleCalculator(keypoint_format="BLAZEPOSE_33")
        
        frame_angles = calc.calculate_frame_angles(keypoints_frame, frame_index=frame_index)
        
        assert frame_angles.frame_index == frame_index
        
        # All angles in the frame should have the same frame index
        for angle in frame_angles.angles.values():
            assert angle.frame_index == frame_index
    
    @given(keypoints_sequence=keypoints_sequence_strategy(min_frames=2, max_frames=10))
    @settings(max_examples=20, deadline=None)
    def test_statistics_consistency_property(self, keypoints_sequence):
        """
        Property: Statistics should be consistent with raw angle values.
        
        The mean, min, max calculated from statistics should match direct
        calculation from the angle series.
        """
        result = get_joint_angles(keypoints_sequence)
        
        # Check for any joint that has angles
        for frame in result.frames:
            if frame.angles:
                joint_name = list(frame.angles.keys())[0]
                break
        else:
            # No angles calculated, skip test
            return
        
        # Get statistics
        stats = result.get_statistics(joint_name)
        
        # Get raw angles
        angles = result.get_joint_angle_series(joint_name)
        valid_angles = angles[~np.isnan(angles)]
        
        if len(valid_angles) > 0:
            assert stats["mean"] == pytest.approx(np.mean(valid_angles), abs=0.01)
            assert stats["min"] == pytest.approx(np.min(valid_angles), abs=0.01)
            assert stats["max"] == pytest.approx(np.max(valid_angles), abs=0.01)
            assert stats["valid_count"] == len(valid_angles)
    
    @given(
        keypoints_sequence=keypoints_sequence_strategy(min_frames=1, max_frames=5),
        confidence_threshold=st.floats(min_value=0.0, max_value=1.0)
    )
    @settings(max_examples=20, deadline=None)
    def test_confidence_threshold_filtering_property(self, keypoints_sequence, confidence_threshold):
        """
        Property: Angles below confidence threshold should be filtered out.
        
        All angles in the output should have confidence >= threshold.
        """
        result = get_joint_angles(
            keypoints_sequence,
            confidence_threshold=confidence_threshold
        )
        
        for frame in result.frames:
            for angle in frame.angles.values():
                assert angle.confidence >= confidence_threshold - 0.01  # Small tolerance
    
    @given(keypoints_sequence=keypoints_sequence_strategy(min_frames=3, max_frames=10))
    @settings(max_examples=15, deadline=None)
    def test_temporal_ordering_property(self, keypoints_sequence):
        """
        Property: Frame indices should be monotonically increasing.
        
        Frames in the sequence should maintain temporal order.
        """
        result = get_joint_angles(keypoints_sequence)
        
        frame_indices = [frame.frame_index for frame in result.frames]
        
        # Check monotonic increase
        for i in range(len(frame_indices) - 1):
            assert frame_indices[i] < frame_indices[i + 1]
    
    @given(
        keypoints_frame=keypoints_frame_strategy(num_keypoints=33),
        fps=st.floats(min_value=1.0, max_value=120.0)
    )
    @settings(max_examples=20, deadline=None)
    def test_timestamp_calculation_property(self, keypoints_frame, fps):
        """
        Property: Timestamps should be correctly calculated from frame index and FPS.
        
        timestamp = frame_index / fps
        """
        calc = JointAngleCalculator(keypoint_format="BLAZEPOSE_33")
        
        frame_index = 10
        frame_angles = calc.calculate_frame_angles(
            keypoints_frame,
            frame_index=frame_index,
            timestamp=frame_index / fps
        )
        
        expected_timestamp = frame_index / fps
        assert frame_angles.timestamp == pytest.approx(expected_timestamp, abs=0.001)


class TestSymmetryProperties:
    """Property-based tests for symmetry in joint angle calculations."""
    
    @given(
        x=st.floats(min_value=100, max_value=900, allow_nan=False, allow_infinity=False),
        y=st.floats(min_value=100, max_value=900, allow_nan=False, allow_infinity=False),
        offset=st.floats(min_value=10, max_value=100, allow_nan=False, allow_infinity=False)
    )
    @settings(max_examples=30, deadline=None)
    def test_angle_symmetry_property(self, x, y, offset):
        """
        Property: Angle calculation should be symmetric.
        
        Swapping p1 and p3 should give the same angle (angle is independent of
        which arm of the angle we consider first).
        """
        calc = JointAngleCalculator()
        
        p1 = np.array([x - offset, y])
        p2 = np.array([x, y])  # Vertex
        p3 = np.array([x + offset, y + offset])
        
        angle1, _ = calc.calculate_angle(p1, p2, p3, 1.0, 1.0, 1.0)
        angle2, _ = calc.calculate_angle(p3, p2, p1, 1.0, 1.0, 1.0)
        
        if not np.isnan(angle1) and not np.isnan(angle2):
            assert angle1 == pytest.approx(angle2, abs=0.01)


class TestRobustnessProperties:
    """Property-based tests for robustness and error handling."""
    
    @given(keypoints_sequence=keypoints_sequence_strategy(min_frames=1, max_frames=5))
    @settings(max_examples=20, deadline=None)
    def test_no_exceptions_property(self, keypoints_sequence):
        """
        Property: Valid input should never raise exceptions.
        
        For any valid keypoint sequence, the function should complete without
        raising exceptions (though it may return empty results).
        """
        try:
            result = get_joint_angles(keypoints_sequence)
            assert isinstance(result, JointAngleSequence)
        except Exception as e:
            pytest.fail(f"Unexpected exception: {e}")
    
    @given(
        num_keypoints=st.integers(min_value=0, max_value=50),
        num_frames=st.integers(min_value=0, max_value=10)
    )
    @settings(max_examples=30, deadline=None)
    def test_variable_keypoint_count_property(self, num_keypoints, num_frames):
        """
        Property: Should handle variable keypoint counts gracefully.
        
        The function should work with any number of keypoints (though results
        may be empty if insufficient keypoints are provided).
        """
        keypoints_sequence = []
        for _ in range(num_frames):
            frame = []
            for _ in range(num_keypoints):
                frame.append({
                    "x": 500.0,
                    "y": 500.0,
                    "confidence": 0.9
                })
            keypoints_sequence.append(frame)
        
        try:
            result = get_joint_angles(keypoints_sequence)
            assert len(result.frames) == num_frames
        except Exception as e:
            pytest.fail(f"Should handle variable keypoint counts: {e}")


@pytest.mark.property
class TestJointAngleInvariants:
    """Test mathematical invariants of joint angle calculations."""
    
    def test_angle_sum_triangle_invariant(self):
        """
        Invariant: For any triangle, sum of angles should be 180 degrees.
        
        This is a fundamental geometric invariant.
        """
        calc = JointAngleCalculator()
        
        # Create a triangle
        p1 = np.array([0.0, 0.0])
        p2 = np.array([1.0, 0.0])
        p3 = np.array([0.5, 0.866])  # Equilateral triangle
        
        # Calculate all three angles
        angle1, _ = calc.calculate_angle(p2, p1, p3, 1.0, 1.0, 1.0)
        angle2, _ = calc.calculate_angle(p1, p2, p3, 1.0, 1.0, 1.0)
        angle3, _ = calc.calculate_angle(p1, p3, p2, 1.0, 1.0, 1.0)
        
        total = angle1 + angle2 + angle3
        assert total == pytest.approx(180.0, abs=0.5)
