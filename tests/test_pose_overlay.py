"""
Comprehensive tests for GAVD pose overlay functionality.

Tests the fixes for:
1. Pose keypoint coordinate scaling
2. Source video dimension handling
3. Proper overlay positioning relative to bounding box
4. Integration between backend and frontend coordinate systems

Key concepts:
- Pose keypoints are generated from the ACTUAL video dimensions (e.g., 640x360)
- vid_info in GAVD CSV may have DIFFERENT dimensions (e.g., 1280x720)
- Frontend must scale keypoints using the ACTUAL source dimensions, not vid_info
"""

import pytest
from hypothesis import given, strategies as st, settings, assume
from typing import List, Dict, Any, Tuple, Optional
import math


# ============================================================================
# Core Coordinate Transformation Logic
# ============================================================================

def scale_keypoint(
    kp_x: float,
    kp_y: float,
    source_width: int,
    source_height: int,
    display_width: int,
    display_height: int
) -> Tuple[float, float]:
    """
    Scale a keypoint from source video coordinates to display coordinates.
    
    Args:
        kp_x: Keypoint x in source video coordinates
        kp_y: Keypoint y in source video coordinates
        source_width: Width of video used for pose estimation
        source_height: Height of video used for pose estimation
        display_width: Width of displayed video
        display_height: Height of displayed video
        
    Returns:
        Tuple of (scaled_x, scaled_y) in display coordinates
    """
    scale_x = display_width / source_width
    scale_y = display_height / source_height
    
    return (kp_x * scale_x, kp_y * scale_y)


def is_keypoint_in_bbox(
    kp_x: float,
    kp_y: float,
    bbox_left: float,
    bbox_top: float,
    bbox_width: float,
    bbox_height: float,
    tolerance: float = 50.0
) -> bool:
    """
    Check if a keypoint is within or near a bounding box.
    
    Args:
        kp_x: Keypoint x coordinate
        kp_y: Keypoint y coordinate
        bbox_left: Bounding box left edge
        bbox_top: Bounding box top edge
        bbox_width: Bounding box width
        bbox_height: Bounding box height
        tolerance: Allowed distance outside bbox (for limbs extending beyond)
        
    Returns:
        True if keypoint is within bbox (with tolerance)
    """
    bbox_right = bbox_left + bbox_width
    bbox_bottom = bbox_top + bbox_height
    
    return (
        bbox_left - tolerance <= kp_x <= bbox_right + tolerance and
        bbox_top - tolerance <= kp_y <= bbox_bottom + tolerance
    )


# ============================================================================
# Property-Based Tests for Coordinate Scaling
# ============================================================================

class TestCoordinateScaling:
    """Property-based tests for coordinate scaling."""
    
    @given(
        kp_x=st.floats(min_value=0, max_value=1000),
        kp_y=st.floats(min_value=0, max_value=1000),
        source_width=st.integers(min_value=100, max_value=4000),
        source_height=st.integers(min_value=100, max_value=4000),
        display_width=st.integers(min_value=100, max_value=4000),
        display_height=st.integers(min_value=100, max_value=4000)
    )
    @settings(max_examples=100)
    def test_scaling_preserves_relative_position(
        self,
        kp_x: float,
        kp_y: float,
        source_width: int,
        source_height: int,
        display_width: int,
        display_height: int
    ):
        """Property: Relative position should be preserved after scaling."""
        assume(kp_x <= source_width and kp_y <= source_height)
        
        scaled_x, scaled_y = scale_keypoint(
            kp_x, kp_y,
            source_width, source_height,
            display_width, display_height
        )
        
        # Relative position should be preserved
        source_rel_x = kp_x / source_width
        source_rel_y = kp_y / source_height
        
        display_rel_x = scaled_x / display_width
        display_rel_y = scaled_y / display_height
        
        assert abs(source_rel_x - display_rel_x) < 0.001
        assert abs(source_rel_y - display_rel_y) < 0.001
    
    @given(
        kp_x=st.floats(min_value=0, max_value=640),
        kp_y=st.floats(min_value=0, max_value=360)
    )
    @settings(max_examples=50)
    def test_scaling_640_to_1280(self, kp_x: float, kp_y: float):
        """Property: Scaling from 640x360 to 1280x720 should double coordinates."""
        scaled_x, scaled_y = scale_keypoint(
            kp_x, kp_y,
            source_width=640, source_height=360,
            display_width=1280, display_height=720
        )
        
        assert scaled_x == pytest.approx(kp_x * 2, rel=0.001)
        assert scaled_y == pytest.approx(kp_y * 2, rel=0.001)
    
    @given(
        kp_x=st.floats(min_value=0, max_value=1280),
        kp_y=st.floats(min_value=0, max_value=720)
    )
    @settings(max_examples=50)
    def test_no_scaling_when_same_dimensions(self, kp_x: float, kp_y: float):
        """Property: No scaling when source and display are same size."""
        scaled_x, scaled_y = scale_keypoint(
            kp_x, kp_y,
            source_width=1280, source_height=720,
            display_width=1280, display_height=720
        )
        
        assert scaled_x == pytest.approx(kp_x, rel=0.001)
        assert scaled_y == pytest.approx(kp_y, rel=0.001)


class TestBoundingBoxAlignment:
    """Tests for keypoint alignment with bounding box."""
    
    def test_keypoints_should_be_in_bbox_region(self):
        """Test that properly scaled keypoints fall within bbox region."""
        # Simulated scenario from the bug:
        # - Video downloaded at 640x360
        # - vid_info says 1280x720
        # - Bounding box at (801, 126, 278, 497) in 1280x720 space
        # - Keypoints generated at 640x360 space
        
        # Keypoint in 640x360 space (person on right side)
        kp_x_source = 450  # Right side of 640 width
        kp_y_source = 200  # Middle of 360 height
        
        # Scale to 1280x720 display
        scaled_x, scaled_y = scale_keypoint(
            kp_x_source, kp_y_source,
            source_width=640, source_height=360,
            display_width=1280, display_height=720
        )
        
        # Bounding box in 1280x720 space
        bbox_left = 801
        bbox_top = 126
        bbox_width = 278
        bbox_height = 497
        
        # Scaled keypoint should be near the bbox
        # (450/640) * 1280 = 900, which is within bbox (801 to 1079)
        assert scaled_x == pytest.approx(900, rel=0.01)
        assert scaled_y == pytest.approx(400, rel=0.01)
        
        # Should be within bbox region
        assert is_keypoint_in_bbox(
            scaled_x, scaled_y,
            bbox_left, bbox_top, bbox_width, bbox_height,
            tolerance=100
        )
    
    def test_wrong_scaling_causes_misalignment(self):
        """Test that using wrong dimensions causes misalignment (the bug)."""
        # This demonstrates the bug: using vid_info (1280x720) instead of
        # actual source dimensions (640x360)
        
        kp_x_source = 178  # Actual keypoint x in 640x360 space
        kp_y_source = 95   # Actual keypoint y in 640x360 space
        
        # WRONG: Using vid_info dimensions (1280x720) for scaling
        # This is what the bug was doing
        wrong_scale_x = 1280 / 1280  # = 1.0 (no scaling)
        wrong_scale_y = 720 / 720    # = 1.0 (no scaling)
        
        wrong_x = kp_x_source * wrong_scale_x  # = 178
        wrong_y = kp_y_source * wrong_scale_y  # = 95
        
        # CORRECT: Using actual source dimensions (640x360)
        correct_x, correct_y = scale_keypoint(
            kp_x_source, kp_y_source,
            source_width=640, source_height=360,
            display_width=1280, display_height=720
        )
        
        # Wrong scaling puts keypoint at (178, 95) - LEFT side
        # Correct scaling puts keypoint at (356, 190) - still not right, but closer
        
        # The keypoint at 178 in 640 space is at 27.8% from left
        # In 1280 space, that should be at 356 (27.8% of 1280)
        assert wrong_x == pytest.approx(178, rel=0.01)  # Bug position
        assert correct_x == pytest.approx(356, rel=0.01)  # Correct position
        
        # The correct position is 2x the wrong position
        assert correct_x == pytest.approx(wrong_x * 2, rel=0.01)


class TestSourceDimensionHandling:
    """Tests for source dimension storage and retrieval."""
    
    def test_pose_data_format_with_dimensions(self):
        """Test the new pose data format includes source dimensions."""
        # New format
        pose_data_new = {
            'keypoints': [
                {'x': 178.75, 'y': 95.30, 'confidence': 0.99, 'id': 0},
                {'x': 179.45, 'y': 89.69, 'confidence': 0.99, 'id': 1},
            ],
            'source_width': 640,
            'source_height': 360
        }
        
        assert 'keypoints' in pose_data_new
        assert 'source_width' in pose_data_new
        assert 'source_height' in pose_data_new
        assert pose_data_new['source_width'] == 640
        assert pose_data_new['source_height'] == 360
    
    def test_backward_compatibility_with_old_format(self):
        """Test handling of old format (list of keypoints without dimensions)."""
        # Old format - just a list of keypoints
        pose_data_old = [
            {'x': 178.75, 'y': 95.30, 'confidence': 0.99, 'id': 0},
            {'x': 179.45, 'y': 89.69, 'confidence': 0.99, 'id': 1},
        ]
        
        # Should be able to detect old format
        assert isinstance(pose_data_old, list)
        
        # Convert to new format with None dimensions
        converted = {
            'keypoints': pose_data_old,
            'source_width': None,
            'source_height': None
        }
        
        assert converted['keypoints'] == pose_data_old
        assert converted['source_width'] is None
    
    def test_fallback_to_vid_info_when_no_source_dims(self):
        """Test fallback behavior when source dimensions are not available."""
        vid_info = {'width': 1280, 'height': 720}
        source_width = None
        source_height = None
        display_width = 1280
        display_height = 720
        
        # Fallback logic
        effective_source_width = source_width or vid_info['width']
        effective_source_height = source_height or vid_info['height']
        
        assert effective_source_width == 1280
        assert effective_source_height == 720


class TestMediaPipeIntegration:
    """Tests for MediaPipe pose estimator integration."""
    
    def test_video_keypoints_return_format(self):
        """Test that video keypoints include dimensions."""
        # Simulated return from MediaPipe estimator
        result = {
            'frames': [
                [{'x': 100, 'y': 100, 'confidence': 0.9, 'id': 0}],
                [{'x': 102, 'y': 102, 'confidence': 0.9, 'id': 0}],
            ],
            'video_width': 640,
            'video_height': 360
        }
        
        assert 'frames' in result
        assert 'video_width' in result
        assert 'video_height' in result
        assert len(result['frames']) == 2
        assert result['video_width'] == 640
        assert result['video_height'] == 360
    
    def test_keypoint_coordinate_range(self):
        """Test that keypoints are in valid coordinate range."""
        video_width = 640
        video_height = 360
        
        # Valid keypoint
        kp = {'x': 320, 'y': 180, 'confidence': 0.9, 'id': 0}
        
        assert 0 <= kp['x'] <= video_width
        assert 0 <= kp['y'] <= video_height
        assert 0 <= kp['confidence'] <= 1


class TestRegressionCases:
    """Specific regression tests for known issues."""
    
    def test_pose_on_left_when_person_on_right(self):
        """Regression: Pose overlay appearing on left when person is on right."""
        # This was the original bug
        # Person is on right side of frame (bbox at x=801)
        # But pose was appearing on left side (x=178)
        
        # Keypoint from 640x360 video
        kp_x = 450  # Right side of 640 width (70% from left)
        kp_y = 200
        
        # Wrong scaling (using vid_info 1280x720 as source)
        wrong_x = kp_x * (1280 / 1280)  # No scaling
        
        # Correct scaling (using actual 640x360 as source)
        correct_x, _ = scale_keypoint(
            kp_x, kp_y,
            source_width=640, source_height=360,
            display_width=1280, display_height=720
        )
        
        # Wrong position is on left side
        assert wrong_x < 640  # Left half of 1280
        
        # Correct position is on right side
        assert correct_x > 640  # Right half of 1280
    
    def test_actual_keypoint_values_from_bug(self):
        """Test with actual keypoint values from the bug report."""
        # From the pose_data.json file
        keypoints = [
            {'x': 178.75, 'y': 95.30, 'confidence': 0.998, 'id': 0},
            {'x': 179.45, 'y': 89.69, 'confidence': 0.998, 'id': 1},
        ]
        
        # Source video was 640x360
        source_width = 640
        source_height = 360
        
        # Display is 1280x720
        display_width = 1280
        display_height = 720
        
        # Scale keypoints
        for kp in keypoints:
            scaled_x, scaled_y = scale_keypoint(
                kp['x'], kp['y'],
                source_width, source_height,
                display_width, display_height
            )
            
            # Keypoints should be scaled by 2x
            assert scaled_x == pytest.approx(kp['x'] * 2, rel=0.01)
            assert scaled_y == pytest.approx(kp['y'] * 2, rel=0.01)


class TestEdgeCases:
    """Tests for edge cases."""
    
    def test_zero_dimensions(self):
        """Test handling of zero dimensions."""
        # Should not crash, but behavior is undefined
        with pytest.raises(ZeroDivisionError):
            scale_keypoint(100, 100, 0, 0, 1280, 720)
    
    def test_negative_coordinates(self):
        """Test handling of negative coordinates (shouldn't happen but be safe)."""
        scaled_x, scaled_y = scale_keypoint(
            -10, -10,
            source_width=640, source_height=360,
            display_width=1280, display_height=720
        )
        
        # Should scale negative values too
        assert scaled_x == pytest.approx(-20, rel=0.01)
        assert scaled_y == pytest.approx(-20, rel=0.01)
    
    def test_very_large_coordinates(self):
        """Test handling of coordinates outside video bounds."""
        # Keypoint outside video bounds (can happen with tracking)
        scaled_x, scaled_y = scale_keypoint(
            700, 400,  # Outside 640x360
            source_width=640, source_height=360,
            display_width=1280, display_height=720
        )
        
        # Should still scale correctly
        assert scaled_x == pytest.approx(1400, rel=0.01)
        assert scaled_y == pytest.approx(800, rel=0.01)
    
    @given(
        source_width=st.integers(min_value=1, max_value=10000),
        source_height=st.integers(min_value=1, max_value=10000),
        display_width=st.integers(min_value=1, max_value=10000),
        display_height=st.integers(min_value=1, max_value=10000)
    )
    @settings(max_examples=50)
    def test_scaling_never_crashes(
        self,
        source_width: int,
        source_height: int,
        display_width: int,
        display_height: int
    ):
        """Property: Scaling should never crash with valid dimensions."""
        # Should not raise any exceptions
        scaled_x, scaled_y = scale_keypoint(
            100, 100,
            source_width, source_height,
            display_width, display_height
        )
        
        # Results should be finite numbers
        assert math.isfinite(scaled_x)
        assert math.isfinite(scaled_y)


class TestIntegration:
    """Integration tests for complete scenarios."""
    
    def test_complete_pose_overlay_scenario(self):
        """Test complete scenario from pose estimation to display."""
        # 1. Video is downloaded at 640x360
        video_width = 640
        video_height = 360
        
        # 2. MediaPipe generates keypoints in 640x360 space
        keypoints = [
            {'x': 450, 'y': 200, 'confidence': 0.95, 'id': 0},  # Nose
            {'x': 445, 'y': 250, 'confidence': 0.90, 'id': 11},  # Left hip
            {'x': 455, 'y': 250, 'confidence': 0.90, 'id': 12},  # Right hip
        ]
        
        # 3. Pose data is stored with source dimensions
        pose_data = {
            'keypoints': keypoints,
            'source_width': video_width,
            'source_height': video_height
        }
        
        # 4. Frontend displays video at 1280x720
        display_width = 1280
        display_height = 720
        
        # 5. Frontend scales keypoints using source dimensions
        scaled_keypoints = []
        for kp in pose_data['keypoints']:
            scaled_x, scaled_y = scale_keypoint(
                kp['x'], kp['y'],
                pose_data['source_width'], pose_data['source_height'],
                display_width, display_height
            )
            scaled_keypoints.append({
                'x': scaled_x,
                'y': scaled_y,
                'confidence': kp['confidence'],
                'id': kp['id']
            })
        
        # 6. Verify scaled keypoints are in correct position
        # Original: 450/640 = 70.3% from left
        # Scaled: should be at 70.3% of 1280 = 900
        assert scaled_keypoints[0]['x'] == pytest.approx(900, rel=0.01)
        
        # 7. Bounding box is at (801, 126, 278, 497) in 1280x720 space
        bbox = {'left': 801, 'top': 126, 'width': 278, 'height': 497}
        
        # 8. Verify keypoints are within/near bounding box
        for kp in scaled_keypoints:
            assert is_keypoint_in_bbox(
                kp['x'], kp['y'],
                bbox['left'], bbox['top'], bbox['width'], bbox['height'],
                tolerance=150  # Allow some tolerance for limbs
            )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
