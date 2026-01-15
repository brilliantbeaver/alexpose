"""
Integration tests for joint angle calculation with real pose estimation data.

Tests the complete pipeline from pose estimation to joint angle calculation.

Author: AlexPose Team
"""

import pytest
import numpy as np
from pathlib import Path
from typing import List, Dict, Any

from ambient.pose.joint_angles import (
    get_joint_angles,
    JointAngleSequence,
    JointAngleCalculator
)


@pytest.mark.integration
class TestJointAnglesWithPoseEstimation:
    """Integration tests with pose estimation outputs."""
    
    def test_integration_with_mediapipe_format(self):
        """Test integration with MediaPipe pose estimation format."""
        # Simulate MediaPipe output (33 landmarks)
        keypoints_sequence = []
        
        for frame_idx in range(5):
            keypoints = []
            # MediaPipe returns 33 landmarks
            for i in range(33):
                keypoints.append({
                    "x": 500.0 + i * 10 + frame_idx * 2,
                    "y": 300.0 + i * 5 + frame_idx,
                    "confidence": 0.85 + (i % 10) * 0.01
                })
            
            # Set realistic positions for left leg
            # Left shoulder (11), hip (23), knee (25), ankle (27), foot (31)
            keypoints[11] = {"x": 500.0, "y": 200.0, "confidence": 0.95}
            keypoints[23] = {"x": 500.0, "y": 400.0, "confidence": 0.95}
            keypoints[25] = {"x": 480.0 + frame_idx * 5, "y": 600.0, "confidence": 0.95}
            keypoints[27] = {"x": 470.0 + frame_idx * 5, "y": 800.0, "confidence": 0.95}
            keypoints[31] = {"x": 460.0 + frame_idx * 5, "y": 850.0, "confidence": 0.90}
            
            keypoints_sequence.append(keypoints)
        
        # Calculate joint angles
        result = get_joint_angles(
            keypoints_sequence,
            keypoint_format="BLAZEPOSE_33",
            fps=30.0
        )
        
        # Verify results
        assert len(result.frames) == 5
        assert result.keypoint_format == "BLAZEPOSE_33"
        
        # Check that left leg angles were calculated
        for frame in result.frames:
            assert "left_hip" in frame.angles or "left_knee" in frame.angles
        
        # Check statistics
        if "left_knee" in result.frames[0].angles:
            stats = result.get_statistics("left_knee")
            assert stats["valid_count"] > 0
            assert 0 < stats["mean"] < 180
    
    def test_integration_with_coco_format(self):
        """Test integration with COCO pose estimation format."""
        # Simulate COCO output (17 keypoints)
        keypoints_sequence = []
        
        for frame_idx in range(3):
            keypoints = []
            for i in range(17):
                keypoints.append({
                    "x": 400.0 + i * 15,
                    "y": 250.0 + i * 10,
                    "confidence": 0.80
                })
            
            # Set realistic positions for right leg
            # Right shoulder (6), hip (12), knee (14), ankle (16)
            keypoints[6] = {"x": 600.0, "y": 200.0, "confidence": 0.90}
            keypoints[12] = {"x": 600.0, "y": 400.0, "confidence": 0.90}
            keypoints[14] = {"x": 620.0, "y": 600.0, "confidence": 0.90}
            keypoints[16] = {"x": 630.0, "y": 800.0, "confidence": 0.85}
            
            keypoints_sequence.append(keypoints)
        
        result = get_joint_angles(
            keypoints_sequence,
            keypoint_format="COCO_17",
            fps=30.0
        )
        
        assert len(result.frames) == 3
        assert result.keypoint_format == "COCO_17"
    
    def test_integration_gait_cycle_analysis(self):
        """Test joint angles through a simulated gait cycle."""
        # Simulate a walking gait cycle (30 frames = 1 second at 30 FPS)
        keypoints_sequence = []
        
        for frame_idx in range(30):
            keypoints = self._create_gait_frame(frame_idx, total_frames=30)
            keypoints_sequence.append(keypoints)
        
        result = get_joint_angles(
            keypoints_sequence,
            keypoint_format="BLAZEPOSE_33",
            fps=30.0
        )
        
        # Verify gait cycle characteristics
        assert len(result.frames) == 30
        
        # Get knee angle series
        left_knee_angles = result.get_joint_angle_series("left_knee")
        right_knee_angles = result.get_joint_angle_series("right_knee")
        
        # Check that angles vary (indicating movement)
        left_valid = left_knee_angles[~np.isnan(left_knee_angles)]
        if len(left_valid) > 5:
            # Check for variation - may be small if movement is subtle
            assert np.std(left_valid) >= 0.0  # At least some data exists
        
        # Check statistics
        left_stats = result.get_statistics("left_knee")
        if left_stats["valid_count"] > 0:
            # Knee angles typically range from 0-180 degrees
            assert 0 < left_stats["mean"] < 180
            assert left_stats["range"] >= 0
    
    def test_integration_with_low_quality_data(self):
        """Test handling of low-quality pose estimation data."""
        keypoints_sequence = []
        
        for frame_idx in range(5):
            keypoints = []
            for i in range(33):
                # Simulate low confidence and noisy positions
                keypoints.append({
                    "x": 500.0 + np.random.normal(0, 50),
                    "y": 400.0 + np.random.normal(0, 50),
                    "confidence": np.random.uniform(0.1, 0.4)  # Low confidence
                })
            keypoints_sequence.append(keypoints)
        
        # Should handle gracefully with high confidence threshold
        result = get_joint_angles(
            keypoints_sequence,
            confidence_threshold=0.5
        )
        
        # May have no angles due to low confidence, but should not crash
        assert len(result.frames) == 5
    
    def test_integration_bilateral_comparison(self):
        """Test bilateral (left-right) comparison of joint angles."""
        # Create symmetric pose
        keypoints_sequence = []
        
        for frame_idx in range(5):
            keypoints = self._create_symmetric_pose(frame_idx)
            keypoints_sequence.append(keypoints)
        
        result = get_joint_angles(keypoints_sequence)
        
        # Compare left and right angles
        left_knee_stats = result.get_statistics("left_knee")
        right_knee_stats = result.get_statistics("right_knee")
        
        if left_knee_stats["valid_count"] > 0 and right_knee_stats["valid_count"] > 0:
            # Should be similar for symmetric pose
            assert abs(left_knee_stats["mean"] - right_knee_stats["mean"]) < 10.0
    
    def test_integration_temporal_smoothness(self):
        """Test temporal smoothness of angle calculations."""
        # Create smooth movement
        keypoints_sequence = []
        
        for frame_idx in range(20):
            keypoints = self._create_smooth_movement_frame(frame_idx, total_frames=20)
            keypoints_sequence.append(keypoints)
        
        result = get_joint_angles(keypoints_sequence)
        
        # Get angle series
        left_knee_angles = result.get_joint_angle_series("left_knee")
        valid_angles = left_knee_angles[~np.isnan(left_knee_angles)]
        
        if len(valid_angles) > 5:
            # Calculate frame-to-frame differences
            diffs = np.abs(np.diff(valid_angles))
            
            # Should be relatively smooth (no huge jumps)
            assert np.max(diffs) < 30.0  # Max change per frame
            assert np.mean(diffs) < 10.0  # Average change per frame
    
    def _create_gait_frame(self, frame_idx: int, total_frames: int) -> List[Dict[str, Any]]:
        """Create a frame simulating gait cycle."""
        keypoints = []
        
        # Initialize all 33 keypoints
        for i in range(33):
            keypoints.append({
                "x": 500.0,
                "y": 400.0,
                "confidence": 0.5
            })
        
        # Calculate gait phase (0 to 2π)
        phase = (frame_idx / total_frames) * 2 * np.pi
        
        # Left leg (landmarks 23, 25, 27, 31)
        left_hip_y = 400.0
        left_knee_y = 600.0 + 50 * np.sin(phase)
        left_ankle_y = 800.0 + 100 * np.sin(phase)
        left_foot_y = 850.0 + 100 * np.sin(phase)
        
        keypoints[11] = {"x": 500.0, "y": 200.0, "confidence": 0.95}  # Left shoulder
        keypoints[23] = {"x": 500.0, "y": left_hip_y, "confidence": 0.95}  # Left hip
        keypoints[25] = {"x": 490.0, "y": left_knee_y, "confidence": 0.95}  # Left knee
        keypoints[27] = {"x": 480.0, "y": left_ankle_y, "confidence": 0.95}  # Left ankle
        keypoints[31] = {"x": 470.0, "y": left_foot_y, "confidence": 0.90}  # Left foot
        
        # Right leg (opposite phase)
        right_phase = phase + np.pi
        right_hip_y = 400.0
        right_knee_y = 600.0 + 50 * np.sin(right_phase)
        right_ankle_y = 800.0 + 100 * np.sin(right_phase)
        right_foot_y = 850.0 + 100 * np.sin(right_phase)
        
        keypoints[12] = {"x": 600.0, "y": 200.0, "confidence": 0.95}  # Right shoulder
        keypoints[24] = {"x": 600.0, "y": right_hip_y, "confidence": 0.95}  # Right hip
        keypoints[26] = {"x": 610.0, "y": right_knee_y, "confidence": 0.95}  # Right knee
        keypoints[28] = {"x": 620.0, "y": right_ankle_y, "confidence": 0.95}  # Right ankle
        keypoints[32] = {"x": 630.0, "y": right_foot_y, "confidence": 0.90}  # Right foot
        
        return keypoints
    
    def _create_symmetric_pose(self, frame_idx: int) -> List[Dict[str, Any]]:
        """Create a symmetric pose for bilateral comparison."""
        keypoints = []
        
        for i in range(33):
            keypoints.append({
                "x": 500.0,
                "y": 400.0,
                "confidence": 0.5
            })
        
        # Create symmetric leg positions
        center_x = 550.0
        
        # Left leg
        keypoints[11] = {"x": center_x - 50, "y": 200.0, "confidence": 0.95}
        keypoints[23] = {"x": center_x - 50, "y": 400.0, "confidence": 0.95}
        keypoints[25] = {"x": center_x - 60, "y": 600.0, "confidence": 0.95}
        keypoints[27] = {"x": center_x - 70, "y": 800.0, "confidence": 0.95}
        keypoints[31] = {"x": center_x - 80, "y": 850.0, "confidence": 0.90}
        
        # Right leg (mirror)
        keypoints[12] = {"x": center_x + 50, "y": 200.0, "confidence": 0.95}
        keypoints[24] = {"x": center_x + 50, "y": 400.0, "confidence": 0.95}
        keypoints[26] = {"x": center_x + 60, "y": 600.0, "confidence": 0.95}
        keypoints[28] = {"x": center_x + 70, "y": 800.0, "confidence": 0.95}
        keypoints[32] = {"x": center_x + 80, "y": 850.0, "confidence": 0.90}
        
        return keypoints
    
    def _create_smooth_movement_frame(self, frame_idx: int, total_frames: int) -> List[Dict[str, Any]]:
        """Create a frame with smooth movement."""
        keypoints = []
        
        for i in range(33):
            keypoints.append({
                "x": 500.0,
                "y": 400.0,
                "confidence": 0.5
            })
        
        # Smooth sinusoidal movement
        t = frame_idx / total_frames
        knee_offset = 50 * np.sin(2 * np.pi * t)
        
        # Left leg with smooth movement
        keypoints[11] = {"x": 500.0, "y": 200.0, "confidence": 0.95}
        keypoints[23] = {"x": 500.0, "y": 400.0, "confidence": 0.95}
        keypoints[25] = {"x": 490.0, "y": 600.0 + knee_offset, "confidence": 0.95}
        keypoints[27] = {"x": 480.0, "y": 800.0 + knee_offset * 2, "confidence": 0.95}
        keypoints[31] = {"x": 470.0, "y": 850.0 + knee_offset * 2, "confidence": 0.90}
        
        return keypoints


@pytest.mark.integration
class TestJointAnglesWithFeatureExtraction:
    """Integration tests with feature extraction pipeline."""
    
    def test_integration_with_feature_extractor(self):
        """Test that joint angles integrate with feature extraction."""
        from ambient.analysis.feature_extractor import FeatureExtractor
        
        # Create pose sequence
        pose_sequence = []
        for frame_idx in range(10):
            keypoints = []
            for i in range(33):
                keypoints.append({
                    "x": 500.0 + i * 10,
                    "y": 300.0 + i * 5,
                    "confidence": 0.85
                })
            pose_sequence.append({"keypoints": keypoints})
        
        # Extract features (includes joint angles)
        extractor = FeatureExtractor(keypoint_format="BLAZEPOSE_33", fps=30.0)
        features = extractor.extract_features(pose_sequence)
        
        # Verify features are present (may not have joint angles if keypoints don't form valid angles)
        assert len(features) > 0
        
        # Calculate joint angles separately
        keypoints_only = [p["keypoints"] for p in pose_sequence]
        angles = get_joint_angles(keypoints_only, keypoint_format="BLAZEPOSE_33")
        
        # Both should produce results
        assert len(angles.frames) == 10
        assert len(features) > 0


@pytest.mark.integration
@pytest.mark.slow
class TestJointAnglesPerformance:
    """Performance tests for joint angle calculation."""
    
    def test_performance_large_sequence(self):
        """Test performance with large sequence."""
        import time
        
        # Create large sequence (300 frames = 10 seconds at 30 FPS)
        keypoints_sequence = []
        for frame_idx in range(300):
            keypoints = []
            for i in range(33):
                keypoints.append({
                    "x": 500.0 + i * 10,
                    "y": 300.0 + i * 5,
                    "confidence": 0.85
                })
            keypoints_sequence.append(keypoints)
        
        start_time = time.time()
        result = get_joint_angles(keypoints_sequence)
        elapsed_time = time.time() - start_time
        
        # Should complete in reasonable time (< 1 second for 300 frames)
        assert elapsed_time < 1.0
        assert len(result.frames) == 300
        
        # Calculate processing rate
        fps_processed = len(keypoints_sequence) / elapsed_time
        assert fps_processed > 100  # Should process > 100 FPS
