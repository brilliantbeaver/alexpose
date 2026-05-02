"""
Unit tests for enhanced feature extraction integration.

This test suite validates the complete integration between the enhanced
analyzer components and the GaitFeatureVector class.

Author: AlexPose Team
"""

import pytest
import numpy as np
from typing import Dict, Any, List

from ambient.analysis.gait_analyzer import EnhancedGaitAnalyzer
from ambient.classification.features import GaitFeatureVector


class TestEnhancedFeatureIntegration:
    """Test suite for enhanced feature extraction integration."""
    
    @pytest.fixture
    def sample_pose_sequence(self) -> List[Dict[str, Any]]:
        """Create a sample pose sequence for testing."""
        poses = []
        
        for frame_idx in range(60):  # Shorter sequence for faster tests
            t = frame_idx / 30.0
            
            # Create realistic walking motion
            hip_y = 200 + 10 * np.sin(2 * np.pi * t)
            knee_y = hip_y + 50 + 20 * np.sin(4 * np.pi * t)
            ankle_y = knee_y + 50 + 30 * np.sin(6 * np.pi * t)
            x_offset = t * 20
            
            keypoints = [
                {"x": 320, "y": 100, "confidence": 0.9},  # nose
                {"x": 310, "y": 110, "confidence": 0.8},  # left_eye
                {"x": 330, "y": 110, "confidence": 0.8},  # right_eye
                {"x": 300, "y": 120, "confidence": 0.7},  # left_ear
                {"x": 340, "y": 120, "confidence": 0.7},  # right_ear
                {"x": 280 + x_offset, "y": 150, "confidence": 0.9},  # left_shoulder
                {"x": 360 + x_offset, "y": 150, "confidence": 0.9},  # right_shoulder
                {"x": 270 + x_offset, "y": 180, "confidence": 0.8},  # left_elbow
                {"x": 370 + x_offset, "y": 180, "confidence": 0.8},  # right_elbow
                {"x": 260 + x_offset, "y": 210, "confidence": 0.7},  # left_wrist
                {"x": 380 + x_offset, "y": 210, "confidence": 0.7},  # right_wrist
                {"x": 290 + x_offset, "y": hip_y, "confidence": 0.9},  # left_hip
                {"x": 350 + x_offset, "y": hip_y, "confidence": 0.9},  # right_hip
                {"x": 285 + x_offset, "y": knee_y, "confidence": 0.9},  # left_knee
                {"x": 355 + x_offset, "y": knee_y, "confidence": 0.9},  # right_knee
                {"x": 280 + x_offset, "y": ankle_y, "confidence": 0.9},  # left_ankle
                {"x": 360 + x_offset, "y": ankle_y, "confidence": 0.9},  # right_ankle
            ]
            
            poses.append({
                "keypoints": keypoints,
                "frame_index": frame_idx,
                "timestamp": t
            })
        
        return poses
    
    @pytest.fixture
    def asymmetric_pose_sequence(self) -> List[Dict[str, Any]]:
        """Create an asymmetric pose sequence for testing."""
        poses = []
        
        for frame_idx in range(60):
            t = frame_idx / 30.0
            
            # Create asymmetric motion (left side reduced)
            left_hip_y = 200 + 6 * np.sin(2 * np.pi * t)
            right_hip_y = 200 + 12 * np.sin(2 * np.pi * t + 0.2)
            left_knee_y = left_hip_y + 45 + 15 * np.sin(4 * np.pi * t)
            right_knee_y = right_hip_y + 55 + 25 * np.sin(4 * np.pi * t + 0.2)
            left_ankle_y = left_knee_y + 45 + 20 * np.sin(6 * np.pi * t)
            right_ankle_y = right_knee_y + 55 + 35 * np.sin(6 * np.pi * t + 0.2)
            
            x_offset = t * 20
            left_x_offset = x_offset * 0.9
            right_x_offset = x_offset * 1.1
            
            keypoints = [
                {"x": 320, "y": 100, "confidence": 0.9},  # nose
                {"x": 310, "y": 110, "confidence": 0.8},  # left_eye
                {"x": 330, "y": 110, "confidence": 0.8},  # right_eye
                {"x": 300, "y": 120, "confidence": 0.7},  # left_ear
                {"x": 340, "y": 120, "confidence": 0.7},  # right_ear
                {"x": 280 + left_x_offset, "y": 150, "confidence": 0.9},  # left_shoulder
                {"x": 360 + right_x_offset, "y": 148, "confidence": 0.9},  # right_shoulder
                {"x": 270 + left_x_offset, "y": 180, "confidence": 0.8},  # left_elbow
                {"x": 370 + right_x_offset, "y": 180, "confidence": 0.8},  # right_elbow
                {"x": 260 + left_x_offset, "y": 210, "confidence": 0.7},  # left_wrist
                {"x": 380 + right_x_offset, "y": 210, "confidence": 0.7},  # right_wrist
                {"x": 290 + left_x_offset, "y": left_hip_y, "confidence": 0.9},  # left_hip
                {"x": 350 + right_x_offset, "y": right_hip_y, "confidence": 0.9},  # right_hip
                {"x": 285 + left_x_offset, "y": left_knee_y, "confidence": 0.9},  # left_knee
                {"x": 355 + right_x_offset, "y": right_knee_y, "confidence": 0.9},  # right_knee
                {"x": 280 + left_x_offset, "y": left_ankle_y, "confidence": 0.9},  # left_ankle
                {"x": 360 + right_x_offset, "y": right_ankle_y, "confidence": 0.9},  # right_ankle
            ]
            
            poses.append({
                "keypoints": keypoints,
                "frame_index": frame_idx,
                "timestamp": t
            })
        
        return poses
    
    def test_enhanced_analyzer_initialization(self):
        """Test that EnhancedGaitAnalyzer initializes correctly."""
        analyzer = EnhancedGaitAnalyzer(keypoint_format="COCO_17", fps=30.0)
        
        assert analyzer.keypoint_format == "COCO_17"
        assert analyzer.fps == 30.0
        assert analyzer.feature_extractor is not None
        assert analyzer.temporal_analyzer is not None
        assert analyzer.symmetry_analyzer is not None
    
    def test_analysis_results_structure(self, sample_pose_sequence):
        """Test that analysis results have the expected structure."""
        analyzer = EnhancedGaitAnalyzer()
        results = analyzer.analyze_gait_sequence(sample_pose_sequence)
        
        # Check required keys
        required_keys = [
            "metadata", "sequence_info", "features", "gait_cycles",
            "timing_analysis", "phase_features", "symmetry_analysis", "summary"
        ]
        
        for key in required_keys:
            assert key in results, f"Missing required key: {key}"
        
        # Check that features are extracted
        assert len(results["features"]) > 0
        assert "left_hip_mean" in results["features"]
        assert "right_hip_mean" in results["features"]
    
    def test_feature_vector_creation(self, sample_pose_sequence):
        """Test that feature vector is created successfully from analysis results."""
        analyzer = EnhancedGaitAnalyzer()
        results = analyzer.analyze_gait_sequence(sample_pose_sequence)
        
        feature_vector = GaitFeatureVector.from_analysis_results(
            results, "test_sample", "normal"
        )
        
        assert feature_vector is not None
        assert feature_vector.sample_id == "test_sample"
        assert feature_vector.condition_label == "normal"
    
    def test_feature_vector_validation(self, sample_pose_sequence):
        """Test that feature vector validation works correctly."""
        analyzer = EnhancedGaitAnalyzer()
        results = analyzer.analyze_gait_sequence(sample_pose_sequence)
        feature_vector = GaitFeatureVector.from_analysis_results(results)
        
        is_valid, issues = feature_vector.validate(check_all_groups=True)
        
        # Should be valid for normal test data
        assert is_valid, f"Validation failed with issues: {issues}"
    
    def test_feature_group_selection(self, sample_pose_sequence):
        """Test that feature group selection works correctly."""
        analyzer = EnhancedGaitAnalyzer()
        results = analyzer.analyze_gait_sequence(sample_pose_sequence)
        feature_vector = GaitFeatureVector.from_analysis_results(results)
        
        # Test different feature group combinations
        core_features = feature_vector.to_array(feature_groups=["core_angles"])
        assert len(core_features) == 15
        
        spatio_features = feature_vector.to_array(feature_groups=["spatiotemporal"])
        assert len(spatio_features) == 4
        
        combined_features = feature_vector.to_array(feature_groups=["core_angles", "spatiotemporal"])
        assert len(combined_features) == 19
        
        all_features = feature_vector.to_array()
        assert len(all_features) == 34
    
    def test_backward_compatibility(self):
        """Test that backward compatibility is maintained."""
        # Create a legacy-style feature vector
        legacy_feature = GaitFeatureVector(
            left_hip_mean=45.0,
            left_knee_mean=90.0,
            left_ankle_mean=15.0,
            right_hip_mean=47.0,
            right_knee_mean=88.0,
            right_ankle_mean=17.0,
            sample_id="legacy_test",
            condition_label="normal"
        )
        
        # Test that legacy behavior works
        legacy_array = legacy_feature.to_array(feature_groups=["core_angles"])
        assert len(legacy_array) == 15
        
        # Test that asymmetry is calculated automatically
        assert legacy_feature.hip_asymmetry == 2.0
        assert legacy_feature.knee_asymmetry == 2.0
        assert legacy_feature.ankle_asymmetry == 2.0
    
    def test_asymmetry_detection(self, asymmetric_pose_sequence):
        """Test that asymmetry is properly detected and calculated."""
        analyzer = EnhancedGaitAnalyzer()
        results = analyzer.analyze_gait_sequence(asymmetric_pose_sequence)
        feature_vector = GaitFeatureVector.from_analysis_results(results)
        
        # Should detect some asymmetry
        assert feature_vector.hip_asymmetry > 0.5  # Some joint asymmetry
        assert feature_vector.stride_length_si > 10.0  # Some SI asymmetry
        
        # At least one SI should be above pathological threshold (16%)
        si_values = [
            feature_vector.stride_length_si,
            feature_vector.hip_angle_si,
            feature_vector.knee_angle_si,
            feature_vector.ankle_angle_si
        ]
        
        assert any(si > 16.0 for si in si_values), "No pathological asymmetry detected"
    
    def test_temporal_features(self, sample_pose_sequence):
        """Test that temporal features are properly extracted."""
        analyzer = EnhancedGaitAnalyzer()
        results = analyzer.analyze_gait_sequence(sample_pose_sequence)
        feature_vector = GaitFeatureVector.from_analysis_results(results)
        
        # Check temporal phase features
        assert 50 <= feature_vector.stance_percentage <= 80  # Reasonable stance %
        assert 20 <= feature_vector.swing_percentage <= 50   # Reasonable swing %
        assert 1.0 <= feature_vector.stance_swing_ratio <= 3.0  # Reasonable ratio
        
        # Check spatiotemporal features
        assert feature_vector.walking_speed_ms > 0
        assert feature_vector.cadence_steps_min > 0
        assert feature_vector.stride_length_m > 0
    
    def test_feature_names_consistency(self):
        """Test that feature names are consistent with array output."""
        # Create a sample feature vector
        feature_vector = GaitFeatureVector()
        
        # Test that feature names match array length for each group
        feature_groups = GaitFeatureVector.get_feature_groups()
        
        for group_name, expected_names in feature_groups.items():
            actual_array = feature_vector.to_array(feature_groups=[group_name])
            assert len(actual_array) == len(expected_names), \
                f"Mismatch in {group_name}: {len(actual_array)} vs {len(expected_names)}"
        
        # Test total feature count
        all_names = GaitFeatureVector.get_feature_names()
        all_array = feature_vector.to_array()
        assert len(all_array) == len(all_names) == 34
    
    def test_feature_summary_generation(self, sample_pose_sequence):
        """Test that feature summary is generated correctly."""
        analyzer = EnhancedGaitAnalyzer()
        results = analyzer.analyze_gait_sequence(sample_pose_sequence)
        feature_vector = GaitFeatureVector.from_analysis_results(results)
        
        # Test summary generation
        summary = feature_vector.get_feature_summary(include_all_groups=True)
        
        assert isinstance(summary, str)
        assert len(summary) > 100  # Should be substantial
        assert "CORE JOINT ANGLES" in summary
        assert "SYMMETRY INDICES" in summary
        assert "SPATIOTEMPORAL" in summary
    
    def test_error_handling(self):
        """Test error handling for invalid inputs."""
        analyzer = EnhancedGaitAnalyzer()
        
        # Test with empty pose sequence
        results = analyzer.analyze_gait_sequence([])
        assert "error" in results
        
        # Test feature vector creation with empty results
        feature_vector = GaitFeatureVector.from_analysis_results({})
        assert feature_vector is None
    
    @pytest.mark.parametrize("keypoint_format", ["COCO_17", "BODY_25"])
    def test_different_keypoint_formats(self, keypoint_format, sample_pose_sequence):
        """Test that different keypoint formats work correctly."""
        analyzer = EnhancedGaitAnalyzer(keypoint_format=keypoint_format)
        results = analyzer.analyze_gait_sequence(sample_pose_sequence)
        
        # Should complete without errors
        assert "features" in results
        assert len(results["features"]) > 0
    
    def test_performance_benchmarks(self, sample_pose_sequence):
        """Test that analysis completes within reasonable time."""
        import time
        
        analyzer = EnhancedGaitAnalyzer()
        
        start_time = time.time()
        results = analyzer.analyze_gait_sequence(sample_pose_sequence)
        feature_vector = GaitFeatureVector.from_analysis_results(results)
        end_time = time.time()
        
        analysis_time = end_time - start_time
        
        # Should complete within 1 second for 60 frames
        assert analysis_time < 1.0, f"Analysis took too long: {analysis_time:.2f}s"
        assert feature_vector is not None