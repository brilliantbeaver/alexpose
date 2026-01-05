"""
Comprehensive unit tests for data models in ambient.core.data_models.

Tests all data structures used in the gait analysis system including
GaitMetrics, ClassificationResult, VideoMetadata, and related models.
"""

import pytest
import numpy as np
from pathlib import Path
from dataclasses import asdict
from typing import Dict, List

from tests.conftest import skip_if_no_ambient
from tests.utils.test_helpers import FileManager, PerformanceProfiler
from tests.utils.assertions import assert_gait_metrics_realistic, AssertionHelpers

try:
    from ambient.core.data_models import (
        Keypoint, GaitFeatures, BalanceMetrics, TemporalMetrics, GaitMetrics,
        ConditionPrediction, ClassificationResult, VideoMetadata, ProcessingMetadata,
        GaitCycle, AnalysisResult, TrainingDataSample, DatasetInfo
    )
    AMBIENT_AVAILABLE = True
except ImportError:
    AMBIENT_AVAILABLE = False


@skip_if_no_ambient()
class TestKeypoint:
    """Test Keypoint data structure."""
    
    def test_keypoint_initialization(self):
        """Test Keypoint initialization with required fields."""
        keypoint = Keypoint(x=100.5, y=200.3, confidence=0.95)
        
        assert keypoint.x == 100.5
        assert keypoint.y == 200.3
        assert keypoint.confidence == 0.95
        assert keypoint.visibility == 1.0  # Default value
        assert keypoint.name == ""  # Default value
    
    def test_keypoint_with_all_fields(self):
        """Test Keypoint initialization with all fields."""
        keypoint = Keypoint(
            x=150.0,
            y=250.0,
            confidence=0.87,
            visibility=0.9,
            name="left_ankle"
        )
        
        assert keypoint.x == 150.0
        assert keypoint.y == 250.0
        assert keypoint.confidence == 0.87
        assert keypoint.visibility == 0.9
        assert keypoint.name == "left_ankle"
    
    def test_keypoint_serialization(self):
        """Test Keypoint can be serialized to dict."""
        keypoint = Keypoint(x=75.0, y=125.0, confidence=0.92, name="right_knee")
        keypoint_dict = asdict(keypoint)
        
        expected = {
            'x': 75.0,
            'y': 125.0,
            'confidence': 0.92,
            'visibility': 1.0,
            'name': 'right_knee'
        }
        
        assert keypoint_dict == expected


@skip_if_no_ambient()
class TestGaitFeatures:
    """Test GaitFeatures data structure."""
    
    def test_gait_features_initialization(self):
        """Test GaitFeatures initialization with default values."""
        features = GaitFeatures()
        
        assert features.temporal_features == {}
        assert features.spatial_features == {}
        assert features.kinematic_features == {}
        assert features.symmetry_features == {}
        assert features.frequency_features == {}
    
    def test_gait_features_with_data(self):
        """Test GaitFeatures with actual feature data."""
        features = GaitFeatures(
            temporal_features={"stride_time": 1.2, "stance_time": 0.7},
            spatial_features={"stride_length": 1.5, "step_width": 0.15},
            kinematic_features={"knee_angle": [45.0, 60.0, 30.0]},
            symmetry_features={"left_right_ratio": 0.95},
            frequency_features={"dominant_frequency": 1.8}
        )
        
        assert features.temporal_features["stride_time"] == 1.2
        assert features.spatial_features["stride_length"] == 1.5
        assert features.kinematic_features["knee_angle"] == [45.0, 60.0, 30.0]
        assert features.symmetry_features["left_right_ratio"] == 0.95
        assert features.frequency_features["dominant_frequency"] == 1.8
    
    def test_gait_features_modification(self):
        """Test GaitFeatures can be modified after creation."""
        features = GaitFeatures()
        
        # Add temporal features
        features.temporal_features["cadence"] = 110.0
        features.temporal_features["swing_time"] = 0.4
        
        # Add spatial features
        features.spatial_features["step_length"] = 0.75
        
        assert len(features.temporal_features) == 2
        assert features.temporal_features["cadence"] == 110.0
        assert features.spatial_features["step_length"] == 0.75


@skip_if_no_ambient()
class TestBalanceMetrics:
    """Test BalanceMetrics data structure."""
    
    def test_balance_metrics_defaults(self):
        """Test BalanceMetrics default values."""
        balance = BalanceMetrics()
        
        assert balance.center_of_mass_displacement == 0.0
        assert balance.postural_sway == 0.0
        assert balance.stability_index == 0.0
    
    def test_balance_metrics_with_values(self):
        """Test BalanceMetrics with specific values."""
        balance = BalanceMetrics(
            center_of_mass_displacement=2.5,
            postural_sway=1.8,
            stability_index=0.85
        )
        
        assert balance.center_of_mass_displacement == 2.5
        assert balance.postural_sway == 1.8
        assert balance.stability_index == 0.85


@skip_if_no_ambient()
class TestTemporalMetrics:
    """Test TemporalMetrics data structure."""
    
    def test_temporal_metrics_defaults(self):
        """Test TemporalMetrics default values."""
        temporal = TemporalMetrics()
        
        assert temporal.stance_phase_duration == 0.0
        assert temporal.swing_phase_duration == 0.0
        assert temporal.double_support_time == 0.0
        assert temporal.gait_cycle_time == 0.0
    
    def test_temporal_metrics_with_values(self):
        """Test TemporalMetrics with realistic gait values."""
        temporal = TemporalMetrics(
            stance_phase_duration=0.65,
            swing_phase_duration=0.35,
            double_support_time=0.15,
            gait_cycle_time=1.0
        )
        
        assert temporal.stance_phase_duration == 0.65
        assert temporal.swing_phase_duration == 0.35
        assert temporal.double_support_time == 0.15
        assert temporal.gait_cycle_time == 1.0


@skip_if_no_ambient()
class TestGaitMetrics:
    """Test GaitMetrics data structure."""
    
    def test_gait_metrics_defaults(self):
        """Test GaitMetrics default values."""
        metrics = GaitMetrics()
        
        assert metrics.stride_length == 0.0
        assert metrics.stride_time == 0.0
        assert metrics.cadence == 0.0
        assert metrics.step_width == 0.0
        assert metrics.joint_angles == {}
        assert metrics.symmetry_index == 0.0
        assert isinstance(metrics.balance_metrics, BalanceMetrics)
        assert isinstance(metrics.temporal_metrics, TemporalMetrics)
    
    def test_gait_metrics_realistic_values(self):
        """Test GaitMetrics with realistic gait analysis values."""
        balance = BalanceMetrics(
            center_of_mass_displacement=1.2,
            postural_sway=0.8,
            stability_index=0.92
        )
        
        temporal = TemporalMetrics(
            stance_phase_duration=0.62,
            swing_phase_duration=0.38,
            double_support_time=0.12,
            gait_cycle_time=1.1
        )
        
        metrics = GaitMetrics(
            stride_length=1.4,
            stride_time=1.1,
            cadence=109.0,
            step_width=0.12,
            joint_angles={
                "knee": [15.0, 60.0, 5.0],
                "ankle": [10.0, 20.0, -5.0]
            },
            symmetry_index=0.94,
            balance_metrics=balance,
            temporal_metrics=temporal
        )
        
        assert_gait_metrics_realistic(metrics)
        
        assert metrics.stride_length == 1.4
        assert metrics.cadence == 109.0
        assert metrics.joint_angles["knee"] == [15.0, 60.0, 5.0]
        assert metrics.balance_metrics.stability_index == 0.92
        assert metrics.temporal_metrics.gait_cycle_time == 1.1


@skip_if_no_ambient()
class TestConditionPrediction:
    """Test ConditionPrediction data structure."""
    
    def test_condition_prediction_minimal(self):
        """Test ConditionPrediction with minimal required fields."""
        prediction = ConditionPrediction(
            condition_name="Parkinson's Disease",
            confidence=0.78
        )
        
        assert prediction.condition_name == "Parkinson's Disease"
        assert prediction.confidence == 0.78
        assert prediction.severity == "unknown"  # Default
        assert prediction.supporting_evidence == []  # Default
    
    def test_condition_prediction_complete(self):
        """Test ConditionPrediction with all fields."""
        prediction = ConditionPrediction(
            condition_name="Hemiplegia",
            confidence=0.85,
            severity="moderate",
            supporting_evidence=[
                "Asymmetric gait pattern",
                "Reduced swing phase on affected side",
                "Compensatory hip hiking"
            ]
        )
        
        assert prediction.condition_name == "Hemiplegia"
        assert prediction.confidence == 0.85
        assert prediction.severity == "moderate"
        assert len(prediction.supporting_evidence) == 3
        assert "Asymmetric gait pattern" in prediction.supporting_evidence


@skip_if_no_ambient()
class TestClassificationResult:
    """Test ClassificationResult data structure."""
    
    def test_classification_result_normal_gait(self):
        """Test ClassificationResult for normal gait."""
        result = ClassificationResult(
            is_normal=True,
            confidence=0.92,
            explanation="Gait pattern shows normal temporal and spatial parameters."
        )
        
        assert result.is_normal is True
        assert result.confidence == 0.92
        assert result.identified_conditions == []  # Default
        assert "normal temporal" in result.explanation
        assert result.feature_importance == {}  # Default
    
    def test_classification_result_abnormal_gait(self):
        """Test ClassificationResult for abnormal gait with conditions."""
        conditions = [
            ConditionPrediction("Parkinson's Disease", 0.75, "mild"),
            ConditionPrediction("Gait Instability", 0.68, "moderate")
        ]
        
        result = ClassificationResult(
            is_normal=False,
            confidence=0.82,
            identified_conditions=conditions,
            explanation="Multiple gait abnormalities detected.",
            feature_importance={
                "stride_variability": 0.85,
                "cadence_irregularity": 0.72,
                "balance_metrics": 0.68
            }
        )
        
        assert result.is_normal is False
        assert result.confidence == 0.82
        assert len(result.identified_conditions) == 2
        assert result.identified_conditions[0].condition_name == "Parkinson's Disease"
        assert result.feature_importance["stride_variability"] == 0.85


@skip_if_no_ambient()
class TestVideoMetadata:
    """Test VideoMetadata data structure."""
    
    def test_video_metadata_minimal(self, tmp_path):
        """Test VideoMetadata with minimal required fields."""
        video_file = tmp_path / "test_video.mp4"
        
        metadata = VideoMetadata(file_path=video_file)
        
        assert metadata.file_path == video_file
        assert metadata.duration == 0.0  # Default
        assert metadata.frame_rate == 30.0  # Default
        assert metadata.width == 0  # Default
        assert metadata.height == 0  # Default
        assert metadata.format == "unknown"  # Default
        assert metadata.file_size == 0  # Default
        assert metadata.codec == "unknown"  # Default
    
    def test_video_metadata_complete(self, tmp_path):
        """Test VideoMetadata with all fields."""
        video_file = tmp_path / "complete_video.mp4"
        
        metadata = VideoMetadata(
            file_path=video_file,
            duration=30.5,
            frame_rate=60.0,
            width=1920,
            height=1080,
            format="mp4",
            file_size=1024000,
            codec="h264"
        )
        
        assert metadata.file_path == video_file
        assert metadata.duration == 30.5
        assert metadata.frame_rate == 60.0
        assert metadata.width == 1920
        assert metadata.height == 1080
        assert metadata.format == "mp4"
        assert metadata.file_size == 1024000
        assert metadata.codec == "h264"


@skip_if_no_ambient()
class TestProcessingMetadata:
    """Test ProcessingMetadata data structure."""
    
    def test_processing_metadata_defaults(self):
        """Test ProcessingMetadata default values."""
        metadata = ProcessingMetadata()
        
        assert metadata.processing_time == 0.0
        assert metadata.algorithm_version == "unknown"
        assert metadata.parameters == {}
        assert metadata.quality_metrics == {}
        assert metadata.timestamp is None
    
    def test_processing_metadata_complete(self):
        """Test ProcessingMetadata with all fields."""
        metadata = ProcessingMetadata(
            processing_time=45.2,
            algorithm_version="v2.1.0",
            parameters={
                "pose_model": "alexpose_v2",
                "confidence_threshold": 0.5,
                "smoothing_window": 5
            },
            quality_metrics={
                "pose_detection_accuracy": 0.94,
                "tracking_consistency": 0.89,
                "temporal_smoothness": 0.92
            },
            timestamp="2024-01-15T10:30:00Z"
        )
        
        assert metadata.processing_time == 45.2
        assert metadata.algorithm_version == "v2.1.0"
        assert metadata.parameters["pose_model"] == "alexpose_v2"
        assert metadata.quality_metrics["pose_detection_accuracy"] == 0.94
        assert metadata.timestamp == "2024-01-15T10:30:00Z"


@skip_if_no_ambient()
class TestGaitCycle:
    """Test GaitCycle data structure."""
    
    def test_gait_cycle_minimal(self):
        """Test GaitCycle with minimal required fields."""
        cycle = GaitCycle(start_frame=100, end_frame=150, duration=1.2)
        
        assert cycle.start_frame == 100
        assert cycle.end_frame == 150
        assert cycle.duration == 1.2
        assert cycle.left_heel_strikes == []  # Default
        assert cycle.right_heel_strikes == []  # Default
        assert cycle.left_toe_offs == []  # Default
        assert cycle.right_toe_offs == []  # Default
    
    def test_gait_cycle_complete(self):
        """Test GaitCycle with all gait events."""
        cycle = GaitCycle(
            start_frame=200,
            end_frame=280,
            duration=1.33,
            left_heel_strikes=[200, 240],
            right_heel_strikes=[220, 260],
            left_toe_offs=[210, 250],
            right_toe_offs=[230, 270]
        )
        
        assert cycle.start_frame == 200
        assert cycle.end_frame == 280
        assert cycle.duration == 1.33
        assert cycle.left_heel_strikes == [200, 240]
        assert cycle.right_heel_strikes == [220, 260]
        assert cycle.left_toe_offs == [210, 250]
        assert cycle.right_toe_offs == [230, 270]
    
    def test_gait_cycle_validation(self):
        """Test gait cycle temporal consistency."""
        cycle = GaitCycle(
            start_frame=100,
            end_frame=160,
            duration=1.0,
            left_heel_strikes=[100],
            right_heel_strikes=[130],
            left_toe_offs=[115],
            right_toe_offs=[145]
        )
        
        # Validate temporal ordering
        assert cycle.start_frame < cycle.end_frame
        assert all(cycle.start_frame <= frame <= cycle.end_frame 
                  for frame in cycle.left_heel_strikes)
        assert all(cycle.start_frame <= frame <= cycle.end_frame 
                  for frame in cycle.right_heel_strikes)


@skip_if_no_ambient()
class TestAnalysisResult:
    """Test AnalysisResult data structure."""
    
    def test_analysis_result_complete(self, tmp_path):
        """Test AnalysisResult with all components."""
        video_file = tmp_path / "analysis_video.mp4"
        
        video_metadata = VideoMetadata(
            file_path=video_file,
            duration=15.0,
            frame_rate=30.0,
            width=640,
            height=480
        )
        
        gait_metrics = GaitMetrics(
            stride_length=1.3,
            stride_time=1.1,
            cadence=105.0,
            step_width=0.11
        )
        
        classification = ClassificationResult(
            is_normal=True,
            confidence=0.89
        )
        
        gait_cycles = [
            GaitCycle(start_frame=0, end_frame=30, duration=1.0),
            GaitCycle(start_frame=30, end_frame=60, duration=1.0)
        ]
        
        processing_metadata = ProcessingMetadata(
            processing_time=12.5,
            algorithm_version="v2.0.0"
        )
        
        result = AnalysisResult(
            sequence_id="test_sequence_001",
            video_metadata=video_metadata,
            gait_metrics=gait_metrics,
            classification_result=classification,
            gait_cycles=gait_cycles,
            processing_metadata=processing_metadata
        )
        
        assert result.sequence_id == "test_sequence_001"
        assert result.video_metadata.file_path == video_file
        assert result.gait_metrics.stride_length == 1.3
        assert result.classification_result.is_normal is True
        assert len(result.gait_cycles) == 2
        assert result.processing_metadata.processing_time == 12.5
        assert result.raw_pose_data is None  # Default


@skip_if_no_ambient()
class TestTrainingDataSample:
    """Test TrainingDataSample data structure."""
    
    def test_training_data_sample_minimal(self):
        """Test TrainingDataSample with minimal fields."""
        features = GaitFeatures(
            temporal_features={"stride_time": 1.1},
            spatial_features={"stride_length": 1.4}
        )
        
        sample = TrainingDataSample(
            sample_id="sample_001",
            features=features,
            ground_truth_label="normal"
        )
        
        assert sample.sample_id == "sample_001"
        assert sample.features.temporal_features["stride_time"] == 1.1
        assert sample.ground_truth_label == "normal"
        assert sample.condition_labels == []  # Default
        assert sample.metadata == {}  # Default
    
    def test_training_data_sample_complete(self):
        """Test TrainingDataSample with all fields."""
        features = GaitFeatures(
            temporal_features={"stride_time": 1.3, "cadence": 95.0},
            spatial_features={"stride_length": 1.2, "step_width": 0.15}
        )
        
        sample = TrainingDataSample(
            sample_id="sample_pathological_001",
            features=features,
            ground_truth_label="pathological",
            condition_labels=["parkinson", "gait_instability"],
            metadata={
                "patient_age": 68,
                "patient_gender": "male",
                "disease_duration": 5
            }
        )
        
        assert sample.sample_id == "sample_pathological_001"
        assert sample.ground_truth_label == "pathological"
        assert "parkinson" in sample.condition_labels
        assert sample.metadata["patient_age"] == 68


@skip_if_no_ambient()
class TestDatasetInfo:
    """Test DatasetInfo data structure."""
    
    def test_dataset_info_minimal(self, tmp_path):
        """Test DatasetInfo with minimal fields."""
        dataset_path = tmp_path / "gait_dataset"
        
        info = DatasetInfo(
            dataset_name="GAVD Clinical Dataset",
            version="1.0",
            source_path=dataset_path,
            sample_count=150
        )
        
        assert info.dataset_name == "GAVD Clinical Dataset"
        assert info.version == "1.0"
        assert info.source_path == dataset_path
        assert info.sample_count == 150
        assert info.condition_distribution == {}  # Default
        assert info.metadata == {}  # Default
        assert info.created_at is None  # Default
        assert info.updated_at is None  # Default
    
    def test_dataset_info_complete(self, tmp_path):
        """Test DatasetInfo with all fields."""
        dataset_path = tmp_path / "comprehensive_gait_dataset"
        
        info = DatasetInfo(
            dataset_name="Comprehensive Gait Analysis Dataset",
            version="2.1.0",
            source_path=dataset_path,
            sample_count=500,
            condition_distribution={
                "normal": 200,
                "parkinson": 150,
                "hemiplegia": 100,
                "other": 50
            },
            metadata={
                "collection_period": "2023-2024",
                "institutions": ["Hospital A", "Clinic B"],
                "ethics_approval": "IRB-2023-001"
            },
            created_at="2023-01-15T09:00:00Z",
            updated_at="2024-01-15T14:30:00Z"
        )
        
        assert info.dataset_name == "Comprehensive Gait Analysis Dataset"
        assert info.sample_count == 500
        assert info.condition_distribution["normal"] == 200
        assert info.condition_distribution["parkinson"] == 150
        assert "Hospital A" in info.metadata["institutions"]
        assert info.created_at == "2023-01-15T09:00:00Z"


@pytest.mark.performance
@skip_if_no_ambient()
class TestDataModelsPerformance:
    """Test performance characteristics of data models."""
    
    def test_gait_metrics_serialization_performance(self):
        """Test performance of GaitMetrics serialization."""
        profiler = PerformanceProfiler()
        
        # Create complex GaitMetrics
        metrics = GaitMetrics(
            stride_length=1.4,
            stride_time=1.1,
            cadence=109.0,
            step_width=0.12,
            joint_angles={
                f"joint_{i}": [float(j) for j in range(100)]
                for i in range(20)
            },
            symmetry_index=0.94
        )
        
        with profiler.profile("serialize_gait_metrics"):
            serialized = asdict(metrics)
        
        with profiler.profile("access_serialized_data"):
            _ = serialized["joint_angles"]["joint_5"][50]
        
        serialize_time = profiler.get_metrics("serialize_gait_metrics")["execution_time"]
        access_time = profiler.get_metrics("access_serialized_data")["execution_time"]
        
        # Serialization should be fast even for complex data
        assert serialize_time < 0.1  # Less than 100ms
        assert access_time < 0.001  # Less than 1ms
    
    def test_analysis_result_memory_usage(self):
        """Test memory efficiency of AnalysisResult."""
        profiler = PerformanceProfiler()
        
        # Create multiple analysis results
        results = []
        
        with profiler.profile("create_analysis_results"):
            for i in range(100):
                video_metadata = VideoMetadata(
                    file_path=Path(f"video_{i}.mp4"),
                    duration=10.0,
                    frame_rate=30.0
                )
                
                gait_metrics = GaitMetrics(
                    stride_length=1.3 + i * 0.01,
                    cadence=100.0 + i
                )
                
                classification = ClassificationResult(
                    is_normal=i % 2 == 0,
                    confidence=0.8 + i * 0.001
                )
                
                result = AnalysisResult(
                    sequence_id=f"seq_{i:03d}",
                    video_metadata=video_metadata,
                    gait_metrics=gait_metrics,
                    classification_result=classification
                )
                
                results.append(result)
        
        creation_time = profiler.get_metrics("create_analysis_results")["execution_time"]
        
        # Should be able to create 100 results quickly
        assert creation_time < 1.0  # Less than 1 second
        assert len(results) == 100
        
        # Verify data integrity
        assert results[50].sequence_id == "seq_050"
        assert results[50].gait_metrics.stride_length == 1.3 + 50 * 0.01


@skip_if_no_ambient()
class TestDataModelsIntegration:
    """Test integration between different data models."""
    
    def test_complete_analysis_workflow(self, tmp_path):
        """Test complete analysis workflow using all data models."""
        # Create video metadata
        video_file = tmp_path / "workflow_test.mp4"
        video_metadata = VideoMetadata(
            file_path=video_file,
            duration=20.0,
            frame_rate=30.0,
            width=1280,
            height=720,
            format="mp4"
        )
        
        # Create gait features
        features = GaitFeatures(
            temporal_features={
                "stride_time": 1.15,
                "stance_time": 0.68,
                "swing_time": 0.47
            },
            spatial_features={
                "stride_length": 1.35,
                "step_width": 0.13,
                "step_length": 0.67
            },
            kinematic_features={
                "knee_angle": [5.0, 65.0, 10.0],
                "ankle_angle": [10.0, 25.0, -5.0]
            }
        )
        
        # Create gait metrics
        balance = BalanceMetrics(
            center_of_mass_displacement=1.8,
            postural_sway=0.9,
            stability_index=0.88
        )
        
        temporal = TemporalMetrics(
            stance_phase_duration=0.68,
            swing_phase_duration=0.47,
            double_support_time=0.14,
            gait_cycle_time=1.15
        )
        
        gait_metrics = GaitMetrics(
            stride_length=1.35,
            stride_time=1.15,
            cadence=104.3,
            step_width=0.13,
            joint_angles={
                "knee": [5.0, 65.0, 10.0],
                "ankle": [10.0, 25.0, -5.0]
            },
            symmetry_index=0.91,
            balance_metrics=balance,
            temporal_metrics=temporal
        )
        
        # Create classification result
        condition = ConditionPrediction(
            condition_name="Mild Gait Instability",
            confidence=0.72,
            severity="mild",
            supporting_evidence=[
                "Slightly increased step width variability",
                "Minor balance perturbations"
            ]
        )
        
        classification = ClassificationResult(
            is_normal=False,
            confidence=0.75,
            identified_conditions=[condition],
            explanation="Mild gait abnormalities detected, possibly age-related.",
            feature_importance={
                "balance_metrics": 0.82,
                "step_width_variability": 0.71,
                "temporal_consistency": 0.65
            }
        )
        
        # Create gait cycles
        gait_cycles = [
            GaitCycle(
                start_frame=0,
                end_frame=34,
                duration=1.13,
                left_heel_strikes=[0, 17],
                right_heel_strikes=[8, 25],
                left_toe_offs=[5, 22],
                right_toe_offs=[13, 30]
            ),
            GaitCycle(
                start_frame=34,
                end_frame=69,
                duration=1.17,
                left_heel_strikes=[34, 52],
                right_heel_strikes=[43, 60],
                left_toe_offs=[39, 57],
                right_toe_offs=[48, 65]
            )
        ]
        
        # Create processing metadata
        processing_metadata = ProcessingMetadata(
            processing_time=28.5,
            algorithm_version="alexpose_v2.1.0",
            parameters={
                "pose_model": "alexpose_clinical",
                "confidence_threshold": 0.6,
                "smoothing_window": 7,
                "gait_cycle_detection": "heel_strike"
            },
            quality_metrics={
                "pose_detection_accuracy": 0.93,
                "tracking_consistency": 0.87,
                "temporal_smoothness": 0.91,
                "overall_quality": 0.90
            },
            timestamp="2024-01-15T15:45:30Z"
        )
        
        # Create complete analysis result
        analysis_result = AnalysisResult(
            sequence_id="workflow_test_001",
            video_metadata=video_metadata,
            gait_metrics=gait_metrics,
            classification_result=classification,
            gait_cycles=gait_cycles,
            processing_metadata=processing_metadata,
            raw_pose_data=[
                {"frame": 0, "keypoints": [{"x": 100, "y": 200, "confidence": 0.9}]},
                {"frame": 1, "keypoints": [{"x": 102, "y": 198, "confidence": 0.91}]}
            ]
        )
        
        # Validate complete workflow
        assert analysis_result.sequence_id == "workflow_test_001"
        assert analysis_result.video_metadata.duration == 20.0
        assert analysis_result.gait_metrics.cadence == 104.3
        assert analysis_result.classification_result.is_normal is False
        assert len(analysis_result.gait_cycles) == 2
        assert analysis_result.processing_metadata.algorithm_version == "alexpose_v2.1.0"
        assert len(analysis_result.raw_pose_data) == 2
        
        # Test realistic gait metrics
        assert_gait_metrics_realistic(analysis_result.gait_metrics)
        
        # Test data consistency
        assert (analysis_result.gait_metrics.stride_time == 
                analysis_result.gait_metrics.temporal_metrics.gait_cycle_time)
        
        # Test serialization of complete result
        serialized = asdict(analysis_result)
        assert serialized["sequence_id"] == "workflow_test_001"
        assert serialized["gait_metrics"]["cadence"] == 104.3


if __name__ == "__main__":
    pytest.main([__file__, "-v"])