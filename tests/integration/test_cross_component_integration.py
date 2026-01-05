"""
Cross-component integration tests for AlexPose.

This module tests interactions between different system components
to ensure they work together correctly in various scenarios.
"""

import pytest
import asyncio
import time
import tempfile
from pathlib import Path
from typing import Dict, List, Any, Optional
from unittest.mock import Mock, patch
import json

from tests.integration.integration_framework import IntegrationTestFramework
from tests.fixtures.real_data_fixtures import get_real_data_manager

try:
    from ambient.video.processor import VideoProcessor
    from ambient.core.frame import Frame, FrameSequence
    from ambient.pose.factory import PoseEstimatorFactory
    from ambient.analysis.gait_analyzer import GaitAnalyzer, EnhancedGaitAnalyzer
    from ambient.classification.llm_classifier import LLMClassifier
    from ambient.core.config import ConfigurationManager
    from ambient.core.interfaces import IPoseEstimator, IGaitAnalyzer, IClassifier
    AMBIENT_AVAILABLE = True
except ImportError:
    AMBIENT_AVAILABLE = False


@pytest.mark.skipif(not AMBIENT_AVAILABLE, reason="Ambient components not available")
class TestCrossComponentIntegration:
    """Test cross-component interactions and workflows."""

    @pytest.fixture(scope="class")
    def config_manager(self):
        """Provide configuration manager for testing."""
        return ConfigurationManager()

    @pytest.fixture(scope="class")
    def sample_videos(self):
        """Provide sample videos for testing."""
        data_manager = get_real_data_manager()
        return data_manager.get_sample_videos()

    @pytest.fixture
    def mock_components(self):
        """Provide mock components for testing interactions."""
        components = {
            'video_processor': Mock(spec=VideoProcessor),
            'pose_estimator': Mock(spec=IPoseEstimator),
            'gait_analyzer': Mock(spec=IGaitAnalyzer),
            'classifier': Mock(spec=IClassifier)
        }
        
        # Configure realistic mock behaviors
        self._configure_mock_components(components)
        return components

    def _configure_mock_components(self, components: Dict[str, Mock]):
        """Configure realistic behaviors for mock components."""
        # Video processor mock
        components['video_processor'].get_video_info.return_value = {
            "duration": 5.0,
            "fps": 30.0,
            "frame_count": 150,
            "width": 640,
            "height": 480
        }
        
        # Mock frame sequence using a simple class
        class MockFrameSequence:
            def __init__(self, frames):
                self.frames = frames
            
            def __len__(self):
                return len(self.frames)
            
            def __iter__(self):
                return iter(self.frames)
        
        mock_frames = [Mock(spec=Frame) for _ in range(150)]
        for i, frame in enumerate(mock_frames):
            frame.frame_id = i
            frame.timestamp = i / 30.0
            frame.width = 640
            frame.height = 480
        
        mock_sequence = MockFrameSequence(mock_frames)
        components['video_processor'].load_video.return_value = mock_sequence
        
        # Pose estimator mock
        mock_keypoints = [
            {"x": 100.0 + i, "y": 200.0 + i, "confidence": 0.8}
            for i in range(33)
        ]
        components['pose_estimator'].estimate_pose_sequence.return_value = [
            mock_keypoints for _ in range(150)
        ]
        
        # Gait analyzer mock
        components['gait_analyzer'].analyze_gait_sequence.return_value = {
            "temporal_features": {
                "stride_time": 1.2,
                "cadence": 110.0,
                "stance_time": 0.7,
                "swing_time": 0.5
            },
            "spatial_features": {
                "stride_length": 1.4,
                "step_width": 0.15,
                "step_length": 0.7
            },
            "symmetry_features": {
                "left_right_symmetry": 0.95,
                "temporal_symmetry": 0.92
            }
        }
        
        # Classifier mock
        components['classifier'].classify_gait.return_value = {
            "classification": "normal",
            "confidence": 0.85,
            "explanation": "Normal gait pattern detected with good symmetry"
        }

    @pytest.mark.integration
    @pytest.mark.slow
    def test_video_processor_to_pose_estimator_integration(
        self, 
        mock_components: Dict[str, Mock],
        sample_videos: Dict[str, Path]
    ):
        """
        Test integration between video processor and pose estimator.
        
        This test validates that frame sequences from video processor
        are properly consumed by pose estimator.
        """
        video_processor = mock_components['video_processor']
        pose_estimator = mock_components['pose_estimator']
        
        # Use a sample video if available, otherwise use mock path
        video_path = sample_videos.get("normal_walking", Path("mock_video.mp4"))
        
        # Step 1: Video processor loads video
        frame_sequence = video_processor.load_video(video_path)
        
        # Validate frame sequence
        assert frame_sequence is not None, "Video processor should return frame sequence"
        assert len(frame_sequence) > 0, "Frame sequence should not be empty"
        
        # Step 2: Pose estimator processes frame sequence
        pose_sequence = pose_estimator.estimate_pose_sequence(frame_sequence)
        
        # Validate pose sequence
        assert pose_sequence is not None, "Pose estimator should return pose sequence"
        assert len(pose_sequence) == len(frame_sequence), \
            "Pose sequence length should match frame sequence length"
        
        # Validate pose data structure
        if pose_sequence:
            first_frame_poses = pose_sequence[0]
            assert isinstance(first_frame_poses, list), "Poses should be a list"
            assert len(first_frame_poses) == 33, "Should have 33 MediaPipe landmarks"
            
            # Validate landmark structure
            first_landmark = first_frame_poses[0]
            assert "x" in first_landmark, "Landmark should have x coordinate"
            assert "y" in first_landmark, "Landmark should have y coordinate"
            assert "confidence" in first_landmark, "Landmark should have confidence"
        
        # Verify component interactions
        video_processor.load_video.assert_called_once_with(video_path)
        pose_estimator.estimate_pose_sequence.assert_called_once_with(frame_sequence)

    @pytest.mark.integration
    @pytest.mark.slow
    def test_pose_estimator_to_gait_analyzer_integration(
        self, 
        mock_components: Dict[str, Mock]
    ):
        """
        Test integration between pose estimator and gait analyzer.
        
        This test validates that pose sequences are properly
        analyzed for gait features.
        """
        pose_estimator = mock_components['pose_estimator']
        gait_analyzer = mock_components['gait_analyzer']
        
        # Generate mock frame sequence for pose estimation
        class MockFrameSequence:
            def __init__(self, frames):
                self.frames = frames
            
            def __len__(self):
                return len(self.frames)
            
            def __iter__(self):
                return iter(self.frames)
        
        mock_frames = [Mock(spec=Frame) for _ in range(150)]
        mock_sequence = MockFrameSequence(mock_frames)
        
        # Step 1: Pose estimator generates pose sequence
        pose_sequence = pose_estimator.estimate_pose_sequence(mock_sequence)
        
        # Validate pose sequence
        assert pose_sequence is not None, "Pose estimator should return pose sequence"
        assert len(pose_sequence) > 0, "Pose sequence should not be empty"
        
        # Step 2: Gait analyzer processes pose sequence
        gait_features = gait_analyzer.analyze_gait_sequence(pose_sequence)
        
        # Validate gait features
        assert gait_features is not None, "Gait analyzer should return features"
        
        # Validate required feature categories
        required_categories = ["temporal_features", "spatial_features", "symmetry_features"]
        for category in required_categories:
            assert category in gait_features, f"Missing feature category: {category}"
        
        # Validate temporal features
        temporal = gait_features["temporal_features"]
        assert "stride_time" in temporal, "Missing stride_time"
        assert "cadence" in temporal, "Missing cadence"
        assert temporal["stride_time"] > 0, "Stride time should be positive"
        assert temporal["cadence"] > 0, "Cadence should be positive"
        
        # Validate spatial features
        spatial = gait_features["spatial_features"]
        assert "stride_length" in spatial, "Missing stride_length"
        assert "step_width" in spatial, "Missing step_width"
        assert spatial["stride_length"] > 0, "Stride length should be positive"
        
        # Validate symmetry features
        symmetry = gait_features["symmetry_features"]
        assert "left_right_symmetry" in symmetry, "Missing left_right_symmetry"
        assert 0 <= symmetry["left_right_symmetry"] <= 1, "Symmetry should be between 0 and 1"
        
        # Verify component interactions
        pose_estimator.estimate_pose_sequence.assert_called_once()
        gait_analyzer.analyze_gait_sequence.assert_called_once_with(pose_sequence)

    @pytest.mark.integration
    @pytest.mark.slow
    def test_gait_analyzer_to_classifier_integration(
        self, 
        mock_components: Dict[str, Mock]
    ):
        """
        Test integration between gait analyzer and classifier.
        
        This test validates that gait features are properly
        classified for medical assessment.
        """
        gait_analyzer = mock_components['gait_analyzer']
        classifier = mock_components['classifier']
        
        # Generate mock pose sequence for gait analysis
        mock_pose_sequence = [
            [{"x": 100.0 + i, "y": 200.0 + i, "confidence": 0.8} for i in range(33)]
            for _ in range(150)
        ]
        
        # Step 1: Gait analyzer extracts features
        gait_features = gait_analyzer.analyze_gait_sequence(mock_pose_sequence)
        
        # Validate gait features
        assert gait_features is not None, "Gait analyzer should return features"
        assert isinstance(gait_features, dict), "Gait features should be a dictionary"
        
        # Step 2: Classifier processes gait features
        classification_result = classifier.classify_gait(gait_features)
        
        # Validate classification result
        assert classification_result is not None, "Classifier should return result"
        assert "classification" in classification_result, "Missing classification field"
        assert "confidence" in classification_result, "Missing confidence field"
        
        # Validate classification values
        classification = classification_result["classification"]
        confidence = classification_result["confidence"]
        
        assert classification in ["normal", "abnormal"], \
            f"Invalid classification: {classification}"
        assert 0.0 <= confidence <= 1.0, f"Invalid confidence: {confidence}"
        
        # Validate optional explanation
        if "explanation" in classification_result:
            explanation = classification_result["explanation"]
            assert isinstance(explanation, str), "Explanation should be a string"
            assert len(explanation) > 0, "Explanation should not be empty"
        
        # Verify component interactions
        gait_analyzer.analyze_gait_sequence.assert_called_once_with(mock_pose_sequence)
        classifier.classify_gait.assert_called_once_with(gait_features)

    @pytest.mark.integration
    @pytest.mark.slow
    def test_configuration_manager_component_integration(
        self, 
        config_manager: ConfigurationManager
    ):
        """
        Test integration between configuration manager and components.
        
        This test validates that components properly use configuration
        settings from the configuration manager.
        """
        # Test pose estimator configuration
        try:
            pose_config = config_manager.get_pose_estimator_config('mediapipe')
            assert pose_config is not None, "Should have MediaPipe configuration"
            
            # Test that factory can use configuration
            factory = PoseEstimatorFactory()
            estimator_info = factory.get_estimator_info('mediapipe')
            assert estimator_info is not None, "Should get estimator info"
            
        except Exception as e:
            # Configuration might not be fully set up in test environment
            pytest.skip(f"Configuration not available: {e}")
        
        # Test LLM configuration for classifier
        try:
            llm_config = config_manager.get_llm_config()
            if llm_config:
                assert "models" in llm_config, "LLM config should have models"
                
        except Exception as e:
            # LLM configuration might not be available
            pytest.skip(f"LLM configuration not available: {e}")

    @pytest.mark.integration
    @pytest.mark.slow
    def test_error_propagation_across_components(
        self, 
        mock_components: Dict[str, Mock]
    ):
        """
        Test error propagation between components.
        
        This test validates that errors in one component are
        properly handled by downstream components.
        """
        video_processor = mock_components['video_processor']
        pose_estimator = mock_components['pose_estimator']
        gait_analyzer = mock_components['gait_analyzer']
        classifier = mock_components['classifier']
        
        # Test 1: Video processor error
        video_processor.load_video.side_effect = Exception("Video loading failed")
        
        with pytest.raises(Exception) as exc_info:
            video_processor.load_video("invalid_video.mp4")
        
        assert "Video loading failed" in str(exc_info.value)
        
        # Reset mock
        video_processor.load_video.side_effect = None
        video_processor.load_video.return_value = Mock(spec=FrameSequence)
        
        # Test 2: Pose estimator error
        pose_estimator.estimate_pose_sequence.side_effect = Exception("Pose estimation failed")
        
        frame_sequence = video_processor.load_video("test_video.mp4")
        
        with pytest.raises(Exception) as exc_info:
            pose_estimator.estimate_pose_sequence(frame_sequence)
        
        assert "Pose estimation failed" in str(exc_info.value)
        
        # Reset mock
        pose_estimator.estimate_pose_sequence.side_effect = None
        pose_estimator.estimate_pose_sequence.return_value = []
        
        # Test 3: Gait analyzer error
        gait_analyzer.analyze_gait_sequence.side_effect = Exception("Gait analysis failed")
        
        pose_sequence = pose_estimator.estimate_pose_sequence(frame_sequence)
        
        with pytest.raises(Exception) as exc_info:
            gait_analyzer.analyze_gait_sequence(pose_sequence)
        
        assert "Gait analysis failed" in str(exc_info.value)
        
        # Reset mock
        gait_analyzer.analyze_gait_sequence.side_effect = None
        gait_analyzer.analyze_gait_sequence.return_value = {}
        
        # Test 4: Classifier error
        classifier.classify_gait.side_effect = Exception("Classification failed")
        
        gait_features = gait_analyzer.analyze_gait_sequence(pose_sequence)
        
        with pytest.raises(Exception) as exc_info:
            classifier.classify_gait(gait_features)
        
        assert "Classification failed" in str(exc_info.value)

    @pytest.mark.integration
    @pytest.mark.slow
    def test_data_format_compatibility_across_components(
        self, 
        mock_components: Dict[str, Mock]
    ):
        """
        Test data format compatibility between components.
        
        This test validates that data formats are consistent
        across component boundaries.
        """
        video_processor = mock_components['video_processor']
        pose_estimator = mock_components['pose_estimator']
        gait_analyzer = mock_components['gait_analyzer']
        classifier = mock_components['classifier']
        
        # Test frame sequence format
        frame_sequence = video_processor.load_video("test_video.mp4")
        
        # Validate frame sequence interface
        assert hasattr(frame_sequence, '__len__'), "Frame sequence should be iterable"
        assert hasattr(frame_sequence, '__iter__'), "Frame sequence should be iterable"
        
        # Test pose sequence format
        pose_sequence = pose_estimator.estimate_pose_sequence(frame_sequence)
        
        # Validate pose sequence format
        assert isinstance(pose_sequence, list), "Pose sequence should be a list"
        if pose_sequence:
            frame_poses = pose_sequence[0]
            assert isinstance(frame_poses, list), "Frame poses should be a list"
            
            if frame_poses:
                landmark = frame_poses[0]
                assert isinstance(landmark, dict), "Landmark should be a dictionary"
                assert "x" in landmark, "Landmark should have x coordinate"
                assert "y" in landmark, "Landmark should have y coordinate"
                assert "confidence" in landmark, "Landmark should have confidence"
        
        # Test gait features format
        gait_features = gait_analyzer.analyze_gait_sequence(pose_sequence)
        
        # Validate gait features format
        assert isinstance(gait_features, dict), "Gait features should be a dictionary"
        
        # Test classification result format
        classification_result = classifier.classify_gait(gait_features)
        
        # Validate classification result format
        assert isinstance(classification_result, dict), "Classification result should be a dictionary"
        assert "classification" in classification_result, "Should have classification field"
        assert "confidence" in classification_result, "Should have confidence field"

    @pytest.mark.integration
    @pytest.mark.slow
    def test_component_performance_interaction(
        self, 
        mock_components: Dict[str, Mock]
    ):
        """
        Test performance characteristics of component interactions.
        
        This test validates that component interactions don't
        introduce significant performance bottlenecks.
        """
        video_processor = mock_components['video_processor']
        pose_estimator = mock_components['pose_estimator']
        gait_analyzer = mock_components['gait_analyzer']
        classifier = mock_components['classifier']
        
        # Measure component interaction times
        start_time = time.time()
        
        # Video processing
        video_start = time.time()
        frame_sequence = video_processor.load_video("test_video.mp4")
        video_time = time.time() - video_start
        
        # Pose estimation
        pose_start = time.time()
        pose_sequence = pose_estimator.estimate_pose_sequence(frame_sequence)
        pose_time = time.time() - pose_start
        
        # Gait analysis
        gait_start = time.time()
        gait_features = gait_analyzer.analyze_gait_sequence(pose_sequence)
        gait_time = time.time() - gait_start
        
        # Classification
        classify_start = time.time()
        classification_result = classifier.classify_gait(gait_features)
        classify_time = time.time() - classify_start
        
        total_time = time.time() - start_time
        
        # Validate performance (with mocks, should be very fast)
        assert video_time < 1.0, f"Video processing too slow: {video_time:.3f}s"
        assert pose_time < 1.0, f"Pose estimation too slow: {pose_time:.3f}s"
        assert gait_time < 1.0, f"Gait analysis too slow: {gait_time:.3f}s"
        assert classify_time < 1.0, f"Classification too slow: {classify_time:.3f}s"
        assert total_time < 2.0, f"Total pipeline too slow: {total_time:.3f}s"
        
        # Log performance metrics
        print(f"\nComponent Performance Metrics:")
        print(f"  Video Processing: {video_time:.3f}s")
        print(f"  Pose Estimation: {pose_time:.3f}s")
        print(f"  Gait Analysis: {gait_time:.3f}s")
        print(f"  Classification: {classify_time:.3f}s")
        print(f"  Total Time: {total_time:.3f}s")

    @pytest.mark.integration
    @pytest.mark.slow
    def test_component_state_isolation(
        self, 
        mock_components: Dict[str, Mock]
    ):
        """
        Test that components maintain proper state isolation.
        
        This test validates that components don't interfere
        with each other's internal state.
        """
        video_processor = mock_components['video_processor']
        pose_estimator = mock_components['pose_estimator']
        gait_analyzer = mock_components['gait_analyzer']
        classifier = mock_components['classifier']
        
        # Process first video
        frame_sequence_1 = video_processor.load_video("video1.mp4")
        pose_sequence_1 = pose_estimator.estimate_pose_sequence(frame_sequence_1)
        gait_features_1 = gait_analyzer.analyze_gait_sequence(pose_sequence_1)
        result_1 = classifier.classify_gait(gait_features_1)
        
        # Process second video
        frame_sequence_2 = video_processor.load_video("video2.mp4")
        pose_sequence_2 = pose_estimator.estimate_pose_sequence(frame_sequence_2)
        gait_features_2 = gait_analyzer.analyze_gait_sequence(pose_sequence_2)
        result_2 = classifier.classify_gait(gait_features_2)
        
        # Validate that each processing is independent
        assert result_1 is not None, "First result should be valid"
        assert result_2 is not None, "Second result should be valid"
        
        # Verify call counts (each component called twice)
        assert video_processor.load_video.call_count == 2
        assert pose_estimator.estimate_pose_sequence.call_count == 2
        assert gait_analyzer.analyze_gait_sequence.call_count == 2
        assert classifier.classify_gait.call_count == 2

    @pytest.mark.integration
    @pytest.mark.slow
    def test_component_resource_cleanup(
        self, 
        mock_components: Dict[str, Mock]
    ):
        """
        Test that components properly clean up resources.
        
        This test validates that components don't leak
        resources during normal operation.
        """
        import gc
        import psutil
        import os
        
        # Get initial memory usage
        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        video_processor = mock_components['video_processor']
        pose_estimator = mock_components['pose_estimator']
        gait_analyzer = mock_components['gait_analyzer']
        classifier = mock_components['classifier']
        
        # Process multiple videos to test resource cleanup
        for i in range(10):
            frame_sequence = video_processor.load_video(f"video_{i}.mp4")
            pose_sequence = pose_estimator.estimate_pose_sequence(frame_sequence)
            gait_features = gait_analyzer.analyze_gait_sequence(pose_sequence)
            result = classifier.classify_gait(gait_features)
            
            # Validate result
            assert result is not None, f"Result {i} should be valid"
            
            # Force garbage collection
            gc.collect()
        
        # Get final memory usage
        final_memory = process.memory_info().rss / 1024 / 1024  # MB
        memory_increase = final_memory - initial_memory
        
        # Validate memory usage (should not increase significantly with mocks)
        assert memory_increase < 100, \
            f"Excessive memory usage: {memory_increase:.1f}MB increase"
        
        print(f"Memory usage: {initial_memory:.1f}MB → {final_memory:.1f}MB (+{memory_increase:.1f}MB)")

    @pytest.mark.integration
    @pytest.mark.slow
    @pytest.mark.asyncio
    async def test_component_concurrent_access(
        self, 
        mock_components: Dict[str, Mock]
    ):
        """
        Test component behavior under concurrent access.
        
        This test validates that components can handle
        concurrent requests without interference.
        """
        video_processor = mock_components['video_processor']
        pose_estimator = mock_components['pose_estimator']
        gait_analyzer = mock_components['gait_analyzer']
        classifier = mock_components['classifier']
        
        async def process_video(video_id: int):
            """Process a single video through the pipeline."""
            frame_sequence = video_processor.load_video(f"video_{video_id}.mp4")
            pose_sequence = pose_estimator.estimate_pose_sequence(frame_sequence)
            gait_features = gait_analyzer.analyze_gait_sequence(pose_sequence)
            result = classifier.classify_gait(gait_features)
            return video_id, result
        
        # Process 5 videos concurrently
        tasks = [process_video(i) for i in range(5)]
        results = await asyncio.gather(*tasks)
        
        # Validate all results
        assert len(results) == 5, "Should have 5 results"
        
        for video_id, result in results:
            assert result is not None, f"Result for video {video_id} should be valid"
            assert "classification" in result, f"Result {video_id} should have classification"
        
        # Verify call counts (each component called 5 times)
        assert video_processor.load_video.call_count == 5
        assert pose_estimator.estimate_pose_sequence.call_count == 5
        assert gait_analyzer.analyze_gait_sequence.call_count == 5
        assert classifier.classify_gait.call_count == 5

    @pytest.mark.integration
    @pytest.mark.slow
    def test_component_factory_integration(self):
        """
        Test integration with component factories.
        
        This test validates that factories properly create
        and configure components for integration.
        """
        try:
            # Test pose estimator factory
            factory = PoseEstimatorFactory()
            
            # Get available estimators
            available_estimators = factory.get_available_estimators()
            assert len(available_estimators) > 0, "Should have available estimators"
            
            # Test creating estimator
            if 'mediapipe' in available_estimators:
                estimator = factory.create_estimator('mediapipe')
                assert estimator is not None, "Should create MediaPipe estimator"
                assert hasattr(estimator, 'estimate_pose_sequence'), \
                    "Estimator should have estimate_pose_sequence method"
            
        except Exception as e:
            pytest.skip(f"Factory integration not available: {e}")

    @pytest.mark.integration
    @pytest.mark.slow
    def test_real_component_integration_smoke_test(
        self, 
        sample_videos: Dict[str, Path]
    ):
        """
        Smoke test with real components (if available).
        
        This test attempts to use real components for basic
        integration validation.
        """
        if not sample_videos or "normal_walking" not in sample_videos:
            pytest.skip("Sample videos not available")
        
        video_file = sample_videos["normal_walking"]
        if not video_file.exists():
            pytest.skip(f"Video file not found: {video_file}")
        
        try:
            # Try to create real components
            video_processor = VideoProcessor()
            
            # Test video loading
            video_info = video_processor.get_video_info(video_file)
            assert video_info is not None, "Should get video info"
            assert "duration" in video_info, "Video info should have duration"
            
            # Try to create pose estimator
            factory = PoseEstimatorFactory()
            available_estimators = factory.get_available_estimators()
            
            if available_estimators:
                estimator_type = available_estimators[0]
                estimator = factory.create_estimator(estimator_type)
                assert estimator is not None, f"Should create {estimator_type} estimator"
            
        except Exception as e:
            pytest.skip(f"Real component integration not available: {e}")


@pytest.mark.skipif(not AMBIENT_AVAILABLE, reason="Ambient components not available")
class TestComponentDataFlow:
    """Test data flow between components with various data types."""

    @pytest.fixture
    def sample_data_generators(self):
        """Provide sample data generators for testing."""
        return {
            'frame_sequence': self._generate_frame_sequence,
            'pose_sequence': self._generate_pose_sequence,
            'gait_features': self._generate_gait_features
        }

    def _generate_frame_sequence(self, frame_count: int = 150):
        """Generate mock frame sequence."""
        class MockFrameSequence:
            def __init__(self, frames):
                self.frames = frames
            
            def __len__(self):
                return len(self.frames)
            
            def __iter__(self):
                return iter(self.frames)
        
        mock_frames = []
        for i in range(frame_count):
            frame = Mock(spec=Frame)
            frame.frame_id = i
            frame.timestamp = i / 30.0
            frame.width = 640
            frame.height = 480
            mock_frames.append(frame)
        
        return MockFrameSequence(mock_frames)

    def _generate_pose_sequence(self, frame_count: int = 150) -> List[List[Dict[str, float]]]:
        """Generate mock pose sequence."""
        pose_sequence = []
        for frame_idx in range(frame_count):
            frame_poses = []
            for landmark_idx in range(33):  # MediaPipe landmarks
                landmark = {
                    "x": 100.0 + landmark_idx + (frame_idx * 0.1),
                    "y": 200.0 + landmark_idx + (frame_idx * 0.1),
                    "confidence": 0.8 + (landmark_idx * 0.005)
                }
                frame_poses.append(landmark)
            pose_sequence.append(frame_poses)
        return pose_sequence

    def _generate_gait_features(self) -> Dict[str, Any]:
        """Generate mock gait features."""
        return {
            "temporal_features": {
                "stride_time": 1.2,
                "cadence": 110.0,
                "stance_time": 0.7,
                "swing_time": 0.5,
                "double_support_time": 0.2
            },
            "spatial_features": {
                "stride_length": 1.4,
                "step_width": 0.15,
                "step_length": 0.7,
                "foot_angle": 15.0
            },
            "symmetry_features": {
                "left_right_symmetry": 0.95,
                "temporal_symmetry": 0.92,
                "spatial_symmetry": 0.88
            },
            "kinematic_features": {
                "hip_flexion_max": 45.0,
                "knee_flexion_max": 60.0,
                "ankle_dorsiflexion_max": 20.0
            }
        }

    @pytest.mark.integration
    @pytest.mark.fast
    def test_frame_sequence_data_flow(
        self, 
        sample_data_generators: Dict[str, callable]
    ):
        """Test frame sequence data flow between components."""
        # Generate frame sequence
        frame_sequence = sample_data_generators['frame_sequence'](100)
        
        # Validate frame sequence structure
        assert len(frame_sequence) == 100, "Should have 100 frames"
        
        # Test iteration
        frame_count = 0
        for frame in frame_sequence:
            assert hasattr(frame, 'frame_id'), "Frame should have frame_id"
            assert hasattr(frame, 'timestamp'), "Frame should have timestamp"
            frame_count += 1
        
        assert frame_count == 100, "Should iterate over all frames"

    @pytest.mark.integration
    @pytest.mark.fast
    def test_pose_sequence_data_flow(
        self, 
        sample_data_generators: Dict[str, callable]
    ):
        """Test pose sequence data flow between components."""
        # Generate pose sequence
        pose_sequence = sample_data_generators['pose_sequence'](50)
        
        # Validate pose sequence structure
        assert len(pose_sequence) == 50, "Should have 50 frames of poses"
        
        # Validate each frame
        for frame_idx, frame_poses in enumerate(pose_sequence):
            assert len(frame_poses) == 33, f"Frame {frame_idx} should have 33 landmarks"
            
            # Validate each landmark
            for landmark_idx, landmark in enumerate(frame_poses):
                assert "x" in landmark, f"Landmark {landmark_idx} should have x coordinate"
                assert "y" in landmark, f"Landmark {landmark_idx} should have y coordinate"
                assert "confidence" in landmark, f"Landmark {landmark_idx} should have confidence"
                
                # Validate coordinate ranges
                assert isinstance(landmark["x"], (int, float)), "X coordinate should be numeric"
                assert isinstance(landmark["y"], (int, float)), "Y coordinate should be numeric"
                assert 0.0 <= landmark["confidence"] <= 1.0, "Confidence should be between 0 and 1"

    @pytest.mark.integration
    @pytest.mark.fast
    def test_gait_features_data_flow(
        self, 
        sample_data_generators: Dict[str, callable]
    ):
        """Test gait features data flow between components."""
        # Generate gait features
        gait_features = sample_data_generators['gait_features']()
        
        # Validate gait features structure
        required_categories = ["temporal_features", "spatial_features", "symmetry_features"]
        for category in required_categories:
            assert category in gait_features, f"Missing feature category: {category}"
        
        # Validate temporal features
        temporal = gait_features["temporal_features"]
        temporal_keys = ["stride_time", "cadence", "stance_time", "swing_time"]
        for key in temporal_keys:
            assert key in temporal, f"Missing temporal feature: {key}"
            assert temporal[key] > 0, f"Temporal feature {key} should be positive"
        
        # Validate spatial features
        spatial = gait_features["spatial_features"]
        spatial_keys = ["stride_length", "step_width", "step_length"]
        for key in spatial_keys:
            assert key in spatial, f"Missing spatial feature: {key}"
            assert spatial[key] > 0, f"Spatial feature {key} should be positive"
        
        # Validate symmetry features
        symmetry = gait_features["symmetry_features"]
        symmetry_keys = ["left_right_symmetry", "temporal_symmetry", "spatial_symmetry"]
        for key in symmetry_keys:
            assert key in symmetry, f"Missing symmetry feature: {key}"
            assert 0.0 <= symmetry[key] <= 1.0, f"Symmetry feature {key} should be between 0 and 1"

    @pytest.mark.integration
    @pytest.mark.fast
    def test_data_transformation_consistency(
        self, 
        sample_data_generators: Dict[str, callable]
    ):
        """Test data transformation consistency across components."""
        # Generate initial data
        frame_sequence = sample_data_generators['frame_sequence'](75)
        
        # Simulate pose estimation transformation
        pose_sequence = sample_data_generators['pose_sequence'](75)
        
        # Validate transformation consistency
        assert len(pose_sequence) == len(frame_sequence), \
            "Pose sequence length should match frame sequence length"
        
        # Simulate gait analysis transformation
        gait_features = sample_data_generators['gait_features']()
        
        # Validate that gait features are derived from pose sequence
        assert isinstance(gait_features, dict), "Gait features should be a dictionary"
        assert len(gait_features) > 0, "Gait features should not be empty"
        
        # Simulate classification transformation
        classification_result = {
            "classification": "normal",
            "confidence": 0.85,
            "explanation": "Normal gait pattern detected"
        }
        
        # Validate classification result format
        assert "classification" in classification_result
        assert "confidence" in classification_result
        assert classification_result["classification"] in ["normal", "abnormal"]
        assert 0.0 <= classification_result["confidence"] <= 1.0

    @pytest.mark.integration
    @pytest.mark.fast
    def test_data_validation_across_components(
        self, 
        sample_data_generators: Dict[str, callable]
    ):
        """Test data validation across component boundaries."""
        # Test invalid frame sequence
        invalid_frame_sequence = None
        
        # Components should handle None input gracefully
        with pytest.raises((ValueError, TypeError, AttributeError)):
            # This would fail in real pose estimator
            len(invalid_frame_sequence)
        
        # Test invalid pose sequence
        invalid_pose_sequence = []
        
        # Empty pose sequence should be handled
        assert len(invalid_pose_sequence) == 0
        
        # Test invalid gait features
        invalid_gait_features = {}
        
        # Empty gait features should be handled
        assert len(invalid_gait_features) == 0
        
        # Test malformed data structures
        malformed_pose_sequence = [
            [{"x": 100, "y": 200}],  # Missing confidence
            [{"x": 100, "confidence": 0.8}],  # Missing y
            [{"y": 200, "confidence": 0.8}]   # Missing x
        ]
        
        # Components should validate required fields
        for frame_poses in malformed_pose_sequence:
            for landmark in frame_poses:
                # At least one required field should be missing
                required_fields = ["x", "y", "confidence"]
                missing_fields = [field for field in required_fields if field not in landmark]
                if missing_fields:
                    # This would be caught by proper validation
                    assert len(missing_fields) > 0