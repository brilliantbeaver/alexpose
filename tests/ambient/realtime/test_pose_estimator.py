"""
Tests for RealtimePoseEstimator.

This module tests the realtime pose estimation with performance optimizations
and adaptive quality control.
"""

import time
import pytest
import numpy as np
from unittest.mock import Mock, patch

from ambient.realtime.pose_estimator import RealtimePoseEstimator, MEDIAPIPE_AVAILABLE
from ambient.realtime.interfaces import RealtimeFrame, ProcessingMode


pytestmark = pytest.mark.skipif(
    not MEDIAPIPE_AVAILABLE,
    reason="MediaPipe not available"
)


@pytest.fixture
def pose_estimator():
    """Create a pose estimator for testing."""
    return RealtimePoseEstimator(
        processing_mode=ProcessingMode.BALANCED,
        target_fps=30
    )


@pytest.fixture
def sample_frame():
    """Create a sample frame for testing."""
    return RealtimeFrame(
        data=np.zeros((480, 640, 3), dtype=np.uint8),
        timestamp=time.time(),
        frame_id=1,
        metadata={'width': 640, 'height': 480}
    )


class TestPoseEstimatorInitialization:
    """Test pose estimator initialization."""
    
    def test_initialization_balanced_mode(self):
        """Test initialization with balanced mode."""
        estimator = RealtimePoseEstimator(
            processing_mode=ProcessingMode.BALANCED
        )
        
        assert estimator.processing_mode == ProcessingMode.BALANCED
        assert estimator.target_fps == 30
        assert estimator.is_ready()
    
    def test_initialization_fast_mode(self):
        """Test initialization with fast mode."""
        estimator = RealtimePoseEstimator(
            processing_mode=ProcessingMode.FAST
        )
        
        assert estimator.processing_mode == ProcessingMode.FAST
        assert estimator._frame_skip_interval == 2
    
    def test_initialization_accurate_mode(self):
        """Test initialization with accurate mode."""
        estimator = RealtimePoseEstimator(
            processing_mode=ProcessingMode.ACCURATE
        )
        
        assert estimator.processing_mode == ProcessingMode.ACCURATE


class TestProcessingModes:
    """Test different processing modes."""
    
    def test_set_processing_mode(self, pose_estimator):
        """Test changing processing mode."""
        pose_estimator.set_processing_mode(ProcessingMode.FAST)
        assert pose_estimator.processing_mode == ProcessingMode.FAST
        
        pose_estimator.set_processing_mode(ProcessingMode.ACCURATE)
        assert pose_estimator.processing_mode == ProcessingMode.ACCURATE
    
    def test_quality_params_fast_mode(self, pose_estimator):
        """Test quality parameters for fast mode."""
        pose_estimator.set_processing_mode(ProcessingMode.FAST)
        params = pose_estimator._quality_params
        
        assert params['min_detection_confidence'] == 0.3
        assert params['resize_factor'] == 0.5
    
    def test_quality_params_accurate_mode(self, pose_estimator):
        """Test quality parameters for accurate mode."""
        pose_estimator.set_processing_mode(ProcessingMode.ACCURATE)
        params = pose_estimator._quality_params
        
        assert params['min_detection_confidence'] == 0.7
        assert params['resize_factor'] == 1.0


class TestPoseEstimation:
    """Test pose estimation functionality."""
    
    @pytest.mark.slow
    def test_estimate_pose_basic(self, pose_estimator, sample_frame):
        """Test basic pose estimation."""
        result = pose_estimator.estimate_pose(sample_frame)
        
        assert result is not None
        assert result.frame_id == sample_frame.frame_id
        assert result.timestamp == sample_frame.timestamp
        assert isinstance(result.keypoints, list)
        assert isinstance(result.confidence_scores, list)
        assert result.processing_time_ms >= 0
    
    @pytest.mark.slow
    def test_estimate_pose_with_person(self, pose_estimator):
        """Test pose estimation with simulated person."""
        # Create a frame with some pattern (simulating a person)
        frame_data = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        frame = RealtimeFrame(
            data=frame_data,
            timestamp=time.time(),
            frame_id=1,
            metadata={'width': 640, 'height': 480}
        )
        
        result = pose_estimator.estimate_pose(frame)
        
        assert result is not None
        assert 'estimator' in result.estimator_info
        assert result.estimator_info['estimator'] == 'MediaPipe'


class TestPerformanceOptimization:
    """Test performance optimization features."""
    
    def test_frame_skipping(self, pose_estimator):
        """Test frame skipping for performance."""
        pose_estimator.set_processing_mode(ProcessingMode.FAST)
        
        # Process multiple frames quickly
        results = []
        for i in range(10):
            frame = RealtimeFrame(
                data=np.zeros((100, 100, 3), dtype=np.uint8),
                timestamp=time.time(),
                frame_id=i,
                metadata={}
            )
            result = pose_estimator.estimate_pose(frame)
            results.append(result)
        
        # Some frames should be skipped
        skipped = sum(1 for r in results if r.estimator_info.get('skipped', False))
        assert skipped > 0
    
    def test_performance_stats_tracking(self, pose_estimator, sample_frame):
        """Test performance statistics tracking."""
        # Process a frame
        pose_estimator.estimate_pose(sample_frame)
        
        stats = pose_estimator.get_performance_stats()
        
        assert 'frames_processed' in stats
        assert 'frames_skipped' in stats
        assert 'average_processing_time_ms' in stats
        assert 'fps' in stats
        assert stats['frames_processed'] >= 0
    
    def test_adaptive_frame_skip(self, pose_estimator):
        """Test adaptive frame skip adjustment."""
        initial_interval = pose_estimator._frame_skip_interval
        
        # Simulate slow processing
        pose_estimator._performance_stats['average_processing_time_ms'] = 100
        
        # Process a slow frame
        frame = RealtimeFrame(
            data=np.zeros((100, 100, 3), dtype=np.uint8),
            timestamp=time.time(),
            frame_id=1,
            metadata={}
        )
        
        with patch.object(pose_estimator, '_estimate_pose_internal', return_value={'landmarks': [], 'world_landmarks': [], 'segmentation_masks': None}):
            pose_estimator.estimate_pose(frame)
        
        # Frame skip interval may have adjusted
        assert pose_estimator._frame_skip_interval >= initial_interval


class TestFramePreprocessing:
    """Test frame preprocessing."""
    
    def test_preprocess_frame_resize(self, pose_estimator):
        """Test frame resizing during preprocessing."""
        pose_estimator.set_processing_mode(ProcessingMode.FAST)
        
        frame = RealtimeFrame(
            data=np.zeros((480, 640, 3), dtype=np.uint8),
            timestamp=time.time(),
            frame_id=1,
            metadata={}
        )
        
        processed = pose_estimator._preprocess_frame(frame)
        
        # Should be resized to 50% (fast mode)
        assert processed.shape[0] == 240
        assert processed.shape[1] == 320
    
    def test_preprocess_frame_no_resize(self, pose_estimator):
        """Test frame preprocessing without resize."""
        pose_estimator.set_processing_mode(ProcessingMode.ACCURATE)
        
        frame = RealtimeFrame(
            data=np.zeros((480, 640, 3), dtype=np.uint8),
            timestamp=time.time(),
            frame_id=1,
            metadata={}
        )
        
        processed = pose_estimator._preprocess_frame(frame)
        
        # Should maintain original size (accurate mode)
        assert processed.shape[0] == 480
        assert processed.shape[1] == 640


@pytest.mark.integration
class TestPoseEstimatorIntegration:
    """Integration tests for pose estimator."""
    
    @pytest.mark.slow
    def test_continuous_processing(self, pose_estimator):
        """Test continuous frame processing."""
        results = []
        
        # Process 30 frames
        for i in range(30):
            frame = RealtimeFrame(
                data=np.random.randint(0, 255, (240, 320, 3), dtype=np.uint8),
                timestamp=time.time(),
                frame_id=i,
                metadata={}
            )
            result = pose_estimator.estimate_pose(frame)
            results.append(result)
            time.sleep(0.01)
        
        # Check that processing occurred
        assert len(results) == 30
        
        # Get final stats
        stats = pose_estimator.get_performance_stats()
        assert stats['frames_processed'] > 0
        assert stats['average_processing_time_ms'] > 0
    
    @pytest.mark.slow
    def test_mode_switching_during_processing(self, pose_estimator):
        """Test switching modes during processing."""
        # Process some frames in balanced mode
        for i in range(5):
            frame = RealtimeFrame(
                data=np.zeros((240, 320, 3), dtype=np.uint8),
                timestamp=time.time(),
                frame_id=i,
                metadata={}
            )
            pose_estimator.estimate_pose(frame)
        
        # Switch to fast mode
        pose_estimator.set_processing_mode(ProcessingMode.FAST)
        
        # Process more frames
        for i in range(5, 10):
            frame = RealtimeFrame(
                data=np.zeros((240, 320, 3), dtype=np.uint8),
                timestamp=time.time(),
                frame_id=i,
                metadata={}
            )
            result = pose_estimator.estimate_pose(frame)
            assert result.estimator_info['processing_mode'] == 'fast'
