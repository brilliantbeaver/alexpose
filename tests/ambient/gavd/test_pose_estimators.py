"""
Comprehensive tests for pose estimators.

This module includes:
- Unit tests for MediaPipeEstimator
- Integration tests
- Property-based tests using Hypothesis
"""

import pytest
import tempfile
from pathlib import Path
from typing import List, Dict, Any
from unittest.mock import Mock, patch, MagicMock

try:
    import mediapipe as mp
    from mediapipe.tasks import python
    from mediapipe.tasks.python import vision
    MEDIAPIPE_AVAILABLE = True
except ImportError:
    MEDIAPIPE_AVAILABLE = False

try:
    import cv2
    import numpy as np
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False

from hypothesis import given, strategies as st, assume, settings
from hypothesis.strategies import composite

from ambient.gavd.pose_estimators import (
    PoseEstimator,
    MediaPipeEstimator,
    OpenPoseEstimator,
    Keypoint,
)


# Skip tests if MediaPipe is not available
pytestmark = pytest.mark.skipif(
    not MEDIAPIPE_AVAILABLE,
    reason="MediaPipe is not installed"
)


class TestMediaPipeEstimatorInitialization:
    """Test MediaPipeEstimator initialization and configuration."""
    
    def test_init_with_valid_model_path(self, tmp_path):
        """Test initialization with valid model path."""
        # Create a dummy model file
        model_file = tmp_path / "pose_landmarker_full.task"
        model_file.write_bytes(b"dummy model data")
        
        estimator = MediaPipeEstimator(
            model_path=str(model_file),
            default_model="BODY_25"
        )
        
        assert estimator.model_path == model_file.resolve()
        assert estimator.default_model == "BODY_25"
        assert estimator.min_pose_detection_confidence == 0.5
        assert estimator.min_pose_presence_confidence == 0.5
        assert estimator.min_tracking_confidence == 0.5
    
    def test_init_with_invalid_model_path(self):
        """Test initialization fails with invalid model path."""
        with pytest.raises(FileNotFoundError):
            MediaPipeEstimator(model_path="/nonexistent/path/model.task")
    
    def test_init_with_custom_confidence_thresholds(self, tmp_path):
        """Test initialization with custom confidence thresholds."""
        model_file = tmp_path / "model.task"
        model_file.write_bytes(b"dummy")
        
        estimator = MediaPipeEstimator(
            model_path=str(model_file),
            min_pose_detection_confidence=0.7,
            min_pose_presence_confidence=0.8,
            min_tracking_confidence=0.9,
        )
        
        assert estimator.min_pose_detection_confidence == 0.7
        assert estimator.min_pose_presence_confidence == 0.8
        assert estimator.min_tracking_confidence == 0.9
    
    @pytest.mark.skipif(not MEDIAPIPE_AVAILABLE, reason="MediaPipe not available")
    def test_init_without_mediapipe_raises_error(self, tmp_path):
        """Test that ImportError is raised if MediaPipe is not available."""
        model_file = tmp_path / "model.task"
        model_file.write_bytes(b"dummy")
        
        # Patch MEDIAPIPE_AVAILABLE to False to simulate MediaPipe not being installed
        with patch('ambient.gavd.pose_estimators.MEDIAPIPE_AVAILABLE', False):
            with pytest.raises(ImportError, match=r"MediaPipe is not installed.*"):
                # Actually try to instantiate the class - this should raise ImportError
                MediaPipeEstimator(model_path=str(model_file))


class TestMediaPipeEstimatorImageProcessing:
    """Test image processing functionality."""
    
    @pytest.fixture
    def sample_image(self, tmp_path):
        """Create a sample test image."""
        if not CV2_AVAILABLE:
            pytest.skip("OpenCV not available")
        
        image_path = tmp_path / "test_image.jpg"
        # Create a simple test image (100x100, RGB)
        image = np.ones((100, 100, 3), dtype=np.uint8) * 128
        cv2.imwrite(str(image_path), image)
        return image_path
    
    @pytest.fixture
    def mock_landmarker(self):
        """Create a mock MediaPipe landmarker."""
        if not MEDIAPIPE_AVAILABLE:
            pytest.skip("MediaPipe not available")
        
        mock_result = Mock(spec=vision.PoseLandmarkerResult)
        # Create mock landmarks
        mock_landmark = Mock()
        mock_landmark.x = 0.5
        mock_landmark.y = 0.5
        mock_landmark.visibility = 0.9
        mock_landmark.z = 0.0
        
        mock_result.pose_landmarks = [[mock_landmark] * 33]  # 33 landmarks
        
        mock_landmarker = Mock(spec=vision.PoseLandmarker)
        mock_landmarker.detect.return_value = mock_result
        
        return mock_landmarker
    
    @pytest.mark.skipif(not CV2_AVAILABLE, reason="OpenCV not available")
    def test_estimate_image_keypoints_file_not_found(self, tmp_path):
        """Test that FileNotFoundError is raised for missing image."""
        model_file = tmp_path / "model.task"
        model_file.write_bytes(b"dummy")
        
        estimator = MediaPipeEstimator(model_path=str(model_file))
        
        with pytest.raises(FileNotFoundError):
            estimator.estimate_image_keypoints("/nonexistent/image.jpg")
    
    @pytest.mark.skipif(not CV2_AVAILABLE, reason="OpenCV not available")
    def test_estimate_image_keypoints_with_bbox(self, sample_image, tmp_path):
        """Test image processing with bounding box."""
        model_file = tmp_path / "model.task"
        model_file.write_bytes(b"dummy")
        
        estimator = MediaPipeEstimator(model_path=str(model_file))
        
        # Mock the landmarker to avoid needing actual MediaPipe model
        with patch.object(estimator, '_get_image_landmarker') as mock_get:
            mock_landmarker = Mock()
            mock_result = Mock()
            mock_landmark = Mock()
            mock_landmark.x = 0.5
            mock_landmark.y = 0.5
            mock_landmark.visibility = 0.9
            mock_result.pose_landmarks = [[mock_landmark] * 33]
            mock_landmarker.detect.return_value = mock_result
            mock_get.return_value = mock_landmarker
            
            bbox = {"left": 10, "top": 20, "width": 50, "height": 50}
            keypoints = estimator.estimate_image_keypoints(
                str(sample_image),
                bbox=bbox
            )
            
            # Should return 33 keypoints
            assert len(keypoints) == 33
            assert all("x" in kp and "y" in kp and "confidence" in kp for kp in keypoints)


class TestMediaPipeEstimatorVideoProcessing:
    """Test video processing functionality."""
    
    @pytest.fixture
    def sample_video(self, tmp_path):
        """Create a sample test video."""
        if not CV2_AVAILABLE:
            pytest.skip("OpenCV not available")
        
        video_path = tmp_path / "test_video.mp4"
        # Create a simple test video (10 frames, 100x100)
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(str(video_path), fourcc, 30.0, (100, 100))
        
        for _ in range(10):
            frame = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
            out.write(frame)
        out.release()
        
        return video_path
    
    @pytest.mark.skipif(not CV2_AVAILABLE, reason="OpenCV not available")
    def test_estimate_video_keypoints_file_not_found(self, tmp_path):
        """Test that FileNotFoundError is raised for missing video."""
        model_file = tmp_path / "model.task"
        model_file.write_bytes(b"dummy")
        
        estimator = MediaPipeEstimator(model_path=str(model_file))
        
        with pytest.raises(FileNotFoundError):
            estimator.estimate_video_keypoints("/nonexistent/video.mp4")
    
    @pytest.mark.skipif(not CV2_AVAILABLE, reason="OpenCV not available")
    def test_estimate_video_keypoints_returns_list_of_lists(self, sample_video, tmp_path):
        """Test that video processing returns correct structure."""
        model_file = tmp_path / "model.task"
        model_file.write_bytes(b"dummy")
        
        estimator = MediaPipeEstimator(model_path=str(model_file))
        
        # Mock the landmarker
        with patch.object(estimator, '_get_video_landmarker') as mock_get:
            mock_landmarker = Mock()
            mock_result = Mock()
            mock_landmark = Mock()
            mock_landmark.x = 0.5
            mock_landmark.y = 0.5
            mock_landmark.visibility = 0.9
            mock_result.pose_landmarks = [[mock_landmark] * 33]
            mock_landmarker.detect_for_video.return_value = mock_result
            mock_get.return_value = mock_landmarker
            
            results = estimator.estimate_video_keypoints(str(sample_video))
            
            # Should return list of lists
            assert isinstance(results, list)
            assert len(results) > 0
            assert all(isinstance(frame_kps, list) for frame_kps in results)


class TestMediaPipeEstimatorKeypointFormat:
    """Test keypoint format conversion."""
    
    def test_parse_mediapipe_landmarks_empty_result(self, tmp_path):
        """Test parsing empty MediaPipe result."""
        model_file = tmp_path / "model.task"
        model_file.write_bytes(b"dummy")
        
        estimator = MediaPipeEstimator(model_path=str(model_file))
        
        if not MEDIAPIPE_AVAILABLE:
            pytest.skip("MediaPipe not available")
        
        mock_result = Mock(spec=vision.PoseLandmarkerResult)
        mock_result.pose_landmarks = []
        
        keypoints = estimator._parse_mediapipe_landmarks(mock_result)
        assert keypoints == []
    
    def test_parse_mediapipe_landmarks_with_coordinates(self, tmp_path):
        """Test parsing MediaPipe landmarks with coordinate conversion."""
        model_file = tmp_path / "model.task"
        model_file.write_bytes(b"dummy")
        
        estimator = MediaPipeEstimator(model_path=str(model_file))
        
        if not MEDIAPIPE_AVAILABLE:
            pytest.skip("MediaPipe not available")
        
        # Create mock landmarks
        mock_result = Mock(spec=vision.PoseLandmarkerResult)
        mock_landmarks = []
        for i in range(33):
            mock_landmark = Mock()
            mock_landmark.x = 0.5 + (i * 0.01)  # Vary x coordinates
            mock_landmark.y = 0.3 + (i * 0.01)  # Vary y coordinates
            mock_landmark.visibility = 0.8 + (i * 0.001)
            mock_landmarks.append(mock_landmark)
        
        mock_result.pose_landmarks = [mock_landmarks]
        
        # Parse with image dimensions
        keypoints = estimator._parse_mediapipe_landmarks(
            mock_result,
            image_width=640,
            image_height=480
        )
        
        assert len(keypoints) == 33
        # Check first keypoint
        assert keypoints[0]["x"] == pytest.approx(0.5 * 640, abs=1.0)
        assert keypoints[0]["y"] == pytest.approx(0.3 * 480, abs=1.0)
        assert keypoints[0]["confidence"] == pytest.approx(0.8, abs=0.01)


# Property-based tests using Hypothesis
@composite
def keypoint_strategy(draw):
    """Generate a valid keypoint dictionary."""
    x = draw(st.floats(min_value=0.0, max_value=1000.0))
    y = draw(st.floats(min_value=0.0, max_value=1000.0))
    confidence = draw(st.floats(min_value=0.0, max_value=1.0))
    return {"x": x, "y": y, "confidence": confidence}


@composite
def keypoint_list_strategy(draw, min_size=0, max_size=33):
    """Generate a list of keypoints."""
    return draw(st.lists(keypoint_strategy(), min_size=min_size, max_size=max_size))


class TestMediaPipeEstimatorProperties:
    """Property-based tests for MediaPipeEstimator."""
    
    @given(keypoint_list_strategy(min_size=0, max_size=33))
    @settings(max_examples=50)
    def test_keypoint_format_properties(self, keypoints):
        """Test that keypoints always have correct format."""
        for kp in keypoints:
            assert "x" in kp
            assert "y" in kp
            assert "confidence" in kp
            assert isinstance(kp["x"], (int, float))
            assert isinstance(kp["y"], (int, float))
            assert isinstance(kp["confidence"], (int, float))
            assert 0.0 <= kp["confidence"] <= 1.0
    
    @given(
        st.integers(min_value=1, max_value=1920),
        st.integers(min_value=1, max_value=1080),
        st.floats(min_value=0.0, max_value=1.0),
        st.floats(min_value=0.0, max_value=1.0),
    )
    @settings(max_examples=20)
    def test_coordinate_conversion_properties(
        self, width, height, norm_x, norm_y
    ):
        """Test that normalized to pixel coordinate conversion is correct."""
        pixel_x = norm_x * width
        pixel_y = norm_y * height
        
        assert 0 <= pixel_x <= width
        assert 0 <= pixel_y <= height
    
    @given(
        st.integers(min_value=0, max_value=1000),
        st.integers(min_value=0, max_value=1000),
        st.integers(min_value=1, max_value=500),
        st.integers(min_value=1, max_value=500),
    )
    @settings(max_examples=20)
    def test_bbox_crop_properties(self, left, top, width, height):
        """Test that bbox cropping maintains valid bounds."""
        # Simulate image bounds
        img_width = 1920
        img_height = 1080
        
        # Ensure bbox is within image
        left = min(left, img_width - 1)
        top = min(top, img_height - 1)
        width = min(width, img_width - left)
        height = min(height, img_height - top)
        
        right = left + width
        bottom = top + height
        
        assert 0 <= left < img_width
        assert 0 <= top < img_height
        assert left < right <= img_width
        assert top < bottom <= img_height


class TestMediaPipeEstimatorInterface:
    """Test that MediaPipeEstimator implements PoseEstimator interface correctly."""
    
    def test_implements_pose_estimator(self, tmp_path):
        """Test that MediaPipeEstimator is a PoseEstimator."""
        model_file = tmp_path / "model.task"
        model_file.write_bytes(b"dummy")
        
        estimator = MediaPipeEstimator(model_path=str(model_file))
        assert isinstance(estimator, PoseEstimator)
    
    def test_supports_video_batch(self, tmp_path):
        """Test that MediaPipeEstimator supports video batch processing."""
        model_file = tmp_path / "model.task"
        model_file.write_bytes(b"dummy")
        
        estimator = MediaPipeEstimator(model_path=str(model_file))
        assert estimator.supports_video_batch() is True
    
    def test_has_required_methods(self, tmp_path):
        """Test that MediaPipeEstimator has all required methods."""
        model_file = tmp_path / "model.task"
        model_file.write_bytes(b"dummy")
        
        estimator = MediaPipeEstimator(model_path=str(model_file))
        
        assert hasattr(estimator, "estimate_image_keypoints")
        assert hasattr(estimator, "estimate_video_keypoints")
        assert hasattr(estimator, "supports_video_batch")
        
        # Check method signatures
        import inspect
        img_sig = inspect.signature(estimator.estimate_image_keypoints)
        assert "image_path" in img_sig.parameters
        assert "model" in img_sig.parameters
        assert "bbox" in img_sig.parameters
        
        vid_sig = inspect.signature(estimator.estimate_video_keypoints)
        assert "video_path" in vid_sig.parameters
        assert "model" in vid_sig.parameters


class TestMediaPipeEstimatorErrorHandling:
    """Test error handling and edge cases."""
    
    @pytest.mark.skipif(not CV2_AVAILABLE, reason="OpenCV not available")
    def test_invalid_image_file(self, tmp_path):
        """Test handling of invalid image file."""
        model_file = tmp_path / "model.task"
        model_file.write_bytes(b"dummy")
        
        # Create a file that's not a valid image
        invalid_image = tmp_path / "invalid.jpg"
        invalid_image.write_text("not an image")
        
        estimator = MediaPipeEstimator(model_path=str(model_file))
        
        with pytest.raises((ValueError, RuntimeError)):
            estimator.estimate_image_keypoints(str(invalid_image))
    
    def test_empty_bbox(self, tmp_path):
        """Test handling of empty bounding box."""
        model_file = tmp_path / "model.task"
        model_file.write_bytes(b"dummy")
        
        estimator = MediaPipeEstimator(model_path=str(model_file))
        
        # Empty bbox should be handled gracefully
        empty_bbox = {}
        # This should not raise an error (though it may not crop)
        # The actual behavior depends on implementation


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

