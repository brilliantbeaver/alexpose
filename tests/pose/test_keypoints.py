"""
Tests for ambient.pose.keypoints module

This test suite covers:
- BoundingBoxProcessor functionality
- KeypointGenerator functionality
- PoseKeypointExtractor functionality
- MediaPipeModelManager functionality
- PoseLandmarkerFactory functionality
- SequenceKeypointExtractor functionality
- KeypointVisualizer functionality
"""

import pytest
import numpy as np
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock

from ambient.pose.keypoints import (
    BoundingBoxProcessor,
    KeypointGenerator,
    PoseKeypointExtractor,
    MediaPipeModelManager,
    PoseLandmarkerFactory,
    SequenceKeypointExtractor,
    KeypointVisualizer,
    POSE_LANDMARK_NAMES,
    ensure_model_downloaded,
    get_keypoints,
    create_pose_landmarker,
)


class TestBoundingBoxProcessor:
    """Test BoundingBoxProcessor class."""
    
    def test_calculate_center_valid_bbox(self):
        """Test center calculation with valid bounding box."""
        processor = BoundingBoxProcessor()
        bbox = {"left": 10, "top": 20, "width": 100, "height": 200}
        
        center_x, center_y = processor.calculate_center(bbox)
        
        assert center_x == 60.0  # 10 + 100/2
        assert center_y == 120.0  # 20 + 200/2
    
    def test_calculate_center_zero_bbox(self):
        """Test center calculation with zero-sized bbox."""
        processor = BoundingBoxProcessor()
        bbox = {"left": 0, "top": 0, "width": 0, "height": 0}
        
        center_x, center_y = processor.calculate_center(bbox)
        
        assert center_x == 0.0
        assert center_y == 0.0
    
    def test_calculate_center_missing_keys(self):
        """Test center calculation with missing keys uses defaults."""
        processor = BoundingBoxProcessor()
        bbox = {"left": 10}  # Missing other keys
        
        center_x, center_y = processor.calculate_center(bbox)
        
        assert center_x == 10.0  # left + 0/2
        assert center_y == 0.0   # 0 + 0/2
    
    def test_calculate_center_invalid_bbox(self):
        """Test center calculation with invalid bbox raises error."""
        processor = BoundingBoxProcessor()
        
        with pytest.raises(ValueError, match="Bounding box must be a non-empty dictionary"):
            processor.calculate_center(None)
        
        with pytest.raises(ValueError, match="Bounding box must be a non-empty dictionary"):
            processor.calculate_center({})


class TestKeypointGenerator:
    """Test KeypointGenerator class."""
    
    def test_create_keypoint_basic(self):
        """Test basic keypoint creation."""
        kp = KeypointGenerator.create_keypoint(100.0, 200.0, 0.9)
        
        assert kp["x"] == 100.0
        assert kp["y"] == 200.0
        assert kp["confidence"] == 0.9
    
    def test_create_keypoint_confidence_clamping(self):
        """Test confidence is clamped to valid range."""
        # Test upper bound
        kp_high = KeypointGenerator.create_keypoint(0, 0, 1.5)
        assert kp_high["confidence"] == 1.0
        
        # Test lower bound
        kp_low = KeypointGenerator.create_keypoint(0, 0, -0.5)
        assert kp_low["confidence"] == 0.0
    
    def test_generate_grid_keypoints_default(self):
        """Test grid keypoint generation with defaults."""
        keypoints = KeypointGenerator.generate_grid_keypoints(100.0, 200.0)
        
        assert len(keypoints) == 25  # Default num_keypoints
        assert all("x" in kp and "y" in kp and "confidence" in kp for kp in keypoints)
    
    def test_generate_grid_keypoints_custom(self):
        """Test grid keypoint generation with custom parameters."""
        keypoints = KeypointGenerator.generate_grid_keypoints(
            center_x=50.0,
            center_y=50.0,
            num_keypoints=9,
            grid_spacing=10.0,
            confidence=0.7
        )
        
        assert len(keypoints) == 9
        assert all(kp["confidence"] == 0.7 for kp in keypoints)
        
        # Check that keypoints are distributed around center
        x_coords = [kp["x"] for kp in keypoints]
        y_coords = [kp["y"] for kp in keypoints]
        assert min(x_coords) < 50.0 < max(x_coords)
        assert min(y_coords) < 50.0 < max(y_coords)


class TestPoseKeypointExtractor:
    """Test PoseKeypointExtractor class."""
    
    def test_extract_from_bbox_valid(self):
        """Test keypoint extraction from valid bbox."""
        extractor = PoseKeypointExtractor()
        bbox = {"left": 10, "top": 20, "width": 100, "height": 200}
        
        keypoints = extractor.extract_from_bbox(bbox)
        
        assert len(keypoints) == 25  # Default
        assert all("x" in kp and "y" in kp for kp in keypoints)
    
    def test_extract_from_bbox_custom_params(self):
        """Test extraction with custom parameters."""
        extractor = PoseKeypointExtractor()
        bbox = {"left": 0, "top": 0, "width": 100, "height": 100}
        
        keypoints = extractor.extract_from_bbox(
            bbox,
            num_keypoints=16,
            grid_spacing=8.0,
            confidence=0.95
        )
        
        assert len(keypoints) == 16
        assert all(kp["confidence"] == 0.95 for kp in keypoints)
    
    def test_extract_from_bbox_invalid(self):
        """Test extraction with invalid bbox raises error."""
        extractor = PoseKeypointExtractor()
        
        with pytest.raises(ValueError):
            extractor.extract_from_bbox(None)
        
        with pytest.raises(ValueError):
            extractor.extract_from_bbox({})


class TestMediaPipeModelManager:
    """Test MediaPipeModelManager class."""
    
    def test_init_default_path(self):
        """Test initialization with default path."""
        manager = MediaPipeModelManager()
        
        assert manager.models_dir.name == "models"
        assert "data" in str(manager.models_dir)
    
    def test_init_custom_path(self, tmp_path):
        """Test initialization with custom path."""
        custom_dir = tmp_path / "custom_models"
        manager = MediaPipeModelManager(custom_dir)
        
        assert manager.models_dir == custom_dir
        assert custom_dir.exists()
    
    def test_get_model_path(self, tmp_path):
        """Test getting model path."""
        manager = MediaPipeModelManager(tmp_path)
        
        model_path = manager.get_model_path("test_model.task")
        
        assert model_path == tmp_path / "test_model.task"
    
    def test_is_model_downloaded(self, tmp_path):
        """Test checking if model is downloaded."""
        manager = MediaPipeModelManager(tmp_path)
        
        # Model doesn't exist
        assert not manager.is_model_downloaded("test_model.task")
        
        # Create model file
        model_path = tmp_path / "test_model.task"
        model_path.write_text("fake model")
        
        # Model exists
        assert manager.is_model_downloaded("test_model.task")
    
    @patch('urllib.request.urlretrieve')
    def test_download_model_success(self, mock_retrieve, tmp_path):
        """Test successful model download."""
        manager = MediaPipeModelManager(tmp_path)
        model_path = tmp_path / "test_model.task"
        
        # Simulate successful download
        def create_file(*args):
            model_path.write_bytes(b"fake model data" * 1000)
        
        mock_retrieve.side_effect = create_file
        
        result = manager.download_model(
            model_url="http://example.com/model.task",
            model_name="test_model.task"
        )
        
        assert result == str(model_path)
        assert model_path.exists()
        mock_retrieve.assert_called_once()
    
    def test_download_model_already_exists(self, tmp_path):
        """Test download when model already exists."""
        manager = MediaPipeModelManager(tmp_path)
        model_path = tmp_path / "test_model.task"
        model_path.write_text("existing model")
        
        result = manager.download_model(model_name="test_model.task")
        
        assert result == str(model_path)
    
    @patch('urllib.request.urlretrieve')
    def test_download_model_failure(self, mock_retrieve, tmp_path):
        """Test download failure handling."""
        manager = MediaPipeModelManager(tmp_path)
        
        mock_retrieve.side_effect = Exception("Download failed")
        
        result = manager.download_model(
            model_url="http://example.com/model.task",
            model_name="test_model.task"
        )
        
        assert result is None
    
    def test_ensure_model_available_exists(self, tmp_path):
        """Test ensuring model is available when it exists."""
        manager = MediaPipeModelManager(tmp_path)
        model_path = tmp_path / "test_model.task"
        model_path.write_text("existing model")
        
        result = manager.ensure_model_available("test_model.task")
        
        assert result == str(model_path)


class TestPoseLandmarkerFactory:
    """Test PoseLandmarkerFactory class."""
    
    @pytest.mark.skipif(
        not hasattr(PoseLandmarkerFactory, 'create_landmarker'),
        reason="MediaPipe not available"
    )
    @patch('ambient.pose.keypoints.vision.PoseLandmarker.create_from_options')
    def test_create_landmarker_success(self, mock_create):
        """Test successful landmarker creation."""
        mock_landmarker = Mock()
        mock_create.return_value = mock_landmarker
        
        result = PoseLandmarkerFactory.create_landmarker("fake_model.task")
        
        assert result == mock_landmarker
        mock_create.assert_called_once()
    
    @patch('ambient.pose.keypoints.MEDIAPIPE_AVAILABLE', False)
    def test_create_landmarker_no_mediapipe(self):
        """Test landmarker creation without MediaPipe."""
        with pytest.raises(ImportError, match="MediaPipe is not available"):
            PoseLandmarkerFactory.create_landmarker("fake_model.task")


class TestSequenceKeypointExtractor:
    """Test SequenceKeypointExtractor class."""
    
    def test_init_default(self):
        """Test initialization with defaults."""
        extractor = SequenceKeypointExtractor()
        
        assert extractor.model_manager is not None
        assert extractor.landmarker_factory is not None
        assert extractor._landmarker is None
    
    def test_init_custom(self):
        """Test initialization with custom components."""
        custom_manager = MediaPipeModelManager()
        custom_factory = PoseLandmarkerFactory()
        
        extractor = SequenceKeypointExtractor(
            model_manager=custom_manager,
            landmarker_factory=custom_factory
        )
        
        assert extractor.model_manager == custom_manager
        assert extractor.landmarker_factory == custom_factory


class TestKeypointVisualizer:
    """Test KeypointVisualizer class."""
    
    def test_add_landmark_names(self):
        """Test adding landmark names to keypoints."""
        keypoints = [
            {"id": 0, "x": 100, "y": 200},
            {"id": 1, "x": 150, "y": 250},
        ]
        
        result = KeypointVisualizer.add_landmark_names(keypoints)
        
        assert result[0]["name"] == POSE_LANDMARK_NAMES[0]
        assert result[1]["name"] == POSE_LANDMARK_NAMES[1]
    
    def test_add_landmark_names_preserves_existing(self):
        """Test that existing names are preserved."""
        keypoints = [
            {"id": 0, "x": 100, "y": 200, "name": "CUSTOM_NAME"},
        ]
        
        result = KeypointVisualizer.add_landmark_names(keypoints)
        
        assert result[0]["name"] == "CUSTOM_NAME"
    
    def test_draw_keypoints(self):
        """Test drawing keypoints on image."""
        image = np.zeros((480, 640, 3), dtype=np.uint8)
        keypoints = [
            {"x": 100, "y": 200, "confidence": 0.9},
            {"x": 150, "y": 250, "confidence": 0.3},  # Below threshold
        ]
        
        result = KeypointVisualizer.draw_keypoints(
            image, keypoints, confidence_threshold=0.5
        )
        
        assert result.shape == image.shape
        assert not np.array_equal(result, image)  # Image was modified
    
    def test_get_summary_stats_empty(self):
        """Test summary stats with empty keypoints."""
        stats = KeypointVisualizer.get_summary_stats([])
        
        assert stats["total_landmarks"] == 0
        assert stats["visible_landmarks"] == 0
        assert stats["avg_confidence"] == 0.0
    
    def test_get_summary_stats_valid(self):
        """Test summary stats with valid keypoints."""
        keypoints = [
            {"confidence": 0.9},
            {"confidence": 0.7},
            {"confidence": 0.3},
        ]
        
        stats = KeypointVisualizer.get_summary_stats(keypoints)
        
        assert stats["total_landmarks"] == 3
        assert stats["visible_landmarks"] == 2  # > 0.5
        assert 0.6 < stats["avg_confidence"] < 0.7
        assert stats["min_confidence"] == 0.3
        assert stats["max_confidence"] == 0.9


class TestConvenienceFunctions:
    """Test convenience functions."""
    
    @patch('ambient.pose.keypoints.MediaPipeModelManager')
    def test_ensure_model_downloaded(self, mock_manager_class, tmp_path):
        """Test ensure_model_downloaded convenience function."""
        mock_manager = Mock()
        mock_manager.ensure_model_available.return_value = str(tmp_path / "model.task")
        mock_manager_class.return_value = mock_manager
        
        result = ensure_model_downloaded(tmp_path)
        
        assert result == str(tmp_path / "model.task")
        mock_manager.ensure_model_available.assert_called_once()
    
    @patch('ambient.pose.keypoints.PoseLandmarkerFactory')
    def test_create_pose_landmarker(self, mock_factory_class):
        """Test create_pose_landmarker convenience function."""
        mock_factory = Mock()
        mock_landmarker = Mock()
        mock_factory.create_landmarker.return_value = mock_landmarker
        mock_factory_class.return_value = mock_factory
        
        result = create_pose_landmarker("model.task")
        
        assert result == mock_landmarker


class TestGetKeypointsFunction:
    """Test get_keypoints convenience function."""
    
    @patch('ambient.pose.keypoints.SequenceKeypointExtractor')
    def test_get_keypoints_with_dataframe(self, mock_extractor_class, tmp_path):
        """Test get_keypoints with DataFrame input."""
        import pandas as pd
        
        # Mock the extractor
        mock_extractor = Mock()
        mock_keypoints = [Mock(), Mock()]  # Mock keypoint objects
        mock_extractor.extract_from_sequence.return_value = mock_keypoints
        mock_extractor_class.return_value = mock_extractor
        
        # Create test DataFrame
        sequence_data = pd.DataFrame({
            'frame_num': [1, 2, 3],
            'url': ['http://example.com/video1', 'http://example.com/video1', 'http://example.com/video1'],
            'seq': ['seq1', 'seq1', 'seq1']
        })
        
        result = get_keypoints(tmp_path, sequence_data, verbose=False)
        
        assert result == mock_keypoints
        mock_extractor.extract_from_sequence.assert_called_once_with(
            sequence_data,
            tmp_path / "data" / "youtube",
            None,
            False
        )
    
    def test_get_keypoints_with_invalid_input(self, tmp_path):
        """Test get_keypoints with invalid input types."""
        
        # Test with dict (no longer supported)
        with pytest.raises(TypeError, match="sequence_data must be pd.DataFrame"):
            get_keypoints(tmp_path, {"seq1": "not_a_dataframe"})
        
        # Test with string
        with pytest.raises(TypeError, match="sequence_data must be pd.DataFrame"):
            get_keypoints(tmp_path, "invalid_input")
        
        # Test with None
        with pytest.raises(TypeError, match="sequence_data must be pd.DataFrame"):
            get_keypoints(tmp_path, None)


# Property-based tests using hypothesis
try:
    from hypothesis import given, strategies as st
    
    class TestKeypointPropertiesHypothesis:
        """Property-based tests for keypoint operations."""
        
        @given(
            x=st.floats(min_value=-1000, max_value=1000),
            y=st.floats(min_value=-1000, max_value=1000),
            confidence=st.floats(min_value=-1, max_value=2)
        )
        def test_create_keypoint_always_valid_confidence(self, x, y, confidence):
            """Property: Created keypoints always have valid confidence."""
            kp = KeypointGenerator.create_keypoint(x, y, confidence)
            
            assert 0.0 <= kp["confidence"] <= 1.0
        
        @given(
            left=st.floats(min_value=0, max_value=1000),
            top=st.floats(min_value=0, max_value=1000),
            width=st.floats(min_value=0, max_value=1000),
            height=st.floats(min_value=0, max_value=1000)
        )
        def test_bbox_center_within_bounds(self, left, top, width, height):
            """Property: Bbox center is always within bbox bounds."""
            bbox = {"left": left, "top": top, "width": width, "height": height}
            processor = BoundingBoxProcessor()
            
            center_x, center_y = processor.calculate_center(bbox)
            
            assert left <= center_x <= left + width
            assert top <= center_y <= top + height

except ImportError:
    # Hypothesis not available, skip property tests
    pass


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
