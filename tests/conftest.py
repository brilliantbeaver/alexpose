"""
Pytest configuration and shared fixtures for all tests.
"""

import pytest
import tempfile
from pathlib import Path
import numpy as np

try:
    import cv2
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False

try:
    import mediapipe as mp
    MEDIAPIPE_AVAILABLE = True
except ImportError:
    MEDIAPIPE_AVAILABLE = False


@pytest.fixture
def tmp_model_file(tmp_path):
    """Create a temporary MediaPipe model file for testing."""
    model_file = tmp_path / "pose_landmarker_full.task"
    # Write dummy data (actual model would be binary)
    model_file.write_bytes(b"dummy mediapipe model data" * 100)
    return model_file


@pytest.fixture
def sample_image_file(tmp_path):
    """Create a sample test image file."""
    if not CV2_AVAILABLE:
        pytest.skip("OpenCV not available")
    
    image_path = tmp_path / "test_image.jpg"
    # Create a simple test image (200x200, RGB)
    image = np.ones((200, 200, 3), dtype=np.uint8) * 128
    # Add some variation
    image[50:150, 50:150] = [255, 0, 0]  # Red square
    cv2.imwrite(str(image_path), image)
    return image_path


@pytest.fixture
def sample_video_file(tmp_path):
    """Create a sample test video file."""
    if not CV2_AVAILABLE:
        pytest.skip("OpenCV not available")
    
    video_path = tmp_path / "test_video.mp4"
    # Create a simple test video (20 frames, 200x200, 30 fps)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(str(video_path), fourcc, 30.0, (200, 200))
    
    for i in range(20):
        # Create frames with slight variation
        frame = np.ones((200, 200, 3), dtype=np.uint8) * (128 + i)
        out.write(frame)
    out.release()
    
    return video_path

