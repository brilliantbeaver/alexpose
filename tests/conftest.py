"""
Pytest configuration and shared fixtures for all tests.
"""

import pytest
import tempfile
import os
import shutil
from pathlib import Path
import numpy as np
from typing import Dict, List, Any, Optional
from unittest.mock import Mock, MagicMock
import yaml
import json

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

try:
    from hypothesis import settings, Verbosity
    # Configure Hypothesis profiles
    settings.register_profile("dev", max_examples=10, deadline=1000, verbosity=Verbosity.normal)
    settings.register_profile("ci", max_examples=100, deadline=5000, verbosity=Verbosity.normal)
    settings.register_profile("thorough", max_examples=1000, deadline=10000, verbosity=Verbosity.verbose)
    
    # Load profile from environment or default to dev
    profile = os.getenv("HYPOTHESIS_PROFILE", "dev")
    settings.load_profile(profile)
    
    HYPOTHESIS_AVAILABLE = True
except ImportError:
    HYPOTHESIS_AVAILABLE = False

# Import ambient components for testing
try:
    from ambient.core.frame import Frame, FrameSequence
    from ambient.core.data_models import GaitFeatures, GaitMetrics, Keypoint
    from ambient.core.config import ConfigurationManager
    AMBIENT_AVAILABLE = True
except ImportError:
    AMBIENT_AVAILABLE = False


# ============================================================================
# Test Configuration and Setup
# ============================================================================

def pytest_configure(config):
    """Configure pytest with custom markers and settings."""
    config.addinivalue_line("markers", "slow: marks tests as slow")
    config.addinivalue_line("markers", "performance: marks tests as performance tests")
    config.addinivalue_line("markers", "property: marks tests as property-based tests")
    config.addinivalue_line("markers", "integration: marks tests as integration tests")
    config.addinivalue_line("markers", "hardware: marks tests requiring specific hardware")


def pytest_collection_modifyitems(config, items):
    """Automatically mark tests based on their characteristics."""
    for item in items:
        # Mark property-based tests
        if "property" in item.name.lower() or "hypothesis" in str(item.function):
            item.add_marker(pytest.mark.property)
        
        # Mark integration tests
        if "integration" in str(item.fspath) or "pipeline" in item.name.lower():
            item.add_marker(pytest.mark.integration)
        
        # Mark performance tests
        if "performance" in str(item.fspath) or "benchmark" in item.name.lower():
            item.add_marker(pytest.mark.performance)


# ============================================================================
# Basic Fixtures
# ============================================================================

@pytest.fixture(scope="session")
def test_data_dir():
    """Create and provide test data directory."""
    data_dir = Path("data/test_data")
    data_dir.mkdir(parents=True, exist_ok=True)
    return data_dir


@pytest.fixture
def tmp_config_dir(tmp_path):
    """Create temporary configuration directory with test config files."""
    config_dir = tmp_path / "config"
    config_dir.mkdir()
    
    # Create basic test configuration
    test_config = {
        "video_processing": {
            "supported_formats": ["mp4", "avi", "mov", "webm"],
            "default_frame_rate": 30.0,
            "max_video_size_mb": 500,
            "ffmpeg_enabled": True
        },
        "pose_estimation": {
            "estimators": {
                "mediapipe": {
                    "enabled": True,
                    "model_complexity": 1,
                    "min_detection_confidence": 0.5,
                    "min_tracking_confidence": 0.5
                }
            },
            "default_estimator": "mediapipe",
            "confidence_threshold": 0.5
        },
        "gait_analysis": {
            "min_sequence_length": 10,
            "gait_cycle_detection_method": "heel_strike",
            "feature_extraction": {
                "window_size": 30,
                "overlap": 0.5,
                "normalize_features": True
            }
        },
        "classification": {
            "llm": {
                "provider": "openai",
                "model": "gpt-4o-mini",
                "temperature": 0.1,
                "max_tokens": 1000,
                "enabled": True,
                "multimodal_enabled": True
            },
            "normal_abnormal_threshold": 0.7,
            "condition_confidence_threshold": 0.6
        }
    }
    
    with open(config_dir / "alexpose.yaml", 'w') as f:
        yaml.dump(test_config, f)
    
    return config_dir


@pytest.fixture
def tmp_model_file(tmp_path):
    """Create a temporary MediaPipe model file for testing."""
    model_file = tmp_path / "pose_landmarker_full.task"
    # Write dummy data (actual model would be binary)
    model_file.write_bytes(b"dummy mediapipe model data" * 100)
    return model_file


# ============================================================================
# Video and Image Fixtures
# ============================================================================

@pytest.fixture
def sample_image_file(tmp_path):
    """Create a sample test image file."""
    if not CV2_AVAILABLE:
        pytest.skip("OpenCV not available")
    
    image_path = tmp_path / "test_image.jpg"
    # Create a simple test image (200x200, RGB)
    image = np.ones((200, 200, 3), dtype=np.uint8) * 128
    # Add some variation - red square in center
    image[50:150, 50:150] = [255, 0, 0]
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
        frame = np.ones((200, 200, 3), dtype=np.uint8) * (128 + (i * 2))
        # Add moving element to simulate motion
        cv2.circle(frame, (100 + i * 5, 100), 20, (0, 255, 0), -1)
        out.write(frame)
    out.release()
    
    return video_path


@pytest.fixture
def sample_gait_videos(test_data_dir):
    """Provide sample gait videos for testing."""
    videos = {}
    
    if CV2_AVAILABLE:
        # Create normal walking pattern video
        normal_video = test_data_dir / "normal_walking.mp4"
        if not normal_video.exists():
            _create_synthetic_gait_video(normal_video, gait_type="normal")
        videos["normal"] = normal_video
        
        # Create abnormal gait pattern video
        abnormal_video = test_data_dir / "abnormal_gait.mp4"
        if not abnormal_video.exists():
            _create_synthetic_gait_video(abnormal_video, gait_type="abnormal")
        videos["abnormal"] = abnormal_video
    
    return videos


def _create_synthetic_gait_video(video_path: Path, gait_type: str = "normal"):
    """Create synthetic gait video with realistic motion patterns."""
    if not CV2_AVAILABLE:
        return
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(str(video_path), fourcc, 30.0, (640, 480))
    
    # Simulate walking motion over 3 seconds (90 frames)
    for frame_num in range(90):
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        
        # Simulate person walking from left to right
        person_x = int(50 + (frame_num * 6))  # Move across screen
        person_y = 240  # Center vertically
        
        # Add gait-specific motion patterns
        if gait_type == "normal":
            # Normal symmetric gait
            leg_offset = int(20 * np.sin(frame_num * 0.3))
        else:
            # Abnormal asymmetric gait
            leg_offset = int(30 * np.sin(frame_num * 0.2) + 10 * np.sin(frame_num * 0.7))
        
        # Draw simplified person (head, body, legs)
        cv2.circle(frame, (person_x, person_y - 60), 15, (255, 255, 255), -1)  # Head
        cv2.rectangle(frame, (person_x - 10, person_y - 45), (person_x + 10, person_y + 10), (255, 255, 255), -1)  # Body
        
        # Legs with motion
        cv2.line(frame, (person_x, person_y + 10), (person_x - 15 + leg_offset, person_y + 50), (255, 255, 255), 3)  # Left leg
        cv2.line(frame, (person_x, person_y + 10), (person_x + 15 - leg_offset, person_y + 50), (255, 255, 255), 3)  # Right leg
        
        out.write(frame)
    
    out.release()


# ============================================================================
# Data Model Fixtures
# ============================================================================

@pytest.fixture
def sample_frame():
    """Create a sample Frame object for testing."""
    if not AMBIENT_AVAILABLE:
        pytest.skip("Ambient package not available")
    
    return Frame(
        frame_id="test_frame_001",
        sequence_id="test_sequence",
        frame_number=1,
        timestamp=0.033,
        width=640,
        height=480,
        channels=3,
        format='RGB',
        data=np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    )


@pytest.fixture
def sample_frame_sequence():
    """Create a sample FrameSequence for testing."""
    if not AMBIENT_AVAILABLE:
        pytest.skip("Ambient package not available")
    
    frames = []
    for i in range(10):
        frame = Frame(
            frame_id=f"test_frame_{i:03d}",
            sequence_id="test_sequence",
            frame_number=i,
            timestamp=i * 0.033,
            width=640,
            height=480,
            channels=3,
            format='RGB',
            data=np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        )
        frames.append(frame)
    
    return FrameSequence(
        sequence_id="test_sequence",
        frames=frames,
        metadata={"fps": 30.0, "duration": 0.33}
    )


@pytest.fixture
def sample_keypoints():
    """Create sample keypoints for testing."""
    keypoints = []
    # Create 33 MediaPipe-style keypoints
    for i in range(33):
        keypoint = Keypoint(
            x=float(100 + i * 10),
            y=float(200 + i * 5),
            confidence=0.8 + (i * 0.005),
            visibility=0.9,
            name=f"landmark_{i}"
        )
        keypoints.append(keypoint)
    return keypoints


@pytest.fixture
def sample_gait_features():
    """Create sample gait features for testing."""
    if not AMBIENT_AVAILABLE:
        pytest.skip("Ambient package not available")
    
    return GaitFeatures(
        temporal_features={
            "stride_time": 1.2,
            "cadence": 110.0,
            "stance_phase_duration": 0.65,
            "swing_phase_duration": 0.35
        },
        spatial_features={
            "stride_length": 1.4,
            "step_width": 0.15,
            "step_length_left": 0.7,
            "step_length_right": 0.7
        },
        kinematic_features={
            "hip_flexion_max": [25.0, 23.0, 26.0],
            "knee_flexion_max": [60.0, 58.0, 62.0],
            "ankle_dorsiflexion_max": [15.0, 14.0, 16.0]
        },
        symmetry_features={
            "left_right_symmetry": 0.95,
            "temporal_symmetry": 0.92,
            "spatial_symmetry": 0.94
        },
        frequency_features={
            "dominant_frequency": 1.8,
            "harmonic_ratio": 0.85,
            "spectral_centroid": 2.1
        }
    )


@pytest.fixture
def sample_gait_metrics():
    """Create sample gait metrics for testing."""
    if not AMBIENT_AVAILABLE:
        pytest.skip("Ambient package not available")
    
    return GaitMetrics(
        stride_length=1.4,
        stride_time=1.2,
        cadence=110.0,
        step_width=0.15,
        joint_angles={
            "hip_flexion": [20.0, 25.0, 15.0],
            "knee_flexion": [55.0, 60.0, 50.0],
            "ankle_dorsiflexion": [12.0, 15.0, 10.0]
        },
        symmetry_index=0.95,
        balance_metrics={
            "center_of_mass_displacement": 0.05,
            "postural_sway": 0.03,
            "stability_index": 0.92
        },
        temporal_metrics={
            "stance_phase_duration": 0.65,
            "swing_phase_duration": 0.35,
            "double_support_time": 0.12,
            "gait_cycle_time": 1.2
        }
    )


# ============================================================================
# Mock Fixtures
# ============================================================================

@pytest.fixture
def mock_mediapipe_estimator():
    """Create a mock MediaPipe estimator for testing."""
    mock_estimator = Mock()
    mock_estimator.estimate_image_keypoints.return_value = [
        {"x": 100.0, "y": 200.0, "confidence": 0.9} for _ in range(33)
    ]
    mock_estimator.estimate_video_keypoints.return_value = [
        [{"x": 100.0, "y": 200.0, "confidence": 0.9} for _ in range(33)]
        for _ in range(10)
    ]
    mock_estimator.supports_video_batch.return_value = True
    return mock_estimator


@pytest.fixture
def mock_llm_classifier():
    """Create a mock LLM classifier for testing."""
    mock_classifier = Mock()
    mock_classifier.classify_normal_abnormal.return_value = {
        "is_normal": True,
        "confidence": 0.85,
        "explanation": "Normal gait pattern detected"
    }
    mock_classifier.identify_condition.return_value = {
        "condition": "normal",
        "confidence": 0.85,
        "supporting_evidence": ["Symmetric stride pattern", "Normal cadence"]
    }
    return mock_classifier


@pytest.fixture
def mock_gait_analyzer():
    """Create a mock gait analyzer for testing."""
    mock_analyzer = Mock()
    mock_analyzer.analyze_sequence.return_value = GaitMetrics(
        stride_length=1.4,
        stride_time=1.2,
        cadence=110.0,
        step_width=0.15,
        joint_angles={"hip": [20.0], "knee": [60.0], "ankle": [15.0]},
        symmetry_index=0.95,
        balance_metrics=Mock(),
        temporal_metrics=Mock()
    )
    return mock_analyzer


# ============================================================================
# Environment and API Fixtures
# ============================================================================

@pytest.fixture
def clean_environment():
    """Provide clean environment for testing."""
    # Store original environment
    original_env = dict(os.environ)
    
    # Clear test-related environment variables
    test_vars = [
        "OPENAI_API_KEY", "GOOGLE_API_KEY", "GEMINI_API_KEY",
        "ENVIRONMENT", "JWT_SECRET_KEY", "DATABASE_URL"
    ]
    
    for var in test_vars:
        if var in os.environ:
            del os.environ[var]
    
    yield
    
    # Restore original environment
    os.environ.clear()
    os.environ.update(original_env)


@pytest.fixture
def test_api_keys():
    """Provide test API keys for testing."""
    return {
        "OPENAI_API_KEY": "sk-test-key-for-testing-12345678901234567890",
        "GOOGLE_API_KEY": "AItest-google-api-key-for-testing-123456789",
        "GEMINI_API_KEY": "test-gemini-api-key-for-testing-123456789"
    }


@pytest.fixture
def set_test_env_vars(test_api_keys):
    """Set test environment variables."""
    for key, value in test_api_keys.items():
        os.environ[key] = value
    
    yield
    
    # Clean up
    for key in test_api_keys.keys():
        if key in os.environ:
            del os.environ[key]


# ============================================================================
# Performance Testing Fixtures
# ============================================================================

@pytest.fixture
def performance_monitor():
    """Monitor performance metrics during tests."""
    import time
    import psutil
    
    class PerformanceMonitor:
        def __init__(self):
            self.start_time = None
            self.start_memory = None
            self.process = psutil.Process()
        
        def start(self):
            self.start_time = time.time()
            self.start_memory = self.process.memory_info().rss
        
        def stop(self):
            end_time = time.time()
            end_memory = self.process.memory_info().rss
            
            return {
                "execution_time": end_time - self.start_time,
                "memory_delta": end_memory - self.start_memory,
                "peak_memory": self.process.memory_info().rss
            }
    
    return PerformanceMonitor()


# Import fixtures from other modules
from tests.fixtures.real_data_fixtures import (
    gavd_test_subset, gavd_video_urls, sample_gait_videos,
    real_pose_keypoints, real_gait_features, property_test_keypoints,
    property_test_gait_features, property_test_classification_results,
    enhanced_real_data_dir, property_test_data_manager
)


# ============================================================================
# Utility Functions
# ============================================================================

def skip_if_no_opencv():
    """Skip test if OpenCV is not available."""
    return pytest.mark.skipif(not CV2_AVAILABLE, reason="OpenCV not available")


def skip_if_no_mediapipe():
    """Skip test if MediaPipe is not available."""
    return pytest.mark.skipif(not MEDIAPIPE_AVAILABLE, reason="MediaPipe not available")


def skip_if_no_hypothesis():
    """Skip test if Hypothesis is not available."""
    return pytest.mark.skipif(not HYPOTHESIS_AVAILABLE, reason="Hypothesis not available")


def skip_if_no_ambient():
    """Skip test if Ambient package is not available."""
    return pytest.mark.skipif(not AMBIENT_AVAILABLE, reason="Ambient package not available")

