"""
Custom assertion helpers for AlexPose testing.
"""

import numpy as np
from typing import Dict, List, Any, Optional, Union
from pathlib import Path

try:
    from ambient.core.frame import Frame, FrameSequence
    from ambient.core.data_models import GaitFeatures, GaitMetrics, Keypoint
    AMBIENT_AVAILABLE = True
except ImportError:
    AMBIENT_AVAILABLE = False


class AssertionHelpers:
    """Custom assertion helpers for domain-specific testing."""
    
    @staticmethod
    def assert_video_format_supported(file_path: Union[str, Path], supported_formats: List[str]):
        """Assert that video format is supported."""
        path = Path(file_path)
        extension = path.suffix.lower().lstrip('.')
        
        assert extension in supported_formats, (
            f"Video format '{extension}' not supported. "
            f"Supported formats: {supported_formats}"
        )
    
    @staticmethod
    def assert_frame_extraction_count(
        extracted_frames: int,
        video_duration: float,
        frame_rate: float,
        tolerance: float = 0.1
    ):
        """Assert that extracted frame count matches expected count."""
        expected_frames = int(video_duration * frame_rate)
        tolerance_frames = int(expected_frames * tolerance)
        
        assert abs(extracted_frames - expected_frames) <= tolerance_frames, (
            f"Extracted {extracted_frames} frames, expected {expected_frames} "
            f"(±{tolerance_frames}) for {video_duration}s video at {frame_rate}fps"
        )
    
    @staticmethod
    def assert_pose_landmarks_count(landmarks: List[Dict], expected_count: int):
        """Assert correct number of pose landmarks."""
        assert len(landmarks) == expected_count, (
            f"Expected {expected_count} pose landmarks, got {len(landmarks)}"
        )
    
    @staticmethod
    def assert_pose_landmark_format(landmark: Dict[str, Any]):
        """Assert pose landmark has correct format."""
        required_fields = ["x", "y", "confidence"]
        
        for field in required_fields:
            assert field in landmark, f"Landmark missing required field: {field}"
        
        assert isinstance(landmark["x"], (int, float)), "Landmark x must be numeric"
        assert isinstance(landmark["y"], (int, float)), "Landmark y must be numeric"
        assert isinstance(landmark["confidence"], (int, float)), "Landmark confidence must be numeric"
        
        assert 0.0 <= landmark["confidence"] <= 1.0, (
            f"Landmark confidence {landmark['confidence']} not in range [0, 1]"
        )
    
    @staticmethod
    def assert_gait_cycle_detection(
        detected_cycles: List[Dict],
        min_cycles: int = 1,
        max_cycle_duration: float = 2.0
    ):
        """Assert gait cycle detection results."""
        assert len(detected_cycles) >= min_cycles, (
            f"Expected at least {min_cycles} gait cycles, detected {len(detected_cycles)}"
        )
        
        for i, cycle in enumerate(detected_cycles):
            assert "start_frame" in cycle, f"Cycle {i} missing start_frame"
            assert "end_frame" in cycle, f"Cycle {i} missing end_frame"
            assert "duration" in cycle, f"Cycle {i} missing duration"
            
            assert cycle["end_frame"] > cycle["start_frame"], (
                f"Cycle {i} end_frame must be greater than start_frame"
            )
            
            assert cycle["duration"] <= max_cycle_duration, (
                f"Cycle {i} duration {cycle['duration']}s exceeds maximum {max_cycle_duration}s"
            )
    
    @staticmethod
    def assert_classification_completeness(result: Dict[str, Any]):
        """Assert classification result completeness."""
        required_fields = ["is_normal", "confidence"]
        
        for field in required_fields:
            assert field in result, f"Classification result missing: {field}"
        
        assert isinstance(result["is_normal"], bool), "is_normal must be boolean"
        assert isinstance(result["confidence"], (int, float)), "confidence must be numeric"
        assert 0.0 <= result["confidence"] <= 1.0, (
            f"Confidence {result['confidence']} not in range [0, 1]"
        )
        
        # If abnormal, should have condition information
        if not result["is_normal"] and "conditions" in result:
            assert isinstance(result["conditions"], list), "conditions must be a list"
            for condition in result["conditions"]:
                assert "name" in condition, "Condition missing name"
                assert "confidence" in condition, "Condition missing confidence"
    
    @staticmethod
    def assert_temporal_sequence_order(sequence: List[Dict], timestamp_field: str = "timestamp"):
        """Assert temporal sequence is properly ordered."""
        timestamps = [item[timestamp_field] for item in sequence if timestamp_field in item]
        
        if len(timestamps) > 1:
            assert timestamps == sorted(timestamps), (
                "Temporal sequence not properly ordered by timestamp"
            )
    
    @staticmethod
    def assert_bounding_box_valid(bbox: Dict[str, float], image_width: int, image_height: int):
        """Assert bounding box coordinates are valid."""
        required_fields = ["left", "top", "width", "height"]
        
        for field in required_fields:
            assert field in bbox, f"Bounding box missing: {field}"
            assert isinstance(bbox[field], (int, float)), f"Bounding box {field} must be numeric"
        
        # Check bounds
        assert 0 <= bbox["left"] < image_width, f"Bounding box left {bbox['left']} out of bounds"
        assert 0 <= bbox["top"] < image_height, f"Bounding box top {bbox['top']} out of bounds"
        assert bbox["width"] > 0, f"Bounding box width must be positive"
        assert bbox["height"] > 0, f"Bounding box height must be positive"
        
        # Check that box fits within image
        assert bbox["left"] + bbox["width"] <= image_width, "Bounding box extends beyond image width"
        assert bbox["top"] + bbox["height"] <= image_height, "Bounding box extends beyond image height"
    
    @staticmethod
    def assert_symmetry_analysis(symmetry_metrics: Dict[str, float]):
        """Assert symmetry analysis results."""
        required_metrics = ["left_right_symmetry"]
        
        for metric in required_metrics:
            if metric in symmetry_metrics:
                value = symmetry_metrics[metric]
                assert isinstance(value, (int, float)), f"Symmetry metric {metric} must be numeric"
                assert 0.0 <= value <= 1.0, f"Symmetry metric {metric} must be in range [0, 1]"
    
    @staticmethod
    def assert_feature_extraction_completeness(features: Dict[str, Any]):
        """Assert feature extraction completeness."""
        required_categories = ["temporal_features", "spatial_features"]
        
        for category in required_categories:
            assert category in features, f"Missing feature category: {category}"
            assert isinstance(features[category], dict), f"{category} must be a dictionary"
            assert len(features[category]) > 0, f"{category} cannot be empty"
    
    @staticmethod
    def assert_performance_bounds(
        execution_time: float,
        memory_usage_mb: float,
        max_time: Optional[float] = None,
        max_memory_mb: Optional[float] = None
    ):
        """Assert performance metrics are within acceptable bounds."""
        if max_time is not None:
            assert execution_time <= max_time, (
                f"Execution time {execution_time:.2f}s exceeds maximum {max_time}s"
            )
        
        if max_memory_mb is not None:
            assert memory_usage_mb <= max_memory_mb, (
                f"Memory usage {memory_usage_mb:.2f}MB exceeds maximum {max_memory_mb}MB"
            )
    
    @staticmethod
    def assert_configuration_valid(config: Dict[str, Any]):
        """Assert configuration structure and values are valid."""
        required_sections = ["video_processing", "pose_estimation", "gait_analysis"]
        
        for section in required_sections:
            assert section in config, f"Configuration missing section: {section}"
            assert isinstance(config[section], dict), f"Configuration section {section} must be dict"
        
        # Validate video processing config
        video_config = config["video_processing"]
        if "supported_formats" in video_config:
            assert isinstance(video_config["supported_formats"], list), "supported_formats must be list"
            assert len(video_config["supported_formats"]) > 0, "supported_formats cannot be empty"
        
        if "default_frame_rate" in video_config:
            assert video_config["default_frame_rate"] > 0, "default_frame_rate must be positive"
        
        # Validate pose estimation config
        pose_config = config["pose_estimation"]
        if "confidence_threshold" in pose_config:
            threshold = pose_config["confidence_threshold"]
            assert 0.0 <= threshold <= 1.0, f"confidence_threshold {threshold} not in [0, 1]"
    
    @staticmethod
    def assert_api_response_format(response: Dict[str, Any], expected_status: str):
        """Assert API response has correct format."""
        assert "status" in response, "API response missing status"
        assert response["status"] == expected_status, (
            f"Expected status '{expected_status}', got '{response['status']}'"
        )
        
        if response["status"] == "error":
            assert "error_message" in response, "Error response missing error_message"
            assert "error_code" in response, "Error response missing error_code"
        elif response["status"] == "success":
            assert "data" in response, "Success response missing data"
    
    @staticmethod
    def assert_data_persistence_integrity(
        original_data: Dict[str, Any],
        retrieved_data: Dict[str, Any],
        ignore_fields: Optional[List[str]] = None
    ):
        """Assert data persistence maintains integrity."""
        ignore_fields = ignore_fields or ["created_at", "updated_at", "id"]
        
        for key, value in original_data.items():
            if key in ignore_fields:
                continue
            
            assert key in retrieved_data, f"Retrieved data missing field: {key}"
            
            if isinstance(value, (int, float)):
                assert abs(retrieved_data[key] - value) < 1e-6, (
                    f"Numeric field {key} changed: {value} -> {retrieved_data[key]}"
                )
            else:
                assert retrieved_data[key] == value, (
                    f"Field {key} changed: {value} -> {retrieved_data[key]}"
                )


# Convenience functions for common assertions
def assert_mediapipe_landmarks(landmarks: List[Dict]):
    """Assert MediaPipe landmarks format (33 landmarks)."""
    AssertionHelpers.assert_pose_landmarks_count(landmarks, 33)
    for landmark in landmarks:
        AssertionHelpers.assert_pose_landmark_format(landmark)


def assert_openpose_keypoints(keypoints: List[Dict]):
    """Assert OpenPose keypoints format (25 keypoints)."""
    AssertionHelpers.assert_pose_landmarks_count(keypoints, 25)
    for keypoint in keypoints:
        AssertionHelpers.assert_pose_landmark_format(keypoint)


def assert_normal_classification(result: Dict[str, Any], min_confidence: float = 0.7):
    """Assert normal gait classification result."""
    AssertionHelpers.assert_classification_completeness(result)
    assert result["is_normal"] is True, "Expected normal classification"
    assert result["confidence"] >= min_confidence, (
        f"Normal classification confidence {result['confidence']} below minimum {min_confidence}"
    )


def assert_abnormal_classification(result: Dict[str, Any], min_confidence: float = 0.6):
    """Assert abnormal gait classification result."""
    AssertionHelpers.assert_classification_completeness(result)
    assert result["is_normal"] is False, "Expected abnormal classification"
    assert result["confidence"] >= min_confidence, (
        f"Abnormal classification confidence {result['confidence']} below minimum {min_confidence}"
    )


def assert_frame_sequence_valid(sequence):
    """Assert FrameSequence is valid."""
    if not AMBIENT_AVAILABLE:
        return
    
    assert isinstance(sequence, FrameSequence), "Expected FrameSequence object"
    assert len(sequence.frames) > 0, "FrameSequence cannot be empty"
    
    # Check frame ordering if frame_number is available in metadata
    frame_numbers = []
    for f in sequence.frames:
        if hasattr(f, 'metadata') and f.metadata and 'frame_number' in f.metadata:
            frame_numbers.append(f.metadata['frame_number'])
    
    if frame_numbers and len(frame_numbers) == len(sequence.frames):
        assert frame_numbers == sorted(frame_numbers), "Frames not in correct order"


def assert_gait_metrics_realistic(metrics):
    """Assert gait metrics are within realistic ranges."""
    if not AMBIENT_AVAILABLE:
        return
    
    assert isinstance(metrics, GaitMetrics), "Expected GaitMetrics object"
    
    # Check realistic ranges
    assert 0.5 <= metrics.stride_length <= 3.0, f"Unrealistic stride_length: {metrics.stride_length}"
    assert 0.8 <= metrics.stride_time <= 2.0, f"Unrealistic stride_time: {metrics.stride_time}"
    assert 60 <= metrics.cadence <= 180, f"Unrealistic cadence: {metrics.cadence}"
    assert 0.05 <= metrics.step_width <= 0.4, f"Unrealistic step_width: {metrics.step_width}"
    assert 0.0 <= metrics.symmetry_index <= 1.0, f"Invalid symmetry_index: {metrics.symmetry_index}"


# Additional assertion functions for gait analysis and classification
def assert_gait_features(features: Dict[str, Any]):
    """Assert gait features structure and validity."""
    required_categories = ["temporal_features", "spatial_features", "symmetry_features"]
    
    for category in required_categories:
        assert category in features, f"Missing feature category: {category}"
        assert isinstance(features[category], dict), f"{category} must be a dictionary"
        assert len(features[category]) > 0, f"{category} cannot be empty"
    
    # Validate temporal features
    temporal = features["temporal_features"]
    if "cadence" in temporal:
        assert 50 <= temporal["cadence"] <= 200, f"Unrealistic cadence: {temporal['cadence']}"
    if "stride_time" in temporal:
        assert 0.5 <= temporal["stride_time"] <= 3.0, f"Unrealistic stride_time: {temporal['stride_time']}"
    
    # Validate spatial features
    spatial = features["spatial_features"]
    if "stride_length" in spatial:
        assert 0.5 <= spatial["stride_length"] <= 3.0, f"Unrealistic stride_length: {spatial['stride_length']}"
    if "step_width" in spatial:
        assert 0.02 <= spatial["step_width"] <= 0.5, f"Unrealistic step_width: {spatial['step_width']}"
    
    # Validate symmetry features
    symmetry = features["symmetry_features"]
    for key, value in symmetry.items():
        if "symmetry" in key.lower():
            assert 0.0 <= value <= 1.0, f"Symmetry metric {key} = {value} not in [0, 1]"


def assert_temporal_features(features: Dict[str, Any]):
    """Assert temporal features validity."""
    required_features = ["stride_time", "cadence"]
    
    for feature in required_features:
        if feature in features:
            value = features[feature]
            assert isinstance(value, (int, float)), f"Temporal feature {feature} must be numeric"
            
            if feature == "stride_time":
                assert 0.5 <= value <= 3.0, f"Stride time {value} outside realistic range [0.5, 3.0]"
            elif feature == "cadence":
                assert 50 <= value <= 200, f"Cadence {value} outside realistic range [50, 200]"


def assert_spatial_features(features: Dict[str, Any]):
    """Assert spatial features validity."""
    required_features = ["stride_length", "step_width"]
    
    for feature in required_features:
        if feature in features:
            value = features[feature]
            assert isinstance(value, (int, float)), f"Spatial feature {feature} must be numeric"
            assert value > 0, f"Spatial feature {feature} must be positive"
            
            if feature == "stride_length":
                assert 0.5 <= value <= 3.0, f"Stride length {value} outside realistic range [0.5, 3.0]"
            elif feature == "step_width":
                assert 0.02 <= value <= 0.5, f"Step width {value} outside realistic range [0.02, 0.5]"


def assert_symmetry_features(features: Dict[str, Any]):
    """Assert symmetry features validity."""
    for key, value in features.items():
        if "symmetry" in key.lower():
            assert isinstance(value, (int, float)), f"Symmetry feature {key} must be numeric"
            assert 0.0 <= value <= 1.0, f"Symmetry feature {key} = {value} not in [0, 1]"


def assert_classification_result(result: Dict[str, Any]):
    """Assert classification result structure and validity."""
    required_fields = ["classification", "confidence"]
    
    for field in required_fields:
        assert field in result, f"Classification result missing: {field}"
    
    # Validate classification
    classification = result["classification"]
    assert classification in ["normal", "abnormal"], (
        f"Invalid classification: {classification}. Must be 'normal' or 'abnormal'"
    )
    
    # Validate confidence
    confidence = result["confidence"]
    assert isinstance(confidence, (int, float)), "Confidence must be numeric"
    assert 0.0 <= confidence <= 1.0, f"Confidence {confidence} not in range [0, 1]"
    
    # Check for reasoning if present
    if "reasoning" in result:
        assert isinstance(result["reasoning"], str), "Reasoning must be string"
        assert len(result["reasoning"]) > 0, "Reasoning cannot be empty"


def assert_confidence_bounds(confidence: float, min_val: float = 0.0, max_val: float = 1.0):
    """Assert confidence score is within valid bounds."""
    assert isinstance(confidence, (int, float)), "Confidence must be numeric"
    assert min_val <= confidence <= max_val, (
        f"Confidence {confidence} not in range [{min_val}, {max_val}]"
    )