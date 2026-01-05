"""
Utility functions and helpers for property-based testing.

This module provides common utilities, assertions, and helper functions
specifically designed for property-based testing in the AlexPose system.
"""

import numpy as np
from typing import Any, Dict, List, Optional, Union, Callable, Tuple
from pathlib import Path
import json
import tempfile
from hypothesis import strategies as st, assume
from hypothesis.strategies import composite

try:
    from ambient.core.frame import Frame, FrameSequence
    from ambient.core.data_models import GaitFeatures, GaitMetrics, Keypoint
    AMBIENT_AVAILABLE = True
except ImportError:
    AMBIENT_AVAILABLE = False


# ============================================================================
# Property Test Assertions
# ============================================================================

def assert_keypoints_valid(
    keypoints: List[Dict[str, Any]], 
    expected_count: Optional[int] = None,
    confidence_threshold: float = 0.0
) -> None:
    """Assert that keypoints are valid with proper structure and values."""
    assert isinstance(keypoints, list), "Keypoints must be a list"
    
    if expected_count is not None:
        assert len(keypoints) == expected_count, \
            f"Expected {expected_count} keypoints, got {len(keypoints)}"
    
    for i, kp in enumerate(keypoints):
        assert isinstance(kp, dict), f"Keypoint {i} must be a dictionary"
        
        # Check required fields
        required_fields = ["x", "y", "confidence"]
        for field in required_fields:
            assert field in kp, f"Keypoint {i} missing required field '{field}'"
        
        # Check data types
        assert isinstance(kp["x"], (int, float)), \
            f"Keypoint {i} 'x' must be numeric, got {type(kp['x'])}"
        assert isinstance(kp["y"], (int, float)), \
            f"Keypoint {i} 'y' must be numeric, got {type(kp['y'])}"
        assert isinstance(kp["confidence"], (int, float)), \
            f"Keypoint {i} 'confidence' must be numeric, got {type(kp['confidence'])}"
        
        # Check value ranges
        assert 0.0 <= kp["confidence"] <= 1.0, \
            f"Keypoint {i} confidence {kp['confidence']} not in range [0.0, 1.0]"
        assert kp["confidence"] >= confidence_threshold, \
            f"Keypoint {i} confidence {kp['confidence']} below threshold {confidence_threshold}"
        
        # Check for NaN/infinity
        for coord in ["x", "y", "confidence"]:
            value = kp[coord]
            assert not (isinstance(value, float) and (np.isnan(value) or np.isinf(value))), \
                f"Keypoint {i} '{coord}' contains NaN or infinity: {value}"


def assert_gait_features_valid(features: Dict[str, Any]) -> None:
    """Assert that gait features are valid with proper structure and physiological ranges."""
    assert isinstance(features, dict), "Features must be a dictionary"
    
    # Check required categories
    required_categories = ["temporal_features", "spatial_features"]
    for category in required_categories:
        assert category in features, f"Missing required category: {category}"
        assert isinstance(features[category], dict), f"{category} must be a dictionary"
    
    # Validate temporal features
    temporal = features["temporal_features"]
    if "stride_time" in temporal:
        stride_time = temporal["stride_time"]
        assert isinstance(stride_time, (int, float)), "stride_time must be numeric"
        assert 0.5 <= stride_time <= 3.0, \
            f"stride_time {stride_time} outside physiological range [0.5, 3.0]"
    
    if "cadence" in temporal:
        cadence = temporal["cadence"]
        assert isinstance(cadence, (int, float)), "cadence must be numeric"
        assert 50 <= cadence <= 200, \
            f"cadence {cadence} outside physiological range [50, 200]"
    
    if "stance_phase_duration" in temporal:
        stance = temporal["stance_phase_duration"]
        assert isinstance(stance, (int, float)), "stance_phase_duration must be numeric"
        assert 0.4 <= stance <= 0.8, \
            f"stance_phase_duration {stance} outside physiological range [0.4, 0.8]"
    
    # Validate spatial features
    spatial = features["spatial_features"]
    if "stride_length" in spatial:
        stride_length = spatial["stride_length"]
        assert isinstance(stride_length, (int, float)), "stride_length must be numeric"
        assert 0.5 <= stride_length <= 3.0, \
            f"stride_length {stride_length} outside physiological range [0.5, 3.0]"
    
    if "step_width" in spatial:
        step_width = spatial["step_width"]
        assert isinstance(step_width, (int, float)), "step_width must be numeric"
        assert 0.05 <= step_width <= 0.5, \
            f"step_width {step_width} outside physiological range [0.05, 0.5]"
    
    # Validate symmetry features if present
    if "symmetry_features" in features:
        symmetry = features["symmetry_features"]
        assert isinstance(symmetry, dict), "symmetry_features must be a dictionary"
        
        for sym_key, sym_value in symmetry.items():
            if isinstance(sym_value, (int, float)):
                assert 0.0 <= sym_value <= 1.0, \
                    f"symmetry feature {sym_key} value {sym_value} not in range [0.0, 1.0]"


def assert_classification_result_valid(result: Dict[str, Any]) -> None:
    """Assert that classification result is valid with proper structure."""
    assert isinstance(result, dict), "Classification result must be a dictionary"
    
    # Check required fields
    required_fields = ["is_normal", "confidence"]
    for field in required_fields:
        assert field in result, f"Missing required field: {field}"
    
    # Validate field types and values
    assert isinstance(result["is_normal"], bool), \
        f"is_normal must be boolean, got {type(result['is_normal'])}"
    
    confidence = result["confidence"]
    assert isinstance(confidence, (int, float)), \
        f"confidence must be numeric, got {type(confidence)}"
    assert 0.0 <= confidence <= 1.0, \
        f"confidence {confidence} not in range [0.0, 1.0]"
    
    # If abnormal classification, check for condition information
    if not result["is_normal"] and "conditions" in result:
        conditions = result["conditions"]
        assert isinstance(conditions, list), "conditions must be a list"
        
        for i, condition in enumerate(conditions):
            assert isinstance(condition, dict), f"condition {i} must be a dictionary"
            if "confidence" in condition:
                cond_conf = condition["confidence"]
                assert isinstance(cond_conf, (int, float)), \
                    f"condition {i} confidence must be numeric"
                assert 0.0 <= cond_conf <= 1.0, \
                    f"condition {i} confidence {cond_conf} not in range [0.0, 1.0]"


def assert_video_metadata_valid(metadata: Dict[str, Any]) -> None:
    """Assert that video metadata is valid."""
    assert isinstance(metadata, dict), "Video metadata must be a dictionary"
    
    # Check common metadata fields
    if "duration" in metadata:
        duration = metadata["duration"]
        assert isinstance(duration, (int, float)), "duration must be numeric"
        assert duration > 0, f"duration {duration} must be positive"
    
    if "frame_rate" in metadata:
        fps = metadata["frame_rate"]
        assert isinstance(fps, (int, float)), "frame_rate must be numeric"
        assert 1.0 <= fps <= 120.0, f"frame_rate {fps} outside reasonable range [1.0, 120.0]"
    
    if "resolution" in metadata:
        resolution = metadata["resolution"]
        assert isinstance(resolution, (list, tuple)), "resolution must be list or tuple"
        assert len(resolution) == 2, f"resolution must have 2 elements, got {len(resolution)}"
        width, height = resolution
        assert isinstance(width, int) and isinstance(height, int), \
            "resolution dimensions must be integers"
        assert width > 0 and height > 0, "resolution dimensions must be positive"


# ============================================================================
# Property Test Utilities
# ============================================================================

def filter_valid_keypoints(keypoints: List[Dict[str, Any]], min_confidence: float = 0.5) -> List[Dict[str, Any]]:
    """Filter keypoints to only include those above confidence threshold."""
    return [kp for kp in keypoints if kp.get("confidence", 0.0) >= min_confidence]


def calculate_keypoint_distances(kp1: Dict[str, Any], kp2: Dict[str, Any]) -> float:
    """Calculate Euclidean distance between two keypoints."""
    dx = kp1["x"] - kp2["x"]
    dy = kp1["y"] - kp2["y"]
    return np.sqrt(dx * dx + dy * dy)


def validate_gait_cycle_consistency(temporal_features: Dict[str, Any]) -> bool:
    """Validate that temporal gait features are internally consistent."""
    if "stance_phase_duration" in temporal_features and "swing_phase_duration" in temporal_features:
        stance = temporal_features["stance_phase_duration"]
        swing = temporal_features["swing_phase_duration"]
        total = stance + swing
        
        # Total should be close to 1.0 (100% of gait cycle)
        if not (0.9 <= total <= 1.1):
            return False
    
    if "stride_time" in temporal_features and "cadence" in temporal_features:
        stride_time = temporal_features["stride_time"]
        cadence = temporal_features["cadence"]
        
        # Cadence (steps/min) should be approximately 60 / stride_time
        expected_cadence = 60.0 / stride_time
        if not (0.8 * expected_cadence <= cadence <= 1.2 * expected_cadence):
            return False
    
    return True


def validate_spatial_symmetry(spatial_features: Dict[str, Any], tolerance: float = 0.2) -> bool:
    """Validate spatial symmetry in gait features."""
    if "step_length_left" in spatial_features and "step_length_right" in spatial_features:
        left = spatial_features["step_length_left"]
        right = spatial_features["step_length_right"]
        
        if left > 0 and right > 0:
            ratio = min(left, right) / max(left, right)
            if ratio < (1.0 - tolerance):
                return False
    
    return True


def create_minimal_reproduction_case(
    failing_input: Any,
    test_function: Callable[[Any], bool],
    max_attempts: int = 100
) -> Optional[Any]:
    """Create a minimal reproduction case for a failing property test."""
    if not isinstance(failing_input, dict):
        return failing_input
    
    # Try to minimize dictionary by removing keys
    minimal_input = failing_input.copy()
    
    for key in list(minimal_input.keys()):
        temp_input = minimal_input.copy()
        del temp_input[key]
        
        try:
            # If test still fails without this key, remove it
            if not test_function(temp_input):
                minimal_input = temp_input
        except Exception:
            # If removing key causes different error, keep it
            continue
    
    return minimal_input


# ============================================================================
# Data Generation Helpers
# ============================================================================

@composite
def realistic_pose_keypoints(draw, num_keypoints: int = 33, confidence_range: Tuple[float, float] = (0.5, 1.0)):
    """Generate realistic pose keypoints with spatial relationships."""
    keypoints = []
    
    # Generate a central body position
    center_x = draw(st.floats(min_value=200, max_value=800))
    center_y = draw(st.floats(min_value=200, max_value=600))
    
    for i in range(num_keypoints):
        # Generate keypoints around the center with realistic spread
        offset_x = draw(st.floats(min_value=-100, max_value=100))
        offset_y = draw(st.floats(min_value=-150, max_value=150))
        
        keypoint = {
            "x": center_x + offset_x,
            "y": center_y + offset_y,
            "confidence": draw(st.floats(min_value=confidence_range[0], max_value=confidence_range[1]))
        }
        keypoints.append(keypoint)
    
    return keypoints


@composite
def physiologically_consistent_gait_features(draw, gait_type: str = "normal"):
    """Generate physiologically consistent gait features."""
    if gait_type == "normal":
        # Normal gait parameters
        stride_time = draw(st.floats(min_value=1.0, max_value=1.4))
        cadence = 60.0 / stride_time + draw(st.floats(min_value=-5, max_value=5))
        
        temporal_features = {
            "stride_time": stride_time,
            "cadence": max(50, min(200, cadence)),  # Clamp to valid range
            "stance_phase_duration": draw(st.floats(min_value=0.6, max_value=0.7)),
            "swing_phase_duration": draw(st.floats(min_value=0.3, max_value=0.4))
        }
        
        # Ensure phases sum to approximately 1.0
        total_phase = temporal_features["stance_phase_duration"] + temporal_features["swing_phase_duration"]
        if total_phase > 0:
            temporal_features["stance_phase_duration"] /= total_phase
            temporal_features["swing_phase_duration"] /= total_phase
        
        stride_length = draw(st.floats(min_value=1.2, max_value=1.6))
        spatial_features = {
            "stride_length": stride_length,
            "step_width": draw(st.floats(min_value=0.1, max_value=0.2)),
            "step_length_left": stride_length / 2 + draw(st.floats(min_value=-0.05, max_value=0.05)),
            "step_length_right": stride_length / 2 + draw(st.floats(min_value=-0.05, max_value=0.05))
        }
        
        symmetry_features = {
            "left_right_symmetry": draw(st.floats(min_value=0.9, max_value=1.0)),
            "temporal_symmetry": draw(st.floats(min_value=0.85, max_value=0.98)),
            "spatial_symmetry": draw(st.floats(min_value=0.88, max_value=0.98))
        }
        
    else:  # abnormal gait
        stride_time = draw(st.floats(min_value=1.4, max_value=2.0))
        cadence = 60.0 / stride_time + draw(st.floats(min_value=-10, max_value=10))
        
        temporal_features = {
            "stride_time": stride_time,
            "cadence": max(50, min(200, cadence)),
            "stance_phase_duration": draw(st.floats(min_value=0.7, max_value=0.8)),
            "swing_phase_duration": draw(st.floats(min_value=0.2, max_value=0.3))
        }
        
        stride_length = draw(st.floats(min_value=0.8, max_value=1.2))
        spatial_features = {
            "stride_length": stride_length,
            "step_width": draw(st.floats(min_value=0.2, max_value=0.4)),
            "step_length_left": stride_length / 2 + draw(st.floats(min_value=-0.2, max_value=0.2)),
            "step_length_right": stride_length / 2 + draw(st.floats(min_value=-0.2, max_value=0.2))
        }
        
        symmetry_features = {
            "left_right_symmetry": draw(st.floats(min_value=0.5, max_value=0.85)),
            "temporal_symmetry": draw(st.floats(min_value=0.4, max_value=0.8)),
            "spatial_symmetry": draw(st.floats(min_value=0.45, max_value=0.82))
        }
    
    return {
        "temporal_features": temporal_features,
        "spatial_features": spatial_features,
        "symmetry_features": symmetry_features
    }


def assume_valid_keypoints(keypoints: List[Dict[str, Any]], min_confidence: float = 0.1) -> None:
    """Hypothesis assume() helper for valid keypoints."""
    assume(isinstance(keypoints, list))
    assume(len(keypoints) > 0)
    
    for kp in keypoints:
        assume(isinstance(kp, dict))
        assume("x" in kp and "y" in kp and "confidence" in kp)
        assume(isinstance(kp["x"], (int, float)))
        assume(isinstance(kp["y"], (int, float)))
        assume(isinstance(kp["confidence"], (int, float)))
        assume(0.0 <= kp["confidence"] <= 1.0)
        assume(kp["confidence"] >= min_confidence)
        assume(not (isinstance(kp["x"], float) and (np.isnan(kp["x"]) or np.isinf(kp["x"]))))
        assume(not (isinstance(kp["y"], float) and (np.isnan(kp["y"]) or np.isinf(kp["y"]))))


def assume_valid_gait_features(features: Dict[str, Any]) -> None:
    """Hypothesis assume() helper for valid gait features."""
    assume(isinstance(features, dict))
    assume("temporal_features" in features)
    assume("spatial_features" in features)
    
    temporal = features["temporal_features"]
    assume(isinstance(temporal, dict))
    
    spatial = features["spatial_features"]
    assume(isinstance(spatial, dict))
    
    # Assume physiological ranges
    if "stride_time" in temporal:
        assume(0.5 <= temporal["stride_time"] <= 3.0)
    if "cadence" in temporal:
        assume(50 <= temporal["cadence"] <= 200)
    if "stride_length" in spatial:
        assume(0.5 <= spatial["stride_length"] <= 3.0)
    if "step_width" in spatial:
        assume(0.05 <= spatial["step_width"] <= 0.5)


# ============================================================================
# Test Data Management
# ============================================================================

class PropertyTestDataManager:
    """Manager for property test data and fixtures."""
    
    def __init__(self, base_dir: Optional[Path] = None):
        self.base_dir = base_dir or Path(tempfile.gettempdir()) / "property_test_data"
        self.base_dir.mkdir(exist_ok=True)
        self.created_files: List[Path] = []
    
    def create_test_keypoints_file(self, filename: str, keypoints: List[Dict[str, Any]]) -> Path:
        """Create a JSON file with test keypoints."""
        file_path = self.base_dir / filename
        with open(file_path, 'w') as f:
            json.dump(keypoints, f, indent=2)
        self.created_files.append(file_path)
        return file_path
    
    def create_test_gait_features_file(self, filename: str, features: Dict[str, Any]) -> Path:
        """Create a JSON file with test gait features."""
        file_path = self.base_dir / filename
        with open(file_path, 'w') as f:
            json.dump(features, f, indent=2)
        self.created_files.append(file_path)
        return file_path
    
    def cleanup(self):
        """Clean up all created test files."""
        for file_path in self.created_files:
            if file_path.exists():
                file_path.unlink()
        self.created_files.clear()
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.cleanup()


# ============================================================================
# Performance Helpers
# ============================================================================

def measure_property_test_performance(test_function: Callable, *args, **kwargs) -> Dict[str, float]:
    """Measure performance metrics for a property test."""
    import time
    import psutil
    
    process = psutil.Process()
    start_time = time.time()
    start_memory = process.memory_info().rss
    
    try:
        result = test_function(*args, **kwargs)
        success = True
    except Exception:
        result = None
        success = False
    
    end_time = time.time()
    end_memory = process.memory_info().rss
    
    return {
        "execution_time": end_time - start_time,
        "memory_delta": end_memory - start_memory,
        "memory_delta_mb": (end_memory - start_memory) / (1024 * 1024),
        "success": success,
        "result": result
    }


# ============================================================================
# Additional Helper Classes for Gait Analysis and Classification
# ============================================================================

class GaitFeatureGenerator:
    """Helper class for generating realistic gait features."""
    
    @staticmethod
    def generate_normal_features() -> Dict[str, Any]:
        """Generate features typical of normal gait."""
        return {
            "temporal_features": {
                "stride_time": np.random.uniform(1.0, 1.4),
                "cadence": np.random.uniform(100, 130),
                "stance_phase_duration": np.random.uniform(0.6, 0.7),
                "swing_phase_duration": np.random.uniform(0.3, 0.4)
            },
            "spatial_features": {
                "stride_length": np.random.uniform(1.2, 1.6),
                "step_width": np.random.uniform(0.1, 0.2),
                "step_length": np.random.uniform(0.6, 0.8)
            },
            "symmetry_features": {
                "left_right_symmetry": np.random.uniform(0.9, 1.0),
                "temporal_symmetry": np.random.uniform(0.85, 0.98)
            }
        }
    
    @staticmethod
    def generate_abnormal_features() -> Dict[str, Any]:
        """Generate features typical of abnormal gait."""
        return {
            "temporal_features": {
                "stride_time": np.random.uniform(1.4, 2.0),
                "cadence": np.random.uniform(60, 100),
                "stance_phase_duration": np.random.uniform(0.7, 0.8),
                "swing_phase_duration": np.random.uniform(0.2, 0.3)
            },
            "spatial_features": {
                "stride_length": np.random.uniform(0.8, 1.2),
                "step_width": np.random.uniform(0.2, 0.4),
                "step_length": np.random.uniform(0.4, 0.6)
            },
            "symmetry_features": {
                "left_right_symmetry": np.random.uniform(0.5, 0.85),
                "temporal_symmetry": np.random.uniform(0.4, 0.8)
            }
        }


class PhysiologicalBounds:
    """Helper class for physiological parameter bounds."""
    
    TEMPORAL_BOUNDS = {
        "stride_time": (0.5, 3.0),
        "cadence": (50, 200),
        "stance_phase_duration": (0.4, 0.8),
        "swing_phase_duration": (0.2, 0.6),
        "double_support_time": (0.05, 0.4)
    }
    
    SPATIAL_BOUNDS = {
        "stride_length": (0.5, 3.0),
        "step_width": (0.05, 0.5),
        "step_length": (0.3, 1.5),
        "foot_angle": (-45, 45)
    }
    
    SYMMETRY_BOUNDS = {
        "left_right_symmetry": (0.0, 1.0),
        "temporal_symmetry": (0.0, 1.0),
        "spatial_symmetry": (0.0, 1.0)
    }
    
    @classmethod
    def is_within_bounds(cls, feature_name: str, value: float, category: str = "temporal") -> bool:
        """Check if a feature value is within physiological bounds."""
        bounds_dict = getattr(cls, f"{category.upper()}_BOUNDS", {})
        if feature_name in bounds_dict:
            min_val, max_val = bounds_dict[feature_name]
            return min_val <= value <= max_val
        return True  # Unknown features are assumed valid
    
    @classmethod
    def validate_features(cls, features: Dict[str, Any]) -> List[str]:
        """Validate all features and return list of violations."""
        violations = []
        
        for category, category_features in features.items():
            if not isinstance(category_features, dict):
                continue
                
            category_name = category.replace("_features", "")
            for feature_name, value in category_features.items():
                if isinstance(value, (int, float)):
                    if not cls.is_within_bounds(feature_name, value, category_name):
                        bounds_dict = getattr(cls, f"{category_name.upper()}_BOUNDS", {})
                        bounds = bounds_dict.get(feature_name, (None, None))
                        violations.append(
                            f"{category}.{feature_name} = {value} outside bounds {bounds}"
                        )
        
        return violations


class GaitCycleValidator:
    """Helper class for validating gait cycle detection."""
    
    @staticmethod
    def validate_cycle_boundaries(boundaries: List[int], total_frames: int) -> List[str]:
        """Validate gait cycle boundaries."""
        errors = []
        
        if not boundaries:
            errors.append("No cycle boundaries detected")
            return errors
        
        # Check boundary order
        if boundaries != sorted(boundaries):
            errors.append("Cycle boundaries not in chronological order")
        
        # Check boundary range
        for i, boundary in enumerate(boundaries):
            if boundary < 0 or boundary >= total_frames:
                errors.append(f"Boundary {i} ({boundary}) outside frame range [0, {total_frames})")
        
        # Check minimum distance between boundaries
        if len(boundaries) > 1:
            min_distance = total_frames // 10  # Minimum 10% of video length
            for i in range(len(boundaries) - 1):
                distance = boundaries[i + 1] - boundaries[i]
                if distance < min_distance:
                    errors.append(
                        f"Boundaries {i} and {i+1} too close: {distance} frames < {min_distance}"
                    )
        
        return errors
    
    @staticmethod
    def estimate_cycle_quality(boundaries: List[int], frame_rate: float) -> float:
        """Estimate quality of cycle detection (0.0 to 1.0)."""
        if len(boundaries) < 2:
            return 0.0
        
        # Calculate cycle durations
        durations = []
        for i in range(len(boundaries) - 1):
            duration = (boundaries[i + 1] - boundaries[i]) / frame_rate
            durations.append(duration)
        
        if not durations:
            return 0.0
        
        # Quality based on consistency of cycle durations
        mean_duration = np.mean(durations)
        std_duration = np.std(durations)
        
        # Good cycles should have consistent durations (low std relative to mean)
        if mean_duration > 0:
            coefficient_of_variation = std_duration / mean_duration
            quality = max(0.0, 1.0 - coefficient_of_variation)
        else:
            quality = 0.0
        
        return quality


class ClassificationValidator:
    """Helper class for validating classification results."""
    
    VALID_CLASSIFICATIONS = {"normal", "abnormal"}
    
    @staticmethod
    def validate_result(result: Dict[str, Any]) -> List[str]:
        """Validate classification result structure and values."""
        errors = []
        
        # Check required fields
        if "classification" not in result:
            errors.append("Missing 'classification' field")
        elif result["classification"] not in ClassificationValidator.VALID_CLASSIFICATIONS:
            errors.append(
                f"Invalid classification '{result['classification']}'. "
                f"Must be one of: {ClassificationValidator.VALID_CLASSIFICATIONS}"
            )
        
        if "confidence" not in result:
            errors.append("Missing 'confidence' field")
        elif not isinstance(result["confidence"], (int, float)):
            errors.append("Confidence must be numeric")
        elif not (0.0 <= result["confidence"] <= 1.0):
            errors.append(f"Confidence {result['confidence']} not in range [0.0, 1.0]")
        
        # Check optional fields
        if "reasoning" in result:
            if not isinstance(result["reasoning"], str):
                errors.append("Reasoning must be string")
            elif len(result["reasoning"]) == 0:
                errors.append("Reasoning cannot be empty")
        
        return errors
    
    @staticmethod
    def assess_explanation_quality(explanation: str) -> Dict[str, Any]:
        """Assess quality of classification explanation."""
        if not isinstance(explanation, str):
            return {"quality_score": 0.0, "issues": ["Explanation is not a string"]}
        
        issues = []
        quality_score = 1.0
        
        # Check length
        if len(explanation) < 10:
            issues.append("Explanation too short")
            quality_score -= 0.3
        elif len(explanation) > 1000:
            issues.append("Explanation too long")
            quality_score -= 0.1
        
        # Check for key terms
        key_terms = ["gait", "analysis", "features", "temporal", "spatial", "symmetry"]
        terms_found = sum(1 for term in key_terms if term.lower() in explanation.lower())
        if terms_found < 2:
            issues.append("Explanation lacks domain-specific terminology")
            quality_score -= 0.2
        
        # Check for quantitative information
        import re
        numbers = re.findall(r'\d+\.?\d*', explanation)
        if len(numbers) == 0:
            issues.append("Explanation lacks quantitative information")
            quality_score -= 0.2
        
        # Check for reasoning structure
        reasoning_indicators = ["because", "due to", "indicates", "suggests", "shows"]
        reasoning_found = any(indicator in explanation.lower() for indicator in reasoning_indicators)
        if not reasoning_found:
            issues.append("Explanation lacks clear reasoning structure")
            quality_score -= 0.2
        
        return {
            "quality_score": max(0.0, quality_score),
            "issues": issues,
            "length": len(explanation),
            "key_terms_found": terms_found,
            "has_numbers": len(numbers) > 0,
            "has_reasoning": reasoning_found
        }


class LLMResponseAnalyzer:
    """Helper class for analyzing LLM response consistency and quality."""
    
    @staticmethod
    def analyze_consistency(responses: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze consistency across multiple LLM responses."""
        if not responses:
            return {"consistency_score": 0.0, "analysis": "No responses to analyze"}
        
        # Extract classifications and confidences
        classifications = [r.get("classification") for r in responses if "classification" in r]
        confidences = [r.get("confidence") for r in responses if "confidence" in r and isinstance(r["confidence"], (int, float))]
        
        analysis = {}
        
        # Classification consistency
        if classifications:
            unique_classifications = set(classifications)
            most_common = max(unique_classifications, key=classifications.count)
            agreement_ratio = classifications.count(most_common) / len(classifications)
            analysis["classification_consistency"] = {
                "agreement_ratio": agreement_ratio,
                "most_common": most_common,
                "unique_count": len(unique_classifications)
            }
        
        # Confidence consistency
        if confidences:
            confidence_std = np.std(confidences)
            confidence_mean = np.mean(confidences)
            analysis["confidence_consistency"] = {
                "mean": confidence_mean,
                "std": confidence_std,
                "coefficient_of_variation": confidence_std / confidence_mean if confidence_mean > 0 else float('inf')
            }
        
        # Overall consistency score
        consistency_score = 1.0
        if "classification_consistency" in analysis:
            consistency_score *= analysis["classification_consistency"]["agreement_ratio"]
        if "confidence_consistency" in analysis:
            cv = analysis["confidence_consistency"]["coefficient_of_variation"]
            if cv < 0.1:
                consistency_score *= 1.0
            elif cv < 0.2:
                consistency_score *= 0.8
            else:
                consistency_score *= 0.5
        
        analysis["consistency_score"] = consistency_score
        return analysis
    
    @staticmethod
    def detect_response_patterns(responses: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Detect patterns in LLM responses."""
        patterns = {
            "identical_responses": 0,
            "similar_reasoning": 0,
            "confidence_clustering": [],
            "response_length_stats": {}
        }
        
        if len(responses) < 2:
            return patterns
        
        # Check for identical responses
        response_strings = []
        for r in responses:
            response_str = str(sorted(r.items()))
            response_strings.append(response_str)
        
        unique_responses = set(response_strings)
        patterns["identical_responses"] = len(response_strings) - len(unique_responses)
        
        # Analyze reasoning similarity
        reasonings = [r.get("reasoning", "") for r in responses if "reasoning" in r]
        if reasonings:
            # Simple similarity check based on common words
            all_words = set()
            for reasoning in reasonings:
                words = reasoning.lower().split()
                all_words.update(words)
            
            similarity_scores = []
            for i in range(len(reasonings)):
                for j in range(i + 1, len(reasonings)):
                    words1 = set(reasonings[i].lower().split())
                    words2 = set(reasonings[j].lower().split())
                    if words1 or words2:
                        similarity = len(words1 & words2) / len(words1 | words2)
                        similarity_scores.append(similarity)
            
            if similarity_scores:
                patterns["similar_reasoning"] = np.mean(similarity_scores)
        
        # Response length statistics
        if reasonings:
            lengths = [len(r) for r in reasonings]
            patterns["response_length_stats"] = {
                "mean": np.mean(lengths),
                "std": np.std(lengths),
                "min": min(lengths),
                "max": max(lengths)
            }
        
        return patterns


class PropertyTestValidator:
    """Validator for property-based testing framework components."""
    
    def __init__(self):
        self.validation_errors = []
        self.validation_warnings = []
    
    def validate_property_registry(self, registry) -> Dict[str, Any]:
        """Validate property registry completeness and correctness."""
        results = {
            "is_valid": True,
            "errors": [],
            "warnings": [],
            "statistics": {}
        }
        
        try:
            all_properties = registry.get_all_properties()
            
            # Check minimum property count
            if len(all_properties) < 18:
                results["errors"].append(f"Expected at least 18 properties, found {len(all_properties)}")
                results["is_valid"] = False
            
            # Check property completeness
            for prop in all_properties:
                if not prop.name or len(prop.name.strip()) == 0:
                    results["errors"].append("Property with empty name found")
                    results["is_valid"] = False
                
                if not prop.description or len(prop.description.strip()) < 10:
                    results["warnings"].append(f"Property '{prop.name}' has insufficient description")
                
                if not prop.requirements:
                    results["warnings"].append(f"Property '{prop.name}' has no requirement traceability")
            
            # Check category coverage
            from tests.property.property_registry import PropertyCategory
            categories_with_properties = set(p.category for p in all_properties)
            expected_categories = set(PropertyCategory)
            missing_categories = expected_categories - categories_with_properties
            
            if missing_categories:
                results["warnings"].append(f"Missing properties for categories: {missing_categories}")
            
            # Generate statistics
            results["statistics"] = {
                "total_properties": len(all_properties),
                "categories_covered": len(categories_with_properties),
                "properties_with_requirements": len([p for p in all_properties if p.requirements]),
                "enabled_properties": len([p for p in all_properties if p.enabled])
            }
            
        except Exception as e:
            results["errors"].append(f"Registry validation failed: {str(e)}")
            results["is_valid"] = False
        
        return results
    
    def validate_test_data_generation(self, data_manager) -> Dict[str, Any]:
        """Validate test data generation capabilities."""
        results = {
            "is_valid": True,
            "errors": [],
            "warnings": [],
            "capabilities": {}
        }
        
        # Test keypoints generation
        try:
            keypoints = data_manager.create_property_test_data("keypoints", count=5, variation="mixed")
            if len(keypoints) != 5:
                results["errors"].append(f"Keypoints generation: expected 5, got {len(keypoints)}")
                results["is_valid"] = False
            else:
                # Validate keypoint structure
                for i, kp_list in enumerate(keypoints):
                    if not isinstance(kp_list, list):
                        results["errors"].append(f"Keypoint set {i} is not a list")
                        results["is_valid"] = False
                        continue
                    
                    for j, kp in enumerate(kp_list):
                        if not isinstance(kp, dict):
                            results["errors"].append(f"Keypoint {i}.{j} is not a dictionary")
                            results["is_valid"] = False
                        elif not all(field in kp for field in ["x", "y", "confidence"]):
                            results["errors"].append(f"Keypoint {i}.{j} missing required fields")
                            results["is_valid"] = False
                        elif not (0.0 <= kp["confidence"] <= 1.0):
                            results["errors"].append(f"Keypoint {i}.{j} confidence out of range: {kp['confidence']}")
                            results["is_valid"] = False
            
            results["capabilities"]["keypoints_generation"] = True
            
        except Exception as e:
            results["errors"].append(f"Keypoints generation failed: {str(e)}")
            results["is_valid"] = False
            results["capabilities"]["keypoints_generation"] = False
        
        # Test gait features generation
        try:
            features = data_manager.create_property_test_data("gait_features", count=3, variation="mixed")
            if len(features) != 3:
                results["errors"].append(f"Gait features generation: expected 3, got {len(features)}")
                results["is_valid"] = False
            else:
                for i, feature_set in enumerate(features):
                    if not isinstance(feature_set, dict):
                        results["errors"].append(f"Gait feature set {i} is not a dictionary")
                        results["is_valid"] = False
                    elif not any(cat in feature_set for cat in ["temporal_features", "spatial_features", "symmetry_features"]):
                        results["warnings"].append(f"Gait feature set {i} missing standard categories")
            
            results["capabilities"]["gait_features_generation"] = True
            
        except Exception as e:
            results["errors"].append(f"Gait features generation failed: {str(e)}")
            results["is_valid"] = False
            results["capabilities"]["gait_features_generation"] = False
        
        # Test classification results generation
        try:
            classifications = data_manager.create_property_test_data("classification_results", count=2, variation="mixed")
            if len(classifications) != 2:
                results["errors"].append(f"Classification results generation: expected 2, got {len(classifications)}")
                results["is_valid"] = False
            else:
                for i, result in enumerate(classifications):
                    if not isinstance(result, dict):
                        results["errors"].append(f"Classification result {i} is not a dictionary")
                        results["is_valid"] = False
                    elif not all(field in result for field in ["is_normal", "confidence"]):
                        results["errors"].append(f"Classification result {i} missing required fields")
                        results["is_valid"] = False
                    elif not isinstance(result["is_normal"], bool):
                        results["errors"].append(f"Classification result {i} 'is_normal' is not boolean")
                        results["is_valid"] = False
                    elif not (0.0 <= result["confidence"] <= 1.0):
                        results["errors"].append(f"Classification result {i} confidence out of range: {result['confidence']}")
                        results["is_valid"] = False
            
            results["capabilities"]["classification_results_generation"] = True
            
        except Exception as e:
            results["errors"].append(f"Classification results generation failed: {str(e)}")
            results["is_valid"] = False
            results["capabilities"]["classification_results_generation"] = False
        
        return results
    
    def validate_performance_monitoring(self, monitor) -> Dict[str, Any]:
        """Validate performance monitoring capabilities."""
        results = {
            "is_valid": True,
            "errors": [],
            "warnings": [],
            "features": {}
        }
        
        try:
            # Test context manager
            with monitor.start_monitoring("validation_test"):
                import time
                time.sleep(0.01)  # Simulate work
            
            if len(monitor.current_metrics) == 0:
                results["errors"].append("Performance monitoring did not record any metrics")
                results["is_valid"] = False
            else:
                latest_metric = monitor.current_metrics[-1]
                if latest_metric.test_name != "validation_test":
                    results["errors"].append(f"Metric test name mismatch: expected 'validation_test', got '{latest_metric.test_name}'")
                    results["is_valid"] = False
                
                if latest_metric.execution_time <= 0:
                    results["errors"].append(f"Invalid execution time: {latest_metric.execution_time}")
                    results["is_valid"] = False
            
            results["features"]["context_manager"] = True
            
        except Exception as e:
            results["errors"].append(f"Performance monitoring context manager failed: {str(e)}")
            results["is_valid"] = False
            results["features"]["context_manager"] = False
        
        try:
            # Test report generation
            report = monitor.generate_report()
            if not isinstance(report, dict):
                results["errors"].append("Performance report is not a dictionary")
                results["is_valid"] = False
            elif "total_tests" not in report:
                results["errors"].append("Performance report missing 'total_tests' field")
                results["is_valid"] = False
            
            results["features"]["report_generation"] = True
            
        except Exception as e:
            results["errors"].append(f"Performance report generation failed: {str(e)}")
            results["is_valid"] = False
            results["features"]["report_generation"] = False
        
        return results
    
    def validate_framework_integration(self, registry, data_manager, monitor) -> Dict[str, Any]:
        """Validate integration between framework components."""
        results = {
            "is_valid": True,
            "errors": [],
            "warnings": [],
            "integration_tests": {}
        }
        
        # Test registry-data manager integration
        try:
            all_properties = registry.get_all_properties()
            test_data = data_manager.create_property_test_data("keypoints", count=1)
            
            if len(all_properties) > 0 and len(test_data) > 0:
                results["integration_tests"]["registry_data_manager"] = True
            else:
                results["warnings"].append("Registry-data manager integration test inconclusive")
                results["integration_tests"]["registry_data_manager"] = False
                
        except Exception as e:
            results["errors"].append(f"Registry-data manager integration failed: {str(e)}")
            results["is_valid"] = False
            results["integration_tests"]["registry_data_manager"] = False
        
        # Test data manager-monitor integration
        try:
            with monitor.start_monitoring("integration_test"):
                test_data = data_manager.create_property_test_data("gait_features", count=1)
            
            if len(monitor.current_metrics) > 0 and len(test_data) > 0:
                results["integration_tests"]["data_manager_monitor"] = True
            else:
                results["warnings"].append("Data manager-monitor integration test inconclusive")
                results["integration_tests"]["data_manager_monitor"] = False
                
        except Exception as e:
            results["errors"].append(f"Data manager-monitor integration failed: {str(e)}")
            results["is_valid"] = False
            results["integration_tests"]["data_manager_monitor"] = False
        
        # Test complete workflow
        try:
            # Get a property from registry
            properties = registry.get_all_properties()
            if properties:
                test_property = properties[0]
                
                # Generate test data
                with monitor.start_monitoring(f"workflow_test_{test_property.name}"):
                    if "keypoint" in test_property.name.lower():
                        test_data = data_manager.create_property_test_data("keypoints", count=1)
                    elif "gait" in test_property.name.lower():
                        test_data = data_manager.create_property_test_data("gait_features", count=1)
                    else:
                        test_data = data_manager.create_property_test_data("classification_results", count=1)
                
                # Verify workflow completed
                if len(test_data) > 0 and len(monitor.current_metrics) > 0:
                    results["integration_tests"]["complete_workflow"] = True
                else:
                    results["warnings"].append("Complete workflow test inconclusive")
                    results["integration_tests"]["complete_workflow"] = False
            else:
                results["warnings"].append("No properties available for workflow test")
                results["integration_tests"]["complete_workflow"] = False
                
        except Exception as e:
            results["errors"].append(f"Complete workflow integration failed: {str(e)}")
            results["is_valid"] = False
            results["integration_tests"]["complete_workflow"] = False
        
        return results
    
    def generate_validation_report(self, registry, data_manager, monitor) -> Dict[str, Any]:
        """Generate comprehensive validation report for the testing framework."""
        report = {
            "timestamp": time.time(),
            "framework_version": "1.0.0",
            "overall_status": "unknown",
            "component_validations": {},
            "integration_validation": {},
            "summary": {},
            "recommendations": []
        }
        
        # Validate individual components
        report["component_validations"]["registry"] = self.validate_property_registry(registry)
        report["component_validations"]["data_generation"] = self.validate_test_data_generation(data_manager)
        report["component_validations"]["performance_monitoring"] = self.validate_performance_monitoring(monitor)
        
        # Validate integration
        report["integration_validation"] = self.validate_framework_integration(registry, data_manager, monitor)
        
        # Determine overall status
        all_valid = all(
            validation["is_valid"] 
            for validation in report["component_validations"].values()
        ) and report["integration_validation"]["is_valid"]
        
        report["overall_status"] = "valid" if all_valid else "invalid"
        
        # Generate summary
        total_errors = sum(
            len(validation["errors"]) 
            for validation in report["component_validations"].values()
        ) + len(report["integration_validation"]["errors"])
        
        total_warnings = sum(
            len(validation["warnings"]) 
            for validation in report["component_validations"].values()
        ) + len(report["integration_validation"]["warnings"])
        
        report["summary"] = {
            "total_errors": total_errors,
            "total_warnings": total_warnings,
            "components_tested": len(report["component_validations"]),
            "integration_tests_passed": sum(
                1 for test_result in report["integration_validation"]["integration_tests"].values() 
                if test_result
            ),
            "integration_tests_total": len(report["integration_validation"]["integration_tests"])
        }
        
        # Generate recommendations
        if total_errors > 0:
            report["recommendations"].append("Address all validation errors before using the framework")
        
        if total_warnings > 5:
            report["recommendations"].append("Consider addressing validation warnings to improve framework quality")
        
        registry_stats = report["component_validations"]["registry"].get("statistics", {})
        if registry_stats.get("total_properties", 0) < 20:
            report["recommendations"].append("Consider adding more property tests for better coverage")
        
        return report


# Global validator instance
framework_validator = PropertyTestValidator()


def validate_framework_setup(registry=None, data_manager=None, monitor=None):
    """Convenience function to validate framework setup."""
    if registry is None:
        from tests.property.property_registry import PropertyTestRegistry
        registry = PropertyTestRegistry()
    
    if data_manager is None:
        from tests.fixtures.real_data_fixtures import RealDataManager
        data_manager = RealDataManager()
    
    if monitor is None:
        from tests.utils.test_performance import TestPerformanceMonitor
        monitor = TestPerformanceMonitor()
    
    return framework_validator.generate_validation_report(registry, data_manager, monitor)