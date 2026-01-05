"""
Hypothesis strategies for property-based testing of AlexPose components.
"""

import numpy as np
from pathlib import Path
from typing import Dict, List, Any, Optional
from hypothesis import strategies as st
from hypothesis.strategies import composite

try:
    from ambient.core.frame import Frame, FrameSequence
    from ambient.core.data_models import GaitFeatures, GaitMetrics, Keypoint
    AMBIENT_AVAILABLE = True
except ImportError:
    AMBIENT_AVAILABLE = False


# ============================================================================
# Basic Data Type Strategies
# ============================================================================

# Video format strategies
video_formats = st.sampled_from(['mp4', 'avi', 'mov', 'webm'])
invalid_formats = st.sampled_from(['txt', 'pdf', 'jpg', 'png', 'docx', 'exe'])
all_formats = st.one_of(video_formats, invalid_formats)

# Coordinate and dimension strategies
coordinates = st.floats(min_value=0.0, max_value=2000.0, allow_nan=False, allow_infinity=False)
confidence_scores = st.floats(min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False)
frame_rates = st.floats(min_value=1.0, max_value=120.0, allow_nan=False, allow_infinity=False)
video_durations = st.floats(min_value=0.1, max_value=300.0, allow_nan=False, allow_infinity=False)

# Image dimensions
image_widths = st.integers(min_value=100, max_value=4096)
image_heights = st.integers(min_value=100, max_value=2160)
image_channels = st.sampled_from([1, 3, 4])

# Realistic gait parameter ranges
stride_times = st.floats(min_value=0.8, max_value=2.0, allow_nan=False, allow_infinity=False)
cadences = st.floats(min_value=60, max_value=180, allow_nan=False, allow_infinity=False)
stride_lengths = st.floats(min_value=0.8, max_value=2.5, allow_nan=False, allow_infinity=False)
step_widths = st.floats(min_value=0.05, max_value=0.4, allow_nan=False, allow_infinity=False)
symmetry_indices = st.floats(min_value=0.5, max_value=1.0, allow_nan=False, allow_infinity=False)


# ============================================================================
# Keypoint and Pose Strategies
# ============================================================================

@composite
def keypoint_strategy(draw):
    """Generate a single keypoint with realistic coordinates and confidence."""
    return {
        "x": draw(coordinates),
        "y": draw(coordinates),
        "confidence": draw(confidence_scores)
    }


@composite
def mediapipe_landmarks_strategy(draw):
    """Generate MediaPipe-style landmarks (33 keypoints)."""
    return draw(st.lists(keypoint_strategy(), min_size=33, max_size=33))


@composite
def openpose_keypoints_strategy(draw):
    """Generate OpenPose-style keypoints (25 keypoints)."""
    return draw(st.lists(keypoint_strategy(), min_size=25, max_size=25))


@composite
def variable_keypoints_strategy(draw, min_keypoints=1, max_keypoints=50):
    """Generate variable number of keypoints."""
    return draw(st.lists(keypoint_strategy(), min_size=min_keypoints, max_size=max_keypoints))


@composite
def pose_sequence_strategy(draw, min_frames=5, max_frames=100):
    """Generate a sequence of pose estimations."""
    num_frames = draw(st.integers(min_value=min_frames, max_value=max_frames))
    return [draw(mediapipe_landmarks_strategy()) for _ in range(num_frames)]


@composite
def invalid_keypoint_strategy(draw):
    """Generate invalid keypoint data for error testing."""
    return draw(st.one_of(
        # Non-dict types - these are truly invalid
        st.none(),
        st.text(),
        st.lists(st.floats()),
        st.integers(),
        
        # Missing fields - create incomplete dictionaries
        st.just({"x": 100.0}),  # Missing y and confidence
        st.just({"y": 200.0}),  # Missing x and confidence
        st.just({"x": 100.0, "y": 200.0}),  # Missing confidence
        
        # Invalid field types
        st.just({"x": "invalid", "y": 200.0, "confidence": 0.5}),
        st.just({"x": 100.0, "y": "invalid", "confidence": 0.5}),
        st.just({"x": 100.0, "y": 200.0, "confidence": "invalid"}),
        st.just({"x": None, "y": 200.0, "confidence": 0.5}),
        st.just({"x": 100.0, "y": None, "confidence": 0.5}),
        st.just({"x": 100.0, "y": 200.0, "confidence": None}),
        
        # Invalid confidence ranges
        st.just({"x": 100.0, "y": 200.0, "confidence": 1.5}),  # > 1.0
        st.just({"x": 100.0, "y": 200.0, "confidence": -0.5}),  # < 0.0
        st.just({"x": 100.0, "y": 200.0, "confidence": 10.0}),  # Way > 1.0
        st.just({"x": 100.0, "y": 200.0, "confidence": -10.0}),  # Way < 0.0
    ))


# ============================================================================
# Gait Analysis Strategies
# ============================================================================

@composite
def temporal_features_strategy(draw, gait_type="normal"):
    """Generate temporal gait features."""
    if gait_type == "normal":
        return {
            "stride_time": draw(st.floats(min_value=1.0, max_value=1.4)),
            "cadence": draw(st.floats(min_value=100, max_value=130)),
            "stance_phase_duration": draw(st.floats(min_value=0.6, max_value=0.7)),
            "swing_phase_duration": draw(st.floats(min_value=0.3, max_value=0.4)),
            "double_support_time": draw(st.floats(min_value=0.1, max_value=0.15))
        }
    elif gait_type == "abnormal":
        return {
            "stride_time": draw(st.floats(min_value=1.4, max_value=2.0)),
            "cadence": draw(st.floats(min_value=60, max_value=100)),
            "stance_phase_duration": draw(st.floats(min_value=0.7, max_value=0.8)),
            "swing_phase_duration": draw(st.floats(min_value=0.2, max_value=0.3)),
            "double_support_time": draw(st.floats(min_value=0.15, max_value=0.25))
        }
    else:  # random
        return {
            "stride_time": draw(stride_times),
            "cadence": draw(cadences),
            "stance_phase_duration": draw(st.floats(min_value=0.5, max_value=0.8)),
            "swing_phase_duration": draw(st.floats(min_value=0.2, max_value=0.5)),
            "double_support_time": draw(st.floats(min_value=0.05, max_value=0.3))
        }


@composite
def spatial_features_strategy(draw, gait_type="normal"):
    """Generate spatial gait features."""
    if gait_type == "normal":
        stride_length = draw(st.floats(min_value=1.2, max_value=1.6))
        return {
            "stride_length": stride_length,
            "step_width": draw(st.floats(min_value=0.1, max_value=0.2)),
            "step_length_left": stride_length / 2 + draw(st.floats(min_value=-0.05, max_value=0.05)),
            "step_length_right": stride_length / 2 + draw(st.floats(min_value=-0.05, max_value=0.05))
        }
    elif gait_type == "abnormal":
        stride_length = draw(st.floats(min_value=0.8, max_value=1.2))
        return {
            "stride_length": stride_length,
            "step_width": draw(st.floats(min_value=0.2, max_value=0.4)),
            "step_length_left": stride_length / 2 + draw(st.floats(min_value=-0.2, max_value=0.2)),
            "step_length_right": stride_length / 2 + draw(st.floats(min_value=-0.2, max_value=0.2))
        }
    else:  # random
        stride_length = draw(stride_lengths)
        return {
            "stride_length": stride_length,
            "step_width": draw(step_widths),
            "step_length_left": draw(st.floats(min_value=0.3, max_value=1.5)),
            "step_length_right": draw(st.floats(min_value=0.3, max_value=1.5))
        }


@composite
def symmetry_features_strategy(draw, gait_type="normal"):
    """Generate symmetry gait features."""
    if gait_type == "normal":
        return {
            "left_right_symmetry": draw(st.floats(min_value=0.9, max_value=1.0)),
            "temporal_symmetry": draw(st.floats(min_value=0.85, max_value=0.98)),
            "spatial_symmetry": draw(st.floats(min_value=0.88, max_value=0.98))
        }
    elif gait_type == "abnormal":
        return {
            "left_right_symmetry": draw(st.floats(min_value=0.5, max_value=0.85)),
            "temporal_symmetry": draw(st.floats(min_value=0.4, max_value=0.8)),
            "spatial_symmetry": draw(st.floats(min_value=0.45, max_value=0.82))
        }
    else:  # random
        return {
            "left_right_symmetry": draw(symmetry_indices),
            "temporal_symmetry": draw(symmetry_indices),
            "spatial_symmetry": draw(symmetry_indices)
        }


@composite
def gait_features_strategy(draw, gait_type="random"):
    """Generate complete gait features."""
    return {
        "temporal_features": draw(temporal_features_strategy(gait_type)),
        "spatial_features": draw(spatial_features_strategy(gait_type)),
        "symmetry_features": draw(symmetry_features_strategy(gait_type)),
        "kinematic_features": {
            "hip_flexion_max": draw(st.lists(
                st.floats(min_value=10.0, max_value=40.0), 
                min_size=3, max_size=10
            )),
            "knee_flexion_max": draw(st.lists(
                st.floats(min_value=40.0, max_value=80.0), 
                min_size=3, max_size=10
            )),
            "ankle_dorsiflexion_max": draw(st.lists(
                st.floats(min_value=5.0, max_value=25.0), 
                min_size=3, max_size=10
            ))
        },
        "frequency_features": {
            "dominant_frequency": draw(st.floats(min_value=1.0, max_value=3.0)),
            "harmonic_ratio": draw(st.floats(min_value=0.5, max_value=1.0)),
            "spectral_centroid": draw(st.floats(min_value=1.5, max_value=3.5))
        }
    }


# ============================================================================
# Frame and Video Strategies
# ============================================================================

@composite
def frame_metadata_strategy(draw):
    """Generate Frame metadata."""
    return {
        "frame_id": f"frame_{draw(st.integers(min_value=0, max_value=9999)):04d}",
        "sequence_id": f"seq_{draw(st.integers(min_value=0, max_value=999)):03d}",
        "frame_number": draw(st.integers(min_value=0, max_value=10000)),
        "timestamp": draw(st.floats(min_value=0.0, max_value=300.0)),
        "width": draw(image_widths),
        "height": draw(image_heights),
        "channels": draw(image_channels),
        "format": draw(st.sampled_from(['RGB', 'BGR', 'RGBA', 'GRAY']))
    }


@composite
def bounding_box_strategy(draw, max_width=1920, max_height=1080):
    """Generate valid bounding box coordinates."""
    left = draw(st.integers(min_value=0, max_value=max_width - 100))
    top = draw(st.integers(min_value=0, max_value=max_height - 100))
    width = draw(st.integers(min_value=50, max_value=min(500, max_width - left)))
    height = draw(st.integers(min_value=50, max_value=min(500, max_height - top)))
    
    return {
        "left": float(left),
        "top": float(top),
        "width": float(width),
        "height": float(height)
    }


@composite
def invalid_bounding_box_strategy(draw, max_width=1920, max_height=1080):
    """Generate invalid bounding box coordinates."""
    return draw(st.one_of(
        # Negative coordinates
        st.builds(dict,
            left=st.floats(min_value=-100, max_value=-1),
            top=st.integers(min_value=0, max_value=100),
            width=st.integers(min_value=50, max_value=200),
            height=st.integers(min_value=50, max_value=200)
        ),
        
        # Zero or negative dimensions
        st.builds(dict,
            left=st.integers(min_value=0, max_value=100),
            top=st.integers(min_value=0, max_value=100),
            width=st.integers(min_value=-50, max_value=0),
            height=st.integers(min_value=50, max_value=200)
        ),
        
        # Out of bounds
        st.builds(dict,
            left=st.integers(min_value=max_width, max_value=max_width + 100),
            top=st.integers(min_value=0, max_value=100),
            width=st.integers(min_value=50, max_value=200),
            height=st.integers(min_value=50, max_value=200)
        ),
        
        # Missing fields
        st.builds(dict,
            left=st.integers(min_value=0, max_value=100),
            top=st.integers(min_value=0, max_value=100),
            width=st.integers(min_value=50, max_value=200)
            # Missing height
        )
    ))


# ============================================================================
# Configuration Strategies
# ============================================================================

@composite
def video_processing_config_strategy(draw):
    """Generate video processing configuration."""
    return {
        "supported_formats": draw(st.lists(video_formats, min_size=1, max_size=10, unique=True)),
        "default_frame_rate": draw(frame_rates),
        "max_video_size_mb": draw(st.integers(min_value=10, max_value=2000)),
        "ffmpeg_enabled": draw(st.booleans())
    }


@composite
def pose_estimation_config_strategy(draw):
    """Generate pose estimation configuration."""
    estimators = ["mediapipe", "openpose", "ultralytics", "alphapose"]
    
    return {
        "estimators": {
            estimator: {
                "enabled": draw(st.booleans()),
                "confidence_threshold": draw(confidence_scores)
            }
            for estimator in draw(st.lists(st.sampled_from(estimators), min_size=1, unique=True))
        },
        "default_estimator": draw(st.sampled_from(estimators)),
        "confidence_threshold": draw(confidence_scores)
    }


@composite
def classification_config_strategy(draw):
    """Generate classification configuration."""
    return {
        "llm": {
            "provider": draw(st.sampled_from(["openai", "gemini", "anthropic"])),
            "model": draw(st.sampled_from(["gpt-4o-mini", "gpt-4", "gemini-pro", "claude-3"])),
            "temperature": draw(st.floats(min_value=0.0, max_value=2.0)),
            "max_tokens": draw(st.integers(min_value=100, max_value=4000)),
            "enabled": draw(st.booleans())
        },
        "normal_abnormal_threshold": draw(confidence_scores),
        "condition_confidence_threshold": draw(confidence_scores)
    }


# ============================================================================
# Error and Edge Case Strategies
# ============================================================================

@composite
def corrupted_data_strategy(draw):
    """Generate corrupted or invalid data for error testing."""
    return draw(st.one_of(
        st.none(),
        st.text(min_size=0, max_size=10),
        st.binary(min_size=0, max_size=100),
        st.lists(st.integers(), min_size=0, max_size=5),
        st.dictionaries(st.text(), st.integers(), min_size=0, max_size=3)
    ))


@composite
def edge_case_values_strategy(draw):
    """Generate edge case values for boundary testing."""
    return draw(st.one_of(
        st.just(0),
        st.just(0.0),
        st.just(-1),
        st.just(float('inf')),
        st.just(float('-inf')),
        st.just(float('nan')),
        st.just(""),
        st.just([]),
        st.just({}),
        st.just(None)
    ))


# ============================================================================
# Classification Strategies
# ============================================================================

@composite
def classification_result_strategy(draw):
    """Generate classification results."""
    classification = draw(st.sampled_from(["normal", "abnormal"]))
    confidence = draw(confidence_scores)
    
    return {
        "classification": classification,
        "confidence": confidence,
        "reasoning": f"Classification based on analysis with {confidence:.2f} confidence",
        "timestamp": draw(st.floats(min_value=0.0, max_value=1000.0)),
        "model_version": draw(st.sampled_from(["v1.0", "v1.1", "v2.0"])),
        "features_analyzed": draw(st.lists(
            st.sampled_from(["temporal", "spatial", "symmetry", "kinematic"]),
            min_size=1, max_size=4, unique=True
        ))
    }


@composite
def llm_response_strategy(draw):
    """Generate LLM response data."""
    classification = draw(st.sampled_from(["normal", "abnormal"]))
    confidence = draw(confidence_scores)
    
    # Generate reasoning based on classification
    if classification == "abnormal":
        reasoning_templates = [
            "Gait analysis reveals significant deviations from normal parameters",
            "Multiple abnormal indicators detected in temporal and spatial features",
            "Asymmetry patterns suggest potential gait abnormalities"
        ]
    else:
        reasoning_templates = [
            "Gait parameters fall within normal physiological ranges",
            "No significant abnormalities detected in gait pattern",
            "Temporal and spatial features indicate healthy gait"
        ]
    
    reasoning = draw(st.sampled_from(reasoning_templates))
    
    return {
        "classification": classification,
        "confidence": confidence,
        "reasoning": reasoning,
        "detailed_analysis": {
            "temporal_assessment": f"Temporal features show {classification} patterns",
            "spatial_assessment": f"Spatial measurements within {classification} range",
            "symmetry_assessment": f"Symmetry indices indicate {classification} gait"
        },
        "recommendations": draw(st.lists(
            st.text(min_size=10, max_size=100),
            min_size=0, max_size=3
        )),
        "model_info": {
            "model": draw(st.sampled_from(["gpt-4o-mini", "gpt-4", "claude-3"])),
            "temperature": draw(st.floats(min_value=0.0, max_value=1.0)),
            "tokens_used": draw(st.integers(min_value=100, max_value=2000))
        }
    }


@composite
def gait_cycle_strategy(draw):
    """Generate gait cycle data."""
    cycle_duration = draw(st.floats(min_value=0.8, max_value=1.8))
    frame_rate = draw(st.floats(min_value=20.0, max_value=60.0))
    
    # Calculate cycle boundaries
    cycle_frames = int(cycle_duration * frame_rate)
    
    return {
        "duration": cycle_duration,
        "frame_rate": frame_rate,
        "total_frames": cycle_frames,
        "heel_strike_frames": [0, cycle_frames],
        "toe_off_frames": [int(cycle_frames * 0.6)],
        "stance_phase": {
            "start_frame": 0,
            "end_frame": int(cycle_frames * 0.6),
            "duration_ratio": 0.6
        },
        "swing_phase": {
            "start_frame": int(cycle_frames * 0.6),
            "end_frame": cycle_frames,
            "duration_ratio": 0.4
        },
        "quality_score": draw(st.floats(min_value=0.5, max_value=1.0))
    }


# ============================================================================
# Composite Strategies for Complex Scenarios
# ============================================================================

@composite
def realistic_gait_analysis_scenario(draw):
    """Generate a realistic gait analysis scenario."""
    gait_type = draw(st.sampled_from(["normal", "abnormal"]))
    
    return {
        "video_metadata": {
            "duration": draw(st.floats(min_value=3.0, max_value=30.0)),
            "frame_rate": draw(st.sampled_from([25.0, 30.0, 60.0])),
            "resolution": draw(st.sampled_from([(640, 480), (1280, 720), (1920, 1080)]))
        },
        "pose_sequence": draw(pose_sequence_strategy(min_frames=30, max_frames=300)),
        "gait_features": draw(gait_features_strategy(gait_type)),
        "expected_classification": gait_type,
        "bounding_boxes": draw(st.lists(
            bounding_box_strategy(), 
            min_size=1, 
            max_size=10
        ))
    }


@composite
def multi_subject_scenario(draw):
    """Generate multi-subject gait analysis scenario."""
    num_subjects = draw(st.integers(min_value=1, max_value=5))
    
    subjects = []
    for i in range(num_subjects):
        gait_type = draw(st.sampled_from(["normal", "abnormal"]))
        subjects.append({
            "subject_id": f"subject_{i}",
            "gait_type": gait_type,
            "features": draw(gait_features_strategy(gait_type)),
            "bounding_box": draw(bounding_box_strategy())
        })
    
    return {
        "subjects": subjects,
        "video_metadata": {
            "duration": draw(st.floats(min_value=5.0, max_value=60.0)),
            "frame_rate": 30.0,
            "resolution": (1920, 1080)
        }
    }


# ============================================================================
# Utility Functions
# ============================================================================

def generate_test_cases(strategy, num_cases=10):
    """Generate a list of test cases from a strategy."""
    from hypothesis import given, example
    
    test_cases = []
    
    @given(strategy)
    def collect_case(case):
        test_cases.append(case)
    
    # Generate cases
    for _ in range(num_cases):
        try:
            collect_case()
        except Exception:
            continue  # Skip invalid cases
    
    return test_cases[:num_cases]


def create_property_test_data(strategy_name: str, **kwargs):
    """Create property test data for a specific strategy."""
    strategies_map = {
        "keypoints": mediapipe_landmarks_strategy(),
        "gait_features": gait_features_strategy(),
        "video_config": video_processing_config_strategy(),
        "pose_config": pose_estimation_config_strategy(),
        "classification_config": classification_config_strategy(),
        "bounding_box": bounding_box_strategy(),
        "realistic_scenario": realistic_gait_analysis_scenario()
    }
    
    if strategy_name not in strategies_map:
        raise ValueError(f"Unknown strategy: {strategy_name}")
    
    return strategies_map[strategy_name]

# ============================================================================
# Framework Testing Strategies (for Task 2.1)
# ============================================================================

# Basic framework data strategies
@composite
def property_name_strategy(draw):
    """Generate valid property names for testing."""
    # Property names should be descriptive and follow naming conventions
    prefix = draw(st.sampled_from([
        "test_", "validate_", "check_", "verify_", "ensure_"
    ]))
    
    main_part = draw(st.text(
        min_size=5, 
        max_size=30,
        alphabet=st.characters(whitelist_categories=('Lu', 'Ll', 'Nd')),
    ))
    
    suffix = draw(st.sampled_from([
        "_property", "_validation", "_consistency", "_bounds", "_completeness", ""
    ]))
    
    return f"{prefix}{main_part}{suffix}".lower().replace(" ", "_")


@composite
def test_data_strategy(draw):
    """Generate test data for framework validation."""
    data_type = draw(st.sampled_from([
        "keypoints", "gait_features", "classification_result", "video_metadata"
    ]))
    
    if data_type == "keypoints":
        num_keypoints = draw(st.integers(min_value=1, max_value=33))
        keypoints = []
        
        for _ in range(num_keypoints):
            keypoint = {
                "x": draw(st.floats(min_value=0.0, max_value=1920.0)),
                "y": draw(st.floats(min_value=0.0, max_value=1080.0)),
                "confidence": draw(st.floats(min_value=0.0, max_value=1.0))
            }
            keypoints.append(keypoint)
        
        return {
            "type": data_type,
            "data": keypoints,
            "metadata": {
                "source": "synthetic",
                "frame_number": draw(st.integers(min_value=0, max_value=1000))
            }
        }
    
    elif data_type == "gait_features":
        temporal_features = {
            "stride_time": draw(st.floats(min_value=0.5, max_value=3.0)),
            "cadence": draw(st.integers(min_value=40, max_value=200)),
            "stance_phase_duration": draw(st.floats(min_value=0.4, max_value=0.8)),
            "swing_phase_duration": draw(st.floats(min_value=0.2, max_value=0.6))
        }
        
        spatial_features = {
            "stride_length": draw(st.floats(min_value=0.3, max_value=3.0)),
            "step_width": draw(st.floats(min_value=0.05, max_value=0.5)),
            "step_length_left": draw(st.floats(min_value=0.2, max_value=1.5)),
            "step_length_right": draw(st.floats(min_value=0.2, max_value=1.5))
        }
        
        symmetry_features = {
            "left_right_symmetry": draw(st.floats(min_value=0.5, max_value=1.0)),
            "temporal_symmetry": draw(st.floats(min_value=0.5, max_value=1.0)),
            "spatial_symmetry": draw(st.floats(min_value=0.5, max_value=1.0))
        }
        
        return {
            "type": data_type,
            "data": {
                "temporal_features": temporal_features,
                "spatial_features": spatial_features,
                "symmetry_features": symmetry_features
            },
            "metadata": {
                "subject_id": draw(st.text(min_size=3, max_size=10)),
                "analysis_timestamp": draw(st.floats(min_value=1000000000, max_value=2000000000))
            }
        }
    
    elif data_type == "classification_result":
        is_normal = draw(st.booleans())
        confidence = draw(st.floats(min_value=0.0, max_value=1.0))
        
        conditions = []
        if not is_normal:
            num_conditions = draw(st.integers(min_value=1, max_value=3))
            condition_names = draw(st.lists(
                st.sampled_from(["parkinson", "hemiplegia", "neuropathy", "myopathy", "ataxia"]),
                min_size=num_conditions,
                max_size=num_conditions,
                unique=True
            ))
            
            for condition in condition_names:
                conditions.append({
                    "name": condition,
                    "confidence": draw(st.floats(min_value=0.3, max_value=1.0))
                })
        
        return {
            "type": data_type,
            "data": {
                "is_normal": is_normal,
                "confidence": confidence,
                "explanation": draw(st.text(min_size=10, max_size=200)),
                "conditions": conditions,
                "features_analyzed": draw(st.lists(
                    st.sampled_from(["temporal", "spatial", "symmetry", "kinematic"]),
                    min_size=1,
                    max_size=4,
                    unique=True
                ))
            },
            "metadata": {
                "model_version": draw(st.text(min_size=3, max_size=10)),
                "processing_time": draw(st.floats(min_value=0.1, max_value=30.0))
            }
        }
    
    elif data_type == "video_metadata":
        return {
            "type": data_type,
            "data": {
                "duration": draw(st.floats(min_value=1.0, max_value=300.0)),
                "frame_rate": draw(st.floats(min_value=15.0, max_value=60.0)),
                "width": draw(st.integers(min_value=320, max_value=1920)),
                "height": draw(st.integers(min_value=240, max_value=1080)),
                "total_frames": draw(st.integers(min_value=30, max_value=18000)),
                "format": draw(st.sampled_from(["mp4", "avi", "mov", "mkv"])),
                "codec": draw(st.sampled_from(["h264", "h265", "xvid", "vp9"]))
            },
            "metadata": {
                "file_size": draw(st.integers(min_value=1024, max_value=1073741824)),  # 1KB to 1GB
                "creation_date": draw(st.text(min_size=10, max_size=20))
            }
        }


@composite
def framework_config_strategy(draw):
    """Generate framework configuration data."""
    return {
        "test_execution": {
            "max_examples": draw(st.integers(min_value=10, max_value=1000)),
            "deadline": draw(st.integers(min_value=1000, max_value=30000)),
            "timeout": draw(st.integers(min_value=5, max_value=300))
        },
        "performance": {
            "memory_limit_mb": draw(st.integers(min_value=512, max_value=8192)),
            "cpu_limit_percent": draw(st.integers(min_value=50, max_value=100)),
            "parallel_workers": draw(st.integers(min_value=1, max_value=16))
        },
        "data_generation": {
            "use_real_data_ratio": draw(st.floats(min_value=0.0, max_value=1.0)),
            "synthetic_data_quality": draw(st.sampled_from(["low", "medium", "high"])),
            "cache_generated_data": draw(st.booleans())
        },
        "reporting": {
            "verbose_output": draw(st.booleans()),
            "save_artifacts": draw(st.booleans()),
            "generate_coverage_report": draw(st.booleans())
        }
    }

