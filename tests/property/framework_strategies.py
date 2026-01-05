"""
Hypothesis strategies for testing framework validation.

This module provides specialized strategies for generating test data
specifically for validating the testing framework itself.
"""

from hypothesis import strategies as st
from typing import Dict, List, Any, Optional
import string
from tests.property.property_registry import PropertyCategory, PropertyPriority


# Basic framework data strategies
@st.composite
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


@st.composite
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


@st.composite
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


@st.composite
def property_definition_strategy(draw):
    """Generate property test definitions."""
    name = draw(property_name_strategy())
    category = draw(st.sampled_from(list(PropertyCategory)))
    priority = draw(st.sampled_from(list(PropertyPriority)))
    
    # Generate requirements
    num_requirements = draw(st.integers(min_value=0, max_value=5))
    requirements = []
    for i in range(num_requirements):
        req_major = draw(st.integers(min_value=1, max_value=10))
        req_minor = draw(st.integers(min_value=1, max_value=20))
        requirements.append(f"REQ-{req_major}.{req_minor}")
    
    # Generate tags
    available_tags = [
        "validation", "consistency", "bounds", "performance", "integration",
        "error_handling", "edge_cases", "determinism", "completeness"
    ]
    num_tags = draw(st.integers(min_value=0, max_value=4))
    tags = draw(st.lists(
        st.sampled_from(available_tags),
        min_size=num_tags,
        max_size=num_tags,
        unique=True
    ))
    
    return {
        "name": name,
        "description": f"Property test for {name.replace('_', ' ')}",
        "category": category,
        "priority": priority,
        "requirements": requirements,
        "tags": set(tags),
        "expected_examples": draw(st.integers(min_value=10, max_value=500)),
        "timeout_seconds": draw(st.integers(min_value=5, max_value=120))
    }


@st.composite
def test_execution_context_strategy(draw):
    """Generate test execution context data."""
    return {
        "test_id": draw(st.text(min_size=5, max_size=20, alphabet=string.ascii_letters + string.digits)),
        "execution_mode": draw(st.sampled_from(["sequential", "parallel", "distributed"])),
        "worker_count": draw(st.integers(min_value=1, max_value=16)),
        "timeout_seconds": draw(st.integers(min_value=10, max_value=300)),
        "memory_limit_mb": draw(st.integers(min_value=256, max_value=4096)),
        "environment": {
            "python_version": draw(st.sampled_from(["3.11", "3.12", "3.13"])),
            "platform": draw(st.sampled_from(["linux", "windows", "macos"])),
            "ci_environment": draw(st.booleans())
        },
        "data_sources": {
            "real_data_available": draw(st.booleans()),
            "synthetic_data_quality": draw(st.sampled_from(["low", "medium", "high"])),
            "cache_enabled": draw(st.booleans())
        }
    }


@st.composite
def performance_metrics_strategy(draw):
    """Generate performance metrics data."""
    return {
        "execution_time": draw(st.floats(min_value=0.001, max_value=300.0)),
        "memory_usage_mb": draw(st.floats(min_value=10.0, max_value=2048.0)),
        "cpu_usage_percent": draw(st.floats(min_value=0.0, max_value=100.0)),
        "iterations_completed": draw(st.integers(min_value=1, max_value=1000)),
        "failures_count": draw(st.integers(min_value=0, max_value=10)),
        "success_rate": draw(st.floats(min_value=0.0, max_value=1.0)),
        "throughput_per_second": draw(st.floats(min_value=0.1, max_value=1000.0)),
        "latency_percentiles": {
            "p50": draw(st.floats(min_value=0.001, max_value=1.0)),
            "p95": draw(st.floats(min_value=0.01, max_value=5.0)),
            "p99": draw(st.floats(min_value=0.1, max_value=10.0))
        }
    }


@st.composite
def error_scenario_strategy(draw):
    """Generate error scenarios for testing framework robustness."""
    error_type = draw(st.sampled_from([
        "timeout", "memory_exhaustion", "invalid_input", "network_failure",
        "file_not_found", "permission_denied", "resource_busy", "corruption"
    ]))
    
    base_scenario = {
        "error_type": error_type,
        "severity": draw(st.sampled_from(["low", "medium", "high", "critical"])),
        "recoverable": draw(st.booleans()),
        "retry_count": draw(st.integers(min_value=0, max_value=5))
    }
    
    # Add error-specific details
    if error_type == "timeout":
        base_scenario.update({
            "timeout_duration": draw(st.floats(min_value=0.1, max_value=60.0)),
            "operation": draw(st.sampled_from(["data_loading", "computation", "network_request"]))
        })
    elif error_type == "memory_exhaustion":
        base_scenario.update({
            "memory_limit_mb": draw(st.integers(min_value=128, max_value=2048)),
            "actual_usage_mb": draw(st.integers(min_value=2048, max_value=8192))
        })
    elif error_type == "invalid_input":
        base_scenario.update({
            "input_type": draw(st.sampled_from(["malformed_json", "wrong_type", "out_of_bounds", "null_value"])),
            "validation_failed": draw(st.booleans())
        })
    
    return base_scenario


@st.composite
def test_coverage_strategy(draw):
    """Generate test coverage data."""
    total_lines = draw(st.integers(min_value=100, max_value=10000))
    covered_lines = draw(st.integers(min_value=0, max_value=total_lines))
    
    return {
        "total_lines": total_lines,
        "covered_lines": covered_lines,
        "coverage_percentage": (covered_lines / total_lines) * 100 if total_lines > 0 else 0.0,
        "branch_coverage": draw(st.floats(min_value=0.0, max_value=100.0)),
        "function_coverage": draw(st.floats(min_value=0.0, max_value=100.0)),
        "uncovered_lines": draw(st.lists(
            st.integers(min_value=1, max_value=total_lines),
            max_size=min(50, total_lines - covered_lines)
        )),
        "files": draw(st.lists(
            st.text(min_size=5, max_size=50, alphabet=string.ascii_letters + "/_." + string.digits),
            min_size=1,
            max_size=20
        ))
    }


# Composite strategies for complex framework scenarios
@st.composite
def framework_integration_scenario_strategy(draw):
    """Generate complete framework integration scenarios."""
    return {
        "config": draw(framework_config_strategy()),
        "properties": draw(st.lists(
            property_definition_strategy(),
            min_size=5,
            max_size=25
        )),
        "execution_context": draw(test_execution_context_strategy()),
        "expected_performance": draw(performance_metrics_strategy()),
        "error_scenarios": draw(st.lists(
            error_scenario_strategy(),
            min_size=0,
            max_size=5
        )),
        "coverage_targets": draw(test_coverage_strategy())
    }


# Validation strategies
def valid_property_name():
    """Strategy for generating valid property names."""
    return st.text(
        min_size=5,
        max_size=50,
        alphabet=st.characters(whitelist_categories=('Lu', 'Ll', 'Nd', 'Pc'))
    ).filter(lambda x: x.replace('_', '').isalnum() and not x.startswith('_') and not x.endswith('_'))


def valid_requirement_id():
    """Strategy for generating valid requirement IDs."""
    return st.builds(
        lambda major, minor: f"REQ-{major}.{minor}",
        major=st.integers(min_value=1, max_value=20),
        minor=st.integers(min_value=1, max_value=50)
    )


def realistic_performance_bounds():
    """Strategy for generating realistic performance bounds."""
    return st.builds(
        dict,
        max_execution_time=st.floats(min_value=0.1, max_value=300.0),
        max_memory_mb=st.integers(min_value=64, max_value=4096),
        min_success_rate=st.floats(min_value=0.8, max_value=1.0),
        max_error_rate=st.floats(min_value=0.0, max_value=0.2)
    )