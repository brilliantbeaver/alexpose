"""
Hypothesis strategies for integration testing validation.

This module provides specialized strategies for generating test data
specifically for validating integration testing across system components.
"""

from hypothesis import strategies as st
from hypothesis.strategies import composite
from typing import Dict, List, Any, Optional
import string
import numpy as np


# API Integration Strategies
@composite
def api_endpoint_strategy(draw):
    """Generate API endpoint configurations."""
    endpoint_types = {
        "/analyze": {
            "methods": ["POST"],
            "required_fields": ["video_data", "analysis_type"],
            "response_fields": ["analysis_result", "processing_time"]
        },
        "/classify": {
            "methods": ["POST"],
            "required_fields": ["features", "model_version"],
            "response_fields": ["classification", "confidence", "explanation"]
        },
        "/extract": {
            "methods": ["POST"],
            "required_fields": ["video_url", "extraction_params"],
            "response_fields": ["keypoints", "metadata"]
        },
        "/status": {
            "methods": ["GET"],
            "required_fields": [],
            "response_fields": ["system_status", "uptime", "version"]
        },
        "/health": {
            "methods": ["GET"],
            "required_fields": [],
            "response_fields": ["status", "checks"]
        }
    }
    
    endpoint = draw(st.sampled_from(list(endpoint_types.keys())))
    endpoint_config = endpoint_types[endpoint]
    
    return {
        "endpoint": endpoint,
        "method": draw(st.sampled_from(endpoint_config["methods"])),
        "required_fields": endpoint_config["required_fields"],
        "expected_response_fields": endpoint_config["response_fields"],
        "timeout": draw(st.integers(min_value=5, max_value=60)),
        "retry_count": draw(st.integers(min_value=0, max_value=3)),
        "rate_limit": draw(st.integers(min_value=10, max_value=1000))
    }


@composite
def api_request_payload_strategy(draw):
    """Generate API request payloads."""
    payload_type = draw(st.sampled_from(["video_analysis", "classification", "extraction", "status"]))
    
    if payload_type == "video_analysis":
        return {
            "type": payload_type,
            "video_data": {
                "url": draw(st.text(min_size=10, max_size=100)),
                "format": draw(st.sampled_from(["mp4", "avi", "mov"])),
                "duration": draw(st.floats(min_value=1.0, max_value=300.0)),
                "resolution": draw(st.tuples(
                    st.integers(min_value=320, max_value=1920),
                    st.integers(min_value=240, max_value=1080)
                ))
            },
            "analysis_params": {
                "extract_keypoints": draw(st.booleans()),
                "analyze_gait": draw(st.booleans()),
                "quality_threshold": draw(st.floats(min_value=0.3, max_value=1.0))
            }
        }
    
    elif payload_type == "classification":
        return {
            "type": payload_type,
            "features": {
                "temporal": draw(st.dictionaries(
                    st.text(min_size=5, max_size=20),
                    st.floats(min_value=0.1, max_value=10.0),
                    min_size=3, max_size=8
                )),
                "spatial": draw(st.dictionaries(
                    st.text(min_size=5, max_size=20),
                    st.floats(min_value=0.1, max_value=5.0),
                    min_size=3, max_size=8
                )),
                "symmetry": draw(st.dictionaries(
                    st.text(min_size=5, max_size=20),
                    st.floats(min_value=0.0, max_value=1.0),
                    min_size=2, max_size=5
                ))
            },
            "model_config": {
                "version": draw(st.text(min_size=3, max_size=10)),
                "confidence_threshold": draw(st.floats(min_value=0.5, max_value=0.9))
            }
        }
    
    elif payload_type == "extraction":
        return {
            "type": payload_type,
            "source": {
                "type": draw(st.sampled_from(["file", "url", "stream"])),
                "location": draw(st.text(min_size=10, max_size=100)),
                "format": draw(st.sampled_from(["mp4", "avi", "mov", "webm"]))
            },
            "extraction_params": {
                "keypoint_model": draw(st.sampled_from(["mediapipe", "openpose", "alphapose"])),
                "confidence_threshold": draw(st.floats(min_value=0.1, max_value=0.9)),
                "frame_skip": draw(st.integers(min_value=1, max_value=5))
            }
        }
    
    else:  # status
        return {
            "type": payload_type,
            "include_details": draw(st.booleans()),
            "component_filter": draw(st.lists(
                st.sampled_from(["database", "api", "processor", "classifier"]),
                min_size=0, max_size=4, unique=True
            ))
        }


@composite
def api_response_strategy(draw):
    """Generate API response structures."""
    status = draw(st.sampled_from(["success", "error", "timeout", "rate_limited"]))
    
    base_response = {
        "status": status,
        "timestamp": draw(st.floats(min_value=1600000000, max_value=2000000000)),
        "request_id": draw(st.text(min_size=8, max_size=32, alphabet=string.ascii_letters + string.digits)),
        "processing_time": draw(st.floats(min_value=0.001, max_value=30.0))
    }
    
    if status == "success":
        # Add success-specific fields
        base_response.update({
            "data": draw(st.dictionaries(
                st.text(min_size=3, max_size=20),
                st.one_of(
                    st.text(min_size=1, max_size=100),
                    st.floats(min_value=0.0, max_value=1.0),
                    st.integers(min_value=0, max_value=1000),
                    st.booleans()
                ),
                min_size=1, max_size=10
            )),
            "metadata": {
                "version": draw(st.text(min_size=3, max_size=10)),
                "model_used": draw(st.text(min_size=5, max_size=20))
            }
        })
    
    elif status == "error":
        base_response.update({
            "error": {
                "code": draw(st.integers(min_value=400, max_value=599)),
                "message": draw(st.text(min_size=10, max_size=200)),
                "details": draw(st.dictionaries(
                    st.text(min_size=3, max_size=15),
                    st.text(min_size=5, max_size=50),
                    min_size=0, max_size=5
                ))
            }
        })
    
    return base_response


# Database Integration Strategies
@composite
def database_transaction_strategy(draw):
    """Generate database transaction configurations."""
    operation = draw(st.sampled_from(["SELECT", "INSERT", "UPDATE", "DELETE"]))
    
    base_transaction = {
        "operation": operation,
        "table": draw(st.sampled_from(["analyses", "results", "users", "sessions", "videos"])),
        "transaction_id": draw(st.text(min_size=8, max_size=32, alphabet=string.ascii_letters + string.digits)),
        "isolation_level": draw(st.sampled_from(["READ_COMMITTED", "SERIALIZABLE", "REPEATABLE_READ"])),
        "timeout": draw(st.integers(min_value=5, max_value=300))
    }
    
    if operation in ["INSERT", "UPDATE"]:
        base_transaction["data"] = draw(st.dictionaries(
            st.text(min_size=3, max_size=20, alphabet=string.ascii_letters + "_"),
            st.one_of(
                st.text(min_size=1, max_size=100),
                st.integers(min_value=0, max_value=1000000),
                st.floats(min_value=0.0, max_value=1000.0),
                st.booleans()
            ),
            min_size=1, max_size=15
        ))
        base_transaction["record_count"] = draw(st.integers(min_value=1, max_value=1000))
    
    elif operation == "DELETE":
        base_transaction["where_clause"] = draw(st.dictionaries(
            st.text(min_size=3, max_size=20, alphabet=string.ascii_letters + "_"),
            st.one_of(
                st.text(min_size=1, max_size=50),
                st.integers(min_value=0, max_value=1000),
                st.floats(min_value=0.0, max_value=100.0)
            ),
            min_size=1, max_size=5
        ))
        base_transaction["max_affected"] = draw(st.integers(min_value=1, max_value=100))
    
    elif operation == "SELECT":
        base_transaction["columns"] = draw(st.lists(
            st.text(min_size=3, max_size=20, alphabet=string.ascii_letters + "_"),
            min_size=1, max_size=10, unique=True
        ))
        base_transaction["limit"] = draw(st.integers(min_value=1, max_value=1000))
        base_transaction["offset"] = draw(st.integers(min_value=0, max_value=10000))
    
    return base_transaction


@composite
def database_connection_strategy(draw):
    """Generate database connection configurations."""
    return {
        "host": draw(st.text(min_size=5, max_size=50)),
        "port": draw(st.integers(min_value=1024, max_value=65535)),
        "database": draw(st.text(min_size=3, max_size=30, alphabet=string.ascii_letters + string.digits + "_")),
        "username": draw(st.text(min_size=3, max_size=20, alphabet=string.ascii_letters + string.digits)),
        "pool_size": draw(st.integers(min_value=1, max_value=50)),
        "max_connections": draw(st.integers(min_value=10, max_value=200)),
        "connection_timeout": draw(st.integers(min_value=5, max_value=60)),
        "retry_attempts": draw(st.integers(min_value=1, max_value=5)),
        "ssl_enabled": draw(st.booleans())
    }


# Component Integration Strategies
@composite
def component_integration_strategy(draw):
    """Generate component integration configurations."""
    components = [
        "video_processor", "pose_estimator", "feature_extractor",
        "gait_analyzer", "classifier", "reporter", "api_gateway"
    ]
    
    source = draw(st.sampled_from(components))
    target = draw(st.sampled_from([c for c in components if c != source]))
    
    return {
        "source_component": source,
        "target_component": target,
        "data_format": draw(st.sampled_from(["json", "numpy", "pandas", "protobuf", "custom"])),
        "serialization": draw(st.sampled_from(["json", "pickle", "msgpack", "protobuf"])),
        "compression": draw(st.sampled_from(["none", "gzip", "lz4", "zstd"])),
        "validation_schema": draw(st.booleans()),
        "transformation_required": draw(st.booleans()),
        "async_processing": draw(st.booleans()),
        "batch_size": draw(st.integers(min_value=1, max_value=100)),
        "timeout": draw(st.integers(min_value=1, max_value=300)),
        "retry_policy": {
            "max_retries": draw(st.integers(min_value=0, max_value=5)),
            "backoff_factor": draw(st.floats(min_value=1.0, max_value=3.0)),
            "retry_on_timeout": draw(st.booleans())
        }
    }


@composite
def data_flow_strategy(draw):
    """Generate data flow configurations between components."""
    return {
        "flow_id": draw(st.text(min_size=8, max_size=32, alphabet=string.ascii_letters + string.digits)),
        "source_format": draw(st.sampled_from(["video", "keypoints", "features", "classification"])),
        "target_format": draw(st.sampled_from(["keypoints", "features", "classification", "report"])),
        "transformations": draw(st.lists(
            st.sampled_from(["normalize", "filter", "aggregate", "validate", "enrich"]),
            min_size=0, max_size=5, unique=True
        )),
        "quality_checks": draw(st.lists(
            st.sampled_from(["completeness", "accuracy", "consistency", "timeliness"]),
            min_size=1, max_size=4, unique=True
        )),
        "error_handling": draw(st.sampled_from(["fail_fast", "skip_invalid", "retry", "fallback"])),
        "monitoring": {
            "track_latency": draw(st.booleans()),
            "track_throughput": draw(st.booleans()),
            "track_errors": draw(st.booleans()),
            "alert_thresholds": {
                "max_latency_ms": draw(st.integers(min_value=100, max_value=10000)),
                "min_throughput": draw(st.integers(min_value=1, max_value=1000)),
                "max_error_rate": draw(st.floats(min_value=0.01, max_value=0.5))
            }
        }
    }


# Test Suite Configuration Strategies
@composite
def test_suite_strategy(draw):
    """Generate test suite configurations."""
    return {
        "suite_name": draw(st.text(min_size=5, max_size=30, alphabet=string.ascii_letters + "_")),
        "total_tests": draw(st.integers(min_value=10, max_value=500)),
        "test_categories": draw(st.dictionaries(
            st.sampled_from(["unit", "integration", "performance", "property", "e2e"]),
            st.integers(min_value=1, max_value=100),
            min_size=2, max_size=5
        )),
        "data_requirements": {
            "real_data_ratio": draw(st.floats(min_value=0.3, max_value=0.9)),
            "synthetic_data_quality": draw(st.sampled_from(["low", "medium", "high"])),
            "data_volume": draw(st.sampled_from(["small", "medium", "large"])),
            "data_variety": draw(st.integers(min_value=1, max_value=10))
        },
        "execution_config": {
            "parallel_execution": draw(st.booleans()),
            "max_workers": draw(st.integers(min_value=1, max_value=16)),
            "timeout_per_test": draw(st.integers(min_value=5, max_value=300)),
            "retry_failed_tests": draw(st.booleans()),
            "fail_fast": draw(st.booleans())
        },
        "reporting": {
            "detailed_logs": draw(st.booleans()),
            "performance_metrics": draw(st.booleans()),
            "coverage_analysis": draw(st.booleans()),
            "export_format": draw(st.sampled_from(["json", "xml", "html", "csv"]))
        }
    }


@composite
def real_data_usage_strategy(draw):
    """Generate real data usage configurations."""
    return {
        "target_ratio": draw(st.floats(min_value=0.5, max_value=0.95)),
        "fallback_strategy": draw(st.sampled_from(["synthetic", "cached", "skip", "fail"])),
        "data_sources": {
            "gavd_dataset": draw(st.booleans()),
            "local_videos": draw(st.booleans()),
            "synthetic_generator": draw(st.booleans()),
            "cached_results": draw(st.booleans())
        },
        "quality_requirements": {
            "min_confidence": draw(st.floats(min_value=0.3, max_value=0.9)),
            "min_completeness": draw(st.floats(min_value=0.7, max_value=1.0)),
            "max_noise_level": draw(st.floats(min_value=0.0, max_value=0.3))
        },
        "usage_tracking": {
            "log_data_source": draw(st.booleans()),
            "track_quality_metrics": draw(st.booleans()),
            "monitor_ratio_compliance": draw(st.booleans())
        }
    }


# Pipeline Integration Strategies
@composite
def pipeline_configuration_strategy(draw):
    """Generate end-to-end pipeline configurations."""
    stages = ["input", "preprocessing", "analysis", "classification", "output"]
    
    return {
        "pipeline_id": draw(st.text(min_size=8, max_size=32, alphabet=string.ascii_letters + string.digits)),
        "stages": draw(st.lists(
            st.sampled_from(stages),
            min_size=3, max_size=len(stages), unique=True
        )),
        "input_config": {
            "source_type": draw(st.sampled_from(["file", "url", "stream", "batch"])),
            "format": draw(st.sampled_from(["mp4", "avi", "mov", "json", "csv"])),
            "validation": draw(st.booleans())
        },
        "processing_config": {
            "parallel_stages": draw(st.booleans()),
            "stage_timeout": draw(st.integers(min_value=30, max_value=600)),
            "error_recovery": draw(st.sampled_from(["retry", "skip", "fail", "fallback"])),
            "quality_gates": draw(st.booleans())
        },
        "output_config": {
            "format": draw(st.sampled_from(["json", "csv", "pdf", "html"])),
            "destination": draw(st.sampled_from(["file", "database", "api", "stream"])),
            "include_metadata": draw(st.booleans()),
            "compression": draw(st.booleans())
        },
        "monitoring": {
            "track_progress": draw(st.booleans()),
            "log_intermediate_results": draw(st.booleans()),
            "performance_profiling": draw(st.booleans()),
            "alert_on_failure": draw(st.booleans())
        }
    }


# Error Scenario Strategies
@composite
def integration_error_strategy(draw):
    """Generate integration error scenarios."""
    error_type = draw(st.sampled_from([
        "network_timeout", "service_unavailable", "data_corruption",
        "format_mismatch", "authentication_failure", "rate_limit_exceeded",
        "resource_exhaustion", "dependency_failure"
    ]))
    
    base_error = {
        "error_type": error_type,
        "severity": draw(st.sampled_from(["low", "medium", "high", "critical"])),
        "recoverable": draw(st.booleans()),
        "retry_recommended": draw(st.booleans()),
        "estimated_recovery_time": draw(st.integers(min_value=1, max_value=3600))
    }
    
    # Add error-specific details
    if error_type == "network_timeout":
        base_error.update({
            "timeout_duration": draw(st.integers(min_value=1, max_value=300)),
            "endpoint": draw(st.text(min_size=5, max_size=50)),
            "retry_count": draw(st.integers(min_value=0, max_value=5))
        })
    
    elif error_type == "data_corruption":
        base_error.update({
            "corruption_type": draw(st.sampled_from(["partial", "complete", "checksum_mismatch"])),
            "affected_fields": draw(st.lists(
                st.text(min_size=3, max_size=20),
                min_size=1, max_size=10, unique=True
            )),
            "data_size": draw(st.integers(min_value=100, max_value=1000000))
        })
    
    elif error_type == "format_mismatch":
        base_error.update({
            "expected_format": draw(st.sampled_from(["json", "xml", "csv", "binary"])),
            "actual_format": draw(st.sampled_from(["json", "xml", "csv", "binary", "unknown"])),
            "conversion_possible": draw(st.booleans())
        })
    
    return base_error


# Validation strategies for integration testing
def valid_api_endpoint():
    """Strategy for valid API endpoints."""
    return st.sampled_from([
        "/api/v1/analyze", "/api/v1/classify", "/api/v1/extract",
        "/api/v1/status", "/api/v1/health", "/api/v1/metrics"
    ])


def valid_http_method():
    """Strategy for valid HTTP methods."""
    return st.sampled_from(["GET", "POST", "PUT", "DELETE", "PATCH"])


def valid_database_table():
    """Strategy for valid database table names."""
    return st.sampled_from([
        "analyses", "results", "users", "sessions", "videos",
        "keypoints", "features", "classifications", "reports"
    ])


def realistic_processing_time():
    """Strategy for realistic processing times."""
    return st.floats(min_value=0.001, max_value=300.0)


def realistic_data_size():
    """Strategy for realistic data sizes in bytes."""
    return st.integers(min_value=100, max_value=100_000_000)  # 100B to 100MB