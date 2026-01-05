#!/usr/bin/env python3
"""
Tests for the new logging utilities.

This module tests the logging utilities added to ambient.utils.logging
to ensure they work correctly and provide consistent structured logging.
"""

import pytest
import tempfile
import time
import asyncio
from pathlib import Path
from unittest.mock import patch, MagicMock

from ambient.utils.logging import (
    setup_logging,
    create_component_logger,
    log_function_execution,
    log_async_function_execution,
    BatchOperationLogger,
    create_operation_logger,
    create_progress_logger,
    log_video_processing,
    log_pose_estimation,
    log_gait_analysis,
    log_classification_result,
    log_model_operation,
    log_resource_usage
)

# Import loguru to clean up handlers after tests
from loguru import logger


def cleanup_loguru_handlers():
    """Clean up loguru handlers to prevent file locking issues."""
    try:
        # Remove all handlers except the default one
        logger.remove()
        # Add back a simple console handler for basic logging
        logger.add(
            lambda msg: None,  # Null handler to prevent output during tests
            level="DEBUG"
        )
    except Exception:
        pass  # Ignore cleanup errors


class TestComponentLogger:
    """Test component logger creation and usage."""
    
    def test_create_component_logger(self):
        """Test creating a component logger."""
        logger = create_component_logger("test_component")
        assert logger is not None
        assert logger.component == "test_component"
    
    def test_component_logger_methods(self):
        """Test component logger methods."""
        logger = create_component_logger("test_component")
        
        # Test that methods exist and can be called
        logger.debug("Debug message", operation="test")
        logger.info("Info message", operation="test")
        logger.warning("Warning message", operation="test")
        logger.error("Error message", operation="test")


class TestFunctionLogging:
    """Test function execution logging decorators."""
    
    def test_function_execution_decorator(self):
        """Test the function execution logging decorator."""
        
        @log_function_execution(component="test", log_args=True, log_duration=True)
        def test_function(arg1: str, arg2: int = 10):
            time.sleep(0.01)  # Small delay to test duration logging
            return f"result_{arg1}_{arg2}"
        
        result = test_function("test", 20)
        assert result == "result_test_20"
    
    def test_function_execution_with_error(self):
        """Test function execution logging with error handling."""
        
        @log_function_execution(component="test", log_args=True)
        def failing_function(should_fail: bool = True):
            if should_fail:
                raise ValueError("Test error")
            return "success"
        
        with pytest.raises(ValueError, match="Test error"):
            failing_function(True)


class TestBatchOperationLogger:
    """Test batch operation logging."""
    
    def test_batch_operation_logger_creation(self):
        """Test creating a batch operation logger."""
        batch_logger = BatchOperationLogger("test_component", "test_operation", total_items=10)
        assert batch_logger.component == "test_component"
        assert batch_logger.operation == "test_operation"
        assert batch_logger.total_items == 10
    
    def test_batch_operation_logging(self):
        """Test batch operation logging workflow."""
        batch_logger = BatchOperationLogger("test_component", "test_batch", total_items=5)
        batch_logger.start_batch(test_context="test_value")
        
        # Process items with some successes and failures
        for i in range(5):
            try:
                with batch_logger.log_item(i, {"item_id": f"item_{i}"}):
                    if i == 2:  # Simulate failure
                        raise ValueError(f"Test error for item {i}")
                    time.sleep(0.001)  # Small delay
            except ValueError:
                pass  # Expected error
        
        batch_logger.log_completion()
        
        # Verify statistics
        assert batch_logger.processed_count == 5
        assert batch_logger.success_count == 4
        assert batch_logger.error_count == 1
        assert len(batch_logger.errors) == 1


class TestOperationLogger:
    """Test operation-specific logging."""
    
    def test_create_operation_logger(self):
        """Test creating an operation logger."""
        op_logger = create_operation_logger("test_component", "test_operation")
        assert op_logger is not None
        
        # Test that methods exist and can be called
        op_logger.info("Test message", test_param="value")
        op_logger.debug("Debug message", debug_info=True)
        op_logger.warning("Warning message", warning_level="low")
        op_logger.error("Error message", error_code=500)


class TestProgressLogger:
    """Test progress logging."""
    
    def test_create_progress_logger(self):
        """Test creating a progress logger."""
        progress = create_progress_logger("test_component", "test_operation", total_items=100)
        assert progress is not None
        assert progress.component == "test_component"
        assert progress.operation == "test_operation"
        assert progress.total_items == 100
    
    def test_progress_logging_workflow(self):
        """Test progress logging workflow."""
        progress = create_progress_logger("test_component", "test_progress", total_items=10)
        
        progress.start(test_context="test_value")
        
        for i in range(10):
            progress.update(i + 1, {"item_number": i + 1})
            time.sleep(0.001)  # Small delay
        
        progress.complete(final_status="success")


class TestSpecializedLogging:
    """Test specialized logging functions."""
    
    def test_log_video_processing(self):
        """Test video processing logging."""
        log_video_processing(
            "video_processor", 
            "test_video.mp4", 
            "extract_frames",
            frame_count=100,
            duration=5.0,
            resolution="1920x1080"
        )
    
    def test_log_pose_estimation(self):
        """Test pose estimation logging."""
        log_pose_estimation(
            "pose_estimator",
            "mediapipe",
            "estimate_poses",
            frames_processed=100,
            keypoints_per_frame=17,
            avg_confidence=0.85
        )
    
    def test_log_gait_analysis(self):
        """Test gait analysis logging."""
        log_gait_analysis(
            "gait_analyzer",
            "temporal_analysis",
            "extract_features",
            features_extracted=15,
            gait_cycles_detected=3,
            symmetry_index=0.92
        )
    
    def test_log_classification_result(self):
        """Test classification result logging."""
        result = {
            "prediction": "abnormal",
            "confidence": 0.78,
            "condition": "mild_limp"
        }
        log_classification_result(
            "classifier",
            "binary_classifier",
            result,
            model_version="v1.0"
        )
    
    def test_log_model_operation(self):
        """Test model operation logging."""
        log_model_operation(
            "pose_estimator",
            "mediapipe_pose",
            "load",
            model_size_mb=25.0,
            load_time=2.3,
            device="cpu"
        )
    
    def test_log_resource_usage(self):
        """Test resource usage logging."""
        log_resource_usage(
            "video_processor",
            "extract_frames",
            memory_mb=512,
            cpu_percent=85,
            processing_time=30.5
        )


class TestLoggingSetup:
    """Test logging setup and configuration."""
    
    def test_setup_logging_with_temp_dir(self):
        """Test setting up logging with a temporary directory."""
        with tempfile.TemporaryDirectory() as temp_dir:
            try:
                setup_logging(
                    log_level="DEBUG",
                    log_dir=temp_dir,
                    format_type="structured",
                    environment="testing"
                )
                
                # Verify log directory was created
                log_path = Path(temp_dir)
                assert log_path.exists()
                
                # Test that logging works
                test_logger = create_component_logger("test")
                test_logger.info("Test message", operation="test_setup")
                
            finally:
                # Clean up handlers to release file locks
                cleanup_loguru_handlers()


def test_logging_utilities_integration():
    """Integration test for logging utilities."""
    with tempfile.TemporaryDirectory() as temp_dir:
        try:
            # Setup logging
            setup_logging(
                log_level="DEBUG",
                log_dir=temp_dir,
                format_type="structured",
                environment="testing"
            )
            
            # Test various logging utilities together
            test_logger = create_component_logger("integration_test")
            test_logger.info("Starting integration test", operation="integration_test")
            
            # Test function decorator
            @log_function_execution(component="integration_test", log_duration=True)
            def test_function():
                return "integration_success"
            
            result = test_function()
            assert result == "integration_success"
            
            # Test batch logging
            batch_logger = BatchOperationLogger("integration_test", "test_batch", total_items=3)
            batch_logger.start_batch()
            
            for i in range(3):
                with batch_logger.log_item(i):
                    time.sleep(0.001)
            
            batch_logger.log_completion()
            
            # Test specialized logging
            log_video_processing("integration_test", "test.mp4", "process", frame_count=10)
            
            test_logger.info("Integration test completed", operation="integration_test")
            
        finally:
            # Clean up handlers to release file locks
            cleanup_loguru_handlers()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])