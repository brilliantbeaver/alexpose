#!/usr/bin/env python3
"""
Example demonstrating the new logging utilities for consistent usage across components.

This example shows how to use the various logging utilities provided by the
ambient.utils.logging module for consistent structured logging across the
AlexPose system.

Usage:
    python examples/logging_utilities_example.py
"""

import asyncio
import time
import sys
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Import the new logging utilities
from ambient.utils.logging import (
    setup_logging,
    create_component_logger,
    log_function_execution,
    log_async_function_execution,
    BatchOperationLogger,
    APIRequestLogger,
    LoggingContext,
    create_operation_logger,
    create_progress_logger,
    log_video_processing,
    log_pose_estimation,
    log_gait_analysis,
    log_classification_result,
    log_file_operation,
    log_model_operation,
    log_resource_usage,
    log_configuration_change
)


def main():
    """Main example function demonstrating logging utilities."""
    
    # 1. Setup logging for the example
    print("Setting up logging...")
    setup_logging(
        log_level="DEBUG",
        log_dir="logs/examples",
        format_type="structured",
        environment="development"
    )
    
    # 2. Create component loggers
    print("\n=== Component Logger Examples ===")
    video_logger = create_component_logger("video_processor")
    pose_logger = create_component_logger("pose_estimator")
    api_logger = create_component_logger("api_handler")
    
    video_logger.info("Video processor initialized", operation="init", 
                     supported_formats=["mp4", "avi", "mov"])
    pose_logger.debug("Loading pose estimation model", operation="model_load",
                     model_type="mediapipe", confidence_threshold=0.7)
    
    # 3. Function execution logging with decorators
    print("\n=== Function Execution Logging ===")
    
    @log_function_execution(component="video_processor", log_args=True, log_duration=True)
    def process_video_frames(video_path: str, frame_rate: float = 30.0):
        """Example function with automatic logging."""
        time.sleep(0.5)  # Simulate processing
        return {"frames_extracted": 150, "duration": 5.0}
    
    @log_async_function_execution(component="api_handler", log_result=True)
    async def upload_video_async(filename: str, size_mb: float):
        """Example async function with automatic logging."""
        await asyncio.sleep(0.3)  # Simulate async processing
        return {"upload_id": "12345", "status": "completed"}
    
    # Test the decorated functions
    result = process_video_frames("sample_video.mp4", 25.0)
    print(f"Video processing result: {result}")
    
    # Run async example
    async def run_async_example():
        result = await upload_video_async("test_video.mp4", 45.2)
        print(f"Async upload result: {result}")
    
    asyncio.run(run_async_example())
    
    # 4. Batch operation logging
    print("\n=== Batch Operation Logging ===")
    
    batch_logger = BatchOperationLogger("video_processor", "process_video_batch", total_items=10)
    batch_logger.start_batch(batch_id="batch_001", source="user_upload")
    
    for i in range(10):
        try:
            with batch_logger.log_item(i, {"video_id": f"video_{i:03d}"}):
                time.sleep(0.1)  # Simulate processing
                if i == 7:  # Simulate an error
                    raise ValueError(f"Processing failed for video_{i:03d}")
        except ValueError:
            # Expected error for demonstration
            pass
    
    batch_logger.log_completion(total_size_mb=500.0)
    
    # 5. Operation-specific logging
    print("\n=== Operation-Specific Logging ===")
    
    op_logger = create_operation_logger("gait_analyzer", "analyze_gait_sequence")
    op_logger.info("Starting gait analysis", sequence_length=100, subject_id="S001")
    op_logger.debug("Extracting temporal features", feature_count=25)
    op_logger.warning("Low confidence detected", confidence=0.45, threshold=0.7)
    
    # 6. Progress logging
    print("\n=== Progress Logging ===")
    
    progress = create_progress_logger("pose_estimator", "estimate_poses", total_items=50)
    progress.start(model_type="mediapipe", batch_size=5)
    
    for i in range(50):
        time.sleep(0.02)  # Simulate processing
        progress.update(i + 1, {"frame_number": i, "keypoints_detected": 17})
    
    progress.complete(total_keypoints=850, avg_confidence=0.82)
    
    # 7. Specialized logging functions
    print("\n=== Specialized Logging Functions ===")
    
    # Video processing logging
    log_video_processing("video_handler", "sample_video.mp4", "extract_frames",
                        frame_count=120, duration=4.0, resolution="1920x1080")
    
    # Pose estimation logging
    log_pose_estimation("pose_estimator", "mediapipe", "batch_estimate",
                       frames_processed=120, keypoints_per_frame=17, avg_confidence=0.85)
    
    # Gait analysis logging
    log_gait_analysis("gait_analyzer", "temporal_analysis", "extract_features",
                     features_extracted=15, gait_cycles_detected=3, symmetry_index=0.92)
    
    # Classification result logging
    classification_result = {
        "prediction": "abnormal",
        "confidence": 0.78,
        "condition": "mild_limp",
        "severity": "low"
    }
    log_classification_result("classifier", "binary_classifier", classification_result,
                             model_version="v1.2", processing_time=0.15)
    
    # File operation logging
    log_file_operation("storage_manager", "save", "data/analysis/result_001.json",
                      file_size_mb=0.5, compression="none")
    
    # Model operation logging
    log_model_operation("pose_estimator", "mediapipe_pose", "load",
                       model_size_mb=25.0, load_time=2.3, device="cpu")
    
    # Resource usage logging
    log_resource_usage("video_processor", "extract_frames",
                      memory_mb=512, cpu_percent=85, gpu_utilization=45,
                      processing_time=30.5, frames_per_second=4.0)
    
    # Configuration change logging
    log_configuration_change("pose_estimator", "confidence_threshold", 0.5, 0.7,
                            reason="improved_accuracy_requirements")
    
    # 8. Context manager logging
    print("\n=== Context Manager Logging ===")
    
    with LoggingContext("full_video_analysis", "video_processor", 
                       video_path="sample_video.mp4", analysis_type="comprehensive") as ctx:
        
        ctx.log_progress("Loading video", progress=10, frames_loaded=0)
        time.sleep(0.2)
        
        ctx.log_progress("Extracting frames", progress=30, frames_extracted=50)
        time.sleep(0.3)
        
        ctx.log_progress("Estimating poses", progress=60, poses_estimated=50)
        time.sleep(0.2)
        
        ctx.log_progress("Analyzing gait", progress=90, gait_cycles=3)
        time.sleep(0.1)
        
        ctx.log_progress("Generating report", progress=100, report_generated=True)
    
    print("\n=== Logging Utilities Example Completed ===")
    print("Check the logs/examples/ directory for generated log files.")
    print("Log files include structured data for easy parsing and analysis.")


if __name__ == "__main__":
    main()