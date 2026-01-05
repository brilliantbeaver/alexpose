#!/usr/bin/env python3
"""
Example demonstrating the structured logging system.

This example shows how to use the AlexPose structured logging utilities
for consistent logging across components.
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from ambient.utils.logging import (
    setup_logging,
    create_component_logger,
    LoggingContext,
    log_performance,
    log_error_with_context,
    log_system_event,
    log_data_operation,
    configure_development_logging
)


def main():
    """Demonstrate structured logging features."""
    
    # Setup structured logging
    configure_development_logging()
    
    # Create component-specific loggers
    video_logger = create_component_logger("video_processor")
    pose_logger = create_component_logger("pose_estimator")
    gait_logger = create_component_logger("gait_analyzer")
    
    # Basic structured logging
    video_logger.info("Starting video processing", operation="process_video",
                     filename="sample.mp4", duration=30.5, resolution="1920x1080")
    
    pose_logger.debug("Loading pose estimation model", operation="model_init",
                     model_type="mediapipe", confidence_threshold=0.7)
    
    # Using logging context for operations
    with LoggingContext("gait_analysis", component="gait_analyzer", 
                       video_id="sample_123", subject_id="patient_456") as ctx:
        
        ctx.log_progress("Extracting frames", progress=25.0, frames_extracted=75)
        ctx.log_progress("Running pose estimation", progress=50.0, poses_detected=68)
        ctx.log_progress("Analyzing gait patterns", progress=75.0, cycles_detected=3)
        ctx.log_progress("Generating report", progress=100.0, analysis_complete=True)
    
    # Performance logging
    log_performance("frame_extraction", 2.45, "video_processor", {
        "frame_count": 915,
        "fps": 30,
        "extraction_method": "ffmpeg",
        "quality": "high"
    })
    
    # Data operation logging
    log_data_operation("process", "pose_keypoints", 915, "pose_estimator", {
        "batch_size": 32,
        "model_accuracy": 0.94,
        "processing_time": 12.3
    })
    
    # System event logging
    log_system_event("model_loaded", "Pose estimation model loaded successfully",
                    "pose_estimator", {
                        "model_name": "mediapipe_pose_v1",
                        "model_size": "23.4MB",
                        "load_time": 1.2
                    })
    
    # Error logging with context
    try:
        # Simulate an error
        raise ValueError("Invalid video format: unsupported codec")
    except ValueError as e:
        log_error_with_context(e, "video_processor", {
            "filename": "corrupted_video.avi",
            "codec": "unknown",
            "file_size": "45.2MB",
            "attempted_operation": "decode_video"
        }, "video_decoding")
    
    # Component-specific error logging
    try:
        # Simulate another error
        raise RuntimeError("Pose estimation failed: insufficient keypoints detected")
    except RuntimeError as e:
        pose_logger.error("Pose estimation failed", operation="estimate_pose",
                         error=e, frame_number=123, keypoints_detected=8,
                         minimum_required=17)
    
    # Warning with context
    gait_logger.warning("Low confidence gait analysis", operation="analyze_gait",
                       confidence_score=0.45, threshold=0.7, 
                       recommendation="increase_video_quality")
    
    print("\n✓ Structured logging example completed!")
    print("Check the logs/ directory for generated log files:")
    print("  - alexpose_YYYY-MM-DD.log (main application log)")
    print("  - errors_YYYY-MM-DD.log (error-specific log)")
    print("  - performance_YYYY-MM-DD.log (performance metrics)")


if __name__ == "__main__":
    main()