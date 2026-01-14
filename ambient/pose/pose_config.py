"""
Pose estimation configuration and logging utilities.

This module provides utilities to suppress verbose C++ logs from pose estimation
libraries (MediaPipe, OpenPose, etc.) and their dependencies (TensorFlow Lite, etc.)
that cannot be controlled via Python logging.

Supports:
- MediaPipe (TensorFlow Lite backend)
- OpenPose (Caffe backend)
- Ultralytics (PyTorch backend)
- AlphaPose (PyTorch backend)
"""

import os
import sys
import contextlib
from typing import Generator


def configure_pose_environment() -> None:
    """
    Configure environment variables to suppress pose estimation library logs.
    
    This must be called BEFORE importing pose estimation libraries to be effective.
    Sets environment variables that control C++ level logging for various backends.
    
    Environment variables set:
        - TF_CPP_MIN_LOG_LEVEL: Controls TensorFlow C++ logging (MediaPipe, Ultralytics)
        - GLOG_minloglevel: Controls Google logging (MediaPipe, Caffe/OpenPose)
        - OPENCV_LOG_LEVEL: Controls OpenCV logging (used by multiple backends)
    
    Supported backends:
        - MediaPipe: Uses TensorFlow Lite (TF_CPP_MIN_LOG_LEVEL, GLOG_minloglevel)
        - OpenPose: Uses Caffe (GLOG_minloglevel)
        - Ultralytics: Uses PyTorch + TensorFlow (TF_CPP_MIN_LOG_LEVEL)
        - AlphaPose: Uses PyTorch (minimal C++ logging)
    
    Note:
        These warnings are typically safe to suppress:
        - "GL version" logs are informational about GPU initialization
        - "Feedback manager" warnings indicate TFLite features not needed
        - Caffe initialization logs are purely informational
    """
    # Suppress TensorFlow Lite logs (MediaPipe, Ultralytics)
    # 0=all, 1=filter INFO, 2=filter INFO+WARNING, 3=filter INFO+WARNING+ERROR
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    
    # Suppress Google logging (MediaPipe, OpenPose/Caffe)
    os.environ['GLOG_minloglevel'] = '3'
    os.environ['GLOG_logtostderr'] = '0'  # Disable GLOG stderr output
    
    # Suppress Abseil logging (used internally by TFLite)
    # This targets inference_feedback_manager.cc warnings
    os.environ['ABSL_MIN_LOG_LEVEL'] = '3'
    
    # Suppress OpenCV logs (used by all backends)
    os.environ['OPENCV_LOG_LEVEL'] = 'ERROR'


@contextlib.contextmanager
def suppress_stderr() -> Generator[None, None, None]:
    """
    Context manager to temporarily suppress stderr output at the OS level.
    
    This redirects file descriptor 2 (stderr) to /dev/null, which is necessary
    to suppress C++ level warnings from pose estimation libraries that bypass
    Python's sys.stderr.
    
    Use cases:
        - MediaPipe: Suppress TensorFlow Lite initialization warnings
        - OpenPose: Suppress Caffe initialization logs
        - Ultralytics: Suppress YOLO model loading messages
        - Any C++ library that writes directly to stderr
    
    Example:
        >>> with suppress_stderr():
        ...     estimator = PoseEstimator(model_path)
    
    Yields:
        None
    """
    # Save the original stderr file descriptor
    stderr_fd = sys.stderr.fileno()
    
    # Save a copy of the original stderr fd
    saved_stderr_fd = os.dup(stderr_fd)
    
    # Open /dev/null
    devnull_fd = os.open(os.devnull, os.O_WRONLY)
    
    try:
        # Flush Python's stderr buffer
        sys.stderr.flush()
        
        # Redirect stderr fd to /dev/null at the OS level
        os.dup2(devnull_fd, stderr_fd)
        
        yield
        
    finally:
        # Flush again before restoring
        sys.stderr.flush()
        
        # Restore the original stderr fd
        os.dup2(saved_stderr_fd, stderr_fd)
        
        # Close the file descriptors we opened
        os.close(devnull_fd)
        os.close(saved_stderr_fd)


@contextlib.contextmanager
def suppress_pose_logs() -> Generator[None, None, None]:
    """
    Context manager to completely suppress pose estimation library logs.
    
    Combines environment variable configuration with stderr redirection
    for maximum log suppression across all pose estimation backends.
    
    Example:
        >>> with suppress_pose_logs():
        ...     estimator = MediaPipeEstimator(model_path="model.task")
        ...     result = estimator.estimate_image_keypoints("image.jpg")
    
    Yields:
        None
    """
    # Save original environment
    old_tf_level = os.environ.get('TF_CPP_MIN_LOG_LEVEL')
    old_glog_level = os.environ.get('GLOG_minloglevel')
    old_opencv_level = os.environ.get('OPENCV_LOG_LEVEL')
    
    try:
        # Configure environment
        configure_pose_environment()
        
        # Also suppress stderr
        with suppress_stderr():
            yield
    finally:
        # Restore original environment
        if old_tf_level is not None:
            os.environ['TF_CPP_MIN_LOG_LEVEL'] = old_tf_level
        elif 'TF_CPP_MIN_LOG_LEVEL' in os.environ:
            del os.environ['TF_CPP_MIN_LOG_LEVEL']
            
        if old_glog_level is not None:
            os.environ['GLOG_minloglevel'] = old_glog_level
        elif 'GLOG_minloglevel' in os.environ:
            del os.environ['GLOG_minloglevel']
            
        if old_opencv_level is not None:
            os.environ['OPENCV_LOG_LEVEL'] = old_opencv_level
        elif 'OPENCV_LOG_LEVEL' in os.environ:
            del os.environ['OPENCV_LOG_LEVEL']


# Configure environment on module import
# This ensures the environment is set before any pose estimation library imports
configure_pose_environment()
