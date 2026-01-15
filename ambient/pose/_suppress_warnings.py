"""
Warning suppression for MediaPipe and other C++ libraries.

This module MUST be imported before any MediaPipe imports to suppress C++ warnings.
It sets environment variables at module import time.

CRITICAL: This module is automatically imported by ambient/__init__.py to ensure
warnings are suppressed throughout the entire application.
"""

import os
import sys
import warnings
import contextlib

# Suppress all Python warnings
warnings.filterwarnings('ignore')

# Configure environment to suppress C++ logs
# These MUST be set before any TensorFlow/MediaPipe imports
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'           # TensorFlow: 0=all, 1=info, 2=warning, 3=error
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'          # Disable oneDNN custom operations
os.environ['GLOG_minloglevel'] = '4'               # Google logging: 0=INFO, 1=WARNING, 2=ERROR, 3=FATAL, 4=NONE
os.environ['GLOG_logtostderr'] = '0'               # Don't log to stderr
os.environ['GLOG_stderrthreshold'] = '4'           # Never log to stderr
os.environ['GLOG_v'] = '0'                         # Verbose level 0
os.environ['ABSL_MIN_LOG_LEVEL'] = '3'             # Abseil logging
os.environ['OPENCV_LOG_LEVEL'] = 'SILENT'          # OpenCV - use SILENT instead of ERROR
os.environ['MEDIAPIPE_DISABLE_GPU'] = '1'          # Disable GPU to reduce warnings

# Additional TensorFlow settings
os.environ['TF_CPP_VMODULE'] = 'inference_feedback_manager=0'  # Specifically suppress feedback manager
os.environ['TF_CPP_MIN_VLOG_LEVEL'] = '4'          # Minimum verbose log level
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'          # Disable CUDA warnings

# Suppress TensorFlow Lite warnings
os.environ['TFLITE_LOG_LEVEL'] = '3'

# Additional aggressive suppression
os.environ['PYTHONWARNINGS'] = 'ignore'

# Mark that suppression has been initialized
_SUPPRESSION_INITIALIZED = True


@contextlib.contextmanager
def suppress_stderr_fd():
    """
    Context manager to suppress stderr at the file descriptor level.
    This is more aggressive than Python-level suppression.
    """
    try:
        # Save original stderr
        stderr_fd = sys.stderr.fileno()
        saved_stderr_fd = os.dup(stderr_fd)
        
        # Redirect stderr to devnull
        devnull_fd = os.open(os.devnull, os.O_WRONLY)
        os.dup2(devnull_fd, stderr_fd)
        
        try:
            yield
        finally:
            # Restore stderr
            os.dup2(saved_stderr_fd, stderr_fd)
            os.close(devnull_fd)
            os.close(saved_stderr_fd)
    except (AttributeError, OSError):
        # Fallback: just yield without suppression
        yield


# Apply stderr suppression globally for MediaPipe imports
# This will catch warnings that happen during module initialization
_original_stderr = sys.stderr
try:
    # Temporarily redirect stderr during initial setup
    sys.stderr = open(os.devnull, 'w')
except Exception:
    pass  # If this fails, continue anyway
