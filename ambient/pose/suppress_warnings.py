"""
Warning suppression for MediaPipe and TensorFlow.

This module provides comprehensive warning suppression for MediaPipe, TensorFlow Lite,
and related C++ libraries. It uses multiple strategies:

1. Environment variables (set at module import time)
2. Python warnings module
3. File descriptor level redirection (most aggressive)

CRITICAL: This module sets environment variables at import time to suppress
C++ warnings. Import this module before importing MediaPipe for best results.

Usage:
    from ambient.pose.suppress_warnings import suppress_stderr_fd
    
    # Suppress C++ warnings during MediaPipe operations
    with suppress_stderr_fd():
        # Your MediaPipe code here
        pass
"""

import os
import sys
import io
import warnings
import contextlib
from typing import Optional

# ============================================================================
# Module-level initialization: Set environment variables BEFORE any imports
# ============================================================================
# These environment variables MUST be set before TensorFlow/MediaPipe imports
# to effectively suppress C++ level warnings.

# Suppress all Python warnings
warnings.filterwarnings('ignore')

# TensorFlow settings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'           # 0=all, 1=info, 2=warning, 3=error
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'          # Disable oneDNN custom operations
os.environ['TF_CPP_VMODULE'] = 'inference_feedback_manager=0'  # Suppress feedback manager
os.environ['TF_CPP_MIN_VLOG_LEVEL'] = '4'          # Minimum verbose log level
os.environ['TFLITE_LOG_LEVEL'] = '3'               # TensorFlow Lite log level

# Google logging (GLOG) - used by MediaPipe
os.environ['GLOG_minloglevel'] = '4'               # 0=INFO, 1=WARNING, 2=ERROR, 3=FATAL, 4=NONE
os.environ['GLOG_logtostderr'] = '0'               # Don't log to stderr
os.environ['GLOG_stderrthreshold'] = '4'           # Never log to stderr
os.environ['GLOG_v'] = '0'                         # Verbose level 0

# Abseil logging
os.environ['ABSL_MIN_LOG_LEVEL'] = '3'             # Abseil logging level

# OpenCV settings
os.environ['OPENCV_LOG_LEVEL'] = 'SILENT'          # OpenCV - use SILENT instead of ERROR

# MediaPipe settings
os.environ['MEDIAPIPE_DISABLE_GPU'] = '1'          # Disable GPU to reduce warnings

# CUDA settings
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'          # Disable CUDA warnings

# Python warnings
os.environ['PYTHONWARNINGS'] = 'ignore'

# Mark that suppression has been initialized
_SUPPRESSION_INITIALIZED = True


@contextlib.contextmanager
def suppress_stderr_fd():
    """
    Suppress stderr at the file descriptor level.
    
    This is the most aggressive form of stderr suppression and will catch
    C++ level output from TensorFlow Lite, MediaPipe, and GLOG that cannot
    be suppressed through Python's warnings module or sys.stderr redirection.
    
    Works on both Unix and Windows systems, and is compatible with Jupyter notebooks.
    
    Example:
        with suppress_stderr_fd():
            # C++ warnings are completely suppressed
            import mediapipe as mp
            landmarker = mp.solutions.pose.Pose()
    """
    # Check if we're in a Jupyter notebook or if stderr doesn't have a file descriptor
    try:
        stderr_fd = sys.stderr.fileno()
    except (AttributeError, io.UnsupportedOperation):
        # We're in Jupyter or another environment without real file descriptors
        # Fall back to Python-level suppression only
        old_stderr = sys.stderr
        try:
            sys.stderr = open(os.devnull, 'w')
            yield
        finally:
            sys.stderr.close()
            sys.stderr = old_stderr
        return
    
    # Normal file descriptor-based suppression for regular Python environments
    # Duplicate the stderr file descriptor to restore it later
    with os.fdopen(os.dup(stderr_fd), 'wb') as copied:
        # Flush any pending output
        sys.stderr.flush()
        
        try:
            # Determine the null device for this platform
            null_file = 'NUL' if sys.platform == 'win32' else '/dev/null'
            
            # Redirect stderr to the null device
            with open(null_file, 'wb') as devnull:
                os.dup2(devnull.fileno(), stderr_fd)
            
            yield
            
        finally:
            # Restore the original stderr
            sys.stderr.flush()
            os.dup2(copied.fileno(), stderr_fd)


@contextlib.contextmanager
def suppress_stdout_fd():
    """
    Suppress stdout at the file descriptor level.
    
    Similar to suppress_stderr_fd but for stdout.
    Compatible with Jupyter notebooks.
    
    Example:
        with suppress_stdout_fd():
            print("This won't be displayed")
    """
    # Check if we're in a Jupyter notebook or if stdout doesn't have a file descriptor
    try:
        stdout_fd = sys.stdout.fileno()
    except (AttributeError, io.UnsupportedOperation):
        # We're in Jupyter or another environment without real file descriptors
        # Fall back to Python-level suppression only
        old_stdout = sys.stdout
        try:
            sys.stdout = open(os.devnull, 'w')
            yield
        finally:
            sys.stdout.close()
            sys.stdout = old_stdout
        return
    
    # Normal file descriptor-based suppression for regular Python environments
    with os.fdopen(os.dup(stdout_fd), 'wb') as copied:
        sys.stdout.flush()
        
        try:
            null_file = 'NUL' if sys.platform == 'win32' else '/dev/null'
            
            with open(null_file, 'wb') as devnull:
                os.dup2(devnull.fileno(), stdout_fd)
            
            yield
            
        finally:
            sys.stdout.flush()
            os.dup2(copied.fileno(), stdout_fd)


@contextlib.contextmanager
def suppress_all_output_fd():
    """
    Suppress both stdout and stderr at the file descriptor level.
    
    This is the nuclear option - completely silences all output including
    C++ level logging from TensorFlow, MediaPipe, etc.
    
    Example:
        with suppress_all_output_fd():
            # All output is completely suppressed
            import mediapipe as mp
            print("This won't be displayed")
    """
    with suppress_stdout_fd():
        with suppress_stderr_fd():
            yield


class SuppressOutput:
    """
    Context manager and decorator for suppressing output at file descriptor level.
    
    This is more aggressive than Python-level suppression and will catch
    C++ output from libraries like TensorFlow and MediaPipe.
    
    Example as context manager:
        with SuppressOutput():
            # Your code here
            pass
    
    Example as decorator:
        @SuppressOutput()
        def my_function():
            # Your code here
            pass
    """
    
    def __init__(
        self, 
        suppress_stderr: bool = True, 
        suppress_stdout: bool = False,
        suppress_python_warnings: bool = True
    ):
        """
        Initialize the output suppressor.
        
        Args:
            suppress_stderr: Whether to suppress stderr (C++ warnings)
            suppress_stdout: Whether to suppress stdout (print statements)
            suppress_python_warnings: Whether to suppress Python warnings
        """
        self.suppress_stderr = suppress_stderr
        self.suppress_stdout = suppress_stdout
        self.suppress_python_warnings = suppress_python_warnings
        self._stderr_fd = None
        self._stdout_fd = None
        self._stderr_copy = None
        self._stdout_copy = None
        self._warning_filters = None
    
    def __enter__(self):
        """Enter the context manager."""
        # Suppress Python warnings
        if self.suppress_python_warnings:
            self._warning_filters = warnings.filters[:]
            warnings.filterwarnings('ignore')
        
        # Suppress stderr at fd level
        if self.suppress_stderr:
            try:
                self._stderr_fd = sys.stderr.fileno()
                self._stderr_copy = os.dup(self._stderr_fd)
                sys.stderr.flush()
                
                null_file = 'NUL' if sys.platform == 'win32' else '/dev/null'
                devnull = os.open(null_file, os.O_WRONLY)
                os.dup2(devnull, self._stderr_fd)
                os.close(devnull)
            except (AttributeError, io.UnsupportedOperation):
                # Jupyter environment - use Python-level suppression
                self._stderr_fd = None
                self._stderr_copy = None
        
        # Suppress stdout at fd level
        if self.suppress_stdout:
            try:
                self._stdout_fd = sys.stdout.fileno()
                self._stdout_copy = os.dup(self._stdout_fd)
                sys.stdout.flush()
                
                null_file = 'NUL' if sys.platform == 'win32' else '/dev/null'
                devnull = os.open(null_file, os.O_WRONLY)
                os.dup2(devnull, self._stdout_fd)
                os.close(devnull)
            except (AttributeError, io.UnsupportedOperation):
                # Jupyter environment - use Python-level suppression
                self._stdout_fd = None
                self._stdout_copy = None
        
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Exit the context manager."""
        # Restore stderr
        if self.suppress_stderr and self._stderr_copy is not None:
            sys.stderr.flush()
            os.dup2(self._stderr_copy, self._stderr_fd)
            os.close(self._stderr_copy)
        
        # Restore stdout
        if self.suppress_stdout and self._stdout_copy is not None:
            sys.stdout.flush()
            os.dup2(self._stdout_copy, self._stdout_fd)
            os.close(self._stdout_copy)
        
        # Restore Python warnings
        if self.suppress_python_warnings and self._warning_filters is not None:
            warnings.filters[:] = self._warning_filters
        
        return False
    
    def __call__(self, func):
        """Allow use as a decorator."""
        def wrapper(*args, **kwargs):
            with self:
                return func(*args, **kwargs)
        return wrapper


def suppress_all_warnings():
    """
    Globally suppress Python warnings.
    
    Note: This only suppresses Python-level warnings. For C++ warnings
    from MediaPipe/TensorFlow, use suppress_stderr_fd() context manager.
    
    Example:
        from ambient.pose.suppress_warnings import suppress_all_warnings
        
        suppress_all_warnings()
        
        # Python warnings are now suppressed
        import warnings
        warnings.warn("This won't be displayed")
    """
    warnings.filterwarnings('ignore')
    print("[OK] Python warnings suppressed")
    print("[BULB] For C++ warnings, use suppress_stderr_fd() context manager")

