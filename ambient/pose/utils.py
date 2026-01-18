"""
Pose Utilities Module

Common utility functions for pose estimation and processing.

Author: AlexPose Team
"""

import os
import sys
import contextlib


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
    import io
    
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
