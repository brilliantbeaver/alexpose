"""
Notebook setup utilities for suppressing MediaPipe/TensorFlow logs.

This module provides utilities specifically designed for Jupyter notebooks
to suppress C++ level logging from MediaPipe and TensorFlow.

CRITICAL: Import this module and call setup_notebook() BEFORE any other imports
that might trigger MediaPipe or TensorFlow initialization.

Usage:
    # First cell in notebook
    from ambient.utils.notebook_setup import setup_notebook
    setup_notebook()
    
    # Now import other modules
    from ambient.pose import MediaPipeEstimator
"""

import os
import sys
import warnings
import logging


def setup_notebook(verbose: bool = False):
    """
    Configure environment for clean notebook execution.
    
    This function:
    1. Sets all necessary environment variables to suppress C++ logging
    2. Configures Python logging to suppress verbose output
    3. Suppresses Python warnings
    4. Redirects C++ stderr to /dev/null (Unix) or NUL (Windows)
    
    Args:
        verbose: If True, print confirmation messages
    
    Returns:
        None
    
    Example:
        # First cell in your notebook
        from ambient.utils.notebook_setup import setup_notebook
        setup_notebook()
        
        # Now safe to import MediaPipe-dependent modules
        from ambient.pose import MediaPipeEstimator
    """
    # ========================================================================
    # Step 1: Environment Variables (must be set before any imports)
    # ========================================================================
    
    # TensorFlow settings
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
    os.environ['TF_CPP_VMODULE'] = 'inference_feedback_manager=0,gl_context=0'
    os.environ['TF_CPP_MIN_VLOG_LEVEL'] = '4'
    os.environ['TFLITE_LOG_LEVEL'] = '3'
    
    # Google logging (GLOG) - used by MediaPipe
    os.environ['GLOG_minloglevel'] = '4'  # 4 = NONE (higher than FATAL)
    os.environ['GLOG_logtostderr'] = '0'
    os.environ['GLOG_stderrthreshold'] = '4'
    os.environ['GLOG_v'] = '0'
    os.environ['GLOG_alsologtostderr'] = '0'
    
    # Abseil logging
    os.environ['ABSL_MIN_LOG_LEVEL'] = '3'
    
    # OpenCV settings
    os.environ['OPENCV_LOG_LEVEL'] = 'SILENT'
    
    # MediaPipe settings
    os.environ['MEDIAPIPE_DISABLE_GPU'] = '0'  # Keep GPU enabled for performance
    
    # Python warnings
    os.environ['PYTHONWARNINGS'] = 'ignore'
    
    # ========================================================================
    # Step 2: Python-level suppression
    # ========================================================================
    
    # Suppress all Python warnings
    warnings.filterwarnings('ignore')
    
    # Configure Python logging to be less verbose
    logging.getLogger().setLevel(logging.ERROR)
    
    # Suppress specific loggers that might be noisy
    for logger_name in ['tensorflow', 'mediapipe', 'absl', 'h5py']:
        logging.getLogger(logger_name).setLevel(logging.ERROR)
    
    # ========================================================================
    # Step 3: Jupyter-specific stderr redirection
    # ========================================================================
    
    # In Jupyter, we need to redirect stderr at the Python level
    # since file descriptor manipulation doesn't work reliably
    try:
        # Try to redirect stderr to devnull
        # This catches some (but not all) C++ output in Jupyter
        import io
        
        # Create a custom stderr that discards C++ log patterns
        class FilteredStderr:
            """Custom stderr that filters out known MediaPipe/TensorFlow log patterns."""
            
            def __init__(self, original_stderr):
                self.original_stderr = original_stderr
                self.patterns_to_filter = [
                    'GL version:',
                    'renderer:',
                    'Feedback manager requires',
                    'Created TensorFlow Lite',
                    'XNNPACK delegate',
                    'inference_feedback_manager',
                    'gl_context.cc',
                ]
            
            def write(self, text):
                # Filter out known log patterns
                if any(pattern in text for pattern in self.patterns_to_filter):
                    return
                # Pass through everything else
                self.original_stderr.write(text)
            
            def flush(self):
                self.original_stderr.flush()
            
            def fileno(self):
                return self.original_stderr.fileno()
        
        # Replace stderr with filtered version
        sys.stderr = FilteredStderr(sys.stderr)
        
    except Exception as e:
        if verbose:
            print(f"Warning: Could not set up stderr filtering: {e}")
    
    if verbose:
        print("✅ Notebook environment configured")
        print("   - TensorFlow logging suppressed")
        print("   - MediaPipe logging suppressed")
        print("   - Python warnings suppressed")
        print("   - C++ output filtered")


def restore_stderr():
    """
    Restore original stderr if it was replaced by setup_notebook().
    
    This is useful if you need to see error messages for debugging.
    
    Example:
        from ambient.utils.notebook_setup import restore_stderr
        restore_stderr()
    """
    if hasattr(sys.stderr, 'original_stderr'):
        sys.stderr = sys.stderr.original_stderr
        print("✅ Original stderr restored")
    else:
        print("ℹ️  stderr was not modified or already restored")


# Auto-configure on import if AMBIENT_AUTO_SETUP is set
if os.environ.get('AMBIENT_AUTO_SETUP', '').lower() in ('1', 'true', 'yes'):
    setup_notebook(verbose=False)
