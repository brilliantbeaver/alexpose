"""
Helper script to suppress MediaPipe warnings.

This script sets environment variables before importing MediaPipe to suppress
C++ warnings including the "feedback tensors" warning.

Usage:
    python scripts/suppress_mediapipe_warnings.py

Or set these in your shell before running Python:
    Windows (PowerShell):
        $env:TF_CPP_MIN_LOG_LEVEL="3"
        $env:GLOG_minloglevel="3"
        
    Linux/Mac:
        export TF_CPP_MIN_LOG_LEVEL=3
        export GLOG_minloglevel=3
"""

import os
import sys

# Set environment variables to suppress warnings
env_vars = {
    'TF_CPP_MIN_LOG_LEVEL': '3',
    'GLOG_minloglevel': '3',
    'GLOG_logtostderr': '0',
    'ABSL_MIN_LOG_LEVEL': '3',
    'OPENCV_LOG_LEVEL': 'ERROR',
    'MEDIAPIPE_DISABLE_GPU': '1',
    'TF_ENABLE_ONEDNN_OPTS': '0',
}

for key, value in env_vars.items():
    os.environ[key] = value
    print(f"Set {key}={value}")

print("\nEnvironment configured to suppress MediaPipe warnings.")
print("You can now import MediaPipe without warnings.")
print("\nExample:")
print("  from ambient.pose import MediaPipeEstimator")
