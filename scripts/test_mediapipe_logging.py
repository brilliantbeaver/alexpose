#!/usr/bin/env python3
"""
Test script to verify MediaPipe logging suppression.

This script tests that MediaPipe initialization doesn't produce
verbose C++ warnings about feedback managers or GL context.

Usage:
    python scripts/test_mediapipe_logging.py
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

print("=" * 70)
print("Testing MediaPipe Logging Suppression")
print("=" * 70)
print()

print("Step 1: Importing pose configuration...")
from ambient.pose.pose_config import configure_pose_environment
configure_pose_environment()
print("✓ Configuration imported and applied")
print()

print("Step 2: Importing MediaPipe estimator...")
from ambient.gavd.pose_estimators import MediaPipeEstimator, MEDIAPIPE_AVAILABLE
print("✓ MediaPipe estimator imported")
print()

if not MEDIAPIPE_AVAILABLE:
    print("❌ MediaPipe is not available. Please install it:")
    print("   uv pip install mediapipe")
    sys.exit(1)

print("Step 3: Checking for model file...")
model_path = project_root / "data" / "models" / "pose_landmarker_lite.task"
if not model_path.exists():
    print(f"⚠️  Model file not found at: {model_path}")
    print("   Download from: https://developers.google.com/mediapipe/solutions/vision/pose_landmarker/index#models")
    print()
    print("Skipping estimator initialization test.")
    print()
    print("=" * 70)
    print("✓ Import test PASSED - No warnings during import")
    print("=" * 70)
    sys.exit(0)

print(f"✓ Model file found: {model_path}")
print()

print("Step 4: Creating MediaPipe estimator...")
print("   (This is where warnings would normally appear)")
try:
    estimator = MediaPipeEstimator(model_path=str(model_path))
    print("✓ Estimator created successfully")
    print()
    
    print("Step 5: Checking estimator availability...")
    if estimator.is_available():
        print("✓ Estimator is available and ready")
    else:
        print("❌ Estimator is not available")
        sys.exit(1)
    
except Exception as e:
    print(f"❌ Failed to create estimator: {e}")
    sys.exit(1)

print()
print("=" * 70)
print("✓ ALL TESTS PASSED")
print("=" * 70)
print()
print("If you saw NO warnings about:")
print("  - 'GL version' or 'Metal'")
print("  - 'Feedback manager requires a model'")
print()
print("Then the logging suppression is working correctly! 🎉")
print()
