#!/usr/bin/env python3
"""
Test script to verify imports work in Jupyter-like environments.

This script simulates Jupyter's IO redirection to test that our
suppress_warnings module handles it correctly.
"""

import sys
import io


class MockJupyterStream:
    """Mock Jupyter's custom IO stream that doesn't have fileno()."""
    
    def __init__(self, original_stream):
        self.original_stream = original_stream
    
    def write(self, text):
        return self.original_stream.write(text)
    
    def flush(self):
        return self.original_stream.flush()
    
    def fileno(self):
        """Raise UnsupportedOperation like Jupyter does."""
        raise io.UnsupportedOperation("fileno")


def test_imports_with_mock_jupyter():
    """Test imports with Jupyter-like IO streams."""
    print("=" * 60)
    print("Testing imports with Jupyter-like IO streams")
    print("=" * 60)
    
    # Save original streams
    original_stdout = sys.stdout
    original_stderr = sys.stderr
    
    try:
        # Replace with mock Jupyter streams
        sys.stdout = MockJupyterStream(original_stdout)
        sys.stderr = MockJupyterStream(original_stderr)
        
        print("\n[TEST 1] Testing suppress_warnings import...")
        from ambient.pose.suppress_warnings import suppress_stderr_fd
        print("[OK] suppress_warnings imported successfully")
        
        print("\n[TEST 2] Testing suppress_stderr_fd context manager...")
        with suppress_stderr_fd():
            print("[OK] suppress_stderr_fd works in Jupyter environment")
        
        print("\n[TEST 3] Testing MediaPipe import...")
        from ambient.pose.mediapipe_estimator import MediaPipeEstimator
        print("[OK] MediaPipeEstimator imported successfully")
        
        print("\n[TEST 4] Testing GAVD imports...")
        from ambient.gavd import GaitDataProcessor, GAVDDataLoader, PoseDataConverter
        print("[OK] GAVD modules imported successfully")
        
        print("\n[TEST 5] Testing pose estimators import...")
        from ambient.pose.pose_estimators import OpenPoseEstimator
        print("[OK] OpenPoseEstimator imported successfully")
        
        print("\n" + "=" * 60)
        print("[SUCCESS] All imports work in Jupyter-like environment!")
        print("=" * 60)
        
    except Exception as e:
        print(f"\n[ERROR] Import failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    finally:
        # Restore original streams
        sys.stdout = original_stdout
        sys.stderr = original_stderr
    
    return True


def test_imports_normal():
    """Test imports in normal Python environment."""
    print("\n" + "=" * 60)
    print("Testing imports in normal Python environment")
    print("=" * 60)
    
    try:
        print("\n[TEST 1] Testing all imports...")
        from ambient.gavd import GaitDataProcessor, GAVDDataLoader, PoseDataConverter
        from ambient.pose.pose_estimators import OpenPoseEstimator
        from ambient.utils.csv_parser import parse_csv_with_dicts
        
        print("[OK] All imports successful in normal environment")
        print("=" * 60)
        
    except Exception as e:
        print(f"\n[ERROR] Import failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    return True


if __name__ == "__main__":
    print("\nJupyter Import Compatibility Test")
    print("=" * 60)
    
    # Test normal environment first
    success_normal = test_imports_normal()
    
    # Test Jupyter-like environment
    success_jupyter = test_imports_with_mock_jupyter()
    
    print("\n" + "=" * 60)
    print("FINAL RESULTS")
    print("=" * 60)
    print(f"Normal environment: {'[PASS]' if success_normal else '[FAIL]'}")
    print(f"Jupyter environment: {'[PASS]' if success_jupyter else '[FAIL]'}")
    
    if success_normal and success_jupyter:
        print("\n[SUCCESS] All tests passed!")
        sys.exit(0)
    else:
        print("\n[FAILURE] Some tests failed")
        sys.exit(1)
