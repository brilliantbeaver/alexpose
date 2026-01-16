"""
Test script to simulate Jupyter notebook imports.
This simulates the Jupyter environment where sys.stderr is a special object.
"""
import sys
import io

# Simulate Jupyter's stderr (which doesn't have fileno())
class JupyterStderr:
    def write(self, text):
        pass
    
    def flush(self):
        pass
    
    def fileno(self):
        raise io.UnsupportedOperation("fileno")

# Replace stderr with Jupyter-like object
original_stderr = sys.stderr
sys.stderr = JupyterStderr()

try:
    print("Testing imports in Jupyter-like environment...")
    
    # Test the imports from the notebook
    from ambient.utils.csv_parser import parse_csv_with_dicts
    print("✓ csv_parser imported")
    
    from ambient.gavd import GaitDataProcessor, GAVDDataLoader, PoseDataConverter
    print("✓ gavd imports successful")
    
    from ambient.pose import OpenPoseEstimator
    print("✓ OpenPoseEstimator imported")
    
    # Test direct imports
    from ambient.pose import MediaPipeEstimator, PoseEstimator
    print("✓ MediaPipeEstimator imported directly")
    
    print("\n✅ All imports successful in Jupyter-like environment!")
    
except Exception as e:
    print(f"\n❌ Import failed: {e}")
    import traceback
    traceback.print_exc()
    
finally:
    # Restore original stderr
    sys.stderr = original_stderr
