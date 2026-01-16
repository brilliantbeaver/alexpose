"""
Test script to verify the get_keypoints fix.

This script tests that get_keypoints() correctly returns a tuple of (keypoints, frame)
when called with a dictionary of sequences.
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

print("Testing get_keypoints fix...")
print("=" * 60)

# Test 1: Import the function
print("\n[TEST 1] Importing get_keypoints from ambient.utils.eval_keypoints...")
try:
    from ambient.utils.eval_keypoints import get_keypoints
    print("[OK] Import successful")
except Exception as e:
    print(f"[ERROR] Import failed: {e}")
    sys.exit(1)

# Test 2: Load GAVD data
print("\n[TEST 2] Loading GAVD data...")
try:
    from ambient.gavd import GAVDDataLoader
    
    loader = GAVDDataLoader()
    ONE_SEQUENCE_PATH = project_root / "data" / "GAVD_Clinical_Annotations_1.1.csv"
    
    if not ONE_SEQUENCE_PATH.exists():
        print(f"[SKIP] Test data not found: {ONE_SEQUENCE_PATH}")
        print("[INFO] This is expected if you don't have the GAVD dataset")
        sys.exit(0)
    
    df = loader.load_gavd_data(ONE_SEQUENCE_PATH)
    sequences = loader.organize_by_sequence(df)
    print(f"[OK] Loaded {len(sequences)} sequences")
except Exception as e:
    print(f"[ERROR] Failed to load data: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 3: Call get_keypoints with dict
print("\n[TEST 3] Calling get_keypoints with dictionary of sequences...")
try:
    keypoints, frame = get_keypoints(project_root, sequences, verbose=False)
    print(f"[OK] Function returned tuple: (keypoints, frame)")
    print(f"     - keypoints type: {type(keypoints)}")
    print(f"     - keypoints length: {len(keypoints) if keypoints else 0}")
    print(f"     - frame type: {type(frame)}")
    print(f"     - frame shape: {frame.shape if frame is not None else 'None'}")
except Exception as e:
    print(f"[ERROR] Function call failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 4: Verify return types
print("\n[TEST 4] Verifying return types...")
try:
    import numpy as np
    
    assert isinstance(keypoints, list), f"Expected list, got {type(keypoints)}"
    assert frame is None or isinstance(frame, np.ndarray), f"Expected np.ndarray or None, got {type(frame)}"
    print("[OK] Return types are correct")
except AssertionError as e:
    print(f"[ERROR] Type check failed: {e}")
    sys.exit(1)

print("\n" + "=" * 60)
print("[SUCCESS] All tests passed!")
print("The get_keypoints fix is working correctly.")
