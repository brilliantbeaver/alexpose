# Jupyter Compatibility Fix - January 15, 2026

## Problem

Jupyter notebooks were failing with `UnsupportedOperation` error when importing ambient modules:

```python
UnsupportedOperation: fileno
  File ambient/pose/suppress_warnings.py:88, in suppress_stderr_fd()
    stderr_fd = sys.stderr.fileno()
  File ipykernel/iostream.py:372, in OutStream.fileno(self)
    raise io.UnsupportedOperation(msg)
```

## Root Cause Analysis

### Why This Happened

1. **Jupyter's Custom IO Streams**: Jupyter notebooks use custom IO streams (`ipykernel.iostream.OutStream`) that don't have real file descriptors
2. **File Descriptor Suppression**: Our `suppress_stderr_fd()` function tried to use `sys.stderr.fileno()` to suppress C++ warnings from MediaPipe/TensorFlow
3. **Incompatibility**: Jupyter's streams raise `io.UnsupportedOperation` when `fileno()` is called

### Technical Details

**Normal Python Environment:**
- `sys.stderr` → real file descriptor (integer like 2)
- Can use `os.dup2()` to redirect at OS level
- Catches C++ warnings from MediaPipe/TensorFlow

**Jupyter Environment:**
- `sys.stderr` → `ipykernel.iostream.OutStream` (custom object)
- No real file descriptor
- `fileno()` raises `io.UnsupportedOperation`

## Solution Implemented

### 1. Made `suppress_stderr_fd()` Jupyter-Compatible

Added fallback logic to detect Jupyter environments:

```python
@contextlib.contextmanager
def suppress_stderr_fd():
    # Try to get file descriptor
    try:
        stderr_fd = sys.stderr.fileno()
    except (AttributeError, io.UnsupportedOperation):
        # Jupyter environment - use Python-level suppression
        old_stderr = sys.stderr
        try:
            sys.stderr = open(os.devnull, 'w')
            yield
        finally:
            sys.stderr.close()
            sys.stderr = old_stderr
        return
    
    # Normal file descriptor-based suppression for regular Python
    # ... (existing code)
```

### 2. Applied Same Fix to All Suppression Functions

- `suppress_stderr_fd()` - Fixed ✓
- `suppress_stdout_fd()` - Fixed ✓
- `SuppressOutput.__enter__()` - Fixed ✓

### 3. Added `io` Module Import

```python
import io  # Required for io.UnsupportedOperation
```

## Testing

### Created Comprehensive Test Suite

**File:** `scripts/test_jupyter_imports.py`

Tests both environments:
1. **Normal Python environment** - Uses real file descriptors
2. **Mock Jupyter environment** - Simulates Jupyter's custom IO streams

**Test Results:**
```
Normal environment: [PASS]
Jupyter environment: [PASS]
[SUCCESS] All tests passed!
```

### Verified Imports

All imports now work in both environments:
```python
from ambient.gavd import GaitDataProcessor, GAVDDataLoader, PoseDataConverter
from ambient.pose.pose_estimators import OpenPoseEstimator
from ambient.utils.csv_parser import parse_csv_with_dicts
```

## Impact

### What Changed

**Before:**
- ❌ Imports failed in Jupyter notebooks
- ❌ `UnsupportedOperation` error on import
- ❌ Notebooks unusable

**After:**
- ✅ Imports work in Jupyter notebooks
- ✅ Imports work in normal Python
- ✅ Graceful fallback for environments without file descriptors
- ✅ C++ warnings still suppressed (where possible)

### Suppression Behavior

**Normal Python:**
- Full file descriptor-level suppression
- Catches C++ warnings from MediaPipe/TensorFlow
- Most aggressive suppression

**Jupyter Notebooks:**
- Python-level suppression only
- May see some C++ warnings (unavoidable in Jupyter)
- Still suppresses Python warnings

## Files Modified

1. **ambient/pose/suppress_warnings.py**
   - Added `io` import
   - Made `suppress_stderr_fd()` Jupyter-compatible
   - Made `suppress_stdout_fd()` Jupyter-compatible
   - Made `SuppressOutput.__enter__()` Jupyter-compatible

2. **scripts/test_jupyter_imports.py** (new)
   - Comprehensive test suite
   - Tests both normal and Jupyter environments
   - Mock Jupyter IO streams for testing

3. **notes/JUPYTER_COMPATIBILITY_FIX.md** (this file)
   - Complete documentation of the fix

## Usage in Jupyter Notebooks

### Standard Import Pattern

```python
# Cell 1: Setup
%load_ext autoreload
%autoreload 2

import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Cell 2: Imports (now work!)
from ambient.gavd import GaitDataProcessor, GAVDDataLoader
from ambient.pose import MediaPipeEstimator, get_pose_estimator
from ambient.utils.csv_parser import parse_csv_with_dicts

print("[OK] All imports successful")
```

### Expected Behavior

- Imports complete successfully
- May see "MediaPipe detects 33 landmarks" message (normal)
- Some C++ warnings may appear (unavoidable in Jupyter)
- Python warnings are suppressed

## Prevention

### Design Principles Applied

1. **Environment Detection**: Always check if file descriptors are available
2. **Graceful Fallback**: Provide alternative suppression methods
3. **Exception Handling**: Catch `AttributeError` and `io.UnsupportedOperation`
4. **Cross-Platform**: Works on Windows, Linux, macOS, and Jupyter

### Code Pattern for File Descriptor Operations

```python
try:
    fd = sys.stderr.fileno()
    # Use file descriptor operations
except (AttributeError, io.UnsupportedOperation):
    # Fallback for Jupyter/environments without FDs
    pass
```

## Related Issues

- **Previous Fix**: Stale Python cache (cleared with `scripts/clear_python_cache.py`)
- **Previous Fix**: Circular import (renamed `logging.py` to `log_config.py`)
- **Previous Fix**: Unicode errors (removed emoji characters)

## Verification Commands

```bash
# Test in normal Python
python -c "from ambient.gavd import GaitDataProcessor; print('[OK]')"

# Test with mock Jupyter environment
python scripts/test_jupyter_imports.py

# Test in actual Jupyter notebook
jupyter notebook notebooks/explore3\ -\ extract\ features.ipynb
```

## Technical Notes

### Why File Descriptors Matter

File descriptors allow OS-level redirection of output, which is the only way to suppress C++ warnings from libraries like MediaPipe and TensorFlow that write directly to stderr at the C++ level.

### Jupyter's Architecture

Jupyter uses ZeroMQ for communication between the kernel and frontend. The kernel's stdout/stderr are captured and sent as messages, not written to real file descriptors. This is why `fileno()` doesn't work.

### Trade-offs

**Normal Python:**
- ✅ Complete suppression of C++ warnings
- ✅ Clean output

**Jupyter:**
- ⚠️ Some C++ warnings may appear
- ✅ Python warnings suppressed
- ✅ Imports work correctly

This is an acceptable trade-off since Jupyter's architecture makes complete C++ warning suppression impossible without kernel modifications.

## Success Criteria

- ✅ Imports work in Jupyter notebooks
- ✅ Imports work in normal Python
- ✅ No `UnsupportedOperation` errors
- ✅ Graceful fallback for environments without file descriptors
- ✅ Comprehensive test coverage
- ✅ Documentation complete

## Conclusion

The fix successfully makes the ambient package compatible with Jupyter notebooks while maintaining full functionality in normal Python environments. The solution is robust, well-tested, and properly documented.


---

## Update: Additional Fix in keypoints.py

**Date**: January 15, 2026

### Additional Location Fixed

The same Jupyter compatibility issue was found in `ambient/pose/keypoints.py`. This file had its own copy of `suppress_stderr_fd()` that also needed the Jupyter compatibility fix.

### Files Fixed (Complete List)

1. ✅ `ambient/pose/suppress_warnings.py` - Fixed in Task 5
2. ✅ `ambient/pose/keypoints.py` - Fixed in this update

### Why Two Locations?

The `keypoints.py` module has its own implementation of `suppress_stderr_fd()` for local use within the module. Both implementations have now been updated with Jupyter compatibility.

### Verification

After this fix, the notebook can successfully call:
```python
from ambient.utils.eval_keypoints import get_keypoints
keypoints, frame = get_keypoints(project_root, sequences)
```

All Jupyter compatibility issues are now resolved!
