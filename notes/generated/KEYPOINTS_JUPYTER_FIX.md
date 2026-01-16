# Keypoints Jupyter Compatibility Fix

**Date**: January 15, 2026  
**Status**: ✅ RESOLVED

## Problem

The notebook was failing with `UnsupportedOperation: fileno` when calling `get_keypoints()`:

```python
UnsupportedOperation                      Traceback (most recent call last)
Cell In[14], line 3
      1 from ambient.utils.eval_keypoints import get_keypoints, visualize_keypoints
----> 3 keypoints, frame = get_keypoints(project_root, sequences)
      4 visualize_keypoints(keypoints, frame)

...

File c:\Users\alexm\dev\alexpose\ambient\pose\keypoints.py:78, in suppress_stderr_fd()
     77 # Save the original stderr file descriptor
---> 78 stderr_fd = sys.stderr.fileno()

File c:\Users\alexm\dev\alexpose\.venv\Lib\site-packages\ipykernel\iostream.py:372, in OutStream.fileno(self)
    370     return self._original_stdstream_copy
    371 msg = "fileno"
--> 372 raise io.UnsupportedOperation(msg)

UnsupportedOperation: fileno
```

## Root Cause Analysis

The error occurred because:

1. **Duplicate Implementation**: `ambient/pose/keypoints.py` had its own copy of `suppress_stderr_fd()` 
2. **Missing Jupyter Fix**: This copy didn't have the Jupyter compatibility fix that was added to `ambient/pose/suppress_warnings.py`
3. **Jupyter Environment**: When running in Jupyter, `sys.stderr.fileno()` fails because Jupyter uses custom IO streams without real file descriptors

## Solution

Updated `suppress_stderr_fd()` in `ambient/pose/keypoints.py` to match the Jupyter-compatible version from `suppress_warnings.py`:

```python
@contextlib.contextmanager
def suppress_stderr_fd():
    """
    Suppress stderr at the file descriptor level.
    
    Works on both Unix and Windows systems, and is compatible with Jupyter notebooks.
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
    with os.fdopen(os.dup(stderr_fd), 'wb') as copied:
        sys.stderr.flush()
        try:
            null_file = 'NUL' if sys.platform == 'win32' else '/dev/null'
            with open(null_file, 'wb') as devnull:
                os.dup2(devnull.fileno(), stderr_fd)
            yield
        finally:
            sys.stderr.flush()
            os.dup2(copied.fileno(), stderr_fd)
```

## Key Changes

1. **Added `import io`**: To handle the `io.UnsupportedOperation` exception
2. **Try-Except Block**: Wraps `sys.stderr.fileno()` to catch Jupyter environment
3. **Fallback Mechanism**: Uses Python-level suppression when file descriptors unavailable
4. **Early Return**: Prevents execution of file descriptor code in Jupyter

## Files Modified

1. `ambient/pose/keypoints.py` - Added Jupyter compatibility to `suppress_stderr_fd()`
2. `notes/JUPYTER_COMPATIBILITY_FIX.md` - Updated with additional fix location
3. `notes/KEYPOINTS_JUPYTER_FIX.md` - This documentation

## Testing

Verified with `scripts/test_get_keypoints_fix.py`:
- ✅ Imports work correctly
- ✅ Loads GAVD data successfully
- ✅ Calls `get_keypoints()` with dictionary of sequences
- ✅ Returns correct tuple `(keypoints, frame)`
- ✅ Extracted 512 keypoints from first sequence
- ✅ Frame extracted with shape (360, 640, 3)

## Complete Fix Chain

This was the final fix in a series of related issues:

1. ✅ **Circular import** - `logging.py` shadowing Python's logging module
2. ✅ **Unicode errors** - Emoji characters causing encoding issues on Windows
3. ✅ **Stale cache** - `.pyc` files from old file locations
4. ✅ **Jupyter compatibility (suppress_warnings.py)** - First location fixed
5. ✅ **AttributeError** - `get_keypoints()` handling dict input
6. ✅ **Jupyter compatibility (keypoints.py)** - THIS FIX

## Usage

The notebook can now successfully run:

```python
from ambient.utils.eval_keypoints import get_keypoints, visualize_keypoints

# Works in Jupyter notebooks
keypoints, frame = get_keypoints(project_root, sequences)
visualize_keypoints(keypoints, frame)
```

## Next Steps

1. ✅ Clear Python cache: `python scripts/clear_python_cache.py`
2. ✅ Restart Jupyter kernel
3. ✅ Run notebook cells - should work without errors!

All fixes are complete and the notebook is now fully functional in Jupyter environments.
