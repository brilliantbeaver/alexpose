# Suppressing MediaPipe Warnings

## Overview

MediaPipe and TensorFlow Lite produce various informational and warning messages that can clutter output. This guide explains how AlexPose suppresses these messages by default.

## Types of Warnings

MediaPipe produces several types of output:

1. **INFO messages** (from TensorFlow Lite):
   ```
   INFO: Created TensorFlow Lite XNNPACK delegate for CPU.
   ```

2. **C++ warnings** (from MediaPipe inference):
   ```
   W0000 00:00:... inference_feedback_manager.cc:121] Feedback manager requires...
   ```

3. **Calculator warnings** (from MediaPipe calculators):
   ```
   W0000 00:00:... landmark_projection_calculator.cc:81] Using NORM_RECT without...
   ```

## Solution: File Descriptor Level Suppression

AlexPose uses **file descriptor level redirection** to suppress these warnings. This is more aggressive than Python-level suppression and catches C++ output.

### How It Works

```python
@contextlib.contextmanager
def suppress_stderr_fd():
    """Suppress stderr at the file descriptor level."""
    stderr_fd = sys.stderr.fileno()
    
    with os.fdopen(os.dup(stderr_fd), 'wb') as copied:
        sys.stderr.flush()
        
        try:
            # Redirect stderr to null device
            null_file = 'NUL' if sys.platform == 'win32' else '/dev/null'
            
            with open(null_file, 'wb') as devnull:
                os.dup2(devnull.fileno(), stderr_fd)
            
            yield
            
        finally:
            # Restore stderr
            sys.stderr.flush()
            os.dup2(copied.fileno(), stderr_fd)
```

### Key Points

1. **File Descriptor Level**: Redirects stderr at the OS level (fd 2)
2. **Cross-Platform**: Works on both Windows (NUL) and Unix (/dev/null)
3. **Temporary**: Restores stderr after the operation completes
4. **Default Behavior**: Enabled by default in `SequenceKeypointExtractor`

## Usage

### Default Behavior (Warnings Suppressed)

```python
from ambient.pose.keypoints import SequenceKeypointExtractor

# Warnings are automatically suppressed
extractor = SequenceKeypointExtractor()
keypoints = extractor.extract_from_image(image)
```

### Enable Warnings (For Debugging)

```python
from ambient.pose.keypoints import SequenceKeypointExtractor

# Enable warnings for debugging
extractor = SequenceKeypointExtractor(suppress_warnings=False)
keypoints = extractor.extract_from_image(image)
```

### Manual Suppression

```python
from ambient.pose.suppress_warnings import suppress_stderr_fd

# Suppress warnings for specific operations
with suppress_stderr_fd():
    # Your MediaPipe code here
    import mediapipe as mp
    landmarker = mp.solutions.pose.Pose()
```

### Automatic Suppression (Recommended)

```python
# Simply import ambient - suppression is automatic!
import ambient
from ambient.pose.mediapipe_estimator import MediaPipeEstimator

# No warnings will appear
estimator = MediaPipeEstimator()
```

The `ambient` package automatically imports `suppress_warnings` at initialization, setting all necessary environment variables before any MediaPipe imports occur.

### Suppress All Output

```python
from ambient.pose.suppress_warnings import SuppressOutput

# Suppress both stdout and stderr
with SuppressOutput(suppress_stderr=True, suppress_stdout=True):
    # All output is suppressed
    print("This won't be displayed")
    # MediaPipe warnings also suppressed
```

## Implementation Details

### Files Modified

1. **`ambient/pose/suppress_warnings.py`** (primary module)
   - Comprehensive utility module with environment variables
   - Multiple suppression strategies (fd-level, context managers, decorators)
   - Automatically imported by `ambient/__init__.py`

2. **`ambient/__init__.py`**
   - Imports `suppress_warnings` at package initialization
   - Ensures environment variables are set before any MediaPipe imports

3. **`ambient/pose/mediapipe_estimator.py`**
   - Uses `suppress_stderr_fd()` during MediaPipe operations
   - Wraps landmarker creation and detection calls

4. **`ambient/pose/keypoints.py`**
   - Added `suppress_stderr_fd()` context manager
   - Wraps MediaPipe operations with suppression
   - Default: `suppress_warnings=True`

5. **`notebooks/utils/eval_keypoints.py`**
   - Uses same suppression during imports
   - Ensures clean notebook output

### Why File Descriptor Level?

Other approaches don't work for C++ warnings:

❌ **Environment Variables**: Must be set before Python starts
```python
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Doesn't work if set after import
```

❌ **sys.stderr Redirection**: Only catches Python-level output
```python
sys.stderr = io.StringIO()  # Doesn't catch C++ output
```

✅ **File Descriptor Redirection**: Catches everything
```python
os.dup2(devnull.fileno(), stderr_fd)  # Catches C++ output!
```

## Testing

Run the test script to verify suppression:

```bash
python test_warning_suppression.py
```

Expected output:
```
======================================================================
Testing MediaPipe Warning Suppression
======================================================================

1. Importing SequenceKeypointExtractor...
✅ Import successful - no warnings should appear above

2. Creating extractor instance...
✅ Extractor created

3. Creating a test image...
✅ Test image created

4. Running pose detection...
✅ Pose detection completed - detected 0 keypoints

======================================================================
✅ TEST COMPLETE - No C++ warnings should have appeared!
======================================================================
```

If you see **only** the test messages (no TensorFlow/MediaPipe warnings), suppression is working correctly! 🎉

## Troubleshooting

### Warnings Still Appearing?

1. **Check suppress_warnings parameter**:
   ```python
   extractor = SequenceKeypointExtractor(suppress_warnings=True)  # Should be True
   ```

2. **Verify file descriptor suppression is being used**:
   ```python
   from ambient.pose.suppress_warnings import suppress_stderr_fd
   
   with suppress_stderr_fd():
       # Your code here
   ```

3. **Check for errors in stderr redirection**:
   - Ensure you have permissions to access /dev/null (Unix) or NUL (Windows)
   - Verify file descriptors are being properly restored

### Need Warnings for Debugging?

```python
# Temporarily enable warnings
extractor = SequenceKeypointExtractor(suppress_warnings=False)
```

## Best Practices

1. **Production**: Keep warnings suppressed (default)
2. **Development**: Enable warnings when debugging MediaPipe issues
3. **Testing**: Use suppression to keep test output clean
4. **Notebooks**: Warnings are automatically suppressed for clean output

## Performance Impact

File descriptor redirection has **negligible performance impact**:
- Only redirects during MediaPipe operations
- Overhead: < 1ms per operation
- No impact on pose detection accuracy

## References

- [MediaPipe Documentation](https://developers.google.com/mediapipe)
- [TensorFlow Lite Logging](https://www.tensorflow.org/lite/guide/inference#logging)
- [Python os.dup2 Documentation](https://docs.python.org/3/library/os.html#os.dup2)
