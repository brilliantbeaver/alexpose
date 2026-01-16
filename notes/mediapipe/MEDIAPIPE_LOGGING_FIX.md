# MediaPipe Logging Suppression - Implementation Summary

## Problem
MediaPipe was generating verbose C++ warnings during initialization:
- GL version/Metal renderer info logs
- "Feedback manager requires a model with a single signature inference" warnings

These warnings are **harmless** but clutter the output and concern users.

## Root Cause
- Warnings come from C++ code in TensorFlow Lite/MediaPipe
- They write directly to **file descriptor 2** (OS-level stderr), not Python's `sys.stderr` object
- Environment variables must be set **before** importing MediaPipe to take effect
- **Critical**: Python's `sys.stderr` redirection doesn't affect C++ code - requires OS-level fd manipulation

## Solution Implemented

### 1. Created Centralized Configuration Module
**File**: `ambient/pose/mediapipe_config.py`

Provides:
- `configure_mediapipe_environment()` - Sets environment variables
- `suppress_stderr()` - Context manager for **OS-level file descriptor redirection**
- `suppress_mediapipe_logs()` - Combined suppression utility

**Key Innovation**: Uses `os.dup2()` to redirect file descriptor 2 to `/dev/null`, which catches C++ logs that bypass Python's stderr.

### 2. Updated Core Estimator
**File**: `ambient/gavd/pose_estimators.py`

Changes:
- Imports and applies configuration before MediaPipe imports
- Wraps landmarker creation with `suppress_stderr()` context manager
- Ensures clean initialization for both IMAGE and VIDEO modes

### 3. Updated Notebook Utilities
**File**: `notebooks/utils/eval_keypoints.py`

Changes:
- Calls `configure_mediapipe_environment()` at the top
- Ensures notebooks run without warnings

### 4. Created Documentation
**File**: `docs/guides/mediapipe-logging-suppression.md`

Comprehensive guide covering:
- Why warnings occur
- Why they're safe to suppress
- Implementation details
- Usage patterns
- Best practices
- Debugging tips

### 5. Created Test Script
**File**: `scripts/test_mediapipe_logging.py`

Verifies:
- Configuration imports correctly
- MediaPipe imports without warnings
- Estimator initializes cleanly

## How It Works

### Three-Layer Defense Strategy

**Layer 1: Environment Variables**
```python
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Suppress TF Lite
os.environ['GLOG_minloglevel'] = '3'      # Suppress Google logging
```

**Layer 2: Centralized Configuration**
```python
from ambient.pose.mediapipe_config import configure_mediapipe_environment
configure_mediapipe_environment()  # Before MediaPipe imports
```

**Layer 3: OS-Level File Descriptor Redirection**
```python
with suppress_stderr():
    # Redirects fd 2 to /dev/null at OS level
    landmarker = vision.PoseLandmarker.create_from_options(options)
```

**Why Layer 3 is Critical**: C++ code writes to file descriptor 2, not Python's `sys.stderr`. Only OS-level redirection using `os.dup2()` can intercept these logs.

## Testing

Run the test script:
```bash
python scripts/test_mediapipe_logging.py
```

Expected output: No warnings, all tests pass ✓

## Usage for New Code

```python
# At the top of any file that imports MediaPipe
from ambient.pose.mediapipe_config import configure_mediapipe_environment
configure_mediapipe_environment()

import mediapipe as mp
# ... rest of your code
```

## Files Modified

1. ✅ `ambient/pose/mediapipe_config.py` (NEW)
2. ✅ `ambient/gavd/pose_estimators.py` (UPDATED)
3. ✅ `notebooks/utils/eval_keypoints.py` (UPDATED)
4. ✅ `docs/guides/mediapipe-logging-suppression.md` (NEW)
5. ✅ `scripts/test_mediapipe_logging.py` (NEW)

## Best Practices

### ✅ DO
- Use `ambient.pose.mediapipe_config` for consistency
- Set environment variables before MediaPipe imports
- Document why logs are suppressed

### ❌ DON'T
- Try to suppress logs after importing MediaPipe
- Suppress logs globally if debugging
- Worry about these warnings - they're harmless

## Why These Warnings Are Safe to Suppress

1. **GL Version Logs**: Purely informational about GPU initialization
2. **Feedback Manager Warnings**: Indicates a TFLite feature that pose estimation doesn't need
3. **No Impact**: MediaPipe works perfectly without these features
4. **Industry Standard**: Common practice to suppress these in production

## Debugging

To temporarily enable logs for debugging:
```python
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '0'  # Show all logs
os.environ['GLOG_minloglevel'] = '0'
```

## References

- [TensorFlow Logging Control](https://www.tensorflow.org/api_docs/python/tf/get_logger)
- [MediaPipe GPU Support](https://ai.google.dev/edge/mediapipe/framework/getting_started/gpu_support)
- [Stack Overflow: Suppress TensorFlow Warnings](https://stackoverflow.com/questions/35869137/avoid-tensorflow-print-on-standard-error)

---

**Status**: ✅ Implemented and tested
**Impact**: Clean, professional output without verbose C++ warnings
**Maintainability**: Centralized configuration makes it easy to adjust
