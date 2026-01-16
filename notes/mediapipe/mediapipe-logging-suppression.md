# MediaPipe Logging Suppression Guide

## Problem Overview

MediaPipe and TensorFlow Lite generate verbose C++ level logs that appear as warnings during initialization:

```
I0000 00:00:1768369554.599845 6034739 gl_context.cc:407] GL version: 2.1 (2.1 Metal - 90.5), renderer: Apple M1 Pro
W0000 00:00:1768369554.535813 6034725 inference_feedback_manager.cc:121] Feedback manager requires a model with a single signature inference. Disabling support for feedback tensors.
```

### Why These Warnings Occur

1. **GL Version Log (I0000)**: Informational message about OpenGL/Metal GPU initialization
   - On M1/M2 Macs, MediaPipe uses Metal for GPU acceleration
   - This is purely informational and indicates successful initialization

2. **Feedback Manager Warning (W0000)**: TensorFlow Lite feature incompatibility
   - MediaPipe's pose models don't support "feedback tensors" (a stateful model feature)
   - The warning appears twice (once for IMAGE mode, once for VIDEO mode)
   - MediaPipe automatically disables this feature and continues normally
   - **This is completely harmless** - pose estimation doesn't need feedback tensors

### Why Standard Python Logging Doesn't Work

These logs come from C++ code in TensorFlow Lite and MediaPipe's internal libraries. They bypass Python's logging system and write directly to **file descriptor 2 (OS-level stderr)**, which is why:

- Python's `logging` module can't control them
- They appear even with `logger.setLevel(ERROR)`
- They're printed before Python code can intercept them
- **Critical**: Python's `sys.stderr` redirection doesn't affect C++ code
- Environment variables sometimes fail because the C++ library is already initialized

**The Real Problem**: C++ libraries write to the OS-level file descriptor, not Python's stderr object. This requires OS-level file descriptor manipulation to suppress.

## Solution Architecture

We've implemented a **three-layer defense** strategy:

### Layer 1: Environment Variables (Before Import)

Set environment variables **before** importing MediaPipe:

```python
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Suppress TF Lite logs
os.environ['GLOG_minloglevel'] = '3'      # Suppress Google logging
```

**Critical**: These must be set before `import mediapipe` to take effect.

### Layer 2: Centralized Configuration Module

We created `ambient/pose/mediapipe_config.py` that:
- Configures environment variables consistently
- Provides context managers for selective suppression
- Auto-configures on module import

```python
from ambient.pose.mediapipe_config import configure_mediapipe_environment

# Call before MediaPipe imports
configure_mediapipe_environment()
```

### Layer 3: OS-Level File Descriptor Redirection (During Initialization)

For stubborn logs that bypass environment variables, we redirect **file descriptor 2 (stderr)** at the OS level during landmarker creation:

```python
from ambient.pose.mediapipe_config import suppress_stderr

with suppress_stderr():
    landmarker = vision.PoseLandmarker.create_from_options(options)
```

**Why this works**: 
- C++ code writes to file descriptor 2, not Python's `sys.stderr` object
- We use `os.dup2()` to redirect fd 2 to `/dev/null` at the OS level
- This catches logs that environment variables miss
- After the context exits, stderr is restored to normal

## Implementation Details

### Files Modified

1. **`ambient/pose/mediapipe_config.py`** (NEW)
   - Centralized configuration utilities
   - Environment variable management
   - Context managers for log suppression

2. **`ambient/gavd/pose_estimators.py`**
   - Imports and uses `mediapipe_config`
   - Wraps landmarker creation with `suppress_stderr()`
   - Ensures clean initialization

3. **`notebooks/utils/eval_keypoints.py`**
   - Calls `configure_mediapipe_environment()` before imports
   - Ensures notebooks run cleanly

### Usage Patterns

#### For New Code

```python
# At the top of your file, before any MediaPipe imports
from ambient.pose.mediapipe_config import configure_mediapipe_environment
configure_mediapipe_environment()

import mediapipe as mp
# ... rest of your code
```

#### For Selective Suppression

```python
from ambient.pose.mediapipe_config import suppress_mediapipe_logs

# Suppress logs only during specific operations
with suppress_mediapipe_logs():
    estimator = MediaPipeEstimator(model_path="model.task")
    result = estimator.estimate_image_keypoints("image.jpg")
```

#### For Temporary Stderr Suppression

```python
from ambient.pose.mediapipe_config import suppress_stderr

# Suppress only stderr during initialization
with suppress_stderr():
    landmarker = vision.PoseLandmarker.create_from_options(options)
```

## Best Practices

### ✅ DO

- Set environment variables **before** importing MediaPipe
- Use `ambient.pose.mediapipe_config` for consistency
- Document why logs are being suppressed
- Keep suppression scoped to initialization code

### ❌ DON'T

- Try to suppress logs after importing MediaPipe (won't work)
- Use `setdefault()` - use direct assignment for reliability
- Suppress logs globally if you need them for debugging
- Forget that these warnings are harmless

## Debugging

If you need to see MediaPipe logs for debugging:

```python
import os

# Temporarily enable all logs
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '0'
os.environ['GLOG_minloglevel'] = '0'

# Then import and use MediaPipe
import mediapipe as mp
```

Or use the context manager approach:

```python
# Logs suppressed by default
from ambient.pose.mediapipe_config import suppress_mediapipe_logs

# Normal operation - logs suppressed
estimator = MediaPipeEstimator()

# Debug mode - logs visible
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '0'
debug_estimator = MediaPipeEstimator()
```

## Testing

To verify the suppression is working:

```bash
# Run a notebook that uses MediaPipe
jupyter notebook notebooks/explore3\ -\ extract\ features.ipynb

# Or run Python code
python -c "from ambient.gavd.pose_estimators import MediaPipeEstimator; e = MediaPipeEstimator('data/models/pose_landmarker_lite.task')"
```

You should see **no** warnings about "feedback manager" or GL version.

## Technical References

- [TensorFlow Logging Control](https://www.tensorflow.org/api_docs/python/tf/get_logger)
- [MediaPipe GPU Support](https://ai.google.dev/edge/mediapipe/framework/getting_started/gpu_support)
- [TensorFlow Lite Delegates](https://www.tensorflow.org/lite/performance/delegates)

## Summary

The warnings are **safe to suppress** because:
1. GL version logs are purely informational
2. Feedback tensor warnings indicate a feature that pose estimation doesn't use
3. MediaPipe continues to work perfectly without these features
4. The warnings don't indicate any actual problems

Our solution provides clean, professional output while maintaining the ability to enable logs for debugging when needed.
