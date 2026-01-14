# Pose Estimation Logging Suppression Guide

## Problem Overview

Pose estimation libraries (MediaPipe, OpenPose, Ultralytics, etc.) and their C++ dependencies generate verbose logs during initialization:

**MediaPipe/TensorFlow Lite:**
```
I0000 00:00:... gl_context.cc:407] GL version: 2.1 (2.1 Metal - 90.5), renderer: Apple M1 Pro
W0000 00:00:... inference_feedback_manager.cc:121] Feedback manager requires a model with a single signature inference...
```

**OpenPose/Caffe:**
```
I0000 00:00:... Caffe: setting CPU mode
I0000 00:00:... Net initialization done
```

**Ultralytics/YOLO:**
```
Ultralytics YOLOv8.0.0 Python-3.12.0 torch-2.0.0...
```

### Why These Warnings Occur

1. **GL Version Logs**: Informational messages about GPU initialization (Metal on M1/M2 Macs, CUDA on NVIDIA)
2. **Feedback Manager Warnings**: TensorFlow Lite feature incompatibility (harmless for pose estimation)
3. **Caffe Initialization**: OpenPose backend startup messages
4. **Model Loading**: Various libraries log model initialization details

### Why Standard Python Logging Doesn't Work

These logs come from C++ code that writes directly to **file descriptor 2 (OS-level stderr)**, not Python's `sys.stderr` object:

- Python's `logging` module can't control them
- They appear even with `logger.setLevel(ERROR)`
- They're printed before Python code can intercept them
- **Critical**: Python's `sys.stderr` redirection doesn't affect C++ code
- Environment variables sometimes fail because the C++ library is already initialized

**The Real Problem**: C++ libraries write to the OS-level file descriptor, not Python's stderr object. This requires OS-level file descriptor manipulation to suppress.

## Solution Architecture

We've implemented a **three-layer defense** strategy in `ambient/pose/pose_config.py`:

### Layer 1: Environment Variables (Before Import)

Set environment variables **before** importing pose estimation libraries:

```python
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # TensorFlow Lite (MediaPipe, Ultralytics)
os.environ['GLOG_minloglevel'] = '3'      # Google logging (MediaPipe, OpenPose)
os.environ['OPENCV_LOG_LEVEL'] = 'ERROR'  # OpenCV (all backends)
```

**Critical**: These must be set before `import mediapipe` or other libraries to take effect.

### Layer 2: Centralized Configuration Module

We created `ambient/pose/pose_config.py` that:
- Configures environment variables consistently across all pose estimators
- Provides reusable context managers for selective suppression
- Auto-configures on module import

```python
from ambient.pose.pose_config import configure_pose_environment

# Call before pose estimation library imports
configure_pose_environment()
```

### Layer 3: OS-Level File Descriptor Redirection (During Initialization)

For stubborn logs that bypass environment variables, we redirect **file descriptor 2 (stderr)** at the OS level during estimator creation:

```python
from ambient.pose.pose_config import suppress_stderr

with suppress_stderr():
    estimator = PoseEstimator(model_path)
```

**Why this works**: 
- C++ code writes to file descriptor 2, not Python's `sys.stderr` object
- We use `os.dup2()` to redirect fd 2 to `/dev/null` at the OS level
- This catches logs that environment variables miss
- After the context exits, stderr is restored to normal

## Implementation Details

### Files Modified

1. **`ambient/pose/pose_config.py`** (Centralized configuration)
   - `configure_pose_environment()` - Sets environment variables for all backends
   - `suppress_stderr()` - OS-level file descriptor redirection
   - `suppress_pose_logs()` - Combined suppression utility

2. **`ambient/gavd/pose_estimators.py`** (MediaPipe implementation)
   - Imports and uses `pose_config`
   - Wraps landmarker creation with `suppress_stderr()`

3. **`notebooks/utils/keypoints.py`** (Notebook utilities)
   - Calls `configure_pose_environment()` before imports

### Supported Backends

| Backend | C++ Library | Environment Variables | Suppression |
|---------|-------------|----------------------|-------------|
| MediaPipe | TensorFlow Lite | `TF_CPP_MIN_LOG_LEVEL`, `GLOG_minloglevel` | ✅ Full |
| OpenPose | Caffe | `GLOG_minloglevel` | ✅ Full |
| Ultralytics | PyTorch + TF | `TF_CPP_MIN_LOG_LEVEL` | ✅ Full |
| AlphaPose | PyTorch | `OPENCV_LOG_LEVEL` | ✅ Partial |

### Usage Patterns

#### For New Pose Estimators

```python
# At the top of your estimator file, before any C++ library imports
from ambient.pose.pose_config import configure_pose_environment
configure_pose_environment()

# Then import your pose estimation library
import your_pose_library
```

#### For Selective Suppression

```python
from ambient.pose.pose_config import suppress_pose_logs

# Suppress logs only during specific operations
with suppress_pose_logs():
    estimator = YourPoseEstimator(model_path="model.pth")
    result = estimator.estimate(image)
```

#### For Temporary Stderr Suppression

```python
from ambient.pose.pose_config import suppress_stderr

# Suppress only stderr during initialization
with suppress_stderr():
    model = load_pose_model(model_path)
```

## Best Practices

### ✅ DO

- Set environment variables **before** importing pose libraries
- Use `ambient.pose.pose_config` for consistency across all estimators
- Document why logs are being suppressed
- Keep suppression scoped to initialization code
- Test that real errors are still visible during inference

### ❌ DON'T

- Try to suppress logs after importing libraries (won't work)
- Use `setdefault()` - use direct assignment for reliability
- Suppress logs globally if you need them for debugging
- Forget that these warnings are usually harmless
- Suppress stderr during inference (hides real errors)

## Debugging

If you need to see pose estimation logs for debugging:

```python
import os

# Temporarily enable all logs
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '0'
os.environ['GLOG_minloglevel'] = '0'
os.environ['OPENCV_LOG_LEVEL'] = 'INFO'

# Then import and use pose libraries
import mediapipe as mp
```

Or use the context manager approach:

```python
# Logs suppressed by default
from ambient.pose.pose_config import suppress_pose_logs

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
# Run the test script
python scripts/test_mediapipe_logging.py

# Or test directly
python -c "from ambient.gavd.pose_estimators import MediaPipeEstimator; e = MediaPipeEstimator('data/models/pose_landmarker_lite.task')"
```

You should see **no** warnings about "feedback manager", "GL version", or Caffe initialization.

## Technical References

- [TensorFlow Logging Control](https://www.tensorflow.org/api_docs/python/tf/get_logger)
- [MediaPipe GPU Support](https://ai.google.dev/edge/mediapipe/framework/getting_started/gpu_support)
- [TensorFlow Lite Delegates](https://www.tensorflow.org/lite/performance/delegates)
- [Caffe Logging](https://caffe.berkeleyvision.org/tutorial/interfaces.html)
- [Unix File Descriptors](https://en.wikipedia.org/wiki/File_descriptor)

## Summary

The warnings are **safe to suppress** because:
1. GL version logs are purely informational
2. Feedback tensor warnings indicate features that pose estimation doesn't use
3. Caffe initialization logs are standard startup messages
4. The warnings don't indicate any actual problems

Our solution provides clean, professional output while maintaining the ability to enable logs for debugging when needed. The centralized `pose_config.py` module makes it easy to add support for new pose estimation backends in the future.
