# Pose Logging Suppression - Quick Reference

## ✅ Status: COMPLETE & ALL WARNINGS ELIMINATED

## What Was Fixed

Eliminated **all** verbose C++ warnings from pose estimation libraries:
1. **Initialization warnings**: GL context, feedback manager
2. **Inference warnings**: NORM_RECT, landmark projection

## Warnings Suppressed

```
✅ I0000 ... gl_context.cc:407] GL version: 2.1 (2.1 Metal - 90.5)...
✅ W0000 ... inference_feedback_manager.cc:121] Feedback manager requires...
✅ W0000 ... landmark_projection_calculator.cc:81] Using NORM_RECT without IMAGE_DIMENSIONS...
```

## Key Files

| File | Purpose |
|------|---------|
| `ambient/pose/pose_config.py` | Generic configuration for all pose backends |
| `ambient/gavd/pose_estimators.py` | MediaPipe implementation with local config |
| `docs/guides/pose-logging-suppression.md` | User guide |
| `POSE_LOGGING_FIX.md` | Complete solution summary |

## Quick Test

```bash
# Verify everything works
python -c "from ambient.gavd.pose_estimators import MediaPipeEstimator; print('✓ Works!')"

# Run test script
python scripts/test_mediapipe_logging.py
```

## For New Pose Estimators

### If No Circular Import Issues
```python
from ambient.pose.pose_config import configure_pose_environment, suppress_stderr

configure_pose_environment()  # Before imports
import your_pose_library

with suppress_stderr():
    estimator = YourEstimator(model_path)
```

### If Circular Import Issues (like MediaPipe)
```python
import os
import sys
import contextlib

# Set environment variables directly
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['GLOG_minloglevel'] = '3'
os.environ['OPENCV_LOG_LEVEL'] = 'ERROR'

# Local suppress_stderr implementation
@contextlib.contextmanager
def _suppress_stderr():
    stderr_fd = sys.stderr.fileno()
    saved_stderr_fd = os.dup(stderr_fd)
    devnull_fd = os.open(os.devnull, os.O_WRONLY)
    try:
        sys.stderr.flush()
        os.dup2(devnull_fd, stderr_fd)
        yield
    finally:
        sys.stderr.flush()
        os.dup2(saved_stderr_fd, stderr_fd)
        os.close(devnull_fd)
        os.close(saved_stderr_fd)

# Use it
with _suppress_stderr():
    estimator = YourEstimator(model_path)
```

## Environment Variables

```python
TF_CPP_MIN_LOG_LEVEL='3'  # TensorFlow Lite (MediaPipe, Ultralytics)
GLOG_minloglevel='3'      # Google logging (MediaPipe, OpenPose)
OPENCV_LOG_LEVEL='ERROR'  # OpenCV (all backends)
```

## Debugging

To see logs again:
```python
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '0'
os.environ['GLOG_minloglevel'] = '0'
os.environ['OPENCV_LOG_LEVEL'] = 'INFO'
```

## Key Insight

**C++ code writes to file descriptor 2 (OS-level stderr), not Python's `sys.stderr` object.**

Only OS-level redirection using `os.dup2()` can intercept these logs.

## Documentation

- Full guide: `docs/guides/pose-logging-suppression.md`
- Technical details: `POSE_LOGGING_FIX.md`
- Refactoring notes: `REFACTORING_SUMMARY.md`

---

**Questions?** Check `FINAL_SUMMARY.md` for complete details.
