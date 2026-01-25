# Suppressing TensorFlow and MediaPipe Logs

## Problem
When running scripts that use MediaPipe, you see verbose logs like:
```
I0000 00:00:1768862377.618457 6144396 gl_context.cc:407] GL version: 2.1 (2.1 Metal - 90.5)
INFO: Created TensorFlow Lite XNNPACK delegate for CPU.
W0000 00:00:1768862377.667384 6144398 inference_feedback_manager.cc:121] Feedback manager requires...
```

## Solution
Set environment variables **before** importing any MediaPipe or TensorFlow modules:

```python
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Suppress TensorFlow logs
os.environ['GLOG_minloglevel'] = '3'      # Suppress MediaPipe logs

# Now import other modules
import sys
from pathlib import Path
from ambient.pose.keypoint_extractor import SequenceKeypointExtractor
```

## Environment Variable Levels

### TF_CPP_MIN_LOG_LEVEL
Controls TensorFlow C++ logging:
- `0` = All logs (default)
- `1` = Filter out INFO logs
- `2` = Filter out WARNING logs
- `3` = Filter out ERROR logs (only FATAL)

### GLOG_minloglevel
Controls Google logging (used by MediaPipe):
- `0` = INFO and above
- `1` = WARNING and above
- `2` = ERROR and above
- `3` = FATAL only

## Files Updated

1. ✅ `experiments/exp2/src/process3_per_set_conditions.py`
2. ✅ `notebooks/tutorial2 - train classifier.ipynb`

## Alternative: Suppress at Runtime

If you can't modify the script, set environment variables before running:

```bash
# Bash/Zsh
export TF_CPP_MIN_LOG_LEVEL=3
export GLOG_minloglevel=3
python your_script.py

# Or inline
TF_CPP_MIN_LOG_LEVEL=3 GLOG_minloglevel=3 python your_script.py
```

## Note
These environment variables must be set **before** the libraries are imported. Setting them after import has no effect.

## Verification
Run your script - you should no longer see:
- ❌ `GL version: 2.1 (2.1 Metal - 90.5)`
- ❌ `Created TensorFlow Lite XNNPACK delegate for CPU`
- ❌ `Feedback manager requires a model with a single signature`

You'll only see your application's output:
- ✅ `Examining 4 number of conditions ...`
- ✅ `Average joint angles for 145 frames:`
- ✅ `WARNING: First frame has no joint angles`
