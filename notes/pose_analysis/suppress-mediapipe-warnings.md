# Suppressing MediaPipe Warnings

MediaPipe may display C++ warnings during initialization, such as:

```
W0000 00:00:1768456500.575937 9636 inference_feedback_manager.cc:121] 
Feedback manager requires a model with a single signature inference. 
Disabling support for feedback tensors.
```

These warnings are **informational only** and don't affect functionality. However, if you want to suppress them, here are several methods:

## Method 1: Environment Variables (Recommended)

Set these environment variables **before** starting Python:

### Windows (PowerShell)
```powershell
$env:TF_CPP_MIN_LOG_LEVEL="3"
$env:GLOG_minloglevel="3"
$env:GLOG_logtostderr="0"
$env:ABSL_MIN_LOG_LEVEL="3"

# Then run your Python script
python your_script.py
```

### Windows (CMD)
```cmd
set TF_CPP_MIN_LOG_LEVEL=3
set GLOG_minloglevel=3
set GLOG_logtostderr=0
set ABSL_MIN_LOG_LEVEL=3

python your_script.py
```

### Linux/macOS
```bash
export TF_CPP_MIN_LOG_LEVEL=3
export GLOG_minloglevel=3
export GLOG_logtostderr=0
export ABSL_MIN_LOG_LEVEL=3

python your_script.py
```

## Method 2: Add to .env File

Add these to your `.env` file in the project root:

```env
TF_CPP_MIN_LOG_LEVEL=3
GLOG_minloglevel=3
GLOG_logtostderr=0
ABSL_MIN_LOG_LEVEL=3
OPENCV_LOG_LEVEL=ERROR
```

Then load them before importing:

```python
from dotenv import load_dotenv
load_dotenv()  # Load before importing MediaPipe

from ambient.pose import MediaPipeEstimator
```

## Method 3: Set in Python (Limited Effectiveness)

The AlexPose library already sets these in `mediapipe_estimator.py`, but some warnings may still appear because they're printed during C++ library initialization:

```python
import os

# Set before any MediaPipe imports
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['GLOG_minloglevel'] = '3'
os.environ['GLOG_logtostderr'] = '0'

from ambient.pose import MediaPipeEstimator
```

## Method 4: Jupyter Notebooks

For Jupyter notebooks, add this to the **first cell**:

```python
import os
import warnings

# Suppress environment warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['GLOG_minloglevel'] = '3'
os.environ['GLOG_logtostderr'] = '0'
os.environ['ABSL_MIN_LOG_LEVEL'] = '3'

# Suppress Python warnings
warnings.filterwarnings('ignore')

# Now import MediaPipe
from ambient.pose import MediaPipeEstimator
```

## Understanding the Warning

The "feedback tensors" warning appears because:

1. MediaPipe's inference feedback manager expects models with a single signature
2. The pose landmarker model has multiple signatures
3. MediaPipe automatically disables feedback tensor support (which is fine)

**This is normal behavior and doesn't affect pose estimation accuracy or performance.**

## Verification

To verify warnings are suppressed:

```python
from ambient.pose import MediaPipeEstimator

# Should initialize without warnings
estimator = MediaPipeEstimator(
    model_path="data/models/pose_landmarker_lite.task"
)

print("✓ MediaPipe initialized successfully")
```

## Troubleshooting

If warnings still appear:

1. **Check environment variables are set** before Python starts
2. **Restart your Python kernel** (for Jupyter notebooks)
3. **Use Method 1** (shell environment variables) for most reliable suppression
4. **Accept the warnings** - they're harmless and only appear once during initialization

## Additional Resources

- [MediaPipe Documentation](https://developers.google.com/mediapipe)
- [TensorFlow Logging](https://www.tensorflow.org/api_docs/python/tf/get_logger)
- [Google Logging (GLOG)](https://github.com/google/glog)
