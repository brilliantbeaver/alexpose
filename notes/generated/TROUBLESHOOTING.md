# Notebook Troubleshooting Guide

Quick reference for common issues when running AlexPose notebooks.

## Quick Fix Checklist

If you encounter errors in the notebooks, follow these steps:

### 1. Clear Python Cache
```bash
python scripts/clear_python_cache.py
```

### 2. Restart Jupyter Kernel
In Jupyter: `Kernel` → `Restart Kernel`

### 3. Verify Imports
```python
from ambient.utils.eval_keypoints import get_keypoints, visualize_keypoints
print("[OK] Imports successful")
```

---

## Common Errors

### Error: `UnsupportedOperation: fileno`

**Symptom**: Error when importing or using pose estimation functions in Jupyter

**Cause**: Jupyter's stderr doesn't have a real file descriptor

**Solution**: Already fixed in the codebase. If you still see this:
1. Clear Python cache: `python scripts/clear_python_cache.py`
2. Restart Jupyter kernel
3. Re-run the cell

**Documentation**: `notes/JUPYTER_COMPATIBILITY_FIX.md`

---

### Error: `AttributeError: 'KeypointVisualizer' has no attribute 'add_landmark_names'`

**Symptom**: Error when calling `visualize_keypoints()`

**Cause**: Old code trying to call non-existent method

**Solution**: Already fixed in the codebase. If you still see this:
1. Clear Python cache: `python scripts/clear_python_cache.py`
2. Restart Jupyter kernel
3. Verify you're using the latest code

**Documentation**: `notes/VISUALIZE_KEYPOINTS_FIX.md`

---

### Error: `AttributeError: 'dict' object has no attribute 'empty'`

**Symptom**: Error when calling `get_keypoints()` with sequences dictionary

**Cause**: Function expected DataFrame but received dict

**Solution**: Already fixed in the codebase. The function now handles both:
- `pd.DataFrame` - Single sequence
- `Dict[str, pd.DataFrame]` - Multiple sequences (uses first one)

**Documentation**: `notes/GET_KEYPOINTS_FIX.md`

---

### Error: `UnicodeEncodeError` with emoji characters

**Symptom**: Error when printing output with emoji characters on Windows

**Cause**: Windows console encoding issues

**Solution**: Already fixed - all emoji characters removed from code

**Documentation**: `notes/IMPORT_ERROR_FIX.md`

---

## Correct Usage Examples

### Extract Keypoints from Video Sequence

```python
from pathlib import Path
from ambient.utils.eval_keypoints import get_keypoints, visualize_keypoints
from ambient.gavd import GAVDDataLoader

# Setup
project_root = Path.cwd().parent  # Adjust if needed

# Load GAVD data
loader = GAVDDataLoader()
csv_path = project_root / "data" / "GAVD_Clinical_Annotations_1.1.csv"
df = loader.load_gavd_data(csv_path)
sequences = loader.organize_by_sequence(df)

# Extract keypoints (returns tuple: keypoints, frame)
keypoints, frame = get_keypoints(project_root, sequences)

# Visualize results
visualize_keypoints(keypoints, frame)
```

### Expected Output

```
Processing sequence: <sequence_id>
Number of frames: <N>
<frame_numbers>...
[OK] Sequence processing complete

[OK] SUCCESS! Detected 33 landmarks
[INFO] 28 landmarks are visible (confidence > 0.5)
[INFO] Average confidence: 0.847
[INFO] Detection quality: 0.823

[INFO] Sample keypoints:
  NOSE: (320.5, 145.2) confidence=0.950
  LEFT_EYE: (310.3, 138.7) confidence=0.920
  ...
```

Plus a side-by-side visualization showing:
- Left: Original video frame
- Right: Frame with detected pose skeleton

---

## Data Structures

### KeypointSet
```python
# Each frame's keypoints are stored in a KeypointSet object
keypoint_set = keypoints[0]  # First frame

# Access properties
print(f"Format: {keypoint_set.format.value}")  # e.g., "mediapipe_33"
print(f"Landmarks: {len(keypoint_set)}")       # e.g., 33
print(f"Avg confidence: {keypoint_set.avg_confidence:.3f}")
print(f"Visible: {len(keypoint_set.visible_keypoints)}")

# Access individual keypoints
for kp in keypoint_set.keypoints[:5]:
    print(f"{kp.name}: ({kp.x:.1f}, {kp.y:.1f}) conf={kp.confidence:.3f}")
```

### Keypoint
```python
# Each keypoint has rich metadata
kp = keypoint_set[0]  # First keypoint (NOSE)

print(f"ID: {kp.id}")                    # 0
print(f"Name: {kp.name}")                # "NOSE"
print(f"Position: ({kp.x}, {kp.y})")    # Pixel coordinates
print(f"Confidence: {kp.confidence}")    # 0.0 to 1.0
print(f"Visible: {kp.is_visible}")       # True/False
print(f"Reliable: {kp.is_reliable}")     # True/False
```

---

## Test Scripts

Run these to verify functionality:

```bash
# Test Jupyter compatibility
python scripts/test_jupyter_imports.py

# Test keypoint extraction
python scripts/test_get_keypoints_fix.py

# Test visualization functions
python scripts/test_visualize_keypoints.py
```

All tests should pass with `[OK] ALL TESTS PASSED!`

---

## Environment Setup

### Required Packages
```bash
# Install dependencies
uv sync

# Or with pip
pip install -r requirements.txt
```

### Required Environment Variables
```bash
# Create .env file (optional for notebooks)
OPENAI_API_KEY=sk-...
GOOGLE_API_KEY=AI...
```

### MediaPipe Model
The pose estimation model will be automatically downloaded to `data/models/` on first use.

---

## Getting Help

### Documentation Files
- `notes/NOTEBOOK_FIXES_SUMMARY.md` - Complete fix summary
- `notes/JUPYTER_COMPATIBILITY_FIX.md` - Jupyter-specific fixes
- `notes/VISUALIZE_KEYPOINTS_FIX.md` - Visualization function details
- `notes/GET_KEYPOINTS_FIX.md` - Keypoint extraction details

### Project Structure
- `ambient/pose/` - Pose estimation core
- `ambient/utils/` - Utility functions for notebooks
- `notebooks/` - Jupyter notebooks
- `scripts/` - Helper scripts

### Tech Stack
- Python 3.12+
- MediaPipe for pose estimation
- OpenCV for video processing
- Pandas for data management
- Matplotlib for visualization

---

## Best Practices

1. **Always clear cache after code changes**
   ```bash
   python scripts/clear_python_cache.py
   ```

2. **Restart kernel after clearing cache**
   - Jupyter: `Kernel` → `Restart Kernel`

3. **Use the wrapper functions in notebooks**
   - `from ambient.utils.eval_keypoints import get_keypoints, visualize_keypoints`
   - These provide notebook-friendly interfaces

4. **Check test scripts before debugging**
   - Run test scripts to verify the environment is set up correctly

5. **Keep documentation updated**
   - If you find new issues, document them in `notes/`
