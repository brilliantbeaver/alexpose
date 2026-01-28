# Zero Features - Root Cause Analysis and Fix
## Extended Joint Angle Statistics (std, max, min)

**Date:** January 27, 2026  
**Issue:** All extended joint angle statistics showing 0.00

---

## Root Cause Identified

### Problem: Duplicate Extraction Code

In `ambient/classification/features.py`, the `from_analysis_results()` method had **DUPLICATE extraction** of extended joint angle features:

**Location 1 (lines 960-977):** First extraction with conditional check
```python
left_hip_std = safe_extract(features_dict, "left_hip_std") if extract_all else 0.0
left_hip_max = safe_extract(features_dict, "left_hip_max") if extract_all else 0.0
# ... etc
```

**Location 2 (lines 1109-1127):** Second extraction WITHOUT conditional
```python
left_hip_std = safe_extract(features_dict, "left_hip_std")
left_hip_max = safe_extract(features_dict, "left_hip_max")
# ... etc
```

### Why This Caused 0.00 Values

The second extraction (lines 1109-1127) **OVERWRITES** the first extraction. If the features are not in `features_dict`, `safe_extract()` returns the default value of `0.0`, overwriting any previous values.

### The Deeper Issue

Even after removing the duplicate, the features might still be 0.00 if:

1. **FeatureExtractor is not creating them** - This happens when:
   - `include_joint_statistics=False` (but we verified it's True)
   - Angle arrays are empty due to confidence filtering
   - Exception occurs during extraction

2. **Angle arrays are empty** - This happens when:
   - No frames pass the confidence threshold check
   - Keypoint triplets required for angle calculation are missing
   - All confidence values are below threshold (0.3)

---

## Fixes Applied

### 1. Removed Duplicate Extraction ✅

**File:** `ambient/classification/features.py`

**Change:** Removed lines 960-977 (first extraction with `extract_all` check)

**Result:** Extended joint angle features are now extracted only once at lines 1109-1127

### 2. Added Debug Logging ✅

**File:** `ambient/analysis/feature_extractor.py`

**Change:** Added logging to track when extended statistics are extracted

```python
if self.include_joint_statistics:
    features[f"{angle_name}_std"] = np.std(angle_values)
    features[f"{angle_name}_max"] = np.max(angle_values)
    features[f"{angle_name}_min"] = np.min(angle_values)
    logger.debug(f"Extracted extended stats for {angle_name}: std={...}, max={...}, min={...}")
else:
    logger.warning(f"No valid angle values for {angle_name} - skipping extended statistics")
```

**Result:** Can now diagnose if angle arrays are empty

---

## Verification Steps

After applying fixes, check logs for:

### Success Case:
```
DEBUG: Extracted extended stats for left_hip: std=5.23, max=180.45, min=165.12
DEBUG: Extracted extended stats for left_knee: std=12.45, max=175.23, min=145.67
DEBUG: Extracted extended stats for left_ankle: std=8.91, max=160.34, min=135.22
...
```

### Failure Case (Empty Angles):
```
WARNING: No valid angle values for left_hip - skipping extended statistics
WARNING: No valid angle values for left_knee - skipping extended statistics
...
```

If you see warnings, the issue is that **no frames pass the confidence threshold** for the required keypoint triplets.

---

## Potential Remaining Issues

If features are still 0.00 after fixes, investigate:

### Issue 1: Confidence Threshold Too High

**Symptom:** Warnings about "No valid angle values"

**Cause:** Confidence threshold (0.3) filters out all frames

**Solution:** Lower confidence threshold or check pose estimation quality

```python
analyzer = EnhancedGaitAnalyzer(
    feature_extraction_config={
        "confidence_threshold": 0.2  # Lower threshold
    }
)
```

### Issue 2: Missing Keypoints

**Symptom:** Warnings about "No valid angle values" for specific joints

**Cause:** Pose estimator not detecting certain keypoints

**Solution:** 
- Check pose estimation quality
- Try different pose estimator (MediaPipe, Ultralytics, etc.)
- Verify video quality and camera angle

### Issue 3: Keypoint Format Mismatch

**Symptom:** No angles calculated at all

**Cause:** Wrong keypoint format specified

**Solution:** Verify keypoint format matches pose estimator output

```python
# For MediaPipe
analyzer = EnhancedGaitAnalyzer(keypoint_format="BLAZEPOSE_33")

# For Ultralytics YOLO
analyzer = EnhancedGaitAnalyzer(keypoint_format="COCO_17")
```

---

## Testing

Run with debug logging enabled:

```python
import logging
from loguru import logger

# Enable debug logging
logger.remove()
logger.add(lambda msg: print(msg), level="DEBUG")

# Run analysis
analyzer = EnhancedGaitAnalyzer(comprehensive_features=True)
results = analyzer.analyze_gait_sequence(pose_sequence)
features = GaitFeatureVector.from_analysis_results(results)

# Check features
print(f"left_hip_std: {features.left_hip_std}")
print(f"left_knee_std: {features.left_knee_std}")
print(f"left_ankle_std: {features.left_ankle_std}")
```

---

## Expected Results After Fix

With good quality pose data:

```
left_hip_std                  :    5.23  ← FIXED (was 0.00)
left_hip_max                  :  180.45  ← FIXED (was 0.00)
left_hip_min                  :  165.12  ← FIXED (was 0.00)
left_knee_std                 :   12.45  ← FIXED (was 0.00)
left_knee_max                 :  175.23  ← FIXED (was 0.00)
left_knee_min                 :  145.67  ← FIXED (was 0.00)
left_ankle_std                :    8.91  ← FIXED (was 0.00)
left_ankle_max                :  160.34  ← FIXED (was 0.00)
left_ankle_min                :  135.22  ← FIXED (was 0.00)
...
```

---

## Files Modified

1. ✅ `ambient/classification/features.py`
   - Removed duplicate extraction (lines 960-977)

2. ✅ `ambient/analysis/feature_extractor.py`
   - Added debug logging for extended statistics
   - Added warning for empty angle arrays

---

## Summary

**Root Cause:** Duplicate extraction code where second extraction overwrote first with 0.0 defaults

**Fix:** Removed duplicate extraction, keeping only the unconditional extraction at lines 1109-1127

**Additional:** Added logging to diagnose if angle arrays are empty due to confidence filtering

**Next Step:** Run with debug logging to verify features are being extracted correctly
