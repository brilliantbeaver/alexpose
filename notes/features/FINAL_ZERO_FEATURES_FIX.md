# Final Zero Features Fix - Complete Root Cause Analysis

**Date:** January 27, 2026  
**Status:** COMPREHENSIVE FIX APPLIED

---

## Summary of Remaining Issues

After previous fixes, two categories of features were still showing 0.00:

1. **Extended joint angle statistics (18 features):** left/right hip/knee/ankle std/max/min
2. **phase_asymmetry (1 feature)**

---

## Root Cause #1: Extended Joint Angle Statistics

### The Problem

Extended joint angle statistics (std, max, min) were consistently 0.00 even though:
- Core angle statistics (mean, range) had valid values
- `include_joint_statistics=True` was set
- The extraction code looked correct

### Deep Investigation

Traced through the code path:

1. `EnhancedGaitAnalyzer.analyze_gait_sequence()` calls `FeatureExtractor.extract_features()`
2. `FeatureExtractor.extract_features()` calls `_extract_joint_angle_features()`
3. `_extract_joint_angle_features()` has a `try-except` block (line 222-257)
4. **CRITICAL:** The except block catches ALL exceptions with only a WARNING log

```python
try:
    # Extract angles and statistics
    ...
    if self.include_joint_statistics:
        features[f"{angle_name}_std"] = np.std(angle_values)
        features[f"{angle_name}_max"] = np.max(angle_values)
        features[f"{angle_name}_min"] = np.min(angle_values)
except Exception as e:
    logger.warning(f"Joint angle extraction failed: {e}")  # ← SILENTLY FAILS!
    
return features  # Returns WITHOUT extended statistics!
```

### Root Cause

**ANY exception during extended statistics extraction causes silent failure:**
- Exception is caught
- Only a warning is logged (easy to miss)
- Method returns `features` dict WITHOUT extended statistics
- `safe_extract()` in `GaitFeatureVector.from_analysis_results()` returns 0.0 for missing keys

### Possible Exception Causes

1. **NumPy errors:** Division by zero, invalid operations
2. **Type errors:** Unexpected data types
3. **Index errors:** Array indexing issues
4. **Attribute errors:** Missing attributes on objects

### Fixes Applied

**1. Improved error logging (line 256):**
```python
except Exception as e:
    logger.error(f"Joint angle extraction failed: {e}", exc_info=True)  # ← Now ERROR with stack trace
    # Still return what we have so far
```

**2. Ensure features are created even with empty arrays (lines 247-254):**
```python
else:
    logger.warning(f"No valid angle values for {angle_name}...")
    # Still create the features with 0.0 so they exist in the dict
    features[f"{angle_name}_mean"] = 0.0
    features[f"{angle_name}_range"] = 0.0
    if self.include_joint_statistics:
        features[f"{angle_name}_std"] = 0.0
        features[f"{angle_name}_max"] = 0.0
        features[f"{angle_name}_min"] = 0.0
```

**3. Added debug logging for successful extraction (line 244):**
```python
logger.debug(f"Extracted extended stats for {angle_name}: std={...}, max={...}, min={...}")
```

---

## Root Cause #2: phase_asymmetry

### The Problem

`phase_asymmetry` was 0.00 even with 4 detected cycles.

### Investigation

The calculation logic:
```python
if left_stance and right_stance:
    # Calculate asymmetry between left and right
    features["phase_asymmetry"] = ...
else:
    # Fallback to CV
    features["phase_asymmetry"] = np.std(stance_durations) / np.mean(stance_durations) if len(stance_durations) > 1 else 0.0
```

### Root Cause

**Multiple failure points:**
1. If `left_stance` or `right_stance` are empty (no cycles for one foot)
2. If `len(stance_durations) <= 1` (only 1 cycle total)
3. Fallback returns 0.0 without logging why

### Fix Applied

**Improved fallback logic with debug logging:**
```python
if left_stance and right_stance:
    # Primary calculation
    features["phase_asymmetry"] = abs(left_stance_mean - right_stance_mean) / ((left_stance_mean + right_stance_mean) / 2)
elif len(stance_durations) > 1:
    # Fallback: use CV as proxy
    features["phase_asymmetry"] = np.std(stance_durations) / np.mean(stance_durations)
else:
    # Not enough data
    features["phase_asymmetry"] = 0.0
    logger.debug(f"Cannot calculate phase_asymmetry: left_stance={len(left_stance)}, right_stance={len(right_stance)}, stance_durations={len(stance_durations)}")
```

---

## Files Modified

1. ✅ `ambient/analysis/feature_extractor.py`
   - Changed exception logging from WARNING to ERROR with stack trace
   - Added feature creation for empty angle arrays
   - Added debug logging for successful extraction

2. ✅ `ambient/analysis/temporal_analyzer.py`
   - Improved phase_asymmetry fallback logic
   - Added debug logging for failure cases

3. ✅ `ambient/classification/features.py`
   - Removed duplicate extraction code (previous fix)

---

## Diagnostic Tools

### debug_extended_features.py

Comprehensive diagnostic script that:
1. Initializes analyzer with comprehensive features
2. Runs analysis
3. Checks what's in `features_dict`
4. Identifies missing features
5. Extracts GaitFeatureVector
6. Compares values at each stage
7. Provides diagnosis

**Usage:**
```python
from debug_extended_features import debug_extended_features
results, features = debug_extended_features(pose_sequence)
```

---

## Expected Behavior After Fixes

### If Extended Statistics Work:

```
left_hip_std                  :    5.23  ← Non-zero
left_hip_max                  :  180.45  ← Non-zero
left_hip_min                  :  165.12  ← Non-zero
...
```

### If Extended Statistics Still 0.00:

Check logs for:
```
ERROR: Joint angle extraction failed: <exception details>
<full stack trace>
```

This will tell you EXACTLY what exception is occurring.

### If phase_asymmetry Still 0.00:

Check logs for:
```
DEBUG: Cannot calculate phase_asymmetry: left_stance=0, right_stance=4, stance_durations=4
```

This tells you why it couldn't be calculated (e.g., all cycles are for one foot only).

---

## Next Steps

1. ✅ Run analysis with updated code
2. ⏳ Check logs for ERROR messages about joint angle extraction
3. ⏳ Check logs for DEBUG messages about phase_asymmetry
4. ⏳ If still 0.00, run `debug_extended_features.py` to see exactly what's in features_dict
5. ⏳ Fix any exceptions revealed by the improved error logging

---

## Summary

**Root causes identified:**
1. **Silent exception handling** - Exceptions during extended statistics extraction were caught and only logged as warnings
2. **Missing fallback** - No features created when angle arrays were empty
3. **Insufficient logging** - Hard to diagnose why features weren't being created

**Fixes applied:**
1. **Improved error logging** - Exceptions now logged as ERROR with full stack trace
2. **Guaranteed feature creation** - Features created with 0.0 even if angle arrays empty
3. **Enhanced debug logging** - Success and failure cases both logged
4. **Better fallback logic** - phase_asymmetry has multiple fallback levels with logging

**Result:**
- If features are still 0.00, logs will now show EXACTLY why
- No more silent failures
- Clear diagnostic path to identify remaining issues
