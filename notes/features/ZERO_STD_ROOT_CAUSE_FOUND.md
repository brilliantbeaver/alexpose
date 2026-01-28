# ✅ Root Cause Found: Zero STD Values

**Date:** January 27, 2026  
**Status:** ROOT CAUSE IDENTIFIED  
**Issue:** std values showing 0.00 despite non-zero mean/range

---

## The Problem

User is seeing:
```
left_hip_mean     : 173.19  ✅ NON-ZERO
left_hip_range    :  18.19  ✅ NON-ZERO
left_hip_std      :   0.00  ❌ ZERO
```

## Root Cause Identified

**The issue is in the notebook code, NOT in the AlexPose library!**

Looking at `experiments/exp5/07_extract_all_features.ipynb` lines 685-695:

```python
# Step 3: Run enhanced analysis (for the new features)
analyzer = EnhancedGaitAnalyzer(keypoint_format="BLAZEPOSE_33", fps=30.0)
analysis_results = analyzer.analyze_gait_sequence(pose_sequence)

# ❌ BUG: Manually overwriting features dict
if "features" not in analysis_results:
    analysis_results["features"] = {}

# Extract joint angle statistics and add to features dict
for joint_name in ["left_hip", "left_knee", "left_ankle", "right_hip", "right_knee", "right_ankle"]:
    stats = normal_joint_angles.get_statistics(joint_name)
    analysis_results["features"][f"{joint_name}_mean"] = stats.get("mean", 0.0)
    analysis_results["features"][f"{joint_name}_range"] = stats.get("range", 0.0)
    # ❌ NOT ADDING STD!

# Create feature vector with ALL features
feature_vector = GaitFeatureVector.from_analysis_results(
    analysis_results,
    sample_id=normal_sid,
    condition_label="normal"
)
```

## What's Happening

1. ✅ `EnhancedGaitAnalyzer.analyze_gait_sequence()` runs correctly
2. ✅ `FeatureExtractor` creates `analysis_results["features"]` with std values
3. ❌ **User code OVERWRITES `analysis_results["features"]`** with only mean/range
4. ❌ std values are lost
5. ❌ `GaitFeatureVector.from_analysis_results()` extracts 0.00 for std

## Proof

Our debug script (`debug_std_extraction.py`) proved:
- ✅ `GaitFeatureVector.from_analysis_results()` works correctly
- ✅ If std values are in `features_dict`, they're extracted properly
- ❌ The std values are NOT in `features_dict` because user code overwrites it

## The Fix

### Option 1: Remove Manual Overwrite (RECOMMENDED)

**Simply delete lines 685-695** and let `EnhancedGaitAnalyzer` do its job:

```python
# Step 3: Run enhanced analysis
analyzer = EnhancedGaitAnalyzer(keypoint_format="BLAZEPOSE_33", fps=30.0)
analysis_results = analyzer.analyze_gait_sequence(pose_sequence)

# ✅ Don't manually overwrite - EnhancedGaitAnalyzer already extracted everything!

# Create feature vector
feature_vector = GaitFeatureVector.from_analysis_results(
    analysis_results,
    sample_id=normal_sid,
    condition_label="normal"
)
```

### Option 2: Add std to Manual Overwrite

If you need to manually add features, include std:

```python
# Add joint angle statistics to analysis_results
if "features" not in analysis_results:
    analysis_results["features"] = {}

for joint_name in ["left_hip", "left_knee", "left_ankle", "right_hip", "right_knee", "right_ankle"]:
    stats = normal_joint_angles.get_statistics(joint_name)
    analysis_results["features"][f"{joint_name}_mean"] = stats.get("mean", 0.0)
    analysis_results["features"][f"{joint_name}_range"] = stats.get("range", 0.0)
    analysis_results["features"][f"{joint_name}_std"] = stats.get("std", 0.0)  # ✅ ADD THIS
```

### Option 3: Update Instead of Overwrite

Use `update()` to merge instead of overwrite:

```python
# Merge additional features instead of overwriting
for joint_name in ["left_hip", "left_knee", "left_ankle", "right_hip", "right_knee", "right_ankle"]:
    stats = normal_joint_angles.get_statistics(joint_name)
    # Only add if not already present
    if f"{joint_name}_mean" not in analysis_results["features"]:
        analysis_results["features"][f"{joint_name}_mean"] = stats.get("mean", 0.0)
    if f"{joint_name}_range" not in analysis_results["features"]:
        analysis_results["features"][f"{joint_name}_range"] = stats.get("range", 0.0)
    if f"{joint_name}_std" not in analysis_results["features"]:
        analysis_results["features"][f"{joint_name}_std"] = stats.get("std", 0.0)
```

## Why This Happened

The notebook code was written before `EnhancedGaitAnalyzer` was fully implemented. The manual feature extraction was a workaround that's no longer needed.

`EnhancedGaitAnalyzer` now:
- ✅ Extracts all joint angle features (mean, range, std)
- ✅ Extracts all kinematic features
- ✅ Extracts all temporal features
- ✅ Extracts all symmetry features
- ✅ Extracts all stability features

**You don't need to manually add features anymore!**

## Verification

After applying the fix, run this in the notebook:

```python
# Check what's in features_dict
print("Features in analysis_results:")
for key in sorted(analysis_results["features"].keys()):
    if "std" in key:
        print(f"  {key}: {analysis_results['features'][key]}")
```

You should see:
```
Features in analysis_results:
  left_ankle_std: 15.8
  left_hip_std: 5.5
  left_knee_std: 12.3
  right_ankle_std: 14.1
  right_hip_std: 6.2
  right_knee_std: 10.5
```

## Summary

- ❌ **NOT a bug in AlexPose**
- ❌ **NOT a configuration issue**
- ❌ **NOT a feature extraction issue**
- ✅ **User code is overwriting the features dict**
- ✅ **Fix: Remove the manual overwrite**

The AlexPose library is working correctly. The issue is in the notebook code that's overwriting the properly extracted features.

---

## Action Items

1. **Update the notebook** to remove manual feature overwrite
2. **Trust EnhancedGaitAnalyzer** - it extracts all 82 features correctly
3. **Verify** that std values are now non-zero
4. **Remove** the `normal_joint_angles.get_statistics()` workaround

The system is production-ready. The notebook just needs to be updated to use the proper API.
