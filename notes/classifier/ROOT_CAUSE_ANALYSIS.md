# Root Cause Analysis: Why New Features Show 0.00

## Issue Summary

When using `GaitFeatureVector.from_joint_angles()` in notebooks, all 19 new enhanced features (spatiotemporal, temporal phases, symmetry indices, variability, postural) display as 0.00, while the 15 core joint angle features show correct values.

## Root Cause

The issue stems from using the **wrong factory method** for feature extraction. There are two different factory methods with different purposes:

### Method 1: `from_joint_angles()` - LEGACY (15 features)

```python
feature_vector = GaitFeatureVector.from_joint_angles(
    joint_angle_sequence,  # JointAngleSequence object
    sample_id="sample_001",
    condition_label="normal"
)
```

**What it does:**
- Takes a `JointAngleSequence` object (from `get_joint_angles()`)
- Extracts ONLY 15 core joint angle features:
  - Mean angles (6): left/right hip, knee, ankle
  - Asymmetry (3): hip, knee, ankle differences
  - Range of motion (6): left/right hip, knee, ankle ranges
- All other 19 features default to 0.0
- Maintained for backward compatibility

**Source code location:** `ambient/classification/features.py:370-448`

```python
@classmethod
def from_joint_angles(cls, joint_angle_sequence, sample_id="", condition_label=""):
    # Extract statistics for each joint
    left_hip_stats = joint_angle_sequence.get_statistics("left_hip")
    # ... only extracts joint angle statistics
    
    return cls(
        left_hip_mean=left_hip_mean,
        # ... only 15 core features
        # All new features default to 0.0
    )
```

### Method 2: `from_analysis_results()` - ENHANCED (34 features)

```python
feature_vector = GaitFeatureVector.from_analysis_results(
    analysis_results,  # Dictionary from EnhancedGaitAnalyzer
    sample_id="sample_001",
    condition_label="normal"
)
```

**What it does:**
- Takes a dictionary from `EnhancedGaitAnalyzer.analyze_gait_sequence()`
- Extracts ALL 34 features from multiple analyzer components:
  - FeatureExtractor: kinematic, joint angles, temporal, stride, symmetry, stability
  - TemporalAnalyzer: gait cycles, timing, phase features, variability
  - SymmetryAnalyzer: positional, movement, temporal, angular symmetry
- Provides comprehensive gait analysis

**Source code location:** `ambient/classification/features.py:450-690`

```python
@classmethod
def from_analysis_results(cls, analysis_results, sample_id="", condition_label=""):
    # Extract from multiple analyzer components
    features_dict = analysis_results.get("features", {})
    timing_analysis = analysis_results.get("timing_analysis", {})
    phase_features = analysis_results.get("phase_features", {})
    symmetry_analysis = analysis_results.get("symmetry_analysis", {})
    
    # Extract all 34 features
    # ... comprehensive feature extraction
```

## Why the Confusion?

The notebook code was using the legacy workflow:

```python
# Step 1: Get joint angles
joint_angles = get_joint_angles(keypoints_array, ...)

# Step 2: Extract features using LEGACY method
feature_vector = GaitFeatureVector.from_joint_angles(joint_angles, ...)
# ❌ This only extracts 15 features!
```

When trying to fix it by using `from_analysis_results()`:

```python
# ❌ WRONG: Passing JointAngleSequence to from_analysis_results()
feature_vector = GaitFeatureVector.from_analysis_results(
    normal_joint_angles,  # This is a JointAngleSequence, not a dict!
    sample_id=normal_sid,
    condition_label="normal"
)
```

**Error:** `'JointAngleSequence' object has no attribute 'get'`

This fails because `from_analysis_results()` expects a dictionary with keys like `"features"`, `"timing_analysis"`, etc., but receives a `JointAngleSequence` object instead.

## The Correct Workflow

### Legacy Workflow (15 features)
```
Keypoints Array
    ↓
get_joint_angles()
    ↓
JointAngleSequence
    ↓
from_joint_angles()
    ↓
GaitFeatureVector (15 features)
```

### Enhanced Workflow (34 features)
```
Keypoints Array
    ↓
Convert to pose sequence format
    ↓
EnhancedGaitAnalyzer.analyze_gait_sequence()
    ↓
Analysis Results Dictionary
    ↓
from_analysis_results()
    ↓
GaitFeatureVector (34 features)
```

## Solution

Use the helper function that handles the conversion:

```python
from ambient.utils.notebook_helpers import extract_enhanced_features_from_keypoints

# One-step extraction of all 34 features
feature_vector = extract_enhanced_features_from_keypoints(
    keypoints_array,
    sample_id="sample_001",
    condition_label="normal"
)
```

**What the helper does:**
1. Converts `KeypointData` array to pose sequence format
2. Runs `EnhancedGaitAnalyzer.analyze_gait_sequence()`
3. Calls `from_analysis_results()` with the dictionary
4. Returns `GaitFeatureVector` with all 34 features

## Technical Details

### Why from_joint_angles() Only Extracts 15 Features

The `JointAngleSequence` object only contains:
- Frame-by-frame joint angles
- Statistical methods: `get_statistics()`, `get_joint_angle_series()`

It does NOT contain:
- Gait cycle information
- Temporal phase data
- Symmetry analysis results
- Variability metrics
- Postural features

These require the full analysis pipeline with:
- `FeatureExtractor` for spatiotemporal parameters
- `TemporalAnalyzer` for gait cycle detection and phase analysis
- `SymmetryAnalyzer` for evidence-based symmetry indices

### Why from_analysis_results() Needs a Dictionary

The `from_analysis_results()` method expects a dictionary with this structure:

```python
{
    "features": {
        "left_hip_mean": 45.0,
        "walking_speed_pixels_per_sec": 120.0,
        "estimated_cadence": 110.0,
        # ... 62 raw features from FeatureExtractor
    },
    "timing_analysis": {
        "cadence_steps_per_minute": 110.0,
        "stride_time_cv": 0.05,
        # ... 25 timing features from TemporalAnalyzer
    },
    "phase_features": {
        "stance_percentage_mean": 60.0,
        "swing_percentage_mean": 40.0,
        # ... phase features
    },
    "symmetry_analysis": {
        "stride_length_si": 8.5,
        "hip_angle_si": 5.2,
        # ... 54 symmetry features from SymmetryAnalyzer
    }
}
```

This dictionary is produced by `EnhancedGaitAnalyzer.analyze_gait_sequence()`.

## Design Rationale

### Why Keep Two Methods?

1. **Backward Compatibility**: Existing code using `from_joint_angles()` continues to work
2. **Gradual Migration**: Users can migrate at their own pace
3. **Flexibility**: Some use cases only need basic joint angles
4. **Clear Separation**: Legacy vs enhanced workflows are explicit

### Why Not Auto-Upgrade from_joint_angles()?

We considered making `from_joint_angles()` automatically run the enhanced analysis, but decided against it because:

1. **Breaking Changes**: Would change behavior of existing code
2. **Performance**: Enhanced analysis is slower (9ms vs <1ms)
3. **Dependencies**: Would require converting data formats
4. **Clarity**: Explicit methods make the difference clear

## Verification

To verify you're getting enhanced features:

```python
# Check feature count
X = feature_vector.to_array()
print(f"Feature count: {len(X)}")
# Should print: Feature count: 34

# Check specific new features
print(f"Walking speed: {feature_vector.walking_speed_ms:.2f} m/s")
print(f"Cadence: {feature_vector.cadence_steps_min:.1f} steps/min")
print(f"Stance %: {feature_vector.stance_percentage:.1f}%")
print(f"Stride length SI: {feature_vector.stride_length_si:.1f}%")

# If these are all 0.00, you're using the legacy method
```

## Summary

| Aspect | Legacy Method | Enhanced Method |
|--------|--------------|-----------------|
| Function | `from_joint_angles()` | `from_analysis_results()` |
| Input | `JointAngleSequence` | Dictionary from analyzer |
| Features | 15 core angles | 34 comprehensive |
| Speed | <1ms | ~9ms |
| Use Case | Basic joint angles | Full gait analysis |
| Helper | None | `extract_enhanced_features_from_keypoints()` |

**Bottom Line:** Use `extract_enhanced_features_from_keypoints()` in notebooks to get all 34 features with real values instead of 0.00.
