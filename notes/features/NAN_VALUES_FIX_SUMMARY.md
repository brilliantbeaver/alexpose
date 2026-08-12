# NaN Values in Training Data - Root Cause Analysis and Solution

## Problem Summary

The KNN classifier training fails with:
```
ValueError: Input X contains NaN. KNeighborsClassifier does not accept missing values encoded as NaN natively.
```

This occurs when feature vectors contain NaN (Not a Number) values, which sklearn's KNeighborsClassifier cannot handle.

## Root Cause Analysis

### The Data Flow

1. **Keypoint Extraction** → `extract_from_sequence()` returns KeypointSet objects
2. **Joint Angle Calculation** → `get_joint_angles()` calculates angles from keypoints
3. **Statistics Calculation** → `get_statistics()` computes mean, std, range, etc.
4. **Feature Vector Creation** → `from_joint_angles()` creates GaitFeatureVector
5. **Training** → `train()` converts features to numpy array and trains model

### Where NaN Values Originate

#### Source 1: Empty Keypoint Extraction
When `extract_from_sequence()` returns an empty array (all frames failed):
```python
keypoints_array = []  # No frames extracted
joint_angles = get_joint_angles(keypoints_array, ...)  # Creates empty sequence
stats = joint_angles.get_statistics("left_hip")  # Returns {"mean": np.nan, ...}
```

#### Source 2: No Valid Angles
When joint angle calculation fails for all frames:
```python
# All angles are NaN due to missing keypoints
valid_angles = angles[~np.isnan(angles)]  # Empty array
if len(valid_angles) == 0:
    return {"mean": np.nan, "std": np.nan, ...}  # ← NaN values returned
```

#### Source 3: Insufficient Keypoints
When pose detection succeeds but specific joints are not visible:
```python
# Example: Person filmed from side, one hip not visible
left_hip_angles = [NaN, NaN, NaN, ...]  # All NaN
stats = get_statistics("left_hip")  # Returns NaN
```

### The Bug in `from_joint_angles()`

**Original Code:**
```python
left_hip_mean=left_hip_stats.get("mean", 0)  # ← BUG!
```

**Problem:** `.get("mean", 0)` returns the value if key exists, even if it's NaN!
- Key "mean" exists in the dict
- Value is `np.nan`
- Default value `0` is never used
- NaN propagates to feature vector

## Fixes Applied

### Fix 1: Safe Value Extraction in `from_joint_angles()`

**File:** `ambient/classification/knn_classifier.py`

Added `safe_get()` helper function:
```python
def safe_get(stats_dict, key, default=0.0):
    """Get value from dict, replacing NaN with default"""
    value = stats_dict.get(key, default)
    return default if np.isnan(value) else value

# Usage
left_hip_mean = safe_get(left_hip_stats, "mean")  # Returns 0 if NaN
```

**Why this works:**
- Explicitly checks for NaN values
- Replaces NaN with 0 (neutral value for angles)
- Prevents NaN from entering feature vectors

### Fix 2: Comprehensive Validation in `train()`

**File:** `ambient/classification/knn_classifier.py`

Added detailed NaN detection and reporting:
```python
# Check for NaN values
nan_mask = np.isnan(X)
if np.any(nan_mask):
    nan_samples = np.where(nan_mask.any(axis=1))[0]
    nan_features = np.where(nan_mask.any(axis=0))[0]
    
    # Detailed error message with:
    # - Which samples have NaN
    # - Which features have NaN
    # - Sample IDs if available
    # - Common causes and solutions
    raise ValueError("Training data contains NaN values. See log for details.")
```

**Benefits:**
- Catches any NaN that slips through
- Provides actionable diagnostics
- Shows which samples and features are affected
- Suggests solutions

### Fix 3: Feature Validation Helper

**File:** `ambient/classification/knn_classifier.py`

Added `validate_features()` static method:
```python
valid_features, invalid_indices = KNNGaitClassifier.validate_features(
    features, 
    remove_invalid=True  # Automatically remove bad samples
)
```

**Features:**
- Validates all feature vectors
- Detects NaN and Inf values
- Can remove invalid samples automatically
- Returns indices of invalid samples for debugging

## How to Use

### Recommended Workflow

```python
from ambient.pose.keypoint_extractor import SequenceKeypointExtractor
from ambient.pose.joint_angles import get_joint_angles
from ambient.classification.knn_classifier import (
    KNNGaitClassifier,
    GaitFeatureVector
)

# 1. Extract keypoints
extractor = SequenceKeypointExtractor()
keypoints_array = extractor.extract_from_sequence(
    sequence_data=df,
    video_base_path=video_base_path,
    filter_empty=True,
    min_keypoints=25
)

# 2. Check if extraction succeeded
if not keypoints_array:
    print(f"⚠️  WARNING: No keypoints extracted for {sample_id}")
    continue  # Skip this sample

# 3. Calculate joint angles
joint_angles = get_joint_angles(
    keypoints_array=keypoints_array,
    keypoint_format="BLAZEPOSE_33",
    fps=30.0
)

# 4. Create feature vector (now handles NaN safely)
feature_vector = GaitFeatureVector.from_joint_angles(
    joint_angles,
    sample_id=sample_id,
    condition_label=condition
)

# 5. Collect all features
all_features.append(feature_vector)

# 6. Validate before training (optional but recommended)
valid_features, invalid_indices = KNNGaitClassifier.validate_features(
    all_features,
    remove_invalid=True  # Remove invalid samples
)

if invalid_indices:
    print(f"⚠️  Removed {len(invalid_indices)} invalid samples")
    print(f"   Invalid sample IDs: {[all_features[i].sample_id for i in invalid_indices[:5]]}")

# 7. Train classifier
classifier = KNNGaitClassifier()
metrics = classifier.train(valid_features, validate=True)
```

### Handling Empty Extractions

```python
# Process multiple samples
train_features = []

for csv_file in csv_files:
    df = loader.load_gavd_data(csv_file)
    
    # Extract keypoints
    keypoints = extractor.extract_from_sequence(
        df, video_base_path, 
        filter_empty=True, 
        min_keypoints=25
    )
    
    # ALWAYS check if extraction succeeded
    if not keypoints:
        logger.warning(f"Skipping {csv_file.name}: no valid keypoints")
        continue
    
    # Calculate angles
    angles = get_joint_angles(keypoints, "BLAZEPOSE_33", fps=30.0)
    
    # Create feature (NaN-safe)
    feature = GaitFeatureVector.from_joint_angles(
        angles,
        sample_id=csv_file.stem,
        condition_label=condition
    )
    
    train_features.append(feature)

# Validate and train
if len(train_features) < 5:
    raise ValueError(f"Insufficient training samples: {len(train_features)}")

valid_features, _ = KNNGaitClassifier.validate_features(
    train_features, 
    remove_invalid=True
)

classifier.train(valid_features)
```

## Common Scenarios and Solutions

### Scenario 1: All Samples Have NaN

**Symptom:**
```
ValueError: Training data contains NaN values in 50 samples
```

**Cause:** Keypoint extraction failed for all samples

**Solutions:**
1. Check video files exist: `ls data/youtube/*.mp4`
2. Verify video_base_path is correct
3. Check extraction logs for errors
4. Try lowering `min_keypoints` threshold

### Scenario 2: Some Samples Have NaN

**Symptom:**
```
WARNING: Removed 5 invalid feature vectors out of 50
```

**Cause:** Extraction failed for specific samples

**Solutions:**
1. Use `validate_features(remove_invalid=True)` to auto-remove
2. Check which samples failed (logged with sample_id)
3. Investigate those specific videos
4. Ensure minimum sample count after removal

### Scenario 3: Specific Features Always NaN

**Symptom:**
```
Affected features: ['left_ankle_mean', 'left_ankle_range']
```

**Cause:** Specific joint not visible in videos

**Solutions:**
1. Check camera angle (e.g., ankles not visible from certain angles)
2. Consider using different features
3. Verify pose estimation model quality
4. Check if videos are cropped

## Prevention Best Practices

### 1. Always Validate Extraction Results

```python
if not keypoints_array:
    logger.error(f"Extraction failed for {sample_id}")
    continue  # Don't create feature vector
```

### 2. Use Feature Validation

```python
# Before training
valid_features, invalid = KNNGaitClassifier.validate_features(
    features, 
    remove_invalid=True
)
```

### 3. Check Sample Counts

```python
if len(valid_features) < 5:
    raise ValueError("Need at least 5 samples for training")
```

### 4. Log Extraction Statistics

```python
if keypoints_array:
    extractor.print_extraction_statistics(keypoints_array, sample_id)
else:
    logger.warning(f"No keypoints for {sample_id}")
```

### 5. Handle Edge Cases

```python
# Check for all-zero features (might indicate issues)
feature_array = feature.to_array()
if np.all(feature_array == 0):
    logger.warning(f"All-zero features for {sample_id}")
```

## Testing

Create a test to verify NaN handling:

```python
def test_nan_handling():
    """Test that NaN values are handled correctly"""
    from ambient.classification.knn_classifier import GaitFeatureVector
    from ambient.pose.joint_angles import JointAngleSequence, JointAngleFrame
    
    # Create sequence with no valid angles (all NaN)
    frames = [
        JointAngleFrame(
            frame_number=i,
            timestamp=i/30.0,
            angles={}  # Empty angles
        )
        for i in range(10)
    ]
    
    sequence = JointAngleSequence(
        sequence_id="test",
        keypoint_format="BLAZEPOSE_33",
        fps=30.0,
        frames=frames
    )
    
    # Create feature vector
    feature = GaitFeatureVector.from_joint_angles(
        sequence,
        sample_id="test",
        condition_label="test"
    )
    
    # Verify no NaN in feature array
    feature_array = feature.to_array()
    assert not np.any(np.isnan(feature_array)), "Feature vector contains NaN"
    assert not np.any(np.isinf(feature_array)), "Feature vector contains Inf"
    
    print("✓ NaN handling test passed")

test_nan_handling()
```

## Files Modified

1. **`ambient/classification/knn_classifier.py`**
   - `from_joint_angles()`: Added `safe_get()` helper to handle NaN
   - `train()`: Added comprehensive NaN validation and diagnostics
   - `validate_features()`: New static method for feature validation

2. **`NAN_VALUES_FIX_SUMMARY.md`**: This documentation

## Summary

The NaN issue was caused by:
1. `get_statistics()` returning NaN when no valid angles exist
2. `from_joint_angles()` not checking for NaN values
3. No validation before training

The fix:
1. Safely extracts values, replacing NaN with 0
2. Validates training data with detailed diagnostics
3. Provides helper methods for feature validation
4. Gives clear error messages and solutions

**Key Takeaway:** Always check if keypoint extraction succeeded before creating feature vectors, and validate features before training.
