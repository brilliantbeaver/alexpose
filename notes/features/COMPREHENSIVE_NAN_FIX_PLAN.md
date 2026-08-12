# Comprehensive NaN Fix Plan - Deep Root Cause Analysis

## Current Situation

**Error:** Sample `cljas5esv00fn3n6lewd5xqdl` (index 53) has NaN in ALL 15 features.

**Critical Insight:** The fix was applied to `from_joint_angles()`, but the error persists. This means either:
1. The notebook is using cached/old code (cells run before fix)
2. There's a different code path creating features
3. The joint_angles object itself is problematic

## Deep Root Cause Analysis

### Why ALL Features Are NaN

When ALL features are NaN, it means:
```python
joint_angles.get_statistics("left_hip")  # Returns {"mean": NaN, "std": NaN, ...}
joint_angles.get_statistics("left_knee") # Returns {"mean": NaN, "std": NaN, ...}
# ... all joints return NaN
```

This happens when:
1. **Empty keypoints array** passed to `get_joint_angles()`
2. **All frames have no valid angles** (all angles are NaN)
3. **Joint angle calculation completely failed**

### The Real Problem

Looking at the error, sample `cljas5esv00fn3n6lewd5xqdl` likely has:
- Keypoint extraction succeeded (some frames extracted)
- BUT joint angle calculation failed for ALL frames
- OR keypoints array was empty but no error was raised

## Comprehensive Fix Strategy

### Phase 1: Prevent NaN at Source (Joint Angles)

**Problem:** `get_joint_angles()` should not create sequences with all-NaN angles

**Solution:** Add validation in `get_joint_angles()` to warn/error when no valid angles

### Phase 2: Validate Before Feature Creation

**Problem:** No check if joint_angles has any valid data before creating features

**Solution:** Add validation method to JointAngleSequence

### Phase 3: Automatic Filtering

**Problem:** User has to manually check each sample

**Solution:** Provide automatic filtering of invalid samples

### Phase 4: Better Error Messages

**Problem:** Error doesn't tell user HOW to fix it

**Solution:** Provide actionable guidance with sample-specific diagnostics

## Implementation Plan

### Fix 1: Add Validation to JointAngleSequence

**File:** `ambient/pose/joint_angles.py`

```python
def has_valid_data(self) -> bool:
    """
    Check if sequence has any valid angle data.
    
    Returns:
        True if at least one joint has at least one valid angle
    """
    for frame in self.frames:
        if frame.angles:  # Has any angles
            for angle_obj in frame.angles.values():
                if angle_obj and not np.isnan(angle_obj.angle_degrees):
                    return True
    return False

def get_valid_frame_count(self) -> int:
    """Get count of frames with at least one valid angle."""
    count = 0
    for frame in self.frames:
        if any(not np.isnan(a.angle_degrees) for a in frame.angles.values() if a):
            count += 1
    return count
```

### Fix 2: Validate in from_joint_angles()

**File:** `ambient/classification/knn_classifier.py`

```python
@classmethod
def from_joint_angles(
    cls, joint_angle_sequence, sample_id: str = "", condition_label: str = ""
) -> Optional["GaitFeatureVector"]:
    """
    Create feature vector from JointAngleSequence.
    
    Returns:
        GaitFeatureVector with computed features, or None if no valid data
    """
    # Check if sequence has any valid data
    if hasattr(joint_angle_sequence, 'has_valid_data'):
        if not joint_angle_sequence.has_valid_data():
            logger.warning(
                f"Sample {sample_id}: No valid angle data in sequence. "
                f"Frames: {len(joint_angle_sequence.frames)}, "
                f"Valid frames: {joint_angle_sequence.get_valid_frame_count()}"
            )
            return None
    
    # ... rest of existing code with safe_get()
```

### Fix 3: Filter Invalid Features Automatically

**File:** `ambient/classification/knn_classifier.py`

```python
@staticmethod
def create_features_from_sequences(
    sequences: List[Tuple[JointAngleSequence, str, str]],
    skip_invalid: bool = True
) -> Tuple[List[GaitFeatureVector], List[str]]:
    """
    Create feature vectors from multiple joint angle sequences.
    
    Args:
        sequences: List of (joint_angles, sample_id, condition_label) tuples
        skip_invalid: If True, skip sequences with no valid data
        
    Returns:
        Tuple of (valid_features, skipped_sample_ids)
    """
    features = []
    skipped = []
    
    for joint_angles, sample_id, condition in sequences:
        feature = GaitFeatureVector.from_joint_angles(
            joint_angles, sample_id, condition
        )
        
        if feature is None:
            skipped.append(sample_id)
            if skip_invalid:
                logger.warning(f"Skipping sample {sample_id}: no valid data")
                continue
            else:
                raise ValueError(f"Sample {sample_id} has no valid angle data")
        
        # Double-check for NaN
        if np.any(np.isnan(feature.to_array())):
            skipped.append(sample_id)
            if skip_invalid:
                logger.warning(f"Skipping sample {sample_id}: contains NaN")
                continue
            else:
                raise ValueError(f"Sample {sample_id} contains NaN values")
        
        features.append(feature)
    
    if skipped:
        logger.info(f"Skipped {len(skipped)} invalid samples: {skipped[:5]}")
    
    return features, skipped
```

### Fix 4: Enhanced train() Method

**File:** `ambient/classification/knn_classifier.py`

```python
def train(
    self,
    features: List[GaitFeatureVector],
    labels: Optional[List[str]] = None,
    validate: bool = True,
    auto_remove_invalid: bool = False  # NEW PARAMETER
) -> Dict[str, Any]:
    """
    Train the KNN classifier.
    
    Args:
        auto_remove_invalid: If True, automatically remove samples with NaN/Inf
    """
    # ... existing validation code ...
    
    # Check for NaN values
    nan_mask = np.isnan(X)
    if np.any(nan_mask):
        nan_samples = np.where(nan_mask.any(axis=1))[0]
        
        if auto_remove_invalid:
            # Remove invalid samples
            valid_mask = ~nan_mask.any(axis=1)
            X = X[valid_mask]
            y = y[valid_mask]
            
            removed_ids = [features[i].sample_id for i in nan_samples]
            logger.warning(
                f"Automatically removed {len(nan_samples)} samples with NaN: "
                f"{removed_ids[:5]}"
            )
            
            if len(X) < 5:
                raise ValueError(
                    f"After removing invalid samples, only {len(X)} remain. "
                    f"Need at least 5 for training."
                )
        else:
            # Existing detailed error message
            # ... existing code ...
            raise ValueError("Training data contains NaN values. See log for details.")
```

## Immediate User Workaround

### Option 1: Restart Kernel and Re-run

```python
# In Jupyter notebook
# 1. Kernel -> Restart Kernel
# 2. Run all cells from beginning
# This ensures the fixed code is loaded
```

### Option 2: Manual Filtering

```python
# Before training, filter out invalid samples
valid_features = []
invalid_samples = []

for feature in train_features:
    feature_array = feature.to_array()
    if np.any(np.isnan(feature_array)) or np.any(np.isinf(feature_array)):
        invalid_samples.append(feature.sample_id)
        print(f"⚠️  Skipping invalid sample: {feature.sample_id}")
    else:
        valid_features.append(feature)

print(f"\nFiltered: {len(valid_features)} valid, {len(invalid_samples)} invalid")
print(f"Invalid samples: {invalid_samples}")

# Train with valid features only
classifier.train(valid_features, validate=True)
```

### Option 3: Investigate Specific Sample

```python
# Find the problematic sample
problem_sample_id = "cljas5esv00fn3n6lewd5xqdl"

# Check if keypoints were extracted
csv_file = data_root / "cerebralpalsy" / f"{problem_sample_id}.csv"
df = loader.load_gavd_data(csv_file)

keypoints = extractor.extract_from_sequence(
    df, video_base_path,
    verbose=True,
    filter_empty=False  # Don't filter to see what's happening
)

print(f"\nKeypoints extracted: {len(keypoints)}")
if keypoints:
    extractor.print_extraction_statistics(keypoints, problem_sample_id)
    
    # Try joint angles
    joint_angles = get_joint_angles(keypoints, "BLAZEPOSE_33", fps=30.0)
    print(f"Frames with angles: {len(joint_angles.frames)}")
    
    # Check statistics
    for joint in ["left_hip", "left_knee", "left_ankle"]:
        stats = joint_angles.get_statistics(joint)
        print(f"{joint}: mean={stats['mean']}, valid_count={stats['valid_count']}")
else:
    print("⚠️  No keypoints extracted - this is the problem!")
```

## Root Cause Checklist

For sample `cljas5esv00fn3n6lewd5xqdl`, check:

1. **Video file exists?**
   ```bash
   ls data/youtube/wRntYsztIEY.mp4  # Extract video ID from CSV
   ```

2. **Keypoint extraction succeeded?**
   - Check if `extract_from_sequence()` returned empty array
   - Check extraction logs for errors

3. **Joint angles calculated?**
   - Check if `get_joint_angles()` created valid angles
   - Verify frames have angle data

4. **Feature creation?**
   - Check if `from_joint_angles()` was called with valid data
   - Verify safe_get() is being used

## Prevention Strategy

### 1. Validate at Each Step

```python
# Step 1: Extract keypoints
keypoints = extractor.extract_from_sequence(...)
if not keypoints:
    logger.error(f"No keypoints for {sample_id}")
    continue  # Skip this sample

# Step 2: Calculate angles
joint_angles = get_joint_angles(keypoints, ...)
if not joint_angles.has_valid_data():  # NEW METHOD
    logger.error(f"No valid angles for {sample_id}")
    continue  # Skip this sample

# Step 3: Create feature
feature = GaitFeatureVector.from_joint_angles(...)
if feature is None:  # NEW: returns None if invalid
    logger.error(f"Could not create feature for {sample_id}")
    continue  # Skip this sample

# Step 4: Validate feature
if np.any(np.isnan(feature.to_array())):
    logger.error(f"Feature has NaN for {sample_id}")
    continue  # Skip this sample

# Only add if all checks pass
train_features.append(feature)
```

### 2. Use Batch Processing Helper

```python
# NEW: Use helper method
sequences = [
    (joint_angles1, "sample1", "normal"),
    (joint_angles2, "sample2", "stroke"),
    # ...
]

valid_features, skipped = KNNGaitClassifier.create_features_from_sequences(
    sequences,
    skip_invalid=True  # Automatically skip bad samples
)

print(f"Created {len(valid_features)} features, skipped {len(skipped)}")
```

## Testing Plan

### Test 1: Empty Keypoints
```python
# Simulate empty extraction
empty_keypoints = []
joint_angles = get_joint_angles(empty_keypoints, "BLAZEPOSE_33", fps=30.0)
feature = GaitFeatureVector.from_joint_angles(joint_angles, "test", "test")
assert feature is None or not np.any(np.isnan(feature.to_array()))
```

### Test 2: All-NaN Angles
```python
# Simulate failed angle calculation
# (create sequence with empty angles dict)
```

### Test 3: Mixed Valid/Invalid
```python
# Test automatic filtering
```

## Summary

**The core issue:** The fix was applied, but:
1. User may be running old code (notebook cells executed before fix)
2. Need validation at EVERY step, not just in feature creation
3. Need automatic filtering options
4. Need better diagnostics for specific samples

**Immediate action:**
1. Restart kernel and re-run notebook
2. Add manual filtering before training
3. Investigate the specific problematic sample

**Long-term solution:**
1. Add `has_valid_data()` to JointAngleSequence
2. Make `from_joint_angles()` return None for invalid data
3. Add batch processing helper with automatic filtering
4. Add `auto_remove_invalid` parameter to `train()`
