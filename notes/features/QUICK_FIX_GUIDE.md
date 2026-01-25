# Quick Fix Guide for NaN Errors

## Immediate Solution (Use This Now!)

### Option 1: Auto-Remove Invalid Samples (Recommended)

```python
# Simply add auto_remove_invalid=True to your training call
classifier = KNNGaitClassifier(config=config)
metrics = classifier.train(
    train_features, 
    validate=True,
    auto_remove_invalid=True  # ← Add this parameter
)
```

This will:
- Automatically detect and remove samples with NaN values
- Log which samples were removed
- Continue training with valid samples only
- Warn if too few samples remain

### Option 2: Manual Filtering (More Control)

```python
# Filter out invalid samples before training
valid_features = []
invalid_samples = []

for feature in train_features:
    # Check if feature is None (from failed extraction)
    if feature is None:
        continue
    
    # Check for NaN/Inf in feature array
    feature_array = feature.to_array()
    if np.any(np.isnan(feature_array)) or np.any(np.isinf(feature_array)):
        invalid_samples.append(feature.sample_id)
    else:
        valid_features.append(feature)

print(f"Valid: {len(valid_features)}, Invalid: {len(invalid_samples)}")
if invalid_samples:
    print(f"Skipped samples: {invalid_samples}")

# Train with valid features
classifier.train(valid_features, validate=True)
```

## Why This Happens

Sample `cljas5esv00fn3n6lewd5xqdl` has NaN in all features because:
1. Keypoint extraction may have failed
2. Joint angle calculation returned all NaN
3. The sequence has no valid angle data

## Prevention (For Future Runs)

### Step 1: Check Keypoints

```python
keypoints = extractor.extract_from_sequence(
    df, video_base_path,
    filter_empty=True,
    min_keypoints=25
)

# ALWAYS check if extraction succeeded
if not keypoints:
    print(f"⚠️  Skipping {sample_id}: no keypoints extracted")
    continue  # Skip this sample
```

### Step 2: Validate Joint Angles

```python
joint_angles = get_joint_angles(keypoints, "BLAZEPOSE_33", fps=30.0)

# NEW: Check if sequence has valid data
if not joint_angles.has_valid_data():
    print(f"⚠️  Skipping {sample_id}: no valid angles")
    continue  # Skip this sample
```

### Step 3: Check Feature Creation

```python
feature = GaitFeatureVector.from_joint_angles(
    joint_angles,
    sample_id=sample_id,
    condition_label=condition
)

# NEW: from_joint_angles() returns None if no valid data
if feature is None:
    print(f"⚠️  Skipping {sample_id}: could not create feature")
    continue  # Skip this sample

train_features.append(feature)
```

## Complete Example

```python
from ambient.pose.keypoint_extractor import SequenceKeypointExtractor
from ambient.pose.joint_angles import get_joint_angles
from ambient.classification.knn_classifier import (
    KNNGaitClassifier,
    KNNClassifierConfig
)
from ambient.classification.features import GaitFeatureVector

# Process all samples
train_features = []
skipped_samples = []

for csv_file in csv_files:
    sample_id = csv_file.stem
    df = loader.load_gavd_data(csv_file)
    
    # Extract keypoints
    keypoints = extractor.extract_from_sequence(
        df, video_base_path,
        filter_empty=True,
        min_keypoints=25
    )
    
    if not keypoints:
        skipped_samples.append((sample_id, "no_keypoints"))
        continue
    
    # Calculate joint angles
    joint_angles = get_joint_angles(keypoints, "BLAZEPOSE_33", fps=30.0)
    
    # Check if valid
    if not joint_angles.has_valid_data():
        skipped_samples.append((sample_id, "no_valid_angles"))
        continue
    
    # Create feature
    feature = GaitFeatureVector.from_joint_angles(
        joint_angles,
        sample_id=sample_id,
        condition_label=condition
    )
    
    if feature is None:
        skipped_samples.append((sample_id, "feature_creation_failed"))
        continue
    
    train_features.append(feature)

print(f"\nProcessed: {len(train_features)} valid, {len(skipped_samples)} skipped")
if skipped_samples:
    print("\nSkipped samples:")
    for sid, reason in skipped_samples[:10]:
        print(f"  {sid}: {reason}")

# Train classifier
if len(train_features) >= 5:
    config = KNNClassifierConfig(
        n_neighbors=5,
        weights="distance",
        metric="euclidean",
        normalize_features=True
    )
    
    classifier = KNNGaitClassifier(config=config)
    metrics = classifier.train(
        train_features,
        validate=True,
        auto_remove_invalid=True  # Extra safety net
    )
    
    print(f"\nTraining Results:")
    print(f"  Accuracy: {metrics['train_accuracy']:.3f}")
    print(f"  Samples: {metrics['n_samples']}")
else:
    print(f"⚠️  Not enough valid samples: {len(train_features)}")
```

## Restart Kernel

**IMPORTANT:** If you've already run cells in your notebook:

1. **Kernel → Restart Kernel** (or Restart & Run All)
2. This ensures the updated code is loaded
3. Re-run all cells from the beginning

## Investigate Specific Sample

To understand why `cljas5esv00fn3n6lewd5xqdl` failed:

```python
problem_id = "cljas5esv00fn3n6lewd5xqdl"
csv_file = data_root / "cerebralpalsy" / f"{problem_id}.csv"

# Load and check
df = loader.load_gavd_data(csv_file)
print(f"CSV rows: {len(df)}")
print(f"Video ID: {df.iloc[0]['id']}")

# Try extraction
keypoints = extractor.extract_from_sequence(
    df, video_base_path,
    verbose=True,
    filter_empty=False  # Don't filter to see what happens
)

print(f"\nKeypoints: {len(keypoints)}")
if keypoints:
    extractor.print_extraction_statistics(keypoints, problem_id)
    
    # Try angles
    angles = get_joint_angles(keypoints, "BLAZEPOSE_33", fps=30.0)
    print(f"\nAngle frames: {len(angles.frames)}")
    print(f"Has valid data: {angles.has_valid_data()}")
    print(f"Valid frames: {angles.get_valid_frame_count()}")
    
    # Check each joint
    for joint in ["left_hip", "left_knee", "left_ankle"]:
        stats = angles.get_statistics(joint)
        print(f"{joint}: valid_count={stats['valid_count']}, mean={stats['mean']}")
```

## Summary

**Immediate fix:** Add `auto_remove_invalid=True` to your `train()` call.

**Long-term fix:** Add validation checks at each step (keypoints → angles → features).

**Root cause:** Sample has no valid angle data, likely due to failed keypoint extraction or joint angle calculation.
