# Why Frames Have 0 Keypoints - Complete Guide

## The Issue

You're seeing frames with `len(keypoints) == 0` in your `normal_keypoints_array`. This is **expected behavior**, not a bug.

## Root Cause

From the diagnostic, we found:
- **80% of frames (8 out of 10) had 0 keypoints**
- **Bounding box extends beyond frame**: `left: 489` but frame width is only `494`
- This means the person is **partially or completely out of frame**

## Why MediaPipe Returns 0 Keypoints

MediaPipe returns an empty keypoint set when:

1. **Person not visible** - Out of frame, occluded, or behind objects
2. **Poor lighting** - Too dark or too bright
3. **Person too small** - Too far from camera
4. **Low quality** - Blurry, pixelated, or compressed
5. **Unusual pose** - Lying down, bent over, or extreme angles
6. **Multiple people** - MediaPipe picks strongest detection, may miss target
7. **Bad bounding box** - GAVD bbox data may be incorrect or outdated

## Your Specific Case

```
Frame 0: bbox = {left: 489, width: 145}
Frame width: 494 pixels
Person position: 489 + 145 = 634 pixels (extends 140 pixels beyond frame!)
```

The person is **mostly out of frame** on the right side, so MediaPipe correctly returns 0 keypoints.

## Solution: Filter Valid Frames

### Option 1: Use the Filtering Helper (Recommended)

```python
# Load the helper
exec(open('filter_valid_frames.py').read())

# Extract with automatic filtering
normal_keypoints_array, normal_df_filtered, stats = extract_with_filtering(
    normal_df, 
    video_base_path, 
    extractor,
    min_keypoints=25,  # Require at least 25 out of 33 keypoints
    verbose=True
)

# Now normal_keypoints_array only contains valid frames
# And normal_df_filtered is aligned with it (same indices)
```

This will:
- Extract keypoints from all frames
- Show detailed statistics
- Filter to only frames with ≥25 keypoints
- Return aligned keypoints and DataFrame

### Option 2: Manual Filtering

```python
# Extract all frames
all_keypoints = extractor.extract_from_sequence(
    sequence_data=normal_df,
    video_base_path=video_base_path,
    verbose=True
)

# Filter to valid frames
valid_keypoints = []
valid_indices = []

for i, kp_set in enumerate(all_keypoints):
    if kp_set is not None and len(kp_set.keypoints) >= 25:
        valid_keypoints.append(kp_set)
        valid_indices.append(i)

# Filter DataFrame to match
normal_df_filtered = normal_df.iloc[valid_indices].reset_index(drop=True)

print(f"Kept {len(valid_keypoints)} out of {len(all_keypoints)} frames")
```

### Option 3: Check Statistics First

```python
# Load helper
exec(open('filter_valid_frames.py').read())

# Extract keypoints
normal_keypoints_array = extractor.extract_from_sequence(
    sequence_data=normal_df,
    video_base_path=video_base_path,
    verbose=True
)

# Print detailed statistics
print_keypoint_stats(normal_keypoints_array, "Normal Gait")

# Then decide if you need to filter
```

## Understanding the Statistics

When you run `print_keypoint_stats()`, you'll see:

```
======================================================================
Keypoint Statistics: Normal Gait
======================================================================
Total frames: 80
  ✅ Valid detections: 16 (20.0%)
  ⚠️  Empty detections: 64 (80.0%)
  ❌ Failed extractions: 0 (0.0%)

Keypoint counts (valid frames only):
  Average: 33.0
  Range: 33 - 33

Quality breakdown:
  🟢 Full (33 keypoints): 16
  🟡 Partial (20-32): 0
  🔴 Poor (<20): 0
======================================================================
```

This tells you:
- **20% success rate** - Only 1 in 5 frames has a person visible
- **All valid frames are high quality** - When detected, all 33 keypoints found
- **No partial detections** - Person is either fully visible or not at all

## Why This Happens with GAVD Data

GAVD dataset has some quirks:

1. **Bounding boxes may be inaccurate** - Annotated manually, may have errors
2. **Videos have scene transitions** - Person walks in/out of frame
3. **Camera angles change** - Person may be occluded at certain angles
4. **Multiple people in video** - GAVD focuses on one person, others may interfere

## Best Practices

### 1. Always Filter Before Analysis

```python
# ❌ BAD - Includes empty frames
joint_angles = get_joint_angles(all_keypoints, ...)

# ✅ GOOD - Only valid frames
valid_keypoints, valid_df, _ = extract_with_filtering(df, video_base_path, extractor)
joint_angles = get_joint_angles(valid_keypoints, ...)
```

### 2. Set Appropriate Thresholds

```python
# Strict - Require all 33 keypoints
valid_kp, valid_df, _ = extract_with_filtering(df, video_base_path, extractor, min_keypoints=33)

# Moderate - Allow some missing keypoints (recommended)
valid_kp, valid_df, _ = extract_with_filtering(df, video_base_path, extractor, min_keypoints=25)

# Lenient - Accept partial detections
valid_kp, valid_df, _ = extract_with_filtering(df, video_base_path, extractor, min_keypoints=20)
```

### 3. Check Success Rate

```python
# If success rate is too low, investigate
if len(valid_keypoints) / len(all_keypoints) < 0.3:  # Less than 30%
    print("⚠️  Low success rate - check video quality or bounding boxes")
```

### 4. Keep DataFrames Aligned

```python
# ✅ GOOD - DataFrame and keypoints stay aligned
valid_keypoints, valid_df, _ = extract_with_filtering(df, video_base_path, extractor)

# Now you can safely use:
for i, kp_set in enumerate(valid_keypoints):
    frame_num = valid_df.iloc[i]['frame_num']  # Correct alignment
    # ... analyze frame
```

## Quick Reference

| Keypoint Count | Quality | Action |
|---------------|---------|--------|
| 33 | 🟢 Perfect | Use for analysis |
| 25-32 | 🟡 Good | Use with caution |
| 20-24 | 🟠 Fair | Consider filtering |
| 1-19 | 🔴 Poor | Filter out |
| 0 | ⚠️ Empty | Always filter out |

## Example: Complete Workflow

```python
# 1. Load helpers
exec(open('filter_valid_frames.py').read())

# 2. Load data
gavd_loader = GAVDDataLoader()
normal_df = gavd_loader.load_gavd_data(normal_csv)

# 3. Extract with filtering
extractor = SequenceKeypointExtractor()
normal_keypoints, normal_df_clean, stats = extract_with_filtering(
    normal_df,
    video_base_path,
    extractor,
    min_keypoints=25,
    verbose=True
)

# 4. Verify results
print(f"\nFinal dataset:")
print(f"  Frames: {len(normal_keypoints)}")
print(f"  Success rate: {stats['valid']/stats['total']*100:.1f}%")

# 5. Proceed with analysis (only valid frames)
joint_angles = get_joint_angles(
    keypoints_array=normal_keypoints,
    keypoint_format="BLAZEPOSE_33",
    fps=30.0,
    confidence_threshold=0.3,
    sequence_id=normal_df_clean['seq'].iloc[0]
)
```

## Summary

- **0 keypoints is normal** - MediaPipe correctly detects when person is not visible
- **Always filter** - Remove empty frames before analysis
- **Use the helpers** - `extract_with_filtering()` handles everything automatically
- **Check statistics** - Understand your data quality
- **Keep aligned** - Use filtered DataFrame with filtered keypoints

The refactored code now handles this gracefully and provides tools to work with real-world data where not every frame has a visible person.
