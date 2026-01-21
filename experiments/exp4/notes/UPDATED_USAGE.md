# Updated extract_from_sequence() Usage Guide

## What Changed

The `extract_from_sequence()` method now has **built-in filtering** and **statistics** capabilities.

## New Parameters

```python
def extract_from_sequence(
    self,
    sequence_data: pd.DataFrame,
    video_base_path: Path,
    model_path: Optional[str] = None,
    verbose: bool = False,
    filter_empty: bool = False,      # ← NEW: Auto-filter empty frames
    min_keypoints: int = 25           # ← NEW: Minimum keypoints threshold
) -> List[KeypointSet]:
```

## Usage Examples

### Example 1: Get All Frames (Original Behavior)

```python
extractor = SequenceKeypointExtractor()

# Returns ALL frames, including those with 0 keypoints
all_keypoints = extractor.extract_from_sequence(
    sequence_data=normal_df,
    video_base_path=video_base_path,
    verbose=True
)

print(f"Total frames: {len(all_keypoints)}")
# Output: Total frames: 80 (includes empty frames)
```

### Example 2: Auto-Filter Empty Frames (Recommended)

```python
extractor = SequenceKeypointExtractor()

# Returns ONLY frames with >= 25 keypoints
valid_keypoints = extractor.extract_from_sequence(
    sequence_data=normal_df,
    video_base_path=video_base_path,
    verbose=True,
    filter_empty=True,      # ← Enable filtering
    min_keypoints=25        # ← Require at least 25 keypoints
)

print(f"Valid frames: {len(valid_keypoints)}")
# Output: Valid frames: 16 (only frames with person visible)
```

### Example 3: Custom Threshold

```python
# Strict - require all 33 keypoints
strict_keypoints = extractor.extract_from_sequence(
    sequence_data=normal_df,
    video_base_path=video_base_path,
    filter_empty=True,
    min_keypoints=33
)

# Lenient - accept partial detections
lenient_keypoints = extractor.extract_from_sequence(
    sequence_data=normal_df,
    video_base_path=video_base_path,
    filter_empty=True,
    min_keypoints=20
)
```

### Example 4: Get Statistics

```python
extractor = SequenceKeypointExtractor()

# Extract all frames first
all_keypoints = extractor.extract_from_sequence(
    sequence_data=normal_df,
    video_base_path=video_base_path
)

# Print detailed statistics
extractor.print_extraction_statistics(all_keypoints, "Normal Gait")

# Or get stats as dictionary
stats = extractor.get_extraction_statistics(all_keypoints)
print(f"Success rate: {stats['success_rate']:.1f}%")
print(f"Valid frames: {stats['valid']}")
print(f"Empty frames: {stats['empty']}")
```

## Complete Workflow

### Old Way (Manual Filtering Required)

```python
# Extract all frames
all_keypoints = extractor.extract_from_sequence(
    sequence_data=normal_df,
    video_base_path=video_base_path
)

# Manually filter
valid_keypoints = []
valid_indices = []
for i, kp in enumerate(all_keypoints):
    if kp is not None and len(kp.keypoints) >= 25:
        valid_keypoints.append(kp)
        valid_indices.append(i)

# Manually align DataFrame
normal_df_filtered = normal_df.iloc[valid_indices].reset_index(drop=True)
```

### New Way (Built-in Filtering)

```python
# Extract with automatic filtering
valid_keypoints = extractor.extract_from_sequence(
    sequence_data=normal_df,
    video_base_path=video_base_path,
    verbose=True,
    filter_empty=True,
    min_keypoints=25
)

# Note: DataFrame alignment must still be done manually if needed
# See "DataFrame Alignment" section below
```

## Important: DataFrame Alignment

When using `filter_empty=True`, the returned keypoints array is **shorter** than the input DataFrame. You need to align them:

### Option 1: Extract Twice (Simple but Slower)

```python
# First pass: get all frames to identify valid indices
all_keypoints = extractor.extract_from_sequence(
    sequence_data=normal_df,
    video_base_path=video_base_path
)

# Find valid indices
valid_indices = [
    i for i, kp in enumerate(all_keypoints)
    if kp is not None and len(kp.keypoints) >= 25
]

# Filter DataFrame
normal_df_filtered = normal_df.iloc[valid_indices].reset_index(drop=True)

# Second pass: get filtered keypoints
valid_keypoints = extractor.extract_from_sequence(
    sequence_data=normal_df_filtered,
    video_base_path=video_base_path
)
```

### Option 2: Manual Alignment (Faster)

```python
# Extract all frames
all_keypoints = extractor.extract_from_sequence(
    sequence_data=normal_df,
    video_base_path=video_base_path
)

# Filter both keypoints and DataFrame together
valid_keypoints = []
valid_indices = []

for i, kp in enumerate(all_keypoints):
    if kp is not None and len(kp.keypoints) >= 25:
        valid_keypoints.append(kp)
        valid_indices.append(i)

normal_df_filtered = normal_df.iloc[valid_indices].reset_index(drop=True)
```

### Option 3: Use Helper Function (Recommended)

```python
# Load helper (still useful for DataFrame alignment)
exec(open('filter_valid_frames.py').read())

# This handles both keypoints and DataFrame alignment
valid_keypoints, normal_df_filtered, stats = extract_with_filtering(
    normal_df,
    video_base_path,
    extractor,
    min_keypoints=25
)
```

## Recommended Approach for Notebook

```python
# 1. Create extractor
extractor = SequenceKeypointExtractor()

# 2. Extract with filtering
normal_keypoints = extractor.extract_from_sequence(
    sequence_data=normal_df,
    video_base_path=video_base_path,
    verbose=True,
    filter_empty=True,      # ← Auto-filter
    min_keypoints=25        # ← Quality threshold
)

# 3. Print statistics (optional)
extractor.print_extraction_statistics(normal_keypoints, "Normal Gait")

# 4. If you need aligned DataFrame, use helper or manual filtering
# See "DataFrame Alignment" section above
```

## Output Example

```
Processing sequence: cljo30lnz001q3n6lopfty7q5
Number of frames: 80
        frame 311 (1/80)
        frame 321 (11/80)
        ...
Sequence processing complete: 80/80 frames successful
Filtered to 16/80 frames (min 25 keypoints, removed 64)

======================================================================
Keypoint Statistics: Normal Gait
======================================================================
Total frames: 16
  ✅ Valid detections: 16 (100.0%)
  ⚠️  Empty detections: 0 (0.0%)
  ❌ Failed extractions: 0 (0.0%)

Keypoint counts (valid frames only):
  Average: 33.0

Quality breakdown:
  🟢 Full (33 keypoints): 16
  🟡 Partial (20-32): 0
  🔴 Poor (<20): 0
======================================================================
```

## Migration Guide

### Before (External Helpers)

```python
exec(open('filter_valid_frames.py').read())
keypoints, df_filtered, stats = extract_with_filtering(
    normal_df, video_base_path, extractor, min_keypoints=25
)
```

### After (Built-in)

```python
# Just keypoints (no DataFrame alignment)
keypoints = extractor.extract_from_sequence(
    sequence_data=normal_df,
    video_base_path=video_base_path,
    filter_empty=True,
    min_keypoints=25
)

# With statistics
extractor.print_extraction_statistics(keypoints, "Normal Gait")
```

## Summary

- ✅ **`filter_empty=True`** - Automatically removes frames with too few keypoints
- ✅ **`min_keypoints=25`** - Customizable quality threshold
- ✅ **`get_extraction_statistics()`** - Get detailed stats as dictionary
- ✅ **`print_extraction_statistics()`** - Print formatted statistics
- ⚠️ **DataFrame alignment** - Still needs manual handling (use helpers if needed)

The method is now more powerful while maintaining backward compatibility (default behavior unchanged).
