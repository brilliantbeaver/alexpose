# ZeroDivisionError Fix - Root Cause Analysis and Solution

## Problem Summary

The `ZeroDivisionError` occurs in `print_extraction_statistics()` when `stats['total']` is 0, which happens when `extract_from_sequence()` returns an empty array.

## Root Causes Identified

### 1. **Aggressive Filtering**
When `filter_empty=True` with `min_keypoints=25`, ALL frames can be filtered out if:
- Pose detection fails for all frames
- All detected poses have < 25 keypoints (e.g., partial body visibility)
- Video file is missing or corrupted

### 2. **Silent Failures**
The original code didn't provide clear warnings when:
- All frames were filtered out
- Video files were missing
- Extraction completely failed

### 3. **No Validation of Results**
The code didn't check if the result array was empty before attempting statistics calculations.

## Fixes Applied

### Fix 1: Enhanced `print_extraction_statistics()` 
**File**: `ambient/pose/keypoint_extractor.py`

Added early return when `stats['total'] == 0`:
```python
if stats['total'] == 0:
    print("  ⚠️  No frames to analyze")
    print(f"{'='*70}\n")
    return
```

### Fix 2: Improved `extract_from_sequence()`
**File**: `ambient/pose/keypoint_extractor.py`

Added validation and warnings:
```python
# Check if we got any results at all
if not keypoints_array:
    logger.warning("No frames were processed - extraction completely failed")
    return []

# Apply filtering if requested
if filter_empty:
    filtered = self._filter_keypoints(keypoints_array, min_keypoints, verbose)
    if not filtered:
        logger.warning(
            f"All {len(keypoints_array)} frames were filtered out "
            f"(min_keypoints={min_keypoints}). Consider lowering min_keypoints threshold."
        )
    return filtered
```

### Fix 3: Better Error Logging in `_process_frame()`
**File**: `ambient/pose/keypoint_extractor.py`

Added detailed logging for first few failures:
```python
# Log if extraction returned None or empty keypoints
if keypoints is None:
    if idx == 0:
        logger.warning(f"Keypoint extraction returned None for frame {frame_num}")
elif len(keypoints.keypoints) == 0:
    if idx == 0:
        logger.warning(f"Keypoint extraction returned 0 keypoints for frame {frame_num}")
```

### Fix 4: Enhanced Validation Logging
**File**: `ambient/pose/keypoint_extractor.py`

Added detailed diagnostics when validation fails:
```python
if not is_valid:
    logger.error(f"Sequence validation failed: {message}")
    logger.error(f"  DataFrame shape: {sequence_data.shape if not sequence_data.empty else 'empty'}")
    logger.error(f"  Video base path: {video_base_path}")
    if not sequence_data.empty and 'url' in sequence_data.columns:
        sample_url = sequence_data['url'].iloc[0] if len(sequence_data) > 0 else 'N/A'
        logger.error(f"  Sample URL: {sample_url}")
```

## How to Use

### Recommended Approach

```python
from ambient.pose.keypoint_extractor import SequenceKeypointExtractor

extractor = SequenceKeypointExtractor()

# Extract keypoints
keypoints_array = extractor.extract_from_sequence(
    sequence_data=df,
    video_base_path=video_base_path,
    verbose=True,
    filter_empty=True,
    min_keypoints=25  # Adjust based on your needs
)

# ALWAYS check if array is empty before using
if not keypoints_array:
    print(f"⚠️  WARNING: No keypoints extracted for {sequence_name}")
    print("   Possible reasons:")
    print("   - Video file missing")
    print("   - All frames filtered out (try lower min_keypoints)")
    print("   - Pose detection failed")
else:
    # Safe to use
    extractor.print_extraction_statistics(keypoints_array, sequence_name)
```

### Adjusting Filtering Threshold

If you're getting empty results, try:

```python
# Option 1: Lower the threshold
keypoints_array = extractor.extract_from_sequence(
    sequence_data=df,
    video_base_path=video_base_path,
    filter_empty=True,
    min_keypoints=15  # Lower threshold
)

# Option 2: Don't filter, handle None values yourself
keypoints_array = extractor.extract_from_sequence(
    sequence_data=df,
    video_base_path=video_base_path,
    filter_empty=False  # Keep all frames
)
# Then filter manually
valid_keypoints = [kp for kp in keypoints_array if kp is not None and len(kp.keypoints) >= 15]
```

## Testing

Run the diagnostic script to test specific CSV files:

```bash
python test_extraction_diagnosis.py
```

This will show:
- DataFrame structure
- Video file existence
- Extraction results with and without filtering
- Detailed keypoint statistics

## Common Issues and Solutions

### Issue 1: "All frames filtered out"
**Cause**: `min_keypoints` threshold too high
**Solution**: Lower `min_keypoints` or check video quality

### Issue 2: "Video file not found"
**Cause**: Missing video file or incorrect `video_base_path`
**Solution**: Verify video files exist and path is correct

### Issue 3: "No frames were processed"
**Cause**: DataFrame validation failed or all frames failed extraction
**Solution**: Check logs for validation errors, verify CSV structure

## Prevention

Always wrap statistics calls with empty checks:

```python
if keypoints_array:
    extractor.print_extraction_statistics(keypoints_array, name)
else:
    print(f"⚠️  No data to analyze for {name}")
```

## Files Modified

1. `ambient/pose/keypoint_extractor.py` - Core extraction logic
2. `test_extraction_diagnosis.py` - Diagnostic tool (new)
3. `EXTRACTION_FIX_SUMMARY.md` - This documentation (new)
