# Keypoint Extraction Bug Fix

## Problem
The joint angle visualizations in the Jupyter notebook were showing flat lines with no data because only **1 frame** was being processed instead of all frames in the sequence.

## Root Cause
In `ambient/pose/keypoint_extractor.py`, the `extract_from_sequence()` method had an **indentation bug**. The `return keypoints_array` statement was inside the for loop instead of outside it, causing the function to return after processing only the first frame.

### Before (Buggy Code)
```python
for fnum in range(num_frames):
    # ... process frame ...
    keypoints_array.append(keypoints)
    
    if verbose:
        logger.info("Sequence processing complete")
    
    return keypoints_array  # ❌ WRONG: Returns after first iteration
```

### After (Fixed Code)
```python
for fnum in range(num_frames):
    # ... process frame ...
    keypoints_array.append(keypoints)

if verbose:
    logger.info("Sequence processing complete")

return keypoints_array  # ✅ CORRECT: Returns after all iterations
```

## Impact
- **Before**: Only 1 frame processed → flat plots with no temporal data
- **After**: All 145 frames processed → proper time-series plots with joint angle variations

## Testing
Verified with test sequence `cljawh4c7000m3n6lkz9es9bl.csv`:
- Expected frames: 145
- Extracted frames: 145 ✅
- Processing time: ~35 seconds

## Next Steps
Re-run the Jupyter notebook cells that extract keypoints and calculate joint angles. The visualizations should now display proper time-series data showing how joint angles change over time during the gait sequence.
