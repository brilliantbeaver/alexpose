# Fix: pose_estimation_for_frames() - KeypointSet Compatibility

## Issue
`TypeError: 'Keypoint' object is not subscriptable` when calling `pose_estimation_for_frames()`

## Root Cause
Line 454 in `ambient/utils/eval_keypoints.py` was trying to access keypoint confidence using dictionary syntax:
```python
'avg_confidence': np.mean([kp['confidence'] for kp in keypoints])
```

However, `keypoints` is now a `KeypointSet` object (not a list of dictionaries), and individual keypoints are `Keypoint` objects that don't support subscript access.

## Solution Applied
Changed line 454 to use the `KeypointSet.avg_confidence` property:
```python
'avg_confidence': keypoints.avg_confidence if keypoints and len(keypoints) > 0 else 0.0
```

## Files Modified
- `ambient/utils/eval_keypoints.py` (line 454)

## Verification Steps
1. ✅ Python cache cleared: `python scripts/clear_python_cache.py`
2. ⏳ **USER ACTION REQUIRED**: Restart Jupyter kernel
3. ⏳ **USER ACTION REQUIRED**: Re-run notebook cell with `pose_estimation_for_frames()`

## Expected Output
After restarting the kernel, the function should successfully process multiple frames and return statistics:
```python
results = pose_estimation_for_frames(
    project_root=project_root,
    sequence_data=sequences[first_seq_id],
    frame_indices=[0, 100, 200]
)
```

Should output:
- Frame-by-frame processing logs
- Summary with successful detections count
- Average landmarks and confidence statistics
- List of result dictionaries with metadata for each frame

## Related Fixes
This is part of the KeypointSet migration:
1. ✅ `visualize_keypoints()` - Fixed AttributeError with `add_landmark_names()`
2. ✅ `extract_pose_from_sequence()` - Fixed missing import and format conversion
3. ✅ `pose_estimation_for_frames()` - Fixed KeypointSet property access (this fix)

## Date
2026-01-16
