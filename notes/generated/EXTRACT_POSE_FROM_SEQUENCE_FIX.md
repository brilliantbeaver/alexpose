# Extract Pose From Sequence Fix

**Date**: January 15, 2026  
**Issue**: `NameError: name 'KeypointSet' is not defined` and `AttributeError: 'Keypoint' object has no attribute 'get'`

## Problem

The `extract_pose_from_sequence()` function in `ambient/utils/eval_keypoints.py` had two issues:

1. **Missing Import**: Used `KeypointSet` in type hint but didn't import it
2. **Type Conversion**: Returned `KeypointSet` but `visualize_pose_with_skeleton()` expected old dictionary format

### Error Details

```python
NameError: name 'KeypointSet' is not defined
  File "ambient/utils/eval_keypoints.py", line 224
    ) -> Optional[Tuple[KeypointSet, np.ndarray, Dict]]:
                        ^^^^^^^^^^^
```

```python
AttributeError: 'Keypoint' object has no attribute 'get'
  File "ambient/utils/viz.py", line 333, in visualize_pose_with_skeleton
    confidence = kp.get('confidence', 0)
                 ^^^^^^
```

## Root Causes

1. **Type Hint Without Import**: The function signature used `KeypointSet` in the return type but the import was missing
2. **Format Mismatch**: `extract_from_image()` returns a `KeypointSet` object, but `visualize_pose_with_skeleton()` expects a list of dictionaries with keys like `'x'`, `'y'`, `'confidence'`

## Solution

### 1. Added Missing Import

```python
from ambient.pose.keypoint_data import KeypointSet  # For type hints
```

### 2. Convert KeypointSet to Dict Format

Before passing to `visualize_pose_with_skeleton()`, convert the KeypointSet to dictionary format:

```python
# Visualize if requested
if show_visualization:
    # Check if we have valid keypoints (KeypointSet with landmarks)
    if keypoints is not None and len(keypoints) > 0:
        title = f"{title_prefix} - Seq: {sequence_id[:8]}... | Frame: {actual_frame_num}"
        vid_info = frame_row.get('vid_info', {})
        
        # Convert KeypointSet to dict format for visualization
        # The KeypointSet.to_dict_list() method returns List[Dict]
        keypoints_dict = keypoints.to_dict_list()
        
        visualize_pose_with_skeleton(
            frame_rgb, keypoints_dict, bbox, title, 
            vid_info=vid_info, frame_shape=frame_rgb.shape
        )
```

### 3. Updated Type Hint

```python
def extract_pose_from_sequence(
    project_root: Path, 
    sequence_data: pd.DataFrame, 
    frame_index: Optional[int] = None,
    frame_num: Optional[int] = None, 
    use_bbox: bool = True,
    show_visualization: bool = True, 
    confidence_threshold: float = 0.3,
    title_prefix: str = "MediaPipe Pose"
) -> Optional[Tuple[KeypointSet, np.ndarray, Dict]]:  # Now correctly typed
```

## Files Modified

1. **`ambient/utils/eval_keypoints.py`**:
   - Added `from ambient.pose.keypoint_data import KeypointSet` import
   - Updated `extract_pose_from_sequence()` to convert KeypointSet to dict before visualization
   - Fixed type hint to use `KeypointSet` instead of `List[Dict]`

## Usage

After clearing Python cache and restarting Jupyter kernel:

```python
from ambient.utils.eval_keypoints import extract_pose_from_sequence

# This returns: (KeypointSet, frame_rgb, metadata)
result = extract_pose_from_sequence(
    project_root,
    sequence_data=sequences[first_seq_id],
    frame_num=1800,
    show_visualization=True  # Handles visualization internally
)

if result:
    keypoint_set, frame_rgb, metadata = result
    
    # Access keypoint data
    print(f"Detected {len(keypoint_set)} landmarks")
    print(f"Average confidence: {keypoint_set.avg_confidence:.3f}")
    
    # Access individual keypoints
    for kp in keypoint_set.keypoints[:5]:
        print(f"{kp.name}: ({kp.x:.1f}, {kp.y:.1f}) conf={kp.confidence:.3f}")
```

## Key Points

1. **KeypointSet vs Dict**: The new data structure uses `KeypointSet` objects internally, but some visualization functions still expect the old dictionary format
2. **Conversion Method**: Use `keypoint_set.to_dict_list()` to convert to the old format when needed
3. **Type Safety**: Always import types used in type hints to avoid `NameError`
4. **Cache Clearing**: After fixing imports, always clear Python cache and restart Jupyter kernel

## Testing

After the fix:
1. Clear Python cache: `python scripts/clear_python_cache.py`
2. Restart Jupyter kernel
3. Run the notebook cell with `extract_pose_from_sequence()`
4. Visualization should display correctly with skeleton overlay

## Related Issues

This fix completes the series of updates for the KeypointSet migration:
- ✅ Task 6: Simplified `get_keypoints()` to DataFrame-only
- ✅ Task 7: Fixed Jupyter compatibility in `keypoints.py`
- ✅ Task 8: Fixed `visualize_keypoints()` function
- ✅ Task 9: Fixed `extract_pose_from_sequence()` (THIS FIX)

## Future Improvements

Consider updating `visualize_pose_with_skeleton()` in `ambient/utils/viz.py` to accept `KeypointSet` objects directly, eliminating the need for conversion.
