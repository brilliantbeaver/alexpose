# Bounding Box Alignment Fix

## Problem Identified ✅

The bounding box in the "Original Image" panel is **not aligned with the person** because of a **coordinate scaling issue**.

### Root Cause Analysis

1. **CSV Annotations**: Created at **1280x720** resolution
2. **Actual Video**: Downloaded at **640x360** resolution (50% scale)
3. **Scale Factors**: x=0.5, y=0.5
4. **Issue**: Bounding box coordinates were **not being scaled** in the visualization

### Evidence from Screenshot

- Frame: 1957 (index 200)
- Sequence: `cljan9b4p00043n6ligceanyp`
- Frame extracted: (360, 640, 3)
- Video resolution: 640x360
- Annotation resolution: 1280x720
- Scale factors: x=0.500, y=0.500

The red bounding box appears **too far to the right** because it's using coordinates for a 1280x720 image on a 640x360 image.

## The Fix 🔧

The solution is to apply the **same scaling logic** that's already implemented in the `visualize_frame` function to the `visualize_pose_with_skeleton` function.

### Code Changes Required

#### 1. Update Function Signature

```python
# OLD
def visualize_pose_with_skeleton(image, keypoints, bbox=None, title="Pose Detection"):

# NEW  
def visualize_pose_with_skeleton(image, keypoints, bbox=None, title="Pose Detection",
                               vid_info=None, frame_shape=None):
```

#### 2. Update Bounding Box Drawing Code

```python
# OLD CODE (WRONG)
if bbox and isinstance(bbox, dict):
    left = bbox.get('left', 0)      # ← Using original coordinates!
    top = bbox.get('top', 0)        # ← Using original coordinates!
    width = bbox.get('width', 0)    # ← Using original coordinates!
    height = bbox.get('height', 0)  # ← Using original coordinates!

# NEW CODE (FIXED)
if bbox and isinstance(bbox, dict):
    # Get actual frame dimensions
    actual_height, actual_width = frame_shape[:2] if frame_shape else image.shape[:2]
    
    # Get annotation dimensions from vid_info
    annotation_width = vid_info.get('width', actual_width) if vid_info else actual_width
    annotation_height = vid_info.get('height', actual_height) if vid_info else actual_height
    
    # Calculate scaling factors (SAME LOGIC AS visualize_frame)
    scale_x = actual_width / annotation_width if annotation_width > 0 else 1.0
    scale_y = actual_height / annotation_height if annotation_height > 0 else 1.0
    
    # Apply scaling to bbox coordinates
    left = bbox.get('left', 0) * scale_x      # ← Now properly scaled!
    top = bbox.get('top', 0) * scale_y        # ← Now properly scaled!
    width = bbox.get('width', 0) * scale_x    # ← Now properly scaled!
    height = bbox.get('height', 0) * scale_y  # ← Now properly scaled!
```

#### 3. Update Test Function Call

```python
# OLD
visualize_pose_with_skeleton(frame_rgb, keypoints, bbox, title)

# NEW
vid_info = frame_row.get('vid_info', {})  # Extract vid_info
visualize_pose_with_skeleton(frame_rgb, keypoints, bbox, title,
                           vid_info=vid_info, frame_shape=frame_rgb.shape)
```

## Example with Real Numbers

### Before Fix (Wrong)
- Original bbox: left=312, top=250, width=238, height=500
- **No scaling applied**
- Result: Box appears at (312, 250) on 640x360 image → **Too far right!**

### After Fix (Correct)  
- Original bbox: left=312, top=250, width=238, height=500
- Scale factors: x=0.5, y=0.5
- **Scaling applied**: left=156, top=125, width=119, height=250
- Result: Box appears at (156, 125) on 640x360 image → **Correctly positioned!**

## Why This Happens

1. **MediaPipe** automatically handles coordinate scaling for pose keypoints
2. **Bounding box coordinates** come from CSV data and need manual scaling
3. The `visualize_frame` function **already has this scaling logic**
4. The `visualize_pose_with_skeleton` function was **missing this scaling logic**

## Implementation Status

- ✅ **Problem identified**: Coordinate scaling mismatch
- ✅ **Root cause found**: Missing scaling in visualization function  
- ✅ **Solution designed**: Apply same scaling as visualize_frame
- ✅ **Code changes specified**: Function signature and bbox drawing logic
- ⏳ **Implementation needed**: Apply the code changes to the notebook

## Files Created

1. `notebooks/fixed_bbox_visualization.py` - Complete fixed visualization function
2. `notebooks/bbox_fix_patch.py` - Patch showing the exact changes needed
3. `notebooks/test_bbox_fix.py` - Demonstration of the scaling issue
4. `notebooks/BBOX_ALIGNMENT_FIX.md` - This comprehensive analysis

## Next Steps

1. Apply the code changes to the notebook
2. Test with the same frame (1957) to verify the fix
3. Confirm the bounding box now properly surrounds the person

The fix is straightforward and follows the same pattern already established in the codebase. The bounding box will be properly aligned once the scaling is applied.