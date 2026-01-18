# Pose Overlay Offset Fix - Complete

## Issue
Pose skeleton overlay was offset to the left and scaled smaller than the actual person in GAVD video visualization.

## Root Cause Found ✅
**Bug Location**: `ambient/gavd/gavd_processor.py` lines 1152-1163

The batch video processing code path was converting `KeypointSet` objects to dictionary format but **dropping the source frame dimensions** (`frame_width` and `frame_height`). This caused the frontend to fall back to incorrect `vid_info` dimensions for scaling, resulting in misaligned overlays.

## Fix Applied ✅
Modified the KeypointSet → dict conversion to preserve source dimensions:

```python
# Extract source dimensions from KeypointSet
source_width = kp_set.frame_width
source_height = kp_set.frame_height

# Include in each keypoint
for kp in kp_set.keypoints:
    keypoints.append({
        "x": kp.x,
        "y": kp.y,
        "confidence": kp.confidence,
        "source_width": source_width,   # ← ADDED
        "source_height": source_height,  # ← ADDED
    })
```

## Next Steps Required

### 1. Reprocess Your Dataset
The code fix is complete, but existing data needs to be regenerated:

```bash
python -m ambient.cli process-gavd <dataset_id>
```

Replace `<dataset_id>` with your actual dataset ID (e.g., `cljar9bqg00c43n6lmh1qhydd`).

### 2. Verify the Fix
After reprocessing, run the verification script:

```bash
python scripts/verify_pose_source_dimensions.py <dataset_id>
```

This will check if the new data includes source dimensions.

### 3. Check in Browser
Open the GAVD visualization page and check the browser console:
- ✅ Should see: `"Using stored source dimensions: 640x360"` (or similar)
- ✅ Scale factors should be close to 1.0x
- ✅ Pose skeleton should align perfectly with the person

## Technical Details

### Why This Happened
- Videos are downloaded at varying resolutions (640x360, 854x480, etc.)
- Pose keypoints are extracted in the actual video's coordinate space
- Frontend needs to know the source dimensions to scale correctly
- The batch processing path was losing this critical information

### What Changed
- **Before**: KeypointSet dimensions were dropped during conversion
- **After**: Source dimensions are preserved in each keypoint
- **Impact**: Frontend can now scale pose overlays correctly

## Files Modified
1. `ambient/gavd/gavd_processor.py` - Fixed KeypointSet conversion
2. `notes/BBOX_POSE_OFFSET_FINAL_DIAGNOSIS.md` - Updated diagnosis
3. `scripts/verify_pose_source_dimensions.py` - New verification tool

## Status
- ✅ Code fix complete
- ✅ Python cache cleared
- ⚠️ Data reprocessing required
- ⏳ Awaiting user to reprocess dataset

## Expected Result
After reprocessing, the pose skeleton overlay will be perfectly aligned with the person in the video, with correct size and position.
