# Bounding Box and Pose Offset Issue - Root Cause Fixed

## Problem
Both bounding box and pose skeleton are offset to the left and smaller than the actual person in the video.

## Root Cause Identified ✅

### The Bug
There are **two code paths** for pose extraction in `ambient/gavd/gavd_processor.py`:

1. **Path 1: `extract_from_image_and_bbox()` (line 955)** ✅
   - Used when processing with image data
   - CORRECTLY includes `source_width` and `source_height` in each keypoint

2. **Path 2: `extract_from_video_frame()` (lines 1152-1163)** ❌ **BUG HERE**
   - Used for batch video processing (the active code path)
   - Receives `KeypointSet` object with `frame_width` and `frame_height`
   - Converts to dictionary format but **DROPS these dimensions**
   - Only extracts x, y, confidence - losing critical source information

### Why This Causes Offset/Scaling Issues

1. **Video Download**: Videos downloaded at varying resolutions (640x360, 854x480, etc.)
2. **Pose Extraction**: Keypoints extracted from actual video frames (e.g., 640x360)
3. **Data Storage**: Source dimensions LOST during KeypointSet → dict conversion
4. **Frontend Display**: Falls back to `vid_info` dimensions (1280x720) for scaling
5. **Result**: Keypoints scaled from wrong coordinate space → offset and smaller

**Example:**
- Video actual size: 640x360
- Keypoints extracted in: 640x360 space
- Source dims lost, frontend uses: 1280x720 (from vid_info)
- Frontend applies 0.5x scale when it should apply 1.0x
- Result: Pose appears 50% smaller and offset

## The Fix ✅

Modified `ambient/gavd/gavd_processor.py` lines 1156-1169:

```python
# BEFORE (BUG):
for kp in kp_set.keypoints:
    keypoints.append({
        "x": kp.x,
        "y": kp.y,
        "confidence": kp.confidence,
    })

# AFTER (FIXED):
source_width = kp_set.frame_width
source_height = kp_set.frame_height

for kp in kp_set.keypoints:
    keypoints.append({
        "x": kp.x,
        "y": kp.y,
        "confidence": kp.confidence,
        "source_width": source_width,   # ← ADDED
        "source_height": source_height,  # ← ADDED
    })
```

## Verification Steps

1. **Reprocess your GAVD dataset** with the fixed code:
   ```bash
   python -m ambient.cli process-gavd <dataset_id>
   ```

2. **Verify the fix worked**:
   ```bash
   python scripts/verify_pose_source_dimensions.py <dataset_id>
   ```

3. **Check in browser console**:
   - Should see: `"✓ Source video dimensions for frame X: WxH"`
   - Should see: `"Using stored source dimensions: WxH"`
   - Scale factors should be close to 1.0x

## Expected Behavior After Fix

### Before Fix ❌
- Pose offset to left and smaller
- Console: `"Using vid_info dimensions: 1280x720"`
- Scale factors: ~0.5x (if video is 640x360)
- Keypoints don't align with person

### After Fix ✅
- Pose perfectly aligned with person
- Console: `"Using stored source dimensions: 640x360"`
- Scale factors: ~1.0x (no scaling needed)
- Keypoints match skeletal structure

## Technical Details

### Data Flow (Fixed)
1. **Video Download**: `best[height<=720]` → actual resolution varies
2. **Frame Extraction**: FFmpeg extracts frame at actual resolution
3. **Pose Estimation**: MediaPipe processes frame → `KeypointSet` with `frame_width/height`
4. **Conversion**: **NOW PRESERVES** source dimensions in each keypoint
5. **Storage**: JSON includes `source_width/height` at frame and keypoint level
6. **Frontend**: Uses correct source dimensions for scaling

### Why Both Paths Exist
- **Path 1** (`extract_from_image_and_bbox`): For processing with pre-loaded images
- **Path 2** (`extract_from_video_frame`): For efficient batch video processing
- Both paths now correctly preserve source dimensions

## Summary

**Root Cause**: KeypointSet → dict conversion dropped frame dimensions in batch processing path

**Solution**: Preserve `frame_width` and `frame_height` from KeypointSet when converting to dict

**Status**: Code fixed ✅ | Data reprocessing required ⚠️

**Impact**: Fixes pose overlay offset/scaling issues for all GAVD processing
