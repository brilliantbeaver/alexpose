# Pose Overlay Offset and Scaling Issue - Root Cause Analysis

## Problem
Pose skeleton and bounding box are significantly offset to the left and smaller than the actual person in the video frame.

## Root Cause

### Coordinate Space Mismatch

1. **GAVD Annotations** (`vid_info`): 
   - Bounding boxes are annotated in 1280x720 coordinate space
   - This is the "reference" resolution for GAVD dataset

2. **Downloaded Videos**:
   - Videos are downloaded at `"best[height<=720]"` quality
   - Actual resolution can be 640x360, 854x480, 1280x720, etc.
   - **NOT always 1280x720!**

3. **Pose Keypoint Extraction**:
   - Keypoints are extracted from the **FULL downloaded video frame**
   - Coordinates are in the **actual video resolution** (e.g., 640x360)
   - NOT in the GAVD annotation space (1280x720)

4. **Frontend Scaling**:
   - Frontend receives keypoints in actual video space (640x360)
   - But scales them using `vid_info` dimensions (1280x720)
   - **This causes the offset and scaling issues!**

### Example Scenario

```
GAVD Annotation (vid_info):  1280x720
Downloaded Video:             640x360  (50% scale)
Pose Keypoint at:            (320, 180) in 640x360 space
Frontend scales as if:       (320, 180) in 1280x720 space
Result on 640x360 display:   (160, 90)  <- WRONG! Offset to left and smaller
Correct result should be:    (320, 180) <- No scaling needed!
```

## Solution

### Phase 1: Store Source Dimensions with Keypoints
- When extracting keypoints, capture the actual video frame dimensions
- Store `source_width` and `source_height` with each keypoint set
- Pass these through the storage layer

### Phase 2: API Changes
- Include `source_width` and `source_height` in pose data API responses
- Ensure backward compatibility with old data

### Phase 3: Frontend Fixes
- Use `pose_source_width/height` if available (new format)
- Fall back to `vid_info` dimensions only if source dims not available
- Calculate correct scale factors: `scaleX = displayWidth / sourceWidth`

## Files to Fix

1. `ambient/gavd/gavd_processor.py` - Store source dimensions when extracting
2. `ambient/pose/keypoint_extractor.py` - Return source dimensions with keypoints
3. `server/services/gavd_service.py` - Pass source dimensions through API
4. `frontend/components/GAVDVideoPlayer.tsx` - Use correct source dimensions for scaling

## Testing

After fix, verify:
- Pose skeleton aligns with person's body
- Bounding box matches person's size and position
- Works for videos at different resolutions (360p, 480p, 720p)
- Backward compatible with old data (falls back gracefully)
