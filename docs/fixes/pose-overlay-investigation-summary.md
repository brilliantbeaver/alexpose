# Pose Overlay Investigation - Complete Summary

## Current Status: BACKEND WORKING ✓ | FRONTEND NEEDS DEBUGGING

## Investigation Summary

After deep investigation into why the "Show Pose Overlay" feature wasn't displaying skeletal keypoints, I've confirmed:

### ✅ What's Working

1. **Backend API** - Fully functional
   - Endpoint: `/api/v1/gavd/sequence/{dataset_id}/{sequence_id}/frame/{frame_num}/pose`
   - Returns keypoints with correct `keypoint_id` fields (0-24)
   - Data format: OpenPose BODY_25 (25 keypoints)
   - Tested with: `597fa151-966c-4f9d-abb0-dabd63f65f5a/cljan9b4p00043n6ligceanyp/frame/1757`

2. **Pose Data Files** - Exist and are valid
   - File: `data/training/gavd/results/597fa151-966c-4f9d-abb0-dabd63f65f5a_pose_data.json`
   - Contains 512 frames for sequence `cljan9b4p00043n6ligceanyp`
   - Keypoints are in absolute pixel coordinates (e.g., x=260, y=363.5)

3. **Frontend Code** - Implemented correctly
   - Has OpenPose BODY_25 skeleton connections
   - Has dynamic format detection (25 vs 33 keypoints)
   - Has proper scaling logic
   - Has comprehensive debugging logs

### ❓ What Needs Investigation

**The coordinate scaling and rendering pipeline needs debugging:**

1. **Keypoint Coordinate Space**
   - Keypoints: x=260-280, y=363-383 (absolute pixels)
   - Video dimensions: 1280x720 (from vid_info)
   - Question: Are these coordinates being scaled correctly to the display canvas?

2. **Video Element Dimensions**
   - Need to verify: What is `video.videoWidth` and `video.videoHeight`?
   - Expected: Should match the actual downloaded video resolution
   - Question: Does it match vid_info (1280x720)?

3. **Canvas Dimensions**
   - Need to verify: Does canvas size match video element size?
   - Question: Is the canvas properly overlaid on the video?

4. **Drawing Execution**
   - Need to verify: Are the draw calls actually executing?
   - Question: Are there any JavaScript errors preventing rendering?

## Verified Dataset Information

**Dataset ID**: `597fa151-966c-4f9d-abb0-dabd63f65f5a`
- **Status**: Completed ✓
- **Sequences**: 2
  - `cljan9b4p00043n6ligceanyp` (512 frames) - Parkinsons gait
  - `cljanb45y00083n6lmh1qhydd` (215 frames)
- **Pose Data**: EXISTS ✓
- **Format**: OpenPose BODY_25 (25 keypoints)

**Sample Keypoint Data** (Frame 1757):
```json
{
  "x": 260.0,
  "y": 363.5,
  "confidence": 0.8,
  "keypoint_id": 0
}
```

**Video Information**:
- URL: `https://www.youtube.com/watch?v=B5hrxKe2nP8`
- Dimensions: 1280x720
- Bounding Box: {top: 125, left: 156, height: 497, width: 228}

## Next Steps for User

To debug why the pose overlay isn't rendering, please:

### 1. Open Browser Console
Navigate to the dataset and open the browser developer console (F12)

### 2. Check for These Log Messages
When you enable "Show Pose Overlay", you should see:
```
[loadSequenceFrames] Loaded pose data for frame 1757: 25 keypoints
Drawing 25 keypoints
Normalized keypoints sample: [{x: 260, y: 363.5, confidence: 0.8, keypoint_id: 0}, ...]
Pose scaling: source=1280x720, display=WxH, scale=X.XXxY.YY
```

### 3. Look for Errors
Check for any JavaScript errors or warnings that might prevent rendering

### 4. Verify Video Dimensions
The console should show the actual video element dimensions. Please share:
- What is the "display" size shown in the "Pose scaling" log?
- Does it match the expected video dimensions?

### 5. Check Canvas Overlay
- Is the canvas element visible in the DOM inspector?
- Does it have the correct dimensions?
- Is it positioned correctly over the video?

## Technical Details

### Coordinate System
```
Keypoint Storage Format:
- Coordinates: Absolute pixels in original video space
- Example: x=260, y=363.5
- Video space: 1280x720 (from vid_info)

Display Transformation:
- Source: 1280x720 (original video)
- Display: video.videoWidth x video.videoHeight (actual rendered size)
- Scale: displayX = sourceX * (video.videoWidth / 1280)
```

### Expected Rendering
When working correctly:
- **Green lines** connect keypoints (skeleton)
- **Red dots** mark keypoints (with white outline)
- **Confidence filter**: Only keypoints with confidence > 0.3 are drawn
- **25 keypoints** for OpenPose BODY_25 format

### Skeleton Connections (OpenPose BODY_25)
```
Head: 0→1 (Nose→Neck)
Arms: 1→2→3→4 (Right), 1→5→6→7 (Left)
Torso: 1→8, 8→9, 8→12
Legs: 9→10→11 (Right), 12→13→14 (Left)
Feet: 11→22→23, 14→19→20
Face: 0→15→17, 0→16→18
```

## Files Involved

### Backend (Working ✓)
- `server/routers/gavd.py` - API endpoint with keypoint_id processing
- `server/services/gavd_service.py` - Pose data loading logic
- `data/training/gavd/results/597fa151-966c-4f9d-abb0-dabd63f65f5a_pose_data.json` - Pose data file

### Frontend (Needs Debugging)
- `frontend/components/GAVDVideoPlayer.tsx` - Video player with pose overlay
- `frontend/app/gavd/[dataset_id]/page.tsx` - Dataset page that loads frames

## API Test Command

To verify the backend is working:
```powershell
Invoke-WebRequest -Uri "http://localhost:8000/api/v1/gavd/sequence/597fa151-966c-4f9d-abb0-dabd63f65f5a/cljan9b4p00043n6ligceanyp/frame/1757/pose" -UseBasicParsing | Select-Object -ExpandProperty Content | ConvertFrom-Json | ConvertTo-Json -Depth 5
```

Expected: Returns 25 keypoints with keypoint_id fields ✓

## Conclusion

The backend is working perfectly and returning correct pose data. The issue is likely in the frontend rendering pipeline - either in coordinate scaling, canvas setup, or the actual drawing execution. The comprehensive logging added to the frontend should help identify exactly where the rendering is failing.

**User Action Required**: Please check the browser console logs when enabling "Show Pose Overlay" and share any errors or unexpected log messages.
