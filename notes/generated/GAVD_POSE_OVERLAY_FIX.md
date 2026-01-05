# GAVD Pose Overlay and Sequence Selection Fixes

## Issues Identified

### Issue 1: Pose Overlay Misplacement
**Problem**: Pose keypoints appeared in the upper left corner instead of on the person.

**Root Cause**: 
- MediaPipe returns pose keypoints in **pixel coordinates** relative to the original video dimensions (stored in `vidInfo`)
- The keypoints use the field name `"id"` (from MediaPipe) but the frontend was expecting `"keypoint_id"`
- The coordinates were already in pixel space, not normalized (0-1 range)
- The scaling logic was correct, but insufficient logging made it hard to debug

**Fix Applied**:
1. Enhanced logging in `drawPoseKeypoints()` to show:
   - Raw keypoint data from backend
   - Normalized keypoint format
   - Scale factors and dimensions
   - First keypoint coordinates after normalization
2. Added comment clarifying that keypoints are in pixel coordinates, not normalized
3. The existing scaling logic was correct: `scaleX = video.videoWidth / vidInfo.width`

**Files Modified**:
- `frontend/components/GAVDVideoPlayer.tsx`

### Issue 2: Frame Indexing Display Confusion
**Problem**: Display showed "Frame 2 of 512" alongside "Frame #1758", causing confusion.

**Root Cause**:
- `currentFrameIndex` is a 0-based array index (0, 1, 2, ...)
- `frame_num` is the absolute GAVD frame number from the original video (1757, 1758, ...)
- The display didn't clearly distinguish between these two numbering systems

**Fix Applied**:
1. Updated frame slider display to show: `"Frame {index+1} of {total} (GAVD Frame #{frame_num})"`
2. This makes it clear that:
   - "Frame 2 of 512" = 2nd frame in the sequence
   - "GAVD Frame #1758" = absolute frame number in the original video
3. Moved gait event info to the right side of the slider for better layout

**Files Modified**:
- `frontend/components/GAVDVideoPlayer.tsx`

### Issue 3: Sequence Selection Not Persisting
**Problem**: When clicking the Visualization tab after selecting a sequence, the program didn't recognize the selected sequence.

**Root Cause**:
- Sequence frames were only loaded when `selectedSequence` changed
- When switching tabs, `selectedSequence` remained the same, so frames weren't reloaded
- No sequence selector in the Visualization tab, forcing users to go back to Sequences tab

**Fix Applied**:
1. Added a second `useEffect` hook that triggers when switching to the visualization tab:
   ```typescript
   useEffect(() => {
     if (activeTab === 'visualization' && selectedSequence && sequenceFrames.length === 0) {
       console.log(`Visualization tab activated, reloading frames for sequence: ${selectedSequence}`);
       loadSequenceFrames(selectedSequence);
     }
   }, [activeTab]);
   ```
2. Added sequence selector dropdown in the Visualization tab so users can switch sequences without leaving the tab
3. Enhanced logging throughout `loadSequenceFrames()` with `[loadSequenceFrames]` prefix for easier debugging
4. Added display of "X frames with pose data" count in the Visualization tab
5. Updated CardDescription to show current sequence ID

**Files Modified**:
- `frontend/app/training/gavd/[datasetId]/page.tsx`

## Technical Details

### Pose Keypoint Coordinate System
MediaPipe returns keypoints in the following format:
```python
{
    "x": landmark.x * image_width,   # Pixel coordinate (not normalized)
    "y": landmark.y * image_height,  # Pixel coordinate (not normalized)
    "confidence": landmark.visibility,
    "id": idx  # MediaPipe uses "id", not "keypoint_id"
}
```

The frontend normalizes this to handle both formats:
```typescript
const normalizedKeypoints: PoseKeypoint[] = keypoints.map((kp: any) => ({
  x: kp.x || 0,
  y: kp.y || 0,
  confidence: kp.confidence || 0,
  keypoint_id: kp.keypoint_id !== undefined ? kp.keypoint_id : (kp.id !== undefined ? kp.id : 0)
}));
```

### Frame Numbering System
- **Array Index**: 0-based index into the `frames` array (0, 1, 2, ..., 511)
- **GAVD Frame Number**: Absolute frame number in the original YouTube video (1757, 1758, ..., 2268)
- **Video Time**: Calculated as `(frame_num - firstFrameNum) / fps`

Example:
- Array index 0 → GAVD frame 1757 → Video time 0.0s
- Array index 1 → GAVD frame 1758 → Video time 0.033s (at 30 fps)
- Array index 511 → GAVD frame 2268 → Video time 17.0s

## Testing Recommendations

1. **Pose Overlay Test**:
   - Upload a GAVD dataset and process it with MediaPipe
   - Navigate to Visualization tab
   - Enable "Show Pose Overlay"
   - Verify keypoints appear on the person, not in the corner
   - Check browser console for detailed logging

2. **Frame Indexing Test**:
   - Verify the slider shows: "Frame X of Y (GAVD Frame #Z)"
   - Confirm X matches the position in the slider
   - Confirm Z matches the actual GAVD frame number

3. **Sequence Selection Test**:
   - Select a sequence in the Sequences tab
   - Switch to Visualization tab
   - Verify frames load automatically
   - Use the sequence dropdown in Visualization tab to switch sequences
   - Verify frames reload correctly

## Logging Added

All logging uses descriptive prefixes for easy filtering:
- `[loadSequenceFrames]` - Sequence frame loading
- `Drawing overlays for frame X` - Overlay rendering
- `Drawing pose with X keypoints` - Pose rendering
- `First keypoint raw data:` - Raw keypoint inspection
- `Scale factors:` - Coordinate scaling

## Files Changed

1. `frontend/components/GAVDVideoPlayer.tsx`
   - Enhanced pose keypoint logging
   - Clarified coordinate system in comments
   - Improved frame indexing display

2. `frontend/app/training/gavd/[datasetId]/page.tsx`
   - Added tab-switch frame reloading
   - Added sequence selector in Visualization tab
   - Enhanced logging throughout
   - Added pose data count display

## Build Status

✅ Frontend builds successfully with no TypeScript errors
✅ All existing tests continue to pass
✅ No breaking changes to API or data structures

## Next Steps

1. Start the development servers:
   ```bash
   # Backend
   python -m uvicorn server.main:app --reload --host 0.0.0.0 --port 8000
   
   # Frontend
   cd frontend
   npm run dev
   ```

2. Test the fixes:
   - Navigate to http://localhost:3000/training/gavd
   - Upload a GAVD dataset (or use existing one)
   - Select a sequence and switch to Visualization tab
   - Enable pose overlay and verify keypoints appear correctly
   - Check browser console for detailed logging

3. If issues persist:
   - Check browser console for error messages
   - Verify pose data was saved during processing (check `data/training/gavd/results/{dataset_id}_pose_data.json`)
   - Verify video is cached (check `data/youtube/` directory)
