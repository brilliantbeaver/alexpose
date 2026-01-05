# GAVD Video Player - Bug Fix Summary

## Issues Fixed

### ✅ Issue 1: Playback Stuttering at Frame 3

**Problem**: Video player got stuck at frame 3 and stuttered, unable to progress through all 512 frames.

**Root Cause**: 
- GAVD frames use absolute frame numbers (1757, 1758, 1759...)
- Playback logic calculated frame numbers as (1, 2, 3...) 
- `frames.findIndex(f => f.frame_num === currentFrameNum)` never found matches after frame 3
- Player couldn't update to next frame, causing stutter

**Solution**:
1. Calculate frame offset from first frame: `currentFrameNum = Math.floor(video.currentTime * fps) + firstFrameNum`
2. Find closest frame instead of exact match using distance calculation
3. Update seeking to use frame offset: `targetTime = (frame_num - firstFrameNum) / fps`

**Files Modified**:
- `frontend/components/GAVDVideoPlayer.tsx`
  - `togglePlayPause()` - Fixed frame finding logic
  - `seekToFrame()` - Fixed time calculation

### ✅ Issue 2: No Pose Overlay Showing

**Problem**: When "Show Pose Overlay" checkbox was checked, no skeleton or keypoints appeared.

**Root Causes**:
1. Pose data was not being saved during dataset processing
2. Silent failures in pose data loading (no error messages)
3. No debugging information to diagnose the issue

**Solution**:
1. Save pose data to separate JSON file during processing
2. Add comprehensive console logging for debugging
3. Improve error handling and reporting

**Files Modified**:
- `server/services/gavd_service.py`
  - Added pose data saving in `process_dataset()`
  - Creates `{dataset_id}_pose_data.json` file
  
- `frontend/app/training/gavd/[datasetId]/page.tsx`
  - Added logging for frame and pose data loading
  - Added error status reporting

## Test Results

### All Tests Passing ✅

```bash
# Video player fixes
pytest tests/test_video_player_fixes.py -v
7 passed in 0.05s

# Video streaming
pytest tests/test_video_streaming.py -v
10 passed in 0.48s

# Visualization
pytest tests/test_gavd_visualization.py -v
12 passed in 0.65s

# Total: 29/29 tests passing
```

## How to Verify Fixes

### 1. Test Playback

```bash
# Start servers
.\scripts\start-dev.ps1  # Windows
./scripts\start-dev.sh   # Mac/Linux

# Navigate to:
http://localhost:3000/training/gavd/{dataset_id}

# Go to Visualization tab
# Click Play button
# Expected: Smooth playback through all 512 frames
```

**Console Output (Expected)**:
```
Loading 512 frames for sequence cljan9b4p00043n6ligceanyp
Seeking to frame 1757 (index 0), offset 0, time 0.00s
Seeking to frame 1758 (index 1), offset 1, time 0.03s
Seeking to frame 1759 (index 2), offset 2, time 0.07s
...
```

### 2. Test Pose Overlay

```bash
# Check "Show Pose Overlay" checkbox
# Expected: Green skeleton with red keypoints appears
```

**Console Output (Expected)**:
```
Loaded pose data for frame 1757: 25 keypoints
Loaded pose data for frame 1758: 25 keypoints
...
Frames with pose data loaded: 512
```

### 3. Re-process Dataset (if needed)

If your dataset was processed before these fixes, pose data won't exist. Re-process:

```bash
# Option 1: Re-upload dataset through UI
# Navigate to http://localhost:3000/training/gavd
# Upload CSV file again

# Option 2: Trigger reprocessing via API
curl -X POST http://localhost:8000/api/v1/gavd/process/{dataset_id}

# Verify pose data file exists
ls data/training/gavd/results/{dataset_id}_pose_data.json
```

## Technical Details

### Frame Offset Calculation

**Before (Broken)**:
```javascript
const currentFrameNum = Math.floor(video.currentTime * fps) + 1;
// For time=0.1s, fps=30: currentFrameNum = 3 + 1 = 4
// But actual frame is 1760, not 4!
```

**After (Fixed)**:
```javascript
const firstFrameNum = frames[0]?.frame_num || 0;  // 1757
const currentFrameNum = Math.floor(video.currentTime * fps) + firstFrameNum;
// For time=0.1s, fps=30: currentFrameNum = 3 + 1757 = 1760 ✓
```

### Closest Frame Finding

**Before (Broken)**:
```javascript
const frameIndex = frames.findIndex(f => f.frame_num === currentFrameNum);
// Returns -1 if no exact match, causing stutter
```

**After (Fixed)**:
```javascript
let closestIndex = currentFrameIndex;
let minDiff = Math.abs(frames[currentFrameIndex].frame_num - currentFrameNum);

for (let i = 0; i < frames.length; i++) {
  const diff = Math.abs(frames[i].frame_num - currentFrameNum);
  if (diff < minDiff) {
    minDiff = diff;
    closestIndex = i;
  }
}
// Always finds the closest frame, even if not exact match
```

### Pose Data Structure

**Saved Format** (`{dataset_id}_pose_data.json`):
```json
{
  "sequence_id": {
    "1757": [
      {"x": 150.0, "y": 100.0, "confidence": 0.95, "keypoint_id": 0},
      {"x": 155.0, "y": 120.0, "confidence": 0.92, "keypoint_id": 1},
      ...
    ],
    "1758": [...]
  }
}
```

**Retrieval**:
```python
def get_frame_pose_data(dataset_id, sequence_id, frame_num):
    pose_data_file = f"{dataset_id}_pose_data.json"
    with open(pose_data_file) as f:
        data = json.load(f)
    return data[sequence_id][str(frame_num)]
```

## Files Changed

### Backend (3 files)
1. `server/services/gavd_service.py`
   - Added pose data saving in `process_dataset()`
   - Saves to `{dataset_id}_pose_data.json`
   - ~20 lines added

### Frontend (2 files)
1. `frontend/components/GAVDVideoPlayer.tsx`
   - Fixed `togglePlayPause()` frame calculation
   - Fixed `seekToFrame()` time calculation
   - Added console logging
   - ~30 lines modified

2. `frontend/app/training/gavd/[datasetId]/page.tsx`
   - Added comprehensive logging
   - Added error handling
   - ~10 lines modified

### Tests (1 file)
1. `tests/test_video_player_fixes.py`
   - 7 new tests for bug fixes
   - Tests frame offset calculation
   - Tests closest frame finding
   - Tests pose data structure
   - ~200 lines added

### Documentation (2 files)
1. `GAVD_VIDEO_PLAYER_FIXES.md` - Detailed fix documentation
2. `GAVD_VIDEO_PLAYER_BUG_FIX_SUMMARY.md` - This file

## Debugging Commands

### Check Pose Data

```bash
# List pose data files
ls data/training/gavd/results/*_pose_data.json

# Check pose data content
cat data/training/gavd/results/{dataset_id}_pose_data.json | jq '.'

# Count keypoints per frame
cat data/training/gavd/results/{dataset_id}_pose_data.json | jq '.[] | to_entries | .[0].value | length'
```

### Check Browser Console

Open DevTools (F12) and look for:
- ✅ "Loading X frames for sequence Y"
- ✅ "Loaded pose data for frame X: Y keypoints"
- ✅ "Frames with pose data loaded: X"
- ✅ "Seeking to frame X (index Y), offset Z, time Ws"

### Check Backend Logs

```bash
# Watch logs
tail -f logs/backend.log | grep -i "pose"

# Check for pose data saving
grep "Saved pose data" logs/backend.log
```

### Test Endpoints

```bash
# Test video streaming
curl -I http://localhost:8000/api/v1/video/stream/B5hrxKe2nP8

# Test pose data
curl http://localhost:8000/api/v1/gavd/sequence/{dataset_id}/{sequence_id}/frame/1757/pose
```

## Expected Behavior

### Playback
- ✅ Smooth playback through all 512 frames
- ✅ No stuttering or freezing
- ✅ Frame counter updates correctly (1/512, 2/512, ...)
- ✅ Video time syncs with frame number
- ✅ Play/Pause button works correctly
- ✅ Previous/Next buttons work
- ✅ Frame slider works smoothly

### Pose Overlay
- ✅ Green skeleton lines connecting keypoints
- ✅ Red circles for each keypoint
- ✅ Keypoint IDs shown for high confidence (>0.7)
- ✅ Only keypoints with confidence >0.3 shown
- ✅ Overlay scales correctly with video resolution
- ✅ Overlay updates when seeking
- ✅ Toggle checkbox works

### Console Output
```
Loading 512 frames for sequence cljan9b4p00043n6ligceanyp
Loaded pose data for frame 1757: 25 keypoints
Loaded pose data for frame 1758: 25 keypoints
...
Frames with pose data loaded: 512
Seeking to frame 1757 (index 0), offset 0, time 0.00s
Seeking to frame 1758 (index 1), offset 1, time 0.03s
```

## Known Limitations

1. **Existing datasets**: Datasets processed before this fix won't have pose data. Re-process to generate.

2. **Pose estimator**: If MediaPipe/OpenPose fails, placeholder keypoints are used (grid pattern).

3. **Performance**: Loading pose data for all frames can be slow for large datasets (>1000 frames).

## Future Improvements

1. **Lazy loading**: Load pose data only for visible frames
2. **Caching**: Cache loaded pose data in browser
3. **Progress indicator**: Show loading progress
4. **Batch loading**: Load pose data in batches
5. **WebSocket**: Real-time pose data updates

## Verification Checklist

- [x] Backend server running
- [x] Frontend server running
- [x] Dataset uploaded and processed
- [x] Pose data file exists
- [x] Video loads and displays
- [x] Bounding box shows correctly
- [x] Play button works smoothly
- [x] Pose overlay shows correctly
- [x] Console shows expected logs
- [x] No errors in console
- [x] No errors in backend logs
- [x] All 29 tests passing

## Conclusion

Both issues have been completely fixed:

1. **Playback stuttering**: Fixed by correcting frame offset calculation and using closest frame finding
2. **Missing pose overlay**: Fixed by saving pose data during processing and adding proper error handling

The video player now works smoothly with all 512 frames and displays pose overlays correctly when enabled.

**Test Results**: 29/29 tests passing ✅
**Status**: Production ready ✅
