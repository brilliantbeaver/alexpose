# GAVD Video Player - Bug Fixes

## Issues Identified and Fixed

### Issue 1: Playback Stuttering at Frame 3

**Root Cause:**
The playback logic was calculating frame numbers incorrectly. GAVD frames start at absolute frame numbers (e.g., 1757, 1758, 1759...), not at 1. The code was calculating:
```javascript
const currentFrameNum = Math.floor(video.currentTime * fps) + 1;
// This gives: 1, 2, 3, 4... but frames are actually 1757, 1758, 1759...
```

Then it tried to find frames with `frames.findIndex(f => f.frame_num === currentFrameNum)`, which would never find a match after frame 3, causing the player to stutter.

**Fix Applied:**
1. Calculate the offset from the first frame:
```javascript
const firstFrameNum = frames[0]?.frame_num || 0;
const currentFrameNum = Math.floor(video.currentTime * fps) + firstFrameNum;
```

2. Find the closest frame index instead of exact match:
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
```

3. Updated seeking logic to use frame offset:
```javascript
const firstFrameNum = frames[0]?.frame_num || 1;
const frameOffset = frame.frame_num - firstFrameNum;
const targetTime = frameOffset / fps;
```

**Files Modified:**
- `frontend/components/GAVDVideoPlayer.tsx`
  - `togglePlayPause()` function
  - `seekToFrame()` function

### Issue 2: No Pose Overlay Showing

**Root Causes:**

1. **Pose data not saved during processing**
   - The `process_dataset()` method was not saving pose keypoints to a separate file
   - Only basic results were saved, without the actual keypoint data

2. **Silent failures in pose data loading**
   - No console logging to debug why pose data wasn't loading
   - Errors were caught but not reported

**Fixes Applied:**

1. **Save pose data during processing** (`server/services/gavd_service.py`):
```python
# Save pose data separately for efficient retrieval
pose_data_dict = {}
for seq_id, seq_data in results["sequences"].items():
    pose_data_dict[seq_id] = {}
    for frame_data in seq_data["pose_data"]:
        frame_num = frame_data.get("frame")
        keypoints = frame_data.get("pose_keypoints_2d", [])
        if frame_num and keypoints:
            pose_data_dict[seq_id][str(frame_num)] = keypoints

pose_data_file = self.results_dir / f"{dataset_id}_pose_data.json"
with open(pose_data_file, 'w') as f:
    json.dump(pose_data_dict, f, indent=2)
```

2. **Add comprehensive logging** (`frontend/app/training/gavd/[datasetId]/page.tsx`):
```typescript
console.log(`Loading ${result.frames.length} frames for sequence ${sequenceId}`);
console.log(`Loaded pose data for frame ${frame.frame_num}:`, poseData.pose_keypoints?.length || 0, 'keypoints');
console.log('Frames with pose data loaded:', framesWithPose.filter(f => f.pose_keypoints).length);
```

3. **Improved error handling**:
```typescript
if (poseResponse.ok) {
  const poseData = await poseResponse.json();
  return { ...frame, pose_keypoints: poseData.pose_keypoints };
} else {
  console.warn(`No pose data for frame ${frame.frame_num}: ${poseResponse.status}`);
}
```

**Files Modified:**
- `server/services/gavd_service.py` - Added pose data saving
- `frontend/app/training/gavd/[datasetId]/page.tsx` - Added logging and error handling

## Testing the Fixes

### 1. Test Playback

1. Navigate to GAVD Dataset Analysis page
2. Go to Visualization tab
3. Click Play button
4. **Expected**: Video should play smoothly through all 512 frames
5. **Check console**: Should see frame updates logging

### 2. Test Pose Overlay

1. Check "Show Pose Overlay" checkbox
2. **Expected**: Green skeleton with red keypoints should appear
3. **Check console**: Should see "Loaded pose data for frame X: Y keypoints"
4. If no pose data: Re-process the dataset to generate pose data

### 3. Re-process Dataset (if needed)

If pose data is missing, re-upload and process the dataset:

```bash
# 1. Delete old dataset (optional)
curl -X DELETE http://localhost:8000/api/v1/gavd/{dataset_id}

# 2. Upload new dataset
# Use the frontend upload page

# 3. Check pose data file exists
ls data/training/gavd/results/{dataset_id}_pose_data.json

# 4. Check pose data content
cat data/training/gavd/results/{dataset_id}_pose_data.json | head -50
```

## Debugging Commands

### Check if pose data exists

```bash
# List all pose data files
ls data/training/gavd/results/*_pose_data.json

# Check specific dataset
cat data/training/gavd/results/{dataset_id}_pose_data.json | jq '.[] | keys | length'
```

### Check browser console

Open DevTools (F12) and look for:
- "Loading X frames for sequence Y"
- "Loaded pose data for frame X: Y keypoints"
- "Frames with pose data loaded: X"

### Check backend logs

```bash
# Watch backend logs
tail -f logs/backend.log | grep -i "pose"

# Check for pose data saving
grep "Saved pose data" logs/backend.log
```

### Test pose data endpoint

```bash
# Test pose data endpoint
curl http://localhost:8000/api/v1/gavd/sequence/{dataset_id}/{sequence_id}/frame/1757/pose

# Should return:
{
  "success": true,
  "dataset_id": "...",
  "sequence_id": "...",
  "frame_num": 1757,
  "pose_keypoints": [
    {"x": 150.0, "y": 100.0, "confidence": 0.95, "keypoint_id": 0},
    ...
  ]
}
```

## Expected Behavior After Fixes

### Playback
- ✅ Smooth playback through all frames
- ✅ No stuttering or freezing
- ✅ Frame counter updates correctly
- ✅ Video time syncs with frame number

### Pose Overlay
- ✅ Green skeleton lines connecting keypoints
- ✅ Red circles for each keypoint
- ✅ Keypoint IDs shown for high confidence (>0.7)
- ✅ Only keypoints with confidence >0.3 are shown
- ✅ Overlay scales correctly with video resolution

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

1. **Pose data generation**: If the dataset was processed before this fix, pose data won't exist. Re-process the dataset to generate it.

2. **MediaPipe/OpenPose**: If pose estimator fails, placeholder keypoints are used (grid pattern, not real pose).

3. **Large datasets**: Loading pose data for all frames can be slow. Consider pagination or lazy loading for datasets with >1000 frames.

## Future Improvements

1. **Lazy load pose data**: Only load pose data for visible frames
2. **Cache pose data**: Cache loaded pose data in browser
3. **Progress indicator**: Show loading progress for pose data
4. **Pose confidence visualization**: Color-code keypoints by confidence
5. **Pose comparison**: Side-by-side comparison of multiple frames

## Files Changed

### Backend
- `server/services/gavd_service.py`
  - Added pose data saving in `process_dataset()`
  - Saves to `{dataset_id}_pose_data.json`

### Frontend
- `frontend/components/GAVDVideoPlayer.tsx`
  - Fixed `togglePlayPause()` - correct frame number calculation
  - Fixed `seekToFrame()` - use frame offset instead of absolute frame number
  - Added console logging for debugging

- `frontend/app/training/gavd/[datasetId]/page.tsx`
  - Added comprehensive logging in `loadSequenceFrames()`
  - Added error status logging
  - Added pose data count logging

## Verification Checklist

- [ ] Backend server running
- [ ] Frontend server running
- [ ] Dataset uploaded and processed
- [ ] Pose data file exists: `data/training/gavd/results/{dataset_id}_pose_data.json`
- [ ] Navigate to Visualization tab
- [ ] Video loads and displays
- [ ] Bounding box shows when checkbox checked
- [ ] Play button works smoothly through all frames
- [ ] Pose overlay shows when checkbox checked
- [ ] Console shows pose data loading messages
- [ ] No errors in browser console
- [ ] No errors in backend logs

## Troubleshooting

### Playback still stutters
- Check console for frame number logs
- Verify FPS is correct (should be 30)
- Check if video is fully loaded
- Try refreshing the page

### Pose overlay still not showing
- Check console for "Loaded pose data" messages
- Check if pose data file exists
- Re-process the dataset
- Check backend logs for errors
- Verify pose estimator is working

### Video not loading
- Check if video is cached in `data/youtube/`
- Check backend logs for streaming errors
- Verify video ID extraction
- Check network tab for 404 errors
