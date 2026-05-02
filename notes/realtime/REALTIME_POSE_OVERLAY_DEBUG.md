# Realtime Pose Overlay Debugging Guide

## Issue
Pose keypoints and skeleton overlay not appearing on live camera feed after clicking "Start Analysis".

## Root Cause Analysis

### Critical Bug Found
**Location**: `ambient/realtime/pose_estimator.py` - `_postprocess_result()` method

**Problem**: Duplicate `return` statement was preventing keypoints from being properly formatted and returned.

```python
# BEFORE (BROKEN):
def _get_landmark_name(self, idx: int) -> str:
    landmark_names = [...]
    return landmark_names[idx] if idx < len(landmark_names) else f'landmark_{idx}'
    
    # This return statement was unreachable!
    return RealtimePoseResult(
        keypoints=keypoints,
        ...
    )

# AFTER (FIXED):
def _get_landmark_name(self, idx: int) -> str:
    landmark_names = [...]
    return landmark_names[idx] if idx < len(landmark_names) else f'landmark_{idx}'

# Return statement moved to correct location in _postprocess_result()
```

## Changes Made

### 1. Backend Fixes

#### `ambient/realtime/pose_estimator.py`
- **Fixed**: Removed duplicate return statement in `_get_landmark_name()` method
- **Added**: Debug logging to track keypoint processing
- **Added**: Logging for landmark count and frame dimensions

```python
logger.debug(f"Processing {len(landmarks)} landmarks for frame {frame_width}x{frame_height}")
logger.debug(f"Created {len(keypoints)} keypoints with pixel coordinates")
logger.warning("No landmarks detected in pose result")
```

#### `ambient/realtime/stream_processor.py`
- **Added**: Logging to track keypoints being sent to frontend
- **Added**: `frame_id` and `timestamp` to pose response for better tracking

```python
if pose_result.keypoints:
    logger.debug(f"Sending {len(pose_result.keypoints)} keypoints to frontend")
else:
    logger.warning("No keypoints in pose result - pose may not have been detected")
```

### 2. Frontend Debugging

#### `frontend/hooks/useRealtimeAnalysis.ts`
- **Added**: Console logging for received pose data
- **Added**: Logging for keypoint count and first keypoint

```typescript
console.log('Received pose data:', {
  keypoints: message.data.pose.keypoints?.length || 0,
  hasKeypoints: !!message.data.pose.keypoints,
  firstKeypoint: message.data.pose.keypoints?.[0]
});
```

#### `frontend/components/realtime/RealtimeCamera.tsx`
- **Added**: Console logging for pose overlay drawing
- **Added**: Logging to track when pose data is missing or overlay is hidden

```typescript
console.log('Drawing pose overlay:', {
    hasKeypoints: !!currentPose.keypoints,
    keypointCount: currentPose.keypoints?.length || 0,
    firstKeypoint: currentPose.keypoints?.[0]
});
```

## Data Flow

### Complete Pipeline
```
1. Frontend Camera
   └─> Captures frame from webcam
   └─> Converts to base64 JPEG
   └─> Sends via WebSocket

2. Backend WebSocket Handler (server/routers/realtime.py)
   └─> Receives frame message
   └─> Calls service.handle_frame()

3. Realtime Service (server/services/realtime_service.py)
   └─> Forwards to stream_processor.process_frame()

4. Stream Processor (ambient/realtime/stream_processor.py)
   └─> Decodes base64 frame
   └─> Calls pose_estimator.estimate_pose()

5. Pose Estimator (ambient/realtime/pose_estimator.py)
   └─> Preprocesses frame (resize, blur)
   └─> Calls MediaPipe pose detection
   └─> Postprocesses results (converts normalized to pixel coords)
   └─> Returns RealtimePoseResult with keypoints

6. Back to Stream Processor
   └─> Packages pose result
   └─> Returns to service

7. Back to WebSocket Handler
   └─> Sends pose_result message to frontend

8. Frontend WebSocket Handler (frontend/hooks/useRealtimeAnalysis.ts)
   └─> Receives pose_result message
   └─> Updates currentPose state

9. Frontend Camera Component (frontend/components/realtime/RealtimeCamera.tsx)
   └─> useEffect detects currentPose change
   └─> Calls drawPoseOverlay()
   └─> Draws keypoints and skeleton on canvas
```

## Expected Data Format

### Backend Response
```json
{
  "type": "pose_result",
  "data": {
    "success": true,
    "frame_id": 123,
    "timestamp": 1706198400.123,
    "pose": {
      "keypoints": [
        {
          "x": 320.5,
          "y": 240.3,
          "z": 0.0,
          "confidence": 0.95,
          "id": 0,
          "name": "nose"
        },
        // ... 32 more keypoints
      ],
      "confidence_scores": [0.95, 0.92, ...],
      "processing_time_ms": 15.2,
      "frame_id": 123,
      "timestamp": 1706198400.123,
      "estimator_info": {
        "estimator": "MediaPipe",
        "processing_mode": "balanced",
        "num_keypoints": 33
      }
    },
    "processing_time_ms": 18.5
  }
}
```

### Frontend Pose Interface
```typescript
interface PoseKeypoint {
  x: number;        // Pixel coordinate
  y: number;        // Pixel coordinate
  confidence: number;
  id: number;
}

interface PoseResult {
  keypoints: PoseKeypoint[];
  confidence_scores: number[];
  processing_time_ms: number;
  frame_id: number;
  timestamp: number;
  estimator_info: any;
}
```

## Debugging Steps

### 1. Check Backend Logs
```bash
# Look for these log messages:
tail -f logs/alexpose_*.log | grep -E "(Processing|keypoints|landmarks)"
```

Expected output:
```
Processing 33 landmarks for frame 1280x720
Created 33 keypoints with pixel coordinates
Sending 33 keypoints to frontend
```

### 2. Check Frontend Console
Open browser DevTools Console and look for:
```
Received pose data: { keypoints: 33, hasKeypoints: true, firstKeypoint: {...} }
Drawing pose overlay: { hasKeypoints: true, keypointCount: 33, firstKeypoint: {...} }
```

### 3. Check WebSocket Connection
In browser DevTools Network tab:
- Filter by "WS" (WebSocket)
- Look for connection to `ws://localhost:8000/api/realtime/stream`
- Check messages tab for `pose_result` messages

### 4. Verify MediaPipe Model
```bash
# Check if model file exists
ls -lh data/models/pose_landmarker_*.task
```

### 5. Test Backend Directly
```bash
# Use the test endpoint
curl -X POST http://localhost:8000/api/realtime/test-frame \
  -H "Content-Type: application/json" \
  -d '{"data": "base64_encoded_image_here"}'
```

## Common Issues & Solutions

### Issue 1: No Keypoints Detected
**Symptoms**: `keypoints: []` in logs
**Causes**:
- Person not visible in frame
- Poor lighting
- Camera too far away
- MediaPipe model not loaded

**Solutions**:
- Ensure person is centered in frame
- Improve lighting
- Move closer to camera
- Check model file exists

### Issue 2: Keypoints Not Drawing
**Symptoms**: Keypoints received but not visible
**Causes**:
- Canvas not sized correctly
- Overlay hidden
- Coordinate scaling wrong

**Solutions**:
- Check canvas dimensions match video
- Verify `showOverlay` is true
- Check coordinate conversion (normalized → pixel)

### Issue 3: WebSocket Disconnects
**Symptoms**: Connection drops frequently
**Causes**:
- Backend not running
- Port conflict
- Network issues

**Solutions**:
- Verify backend running on port 8000
- Check for port conflicts
- Restart both servers

### Issue 4: Slow Performance
**Symptoms**: Low FPS, laggy overlay
**Causes**:
- Processing mode too accurate
- High resolution video
- CPU overload

**Solutions**:
- Switch to "fast" processing mode
- Reduce video resolution
- Close other applications

## Testing Checklist

- [ ] Backend server running on port 8000
- [ ] Frontend server running on port 3000
- [ ] WebSocket connection established
- [ ] Camera permission granted
- [ ] Video feed visible
- [ ] "Start Analysis" button clicked
- [ ] Backend logs show keypoint processing
- [ ] Frontend console shows pose data received
- [ ] Canvas overlay visible
- [ ] Keypoints appear on video
- [ ] Skeleton connections drawn

## Performance Metrics

### Expected Values
- **Frame Processing**: 15-30ms per frame
- **WebSocket Latency**: < 50ms
- **FPS**: 20-30 frames per second
- **Keypoint Detection Rate**: > 90%

### Monitor in UI
- Processing Time: Shown in pose info overlay
- FPS: Shown in performance stats
- Frames Processed: Shown in frame processing section

## Next Steps

1. **Restart Backend Server**
   ```bash
   # Kill existing process
   pkill -f "uvicorn server.main"
   
   # Start fresh
   uvicorn server.main:app --reload --port 8000
   ```

2. **Clear Browser Cache**
   - Hard refresh: Cmd+Shift+R (Mac) or Ctrl+Shift+R (Windows)
   - Or clear cache in DevTools

3. **Test with Simple Frame**
   - Use test endpoint with known good image
   - Verify keypoints are detected

4. **Check Logs**
   - Backend: `logs/alexpose_*.log`
   - Frontend: Browser console
   - Look for errors or warnings

## Files Modified

1. `ambient/realtime/pose_estimator.py` - Fixed duplicate return, added logging
2. `ambient/realtime/stream_processor.py` - Added keypoint logging
3. `frontend/hooks/useRealtimeAnalysis.ts` - Added console logging
4. `frontend/components/realtime/RealtimeCamera.tsx` - Added overlay logging

## Conclusion

The main issue was a duplicate return statement preventing keypoints from being properly formatted. With the fix applied and comprehensive logging added, you should now be able to:

1. See keypoints being detected in backend logs
2. See pose data being received in frontend console
3. See keypoints and skeleton overlay on the live camera feed

If issues persist after restarting servers, use the debugging steps above to trace where the data flow breaks.
