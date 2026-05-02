# Realtime Pose Overlay - Deep Investigation

## Investigation Approach

I've added comprehensive logging throughout the entire data pipeline to trace exactly where the issue occurs.

## Changes Made for Debugging

### 1. Backend Test Script
Created `test_realtime_data_flow.py` to verify:
- Pose detection is working
- Keypoint format is correct
- Data structure matches expectations

### 2. Enhanced Frontend Logging

#### `frontend/hooks/useRealtimeAnalysis.ts`
Added detailed WebSocket message logging:
```typescript
console.log('[WebSocket] Received message type:', message.type);
console.log('[WebSocket] Pose result received:', {
  success: message.data?.success,
  hasPose: !!message.data?.pose,
  poseKeys: message.data?.pose ? Object.keys(message.data.pose) : [],
  keypointsType: message.data?.pose?.keypoints ? typeof message.data.pose.keypoints : 'undefined',
  keypointsIsArray: Array.isArray(message.data?.pose?.keypoints),
  keypointsLength: message.data?.pose?.keypoints?.length || 0
});
```

#### `frontend/components/realtime/RealtimeCamera.tsx`
Added logging at multiple points:
1. When pose data changes (useEffect)
2. When drawPoseOverlay is called
3. During keypoint filtering
4. When drawing skeleton and keypoints

## How to Use This Debugging Setup

### Step 1: Restart Backend
```bash
pkill -f "uvicorn server.main"
uvicorn server.main:app --reload --port 8000
```

### Step 2: Open Browser Console
1. Open http://localhost:3000/realtime
2. Open DevTools (F12)
3. Go to Console tab
4. Clear console

### Step 3: Start Analysis
1. Click "Start Analysis"
2. Allow camera permissions
3. Watch the console output

## What to Look For in Console

### Expected Log Sequence

#### 1. WebSocket Connection
```
[WebSocket] Connecting to WebSocket: ws://localhost:8000/api/realtime/stream
[WebSocket] WebSocket connected
[WebSocket] Session started: <session-id>
```

#### 2. Frame Capture
```
Starting frame capture...
Captured 30 frames
Captured 60 frames
...
```

#### 3. Pose Results (Every Frame)
```
[WebSocket] Received message type: pose_result
[WebSocket] Pose result received: {
  success: true,
  hasPose: true,
  poseKeys: ['keypoints', 'confidence_scores', 'processing_time_ms', ...],
  keypointsType: 'object',
  keypointsIsArray: true,
  keypointsLength: 33  // <-- THIS SHOULD BE 33 if person detected
}
[WebSocket] Setting currentPose with keypoints: 33
[WebSocket] First keypoint: {x: 320.5, y: 240.3, confidence: 0.95, id: 0}
```

#### 4. Pose Drawing
```
[RealtimeCamera] Pose data changed: {
  hasPose: true,
  showOverlay: true,
  keypointsType: 'object',
  keypointsIsArray: true,
  keypointsLength: 33,
  firstKeypoint: {x: 320.5, y: 240.3, ...}
}
[RealtimeCamera] Drawing pose overlay with 33 keypoints
[drawPoseOverlay] Called with pose: {
  hasKeypoints: true,
  keypointsLength: 33,
  keypointsType: 'object',
  isArray: true,
  firstKeypoint: {x: 320.5, y: 240.3, ...}
}
[drawPoseOverlay] Valid keypoints after filtering: 33
[drawPoseOverlay] Drawing skeleton
[drawPoseOverlay] Drawing keypoints
[drawPoseOverlay] Drawing complete
```

## Diagnostic Scenarios

### Scenario 1: No Keypoints Detected
**Console shows:**
```
[WebSocket] Pose result received: { keypointsLength: 0 }
[RealtimeCamera] No current pose data
```

**Possible Causes:**
1. **No person in frame** - Most common
   - Solution: Ensure you're visible in camera
   - Stand 3-6 feet away
   - Face camera directly
   - Good lighting

2. **MediaPipe not detecting person**
   - Check backend logs for warnings
   - Try different processing mode
   - Ensure model file exists

3. **Camera angle/position**
   - Try different camera angles
   - Ensure full body visible
   - Avoid backlighting

### Scenario 2: Keypoints Received But Not Drawing
**Console shows:**
```
[WebSocket] Setting currentPose with keypoints: 33
[RealtimeCamera] Drawing pose overlay with 33 keypoints
[drawPoseOverlay] Called with pose: { keypointsLength: 33 }
[drawPoseOverlay] Valid keypoints after filtering: 0  // <-- PROBLEM
```

**Possible Causes:**
1. **Confidence threshold too high**
   - All keypoints filtered out
   - Solution: Lower confidence threshold in settings
   - Default is 0.5, try 0.3

2. **Low confidence detections**
   - Poor lighting
   - Person too far away
   - Partial occlusion

### Scenario 3: Canvas Not Visible
**Console shows:**
```
[drawPoseOverlay] No canvas or context
```

**Possible Causes:**
1. **Canvas ref not set**
   - Component not mounted properly
   - Check React DevTools

2. **Canvas dimensions wrong**
   - Check canvas.width and canvas.height
   - Should match video dimensions

### Scenario 4: Keypoints Wrong Format
**Console shows:**
```
[WebSocket] Pose result received: {
  keypointsType: 'undefined',  // <-- PROBLEM
  keypointsIsArray: false
}
```

**Possible Causes:**
1. **Backend not sending keypoints**
   - Check backend logs
   - Verify pose_estimator is working

2. **Data transformation error**
   - Check stream_processor
   - Verify JSON serialization

## Backend Logs to Check

### Terminal Running Backend
Look for these messages:

#### Success Case:
```
[DEBUG] Processing 33 landmarks for frame 1280x720
[DEBUG] Created 33 keypoints with pixel coordinates
[DEBUG] Sending 33 keypoints to frontend
```

#### No Detection:
```
[WARNING] No landmarks detected in pose result
[WARNING] No keypoints in pose result - pose may not have been detected
```

#### Error Case:
```
[ERROR] Frame processing failed: <error message>
[ERROR] Pose estimation failed: <error message>
```

## Network Tab Inspection

### WebSocket Messages
1. Open DevTools → Network tab
2. Filter by "WS"
3. Click on the WebSocket connection
4. Go to "Messages" tab
5. Look for `pose_result` messages

**Expected Message:**
```json
{
  "type": "pose_result",
  "data": {
    "success": true,
    "pose": {
      "keypoints": [
        {"x": 320.5, "y": 240.3, "confidence": 0.95, "id": 0},
        ...33 total keypoints
      ],
      "confidence_scores": [0.95, 0.92, ...],
      "processing_time_ms": 15.2
    }
  }
}
```

## Quick Fixes to Try

### Fix 1: Lower Confidence Threshold
In Settings panel, change confidence threshold from 0.5 to 0.3

### Fix 2: Change Processing Mode
Try "Fast" mode for better performance

### Fix 3: Improve Lighting
- Turn on more lights
- Avoid backlighting
- Use natural light if possible

### Fix 4: Camera Position
- Move closer (3-4 feet)
- Ensure full body visible
- Face camera directly
- Stand against plain background

### Fix 5: Clear Browser Cache
Hard refresh: Cmd+Shift+R (Mac) or Ctrl+Shift+R (Windows)

### Fix 6: Restart Everything
```bash
# Backend
pkill -f "uvicorn server.main"
uvicorn server.main:app --reload --port 8000

# Frontend
# In frontend directory
npm run dev
```

## Test Checklist

- [ ] Backend server running on port 8000
- [ ] Frontend server running on port 3000
- [ ] WebSocket connected (check Network tab)
- [ ] Camera permission granted
- [ ] Video feed visible
- [ ] "Start Analysis" clicked
- [ ] Person visible in frame
- [ ] Good lighting
- [ ] Console shows keypoints received
- [ ] Console shows drawing functions called
- [ ] Canvas overlay visible

## Expected Results

When everything is working:
1. Console shows 33 keypoints received every frame
2. Console shows drawing functions called
3. Visual overlay appears on video:
   - 33 colored dots on body joints
   - Lines connecting joints (skeleton)
   - Color-coded by body part
4. Overlay updates in real-time (20-30 FPS)
5. Performance stats show processing times < 30ms

## If Still Not Working

After checking all the above, if overlay still doesn't appear:

1. **Copy console output** and share it
2. **Check backend terminal** for errors
3. **Inspect WebSocket messages** in Network tab
4. **Try test script**: `python test_realtime_data_flow.py`
5. **Check browser compatibility** (Chrome/Edge recommended)

## Files Modified for Debugging

1. `test_realtime_data_flow.py` - Backend test script
2. `frontend/hooks/useRealtimeAnalysis.ts` - Enhanced WebSocket logging
3. `frontend/components/realtime/RealtimeCamera.tsx` - Enhanced drawing logging

All logging can be easily removed by searching for `console.log('[` and `logger.debug` in the respective files.
