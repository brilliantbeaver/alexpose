# Real-time Pose Overlay Debugging Guide

## Issue
Pose keypoints and skeletal overlay are not showing on the real-time camera feed after the metrics normalization fix.

## Debug Steps Added

### 1. Frontend Hook Debug Logging
**File**: `frontend/hooks/useRealtimeAnalysis.ts`

Added console logging in the `pose_result` message handler to track:
- Number of keypoints received
- Whether pose data exists in the message
- Success status of pose results

```typescript
console.log('[DEBUG] Received pose with', message.data.pose.keypoints?.length || 0, 'keypoints');
```

### 2. Camera Component Debug Logging
**File**: `frontend/components/realtime/RealtimeCamera.tsx`

Added console logging in the overlay drawing effect to track:
- Whether currentPose exists
- Number of keypoints to draw
- Whether overlay is visible

```typescript
console.log('[DEBUG] Drawing pose overlay with', currentPose.keypoints?.length || 0, 'keypoints');
```

## How to Debug

### Step 1: Start the Backend
```bash
uvicorn server.main:app --reload
```

### Step 2: Start the Frontend
```bash
cd frontend
npm run dev
```

### Step 3: Open Browser Console
1. Navigate to `http://localhost:3000/realtime`
2. Open browser DevTools (F12)
3. Go to Console tab
4. Click "Start Analysis"

### Step 4: Check Debug Output

Look for these debug messages in the console:

**Expected Flow:**
1. `Connecting to WebSocket: ws://localhost:8000/api/realtime/stream`
2. `WebSocket connected`
3. `[DEBUG] Received pose with X keypoints` (should be 33 for MediaPipe)
4. `[DEBUG] Drawing pose overlay with X keypoints`

**Problem Indicators:**
- `[DEBUG] No pose data in message` → Backend not sending pose data
- `[DEBUG] Pose result not successful` → Backend processing failed
- `[DEBUG] No currentPose to draw` → State not updating
- `[DEBUG] Overlay hidden` → Overlay toggle is off
- `[DEBUG] Received pose with 0 keypoints` → No person detected or pose estimation failed

## Common Issues & Solutions

### Issue 1: No Keypoints Received (0 keypoints)
**Possible Causes:**
- No person in camera view
- Poor lighting conditions
- MediaPipe model not loaded
- Confidence threshold too high

**Solutions:**
- Ensure person is fully visible in camera
- Improve lighting
- Check backend logs for MediaPipe errors
- Lower confidence threshold in config

### Issue 2: Keypoints Received But Not Drawn
**Possible Causes:**
- Canvas not properly sized
- Overlay toggle is off
- Drawing function error
- Coordinate mismatch

**Solutions:**
- Check canvas dimensions match video (640x480)
- Verify `showOverlay` state is true
- Check browser console for drawing errors
- Verify keypoint coordinates are in pixel space (not normalized)

### Issue 3: WebSocket Connection Failed
**Possible Causes:**
- Backend not running
- Port mismatch
- CORS issues

**Solutions:**
- Verify backend is running on port 8000
- Check WebSocket URL in hook
- Check backend CORS configuration

### Issue 4: Overlay Misaligned
**Possible Causes:**
- Canvas size doesn't match coordinate space
- Video dimensions don't match processed frame size

**Solutions:**
- Ensure canvas internal dimensions are 640x480
- Verify keypoints are scaled to 640x480 in backend

## Backend Verification

### Check Backend Logs
Look for these in backend console:
```
StreamProcessor initialized: mode=balanced, buffer_size=30, tracking=True
Session started: <session_id>
```

### Check for Errors
Look for these error patterns:
```
ERROR | Failed to decode frame
ERROR | Pose estimation failed
ERROR | Frame processing failed
```

### Test Backend Directly
Use the test endpoint:
```bash
curl -X POST http://localhost:8000/api/realtime/test-frame \
  -H "Content-Type: application/json" \
  -d '{"data": "<base64_image_data>"}'
```

## Data Flow Verification

### 1. Frame Capture (Frontend)
- Video element captures webcam
- Canvas captures frame at target FPS
- Frame encoded as JPEG base64
- Sent via WebSocket

### 2. Backend Processing
- WebSocket receives frame
- Base64 decoded to image
- MediaPipe processes image
- Keypoints extracted (33 points)
- Coordinates scaled to pixel space
- Result sent back via WebSocket

### 3. Frontend Rendering
- WebSocket receives pose result
- Hook updates `currentPose` state
- React effect triggers
- Canvas overlay draws keypoints and skeleton

## Quick Fixes to Try

### 1. Toggle Overlay
Click the eye icon in top-right to toggle overlay on/off.

### 2. Restart Analysis
Stop and start the analysis to reset state.

### 3. Refresh Page
Hard refresh (Ctrl+Shift+R) to clear any cached state.

### 4. Check Camera Permissions
Ensure browser has camera permissions granted.

### 5. Try Different Processing Mode
Switch between Fast/Balanced/Accurate modes.

## Next Steps After Debugging

Once you identify the issue from the debug logs:

1. **If no keypoints received**: Check backend pose estimation
2. **If keypoints received but not drawn**: Check frontend rendering
3. **If coordinates wrong**: Check coordinate scaling
4. **If performance issue**: Adjust processing mode or FPS

## Removing Debug Logs

After identifying the issue, remove debug logs:

```typescript
// Remove these lines from useRealtimeAnalysis.ts
console.log('[DEBUG] Received pose with', ...);
console.log('[DEBUG] No pose data in message');
console.log('[DEBUG] Pose result not successful:', ...);

// Remove these lines from RealtimeCamera.tsx
console.log('[DEBUG] Drawing pose overlay with', ...);
console.log('[DEBUG] No currentPose to draw');
console.log('[DEBUG] Overlay hidden');
```

## Technical Details

### Coordinate System
- MediaPipe returns normalized coordinates (0-1)
- Backend scales to pixel coordinates (640x480)
- Frontend canvas matches this coordinate space
- Browser scales canvas to display size

### Performance Targets
- Target FPS: 30 (balanced mode)
- Processing latency: 20-40ms
- Frame throttling: One frame in flight at a time

### MediaPipe Model
- Model: pose_landmarker_lite.task
- Keypoints: 33 landmarks
- Running mode: VIDEO (not IMAGE)
- Persistent landmarker instance
