# Real-time Pose Overlay Investigation Summary

## Current Status
Added comprehensive debug logging to identify why pose keypoints and skeletal overlay are not displaying on the real-time camera feed.

## Changes Made

### 1. Debug Logging Added

#### Frontend Hook (`frontend/hooks/useRealtimeAnalysis.ts`)
- Added logging when pose results are received
- Tracks number of keypoints in each message
- Logs when pose data is missing or unsuccessful

#### Frontend Camera Component (`frontend/components/realtime/RealtimeCamera.tsx`)
- Added logging when attempting to draw overlay
- Tracks whether currentPose exists
- Logs overlay visibility state

### 2. Metrics Normalization (Previous Fix)
- Walking speed normalized to 0-5 scale
- Step/stride lengths normalized by body height
- All metrics clamped to reasonable ranges

## Testing Instructions

### Step 1: Start Backend Server
```bash
# From project root
uvicorn server.main:app --reload
```

Expected output:
```
INFO:     Started server process
INFO:     Waiting for application startup.
INFO:     Application startup complete.
INFO:     Uvicorn running on http://127.0.0.1:8000
```

### Step 2: Start Frontend Server
```bash
# From project root
cd frontend
npm run dev
```

Expected output:
```
▲ Next.js 16.1.1
- Local:        http://localhost:3000
```

### Step 3: Open Browser and Test
1. Navigate to `http://localhost:3000/realtime`
2. Open Browser DevTools (F12 or Cmd+Option+I)
3. Go to Console tab
4. Click "Start Analysis" button
5. Allow camera permissions when prompted
6. Stand in front of camera with full body visible

### Step 4: Check Debug Output

**Look for these console messages:**

✅ **Success Pattern:**
```
Connecting to WebSocket: ws://localhost:8000/api/realtime/stream
WebSocket connected
[DEBUG] Received pose with 33 keypoints
[DEBUG] Drawing pose overlay with 33 keypoints
```

❌ **Problem Patterns:**

**No Keypoints:**
```
[DEBUG] Received pose with 0 keypoints
[DEBUG] No currentPose to draw
```
→ Person not detected or pose estimation failed

**No Pose Data:**
```
[DEBUG] No pose data in message
```
→ Backend not sending pose data correctly

**Processing Failed:**
```
[DEBUG] Pose result not successful: {error details}
```
→ Backend processing error

**Overlay Hidden:**
```
[DEBUG] Overlay hidden
```
→ Overlay toggle is off (click eye icon to enable)

### Step 5: Check Backend Logs

**Look for these in backend console:**

✅ **Success Pattern:**
```
StreamProcessor initialized: mode=balanced, buffer_size=30, tracking=True
Session started: <uuid>
```

❌ **Problem Patterns:**
```
ERROR | Failed to decode frame
ERROR | Pose estimation failed
ERROR | Frame processing failed
```

## Possible Issues & Solutions

### Issue 1: No Person Detected (0 keypoints)
**Symptoms:**
- `[DEBUG] Received pose with 0 keypoints`
- No overlay appears

**Solutions:**
- Ensure full body is visible in camera
- Improve lighting (bright, even lighting works best)
- Move closer or farther from camera
- Try different background (plain background works best)
- Check if MediaPipe model is loaded (backend logs)

### Issue 2: Overlay Toggle Off
**Symptoms:**
- `[DEBUG] Overlay hidden`
- Keypoints received but not drawn

**Solutions:**
- Click the eye icon in top-right of video
- Should toggle from EyeOff to Eye icon

### Issue 3: WebSocket Connection Failed
**Symptoms:**
- No debug messages appear
- "Connection error occurred" alert

**Solutions:**
- Verify backend is running on port 8000
- Check for CORS errors in console
- Restart both servers
- Clear browser cache

### Issue 4: Backend Processing Error
**Symptoms:**
- `[DEBUG] Pose result not successful`
- Backend error logs

**Solutions:**
- Check backend logs for specific error
- Verify MediaPipe model file exists: `data/models/pose_landmarker_lite.task`
- Restart backend server
- Check Python dependencies are installed

### Issue 5: Canvas/Coordinate Mismatch
**Symptoms:**
- Keypoints received but drawn in wrong location
- Overlay appears but misaligned

**Solutions:**
- Verify canvas dimensions are 640x480
- Check keypoint coordinates are in pixel space (not normalized)
- Verify video element is displaying correctly

## Data Flow Verification

### Complete Flow:
1. **Camera Capture** → Video element captures webcam at 30 FPS
2. **Frame Encoding** → Canvas captures frame, encodes as JPEG base64
3. **WebSocket Send** → Frame sent to backend via WebSocket
4. **Backend Decode** → Base64 decoded to OpenCV image
5. **Pose Estimation** → MediaPipe processes image (VIDEO mode)
6. **Keypoint Extraction** → 33 landmarks extracted
7. **Coordinate Scaling** → Normalized (0-1) → Pixel (640x480)
8. **WebSocket Response** → Pose result sent back to frontend
9. **State Update** → React hook updates `currentPose` state
10. **Overlay Rendering** → Canvas draws keypoints and skeleton

### Breakpoints to Check:
- [ ] Frame captured from video
- [ ] Frame sent via WebSocket
- [ ] Backend receives frame
- [ ] MediaPipe processes frame
- [ ] Keypoints extracted (33 points)
- [ ] Coordinates scaled correctly
- [ ] Response sent to frontend
- [ ] Frontend receives response
- [ ] currentPose state updated
- [ ] Canvas draws overlay

## Quick Diagnostic Commands

### Check Backend Health
```bash
curl http://localhost:8000/api/realtime/health
```

Expected response:
```json
{
  "success": true,
  "health": {
    "service": "realtime",
    "status": "healthy",
    "ready": true
  }
}
```

### Check Processing Modes
```bash
curl http://localhost:8000/api/realtime/processing-modes
```

### Check Model Info
```bash
curl http://localhost:8000/api/realtime/model-info
```

## Performance Expectations

### Normal Operation:
- **FPS**: 25-30 frames per second
- **Latency**: 20-40ms per frame
- **Keypoints**: 33 landmarks detected
- **Confidence**: 70-95% average
- **CPU Usage**: 30-50% (balanced mode)

### Degraded Performance:
- **FPS**: < 20 frames per second
- **Latency**: > 50ms per frame
- **Keypoints**: < 20 landmarks detected
- **Confidence**: < 50% average

## Next Steps

### If Overlay Works:
1. Remove debug logging from both files
2. Test with different lighting conditions
3. Test with different camera distances
4. Verify gait metrics are calculating correctly

### If Overlay Doesn't Work:
1. Share console debug output
2. Share backend log output
3. Share screenshot of browser console
4. Share screenshot of video feed
5. Describe what you see (or don't see)

## Files Modified

1. `frontend/hooks/useRealtimeAnalysis.ts` - Added debug logging
2. `frontend/components/realtime/RealtimeCamera.tsx` - Added debug logging
3. `ambient/realtime/gait_analyzer.py` - Normalized metrics (previous fix)
4. `frontend/components/realtime/RealtimeMetrics.tsx` - Updated ranges (previous fix)

## Debug Logs to Remove Later

Once issue is identified, remove these console.log statements:

**In `useRealtimeAnalysis.ts`:**
```typescript
console.log('[DEBUG] Received pose with', ...);
console.log('[DEBUG] No pose data in message');
console.log('[DEBUG] Pose result not successful:', ...);
```

**In `RealtimeCamera.tsx`:**
```typescript
console.log('[DEBUG] Drawing pose overlay with', ...);
console.log('[DEBUG] No currentPose to draw');
console.log('[DEBUG] Overlay hidden');
```

## Technical Architecture

### Frontend Components:
- `RealtimeCamera` - Video display and overlay rendering
- `useRealtimeAnalysis` - WebSocket management and state
- `RealtimeMetrics` - Gait metrics display
- `RealtimeStats` - Performance statistics

### Backend Components:
- `RealtimeService` - Session management
- `StreamProcessor` - Frame processing coordination
- `RealtimePoseEstimator` - MediaPipe wrapper
- `RealtimeGaitAnalyzer` - Gait metrics calculation

### Communication:
- WebSocket protocol for bidirectional real-time communication
- JSON message format
- Base64 encoded JPEG frames
- Pixel-space keypoint coordinates (640x480)
