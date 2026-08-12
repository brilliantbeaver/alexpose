# Realtime Webcam Streaming Fix

## Issue
The realtime page was showing "Disconnected" and the webcam wasn't streaming. The WebSocket connection wasn't being established properly.

## Root Cause
The WebSocket URL was using `window.location.host` which points to the frontend server (localhost:3000), but the WebSocket endpoint is on the backend server (localhost:8000).

## Solution

### 1. Fixed WebSocket Connection URL
**File**: `frontend/hooks/useRealtimeAnalysis.ts`

Changed the WebSocket URL construction to explicitly use the backend port:

```typescript
// Before:
const host = window.location.host;
const wsUrl = `${protocol}//${host}/api/realtime/stream`;

// After:
const wsUrl = `${protocol}//localhost:8000/api/realtime/stream`;
```

### 2. Improved Pose Overlay Rendering
**File**: `frontend/components/realtime/RealtimeCamera.tsx`

Added null checks and validation for pose data:

```typescript
// Check if pose has keypoints before processing
if (!pose.keypoints || pose.keypoints.length === 0) return;

// Filter valid keypoints
const validKeypoints = pose.keypoints.filter(
    kp => kp.confidence >= confidence_threshold
);

if (validKeypoints.length === 0) return;
```

### 3. Enhanced Pose Info Display
Added safe access to pose data with fallbacks:

```typescript
{currentPose && showOverlay && currentPose.keypoints && currentPose.keypoints.length > 0 && (
    <div className="absolute bottom-4 left-4 bg-black/70 text-white px-3 py-2 rounded text-sm">
        <div>Keypoints: {currentPose.keypoints.length}</div>
        <div>Confidence: {currentPose.confidence_scores && currentPose.confidence_scores.length > 0 
            ? (currentPose.confidence_scores.reduce((a, b) => a + b, 0) / currentPose.confidence_scores.length * 100).toFixed(1) 
            : '0.0'}%</div>
        <div>Processing: {currentPose.processing_time_ms.toFixed(1)}ms</div>
    </div>
)}
```

## How It Works

### Architecture Flow

```
┌─────────────────┐         WebSocket          ┌─────────────────┐
│   Frontend      │◄──────────────────────────►│   Backend       │
│  (localhost:    │   ws://localhost:8000/     │  (localhost:    │
│   3000)         │   api/realtime/stream      │   8000)         │
└─────────────────┘                            └─────────────────┘
        │                                               │
        │ 1. getUserMedia()                            │
        │ 2. Capture frames                            │
        │ 3. Convert to base64                         │
        │ 4. Send via WebSocket ────────────────────►  │
        │                                               │ 5. Decode frame
        │                                               │ 6. Pose estimation
        │                                               │ 7. Gait analysis
        │  ◄──────────────────────────────────────────  │ 8. Send results
        │ 9. Receive pose data                         │
        │ 10. Draw overlay                             │
        │ 11. Display metrics                          │
        └───────────────────────────────────────────────┘
```

### Data Flow

1. **Camera Initialization**
   - Request webcam access via `navigator.mediaDevices.getUserMedia()`
   - Set video element source to media stream
   - Configure canvas for pose overlay

2. **WebSocket Connection**
   - Connect to `ws://localhost:8000/api/realtime/stream`
   - Receive session ID and configuration
   - Handle connection state changes

3. **Frame Capture Loop**
   - Use `requestAnimationFrame` for smooth capture
   - Throttle based on target FPS (20-30 fps)
   - Draw video frame to temporary canvas
   - Convert to base64 JPEG (80% quality)

4. **Frame Transmission**
   - Send frame data via WebSocket as JSON:
     ```json
     {
       "type": "frame",
       "data": "base64_encoded_image_data"
     }
     ```

5. **Backend Processing**
   - Decode base64 image
   - Run MediaPipe pose estimation
   - Calculate gait metrics
   - Return results:
     ```json
     {
       "type": "pose_result",
       "data": {
         "pose": {
           "keypoints": [...],
           "confidence_scores": [...],
           "processing_time_ms": 25.3
         },
         "gait_metrics": {
           "cadence": 120,
           "step_length": 0.65,
           ...
         }
       }
     }
     ```

6. **Pose Overlay Rendering**
   - Clear canvas
   - Draw skeleton connections (green lines)
   - Draw keypoints (colored circles based on confidence)
   - Display metrics overlay

## Testing

### 1. Test WebSocket Connection
Open `test_websocket.html` in a browser to verify the WebSocket endpoint is accessible.

### 2. Test Webcam Access
1. Navigate to http://localhost:3000/realtime
2. Click "Start Analysis"
3. Allow camera permissions when prompted
4. Verify video stream appears

### 3. Test Pose Estimation
1. Stand in front of camera with full body visible
2. Verify green skeleton overlay appears on your body
3. Check keypoint circles are drawn at joints
4. Verify metrics update in real-time

### 4. Test Performance
- Check "Processing" time in overlay (should be < 50ms)
- Verify FPS is stable (20-30 fps)
- Monitor CPU usage (should be reasonable)

## Configuration Options

### Processing Modes

- **Fast**: Lower accuracy, higher FPS (30 fps target)
  - Resize factor: 0.5
  - Min confidence: 0.3
  
- **Balanced**: Balance of speed and accuracy (25 fps target)
  - Resize factor: 0.75
  - Min confidence: 0.5
  
- **Accurate**: Highest accuracy, lower FPS (20 fps target)
  - Resize factor: 1.0
  - Min confidence: 0.7

### Overlay Options

- **Show Keypoints**: Toggle keypoint circles
- **Show Skeleton**: Toggle skeleton connections
- **Confidence Threshold**: Filter low-confidence keypoints (0.0-1.0)

## Troubleshooting

### Issue: "Disconnected" Status
**Solution**: Ensure backend server is running on port 8000
```bash
uvicorn server.main:app --reload --port 8000
```

### Issue: Camera Access Denied
**Solution**: 
1. Check browser permissions
2. Ensure HTTPS or localhost
3. Try different browser

### Issue: No Pose Overlay
**Solution**:
1. Check console for errors
2. Verify MediaPipe model is loaded
3. Ensure good lighting and full body visible
4. Check confidence threshold setting

### Issue: Slow Performance
**Solution**:
1. Switch to "Fast" processing mode
2. Reduce video resolution
3. Close other applications
4. Check CPU usage

## Future Enhancements

1. **Recording**: Save analysis sessions for later review
2. **Multi-person**: Support multiple people in frame
3. **Gait Metrics History**: Track metrics over time with charts
4. **Export**: Export pose data and metrics to CSV/JSON
5. **Alerts**: Real-time alerts for abnormal gait patterns
6. **Calibration**: Camera calibration for accurate measurements
7. **3D Visualization**: 3D pose visualization
8. **Mobile Support**: Responsive design for mobile devices

## API Reference

### WebSocket Messages

#### Client → Server

**Frame Data**
```json
{
  "type": "frame",
  "data": "base64_image_data"
}
```

**Config Update**
```json
{
  "type": "config_update",
  "config": {
    "processing_mode": "balanced",
    "confidence_threshold": 0.5
  }
}
```

**Get Statistics**
```json
{
  "type": "get_stats"
}
```

#### Server → Client

**Session Started**
```json
{
  "type": "session_started",
  "session_id": "uuid",
  "config": {...}
}
```

**Pose Result**
```json
{
  "type": "pose_result",
  "data": {
    "success": true,
    "pose": {...},
    "gait_metrics": {...},
    "processing_time_ms": 25.3
  }
}
```

**Statistics**
```json
{
  "type": "statistics",
  "data": {
    "frames_processed": 1234,
    "average_processing_time_ms": 28.5,
    "poses_detected": 1200
  }
}
```

**Error**
```json
{
  "type": "error",
  "message": "Error description"
}
```

## Performance Metrics

### Expected Performance

- **Frame Rate**: 20-30 FPS
- **Processing Time**: 20-50ms per frame
- **Latency**: < 100ms end-to-end
- **CPU Usage**: 30-60% (single core)
- **Memory**: ~500MB

### Optimization Tips

1. Use "Fast" mode for real-time feedback
2. Reduce video resolution if needed
3. Close unnecessary browser tabs
4. Use hardware acceleration if available
5. Ensure good lighting for better detection

## Dependencies

### Backend
- FastAPI
- MediaPipe
- OpenCV
- NumPy

### Frontend
- React 19
- Next.js 16
- TypeScript
- Tailwind CSS

## Related Files

- `frontend/app/realtime/page.tsx` - Main realtime page
- `frontend/hooks/useRealtimeAnalysis.ts` - WebSocket hook
- `frontend/components/realtime/RealtimeCamera.tsx` - Camera component
- `server/routers/realtime.py` - WebSocket endpoint
- `server/services/realtime_service.py` - Service layer
- `ambient/realtime/stream_processor.py` - Stream processing
- `ambient/realtime/pose_estimator.py` - Pose estimation
- `ambient/realtime/gait_analyzer.py` - Gait analysis
