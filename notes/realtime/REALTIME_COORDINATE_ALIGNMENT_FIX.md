# Realtime Coordinate Alignment Fix - January 25, 2026

## Problem

The pose keypoints and skeleton overlay were shifted away from the person being tracked, appearing in the upper-left portion of the video feed instead of aligned with the person's body.

## Root Cause

**Coordinate Space Mismatch:**

1. **Frontend** captures frames at 640x480 resolution
2. **Backend** processes frames at 640x480 and returns keypoints scaled to 640x480 coordinates
3. **Frontend** canvas was being sized to the video element's natural dimensions (e.g., 1280x720)
4. **Result**: Keypoints at 640x480 coordinates were drawn on a larger canvas, causing misalignment

### Example of the Problem:
- Video natural size: 1280x720
- Processed frame size: 640x480
- Keypoint at (320, 240) should be at center
- Canvas sized to 1280x720
- Keypoint drawn at (320, 240) on 1280x720 canvas = upper-left quadrant ❌

## Solution

**Match Canvas Dimensions to Processed Frame Dimensions:**

Set the canvas internal dimensions to 640x480 (matching the keypoint coordinate space), while keeping the canvas display size at 100% of the video element. The browser automatically scales the canvas content to fit.

### How It Works:
- Canvas internal dimensions: 640x480 (matches keypoint coordinates)
- Canvas display size: 100% of video element (via CSS `w-full h-full`)
- Browser scales canvas content automatically
- Keypoints align perfectly ✓

## Changes Made

### Frontend (RealtimeCamera.tsx)

**Before:**
```typescript
const { videoWidth, videoHeight } = videoRef.current;
setVideoSize({ width: videoWidth, height: videoHeight });

if (canvasRef.current) {
    canvasRef.current.width = videoWidth;  // ❌ Wrong - uses video's natural size
    canvasRef.current.height = videoHeight;
}
```

**After:**
```typescript
const { videoWidth, videoHeight } = videoRef.current;

// Set canvas to match the PROCESSED frame dimensions (640x480)
// not the video's natural dimensions, since keypoints are scaled to 640x480
const processedWidth = 640;
const processedHeight = 480;

setVideoSize({ width: processedWidth, height: processedHeight });

if (canvasRef.current) {
    canvasRef.current.width = processedWidth;   // ✓ Correct - matches keypoint coordinates
    canvasRef.current.height = processedHeight;
}
```

### Backend (pose_estimator.py)

**Removed DEBUG logs** that were cluttering the console during real-time processing:
- Removed: `logger.debug(f"Processing {len(landmarks)} landmarks for frame {frame_width}x{frame_height}")`
- Removed: `logger.debug(f"Created {len(keypoints)} keypoints with pixel coordinates")`

## Technical Details

### Canvas Coordinate System

HTML Canvas has two dimension concepts:

1. **Internal Dimensions** (`canvas.width`, `canvas.height`):
   - The actual pixel grid for drawing
   - Defines the coordinate space
   - Set to 640x480 to match keypoint coordinates

2. **Display Dimensions** (CSS `width`, `height`):
   - How the canvas is displayed on screen
   - Set to 100% via `w-full h-full` classes
   - Browser scales the internal content to fit

### Data Flow

```
Camera (1280x720 native)
  ↓
Frontend captures at 640x480
  ↓
JPEG encode & send to backend
  ↓
Backend processes at 640x480
  ↓
MediaPipe returns normalized coordinates (0-1)
  ↓
Backend scales to 640x480 pixel coordinates
  ↓
Send keypoints to frontend
  ↓
Frontend canvas: 640x480 internal, 100% display
  ↓
Draw keypoints at their coordinates
  ↓
Browser scales canvas to video size
  ↓
Perfect alignment ✓
```

## Verification

### Before Fix:
- Keypoints appeared in upper-left corner
- Skeleton overlay shifted away from person
- Misalignment increased with larger video display sizes

### After Fix:
- Keypoints perfectly aligned with body parts
- Skeleton overlay tracks person accurately
- Alignment maintained at any display size

## Testing

1. Start backend: `uvicorn server.main:app --reload`
2. Start frontend: `cd frontend && npm run dev`
3. Navigate to `/realtime`
4. Click "Start Analysis"
5. Verify:
   - ✓ Keypoints align with body parts (nose, shoulders, hips, etc.)
   - ✓ Skeleton lines connect correctly
   - ✓ Overlay tracks movement accurately
   - ✓ No shifting or misalignment

## Files Modified

1. **frontend/components/realtime/RealtimeCamera.tsx**
   - Changed canvas dimensions to match processed frame size (640x480)
   - Added explanatory comments

2. **ambient/realtime/pose_estimator.py**
   - Removed DEBUG log statements for cleaner output

## Related Issues

This fix resolves the coordinate alignment issue while maintaining all previous optimizations:
- ✓ Larger keypoints (8px radius)
- ✓ Thicker skeleton lines (6px width)
- ✓ Low latency (~10ms processing)
- ✓ Smooth tracking (no frame skipping)
- ✓ Perfect alignment (coordinate space match)

## Key Takeaway

**Always match canvas internal dimensions to the coordinate space of the data being drawn.**

When processing frames at a different resolution than the display size, the canvas must be sized to match the coordinate space of the processed data, not the display size. The browser will handle scaling automatically.
