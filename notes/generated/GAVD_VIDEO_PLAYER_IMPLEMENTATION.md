# GAVD Video Player Implementation

## Overview

This document describes the robust video player implementation for GAVD dataset analysis, including frame-accurate seeking, bounding box overlay, and pose keypoint visualization.

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    Frontend Video Player                        │
│                  (GAVDVideoPlayer.tsx)                          │
│                                                                 │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐        │
│  │   Video      │  │   Canvas     │  │   Controls   │        │
│  │   Element    │  │   Overlay    │  │   (Play/     │        │
│  │              │  │              │  │   Seek/Nav)  │        │
│  └──────────────┘  └──────────────┘  └──────────────┘        │
└────────────────────────┬────────────────────────────────────────┘
                         │ HTTP/REST API
                         ▼
┌─────────────────────────────────────────────────────────────────┐
│                   Backend Video Streaming                       │
│                  (server/routers/video.py)                      │
│                                                                 │
│  ┌──────────────────────────────────────────────────────────┐ │
│  │ Range Request Support                                    │ │
│  │ - Partial content (206)                                  │ │
│  │ - Efficient seeking                                      │ │
│  │ - Bandwidth optimization                                 │ │
│  └──────────────────────────────────────────────────────────┘ │
└────────────────────────┬────────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────────┐
│                   Cached YouTube Videos                         │
│                    (data/youtube/)                              │
└─────────────────────────────────────────────────────────────────┘
```

## Features

### 1. Video Streaming with Range Requests

**Backend (`server/routers/video.py`):**
- `/api/v1/video/stream/{video_id}` - Stream video with range support
- `/api/v1/video/info/{video_id}` - Get video metadata
- `/api/v1/video/url-to-id` - Extract video ID from YouTube URL

**Key Features:**
- ✅ HTTP Range request support (RFC 7233)
- ✅ Partial content delivery (206 status)
- ✅ Efficient seeking without full download
- ✅ Multiple format support (.mp4, .webm, .mkv, .mov)
- ✅ Caching headers for performance

**Example Range Request:**
```http
GET /api/v1/video/stream/B5hrxKe2nP8
Range: bytes=0-1023

HTTP/1.1 206 Partial Content
Content-Range: bytes 0-1023/5242880
Content-Length: 1024
Content-Type: video/mp4
```

### 2. Frame-Accurate Seeking

**Implementation:**
- Convert frame number to video time: `time = (frame_num - 1) / fps`
- Use HTML5 video `currentTime` property for precise seeking
- Synchronize video position with frame slider
- Handle frame-by-frame navigation

**Frame Navigation:**
- ⏮ Previous Frame: `currentFrameIndex - 1`
- ⏭ Next Frame: `currentFrameIndex + 1`
- 🎯 Direct Seek: Slider to any frame
- ⌨️ Keyboard: Arrow keys for navigation

### 3. Bounding Box Overlay

**Features:**
- ✅ Real-time canvas overlay on video
- ✅ Automatic scaling for resolution differences
- ✅ Red rectangle with frame number label
- ✅ Toggle on/off with checkbox
- ✅ Synchronized with video seeking

**Scaling Algorithm:**
```javascript
// Scale bbox if video resolution differs from annotation
const scaleX = videoWidth / annotationWidth;
const scaleY = videoHeight / annotationHeight;

const scaledLeft = bbox.left * scaleX;
const scaledTop = bbox.top * scaleY;
const scaledWidth = bbox.width * scaleX;
const scaledHeight = bbox.height * scaleY;
```

**Visual Style:**
- Color: Red (#FF0000)
- Line Width: 3px
- Label: Frame number on red background
- Font: 14px Arial

### 4. Pose Keypoint Visualization

**Features:**
- ✅ 25-point BODY_25 skeleton (OpenPose format)
- ✅ Keypoint circles with confidence-based opacity
- ✅ Skeleton connections between keypoints
- ✅ Confidence threshold filtering (>0.3)
- ✅ Toggle on/off with checkbox

**Keypoint Rendering:**
- Circle radius: 4px
- Color: Red with alpha based on confidence
- Border: White 1px stroke
- ID labels for high-confidence keypoints (>0.7)

**Skeleton Connections:**
- Color: Green (#00FF00)
- Line Width: 2px
- Only drawn if both keypoints have confidence >0.3

**BODY_25 Skeleton Structure:**
```
0: Nose
1: Neck
2-4: Right arm (shoulder, elbow, wrist)
5-7: Left arm (shoulder, elbow, wrist)
8-10: Right leg (hip, knee, ankle)
11-13: Left leg (hip, knee, ankle)
14-17: Face (eyes, ears)
19-24: Feet (toes, heels)
```

### 5. Playback Controls

**Controls:**
- ▶️ Play/Pause button
- ⏮ Previous frame button
- ⏭ Next frame button
- 🎚️ Frame slider (scrubbing)
- ⌨️ Keyboard shortcuts

**Keyboard Shortcuts:**
- `Space`: Play/Pause
- `←`: Previous frame
- `→`: Next frame

**Playback Features:**
- Auto-update frame index during playback
- Pause on frame navigation
- Smooth seeking with requestAnimationFrame
- Disabled controls when video not ready

### 6. Status Indicators

**Loading States:**
- Video loading spinner
- "Loading video..." message
- Disabled controls until ready

**Error States:**
- Video not cached error
- Invalid URL error
- Network error handling

**Info Display:**
- Current frame number
- Total frame count
- Video time (current/duration)
- FPS estimate
- Gait event label

## API Endpoints

### Video Streaming

**GET `/api/v1/video/stream/{video_id}`**
- Stream cached YouTube video
- Supports HTTP Range requests
- Returns 206 Partial Content or 200 OK

**GET `/api/v1/video/info/{video_id}`**
- Get video metadata
- Returns file size, format, availability

**GET `/api/v1/video/url-to-id?url={youtube_url}`**
- Extract video ID from YouTube URL
- Returns video ID and stream URL

### GAVD Data

**GET `/api/v1/gavd/sequence/{dataset_id}/{sequence_id}/frames`**
- Get all frames for a sequence
- Returns frame metadata with bbox and vid_info

**GET `/api/v1/gavd/sequence/{dataset_id}/{sequence_id}/frame/{frame_num}/pose`**
- Get pose keypoints for a specific frame
- Returns array of keypoint objects

## Component Props

### GAVDVideoPlayer

```typescript
interface GAVDVideoPlayerProps {
  frames: FrameData[];              // Array of frame data
  currentFrameIndex: number;        // Current frame index (0-based)
  onFrameChange: (index: number) => void;  // Frame change callback
  showBbox?: boolean;               // Show bounding box overlay
  showPose?: boolean;               // Show pose keypoint overlay
  autoPlay?: boolean;               // Auto-play on load
}
```

### FrameData

```typescript
interface FrameData {
  frame_num: number;                // Frame number (1-based)
  bbox: BoundingBox;                // Bounding box coordinates
  vid_info: VideoInfo;              // Video resolution info
  url: string;                      // YouTube URL
  gait_event?: string;              // Gait event label
  cam_view?: string;                // Camera view
  pose_keypoints?: PoseKeypoint[];  // Pose keypoints
}
```

### PoseKeypoint

```typescript
interface PoseKeypoint {
  x: number;                        // X coordinate
  y: number;                        // Y coordinate
  confidence: number;               // Confidence score (0-1)
  keypoint_id: number;              // Keypoint ID (0-24)
}
```

## Usage Example

```tsx
import GAVDVideoPlayer from '@/components/GAVDVideoPlayer';

function AnalysisPage() {
  const [frames, setFrames] = useState<FrameData[]>([]);
  const [currentFrame, setCurrentFrame] = useState(0);
  const [showBbox, setShowBbox] = useState(true);
  const [showPose, setShowPose] = useState(false);

  return (
    <div>
      <div className="controls">
        <label>
          <input
            type="checkbox"
            checked={showBbox}
            onChange={(e) => setShowBbox(e.target.checked)}
          />
          Show Bounding Box
        </label>
        <label>
          <input
            type="checkbox"
            checked={showPose}
            onChange={(e) => setShowPose(e.target.checked)}
          />
          Show Pose Overlay
        </label>
      </div>

      <GAVDVideoPlayer
        frames={frames}
        currentFrameIndex={currentFrame}
        onFrameChange={setCurrentFrame}
        showBbox={showBbox}
        showPose={showPose}
      />
    </div>
  );
}
```

## Testing

### Unit Tests

**Video Streaming (`tests/test_video_streaming.py`):**
- ✅ Full video streaming
- ✅ Range request handling
- ✅ Video not found error
- ✅ Video info retrieval
- ✅ URL to video ID extraction
- ✅ Multiple range requests

**Visualization (`tests/test_gavd_visualization.py`):**
- ✅ Bounding box data retrieval
- ✅ Bbox scaling calculation
- ✅ Pose keypoint retrieval
- ✅ Confidence filtering
- ✅ Frame navigation
- ✅ Data consistency

### Integration Tests

**End-to-End Flow:**
1. Upload GAVD CSV file
2. Process dataset (download videos, extract frames, run pose estimation)
3. Navigate to analysis page
4. Select sequence
5. Video player loads and displays video
6. Toggle bounding box overlay
7. Toggle pose overlay
8. Navigate frames with controls
9. Verify overlays update correctly

### Running Tests

```bash
# Run all tests
pytest tests/test_video_streaming.py tests/test_gavd_visualization.py -v

# Run specific test class
pytest tests/test_video_streaming.py::TestVideoStreaming -v

# Run with coverage
pytest tests/ --cov=server --cov=frontend/components --cov-report=html
```

## Performance Considerations

### Video Streaming
- **Range Requests**: Only download needed portions
- **Chunk Size**: 8KB chunks for smooth streaming
- **Caching**: Browser caches video segments
- **Preload**: Metadata only, not full video

### Canvas Rendering
- **Redraw on Seek**: Only redraw when video position changes
- **RequestAnimationFrame**: Smooth animation during playback
- **Conditional Rendering**: Only draw enabled overlays

### Data Loading
- **Lazy Loading**: Load pose data only when needed
- **Caching**: Cache pose data per frame
- **Batch Requests**: Load all frames at once, pose data on demand

## Browser Compatibility

### Supported Browsers
- ✅ Chrome 90+
- ✅ Firefox 88+
- ✅ Safari 14+
- ✅ Edge 90+

### Required Features
- HTML5 Video with range request support
- Canvas 2D rendering
- ES6+ JavaScript
- Fetch API

## Troubleshooting

### Video Not Loading
- Check if video is cached in `data/youtube/`
- Verify video ID extraction from URL
- Check backend server is running
- Inspect network tab for 404 errors

### Bounding Box Not Showing
- Verify `showBbox` is true
- Check bbox data exists in frame
- Verify canvas overlay is rendering
- Check scaling calculations

### Pose Overlay Not Showing
- Verify `showPose` is true
- Check pose data endpoint returns data
- Verify keypoint confidence >0.3
- Check canvas rendering

### Seeking Not Working
- Verify FPS calculation is correct
- Check frame number to time conversion
- Verify video metadata is loaded
- Check video duration is available

## Future Enhancements

### Short-term
- [ ] Playback speed control (0.5x, 1x, 2x)
- [ ] Frame export to image
- [ ] Zoom and pan controls
- [ ] Side-by-side comparison view

### Long-term
- [ ] Real-time pose estimation
- [ ] Gait cycle visualization
- [ ] Anomaly highlighting
- [ ] Multi-sequence comparison
- [ ] 3D pose visualization
- [ ] Slow-motion analysis

## References

- **HTTP Range Requests**: [RFC 7233](https://tools.ietf.org/html/rfc7233)
- **HTML5 Video**: [MDN Web Docs](https://developer.mozilla.org/en-US/docs/Web/HTML/Element/video)
- **Canvas API**: [MDN Web Docs](https://developer.mozilla.org/en-US/docs/Web/API/Canvas_API)
- **OpenPose BODY_25**: [OpenPose Documentation](https://github.com/CMU-Perceptual-Computing-Lab/openpose)
- **MediaPipe Pose**: [MediaPipe Documentation](https://google.github.io/mediapipe/solutions/pose.html)

## Support

For issues or questions:
1. Check browser console for errors
2. Verify backend logs for streaming errors
3. Test video streaming endpoint directly
4. Review this documentation
5. Check test files for examples
