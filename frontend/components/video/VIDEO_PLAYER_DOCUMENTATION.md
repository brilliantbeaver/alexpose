# Video Player Component - Documentation

## Overview

A fully-featured custom video player built specifically for gait analysis videos with frame-by-frame navigation, playback speed control, and comprehensive controls.

## Features

### Core Playback Controls
✅ **Play/Pause** - Standard video playback control  
✅ **Seek Bar** - Drag to any point in the video  
✅ **Time Display** - Current time and total duration  
✅ **Volume Control** - Adjustable volume with mute toggle  
✅ **Fullscreen** - Expand to fullscreen mode  

### Advanced Features
✅ **Frame-by-Frame Navigation** - Step through individual frames  
✅ **Playback Speed Control** - 0.25x to 2x speed  
✅ **Skip Forward/Backward** - Jump 5 seconds  
✅ **Frame Counter** - Real-time frame number display  
✅ **Loading State** - Visual feedback during video load  
✅ **Error Handling** - Graceful error messages  

### Gait Analysis Specific
✅ **Frame Rate Display** - Shows FPS for accurate analysis  
✅ **Frame Number Overlay** - Always visible frame counter  
✅ **Precise Seeking** - Frame-accurate navigation  
✅ **Slow Motion** - Detailed movement analysis  

## Component API

### Props

```typescript
interface VideoPlayerProps {
  videoUrl: string;           // URL to video file
  videoName: string;          // Display name for video
  frameRate?: number;         // Video frame rate (default: 30)
  onTimeUpdate?: (currentTime: number) => void;  // Callback on time change
  onFrameChange?: (frameNumber: number) => void; // Callback on frame change
}
```

### Usage Example

```tsx
import { VideoPlayer } from '@/components/video/VideoPlayer';

export default function AnalysisPage() {
  return (
    <VideoPlayer
      videoUrl="https://example.com/video.mp4"
      videoName="Walking Test 1"
      frameRate={30}
      onTimeUpdate={(time) => {
        console.log('Current time:', time);
      }}
      onFrameChange={(frame) => {
        console.log('Current frame:', frame);
      }}
    />
  );
}
```

## User Interface

### Video Container
```
┌─────────────────────────────────────────────┐
│ Walking Test 1              Frame: 450/1350 │
│                                             │
│                                             │
│              [Video Content]                │
│                                             │
│              [Play/Pause Overlay]           │
│                                             │
└─────────────────────────────────────────────┘
```

### Control Bar
```
┌─────────────────────────────────────────────┐
│ ████████████░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░ │ Progress Bar
│ 00:15                              00:45    │ Time Display
├─────────────────────────────────────────────┤
│ [▶️] [⏪] [⏮] [⏭] [⏩] [🔊] [━━━━━]         │ Left Controls
│                        [1x ▼] [⛶]          │ Right Controls
├─────────────────────────────────────────────┤
│ Frame Rate: 30 fps • Duration: 00:45       │ Info
│ Speed: 1x • Frame: 450/1350                │
└─────────────────────────────────────────────┘
```

## Controls Reference

### Playback Controls

| Button | Function | Keyboard Shortcut |
|--------|----------|-------------------|
| ▶️/⏸ | Play/Pause | Space |
| ⏪ | Skip backward 5s | Left Arrow |
| ⏮ | Previous frame | , (comma) |
| ⏭ | Next frame | . (period) |
| ⏩ | Skip forward 5s | Right Arrow |

### Volume Controls

| Button | Function | Keyboard Shortcut |
|--------|----------|-------------------|
| 🔊/🔇 | Mute/Unmute | M |
| Slider | Adjust volume | Up/Down Arrow |

### View Controls

| Button | Function | Keyboard Shortcut |
|--------|----------|-------------------|
| ⛶ | Fullscreen | F |
| Speed | Playback rate | +/- |

## Technical Implementation

### Frame Calculation

```typescript
// Convert time to frame number
const timeToFrame = (time: number) => Math.floor(time * frameRate);

// Convert frame number to time
const frameToTime = (frame: number) => frame / frameRate;
```

### State Management

```typescript
const [isPlaying, setIsPlaying] = useState(false);
const [currentTime, setCurrentTime] = useState(0);
const [duration, setDuration] = useState(0);
const [currentFrame, setCurrentFrame] = useState(0);
const [totalFrames, setTotalFrames] = useState(0);
const [playbackRate, setPlaybackRate] = useState(1);
const [volume, setVolume] = useState(1);
```

### Event Handlers

```typescript
// Time update event
video.addEventListener('timeupdate', () => {
  const time = video.currentTime;
  setCurrentTime(time);
  setCurrentFrame(timeToFrame(time));
  onTimeUpdate?.(time);
  onFrameChange?.(timeToFrame(time));
});

// Metadata loaded event
video.addEventListener('loadedmetadata', () => {
  setDuration(video.duration);
  setTotalFrames(Math.floor(video.duration * frameRate));
});
```

## Supported Video Formats

### Browser Support
- **MP4** (H.264) - ✅ All browsers
- **WebM** (VP8/VP9) - ✅ Chrome, Firefox, Edge
- **OGG** (Theora) - ✅ Firefox, Chrome
- **MOV** (H.264) - ⚠️ Safari only

### Recommended Format
**MP4 with H.264 codec** for maximum compatibility.

### Encoding Recommendations
```bash
# FFmpeg command for optimal web playback
ffmpeg -i input.mp4 \
  -c:v libx264 \
  -preset slow \
  -crf 22 \
  -c:a aac \
  -b:a 128k \
  -movflags +faststart \
  output.mp4
```

## Performance Considerations

### Video Loading
- Uses `preload="metadata"` for fast initial load
- Loads full video on play
- Shows loading indicator during buffering

### Memory Management
- Video element properly cleaned up on unmount
- Event listeners removed to prevent memory leaks
- Efficient state updates

### Optimization Tips
1. **Use CDN** for video hosting
2. **Compress videos** before upload
3. **Enable streaming** for large files
4. **Use adaptive bitrate** for varying network speeds

## Accessibility

### ARIA Attributes
```tsx
<video
  aria-label={videoName}
  aria-describedby="video-controls"
/>
```

### Keyboard Navigation
- All controls accessible via keyboard
- Tab through interactive elements
- Space for play/pause
- Arrow keys for seeking

### Screen Reader Support
- Button labels announced
- Time updates announced
- State changes announced

## Error Handling

### Error States

1. **Video Load Error**
```
⚠️
Failed to load video. Please check the video URL.
https://example.com/video.mp4
```

2. **Unsupported Format**
```
⚠️
Your browser does not support this video format.
Try using MP4 format.
```

3. **Network Error**
```
⚠️
Network error. Please check your connection.
```

### Error Recovery
- Displays user-friendly error messages
- Shows video URL for debugging
- Provides fallback UI

## Integration with Analysis

### Time-Based Annotations

```tsx
<VideoPlayer
  videoUrl={videoUrl}
  videoName={name}
  onTimeUpdate={(time) => {
    // Show gait metrics for current time
    updateMetricsDisplay(time);
  }}
/>
```

### Frame-Based Analysis

```tsx
<VideoPlayer
  videoUrl={videoUrl}
  videoName={name}
  frameRate={30}
  onFrameChange={(frame) => {
    // Show pose data for current frame
    updatePoseOverlay(frame);
  }}
/>
```

### Synchronized Displays

```tsx
const [currentFrame, setCurrentFrame] = useState(0);

<VideoPlayer
  onFrameChange={setCurrentFrame}
/>

<MetricsPanel frame={currentFrame} />
<PoseVisualization frame={currentFrame} />
```

## Future Enhancements

### Phase 1: Pose Overlay
- [ ] Draw skeleton overlay on video
- [ ] Highlight key joints
- [ ] Show joint angles
- [ ] Toggle overlay on/off

### Phase 2: Metrics Overlay
- [ ] Real-time gait metrics display
- [ ] Cadence indicator
- [ ] Step length visualization
- [ ] Symmetry comparison

### Phase 3: Comparison Mode
- [ ] Side-by-side video comparison
- [ ] Synchronized playback
- [ ] Difference highlighting
- [ ] Split-screen view

### Phase 4: Advanced Features
- [ ] Video trimming
- [ ] Slow-motion regions
- [ ] Bookmarks/markers
- [ ] Export with annotations

## Styling

### Customization

```tsx
// Custom colors
<VideoPlayer
  className="custom-player"
  videoUrl={url}
  videoName={name}
/>

// CSS
.custom-player video {
  border-radius: 12px;
  box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
}
```

### Responsive Design
- Adapts to container width
- Maintains 16:9 aspect ratio
- Mobile-friendly controls
- Touch-optimized buttons

## Testing

### Unit Tests

```typescript
describe('VideoPlayer', () => {
  it('should render video element', () => {
    render(<VideoPlayer videoUrl="test.mp4" videoName="Test" />);
    expect(screen.getByRole('video')).toBeInTheDocument();
  });

  it('should toggle play/pause', () => {
    render(<VideoPlayer videoUrl="test.mp4" videoName="Test" />);
    const playButton = screen.getByRole('button', { name: /play/i });
    fireEvent.click(playButton);
    expect(video.paused).toBe(false);
  });

  it('should navigate frames', () => {
    const onFrameChange = jest.fn();
    render(
      <VideoPlayer
        videoUrl="test.mp4"
        videoName="Test"
        onFrameChange={onFrameChange}
      />
    );
    const nextButton = screen.getByTitle('Next frame');
    fireEvent.click(nextButton);
    expect(onFrameChange).toHaveBeenCalled();
  });
});
```

### Integration Tests

```typescript
describe('VideoPlayer Integration', () => {
  it('should sync with metrics display', async () => {
    const { getByRole } = render(
      <AnalysisPage videoId="1" />
    );
    
    const video = getByRole('video');
    fireEvent.timeUpdate(video, { currentTime: 5.0 });
    
    await waitFor(() => {
      expect(screen.getByText(/Frame: 150/)).toBeInTheDocument();
    });
  });
});
```

## Browser Compatibility

| Browser | Version | Support |
|---------|---------|---------|
| Chrome | 90+ | ✅ Full |
| Firefox | 88+ | ✅ Full |
| Safari | 14+ | ✅ Full |
| Edge | 90+ | ✅ Full |
| Opera | 76+ | ✅ Full |

### Polyfills
Not required - uses native HTML5 video API.

## Dependencies

### Required
- React 18+
- Shadcn UI components:
  - Button
  - Slider
  - Select
  - Card

### Optional
- None - fully self-contained

## File Structure

```
components/video/
├── VideoPlayer.tsx              # Main component
├── VIDEO_PLAYER_DOCUMENTATION.md # This file
└── __tests__/
    └── VideoPlayer.test.tsx     # Unit tests
```

## Summary

The VideoPlayer component provides a professional, feature-rich video playback experience specifically designed for gait analysis. Key features include:

✅ **Frame-accurate navigation** for precise analysis  
✅ **Variable playback speed** for detailed examination  
✅ **Comprehensive controls** for professional use  
✅ **Error handling** for robust operation  
✅ **Accessibility** for all users  
✅ **Extensible** for future enhancements  

Perfect for medical professionals analyzing gait patterns with precision and ease.

---

**Created**: January 3, 2026  
**Version**: 1.0.0  
**Status**: ✅ Production Ready
