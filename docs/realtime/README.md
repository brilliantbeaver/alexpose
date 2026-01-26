# Realtime Gait Analysis

The realtime gait analysis feature provides live webcam-based pose estimation and gait analysis with immediate visual feedback. This system is optimized for low-latency processing while maintaining accuracy for clinical applications.

## Overview

The realtime analysis system consists of several key components working together:

- **Frontend**: React-based webcam interface with pose overlay visualization
- **Backend**: FastAPI WebSocket server for frame processing
- **Processing Pipeline**: Optimized pose estimation and gait analysis
- **Frame Management**: Circular buffer for efficient memory usage

## Architecture

```
┌─────────────────┐
│   Web Browser   │
│   (Frontend)    │
│                 │
│  ┌───────────┐  │
│  │  Webcam   │  │
│  │  Capture  │  │
│  └─────┬─────┘  │
│        │        │
│  ┌─────▼─────┐  │
│  │  Canvas   │  │
│  │  Overlay  │  │
│  └───────────┘  │
└────────┬────────┘
         │ WebSocket
         │ (Frame Data)
         │
┌────────▼────────┐
│  FastAPI Server │
│                 │
│  ┌───────────┐  │
│  │  Stream   │  │
│  │ Processor │  │
│  └─────┬─────┘  │
│        │        │
│  ┌─────▼─────┐  │
│  │   Pose    │  │
│  │ Estimator │  │
│  └─────┬─────┘  │
│        │        │
│  ┌─────▼─────┐  │
│  │   Gait    │  │
│  │ Analyzer  │  │
│  └───────────┘  │
└─────────────────┘
```

## Features

### Real-time Pose Estimation
- MediaPipe-based pose detection
- 33-point body landmark tracking
- Confidence scoring for each keypoint
- Adaptive quality control

### Visual Feedback
- Live keypoint overlay on webcam feed
- Skeletal connections visualization
- Confidence-based color coding
- Toggleable overlay display

### Gait Analysis
- Cadence calculation (steps per minute)
- Step and stride length estimation
- Walking speed measurement
- Symmetry index (left-right balance)
- Stability score (trunk movement)

### Performance Optimization
- Three processing modes: Fast, Balanced, Accurate
- Adaptive frame skipping
- Circular frame buffer
- Memory management
- Pose tracking for temporal consistency

## Usage

### Starting Realtime Analysis

1. Navigate to the Realtime page in the web interface
2. Grant camera permissions when prompted
3. Click "Start Analysis" to begin processing
4. Walk in front of the camera to generate gait metrics
5. Adjust settings as needed for your use case
6. Click "Stop Analysis" when finished

### Configuration Options

#### Processing Mode
- **Fast**: ~30 FPS, lower accuracy, minimal CPU usage
- **Balanced**: ~25 FPS, good accuracy, moderate CPU usage
- **Accurate**: ~20 FPS, highest accuracy, higher CPU usage

#### Visual Settings
- **Show Keypoints**: Display individual body landmarks
- **Show Skeleton**: Display skeletal connections

#### Performance Settings
- **Confidence Threshold**: Minimum confidence for displaying keypoints (0.1-0.9)
- **Buffer Size**: Number of frames to keep in memory (10-60)
- **Target FPS**: Target frame processing rate (10-30)

#### Advanced Settings
- **Pose Tracking**: Enable temporal smoothing across frames

## API Reference

### WebSocket Endpoint

```
ws://localhost:8000/api/realtime/stream
```

#### Message Types

**Client → Server:**

```typescript
// Send frame for processing
{
  type: "frame",
  data: string  // base64 encoded image
}

// Update configuration
{
  type: "config_update",
  config: {
    processing_mode: "fast" | "balanced" | "accurate",
    confidence_threshold: number,
    buffer_size: number,
    enable_tracking: boolean
  }
}

// Request statistics
{
  type: "get_stats"
}

// Ping for keepalive
{
  type: "ping",
  timestamp: number
}
```

**Server → Client:**

```typescript
// Session started
{
  type: "session_started",
  session_id: string,
  config: object
}

// Pose result
{
  type: "pose_result",
  data: {
    success: boolean,
    frame_id: number,
    timestamp: number,
    pose: {
      keypoints: Array<{x, y, z, confidence, id}>,
      confidence_scores: number[],
      processing_time_ms: number
    },
    gait_metrics?: {
      cadence: number,
      step_length: number,
      stride_length: number,
      walking_speed: number,
      symmetry_index: number,
      stability_score: number,
      confidence: number
    }
  }
}

// Statistics update
{
  type: "statistics",
  data: {
    frames_received: number,
    frames_processed: number,
    average_processing_time_ms: number,
    fps: number,
    poses_detected: number
  }
}

// Error
{
  type: "error",
  message: string
}
```

### HTTP Endpoints

#### Get Configuration
```http
GET /api/realtime/config
```

#### Update Configuration
```http
POST /api/realtime/config
Content-Type: application/json

{
  "processing_mode": "balanced",
  "confidence_threshold": 0.5
}
```

#### Get Statistics
```http
GET /api/realtime/stats
```

#### Health Check
```http
GET /api/realtime/health
```

#### Get Processing Modes
```http
GET /api/realtime/processing-modes
```

## Performance Considerations

### Optimal Conditions
- Good lighting (natural or bright artificial light)
- Clear view of full body
- Solid color background (optional but helpful)
- Stable camera position
- Subject 2-4 meters from camera

### Performance Tips
1. **Use Fast mode** for real-time feedback with minimal lag
2. **Use Balanced mode** for general use with good accuracy
3. **Use Accurate mode** for clinical assessments requiring precision
4. **Reduce buffer size** if experiencing memory issues
5. **Lower target FPS** if CPU usage is too high
6. **Enable pose tracking** for smoother visualizations

### System Requirements
- Modern web browser with WebRTC support
- Webcam (720p or higher recommended)
- CPU: Multi-core processor (4+ cores recommended)
- RAM: 4GB minimum, 8GB recommended
- Network: Low-latency connection to server

## Troubleshooting

### Camera Not Working
- Check browser permissions for camera access
- Ensure no other application is using the camera
- Try refreshing the page
- Check browser console for error messages

### Poor Performance
- Switch to Fast processing mode
- Reduce buffer size
- Lower target FPS
- Close other applications
- Check CPU usage

### Inaccurate Results
- Improve lighting conditions
- Ensure full body is visible
- Move to appropriate distance from camera
- Switch to Accurate processing mode
- Increase confidence threshold

### Connection Issues
- Check network connectivity
- Verify server is running
- Check browser console for WebSocket errors
- Try reconnecting

## Clinical Considerations

### Limitations
- Metrics are in relative units and require calibration for clinical use
- Results should be validated against gold-standard measurement systems
- Not intended as a diagnostic tool without proper validation
- Environmental factors can affect accuracy

### Best Practices
- Conduct assessments in controlled environment
- Use consistent lighting and camera setup
- Record multiple trials for reliability
- Document environmental conditions
- Compare results across sessions

### Data Privacy
- Video frames are processed in real-time and not stored by default
- No video recording unless explicitly enabled
- Pose data can be saved for analysis
- Follow institutional privacy policies

## Development

### Running Tests

```bash
# Run all realtime tests
pytest tests/ambient/realtime/ -v

# Run specific test file
pytest tests/ambient/realtime/test_frame_buffer.py -v

# Run with coverage
pytest tests/ambient/realtime/ --cov=ambient.realtime
```

### Adding New Features

1. Implement in `ambient/realtime/` module
2. Add corresponding tests in `tests/ambient/realtime/`
3. Update API endpoints in `server/routers/realtime.py`
4. Update frontend components in `frontend/components/realtime/`
5. Update documentation

## References

- [MediaPipe Pose](https://google.github.io/mediapipe/solutions/pose.html)
- [WebRTC API](https://developer.mozilla.org/en-US/docs/Web/API/WebRTC_API)
- [WebSocket Protocol](https://developer.mozilla.org/en-US/docs/Web/API/WebSocket)
- [Gait Analysis Fundamentals](../analysis/gait-analysis.md)

## See Also

- [Architecture Documentation](architecture.md)
- [API Reference](api-reference.md)
- [Frontend Components](frontend-components.md)
- [Performance Optimization](performance.md)
