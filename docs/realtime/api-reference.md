# Realtime API Reference

Complete API reference for the realtime gait analysis system.

## Python API

### RealtimePoseEstimator

```python
from ambient.realtime.pose_estimator import RealtimePoseEstimator
from ambient.realtime.interfaces import ProcessingMode
```

#### Constructor

```python
RealtimePoseEstimator(
    model_path: Optional[str] = None,
    processing_mode: ProcessingMode = ProcessingMode.BALANCED,
    target_fps: int = 30,
    max_processing_time_ms: float = 33.0
)
```

**Parameters:**
- `model_path`: Path to MediaPipe pose model file (optional)
- `processing_mode`: Processing mode (FAST, BALANCED, or ACCURATE)
- `target_fps`: Target frame processing rate
- `max_processing_time_ms`: Maximum processing time per frame

#### Methods

**estimate_pose(frame: RealtimeFrame) → RealtimePoseResult**

Estimate pose from a single frame.

```python
result = estimator.estimate_pose(frame)
print(f"Keypoints: {len(result.keypoints)}")
print(f"Processing time: {result.processing_time_ms}ms")
```

**set_processing_mode(mode: ProcessingMode) → None**

Change processing mode.

```python
estimator.set_processing_mode(ProcessingMode.FAST)
```

**get_performance_stats() → Dict[str, Any]**

Get performance statistics.

```python
stats = estimator.get_performance_stats()
print(f"FPS: {stats['fps']}")
print(f"Avg processing time: {stats['average_processing_time_ms']}ms")
```

**is_ready() → bool**

Check if estimator is ready for processing.

```python
if estimator.is_ready():
    result = estimator.estimate_pose(frame)
```

### RealtimeGaitAnalyzer

```python
from ambient.realtime.gait_analyzer import RealtimeGaitAnalyzer
```

#### Constructor

```python
RealtimeGaitAnalyzer(
    window_size: int = 60,
    min_poses_for_analysis: int = 10,
    confidence_threshold: float = 0.5
)
```

**Parameters:**
- `window_size`: Number of poses to keep in sliding window
- `min_poses_for_analysis`: Minimum poses needed for analysis
- `confidence_threshold`: Minimum confidence for pose keypoints

#### Methods

**analyze_pose_sequence(poses: List[RealtimePoseResult]) → RealtimeGaitMetrics**

Analyze a sequence of poses.

```python
metrics = analyzer.analyze_pose_sequence(poses)
print(f"Cadence: {metrics.cadence} steps/min")
print(f"Symmetry: {metrics.symmetry_index}")
```

**update_with_pose(pose: RealtimePoseResult) → Optional[RealtimeGaitMetrics]**

Update analysis with new pose.

```python
metrics = analyzer.update_with_pose(pose)
if metrics:
    print(f"Walking speed: {metrics.walking_speed}")
```

**reset_analysis() → None**

Reset analysis state.

```python
analyzer.reset_analysis()
```

**get_required_pose_count() → int**

Get minimum number of poses needed.

```python
min_poses = analyzer.get_required_pose_count()
```

### FrameBuffer

```python
from ambient.realtime.frame_buffer import FrameBuffer
```

#### Constructor

```python
FrameBuffer(
    max_size: int = 30,
    max_memory_mb: int = 100,
    auto_cleanup: bool = True
)
```

**Parameters:**
- `max_size`: Maximum number of frames to store
- `max_memory_mb`: Maximum memory usage in MB
- `auto_cleanup`: Whether to automatically cleanup old frames

#### Methods

**add_frame(frame: RealtimeFrame) → None**

Add a frame to the buffer.

```python
buffer.add_frame(frame)
```

**get_latest_frame() → Optional[RealtimeFrame]**

Get the most recent frame.

```python
latest = buffer.get_latest_frame()
```

**get_frame_sequence(count: int) → List[RealtimeFrame]**

Get a sequence of recent frames.

```python
recent_frames = buffer.get_frame_sequence(10)
```

**clear() → None**

Clear all frames from buffer.

```python
buffer.clear()
```

**get_buffer_stats() → Dict[str, Any]**

Get buffer statistics.

```python
stats = buffer.get_buffer_stats()
print(f"Current size: {stats['current_size']}")
print(f"Memory usage: {stats['current_memory_mb']}MB")
```

### StreamProcessor

```python
from ambient.realtime.stream_processor import StreamProcessor
```

#### Constructor

```python
StreamProcessor(
    model_path: Optional[str] = None,
    processing_mode: ProcessingMode = ProcessingMode.BALANCED,
    buffer_size: int = 30,
    enable_tracking: bool = True
)
```

**Parameters:**
- `model_path`: Path to pose estimation model
- `processing_mode`: Processing mode for performance/accuracy tradeoff
- `buffer_size`: Size of frame buffer
- `enable_tracking`: Whether to enable pose tracking

#### Methods

**async process_frame(frame_data: bytes) → Dict[str, Any]**

Process a single frame from the stream.

```python
result = await processor.process_frame(frame_data)
if result['success']:
    pose = result['pose']
    gait_metrics = result.get('gait_metrics')
```

**set_processing_parameters(params: Dict[str, Any]) → None**

Set processing parameters.

```python
processor.set_processing_parameters({
    'processing_mode': 'fast',
    'confidence_threshold': 0.6
})
```

**get_processing_stats() → Dict[str, Any]**

Get processing statistics.

```python
stats = processor.get_processing_stats()
```

**start_processing() → None**

Start the processing pipeline.

```python
processor.start_processing()
```

**stop_processing() → None**

Stop the processing pipeline.

```python
processor.stop_processing()
```

### PoseTracker

```python
from ambient.realtime.pose_tracker import PoseTracker
```

#### Constructor

```python
PoseTracker(
    smoothing_factor: float = 0.7,
    max_tracking_distance: float = 50.0,
    confidence_threshold: float = 0.3,
    max_missing_frames: int = 5
)
```

**Parameters:**
- `smoothing_factor`: Factor for temporal smoothing (0-1)
- `max_tracking_distance`: Maximum distance for keypoint tracking
- `confidence_threshold`: Minimum confidence for tracking
- `max_missing_frames`: Maximum frames to track without detection

#### Methods

**track_pose(current_pose: RealtimePoseResult, previous_poses: List[RealtimePoseResult]) → RealtimePoseResult**

Track pose across frames.

```python
tracked_pose = tracker.track_pose(current_pose, previous_poses)
```

**get_tracking_confidence() → float**

Get current tracking confidence.

```python
confidence = tracker.get_tracking_confidence()
```

**reset_tracking() → None**

Reset tracking state.

```python
tracker.reset_tracking()
```

## Data Types

### RealtimeFrame

```python
@dataclass
class RealtimeFrame:
    data: np.ndarray          # Frame image data
    timestamp: float          # Frame timestamp
    frame_id: int            # Unique frame identifier
    metadata: Dict[str, Any] # Additional metadata
```

### RealtimePoseResult

```python
@dataclass
class RealtimePoseResult:
    keypoints: List[Dict[str, Any]]  # Detected keypoints
    confidence_scores: List[float]   # Confidence for each keypoint
    processing_time_ms: float        # Processing time
    frame_id: int                    # Frame identifier
    timestamp: float                 # Result timestamp
    estimator_info: Dict[str, Any]   # Estimator metadata
```

### RealtimeGaitMetrics

```python
@dataclass
class RealtimeGaitMetrics:
    cadence: Optional[float]          # Steps per minute
    step_length: Optional[float]      # Step length
    stride_length: Optional[float]    # Stride length
    walking_speed: Optional[float]    # Walking speed
    symmetry_index: Optional[float]   # Left-right symmetry (0-1)
    stability_score: Optional[float]  # Stability score (0-1)
    confidence: float                 # Overall confidence
    timestamp: float                  # Metrics timestamp
```

### ProcessingMode

```python
class ProcessingMode(Enum):
    FAST = "fast"           # Optimized for speed
    BALANCED = "balanced"   # Balance speed/accuracy
    ACCURATE = "accurate"   # Optimized for accuracy
```

## REST API

### Base URL

```
http://localhost:8000/api/realtime
```

### Endpoints

#### GET /config

Get current configuration.

**Response:**
```json
{
  "success": true,
  "config": {
    "processing_mode": "balanced",
    "buffer_size": 30,
    "enable_tracking": true,
    "confidence_threshold": 0.5,
    "target_fps": 30
  }
}
```

#### POST /config

Update configuration.

**Request:**
```json
{
  "processing_mode": "fast",
  "confidence_threshold": 0.6
}
```

**Response:**
```json
{
  "success": true,
  "config": {
    "processing_mode": "fast",
    "buffer_size": 30,
    "enable_tracking": true,
    "confidence_threshold": 0.6,
    "target_fps": 30
  }
}
```

#### GET /stats

Get processing statistics.

**Response:**
```json
{
  "success": true,
  "statistics": {
    "frames_received": 1234,
    "frames_processed": 1200,
    "frames_failed": 5,
    "average_processing_time_ms": 25.3,
    "poses_detected": 1150,
    "gait_analyses_completed": 45,
    "session_duration_seconds": 120.5,
    "fps": 28.5
  }
}
```

#### GET /health

Health check endpoint.

**Response:**
```json
{
  "success": true,
  "health": {
    "service": "realtime",
    "status": "healthy",
    "ready": true,
    "timestamp": 1706198400.0
  }
}
```

#### GET /processing-modes

Get available processing modes.

**Response:**
```json
{
  "success": true,
  "modes": [
    {
      "value": "fast",
      "label": "Fast",
      "description": "Optimized for speed, lower accuracy",
      "target_fps": 30,
      "cpu_usage": "Low"
    },
    {
      "value": "balanced",
      "label": "Balanced",
      "description": "Balance between speed and accuracy",
      "target_fps": 25,
      "cpu_usage": "Medium"
    },
    {
      "value": "accurate",
      "label": "Accurate",
      "description": "Highest accuracy, slower processing",
      "target_fps": 20,
      "cpu_usage": "High"
    }
  ]
}
```

#### GET /model-info

Get pose estimation model information.

**Response:**
```json
{
  "success": true,
  "model": {
    "name": "MediaPipe Pose",
    "version": "0.10.9",
    "keypoints": 33,
    "input_size": [256, 256],
    "model_path": "/path/to/model.task"
  }
}
```

## WebSocket API

### Connection

```javascript
const ws = new WebSocket('ws://localhost:8000/api/realtime/stream');
```

### Message Types

#### Client Messages

**Send Frame:**
```json
{
  "type": "frame",
  "data": "base64_encoded_image_data"
}
```

**Update Config:**
```json
{
  "type": "config_update",
  "config": {
    "processing_mode": "fast",
    "confidence_threshold": 0.6
  }
}
```

**Request Stats:**
```json
{
  "type": "get_stats"
}
```

**Ping:**
```json
{
  "type": "ping",
  "timestamp": 1706198400000
}
```

#### Server Messages

**Session Started:**
```json
{
  "type": "session_started",
  "session_id": "uuid-string",
  "config": { ... }
}
```

**Pose Result:**
```json
{
  "type": "pose_result",
  "data": {
    "success": true,
    "frame_id": 123,
    "timestamp": 1706198400.0,
    "pose": {
      "keypoints": [...],
      "confidence_scores": [...],
      "processing_time_ms": 25.3
    },
    "gait_metrics": {
      "cadence": 110.5,
      "walking_speed": 1.2,
      "symmetry_index": 0.85,
      "stability_score": 0.92,
      "confidence": 0.87
    },
    "processing_time_ms": 28.1
  }
}
```

**Statistics:**
```json
{
  "type": "statistics",
  "data": {
    "frames_received": 1234,
    "frames_processed": 1200,
    "average_processing_time_ms": 25.3,
    "fps": 28.5
  }
}
```

**Config Updated:**
```json
{
  "type": "config_updated",
  "config": { ... }
}
```

**Error:**
```json
{
  "type": "error",
  "message": "Error description"
}
```

**Keepalive:**
```json
{
  "type": "keepalive"
}
```

## Error Codes

| Code | Description |
|------|-------------|
| 400 | Bad Request - Invalid parameters |
| 403 | Forbidden - Insufficient permissions |
| 404 | Not Found - Resource not found |
| 500 | Internal Server Error - Server error |
| 503 | Service Unavailable - Service not ready |

## Rate Limiting

- WebSocket: No explicit rate limiting, but frame processing is throttled by target FPS
- REST API: 100 requests per minute per IP

## Examples

### Python Client

```python
import asyncio
import websockets
import json
import base64
import cv2

async def realtime_analysis():
    uri = "ws://localhost:8000/api/realtime/stream"
    
    async with websockets.connect(uri) as websocket:
        # Wait for session start
        message = await websocket.recv()
        data = json.loads(message)
        print(f"Session started: {data['session_id']}")
        
        # Open webcam
        cap = cv2.VideoCapture(0)
        
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Encode frame
                _, buffer = cv2.imencode('.jpg', frame)
                frame_data = base64.b64encode(buffer).decode('utf-8')
                
                # Send frame
                await websocket.send(json.dumps({
                    "type": "frame",
                    "data": frame_data
                }))
                
                # Receive result
                message = await websocket.recv()
                result = json.loads(message)
                
                if result['type'] == 'pose_result':
                    pose = result['data']['pose']
                    print(f"Keypoints: {len(pose['keypoints'])}")
                    
                    if 'gait_metrics' in result['data']:
                        metrics = result['data']['gait_metrics']
                        print(f"Cadence: {metrics['cadence']}")
                
                await asyncio.sleep(0.033)  # ~30 FPS
                
        finally:
            cap.release()

asyncio.run(realtime_analysis())
```

### JavaScript Client

```javascript
const ws = new WebSocket('ws://localhost:8000/api/realtime/stream');

ws.onopen = () => {
    console.log('Connected to realtime service');
};

ws.onmessage = (event) => {
    const message = JSON.parse(event.data);
    
    switch (message.type) {
        case 'session_started':
            console.log('Session ID:', message.session_id);
            startCamera();
            break;
            
        case 'pose_result':
            if (message.data.success) {
                updatePoseOverlay(message.data.pose);
                if (message.data.gait_metrics) {
                    updateMetrics(message.data.gait_metrics);
                }
            }
            break;
            
        case 'statistics':
            updateStats(message.data);
            break;
            
        case 'error':
            console.error('Error:', message.message);
            break;
    }
};

function sendFrame(imageData) {
    ws.send(JSON.stringify({
        type: 'frame',
        data: imageData.split(',')[1]  // Remove data URL prefix
    }));
}
```

## See Also

- [Realtime Overview](README.md)
- [Architecture](architecture.md)
- [Frontend Components](frontend-components.md)
- [Performance Optimization](performance.md)
