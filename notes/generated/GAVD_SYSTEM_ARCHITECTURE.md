# GAVD System Architecture

## Overview

The GAVD (Gait Abnormality Video Dataset) processing system is a full-stack application that processes gait analysis datasets from YouTube videos, extracts pose keypoints, and prepares data for machine learning model training.

## System Components

```
┌─────────────────────────────────────────────────────────────────┐
│                         User Interface                          │
│                    (Next.js Frontend)                           │
│                   http://localhost:3000                         │
└────────────────────────┬────────────────────────────────────────┘
                         │ HTTP/REST API
                         ▼
┌─────────────────────────────────────────────────────────────────┐
│                      Backend API Server                         │
│                    (FastAPI + Uvicorn)                          │
│                   http://localhost:8000                         │
│                                                                 │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐        │
│  │   Routers    │  │   Services   │  │  Middleware  │        │
│  │  (gavd.py)   │  │(gavd_service)│  │   (CORS,     │        │
│  │              │  │              │  │   Logging)   │        │
│  └──────────────┘  └──────────────┘  └──────────────┘        │
└────────────────────────┬────────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────────┐
│                   Processing Pipeline                           │
│                  (ambient.gavd module)                          │
│                                                                 │
│  ┌──────────────────────────────────────────────────────────┐ │
│  │ 1. CSV Loading & Validation (GAVDDataLoader)             │ │
│  │    - Parse CSV with dictionary fields                    │ │
│  │    - Validate required columns                           │ │
│  │    - Organize by sequences                               │ │
│  └──────────────────────────────────────────────────────────┘ │
│                         │                                       │
│                         ▼                                       │
│  ┌──────────────────────────────────────────────────────────┐ │
│  │ 2. YouTube Video Download (youtube_cache)                │ │
│  │    - Extract video IDs from URLs                         │ │
│  │    - Download videos using yt-dlp                        │ │
│  │    - Cache in data/youtube/                              │ │
│  └──────────────────────────────────────────────────────────┘ │
│                         │                                       │
│                         ▼                                       │
│  ┌──────────────────────────────────────────────────────────┐ │
│  │ 3. Frame Extraction (PoseDataConverter)                  │ │
│  │    - Use FFmpeg for precise frame extraction             │ │
│  │    - Extract frames at specified frame numbers           │ │
│  │    - Scale bounding boxes for resolution differences     │ │
│  └──────────────────────────────────────────────────────────┘ │
│                         │                                       │
│                         ▼                                       │
│  ┌──────────────────────────────────────────────────────────┐ │
│  │ 4. Pose Estimation (PoseEstimator)                       │ │
│  │    - MediaPipe or OpenPose                               │ │
│  │    - Extract 25 keypoints per frame                      │ │
│  │    - Apply bounding box constraints                      │ │
│  │    - Fallback to placeholder keypoints if needed         │ │
│  └──────────────────────────────────────────────────────────┘ │
│                         │                                       │
│                         ▼                                       │
│  ┌──────────────────────────────────────────────────────────┐ │
│  │ 5. Results Storage (GAVDService)                         │ │
│  │    - Save metadata to JSON                               │ │
│  │    - Store pose data                                     │ │
│  │    - Generate statistics                                 │ │
│  │    - Update processing status                            │ │
│  └──────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────────┐
│                      Data Storage                               │
│                                                                 │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐        │
│  │   Videos     │  │   Metadata   │  │   Results    │        │
│  │ data/youtube/│  │data/training/│  │data/training/│        │
│  │              │  │gavd/metadata/│  │gavd/results/ │        │
│  └──────────────┘  └──────────────┘  └──────────────┘        │
└─────────────────────────────────────────────────────────────────┘
```

## Data Flow

### 1. Upload Request

```
User → Frontend → Backend → Validation → Storage
```

**Frontend (`frontend/app/training/gavd/page.tsx`):**
- User selects/drops CSV file
- FormData created with file and options
- POST request to `/api/v1/gavd/upload`

**Backend (`server/routers/gavd.py`):**
- Receives UploadFile
- Validates CSV structure
- Saves to `data/training/gavd/{dataset_id}.csv`
- Creates metadata entry
- Returns dataset_id and status

### 2. Background Processing

```
Backend → GAVDService → GAVDProcessor → Results
```

**GAVDService (`server/services/gavd_service.py`):**
- Runs in background task
- Updates status: "processing"
- Calls GAVDProcessor
- Saves results
- Updates status: "completed" or "error"

**GAVDProcessor (`ambient/gavd/gavd_processor.py`):**
- Loads CSV data
- Downloads YouTube videos
- Extracts frames
- Runs pose estimation
- Returns processed sequences

### 3. Status Polling

```
Frontend → Backend → Metadata → Frontend
```

**Frontend:**
- Polls `/api/v1/gavd/status/{dataset_id}` every 2 seconds
- Updates UI with progress
- Stops polling when status is "completed" or "error"

**Backend:**
- Reads metadata JSON
- Returns current status and statistics

### 4. Results Display

```
Frontend → Backend → Results → Analysis Page
```

**Frontend:**
- Navigates to `/training/gavd/{dataset_id}`
- Fetches sequences and frames
- Displays pose visualizations
- Shows statistics and metadata

## Key Classes and Modules

### Frontend

**`frontend/app/training/gavd/page.tsx`**
- Main upload page component
- Drag & drop file upload
- Status polling
- Results display

### Backend

**`server/routers/gavd.py`**
- `/upload` - Upload CSV file
- `/process/{dataset_id}` - Start processing
- `/status/{dataset_id}` - Get status
- `/results/{dataset_id}` - Get results
- `/sequences/{dataset_id}` - Get sequences
- `/list` - List all datasets

**`server/services/gavd_service.py`**
- `save_dataset_metadata()` - Save metadata
- `get_dataset_metadata()` - Retrieve metadata
- `process_dataset()` - Background processing
- `get_dataset_results()` - Get results
- `delete_dataset()` - Delete dataset

**`server/utils/file_validation.py`**
- `validate_csv_file()` - Validate CSV structure
- Check required columns
- Count rows and sequences

### Processing Pipeline

**`ambient/gavd/gavd_processor.py`**

**GAVDDataLoader:**
- `load_gavd_data()` - Load and validate CSV
- `organize_by_sequence()` - Group by sequence ID
- `filter_sequences()` - Filter by criteria
- `get_sequence_statistics()` - Generate stats

**PoseDataConverter:**
- `convert_sequence_to_pose_format()` - Convert to pose format
- `_extract_frame_image()` - Extract frame using FFmpeg
- `_scale_bbox_coordinates()` - Scale bounding boxes
- `_resolve_cached_video_path()` - Find cached video

**GAVDProcessor:**
- `process_gavd_file()` - Main processing pipeline
- `process_sequences_with_filtering()` - Process with filters
- `get_processing_statistics()` - Generate statistics

**`ambient/gavd/pose_estimators.py`**
- `PoseEstimator` - Abstract base class
- `MediaPipeEstimator` - MediaPipe implementation
- `OpenPoseEstimator` - OpenPose implementation
- `get_pose_estimator()` - Factory function

**`ambient/utils/youtube_cache.py`**
- `cache_youtube_videos_from_rows()` - Download videos
- `extract_video_id()` - Extract video ID from URL
- `download_youtube_urls()` - Download using yt-dlp

## Configuration

**`ambient/core/config.py`**
- Storage directories
- Logging configuration
- Processing parameters

**Environment Variables:**
- `ENVIRONMENT` - "development" or "production"
- `HOST` - Server host (default: 127.0.0.1)
- `PORT` - Server port (default: 8000)

## Data Structures

### CSV Row Format

```python
{
    "seq": "cljan9b4p00043n6ligceanyp",
    "frame_num": 1757,
    "cam_view": "front",
    "gait_event": "heel_strike",
    "dataset": "GAVD",
    "gait_pat": "normal",
    "bbox": {
        "left": 156,
        "top": 125,
        "width": 238,
        "height": 500
    },
    "vid_info": {
        "width": 1280,
        "height": 720
    },
    "id": "abc123",
    "url": "https://www.youtube.com/watch?v=B5hrxKe2nP8"
}
```

### Pose Data Format

```python
{
    "frame": 1757,
    "person_id": 0,
    "pose_keypoints_2d": [
        {"x": 100.5, "y": 200.3, "confidence": 0.95, "keypoint_id": 0},
        {"x": 105.2, "y": 210.1, "confidence": 0.92, "keypoint_id": 1},
        # ... 25 keypoints total
    ],
    "gavd_metadata": {
        "seq": "cljan9b4p00043n6ligceanyp",
        "gait_pat": "normal",
        "cam_view": "front",
        "gait_event": "heel_strike",
        "dataset": "GAVD",
        "bbox": {...},
        "vid_info": {...},
        "url": "..."
    }
}
```

### Metadata Format

```python
{
    "dataset_id": "uuid-string",
    "original_filename": "GAVD_Clinical_Annotations_1.csv",
    "file_path": "data/training/gavd/uuid.csv",
    "file_size": 1024000,
    "description": "Parkinsons gait dataset",
    "uploaded_at": "2025-01-03T12:00:00Z",
    "status": "completed",  # uploaded, processing, completed, error
    "row_count": 727,
    "sequence_count": 2,
    "processing_started_at": "2025-01-03T12:00:05Z",
    "processing_completed_at": "2025-01-03T12:05:30Z",
    "total_sequences_processed": 2,
    "total_frames_processed": 727,
    "average_frames_per_sequence": 363.5
}
```

## Error Handling

### Frontend
- Network errors → Show connection error message
- Validation errors → Show specific field errors
- Processing errors → Show error details from backend

### Backend
- File validation errors → HTTP 400 with details
- Processing errors → Update metadata with error
- Missing resources → HTTP 404
- Server errors → HTTP 500 with error message

### Processing Pipeline
- Missing videos → Skip frames for that video
- Pose estimation failures → Fallback to placeholder keypoints
- Invalid data → Log warning and continue
- Critical errors → Update status to "error"

## Performance Considerations

### Caching
- YouTube videos cached in `data/youtube/`
- Video keypoints cached per video during processing
- Metadata cached in JSON files

### Async Processing
- File upload is synchronous
- Processing runs in background task
- Status polling for progress updates

### Optimization
- FFmpeg for efficient frame extraction
- Batch pose estimation when possible
- Lazy loading of video data
- Pagination for large datasets

## Security

### Input Validation
- CSV structure validation
- File size limits
- Required columns check
- URL format validation

### CORS
- Configured for localhost development
- Restricted origins in production

### File Storage
- Unique dataset IDs (UUID)
- Isolated storage directories
- No direct file access from frontend

## Testing

### Unit Tests
- CSV parsing and validation
- Bounding box scaling
- Keypoint generation
- Sequence organization

### Integration Tests
- Upload endpoint
- Processing pipeline
- Status polling
- Results retrieval

### End-to-End Tests
- Complete upload flow
- YouTube download
- Pose estimation
- Results display

## Deployment

### Development
```bash
# Start servers
.\scripts\start-dev.ps1  # Windows
./scripts/start-dev.sh   # Mac/Linux

# Stop servers
.\scripts\stop-dev.ps1   # Windows
./scripts/stop-dev.sh    # Mac/Linux
```

### Production
```bash
# Backend
uv run uvicorn server.main:app --host 0.0.0.0 --port 8000 --workers 4

# Frontend
cd frontend
npm run build
npm start
```

## Monitoring

### Logs
- Backend: `logs/backend.log`
- Frontend: `logs/frontend.log`
- Processing: Structured logs with loguru

### Metrics
- Upload count
- Processing time per sequence
- Success/failure rates
- Video download statistics

### Health Checks
- `/health` endpoint
- Database connectivity
- Storage availability
- External service status

## Future Enhancements

### Short-term
- [ ] WebSocket for real-time progress
- [ ] Batch upload support
- [ ] Resume failed processing
- [ ] Export results to various formats

### Long-term
- [ ] Distributed processing
- [ ] GPU acceleration for pose estimation
- [ ] Advanced gait analysis algorithms
- [ ] Machine learning model training
- [ ] Anomaly detection
- [ ] Comparative analysis tools

## References

- **Notebook**: `notebooks/explore2.ipynb` - Processing examples
- **API Docs**: http://localhost:8000/docs - Interactive API documentation
- **Frontend**: http://localhost:3000 - User interface
- **GAVD Dataset**: Original gait abnormality video dataset

## Support

For issues or questions:
1. Check logs in `logs/` directory
2. Review browser DevTools console
3. Check API documentation at `/docs`
4. Review this architecture document
5. Consult the quick fix guide: `GAVD_QUICK_FIX_GUIDE.md`
