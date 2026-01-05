# GAVD Dataset Processing - Quick Start Guide

## Prerequisites

1. **Python Environment**
   ```bash
   # Ensure you have Python 3.12+ installed
   python --version
   ```

2. **Install Dependencies**
   ```bash
   # Install required packages
   pip install fastapi uvicorn pandas loguru python-multipart opencv-python
   
   # Optional: Install MediaPipe for pose estimation
   pip install mediapipe
   ```

3. **Start the Backend Server**
   ```bash
   # From project root
   uvicorn server.main:app --reload --host 127.0.0.1 --port 8000
   ```

4. **Start the Frontend**
   ```bash
   # In a new terminal
   cd frontend
   npm install
   npm run dev
   ```

## Step-by-Step Workflow

### Step 1: Upload GAVD Dataset

1. Navigate to **http://localhost:3000/training/gavd**
2. Click "Choose File" and select your GAVD CSV file
3. (Optional) Add a description
4. Check "Process immediately after upload" if you want automatic processing
5. Click "Upload Dataset"

**Expected CSV Format:**
```csv
seq,frame_num,cam_view,gait_event,dataset,gait_pat,bbox,vid_info,id,url
cljan9b4p00043n6ligceanyp,1757,right side,,Abnormal Gait,parkinsons,"{'top': 125.0, 'left': 156.0, 'height': 497.0, 'width': 228.0}","{'height': 720, 'width': 1280, 'mime_type': 'video/mp4'}",B5hrxKe2nP8,https://www.youtube.com/watch?v=B5hrxKe2nP8
```

### Step 2: Monitor Processing

The system will:
1. ✅ Validate CSV structure
2. ✅ Download YouTube videos (cached in `data/youtube/`)
3. ✅ Extract frames at specified frame numbers
4. ✅ (Optional) Run pose estimation
5. ✅ Organize sequences
6. ✅ Save results

**Processing Time:** Depends on:
- Number of sequences
- Number of unique YouTube videos
- Whether pose estimation is enabled
- Network speed for video downloads

### Step 3: Analyze Results

1. Once processing completes, click on the dataset
2. Navigate to **http://localhost:3000/training/gavd/[dataset-id]**

#### Overview Tab
- View dataset statistics
- See total sequences and frames
- Check processing status

#### Sequences Tab
- Select a sequence from dropdown
- Use timeline slider to navigate frames
- View frame metadata (frame number, camera view, gait events)

#### Visualization Tab
- View frame images
- Toggle bounding box overlay
- Toggle pose overlay (when available)
- See bbox coordinates and video info

#### Pose Analysis Tab
- View pose estimation results
- See keypoint confidence scores
- Analyze skeleton structure

### Step 4: Explore Data

**Navigate Frames:**
- Use the timeline slider
- Click frame numbers
- Use keyboard arrows (coming soon)

**View Metadata:**
- Frame number
- Bounding box coordinates
- Video resolution
- Gait events
- Camera view

**Analyze Sequences:**
- Compare different sequences
- Identify gait patterns
- Check temporal consistency

## API Usage Examples

### Upload via API

```bash
curl -X POST http://localhost:8000/api/v1/gavd/upload \
  -F "file=@data/GAVD_Clinical_Annotations_1.csv" \
  -F "description=Parkinsons dataset" \
  -F "process_immediately=true"
```

### Check Status

```bash
curl http://localhost:8000/api/v1/gavd/status/{dataset_id}
```

### List Datasets

```bash
curl http://localhost:8000/api/v1/gavd/list
```

### Get Sequences

```bash
curl http://localhost:8000/api/v1/gavd/sequences/{dataset_id}
```

### Get Frame Data

```bash
curl http://localhost:8000/api/v1/gavd/sequence/{dataset_id}/{sequence_id}/frames
```

## Troubleshooting

### Issue: CSV Upload Fails

**Error:** "Missing required columns"

**Solution:**
- Ensure CSV has columns: `seq`, `frame_num`, `bbox`, `url`
- Check CSV is UTF-8 encoded
- Verify no empty rows

### Issue: Processing Stuck

**Error:** Status shows "processing" for too long

**Solution:**
1. Check server logs: `logs/alexpose_*.log`
2. Verify YouTube videos are accessible
3. Check disk space in `data/youtube/`
4. Restart processing:
   ```bash
   curl -X POST http://localhost:8000/api/v1/gavd/process/{dataset_id}
   ```

### Issue: Frame Not Displaying

**Error:** "Frame not found" or blank image

**Solution:**
1. Verify video downloaded:
   ```bash
   ls data/youtube/
   ```
2. Check frame number is within video range
3. Verify bbox coordinates are valid
4. Check server logs for FFmpeg errors

### Issue: Pose Estimation Fails

**Error:** "No pose detected"

**Solution:**
1. Install MediaPipe:
   ```bash
   pip install mediapipe
   ```
2. Check image quality
3. Verify person is visible in frame
4. Try different confidence threshold

## Directory Structure

After processing, your directory structure will look like:

```
alexpose/
├── data/
│   ├── training/
│   │   └── gavd/
│   │       ├── {dataset-id}.csv          # Original CSV
│   │       ├── metadata/
│   │       │   └── {dataset-id}.json     # Processing metadata
│   │       └── results/
│   │           └── {dataset-id}_results.json  # Processing results
│   └── youtube/
│       ├── {video-id}.mp4                # Cached videos
│       └── ...
├── logs/
│   └── alexpose_*.log                    # Server logs
└── ...
```

## Performance Tips

### 1. **Batch Processing**

Process multiple datasets in sequence:
```bash
for csv in data/*.csv; do
  curl -X POST http://localhost:8000/api/v1/gavd/upload \
    -F "file=@$csv" \
    -F "process_immediately=true"
done
```

### 2. **Video Caching**

Videos are cached by ID, so:
- Multiple sequences from same video = 1 download
- Reprocessing same dataset = no re-download
- Clear cache to free space: `rm -rf data/youtube/*`

### 3. **Limit Sequences**

For testing, process fewer sequences:
```bash
curl -X POST http://localhost:8000/api/v1/gavd/process/{dataset_id} \
  -F "max_sequences=10"
```

### 4. **Disable Pose Estimation**

For faster processing:
```bash
curl -X POST http://localhost:8000/api/v1/gavd/process/{dataset_id} \
  -F "pose_estimator=none"
```

## Next Steps

1. **Explore Sequences:** Navigate through different gait patterns
2. **Compare Results:** Analyze normal vs. abnormal gaits
3. **Export Data:** Use results for training ML models
4. **Batch Analysis:** Process multiple datasets
5. **Custom Analysis:** Use API for programmatic access

## Support

For issues or questions:
1. Check logs in `logs/` directory
2. Review API documentation at http://localhost:8000/docs
3. See full documentation in `GAVD_INTEGRATION_SUMMARY.md`

## Example Dataset

Use the provided sample dataset:
```bash
# Located at: data/GAVD_Clinical_Annotations_1.csv
# Contains: Parkinsons gait sequences
# Sequences: Multiple sequences with frame-level annotations
```

## Advanced Usage

### Custom Pose Estimator

```python
from ambient.gavd.pose_estimators import PoseEstimator

class CustomEstimator(PoseEstimator):
    def estimate_image_keypoints(self, image_path, model, bbox):
        # Your implementation
        pass
```

### Programmatic Access

```python
from ambient.gavd.gavd_processor import create_gavd_processor

processor = create_gavd_processor()
results = processor.process_gavd_file(
    csv_file_path="data/GAVD_Clinical_Annotations_1.csv",
    max_sequences=10,
    include_metadata=True
)
```

### Batch Frame Extraction

```python
from server.services.gavd_service import GAVDService
from ambient.core.config import ConfigurationManager

config = ConfigurationManager()
service = GAVDService(config)

# Get all frames for a sequence
frames = service.get_sequence_frames(dataset_id, sequence_id)

# Extract images for all frames
for frame in frames:
    image = service.get_frame_image(
        dataset_id,
        sequence_id,
        frame['frame_num'],
        show_bbox=True
    )
```

---

**Happy Analyzing! 🎯**
