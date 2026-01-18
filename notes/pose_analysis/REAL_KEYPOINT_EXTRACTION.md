# Real Keypoint Extraction Implementation

## Overview

The GAVD processor has been upgraded to use **real pose estimation** with MediaPipe instead of generating placeholder grid keypoints. This provides accurate human pose detection from actual video frames.

## Changes Made

### 1. Enhanced `PoseKeypointExtractor` Class

**Location:** `ambient/gavd/gavd_processor.py`

#### Previous Behavior
- Generated synthetic grid keypoints around bounding box center
- No actual pose detection
- Placeholder data for testing only

#### New Behavior
- Uses `SequenceKeypointExtractor` with MediaPipe for real pose detection
- Extracts keypoints from actual image regions defined by bounding boxes
- Falls back to grid keypoints only when real extraction fails or is unavailable

#### Key Methods

**`extract_from_image_and_bbox(image, bbox, model_path=None)`**
- Extracts real pose keypoints from an image region
- Crops image to bounding box
- Runs MediaPipe pose estimation on cropped region
- Transforms coordinates back to original image space
- Returns list of keypoint dictionaries with x, y, confidence

**`extract_from_bbox(bbox, num_keypoints, grid_spacing, confidence)`**
- Backward-compatible fallback method
- Used when only bbox data is available without image
- Generates grid keypoints as before

**`_extract_fallback_grid(bbox, ...)`**
- Internal method for grid keypoint generation
- Used as last resort when real extraction fails

### 2. Enhanced `PoseDataConverter` Class

**Location:** `ambient/gavd/gavd_processor.py`

#### New Method: `_extract_keypoints_fallback()`

This method provides intelligent fallback behavior:

1. **Extract frame from video** using ffmpeg
2. **Load and convert image** to RGB format
3. **Run real pose estimation** using `extract_from_image_and_bbox()`
4. **Clean up temporary files**
5. **Fall back to grid keypoints** only if all above steps fail

#### Updated Conversion Logic

The `convert_sequence_to_pose_format()` method now:

1. **Primary path:** Uses estimator if available (unchanged)
2. **Enhanced fallback:** Tries real extraction with video frames
3. **Final fallback:** Uses grid keypoints only as last resort

## Architecture

```
Video Frame → Extract Frame Image → Crop to BBox → MediaPipe Pose Estimation → Real Keypoints
                                                                                      ↓
                                                                                   (if fails)
                                                                                      ↓
                                                                              Grid Keypoints
```

## Benefits

### 1. Accurate Pose Data
- Real human pose keypoints instead of synthetic placeholders
- Proper joint locations (shoulders, elbows, knees, etc.)
- Confidence scores reflect actual detection quality

### 2. Better Analysis
- Gait analysis features computed from real pose data
- More accurate joint angles and movements
- Reliable symmetry measurements

### 3. Graceful Degradation
- Falls back to grid keypoints if MediaPipe unavailable
- Handles missing videos or extraction failures
- Maintains backward compatibility

### 4. Efficient Processing
- Lazy initialization of SequenceKeypointExtractor
- Reuses extractor instance across frames
- Cleans up temporary files automatically

## Usage Example

```python
from ambient.gavd.gavd_processor import PoseKeypointExtractor
import cv2

# Initialize extractor
extractor = PoseKeypointExtractor()

# Load image
image = cv2.imread("person.jpg")
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# Define bounding box
bbox = {
    "left": 100,
    "top": 50,
    "width": 200,
    "height": 400
}

# Extract real keypoints
keypoints = extractor.extract_from_image_and_bbox(image_rgb, bbox)

# Process keypoints
for kp in keypoints:
    print(f"Point at ({kp['x']:.1f}, {kp['y']:.1f}) "
          f"with confidence {kp['confidence']:.3f}")
```

## Integration with GAVD Pipeline

The changes are transparent to existing code:

```python
from ambient.gavd.gavd_processor import GAVDProcessor

# Create processor
processor = GAVDProcessor()

# Process GAVD file (now uses real keypoints automatically)
result = processor.process_gavd_file(
    "data/gavd/annotations.csv",
    max_sequences=10
)

# Keypoints in result are now real pose detections
for seq_id, seq_data in result['sequences'].items():
    pose_data = seq_data['pose_data']
    # pose_data contains real keypoints from MediaPipe
```

## Dependencies

- **MediaPipe**: For pose estimation
- **OpenCV (cv2)**: For image loading and processing
- **NumPy**: For array operations
- **FFmpeg**: For video frame extraction

## Testing

Run the example script to verify the implementation:

```bash
python examples/test_real_keypoint_extraction.py
```

Run existing tests to ensure backward compatibility:

```bash
pytest tests/ambient/gavd/ -v
```

## Performance Considerations

### Speed
- Real extraction is slower than grid generation
- Typical: ~50-100ms per frame with MediaPipe
- Grid fallback: <1ms per frame

### Memory
- Temporary frame images are cleaned up automatically
- Extractor instance is reused across frames
- No significant memory overhead

### Optimization Tips
1. Use estimator with video batch processing when available
2. Cache extracted frames if processing multiple times
3. Adjust MediaPipe confidence thresholds for speed/accuracy tradeoff

## Future Enhancements

1. **Batch Processing**: Process multiple frames simultaneously
2. **GPU Acceleration**: Use MediaPipe GPU delegate
3. **Model Selection**: Support different MediaPipe models (lite, full, heavy)
4. **Caching**: Cache keypoints to avoid re-extraction
5. **Parallel Processing**: Multi-threaded frame extraction

## Troubleshooting

### MediaPipe Not Available
- System falls back to grid keypoints automatically
- Install MediaPipe: `uv pip install mediapipe`

### Model Download Issues
- MediaPipe model downloads automatically on first use
- Check internet connection
- Verify `data/models/` directory is writable

### Low Confidence Keypoints
- Adjust confidence thresholds in SequenceKeypointExtractor
- Ensure good lighting and clear person visibility in videos
- Try different MediaPipe models

### Frame Extraction Failures
- Verify FFmpeg is installed and in PATH
- Check video file is not corrupted
- Ensure sufficient disk space for temporary files

## Related Files

- `ambient/gavd/gavd_processor.py` - Main implementation
- `ambient/pose/keypoint_extractor.py` - SequenceKeypointExtractor
- `ambient/pose/keypoints.py` - Keypoint utilities
- `ambient/pose/keypoint_data.py` - Data structures
- `examples/test_real_keypoint_extraction.py` - Demo script
- `tests/ambient/gavd/test_pose_estimators.py` - Tests
