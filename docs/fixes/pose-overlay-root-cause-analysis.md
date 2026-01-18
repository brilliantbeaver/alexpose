# Pose Overlay Root Cause Analysis - PLACEHOLDER KEYPOINTS

## Critical Discovery

The pose overlay is displaying correctly, but the data itself is **placeholder/dummy keypoints**, not real pose estimation results.

## Evidence

### Screenshot Analysis
The screenshot shows a small red/pink square icon on the person's torso. This is exactly what we'd expect from placeholder grid keypoints.

### Pose Data Structure
```json
{
  "1757": {
    "keypoints": [
      {"x": 260.0, "y": 363.5, "confidence": 0.8},
      {"x": 265.0, "y": 363.5, "confidence": 0.8},
      {"x": 270.0, "y": 363.5, "confidence": 0.8},
      {"x": 275.0, "y": 363.5, "confidence": 0.8},
      {"x": 280.0, "y": 363.5, "confidence": 0.8},
      {"x": 260.0, "y": 368.5, "confidence": 0.8},
      ...
    ]
  }
}
```

**Pattern**: 5x5 grid of keypoints with 5-pixel spacing
- Row 1: x=260-280 (step=5), y=363.5
- Row 2: x=260-280 (step=5), y=368.5
- Row 3: x=260-280 (step=5), y=373.5
- Row 4: x=260-280 (step=5), y=378.5
- Row 5: x=260-280 (step=5), y=383.5

This is a **perfect grid**, not human pose keypoints!

## Root Cause

### Code Path Analysis

In `ambient/gavd/gavd_processor.py`, the `PoseDataConverter.convert_sequence_to_pose_data()` method:

```python
if self.estimator is not None and has_urls:
    # Try to run real pose estimation
    try:
        pose_keypoints = self.estimator.estimate_video_keypoints(...)
    except Exception:
        # Fallback to placeholder on any failure
        pose_keypoints = self.keypoint_extractor.extract_from_bbox(
            bbox, num_keypoints, grid_spacing, confidence
        )
else:
    # Fallback to placeholder generator
    pose_keypoints = self.keypoint_extractor.extract_from_bbox(
        bbox, num_keypoints, grid_spacing, confidence
    )
```

The `extract_from_bbox()` method calls `KeypointGenerator.generate_grid_keypoints()`:

```python
def generate_grid_keypoints(
    center_x: float,
    center_y: float,
    num_keypoints: int = 25,
    grid_spacing: float = 5.0,
    confidence: float = 0.8
) -> List[Dict[str, float]]:
    """Generate keypoints in a grid pattern."""
    keypoints = []
    grid_size = int(num_keypoints**0.5)  # 5x5 for 25 keypoints
    
    for i in range(num_keypoints):
        row = i // grid_size
        col = i % grid_size
        x = center_x + (col - grid_size // 2) * grid_spacing
        y = center_y + (row - grid_size // 2) * grid_spacing
        keypoints.append({"x": x, "y": y, "confidence": confidence})
    
    return keypoints
```

### Why Placeholders Were Used

The system fell back to placeholders because:
1. **No pose estimator was configured** (`self.estimator is None`), OR
2. **Pose estimation failed** (exception caught), OR
3. **Video files weren't cached** (YouTube videos not downloaded)

## Impact on Frontend

### What the Frontend Does Correctly ✅
1. Loads the keypoints from the API
2. Adds `keypoint_id` fields (0-24)
3. Detects 25 keypoints → uses OpenPose BODY_25 skeleton connections
4. Scales coordinates correctly
5. Draws the keypoints and connections

### What We See in the Screenshot
- **Small red/pink square**: The 25 grid keypoints clustered in a 20x20 pixel area
- **No skeleton**: The connections are drawn, but they're so small they appear as a single icon
- **Correct position**: The grid is centered on the bounding box center (around x=270, y=373)

## Solution

### Option 1: Run Real Pose Estimation (Recommended)

To get actual skeletal pose data, the dataset needs to be reprocessed with a real pose estimator:

```python
from ambient.gavd import GAVDProcessor, PoseDataConverter
from ambient.pose import get_pose_estimator

# Create pose estimator
estimator = get_pose_estimator("mediapipe")  # or "openpose", "ultralytics"

# Create converter with estimator
converter = PoseDataConverter(estimator=estimator)

# Process dataset
processor = GAVDProcessor(data_converter=converter)
processor.process_gavd_file(
    csv_file="data/GAVD_Clinical_Annotations_1.1.csv",
    output_dir="data/training/gavd/results",
    pose_estimator="mediapipe"
)
```

### Option 2: Accept Placeholder Data (Current State)

The frontend is working correctly. The "overlay" is displaying the placeholder keypoints as designed. This is useful for:
- Testing the visualization pipeline
- Verifying bounding box alignment
- Development without expensive pose estimation

## Frontend Status

### ✅ What's Working
- Keypoint loading from API
- Format detection (25 keypoints → OpenPose BODY_25)
- Coordinate scaling
- Skeleton connection drawing
- Keypoint rendering

### ❌ What's Not Working
- **Nothing is broken in the frontend!**
- The issue is the **data quality**, not the visualization

## Recommendations

### Immediate Actions
1. **Document the placeholder data** in the UI
   - Add a badge/indicator showing "Placeholder Pose Data"
   - Explain that real pose estimation hasn't been run

2. **Add pose estimation status** to dataset metadata
   - Track whether real or placeholder keypoints were used
   - Show this in the frontend

### Long-term Solutions
1. **Run pose estimation** on GAVD datasets during processing
2. **Cache pose results** to avoid reprocessing
3. **Add UI controls** to trigger pose estimation on-demand
4. **Show estimation progress** for long-running pose detection

## Testing Real Pose Data

To verify the frontend works with real pose data, we need to:
1. Download a GAVD video (YouTube)
2. Run MediaPipe/OpenPose on it
3. Generate real keypoint data
4. Test the visualization

Expected real keypoints would look like:
```json
{
  "keypoints": [
    {"x": 245.3, "y": 156.7, "confidence": 0.95, "keypoint_id": 0},  // Nose
    {"x": 248.1, "y": 178.2, "confidence": 0.92, "keypoint_id": 1},  // Neck
    {"x": 235.6, "y": 185.4, "confidence": 0.88, "keypoint_id": 2},  // Right shoulder
    ...
  ]
}
```

Notice:
- **Irregular spacing**: Not a perfect grid
- **Anatomical positions**: Follows human body structure
- **Varying confidence**: Different confidence scores per keypoint
- **Realistic coordinates**: Spread across the person's body, not clustered

## Conclusion

**The frontend pose overlay is working correctly.** The small square icon in the screenshot is the accurate visualization of the placeholder grid keypoints in the database. To see a proper skeletal overlay, the dataset needs to be reprocessed with real pose estimation enabled.
