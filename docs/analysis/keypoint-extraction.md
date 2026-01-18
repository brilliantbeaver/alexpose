# Keypoint Extraction Module

## Overview

The `ambient.pose.keypoints` module provides comprehensive utilities for pose keypoint extraction, processing, and analysis. It has been refactored to follow SOLID principles with clear separation of concerns and extensibility.

## Architecture

The module is organized into several specialized classes, each with a single responsibility:

```
ambient.pose.keypoints
├── BoundingBoxProcessor      # Bounding box operations
├── KeypointGenerator          # Synthetic keypoint generation
├── PoseKeypointExtractor      # Bbox-based extraction
├── MediaPipeModelManager      # Model download/management
├── PoseLandmarkerFactory      # MediaPipe landmarker creation
└── KeypointVisualizer         # Visualization utilities

ambient.pose.keypoint_extractor
└── SequenceKeypointExtractor  # Video sequence processing
```

## Core Classes

### BoundingBoxProcessor

Handles bounding box calculations and center point extraction.

```python
from ambient.pose.keypoints import BoundingBoxProcessor

processor = BoundingBoxProcessor()
bbox = {"left": 10, "top": 20, "width": 100, "height": 200}
center_x, center_y = processor.calculate_center(bbox)
```

**Key Methods:**
- `calculate_center(bbox)` - Calculate center point of a bounding box

### KeypointGenerator

Generates synthetic pose keypoints using various strategies.

```python
from ambient.pose.keypoints import KeypointGenerator

generator = KeypointGenerator()

# Create single keypoint
kp = generator.create_keypoint(x=100, y=200, confidence=0.9)

# Generate grid of keypoints
keypoints = generator.generate_grid_keypoints(
    center_x=320,
    center_y=240,
    num_keypoints=25,
    grid_spacing=5.0,
    confidence=0.8
)
```

**Key Methods:**
- `create_keypoint(x, y, confidence)` - Create a single keypoint
- `generate_grid_keypoints(...)` - Generate keypoints in a grid pattern

### MediaPipeModelManager

Manages MediaPipe model downloads and caching.

```python
from ambient.pose.keypoints import MediaPipeModelManager
from pathlib import Path

# Initialize with custom directory
manager = MediaPipeModelManager(Path("data/models"))

# Ensure model is available (downloads if needed)
model_path = manager.ensure_model_available()

# Check if model exists
if manager.is_model_downloaded("pose_landmarker_full.task"):
    print("Model ready!")

# Force re-download
manager.download_model(force=True)
```

**Key Methods:**
- `ensure_model_available(model_name, model_url)` - Ensure model is available
- `is_model_downloaded(model_name)` - Check if model exists
- `download_model(model_url, model_name, force)` - Download a model
- `get_model_path(model_name)` - Get path to model file

### PoseLandmarkerFactory

Factory for creating MediaPipe Pose Landmarker instances.

```python
from ambient.pose.keypoints import PoseLandmarkerFactory

factory = PoseLandmarkerFactory()

# Create landmarker with default settings
landmarker = factory.create_landmarker("data/models/pose_landmarker_full.task")

# Create with custom configuration
landmarker = factory.create_landmarker(
    model_path="data/models/pose_landmarker_full.task",
    num_poses=2,
    min_pose_detection_confidence=0.7,
    min_pose_presence_confidence=0.7,
    min_tracking_confidence=0.7
)
```

**Key Methods:**
- `create_landmarker(model_path, **config)` - Create configured landmarker

### SequenceKeypointExtractor

Extracts pose keypoints from video sequences. This class has been moved to its own module for better organization.

```python
from ambient.pose.keypoint_extractor import SequenceKeypointExtractor
from pathlib import Path
import pandas as pd

extractor = SequenceKeypointExtractor()

# Extract from single image
image = cv2.imread("frame.jpg")
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
keypoints = extractor.extract_from_image(image_rgb)

# Extract from video frame
keypoints = extractor.extract_from_video_frame(
    video_path=Path("video.mp4"),
    frame_number=100
)

# Extract from sequence (DataFrame with frame info)
keypoints_array = extractor.extract_from_sequence(
    sequence_data=sequence_df,
    video_base_path=Path("data/youtube"),
    verbose=True
)
```

**Key Methods:**
- `extract_from_image(image, model_path)` - Extract from RGB image array
- `extract_from_frame_file(image_path, model_path)` - Extract from image file
- `extract_from_video_frame(video_path, frame_number, model_path)` - Extract from video frame
- `extract_from_sequence(sequence_data, video_base_path, model_path, verbose)` - Extract from sequence

**Keypoint Format:**

Each keypoint is a dictionary with the following structure:

```python
{
    'id': 0,                    # Landmark ID (0-32 for BLAZEPOSE_33)
    'name': 'NOSE',             # Landmark name
    'x': 320.5,                 # Pixel x-coordinate
    'y': 240.3,                 # Pixel y-coordinate
    'z': -0.15,                 # Depth (relative)
    'visibility': 0.95,         # Visibility score (0-1)
    'presence': 0.98,           # Presence score (0-1)
    'confidence': 0.95,         # Overall confidence (0-1)
    'x_normalized': 0.5,        # Normalized x (0-1)
    'y_normalized': 0.5         # Normalized y (0-1)
}
```

### KeypointVisualizer

Visualizes pose keypoints on images.

```python
from ambient.pose.keypoints import KeypointVisualizer
import cv2

visualizer = KeypointVisualizer()

# Add landmark names to keypoints
keypoints = visualizer.add_landmark_names(keypoints)

# Draw keypoints on image
annotated = visualizer.draw_keypoints(
    image=image_rgb,
    keypoints=keypoints,
    confidence_threshold=0.5,
    color=(255, 0, 0),
    radius=5
)

# Get summary statistics
stats = visualizer.get_summary_stats(keypoints)
print(f"Total landmarks: {stats['total_landmarks']}")
print(f"Visible landmarks: {stats['visible_landmarks']}")
print(f"Average confidence: {stats['avg_confidence']:.3f}")
```

**Key Methods:**
- `add_landmark_names(keypoints)` - Add landmark names to keypoints
- `draw_keypoints(image, keypoints, ...)` - Draw keypoints on image
- `get_summary_stats(keypoints)` - Get summary statistics

## Convenience Functions

For backward compatibility and ease of use, the module provides convenience functions:

```python
from ambient.pose.keypoints import (
    ensure_model_downloaded,
    get_keypoints,
    create_pose_landmarker
)
from pathlib import Path

# Ensure model is downloaded
project_root = Path.cwd()
model_path = ensure_model_downloaded(project_root)

# Extract keypoints from sequence (simplified interface)
# Note: sequence_data must be a pandas DataFrame
keypoints_array = get_keypoints(
    project_root=project_root,
    sequence_data=sequence_df,  # Must be DataFrame, not dict
    verbose=True
)

# Create landmarker (simplified interface)
landmarker = create_pose_landmarker(model_path)
```

## Landmark Names

The module defines 33 MediaPipe pose landmarks (BLAZEPOSE_33 format):

```python
from ambient.pose.keypoints import POSE_LANDMARK_NAMES

# Access landmark names
print(POSE_LANDMARK_NAMES[0])   # 'NOSE'
print(POSE_LANDMARK_NAMES[11])  # 'LEFT_SHOULDER'
print(POSE_LANDMARK_NAMES[25])  # 'LEFT_KNEE'
```

**Complete Landmark List:**
- Face: NOSE, eyes (inner/outer), ears, mouth
- Upper body: shoulders, elbows, wrists, hands (pinky, index, thumb)
- Lower body: hips, knees, ankles, heels, foot indices

## Usage Examples

### Example 1: Extract Keypoints from Single Image

```python
from ambient.pose.keypoint_extractor import SequenceKeypointExtractor
import cv2

# Initialize extractor
extractor = SequenceKeypointExtractor()

# Load and process image
image = cv2.imread("person.jpg")
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# Extract keypoints
keypoints = extractor.extract_from_image(image_rgb)

# Print results
print(f"Detected {len(keypoints)} landmarks")
for kp in keypoints[:5]:  # Show first 5
    print(f"{kp['name']}: ({kp['x']:.1f}, {kp['y']:.1f}) conf={kp['confidence']:.3f}")
```

### Example 2: Process Video Sequence

```python
from ambient.pose.keypoint_extractor import SequenceKeypointExtractor
from pathlib import Path
import pandas as pd

# Load sequence data
sequence_df = pd.read_csv("sequence_info.csv")

# Initialize extractor
extractor = SequenceKeypointExtractor()

# Extract keypoints from all frames
keypoints_array = extractor.extract_from_sequence(
    sequence_data=sequence_df,
    video_base_path=Path("data/videos"),
    verbose=True
)

print(f"Processed {len(keypoints_array)} frames")
```

### Example 3: Visualize Results

```python
from ambient.pose.keypoint_extractor import SequenceKeypointExtractor
from ambient.pose.keypoints import KeypointVisualizer
import cv2
import matplotlib.pyplot as plt

# Extract keypoints
extractor = SequenceKeypointExtractor()
image = cv2.imread("person.jpg")
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
keypoints = extractor.extract_from_image(image_rgb)

# Visualize
visualizer = KeypointVisualizer()
annotated = visualizer.draw_keypoints(image_rgb, keypoints)

# Display
plt.figure(figsize=(10, 8))
plt.imshow(annotated)
plt.title("Pose Detection Results")
plt.axis('off')
plt.show()

# Print statistics
stats = visualizer.get_summary_stats(keypoints)
print(f"Visible landmarks: {stats['visible_landmarks']}/{stats['total_landmarks']}")
print(f"Average confidence: {stats['avg_confidence']:.3f}")
```

### Example 4: Custom Model Configuration

```python
from ambient.pose.keypoints import (
    MediaPipeModelManager,
    PoseLandmarkerFactory,
)
from ambient.pose.keypoint_extractor import SequenceKeypointExtractor
from pathlib import Path

# Setup custom model manager
model_manager = MediaPipeModelManager(Path("custom/models"))
model_path = model_manager.ensure_model_available()

# Create custom landmarker
factory = PoseLandmarkerFactory()
landmarker = factory.create_landmarker(
    model_path=model_path,
    num_poses=2,  # Detect up to 2 people
    min_pose_detection_confidence=0.7
)

# Use with extractor
extractor = SequenceKeypointExtractor(
    model_manager=model_manager,
    landmarker_factory=factory
)
```

## Design Principles

The refactored module follows SOLID principles:

1. **Single Responsibility Principle (SRP)**
   - Each class has one clear responsibility
   - BoundingBoxProcessor: bbox operations
   - MediaPipeModelManager: model management
   - SequenceKeypointExtractor: keypoint extraction

2. **Open/Closed Principle (OCP)**
   - Classes are open for extension but closed for modification
   - KeypointGenerator supports different generation strategies
   - Factory pattern allows easy landmarker configuration

3. **Liskov Substitution Principle (LSP)**
   - Components can be substituted with custom implementations
   - Dependency injection in SequenceKeypointExtractor

4. **Interface Segregation Principle (ISP)**
   - Focused interfaces for each component
   - Clients only depend on methods they use

5. **Dependency Inversion Principle (DIP)**
   - High-level modules depend on abstractions
   - PoseKeypointExtractor depends on processor/generator interfaces

## Integration with Other Modules

The keypoints module integrates seamlessly with other AlexPose components:

```python
# Integration with joint angle calculation
from ambient.pose.keypoints import get_keypoints
from ambient.pose.joint_angles import get_joint_angles

# Extract keypoints (sequence_df must be DataFrame)
keypoints_array = get_keypoints(project_root, sequence_df)

# Calculate joint angles
joint_angles = get_joint_angles(
    keypoints_array=keypoints_array,
    keypoint_format="BLAZEPOSE_33",
    fps=30.0
)

# Integration with gait analysis
from ambient.analysis.gait_analyzer import EnhancedGaitAnalyzer

analyzer = EnhancedGaitAnalyzer(keypoint_format="BLAZEPOSE_33")
results = analyzer.analyze_gait_sequence(keypoints_array)
```

## Testing

Comprehensive tests are available in `tests/pose/test_keypoints.py`:

```bash
# Run all keypoint tests
pytest tests/pose/test_keypoints.py -v

# Run with coverage
pytest tests/pose/test_keypoints.py --cov=ambient.pose.keypoints

# Run property-based tests
pytest tests/pose/test_keypoints.py -k "Hypothesis" -v
```

## Migration Guide

If you're migrating from the old notebook-based implementation:

**Old Code:**
```python
from notebooks.utils.eval_keypoints import get_keypoints, ensure_model_downloaded

model_path = ensure_model_downloaded(project_root)
keypoints = get_keypoints(project_root, sequence_df)
```

**New Code:**
```python
from ambient.pose.keypoints import get_keypoints, ensure_model_downloaded

# Same interface - no changes needed!
model_path = ensure_model_downloaded(project_root)
keypoints = get_keypoints(project_root, sequence_df)
```

The convenience functions maintain backward compatibility, so existing code continues to work.

## Performance Considerations

- **Model Caching**: Models are downloaded once and cached locally
- **Landmarker Reuse**: Landmarker instances are reused across frames
- **Batch Processing**: Sequence extraction processes frames efficiently
- **Memory Management**: Large video sequences are processed frame-by-frame

## Troubleshooting

### MediaPipe Not Available

```python
ImportError: MediaPipe is not available
```

**Solution:** Install MediaPipe:
```bash
pip install mediapipe
# or
uv add mediapipe
```

### Model Download Fails

```python
❌ Download failed: [error message]
```

**Solutions:**
1. Check internet connection
2. Verify URL is accessible
3. Check disk space in models directory
4. Try manual download and place in `data/models/`

### No Pose Detected

```python
⚠️ No pose detected
```

**Solutions:**
1. Ensure person is fully visible in frame
2. Check image quality and lighting
3. Try adjusting confidence thresholds
4. Verify image is in RGB format (not BGR)

## See Also

- [Joint Angle Analysis](./joint-angle-analysis.md)
- [Gait Analysis Components](./gait-analysis.md)
- [Pose Estimation Backends](../architecture/pose-estimation.md)
- [Testing Strategy](../guides/testing-strategy.md)
