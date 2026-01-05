# Pose Estimation Models Naming Update

## Overview

Updated the naming convention for Ultralytics YOLO pose estimation models to clearly distinguish between YOLOv8 and YOLOv11 architectures, removing ambiguous generic names.

## Problem Statement

The previous naming scheme had:
- `ultralytics` → Generic name using yolov8n-pose.pt
- `ultralytics_yolov8` → Explicit YOLOv8 using yolov8n-pose.pt
- `ultralytics_yolov11` → Explicit YOLOv11 using yolo11n-pose.pt
- `yolo` → Alias for `ultralytics`

This created confusion because:
1. Both "ultralytics" and "yolo" were aliases pointing to the same YOLOv8 model
2. The generic names didn't indicate which YOLO version was being used
3. YOLOv8 (2023) and YOLOv11 (2024) are different architectures with different capabilities

## Solution

### New Naming Convention

**Renamed Models:**
- `yolov8-pose` → YOLOv8 Pose (uses yolov8n-pose.pt, 2023 release)
- `yolov11-pose` → YOLOv11 Pose (uses yolo11n-pose.pt, 2024 release)

**Removed:**
- `ultralytics` (generic name removed)
- `ultralytics_yolov8` (redundant with new yolov8-pose)
- `ultralytics_yolov11` (redundant with new yolov11-pose)
- `yolo` (ambiguous alias removed)

### Model Descriptions

- **YOLOv8 Pose**: Real-time pose estimation using YOLOv8 architecture (2023)
  - 17 COCO keypoints
  - Fast inference speed
  - Good accuracy for real-time applications

- **YOLOv11 Pose**: Advanced pose estimation using YOLOv11 architecture (2024)
  - 17 COCO keypoints
  - Improved accuracy over YOLOv8
  - Better performance on challenging poses

## Files Modified

### 1. Configuration Files

**config/alexpose.yaml**
- Renamed `ultralytics` → `yolov8-pose`
- Renamed `ultralytics_yolov8` → removed (consolidated into yolov8-pose)
- Renamed `ultralytics_yolov11` → `yolov11-pose`

```yaml
pose_estimation:
  estimators:
    yolov8-pose:
      model_name: "yolov8n-pose.pt"
      device: "auto"
      confidence_threshold: 0.5
      enabled: false
    yolov11-pose:
      model_name: "yolo11n-pose.pt"
      device: "auto"
      confidence_threshold: 0.5
      enabled: false
```

### 2. Factory Implementation

**ambient/pose/factory.py**
- Updated `_register_default_estimators()` to register `yolov8-pose` and `yolov11-pose`
- Updated `create_estimator()` to handle new names
- Updated `_create_ultralytics_estimator()` to auto-select correct model based on estimator type
- Updated `_get_default_config()` with separate configs for each version

### 3. Service Layer

**server/services/models_service.py**
- Updated `_get_description()` with accurate descriptions for both YOLO versions

### 4. Documentation

**docs/specs/design.md**
- Updated class diagram to show `YOLOv8PoseEstimator` and `YOLOv11PoseEstimator`
- Updated configuration examples

**docs/specs/requirements.md**
- Updated Introduction to mention YOLOv8 Pose and YOLOv11 Pose
- Updated Glossary entry for Pose_Estimator
- Updated Requirement 2 acceptance criteria to distinguish between YOLOv8 and YOLOv11

**docs/specs/tasks.md**
- Updated Task 2.2 acceptance criteria to mention both YOLO versions

**docs/specs/testing-strategy.md**
- Updated test file names to reflect new naming

## API Response Changes

### Before
```json
{
  "models": [
    {"name": "ultralytics", "display_name": "Ultralytics", "available": true},
    {"name": "yolo", "display_name": "Yolo", "available": true}
  ]
}
```

### After
```json
{
  "models": [
    {"name": "yolov8-pose", "display_name": "Yolov8 Pose", "available": true},
    {"name": "yolov11-pose", "display_name": "Yolov11 Pose", "available": true}
  ]
}
```

## Browse Models Page Display

The Browse Models page now shows:

**Pose Estimation Models:**
- ✅ **MediaPipe** - Google MediaPipe Pose with 33 keypoints
- ✅ **YOLOv8 Pose** - Real-time pose estimation using YOLOv8 architecture (2023)
- ✅ **YOLOv11 Pose** - Advanced pose estimation using YOLOv11 architecture (2024)
- ❌ **OpenPose** - Not yet implemented
- ❌ **AlphaPose** - Requires manual setup

## Benefits

1. **Clarity**: Model names clearly indicate which YOLO version is being used
2. **No Ambiguity**: Removed generic aliases that could cause confusion
3. **Version Awareness**: Users can explicitly choose between YOLOv8 (2023) and YOLOv11 (2024)
4. **Future-Proof**: Easy to add YOLOv12, YOLOv13, etc. when released
5. **Consistency**: Naming follows the pattern `{architecture}-{task}` (e.g., yolov8-pose, yolov11-pose)

## Migration Guide

If you have existing code or configurations using the old names:

### Old Code
```python
# Old way
estimator = factory.create_estimator("ultralytics", config)
estimator = factory.create_estimator("yolo", config)
estimator = factory.create_estimator("ultralytics_yolov8", config)
estimator = factory.create_estimator("ultralytics_yolov11", config)
```

### New Code
```python
# New way - explicit version selection
estimator = factory.create_estimator("yolov8-pose", config)  # For YOLOv8
estimator = factory.create_estimator("yolov11-pose", config)  # For YOLOv11
```

### Configuration Migration
```yaml
# Old config
pose_estimation:
  default_estimator: "ultralytics"  # or "yolo"

# New config
pose_estimation:
  default_estimator: "yolov8-pose"  # or "yolov11-pose"
```

## Testing

All changes have been verified:
- ✅ API endpoints return correct model names
- ✅ Models show as available when properly configured
- ✅ Factory creates correct estimator instances
- ✅ Configuration loading works correctly
- ✅ Browse Models page displays updated names
- ✅ Documentation is consistent across all files

## Technical Details

### Model Architecture Differences

**YOLOv8 (2023)**:
- Anchor-free detection
- Improved backbone with C2f modules
- Decoupled head design
- 17 COCO keypoints for pose estimation

**YOLOv11 (2024)**:
- Enhanced C3k2 modules
- Improved feature pyramid network
- Better small object detection
- More accurate keypoint localization
- 17 COCO keypoints for pose estimation

Both models use the COCO 17-keypoint format:
1. Nose, 2-3. Eyes, 4-5. Ears, 6-7. Shoulders, 8-9. Elbows, 
10-11. Wrists, 12-13. Hips, 14-15. Knees, 16-17. Ankles

## Backward Compatibility

⚠️ **Breaking Change**: The old model names (`ultralytics`, `yolo`, `ultralytics_yolov8`, `ultralytics_yolov11`) are no longer supported.

Users must update their configurations and code to use the new names:
- `yolov8-pose` for YOLOv8
- `yolov11-pose` for YOLOv11

## Future Enhancements

1. Add support for different YOLO model sizes (nano, small, medium, large, xlarge)
2. Add support for custom-trained YOLO pose models
3. Add performance benchmarks comparing YOLOv8 vs YOLOv11
4. Add automatic model selection based on hardware capabilities
