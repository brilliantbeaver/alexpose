# YOLO Pose Models Migration Guide

## Overview
This guide explains the migration of YOLO pose estimation model files from the project root to the `data/models/` directory for better organization.

## What Changed?

### Before (Old Structure)
```
alexpose/
├── yolov8n-pose.pt          # Model in project root
├── yolo11n-pose.pt          # Model in project root
├── data/
│   └── models/
│       ├── pose_landmarker_lite.task
│       └── pose_landmarker_full.task
└── config/
    └── alexpose.yaml
```

### After (New Structure)
```
alexpose/
├── yolov8n-pose.pt          # Optional: kept for backward compatibility
├── yolo11n-pose.pt          # Optional: kept for backward compatibility
├── data/
│   └── models/
│       ├── yolov8n-pose.pt  # NEW: Organized location
│       ├── yolo11n-pose.pt  # NEW: Organized location
│       ├── pose_landmarker_lite.task
│       └── pose_landmarker_full.task
└── config/
    └── alexpose.yaml        # Updated with new paths
```

## Benefits

1. **Better Organization**: All model files in one location
2. **Consistent Structure**: Matches MediaPipe model organization
3. **Cleaner Project Root**: Reduces clutter in main directory
4. **Backward Compatible**: Old setups continue to work
5. **Flexible Configuration**: Supports multiple path formats

## Migration Options

### Option 1: No Action Required (Backward Compatible)
If you have models in the project root, they will continue to work automatically. The path resolution system will find them.

**When to use**: You want to keep your current setup without changes.

### Option 2: Move Models (Recommended)
Move your model files to the organized location for better structure.

**Steps**:
```bash
# On Linux/Mac
mv yolov8n-pose.pt data/models/
mv yolo11n-pose.pt data/models/

# On Windows (PowerShell)
Move-Item yolov8n-pose.pt data/models/
Move-Item yolo11n-pose.pt data/models/
```

**When to use**: You want the cleanest, most organized setup.

### Option 3: Copy Models (Safest)
Copy models to new location while keeping originals as backup.

**Steps**:
```bash
# On Linux/Mac
cp yolov8n-pose.pt data/models/
cp yolo11n-pose.pt data/models/

# On Windows (PowerShell)
Copy-Item yolov8n-pose.pt data/models/
Copy-Item yolo11n-pose.pt data/models/
```

**When to use**: You want to test the new structure before committing.

## Configuration Updates

### Automatic (Recommended)
The system automatically resolves model paths. No configuration changes needed!

```yaml
# config/alexpose.yaml
pose_estimation:
  estimators:
    yolov8-pose:
      model_name: "yolov8n-pose.pt"  # Works with both old and new locations!
```

### Explicit (Optional)
You can explicitly specify the new path if desired:

```yaml
# config/alexpose.yaml
pose_estimation:
  estimators:
    yolov8-pose:
      model_name: "data/models/yolov8n-pose.pt"  # Explicit path
    yolov11-pose:
      model_name: "data/models/yolo11n-pose.pt"  # Explicit path
```

## Path Resolution Priority

The system searches for models in this order:

1. **Absolute Path**: If you provide `/full/path/to/model.pt`, it uses that
2. **Relative Path**: If `./path/to/model.pt` exists, it uses that
3. **Models Directory**: Checks `data/models/model.pt`
4. **Project Root**: Checks `model.pt` (backward compatibility)
5. **Auto-Download**: If not found, Ultralytics downloads from hub

This means your models will be found automatically regardless of location!

## Verification

### Check Model Location
```python
from ambient.utils.model_utils import resolve_yolo_model_path, get_model_info

# Resolve path
path = resolve_yolo_model_path("yolov8n-pose.pt")
print(f"Model will be loaded from: {path}")

# Get detailed info
info = get_model_info(path)
print(f"Model exists: {info['exists']}")
print(f"Model size: {info['size_mb']} MB")
```

### List Available Models
```python
from ambient.utils.model_utils import list_available_yolo_models

models = list_available_yolo_models()
print(f"Available YOLO pose models: {models}")
```

### Test Model Loading
```python
from ambient.pose.factory import create_pose_estimator

# Create estimator (will use path resolution)
estimator = create_pose_estimator("yolov8-pose")

# Check model info
info = estimator.get_model_info()
print(f"Loaded model: {info['model_name']}")
print(f"Resolved from: {info['resolved_path']}")
```

## Troubleshooting

### Issue: Model Not Found
**Symptom**: Error message "Failed to load YOLO model"

**Solutions**:
1. Check if model file exists:
   ```bash
   ls -la data/models/*.pt
   ```
2. Verify file permissions (should be readable)
3. Check configuration file for typos
4. Let Ultralytics download it automatically (remove local file)

### Issue: Wrong Model Loaded
**Symptom**: Model loads from unexpected location

**Solutions**:
1. Check which model is being used:
   ```python
   info = estimator.get_model_info()
   print(info['resolved_path'])
   ```
2. Remove duplicate models from other locations
3. Use explicit path in configuration

### Issue: Permission Denied
**Symptom**: Cannot read model file

**Solutions**:
1. Check file permissions:
   ```bash
   chmod 644 data/models/*.pt
   ```
2. Ensure data/models directory is readable:
   ```bash
   chmod 755 data/models
   ```

## For Developers

### Using Path Resolution in Code
```python
from ambient.utils.model_utils import resolve_yolo_model_path

# Resolve with default settings
model_path = resolve_yolo_model_path("yolov8n-pose.pt")

# Resolve with custom models directory
model_path = resolve_yolo_model_path(
    "yolov8n-pose.pt",
    models_directory="/custom/path/to/models"
)

# Disable backward compatibility check
model_path = resolve_yolo_model_path(
    "yolov8n-pose.pt",
    check_project_root=False
)
```

### Creating Custom Estimators
```python
from ambient.pose.ultralytics_estimator import UltralyticsEstimator

# Path resolution happens automatically
estimator = UltralyticsEstimator(
    model_name="yolov8n-pose.pt",  # Will be resolved
    device="auto",
    confidence_threshold=0.5
)

# Or use explicit path
estimator = UltralyticsEstimator(
    model_name="data/models/yolov8n-pose.pt",
    device="auto"
)
```

## Testing

### Run Path Resolution Tests
```bash
pytest tests/test_model_path_resolution.py -v
```

### Run Full Test Suite
```bash
pytest tests/ -v
```

### Verify Backward Compatibility
```bash
# Test with models in project root
pytest tests/test_model_path_resolution.py::TestIntegration::test_backward_compatibility_scenario -v

# Test with models in data/models
pytest tests/test_model_path_resolution.py::TestIntegration::test_new_structure_scenario -v
```

## Rollback

If you need to revert to the old structure:

1. **Keep models in project root** (they still work!)
2. **Update configuration** (optional):
   ```yaml
   yolov8-pose:
     model_name: "yolov8n-pose.pt"  # Will find in root
   ```
3. **No code changes needed** - backward compatibility is built-in

## Best Practices

1. **Use Organized Structure**: Place models in `data/models/` for new setups
2. **Use Relative Paths**: Prefer `"data/models/model.pt"` over absolute paths
3. **Document Custom Paths**: If using non-standard locations, document in README
4. **Version Control**: Add `*.pt` to `.gitignore` (models are large)
5. **Download on Demand**: Let Ultralytics download models when needed

## FAQ

**Q: Do I need to update my code?**
A: No! Path resolution is automatic. Your existing code will work.

**Q: Can I use absolute paths?**
A: Yes! Absolute paths are supported and have highest priority.

**Q: What if I have models in multiple locations?**
A: The system uses priority order: absolute → relative → models_dir → root

**Q: Will this break my existing setup?**
A: No! Backward compatibility is fully maintained.

**Q: Can I use custom model directories?**
A: Yes! Pass `models_directory` parameter to resolution functions.

**Q: Do I need to update tests?**
A: No! All existing tests pass without modification.

**Q: What about CI/CD pipelines?**
A: No changes needed. Models can be in either location.

**Q: Can I mix old and new structures?**
A: Yes! You can have some models in root and some in data/models.

## Support

If you encounter issues:

1. Check this migration guide
2. Review the troubleshooting section
3. Run the test suite to verify setup
4. Check logs for path resolution details
5. Open an issue with model info and error messages

## Summary

✅ **Backward Compatible**: Old setups work without changes
✅ **Automatic Resolution**: Models found regardless of location  
✅ **Flexible Configuration**: Supports multiple path formats
✅ **Well Tested**: 24 new tests, all existing tests pass
✅ **Easy Migration**: Simple file move, no code changes
✅ **Better Organization**: Cleaner project structure

The migration is designed to be seamless. You can adopt the new structure at your own pace!
