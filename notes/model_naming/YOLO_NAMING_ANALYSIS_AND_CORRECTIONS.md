# YOLO Model Naming Analysis and Corrections

## Investigation Summary

### Official Ultralytics Naming Conventions

Based on official Ultralytics documentation ([https://docs.ultralytics.com/](https://docs.ultralytics.com/)):

#### YOLO11 (2024 Release)
- **Official Name**: YOLO11 (NO "v")
- **Model Files**: `yolo11n-pose.pt`, `yolo11s-pose.pt`, `yolo11m-pose.pt`, etc.
- **Documentation**: Consistently refers to "YOLO11" not "YOLOv11"
- **Example**: "YOLO11n-pose", "YOLO11-pose models"

#### YOLOv8 (2023 Release)
- **Official Name**: YOLOv8 (WITH "v")
- **Model Files**: `yolov8n-pose.pt`, `yolov8s-pose.pt`, `yolov8m-pose.pt`, etc.
- **Documentation**: Consistently refers to "YOLOv8"
- **Example**: "YOLOv8n-pose", "YOLOv8-pose models"

### Why the Difference?

Ultralytics changed their naming convention starting with YOLO11:
- **YOLOv1-v10**: Used "v" prefix (YOLOv8, YOLOv9, YOLOv10)
- **YOLO11+**: Dropped "v" prefix (YOLO11, presumably future YOLO12, etc.)

This is a branding decision by Ultralytics to modernize the naming scheme.

## Current State Analysis

### ✅ CORRECT: Model File Names
Our model files use the correct official names:
- ✅ `yolov8n-pose.pt` (YOLOv8 - with "v")
- ✅ `yolo11n-pose.pt` (YOLO11 - no "v")

### ✅ CORRECT: Estimator Names
Our estimator identifiers are descriptive and acceptable:
- ✅ `yolov8-pose` (estimator name)
- ✅ `yolov11-pose` (estimator name)

**Note**: Using `yolov11-pose` as an estimator name is acceptable for consistency with `yolov8-pose`, even though the official product name is "YOLO11". The key is that the actual model file name is correct.

### ⚠️ NEEDS CORRECTION: Display Names and Documentation

Several places incorrectly refer to "YOLOv11" when it should be "YOLO11":

#### Files Needing Updates:

1. **server/services/models_service.py**
   - Current: `"YOLOv11 Pose - Advanced pose estimation using YOLOv11 architecture (2024)"`
   - Should be: `"YOLO11 Pose - Advanced pose estimation using YOLO11 architecture (2024)"`

2. **Documentation Files**
   - Multiple references to "YOLOv11" should be "YOLO11"
   - Keep "yolov11-pose" as estimator name for consistency

3. **Configuration Comments**
   - `config/alexpose.yaml`: "YOLO v11" should be "YOLO11"

4. **UI Display Names**
   - Any frontend display of "YOLOv11" should be "YOLO11"

## Naming Strategy

### Three-Tier Naming System

1. **Product Name** (Official Ultralytics Name)
   - YOLOv8 (with "v")
   - YOLO11 (no "v")
   - Used in: Documentation, UI display, user-facing text

2. **Estimator Name** (Internal Identifier)
   - `yolov8-pose` (lowercase, with hyphen)
   - `yolov11-pose` (lowercase, with hyphen)
   - Used in: Code, configuration keys, API parameters
   - **Note**: Keeping "v" in `yolov11-pose` for consistency is acceptable

3. **Model File Name** (Ultralytics Official)
   - `yolov8n-pose.pt` (with "v")
   - `yolo11n-pose.pt` (no "v")
   - Used in: File system, model loading

### Rationale for Keeping `yolov11-pose` as Estimator Name

**Pros**:
- Consistency with `yolov8-pose` naming pattern
- Avoids confusion in code (all YOLO estimators follow same pattern)
- Estimator names are internal identifiers, not user-facing
- Easier to parse programmatically (consistent pattern)

**Cons**:
- Slight mismatch with official product name
- Could be seen as incorrect by purists

**Decision**: Keep `yolov11-pose` as estimator name for consistency, but use "YOLO11" in all user-facing text.

## Corrections Needed

### High Priority (User-Facing)

1. **server/services/models_service.py**
   ```python
   # Current
   "yolov11-pose": "YOLOv11 Pose - Advanced pose estimation using YOLOv11 architecture (2024)"
   
   # Corrected
   "yolov11-pose": "YOLO11 Pose - Advanced pose estimation using YOLO11 architecture (2024)"
   ```

2. **config/alexpose.yaml**
   ```yaml
   # Current
   yolov11-pose:
     # YOLO v11 Pose model (2024 release, improved accuracy)
   
   # Corrected
   yolov11-pose:
     # YOLO11 Pose model (2024 release, improved accuracy)
   ```

3. **Frontend Display Names** (if any)
   - Change "YOLOv11" → "YOLO11" in UI components
   - Change "Yolov11" → "YOLO11" in display text

### Medium Priority (Documentation)

4. **Documentation Files**
   - `notes/generated/MODELS_BROWSE_FIX_SUMMARY.md`
   - `notes/generated/POSE_MODELS_NAMING_UPDATE.md`
   - `notes/generated/YOLO_MODELS_MIGRATION_GUIDE.md`
   - `notes/generated/YOLO_MODELS_REFACTORING_COMPLETE.md`
   - `notes/generated/YOLO_MODELS_REFACTORING_PLAN.md`
   - `docs/specs/design.md`
   - `docs/specs/testing-strategy.md`
   
   Update all references:
   - "YOLOv11" → "YOLO11"
   - "YOLOv11 Pose" → "YOLO11 Pose"
   - Keep `yolov11-pose` as estimator name

### Low Priority (Internal)

5. **Code Comments**
   - `ambient/pose/factory.py`: "YOLOv11-pose" → "YOLO11-pose" in comments
   - `ambient/core/config.py`: Update validation messages

## Implementation Plan

### Phase 1: Critical User-Facing Updates
1. Update `server/services/models_service.py` display name
2. Update `config/alexpose.yaml` comments
3. Update any frontend display names

### Phase 2: Documentation Updates
1. Update all markdown documentation files
2. Update code comments
3. Update test descriptions

### Phase 3: Verification
1. Search for remaining "YOLOv11" references
2. Verify UI displays correct names
3. Update tests if needed

## Testing Strategy

### What to Test

1. **Model Loading**
   - Verify `yolov11-pose` estimator loads `yolo11n-pose.pt` correctly
   - Verify `yolov8-pose` estimator loads `yolov8n-pose.pt` correctly

2. **Display Names**
   - Verify UI shows "YOLO11 Pose" not "YOLOv11 Pose"
   - Verify API responses use correct display names

3. **Configuration**
   - Verify config parsing works with `yolov11-pose` key
   - Verify model path resolution works correctly

### Test Cases

```python
def test_yolo11_display_name():
    """Verify YOLO11 uses correct display name (no 'v')."""
    from server.services.models_service import ModelsService
    service = ModelsService()
    models = service.list_models()
    
    yolo11_model = next(m for m in models if m['name'] == 'yolov11-pose')
    assert 'YOLO11' in yolo11_model['display_name']
    assert 'YOLOv11' not in yolo11_model['display_name']

def test_yolo8_display_name():
    """Verify YOLOv8 uses correct display name (with 'v')."""
    from server.services.models_service import ModelsService
    service = ModelsService()
    models = service.list_models()
    
    yolo8_model = next(m for m in models if m['name'] == 'yolov8-pose')
    assert 'YOLOv8' in yolo8_model['display_name']

def test_model_file_names():
    """Verify model files use correct Ultralytics names."""
    from pathlib import Path
    
    models_dir = Path("data/models")
    assert (models_dir / "yolov8n-pose.pt").exists()  # With 'v'
    assert (models_dir / "yolo11n-pose.pt").exists()  # No 'v'
```

## Summary Table

| Aspect | YOLOv8 | YOLO11 | Notes |
|--------|--------|--------|-------|
| **Official Product Name** | YOLOv8 | YOLO11 | Use in documentation, UI |
| **Estimator Name** | `yolov8-pose` | `yolov11-pose` | Internal identifier |
| **Model File Name** | `yolov8n-pose.pt` | `yolo11n-pose.pt` | Ultralytics official |
| **Display Name** | "YOLOv8 Pose" | "YOLO11 Pose" | User-facing text |
| **Config Key** | `yolov8-pose` | `yolov11-pose` | YAML configuration |
| **Code Comments** | "YOLOv8" | "YOLO11" | In-code documentation |

## References

1. **Ultralytics YOLO11 Documentation**: https://docs.ultralytics.com/tasks/pose/
   - Consistently uses "YOLO11" (no "v")
   - Model files: `yolo11n-pose.pt`, `yolo11s-pose.pt`, etc.

2. **Ultralytics YOLOv8 Documentation**: https://docs.ultralytics.com/models/yolov8/
   - Consistently uses "YOLOv8" (with "v")
   - Model files: `yolov8n-pose.pt`, `yolov8s-pose.pt`, etc.

3. **Ultralytics GitHub**: https://github.com/ultralytics/ultralytics
   - Official source for model naming conventions

## Conclusion

Our implementation is **mostly correct**:
- ✅ Model file names are correct
- ✅ Estimator names are acceptable (consistent pattern)
- ⚠️ Display names need correction (YOLOv11 → YOLO11)
- ⚠️ Documentation needs updates

The key correction needed is changing user-facing text from "YOLOv11" to "YOLO11" while keeping the internal estimator name as `yolov11-pose` for consistency.

## Next Steps

1. Apply corrections to high-priority files
2. Update documentation
3. Add tests for display names
4. Verify all user-facing text uses correct names
5. Update this analysis document with completion status
