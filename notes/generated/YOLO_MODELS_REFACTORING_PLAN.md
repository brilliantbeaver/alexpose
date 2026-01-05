# YOLO Pose Models Refactoring Plan

## Overview
Refactor YOLO pose estimation model files from project root to `data/models/` directory with full backward compatibility.

## Current State Analysis

### Current Model Locations
- `yolov8n-pose.pt` - Located at project root
- `yolo11n-pose.pt` - Located at project root

### Current References
1. **Config Files**:
   - `config/alexpose.yaml`: References model names directly
   - `.kiro/specs/gavd-gait-analysis/design.md`: Documentation references
   - `docs/specs/design.md`: Documentation references

2. **Source Code**:
   - `ambient/pose/ultralytics_estimator.py`: Default parameter `model_name="yolov8n-pose.pt"`
   - `ambient/pose/factory.py`: Default model names in `_create_ultralytics_estimator()` and `_get_default_config()`

3. **Model Loading**:
   - `UltralyticsEstimator.__init__()`: Calls `YOLO(model_name)` directly
   - Ultralytics library handles model path resolution

### How Ultralytics YOLO Loads Models

The Ultralytics library's `YOLO()` constructor accepts:
1. **Model name only** (e.g., `"yolov8n-pose.pt"`):
   - Searches in current working directory
   - Downloads from Ultralytics hub if not found locally
   
2. **Absolute path** (e.g., `"/path/to/model.pt"`):
   - Loads directly from specified path
   
3. **Relative path** (e.g., `"data/models/yolov8n-pose.pt"`):
   - Resolves relative to current working directory

## Refactoring Strategy

### Goals
1. ✅ Move model files to `data/models/` directory
2. ✅ Update all code references to use new paths
3. ✅ Maintain backward compatibility
4. ✅ Support both absolute and relative paths
5. ✅ Allow configuration override
6. ✅ Ensure all tests pass

### Approach: Path Resolution with Fallback

Implement a robust path resolution strategy:

```python
def resolve_model_path(model_name: str, models_dir: str = "data/models") -> str:
    """
    Resolve YOLO model path with fallback strategy.
    
    Priority:
    1. If absolute path exists, use it
    2. If relative path exists, use it
    3. Check in models_dir
    4. Check in project root (backward compatibility)
    5. Return model_name (let Ultralytics download)
    """
```

### Benefits
- **Backward Compatible**: Existing deployments with models in root still work
- **Forward Compatible**: New deployments use organized structure
- **Flexible**: Supports custom paths via configuration
- **Robust**: Falls back to auto-download if model not found

## Implementation Plan

### Phase 1: Create Model Path Resolution Utility
**File**: `ambient/utils/model_utils.py` (NEW)

```python
from pathlib import Path
from typing import Union, Optional
from loguru import logger

def resolve_yolo_model_path(
    model_name: str,
    models_directory: Optional[Union[str, Path]] = None,
    check_project_root: bool = True
) -> str:
    """
    Resolve YOLO model path with intelligent fallback.
    
    Args:
        model_name: Model name or path (e.g., "yolov8n-pose.pt")
        models_directory: Directory to search for models (default: "data/models")
        check_project_root: Whether to check project root for backward compatibility
        
    Returns:
        Resolved model path or original model_name
    """
```

### Phase 2: Update UltralyticsEstimator
**File**: `ambient/pose/ultralytics_estimator.py`

Changes:
1. Import `resolve_yolo_model_path` utility
2. Update `__init__()` to resolve model path before loading
3. Store both original model_name and resolved_path
4. Update logging to show resolved path

### Phase 3: Update Factory Defaults
**File**: `ambient/pose/factory.py`

Changes:
1. Update default model paths in `_create_ultralytics_estimator()`
2. Update default config in `_get_default_config()`
3. Use `data/models/` prefix for new defaults

### Phase 4: Update Configuration
**File**: `config/alexpose.yaml`

Changes:
1. Update `yolov8-pose.model_name` to `"data/models/yolov8n-pose.pt"`
2. Update `yolov11-pose.model_name` to `"data/models/yolo11n-pose.pt"`
3. Add comments explaining path resolution

### Phase 5: Move Model Files
**Actions**:
1. Copy `yolov8n-pose.pt` to `data/models/yolov8n-pose.pt`
2. Copy `yolo11n-pose.pt` to `data/models/yolo11n-pose.pt`
3. Keep originals in root for backward compatibility (optional)

### Phase 6: Update Documentation
**Files**:
- `README.md`: Update model installation instructions
- `.kiro/specs/gavd-gait-analysis/design.md`: Update paths
- `docs/specs/design.md`: Update paths
- Create migration guide

### Phase 7: Testing
**Test Coverage**:
1. Unit tests for path resolution utility
2. Integration tests for model loading
3. Backward compatibility tests (models in root)
4. Forward compatibility tests (models in data/models)
5. Configuration override tests

## Backward Compatibility Strategy

### Scenario 1: Models in Project Root (Current)
```python
# Model file: yolov8n-pose.pt (in project root)
# Config: model_name: "yolov8n-pose.pt"
# Result: ✅ Loads from project root
```

### Scenario 2: Models in data/models (New)
```python
# Model file: data/models/yolov8n-pose.pt
# Config: model_name: "data/models/yolov8n-pose.pt"
# Result: ✅ Loads from data/models
```

### Scenario 3: Models in Both Locations
```python
# Model files: yolov8n-pose.pt AND data/models/yolov8n-pose.pt
# Config: model_name: "yolov8n-pose.pt"
# Result: ✅ Prefers data/models, falls back to root
```

### Scenario 4: Model Not Found Locally
```python
# No local model file
# Config: model_name: "yolov8n-pose.pt"
# Result: ✅ Ultralytics downloads from hub
```

## Migration Path for Users

### For New Installations
1. Models automatically placed in `data/models/`
2. Configuration uses new paths
3. No action required

### For Existing Installations
**Option A: Keep Current Setup (No Action)**
- Models remain in project root
- Everything continues to work
- Path resolution finds them automatically

**Option B: Migrate to New Structure (Recommended)**
1. Move models to `data/models/`:
   ```bash
   mv yolov8n-pose.pt data/models/
   mv yolo11n-pose.pt data/models/
   ```
2. Update config (optional - auto-detected):
   ```yaml
   yolov8-pose:
     model_name: "data/models/yolov8n-pose.pt"
   ```
3. Test: `python -m pytest tests/`

## Testing Strategy

### Test Cases

#### 1. Path Resolution Tests
```python
def test_resolve_model_path_absolute()
def test_resolve_model_path_relative()
def test_resolve_model_path_in_models_dir()
def test_resolve_model_path_in_project_root()
def test_resolve_model_path_not_found()
def test_resolve_model_path_priority()
```

#### 2. Model Loading Tests
```python
def test_load_model_from_models_dir()
def test_load_model_from_project_root()
def test_load_model_with_absolute_path()
def test_load_model_with_config_override()
```

#### 3. Backward Compatibility Tests
```python
def test_backward_compat_models_in_root()
def test_backward_compat_config_unchanged()
def test_backward_compat_existing_code()
```

#### 4. Integration Tests
```python
def test_factory_creates_estimator_with_new_paths()
def test_estimator_loads_model_successfully()
def test_pose_estimation_works_with_new_paths()
```

## Risk Assessment

### Low Risk
- ✅ Path resolution with fallback is safe
- ✅ Backward compatibility maintained
- ✅ No breaking changes to API
- ✅ Existing deployments unaffected

### Medium Risk
- ⚠️ Users with custom model paths may need config updates
- ⚠️ Documentation needs to be updated consistently

### Mitigation
- Comprehensive testing before deployment
- Clear migration documentation
- Deprecation warnings for old paths (optional)
- Rollback plan (keep models in both locations temporarily)

## Implementation Checklist

### Code Changes
- [ ] Create `ambient/utils/model_utils.py` with path resolution
- [ ] Update `ambient/pose/ultralytics_estimator.py`
- [ ] Update `ambient/pose/factory.py`
- [ ] Update `config/alexpose.yaml`

### File Operations
- [ ] Copy `yolov8n-pose.pt` to `data/models/`
- [ ] Copy `yolo11n-pose.pt` to `data/models/`
- [ ] Add `.gitignore` entry for `*.pt` files (optional)

### Testing
- [ ] Create `tests/test_model_path_resolution.py`
- [ ] Create `tests/test_yolo_model_loading.py`
- [ ] Run full test suite
- [ ] Manual testing with both path configurations

### Documentation
- [ ] Update `README.md`
- [ ] Update `.kiro/specs/gavd-gait-analysis/design.md`
- [ ] Update `docs/specs/design.md`
- [ ] Create `YOLO_MODELS_MIGRATION_GUIDE.md`

### Verification
- [ ] All tests pass
- [ ] Models load from `data/models/`
- [ ] Backward compatibility verified
- [ ] Configuration override works
- [ ] Documentation is accurate

## Timeline

### Immediate (Phase 1-3)
- Create utility and update code
- Estimated: 1-2 hours

### Short-term (Phase 4-5)
- Update config and move files
- Estimated: 30 minutes

### Medium-term (Phase 6-7)
- Documentation and testing
- Estimated: 2-3 hours

**Total Estimated Time**: 4-6 hours

## Success Criteria

1. ✅ Model files located in `data/models/` directory
2. ✅ All code references updated to use new paths
3. ✅ Backward compatibility maintained (models in root still work)
4. ✅ All existing tests pass
5. ✅ New tests added for path resolution
6. ✅ Documentation updated
7. ✅ No breaking changes to user-facing API

## Rollback Plan

If issues arise:
1. Revert code changes (git revert)
2. Models in project root still work
3. No data loss or corruption risk
4. Quick rollback possible

## Future Enhancements

1. **Model Version Management**: Track model versions in config
2. **Auto-Download**: Automatically download missing models
3. **Model Registry**: Central registry of available models
4. **Model Validation**: Verify model integrity on load
5. **Model Caching**: Cache model metadata for faster loading

## Conclusion

This refactoring plan provides a safe, systematic approach to organizing YOLO model files while maintaining full backward compatibility. The path resolution strategy ensures existing deployments continue to work while new deployments benefit from better organization.
