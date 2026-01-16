# Keypoints Module Rename: keypoints.py → eval_keypoints.py

## Overview

Renamed `notebooks/utils/keypoints.py` to `notebooks/utils/eval_keypoints.py` to better reflect its purpose as an evaluation and testing utility for interactive notebook exploration, rather than core functionality.

## Rationale

The original name `keypoints.py` was ambiguous and could be confused with the core `ambient/pose/keypoints.py` module. The new name `eval_keypoints.py` clearly indicates:

1. **Purpose**: Evaluation and testing utilities for notebooks
2. **Scope**: Notebook-specific wrappers, not core functionality
3. **Usage**: Interactive exploration and visualization

## What Changed

### File Renamed
- **Old**: `notebooks/utils/keypoints.py`
- **New**: `notebooks/utils/eval_keypoints.py`

### Updated Imports

**Notebooks:**
- `notebooks/explore3 - extract features.ipynb` (2 import statements)

**Tests:**
- `tests/test_keypoints_fix.py`

**Documentation (18 files):**
- `docs/analysis/keypoint-extraction.md`
- `docs/gait/JOINT_ANGLES_IMPLEMENTATION_SUMMARY.md`
- `docs/guides/suppressing-mediapipe-warnings.md`
- `notes/mediapipe/MEDIAPIPE_COMPLETE_SOLUTION.md`
- `notes/mediapipe/MEDIAPIPE_FINAL_SUMMARY.md`
- `notes/mediapipe/MEDIAPIPE_KEYERROR_FIX_SUMMARY.md`
- `notes/mediapipe/MEDIAPIPE_LOGGING_FIX.md`
- `notes/mediapipe/MEDIAPIPE_REFACTORING_SUMMARY.md`
- `notes/mediapipe/mediapipe-logging-suppression.md`
- `notes/mediapipe/POSE_LOGGING_FIX.md`
- `notes/mediapipe/pose-logging-suppression.md`

### Module Documentation Updated

Updated the module docstring to clarify:
```python
"""
Notebook Utilities for Keypoint Evaluation and Visualization

Renamed from keypoints.py to eval_keypoints.py to clarify its purpose as an
evaluation/testing utility rather than core functionality.
"""
```

## Migration Guide

### For Notebook Users

**Old import:**
```python
from utils.keypoints import get_keypoints, visualize_keypoints
from utils.keypoints import extract_pose_from_sequence
```

**New import:**
```python
from utils.eval_keypoints import get_keypoints, visualize_keypoints
from utils.eval_keypoints import extract_pose_from_sequence
```

### For Test Scripts

**Old import:**
```python
from notebooks.utils.keypoints import pose_estimation_all_sequences
```

**New import:**
```python
from notebooks.utils.eval_keypoints import pose_estimation_all_sequences
```

## Module Purpose

The `eval_keypoints.py` module provides:

1. **Evaluation Functions**: Test pose detection across sequences
   - `extract_pose_from_sequence()` - Single frame evaluation
   - `pose_estimation_for_frames()` - Multiple frame evaluation
   - `pose_estimation_all_sequences()` - Batch evaluation

2. **Visualization Functions**: Interactive notebook visualization
   - `visualize_keypoints()` - Side-by-side pose visualization
   - Integration with matplotlib for notebook display

3. **Convenience Re-exports**: Easy access to core functionality
   - Re-exports from `ambient.pose.keypoints`
   - Re-exports from `ambient.pose.joint_angles`

4. **GAVD Integration**: Dataset-specific utilities
   - Frame extraction from cached videos
   - Sequence-based processing
   - Metadata handling

## Core vs Evaluation Modules

### Core Module: `ambient/pose/keypoints.py`
- Production-ready pose estimation
- SOLID principles, fully tested
- Used by server and CLI
- Minimal dependencies
- No visualization code

### Evaluation Module: `notebooks/utils/eval_keypoints.py`
- Interactive notebook exploration
- Visualization with matplotlib
- GAVD dataset integration
- Progress reporting
- Convenience wrappers

## Verification

All changes verified:
```bash
# Test imports work
python -c "from notebooks.utils.eval_keypoints import get_keypoints; print('✅ Success')"

# Test file exists
Test-Path "notebooks\utils\eval_keypoints.py"  # True

# Test old file removed
Test-Path "notebooks\utils\keypoints.py"  # False

# Run test suite
python tests/test_keypoints_fix.py  # All tests pass
```

## Files Modified Summary

| Category | Count | Files |
|----------|-------|-------|
| Renamed | 1 | `notebooks/utils/keypoints.py` → `eval_keypoints.py` |
| Notebooks | 1 | `explore3 - extract features.ipynb` |
| Tests | 1 | `tests/test_keypoints_fix.py` |
| Documentation | 11 | Various docs and notes files |
| **Total** | **14** | |

## Benefits

1. **Clarity**: Name clearly indicates evaluation/testing purpose
2. **Organization**: Distinguishes from core `ambient/pose/keypoints.py`
3. **Maintainability**: Easier to understand module relationships
4. **Documentation**: Self-documenting filename

## Backward Compatibility

⚠️ **Breaking Change**: Old imports will fail

Users must update their imports from:
- `from utils.keypoints import ...`

To:
- `from utils.eval_keypoints import ...`

This is intentional to force explicit migration and avoid confusion.

## Related Modules

```
ambient/pose/keypoints.py          # Core pose estimation (production)
    ↓ used by
notebooks/utils/eval_keypoints.py  # Evaluation utilities (notebooks)
    ↓ used by
notebooks/explore3 - extract features.ipynb  # Interactive exploration
```

## Conclusion

The rename from `keypoints.py` to `eval_keypoints.py` improves code organization and clarity by explicitly indicating the module's purpose as an evaluation and testing utility for interactive notebook exploration.

All references have been systematically updated across notebooks, tests, and documentation.
