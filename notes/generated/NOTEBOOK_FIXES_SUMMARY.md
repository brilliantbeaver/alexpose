# Notebook Fixes Summary - Complete Resolution

**Date**: January 15, 2026  
**Notebook**: `notebooks/explore3 - extract features.ipynb`  
**Status**: ✅ ALL ISSUES RESOLVED

## Overview

This document summarizes all fixes applied to resolve issues in the notebook that extracts gait features from video sequences. The fixes address import errors, compatibility issues, and API mismatches that occurred during the migration to the new `KeypointSet` data structure.

---

## Issue Timeline

### Issue 1: Jupyter Compatibility - `UnsupportedOperation: fileno`

**Error**:
```python
UnsupportedOperation: fileno
  File ambient/pose/keypoints.py:78, in suppress_stderr_fd()
```

**Root Cause**: The `suppress_stderr_fd()` function in `keypoints.py` tried to use `sys.stderr.fileno()`, which doesn't work in Jupyter notebooks because Jupyter's stderr is an `IPyKernelApp` stream without a real file descriptor.

**Solution**: Updated `suppress_stderr_fd()` to detect Jupyter environments and fall back to Python-level suppression:

```python
@contextlib.contextmanager
def suppress_stderr_fd():
    import io
    
    # Check if we're in Jupyter or if stderr doesn't have a file descriptor
    try:
        stderr_fd = sys.stderr.fileno()
    except (AttributeError, io.UnsupportedOperation):
        # Fall back to Python-level suppression
        old_stderr = sys.stderr
        try:
            sys.stderr = open(os.devnull, 'w')
            yield
        finally:
            sys.stderr.close()
            sys.stderr = old_stderr
        return
    
    # Normal file descriptor-based suppression for regular Python
    # ... (rest of implementation)
```

**Files Modified**:
- `ambient/pose/keypoints.py`
- `ambient/pose/suppress_warnings.py`

**Documentation**: `notes/JUPYTER_COMPATIBILITY_FIX.md`, `notes/KEYPOINTS_JUPYTER_FIX.md`

---

### Issue 2: Visualization Function - `AttributeError: add_landmark_names`

**Error**:
```python
AttributeError: type object 'KeypointVisualizer' has no attribute 'add_landmark_names'
  File ambient/utils/eval_keypoints.py:175, in visualize_keypoints
```

**Root Cause**: The `visualize_keypoints()` function was written for an older API that expected keypoints as dictionaries. It tried to call a non-existent method `KeypointVisualizer.add_landmark_names()`. The new `KeypointSet` data structure already includes landmark names in each `Keypoint` object.

**Solution**: Updated `visualize_keypoints()` to work with the new `KeypointSet` structure:

```python
def visualize_keypoints(keypoints: list, frame: np.ndarray):
    """Visualize keypoints on a frame with side-by-side comparison."""
    
    if keypoints and len(keypoints) > 0:
        # Get the first frame's keypoints (KeypointSet object)
        first_keypoint_set = keypoints[0]
        
        # Use draw_skeleton for better visualization
        annotated = KeypointVisualizer.draw_skeleton(
            frame_rgb, 
            first_keypoint_set,
            confidence_threshold=0.5,
            keypoint_color=(255, 0, 0),
            line_color=(0, 255, 0)
        )
        
        # Get and display statistics
        stats = KeypointVisualizer.get_summary_stats(first_keypoint_set)
        # ... (display stats)
```

**Key Changes**:
1. Removed call to non-existent `add_landmark_names()` method
2. Fixed to use `draw_skeleton()` with a single `KeypointSet` object
3. Cleaned up print statements (removed emoji characters for Windows compatibility)

**Files Modified**:
- `ambient/utils/eval_keypoints.py`

**Documentation**: `notes/VISUALIZE_KEYPOINTS_FIX.md`

---

## Data Structure Migration

The root cause of both issues was the migration from dictionary-based keypoints to the new `KeypointSet` data structure:

### Old Format (Dictionary-based)
```python
keypoints = [
    {'x': 100, 'y': 150, 'confidence': 0.95},
    {'x': 120, 'y': 160, 'confidence': 0.90},
    # ...
]
```

### New Format (KeypointSet-based)
```python
keypoint_set = KeypointSet(
    keypoints=[
        Keypoint(id=0, name='NOSE', x=100, y=150, confidence=0.95, ...),
        Keypoint(id=1, name='LEFT_EYE', x=120, y=160, confidence=0.90, ...),
        # ...
    ],
    format=KeypointFormat.MEDIAPIPE_33,
    frame_width=640,
    frame_height=480
)
```

### Benefits of New Structure
- **Type Safety**: Strong typing with validation
- **Rich Metadata**: Each keypoint includes name, visibility, presence scores
- **Extensibility**: Easy to add new pose formats
- **Interoperability**: Conversion to/from numpy, pandas, dictionaries
- **Better API**: Clear methods like `visible_keypoints`, `reliable_keypoints`, `avg_confidence`

---

## Testing

### Test Scripts Created

1. **`scripts/test_jupyter_imports.py`** - Tests Jupyter compatibility
2. **`scripts/test_get_keypoints_fix.py`** - Tests keypoint extraction
3. **`scripts/test_visualize_keypoints.py`** - Tests visualization functions

### Test Results

All tests pass successfully:

```
TEST 1: KeypointVisualizer Methods ✅
TEST 2: KeypointSet Structure ✅
TEST 3: Draw Skeleton Function ✅
TEST 4: Get Summary Stats Function ✅
TEST 5: Visualize Keypoints Import ✅

Passed: 5/5
```

---

## How to Use the Fixed Notebook

### 1. Clear Python Cache
```bash
python scripts/clear_python_cache.py
```

### 2. Restart Jupyter Kernel
In Jupyter, click: `Kernel` → `Restart Kernel`

### 3. Run the Notebook Cells
```python
from ambient.utils.eval_keypoints import get_keypoints, visualize_keypoints

# Extract keypoints and first frame
keypoints, frame = get_keypoints(project_root, sequences)

# Visualize results
visualize_keypoints(keypoints, frame)
```

### Expected Output
- Side-by-side visualization showing original frame and detected pose with skeleton
- Statistics about detected landmarks:
  - Total landmarks detected
  - Number of visible landmarks (confidence > 0.5)
  - Average confidence score
  - Overall detection quality
- Sample keypoint coordinates for first 5 landmarks

---

## API Reference

### KeypointSet Structure

```python
@dataclass
class KeypointSet:
    keypoints: List[Keypoint]           # List of detected keypoints
    format: KeypointFormat              # MEDIAPIPE_33, COCO_17, etc.
    frame_width: int                    # Frame width in pixels
    frame_height: int                   # Frame height in pixels
    timestamp: Optional[float]          # Frame number or time
    person_id: Optional[int]            # For multi-person tracking
    metadata: Dict[str, Any]            # Additional metadata
    
    # Properties
    @property
    def visible_keypoints(self) -> List[Keypoint]
    @property
    def reliable_keypoints(self) -> List[Keypoint]
    @property
    def avg_confidence(self) -> float
    @property
    def detection_quality(self) -> float
```

### Keypoint Structure

```python
@dataclass(frozen=True)
class Keypoint:
    id: int                    # Unique identifier (0-indexed)
    name: str                  # Semantic name (e.g., "LEFT_ELBOW")
    x: float                   # X coordinate in pixels
    y: float                   # Y coordinate in pixels
    z: float                   # Depth coordinate
    confidence: float          # Overall confidence [0.0, 1.0]
    visibility: float          # Visibility score [0.0, 1.0]
    presence: float            # Presence score [0.0, 1.0]
    x_normalized: float        # X normalized to [0.0, 1.0]
    y_normalized: float        # Y normalized to [0.0, 1.0]
    
    # Properties
    @property
    def is_visible(self) -> bool
    @property
    def is_reliable(self) -> bool
```

### KeypointVisualizer Methods

```python
class KeypointVisualizer:
    @staticmethod
    def draw_keypoints(
        image: np.ndarray,
        keypoint_set: KeypointSet,
        confidence_threshold: float = 0.5,
        color: Tuple[int, int, int] = (255, 0, 0),
        radius: int = 5
    ) -> np.ndarray
    
    @staticmethod
    def draw_skeleton(
        image: np.ndarray,
        keypoint_set: KeypointSet,
        confidence_threshold: float = 0.5,
        keypoint_color: Tuple[int, int, int] = (255, 0, 0),
        line_color: Tuple[int, int, int] = (0, 255, 0),
        radius: int = 5,
        thickness: int = 2
    ) -> np.ndarray
    
    @staticmethod
    def get_summary_stats(keypoint_set: KeypointSet) -> Dict[str, Union[int, float]]
```

---

## Related Documentation

- `notes/JUPYTER_COMPATIBILITY_FIX.md` - Jupyter stderr suppression fix
- `notes/KEYPOINTS_JUPYTER_FIX.md` - Keypoints.py Jupyter compatibility
- `notes/GET_KEYPOINTS_FIX.md` - Get keypoints function fixes
- `notes/VISUALIZE_KEYPOINTS_FIX.md` - Visualization function fix
- `notes/IMPORT_ERROR_FIX.md` - Python cache clearing guide

---

## Previous Fixes (Context Transfer)

This work builds on previous fixes:

1. ✅ **Task 1**: Migrated `notebooks/utils/` to `ambient/utils/`
2. ✅ **Task 2**: Fixed circular import (logging.py shadowing)
3. ✅ **Task 3**: Fixed Unicode encoding errors (emoji characters)
4. ✅ **Task 4**: Fixed stale Python bytecode cache
5. ✅ **Task 5**: Simplified `get_keypoints()` to DataFrame-only
6. ✅ **Task 6**: Fixed Jupyter compatibility in `suppress_warnings.py`
7. ✅ **Task 7**: Fixed Jupyter compatibility in `keypoints.py`
8. ✅ **Task 8**: Fixed `visualize_keypoints()` function (THIS WORK)

---

## Verification Checklist

- [x] All test scripts pass
- [x] Python cache cleared
- [x] No emoji characters in code (Windows compatibility)
- [x] Jupyter compatibility verified
- [x] KeypointSet API correctly used
- [x] Visualization functions work correctly
- [x] Documentation complete

---

## Next Steps

The notebook should now work correctly. If you encounter any issues:

1. **Clear Python cache**: `python scripts/clear_python_cache.py`
2. **Restart Jupyter kernel**: `Kernel` → `Restart Kernel`
3. **Check test scripts**: Run the test scripts to verify functionality
4. **Review documentation**: Check the related documentation files for details

---

## Contact

For questions or issues, refer to:
- Project structure: `notes/structure.md`
- Tech stack: `notes/tech.md`
- Product overview: `notes/product.md`
