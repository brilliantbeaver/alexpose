# Get Keypoints Function Fix

**Date**: January 15, 2026  
**Status**: ✅ RESOLVED

## Problem

The notebook `notebooks/explore3 - extract features.ipynb` was failing with an `AttributeError` when calling `get_keypoints()`:

```python
AttributeError: 'dict' object has no attribute 'empty'
```

### Root Causes

1. **Type Mismatch**: The notebook passes a `Dict[str, pd.DataFrame]` (from `organize_by_sequence()`) to `get_keypoints()`, but the core function in `ambient/pose/keypoints.py` expected only `pd.DataFrame`.

2. **Return Value Mismatch**: The notebook expects a tuple `(keypoints, frame)` for visualization, but the core function only returns `List[KeypointSet]`.

3. **Incomplete Wrapper**: A wrapper function was started in `ambient/utils/eval_keypoints.py` but was not completed.

## Solution

### 1. Core Function Update (`ambient/pose/keypoints.py`)

The core `get_keypoints()` function was updated to handle both DataFrame and dict inputs:

```python
def get_keypoints(
    project_root: Path,
    sequence_data,  # Can be pd.DataFrame or Dict[str, pd.DataFrame]
    model_path: Optional[str] = None,
    verbose: bool = True
) -> List[KeypointSet]:
    """
    Extract keypoints from a video sequence.
    
    Args:
        sequence_data: DataFrame with sequence information OR dictionary of sequences
                      If dict, uses the first sequence
    
    Returns:
        List of KeypointSet objects, one per frame
    """
    # Handle both DataFrame and dict of DataFrames
    if isinstance(sequence_data, dict):
        if not sequence_data:
            raise ValueError("Empty sequences dictionary provided")
        first_key = list(sequence_data.keys())[0]
        if verbose:
            print(f"[INFO] Using first sequence: {first_key}")
        sequence_df = sequence_data[first_key]
    elif isinstance(sequence_data, pd.DataFrame):
        sequence_df = sequence_data
    else:
        raise TypeError(
            f"sequence_data must be pd.DataFrame or Dict[str, pd.DataFrame], "
            f"got {type(sequence_data)}"
        )
    
    # ... rest of implementation
```

### 2. Wrapper Function (`ambient/utils/eval_keypoints.py`)

Completed the wrapper function that provides notebook-friendly interface:

```python
def get_keypoints(
    project_root: Path,
    sequence_data,  # Can be pd.DataFrame or Dict[str, pd.DataFrame]
    model_path: Optional[str] = None,
    verbose: bool = True
) -> Tuple[List, Optional[np.ndarray]]:
    """
    Extract keypoints from a video sequence (notebook-friendly wrapper).
    
    Returns:
        Tuple of (keypoints, frame) where:
        - keypoints: List of KeypointSet objects
        - frame: First frame as numpy array (BGR) for visualization
    """
    # Use the core function to extract keypoints
    keypoints = _get_keypoints_core(
        project_root=project_root,
        sequence_data=sequence_data,
        model_path=model_path,
        verbose=verbose
    )
    
    # Extract the first frame for visualization
    frame = None
    try:
        # Handle both DataFrame and dict
        if isinstance(sequence_data, dict):
            first_key = list(sequence_data.keys())[0]
            seq_df = sequence_data[first_key]
        else:
            seq_df = sequence_data
        
        # Get first frame info and extract frame from video
        # ... (frame extraction logic)
        
    except Exception as e:
        if verbose:
            print(f"[WARNING] Error extracting frame: {e}")
        frame = None
    
    return keypoints, frame
```

## Key Changes

1. **Import Renaming**: The core function is imported as `_get_keypoints_core` to avoid naming conflicts
2. **Type Handling**: Both functions now handle `Dict[str, pd.DataFrame]` and `pd.DataFrame` inputs
3. **Return Value**: The wrapper returns `(keypoints, frame)` tuple for notebook compatibility
4. **Frame Extraction**: Added logic to extract the first frame from the video for visualization

## Testing

Created `scripts/test_get_keypoints_fix.py` to verify:
- ✅ Function imports correctly
- ✅ Handles dictionary of sequences
- ✅ Returns correct tuple format
- ✅ Types are correct (List[KeypointSet], np.ndarray)

## Usage

### In Notebooks (Interactive)
```python
from ambient.utils.eval_keypoints import get_keypoints, visualize_keypoints

# Returns tuple for visualization
keypoints, frame = get_keypoints(project_root, sequences)
visualize_keypoints(keypoints, frame)
```

### In Scripts (Programmatic)
```python
from ambient.pose.keypoints import get_keypoints

# Returns just keypoints
keypoints = get_keypoints(project_root, sequence_df)
```

## Files Modified

1. `ambient/pose/keypoints.py` - Updated core function to handle dict input
2. `ambient/utils/eval_keypoints.py` - Completed wrapper function
3. `scripts/test_get_keypoints_fix.py` - Created test script
4. `notes/GET_KEYPOINTS_FIX.md` - This documentation

## Related Issues

This fix resolves the final issue in the series:
1. ✅ Circular import (logging.py shadowing)
2. ✅ Unicode encoding errors (emoji characters)
3. ✅ Stale Python bytecode cache
4. ✅ Jupyter compatibility (UnsupportedOperation)
5. ✅ AttributeError in get_keypoints (THIS FIX)

## Next Steps

After clearing Python cache (`python scripts/clear_python_cache.py`), restart the Jupyter kernel and the notebook should work correctly.
