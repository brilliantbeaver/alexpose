# MediaPipe Coordinate Normalization Fix

## Problem Description

The `extract_from_sequence()` function was failing with the following error:

```
[ERROR] Sequence processing failed: y_normalized must be in range [0.0, 1.0], got -0.0019250214099884033
```

This error occurred during pose estimation when MediaPipe returned normalized coordinates that were slightly outside the expected [0.0, 1.0] range due to floating-point precision issues.

## Root Cause Analysis

1. **MediaPipe Precision Issues**: MediaPipe sometimes returns landmark coordinates that are very slightly outside the [0.0, 1.0] range (e.g., -0.0019250214099884033 or 1.0001) due to floating-point arithmetic precision.

2. **Strict Validation**: The `Keypoint` class in `ambient/pose/keypoint_data.py` had strict validation in its `__post_init__` method that rejected any normalized coordinates outside the exact [0.0, 1.0] range.

3. **No Tolerance**: The validation used strict inequality checks without any tolerance for floating-point precision errors.

## Solution Implemented

### 1. Coordinate Clamping in `KeypointSet.from_mediapipe()`

Modified the `from_mediapipe` class method in `ambient/pose/keypoint_data.py` to clamp coordinates to the valid range:

```python
# Clamp normalized coordinates to [0.0, 1.0] range to handle MediaPipe
# floating-point precision issues that can produce values slightly outside
# the expected range (e.g., -0.0019250214099884033)
x_normalized = max(0.0, min(1.0, landmark.x))
y_normalized = max(0.0, min(1.0, landmark.y))
```

### 2. Coordinate Clamping in `Keypoint.from_dict()`

Also added similar clamping to the `from_dict` class method to handle any external data sources:

```python
# Clamp normalized coordinates to [0.0, 1.0] range to handle potential
# floating-point precision issues from external data sources
x_normalized = max(0.0, min(1.0, data.get('x_normalized', 0.0)))
y_normalized = max(0.0, min(1.0, data.get('y_normalized', 0.0)))
```

### 3. Import Fix

Removed the non-existent `get_keypoints` function from the imports in `ambient/utils/__init__.py` to fix import errors in tests.

## Testing

Created comprehensive tests in `tests/ambient/pose/test_keypoint_normalization.py` to verify:

- ✅ Negative coordinates are clamped to 0.0
- ✅ Coordinates > 1.0 are clamped to 1.0
- ✅ Valid coordinates are preserved unchanged
- ✅ MediaPipe integration works with problematic coordinates
- ✅ Edge cases (very small values, exact boundaries) are handled
- ✅ Extreme values (infinity) are properly clamped
- ✅ The original error scenario is fixed

## Impact

- **Backward Compatible**: Valid coordinates (within [0.0, 1.0]) are preserved exactly as before
- **Robust**: The system now handles MediaPipe's floating-point precision issues gracefully
- **Minimal Performance Impact**: Clamping uses simple `max(0.0, min(1.0, value))` operations
- **Comprehensive**: Both MediaPipe integration and general data loading are protected

## Files Modified

1. `ambient/pose/keypoint_data.py` - Added coordinate clamping in `from_mediapipe()` and `from_dict()` methods
2. `ambient/utils/__init__.py` - Removed non-existent `get_keypoints` import
3. `tests/ambient/pose/test_keypoint_normalization.py` - Added comprehensive test coverage

## Verification

The fix was verified by:

1. Testing with the exact problematic coordinate from the original error (-0.0019250214099884033)
2. Running comprehensive unit tests covering various edge cases
3. Confirming that valid coordinates are preserved unchanged
4. Ensuring MediaPipe integration works seamlessly

The original error `y_normalized must be in range [0.0, 1.0], got -0.0019250214099884033` no longer occurs, and the coordinate is properly clamped to 0.0.