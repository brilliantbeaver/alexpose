# Visualize Keypoints Function Fix

**Date**: 2026-01-15  
**Issue**: `AttributeError: type object 'KeypointVisualizer' has no attribute 'add_landmark_names'`

## Problem

The `visualize_keypoints()` function in `ambient/utils/eval_keypoints.py` was trying to call a non-existent method `KeypointVisualizer.add_landmark_names()` on the keypoints list.

### Root Cause

The function was written for an older API that expected keypoints as dictionaries without landmark names. It tried to add names using a method that never existed. The new `KeypointSet` data structure already includes landmark names in each `Keypoint` object, so this step is unnecessary.

### Error Details

```python
# Old code (BROKEN)
keypoints = KeypointVisualizer.add_landmark_names(keypoints)  # Method doesn't exist!
```

The `KeypointVisualizer` class only has these static methods:
- `draw_keypoints()` - Draw keypoints on an image
- `draw_skeleton()` - Draw keypoints with skeleton connections
- `get_summary_stats()` - Get statistics about a KeypointSet

## Solution

Updated `visualize_keypoints()` to work correctly with the new `KeypointSet` data structure:

1. **Removed the non-existent method call** - No need to add landmark names since they're already in the KeypointSet
2. **Fixed the visualization** - Use `draw_skeleton()` correctly with a single KeypointSet object
3. **Cleaned up print statements** - Removed emoji characters that cause encoding issues on Windows

### Changes Made

```python
# NEW CODE (FIXED)
def visualize_keypoints(keypoints: list, frame: np.ndarray):
    """Visualize keypoints on a frame with side-by-side comparison."""
    
    if keypoints and len(keypoints) > 0:
        # Get the first frame's keypoints (KeypointSet object)
        first_keypoint_set = keypoints[0]
        
        # Visualize
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 7))
        
        # Original frame
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        ax1.imshow(frame_rgb)
        ax1.set_title('Original Frame')
        ax1.axis('off')
        
        # Frame with pose - use draw_skeleton for better visualization
        annotated = KeypointVisualizer.draw_skeleton(
            frame_rgb, 
            first_keypoint_set,
            confidence_threshold=0.5,
            keypoint_color=(255, 0, 0),
            line_color=(0, 255, 0)
        )
        ax2.imshow(annotated)
        ax2.set_title(f'MediaPipe Detection ({len(first_keypoint_set)} landmarks)')
        ax2.axis('off')
        
        plt.tight_layout()
        plt.show()
        
        # Get and display statistics
        stats = KeypointVisualizer.get_summary_stats(first_keypoint_set)
        print(f"[OK] SUCCESS! Detected {stats['total_landmarks']} landmarks")
        print(f"[INFO] {stats['visible_landmarks']} landmarks are visible")
        print(f"[INFO] Average confidence: {stats['avg_confidence']:.3f}")
        print(f"[INFO] Detection quality: {stats['detection_quality']:.3f}")
        
        # Show sample keypoints
        print("\n[INFO] Sample keypoints:")
        for i, kp in enumerate(first_keypoint_set.keypoints[:5]):
            name = kp.name if kp.name else f"Point {kp.id}"
            print(f"  {name}: ({kp.x:.1f}, {kp.y:.1f}) confidence={kp.confidence:.3f}")
    else:
        print("[WARNING] No keypoints to visualize")
```

## Key Points

1. **KeypointSet Structure**: Each KeypointSet contains a list of Keypoint objects with:
   - `id`: Unique identifier (0-indexed)
   - `name`: Semantic name (e.g., "LEFT_ELBOW", "NOSE")
   - `x, y, z`: Coordinates
   - `confidence`: Detection confidence
   - `visibility`: Visibility score
   - `presence`: Presence score

2. **No Name Addition Needed**: The `KeypointSet.from_mediapipe()` method already assigns landmark names when creating keypoints from MediaPipe results.

3. **Correct Visualization**: Use `draw_skeleton()` with a single KeypointSet object, not a list.

## Testing

After the fix, the notebook cell should work correctly:

```python
from ambient.utils.eval_keypoints import get_keypoints, visualize_keypoints

keypoints, frame = get_keypoints(project_root, sequences)
visualize_keypoints(keypoints, frame)
```

Expected output:
- Side-by-side visualization showing original frame and detected pose
- Statistics about detected landmarks
- Sample keypoint coordinates

## Related Files

- `ambient/utils/eval_keypoints.py` - Fixed visualization function
- `ambient/pose/keypoints.py` - KeypointVisualizer class definition
- `ambient/pose/keypoint_data.py` - KeypointSet and Keypoint data structures
- `notebooks/explore3 - extract features.ipynb` - Notebook using this function

## Related Issues

This fix is part of a series of updates to migrate the codebase to use the new `KeypointSet` data structure:
- Task 6: Simplified `get_keypoints()` to DataFrame-only input
- Task 7: Fixed Jupyter compatibility in `keypoints.py`
- Task 8: Fixed `visualize_keypoints()` to work with KeypointSet objects
