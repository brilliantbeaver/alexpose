# Investigation Report: Empty Joint Angle Frames

## Problem Statement
Some CSV files result in `joint_angles.frames` being empty, causing this loop to not run:
```python
for joint_name in joint_angles.frames[0].angles.keys():
    avg_joint_angle = joint_angles.get_statistics(joint_name=joint_name)["mean"]
    print(f"\t<joint angle> of {joint_name}: {avg_joint_angle}")
```

## Root Causes

### 1. Frame Extraction Failures (Primary Cause)
**Location**: `ambient/pose/keypoint_extractor.py:617-632`

The `extract_from_sequence()` method returns an empty list when ANY frame extraction fails:
```python
keypoints = self.extract_from_video_frame(video_path, actual_frame_num, model_path)

if keypoints is None:
    logger.error(f"Failed to extract keypoints from frame {actual_frame_num}")
    return []  # ← RETURNS EMPTY LIST, ABANDONING ALL FRAMES
```

**Why frames fail to extract**:
- FFmpeg hangs on certain frame numbers (especially high frame numbers like 1593+)
- Video codec issues
- Corrupted frames in video
- OpenCV fallback also fails
- No timeout mechanism to recover

### 2. MediaPipe Detection Failures
**Location**: `ambient/pose/keypoint_extractor.py:extract_from_image()`

Even when frames extract successfully, MediaPipe may not detect any person:
```python
if not detection_result.pose_landmarks:
    # Return empty result with CUSTOM format (no keypoints detected)
    return KeypointSet(keypoints=[], format=KeypointFormat.CUSTOM, ...)
```

This results in empty keypoint sets that can't produce joint angles.

### 3. Low Confidence Filtering
**Location**: `ambient/pose/joint_angles.py:_calculate_from_keypoint_set()`

Joint angles are only added if confidence meets threshold:
```python
if combined_conf >= self.confidence_threshold and not np.isnan(angle_deg):
    frame_angles.angles[joint_name] = JointAngle(...)
```

If all keypoints have low confidence, no angles are calculated.

### 4. Missing Error Handling in Processing Script
**Location**: `experiments/exp2/src/process3_per_set_conditions.py`

The script doesn't validate results before accessing:
```python
joint_angles = get_joint_angles(...)

# NO CHECK HERE - assumes frames exist!
print(f"Average joint angles for {len(joint_angles.frames)} frames:")
for i in range(len(joint_angles.frames[0].angles)):  # ← CRASHES if frames empty
    ...
```

## Evidence

### Test Results:
- ✅ **Working**: `cljawh4c7000m3n6lkz9es9bl.csv` (145 frames, 6 angles per frame)
- ✅ **Working**: `cljawf4e4000h3n6lri4oz9uy.csv` (188 frames, 6 angles per frame)
- ❌ **Hanging**: `cljr5iki0000j3n6lwi8z5nh6.csv` (137 frames, hangs during extraction)

### Problematic Video:
- Video ID: `8mTHlAIdea0`
- Frame range: 1593-1729
- Issue: Frame extraction hangs/times out at high frame numbers

## Recommended Solutions

### Solution 1: Add Validation and Error Handling (Quick Fix)
```python
joint_angles = get_joint_angles(
    keypoints_array=keypoints_array,
    keypoint_format="BLAZEPOSE_33",
    fps=30.0,
    confidence_threshold=0.3
)

# VALIDATE BEFORE ACCESSING
if len(joint_angles.frames) == 0:
    print(f"WARNING: No joint angle frames computed for sequence")
    continue

if len(joint_angles.frames[0].angles) == 0:
    print(f"WARNING: First frame has no joint angles")
    continue

# NOW SAFE TO ITERATE
print(f"Average joint angles for {len(joint_angles.frames)} frames:")
for joint_name in joint_angles.frames[0].angles.keys():
    avg_joint_angle = joint_angles.get_statistics(joint_name=joint_name)["mean"]
    print(f"\t<joint angle> of {joint_name}: {avg_joint_angle}")
```

### Solution 2: Improve Frame Extraction Resilience (Better Fix)
Modify `extract_from_sequence()` to continue on individual frame failures:
```python
def extract_from_sequence(self, sequence_data, video_base_path, model_path=None, verbose=False):
    keypoints_array = []
    failed_frames = []
    
    for fnum in range(num_frames):
        try:
            keypoints = self.extract_from_video_frame(...)
            
            if keypoints is None:
                logger.warning(f"Frame {actual_frame_num} failed, skipping")
                failed_frames.append(actual_frame_num)
                continue  # ← CONTINUE instead of returning []
            
            keypoints_array.append(keypoints)
            
        except Exception as e:
            logger.error(f"Frame {actual_frame_num} error: {e}")
            failed_frames.append(actual_frame_num)
            continue  # ← CONTINUE instead of crashing
    
    if failed_frames:
        logger.warning(f"Failed to extract {len(failed_frames)} frames: {failed_frames}")
    
    return keypoints_array  # Return what we got, even if incomplete
```

### Solution 3: Add Timeout to Frame Extraction (Best Fix)
Add timeout mechanism to prevent hangs:
```python
def extract_from_video_frame(self, video_path, frame_number, model_path=None, timeout=30):
    """Extract with timeout to prevent hangs"""
    import signal
    
    def timeout_handler(signum, frame):
        raise TimeoutError(f"Frame extraction timed out after {timeout}s")
    
    signal.signal(signal.SIGALRM, timeout_handler)
    signal.alarm(timeout)
    
    try:
        # Existing extraction logic
        result = self._do_extraction(video_path, frame_number, model_path)
        signal.alarm(0)  # Cancel alarm
        return result
    except TimeoutError:
        signal.alarm(0)
        logger.error(f"Frame {frame_number} extraction timed out")
        return None
```

### Solution 4: Batch Processing with Skip-on-Error
Process entire video at once and skip problematic frames:
```python
# Already partially implemented in gavd_processor.py
# But needs better error handling for individual frame failures
```

## Immediate Action Items

1. **Add validation** to `process3_per_set_conditions.py` before accessing `joint_angles.frames[0]`
2. **Log which CSVs fail** and why (video missing, no keypoints, no angles)
3. **Add timeout** to frame extraction to prevent infinite hangs
4. **Change extraction strategy** from "fail-all-on-one-error" to "skip-bad-frames"
5. **Lower confidence threshold** or handle empty angle sets gracefully

## Files to Modify

1. `experiments/exp2/src/process3_per_set_conditions.py` - Add validation
2. `ambient/pose/keypoint_extractor.py` - Add timeout and resilience
3. `ambient/pose/joint_angles.py` - Handle empty keypoint sets better
4. `notebooks/tutorial2 - train classifier.ipynb` - Add validation

## Testing Recommendations

1. Test with known problematic CSV: `cljr5iki0000j3n6lwi8z5nh6.csv`
2. Test with videos that have high frame numbers (1500+)
3. Test with videos where MediaPipe fails to detect poses
4. Add unit tests for empty result handling
