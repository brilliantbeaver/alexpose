# Feature Extraction Quick Reference
## Developer Guide for AlexPose Gait Analysis

**Last Updated:** January 27, 2026  
**Status:** Production-Ready ✅

---

## Quick Start

### Basic Usage

```python
from ambient.analysis.gait_analyzer import EnhancedGaitAnalyzer
from ambient.classification.features import GaitFeatureVector

# Initialize analyzer
analyzer = EnhancedGaitAnalyzer(
    keypoint_format="COCO_17",
    fps=30.0,
    comprehensive_features=True
)

# Analyze pose sequence
results = analyzer.analyze_gait_sequence(pose_sequence)

# Extract feature vector
features = GaitFeatureVector.from_analysis_results(
    results,
    sample_id="video_001",
    condition_label="normal"
)

# Get feature array for ML
feature_array = features.to_array()  # 94 features
feature_names = GaitFeatureVector.get_feature_names()
```

---

## Configuration Options

### Confidence Threshold (Default: 0.3)

```python
analyzer = EnhancedGaitAnalyzer(
    feature_extraction_config={
        "confidence_threshold": 0.3  # Balanced for real-world videos
    }
)
```

**Thresholds:**
- `0.1-0.2`: Very permissive (noisy videos)
- `0.3`: **Recommended** (balanced)
- `0.4-0.5`: Strict (high-quality videos only)

### Feature Extraction Modes

```python
# Comprehensive (94+ features) - Default
features = GaitFeatureVector.from_analysis_results(
    results, 
    feature_extraction_mode="comprehensive"
)

# Standard (43 features)
features = GaitFeatureVector.from_analysis_results(
    results,
    feature_extraction_mode="standard"
)

# Legacy (15 features)
features = GaitFeatureVector.from_analysis_results(
    results,
    feature_extraction_mode="legacy"
)
```

### Feature Group Selection

```python
# Select specific feature groups
features = GaitFeatureVector.create_comprehensive_features(
    results,
    sample_id="video_001",
    condition_label="parkinsons",
    feature_groups=[
        "core_angles",
        "symmetry_indices",
        "stability",
        "temporal_phases"
    ]
)

# Get only selected features
feature_array = features.to_array(
    feature_groups=["core_angles", "symmetry_indices"]
)
```

---

## Feature Groups (94 Total)

### Core Features (43)

1. **Core Angles (15)**
   - Mean, range, asymmetry for hip/knee/ankle
   - Groups: `["core_angles"]`

2. **Spatiotemporal (4)**
   - Walking speed, cadence, stride length, step width
   - Groups: `["spatiotemporal"]`

3. **Temporal Phases (4)**
   - Stance%, swing%, double support%, ratio
   - Groups: `["temporal_phases"]`

4. **Symmetry Indices (6)**
   - Evidence-based SI for stride, stance, swing, joints
   - Groups: `["symmetry_indices"]`

5. **Kinematic (9)**
   - Velocity, acceleration, jerk statistics
   - Groups: `["kinematic"]`

6. **Variability (3)**
   - Stride time CV, step length CV, velocity CV
   - Groups: `["variability"]`

7. **Postural (2)**
   - Trunk lean, pelvic tilt
   - Groups: `["postural"]`

### Extended Features (51)

8. **Extended Angles (18)**
   - Std, max, min for each joint
   - Groups: `["extended_angles"]`

9. **Extended Temporal (12)**
   - Sequence info, cycles, durations, phases
   - Groups: `["temporal_extended"]`

10. **Stability (4)**
    - COM movement, stability index, postural sway
    - Groups: `["stability"]`

11. **Extended Stride (5)**
    - Step width stats, ankle distances
    - Groups: `["stride_extended"]`

12. **Extended Symmetry (10)**
    - Joint-specific + overall symmetry scores
    - Groups: `["symmetry_extended"]`

13. **Extended Kinematic (2)**
    - Pixel-based speed and stride length
    - Groups: `["kinematic_extended"]`

---

## Common Patterns

### Safe Feature Extraction

```python
def safe_extract(source_dict, key, default=0.0):
    """Extract value with NaN/None handling"""
    value = source_dict.get(key, default)
    return default if value is None or np.isnan(value) else float(value)

# Usage
cadence = safe_extract(timing_analysis, "cadence_steps_per_minute", 0.0)
```

### Confidence Filtering

```python
# Only use keypoints with sufficient confidence
if (frame[left_idx, 2] > confidence_threshold and 
    frame[right_idx, 2] > confidence_threshold):
    # Process keypoints
    angle = calculate_angle(frame[left_idx], frame[right_idx])
```

### Division by Zero Protection

```python
# Always add epsilon to denominators
symmetry_index = abs(left - right) / (left + right + 1e-8)
stability_index = std / (mean + 1e-8)
```

---

## Validation

### Feature Validation

```python
# Validate feature vector
is_valid, issues = features.validate(check_all_groups=True)

if not is_valid:
    print("Validation issues:")
    for issue in issues:
        print(f"  - {issue}")
```

### Feature Summary

```python
# Get human-readable summary
summary = features.get_feature_summary(include_all_groups=True)
print(summary)
```

---

## Troubleshooting

### Issue: All features are 0.0

**Causes:**
1. Confidence threshold too high
2. Sequence too short (<15 frames)
3. Missing keypoints

**Solutions:**
```python
# Lower confidence threshold
analyzer = EnhancedGaitAnalyzer(
    feature_extraction_config={"confidence_threshold": 0.2}
)

# Check sequence length
if len(pose_sequence) < 15:
    print("Sequence too short for reliable analysis")

# Check keypoint confidence
avg_confidence = np.mean([
    kp["confidence"] 
    for pose in pose_sequence 
    for kp in pose["keypoints"]
])
print(f"Average confidence: {avg_confidence:.2f}")
```

### Issue: Unrealistic feature values

**Causes:**
1. Incorrect keypoint format
2. Pixel-to-meter calibration mismatch
3. Corrupted pose data

**Solutions:**
```python
# Verify keypoint format
analyzer = EnhancedGaitAnalyzer(keypoint_format="COCO_17")  # or BODY_25, BLAZEPOSE_33

# Check for NaN/Inf
feature_array = features.to_array()
if np.any(np.isnan(feature_array)):
    print("NaN values detected")
if np.any(np.isinf(feature_array)):
    print("Inf values detected")

# Validate feature bounds
is_valid, issues = features.validate()
```

### Issue: Inconsistent results

**Causes:**
1. Variable video quality
2. Different pose estimation backends
3. Frame rate mismatch

**Solutions:**
```python
# Specify correct FPS
analyzer = EnhancedGaitAnalyzer(fps=actual_video_fps)

# Use consistent pose estimator
# Ensure same backend for all videos

# Check video quality
if avg_confidence < 0.3:
    print("Low quality video - results may be unreliable")
```

---

## Performance Tips

### Batch Processing

```python
# Process multiple videos efficiently
results_list = []
for video_path in video_paths:
    pose_sequence = extract_poses(video_path)
    results = analyzer.analyze_gait_sequence(pose_sequence)
    features = GaitFeatureVector.from_analysis_results(results)
    results_list.append(features.to_array())

# Convert to numpy array for ML
X = np.array(results_list)
```

### Memory Optimization

```python
# For large batches, extract only needed features
features = GaitFeatureVector.from_analysis_results(
    results,
    feature_extraction_mode="standard"  # 43 features instead of 94
)

# Or select specific groups
feature_array = features.to_array(
    feature_groups=["core_angles", "symmetry_indices"]
)
```

---

## Clinical Interpretation

### Symmetry Index (SI) Thresholds

```python
# Evidence-based thresholds (Clinical Biomechanics 2022)
if si < 12:
    classification = "Healthy/Symmetric"
elif si < 16:
    classification = "Borderline"
else:
    classification = "Pathological/Asymmetric"
```

### Stance/Swing Ratio

```python
# Normal gait: ~1.5 (60% stance, 40% swing)
if 1.3 <= stance_swing_ratio <= 1.7:
    status = "Normal"
elif stance_swing_ratio > 2.0:
    status = "Prolonged stance (cautious gait)"
else:
    status = "Abnormal"
```

### Cadence

```python
# Normal adult cadence: 100-120 steps/min
if 100 <= cadence <= 120:
    status = "Normal"
elif cadence < 100:
    status = "Slow (elderly, pathological)"
else:
    status = "Fast (running, hurried)"
```

---

## References

- **Full Investigation Report:** `notes/features/FEATURE_EXTRACTION_INVESTIGATION_REPORT.md`
- **Fix Summary:** `notes/features/ZERO_FEATURES_FIX_SUMMARY.md`
- **Documentation:** `docs/analysis/feature-extraction.md`

---

**For questions or issues, refer to the full investigation report.**
