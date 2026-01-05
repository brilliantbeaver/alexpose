# Feature Extraction Documentation

## Overview

The FeatureExtractor component provides comprehensive feature extraction capabilities for gait analysis, extracting 60+ features across multiple domains including kinematic, temporal, symmetry, and stability measures. These features are seamlessly integrated into the AlexPose web interface, providing real-time analysis and visualization capabilities.

## System Integration

### Backend Processing
The FeatureExtractor is integrated into the `EnhancedGaitAnalyzer` and accessed through the `PoseAnalysisServiceAPI`, providing:
- Real-time feature extraction from pose sequences
- Comprehensive caching for performance optimization
- Database persistence for historical analysis
- RESTful API endpoints for frontend integration

### Frontend Visualization
Features are displayed in the web interface through:
- **Interactive Dashboard**: Real-time feature visualization with tooltips
- **Clinical Assessment Cards**: Key metrics displayed with normal range indicators
- **Detailed Analysis Views**: Comprehensive feature breakdowns with explanations
- **Progress Indicators**: Real-time extraction progress during analysis

## FeatureExtractor Class

### Location
`ambient/analysis/feature_extractor.py`

### Initialization

```python
from ambient.analysis.feature_extractor import FeatureExtractor

extractor = FeatureExtractor(
    keypoint_format="COCO_17",  # Keypoint format
    fps=30.0,                   # Video frame rate
    smoothing_window=5          # Smoothing window size
)
```

### Supported Keypoint Formats

1. **COCO_17**: 17 keypoints (nose, eyes, ears, shoulders, elbows, wrists, hips, knees, ankles)
   - **Use Case**: MediaPipe pose estimation, lightweight processing
   - **UI Display**: Standard keypoint visualization overlay

2. **BODY_25**: 25 keypoints including detailed foot landmarks
   - **Use Case**: OpenPose estimation, comprehensive analysis
   - **UI Display**: Enhanced keypoint visualization with foot details

3. **BLAZEPOSE_33**: 33 keypoints with facial and hand landmarks
   - **Use Case**: High-detail analysis, research applications
   - **UI Display**: Full-body keypoint visualization with facial features

## Feature Categories and UI Integration

### 1. Kinematic Features

**Description**: Motion-based features including velocities, accelerations, and movement smoothness.

**Features Extracted**:
- `velocity_mean`: Average movement velocity across all keypoints
- `velocity_std`: Standard deviation of velocities
- `velocity_max`: Maximum velocity observed
- `velocity_min`: Minimum velocity observed
- `acceleration_mean`: Average acceleration magnitude
- `acceleration_std`: Standard deviation of accelerations
- `acceleration_max`: Maximum acceleration observed
- `jerk_mean`: Average jerk (rate of acceleration change)
- `jerk_std`: Standard deviation of jerk

**UI Integration**:
- **Movement Quality Card**: Displays velocity consistency and movement smoothness
- **Color Coding**: Green (good), Yellow (moderate), Red (poor) based on thresholds
- **Interactive Tooltips**: Detailed explanations of clinical significance
- **Real-time Updates**: Features update automatically when sequence changes

**Clinical Significance**:
- Velocity consistency indicates movement smoothness
- High acceleration variability may suggest motor control issues
- Jerk measures reflect movement coordination quality

**UI Display Example**:
```
Movement Quality Card:
┌─────────────────────────────────┐
│ Movement            [?]         │
│                                 │
│ Consistency: [Good    ]         │
│ Smoothness:  [Smooth  ]         │
│                                 │
│ Velocity CV: 0.25               │
└─────────────────────────────────┘
```

### 2. Joint Angle Features

**Description**: Angular measurements at major joints throughout the gait cycle.

**Features Extracted** (per joint):
- `{joint}_mean`: Average joint angle
- `{joint}_std`: Joint angle variability
- `{joint}_range`: Range of motion (max - min)
- `{joint}_max`: Maximum joint angle
- `{joint}_min`: Minimum joint angle

**Joints Analyzed**:
- Left/Right Knee: Hip-Knee-Ankle angle
- Left/Right Hip: Shoulder-Hip-Knee angle  
- Left/Right Ankle: Knee-Ankle-Vertical angle

**UI Integration**:
- **Joint Analysis Section**: Dedicated view for joint-specific metrics
- **Range of Motion Bars**: Visual indicators showing normal vs actual ranges
- **Bilateral Comparison**: Side-by-side left/right joint analysis
- **Asymmetry Highlighting**: Visual emphasis on asymmetric joints

**Clinical Significance**:
- Joint range of motion indicates flexibility and mobility
- Angle patterns reveal gait phase characteristics
- Asymmetric joint angles suggest compensation patterns

**UI Display Example**:
```
Joint Analysis:
┌─────────────────────────────────┐
│ Left Knee    │ Right Knee       │
│ Range: 65°   │ Range: 58°  [!]  │
│ Mean:  145°  │ Mean:  142°      │
│ ████████████ │ ██████████       │
│ Normal Range │ Restricted       │
└─────────────────────────────────┘
```

### 3. Temporal Features

**Description**: Time-based characteristics of the gait sequence.

**Features Extracted**:
- `sequence_length`: Number of frames in sequence
- `duration_seconds`: Total sequence duration
- `fps`: Frame rate used for analysis
- `dominant_frequency`: Primary movement frequency (Hz)
- `estimated_cadence`: Estimated steps per minute

**UI Integration**:
- **Cadence Card**: Large numeric display with normal range indicator
- **Temporal Metrics Panel**: Sequence timing information
- **Progress Indicators**: Real-time analysis progress
- **Performance Metrics**: Analysis speed and efficiency

**Clinical Significance**:
- Cadence is a key gait parameter (normal: 100-130 steps/min)
- Dominant frequency reflects gait rhythm consistency
- Temporal regularity indicates motor control stability

**UI Display Example**:
```
Cadence Card:
┌─────────────────────────────────┐
│ Cadence              [?]        │
│                                 │
│        96.0                     │
│     steps/minute                │
│                                 │
│ [    Slow    ]                  │
│ Normal: 100-130 spm             │
└─────────────────────────────────┘
```

### 4. Stride Features

**Description**: Spatial characteristics of walking pattern.

**Features Extracted**:
- `left_ankle_total_distance`: Total left ankle movement
- `right_ankle_total_distance`: Total right ankle movement
- `ankle_distance_asymmetry`: Difference between left/right movement
- `step_width_mean`: Average distance between ankles
- `step_width_std`: Step width variability
- `step_width_range`: Step width range

**UI Integration**:
- **Stride Analysis Panel**: Spatial gait characteristics
- **Asymmetry Indicators**: Visual highlighting of left/right differences
- **Step Width Visualization**: Graphical representation of step patterns
- **Distance Metrics**: Pixel and calibrated measurements

**Clinical Significance**:
- Step width indicates balance and stability
- Ankle movement asymmetry reveals gait imbalances
- Stride characteristics reflect walking efficiency

**UI Display Example**:
```
Stride Analysis:
┌─────────────────────────────────┐
│ Left Ankle Distance: 1,245 px   │
│ Right Ankle Distance: 1,180 px  │
│ Asymmetry: 0.15 [Mild]          │
│                                 │
│ Step Width: 45.2 ± 8.1 px      │
└─────────────────────────────────┘
```

### 5. Symmetry Features

**Description**: Left-right symmetry measures for bilateral comparison.

**Features Extracted** (per joint pair):
- `{joint}_symmetry_index`: Symmetry measure (0 = perfect symmetry)

**Joint Pairs Analyzed**:
- Shoulder, Elbow, Wrist, Hip, Knee, Ankle

**UI Integration**:
- **Symmetry Assessment Card**: Overall symmetry score and classification
- **Joint-Specific Symmetry**: Individual joint symmetry indicators
- **Most Asymmetric Joints**: Ranked list of asymmetric joints
- **Bilateral Visualization**: Side-by-side comparison views

**Clinical Significance**:
- Symmetry indices identify compensation patterns
- Asymmetry may indicate injury, weakness, or pathology
- Normal gait shows high bilateral symmetry

**UI Display Example**:
```
Symmetry Assessment:
┌─────────────────────────────────┐
│ Symmetry             [?]        │
│                                 │
│ [  Symmetric  ]                 │
│ Score: 0.007                    │
│                                 │
│ Most Asymmetric:                │
│ • Ankle: 0.08 [Low]             │
│ • Knee:  0.05 [Low]             │
└─────────────────────────────────┘
```

### 6. Stability Features

**Description**: Balance and postural control measures.

**Features Extracted**:
- `com_movement_mean`: Average center of mass movement
- `com_movement_std`: Center of mass movement variability
- `com_stability_index`: Stability measure (lower = more stable)
- `postural_sway_area`: Area of postural sway (if stationary)

**UI Integration**:
- **Stability Card**: Level assessment with visual indicators
- **Balance Metrics**: Center of mass movement visualization
- **Fall Risk Indicators**: Stability-based risk assessment
- **Postural Analysis**: Sway pattern visualization

**Clinical Significance**:
- Center of mass movement reflects balance control
- Stability indices indicate fall risk
- Postural sway measures static balance quality

**UI Display Example**:
```
Stability Card:
┌─────────────────────────────────┐
│ Stability            [?]        │
│                                 │
│ [  Moderate  ]                  │
│                                 │
│ Center of mass stability        │
│ Index: 0.35                     │
└─────────────────────────────────┘
```

## Usage Examples

### Basic Feature Extraction

```python
# Extract all features from pose sequence
features = extractor.extract_features(pose_sequence)

# Access specific feature categories
velocity_mean = features['velocity_mean']
left_knee_range = features['left_knee_range']
symmetry_score = features['knee_symmetry_index']
```

### Feature Analysis

```python
# Analyze movement quality
def assess_movement_quality(features):
    velocity_cv = features['velocity_std'] / features['velocity_mean']
    
    if velocity_cv < 0.3:
        return "smooth"
    elif velocity_cv < 0.6:
        return "moderate"
    else:
        return "jerky"

# Assess joint mobility
def assess_joint_mobility(features, joint):
    range_key = f"{joint}_range"
    if range_key in features:
        joint_range = features[range_key]
        
        # Joint-specific normal ranges (degrees)
        normal_ranges = {
            'knee': (60, 80),
            'hip': (40, 60),
            'ankle': (20, 40)
        }
        
        if joint in normal_ranges:
            min_normal, max_normal = normal_ranges[joint]
            if min_normal <= joint_range <= max_normal:
                return "normal"
            elif joint_range < min_normal:
                return "restricted"
            else:
                return "excessive"
    
    return "unknown"
```

### Custom Feature Selection

```python
# Extract only kinematic features
kinematic_features = {
    k: v for k, v in features.items() 
    if any(term in k for term in ['velocity', 'acceleration', 'jerk'])
}

# Extract symmetry features
symmetry_features = {
    k: v for k, v in features.items() 
    if 'symmetry_index' in k
}
```

## Feature Quality Assessment

### Data Quality Indicators

```python
def assess_feature_quality(features):
    quality_indicators = {}
    
    # Check for missing features
    expected_features = ['velocity_mean', 'acceleration_mean']
    missing_features = [f for f in expected_features if f not in features]
    quality_indicators['missing_features'] = missing_features
    
    # Check for invalid values
    invalid_features = []
    for key, value in features.items():
        if isinstance(value, (int, float)):
            if np.isnan(value) or np.isinf(value):
                invalid_features.append(key)
    quality_indicators['invalid_features'] = invalid_features
    
    # Overall quality score
    total_expected = len(expected_features)
    valid_features = total_expected - len(missing_features) - len(invalid_features)
    quality_indicators['quality_score'] = valid_features / total_expected
    
    return quality_indicators
```

### Confidence Scoring

```python
def calculate_feature_confidence(pose_sequence):
    """Calculate confidence based on pose estimation quality."""
    total_confidence = 0
    valid_frames = 0
    
    for pose in pose_sequence:
        keypoints = pose.get('keypoints', [])
        frame_confidence = sum(kp.get('confidence', 0) for kp in keypoints)
        
        if keypoints:
            frame_confidence /= len(keypoints)
            total_confidence += frame_confidence
            valid_frames += 1
    
    return total_confidence / valid_frames if valid_frames > 0 else 0.0
```

## Performance Optimization

### Memory Efficiency

```python
# Process large sequences in chunks
def extract_features_chunked(extractor, pose_sequence, chunk_size=1000):
    features_list = []
    
    for i in range(0, len(pose_sequence), chunk_size):
        chunk = pose_sequence[i:i + chunk_size]
        chunk_features = extractor.extract_features(chunk)
        features_list.append(chunk_features)
    
    # Combine features (implementation depends on requirements)
    return combine_feature_chunks(features_list)
```

### Parallel Processing

```python
from concurrent.futures import ProcessPoolExecutor

def extract_features_parallel(sequences):
    """Extract features from multiple sequences in parallel."""
    with ProcessPoolExecutor() as executor:
        futures = [
            executor.submit(extractor.extract_features, seq) 
            for seq in sequences
        ]
        
        results = [future.result() for future in futures]
    
    return results
```

## Error Handling

### Common Issues

1. **Empty Pose Sequence**
   ```python
   if not pose_sequence:
       return {"error": "Empty pose sequence"}
   ```

2. **Insufficient Keypoints**
   ```python
   if keypoints_array is None or keypoints_array.size == 0:
       return {"error": "No valid keypoints found"}
   ```

3. **Low Confidence Data**
   ```python
   avg_confidence = calculate_feature_confidence(pose_sequence)
   if avg_confidence < 0.3:
       features["warning"] = "Low confidence pose data"
   ```

### Graceful Degradation

```python
def robust_feature_extraction(extractor, pose_sequence):
    """Extract features with error handling."""
    try:
        features = extractor.extract_features(pose_sequence)
        
        # Validate features
        quality = assess_feature_quality(features)
        features.update(quality)
        
        return features
        
    except Exception as e:
        logger.error(f"Feature extraction failed: {e}")
        return {
            "error": str(e),
            "partial_features": {},
            "quality_score": 0.0
        }
```

## Integration Examples

### With Gait Analysis

```python
from ambient.analysis.gait_analyzer import EnhancedGaitAnalyzer

analyzer = EnhancedGaitAnalyzer()
results = analyzer.analyze_gait_sequence(pose_sequence)

# Features are automatically extracted and included
features = results['features']
```

### With Classification

```python
from ambient.classification.llm_classifier import LLMClassifier

# Extract features for classification
features = extractor.extract_features(pose_sequence)

# Use in classification
classifier = LLMClassifier()
classification = classifier.classify_gait({
    'features': features,
    'sequence_info': {...}
})
```

## See Also

- [Gait Analysis](gait-analysis.md) - Main gait analysis documentation
- [Temporal Analysis](temporal-analysis.md) - Gait cycle detection and timing
- [Symmetry Analysis](symmetry-analysis.md) - Detailed symmetry analysis
- [Configuration](../guides/configuration.md) - System configuration options