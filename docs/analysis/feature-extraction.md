# Feature Extraction Documentation

## Overview

The FeatureExtractor component provides comprehensive feature extraction capabilities for gait analysis, extracting **82 features** across multiple domains including kinematic, temporal, symmetry, and stability measures. These features are seamlessly integrated into the AlexPose web interface, providing real-time analysis and visualization capabilities.

**Recent Enhancements (January 2026)**:
- Optimized from 94 to **82 features** by removing redundant max/min values (retained std for unique variability information)
- Added configurable **confidence threshold parameter** (default: 0.3) for improved robustness with real-world data
- Reduced temporal analysis thresholds for better short-sequence support
- Enhanced symmetry analysis with lower confidence requirements
- Improved feature extraction reliability for videos with varying pose confidence

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
    keypoint_format="COCO_17",           # Keypoint format (COCO_17, BODY_25, BLAZEPOSE_33)
    fps=30.0,                            # Video frame rate
    smoothing_window=5,                  # Smoothing window size for calculations
    extract_extended_features=True,      # Extract comprehensive feature set (82 features)
    include_joint_statistics=True,       # Include joint angle std/max/min statistics
    include_stability_features=True,     # Include balance and stability features
    include_advanced_temporal=True,      # Include advanced temporal analysis
    confidence_threshold=0.3             # Minimum confidence for keypoint validity (NEW)
)
```

**Key Parameters**:

- **`confidence_threshold`** (float, default: 0.3): Minimum confidence score for considering a keypoint valid
  - **Purpose**: Filters out low-quality keypoint detections while retaining useful data
  - **Range**: 0.0 to 1.0 (0.0 = accept all, 1.0 = only perfect confidence)
  - **Recommended Values**:
    - `0.3`: Default - good balance for real-world videos with varying quality
    - `0.5`: Higher quality threshold for research-grade data
    - `0.1`: Very permissive for low-quality or challenging videos
  - **Impact**: Lower thresholds extract more features from imperfect data; higher thresholds ensure higher quality but may result in more zero-value features

- **`extract_extended_features`** (bool, default: True): Enable comprehensive 82-feature extraction
  - When `True`: Extracts all available features including extended joint statistics (std only), advanced temporal features, and comprehensive symmetry analysis
  - When `False`: Extracts only core features for faster processing

- **`include_joint_statistics`** (bool, default: True): Include standard deviation, max, and min for each joint angle
  - Adds 18 additional features (3 per joint × 6 joints)
  - Essential for understanding joint angle variability and range of motion

- **`include_stability_features`** (bool, default: True): Include balance and stability analysis
  - Adds center of mass movement, stability indices, and postural sway metrics
  - Critical for fall risk assessment and balance evaluation

- **`include_advanced_temporal`** (bool, default: True): Include advanced temporal analysis
  - Adds frequency analysis, estimated cadence, and enhanced spatiotemporal parameters
  - Provides deeper insights into gait rhythm and timing patterns

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

### 1. Kinematic Features (9 features)

**Description**: Motion-based features including velocities, accelerations, and movement smoothness. These features capture the quality and dynamics of movement, providing insights into motor control, coordination, and neurological function.

**Evidence Base**: Journal of Biomechanics (2024) - Kinematic analysis distinguishes normal from pathological gait patterns. Velocity consistency and jerk measures are validated indicators of movement quality and coordination.

**Features Extracted**:

#### Velocity Features (4 features)
- `velocity_mean`: Average movement velocity across all keypoints (pixels/s)
  - **Normal Range**: 50-150 pixels/s
  - **Clinical Significance**: Overall movement speed and activity level
  
- `velocity_std`: Standard deviation of velocities (pixels/s)
  - **Normal Range**: 20-60 pixels/s
  - **Clinical Significance**: Movement consistency and smoothness
  
- `velocity_max`: Maximum velocity observed (pixels/s)
  - **Normal Range**: 150-300 pixels/s
  - **Clinical Significance**: Peak movement capacity during swing phase
  
- `velocity_min`: Minimum velocity observed (pixels/s)
  - **Normal Range**: 0-20 pixels/s
  - **Clinical Significance**: Baseline movement during stance phase

#### Acceleration Features (4 features)
- `acceleration_mean`: Average acceleration magnitude (pixels/s²)
  - **Normal Range**: 10-50 pixels/s²
  - **Clinical Significance**: Force generation and movement transitions
  
- `acceleration_std`: Standard deviation of accelerations (pixels/s²)
  - **Normal Range**: 5-25 pixels/s²
  - **Clinical Significance**: Consistency of force application
  
- `acceleration_max`: Maximum acceleration observed (pixels/s²)
  - **Normal Range**: 50-150 pixels/s²
  - **Clinical Significance**: Peak force generation capacity

#### Jerk Features (2 features)
- `jerk_mean`: Average jerk - rate of acceleration change (pixels/s³)
  - **Normal Range**: 5-30 pixels/s³
  - **Clinical Significance**: Movement smoothness and coordination quality
  - **Interpretation**: Higher values indicate jerky, poorly coordinated movement
  
- `jerk_std`: Standard deviation of jerk (pixels/s³)
  - **Normal Range**: 2-15 pixels/s³
  - **Clinical Significance**: Consistency of movement coordination

**UI Integration**:
- **Movement Quality Card**: Displays velocity consistency and movement smoothness
  - Velocity CV (Coefficient of Variation) = velocity_std / velocity_mean
  - Good: CV < 0.3, Moderate: CV 0.3-0.6, Poor: CV > 0.6
- **Color Coding**: Green (good), Yellow (moderate), Red (poor) based on thresholds
- **Interactive Tooltips**: Detailed explanations of clinical significance
- **Real-time Updates**: Features update automatically when sequence changes
- **Smoothness Indicator**: Based on jerk_mean values
  - Smooth: jerk_mean < 20, Moderate: 20-40, Jerky: > 40

**Clinical Applications**:

**Parkinson's Disease**:
- Reduced velocity_mean (bradykinesia)
- Increased velocity_std (movement variability)
- Elevated jerk_mean (reduced smoothness)

**Stroke Recovery**:
- Asymmetric velocity patterns between sides
- High velocity_cv indicating poor motor control
- Elevated jerk on affected side

**Cerebellar Ataxia**:
- High velocity_std (uncoordinated movement)
- Elevated velocity_cv (inconsistent control)
- Very high jerk_mean (>40) indicating poor coordination

**Movement Quality Assessment**:
```python
def assess_movement_quality(features):
    velocity_cv = features['velocity_std'] / features['velocity_mean']
    jerk_threshold = 30
    
    if velocity_cv < 0.3 and features['jerk_mean'] < jerk_threshold:
        return "excellent"
    elif velocity_cv < 0.5 and features['jerk_mean'] < jerk_threshold * 1.5:
        return "good"
    elif velocity_cv < 0.7 and features['jerk_mean'] < jerk_threshold * 2:
        return "moderate"
    else:
        return "poor"
```

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
│ Jerk Mean:   18.3 pixels/s³     │
│                                 │
│ Interpretation:                 │
│ • Smooth, coordinated movement  │
│ • Good motor control            │
└─────────────────────────────────┘
```

**See Also**: [Gait Analysis Tutorial - Kinematic Features](../guides/gait-analysis-tutorial.md#36-kinematic-features-9-features---new) for comprehensive details on calculation methods, clinical interpretation, and research applications.

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
from ambient.classification.llm_classifier import LLMClassifier, LLMClassifierConfig

config = LLMClassifierConfig(model_name="gpt-4o-mini")
classifier = LLMClassifier(config)
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