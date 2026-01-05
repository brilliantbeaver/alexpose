# Symmetry Analysis Documentation

## Overview

The SymmetryAnalyzer component provides comprehensive left-right symmetry assessment for gait analysis. It evaluates bilateral movement patterns, identifies asymmetries, and provides clinical insights into gait balance and coordination. The analysis results are seamlessly integrated into the AlexPose web interface with interactive visualizations and detailed clinical interpretations.

## System Integration

### Backend Processing
The SymmetryAnalyzer is integrated into the `EnhancedGaitAnalyzer` and accessed through the `PoseAnalysisServiceAPI`, providing:
- Real-time bilateral symmetry assessment
- Joint-specific asymmetry detection
- Clinical classification of symmetry levels
- Database persistence for longitudinal tracking

### Frontend Visualization
Symmetry analysis results are prominently displayed in the web interface through:
- **Symmetry Assessment Card**: Overall symmetry score with color-coded classification
- **Asymmetry Details Panel**: Ranked list of most asymmetric joints
- **Bilateral Comparison Views**: Side-by-side movement pattern visualization
- **Interactive Tooltips**: Detailed explanations of symmetry metrics and clinical significance

## SymmetryAnalyzer Class

### Location
`ambient/analysis/symmetry_analyzer.py`

### Initialization

```python
from ambient.analysis.symmetry_analyzer import SymmetryAnalyzer

analyzer = SymmetryAnalyzer(
    keypoint_format="COCO_17",        # Keypoint format
    symmetry_threshold=0.1,           # Asymmetry threshold (10%)
    confidence_threshold=0.5,         # Minimum keypoint confidence
    smoothing_window=5                # Smoothing window size
)
```

## Symmetry Metrics

### Overall Symmetry Index

The overall symmetry index provides a single measure of gait symmetry across all bilateral features.

```python
# Calculate overall symmetry
symmetry_result = analyzer.calculate_overall_symmetry(pose_sequence)

symmetry_result = {
    'overall_symmetry_index': 0.007,      # Lower = more symmetric
    'symmetry_classification': 'symmetric', # symmetric, mildly_asymmetric, asymmetric
    'confidence': 0.92                     # Analysis confidence
}
```

### Joint-Specific Symmetry

Individual symmetry measures for each bilateral joint pair.

```python
# Calculate joint symmetries
joint_symmetries = analyzer.calculate_joint_symmetries(pose_sequence)

joint_symmetries = {
    'shoulder_symmetry_index': 0.05,
    'elbow_symmetry_index': 0.08,
    'wrist_symmetry_index': 0.12,
    'hip_symmetry_index': 0.03,
    'knee_symmetry_index': 0.05,
    'ankle_symmetry_index': 0.09
}
```

## Symmetry Analysis Methods

### 1. Amplitude Symmetry

Compares the range of motion between left and right sides.

```python
def calculate_amplitude_symmetry(left_trajectory, right_trajectory):
    """Calculate amplitude-based symmetry measure."""
    left_range = np.max(left_trajectory) - np.min(left_trajectory)
    right_range = np.max(right_trajectory) - np.min(right_trajectory)
    
    if left_range + right_range == 0:
        return 0.0
    
    symmetry_index = abs(left_range - right_range) / (left_range + right_range)
    return symmetry_index
```

### 2. Temporal Symmetry

Compares timing patterns between left and right sides.

```python
def calculate_temporal_symmetry(left_events, right_events):
    """Calculate temporal symmetry between bilateral events."""
    if len(left_events) != len(right_events):
        return 0.5  # High asymmetry for different event counts
    
    time_differences = []
    for left_time, right_time in zip(left_events, right_events):
        time_diff = abs(left_time - right_time)
        time_differences.append(time_diff)
    
    mean_diff = np.mean(time_differences)
    cycle_duration = np.mean([left_events[-1] - left_events[0], 
                             right_events[-1] - right_events[0]])
    
    temporal_symmetry = mean_diff / cycle_duration if cycle_duration > 0 else 1.0
    return min(temporal_symmetry, 1.0)
```

### 3. Cross-Correlation Symmetry

Uses cross-correlation to measure pattern similarity between sides.

```python
def calculate_correlation_symmetry(left_trajectory, right_trajectory):
    """Calculate symmetry using cross-correlation."""
    from scipy.signal import correlate
    
    # Normalize trajectories
    left_norm = (left_trajectory - np.mean(left_trajectory)) / np.std(left_trajectory)
    right_norm = (right_trajectory - np.mean(right_trajectory)) / np.std(right_trajectory)
    
    # Calculate cross-correlation
    correlation = correlate(left_norm, right_norm, mode='full')
    max_correlation = np.max(correlation) / len(left_trajectory)
    
    # Convert correlation to symmetry index (0 = perfect symmetry)
    symmetry_index = 1.0 - max_correlation
    return max(0.0, symmetry_index)
```

## Asymmetry Detection

### Identify Asymmetric Patterns

```python
# Detect specific asymmetries
asymmetries = analyzer.detect_asymmetries(pose_sequence)

asymmetries = [
    {
        'type': 'amplitude',
        'joint': 'knee',
        'severity': 'mild',
        'left_value': 65.2,
        'right_value': 58.7,
        'asymmetry_index': 0.11,
        'clinical_significance': 'May indicate weakness or compensation'
    },
    {
        'type': 'temporal',
        'feature': 'stance_duration',
        'severity': 'moderate',
        'left_value': 0.52,
        'right_value': 0.48,
        'asymmetry_index': 0.08,
        'clinical_significance': 'Possible limping pattern'
    }
]
```

### Asymmetry Classification

```python
def classify_asymmetry_severity(symmetry_index, thresholds=None):
    """Classify asymmetry severity based on symmetry index."""
    if thresholds is None:
        thresholds = {
            'symmetric': 0.05,
            'mildly_asymmetric': 0.15,
            'moderately_asymmetric': 0.30
        }
    
    if symmetry_index <= thresholds['symmetric']:
        return 'symmetric'
    elif symmetry_index <= thresholds['mildly_asymmetric']:
        return 'mildly_asymmetric'
    elif symmetry_index <= thresholds['moderately_asymmetric']:
        return 'moderately_asymmetric'
    else:
        return 'severely_asymmetric'
```

## Coordination Analysis

### Inter-limb Coordination

```python
# Analyze coordination between limbs
coordination = analyzer.analyze_coordination(pose_sequence)

coordination = {
    'phase_coordination_index': 0.92,     # 0-1, higher = better coordination
    'anti_phase_ratio': 0.48,            # Expected ~0.5 for normal gait
    'coordination_variability': 0.08,     # Lower = more consistent
    'coordination_classification': 'good' # good, moderate, poor
}
```

### Gait Symmetry Patterns

```python
def analyze_gait_symmetry_patterns(pose_sequence):
    """Analyze common gait symmetry patterns."""
    patterns = {}
    
    # Step length symmetry
    left_steps = extract_step_lengths(pose_sequence, 'left')
    right_steps = extract_step_lengths(pose_sequence, 'right')
    patterns['step_length_symmetry'] = calculate_amplitude_symmetry(left_steps, right_steps)
    
    # Swing time symmetry
    left_swing = extract_swing_times(pose_sequence, 'left')
    right_swing = extract_swing_times(pose_sequence, 'right')
    patterns['swing_time_symmetry'] = calculate_temporal_symmetry(left_swing, right_swing)
    
    # Stance time symmetry
    left_stance = extract_stance_times(pose_sequence, 'left')
    right_stance = extract_stance_times(pose_sequence, 'right')
    patterns['stance_time_symmetry'] = calculate_temporal_symmetry(left_stance, right_stance)
    
    return patterns
```

## Clinical Interpretation and UI Integration

### Symmetry Assessment Card

The symmetry assessment card provides the primary symmetry display in the web interface:

```
Symmetry Assessment Card:
┌─────────────────────────────────────────┐
│ ⚖️ Symmetry                     [Help?] │
│                                         │
│ [  Symmetric  ]                         │
│ Score: 0.007                            │
│                                         │
│ Confidence: High (0.92)                 │
│                                         │
│ Most Asymmetric:                        │
│ • Ankle: 0.08 [Low]                     │
│ • Knee:  0.05 [Low]                     │
│ • Hip:   0.03 [Low]                     │
└─────────────────────────────────────────┘
```

### Symmetry Classification and Visual Indicators

The UI provides color-coded symmetry classifications with clinical context:

| Classification | Index Range | UI Color | Clinical Significance |
|----------------|-------------|----------|----------------------|
| Symmetric | < 0.05 | Green | Normal bilateral function |
| Mildly Asymmetric | 0.05-0.15 | Yellow | Minor differences, monitor |
| Moderately Asymmetric | 0.15-0.30 | Orange | Noticeable differences, evaluate |
| Severely Asymmetric | > 0.30 | Red | Significant differences, intervention needed |

#### Symmetry Classification Logic for UI
```python
def classify_symmetry_for_ui(symmetry_index):
    """Classify symmetry for UI display with color coding and clinical context."""
    if symmetry_index < 0.05:
        return {
            'classification': 'symmetric',
            'color': 'green',
            'icon': '✅',
            'description': 'Normal bilateral movement',
            'clinical_note': 'Left and right sides show similar movement patterns',
            'recommendation': 'No immediate concerns'
        }
    elif symmetry_index < 0.15:
        return {
            'classification': 'mildly_asymmetric',
            'color': 'yellow',
            'icon': '⚠️',
            'description': 'Minor bilateral differences',
            'clinical_note': 'Small differences between sides, may be within normal variation',
            'recommendation': 'Monitor for progression'
        }
    elif symmetry_index < 0.30:
        return {
            'classification': 'moderately_asymmetric',
            'color': 'orange',
            'icon': '⚠️',
            'description': 'Noticeable bilateral differences',
            'clinical_note': 'Clear differences that may indicate compensation or pathology',
            'recommendation': 'Consider clinical evaluation'
        }
    else:
        return {
            'classification': 'severely_asymmetric',
            'color': 'red',
            'icon': '❌',
            'description': 'Significant bilateral differences',
            'clinical_note': 'Major asymmetry suggesting underlying condition or injury',
            'recommendation': 'Requires clinical evaluation and intervention'
        }
```

### Asymmetry Details Panel

The detailed asymmetry panel shows joint-specific asymmetries:

```
Asymmetry Details Panel:
┌─────────────────────────────────────────┐
│ 🔍 Asymmetry Details                    │
│                                         │
│ Joints showing the most asymmetry       │
│                                         │
│ ┌─────────────────────────────────────┐ │
│ │ Ankle        │ 0.08  │ [Low]  │ ⚠️  │ │
│ │ Knee         │ 0.05  │ [Low]  │ ✅  │ │
│ │ Hip          │ 0.03  │ [Low]  │ ✅  │ │
│ │ Shoulder     │ 0.02  │ [Low]  │ ✅  │ │
│ └─────────────────────────────────────┘ │
│                                         │
│ Clinical Notes:                         │
│ • Ankle asymmetry may indicate         │
│   compensation pattern                  │
│ • Overall symmetry within normal range │
└─────────────────────────────────────────┘
```

### Interactive Symmetry Tooltips

Comprehensive tooltip system provides clinical education:

#### Overall Symmetry Tooltip
```
Symmetry Tooltip:
┌─────────────────────────────────────────┐
│ ⚖️ Gait Symmetry                        │
│                                         │
│ Measures the similarity between left    │
│ and right limb movements during walking.│
│                                         │
│ Calculation:                            │
│ Compares joint angles, step lengths,   │
│ and timing between left and right sides.│
│                                         │
│ Interpretation:                         │
│ • Symmetric: Left and right sides show │
│   similar movement patterns (normal)    │
│ • Mildly Asymmetric: Minor differences │
│   between sides, may be normal          │
│ • Moderately Asymmetric: Noticeable    │
│   differences, may indicate pathology   │
│ • Severely Asymmetric: Significant     │
│   asymmetry suggesting condition        │
│                                         │
│ Clinical: Asymmetry may indicate        │
│ injury, weakness, or neurological       │
│ conditions affecting one side.          │
└─────────────────────────────────────────┘
```

### Normal Symmetry Ranges and UI Display

| Parameter | Normal Range | UI Threshold | Clinical Significance |
|-----------|--------------|--------------|----------------------|
| Overall Symmetry Index | < 0.05 | Green badge | Symmetric gait |
| Joint Symmetry Index | < 0.10 | Individual indicators | Normal bilateral function |
| Step Length Symmetry | < 0.08 | Movement analysis | Balanced step pattern |
| Swing Time Symmetry | < 0.06 | Temporal analysis | Coordinated limb timing |
| Stance Time Symmetry | < 0.06 | Phase analysis | Balanced weight bearing |

### Clinical Assessment Functions for UI

```python
def assess_gait_symmetry_for_ui(symmetry_results):
    """Provide comprehensive clinical assessment for UI display."""
    assessment = {
        'overall_classification': classify_symmetry_for_ui(
            symmetry_results['overall_symmetry_index']
        ),
        'joint_assessments': {},
        'clinical_concerns': [],
        'recommendations': [],
        'confidence_level': 'high'
    }
    
    # Assess individual joints
    joint_thresholds = {
        'ankle': 0.12,
        'knee': 0.10,
        'hip': 0.08,
        'shoulder': 0.15
    }
    
    for joint in ['ankle', 'knee', 'hip', 'shoulder']:
        joint_key = f'{joint}_symmetry_index'
        if joint_key in symmetry_results:
            joint_index = symmetry_results[joint_key]
            threshold = joint_thresholds[joint]
            
            if joint_index > threshold:
                assessment['joint_assessments'][joint] = {
                    'status': 'asymmetric',
                    'severity': 'moderate' if joint_index > threshold * 1.5 else 'mild',
                    'index': joint_index,
                    'color': 'red' if joint_index > threshold * 1.5 else 'yellow'
                }
                assessment['clinical_concerns'].append(
                    f'Asymmetric {joint} movement (index: {joint_index:.3f})'
                )
                assessment['recommendations'].append(
                    f'Evaluate {joint} function and mobility'
                )
            else:
                assessment['joint_assessments'][joint] = {
                    'status': 'symmetric',
                    'severity': 'normal',
                    'index': joint_index,
                    'color': 'green'
                }
    
    # Overall recommendations based on classification
    overall_class = assessment['overall_classification']['classification']
    if overall_class == 'severely_asymmetric':
        assessment['recommendations'].insert(0, 'Immediate clinical evaluation recommended')
    elif overall_class == 'moderately_asymmetric':
        assessment['recommendations'].insert(0, 'Consider gait training or physical therapy')
    elif overall_class == 'mildly_asymmetric':
        assessment['recommendations'].insert(0, 'Monitor for changes over time')
    
    return assessment

def generate_symmetry_summary_for_ui(symmetry_results):
    """Generate a concise summary for UI display."""
    overall_index = symmetry_results['overall_symmetry_index']
    classification = classify_symmetry_for_ui(overall_index)
    
    # Count asymmetric joints
    asymmetric_joints = []
    for key, value in symmetry_results.items():
        if key.endswith('_symmetry_index') and not key.startswith('overall'):
            if value > 0.1:  # Threshold for concern
                joint_name = key.replace('_symmetry_index', '')
                asymmetric_joints.append(joint_name)
    
    summary = {
        'primary_message': f"Gait is {classification['classification'].replace('_', ' ')}",
        'score_display': f"Symmetry score: {overall_index:.3f}",
        'confidence': symmetry_results.get('confidence', 0.0),
        'asymmetric_joint_count': len(asymmetric_joints),
        'most_asymmetric_joints': asymmetric_joints[:3],  # Top 3
        'color_theme': classification['color'],
        'icon': classification['icon']
    }
    
    return summary
```

### Bilateral Comparison Visualization

The UI provides side-by-side comparison views for detailed analysis:

```
Bilateral Comparison View:
┌─────────────────────────────────────────┐
│ 👥 Bilateral Movement Comparison        │
│                                         │
│ Left Side          │ Right Side         │
│ ┌─────────────────┐│┌─────────────────┐ │
│ │ Ankle ROM: 65°  ││ Ankle ROM: 58°  │ │
│ │ ████████████    ││ ██████████      │ │
│ │ Normal Range    ││ Restricted [!]  │ │
│ └─────────────────┘│└─────────────────┘ │
│                                         │
│ Asymmetry: 0.08 (Mild)                  │
│ Clinical: Right ankle shows restricted  │
│ range of motion compared to left        │
└─────────────────────────────────────────┘
```

## Advanced Analysis Features

### Dynamic Symmetry Analysis

```python
def analyze_dynamic_symmetry(pose_sequence, window_size=30):
    """Analyze how symmetry changes over time."""
    symmetry_timeline = []
    
    for i in range(0, len(pose_sequence) - window_size, window_size // 2):
        window = pose_sequence[i:i + window_size]
        window_symmetry = analyzer.calculate_overall_symmetry(window)
        
        symmetry_timeline.append({
            'start_frame': i,
            'end_frame': i + window_size,
            'symmetry_index': window_symmetry['overall_symmetry_index'],
            'timestamp': i / fps
        })
    
    return symmetry_timeline
```

### Fatigue-Related Asymmetry

```python
def detect_fatigue_asymmetry(pose_sequence):
    """Detect asymmetry changes that may indicate fatigue."""
    # Divide sequence into early and late portions
    mid_point = len(pose_sequence) // 2
    early_sequence = pose_sequence[:mid_point]
    late_sequence = pose_sequence[mid_point:]
    
    early_symmetry = analyzer.calculate_overall_symmetry(early_sequence)
    late_symmetry = analyzer.calculate_overall_symmetry(late_sequence)
    
    fatigue_effect = {
        'early_symmetry': early_symmetry['overall_symmetry_index'],
        'late_symmetry': late_symmetry['overall_symmetry_index'],
        'symmetry_change': late_symmetry['overall_symmetry_index'] - early_symmetry['overall_symmetry_index'],
        'fatigue_detected': late_symmetry['overall_symmetry_index'] > early_symmetry['overall_symmetry_index'] + 0.02
    }
    
    return fatigue_effect
```

## Error Handling and Quality Control

### Data Quality Assessment

```python
def assess_symmetry_data_quality(pose_sequence):
    """Assess data quality for symmetry analysis."""
    quality_metrics = {}
    
    # Check bilateral keypoint availability
    bilateral_pairs = [
        ('left_shoulder', 'right_shoulder'),
        ('left_elbow', 'right_elbow'),
        ('left_wrist', 'right_wrist'),
        ('left_hip', 'right_hip'),
        ('left_knee', 'right_knee'),
        ('left_ankle', 'right_ankle')
    ]
    
    missing_pairs = []
    for left_kp, right_kp in bilateral_pairs:
        left_available = sum(1 for pose in pose_sequence if left_kp in pose)
        right_available = sum(1 for pose in pose_sequence if right_kp in pose)
        
        if left_available < len(pose_sequence) * 0.8 or right_available < len(pose_sequence) * 0.8:
            missing_pairs.append((left_kp, right_kp))
    
    quality_metrics['missing_bilateral_pairs'] = missing_pairs
    quality_metrics['bilateral_completeness'] = 1.0 - len(missing_pairs) / len(bilateral_pairs)
    
    # Assess confidence levels
    confidence_scores = []
    for pose in pose_sequence:
        for left_kp, right_kp in bilateral_pairs:
            if left_kp in pose and right_kp in pose:
                left_conf = pose[left_kp].get('confidence', 0)
                right_conf = pose[right_kp].get('confidence', 0)
                confidence_scores.extend([left_conf, right_conf])
    
    quality_metrics['mean_confidence'] = np.mean(confidence_scores) if confidence_scores else 0
    quality_metrics['low_confidence_frames'] = sum(1 for conf in confidence_scores if conf < 0.5)
    
    return quality_metrics
```

### Robust Symmetry Calculation

```python
def robust_symmetry_analysis(analyzer, pose_sequence):
    """Perform robust symmetry analysis with error handling."""
    try:
        # Assess data quality first
        quality = assess_symmetry_data_quality(pose_sequence)
        
        if quality['bilateral_completeness'] < 0.6:
            return {
                'error': 'Insufficient bilateral data for symmetry analysis',
                'quality_metrics': quality
            }
        
        # Perform symmetry analysis
        symmetry_results = analyzer.calculate_overall_symmetry(pose_sequence)
        joint_symmetries = analyzer.calculate_joint_symmetries(pose_sequence)
        
        # Combine results
        results = {
            **symmetry_results,
            **joint_symmetries,
            'quality_metrics': quality
        }
        
        # Add confidence adjustment based on data quality
        confidence_adjustment = quality['bilateral_completeness'] * quality['mean_confidence']
        results['adjusted_confidence'] = results['confidence'] * confidence_adjustment
        
        return results
        
    except Exception as e:
        logger.error(f"Symmetry analysis failed: {e}")
        return {
            'error': str(e),
            'overall_symmetry_index': None,
            'confidence': 0.0
        }
```

## Visualization Support

### Symmetry Visualization

```python
def visualize_symmetry_analysis(pose_sequence, symmetry_results):
    """Create visualizations for symmetry analysis."""
    import matplotlib.pyplot as plt
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # 1. Joint symmetry comparison
    joints = ['shoulder', 'elbow', 'wrist', 'hip', 'knee', 'ankle']
    symmetry_indices = [symmetry_results[f'{joint}_symmetry_index'] for joint in joints]
    
    axes[0, 0].bar(joints, symmetry_indices)
    axes[0, 0].axhline(y=0.1, color='red', linestyle='--', label='Asymmetry Threshold')
    axes[0, 0].set_ylabel('Symmetry Index')
    axes[0, 0].set_title('Joint Symmetry Indices')
    axes[0, 0].legend()
    
    # 2. Left vs Right trajectory comparison (example: knee angles)
    left_knee_angles = extract_joint_angles(pose_sequence, 'left_knee')
    right_knee_angles = extract_joint_angles(pose_sequence, 'right_knee')
    
    axes[0, 1].plot(left_knee_angles, label='Left Knee', alpha=0.7)
    axes[0, 1].plot(right_knee_angles, label='Right Knee', alpha=0.7)
    axes[0, 1].set_xlabel('Frame')
    axes[0, 1].set_ylabel('Knee Angle (degrees)')
    axes[0, 1].set_title('Bilateral Knee Angle Comparison')
    axes[0, 1].legend()
    
    # 3. Symmetry over time
    dynamic_symmetry = analyze_dynamic_symmetry(pose_sequence)
    timestamps = [point['timestamp'] for point in dynamic_symmetry]
    symmetry_values = [point['symmetry_index'] for point in dynamic_symmetry]
    
    axes[1, 0].plot(timestamps, symmetry_values, marker='o')
    axes[1, 0].axhline(y=0.05, color='green', linestyle='--', label='Symmetric')
    axes[1, 0].axhline(y=0.15, color='red', linestyle='--', label='Asymmetric')
    axes[1, 0].set_xlabel('Time (seconds)')
    axes[1, 0].set_ylabel('Symmetry Index')
    axes[1, 0].set_title('Symmetry Over Time')
    axes[1, 0].legend()
    
    # 4. Overall symmetry assessment
    overall_index = symmetry_results['overall_symmetry_index']
    classification = symmetry_results['symmetry_classification']
    
    colors = {'symmetric': 'green', 'mildly_asymmetric': 'yellow', 'asymmetric': 'red'}
    color = colors.get(classification, 'gray')
    
    axes[1, 1].bar(['Overall Symmetry'], [overall_index], color=color, alpha=0.7)
    axes[1, 1].set_ylabel('Symmetry Index')
    axes[1, 1].set_title(f'Overall Assessment: {classification.replace("_", " ").title()}')
    axes[1, 1].set_ylim(0, 0.3)
    
    plt.tight_layout()
    plt.show()
```

## Integration Examples

### With Gait Analysis

```python
from ambient.analysis.gait_analyzer import EnhancedGaitAnalyzer

# Symmetry analysis is automatically included
analyzer = EnhancedGaitAnalyzer()
results = analyzer.analyze_gait_sequence(pose_sequence)

# Access symmetry analysis results
symmetry_analysis = results['symmetry_analysis']
overall_symmetry = symmetry_analysis['overall_symmetry_index']
```

### With Feature Extraction

```python
from ambient.analysis.feature_extractor import FeatureExtractor

# Extract symmetry features
extractor = FeatureExtractor()
features = extractor.extract_features(pose_sequence)

# Symmetry features are included
knee_symmetry = features['knee_symmetry_index']
ankle_asymmetry = features['ankle_distance_asymmetry']
```

## See Also

- [Gait Analysis](gait-analysis.md) - Main gait analysis documentation
- [Feature Extraction](feature-extraction.md) - Comprehensive feature extraction
- [Temporal Analysis](temporal-analysis.md) - Gait cycle detection and timing
- [Configuration](../guides/configuration.md) - System configuration options