# Temporal Analysis Documentation

## Overview

The TemporalAnalyzer component provides comprehensive temporal analysis of gait patterns, including gait cycle detection, timing analysis, and rhythm assessment. It identifies key gait events such as heel strikes and toe-offs to segment gait cycles and extract temporal parameters. The analysis results are seamlessly integrated into the AlexPose web interface, providing real-time visualization and clinical interpretation.

## System Integration

### Backend Processing
The TemporalAnalyzer is integrated into the `EnhancedGaitAnalyzer` and accessed through the `PoseAnalysisServiceAPI`, providing:
- Real-time gait cycle detection from pose sequences
- Comprehensive timing parameter extraction
- Cadence calculation and rhythm analysis
- Database persistence for historical comparison

### Frontend Visualization
Temporal analysis results are displayed in the web interface through:
- **Cadence Card**: Large numeric display with normal range indicators
- **Gait Cycles Card**: Cycle count and average duration with quality metrics
- **Temporal Metrics Panel**: Detailed timing information and phase analysis
- **Interactive Tooltips**: Clinical explanations of temporal parameters

## TemporalAnalyzer Class

### Location
`ambient/analysis/temporal_analyzer.py`

### Initialization

```python
from ambient.analysis.temporal_analyzer import TemporalAnalyzer

analyzer = TemporalAnalyzer(
    fps=30.0,                      # Video frame rate
    min_cycle_duration=0.8,        # Minimum cycle duration (seconds)
    max_cycle_duration=2.5,        # Maximum cycle duration (seconds)
    detection_method="heel_strike", # heel_strike, toe_off, combined
    smoothing_window=5             # Smoothing window for signal processing
)
```

## Gait Cycle Detection

### Detection Methods

#### 1. Heel Strike Detection
**Method**: `heel_strike`
**Description**: Detects gait cycles based on heel contact events
**Algorithm**: 
- Tracks ankle vertical position and velocity
- Identifies local minima in ankle height
- Validates with velocity zero-crossings

```python
# Detect heel strikes
heel_strikes = analyzer.detect_heel_strikes(pose_sequence)

# Results format
heel_strikes = [
    {'frame': 15, 'foot': 'left', 'confidence': 0.85},
    {'frame': 32, 'foot': 'right', 'confidence': 0.92},
    # ...
]
```

#### 2. Toe-Off Detection
**Method**: `toe_off`
**Description**: Detects gait cycles based on toe-off events
**Algorithm**:
- Tracks toe/ankle forward velocity
- Identifies acceleration peaks during push-off
- Validates with ankle trajectory analysis

```python
# Detect toe-offs
toe_offs = analyzer.detect_toe_offs(pose_sequence)
```

#### 3. Combined Detection
**Method**: `combined`
**Description**: Uses both heel strikes and toe-offs for robust cycle detection
**Algorithm**:
- Combines both detection methods
- Cross-validates events for consistency
- Provides most accurate cycle segmentation

### Gait Cycle Segmentation

```python
# Detect complete gait cycles
cycles = analyzer.detect_gait_cycles(pose_sequence)

# Cycle structure
cycle = {
    'cycle_id': 0,
    'start_frame': 10,
    'end_frame': 35,
    'duration_frames': 25,
    'duration_seconds': 0.833,
    'foot': 'left',
    'events': {
        'heel_strike': 10,
        'toe_off': 22,
        'next_heel_strike': 35
    },
    'confidence': 0.88
}
```

## Temporal Parameters

### Basic Timing Metrics

```python
# Calculate temporal parameters
timing_params = analyzer.calculate_temporal_parameters(cycles)

# Parameters extracted
timing_params = {
    'cadence_steps_per_minute': 96.0,
    'cycle_duration_mean': 0.833,
    'cycle_duration_std': 0.045,
    'step_duration_mean': 0.416,
    'step_duration_std': 0.023,
    'stance_phase_duration': 0.500,
    'swing_phase_duration': 0.333,
    'double_support_duration': 0.167
}
```

### Advanced Timing Analysis

```python
# Detailed phase analysis
phase_analysis = analyzer.analyze_gait_phases(cycles)

# Phase breakdown
phase_analysis = {
    'stance_phase': {
        'duration_mean': 0.500,
        'duration_std': 0.032,
        'percentage_of_cycle': 60.0
    },
    'swing_phase': {
        'duration_mean': 0.333,
        'duration_std': 0.028,
        'percentage_of_cycle': 40.0
    },
    'double_support': {
        'duration_mean': 0.167,
        'duration_std': 0.015,
        'percentage_of_cycle': 20.0
    }
}
```

## Rhythm and Regularity Analysis

### Step Regularity

```python
# Calculate step regularity
regularity = analyzer.calculate_step_regularity(cycles)

regularity = {
    'step_regularity_cv': 0.12,        # Coefficient of variation
    'rhythm_consistency': 0.88,        # Rhythm consistency score
    'temporal_symmetry': 0.95,         # Left-right timing symmetry
    'variability_index': 0.08          # Overall variability measure
}
```

### Frequency Analysis

```python
# Analyze gait frequency characteristics
frequency_analysis = analyzer.analyze_gait_frequency(pose_sequence)

frequency_analysis = {
    'dominant_frequency': 1.6,         # Hz
    'frequency_bandwidth': 0.3,        # Hz
    'harmonic_ratio': 2.1,            # Stride/step frequency ratio
    'spectral_centroid': 1.8,         # Hz
    'frequency_stability': 0.92       # Stability measure
}
```

## Clinical Interpretation and UI Integration

### Normal Ranges and Visual Indicators

The web interface provides comprehensive visualization of temporal parameters with clinical context:

| Parameter | Normal Range | Units | UI Display |
|-----------|--------------|-------|------------|
| Cadence | 100-130 | steps/min | Large numeric with color-coded badge |
| Cycle Duration | 0.9-1.3 | seconds | Average display with variability |
| Stance Phase | 55-65 | % of cycle | Phase distribution chart |
| Swing Phase | 35-45 | % of cycle | Phase distribution chart |
| Double Support | 15-25 | % of cycle | Phase distribution chart |
| Step Regularity CV | < 0.1 | - | Regularity assessment badge |

### Cadence Assessment in UI

The cadence card provides the most prominent temporal display:

```
Cadence Card Interface:
┌─────────────────────────────────────────┐
│ 🏃 Cadence                      [Help?] │
│                                         │
│           96.0                          │
│        steps/minute                     │
│                                         │
│ [      Slow      ]                      │
│ Normal: 100-130 spm                     │
│                                         │
│ Clinical: May indicate weakness,        │
│ pain, or neurological issues            │
└─────────────────────────────────────────┘
```

#### Cadence Classification Logic
```python
def assess_cadence_for_ui(cadence_value):
    """Assess cadence for UI display with color coding."""
    if 100 <= cadence_value <= 130:
        return {
            'level': 'normal',
            'color': 'green',
            'description': 'Normal walking pace',
            'clinical_note': 'Within healthy adult range'
        }
    elif cadence_value < 100:
        return {
            'level': 'slow',
            'color': 'red',
            'description': 'Slow walking pace',
            'clinical_note': 'May indicate weakness, pain, or neurological issues'
        }
    else:  # cadence_value > 130
        return {
            'level': 'fast',
            'color': 'yellow',
            'description': 'Fast walking pace',
            'clinical_note': 'May indicate compensation or urgency'
        }
```

### Gait Cycles Visualization

The gait cycles card displays cycle detection results:

```
Gait Cycles Card Interface:
┌─────────────────────────────────────────┐
│ 📈 Gait Cycles                  [Help?] │
│                                         │
│            4                            │
│      detected cycles                    │
│                                         │
│ Avg: 1.12s (Range: 0.98-1.28s)        │
│ Regularity: Good (CV: 0.08)            │
│                                         │
│ Quality: ✅ Reliable detection          │
└─────────────────────────────────────────┘
```

#### Cycle Quality Assessment
```python
def assess_cycle_quality_for_ui(cycles):
    """Assess gait cycle quality for UI display."""
    if len(cycles) < 3:
        return {
            'quality': 'insufficient',
            'icon': '⚠️',
            'message': 'Too few cycles for reliable analysis',
            'color': 'red'
        }
    elif len(cycles) < 6:
        return {
            'quality': 'moderate',
            'icon': '⚡',
            'message': 'Moderate cycle detection reliability',
            'color': 'yellow'
        }
    else:
        return {
            'quality': 'good',
            'icon': '✅',
            'message': 'Reliable cycle detection',
            'color': 'green'
        }
```

### Interactive Tooltips for Temporal Metrics

Comprehensive tooltip system provides clinical context:

#### Cadence Tooltip
```
Cadence Tooltip:
┌─────────────────────────────────────────┐
│ 📊 Cadence (Steps per Minute)           │
│                                         │
│ The number of steps taken per minute,   │
│ indicating walking rhythm and pace.     │
│                                         │
│ Interpretation:                         │
│ • Normal: 100-130 steps/minute          │
│   (typical for healthy adults)          │
│ • Slow: Below 100 steps/minute          │
│   (may indicate weakness, pain, or      │
│   neurological issues)                  │
│ • Fast: Above 130 steps/minute          │
│   (may indicate compensation)           │
│                                         │
│ Clinical: Cadence changes can indicate  │
│ fatigue, pain, balance issues, or       │
│ neurological conditions.                │
└─────────────────────────────────────────┘
```

#### Gait Cycles Tooltip
```
Gait Cycles Tooltip:
┌─────────────────────────────────────────┐
│ 📈 Gait Cycles                          │
│                                         │
│ Complete walking cycles from heel       │
│ strike to heel strike of the same foot. │
│                                         │
│ Each cycle includes:                    │
│ • Stance phase (foot on ground)        │
│ • Swing phase (foot in air)            │
│                                         │
│ More cycles provide better analysis     │
│ reliability. Typical cycle duration     │
│ is 1.0-1.2 seconds for healthy adults. │
│                                         │
│ Regularity indicates motor control      │
│ stability and coordination quality.     │
└─────────────────────────────────────────┘
```

### Assessment Functions for UI Integration

```python
def assess_temporal_parameters_for_ui(timing_params):
    """Assess temporal parameters for UI display with detailed feedback."""
    assessment = {
        'cadence': assess_cadence_for_ui(timing_params['cadence_steps_per_minute']),
        'regularity': assess_regularity_for_ui(timing_params.get('step_regularity_cv', 0)),
        'phase_distribution': assess_phase_distribution_for_ui(timing_params),
        'overall_temporal_quality': 'good'  # Will be calculated based on individual assessments
    }
    
    # Calculate overall temporal quality
    quality_scores = []
    if assessment['cadence']['level'] == 'normal':
        quality_scores.append(1.0)
    elif assessment['cadence']['level'] in ['slow', 'fast']:
        quality_scores.append(0.5)
    
    if assessment['regularity']['level'] == 'good':
        quality_scores.append(1.0)
    elif assessment['regularity']['level'] == 'moderate':
        quality_scores.append(0.7)
    else:
        quality_scores.append(0.3)
    
    overall_score = np.mean(quality_scores)
    if overall_score >= 0.8:
        assessment['overall_temporal_quality'] = 'good'
    elif overall_score >= 0.6:
        assessment['overall_temporal_quality'] = 'moderate'
    else:
        assessment['overall_temporal_quality'] = 'poor'
    
    return assessment

def assess_regularity_for_ui(regularity_cv):
    """Assess step regularity for UI display."""
    if regularity_cv < 0.1:
        return {
            'level': 'good',
            'color': 'green',
            'description': 'Consistent step timing',
            'clinical_note': 'Good motor control stability'
        }
    elif regularity_cv < 0.2:
        return {
            'level': 'moderate',
            'color': 'yellow',
            'description': 'Some step variability',
            'clinical_note': 'Moderate motor control'
        }
    else:
        return {
            'level': 'poor',
            'color': 'red',
            'description': 'Irregular step timing',
            'clinical_note': 'Poor motor control, may indicate neurological issues'
        }
```

## Advanced Features

### Adaptive Cycle Detection

```python
# Adaptive parameters based on sequence characteristics
adaptive_params = analyzer.calculate_adaptive_parameters(pose_sequence)

# Use adaptive parameters for detection
cycles = analyzer.detect_gait_cycles(
    pose_sequence, 
    adaptive_params=adaptive_params
)
```

### Multi-Scale Analysis

```python
# Analyze at different time scales
multi_scale = analyzer.multi_scale_temporal_analysis(pose_sequence)

multi_scale = {
    'short_term': {  # Individual steps
        'variability': 0.08,
        'consistency': 0.92
    },
    'medium_term': {  # Multiple cycles
        'drift': 0.02,
        'stability': 0.95
    },
    'long_term': {  # Entire sequence
        'trend': 'stable',
        'fatigue_index': 0.05
    }
}
```

## Error Handling and Quality Control

### Data Quality Assessment

```python
def assess_temporal_data_quality(cycles):
    """Assess quality of temporal analysis results."""
    quality_metrics = {}
    
    # Check cycle count
    if len(cycles) < 3:
        quality_metrics['warning'] = 'Insufficient cycles for reliable analysis'
        quality_metrics['reliability'] = 'low'
    elif len(cycles) < 6:
        quality_metrics['reliability'] = 'moderate'
    else:
        quality_metrics['reliability'] = 'high'
    
    # Check cycle consistency
    durations = [c['duration_seconds'] for c in cycles]
    cv = np.std(durations) / np.mean(durations)
    
    if cv < 0.15:
        quality_metrics['consistency'] = 'good'
    elif cv < 0.3:
        quality_metrics['consistency'] = 'moderate'
    else:
        quality_metrics['consistency'] = 'poor'
    
    return quality_metrics
```

### Robust Detection

```python
def robust_cycle_detection(analyzer, pose_sequence):
    """Robust gait cycle detection with fallback methods."""
    try:
        # Primary method: heel strike
        cycles = analyzer.detect_gait_cycles(pose_sequence, method='heel_strike')
        
        if len(cycles) < 2:
            # Fallback: toe-off detection
            cycles = analyzer.detect_gait_cycles(pose_sequence, method='toe_off')
        
        if len(cycles) < 2:
            # Fallback: combined method with relaxed parameters
            analyzer.min_cycle_duration = 0.6
            analyzer.max_cycle_duration = 3.0
            cycles = analyzer.detect_gait_cycles(pose_sequence, method='combined')
        
        return cycles
        
    except Exception as e:
        logger.error(f"Cycle detection failed: {e}")
        return []
```

## Performance Optimization

### Efficient Processing

```python
# Process only relevant keypoints for temporal analysis
relevant_keypoints = ['left_ankle', 'right_ankle', 'left_knee', 'right_knee']
filtered_sequence = analyzer.filter_keypoints(pose_sequence, relevant_keypoints)

# Use optimized algorithms for large sequences
if len(pose_sequence) > 1000:
    cycles = analyzer.detect_gait_cycles_optimized(pose_sequence)
else:
    cycles = analyzer.detect_gait_cycles(pose_sequence)
```

### Memory Management

```python
# Process long sequences in segments
def process_long_sequence(analyzer, pose_sequence, segment_size=500):
    """Process long sequences in overlapping segments."""
    segments = []
    overlap = 50  # frames
    
    for i in range(0, len(pose_sequence), segment_size - overlap):
        segment = pose_sequence[i:i + segment_size]
        segment_cycles = analyzer.detect_gait_cycles(segment)
        segments.append(segment_cycles)
    
    # Merge overlapping segments
    return merge_cycle_segments(segments, overlap)
```

## Integration Examples

### With Gait Analysis

```python
from ambient.analysis.gait_analyzer import EnhancedGaitAnalyzer

# Temporal analysis is automatically included
analyzer = EnhancedGaitAnalyzer()
results = analyzer.analyze_gait_sequence(pose_sequence)

# Access temporal analysis results
timing_analysis = results['timing_analysis']
gait_cycles = results['gait_cycles']
```

### With Feature Extraction

```python
from ambient.analysis.feature_extractor import FeatureExtractor

# Extract temporal features
extractor = FeatureExtractor()
features = extractor.extract_features(pose_sequence)

# Temporal features are included
cadence = features['estimated_cadence']
dominant_freq = features['dominant_frequency']
```

## Visualization Support

### Cycle Visualization

```python
def visualize_gait_cycles(pose_sequence, cycles):
    """Create visualization of detected gait cycles."""
    import matplotlib.pyplot as plt
    
    # Plot ankle trajectories with cycle markers
    left_ankle_y = [pose['left_ankle']['y'] for pose in pose_sequence]
    
    plt.figure(figsize=(12, 6))
    plt.plot(left_ankle_y, label='Left Ankle Height')
    
    # Mark cycle boundaries
    for cycle in cycles:
        plt.axvline(cycle['start_frame'], color='red', linestyle='--', alpha=0.7)
        plt.axvline(cycle['end_frame'], color='blue', linestyle='--', alpha=0.7)
    
    plt.xlabel('Frame')
    plt.ylabel('Ankle Height')
    plt.title('Gait Cycles Detection')
    plt.legend()
    plt.show()
```

### Temporal Parameter Plots

```python
def plot_temporal_parameters(timing_params):
    """Create plots for temporal parameter analysis."""
    import matplotlib.pyplot as plt
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    
    # Cadence
    axes[0, 0].bar(['Cadence'], [timing_params['cadence_steps_per_minute']])
    axes[0, 0].set_ylabel('Steps/min')
    axes[0, 0].set_title('Cadence')
    
    # Phase durations
    phases = ['Stance', 'Swing', 'Double Support']
    durations = [
        timing_params['stance_phase_duration'],
        timing_params['swing_phase_duration'],
        timing_params['double_support_duration']
    ]
    axes[0, 1].pie(durations, labels=phases, autopct='%1.1f%%')
    axes[0, 1].set_title('Gait Phase Distribution')
    
    plt.tight_layout()
    plt.show()
```

## See Also

- [Gait Analysis](gait-analysis.md) - Main gait analysis documentation
- [Feature Extraction](feature-extraction.md) - Comprehensive feature extraction
- [Symmetry Analysis](symmetry-analysis.md) - Left-right symmetry assessment
- [Configuration](../guides/configuration.md) - System configuration options