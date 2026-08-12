# Kinematic Features Documentation

## Overview

Kinematic features provide insights into the quality and characteristics of movement during gait. These features capture velocity, acceleration, and jerk (rate of change of acceleration) across all body keypoints, offering a comprehensive view of movement dynamics that complement traditional joint angle and spatiotemporal measurements.

## Clinical Significance

Kinematic features are essential for understanding:
- **Movement Quality**: Smoothness and coordination of motion
- **Motor Control**: Neurological function and movement planning
- **Pathology Detection**: Abnormal movement patterns indicative of disease
- **Treatment Monitoring**: Objective measures of rehabilitation progress

## Evidence Base

Kinematic analysis is supported by extensive research:

**Journal of Biomechanics (2024)** - "Kinematic Analysis of Gait Patterns"
- Velocity and acceleration patterns distinguish normal from pathological gait
- Jerk measures reflect movement coordination quality
- Kinematic variability indicates motor control stability

**Frontiers in Human Neuroscience (2025)** - "Movement Quality Assessment"
- High jerk values correlate with poor motor coordination
- Velocity consistency predicts functional outcomes
- Acceleration patterns reveal compensation strategies

## Feature Categories

### 1. Velocity Features (4 features)

Velocity measures the rate of position change for body keypoints over time.

#### Features Extracted

**velocity_mean** (pixels/second)
- **Description**: Average velocity across all keypoints throughout the sequence
- **Normal Range**: 50-150 pixels/s (depends on video resolution and distance)
- **Clinical Significance**: Overall movement speed and activity level
- **Interpretation**:
  - Low (<50): Slow, cautious movement
  - Normal (50-150): Typical walking speed
  - High (>150): Fast, possibly compensatory movement

**velocity_std** (pixels/second)
- **Description**: Standard deviation of velocity values
- **Normal Range**: 20-60 pixels/s
- **Clinical Significance**: Movement consistency and smoothness
- **Interpretation**:
  - Low (<20): Very consistent, possibly rigid movement
  - Normal (20-60): Natural movement variability
  - High (>60): Jerky, inconsistent movement

**velocity_max** (pixels/second)
- **Description**: Maximum velocity observed during the sequence
- **Normal Range**: 150-300 pixels/s
- **Clinical Significance**: Peak movement speed, often during swing phase
- **Interpretation**:
  - Low (<150): Reduced dynamic range
  - Normal (150-300): Adequate movement capacity
  - High (>300): Excessive or compensatory movements

**velocity_min** (pixels/second)
- **Description**: Minimum velocity observed during the sequence
- **Normal Range**: 0-20 pixels/s
- **Clinical Significance**: Baseline movement, stance phase stability
- **Interpretation**:
  - Near zero: Good stance stability
  - Elevated (>20): Continuous movement, possible instability

#### Velocity Coefficient of Variation (CV)

**Calculation**: `velocity_cv = velocity_std / velocity_mean`

**Interpretation**:
- **Good (CV < 0.3)**: Smooth, consistent movement
- **Moderate (CV 0.3-0.6)**: Some movement variability
- **Poor (CV > 0.6)**: Jerky, inconsistent movement

#### Clinical Applications

**Parkinson's Disease**:
- Reduced velocity_mean (bradykinesia)
- Increased velocity_std (movement variability)
- Low velocity_max (reduced dynamic range)

**Stroke Recovery**:
- Asymmetric velocity patterns between sides
- Reduced velocity_mean on affected side
- High velocity_cv indicating poor motor control

**Cerebellar Ataxia**:
- High velocity_std (uncoordinated movement)
- Elevated velocity_cv (inconsistent control)
- Irregular velocity patterns

### 2. Acceleration Features (4 features)

Acceleration measures the rate of velocity change, reflecting force application and movement transitions.

#### Features Extracted

**acceleration_mean** (pixels/second²)
- **Description**: Average acceleration magnitude across all keypoints
- **Normal Range**: 10-50 pixels/s²
- **Clinical Significance**: Force generation and movement transitions
- **Interpretation**:
  - Low (<10): Gentle, gradual movements
  - Normal (10-50): Typical force application
  - High (>50): Abrupt, forceful movements

**acceleration_std** (pixels/second²)
- **Description**: Standard deviation of acceleration values
- **Normal Range**: 5-25 pixels/s²
- **Clinical Significance**: Consistency of force application
- **Interpretation**:
  - Low (<5): Very smooth force transitions
  - Normal (5-25): Natural force variability
  - High (>25): Erratic force application

**acceleration_max** (pixels/second²)
- **Description**: Maximum acceleration observed during the sequence
- **Normal Range**: 50-150 pixels/s²
- **Clinical Significance**: Peak force generation capacity
- **Interpretation**:
  - Low (<50): Reduced force capacity
  - Normal (50-150): Adequate force generation
  - High (>150): Excessive or compensatory forces

**acceleration_cv** (derived)
- **Calculation**: `acceleration_cv = acceleration_std / acceleration_mean`
- **Normal Range**: 0.3-0.7
- **Clinical Significance**: Force application consistency

#### Clinical Applications

**Muscle Weakness**:
- Reduced acceleration_mean (limited force generation)
- Low acceleration_max (reduced peak capacity)
- May show compensatory patterns

**Spasticity**:
- High acceleration_std (irregular force patterns)
- Elevated acceleration_cv (inconsistent control)
- Abrupt acceleration changes

**Pain-Related Gait**:
- Reduced acceleration on painful side
- Asymmetric acceleration patterns
- Cautious force application

### 3. Jerk Features (2 features)

Jerk measures the rate of acceleration change, reflecting movement smoothness and coordination quality.

#### Features Extracted

**jerk_mean** (pixels/second³)
- **Description**: Average jerk magnitude across all keypoints
- **Normal Range**: 5-30 pixels/s³
- **Clinical Significance**: Movement smoothness and coordination
- **Interpretation**:
  - Low (<5): Very smooth, controlled movement
  - Normal (5-30): Natural movement coordination
  - High (>30): Jerky, poorly coordinated movement

**jerk_std** (pixels/second³)
- **Description**: Standard deviation of jerk values
- **Normal Range**: 2-15 pixels/s³
- **Clinical Significance**: Consistency of movement coordination
- **Interpretation**:
  - Low (<2): Highly consistent coordination
  - Normal (2-15): Natural coordination variability
  - High (>15): Erratic, inconsistent coordination

#### Jerk Coefficient of Variation

**Calculation**: `jerk_cv = jerk_std / jerk_mean`

**Interpretation**:
- **Excellent (CV < 0.4)**: Highly coordinated movement
- **Good (CV 0.4-0.8)**: Adequate coordination
- **Poor (CV > 0.8)**: Poor coordination quality

#### Clinical Applications

**Neurological Conditions**:
- Elevated jerk_mean indicates poor motor control
- High jerk_std suggests inconsistent coordination
- Useful for monitoring neurological rehabilitation

**Movement Disorders**:
- Parkinson's: Variable jerk patterns with freezing episodes
- Huntington's: Elevated jerk due to chorea
- Dystonia: Irregular jerk patterns

**Rehabilitation Monitoring**:
- Decreasing jerk_mean indicates improving coordination
- Reduced jerk_std shows more consistent control
- Objective measure of treatment effectiveness

## Calculation Methodology

### 1. Velocity Calculation

```python
# For each keypoint at each frame
velocity = sqrt((x[t] - x[t-1])² + (y[t] - y[t-1])²) / dt

# Where:
# x[t], y[t] = keypoint position at time t
# dt = time between frames (1/fps)
```

### 2. Acceleration Calculation

```python
# For each keypoint at each frame
acceleration = (velocity[t] - velocity[t-1]) / dt

# Or directly from position:
acceleration = (position[t] - 2*position[t-1] + position[t-2]) / dt²
```

### 3. Jerk Calculation

```python
# For each keypoint at each frame
jerk = (acceleration[t] - acceleration[t-1]) / dt

# Or directly from position:
jerk = (position[t] - 3*position[t-1] + 3*position[t-2] - position[t-3]) / dt³
```

### 4. Aggregation Across Keypoints

```python
# Calculate mean across all keypoints at each frame
frame_velocity_mean = mean(velocity_all_keypoints)

# Then calculate statistics across all frames
velocity_mean = mean(frame_velocity_mean)
velocity_std = std(frame_velocity_mean)
velocity_max = max(frame_velocity_mean)
velocity_min = min(frame_velocity_mean)
```

## Data Quality Considerations

### Smoothing and Filtering

**Recommended Approach**:
- Apply Savitzky-Golay filter (window=5, polynomial=2) to position data
- Reduces noise while preserving signal characteristics
- Essential for accurate derivative calculations

**Without Smoothing**:
- Raw position data contains measurement noise
- Derivatives amplify noise (especially jerk)
- Results may be unreliable

### Confidence Weighting

**Implementation**:
```python
# Weight keypoint contributions by confidence
weighted_velocity = sum(velocity * confidence) / sum(confidence)
```

**Benefits**:
- Reduces impact of low-quality detections
- More robust to occlusions
- Improves feature reliability

### Missing Data Handling

**Strategies**:
- Interpolate short gaps (<5 frames)
- Exclude frames with <50% valid keypoints
- Report data quality metrics alongside features

## Integration with Other Features

### Complementary Feature Groups

**Kinematic + Joint Angles**:
- Kinematic features show movement quality
- Joint angles show movement patterns
- Together provide comprehensive assessment

**Kinematic + Temporal**:
- Kinematic features show how movement occurs
- Temporal features show when movement occurs
- Combined analysis reveals gait dynamics

**Kinematic + Symmetry**:
- Compare kinematic features between sides
- Identify asymmetric movement quality
- Detect compensation patterns

### Feature Correlation Analysis

**Expected Correlations**:
- velocity_mean ↔ walking_speed_ms (strong positive)
- jerk_mean ↔ movement_smoothness (strong negative)
- acceleration_std ↔ gait_variability (moderate positive)

**Unexpected Correlations**:
- May indicate compensation strategies
- Useful for identifying pathological patterns
- Guide clinical interpretation

## Clinical Interpretation Guidelines

### Assessment Framework

**Step 1: Velocity Assessment**
- Check velocity_mean against normal range
- Calculate velocity_cv for consistency
- Compare to age-matched norms

**Step 2: Acceleration Assessment**
- Evaluate acceleration_mean for force capacity
- Check acceleration_cv for control quality
- Identify asymmetric patterns

**Step 3: Jerk Assessment**
- Assess jerk_mean for coordination quality
- Evaluate jerk_std for consistency
- Compare to condition-specific patterns

**Step 4: Integrated Interpretation**
- Consider all kinematic features together
- Relate to joint angles and temporal features
- Generate clinical recommendations

### Condition-Specific Patterns

**Normal Gait**:
- velocity_cv: 0.2-0.4
- acceleration_cv: 0.3-0.6
- jerk_mean: 5-20 pixels/s³

**Parkinson's Disease**:
- velocity_cv: 0.4-0.7 (increased variability)
- acceleration_cv: 0.5-0.9 (poor control)
- jerk_mean: 20-50 (reduced smoothness)

**Stroke (Hemiplegic)**:
- Asymmetric velocity patterns
- Elevated jerk on affected side
- Compensatory acceleration patterns

**Cerebellar Ataxia**:
- velocity_cv: >0.7 (highly variable)
- acceleration_cv: >0.8 (erratic)
- jerk_mean: >40 (very jerky)

## Usage Examples

### Basic Feature Extraction

```python
from ambient.analysis.feature_extractor import FeatureExtractor

# Initialize extractor
extractor = FeatureExtractor(
    keypoint_format="COCO_17",
    fps=30.0,
    smoothing_window=5
)

# Extract features
features = extractor.extract_features(pose_sequence)

# Access kinematic features
velocity_mean = features['velocity_mean']
acceleration_mean = features['acceleration_mean']
jerk_mean = features['jerk_mean']
```

### Movement Quality Assessment

```python
def assess_movement_quality(features):
    """Assess movement quality from kinematic features."""
    
    # Calculate velocity consistency
    velocity_cv = features['velocity_std'] / features['velocity_mean']
    
    # Assess smoothness
    jerk_threshold = 30  # pixels/s³
    
    # Determine quality level
    if velocity_cv < 0.3 and features['jerk_mean'] < jerk_threshold:
        return "excellent"
    elif velocity_cv < 0.5 and features['jerk_mean'] < jerk_threshold * 1.5:
        return "good"
    elif velocity_cv < 0.7 and features['jerk_mean'] < jerk_threshold * 2:
        return "moderate"
    else:
        return "poor"
```

### Asymmetry Detection

```python
def detect_kinematic_asymmetry(left_features, right_features):
    """Detect asymmetry in kinematic features."""
    
    # Calculate velocity asymmetry
    velocity_si = abs(left_features['velocity_mean'] - right_features['velocity_mean']) / \
                  (0.5 * (left_features['velocity_mean'] + right_features['velocity_mean'])) * 100
    
    # Calculate acceleration asymmetry
    accel_si = abs(left_features['acceleration_mean'] - right_features['acceleration_mean']) / \
               (0.5 * (left_features['acceleration_mean'] + right_features['acceleration_mean'])) * 100
    
    # Interpret asymmetry
    if velocity_si > 20 or accel_si > 25:
        return "severe_asymmetry"
    elif velocity_si > 12 or accel_si > 15:
        return "moderate_asymmetry"
    else:
        return "symmetric"
```

## Research Applications

### Biomarker Development

Kinematic features serve as digital biomarkers for:
- Disease progression monitoring
- Treatment response assessment
- Early disease detection
- Functional capacity evaluation

### Clinical Trial Endpoints

**Advantages**:
- Objective, quantitative measures
- Continuous monitoring capability
- Sensitive to subtle changes
- Non-invasive assessment

**Applications**:
- Drug efficacy trials
- Rehabilitation outcome studies
- Surgical intervention assessment
- Device validation studies

## Limitations and Considerations

### Technical Limitations

**Resolution Dependency**:
- Pixel-based measurements depend on video resolution
- Calibration needed for absolute measurements
- Relative measures more robust

**Frame Rate Requirements**:
- Minimum 30 fps for reliable velocity
- 60+ fps preferred for acceleration
- 120+ fps ideal for jerk analysis

**Pose Estimation Quality**:
- Accuracy depends on pose detector quality
- Occlusions affect derivative calculations
- Confidence scores should be considered

### Clinical Limitations

**Individual Variability**:
- Normal ranges vary by age, height, fitness
- Baseline measurements recommended
- Longitudinal tracking more informative

**Context Dependency**:
- Walking speed affects kinematic values
- Footwear and surface influence results
- Environmental factors matter

**Interpretation Complexity**:
- Multiple factors influence kinematic features
- Clinical context essential for interpretation
- Should not be used in isolation

## Future Directions

### Methodological Advances

**3D Kinematic Analysis**:
- Depth cameras for 3D pose estimation
- More accurate derivative calculations
- Better anatomical alignment

**Machine Learning Integration**:
- Automated pattern recognition
- Personalized normal ranges
- Predictive modeling

**Real-Time Analysis**:
- Immediate feedback during assessment
- Biofeedback applications
- Telehealth integration

### Clinical Applications

**Precision Medicine**:
- Individual kinematic signatures
- Personalized treatment planning
- Targeted interventions

**Preventive Healthcare**:
- Early detection of decline
- Fall risk prediction
- Proactive intervention

## See Also

- [Gait Analysis Tutorial - Kinematic Features](../guides/gait-analysis-tutorial.md#36-kinematic-features-9-features---new) - User-friendly guide to kinematic features
- [Feature Extraction](feature-extraction.md) - Complete feature extraction documentation
- [Gait Analysis](gait-analysis.md) - Main gait analysis overview
- [Evidence-Based Features](evidence-based-gait-features-2025.md) - Research foundations
- [Temporal Analysis](temporal-analysis.md) - Gait cycle timing features
- [Symmetry Analysis](symmetry-analysis.md) - Bilateral comparison features

## References

1. **Journal of Biomechanics (2024)** - "Kinematic Analysis of Gait Patterns"
   - Comprehensive review of kinematic gait analysis methods
   - Establishes normal ranges for kinematic parameters

2. **Frontiers in Human Neuroscience (2025)** - "Movement Quality Assessment"
   - Jerk as a measure of movement coordination
   - Clinical applications in neurological rehabilitation

3. **Gait & Posture (2023)** - "Velocity and Acceleration in Pathological Gait"
   - Kinematic signatures of common gait disorders
   - Diagnostic value of derivative measures

4. **IEEE Transactions on Biomedical Engineering (2024)** - "Digital Biomarkers from Gait Kinematics"
   - Machine learning approaches to kinematic analysis
   - Validation of kinematic features as biomarkers
