# KNN Gait Classifier - Design Document

## Executive Summary

This document describes the design and architecture of the KNN (K-Nearest Neighbors) Gait Classifier, a machine learning component for classifying gait patterns into health conditions. The classifier provides a fast, interpretable alternative to LLM-based classification while maintaining high accuracy.

## Design Goals

### Primary Goals

1. **Accuracy**: Achieve >80% classification accuracy on gait conditions
2. **Speed**: Provide sub-millisecond inference time
3. **Interpretability**: Enable understanding of classification decisions
4. **Integration**: Seamlessly integrate with existing AlexPose architecture
5. **Maintainability**: Follow SOLID principles and best practices

### Non-Goals

- Real-time video processing (handled by pose estimation)
- Multi-modal input (video + sensor data)
- Deep learning approaches (separate component)

## Architecture

### System Context

```
┌─────────────────────────────────────────────────────────────┐
│                     AlexPose System                          │
│                                                              │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐ │
│  │   Video      │───▶│     Pose     │───▶│    Joint     │ │
│  │  Processing  │    │  Estimation  │    │   Angles     │ │
│  └──────────────┘    └──────────────┘    └──────┬───────┘ │
│                                                   │          │
│                                                   ▼          │
│                                          ┌──────────────┐   │
│                                          │     KNN      │   │
│                                          │  Classifier  │   │
│                                          └──────┬───────┘   │
│                                                   │          │
│                                                   ▼          │
│                                          ┌──────────────┐   │
│                                          │ Classification│   │
│                                          │    Result    │   │
│                                          └──────────────┘   │
└─────────────────────────────────────────────────────────────┘
```

### Component Architecture

```
ambient/classification/
│
├── knn_classifier.py
│   │
│   ├── GaitFeatureVector
│   │   ├── from_joint_angles()      # Factory method
│   │   ├── to_array()                # Conversion
│   │   └── get_feature_names()       # Metadata
│   │
│   ├── KNNClassifierConfig
│   │   └── [Configuration parameters]
│   │
│   └── KNNGaitClassifier (IClassifier)
│       ├── train()                   # Training
│       ├── classify_gait()           # Prediction
│       ├── evaluate()                # Evaluation
│       ├── tune_hyperparameters()    # Optimization
│       ├── save()                    # Persistence
│       └── load()                    # Loading
```

## Design Principles

### SOLID Principles

**Single Responsibility Principle**
- `GaitFeatureVector`: Feature representation only
- `KNNGaitClassifier`: Classification logic only
- `KNNClassifierConfig`: Configuration only

**Open/Closed Principle**
- Extensible through custom feature extractors
- New distance metrics can be added without modifying core code
- Feature engineering is separate from classification

**Liskov Substitution Principle**
- Implements `IClassifier` interface
- Can be used interchangeably with `LLMClassifier`
- Maintains contract of parent interface

**Interface Segregation Principle**
- Clean, minimal interface
- No unnecessary methods
- Focused on classification tasks

**Dependency Inversion Principle**
- Depends on `IClassifier` abstraction
- Uses `JointAngleSequence` interface
- Decoupled from specific implementations

### Additional Principles

**DRY (Don't Repeat Yourself)**
- Reuses existing joint angle extraction
- Shares data models with other components
- Common utilities in base classes

**YAGNI (You Aren't Gonna Need It)**
- No speculative features
- Focused on current requirements
- Simple, proven algorithms

**Modularity**
- Clear separation of concerns
- Independent components
- Easy to test and maintain

## Feature Engineering

### Feature Selection Rationale

The 15 features were selected based on:

1. **Clinical Relevance**: Used in clinical gait analysis
2. **Discriminative Power**: Differ significantly across conditions
3. **Computational Efficiency**: Fast to compute
4. **Robustness**: Stable across different videos

### Feature Categories

**1. Mean Joint Angles (6 features)**

```python
- left_hip_mean
- left_knee_mean
- left_ankle_mean
- right_hip_mean
- right_knee_mean
- right_ankle_mean
```

**Clinical Significance**:
- Capture overall joint positioning
- Indicate flexion/extension patterns
- Reveal compensation strategies

**2. Asymmetry Measures (3 features)**

```python
- hip_asymmetry = |left_hip_mean - right_hip_mean|
- knee_asymmetry = |left_knee_mean - right_knee_mean|
- ankle_asymmetry = |left_ankle_mean - right_ankle_mean|
```

**Clinical Significance**:
- Key indicator of pathological gait
- Stroke patients show high asymmetry
- Normal gait has minimal asymmetry (<10%)

**3. Range of Motion (4 features)**

```python
- left_hip_range
- left_knee_range
- right_hip_range
- right_knee_range
```

**Clinical Significance**:
- Indicates joint mobility
- Reduced ROM suggests stiffness
- Excessive ROM may indicate instability

### Feature Normalization

Features are normalized using `StandardScaler`:

```python
X_normalized = (X - mean) / std
```

**Rationale**:
- Joint angles have different scales
- Prevents dominance by large-scale features
- Improves KNN distance calculations

## Algorithm Selection

### Why K-Nearest Neighbors?

**Advantages**:
1. **Simplicity**: Easy to understand and implement
2. **No Training Time**: Lazy learning algorithm
3. **Interpretability**: Can examine nearest neighbors
4. **Non-parametric**: No assumptions about data distribution
5. **Effective for Small Datasets**: Works well with limited data

**Disadvantages**:
1. **Memory**: Stores all training data
2. **Prediction Speed**: Slower than parametric models
3. **Curse of Dimensionality**: Performance degrades with many features
4. **Sensitive to Noise**: Outliers affect predictions

### Hyperparameter Selection

**n_neighbors = 5**
- Balances bias and variance
- Odd number prevents ties
- Empirically validated

**weights = "distance"**
- Closer neighbors have more influence
- Reduces impact of distant outliers
- Improves accuracy

**metric = "euclidean"**
- Standard for continuous features
- Intuitive distance measure
- Computationally efficient

**normalize_features = True**
- Essential for fair distance calculations
- Prevents scale dominance
- Improves performance

## Data Flow

### Training Pipeline

```
1. Load GAVD CSV files
   ↓
2. Extract keypoints from videos
   ↓
3. Calculate joint angles
   ↓
4. Create feature vectors
   ↓
5. Normalize features
   ↓
6. Train KNN model
   ↓
7. Cross-validate
   ↓
8. Save model
```

### Prediction Pipeline

```
1. Load video/keypoints
   ↓
2. Calculate joint angles
   ↓
3. Create feature vector
   ↓
4. Normalize features
   ↓
5. Find K nearest neighbors
   ↓
6. Weight by distance
   ↓
7. Aggregate votes
   ↓
8. Return prediction + confidence
```

## Performance Considerations

### Time Complexity

- **Training**: O(1) - just stores data
- **Prediction**: O(n * d) where n = training samples, d = features
- **Memory**: O(n * d) - stores all training data

### Optimization Strategies

1. **Feature Reduction**: Use only 13 most important features
2. **Efficient Distance**: Use optimized sklearn implementation
3. **Caching**: Cache normalized features
4. **Batch Prediction**: Process multiple samples together

### Scalability

**Current Scale**:
- Training: 10-100 samples
- Features: 13 dimensions
- Prediction: <1ms per sample

**Future Scale**:
- Training: 100-1000 samples
- Features: 13-20 dimensions
- Prediction: <10ms per sample

**Scaling Strategy**:
- Use approximate nearest neighbors (ANN) for large datasets
- Consider dimensionality reduction (PCA)
- Implement batch processing

## Error Handling

### Input Validation

```python
# Training
if not features:
    raise ValueError("No training features provided")

if len(X) != len(y):
    raise ValueError("Feature count != label count")

# Prediction
if not self.is_trained:
    raise RuntimeError("Classifier must be trained")
```

### Graceful Degradation

- Return empty results for invalid input
- Log warnings for low-confidence predictions
- Provide fallback to default predictions

## Testing Strategy

### Unit Tests

- Feature vector creation
- Classifier initialization
- Training logic
- Prediction logic
- Model persistence

### Integration Tests

- End-to-end workflow
- Integration with joint angle extraction
- Integration with data loading

### Property-Based Tests

```python
@given(st.lists(st.floats(min_value=0, max_value=180), min_size=13, max_size=13))
def test_feature_vector_properties(angles):
    """Test feature vector with random valid angles."""
    # Property: All angles should be in valid range
    # Property: Asymmetry should be non-negative
    # Property: Array conversion should preserve values
```

### Performance Tests

- Benchmark training time
- Benchmark prediction time
- Memory usage profiling

## Deployment

### Model Versioning

```
models/
├── knn_classifier_v1.0.pkl
├── knn_classifier_v1.1.pkl
└── knn_classifier_latest.pkl -> knn_classifier_v1.1.pkl
```

### Model Metadata

```json
{
  "version": "1.1",
  "created_at": "2026-01-19T10:00:00Z",
  "training_samples": 50,
  "accuracy": 0.85,
  "classes": ["normal", "stroke", "parkinsons", "antalgic"],
  "features": 13,
  "hyperparameters": {
    "n_neighbors": 5,
    "weights": "distance",
    "metric": "euclidean"
  }
}
```

### Monitoring

- Log all predictions
- Track confidence scores
- Monitor accuracy over time
- Alert on low-confidence predictions

## Future Enhancements

### Short Term

1. **Feature Engineering**
   - Add temporal features (cadence, stride time)
   - Add velocity features
   - Add acceleration features

2. **Model Improvements**
   - Ensemble with other classifiers
   - Weighted voting by confidence
   - Active learning for data collection

3. **Usability**
   - Web API endpoint
   - Batch prediction support
   - Confidence calibration

### Long Term

1. **Advanced Algorithms**
   - Random Forest classifier
   - Gradient Boosting
   - Neural network classifier

2. **Multi-Modal**
   - Combine with LLM predictions
   - Integrate sensor data
   - Use video features directly

3. **Personalization**
   - User-specific models
   - Transfer learning
   - Few-shot learning

## References

1. Begg, R., & Kamruzzaman, J. (2005). A machine learning approach for automated recognition of movement patterns. *Journal of Biomechanics*.

2. Verlekar, T. T., et al. (2018). Automatic classification of gait impairments using a markerless 2D video-based system. *Sensors*.

3. Hastie, T., Tibshirani, R., & Friedman, J. (2009). *The Elements of Statistical Learning*. Springer.

4. Martin, R. C. (2017). *Clean Architecture: A Craftsman's Guide to Software Structure and Design*. Prentice Hall.
