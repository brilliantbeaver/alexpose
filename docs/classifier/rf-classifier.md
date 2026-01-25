# Random Forest Gait Classifier

## Overview

The Random Forest (RF) Gait Classifier is a machine learning-based classifier that uses ensemble learning to identify gait abnormalities and classify specific health conditions from extracted gait features.

## Key Features

- **Ensemble Learning**: Uses multiple decision trees for robust predictions
- **Feature Importance**: Provides interpretable rankings of which features contribute most to classification
- **Non-linear Relationships**: Handles complex, non-linear patterns in gait data
- **Balanced Class Weights**: Automatically handles imbalanced datasets
- **Cross-validation**: Built-in support for model validation
- **Hyperparameter Tuning**: Grid search for optimal model parameters

## Architecture

```
GaitFeatureVector (15 features)
    ↓
StandardScaler (optional normalization)
    ↓
Random Forest Classifier (100 trees by default)
    ↓
Prediction + Confidence + Feature Importance
```

## Feature Set

The classifier uses 15 gait features extracted from pose estimation:

### Mean Joint Angles (6 features)
- `left_hip_mean`: Average left hip flexion/extension angle
- `left_knee_mean`: Average left knee flexion angle
- `left_ankle_mean`: Average left ankle dorsiflexion angle
- `right_hip_mean`: Average right hip flexion/extension angle
- `right_knee_mean`: Average right knee flexion angle
- `right_ankle_mean`: Average right ankle dorsiflexion angle

### Asymmetry Features (3 features)
- `hip_asymmetry`: Absolute difference between left and right hip angles
- `knee_asymmetry`: Absolute difference between left and right knee angles
- `ankle_asymmetry`: Absolute difference between left and right ankle angles

### Range of Motion (6 features)
- `left_hip_range`: Range of motion for left hip
- `left_knee_range`: Range of motion for left knee
- `left_ankle_range`: Range of motion for left ankle
- `right_hip_range`: Range of motion for right hip
- `right_knee_range`: Range of motion for right knee
- `right_ankle_range`: Range of motion for right ankle

## Usage

### Basic Training and Classification

```python
from ambient.classification.rf_classifier import RFGaitClassifier, RFClassifierConfig
from ambient.classification.features import GaitFeatureVector

# Configure classifier
config = RFClassifierConfig(
    n_estimators=100,
    max_depth=None,
    random_state=42
)

# Initialize classifier
classifier = RFGaitClassifier(config)

# Train on feature vectors
training_features = [...]  # List of GaitFeatureVector objects
metrics = classifier.train(training_features, validate=True)

print(f"Training accuracy: {metrics['train_accuracy']:.3f}")
print(f"CV accuracy: {metrics['cv_mean_accuracy']:.3f}")

# Classify new sample
test_feature = GaitFeatureVector(...)
result = classifier.classify_gait(test_feature)

print(f"Predicted: {result['predicted_condition']}")
print(f"Confidence: {result['confidence']:.2f}")
```

### Feature Importance Analysis

```python
# Get feature importances
importances = classifier.get_feature_importances()

print("Top 5 most important features:")
for imp in importances[:5]:
    print(f"{imp.rank}. {imp.feature_name}: {imp.importance:.4f}")
```

### Model Evaluation

```python
# Evaluate on test set
test_features = [...]  # List of GaitFeatureVector objects
metrics = classifier.evaluate(test_features)

print(f"Test accuracy: {metrics['accuracy']:.3f}")
print(f"F1 score: {metrics['f1_score']:.3f}")
print(f"Precision: {metrics['precision']:.3f}")
print(f"Recall: {metrics['recall']:.3f}")
```

### Hyperparameter Tuning

```python
# Define parameter grid
param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [10, 20, 30],
    'min_samples_split': [2, 5, 10],
}

# Tune hyperparameters
results = classifier.tune_hyperparameters(
    training_features,
    param_grid=param_grid,
    cv_folds=5
)

print(f"Best parameters: {results['best_params']}")
print(f"Best CV score: {results['best_score']:.3f}")
```

### Model Persistence

```python
# Save trained model
classifier.save('models/rf_gait_classifier.pkl')

# Load model
loaded_classifier = RFGaitClassifier.load('models/rf_gait_classifier.pkl')
```

## Configuration Options

### RFClassifierConfig

```python
@dataclass
class RFClassifierConfig:
    n_estimators: int = 100              # Number of trees in forest
    max_depth: Optional[int] = None      # Maximum tree depth (None = unlimited)
    min_samples_split: int = 2           # Min samples to split internal node
    min_samples_leaf: int = 1            # Min samples in leaf node
    max_features: str = "sqrt"           # Features per split ("sqrt", "log2")
    bootstrap: bool = True               # Use bootstrap sampling
    random_state: int = 42               # Random seed for reproducibility
    n_jobs: int = -1                     # CPU cores (-1 = all cores)
    normalize_features: bool = True      # Apply StandardScaler
    confidence_threshold: float = 0.5    # Confidence threshold
    class_weight: Optional[str] = "balanced"  # Handle imbalanced classes
```

## Classification Output

The `classify_gait()` method returns a dictionary with:

```python
{
    'predicted_condition': str,          # Predicted condition label
    'confidence': float,                 # Confidence score (0-1)
    'probabilities': Dict[str, float],   # Probability per class
    'tree_votes': Dict[str, int],        # Vote count per class
    'n_trees': int,                      # Total number of trees
    'is_normal': bool,                   # True if predicted as normal
    'feature_vector': List[float]        # Input feature values
}
```

## Comparison with KNN Classifier

| Feature | Random Forest | KNN |
|---------|--------------|-----|
| **Algorithm** | Ensemble of decision trees | Distance-based voting |
| **Training Speed** | Slower | Faster |
| **Prediction Speed** | Fast | Slower (computes distances) |
| **Feature Importance** | ✅ Yes | ❌ No |
| **Non-linear Patterns** | ✅ Excellent | ⚠️ Limited |
| **Interpretability** | ⚠️ Moderate (via feature importance) | ✅ High (via neighbors) |
| **Hyperparameters** | Many | Few |
| **Overfitting Risk** | Low (ensemble averaging) | Moderate |
| **Memory Usage** | Moderate | High (stores all training data) |

## Best Practices

### When to Use Random Forest

- **Complex patterns**: When gait abnormalities involve non-linear feature relationships
- **Feature analysis**: When you need to understand which features are most important
- **Imbalanced data**: When some conditions have fewer samples than others
- **Production deployment**: When prediction speed is important

### When to Use KNN Instead

- **Small datasets**: When you have limited training data
- **Explainability**: When you need to show similar cases to justify predictions
- **Simple patterns**: When conditions are linearly separable
- **Quick prototyping**: When you need fast initial results

### Recommended Configuration

For most gait classification tasks:

```python
config = RFClassifierConfig(
    n_estimators=100,        # Good balance of accuracy and speed
    max_depth=20,            # Prevent overfitting
    min_samples_split=5,     # Require meaningful splits
    min_samples_leaf=2,      # Avoid single-sample leaves
    max_features="sqrt",     # Reduce correlation between trees
    class_weight="balanced", # Handle imbalanced conditions
    normalize_features=True  # Standardize feature scales
)
```

## Clinical Interpretation

### Feature Importance for Conditions

Based on typical gait patterns:

**Stroke/Hemiplegic Gait**
- High importance: `hip_asymmetry`, `knee_asymmetry`, `ankle_asymmetry`
- Rationale: Significant left-right differences due to hemiparesis

**Parkinson's Disease**
- High importance: `*_range` features (all ROM features)
- Rationale: Reduced range of motion, shuffling gait

**Antalgic Gait**
- High importance: `hip_asymmetry`, `*_ankle_range`
- Rationale: Compensatory patterns to avoid pain

**Normal Gait**
- Balanced importance across all features
- Low asymmetry values, normal ROM

### Confidence Interpretation

- **> 0.9**: Very confident prediction, strong agreement across trees
- **0.7 - 0.9**: Confident prediction, good agreement
- **0.5 - 0.7**: Moderate confidence, some disagreement
- **< 0.5**: Low confidence, consider manual review

## Performance Metrics

Typical performance on GAVD dataset (with proper train/test split):

- **Accuracy**: 85-92%
- **F1 Score**: 0.83-0.90
- **Training Time**: 2-5 seconds (100 trees, 90 samples)
- **Prediction Time**: < 10ms per sample

## Troubleshooting

### Low Accuracy

1. **Check feature quality**: Ensure pose estimation is accurate
2. **Increase n_estimators**: Try 200-500 trees
3. **Tune hyperparameters**: Use grid search
4. **Add more training data**: Especially for minority classes
5. **Feature engineering**: Consider adding temporal features

### Overfitting

1. **Reduce max_depth**: Limit tree complexity
2. **Increase min_samples_split**: Require more samples for splits
3. **Increase min_samples_leaf**: Prevent tiny leaves
4. **Use cross-validation**: Validate generalization

### Slow Training

1. **Reduce n_estimators**: Use fewer trees
2. **Limit max_depth**: Shallower trees train faster
3. **Reduce n_jobs**: Use fewer CPU cores if memory-constrained
4. **Subsample data**: Use stratified sampling for large datasets

## Integration with AlexPose Pipeline

```python
from ambient.pose.joint_angles import JointAngleSequence
from ambient.classification.features import GaitFeatureVector
from ambient.classification.rf_classifier import RFGaitClassifier

# 1. Extract joint angles from pose estimation
joint_angles = JointAngleSequence(...)

# 2. Convert to feature vector
features = GaitFeatureVector.from_joint_angles(
    joint_angles,
    sample_id="patient_001",
    condition_label="unknown"
)

# 3. Classify
classifier = RFGaitClassifier.load('models/trained_rf.pkl')
result = classifier.classify_gait(features)

# 4. Generate explanation
explanation = classifier.explain_classification(result)
print(explanation)
```

## References

- Breiman, L. (2001). "Random Forests". Machine Learning. 45 (1): 5–32.
- Scikit-learn Random Forest Documentation: https://scikit-learn.org/stable/modules/ensemble.html#forest
- GAVD Dataset: Gait Abnormality Video Dataset for clinical gait analysis

## See Also

- [KNN Classifier](./quickstart.md) - Alternative distance-based classifier
- [Feature Extraction](../analysis/feature-extraction.md) - How gait features are computed
- [Joint Angles](../gait/joint-angle-calculation.md) - Joint angle calculation details
