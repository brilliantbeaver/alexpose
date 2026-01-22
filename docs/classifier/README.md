# KNN Gait Classifier Documentation

## Overview

The KNN (K-Nearest Neighbors) Gait Classifier is a machine learning-based approach for classifying gait patterns into different health conditions. It complements the LLM-based classification system by providing a fast, interpretable, and resource-efficient alternative.

## Table of Contents

1. [Architecture](#architecture)
2. [Features](#features)
3. [Usage Guide](#usage-guide)
4. [API Reference](#api-reference)
5. [Training Guide](#training-guide)
6. [Evaluation Metrics](#evaluation-metrics)
7. [Best Practices](#best-practices)

## Architecture

### Design Principles

The KNN classifier follows SOLID principles:

- **Single Responsibility**: Focused solely on KNN-based gait classification
- **Open/Closed**: Extensible through custom feature extractors
- **Liskov Substitution**: Implements `IClassifier` interface
- **Interface Segregation**: Clean, minimal interface
- **Dependency Inversion**: Depends on abstractions (interfaces)

### Component Structure

```
ambient/classification/
├── knn_classifier.py          # Main KNN classifier implementation
│   ├── KNNGaitClassifier      # Classifier class
│   ├── GaitFeatureVector      # Feature representation
│   └── KNNClassifierConfig    # Configuration
└── __init__.py                # Module exports
```

### Integration with Existing System

The KNN classifier integrates seamlessly with the existing AlexPose architecture:

```python
# Uses existing joint angle extraction
from ambient.pose.joint_angles import get_joint_angles

# Implements standard interface
from ambient.core.interfaces import IClassifier

# Uses existing data models
from ambient.core.data_models import ClassificationResult
```

## Features

### Feature Vector

The classifier uses 15 features extracted from gait sequences:

**Mean Joint Angles (6 features)**:
- Left hip mean angle
- Left knee mean angle
- Left ankle mean angle
- Right hip mean angle
- Right knee mean angle
- Right ankle mean angle

**Asymmetry Measures (3 features)**:
- Hip asymmetry (|left - right|)
- Knee asymmetry (|left - right|)
- Ankle asymmetry (|left - right|)

**Range of Motion (6 features)**:
- Left hip range
- Left knee range
- Left ankle range
- Right hip range
- Right knee range
- Right ankle range

### Clinical Rationale

These features were selected based on clinical gait analysis literature:

1. **Mean Joint Angles**: Capture overall joint positioning during gait
   - Reduced hip/knee flexion indicates Parkinson's or antalgic gait
   - Abnormal ankle angles suggest foot drop or other pathologies

2. **Asymmetry**: Key indicator of pathological gait
   - Stroke patients show significant left-right asymmetry
   - Normal gait has minimal asymmetry (<10%)

3. **Range of Motion**: Indicates joint mobility and function
   - Reduced ROM suggests stiffness or pain avoidance
   - Excessive ROM may indicate instability
   - Ankle ROM is particularly important for detecting foot drop (stroke) and reduced push-off (Parkinson's)

## Usage Guide

### Basic Classification

```python
from ambient.classification.knn_classifier import KNNGaitClassifier
from ambient.pose.joint_angles import get_joint_angles

# Load trained classifier
classifier = KNNGaitClassifier.load("models/knn_classifier.pkl")

# Extract features from gait sequence
joint_angles = get_joint_angles(keypoints_array, keypoint_format="BLAZEPOSE_33")
feature_vector = GaitFeatureVector.from_joint_angles(joint_angles)

# Classify
result = classifier.classify_gait(feature_vector)

print(f"Predicted: {result['predicted_condition']}")
print(f"Confidence: {result['confidence']:.2%}")
print(f"Probabilities: {result['probabilities']}")
```

### Training a New Classifier

```python
from ambient.classification.knn_classifier import (
    KNNGaitClassifier,
    KNNClassifierConfig,
    GaitFeatureVector
)

# Configure classifier
config = KNNClassifierConfig(
    n_neighbors=5,
    weights="distance",
    metric="euclidean",
    normalize_features=True
)

# Create classifier
classifier = KNNGaitClassifier(config=config)

# Train with feature vectors
training_features = [...]  # List of GaitFeatureVector objects
metrics = classifier.train(training_features, validate=True)

# Save trained model
classifier.save("models/my_classifier.pkl")
```

### Hyperparameter Tuning

```python
# Define parameter grid
param_grid = {
    'n_neighbors': [3, 5, 7, 9],
    'weights': ['uniform', 'distance'],
    'metric': ['euclidean', 'manhattan', 'minkowski']
}

# Tune hyperparameters
results = classifier.tune_hyperparameters(
    training_features,
    param_grid=param_grid
)

print(f"Best parameters: {results['best_params']}")
print(f"Best CV score: {results['best_score']:.3f}")
```

## API Reference

### KNNGaitClassifier

Main classifier class implementing the `IClassifier` interface.

#### Methods

**`__init__(config: Optional[KNNClassifierConfig] = None)`**
- Initialize classifier with optional configuration

**`train(features: List[GaitFeatureVector], labels: Optional[List[str]] = None, validate: bool = True) -> Dict[str, Any]`**
- Train the classifier on feature vectors
- Returns training metrics including accuracy and cross-validation scores

**`classify_gait(gait_features: Union[GaitFeatureVector, Dict], context: Optional[Dict] = None) -> Dict[str, Any]`**
- Classify a gait sample
- Returns prediction, confidence, and probabilities

**`evaluate(test_features: List[GaitFeatureVector], test_labels: Optional[List[str]] = None) -> Dict[str, Any]`**
- Evaluate classifier on test data
- Returns accuracy, confusion matrix, and classification report

**`tune_hyperparameters(features: List[GaitFeatureVector], labels: Optional[List[str]] = None, param_grid: Optional[Dict] = None) -> Dict[str, Any]`**
- Perform grid search for optimal hyperparameters
- Returns best parameters and cross-validation scores

**`save(filepath: Union[str, Path]) -> None`**
- Save trained classifier to file

**`load(filepath: Union[str, Path]) -> KNNGaitClassifier`** (classmethod)
- Load trained classifier from file

### GaitFeatureVector

Feature representation for gait classification.

#### Methods

**`from_joint_angles(joint_angle_sequence, sample_id: str = "", condition_label: str = "") -> GaitFeatureVector`** (classmethod)
- Create feature vector from JointAngleSequence

**`to_array() -> np.ndarray`**
- Convert to numpy array for sklearn

**`get_feature_names() -> List[str]`** (classmethod)
- Get ordered list of feature names

### KNNClassifierConfig

Configuration dataclass for KNN classifier.

#### Attributes

- `n_neighbors: int = 5` - Number of neighbors to use
- `weights: str = "distance"` - Weight function ('uniform' or 'distance')
- `metric: str = "euclidean"` - Distance metric
- `algorithm: str = "auto"` - Algorithm for computing neighbors
- `normalize_features: bool = True` - Whether to normalize features
- `confidence_threshold: float = 0.5` - Minimum confidence threshold

## Training Guide

### Data Preparation

1. **Organize Data by Condition**:
```
experiments/exp2/data/
├── normal/
│   ├── sample1.csv
│   └── sample2.csv
├── stroke/
│   ├── sample1.csv
│   └── sample2.csv
├── parkinsons/
│   └── sample1.csv
└── antalgic/
    └── sample1.csv
```

2. **Run Training Script**:
```bash
python experiments/exp2/src/process4_KNN.py
```

3. **Output Files**:
- `experiments/exp2/models/knn_classifier.pkl` - Trained model
- `experiments/exp2/results/training_metrics.json` - Training metrics
- `experiments/exp2/results/evaluation_report.txt` - Evaluation report

### Training Pipeline

The training script follows this pipeline:

1. **Data Loading**: Extract features from GAVD CSV files
2. **Feature Extraction**: Compute joint angles and create feature vectors
3. **Train/Test Split**: 80/20 split with random seed for reproducibility
4. **Training**: Train KNN with cross-validation
5. **Evaluation**: Test on held-out data
6. **Saving**: Save model and metrics

### Recommended Hyperparameters

Based on gait analysis literature and empirical testing:

- **n_neighbors**: 5-7 (balance between bias and variance)
- **weights**: "distance" (closer neighbors have more influence)
- **metric**: "euclidean" (standard for continuous features)
- **normalize_features**: True (features have different scales)

## Evaluation Metrics

### Accuracy

Overall classification accuracy on test set.

**Interpretation**:
- >90%: Excellent
- 80-90%: Good
- 70-80%: Fair
- <70%: Poor (needs improvement)

### Precision, Recall, F1-Score

Per-class metrics from classification report.

**Precision**: Of predicted positives, how many are correct?
**Recall**: Of actual positives, how many were found?
**F1-Score**: Harmonic mean of precision and recall

### Cross-Validation Score

Average accuracy across k-fold cross-validation.

**Purpose**: Estimate generalization performance and detect overfitting

### Confusion Matrix

Shows true vs predicted labels for all classes.

**Use**: Identify which conditions are confused with each other

## Best Practices

### Feature Engineering

1. **Use Sufficient Data**: Minimum 5-10 samples per condition
2. **Balance Classes**: Similar number of samples per condition
3. **Quality Over Quantity**: Clean, high-confidence keypoints

### Model Selection

1. **Start Simple**: Begin with k=5, distance weighting
2. **Tune Systematically**: Use grid search for optimization
3. **Validate Thoroughly**: Always use cross-validation

### Deployment

1. **Version Control**: Track model versions and training data
2. **Monitor Performance**: Log predictions and confidence scores
3. **Update Regularly**: Retrain with new data periodically

### Interpretability

1. **Examine Neighbors**: Check which samples are most similar
2. **Feature Importance**: Analyze which features drive predictions
3. **Confidence Thresholds**: Set appropriate thresholds for clinical use

## Comparison with LLM Classifier

| Aspect | KNN Classifier | LLM Classifier |
|--------|---------------|----------------|
| **Speed** | Fast (<1ms) | Slow (1-5s) |
| **Cost** | Free | API costs |
| **Interpretability** | High (neighbors) | Medium (explanations) |
| **Accuracy** | Good (80-90%) | Excellent (90-95%) |
| **Training Data** | Small (10-100) | Large (100-1000+) |
| **Deployment** | Simple | Complex |
| **Offline Use** | Yes | No (API required) |

### When to Use Each

**Use KNN when**:
- Fast inference is required
- Offline operation is needed
- Limited training data available
- Interpretability is critical
- Cost is a concern

**Use LLM when**:
- Maximum accuracy is required
- Rich explanations are needed
- Large training dataset available
- API access is available
- Cost is not a concern

## Troubleshooting

### Low Accuracy

**Possible causes**:
- Insufficient training data
- Imbalanced classes
- Poor feature quality
- Wrong hyperparameters

**Solutions**:
- Collect more data
- Balance classes or use class weights
- Improve keypoint extraction
- Tune hyperparameters

### High Variance

**Symptoms**: Large gap between train and test accuracy

**Solutions**:
- Increase k (more neighbors)
- Add more training data
- Use distance weighting
- Feature selection

### Poor Generalization

**Symptoms**: Good CV score but poor real-world performance

**Solutions**:
- Ensure test data is representative
- Check for data leakage
- Validate feature extraction pipeline
- Consider domain adaptation

## References

1. Begg, R., & Kamruzzaman, J. (2005). A machine learning approach for automated recognition of movement patterns using basic, kinetic and kinematic gait data. *Journal of Biomechanics*, 38(3), 401-408.

2. Horst, F., Lapuschkin, S., Samek, W., Müller, K. R., & Schöllhorn, W. I. (2019). Explaining the unique nature of individual gait patterns with deep learning. *Scientific Reports*, 9(1), 2391.

3. Mannini, A., & Sabatini, A. M. (2012). Machine learning methods for classifying human physical activity from on-body accelerometers. *Sensors*, 12(2), 1154-1175.

4. Verlekar, T. T., Soares, L. D., & Correia, P. L. (2018). Automatic classification of gait impairments using a markerless 2D video-based system. *Sensors*, 18(9), 2743.
