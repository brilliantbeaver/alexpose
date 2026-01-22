# Gait Classification System

This directory contains documentation for the gait classification system, which uses machine learning to identify gait abnormalities and specific health conditions from extracted gait features.

## Overview

The classification system supports 8 different algorithms, from simple baselines to advanced ensemble methods:

### Available Classifiers

1. **[Logistic Regression](logistic-classifier.md)** - Fast, interpretable baseline
2. **[K-Nearest Neighbors](knn-classifier.md)** - Instance-based learning
3. **[Decision Tree](decisiontree-classifier.md)** - Maximum interpretability
4. **[Support Vector Machine](svm-classifier.md)** - Optimal decision boundaries
5. **[Random Forest](rf-classifier.md)** - Robust ensemble of trees
6. **[XGBoost](xgboost-classifier.md)** - State-of-the-art gradient boosting
7. **[MLP Neural Network](mlp-classifier.md)** - Deep learning approach
8. **[Ensemble](ensemble-classifier.md)** - Combines multiple classifiers

## Quick Comparison

| Classifier | Speed | Accuracy | Interpretability | Best For |
|------------|-------|----------|------------------|----------|
| Logistic | ★★★★★ | ★★★☆☆ | ★★★★★ | Baselines, clinical decisions |
| KNN | ★★★☆☆ | ★★★☆☆ | ★★★★☆ | Quick prototyping, local patterns |
| Decision Tree | ★★★★☆ | ★★★☆☆ | ★★★★★ | Explainable decisions, rules |
| SVM | ★★★☆☆ | ★★★★☆ | ★★☆☆☆ | Non-linear patterns, small data |
| Random Forest | ★★★★☆ | ★★★★☆ | ★★★☆☆ | Robust general-purpose |
| XGBoost | ★★★☆☆ | ★★★★★ | ★★★☆☆ | Maximum accuracy |
| MLP | ★★☆☆☆ | ★★★★★ | ★☆☆☆☆ | Large datasets, online learning |
| Ensemble | ★★☆☆☆ | ★★★★★ | ★★☆☆☆ | Production, critical applications |

## Choosing a Classifier

### For Clinical Applications
- **Primary**: Logistic Regression (interpretable, fast)
- **Secondary**: Decision Tree (explainable rules)
- **Advanced**: Ensemble with agreement analysis

### For Maximum Accuracy
- **Primary**: XGBoost or Ensemble
- **Secondary**: Random Forest or MLP
- **Baseline**: Logistic Regression for comparison

### For Real-Time Systems
- **Primary**: Logistic Regression
- **Secondary**: KNN or Decision Tree
- **Avoid**: MLP, Ensemble (slower)

### For Small Datasets (< 100 samples)
- **Primary**: SVM or KNN
- **Secondary**: Logistic Regression
- **Avoid**: MLP, XGBoost (need more data)

### For Large Datasets (> 10,000 samples)
- **Primary**: XGBoost or MLP
- **Secondary**: Random Forest
- **Avoid**: KNN (slow predictions)

## Quick Start

```python
from ambient.classification.xgboost_classifier import XGBoostGaitClassifier
from ambient.classification.knn_classifier import GaitFeatureVector

# Create classifier
classifier = XGBoostGaitClassifier()

# Prepare training data
training_features = [
    GaitFeatureVector(
        left_hip_mean=45.0, left_knee_mean=60.0, left_ankle_mean=20.0,
        right_hip_mean=45.0, right_knee_mean=60.0, right_ankle_mean=20.0,
        hip_asymmetry=1.0, knee_asymmetry=1.0, ankle_asymmetry=1.0,
        left_hip_range=40.0, left_knee_range=70.0, left_ankle_range=30.0,
        right_hip_range=40.0, right_knee_range=70.0, right_ankle_range=30.0,
        condition_label="normal"
    ),
    # ... more samples
]

# Train with cross-validation
metrics = classifier.train(training_features, validate=True)
print(f"Accuracy: {metrics['train_accuracy']:.3f}")
print(f"CV Score: {metrics['cv_mean_accuracy']:.3f}")

# Classify new sample
result = classifier.classify_gait(test_feature)
print(f"Predicted: {result['predicted_condition']}")
print(f"Confidence: {result['confidence']:.2%}")

# Save model
classifier.save("models/gait_classifier.pkl")
```

## Common Workflows

### 1. Establish Baseline
```python
from ambient.classification.logistic_classifier import LogisticGaitClassifier

# Train simple baseline
baseline = LogisticGaitClassifier()
baseline_metrics = baseline.train(training_features, validate=True)
print(f"Baseline accuracy: {baseline_metrics['cv_mean_accuracy']:.3f}")
```

### 2. Optimize Single Classifier
```python
from ambient.classification.xgboost_classifier import XGBoostGaitClassifier

# Train and tune
classifier = XGBoostGaitClassifier()
tuning_results = classifier.tune_hyperparameters(training_features, cv_folds=5)
print(f"Best params: {tuning_results['best_params']}")
print(f"Best score: {tuning_results['best_score']:.3f}")
```

### 3. Build Ensemble
```python
from ambient.classification.ensemble_classifier import (
    EnsembleGaitClassifier,
    EnsembleClassifierConfig,
    VotingStrategy
)

# Combine best classifiers
ensemble = EnsembleGaitClassifier(
    EnsembleClassifierConfig(
        voting_strategy=VotingStrategy.SOFT,
        classifiers=["rf", "xgboost", "svm"]
    )
)
ensemble_metrics = ensemble.train(training_features, validate=True)
```

## Documentation

### Classifier-Specific Guides
- [Logistic Regression](logistic-classifier.md) - Fast baseline classifier
- [K-Nearest Neighbors](knn-classifier.md) - Instance-based learning
- [Decision Tree](decisiontree-classifier.md) - Interpretable rules
- [Support Vector Machine](svm-classifier.md) - Optimal boundaries
- [Random Forest](rf-classifier.md) - Robust ensemble
- [XGBoost](xgboost-classifier.md) - Gradient boosting
- [MLP Neural Network](mlp-classifier.md) - Deep learning
- [Ensemble](ensemble-classifier.md) - Meta-classifier

### General Documentation
- [Quick Start](quickstart.md) - Getting started guide
- [Design](design.md) - System architecture
- [Implementation](implementation.md) - Technical details
- [Summary](SUMMARY.md) - High-level overview

## Features

All classifiers support:
- ✅ Training with cross-validation
- ✅ Hyperparameter tuning
- ✅ Model persistence (save/load)
- ✅ Probability estimates
- ✅ Feature importance (where applicable)
- ✅ Comprehensive evaluation metrics
- ✅ Consistent API (IClassifier interface)

## Performance Benchmarks

Based on GAVD dataset (45 samples, 3 classes):

| Classifier | Accuracy | Training Time | Prediction Time |
|------------|----------|---------------|-----------------|
| Logistic | 85-90% | < 1s | < 1ms |
| KNN | 80-85% | < 1s | ~10ms |
| Decision Tree | 75-85% | < 1s | < 1ms |
| SVM | 90-95% | ~2s | ~5ms |
| Random Forest | 90-95% | ~3s | ~5ms |
| XGBoost | 92-97% | ~5s | ~5ms |
| MLP | 85-95% | ~10s | ~2ms |
| Ensemble | 93-98% | ~15s | ~20ms |

*Note: Times are approximate and depend on dataset size and hardware*

## Next Steps

1. Read the [quickstart guide](quickstart.md)
2. Choose a classifier based on your requirements
3. Review classifier-specific documentation
4. Explore [design](design.md) and [implementation](implementation.md) details
5. Check examples in `examples/` directory
