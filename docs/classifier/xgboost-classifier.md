# XGBoost Gait Classifier

## Overview

The XGBoost (eXtreme Gradient Boosting) classifier uses gradient boosting with decision trees to achieve state-of-the-art accuracy on gait classification tasks. It often outperforms Random Forest through sequential tree building and advanced regularization techniques.

## Key Features

- **State-of-the-art accuracy**: Sequential boosting improves on weak learners
- **Built-in regularization**: L1/L2 penalties prevent overfitting
- **Handles missing values**: Robust to incomplete data
- **Feature importance**: Identifies most discriminative gait features
- **GPU support**: Fast training on large datasets (optional)
- **Early stopping**: Prevents overfitting during training

## When to Use

XGBoost is ideal when:
- Maximum accuracy is the priority
- You have sufficient training data (100+ samples)
- Feature interactions are important
- You need feature importance rankings
- Training time is not critical

## Configuration

```python
from ambient.classification.xgboost_classifier import (
    XGBoostGaitClassifier,
    XGBoostClassifierConfig
)

# Default configuration
config = XGBoostClassifierConfig(
    n_estimators=100,           # Number of boosting rounds
    max_depth=6,                # Maximum tree depth
    learning_rate=0.3,          # Step size shrinkage
    subsample=0.8,              # Fraction of samples per tree
    colsample_bytree=0.8,       # Fraction of features per tree
    reg_alpha=0.0,              # L1 regularization
    reg_lambda=1.0,             # L2 regularization
    min_child_weight=1,         # Minimum sum of instance weight
    gamma=0.0,                  # Minimum loss reduction for split
    normalize_features=True,    # Standardize features
    random_state=42             # Reproducibility
)

classifier = XGBoostGaitClassifier(config)
```

## Usage Example

```python
from ambient.classification.xgboost_classifier import XGBoostGaitClassifier
from ambient.classification.features import GaitFeatureVector

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
print(f"Training accuracy: {metrics['train_accuracy']:.3f}")
print(f"CV accuracy: {metrics['cv_mean_accuracy']:.3f} ± {metrics['cv_std_accuracy']:.3f}")

# Classify new sample
test_feature = GaitFeatureVector(...)
result = classifier.classify_gait(test_feature)

print(f"Predicted: {result['predicted_condition']}")
print(f"Confidence: {result['confidence']:.2%}")
print(f"Probabilities: {result['probabilities']}")

# Get feature importance
importances = classifier.get_feature_importances(top_n=5)
for imp in importances:
    print(f"{imp.feature_name}: {imp.importance:.3f}")
```

## Hyperparameter Tuning

```python
# Define parameter grid
param_grid = {
    "n_estimators": [50, 100, 200],
    "max_depth": [3, 5, 7],
    "learning_rate": [0.01, 0.1, 0.3],
    "subsample": [0.8, 1.0],
    "colsample_bytree": [0.8, 1.0],
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

## Feature Importance

XGBoost provides feature importance based on how often features are used for splitting:

```python
# Get all feature importances
importances = classifier.get_feature_importances()

# Display top features
print("Top 10 Important Features:")
for imp in importances[:10]:
    print(f"{imp.rank}. {imp.feature_name}: {imp.importance:.3f}")
```

## Model Persistence

```python
# Save trained model
classifier.save("models/xgboost_gait_classifier.pkl")

# Load model
loaded_classifier = XGBoostGaitClassifier.load("models/xgboost_gait_classifier.pkl")
```

## Performance Characteristics

- **Training Time**: Moderate (slower than Logistic, faster than extensive grid search)
- **Prediction Time**: Fast (milliseconds per sample)
- **Memory Usage**: Moderate (stores ensemble of trees)
- **Accuracy**: Excellent (typically highest among tree-based methods)
- **Interpretability**: Moderate (feature importance available)

## Comparison with Other Classifiers

| Aspect | XGBoost | Random Forest | SVM | Logistic |
|--------|---------|---------------|-----|----------|
| Accuracy | ★★★★★ | ★★★★☆ | ★★★★☆ | ★★★☆☆ |
| Speed | ★★★☆☆ | ★★★★☆ | ★★★☆☆ | ★★★★★ |
| Interpretability | ★★★☆☆ | ★★★☆☆ | ★★☆☆☆ | ★★★★★ |
| Robustness | ★★★★★ | ★★★★☆ | ★★★☆☆ | ★★★☆☆ |

## Tips and Best Practices

1. **Start with defaults**: XGBoost's default parameters work well for most cases
2. **Tune learning rate**: Lower values (0.01-0.1) with more estimators often work better
3. **Use early stopping**: Prevents overfitting on validation set
4. **Monitor feature importance**: Helps understand which gait features matter most
5. **Balance regularization**: Increase `reg_alpha` or `reg_lambda` if overfitting
6. **Subsample data**: Values like 0.8 can improve generalization
7. **Cross-validate**: Always use CV to assess true performance

## Common Issues

### Overfitting
- **Symptom**: High training accuracy, low test accuracy
- **Solution**: Increase regularization (`reg_alpha`, `reg_lambda`), reduce `max_depth`, increase `min_child_weight`

### Slow Training
- **Symptom**: Training takes too long
- **Solution**: Reduce `n_estimators`, increase `learning_rate`, use GPU acceleration

### Poor Performance
- **Symptom**: Low accuracy on all sets
- **Solution**: Increase `n_estimators`, tune `max_depth`, check data quality

## References

- [XGBoost Documentation](https://xgboost.readthedocs.io/)
- Chen, T., & Guestrin, C. (2016). XGBoost: A Scalable Tree Boosting System. KDD.
