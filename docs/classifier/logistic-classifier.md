# Logistic Regression Gait Classifier

## Overview

Logistic Regression provides fast, interpretable baseline predictions using linear models. Despite its simplicity, it often performs surprisingly well and serves as an excellent baseline for comparison. The classifier uses regularization to prevent overfitting and handles multi-class problems efficiently.

## Key Features

- **Very fast**: Training and prediction in milliseconds
- **Probabilistic**: Natural probability interpretation
- **Interpretable**: Feature coefficients show importance and direction
- **Low memory**: Minimal storage requirements
- **Regularization**: L1/L2 penalties prevent overfitting
- **Baseline**: Excellent reference for more complex models

## When to Use

Logistic Regression is ideal when:
- You need fast training and prediction
- Interpretability is crucial (clinical decisions)
- Establishing performance baselines
- Real-time prediction requirements
- Understanding linear feature relationships
- Limited computational resources

## Configuration

```python
from ambient.classification.logistic_classifier import (
    LogisticGaitClassifier,
    LogisticClassifierConfig
)

config = LogisticClassifierConfig(
    penalty="l2",               # 'l1', 'l2', 'elasticnet', 'none'
    C=1.0,                      # Inverse regularization strength
    solver="lbfgs",             # Optimization algorithm
    max_iter=1000,              # Maximum iterations
    class_weight="balanced",    # Handle imbalanced classes
    normalize_features=True,
    random_state=42
)

classifier = LogisticGaitClassifier(config)
```

## Usage Example

```python
from ambient.classification.logistic_classifier import LogisticGaitClassifier

# Create and train
classifier = LogisticGaitClassifier()
metrics = classifier.train(training_features, validate=True)

print(f"Training accuracy: {metrics['train_accuracy']:.3f}")
print(f"Top features: {metrics['top_features']}")

# Classify
result = classifier.classify_gait(test_feature)
print(f"Predicted: {result['predicted_condition']}")
print(f"Confidence: {result['confidence']:.2%}")

# Get feature coefficients
coefficients = classifier.get_feature_coefficients(top_n=10)
for coef in coefficients:
    print(f"{coef.feature_name}: {coef.coefficient:+.4f}")

# Get detailed interpretation
interpretation = classifier.get_model_interpretation()
print(f"Feature importance: {interpretation['feature_importance']}")
```

## Hyperparameter Tuning

```python
param_grid = {
    "C": [0.001, 0.01, 0.1, 1, 10, 100],
    "penalty": ["l1", "l2"],
    "solver": ["liblinear", "saga"],  # Support both L1 and L2
}

results = classifier.tune_hyperparameters(
    training_features,
    param_grid=param_grid,
    cv_folds=5
)
```

## Feature Interpretation

Logistic Regression provides the most interpretable feature analysis:

```python
# Get coefficients with interpretation
coefficients = classifier.get_feature_coefficients()

for coef in coefficients[:5]:
    direction = "increases" if coef.coefficient > 0 else "decreases"
    print(f"{coef.feature_name} {direction} probability of condition")
    print(f"  Coefficient: {coef.coefficient:+.4f}")
    print(f"  Importance: {coef.abs_coefficient:.4f}")
```

## Regularization

- **L2 (Ridge)**: Shrinks all coefficients, good default
- **L1 (Lasso)**: Performs feature selection, sets some coefficients to zero
- **ElasticNet**: Combination of L1 and L2
- **None**: No regularization (use with caution)

## Performance Characteristics

- **Training Time**: Very fast (seconds for thousands of samples)
- **Prediction Time**: Extremely fast (microseconds per sample)
- **Memory Usage**: Minimal (only coefficients stored)
- **Accuracy**: Good for linearly separable data
- **Interpretability**: Excellent (clear feature contributions)

## Tips and Best Practices

1. **Always normalize features**: Essential for coefficient interpretation
2. **Start with L2 regularization**: Good default choice
3. **Tune C parameter**: Controls regularization strength
4. **Use as baseline**: Compare other models against it
5. **Interpret coefficients**: Understand feature relationships
6. **Check convergence**: Increase max_iter if needed
7. **Balance classes**: Use class_weight='balanced' for imbalanced data

## Common Issues

### Convergence Warnings
- **Solution**: Increase `max_iter`, normalize features, or adjust `tol`

### Poor Performance
- **Solution**: Data may not be linearly separable, try polynomial features or switch to non-linear model

### Coefficients Too Large
- **Solution**: Decrease C (increase regularization), check feature scaling

## Comparison with Other Classifiers

| Aspect | Logistic | SVM | XGBoost | MLP |
|--------|----------|-----|---------|-----|
| Speed | ★★★★★ | ★★★☆☆ | ★★★☆☆ | ★★☆☆☆ |
| Interpretability | ★★★★★ | ★★☆☆☆ | ★★★☆☆ | ★☆☆☆☆ |
| Non-linearity | ★☆☆☆☆ | ★★★★☆ | ★★★★★ | ★★★★★ |
| Simplicity | ★★★★★ | ★★★☆☆ | ★★☆☆☆ | ★★☆☆☆ |

## Clinical Applications

Logistic Regression is particularly valuable in clinical settings because:
- Coefficients can be explained to clinicians
- Feature contributions are transparent
- Fast enough for real-time screening
- Provides calibrated probabilities
- Regulatory compliance (explainability)

## References

- [Scikit-learn Logistic Regression](https://scikit-learn.org/stable/modules/linear_model.html#logistic-regression)
- Hosmer, D. W., & Lemeshow, S. (2000). Applied Logistic Regression.
