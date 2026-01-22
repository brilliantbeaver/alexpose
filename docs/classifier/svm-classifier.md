# SVM Gait Classifier

## Overview

The Support Vector Machine (SVM) classifier with RBF kernel finds optimal decision boundaries in high-dimensional feature spaces. SVMs are particularly effective for gait patterns due to their ability to capture non-linear relationships while maintaining computational efficiency.

## Key Features

- **Optimal boundaries**: Maximizes margin between classes
- **Kernel trick**: Maps to infinite-dimensional space efficiently
- **Memory efficient**: Uses only support vectors
- **Robust to outliers**: Focus on boundary samples
- **Probabilistic output**: Calibrated probability estimates
- **Multi-class support**: One-vs-rest or one-vs-one strategies

## When to Use

SVM is ideal when:
- You have small to medium datasets (< 10,000 samples)
- Decision boundaries are non-linear
- You need probabilistic predictions
- Interpretability of support vectors is valuable
- Memory efficiency is important

## Configuration

```python
from ambient.classification.svm_classifier import (
    SVMGaitClassifier,
    SVMClassifierConfig
)

config = SVMClassifierConfig(
    kernel="rbf",               # 'linear', 'poly', 'rbf', 'sigmoid'
    C=1.0,                      # Regularization parameter
    gamma="scale",              # Kernel coefficient
    degree=3,                   # Degree for poly kernel
    class_weight="balanced",    # Handle imbalanced classes
    probability=True,           # Enable probability estimates
    max_iter=-1,                # No iteration limit
    normalize_features=True,
    random_state=42
)

classifier = SVMGaitClassifier(config)
```

## Usage Example

```python
from ambient.classification.svm_classifier import SVMGaitClassifier

# Create and train
classifier = SVMGaitClassifier()
metrics = classifier.train(training_features, validate=True)

# Classify
result = classifier.classify_gait(test_feature)
print(f"Predicted: {result['predicted_condition']}")
print(f"Decision values: {result['decision_values']}")

# Get support vector info
sv_info = classifier.get_support_vector_info()
print(f"Support vectors: {sv_info['n_support_vectors']}")
print(f"Per class: {sv_info['support_vectors_per_class']}")
```

## Hyperparameter Tuning

```python
param_grid = {
    "C": [0.1, 1, 10, 100],
    "gamma": ["scale", "auto", 0.001, 0.01, 0.1],
    "kernel": ["rbf", "poly"],
}

results = classifier.tune_hyperparameters(
    training_features,
    param_grid=param_grid,
    cv_folds=5
)
```

## Kernel Selection

- **RBF (Radial Basis Function)**: Default, works well for most cases
- **Linear**: Fast, good for linearly separable data
- **Polynomial**: Captures polynomial relationships
- **Sigmoid**: Similar to neural networks

## Performance Characteristics

- **Training Time**: Moderate to slow (O(n²) to O(n³))
- **Prediction Time**: Fast (depends on support vectors)
- **Memory Usage**: Low (only support vectors stored)
- **Accuracy**: Excellent for non-linear patterns
- **Interpretability**: Moderate (support vectors, decision values)

## Tips and Best Practices

1. **Always normalize features**: SVM is sensitive to feature scales
2. **Start with RBF kernel**: Works well as default
3. **Tune C and gamma together**: They interact significantly
4. **Use class_weight='balanced'**: For imbalanced datasets
5. **Monitor support vectors**: Too many suggests overfitting
6. **Cross-validate**: Essential for parameter selection

## Common Issues

### Slow Training
- **Solution**: Reduce dataset size, use linear kernel, or switch to SGDClassifier

### Poor Performance
- **Solution**: Tune C and gamma, try different kernels, check feature scaling

### Too Many Support Vectors
- **Solution**: Increase C (stronger regularization), simplify kernel

## References

- [Scikit-learn SVM Documentation](https://scikit-learn.org/stable/modules/svm.html)
- Cortes, C., & Vapnik, V. (1995). Support-vector networks. Machine Learning.
