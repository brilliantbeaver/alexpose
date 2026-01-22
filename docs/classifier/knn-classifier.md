# K-Nearest Neighbors Gait Classifier

## Overview

The K-Nearest Neighbors (KNN) classifier is a simple, instance-based learning algorithm that classifies samples based on the majority vote of their k nearest neighbors. It's non-parametric and makes no assumptions about the underlying data distribution.

## Key Features

- **Simple and intuitive**: Easy to understand and implement
- **No training phase**: Lazy learning algorithm
- **Non-parametric**: No assumptions about data distribution
- **Naturally handles multi-class**: No special modifications needed
- **Flexible distance metrics**: Euclidean, Manhattan, Minkowski
- **Adaptive**: Automatically adjusts to local patterns

## When to Use

- Small to medium datasets (< 10,000 samples)
- Quick prototyping and baseline
- Non-linear decision boundaries
- Multi-modal class distributions
- When training time is not critical

## Configuration

```python
from ambient.classification.knn_classifier import (
    KNNGaitClassifier,
    KNNClassifierConfig
)

config = KNNClassifierConfig(
    n_neighbors=5,              # Number of neighbors
    weights="distance",         # 'uniform' or 'distance'
    metric="euclidean",         # Distance metric
    algorithm="auto",           # 'auto', 'ball_tree', 'kd_tree', 'brute'
    leaf_size=30,               # Leaf size for tree algorithms
    normalize_features=True,    # Essential for KNN
    random_state=42
)

classifier = KNNGaitClassifier(config)
```

## Usage Example

```python
from ambient.classification.knn_classifier import KNNGaitClassifier

# Create classifier
classifier = KNNGaitClassifier()

# Train (stores training data)
metrics = classifier.train(training_features, validate=True)

# Classify
result = classifier.classify_gait(test_feature)
print(f"Predicted: {result['predicted_condition']}")
print(f"Confidence: {result['confidence']:.2%}")

# Get nearest neighbors
neighbors = classifier.get_nearest_neighbors(test_feature, n_neighbors=5)
for i, neighbor in enumerate(neighbors):
    print(f"Neighbor {i+1}: {neighbor['condition']} (distance: {neighbor['distance']:.3f})")
```

## Choosing K

```python
# Small k (1-3): More sensitive to noise, complex boundaries
config = KNNClassifierConfig(n_neighbors=1)

# Medium k (5-10): Good balance, default choice
config = KNNClassifierConfig(n_neighbors=5)

# Large k (15-30): Smoother boundaries, more robust
config = KNNClassifierConfig(n_neighbors=15)
```

## Distance Weighting

```python
# Uniform: All neighbors have equal vote
config = KNNClassifierConfig(weights="uniform")

# Distance: Closer neighbors have more influence (recommended)
config = KNNClassifierConfig(weights="distance")
```

## Hyperparameter Tuning

```python
param_grid = {
    "n_neighbors": [3, 5, 7, 9, 11, 15],
    "weights": ["uniform", "distance"],
    "metric": ["euclidean", "manhattan", "minkowski"],
}

results = classifier.tune_hyperparameters(
    training_features,
    param_grid=param_grid,
    cv_folds=5
)

print(f"Best k: {results['best_params']['n_neighbors']}")
print(f"Best score: {results['best_score']:.3f}")
```

## Performance Characteristics

- **Training Time**: Very fast (just stores data)
- **Prediction Time**: Slow (O(n) for each prediction)
- **Memory Usage**: High (stores all training data)
- **Accuracy**: Good for local patterns
- **Interpretability**: High (can show nearest neighbors)

## Tips and Best Practices

1. **Always normalize features**: KNN is very sensitive to feature scales
2. **Use odd k**: Avoids ties in binary classification
3. **Cross-validate k**: Find optimal number of neighbors
4. **Use distance weighting**: Usually improves performance
5. **Consider dimensionality**: Performance degrades in high dimensions
6. **Reduce dataset size**: For faster predictions
7. **Use ball_tree or kd_tree**: For large datasets

## Common Issues

### Slow Predictions
- **Solution**: Reduce training set size, use ball_tree/kd_tree algorithm

### Poor Performance
- **Solution**: Tune k, normalize features, try different distance metrics

### Memory Issues
- **Solution**: Reduce training set size, use approximate methods

## Distance Metrics

- **Euclidean**: Standard choice, L2 norm
- **Manhattan**: L1 norm, less sensitive to outliers
- **Minkowski**: Generalization of Euclidean and Manhattan
- **Cosine**: For directional similarity

## Comparison with Other Classifiers

| Aspect | KNN | SVM | XGBoost | Logistic |
|--------|-----|-----|---------|----------|
| Training speed | ★★★★★ | ★★★☆☆ | ★★★☆☆ | ★★★★★ |
| Prediction speed | ★★☆☆☆ | ★★★★☆ | ★★★★☆ | ★★★★★ |
| Memory usage | ★☆☆☆☆ | ★★★★☆ | ★★★☆☆ | ★★★★★ |
| Interpretability | ★★★★☆ | ★★☆☆☆ | ★★★☆☆ | ★★★★★ |
| Non-linearity | ★★★★☆ | ★★★★☆ | ★★★★★ | ★☆☆☆☆ |

## Curse of Dimensionality

KNN performance degrades in high-dimensional spaces:
- Distances become less meaningful
- All points become equidistant
- Solution: Feature selection, dimensionality reduction (PCA)

## References

- [Scikit-learn KNN](https://scikit-learn.org/stable/modules/neighbors.html)
- Cover, T., & Hart, P. (1967). Nearest neighbor pattern classification.
