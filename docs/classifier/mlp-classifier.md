# MLP Neural Network Gait Classifier

## Overview

Multi-Layer Perceptron (MLP) is a feedforward neural network that can learn complex non-linear patterns. With sufficient data, MLPs can achieve excellent accuracy and support online learning for continuous model updates.

## Key Features

- **Deep learning**: Multiple hidden layers capture complex patterns
- **Non-linear**: Learns arbitrary decision boundaries
- **Online learning**: Supports incremental training
- **Flexible architecture**: Customizable layer sizes
- **Adaptive**: Can improve with more data
- **Regularization**: Dropout and L2 penalties

## When to Use

- Large datasets (1000+ samples)
- Complex non-linear patterns
- Online/incremental learning needed
- Sufficient computational resources
- Maximum accuracy priority

## Configuration

```python
from ambient.classification.mlp_classifier import (
    MLPGaitClassifier,
    MLPClassifierConfig
)

config = MLPClassifierConfig(
    hidden_layer_sizes=(100, 50),   # Two hidden layers
    activation="relu",               # 'relu', 'tanh', 'logistic'
    solver="adam",                   # 'adam', 'sgd', 'lbfgs'
    alpha=0.0001,                    # L2 regularization
    learning_rate="adaptive",        # Learning rate schedule
    max_iter=200,                    # Training epochs
    early_stopping=True,             # Stop when validation plateaus
    validation_fraction=0.1,         # Validation set size
    normalize_features=True,
    random_state=42
)

classifier = MLPGaitClassifier(config)
```

## Usage Example

```python
# Train with early stopping
classifier = MLPGaitClassifier()
metrics = classifier.train(training_features, validate=True)

print(f"Training iterations: {metrics['n_iterations']}")
print(f"Final loss: {metrics['loss']:.4f}")

# Online learning
new_samples = [...]  # New training data
classifier.partial_fit(new_samples)

# Classify
result = classifier.classify_gait(test_feature)
```

## Architecture Design

```python
# Small network (fast, less prone to overfitting)
config = MLPClassifierConfig(hidden_layer_sizes=(50,))

# Medium network (balanced)
config = MLPClassifierConfig(hidden_layer_sizes=(100, 50))

# Large network (high capacity, needs more data)
config = MLPClassifierConfig(hidden_layer_sizes=(200, 100, 50))
```

## Online Learning

```python
# Initial training
classifier.train(initial_features)

# Incremental updates
for new_batch in data_stream:
    classifier.partial_fit(new_batch)
    
# Model continuously improves
```

## Performance Characteristics

- **Training Time**: Slow (minutes for large datasets)
- **Prediction Time**: Fast
- **Memory Usage**: Moderate to high
- **Accuracy**: Excellent with sufficient data
- **Interpretability**: Low (black box)

## Tips and Best Practices

1. **Start small**: Begin with one hidden layer
2. **Use early stopping**: Prevents overfitting
3. **Normalize features**: Essential for neural networks
4. **Tune learning rate**: Critical for convergence
5. **Monitor loss**: Should decrease steadily
6. **Use validation set**: Track generalization
7. **Regularize**: Increase alpha if overfitting

## Common Issues

### Not Converging
- **Solution**: Increase max_iter, adjust learning_rate, normalize features

### Overfitting
- **Solution**: Increase alpha, reduce network size, use early_stopping

### Poor Performance
- **Solution**: Increase network size, train longer, check data quality

## Comparison with Other Classifiers

| Aspect | MLP | XGBoost | SVM | Logistic |
|--------|-----|---------|-----|----------|
| Accuracy (large data) | ★★★★★ | ★★★★★ | ★★★★☆ | ★★★☆☆ |
| Accuracy (small data) | ★★☆☆☆ | ★★★★★ | ★★★★☆ | ★★★☆☆ |
| Training time | ★★☆☆☆ | ★★★☆☆ | ★★★☆☆ | ★★★★★ |
| Interpretability | ★☆☆☆☆ | ★★★☆☆ | ★★☆☆☆ | ★★★★★ |
| Online learning | ★★★★★ | ★☆☆☆☆ | ★☆☆☆☆ | ★★★☆☆ |

## References

- [Scikit-learn MLP](https://scikit-learn.org/stable/modules/neural_networks_supervised.html)
- Goodfellow, I., et al. (2016). Deep Learning. MIT Press.
