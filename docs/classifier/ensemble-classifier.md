# Ensemble Gait Classifier

## Overview

The Ensemble classifier combines predictions from multiple base classifiers using voting or stacking strategies. By leveraging the diverse strengths of different algorithms, ensembles typically achieve better accuracy and robustness than individual classifiers.

## Key Features

- **Higher accuracy**: Combines strengths of multiple models
- **Reduced overfitting**: Diverse models compensate for each other
- **Robust predictions**: Less sensitive to individual model failures
- **Flexible strategies**: Hard, soft, and weighted voting
- **Model diversity**: Combines KNN, RF, XGBoost, SVM, Logistic, Decision Tree, MLP
- **Prediction breakdown**: See individual classifier contributions

## When to Use

- Maximum accuracy is critical
- Robustness to edge cases needed
- Multiple models already trained
- Computational resources available
- Production deployment with high stakes

## Configuration

```python
from ambient.classification.ensemble_classifier import (
    EnsembleGaitClassifier,
    EnsembleClassifierConfig,
    VotingStrategy
)

config = EnsembleClassifierConfig(
    voting_strategy=VotingStrategy.SOFT,    # HARD, SOFT, or WEIGHTED
    classifiers=["rf", "xgboost", "svm", "logistic"],
    classifier_weights=None,                 # Auto-weight by performance
    min_agreement=0.5,                       # Confidence threshold
    normalize_features=True,
    random_state=42
)

classifier = EnsembleGaitClassifier(config)
```

## Voting Strategies

### Hard Voting (Majority Vote)
```python
config = EnsembleClassifierConfig(
    voting_strategy=VotingStrategy.HARD,
    classifiers=["rf", "xgboost", "svm"]
)
# Prediction: class with most votes
```

### Soft Voting (Average Probabilities)
```python
config = EnsembleClassifierConfig(
    voting_strategy=VotingStrategy.SOFT,
    classifiers=["rf", "xgboost", "svm"]
)
# Prediction: class with highest average probability
```

### Weighted Voting (Performance-Based)
```python
config = EnsembleClassifierConfig(
    voting_strategy=VotingStrategy.WEIGHTED,
    classifiers=["rf", "xgboost", "svm"],
    classifier_weights={"rf": 0.3, "xgboost": 0.5, "svm": 0.2}
)
# Prediction: weighted average of probabilities
```

## Usage Example

```python
# Create ensemble with multiple classifiers
ensemble = EnsembleGaitClassifier(
    EnsembleClassifierConfig(
        voting_strategy=VotingStrategy.SOFT,
        classifiers=["rf", "xgboost", "svm", "logistic"]
    )
)

# Train all base classifiers
metrics = ensemble.train(training_features, validate=True)

print(f"Ensemble trained with {metrics['n_classifiers']} classifiers")
for clf_name, clf_metrics in metrics['classifiers'].items():
    print(f"{clf_name}: {clf_metrics['train_accuracy']:.3f}")

# Classify with ensemble
result = ensemble.classify_gait(test_feature)

print(f"Ensemble prediction: {result['predicted_condition']}")
print(f"Confidence: {result['confidence']:.2%}")
print(f"Agreement: {result['agreement']:.2%}")

# See individual predictions
for clf_name, pred in result['individual_predictions'].items():
    print(f"{clf_name}: {pred['predicted_condition']} ({pred['confidence']:.2f})")

# Get detailed breakdown
breakdown = ensemble.get_prediction_breakdown(test_feature)
for clf_name, pred in breakdown.items():
    print(f"\n{clf_name}:")
    print(f"  Prediction: {pred['predicted_condition']}")
    print(f"  Confidence: {pred['confidence']:.2%}")
    print(f"  Weight: {pred['weight']:.2f}")
```

## Classifier Selection

Choose classifiers based on diversity and performance:

```python
# Fast ensemble (quick predictions)
config = EnsembleClassifierConfig(
    classifiers=["logistic", "decisiontree", "knn"]
)

# Accurate ensemble (best performance)
config = EnsembleClassifierConfig(
    classifiers=["rf", "xgboost", "svm", "mlp"]
)

# Balanced ensemble (speed + accuracy)
config = EnsembleClassifierConfig(
    classifiers=["rf", "xgboost", "logistic"]
)

# All classifiers (maximum diversity)
config = EnsembleClassifierConfig(
    classifiers=["knn", "rf", "xgboost", "svm", "logistic", "decisiontree", "mlp"]
)
```

## Performance Characteristics

- **Training Time**: Sum of all base classifiers
- **Prediction Time**: Sum of all base classifiers
- **Memory Usage**: Sum of all base classifiers
- **Accuracy**: Typically best among all methods
- **Interpretability**: Low (multiple models)

## Agreement Analysis

```python
result = ensemble.classify_gait(test_feature)

# High agreement (>0.8): Very confident prediction
# Medium agreement (0.5-0.8): Moderate confidence
# Low agreement (<0.5): Uncertain, review manually

if result['agreement'] < 0.5:
    print("Warning: Low classifier agreement")
    print("Individual predictions:")
    for clf, pred in result['individual_predictions'].items():
        print(f"  {clf}: {pred['predicted_condition']}")
```

## Model Persistence

```python
# Save ensemble (saves all base classifiers)
ensemble.save("models/ensemble_gait_classifier.pkl")

# Load ensemble
loaded_ensemble = EnsembleGaitClassifier.load("models/ensemble_gait_classifier.pkl")
```

## Tips and Best Practices

1. **Use diverse classifiers**: Different algorithms capture different patterns
2. **Soft voting preferred**: Usually outperforms hard voting
3. **Weight by performance**: Use WEIGHTED strategy with validation accuracy
4. **Monitor agreement**: Low agreement indicates uncertainty
5. **Balance speed vs accuracy**: Fewer classifiers = faster predictions
6. **Cross-validate**: Assess true ensemble performance
7. **Update weights**: Retrain and reweight periodically

## Common Issues

### Slow Predictions
- **Solution**: Reduce number of classifiers, use faster base models

### No Improvement Over Best Base Classifier
- **Solution**: Ensure classifier diversity, check for correlated predictions

### High Memory Usage
- **Solution**: Remove redundant classifiers, use simpler base models

## Comparison with Base Classifiers

Ensemble typically provides:
- **+2-5% accuracy** over best base classifier
- **Better calibration** of probability estimates
- **More robust** to outliers and edge cases
- **Higher confidence** in predictions with high agreement

## Advanced: Custom Weights

```python
# Weight by validation performance
val_accuracies = {
    "rf": 0.92,
    "xgboost": 0.95,
    "svm": 0.88,
    "logistic": 0.85
}

# Normalize to sum to 1
total = sum(val_accuracies.values())
weights = {k: v/total for k, v in val_accuracies.items()}

config = EnsembleClassifierConfig(
    voting_strategy=VotingStrategy.WEIGHTED,
    classifiers=list(weights.keys()),
    classifier_weights=weights
)
```

## References

- [Scikit-learn Ensemble Methods](https://scikit-learn.org/stable/modules/ensemble.html)
- Dietterich, T. G. (2000). Ensemble Methods in Machine Learning.
- Zhou, Z. H. (2012). Ensemble Methods: Foundations and Algorithms.
