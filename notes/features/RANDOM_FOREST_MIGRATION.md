# Random Forest Classifier Migration

## Overview

This document describes the migration from KNN (K-Nearest Neighbors) to Random Forest classifier in the GAVD gait analysis experiment.

## Changes Made

### 1. Updated Imports
**File:** `experiments/exp4/02_train_KNN.ipynb`

Changed from:
```python
from ambient.classification.knn_classifier import (
    KNNGaitClassifier,
    KNNClassifierConfig,
    GaitFeatureVector
)
```

To:
```python
from ambient.classification.rf_classifier import (
    RFGaitClassifier,
    RFClassifierConfig,
    GaitFeatureVector
)
```

### 2. Updated Configuration

**Old KNN Configuration:**
```python
config = KNNClassifierConfig(
    n_neighbors=5,
    weights="distance",
    metric="euclidean",
    normalize_features=True
)
```

**New Random Forest Configuration:**
```python
config = RFClassifierConfig(
    n_estimators=100,           # Number of decision trees in the forest
    max_depth=None,             # No limit on tree depth (trees grow until pure)
    min_samples_split=2,        # Minimum samples to split an internal node
    min_samples_leaf=1,         # Minimum samples required at a leaf node
    max_features="sqrt",        # Use sqrt(n_features) for each split
    bootstrap=True,             # Use bootstrap sampling for building trees
    class_weight="balanced",    # Handle imbalanced classes automatically
    normalize_features=True,    # Standardize features before training
    random_state=42             # For reproducible results
)
```

### 3. Updated Classifier Initialization

**Old:**
```python
classifier = KNNGaitClassifier(config=config)
```

**New:**
```python
classifier = RFGaitClassifier(config=config)
```

### 4. Enhanced Training Output

The training output now includes:
- Training accuracy
- Number of samples and features
- Number of trees in the forest
- Class distribution
- Cross-validation results (mean and std deviation)
- **Top 5 most important features** (new!)

## Why Random Forest?

### Advantages over KNN:

1. **Better Performance**: Random Forest typically provides better accuracy on complex datasets
2. **Feature Importance**: Provides insights into which features contribute most to classification
3. **Handles Non-linearity**: Better at capturing complex, non-linear relationships in data
4. **Robust to Outliers**: Less sensitive to noisy data and outliers
5. **No Distance Metric Required**: Doesn't require choosing appropriate distance metrics
6. **Handles Imbalanced Data**: Built-in class weighting for imbalanced datasets
7. **Less Prone to Overfitting**: Ensemble method reduces overfitting compared to single decision trees

### Key Random Forest Parameters:

- **n_estimators**: Number of trees (more = better accuracy but slower)
- **max_depth**: Maximum tree depth (None = unlimited, helps prevent overfitting)
- **min_samples_split**: Minimum samples to split a node (higher = more conservative)
- **min_samples_leaf**: Minimum samples at leaf nodes (higher = smoother boundaries)
- **max_features**: Features considered per split ('sqrt' is recommended)
- **class_weight**: 'balanced' automatically handles imbalanced classes
- **bootstrap**: Use bootstrap sampling for building diverse trees
- **random_state**: Ensures reproducible results

## Feature Importance

One of the key advantages of Random Forest is feature importance analysis. After training, you can see which gait features contribute most to classification:

```python
# Get feature importances
importances = classifier.get_feature_importances()

# Display top 5
for imp in importances[:5]:
    print(f"{imp.rank}. {imp.feature_name}: {imp.importance:.4f}")
```

This helps understand:
- Which joint angles are most discriminative
- Which asymmetry measures matter most
- Which range of motion features are critical

## Usage Example

```python
# Train the classifier
classifier = RFGaitClassifier(config=config)
metrics = classifier.train(train_features, validate=True)

# Classify new gait sample
result = classifier.classify_gait(test_features)
print(f"Predicted: {result['predicted_condition']}")
print(f"Confidence: {result['confidence']:.2f}")

# Get explanation
explanation = classifier.explain_classification(result)
print(explanation)

# Evaluate on test set
test_metrics = classifier.evaluate(test_features, test_labels)
print(f"Test Accuracy: {test_metrics['accuracy']:.3f}")

# Save trained model
classifier.save("models/rf_gait_classifier.pkl")

# Load later
loaded_classifier = RFGaitClassifier.load("models/rf_gait_classifier.pkl")
```

## Hyperparameter Tuning

Random Forest supports automated hyperparameter tuning:

```python
# Define parameter grid
param_grid = {
    "n_estimators": [50, 100, 200],
    "max_depth": [None, 10, 20, 30],
    "min_samples_split": [2, 5, 10],
    "min_samples_leaf": [1, 2, 4],
    "max_features": ["sqrt", "log2"],
}

# Tune hyperparameters
results = classifier.tune_hyperparameters(
    features=train_features,
    labels=train_labels,
    param_grid=param_grid,
    cv_folds=5
)

print(f"Best parameters: {results['best_params']}")
print(f"Best CV score: {results['best_score']:.3f}")
```

## Migration Checklist

- [x] Update imports from knn_classifier to rf_classifier
- [x] Replace KNNClassifierConfig with RFClassifierConfig
- [x] Replace KNNGaitClassifier with RFGaitClassifier
- [x] Update configuration parameters
- [x] Enhance training output display
- [x] Add feature importance display
- [x] Update notebook title
- [x] Add comprehensive documentation

## Next Steps

1. Run the updated notebook to train the Random Forest classifier
2. Compare performance metrics with previous KNN results
3. Analyze feature importances to understand key gait characteristics
4. Consider hyperparameter tuning for optimal performance
5. Evaluate on test set and compare with KNN baseline

## References

- Random Forest Classifier Implementation: `ambient/classification/rf_classifier.py`
- KNN Classifier (for comparison): `ambient/classification/knn_classifier.py`
- Example Usage: `examples/rf_classifier_example.py`
- Documentation: `docs/classifier/rf-classifier.md`
