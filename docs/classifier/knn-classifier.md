# K-Nearest Neighbors (KNN) Classifier

The KNN classifier provides a simple yet effective approach to gait pattern classification using distance-based similarity matching. This non-parametric method is particularly useful for small to medium-sized datasets and provides interpretable results.

## Overview

The KNN classifier works by:
1. Storing all training examples in memory
2. For each new sample, finding the K nearest neighbors in feature space
3. Classifying based on majority vote of neighbors
4. Providing confidence scores based on neighbor distances

## Architecture

```python
from ambient.classification.knn_classifier import KNNClassifier

# Initialize classifier
classifier = KNNClassifier(
    n_neighbors=5,
    weights='distance',
    metric='euclidean'
)

# Train on gait features
classifier.train(X_train, y_train)

# Predict gait patterns
predictions = classifier.predict(X_test)
```

## Features

### Distance Metrics
- **Euclidean**: Standard L2 distance (default)
- **Manhattan**: L1 distance for sparse features
- **Minkowski**: Generalized distance metric
- **Cosine**: Similarity-based distance

### Weighting Schemes
- **Uniform**: All neighbors weighted equally
- **Distance**: Closer neighbors weighted more heavily
- **Custom**: User-defined weighting function

### Optimization
- **KD-Tree**: Fast nearest neighbor search for low dimensions
- **Ball Tree**: Efficient for high-dimensional data
- **Brute Force**: Exhaustive search for small datasets

## Configuration

### Basic Configuration

```yaml
# config/knn_config.yaml
knn_classifier:
  n_neighbors: 5
  weights: 'distance'
  metric: 'euclidean'
  algorithm: 'auto'
  leaf_size: 30
  p: 2  # Power parameter for Minkowski metric
```

### Advanced Configuration

```yaml
knn_classifier:
  n_neighbors: 7
  weights: 'distance'
  metric: 'minkowski'
  algorithm: 'ball_tree'
  leaf_size: 40
  p: 2
  
  # Feature preprocessing
  feature_scaling: 'standard'  # or 'minmax', 'robust'
  feature_selection: true
  n_features: 20
  
  # Cross-validation
  cv_folds: 5
  cv_scoring: 'accuracy'
  
  # Hyperparameter tuning
  tune_hyperparameters: true
  param_grid:
    n_neighbors: [3, 5, 7, 9, 11]
    weights: ['uniform', 'distance']
    metric: ['euclidean', 'manhattan']
```

## Usage Examples

### Basic Classification

```python
from ambient.classification.knn_classifier import KNNClassifier
from ambient.analysis.feature_extractor import FeatureExtractor

# Extract features from gait data
extractor = FeatureExtractor()
features = extractor.extract_features(pose_data)

# Initialize and train classifier
classifier = KNNClassifier(n_neighbors=5)
classifier.train(X_train, y_train)

# Make predictions
prediction = classifier.predict(features)
confidence = classifier.predict_proba(features)

print(f"Predicted class: {prediction}")
print(f"Confidence: {confidence}")
```

### With Feature Scaling

```python
from sklearn.preprocessing import StandardScaler

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train classifier
classifier = KNNClassifier(n_neighbors=5)
classifier.train(X_train_scaled, y_train)

# Predict
predictions = classifier.predict(X_test_scaled)
```

### Hyperparameter Tuning

```python
from sklearn.model_selection import GridSearchCV

# Define parameter grid
param_grid = {
    'n_neighbors': [3, 5, 7, 9, 11],
    'weights': ['uniform', 'distance'],
    'metric': ['euclidean', 'manhattan', 'minkowski']
}

# Perform grid search
classifier = KNNClassifier()
grid_search = GridSearchCV(
    classifier.model,
    param_grid,
    cv=5,
    scoring='accuracy',
    n_jobs=-1
)

grid_search.fit(X_train, y_train)

# Get best parameters
best_params = grid_search.best_params_
print(f"Best parameters: {best_params}")

# Use best model
best_classifier = KNNClassifier(**best_params)
best_classifier.train(X_train, y_train)
```

### Distance-Weighted Predictions

```python
# Use distance weighting for predictions
classifier = KNNClassifier(
    n_neighbors=7,
    weights='distance'  # Closer neighbors have more influence
)

classifier.train(X_train, y_train)

# Get predictions with confidence
predictions = classifier.predict(X_test)
probabilities = classifier.predict_proba(X_test)

# Get nearest neighbors for interpretation
distances, indices = classifier.get_neighbors(X_test[0])
print(f"Nearest neighbors: {indices}")
print(f"Distances: {distances}")
```

## Performance Characteristics

### Advantages
- **Simple and intuitive**: Easy to understand and explain
- **No training time**: Lazy learning approach
- **Non-parametric**: No assumptions about data distribution
- **Naturally multi-class**: Handles multiple classes without modification
- **Interpretable**: Can examine nearest neighbors for explanation

### Disadvantages
- **Slow prediction**: Must compute distances to all training samples
- **Memory intensive**: Stores all training data
- **Sensitive to scale**: Features must be normalized
- **Curse of dimensionality**: Performance degrades with many features
- **Sensitive to noise**: Outliers can affect predictions

### Computational Complexity
- **Training**: O(1) - just stores data
- **Prediction**: O(n × d) where n = training samples, d = dimensions
- **With KD-Tree**: O(d × log n) for low dimensions
- **Memory**: O(n × d) to store training data

## Choosing K (Number of Neighbors)

### Guidelines
- **Small K (3-5)**: More sensitive to noise, complex decision boundaries
- **Large K (10-20)**: Smoother decision boundaries, more robust to noise
- **Odd K**: Avoids ties in binary classification
- **Rule of thumb**: K = sqrt(n) where n is number of training samples

### Cross-Validation Approach

```python
from sklearn.model_selection import cross_val_score
import numpy as np

# Test different K values
k_values = range(1, 31, 2)
cv_scores = []

for k in k_values:
    classifier = KNNClassifier(n_neighbors=k)
    scores = cross_val_score(
        classifier.model,
        X_train,
        y_train,
        cv=5,
        scoring='accuracy'
    )
    cv_scores.append(scores.mean())

# Find optimal K
optimal_k = k_values[np.argmax(cv_scores)]
print(f"Optimal K: {optimal_k}")
```

## Feature Engineering for KNN

### Feature Scaling
```python
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler

# Standard scaling (zero mean, unit variance)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Min-max scaling (0-1 range)
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)

# Robust scaling (resistant to outliers)
scaler = RobustScaler()
X_scaled = scaler.fit_transform(X)
```

### Feature Selection
```python
from sklearn.feature_selection import SelectKBest, f_classif

# Select top K features
selector = SelectKBest(f_classif, k=20)
X_selected = selector.fit_transform(X_train, y_train)

# Get selected feature indices
selected_features = selector.get_support(indices=True)
print(f"Selected features: {selected_features}")
```

### Dimensionality Reduction
```python
from sklearn.decomposition import PCA

# Apply PCA
pca = PCA(n_components=0.95)  # Keep 95% of variance
X_reduced = pca.fit_transform(X_train)

print(f"Original dimensions: {X_train.shape[1]}")
print(f"Reduced dimensions: {X_reduced.shape[1]}")
```

## Evaluation Metrics

### Classification Metrics

```python
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)

# Make predictions
y_pred = classifier.predict(X_test)

# Calculate metrics
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average='weighted')
recall = recall_score(y_test, y_pred, average='weighted')
f1 = f1_score(y_test, y_pred, average='weighted')

print(f"Accuracy: {accuracy:.3f}")
print(f"Precision: {precision:.3f}")
print(f"Recall: {recall:.3f}")
print(f"F1 Score: {f1:.3f}")

# Confusion matrix
cm = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:")
print(cm)

# Detailed report
report = classification_report(y_test, y_pred)
print(report)
```

### Cross-Validation

```python
from sklearn.model_selection import cross_validate

# Perform cross-validation
cv_results = cross_validate(
    classifier.model,
    X_train,
    y_train,
    cv=5,
    scoring=['accuracy', 'precision_weighted', 'recall_weighted', 'f1_weighted'],
    return_train_score=True
)

print(f"CV Accuracy: {cv_results['test_accuracy'].mean():.3f} ± {cv_results['test_accuracy'].std():.3f}")
print(f"CV Precision: {cv_results['test_precision_weighted'].mean():.3f}")
print(f"CV Recall: {cv_results['test_recall_weighted'].mean():.3f}")
print(f"CV F1: {cv_results['test_f1_weighted'].mean():.3f}")
```

## Comparison with Other Classifiers

| Aspect | KNN | Decision Tree | Random Forest | SVM | Neural Network |
|--------|-----|---------------|---------------|-----|----------------|
| Training Speed | Fast | Fast | Medium | Medium | Slow |
| Prediction Speed | Slow | Fast | Fast | Medium | Fast |
| Memory Usage | High | Low | Medium | Medium | Low |
| Interpretability | High | High | Medium | Low | Low |
| Handles Non-linear | Yes | Yes | Yes | Yes | Yes |
| Feature Scaling | Required | Not Required | Not Required | Required | Required |
| Hyperparameters | Few | Many | Many | Many | Many |

## Best Practices

### 1. Always Scale Features
```python
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

# Create pipeline with scaling
pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('classifier', KNNClassifier(n_neighbors=5))
])

pipeline.fit(X_train, y_train)
predictions = pipeline.predict(X_test)
```

### 2. Use Cross-Validation
```python
from sklearn.model_selection import StratifiedKFold

# Use stratified k-fold for imbalanced datasets
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

for train_idx, val_idx in skf.split(X, y):
    X_train, X_val = X[train_idx], X[val_idx]
    y_train, y_val = y[train_idx], y[val_idx]
    
    classifier.train(X_train, y_train)
    score = classifier.score(X_val, y_val)
    print(f"Fold accuracy: {score:.3f}")
```

### 3. Handle Imbalanced Data
```python
from imblearn.over_sampling import SMOTE

# Apply SMOTE for imbalanced classes
smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X_train, y_train)

classifier.train(X_resampled, y_resampled)
```

### 4. Optimize K Value
```python
# Use elbow method to find optimal K
from sklearn.model_selection import cross_val_score
import matplotlib.pyplot as plt

k_range = range(1, 31)
scores = []

for k in k_range:
    classifier = KNNClassifier(n_neighbors=k)
    score = cross_val_score(classifier.model, X_train, y_train, cv=5).mean()
    scores.append(score)

plt.plot(k_range, scores)
plt.xlabel('K Value')
plt.ylabel('Cross-Validation Accuracy')
plt.title('Elbow Method for Optimal K')
plt.show()
```

## Troubleshooting

### Poor Performance
- **Scale features**: Use StandardScaler or MinMaxScaler
- **Reduce dimensions**: Apply PCA or feature selection
- **Tune K**: Try different values of K
- **Check for outliers**: Remove or handle outliers
- **Balance classes**: Use SMOTE or class weights

### Slow Predictions
- **Reduce training set**: Use representative subset
- **Use KD-Tree**: Set algorithm='kd_tree'
- **Reduce dimensions**: Apply PCA
- **Parallelize**: Use n_jobs=-1 for parallel processing

### Memory Issues
- **Reduce training set**: Sample representative examples
- **Reduce features**: Apply feature selection
- **Use sparse matrices**: For sparse feature vectors

## References

- [scikit-learn KNN Documentation](https://scikit-learn.org/stable/modules/neighbors.html)
- [KNN Algorithm Explained](https://en.wikipedia.org/wiki/K-nearest_neighbors_algorithm)
- [Feature Scaling for Machine Learning](https://scikit-learn.org/stable/modules/preprocessing.html)

## See Also

- [Decision Tree Classifier](decisiontree-classifier.md)
- [Random Forest Classifier](rf-classifier.md)
- [SVM Classifier](svm-classifier.md)
- [Feature Extraction](../analysis/feature-extraction.md)
- [Model Evaluation](../guides/model-evaluation.md)
