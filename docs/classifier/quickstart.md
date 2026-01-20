# KNN Gait Classifier - Quick Start Guide

## Installation

The KNN classifier is included in the AlexPose package. No additional installation required.

```bash
# Ensure AlexPose environment is activated
uv sync
```

## 5-Minute Quick Start

### 1. Train a Classifier

```python
from pathlib import Path
from ambient.classification.knn_classifier import (
    KNNGaitClassifier,
    KNNClassifierConfig,
    GaitFeatureVector
)
from ambient.pose.joint_angles import get_joint_angles
from ambient.gavd import GAVDDataLoader
from ambient.pose.keypoint_extractor import SequenceKeypointExtractor

# Load training data
data_root = Path("experiments/exp2/data")
video_base_path = Path("data/youtube")

# Extract features from one condition
condition_path = data_root / "normal"
csv_file = list(condition_path.glob("*.csv"))[0]

gavd_loader = GAVDDataLoader()
df = gavd_loader.load_gavd_data(str(csv_file))
sequences = gavd_loader.organize_by_sequence(df)

features = []
for seq_id in sequences:
    sequence_df = sequences[seq_id]
    
    # Extract keypoints
    extractor = SequenceKeypointExtractor()
    keypoints = extractor.extract_from_sequence(
        sequence_df,
        video_base_path=video_base_path
    )
    
    # Calculate joint angles
    joint_angles = get_joint_angles(
        keypoints,
        keypoint_format="BLAZEPOSE_33"
    )
    
    # Create feature vector
    feature_vector = GaitFeatureVector.from_joint_angles(
        joint_angles,
        sample_id=seq_id,
        condition_label="normal"
    )
    features.append(feature_vector)

# Train classifier
classifier = KNNGaitClassifier()
metrics = classifier.train(features)

print(f"Training accuracy: {metrics['train_accuracy']:.2%}")

# Save model
classifier.save("models/my_classifier.pkl")
```

### 2. Make Predictions

```python
from ambient.classification.knn_classifier import KNNGaitClassifier

# Load trained classifier
classifier = KNNGaitClassifier.load("models/my_classifier.pkl")

# Classify new sample
result = classifier.classify_gait(test_feature_vector)

print(f"Predicted: {result['predicted_condition']}")
print(f"Confidence: {result['confidence']:.2%}")
```

### 3. Evaluate Performance

```python
# Evaluate on test set
metrics = classifier.evaluate(test_features)

print(f"Test Accuracy: {metrics['accuracy']:.2%}")
print("\nClassification Report:")
for condition in metrics['classes']:
    report = metrics['classification_report'][condition]
    print(f"{condition}:")
    print(f"  Precision: {report['precision']:.2%}")
    print(f"  Recall: {report['recall']:.2%}")
```

## Common Use Cases

### Use Case 1: Train on GAVD Dataset

```bash
# Run the training script
python experiments/exp2/src/process4_KNN.py
```

This will:
- Load all condition data from `experiments/exp2/data/`
- Extract features from each sample
- Train a KNN classifier
- Evaluate on test set
- Save model to `experiments/exp2/models/knn_classifier.pkl`

### Use Case 2: Classify Single Video

```python
from pathlib import Path
from ambient.classification.knn_classifier import KNNGaitClassifier, GaitFeatureVector
from ambient.pose.joint_angles import get_joint_angles
from ambient.pose.keypoint_extractor import SequenceKeypointExtractor
import cv2

# Load classifier
classifier = KNNGaitClassifier.load("models/knn_classifier.pkl")

# Process video
video_path = Path("path/to/video.mp4")
extractor = SequenceKeypointExtractor()

# Extract keypoints from video
cap = cv2.VideoCapture(str(video_path))
keypoints = []

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    kp = extractor.extract_from_image(frame_rgb)
    keypoints.append(kp)

cap.release()

# Calculate joint angles
joint_angles = get_joint_angles(keypoints, keypoint_format="BLAZEPOSE_33")

# Create feature vector
features = GaitFeatureVector.from_joint_angles(joint_angles)

# Classify
result = classifier.classify_gait(features)

print(f"Condition: {result['predicted_condition']}")
print(f"Confidence: {result['confidence']:.2%}")
```

### Use Case 3: Hyperparameter Tuning

```python
from ambient.classification.knn_classifier import KNNGaitClassifier

classifier = KNNGaitClassifier()

# Define parameter grid
param_grid = {
    'n_neighbors': [3, 5, 7, 9, 11],
    'weights': ['uniform', 'distance'],
    'metric': ['euclidean', 'manhattan', 'minkowski']
}

# Tune hyperparameters
results = classifier.tune_hyperparameters(
    training_features,
    param_grid=param_grid
)

print(f"Best parameters: {results['best_params']}")
print(f"Best CV score: {results['best_score']:.2%}")

# Classifier is now trained with best parameters
classifier.save("models/tuned_classifier.pkl")
```

### Use Case 4: Batch Prediction

```python
from ambient.classification.knn_classifier import KNNGaitClassifier

classifier = KNNGaitClassifier.load("models/knn_classifier.pkl")

# Classify multiple samples
results = []
for feature_vector in test_features:
    result = classifier.classify_gait(feature_vector)
    results.append(result)

# Analyze results
predictions = [r['predicted_condition'] for r in results]
confidences = [r['confidence'] for r in results]

print(f"Average confidence: {np.mean(confidences):.2%}")
print(f"Predictions: {Counter(predictions)}")
```

## Configuration Options

### Basic Configuration

```python
from ambient.classification.knn_classifier import KNNClassifierConfig

config = KNNClassifierConfig(
    n_neighbors=5,              # Number of neighbors
    weights="distance",         # Weight function
    metric="euclidean",         # Distance metric
    normalize_features=True     # Feature normalization
)

classifier = KNNGaitClassifier(config=config)
```

### Advanced Configuration

```python
config = KNNClassifierConfig(
    n_neighbors=7,
    weights="uniform",          # Equal weight for all neighbors
    metric="manhattan",         # L1 distance
    algorithm="ball_tree",      # Algorithm for neighbor search
    normalize_features=True,
    confidence_threshold=0.6    # Minimum confidence
)
```

## Troubleshooting

### Problem: Low Accuracy

**Solution 1: Collect More Data**
```python
# Aim for at least 10 samples per condition
print(f"Samples per condition: {condition_counts}")
```

**Solution 2: Tune Hyperparameters**
```python
results = classifier.tune_hyperparameters(features)
```

**Solution 3: Check Feature Quality**
```python
# Visualize features
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.DataFrame([f.to_array() for f in features])
sns.pairplot(df, hue='condition')
plt.show()
```

### Problem: Slow Predictions

**Solution 1: Reduce Training Data**
```python
# Use representative subset
from sklearn.model_selection import StratifiedShuffleSplit

splitter = StratifiedShuffleSplit(n_splits=1, train_size=0.5)
indices, _ = next(splitter.split(X, y))
reduced_features = [features[i] for i in indices]
```

**Solution 2: Use Approximate Nearest Neighbors**
```python
# For large datasets, consider using Annoy or FAISS
# (Future enhancement)
```

### Problem: Imbalanced Classes

**Solution: Use Class Weights**
```python
from sklearn.utils.class_weight import compute_class_weight

# Compute class weights
class_weights = compute_class_weight(
    'balanced',
    classes=np.unique(labels),
    y=labels
)

# Apply weights during training
# (Future enhancement - currently not supported)
```

## Next Steps

1. **Read the Full Documentation**: [README.md](README.md)
2. **Follow the Tutorial**: [Tutorial Notebook](../../notebooks/tutorial2%20-%20train%20classifier.ipynb)
3. **Review the Design**: [Design Document](design.md)
4. **Run the Tests**: `pytest tests/ambient/classification/test_knn_classifier.py`

## Getting Help

- **Documentation**: `docs/classifier/`
- **Examples**: `examples/classification/`
- **Issues**: GitHub Issues
- **Community**: Discord/Slack

## Additional Resources

- [scikit-learn KNN Documentation](https://scikit-learn.org/stable/modules/neighbors.html)
- [Gait Analysis Literature](README.md#references)
- [AlexPose Architecture](../architecture/)
