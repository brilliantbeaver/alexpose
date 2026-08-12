# Decision Tree Gait Classifier

## Overview

Decision Tree classifier provides maximum interpretability through a tree of if-then-else rules. Each decision can be traced and explained, making it ideal for clinical applications where transparency is essential.

## Key Features

- **Maximum interpretability**: Clear decision rules
- **No feature scaling needed**: Works with raw features
- **Handles non-linearity**: Captures complex patterns
- **Feature importance**: Shows which features drive decisions
- **Fast prediction**: O(log n) complexity
- **Visual representation**: Can be plotted as tree diagram

## When to Use

- Maximum interpretability required
- Clinical decision support systems
- Regulatory compliance needs
- Educational purposes
- Quick prototyping
- Feature selection

## Configuration

```python
from ambient.classification.decisiontree_classifier import (
    DecisionTreeGaitClassifier,
    DecisionTreeClassifierConfig
)

config = DecisionTreeClassifierConfig(
    criterion="gini",           # 'gini' or 'entropy'
    max_depth=10,               # Maximum tree depth
    min_samples_split=2,        # Minimum samples to split
    min_samples_leaf=1,         # Minimum samples in leaf
    max_features=None,          # Features to consider for split
    class_weight="balanced",
    random_state=42
)

classifier = DecisionTreeGaitClassifier(config)
```

## Usage Example

```python
# Train and get tree structure
classifier = DecisionTreeGaitClassifier()
metrics = classifier.train(training_features)

print(f"Tree depth: {metrics['tree_depth']}")
print(f"Number of leaves: {metrics['n_leaves']}")

# Get decision rules
rules = classifier.get_decision_rules()
for rule in rules[:5]:
    print(f"Rule {rule['rule_id']}: {rule['description']}")

# Get decision path for a sample
result = classifier.classify_gait(test_feature)
path = classifier.get_decision_path(test_feature)
print(f"Decision path: {path}")
```

## Decision Rules

```python
# Extract human-readable rules
rules = classifier.get_decision_rules()

# Example output:
# Rule 1: IF left_knee_range <= 65.0 AND hip_asymmetry > 1.2 THEN parkinsons
# Rule 2: IF left_knee_range > 65.0 AND ankle_asymmetry <= 1.1 THEN normal
```

## Performance Characteristics

- **Training Time**: Fast
- **Prediction Time**: Very fast
- **Memory Usage**: Low
- **Accuracy**: Moderate (prone to overfitting)
- **Interpretability**: Excellent

## Tips and Best Practices

1. **Limit tree depth**: Prevents overfitting (max_depth=5-10)
2. **Prune the tree**: Use min_samples_leaf to simplify
3. **Use as baseline**: Compare with ensemble methods
4. **Visualize tree**: Helps understand decision logic
5. **Combine in ensemble**: Random Forest or boosting
6. **Cross-validate**: Single trees have high variance

## Common Issues

### Overfitting
- **Solution**: Reduce max_depth, increase min_samples_leaf/split

### Unstable Predictions
- **Solution**: Use ensemble methods (Random Forest)

## References

- [Scikit-learn Decision Trees](https://scikit-learn.org/stable/modules/tree.html)
