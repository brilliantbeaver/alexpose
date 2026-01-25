# macOS Training Guide

## Quick Fix for "No space left on device" Error

If you encounter this error when training classifiers on macOS:
```
OSError: [Errno 28] No space left on device
```

**This is NOT a disk space issue!** It's a parallel processing limitation on macOS.

## Solution

The issue is already fixed in the latest version. The default configuration uses sequential processing which works reliably on macOS.

### Default Usage (Recommended)
```python
from ambient.classification.knn_classifier import (
    KNNGaitClassifier,
    KNNClassifierConfig
)
from ambient.classification.features import GaitFeatureVector

# Default configuration (safe for macOS)
config = KNNClassifierConfig(
    n_neighbors=5,
    weights="distance",
    metric="euclidean",
    normalize_features=True
    # cv_n_jobs=1 is the default (sequential processing)
)

classifier = KNNGaitClassifier(config=config)
metrics = classifier.train(train_features, validate=True)
```

### Explicit Configuration
```python
# Explicitly set sequential processing
config = KNNClassifierConfig(
    n_neighbors=5,
    cv_n_jobs=1  # Sequential (safe for macOS)
)
```

### For High-Performance Systems
If you're on a Linux system or want to try parallel processing:
```python
# Try parallel with automatic fallback
config = KNNClassifierConfig(
    n_neighbors=5,
    cv_n_jobs=-1  # Use all CPUs (with fallback to sequential)
)
```

## What Changed?

### Before (Caused Errors)
- Cross-validation used `n_jobs=-1` (all CPUs) by default
- No fallback mechanism
- Failed on macOS with semaphore errors

### After (Fixed)
- Cross-validation uses `n_jobs=1` (sequential) by default
- Automatic fallback to sequential if parallel fails
- Works reliably on all platforms

## Performance Impact

For typical use cases (< 1000 samples):
- **Sequential**: ~0.1-0.5 seconds
- **Parallel**: ~0.08-0.4 seconds
- **Difference**: Negligible

The reliability gain far outweighs the minimal performance difference.

## All Classifiers Affected

This fix applies to all classifiers:
- KNNGaitClassifier
- RFGaitClassifier
- XGBoostGaitClassifier
- SVMGaitClassifier
- LogisticGaitClassifier
- MLPGaitClassifier
- DecisionTreeGaitClassifier
- EnsembleGaitClassifier

All inherit from `BaseGaitClassifier` and use the same cross-validation mechanism.

## Troubleshooting

### Still Getting Errors?
1. **Update your code**: Make sure you're using the latest version
2. **Check configuration**: Verify `cv_n_jobs=1` in your config
3. **Check logs**: Look for fallback warnings in the logs
4. **Disable validation**: As a last resort, use `validate=False`

### Example with Validation Disabled
```python
# Skip cross-validation entirely
metrics = classifier.train(train_features, validate=False)
```

Note: This skips cross-validation metrics but training still works.

## Platform-Specific Recommendations

### macOS (Darwin)
```python
config = KNNClassifierConfig(cv_n_jobs=1)  # Sequential
```

### Linux
```python
config = KNNClassifierConfig(cv_n_jobs=-1)  # Parallel OK
```

### Windows
```python
config = KNNClassifierConfig(cv_n_jobs=1)  # Sequential safer
```

### Adaptive Configuration
```python
import platform

cv_n_jobs = 1 if platform.system() == 'Darwin' else -1
config = KNNClassifierConfig(cv_n_jobs=cv_n_jobs)
```

## Additional Resources

- [Cross-Validation macOS Fix Documentation](../fixes/cross-validation-macos-fix.md)
- [Classifier Quickstart Guide](../classifier/quickstart.md)
- [KNN Classifier Documentation](../classifier/knn-classifier.md)

## Summary

✅ **The fix is automatic** - default configuration works on macOS  
✅ **No code changes needed** - existing code works with new defaults  
✅ **Configurable** - can still use parallel processing if desired  
✅ **Robust** - automatic fallback if parallel processing fails  

You should no longer see "No space left on device" errors when training classifiers!