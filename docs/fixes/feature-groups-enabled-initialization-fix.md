# Fix: GaitFeatureVector _feature_groups_enabled AttributeError

**Date:** 2026-01-27  
**Issue:** AttributeError: 'GaitFeatureVector' object has no attribute '_feature_groups_enabled'  
**Status:** ✅ Fixed

## Problem Description

When training classifiers with directly instantiated `GaitFeatureVector` objects, the following error occurred:

```python
AttributeError: 'GaitFeatureVector' object has no attribute '_feature_groups_enabled'
```

### Error Traceback

```
File ~/dev/alex/alexpose/ambient/classification/base_classifier.py:128, in BaseGaitClassifier.train(self, features, labels, validate, auto_remove_invalid)
    125     raise ValueError("No training features provided")
    127 # Extract features WITHOUT scaling (we'll scale after validation)
--> 128 X = np.array([f.to_array() for f in features])
    129 y = np.array([f.condition_label for f in features])

File ~/dev/alex/alexpose/ambient/classification/features.py:438, in GaitFeatureVector.to_array(self, feature_groups)
    434 # Determine which groups to include
    435 if feature_groups is None:
    436     # Use all enabled groups
    437     groups_to_include = [
--> 438         name for name, enabled in self._feature_groups_enabled.items() if enabled
    439     ]
    440 else:
    441     groups_to_include = feature_groups

AttributeError: 'GaitFeatureVector' object has no attribute '_feature_groups_enabled'
```

## Root Cause

The `_feature_groups_enabled` field was defined in the dataclass with a `default_factory`:

```python
@dataclass
class GaitFeatureVector:
    # ... other fields ...
    
    _feature_groups_enabled: Dict[str, bool] = field(default_factory=lambda: {
        "core_angles": True,
        "spatiotemporal": True,
        # ... etc
    })
```

**Problem:** Python dataclasses don't properly handle mutable defaults when instances are created directly (not through factory methods). When you create an instance like:

```python
fv = GaitFeatureVector(
    left_hip_mean=45.0,
    condition_label="normal"
)
```

The `default_factory` is not always called, leading to the attribute not being initialized.

## Solution

Changed the field definition to use `Optional[Dict[str, bool]]` with `default=None`, then properly initialize it in `__post_init__`:

```python
@dataclass
class GaitFeatureVector:
    # ... other fields ...
    
    # Use Optional to allow None, then initialize in __post_init__
    _feature_groups_enabled: Optional[Dict[str, bool]] = field(default=None)

    def __post_init__(self):
        """Calculate derived features after initialization."""
        # Initialize _feature_groups_enabled if not provided
        if self._feature_groups_enabled is None:
            self._feature_groups_enabled = {
                "core_angles": True,
                "spatiotemporal": True,
                "temporal_phases": True,
                "symmetry_indices": True,
                "kinematic": True,
                "variability": True,
                "postural": True,
                "extended_angles": True,
                "temporal_extended": True,
                "stability": True,
                "stride_extended": True,
                "symmetry_extended": True,
                "kinematic_extended": True,
            }
        
        # ... rest of __post_init__ ...
```

## Why This Works

1. **Explicit initialization:** `__post_init__` is always called after dataclass initialization
2. **None check:** We can check if the field was provided or needs initialization
3. **Mutable default safety:** Each instance gets its own dictionary, avoiding shared mutable state
4. **Backward compatibility:** Existing code that provides `_feature_groups_enabled` still works

## Testing

Created comprehensive test suite in `tests/ambient/classification/test_feature_initialization.py`:

- ✅ Direct instantiation with values
- ✅ Empty instantiation
- ✅ `to_array()` after direct instantiation
- ✅ `to_array()` with feature groups
- ✅ Multiple instances are independent
- ✅ Classifier training scenario (exact bug reproduction)
- ✅ All feature groups enabled by default
- ✅ Custom feature groups can be provided
- ✅ Regression prevention tests

All 11 tests pass.

## Impact

This fix ensures that:

1. **Direct instantiation works:** Users can create `GaitFeatureVector` objects directly without errors
2. **Classifier training works:** The exact scenario from the bug report now works correctly
3. **No breaking changes:** All existing code continues to work
4. **Better reliability:** The initialization is now explicit and predictable

## Files Changed

- `ambient/classification/features.py` - Fixed `_feature_groups_enabled` initialization
- `tests/ambient/classification/test_feature_initialization.py` - Added comprehensive tests
- `docs/fixes/feature-groups-enabled-initialization-fix.md` - This documentation

## Related Issues

This issue was discovered when training KNN classifier with features extracted from GAVD dataset:

```python
knn_metrics = knn_classifier.train(
    features=train_features,
    validate=True,
    auto_remove_invalid=True
)
```

The error occurred in the base classifier's `train()` method when it tried to convert feature vectors to arrays.

## Prevention

To prevent similar issues in the future:

1. **Avoid `default_factory` for complex mutable defaults** in dataclasses
2. **Use `Optional` + `__post_init__`** for fields that need initialization
3. **Add regression tests** for direct instantiation patterns
4. **Test with actual usage patterns** (e.g., classifier training)

## Verification

Run the test suite to verify the fix:

```bash
python3 tests/ambient/classification/test_feature_initialization.py
```

Or test manually:

```python
from ambient.classification.features import GaitFeatureVector
from ambient.classification.knn_classifier import KNNGaitClassifier

# Create features
features = [
    GaitFeatureVector(
        left_hip_mean=45.0,
        condition_label="normal"
    )
    for _ in range(10)
]

# Train classifier (should not raise AttributeError)
classifier = KNNGaitClassifier()
metrics = classifier.train(features)
print(f"✅ Training successful: {metrics['train_accuracy']:.3f}")
```

## References

- Python dataclasses documentation: https://docs.python.org/3/library/dataclasses.html
- Mutable default arguments: https://docs.python-guide.org/writing/gotchas/#mutable-default-arguments
- Issue discovered in: `experiments/exp5/02_classify_features2.ipynb`
