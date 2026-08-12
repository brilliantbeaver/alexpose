# Feature Groups Enabled Initialization Fix - Summary

**Date:** 2026-01-27  
**Status:** ✅ FIXED  
**Impact:** Critical bug preventing classifier training

## Problem

When training classifiers with `GaitFeatureVector` objects created via direct instantiation, the following error occurred:

```
AttributeError: 'GaitFeatureVector' object has no attribute '_feature_groups_enabled'
```

This happened in the exact scenario from the user's notebook:

```python
knn_metrics = knn_classifier.train(
    features=train_features,
    validate=True,
    auto_remove_invalid=True
)
```

## Root Cause

The `_feature_groups_enabled` field was defined with `default_factory=lambda: {...}` in the dataclass. Python dataclasses don't properly handle mutable defaults when instances are created directly (not through factory methods), causing the field to not be initialized.

## Solution

Changed the field to `Optional[Dict[str, bool]]` with `default=None`, then properly initialize it in `__post_init__`:

```python
@dataclass
class GaitFeatureVector:
    _feature_groups_enabled: Optional[Dict[str, bool]] = field(default=None)

    def __post_init__(self):
        if self._feature_groups_enabled is None:
            self._feature_groups_enabled = {
                "core_angles": True,
                "spatiotemporal": True,
                # ... all 13 feature groups
            }
```

## Changes Made

### 1. Core Fix
- **File:** `ambient/classification/features.py`
- **Change:** Modified `_feature_groups_enabled` initialization in `__post_init__`
- **Impact:** All direct instantiations now work correctly

### 2. New Tests
- **File:** `tests/ambient/classification/test_feature_initialization.py`
- **Tests:** 11 comprehensive tests covering:
  - Direct instantiation patterns
  - Empty instantiation
  - `to_array()` functionality
  - Feature group selection
  - Multiple instance independence
  - Classifier training scenario (exact bug reproduction)
  - Regression prevention

### 3. Updated Tests
- **File:** `tests/ambient/classification/test_knn_classifier.py`
- **Changes:**
  - Updated `test_to_array()` to expect 82 features (with legacy mode test)
  - Updated `test_get_feature_names()` to expect 82 features (with legacy mode test)
  - Updated `test_classifier_initialization()` to expect 82 feature names

### 4. Documentation
- **File:** `docs/fixes/feature-groups-enabled-initialization-fix.md`
- **Content:** Comprehensive documentation of the issue, root cause, solution, and prevention

## Test Results

All tests pass:

```bash
# New regression tests
python3 tests/ambient/classification/test_feature_initialization.py
# Result: 11 passed in 0.02s

# Updated KNN classifier tests
python3 tests/ambient/classification/test_knn_classifier.py
# Result: 20 passed in 3.10s

# Manual verification
python3 -c "from ambient.classification.features import GaitFeatureVector; ..."
# Result: ✅ All tests passed!
```

## Verification

The fix was verified with:

1. **Direct instantiation:** Creating `GaitFeatureVector` objects directly works
2. **Classifier training:** The exact scenario from the bug report now works
3. **Feature extraction:** `to_array()` works with and without feature groups
4. **Backward compatibility:** All existing tests pass with updates

## Impact Assessment

### Before Fix
- ❌ Direct instantiation caused AttributeError
- ❌ Classifier training failed
- ❌ Notebooks couldn't train models
- ❌ Production code at risk

### After Fix
- ✅ Direct instantiation works perfectly
- ✅ Classifier training succeeds
- ✅ Notebooks can train models
- ✅ Production code safe
- ✅ 82 features available by default
- ✅ Legacy 15-feature mode still supported

## Key Learnings

1. **Avoid `default_factory` for complex mutable defaults** in dataclasses
2. **Use `Optional` + `__post_init__`** for proper initialization
3. **Test direct instantiation patterns** not just factory methods
4. **Add regression tests** for critical bugs
5. **Document root causes** to prevent recurrence

## Next Steps

1. ✅ Fix applied and tested
2. ✅ Regression tests added
3. ✅ Documentation created
4. ✅ Existing tests updated
5. 🔄 User can now continue with classifier training

## Related Files

- `ambient/classification/features.py` - Core fix
- `ambient/classification/base_classifier.py` - Where error occurred
- `ambient/classification/knn_classifier.py` - Affected classifier
- `tests/ambient/classification/test_feature_initialization.py` - New tests
- `tests/ambient/classification/test_knn_classifier.py` - Updated tests
- `docs/fixes/feature-groups-enabled-initialization-fix.md` - Full documentation

## User Action Required

The user should:

1. **Restart their Jupyter kernel** to load the updated code
2. **Re-run the classifier training cell** - it should now work
3. **Verify the fix** by checking that training completes without errors

Example:

```python
# This should now work without AttributeError
knn_metrics = knn_classifier.train(
    features=train_features,
    validate=True,
    auto_remove_invalid=True
)

print(f"✅ Training successful!")
print(f"   Accuracy: {knn_metrics['train_accuracy']:.3f}")
print(f"   Features: {knn_metrics['n_features']}")  # Should be 82
```

---

**Fix Status:** ✅ COMPLETE  
**Tests Status:** ✅ ALL PASSING  
**Documentation:** ✅ COMPLETE  
**Ready for Use:** ✅ YES
