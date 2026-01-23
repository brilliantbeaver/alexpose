# Ankle Range of Motion Feature Addition

**Date**: January 21, 2026  
**Type**: Feature Enhancement  
**Impact**: KNN Classifier Feature Set Expansion

## Summary

Added `left_ankle_range` and `right_ankle_range` features to the KNN gait classifier, expanding the feature set from 13 to 15 features. This addresses a previous inconsistency where ankle range statistics were being computed but not included in the feature vector.

## Rationale

### Clinical Importance
Ankle range of motion (ROM) is a critical diagnostic feature for gait analysis:
- **Foot Drop Detection**: Reduced ankle dorsiflexion ROM is characteristic of hemiplegic/stroke gait
- **Parkinson's Gait**: Reduced ankle ROM contributes to shuffling gait patterns
- **Antalgic Gait**: Compensatory ankle ROM changes indicate pain avoidance
- **Push-off Power**: Ankle ROM directly correlates with propulsive force generation

### Technical Consistency
- Hip and knee ROM features were already included
- Ankle statistics (including range) were already being computed by `get_statistics()`
- Ankle mean angles and asymmetry were already in the feature set
- This change completes the feature symmetry across all major lower limb joints

## Changes Made

### 1. Core Implementation (`ambient/classification/knn_classifier.py`)

#### GaitFeatureVector Dataclass (moved to `ambient/classification/features.py`)
- Added `left_ankle_range: float = 0.0`
- Added `right_ankle_range: float = 0.0`

#### to_array() Method
- Expanded array from 13 to 15 elements
- Added ankle range values at indices 11 and 14

#### get_feature_names() Method
- Added `"left_ankle_range"` and `"right_ankle_range"` to feature name list
- Updated total count from 13 to 15

#### from_joint_angles() Method
- Now assigns `left_ankle_range` and `right_ankle_range` from statistics
- Previously these values were computed but discarded

#### _dict_to_feature_vector() Method
- Added support for ankle range features in dictionary conversion

### 2. Tests (`tests/ambient/classification/test_knn_classifier.py`)

#### sample_joint_angles Fixture
- Already included ankle range statistics (40.0 for both sides)

#### sample_feature_vectors Fixture
- **Normal gait samples**: Added `left_ankle_range=40.0`, `right_ankle_range=40.0`
- **Stroke gait samples**: Added `left_ankle_range=40.0`, `right_ankle_range=30.0` (reduced)

#### Test Updates
- `test_to_array`: Updated assertion from 13 to 15 features, added ankle range check
- `test_from_joint_angles`: Added assertions for ankle range values
- `test_get_feature_names`: Updated count to 15, added ankle range name checks
- `test_classifier_initialization`: Updated feature count assertion to 15
- `test_classify_with_dict_input`: Added ankle range features to test dictionary

**Test Results**: All 20 tests pass ✅

### 3. Documentation

#### `docs/classifier/README.md`
- Updated feature count from 13 to 15
- Expanded "Range of Motion" section from 4 to 6 features
- Added clinical note about ankle ROM importance for foot drop and push-off detection

#### `docs/classifier/implementation.md`
- Updated feature count from 13 to 15
- Added left_ankle_range (12) and right_ankle_range (15) to numbered feature list
- Added clinical rationale for ankle ROM

#### `docs/classifier/design.md`
- Updated feature selection rationale from 13 to 15 features

### 4. Notebooks (Manual Update Required)

The following notebooks reference the KNN classifier and should be updated when re-run:

#### `experiments/exp4/02_train_KNN.ipynb`
- Uses `GaitFeatureVector.from_joint_angles()` - will automatically include new features
- Feature visualization and statistics will show 15 features instead of 13
- No code changes required, but outputs will differ

#### `notebooks/tutorial2 - train classifier.ipynb`
- Uses `GaitFeatureVector.from_joint_angles()` - will automatically include new features
- Feature name iteration will now include ankle range features
- No code changes required, but outputs will differ

**Note**: Notebooks will automatically pick up the new features when cells are re-executed. The feature extraction is dynamic via `from_joint_angles()`.

### 5. Experiment Scripts

#### `experiments/exp3/process4_KNN.py`
- Uses `GaitFeatureVector.from_joint_angles()` - will automatically include new features
- No code changes required

## Feature Vector Structure

### Before (13 features)
```python
[
    left_hip_mean,      # 0
    left_knee_mean,     # 1
    left_ankle_mean,    # 2
    right_hip_mean,     # 3
    right_knee_mean,    # 4
    right_ankle_mean,   # 5
    hip_asymmetry,      # 6
    knee_asymmetry,     # 7
    ankle_asymmetry,    # 8
    left_hip_range,     # 9
    left_knee_range,    # 10
    right_hip_range,    # 11
    right_knee_range,   # 12
]
```

### After (15 features)
```python
[
    left_hip_mean,       # 0
    left_knee_mean,      # 1
    left_ankle_mean,     # 2
    right_hip_mean,      # 3
    right_knee_mean,     # 4
    right_ankle_mean,    # 5
    hip_asymmetry,       # 6
    knee_asymmetry,      # 7
    ankle_asymmetry,     # 8
    left_hip_range,      # 9
    left_knee_range,     # 10
    left_ankle_range,    # 11  ← NEW
    right_hip_range,     # 12
    right_knee_range,    # 13
    right_ankle_range,   # 14  ← NEW
]
```

## Backward Compatibility

### Breaking Changes
⚠️ **Saved classifier models are NOT compatible** with this update:
- Old models expect 13 features
- New models expect 15 features
- Attempting to load old models will cause dimension mismatch errors

### Migration Path
1. **Retrain all classifiers** with the new 15-feature format
2. Delete or archive old `.pkl` model files
3. Re-run training notebooks/scripts to generate new models

### Data Compatibility
✅ **Training data is fully compatible**:
- `JointAngleSequence` objects already contain ankle range statistics
- `from_joint_angles()` will automatically extract the new features
- No changes needed to existing CSV files or video data

## Validation

### Unit Tests
```bash
pytest tests/ambient/classification/test_knn_classifier.py -v
```
**Result**: 20/20 tests passed ✅

### Integration Tests
The following should be re-run to validate end-to-end functionality:
- [ ] `experiments/exp4/02_train_KNN.ipynb` - Train classifier with new features
- [ ] `notebooks/tutorial2 - train classifier.ipynb` - Tutorial workflow
- [ ] `experiments/exp3/process4_KNN.py` - Batch processing script

## Expected Impact

### Model Performance
- **Potential improvement**: Ankle ROM provides additional discriminative power
- **Stroke detection**: Better capture of foot drop patterns
- **Parkinson's detection**: Better capture of reduced ankle mobility
- **Normal gait**: More complete kinematic profile

### Feature Importance
After retraining, feature importance analysis should reveal:
- Whether ankle ROM contributes significantly to classification
- Which conditions show the most ankle ROM variation
- Correlation between ankle ROM and other features

## Next Steps

1. ✅ Update core implementation
2. ✅ Update tests
3. ✅ Update documentation
4. ⏳ Re-run training notebooks to validate
5. ⏳ Retrain production classifiers
6. ⏳ Analyze feature importance with new features
7. ⏳ Update any saved model files

## Files Modified

### Code
- `ambient/classification/knn_classifier.py`
- `tests/ambient/classification/test_knn_classifier.py`

### Documentation
- `docs/classifier/README.md`
- `docs/classifier/implementation.md`
- `docs/classifier/design.md`

### Notebooks (No changes, but outputs will differ)
- `experiments/exp4/02_train_KNN.ipynb`
- `notebooks/tutorial2 - train classifier.ipynb`

### Scripts (No changes needed)
- `experiments/exp3/process4_KNN.py`

## References

- Joint angle calculation: `ambient/pose/joint_angles.py`
- Feature extraction: `GaitFeatureVector.from_joint_angles()`
- Clinical gait analysis literature on ankle ROM significance
