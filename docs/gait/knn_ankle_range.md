# KNN Classifier: Ankle Range Feature Addition

**Date**: January 21, 2026  
**Status**: ✅ Complete

## Quick Summary

The KNN gait classifier now includes ankle range of motion features, expanding from 13 to 15 features.

## What Changed

### New Features Added
- `left_ankle_range` - Range of motion for left ankle (max - min angle)
- `right_ankle_range` - Range of motion for right ankle (max - min angle)

### Why This Matters
Ankle ROM is clinically significant for:
- **Foot drop** (stroke/hemiplegic gait) - reduced dorsiflexion
- **Parkinson's gait** - reduced ankle mobility, shuffling
- **Antalgic gait** - compensatory ankle ROM changes
- **Push-off power** - ankle ROM correlates with propulsion

## For Developers

### Feature Vector Structure
```python
# Old: 13 features
# New: 15 features (added indices 11 and 14)

GaitFeatureVector(
    # Mean angles (6)
    left_hip_mean, left_knee_mean, left_ankle_mean,
    right_hip_mean, right_knee_mean, right_ankle_mean,
    
    # Asymmetry (3)
    hip_asymmetry, knee_asymmetry, ankle_asymmetry,
    
    # Range of motion (6) - EXPANDED FROM 4
    left_hip_range, left_knee_range, left_ankle_range,  # ← NEW
    right_hip_range, right_knee_range, right_ankle_range  # ← NEW
)
```

### Code Usage (No Changes Required!)
```python
from ambient.classification.knn_classifier import GaitFeatureVector
from ambient.pose.joint_angles import get_joint_angles

# Extract joint angles
joint_angles = get_joint_angles(keypoints_array, keypoint_format="BLAZEPOSE_33")

# Create feature vector - automatically includes new features
feature_vector = GaitFeatureVector.from_joint_angles(
    joint_angles,
    sample_id="sample_001",
    condition_label="normal"
)

# Feature vector now has 15 features instead of 13
print(len(feature_vector.to_array()))  # 15
```

### Breaking Changes ⚠️

**Saved Models**: Old classifier `.pkl` files are NOT compatible
- Old models expect 13 features
- New models expect 15 features
- **Action Required**: Retrain all classifiers

**Training Data**: Fully compatible ✅
- No changes needed to CSV files or videos
- `from_joint_angles()` automatically extracts new features

## For Researchers

### Retraining Required
All saved KNN classifier models must be retrained:

```python
# 1. Load training data (same as before)
features = []
for sequence in training_sequences:
    joint_angles = get_joint_angles(sequence)
    fv = GaitFeatureVector.from_joint_angles(joint_angles, ...)
    features.append(fv)

# 2. Train classifier (same as before)
classifier = KNNGaitClassifier()
classifier.train(features)

# 3. Save new model
classifier.save("models/knn_classifier_v2.pkl")
```

### Feature Importance Analysis
After retraining, analyze the contribution of ankle ROM:
- Compare classification accuracy with/without ankle features
- Check feature importance scores
- Analyze per-condition ankle ROM patterns

### Expected Improvements
- Better stroke/hemiplegic gait detection (foot drop patterns)
- Improved Parkinson's classification (reduced ankle mobility)
- More complete kinematic profile for all conditions

## Testing

All tests pass ✅:
```bash
pytest tests/ambient/classification/test_knn_classifier.py -v
# 20/20 tests passed
```

## Files Modified

### Core Implementation
- ✅ `ambient/classification/knn_classifier.py`
  - Added `left_ankle_range` and `right_ankle_range` fields
  - Updated `to_array()` to include new features
  - Updated `get_feature_names()` to return 15 names
  - Updated `from_joint_angles()` to extract ankle range
  - Updated `_dict_to_feature_vector()` for dict conversion

### Tests
- ✅ `tests/ambient/classification/test_knn_classifier.py`
  - Updated all fixtures to include ankle range
  - Updated assertions for 15 features
  - Added specific ankle range validation

### Documentation
- ✅ `docs/classifier/README.md` - Updated feature count and list
- ✅ `docs/classifier/implementation.md` - Updated feature engineering section
- ✅ `docs/classifier/design.md` - Updated feature selection rationale

### Notebooks (Auto-update)
- ℹ️ `experiments/exp4/02_train_KNN.ipynb` - Will show 15 features when re-run
- ℹ️ `notebooks/tutorial2 - train classifier.ipynb` - Will show 15 features when re-run

## Validation Checklist

- [x] Core implementation updated
- [x] Unit tests updated and passing
- [x] Documentation updated
- [x] Feature extraction verified
- [x] Array conversion verified
- [ ] Notebooks re-run (manual step)
- [ ] Classifiers retrained (manual step)
- [ ] Feature importance analyzed (manual step)

## Next Steps

1. **Re-run Training Notebooks**
   - `experiments/exp4/02_train_KNN.ipynb`
   - `notebooks/tutorial2 - train classifier.ipynb`

2. **Retrain Production Models**
   - Delete old `.pkl` files
   - Run training scripts with new feature set
   - Validate model performance

3. **Analyze Feature Importance**
   - Compare old vs new model accuracy
   - Check ankle ROM contribution per condition
   - Document findings

4. **Update Model Registry**
   - Version new models as v2.0
   - Document feature set changes
   - Archive old models

## References

- Full changelog: `ANKLE_RANGE_FEATURE_ADDITION.md`
- Joint angle calculation: `ambient/pose/joint_angles.py`
- KNN classifier: `ambient/classification/knn_classifier.py`
- Tests: `tests/ambient/classification/test_knn_classifier.py`
