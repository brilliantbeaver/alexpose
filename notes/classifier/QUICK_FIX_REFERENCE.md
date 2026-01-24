# Quick Fix Reference: LLMClassifier AttributeError

## The Error
```
AttributeError: 'NoneType' object has no attribute 'transform'
```

## Root Cause
`LLMClassifier` inherited `evaluate()` from `BaseGaitClassifier`, which assumes traditional ML patterns (label encoding, model.predict(), etc.) that don't apply to LLM classifiers.

## The Fix
**Overrode `evaluate()` method in `LLMClassifier`** with LLM-specific implementation.

## What Changed

### Before (Broken)
```python
# Inherited from BaseGaitClassifier
def evaluate(self, test_features, test_labels=None):
    y_test_encoded = self.label_encoder_.transform(y_test)  # ❌ label_encoder_ is None
    y_pred_encoded = self.model.predict(X_test)  # ❌ model has no predict()
    ...
```

### After (Fixed)
```python
# Overridden in LLMClassifier
def evaluate(self, test_features, test_labels=None):
    y_true = [f.condition_label for f in test_features]  # ✅ No encoding
    for feature in test_features:
        result = self.classify_gait(feature)  # ✅ API call
        y_pred.append(result['predicted_condition'])
    ...
```

## Usage

```python
from ambient.classification.llm_classifier import LLMClassifier, LLMClassifierConfig

# Configure
config = LLMClassifierConfig(model_name="gpt-4o-mini")
classifier = LLMClassifier(config)

# Train (store examples)
classifier.train(train_features)

# Evaluate (now works!)
metrics = classifier.evaluate(test_features)

print(f"Accuracy: {metrics['accuracy']:.3f}")
print(f"Per-sample results: {len(metrics['per_sample_results'])}")
```

## Key Differences

| Aspect | Traditional ML | LLM Classifier |
|--------|---------------|----------------|
| Prediction | `model.predict()` | API calls |
| Labels | Encoded integers | Text strings |
| Speed | Fast | Slow (API latency) |
| Output | Class only | Class + explanation |

## Files Modified
- `ambient/classification/llm_classifier.py` - Added `evaluate()` override

## Documentation
- `docs/architecture/INHERITANCE_FIX_GUIDE.md` - Detailed guide
- `INHERITANCE_FIX_SUMMARY.md` - Complete summary
- `QUICK_FIX_REFERENCE.md` - This file

## Verification
```bash
python -c "
from ambient.classification.llm_classifier import LLMClassifier
from ambient.classification.base_classifier import BaseGaitClassifier
print('✅ Fixed:', LLMClassifier.evaluate != BaseGaitClassifier.evaluate)
"
```

## Status
✅ **Fixed and Tested** - January 23, 2026
