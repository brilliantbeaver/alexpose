# Inheritance Fix Guide: LLMClassifier and BaseGaitClassifier

## Problem Statement

### The Error
```python
AttributeError: 'NoneType' object has no attribute 'transform'
```

### Root Cause
`LLMClassifier` inherits from `BaseGaitClassifier`, but the base class's `evaluate()` method makes assumptions that don't apply to LLM-based classifiers:

1. **Traditional ML Model**: Assumes `self.model.predict()` exists
2. **Label Encoding**: Assumes `self.label_encoder_` exists and is fitted
3. **Feature Scaling**: Assumes `self.scaler` exists for normalization
4. **Training Paradigm**: Assumes traditional model fitting

**LLM classifiers work fundamentally differently:**
- No `.predict()` - uses API calls to LLM services
- No label encoding - works with text labels directly
- No feature scaling - LLM processes raw feature descriptions
- "Training" stores examples, doesn't fit parameters

## Deep Analysis

### Why This Happened

1. **Inheritance for Code Reuse**: `LLMClassifier` inherited from `BaseGaitClassifier` to reuse:
   - Configuration management
   - Save/load infrastructure
   - Common interface (`IClassifier`)

2. **Incomplete Override**: While `train()` and `classify_gait()` were overridden, `evaluate()` was not

3. **Silent Failure**: The issue only manifests when calling `evaluate()`, not during initialization or training

### The Inheritance Hierarchy

```
IClassifier (Interface)
    ↓
BaseGaitClassifier (Abstract Base)
    ↓
├── KNNGaitClassifier (Traditional ML)
├── RFGaitClassifier (Traditional ML)
├── MLPGaitClassifier (Traditional ML)
└── LLMClassifier (API-based) ← Different paradigm!
```

## The Fix

### Solution: Override `evaluate()` in LLMClassifier

```python
class LLMClassifier(BaseGaitClassifier):
    def evaluate(
        self,
        test_features: List[GaitFeatureVector],
        test_labels: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """
        Evaluate LLM classifier on test data.
        
        Unlike traditional ML classifiers, LLM evaluation:
        - Makes API calls for each sample (slower)
        - Doesn't use label encoding or feature scaling
        - Returns detailed explanations for each prediction
        """
        # Custom implementation for LLM
        # 1. No label encoding
        # 2. API calls instead of model.predict()
        # 3. Includes per-sample explanations
        ...
```

### Key Differences in LLM Evaluation

| Aspect | Traditional ML | LLM Classifier |
|--------|---------------|----------------|
| Prediction | `model.predict(X)` | API call per sample |
| Speed | Fast (milliseconds) | Slow (seconds per sample) |
| Labels | Encoded integers | Text strings |
| Features | Scaled numpy arrays | Raw feature dicts |
| Output | Class only | Class + explanation |
| Batch | All at once | One at a time |

## Generalizable Principles

### When to Override Parent Methods

Override a parent method when:

1. **Fundamental Paradigm Difference**: The subclass operates on completely different principles
2. **Incompatible Assumptions**: Parent assumes resources/state that don't exist in subclass
3. **Different I/O Patterns**: Input/output requirements differ significantly
4. **Performance Characteristics**: Timing/resource usage is fundamentally different

### Design Pattern: Template Method with Overrides

```python
class BaseClassifier:
    def evaluate(self, test_data):
        """Template method - can be overridden"""
        # Default implementation for traditional ML
        pass
    
    def _prepare_data(self, data):
        """Hook method - override if needed"""
        pass

class SpecializedClassifier(BaseClassifier):
    def evaluate(self, test_data):
        """Complete override for different paradigm"""
        # Custom implementation
        pass
```

### Checklist for Subclass Implementation

When creating a subclass with different behavior:

- [ ] Identify all parent methods that make assumptions
- [ ] Override methods that are incompatible
- [ ] Document why override is necessary
- [ ] Maintain interface compatibility (same signature)
- [ ] Add subclass-specific features in return values
- [ ] Test both parent and child implementations

## Implementation Details

### What Was Added to LLMClassifier.evaluate()

1. **No Label Encoding**
   ```python
   # Traditional ML
   y_encoded = self.label_encoder_.transform(y_true)
   
   # LLM (no encoding needed)
   y_true = [f.condition_label for f in test_features]
   ```

2. **API-Based Prediction**
   ```python
   # Traditional ML
   y_pred = self.model.predict(X_test)
   
   # LLM (API calls)
   for feature in test_features:
       result = self.classify_gait(feature)  # API call
       y_pred.append(result['predicted_condition'])
   ```

3. **Per-Sample Results**
   ```python
   # LLM-specific: Store detailed results for each sample
   per_sample_results.append({
       "sample_id": feature.sample_id,
       "true_label": y_true[i],
       "predicted_label": predicted,
       "confidence": result.get("confidence", 0.0),
       "explanation": result.get("normal_abnormal_explanation", "")
   })
   ```

4. **Progress Logging**
   ```python
   # LLM evaluation is slow, so log progress
   if (i + 1) % 5 == 0:
       logger.info(f"Processed {i + 1}/{len(test_features)} samples")
   ```

## Testing Strategy

### Unit Tests

```python
def test_llm_evaluate_no_label_encoder():
    """Verify LLM evaluate doesn't use label encoder"""
    config = LLMClassifierConfig(model_name="gpt-4o-mini")
    classifier = LLMClassifier(config)
    
    # Train with examples
    classifier.train(train_features)
    
    # Verify label_encoder_ is None or not used
    assert classifier.label_encoder_ is None or not hasattr(classifier, 'label_encoder_')
    
    # Evaluate should work without label encoder
    metrics = classifier.evaluate(test_features)
    assert 'accuracy' in metrics
```

### Integration Tests

```python
def test_llm_evaluate_with_api():
    """Test full evaluation with API calls"""
    classifier = LLMClassifier(LLMClassifierConfig(model_name="gpt-4o-mini"))
    classifier.train(train_features)
    
    # Should complete without AttributeError
    metrics = classifier.evaluate(test_features[:5])  # Small subset
    
    assert metrics['accuracy'] >= 0
    assert 'per_sample_results' in metrics  # LLM-specific
```

## Similar Issues to Watch For

### Other Methods That May Need Override

1. **`_prepare_features()`**: If feature preparation differs
2. **`save()`/`load()`**: If serialization differs (already overridden)
3. **`get_feature_importance()`**: If importance calculation differs
4. **`cross_validate()`**: If CV doesn't apply (LLM uses few-shot)

### Pattern Recognition

Look for these patterns that indicate override needed:

```python
# Pattern 1: Accessing attributes that don't exist
self.label_encoder_.transform(...)  # LLM doesn't have this

# Pattern 2: Calling methods that don't exist
self.model.predict(...)  # LLM model is API client, not sklearn

# Pattern 3: Assumptions about data format
X_scaled = self.scaler.transform(X)  # LLM doesn't scale

# Pattern 4: Batch operations
predictions = model.predict(X_batch)  # LLM processes one at a time
```

## Best Practices

### 1. Document Paradigm Differences

```python
class LLMClassifier(BaseGaitClassifier):
    """
    LLM-based classifier using API calls.
    
    **Key Differences from Traditional ML:**
    - No model training (uses few-shot learning)
    - No label encoding (works with text)
    - Slower evaluation (API latency)
    - Provides explanations
    """
```

### 2. Maintain Interface Compatibility

```python
# Keep same signature as parent
def evaluate(
    self,
    test_features: List[GaitFeatureVector],
    test_labels: Optional[List[str]] = None,
) -> Dict[str, Any]:  # Same return type
```

### 3. Add Subclass-Specific Features

```python
# Add new fields to return dict, don't break existing ones
return {
    "accuracy": ...,  # Standard
    "precision": ...,  # Standard
    "per_sample_results": ...,  # LLM-specific (added)
    "evaluation_method": "llm_api_calls"  # LLM-specific (added)
}
```

### 4. Handle Errors Gracefully

```python
try:
    result = self.classify_gait(feature)
except Exception as e:
    logger.error(f"Failed to classify sample {i}: {e}")
    # Provide fallback result
    y_pred.append("unknown")
```

## Verification

### How to Verify the Fix

```bash
# 1. Test import
python -c "from ambient.classification.llm_classifier import LLMClassifier; print('✓ Import OK')"

# 2. Test method override
python -c "
from ambient.classification.llm_classifier import LLMClassifier
print('✓ evaluate overridden:', LLMClassifier.evaluate != LLMClassifier.__bases__[0].evaluate)
"

# 3. Test in notebook
# Run the evaluation cell - should complete without AttributeError
```

### Expected Behavior After Fix

✅ No `AttributeError` about `label_encoder_`  
✅ Evaluation completes with API calls  
✅ Returns standard metrics (accuracy, precision, etc.)  
✅ Includes LLM-specific fields (per_sample_results, explanations)  
✅ Logs progress during evaluation  

## Related Files

- `ambient/classification/llm_classifier.py` - LLM classifier implementation
- `ambient/classification/base_classifier.py` - Base class with original evaluate()
- `experiments/exp4/04_LLM_classifier.ipynb` - Notebook using evaluate()
- `tests/ambient/classification/test_llm_classifier.py` - Unit tests

## Future Considerations

### Potential Refactoring

Consider creating separate base classes for different paradigms:

```python
BaseGaitClassifier (Common interface)
    ↓
├── TraditionalMLClassifier (sklearn-based)
│   ├── KNNGaitClassifier
│   ├── RFGaitClassifier
│   └── MLPGaitClassifier
│
└── APIBasedClassifier (API-based)
    └── LLMClassifier
```

This would make assumptions explicit and reduce need for overrides.

---

**Date:** January 23, 2026  
**Issue:** AttributeError in LLMClassifier.evaluate()  
**Status:** Fixed ✅  
**Approach:** Method override with LLM-specific implementation
