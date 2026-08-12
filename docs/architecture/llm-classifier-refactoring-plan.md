# LLM Classifier Refactoring Plan

## Executive Summary

Refactor `LLMClassifier` to inherit from `BaseGaitClassifier` for consistency with other classifiers while preserving its unique LLM-based functionality.

## Current State Analysis

### Problems Identified

1. **No Base Class Inheritance**: `LLMClassifier` directly implements `IClassifier` instead of using `BaseGaitClassifier`
2. **Inconsistent Interface**: Different method signatures and return types compared to other classifiers
3. **No Training Support**: Cannot be trained on labeled data like other classifiers
4. **Different Input Format**: Uses `Dict[str, Any]` instead of `GaitFeatureVector`
5. **No Model Persistence**: Missing `save()` and `load()` methods
6. **No Evaluation Metrics**: Cannot use standard evaluation methods
7. **Inconsistent Configuration**: Uses constructor parameters instead of config dataclass

### Current Architecture

```python
LLMClassifier(IClassifier)
├── __init__(model_name, provider, api_key, ...)  # Many parameters
├── classify_gait(gait_features: Dict, context: Dict) → Dict
├── _classify_normal_abnormal()
├── _identify_conditions()
├── _generate_response()
├── _parse_normal_abnormal_response()
├── _parse_condition_response()
├── get_classification_confidence()
├── explain_classification()
├── get_supported_models()
└── is_model_available()
```

### Desired Architecture

```python
LLMClassifier(BaseGaitClassifier)
├── __init__(config: LLMClassifierConfig)
├── _create_model() → LLM client
├── _get_model_params() → Dict
├── train(features, labels, validate) → Dict  # Inherited
├── classify_gait(gait_features: Union[GaitFeatureVector, Dict]) → Dict  # Inherited
├── evaluate(test_features, test_labels) → Dict  # Inherited
├── save(filepath) → None  # Inherited
├── load(filepath) → LLMClassifier  # Inherited
├── _classify_normal_abnormal()  # LLM-specific
├── _identify_conditions()  # LLM-specific
├── _generate_response()  # LLM-specific
└── ... (other LLM-specific methods)
```

## Refactoring Strategy

### Phase 1: Configuration Dataclass

**Goal**: Create `LLMClassifierConfig` following the pattern of other classifiers

**Implementation**:
```python
@dataclass
class LLMClassifierConfig(BaseClassifierConfig):
    """Configuration for LLM classifier."""
    
    # LLM provider settings
    model_name: str = "gpt-4o-mini"
    provider: str = "openai"  # "openai" or "gemini"
    api_key: Optional[str] = None
    
    # Generation settings
    temperature: float = 0.1
    max_tokens: Optional[int] = None
    
    # Classification settings
    enable_chain_of_thought: bool = True
    two_stage_classification: bool = True
    
    # Prompt settings
    prompt_config_path: Optional[Path] = None
    
    # Override base settings
    normalize_features: bool = False  # LLM doesn't need normalization
    cv_n_jobs: int = 1  # LLM is sequential by nature
```

**Rationale**:
- Consistent with other classifiers
- Easier to serialize/deserialize
- Type-safe configuration
- Clear separation of concerns

### Phase 2: Adapt to BaseGaitClassifier Interface

**Goal**: Make LLMClassifier inherit from BaseGaitClassifier

**Key Changes**:

1. **Constructor**:
```python
def __init__(self, config: Optional[LLMClassifierConfig] = None):
    config = config or LLMClassifierConfig()
    super().__init__(config)
    self.config: LLMClassifierConfig = config
    self.prompt_manager = PromptManager(config.prompt_config_path)
    # Initialize LLM client in _create_model()
```

2. **_create_model()**:
```python
def _create_model(self):
    """Create and return LLM client."""
    if self.config.provider == "openai":
        return self._create_openai_client()
    elif self.config.provider == "gemini":
        return self._create_gemini_client()
    else:
        raise ValueError(f"Unsupported provider: {self.config.provider}")
```

3. **_get_model_params()**:
```python
def _get_model_params(self) -> Dict[str, Any]:
    """Get LLM-specific parameters for saving."""
    return {
        "provider": self.config.provider,
        "model_name": self.config.model_name,
        "temperature": self.config.temperature,
        "enable_chain_of_thought": self.config.enable_chain_of_thought,
    }
```

4. **classify_gait() - Override**:
```python
def classify_gait(
    self,
    gait_features: Union[GaitFeatureVector, Dict[str, Any]],
    context: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Classify gait using LLM.
    
    Overrides base class to support both GaitFeatureVector and Dict inputs.
    """
    # Convert GaitFeatureVector to dict if needed
    if isinstance(gait_features, GaitFeatureVector):
        features_dict = self._feature_vector_to_dict(gait_features)
    else:
        features_dict = gait_features
    
    # Perform LLM classification
    return self._llm_classify(features_dict, context)
```

### Phase 3: Training Support (Optional but Recommended)

**Goal**: Enable LLM to learn from labeled examples via few-shot learning

**Implementation**:
```python
def train(
    self,
    features: List[GaitFeatureVector],
    labels: Optional[List[str]] = None,
    validate: bool = False,  # LLM doesn't need CV
    auto_remove_invalid: bool = True,
) -> Dict[str, Any]:
    """
    'Train' LLM classifier by storing examples for few-shot learning.
    
    Note: LLMs don't train in the traditional sense. This method stores
    labeled examples that will be included in prompts for few-shot learning.
    """
    # Validate and store examples
    self.few_shot_examples = []
    
    for feature in features:
        if auto_remove_invalid:
            is_valid, _ = feature.validate()
            if not is_valid:
                continue
        
        self.few_shot_examples.append({
            "features": feature.to_array().tolist(),
            "label": feature.condition_label,
            "sample_id": feature.sample_id
        })
    
    self.is_trained = True
    
    return {
        "n_examples": len(self.few_shot_examples),
        "classes": list(set(f["label"] for f in self.few_shot_examples)),
        "training_method": "few_shot_learning"
    }
```

### Phase 4: Model Persistence

**Goal**: Support save/load operations

**Implementation**:
```python
def save(self, filepath: Union[str, Path]) -> None:
    """Save LLM classifier configuration and examples."""
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)
    
    model_data = {
        "config": self.config,
        "few_shot_examples": getattr(self, "few_shot_examples", []),
        "is_trained": self.is_trained,
        "model_params": self._get_model_params(),
        "classifier_type": "LLM",
        "version": "1.0"
    }
    
    with open(filepath, "wb") as f:
        pickle.dump(model_data, f)
    
    logger.info(f"LLM classifier saved to {filepath}")

@classmethod
def load(cls, filepath: Union[str, Path]) -> "LLMClassifier":
    """Load LLM classifier from file."""
    filepath = Path(filepath)
    
    with open(filepath, "rb") as f:
        model_data = pickle.load(f)
    
    classifier = cls(config=model_data["config"])
    classifier.few_shot_examples = model_data.get("few_shot_examples", [])
    classifier.is_trained = model_data.get("is_trained", False)
    
    logger.info(f"LLM classifier loaded from {filepath}")
    
    return classifier
```

### Phase 5: Evaluation Support

**Goal**: Enable standard evaluation metrics

**Implementation**:
```python
def evaluate(
    self,
    test_features: List[GaitFeatureVector],
    test_labels: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """
    Evaluate LLM classifier on test data.
    
    Note: This can be slow and expensive for LLMs due to API calls.
    Consider using a subset for evaluation.
    """
    if not self.is_trained:
        logger.warning("Evaluating untrained LLM classifier (no few-shot examples)")
    
    predictions = []
    true_labels = []
    confidences = []
    
    for feature in test_features:
        result = self.classify_gait(feature)
        
        # Extract predicted condition
        predicted = result.get("predicted_condition", "unknown")
        confidence = result.get("confidence", 0.0)
        
        predictions.append(predicted)
        confidences.append(confidence)
        true_labels.append(feature.condition_label)
    
    # Calculate metrics using sklearn
    from sklearn.metrics import (
        accuracy_score,
        precision_score,
        recall_score,
        f1_score,
        classification_report
    )
    
    accuracy = accuracy_score(true_labels, predictions)
    precision = precision_score(true_labels, predictions, average="macro", zero_division=0)
    recall = recall_score(true_labels, predictions, average="macro", zero_division=0)
    f1 = f1_score(true_labels, predictions, average="macro", zero_division=0)
    
    return {
        "accuracy": float(accuracy),
        "precision": float(precision),
        "recall": float(recall),
        "f1_score": float(f1),
        "avg_confidence": float(np.mean(confidences)),
        "n_test_samples": len(test_features),
        "classification_report": classification_report(
            true_labels, predictions, output_dict=True, zero_division=0
        )
    }
```

## Implementation Plan

### Step 1: Create Configuration (1 hour)
- [ ] Create `LLMClassifierConfig` dataclass
- [ ] Add validation for provider and model_name
- [ ] Add default values for all parameters
- [ ] Test configuration creation

### Step 2: Refactor Constructor (1 hour)
- [ ] Update `__init__` to accept config
- [ ] Move client initialization to `_create_model()`
- [ ] Implement `_get_model_params()`
- [ ] Test initialization

### Step 3: Adapt classify_gait() (2 hours)
- [ ] Update signature to match base class
- [ ] Add support for `GaitFeatureVector` input
- [ ] Maintain backward compatibility with Dict input
- [ ] Update return format to match base class
- [ ] Test classification

### Step 4: Implement Training (2 hours)
- [ ] Implement `train()` method for few-shot learning
- [ ] Store labeled examples
- [ ] Update prompts to include examples
- [ ] Test training

### Step 5: Add Persistence (1 hour)
- [ ] Implement `save()` method
- [ ] Implement `load()` class method
- [ ] Test save/load cycle
- [ ] Verify examples are preserved

### Step 6: Add Evaluation (1 hour)
- [ ] Implement `evaluate()` method
- [ ] Calculate standard metrics
- [ ] Test evaluation

### Step 7: Update Documentation (2 hours)
- [ ] Update docstrings
- [ ] Update classifier documentation
- [ ] Create migration guide
- [ ] Update examples

### Step 8: Update Tests (3 hours)
- [ ] Update existing tests
- [ ] Add new tests for base class methods
- [ ] Add integration tests
- [ ] Add property-based tests

### Step 9: Update References (1 hour)
- [ ] Update imports in examples
- [ ] Update CLI commands
- [ ] Update server services
- [ ] Update notebooks

## Testing Strategy

### Unit Tests
```python
def test_llm_classifier_initialization():
    """Test LLM classifier initialization with config."""
    config = LLMClassifierConfig(model_name="gpt-4o-mini")
    classifier = LLMClassifier(config)
    assert classifier.config.model_name == "gpt-4o-mini"
    assert classifier.is_trained == False

def test_llm_classifier_with_feature_vector():
    """Test classification with GaitFeatureVector."""
    config = LLMClassifierConfig()
    classifier = LLMClassifier(config)
    
    feature = GaitFeatureVector(
        left_hip_mean=45.0,
        right_hip_mean=50.0,
        condition_label="test"
    )
    
    result = classifier.classify_gait(feature)
    assert "predicted_condition" in result
    assert "confidence" in result

def test_llm_classifier_training():
    """Test few-shot learning training."""
    config = LLMClassifierConfig()
    classifier = LLMClassifier(config)
    
    features = [
        GaitFeatureVector(condition_label="normal"),
        GaitFeatureVector(condition_label="abnormal")
    ]
    
    metrics = classifier.train(features)
    assert classifier.is_trained == True
    assert metrics["n_examples"] == 2

def test_llm_classifier_persistence():
    """Test save/load functionality."""
    config = LLMClassifierConfig()
    classifier = LLMClassifier(config)
    
    # Train with examples
    features = [GaitFeatureVector(condition_label="normal")]
    classifier.train(features)
    
    # Save
    classifier.save("test_llm.pkl")
    
    # Load
    loaded = LLMClassifier.load("test_llm.pkl")
    assert loaded.is_trained == True
    assert len(loaded.few_shot_examples) == 1
```

### Integration Tests
```python
def test_llm_classifier_with_other_classifiers():
    """Test LLM classifier alongside other classifiers."""
    from ambient.classification.knn_classifier import KNNGaitClassifier
    from ambient.classification.rf_classifier import RFGaitClassifier
    
    # All should have same interface
    classifiers = [
        KNNGaitClassifier(),
        RFGaitClassifier(),
        LLMClassifier()
    ]
    
    feature = GaitFeatureVector(condition_label="normal")
    
    for classifier in classifiers:
        result = classifier.classify_gait(feature)
        assert "predicted_condition" in result
        assert "confidence" in result
```

### Property-Based Tests
```python
from hypothesis import given, strategies as st

@given(
    left_hip=st.floats(min_value=0, max_value=180),
    right_hip=st.floats(min_value=0, max_value=180)
)
def test_llm_classifier_handles_any_valid_angles(left_hip, right_hip):
    """Test LLM classifier with any valid joint angles."""
    config = LLMClassifierConfig()
    classifier = LLMClassifier(config)
    
    feature = GaitFeatureVector(
        left_hip_mean=left_hip,
        right_hip_mean=right_hip
    )
    
    result = classifier.classify_gait(feature)
    assert isinstance(result, dict)
    assert "predicted_condition" in result
```

## Migration Guide

### For Users

**Old approach (no longer supported)**:
```python
from ambient.classification.llm_classifier import LLMClassifier

# This will now raise an error
classifier = LLMClassifier(
    model_name="gpt-4o-mini",
    provider="openai",
    temperature=0.1
)
```

**New approach (required)**:
```python
from ambient.classification.llm_classifier import (
    LLMClassifier,
    LLMClassifierConfig
)

config = LLMClassifierConfig(
    model_name="gpt-4o-mini",
    provider="openai",
    temperature=0.1
)

classifier = LLMClassifier(config)

# Now supports both Dict and GaitFeatureVector
result = classifier.classify_gait(feature_vector)
# or
result = classifier.classify_gait(features_dict, context)
```

### Breaking Changes

To maintain backward compatibility during transition:

```python
def __init__(
    self,
    config: Optional[LLMClassifierConfig] = None,
    # Legacy parameters (deprecated)
    model_name: Optional[str] = None,
    provider: Optional[str] = None,
    **kwargs
):
    """Initialize with config or legacy parameters."""
    if config is None:
        # Legacy initialization
        if model_name or provider or kwargs:
            logger.warning(
                "Legacy initialization is deprecated. "
                "Use LLMClassifierConfig instead."
            )
            config = LLMClassifierConfig(
                model_name=model_name or "gpt-4o-mini",
                provider=provider or "openai",
                **kwargs
            )
        else:
            config = LLMClassifierConfig()
    
    super().__init__(config)
```

## Benefits

### Consistency
- ✅ Same interface as other classifiers
- ✅ Consistent configuration pattern
- ✅ Consistent method signatures
- ✅ Consistent return types

### Functionality
- ✅ Training support (few-shot learning)
- ✅ Model persistence (save/load)
- ✅ Evaluation metrics
- ✅ Feature vector support

### Maintainability
- ✅ Reduced code duplication
- ✅ Easier to test
- ✅ Easier to extend
- ✅ Better documentation

### Interoperability
- ✅ Works with ensemble classifiers
- ✅ Works with evaluation frameworks
- ✅ Works with training pipelines
- ✅ Works with model comparison tools

## Risks and Mitigation

### Risk 1: Breaking Changes
**Mitigation**: Maintain backward compatibility with deprecation warnings

### Risk 2: LLM-Specific Features Lost
**Mitigation**: Preserve all LLM-specific methods as private methods

### Risk 3: Performance Overhead
**Mitigation**: LLM operations are already slow; base class overhead is negligible

### Risk 4: API Cost Increase
**Mitigation**: Add caching and rate limiting in base implementation

## Timeline

- **Total Estimated Time**: 14 hours
- **Phase 1-6**: 10 hours (core refactoring)
- **Phase 7-9**: 4 hours (documentation and updates)

## Success Criteria

- [ ] LLMClassifier inherits from BaseGaitClassifier
- [ ] All base class methods implemented
- [ ] All existing functionality preserved
- [ ] Backward compatibility maintained
- [ ] All tests passing
- [ ] Documentation updated
- [ ] Examples updated
- [ ] Integration tests passing
- [ ] Property-based tests passing

## Conclusion

This refactoring will bring LLMClassifier in line with other classifiers while preserving its unique LLM-based functionality. The result will be a more consistent, maintainable, and feature-rich classifier that integrates seamlessly with the rest of the AlexPose system.