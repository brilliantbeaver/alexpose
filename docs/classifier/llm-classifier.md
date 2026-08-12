# LLM Classifier Documentation

## Overview

The `LLMClassifier` provides gait classification using Large Language Models (LLMs) from OpenAI and Google. It performs two-stage classification:
1. Normal vs abnormal gait detection
2. Specific condition identification for abnormal cases

Unlike traditional ML classifiers, the LLM classifier uses few-shot learning and natural language reasoning to classify gait patterns.

## Features

- **Multiple LLM Providers**: OpenAI GPT and Google Gemini
- **Two-Stage Classification**: Normal/abnormal detection + condition identification
- **Few-Shot Learning**: Learn from labeled examples without traditional training
- **Chain-of-Thought Reasoning**: Enhanced reasoning for complex cases
- **Consistent Interface**: Same API as other classifiers (KNN, RF, XGBoost)
- **Model Persistence**: Save and load trained classifiers
- **Feature Vector Support**: Works with `GaitFeatureVector` objects

## Installation

```bash
# For OpenAI support
pip install openai

# For Google Gemini support
pip install google-generativeai

# Or install both
pip install openai google-generativeai
```

## Quick Start

### Basic Usage

```python
from ambient.classification.llm_classifier import (
    LLMClassifier,
    LLMClassifierConfig
)
from ambient.classification.features import GaitFeatureVector

# Create configuration
config = LLMClassifierConfig(
    model_name="gpt-4o-mini",
    provider="openai",
    temperature=0.1
)

# Initialize classifier
classifier = LLMClassifier(config)

# Classify a gait sample
feature = GaitFeatureVector(
    left_hip_mean=45.0,
    right_hip_mean=50.0,
    left_knee_mean=60.0,
    right_knee_mean=65.0,
    condition_label="test"
)

result = classifier.classify_gait(feature)
print(f"Predicted: {result['predicted_condition']}")
print(f"Confidence: {result['confidence']:.2f}")
```

### With Few-Shot Learning

```python
# Prepare training examples
training_features = [
    GaitFeatureVector(
        left_hip_mean=45.0, right_hip_mean=45.0,
        condition_label="normal", sample_id="normal_001"
    ),
    GaitFeatureVector(
        left_hip_mean=40.0, right_hip_mean=55.0,
        condition_label="hemiplegic", sample_id="stroke_001"
    ),
    GaitFeatureVector(
        left_hip_mean=42.0, right_hip_mean=42.0,
        condition_label="parkinsonian", sample_id="parkinsons_001"
    ),
]

# Train with few-shot examples
metrics = classifier.train(training_features)
print(f"Trained with {metrics['n_examples']} examples")
print(f"Classes: {metrics['classes']}")

# Now classify with improved context
result = classifier.classify_gait(test_feature)
```

### Save and Load

```python
# Save trained classifier
classifier.save("my_llm_classifier.pkl")

# Load later
loaded_classifier = LLMClassifier.load("my_llm_classifier.pkl")

# Use loaded classifier
result = loaded_classifier.classify_gait(test_feature)
```

## Configuration

### LLMClassifierConfig

```python
@dataclass
class LLMClassifierConfig(BaseClassifierConfig):
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
    
    # Base settings
    confidence_threshold: float = 0.7
    normalize_features: bool = False  # LLM doesn't need normalization
    cv_n_jobs: int = 1  # LLM is sequential
```

### Configuration Options

#### model_name
The LLM model to use. Supported models:

**OpenAI**:
- `gpt-4o` - Latest GPT-4 Omni (most capable)
- `gpt-4o-mini` - Cost-effective GPT-4 Omni (recommended)
- `gpt-4-turbo` - GPT-4 Turbo
- `gpt-4` - GPT-4
- `gpt-3.5-turbo` - GPT-3.5 Turbo (fastest, cheapest)

**Google Gemini**:
- `gemini-2.0-flash-exp` - Latest Gemini 2.0 Flash
- `gemini-1.5-pro` - Gemini 1.5 Pro (most capable)
- `gemini-1.5-flash` - Gemini 1.5 Flash (fast, cost-effective)
- `gemini-1.0-pro` - Gemini 1.0 Pro

#### provider
LLM provider: `"openai"` or `"gemini"`

#### api_key
API key for the provider. If not provided, reads from environment:
- OpenAI: `OPENAI_API_KEY`
- Gemini: `GOOGLE_API_KEY`

#### temperature
Controls randomness (0.0-1.0):
- `0.0-0.2`: Deterministic, consistent (recommended for classification)
- `0.3-0.7`: Balanced
- `0.8-1.0`: Creative, varied

#### enable_chain_of_thought
Enable step-by-step reasoning for better accuracy on complex cases.

#### two_stage_classification
Enable two-stage classification (normal/abnormal + condition identification).

## API Reference

### LLMClassifier

#### `__init__(config: LLMClassifierConfig)`
Initialize the classifier with configuration.

#### `train(features, labels=None, validate=False, auto_remove_invalid=True)`
Train classifier using few-shot learning.

**Parameters**:
- `features`: List of `GaitFeatureVector` objects
- `labels`: Optional list of labels (uses `feature.condition_label` if None)
- `validate`: Not used for LLM (kept for interface compatibility)
- `auto_remove_invalid`: Remove features with NaN/Inf values

**Returns**: Dictionary with training metrics

#### `classify_gait(gait_features, context=None)`
Classify a gait sample.

**Parameters**:
- `gait_features`: `GaitFeatureVector` or dict with features
- `context`: Optional context information

**Returns**: Dictionary with classification results:
```python
{
    "predicted_condition": str,
    "confidence": float,
    "is_normal": bool,
    "probabilities": Dict[str, float],
    "normal_abnormal_confidence": float,
    "normal_abnormal_explanation": str,
    "identified_conditions": List[Dict],
    "reasoning": str,
    "model_info": Dict,
    "feature_importance": Dict[str, float]
}
```

#### `save(filepath)`
Save classifier to file.

#### `load(filepath)` (class method)
Load classifier from file.

#### `get_supported_models()`
Get list of supported models by provider.

#### `is_model_available()`
Check if the configured model is available.

#### `explain_classification(result)`
Generate human-readable explanation of classification.

## Examples

### Example 1: Basic Classification

```python
from ambient.classification.llm_classifier import (
    LLMClassifier,
    LLMClassifierConfig
)
from ambient.classification.features import GaitFeatureVector

# Configure
config = LLMClassifierConfig(
    model_name="gpt-4o-mini",
    provider="openai",
    temperature=0.1
)

# Initialize
classifier = LLMClassifier(config)

# Create feature
feature = GaitFeatureVector(
    left_hip_mean=45.0,
    right_hip_mean=50.0,
    left_knee_mean=60.0,
    right_knee_mean=65.0,
    left_ankle_mean=20.0,
    right_ankle_mean=25.0,
)

# Classify
result = classifier.classify_gait(feature)

# Print results
print(f"Condition: {result['predicted_condition']}")
print(f"Confidence: {result['confidence']:.2f}")
print(f"Is Normal: {result['is_normal']}")

# Get explanation
explanation = classifier.explain_classification(result)
print(explanation)
```

### Example 2: Few-Shot Learning

```python
# Prepare training data
training_data = [
    GaitFeatureVector(
        left_hip_mean=45.0, right_hip_mean=45.0,
        left_knee_mean=60.0, right_knee_mean=60.0,
        condition_label="normal"
    ),
    GaitFeatureVector(
        left_hip_mean=40.0, right_hip_mean=55.0,
        left_knee_mean=55.0, right_knee_mean=70.0,
        condition_label="hemiplegic"
    ),
    GaitFeatureVector(
        left_hip_mean=42.0, right_hip_mean=42.0,
        left_knee_mean=50.0, right_knee_mean=50.0,
        condition_label="parkinsonian"
    ),
]

# Train
metrics = classifier.train(training_data)
print(f"Trained with {metrics['n_examples']} examples")

# Classify with improved context
result = classifier.classify_gait(test_feature)
```

### Example 3: Using Gemini

```python
# Configure for Gemini
config = LLMClassifierConfig(
    model_name="gemini-1.5-flash",
    provider="gemini",
    temperature=0.1
)

classifier = LLMClassifier(config)

# Use same API
result = classifier.classify_gait(feature)
```

### Example 4: Batch Classification

```python
# Classify multiple samples
test_features = [
    GaitFeatureVector(...),
    GaitFeatureVector(...),
    GaitFeatureVector(...),
]

results = []
for feature in test_features:
    result = classifier.classify_gait(feature)
    results.append(result)

# Analyze results
for i, result in enumerate(results):
    print(f"Sample {i+1}: {result['predicted_condition']} "
          f"({result['confidence']:.2f})")
```

## Best Practices

### 1. Use Few-Shot Learning
Provide 3-10 labeled examples for better accuracy:
```python
classifier.train(labeled_examples)
```

### 2. Set Low Temperature
Use temperature 0.1-0.2 for consistent classification:
```python
config = LLMClassifierConfig(temperature=0.1)
```

### 3. Enable Chain-of-Thought
For complex cases, enable reasoning:
```python
config = LLMClassifierConfig(enable_chain_of_thought=True)
```

### 4. Cache Results
LLM calls are expensive; cache results for repeated queries:
```python
cache = {}
feature_key = tuple(feature.to_array())
if feature_key in cache:
    result = cache[feature_key]
else:
    result = classifier.classify_gait(feature)
    cache[feature_key] = result
```

### 5. Handle API Errors
Wrap classification in try-except:
```python
try:
    result = classifier.classify_gait(feature)
except Exception as e:
    logger.error(f"Classification failed: {e}")
    result = {"predicted_condition": "unknown", "confidence": 0.0}
```

## Performance Considerations

### Speed
- LLM classification: 1-5 seconds per sample
- Traditional ML: < 0.01 seconds per sample
- Use LLM for complex cases, ML for high-throughput

### Cost
- OpenAI GPT-4o-mini: ~$0.15 per 1M input tokens
- Google Gemini Flash: ~$0.075 per 1M input tokens
- Few-shot examples increase token usage

### Accuracy
- Few-shot learning improves accuracy
- Chain-of-thought helps complex cases
- Consider ensemble with traditional ML

## Troubleshooting

### API Key Not Found
```
ValueError: OpenAI API key required
```
**Solution**: Set environment variable or pass in config:
```bash
export OPENAI_API_KEY=sk-...
```

### Model Not Available
```
Error: Model gpt-5 not found
```
**Solution**: Use supported model:
```python
models = classifier.get_supported_models()
print(models["openai"])
```

### Rate Limit Exceeded
```
RateLimitError: Rate limit exceeded
```
**Solution**: Add retry logic or reduce request rate:
```python
import time
time.sleep(1)  # Wait between requests
```

## Comparison with Traditional ML

| Feature | LLM Classifier | Traditional ML |
|---------|---------------|----------------|
| Training | Few-shot learning | Requires many samples |
| Speed | Slow (1-5s) | Fast (<0.01s) |
| Cost | API costs | Free after training |
| Accuracy | Good with few examples | Better with many examples |
| Interpretability | Natural language explanations | Feature importance |
| Flexibility | Adapts to new conditions | Needs retraining |

## See Also

- [KNN Classifier](knn-classifier.md)
- [Random Forest Classifier](rf-classifier.md)
- [Ensemble Classifier](ensemble-classifier.md)
- [Prompt Manager](../analysis/llm-classification.md)