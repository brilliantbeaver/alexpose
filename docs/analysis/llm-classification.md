# LLM Classification Documentation

## Overview

The LLM Classification system provides AI-powered gait analysis using large language models. It implements a two-stage classification approach: first determining if gait is normal or abnormal, then identifying specific conditions for abnormal cases. The system supports multiple LLM providers with configurable prompts and fallback mechanisms.

## LLMClassifier Class

### Location
`ambient/classification/llm_classifier.py`

### Initialization

```python
from ambient.classification.llm_classifier import LLMClassifier

classifier = LLMClassifier(
    config_manager=config_manager,     # Configuration manager instance
    primary_model="gpt-4o-mini",       # Primary LLM model
    fallback_model="gemini-1.5-pro",   # Fallback model
    temperature=0.1,                   # Model temperature (0.0-1.0)
    max_tokens=1000                    # Maximum response tokens
)
```

## Supported Models

### OpenAI Models
- **gpt-4o-mini** (default): Fast, cost-effective, high-quality
- **gpt-4o**: Most capable model for complex analysis
- **gpt-4-turbo**: Balanced performance and cost
- **gpt-3.5-turbo**: Legacy model, lower cost

### Google Gemini Models
- **gemini-1.5-pro**: High-quality analysis with multimodal support
- **gemini-1.5-flash**: Fast processing for real-time applications
- **gemini-1.0-pro**: Stable baseline model

### Model Configuration

```yaml
# config/alexpose.yaml - LLM Configuration Section
classification:
  llm:
    models:
  openai:
    api_key_env: "OPENAI_API_KEY"
    models:
      gpt-4o-mini:
        max_tokens: 1000
        temperature: 0.1
        supports_multimodal: false
      gpt-4o:
        max_tokens: 2000
        temperature: 0.1
        supports_multimodal: true
  
  gemini:
    api_key_env: "GEMINI_API_KEY"
    models:
      gemini-1.5-pro:
        max_tokens: 1000
        temperature: 0.1
        supports_multimodal: true

default_model: "gpt-4o-mini"
fallback_model: "gemini-1.5-pro"
```

## Two-Stage Classification

### Stage 1: Normal vs Abnormal Classification

```python
# Classify gait as normal or abnormal
normal_abnormal_result = classifier.classify_normal_abnormal(gait_features)

normal_abnormal_result = {
    'classification': 'abnormal',      # 'normal' or 'abnormal'
    'confidence': 0.85,               # Confidence score (0.0-1.0)
    'reasoning': 'Significant asymmetry in step length and timing patterns suggest gait abnormality',
    'key_indicators': [
        'Step length asymmetry: 15%',
        'Cadence variability: High',
        'Stance phase imbalance'
    ],
    'model_used': 'gpt-4o-mini',
    'processing_time': 1.2
}
```

### Stage 2: Condition Identification

```python
# Identify specific condition for abnormal gait
condition_result = classifier.identify_condition(gait_features)

condition_result = {
    'primary_condition': 'Antalgic Gait',
    'confidence': 0.78,
    'alternative_conditions': [
        {'condition': 'Hip Pathology', 'confidence': 0.65},
        {'condition': 'Lower Limb Pain', 'confidence': 0.58}
    ],
    'clinical_indicators': [
        'Shortened stance phase on affected side',
        'Reduced weight bearing time',
        'Compensatory trunk lean'
    ],
    'recommendations': [
        'Clinical examination of hip and lower extremity',
        'Pain assessment and management',
        'Consider imaging studies if indicated'
    ],
    'severity': 'moderate',
    'model_used': 'gpt-4o-mini',
    'processing_time': 1.8
}
```

## Prompt Management

### PromptManager Class

```python
from ambient.classification.prompt_manager import PromptManager

prompt_manager = PromptManager(
    config_path="config/classification_prompts.yaml"
)

# Get prompts for classification
normal_abnormal_prompt = prompt_manager.get_prompt("normal_abnormal_classification")
condition_prompt = prompt_manager.get_prompt("condition_identification")
```

### Prompt Configuration

```yaml
# config/classification_prompts.yaml
prompts:
  normal_abnormal_classification:
    system_prompt: |
      You are an expert gait analysis specialist with extensive clinical experience.
      Your task is to classify gait patterns as either NORMAL or ABNORMAL based on 
      quantitative gait features and biomechanical analysis.
      
      Classification Criteria:
      - NORMAL: Gait parameters within expected ranges, symmetric patterns, smooth movement
      - ABNORMAL: Significant deviations from normal ranges, asymmetries, or irregular patterns
      
      Provide your analysis with confidence scoring and clear reasoning.
    
    user_prompt_template: |
      Analyze the following gait features and classify as NORMAL or ABNORMAL:
      
      Temporal Features:
      - Cadence: {cadence} steps/min (normal: 100-130)
      - Step regularity CV: {step_regularity_cv} (normal: <0.1)
      - Cycle duration: {cycle_duration} seconds (normal: 0.9-1.3)
      
      Symmetry Features:
      - Overall symmetry index: {overall_symmetry_index} (normal: <0.05)
      - Step length asymmetry: {step_length_asymmetry}%
      - Stance time asymmetry: {stance_time_asymmetry}%
      
      Kinematic Features:
      - Velocity consistency: {velocity_cv}
      - Joint angle ranges: {joint_ranges}
      - Movement smoothness: {movement_smoothness}
      
      Provide your classification with:
      1. Classification: NORMAL or ABNORMAL
      2. Confidence: 0.0-1.0
      3. Key indicators supporting your decision
      4. Clinical reasoning
  
  condition_identification:
    system_prompt: |
      You are a clinical gait analysis expert specializing in pathological gait pattern recognition.
      Your task is to identify specific conditions that may cause abnormal gait patterns.
      
      Common Gait Conditions:
      - Antalgic Gait: Pain-related limping
      - Trendelenburg Gait: Hip abductor weakness
      - Steppage Gait: Foot drop compensation
      - Hemiplegic Gait: Stroke-related patterns
      - Parkinsonian Gait: Reduced arm swing, shuffling
      - Ataxic Gait: Cerebellar dysfunction
      
      Provide differential diagnosis with confidence levels and clinical recommendations.
    
    user_prompt_template: |
      Based on the abnormal gait features below, identify the most likely condition:
      
      Gait Abnormalities Detected:
      {abnormality_summary}
      
      Detailed Features:
      {detailed_features}
      
      Asymmetry Patterns:
      {asymmetry_patterns}
      
      Temporal Irregularities:
      {temporal_irregularities}
      
      Provide:
      1. Primary condition with confidence level
      2. Alternative conditions (differential diagnosis)
      3. Key clinical indicators
      4. Recommended clinical actions
      5. Severity assessment
```

## Agentic Classification with Chain-of-Thought

### Advanced Reasoning

```python
# Enable chain-of-thought reasoning
classifier = LLMClassifier(
    enable_chain_of_thought=True,
    reasoning_depth="detailed"  # "basic", "detailed", "comprehensive"
)

# Classification with detailed reasoning
result = classifier.classify_with_reasoning(gait_features)

result = {
    'classification': 'abnormal',
    'confidence': 0.82,
    'reasoning_chain': [
        {
            'step': 1,
            'observation': 'Cadence is 85 steps/min, below normal range (100-130)',
            'implication': 'Suggests slow, cautious gait pattern'
        },
        {
            'step': 2,
            'observation': 'Step length asymmetry of 18% detected',
            'implication': 'Indicates possible unilateral pathology or pain'
        },
        {
            'step': 3,
            'observation': 'Reduced stance phase on right side (45% vs normal 60%)',
            'implication': 'Consistent with antalgic gait pattern'
        },
        {
            'step': 4,
            'conclusion': 'Pattern consistent with pain-related gait abnormality',
            'confidence_factors': ['Multiple converging indicators', 'Clear asymmetry pattern']
        }
    ],
    'final_assessment': 'Abnormal gait with high confidence based on converging evidence'
}
```

### Multi-Modal Analysis

```python
# For models that support images (GPT-4V, Gemini Pro Vision)
multimodal_result = classifier.classify_multimodal(
    gait_features=gait_features,
    gait_video_frames=video_frames,  # Optional video frames
    pose_visualization=pose_plot     # Optional pose visualization
)
```

## Error Handling and Fallback

### Robust Classification

```python
def robust_classification(classifier, gait_features):
    """Perform robust classification with fallback mechanisms."""
    try:
        # Primary classification attempt
        result = classifier.classify_normal_abnormal(gait_features)
        
        if result['confidence'] < 0.6:
            # Low confidence - try fallback model
            logger.warning("Low confidence result, trying fallback model")
            fallback_result = classifier.classify_with_fallback(gait_features)
            
            # Combine results if both available
            if fallback_result:
                result = combine_classification_results(result, fallback_result)
        
        return result
        
    except Exception as e:
        logger.error(f"LLM classification failed: {e}")
        
        # Fallback to rule-based classification
        return rule_based_fallback_classification(gait_features)

def rule_based_fallback_classification(gait_features):
    """Simple rule-based fallback when LLM fails."""
    # Basic rules for normal/abnormal classification
    abnormal_indicators = 0
    
    if gait_features.get('overall_symmetry_index', 0) > 0.1:
        abnormal_indicators += 1
    
    if gait_features.get('cadence', 115) < 90 or gait_features.get('cadence', 115) > 140:
        abnormal_indicators += 1
    
    if gait_features.get('step_regularity_cv', 0.05) > 0.15:
        abnormal_indicators += 1
    
    is_abnormal = abnormal_indicators >= 2
    
    return {
        'classification': 'abnormal' if is_abnormal else 'normal',
        'confidence': 0.7 if abnormal_indicators >= 2 else 0.6,
        'reasoning': f'Rule-based classification: {abnormal_indicators} abnormal indicators detected',
        'method': 'rule_based_fallback',
        'abnormal_indicators': abnormal_indicators
    }
```

### API Error Handling

```python
class LLMClassifier:
    def _handle_api_errors(self, api_call_func, *args, **kwargs):
        """Handle common API errors with appropriate responses."""
        try:
            return api_call_func(*args, **kwargs)
            
        except openai.RateLimitError:
            logger.warning("Rate limit exceeded, waiting and retrying...")
            time.sleep(60)  # Wait 1 minute
            return api_call_func(*args, **kwargs)
            
        except openai.AuthenticationError:
            logger.error("Authentication failed - check API key")
            raise ValueError("Invalid API key configuration")
            
        except openai.APIConnectionError:
            logger.warning("API connection failed, trying fallback model")
            return self._try_fallback_model(*args, **kwargs)
            
        except Exception as e:
            logger.error(f"Unexpected API error: {e}")
            return self._rule_based_fallback(*args, **kwargs)
```

## Performance Optimization

### Batch Processing

```python
# Process multiple analyses in batch
batch_results = classifier.classify_batch([
    gait_features_1,
    gait_features_2,
    gait_features_3
])

# Async processing for better performance
import asyncio

async def classify_async(classifier, gait_features_list):
    """Asynchronous classification for multiple samples."""
    tasks = [
        classifier.classify_async(features) 
        for features in gait_features_list
    ]
    
    results = await asyncio.gather(*tasks)
    return results
```

### Caching

```python
from functools import lru_cache
import hashlib

class LLMClassifier:
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.cache_enabled = kwargs.get('cache_enabled', True)
        self.cache = {}
    
    def _get_cache_key(self, gait_features):
        """Generate cache key for gait features."""
        features_str = json.dumps(gait_features, sort_keys=True)
        return hashlib.md5(features_str.encode()).hexdigest()
    
    def classify_normal_abnormal(self, gait_features):
        """Classify with caching support."""
        if self.cache_enabled:
            cache_key = self._get_cache_key(gait_features)
            if cache_key in self.cache:
                logger.debug("Using cached classification result")
                return self.cache[cache_key]
        
        result = self._perform_classification(gait_features)
        
        if self.cache_enabled:
            self.cache[cache_key] = result
        
        return result
```

## Integration Examples

### With Gait Analysis

```python
from ambient.analysis.gait_analyzer import EnhancedGaitAnalyzer
from ambient.classification.llm_classifier import LLMClassifier

# Complete analysis pipeline
analyzer = EnhancedGaitAnalyzer()
classifier = LLMClassifier()

# Analyze gait sequence
gait_results = analyzer.analyze_gait_sequence(pose_sequence)

# Classify results
classification = classifier.classify_gait(gait_results)

# Combined results
complete_results = {
    **gait_results,
    'classification': classification
}
```

### With Web API

```python
from fastapi import APIRouter
from ambient.classification.llm_classifier import LLMClassifier

router = APIRouter()
classifier = LLMClassifier()

@router.post("/classify-gait")
async def classify_gait_endpoint(gait_data: GaitAnalysisRequest):
    """API endpoint for gait classification."""
    try:
        # Extract features from request
        gait_features = gait_data.features
        
        # Perform classification
        classification = await classifier.classify_async(gait_features)
        
        return {
            'success': True,
            'classification': classification,
            'timestamp': datetime.utcnow()
        }
        
    except Exception as e:
        logger.error(f"Classification endpoint error: {e}")
        return {
            'success': False,
            'error': str(e),
            'fallback_available': True
        }
```

## Monitoring and Analytics

### Classification Metrics

```python
class ClassificationMetrics:
    def __init__(self):
        self.total_classifications = 0
        self.normal_count = 0
        self.abnormal_count = 0
        self.confidence_scores = []
        self.processing_times = []
        self.model_usage = {}
    
    def record_classification(self, result):
        """Record classification metrics."""
        self.total_classifications += 1
        
        if result['classification'] == 'normal':
            self.normal_count += 1
        else:
            self.abnormal_count += 1
        
        self.confidence_scores.append(result['confidence'])
        self.processing_times.append(result.get('processing_time', 0))
        
        model = result.get('model_used', 'unknown')
        self.model_usage[model] = self.model_usage.get(model, 0) + 1
    
    def get_summary(self):
        """Get classification summary statistics."""
        return {
            'total_classifications': self.total_classifications,
            'normal_percentage': (self.normal_count / self.total_classifications) * 100,
            'abnormal_percentage': (self.abnormal_count / self.total_classifications) * 100,
            'average_confidence': np.mean(self.confidence_scores),
            'average_processing_time': np.mean(self.processing_times),
            'model_usage_distribution': self.model_usage
        }
```

## See Also

- [Gait Analysis](gait-analysis.md) - Main gait analysis documentation
- [Feature Extraction](feature-extraction.md) - Feature extraction for classification
- [Configuration](../guides/configuration.md) - System configuration options
- [API Reference](../api/classification.md) - Classification API documentation