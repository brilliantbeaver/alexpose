"""
LLM-based classification engine for gait analysis.

This module provides classification using modern LLM models including
OpenAI GPT models and Google Gemini models with configurable prompts
and two-stage classification (normal/abnormal, then condition identification).

Author: AlexPose Team
"""

import json
import time
from typing import Dict, List, Optional, Any, Union
from pathlib import Path
from dataclasses import dataclass
from loguru import logger

# Load environment variables from .env file
try:
    from dotenv import load_dotenv
    load_dotenv()
    DOTENV_AVAILABLE = True
except ImportError:
    DOTENV_AVAILABLE = False

from ambient.core.interfaces import IClassifier
from ambient.classification.base_classifier import BaseGaitClassifier, BaseClassifierConfig
from ambient.classification.features import GaitFeatureVector
from ambient.classification.prompt_manager import PromptManager

# OpenAI imports with fallback
try:
    import openai
    from openai import OpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    openai = None
    OpenAI = None
    OPENAI_AVAILABLE = False

# Google Gemini imports with fallback
try:
    import google.generativeai as genai
    GEMINI_AVAILABLE = True
except ImportError:
    genai = None
    GEMINI_AVAILABLE = False


@dataclass
class LLMClassifierConfig(BaseClassifierConfig):
    """Configuration for LLM classifier."""
    
    # LLM provider settings
    model_name: str = "gpt-4o-mini"  # Changed default from gpt-5.2 to more stable model
    provider: str = "openai"  # "openai" or "gemini"
    api_key: Optional[str] = None
    
    # Generation settings
    temperature: float = 0.1
    max_tokens: Optional[int] = None  # Token limit (parameter name varies by model)
    
    # Classification settings
    enable_chain_of_thought: bool = True
    two_stage_classification: bool = True
    
    # Prompt settings
    prompt_config_path: Optional[Path] = None
    
    # Override base settings for LLM
    normalize_features: bool = False  # LLM doesn't need feature normalization
    cv_n_jobs: int = 1  # LLM is sequential by nature


class LLMClassifier(BaseGaitClassifier):
    """
    LLM-based classifier for gait analysis.
    
    This classifier uses modern language models to perform two-stage classification:
    1. Normal vs abnormal gait classification
    2. Specific condition identification for abnormal cases
    
    Supports multiple LLM providers with configurable prompts and agentic reasoning.
    Unlike traditional ML classifiers, this uses few-shot learning with labeled examples.
    
    Example:
        >>> from ambient.classification.llm_classifier import (
        ...     LLMClassifier,
        ...     LLMClassifierConfig
        ... )
        >>> from ambient.classification.features import GaitFeatureVector
        >>>
        >>> # Initialize classifier
        >>> config = LLMClassifierConfig(
        ...     model_name="gpt-4o-mini",
        ...     provider="openai",
        ...     temperature=0.1
        ... )
        >>> classifier = LLMClassifier(config)
        >>>
        >>> # Optional: Train with few-shot examples
        >>> training_features = [
        ...     GaitFeatureVector(condition_label="normal"),
        ...     GaitFeatureVector(condition_label="abnormal")
        ... ]
        >>> classifier.train(training_features)
        >>>
        >>> # Classify new sample
        >>> test_feature = GaitFeatureVector(...)
        >>> result = classifier.classify_gait(test_feature)
        >>> print(f"Predicted: {result['predicted_condition']}")
    """
    
    def __init__(
        self,
        config: Optional[LLMClassifierConfig] = None,
        prompt_manager: Optional[PromptManager] = None,
    ):
        """
        Initialize LLM classifier.
        
        Args:
            config: Configuration object. If None, uses default LLMClassifierConfig.
            prompt_manager: Optional custom prompt manager instance.
                          If None, creates default PromptManager.
        
        Example:
            >>> from ambient.classification.llm_classifier import (
            ...     LLMClassifier,
            ...     LLMClassifierConfig
            ... )
            >>> 
            >>> # Use default configuration
            >>> classifier = LLMClassifier()
            >>> 
            >>> # Use custom configuration
            >>> config = LLMClassifierConfig(
            ...     model_name="gpt-4o-mini",
            ...     provider="openai",
            ...     temperature=0.1
            ... )
            >>> classifier = LLMClassifier(config)
        """
        # Use default config if none provided
        if config is None:
            config = LLMClassifierConfig()
        
        # Initialize base class
        super().__init__(config)
        self.config: LLMClassifierConfig = config
        
        # Initialize prompt manager
        if prompt_manager:
            self.prompt_manager = prompt_manager
        else:
            self.prompt_manager = PromptManager(config.prompt_config_path)
        
        # Few-shot learning examples
        self.few_shot_examples: List[Dict[str, Any]] = []
        
        logger.info(
            f"LLM classifier initialized with {config.provider} {config.model_name}"
        )
    
    def _create_model(self):
        """Create and return LLM client."""
        if self.config.provider == "openai":
            return self._create_openai_client()
        elif self.config.provider == "gemini":
            return self._create_gemini_client()
        else:
            raise ValueError(f"Unsupported provider: {self.config.provider}")
    
    def _get_model_params(self) -> Dict[str, Any]:
        """Get LLM-specific parameters for saving."""
        return {
            "provider": self.config.provider,
            "model_name": self.config.model_name,
            "temperature": self.config.temperature,
            "enable_chain_of_thought": self.config.enable_chain_of_thought,
            "two_stage_classification": self.config.two_stage_classification,
            "few_shot_examples_count": len(self.few_shot_examples),
        }
    
    def _create_openai_client(self):
        """Initialize OpenAI client."""
        if not OPENAI_AVAILABLE:
            raise ImportError(
                "OpenAI library not available. Install with: pip install openai"
            )
        
        # Get API key from config or environment
        api_key = self.config.api_key
        if not api_key:
            import os
            api_key = os.getenv("OPENAI_API_KEY")
        
        if not api_key:
            raise ValueError(
                "OpenAI API key required. Set OPENAI_API_KEY environment variable "
                "or pass api_key in config."
            )
        
        return OpenAI(api_key=api_key)
    
    def _create_gemini_client(self):
        """Initialize Gemini client."""
        if not GEMINI_AVAILABLE:
            raise ImportError(
                "Google Generative AI library not available. "
                "Install with: pip install google-generativeai"
            )
        
        # Get API key from config or environment
        api_key = self.config.api_key
        if not api_key:
            import os
            api_key = os.getenv("GOOGLE_API_KEY")
        
        if not api_key:
            raise ValueError(
                "Google API key required. Set GOOGLE_API_KEY environment variable "
                "or pass api_key in config."
            )
        
        genai.configure(api_key=api_key)
        return genai.GenerativeModel(
            model_name=self.config.model_name,
            generation_config=genai.GenerationConfig(
                temperature=self.config.temperature,
                max_output_tokens=self.config.max_tokens
            )
        )
    
    def train(
        self,
        features: List[GaitFeatureVector],
        labels: Optional[List[str]] = None,
        validate: bool = False,  # LLM doesn't use traditional CV
        auto_remove_invalid: bool = True,
    ) -> Dict[str, Any]:
        """
        'Train' LLM classifier by storing examples for few-shot learning.
        
        Note: LLMs don't train in the traditional sense. This method stores
        labeled examples that will be included in prompts for few-shot learning.
        
        Args:
            features: List of GaitFeatureVector objects
            labels: Optional list of condition labels (uses feature.condition_label if None)
            validate: Not used for LLM (kept for interface compatibility)
            auto_remove_invalid: If True, automatically remove samples with NaN/Inf
            
        Returns:
            Dictionary with training metrics
        """
        if not features:
            raise ValueError("No training features provided")
        
        logger.info(f"Training LLM classifier with {len(features)} examples (few-shot learning)")
        
        self.few_shot_examples = []
        invalid_count = 0
        
        for i, feature in enumerate(features):
            # Use provided label or feature's label
            label = labels[i] if labels and i < len(labels) else feature.condition_label
            
            # Validate feature if requested
            if auto_remove_invalid:
                is_valid, issues = feature.validate()
                if not is_valid:
                    logger.warning(
                        f"Skipping invalid feature {feature.sample_id}: {issues}"
                    )
                    invalid_count += 1
                    continue
            
            # Store example for few-shot learning
            self.few_shot_examples.append({
                "features": self._feature_vector_to_dict(feature),
                "label": label,
                "sample_id": feature.sample_id,
                "feature_array": feature.to_array().tolist()
            })
        
        self.is_trained = True
        
        # Get unique classes
        classes = list(set(ex["label"] for ex in self.few_shot_examples))
        
        metrics = {
            "n_examples": len(self.few_shot_examples),
            "n_invalid_removed": invalid_count,
            "classes": classes,
            "training_method": "few_shot_learning",
            "provider": self.config.provider,
            "model_name": self.config.model_name
        }
        
        logger.info(
            f"LLM classifier 'trained' with {len(self.few_shot_examples)} examples "
            f"({invalid_count} invalid removed)"
        )
        logger.info(f"Classes: {classes}")
        
        return metrics
    
    def classify_gait(
        self,
        gait_features: Union[GaitFeatureVector, Dict[str, Any]],
        context: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Classify gait as normal/abnormal and identify conditions.
        
        Overrides base class to support both GaitFeatureVector and Dict inputs.
        
        Args:
            gait_features: GaitFeatureVector or dict with extracted gait features
            context: Optional context information
            
        Returns:
            Dictionary containing classification results with confidence scores
        """
        context = context or {}
        
        # Convert GaitFeatureVector to dict if needed
        if isinstance(gait_features, GaitFeatureVector):
            features_dict = self._feature_vector_to_dict(gait_features)
        else:
            features_dict = gait_features
        
        # Initialize model if not already done
        if self.model is None:
            self.model = self._create_model()
        
        try:
            # Stage 1: Normal vs Abnormal Classification
            logger.info("Performing normal/abnormal classification")
            normal_abnormal_result = self._classify_normal_abnormal(features_dict, context)
            
            # Determine predicted condition and confidence
            is_normal = normal_abnormal_result.get("is_normal", True)
            confidence = normal_abnormal_result.get("confidence", 0.0)
            
            # Stage 2: Condition Identification (if abnormal and two-stage enabled)
            condition_results = []
            predicted_condition = "normal" if is_normal else "abnormal"
            
            if not is_normal and self.config.two_stage_classification:
                logger.info("Performing condition identification")
                condition_results = self._identify_conditions(features_dict, context)
                
                # Use most confident condition as predicted_condition
                if condition_results:
                    most_confident = max(condition_results, key=lambda x: x.get("confidence", 0))
                    predicted_condition = most_confident.get("condition_name", "abnormal")
                    # Update confidence to be weighted average
                    condition_confidence = most_confident.get("confidence", 0.0)
                    confidence = (confidence * 0.7) + (condition_confidence * 0.3)
            
            # Build result in base class format
            classification_result = {
                "predicted_condition": predicted_condition,
                "confidence": float(confidence),
                "is_normal": is_normal,
                "probabilities": self._build_probability_dict(
                    predicted_condition, confidence, condition_results
                ),
                # LLM-specific fields
                "normal_abnormal_confidence": normal_abnormal_result.get("confidence", 0.0),
                "normal_abnormal_explanation": normal_abnormal_result.get("explanation", ""),
                "identified_conditions": condition_results,
                "reasoning": normal_abnormal_result.get("reasoning", ""),
                "classification_timestamp": time.time(),
                "model_info": {
                    "provider": self.config.provider,
                    "model_name": self.config.model_name,
                    "temperature": self.config.temperature,
                    "few_shot_examples": len(self.few_shot_examples)
                }
            }
            
            # Add feature importance if available
            feature_importance = self._calculate_feature_importance(features_dict, classification_result)
            if feature_importance:
                classification_result["feature_importance"] = feature_importance
            
            logger.info(
                f"Classification: {predicted_condition} (confidence: {confidence:.3f})"
            )
            
            return classification_result
            
        except Exception as e:
            logger.error(f"LLM classification failed: {e}")
            return {
                "predicted_condition": "unknown",
                "confidence": 0.0,
                "is_normal": True,  # Default to normal on error
                "probabilities": {"unknown": 1.0},
                "normal_abnormal_confidence": 0.0,
                "normal_abnormal_explanation": f"Classification failed: {str(e)}",
                "identified_conditions": [],
                "error": str(e),
                "classification_timestamp": time.time()
            }
    
    def _feature_vector_to_dict(self, feature: GaitFeatureVector) -> Dict[str, Any]:
        """Convert GaitFeatureVector to dictionary for LLM processing."""
        feature_names = GaitFeatureVector.get_feature_names()
        feature_array = feature.to_array()
        
        features_dict = {
            name: float(value) 
            for name, value in zip(feature_names, feature_array)
        }
        
        # Add metadata
        if feature.sample_id:
            features_dict["sample_id"] = feature.sample_id
        if feature.condition_label:
            features_dict["condition_label"] = feature.condition_label
        
        return features_dict
    
    def _build_probability_dict(
        self,
        predicted_condition: str,
        confidence: float,
        condition_results: List[Dict[str, Any]]
    ) -> Dict[str, float]:
        """Build probability dictionary for base class compatibility."""
        probabilities = {}
        
        # Add predicted condition
        probabilities[predicted_condition] = confidence
        
        # Add other identified conditions
        for condition in condition_results:
            condition_name = condition.get("condition_name", "unknown")
            condition_conf = condition.get("confidence", 0.0)
            if condition_name != predicted_condition:
                probabilities[condition_name] = condition_conf
        
        # Normalize probabilities to sum to 1.0
        total = sum(probabilities.values())
        if total > 0:
            probabilities = {k: v / total for k, v in probabilities.items()}
        
        return probabilities
    
    def _classify_normal_abnormal(
        self, 
        gait_features: Dict[str, Any], 
        context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Perform normal vs abnormal classification."""
        # Get prompt for normal/abnormal classification
        prompt = self.prompt_manager.get_normal_abnormal_prompt(
            gait_features=gait_features,
            context=context,
            enable_chain_of_thought=self.config.enable_chain_of_thought
        )
        
        # Add few-shot examples if available
        if self.few_shot_examples:
            prompt = self._add_few_shot_examples(prompt, "normal_abnormal")
        
        # Generate response
        response = self._generate_response(prompt)
        
        # Parse response
        return self._parse_normal_abnormal_response(response)
    
    def _identify_conditions(
        self, 
        gait_features: Dict[str, Any], 
        context: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Identify specific conditions for abnormal gait."""
        # Get prompt for condition identification
        prompt = self.prompt_manager.get_condition_identification_prompt(
            gait_features=gait_features,
            context=context,
            enable_chain_of_thought=self.config.enable_chain_of_thought
        )
        
        # Add few-shot examples if available
        if self.few_shot_examples:
            prompt = self._add_few_shot_examples(prompt, "condition_identification")
        
        # Generate response
        response = self._generate_response(prompt)
        
        # Parse response
        return self._parse_condition_response(response)
    
    def _add_few_shot_examples(self, prompt: str, task_type: str) -> str:
        """Add few-shot examples to prompt."""
        if not self.few_shot_examples:
            return prompt
        
        examples_text = "\n\nHere are some labeled examples to help with classification:\n\n"
        
        for i, example in enumerate(self.few_shot_examples[:5], 1):  # Limit to 5 examples
            features = example["features"]
            label = example["label"]
            
            examples_text += f"Example {i}:\n"
            examples_text += f"Features: {json.dumps(features, indent=2)}\n"
            examples_text += f"Label: {label}\n\n"
        
        # Insert examples before the main question
        return prompt + examples_text
    
    def _get_token_parameter_name(self, model_name: str) -> str:
        """
        Get the correct token parameter name based on model type.
        
        IMPORTANT: This is for Chat Completions API only. The Responses API 
        uses different parameters but we're not using that API.
        
        For Chat Completions API:
        - Legacy models (GPT-3.5, GPT-4, GPT-4o, GPT-5): max_tokens
        - O-series models (o1, o3, o3-mini): max_completion_tokens
        
        Args:
            model_name: The OpenAI model name
            
        Returns:
            The correct parameter name for token limits in Chat Completions API
        """
        model_lower = model_name.lower()
        
        # O-series models use max_completion_tokens in Chat Completions API
        if any(model_lower.startswith(prefix) for prefix in ["o1", "o3", "o4"]):
            return "max_completion_tokens"
        
        # All other models (including GPT-3.5, GPT-4, GPT-4o, GPT-5) use max_tokens
        # in Chat Completions API
        return "max_tokens"

    def _generate_response(self, prompt: str) -> str:
        """Generate response from LLM."""
        try:
            if self.config.provider == "openai":
                # Build request parameters with model-specific token parameter
                request_params = {
                    "model": self.config.model_name,
                    "messages": [
                        {"role": "system", "content": "You are an expert in gait analysis and medical diagnosis."},
                        {"role": "user", "content": prompt}
                    ],
                    "temperature": self.config.temperature,
                }
                
                # Add token limit parameter based on model type
                if self.config.max_tokens is not None:
                    token_param = self._get_token_parameter_name(self.config.model_name)
                    request_params[token_param] = self.config.max_tokens
                
                response = self.model.chat.completions.create(**request_params)
                return response.choices[0].message.content
                
            elif self.config.provider == "gemini":
                response = self.model.generate_content(prompt)
                return response.text
                
            else:
                raise ValueError(f"Unsupported provider: {self.config.provider}")
                
        except Exception as e:
            logger.error(f"LLM generation failed: {e}")
            raise
    
    def _parse_normal_abnormal_response(self, response: str) -> Dict[str, Any]:
        """Parse normal/abnormal classification response."""
        try:
            # Try to parse as JSON first
            if response.strip().startswith("{"):
                result = json.loads(response)
                return {
                    "is_normal": result.get("is_normal", True),
                    "confidence": float(result.get("confidence", 0.0)),
                    "explanation": result.get("explanation", ""),
                    "reasoning": result.get("reasoning", "")
                }
            
            # Fallback to text parsing
            response_lower = response.lower()
            
            # Determine classification
            is_normal = True
            confidence = 0.5
            
            if "abnormal" in response_lower or "pathological" in response_lower:
                is_normal = False
                confidence = 0.7
            elif "normal" in response_lower:
                is_normal = True
                confidence = 0.7
            
            # Extract confidence if mentioned
            import re
            confidence_match = re.search(r'confidence[:\s]*([0-9.]+)', response_lower)
            if confidence_match:
                confidence = float(confidence_match.group(1))
                if confidence > 1.0:
                    confidence = confidence / 100.0  # Convert percentage
            
            return {
                "is_normal": is_normal,
                "confidence": confidence,
                "explanation": response.strip(),
                "reasoning": ""
            }
            
        except Exception as e:
            logger.error(f"Failed to parse normal/abnormal response: {e}")
            return {
                "is_normal": True,
                "confidence": 0.0,
                "explanation": f"Parse error: {str(e)}",
                "reasoning": ""
            }
    
    def _normalize_condition_label(self, condition_name: str) -> str:
        """
        Normalize condition labels to match expected format.
        
        The test data uses specific label formats (lowercase, no spaces).
        This function maps LLM predictions to the expected labels.
        
        Args:
            condition_name: Raw condition name from LLM
            
        Returns:
            Normalized condition label matching test data format
        """
        if not condition_name:
            return "unknown"
        
        # Convert to lowercase for matching
        name_lower = condition_name.lower().strip()
        
        # Define mapping from LLM predictions to expected labels
        label_mapping = {
            # Cerebral Palsy variations
            "cerebral palsy": "cerebralpalsy",
            "cerebralpalsy": "cerebralpalsy",
            "cp": "cerebralpalsy",
            
            # Parkinson's variations
            "parkinson's disease": "parkinsons",
            "parkinsons disease": "parkinsons", 
            "parkinson disease": "parkinsons",
            "parkinsons": "parkinsons",
            "parkinson": "parkinsons",
            "pd": "parkinsons",
            
            # Myopathic variations
            "myopathic": "myopathic",
            "myopathy": "myopathic",
            "muscular dystrophy": "myopathic",
            "muscle weakness": "myopathic",
            
            # Stroke variations
            "stroke": "stroke",
            "hemiplegia": "stroke",
            "hemiplegic": "stroke",
            "cva": "stroke",
            "cerebrovascular accident": "stroke",
            
            # Normal variations
            "normal": "normal",
            "healthy": "normal",
            "typical": "normal",
            
            # Generic abnormal conditions - try to infer from context
            "gait abnormality": "abnormal",
            "abnormal": "abnormal",
            "pathological": "abnormal",
            "atypical": "abnormal",
        }
        
        # Direct mapping
        if name_lower in label_mapping:
            return label_mapping[name_lower]
        
        # Partial matching for complex names
        for pattern, label in label_mapping.items():
            if pattern in name_lower:
                return label
        
        # If no match found, return the original name normalized
        # (lowercase, no spaces, no special characters)
        normalized = name_lower.replace(" ", "").replace("'", "").replace("-", "")
        return normalized if normalized else "unknown"

    def _parse_condition_response(self, response: str) -> List[Dict[str, Any]]:
        """Parse condition identification response."""
        try:
            # Try to parse as JSON first
            if response.strip().startswith("[") or response.strip().startswith("{"):
                result = json.loads(response)
                
                # Handle single condition as dict
                if isinstance(result, dict):
                    result = [result]
                
                conditions = []
                for condition in result:
                    raw_name = condition.get("condition_name", "Unknown")
                    normalized_name = self._normalize_condition_label(raw_name)
                    conditions.append({
                        "condition_name": normalized_name,
                        "confidence": float(condition.get("confidence", 0.0)),
                        "severity": condition.get("severity", "Unknown"),
                        "supporting_evidence": condition.get("supporting_evidence", []),
                        "explanation": condition.get("explanation", "")
                    })
                
                return conditions
            
            # Fallback to text parsing
            conditions = []
            
            # Look for common condition patterns
            condition_patterns = [
                r'parkinson[\'s]*\s*disease',
                r'parkinsons?',
                r'cerebral\s*palsy',
                r'stroke',
                r'hemiplegia',
                r'hemiplegic',
                r'myopathic',
                r'myopathy',
                r'spinal\s*cord\s*injury',
                r'multiple\s*sclerosis',
                r'muscular\s*dystrophy',
                r'arthritis',
                r'hip\s*replacement',
                r'knee\s*replacement',
                r'normal',
                r'healthy',
                r'typical'
            ]
            
            response_lower = response.lower()
            for pattern in condition_patterns:
                import re
                if re.search(pattern, response_lower):
                    raw_condition_name = re.search(pattern, response_lower).group(0)
                    normalized_name = self._normalize_condition_label(raw_condition_name)
                    conditions.append({
                        "condition_name": normalized_name,
                        "confidence": 0.6,
                        "severity": "Unknown",
                        "supporting_evidence": [],
                        "explanation": response.strip()
                    })
            
            # If no specific conditions found, create generic abnormal condition
            if not conditions:
                conditions.append({
                    "condition_name": self._normalize_condition_label("abnormal"),
                    "confidence": 0.5,
                    "severity": "Unknown",
                    "supporting_evidence": [],
                    "explanation": response.strip()
                })
            
            return conditions
            
        except Exception as e:
            logger.error(f"Failed to parse condition response: {e}")
            return [{
                "condition_name": self._normalize_condition_label("unknown"),
                "confidence": 0.0,
                "severity": "Unknown",
                "supporting_evidence": [],
                "explanation": f"Parse error: {str(e)}"
            }]
    
    def _calculate_feature_importance(
        self, 
        gait_features: Dict[str, Any], 
        classification_result: Dict[str, Any]
    ) -> Dict[str, float]:
        """Calculate feature importance for the classification."""
        # This is a simplified implementation
        # In practice, you might use more sophisticated methods
        
        importance = {}
        
        # Basic heuristics for feature importance
        if not classification_result.get("is_normal", True):
            # Abnormal case - look for key indicators
            
            # Symmetry features
            for key, value in gait_features.items():
                if "asymmetry" in key.lower():
                    if isinstance(value, (int, float)) and value > 0.1:
                        importance[key] = min(value * 2, 1.0)
                
                # Velocity features
                elif "velocity" in key.lower():
                    if isinstance(value, (int, float)):
                        # Abnormal velocities (too fast or too slow)
                        if value < 0.5 or value > 2.0:
                            importance[key] = 0.7
                
                # Stability features
                elif "stability" in key.lower() or "sway" in key.lower():
                    if isinstance(value, (int, float)) and value > 0.3:
                        importance[key] = min(value * 1.5, 1.0)
        
        return importance
    
    def get_classification_confidence(self, result: Dict[str, Any]) -> float:
        """
        Get confidence score for classification result.
        
        Args:
            result: Classification result
            
        Returns:
            Confidence score between 0 and 1
        """
        return result.get("confidence", 0.0)
    
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
        
        Args:
            test_features: List of test feature vectors
            test_labels: Optional test labels (uses feature.condition_label if None)
            
        Returns:
            Dictionary with evaluation metrics including:
            - accuracy, precision, recall, f1_score
            - confusion_matrix
            - classification_report
            - per_sample_results (with LLM explanations)
        """
        if not self.is_trained:
            raise RuntimeError(
                "LLM classifier must be 'trained' (have few-shot examples) before evaluation"
            )
        
        logger.info(f"Evaluating LLM classifier on {len(test_features)} samples")
        logger.info("Note: This will make API calls and may take several minutes")
        
        # Get true labels
        if test_labels:
            y_true = test_labels
        else:
            y_true = [f.condition_label for f in test_features]
        
        # Validate labels
        if not all(y_true):
            raise ValueError("All test features must have condition labels")
        
        # Get predictions (this makes API calls)
        y_pred = []
        per_sample_results = []
        
        for i, feature in enumerate(test_features):
            try:
                result = self.classify_gait(feature)
                predicted = result.get("predicted_condition", "unknown")
                y_pred.append(predicted)
                per_sample_results.append({
                    "sample_id": feature.sample_id,
                    "true_label": y_true[i],
                    "predicted_label": predicted,
                    "confidence": result.get("confidence", 0.0),
                    "is_normal": result.get("is_normal", True),
                    "explanation": result.get("normal_abnormal_explanation", "")[:200]  # Truncate
                })
                
                if (i + 1) % 5 == 0:
                    logger.info(f"Processed {i + 1}/{len(test_features)} samples")
                    
            except Exception as e:
                logger.error(f"Failed to classify sample {i}: {e}")
                y_pred.append("unknown")
                per_sample_results.append({
                    "sample_id": feature.sample_id,
                    "true_label": y_true[i],
                    "predicted_label": "unknown",
                    "confidence": 0.0,
                    "error": str(e)
                })
        
        # Get unique classes from both true and predicted labels
        all_classes = sorted(set(y_true + y_pred))
        
        # Calculate metrics
        from sklearn.metrics import (
            accuracy_score,
            precision_score,
            recall_score,
            f1_score,
            confusion_matrix,
            classification_report,
        )
        
        accuracy = accuracy_score(y_true, y_pred)
        precision = precision_score(
            y_true, y_pred, average="macro", zero_division=0, labels=all_classes
        )
        recall = recall_score(
            y_true, y_pred, average="macro", zero_division=0, labels=all_classes
        )
        f1 = f1_score(
            y_true, y_pred, average="macro", zero_division=0, labels=all_classes
        )
        conf_matrix = confusion_matrix(y_true, y_pred, labels=all_classes)
        class_report = classification_report(
            y_true, y_pred, labels=all_classes, output_dict=True, zero_division=0
        )
        
        metrics = {
            "accuracy": float(accuracy),
            "precision": float(precision),
            "recall": float(recall),
            "f1_score": float(f1),
            "confusion_matrix": conf_matrix.tolist(),
            "classification_report": class_report,
            "n_test_samples": len(test_features),
            "classes": all_classes,
            "per_sample_results": per_sample_results,  # LLM-specific
            "evaluation_method": "llm_api_calls",  # LLM-specific
        }
        
        logger.info(f"LLM evaluation complete: accuracy={accuracy:.3f}, f1={f1:.3f}")
        
        return metrics
    
    def explain_classification(self, result: Dict[str, Any]) -> str:
        """
        Generate explanation for classification result.
        
        Args:
            result: Classification result
            
        Returns:
            Human-readable explanation of the classification
        """
        explanation_parts = []
        
        # Main classification
        predicted = result.get("predicted_condition", "unknown")
        confidence = result.get("confidence", 0.0)
        is_normal = result.get("is_normal", True)
        
        if is_normal:
            explanation_parts.append(
                f"Gait classified as NORMAL with {confidence:.2f} confidence."
            )
        else:
            explanation_parts.append(
                f"Gait classified as {predicted.upper()} with {confidence:.2f} confidence."
            )
        
        # Add normal/abnormal explanation
        na_explanation = result.get("normal_abnormal_explanation", "")
        if na_explanation:
            explanation_parts.append(f"\nReasoning: {na_explanation}")
        
        # Add reasoning if available
        reasoning = result.get("reasoning", "")
        if reasoning and reasoning != na_explanation:
            explanation_parts.append(f"\nDetailed reasoning: {reasoning}")
        
        # Add condition information
        conditions = result.get("identified_conditions", [])
        if conditions:
            explanation_parts.append("\nIdentified conditions:")
            for condition in conditions:
                condition_name = condition.get("condition_name", "Unknown")
                cond_confidence = condition.get("confidence", 0.0)
                severity = condition.get("severity", "Unknown")
                
                explanation_parts.append(
                    f"- {condition_name} (confidence: {cond_confidence:.2f}, severity: {severity})"
                )
                
                condition_explanation = condition.get("explanation", "")
                if condition_explanation:
                    explanation_parts.append(f"  {condition_explanation}")
        
        # Add feature importance
        feature_importance = result.get("feature_importance", {})
        if feature_importance:
            explanation_parts.append("\nKey contributing factors:")
            sorted_features = sorted(
                feature_importance.items(), 
                key=lambda x: x[1], 
                reverse=True
            )
            for feature, imp in sorted_features[:5]:  # Top 5
                explanation_parts.append(f"- {feature}: {imp:.2f}")
        
        return "\n".join(explanation_parts)
    
    def save(self, filepath: Union[str, Path]) -> None:
        """
        Save LLM classifier configuration and few-shot examples.
        
        Note: API keys are not saved for security. Set them via environment
        variables or config when loading.
        
        Args:
            filepath: Path to save the classifier
        """
        if not self.is_trained:
            logger.warning("Saving untrained LLM classifier (no few-shot examples)")
        
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        
        # Create config without API key for security
        safe_config = LLMClassifierConfig(
            model_name=self.config.model_name,
            provider=self.config.provider,
            api_key=None,  # Don't save API key
            temperature=self.config.temperature,
            max_tokens=self.config.max_tokens,
            enable_chain_of_thought=self.config.enable_chain_of_thought,
            two_stage_classification=self.config.two_stage_classification,
            prompt_config_path=self.config.prompt_config_path,
            confidence_threshold=self.config.confidence_threshold,
        )
        
        model_data = {
            "config": safe_config,
            "few_shot_examples": self.few_shot_examples,
            "is_trained": self.is_trained,
            "model_params": self._get_model_params(),
            "classifier_type": "LLM",
            "version": "2.0",  # Updated version with base class
            "feature_names": self.feature_names,
        }
        
        import pickle
        with open(filepath, "wb") as f:
            pickle.dump(model_data, f)
        
        logger.info(f"LLM classifier saved to {filepath}")
        logger.info(f"Saved {len(self.few_shot_examples)} few-shot examples")
    
    @classmethod
    def load(cls, filepath: Union[str, Path]) -> "LLMClassifier":
        """
        Load LLM classifier from file.
        
        Note: API key must be set via environment variable or passed in config
        after loading.
        
        Args:
            filepath: Path to load the classifier from
            
        Returns:
            Loaded LLMClassifier instance
        """
        filepath = Path(filepath)
        
        import pickle
        with open(filepath, "rb") as f:
            model_data = pickle.load(f)
        
        # Create classifier with loaded config
        config = model_data["config"]
        classifier = cls(config=config)
        
        # Restore few-shot examples and training state
        classifier.few_shot_examples = model_data.get("few_shot_examples", [])
        classifier.is_trained = model_data.get("is_trained", False)
        classifier.feature_names = model_data.get("feature_names", GaitFeatureVector.get_feature_names())
        
        logger.info(f"LLM classifier loaded from {filepath}")
        logger.info(f"Loaded {len(classifier.few_shot_examples)} few-shot examples")
        
        return classifier
    
    def get_supported_models(self) -> Dict[str, List[str]]:
        """Get list of supported models by provider."""
        return {
            "openai": [
                # GPT-4 series (use max_tokens)
                "gpt-4o",            # Latest GPT-4 Omni model
                "gpt-4o-mini",       # Cost-effective GPT-4 Omni
                "gpt-4-turbo",       # GPT-4 Turbo
                "gpt-4",             # GPT-4
                "gpt-3.5-turbo",     # GPT-3.5 Turbo
                
                # O-series models (use max_completion_tokens)
                "o1-preview",        # O1 Preview
                "o1-mini",           # O1 Mini
                "o3-mini",           # O3 Mini
                
                # GPT-5 series (use max_output_tokens)
                "gpt-5.2",           # GPT-5.2 (if available)
                "gpt-5.2-chat-latest",  # GPT-5.2 Chat Latest
            ],
            "gemini": [
                "gemini-2.0-flash-exp",      # Latest Gemini 2.0 Flash experimental
                "gemini-1.5-pro",            # Gemini 1.5 Pro
                "gemini-1.5-flash",          # Gemini 1.5 Flash
                "gemini-1.0-pro",            # Gemini 1.0 Pro
            ]
        }
    
    def is_model_available(self) -> bool:
        """Check if the configured model is available."""
        try:
            # Initialize model if not already done
            if self.model is None:
                self.model = self._create_model()
            
            # Test with a simple prompt
            test_prompt = "Respond with 'OK' if you can process this message."
            response = self._generate_response(test_prompt)
            return "ok" in response.lower()
        except Exception as e:
            logger.error(f"Model availability check failed: {e}")
            return False