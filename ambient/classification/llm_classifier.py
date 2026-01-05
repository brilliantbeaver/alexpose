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
from loguru import logger

# Load environment variables from .env file
try:
    from dotenv import load_dotenv
    load_dotenv()
    DOTENV_AVAILABLE = True
except ImportError:
    DOTENV_AVAILABLE = False

from ambient.core.interfaces import IClassifier
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


class LLMClassifier(IClassifier):
    """
    LLM-based classifier for gait analysis.
    
    This classifier uses modern language models to perform two-stage classification:
    1. Normal vs abnormal gait classification
    2. Specific condition identification for abnormal cases
    
    Supports multiple LLM providers with configurable prompts and agentic reasoning.
    """
    
    def __init__(
        self,
        model_name: str = "gpt-5.2",  # Updated to latest model
        provider: str = "openai",
        api_key: Optional[str] = None,
        prompt_manager: Optional[PromptManager] = None,
        temperature: float = 0.1,
        max_tokens: Optional[int] = None,
        confidence_threshold: float = 0.7,
        enable_chain_of_thought: bool = True
    ):
        """
        Initialize LLM classifier.
        
        Args:
            model_name: Name of the LLM model to use
            provider: LLM provider ("openai" or "gemini")
            api_key: API key for the LLM provider
            prompt_manager: Prompt manager instance
            temperature: Temperature for generation (0.0-1.0)
            max_tokens: Maximum tokens to generate
            confidence_threshold: Minimum confidence for classification
            enable_chain_of_thought: Enable chain-of-thought reasoning
        """
        self.model_name = model_name
        self.provider = provider.lower()
        self.api_key = api_key
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.confidence_threshold = confidence_threshold
        self.enable_chain_of_thought = enable_chain_of_thought
        
        # Initialize prompt manager
        self.prompt_manager = prompt_manager or PromptManager()
        
        # Initialize LLM client
        self._initialize_client()
        
        logger.info(f"LLM classifier initialized with {provider} {model_name}")
    
    def _initialize_client(self):
        """Initialize the appropriate LLM client."""
        if self.provider == "openai":
            if not OPENAI_AVAILABLE:
                raise ImportError(
                    "OpenAI library not available. Install with: pip install openai"
                )
            
            # Get API key from environment or parameter
            api_key = self.api_key
            if not api_key:
                import os
                api_key = os.getenv("OPENAI_API_KEY")
            
            if not api_key:
                raise ValueError(
                    "OpenAI API key required. Set OPENAI_API_KEY environment variable "
                    "or pass api_key parameter."
                )
            
            self.client = OpenAI(api_key=api_key)
            
        elif self.provider == "gemini":
            if not GEMINI_AVAILABLE:
                raise ImportError(
                    "Google Generative AI library not available. "
                    "Install with: pip install google-generativeai"
                )
            
            # Get API key from environment or parameter
            api_key = self.api_key
            if not api_key:
                import os
                api_key = os.getenv("GOOGLE_API_KEY")
            
            if not api_key:
                raise ValueError(
                    "Google API key required. Set GOOGLE_API_KEY environment variable "
                    "or pass api_key parameter."
                )
            
            genai.configure(api_key=api_key)
            self.client = genai.GenerativeModel(
                model_name=self.model_name,
                generation_config=genai.GenerationConfig(
                    temperature=self.temperature,
                    max_output_tokens=self.max_tokens
                )
            )
            
        else:
            raise ValueError(f"Unsupported provider: {self.provider}")
    
    def classify_gait(
        self, 
        gait_features: Dict[str, Any], 
        context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Classify gait as normal/abnormal and identify conditions.
        
        Args:
            gait_features: Extracted gait features
            context: Optional context information
            
        Returns:
            Dictionary containing classification results with confidence scores
        """
        context = context or {}
        
        try:
            # Stage 1: Normal vs Abnormal Classification
            logger.info("Performing normal/abnormal classification")
            normal_abnormal_result = self._classify_normal_abnormal(gait_features, context)
            
            # Stage 2: Condition Identification (if abnormal)
            condition_results = []
            if not normal_abnormal_result.get("is_normal", True):
                logger.info("Performing condition identification")
                condition_results = self._identify_conditions(gait_features, context)
            
            # Combine results
            classification_result = {
                "is_normal": normal_abnormal_result.get("is_normal", True),
                "normal_abnormal_confidence": normal_abnormal_result.get("confidence", 0.0),
                "normal_abnormal_explanation": normal_abnormal_result.get("explanation", ""),
                "identified_conditions": condition_results,
                "overall_confidence": self._calculate_overall_confidence(
                    normal_abnormal_result, condition_results
                ),
                "classification_timestamp": time.time(),
                "model_info": {
                    "provider": self.provider,
                    "model_name": self.model_name,
                    "temperature": self.temperature
                }
            }
            
            # Add feature importance if available
            feature_importance = self._calculate_feature_importance(gait_features, classification_result)
            if feature_importance:
                classification_result["feature_importance"] = feature_importance
            
            return classification_result
            
        except Exception as e:
            logger.error(f"LLM classification failed: {e}")
            return {
                "is_normal": True,  # Default to normal on error
                "normal_abnormal_confidence": 0.0,
                "normal_abnormal_explanation": f"Classification failed: {str(e)}",
                "identified_conditions": [],
                "overall_confidence": 0.0,
                "error": str(e),
                "classification_timestamp": time.time()
            }
    
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
            enable_chain_of_thought=self.enable_chain_of_thought
        )
        
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
            enable_chain_of_thought=self.enable_chain_of_thought
        )
        
        # Generate response
        response = self._generate_response(prompt)
        
        # Parse response
        return self._parse_condition_response(response)
    
    def _generate_response(self, prompt: str) -> str:
        """Generate response from LLM."""
        try:
            if self.provider == "openai":
                response = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=[
                        {"role": "system", "content": "You are an expert in gait analysis and medical diagnosis."},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=self.temperature,
                    max_tokens=self.max_tokens
                )
                return response.choices[0].message.content
                
            elif self.provider == "gemini":
                response = self.client.generate_content(prompt)
                return response.text
                
            else:
                raise ValueError(f"Unsupported provider: {self.provider}")
                
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
                    conditions.append({
                        "condition_name": condition.get("condition_name", "Unknown"),
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
                r'cerebral\s*palsy',
                r'stroke',
                r'hemiplegia',
                r'spinal\s*cord\s*injury',
                r'multiple\s*sclerosis',
                r'muscular\s*dystrophy',
                r'arthritis',
                r'hip\s*replacement',
                r'knee\s*replacement'
            ]
            
            response_lower = response.lower()
            for pattern in condition_patterns:
                import re
                if re.search(pattern, response_lower):
                    condition_name = re.search(pattern, response_lower).group(0)
                    conditions.append({
                        "condition_name": condition_name.title(),
                        "confidence": 0.6,
                        "severity": "Unknown",
                        "supporting_evidence": [],
                        "explanation": response.strip()
                    })
            
            # If no specific conditions found, create generic abnormal condition
            if not conditions:
                conditions.append({
                    "condition_name": "Gait Abnormality",
                    "confidence": 0.5,
                    "severity": "Unknown",
                    "supporting_evidence": [],
                    "explanation": response.strip()
                })
            
            return conditions
            
        except Exception as e:
            logger.error(f"Failed to parse condition response: {e}")
            return [{
                "condition_name": "Parse Error",
                "confidence": 0.0,
                "severity": "Unknown",
                "supporting_evidence": [],
                "explanation": f"Parse error: {str(e)}"
            }]
    
    def _calculate_overall_confidence(
        self, 
        normal_abnormal_result: Dict[str, Any], 
        condition_results: List[Dict[str, Any]]
    ) -> float:
        """Calculate overall confidence score."""
        na_confidence = normal_abnormal_result.get("confidence", 0.0)
        
        if not condition_results:
            return na_confidence
        
        # Average condition confidences
        condition_confidences = [c.get("confidence", 0.0) for c in condition_results]
        avg_condition_confidence = sum(condition_confidences) / len(condition_confidences)
        
        # Weighted average (normal/abnormal is more important)
        return (na_confidence * 0.7) + (avg_condition_confidence * 0.3)
    
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
                if "symmetry" in key.lower():
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
        return result.get("overall_confidence", 0.0)
    
    def explain_classification(self, result: Dict[str, Any]) -> str:
        """
        Generate explanation for classification result.
        
        Args:
            result: Classification result
            
        Returns:
            Human-readable explanation of the classification
        """
        explanation_parts = []
        
        # Normal/abnormal explanation
        if result.get("is_normal", True):
            explanation_parts.append(
                f"Gait classified as NORMAL with {result.get('normal_abnormal_confidence', 0.0):.2f} confidence."
            )
        else:
            explanation_parts.append(
                f"Gait classified as ABNORMAL with {result.get('normal_abnormal_confidence', 0.0):.2f} confidence."
            )
        
        # Add normal/abnormal explanation
        na_explanation = result.get("normal_abnormal_explanation", "")
        if na_explanation:
            explanation_parts.append(f"Reasoning: {na_explanation}")
        
        # Add condition information
        conditions = result.get("identified_conditions", [])
        if conditions:
            explanation_parts.append("\nIdentified conditions:")
            for condition in conditions:
                condition_name = condition.get("condition_name", "Unknown")
                confidence = condition.get("confidence", 0.0)
                severity = condition.get("severity", "Unknown")
                
                explanation_parts.append(
                    f"- {condition_name} (confidence: {confidence:.2f}, severity: {severity})"
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
            for feature, importance in sorted_features[:5]:  # Top 5
                explanation_parts.append(f"- {feature}: {importance:.2f}")
        
        return "\n".join(explanation_parts)
    
    def get_supported_models(self) -> Dict[str, List[str]]:
        """Get list of supported models by provider."""
        return {
            "openai": [
                "gpt-5.2",           # Latest flagship model (2025)
                "gpt-5.1",           # Latest performance model (2025)
                "gpt-5-mini",        # Latest cost-effective model (2025)
                "gpt-5-nano",        # Latest ultra-efficient model (2025)
                "gpt-4.1",           # Enhanced GPT-4 model (2025)
                "gpt-4.1-mini",      # Enhanced GPT-4 mini model (2025)
            ],
            "gemini": [
                "gemini-3-pro-preview",      # Latest Gemini 3 Pro preview (2025)
                "gemini-3-pro-image-preview", # Latest Gemini 3 Pro with image support (2025)
                "gemini-3-flash-preview",    # Latest Gemini 3 Flash preview (2025)
                "gemini-2.5-flash",          # Gemini 2.5 Flash (2024)
                "gemini-2.5-flash-image",    # Gemini 2.5 Flash with image support (2024)
                "gemini-2.5-pro"             # Gemini 2.5 Pro (2024)
            ]
        }
    
    def is_model_available(self) -> bool:
        """Check if the configured model is available."""
        try:
            # Test with a simple prompt
            test_prompt = "Respond with 'OK' if you can process this message."
            response = self._generate_response(test_prompt)
            return "ok" in response.lower()
        except Exception as e:
            logger.error(f"Model availability check failed: {e}")
            return False