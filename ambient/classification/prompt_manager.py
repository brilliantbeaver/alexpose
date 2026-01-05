"""
Prompt management system for LLM-based gait classification.

This module manages prompts for different classification tasks,
loading them from YAML configuration files and providing
template formatting capabilities.

Author: AlexPose Team
"""

import os
import yaml
from typing import Any, Dict, Optional
from pathlib import Path
from loguru import logger

from ambient.core.interfaces import IConfigurationManager


class PromptManager:
    """
    Manager for classification prompts with YAML configuration support.
    
    Loads and manages prompts for different classification tasks,
    supporting template formatting and dynamic prompt updates.
    """
    
    def __init__(self, config_manager: Optional[IConfigurationManager] = None):
        """
        Initialize prompt manager.
        
        Args:
            config_manager: Configuration manager instance
        """
        self.config_manager = config_manager
        self.prompts = {}
        
        # Load prompts from configuration
        self._load_prompts()
        
        logger.info("Prompt manager initialized")
    
    def _load_prompts(self):
        """Load prompts from YAML configuration files."""
        try:
            # Get config directory
            if self.config_manager:
                config_dir = Path(self.config_manager.get_config_directory())
            else:
                # Default to config directory relative to project root
                config_dir = Path(__file__).parent.parent.parent / "config"
            
            prompt_file = config_dir / "classification_prompts.yaml"
            
            if prompt_file.exists():
                with open(prompt_file, 'r', encoding='utf-8') as f:
                    self.prompts = yaml.safe_load(f)
                logger.info(f"Loaded prompts from {prompt_file}")
            else:
                logger.warning(f"Prompt file not found: {prompt_file}")
                self._create_default_prompts()
                
        except Exception as e:
            logger.error(f"Failed to load prompts: {e}")
            self._create_default_prompts()
    
    def _create_default_prompts(self):
        """Create default prompts if configuration file is not available."""
        self.prompts = {
            "normal_abnormal_classification": {
                "system": """You are an expert clinical gait analyst with extensive experience in biomechanics and movement disorders. Your task is to analyze gait data and classify it as either NORMAL or ABNORMAL.

Use chain-of-thought reasoning to analyze the provided gait metrics systematically. Consider:

1. TEMPORAL PARAMETERS:
   - Cadence (normal: 100-130 steps/min)
   - Step regularity (CV < 0.1 is good)
   - Cycle duration consistency

2. KINEMATIC FEATURES:
   - Velocity patterns and consistency
   - Acceleration profiles
   - Movement smoothness (jerk analysis)

3. SYMMETRY ANALYSIS:
   - Left-right symmetry indices
   - Joint-specific asymmetries
   - Overall symmetry classification

4. STABILITY MEASURES:
   - Center of mass movement
   - Postural control indicators

Provide your response in JSON format with:
{
  "classification": "NORMAL" or "ABNORMAL",
  "confidence": 0.0-1.0,
  "explanation": "detailed explanation of your reasoning",
  "reasoning_chain": ["step 1", "step 2", "step 3", "conclusion"]
}""",
                
                "user": """Please analyze the following gait data and classify it as NORMAL or ABNORMAL:

{analysis_data}

Provide a thorough analysis using chain-of-thought reasoning, considering all available metrics and their clinical significance."""
            },
            
            "condition_classification": {
                "system": """You are an expert clinical gait analyst specializing in movement disorder diagnosis. Your task is to identify specific gait conditions based on abnormal gait patterns.

Consider these common gait abnormalities and their characteristic patterns:

1. HEMIPLEGIC GAIT:
   - Significant left-right asymmetry
   - Reduced swing phase on affected side
   - Circumduction pattern

2. PARKINSONIAN GAIT:
   - Reduced step length and cadence
   - Increased step-to-step variability
   - Shuffling pattern with reduced arm swing

3. ATAXIC GAIT:
   - High step-to-step variability
   - Wide base of support
   - Irregular timing patterns

4. ANTALGIC GAIT:
   - Asymmetric stance phases
   - Reduced weight-bearing time on affected side
   - Compensatory movements

5. SPASTIC GAIT:
   - Stiff, effortful movement patterns
   - Reduced joint range of motion
   - Increased muscle co-activation

6. TRENDELENBURG GAIT:
   - Hip drop on contralateral side
   - Lateral trunk lean
   - Pelvic instability

Use chain-of-thought reasoning to match the observed patterns with known pathological conditions.

Provide your response in JSON format with:
{
  "classification": "specific condition name",
  "confidence": 0.0-1.0,
  "explanation": "detailed explanation linking observed patterns to the condition",
  "reasoning_chain": ["observation 1", "observation 2", "pattern matching", "conclusion"],
  "differential_diagnosis": ["alternative condition 1", "alternative condition 2"],
  "clinical_recommendations": ["recommendation 1", "recommendation 2"]
}""",
                
                "user": """The gait has been classified as ABNORMAL. Please identify the specific condition based on the following analysis data:

{analysis_data}

Use systematic pattern recognition to match the observed abnormalities with known pathological gait conditions."""
            },
            
            "multimodal_classification": {
                "system": """You are an expert clinical gait analyst with access to both quantitative gait metrics and visual gait analysis. Your task is to provide comprehensive gait classification using both data sources.

When analyzing images/videos, look for:

1. VISUAL GAIT PATTERNS:
   - Step length and width
   - Arm swing patterns
   - Trunk posture and stability
   - Foot clearance and placement
   - Overall coordination

2. TEMPORAL VISUAL CUES:
   - Rhythm and timing
   - Phase transitions
   - Compensatory movements
   - Balance reactions

3. SPATIAL RELATIONSHIPS:
   - Joint alignments
   - Segment orientations
   - Center of mass trajectory

Integrate visual observations with quantitative metrics for comprehensive analysis.

Provide your response in JSON format with:
{
  "classification": "NORMAL/ABNORMAL or specific condition",
  "confidence": 0.0-1.0,
  "explanation": "integrated analysis of visual and quantitative data",
  "reasoning_chain": ["visual observation 1", "metric analysis 1", "integration", "conclusion"],
  "visual_findings": ["key visual observation 1", "key visual observation 2"],
  "metric_findings": ["key metric finding 1", "key metric finding 2"]
}""",
                
                "user": """Please analyze the following gait data and any provided images/videos:

{analysis_data}

Provide comprehensive classification integrating both visual and quantitative analysis."""
            }
        }
        
        logger.info("Created default prompts")
    
    def get_prompt(self, prompt_name: str) -> Dict[str, str]:
        """
        Get a specific prompt by name.
        
        Args:
            prompt_name: Name of the prompt to retrieve
            
        Returns:
            Dictionary containing system and user prompts
            
        Raises:
            KeyError: If prompt name is not found
        """
        if prompt_name not in self.prompts:
            raise KeyError(f"Prompt '{prompt_name}' not found")
        
        return self.prompts[prompt_name]
    
    def get_available_prompts(self) -> list[str]:
        """Get list of available prompt names."""
        return list(self.prompts.keys())
    
    def update_prompt(self, prompt_name: str, system_prompt: str, user_prompt: str):
        """
        Update a specific prompt.
        
        Args:
            prompt_name: Name of the prompt to update
            system_prompt: New system prompt
            user_prompt: New user prompt template
        """
        self.prompts[prompt_name] = {
            "system": system_prompt,
            "user": user_prompt
        }
        
        logger.info(f"Updated prompt: {prompt_name}")
    
    def add_prompt(self, prompt_name: str, system_prompt: str, user_prompt: str):
        """
        Add a new prompt.
        
        Args:
            prompt_name: Name of the new prompt
            system_prompt: System prompt
            user_prompt: User prompt template
        """
        if prompt_name in self.prompts:
            logger.warning(f"Prompt '{prompt_name}' already exists, will be overwritten")
        
        self.prompts[prompt_name] = {
            "system": system_prompt,
            "user": user_prompt
        }
        
        logger.info(f"Added new prompt: {prompt_name}")
    
    def save_prompts(self, file_path: Optional[Path] = None):
        """
        Save current prompts to YAML file.
        
        Args:
            file_path: Optional custom file path, defaults to config directory
        """
        try:
            if file_path is None:
                if self.config_manager:
                    config_dir = Path(self.config_manager.get_config_directory())
                else:
                    config_dir = Path(__file__).parent.parent.parent / "config"
                
                file_path = config_dir / "classification_prompts.yaml"
            
            # Ensure directory exists
            file_path.parent.mkdir(parents=True, exist_ok=True)
            
            with open(file_path, 'w', encoding='utf-8') as f:
                yaml.dump(self.prompts, f, default_flow_style=False, allow_unicode=True)
            
            logger.info(f"Saved prompts to {file_path}")
            
        except Exception as e:
            logger.error(f"Failed to save prompts: {e}")
            raise
    
    def reload_prompts(self):
        """Reload prompts from configuration file."""
        logger.info("Reloading prompts from configuration")
        self._load_prompts()
    
    def format_prompt(
        self, 
        prompt_name: str, 
        **kwargs
    ) -> Dict[str, str]:
        """
        Format a prompt with provided variables.
        
        Args:
            prompt_name: Name of the prompt to format
            **kwargs: Variables to substitute in the prompt template
            
        Returns:
            Dictionary with formatted system and user prompts
        """
        prompt = self.get_prompt(prompt_name)
        
        try:
            # Only format the user prompt, as system prompts contain JSON examples
            formatted_prompt = {
                "system": prompt["system"],  # Keep system prompt as-is
                "user": prompt["user"].format(**kwargs)  # Only format user prompt
            }
            return formatted_prompt
            
        except KeyError as e:
            logger.error(f"Missing template variable in prompt '{prompt_name}': {e}")
            raise
        except Exception as e:
            logger.error(f"Failed to format prompt '{prompt_name}': {e}")
            raise
    
    def get_normal_abnormal_prompt(
        self, 
        gait_features: Dict[str, Any], 
        context: Optional[Dict[str, Any]] = None,
        enable_chain_of_thought: bool = True
    ) -> str:
        """
        Get formatted prompt for normal/abnormal classification.
        
        Args:
            gait_features: Extracted gait features
            context: Optional context information
            enable_chain_of_thought: Enable chain-of-thought reasoning
            
        Returns:
            Formatted prompt string
        """
        context = context or {}
        
        # Format analysis data
        analysis_data = self._format_analysis_data(gait_features, context)
        
        # Get and format prompt
        prompt_data = self.format_prompt(
            "normal_abnormal_classification",
            analysis_data=analysis_data
        )
        
        # Combine system and user prompts
        full_prompt = f"{prompt_data['system']}\n\n{prompt_data['user']}"
        
        if enable_chain_of_thought:
            full_prompt += "\n\nUse step-by-step reasoning to analyze the data systematically."
        
        return full_prompt
    
    def get_condition_identification_prompt(
        self, 
        gait_features: Dict[str, Any], 
        context: Optional[Dict[str, Any]] = None,
        enable_chain_of_thought: bool = True
    ) -> str:
        """
        Get formatted prompt for condition identification.
        
        Args:
            gait_features: Extracted gait features
            context: Optional context information
            enable_chain_of_thought: Enable chain-of-thought reasoning
            
        Returns:
            Formatted prompt string
        """
        context = context or {}
        
        # Format analysis data
        analysis_data = self._format_analysis_data(gait_features, context)
        
        # Get and format prompt
        prompt_data = self.format_prompt(
            "condition_classification",
            analysis_data=analysis_data
        )
        
        # Combine system and user prompts
        full_prompt = f"{prompt_data['system']}\n\n{prompt_data['user']}"
        
        if enable_chain_of_thought:
            full_prompt += "\n\nUse systematic pattern recognition and chain-of-thought reasoning to identify the specific condition."
        
        return full_prompt
    
    def get_multimodal_prompt(
        self, 
        gait_features: Dict[str, Any], 
        context: Optional[Dict[str, Any]] = None,
        enable_chain_of_thought: bool = True
    ) -> str:
        """
        Get formatted prompt for multimodal classification.
        
        Args:
            gait_features: Extracted gait features
            context: Optional context information
            enable_chain_of_thought: Enable chain-of-thought reasoning
            
        Returns:
            Formatted prompt string
        """
        context = context or {}
        
        # Format analysis data
        analysis_data = self._format_analysis_data(gait_features, context)
        
        # Get and format prompt
        prompt_data = self.format_prompt(
            "multimodal_classification",
            analysis_data=analysis_data
        )
        
        # Combine system and user prompts
        full_prompt = f"{prompt_data['system']}\n\n{prompt_data['user']}"
        
        if enable_chain_of_thought:
            full_prompt += "\n\nIntegrate visual and quantitative analysis using systematic reasoning."
        
        return full_prompt
    
    def _format_analysis_data(
        self, 
        gait_features: Dict[str, Any], 
        context: Dict[str, Any]
    ) -> str:
        """
        Format gait features and context into analysis data string.
        
        Args:
            gait_features: Extracted gait features
            context: Context information
            
        Returns:
            Formatted analysis data string
        """
        import json
        
        analysis_sections = []
        
        # Add gait features
        if gait_features:
            analysis_sections.append("GAIT FEATURES:")
            
            # Temporal features
            if "temporal_features" in gait_features:
                analysis_sections.append("Temporal Features:")
                for key, value in gait_features["temporal_features"].items():
                    analysis_sections.append(f"  - {key}: {value}")
            
            # Spatial features
            if "spatial_features" in gait_features:
                analysis_sections.append("Spatial Features:")
                for key, value in gait_features["spatial_features"].items():
                    analysis_sections.append(f"  - {key}: {value}")
            
            # Kinematic features
            if "kinematic_features" in gait_features:
                analysis_sections.append("Kinematic Features:")
                for key, value in gait_features["kinematic_features"].items():
                    if isinstance(value, list):
                        analysis_sections.append(f"  - {key}: {len(value)} measurements")
                    else:
                        analysis_sections.append(f"  - {key}: {value}")
            
            # Symmetry features
            if "symmetry_features" in gait_features:
                analysis_sections.append("Symmetry Features:")
                for key, value in gait_features["symmetry_features"].items():
                    analysis_sections.append(f"  - {key}: {value}")
            
            # Frequency features
            if "frequency_features" in gait_features:
                analysis_sections.append("Frequency Features:")
                for key, value in gait_features["frequency_features"].items():
                    analysis_sections.append(f"  - {key}: {value}")
        
        # Add context information
        if context:
            analysis_sections.append("\nCONTEXT INFORMATION:")
            
            # Patient information (if available)
            if "patient_info" in context:
                patient_info = context["patient_info"]
                analysis_sections.append("Patient Information:")
                for key, value in patient_info.items():
                    analysis_sections.append(f"  - {key}: {value}")
            
            # Video metadata
            if "video_metadata" in context:
                video_meta = context["video_metadata"]
                analysis_sections.append("Video Metadata:")
                for key, value in video_meta.items():
                    analysis_sections.append(f"  - {key}: {value}")
            
            # Processing metadata
            if "processing_metadata" in context:
                proc_meta = context["processing_metadata"]
                analysis_sections.append("Processing Metadata:")
                for key, value in proc_meta.items():
                    analysis_sections.append(f"  - {key}: {value}")
        
        return "\n".join(analysis_sections)
    
    def validate_prompts(self) -> Dict[str, list[str]]:
        """
        Validate all prompts for required fields and template variables.
        
        Returns:
            Dictionary mapping prompt names to lists of validation errors
        """
        validation_results = {}
        
        for prompt_name, prompt_data in self.prompts.items():
            errors = []
            
            # Check required fields
            if "system" not in prompt_data:
                errors.append("Missing 'system' field")
            if "user" not in prompt_data:
                errors.append("Missing 'user' field")
            
            # Check for empty prompts
            if prompt_data.get("system", "").strip() == "":
                errors.append("Empty system prompt")
            if prompt_data.get("user", "").strip() == "":
                errors.append("Empty user prompt")
            
            # Check for common template variables
            user_prompt = prompt_data.get("user", "")
            if "{analysis_data}" not in user_prompt:
                errors.append("User prompt missing '{analysis_data}' template variable")
            
            validation_results[prompt_name] = errors
        
        return validation_results