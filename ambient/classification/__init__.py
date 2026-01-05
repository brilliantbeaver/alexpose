"""
Classification module for gait analysis.

This module provides LLM-based classification capabilities for gait analysis,
including normal/abnormal classification and condition identification.

Author: AlexPose Team
"""

from .llm_classifier import LLMClassifier
from .prompt_manager import PromptManager

__all__ = ["LLMClassifier", "PromptManager"]