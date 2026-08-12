"""
Classification module for gait analysis.

This module provides classification capabilities for gait analysis,
including LLM-based and traditional ML approaches for normal/abnormal
classification and condition identification.

Author: AlexPose Team
"""

from .llm_classifier import LLMClassifier, LLMClassifierConfig
from .prompt_manager import PromptManager
from .features import GaitFeatureVector
from .knn_classifier import (
    KNNGaitClassifier,
    KNNClassifierConfig,
)

__all__ = [
    "LLMClassifier",
    "LLMClassifierConfig",
    "PromptManager",
    "GaitFeatureVector",
    "KNNGaitClassifier",
    "KNNClassifierConfig",
]