"""
Core package for Ambient gait analysis system.

This package contains the core components for the Ambient gait analysis system,
including interfaces, configuration management, analyzers, and output handling.

@Theodore Mui
Monday, July 28, 2025 12:30:00 AM
"""

from .cli import CLIHandler
from .configuration import EnvConfigurationManager, YAMLConfigurationManager
from .interfaces import IAnalyzer, IConfigurationManager, IFileManager, IOutputManager
from .output import OutputManager

# Import data models
from .data_models import (
    Keypoint, GaitFeatures, BalanceMetrics, TemporalMetrics, GaitMetrics,
    ConditionPrediction, ClassificationResult, VideoMetadata, ProcessingMetadata,
    GaitCycle, AnalysisResult, TrainingDataSample, DatasetInfo
)

__all__ = [
    # Interfaces
    "IConfigurationManager",
    "IAnalyzer",
    "IOutputManager",
    "IFileManager",
    # Concrete implementations
    "EnvConfigurationManager",
    "YAMLConfigurationManager",
    "OutputManager",
    "CLIHandler",
    # Data models
    "Keypoint",
    "GaitFeatures", 
    "BalanceMetrics",
    "TemporalMetrics",
    "GaitMetrics",
    "ConditionPrediction",
    "ClassificationResult",
    "VideoMetadata",
    "ProcessingMetadata",
    "GaitCycle",
    "AnalysisResult",
    "TrainingDataSample",
    "DatasetInfo",
]
