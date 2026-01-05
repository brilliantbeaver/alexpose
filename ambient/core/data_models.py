"""
Additional data models for AlexPose gait analysis system.

This module provides data structures that complement the Frame and FrameSequence
models for comprehensive gait analysis workflows.

Key Data Structures:
- GaitMetrics: Comprehensive gait analysis results
- ClassificationResult: LLM-based classification outcomes
- VideoMetadata: Video file metadata and properties
- ProcessingMetadata: Processing operation metadata

Author: AlexPose Team
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
import numpy as np


@dataclass
class Keypoint:
    """Individual keypoint with position and confidence."""
    x: float
    y: float
    confidence: float
    visibility: float = 1.0
    name: str = ""


@dataclass
class GaitFeatures:
    """Extracted features from gait analysis."""
    temporal_features: Dict[str, float] = field(default_factory=dict)
    spatial_features: Dict[str, float] = field(default_factory=dict)
    kinematic_features: Dict[str, List[float]] = field(default_factory=dict)
    symmetry_features: Dict[str, float] = field(default_factory=dict)
    frequency_features: Dict[str, float] = field(default_factory=dict)


@dataclass
class BalanceMetrics:
    """Balance-related gait metrics."""
    center_of_mass_displacement: float = 0.0
    postural_sway: float = 0.0
    stability_index: float = 0.0


@dataclass
class TemporalMetrics:
    """Temporal gait metrics."""
    stance_phase_duration: float = 0.0
    swing_phase_duration: float = 0.0
    double_support_time: float = 0.0
    gait_cycle_time: float = 0.0


@dataclass
class GaitMetrics:
    """Comprehensive gait analysis metrics."""
    stride_length: float = 0.0
    stride_time: float = 0.0
    cadence: float = 0.0
    step_width: float = 0.0
    joint_angles: Dict[str, List[float]] = field(default_factory=dict)
    symmetry_index: float = 0.0
    balance_metrics: BalanceMetrics = field(default_factory=BalanceMetrics)
    temporal_metrics: TemporalMetrics = field(default_factory=TemporalMetrics)


@dataclass
class ConditionPrediction:
    """Prediction for a specific health condition."""
    condition_name: str
    confidence: float
    severity: str = "unknown"
    supporting_evidence: List[str] = field(default_factory=list)


@dataclass
class ClassificationResult:
    """Result of gait classification analysis."""
    is_normal: bool
    confidence: float
    identified_conditions: List[ConditionPrediction] = field(default_factory=list)
    explanation: str = ""
    feature_importance: Dict[str, float] = field(default_factory=dict)


@dataclass
class VideoMetadata:
    """Metadata for video files."""
    file_path: Path
    duration: float = 0.0
    frame_rate: float = 30.0
    width: int = 0
    height: int = 0
    format: str = "unknown"
    file_size: int = 0
    codec: str = "unknown"


@dataclass
class ProcessingMetadata:
    """Metadata for processing operations."""
    processing_time: float = 0.0
    algorithm_version: str = "unknown"
    parameters: Dict[str, Any] = field(default_factory=dict)
    quality_metrics: Dict[str, float] = field(default_factory=dict)
    timestamp: Optional[str] = None


@dataclass
class GaitCycle:
    """Individual gait cycle information."""
    start_frame: int
    end_frame: int
    duration: float
    left_heel_strikes: List[int] = field(default_factory=list)
    right_heel_strikes: List[int] = field(default_factory=list)
    left_toe_offs: List[int] = field(default_factory=list)
    right_toe_offs: List[int] = field(default_factory=list)


@dataclass
class AnalysisResult:
    """Complete analysis result for a video sequence."""
    sequence_id: str
    video_metadata: VideoMetadata
    gait_metrics: GaitMetrics
    classification_result: ClassificationResult
    gait_cycles: List[GaitCycle] = field(default_factory=list)
    processing_metadata: ProcessingMetadata = field(default_factory=ProcessingMetadata)
    raw_pose_data: Optional[List[Dict[str, Any]]] = None


@dataclass
class TrainingDataSample:
    """Individual training data sample."""
    sample_id: str
    features: GaitFeatures
    ground_truth_label: str
    condition_labels: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class DatasetInfo:
    """Information about a training dataset."""
    dataset_name: str
    version: str
    source_path: Path
    sample_count: int
    condition_distribution: Dict[str, int] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: Optional[str] = None
    updated_at: Optional[str] = None