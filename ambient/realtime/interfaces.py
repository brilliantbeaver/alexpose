"""
Interfaces for realtime gait analysis components.

This module defines the contracts that realtime components must implement,
following the Interface Segregation Principle.
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Any, Tuple
import numpy as np
from dataclasses import dataclass
from enum import Enum


class ProcessingMode(Enum):
    """Processing mode for realtime analysis."""
    FAST = "fast"
    BALANCED = "balanced"
    ACCURATE = "accurate"


@dataclass
class RealtimeFrame:
    """Represents a frame in the realtime processing pipeline."""
    data: np.ndarray
    timestamp: float
    frame_id: int
    metadata: Dict[str, Any]


@dataclass
class RealtimePoseResult:
    """Result of realtime pose estimation."""
    keypoints: List[Dict[str, Any]]
    confidence_scores: List[float]
    processing_time_ms: float
    frame_id: int
    timestamp: float
    estimator_info: Dict[str, Any]


@dataclass
class RealtimeGaitMetrics:
    """Real-time gait analysis metrics."""
    cadence: Optional[float]
    step_length: Optional[float]
    stride_length: Optional[float]
    walking_speed: Optional[float]
    symmetry_index: Optional[float]
    stability_score: Optional[float]
    confidence: float
    timestamp: float


class IRealtimePoseEstimator(ABC):
    """Interface for realtime pose estimation."""
    
    @abstractmethod
    def estimate_pose(self, frame: RealtimeFrame) -> RealtimePoseResult:
        """Estimate pose from a single frame."""
        pass
    
    @abstractmethod
    def set_processing_mode(self, mode: ProcessingMode) -> None:
        """Set processing mode for performance optimization."""
        pass
    
    @abstractmethod
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get performance statistics."""
        pass
    
    @abstractmethod
    def is_ready(self) -> bool:
        """Check if estimator is ready for processing."""
        pass


class IRealtimeGaitAnalyzer(ABC):
    """Interface for realtime gait analysis."""
    
    @abstractmethod
    def analyze_pose_sequence(
        self, 
        poses: List[RealtimePoseResult]
    ) -> RealtimeGaitMetrics:
        """Analyze a sequence of poses for gait metrics."""
        pass
    
    @abstractmethod
    def update_with_pose(self, pose: RealtimePoseResult) -> Optional[RealtimeGaitMetrics]:
        """Update analysis with a new pose, return metrics if available."""
        pass
    
    @abstractmethod
    def reset_analysis(self) -> None:
        """Reset analysis state."""
        pass
    
    @abstractmethod
    def get_required_pose_count(self) -> int:
        """Get minimum number of poses needed for analysis."""
        pass


class IFrameBuffer(ABC):
    """Interface for frame buffer management."""
    
    @abstractmethod
    def add_frame(self, frame: RealtimeFrame) -> None:
        """Add a frame to the buffer."""
        pass
    
    @abstractmethod
    def get_latest_frame(self) -> Optional[RealtimeFrame]:
        """Get the most recent frame."""
        pass
    
    @abstractmethod
    def get_frame_sequence(self, count: int) -> List[RealtimeFrame]:
        """Get a sequence of recent frames."""
        pass
    
    @abstractmethod
    def clear(self) -> None:
        """Clear all frames from buffer."""
        pass
    
    @abstractmethod
    def get_buffer_stats(self) -> Dict[str, Any]:
        """Get buffer statistics."""
        pass


class IPoseTracker(ABC):
    """Interface for pose tracking across frames."""
    
    @abstractmethod
    def track_pose(
        self, 
        current_pose: RealtimePoseResult,
        previous_poses: List[RealtimePoseResult]
    ) -> RealtimePoseResult:
        """Track pose across frames for consistency."""
        pass
    
    @abstractmethod
    def get_tracking_confidence(self) -> float:
        """Get current tracking confidence."""
        pass
    
    @abstractmethod
    def reset_tracking(self) -> None:
        """Reset tracking state."""
        pass


class IStreamProcessor(ABC):
    """Interface for stream processing coordination."""
    
    @abstractmethod
    async def process_frame(self, frame_data: bytes) -> Dict[str, Any]:
        """Process a single frame from the stream."""
        pass
    
    @abstractmethod
    def set_processing_parameters(self, params: Dict[str, Any]) -> None:
        """Set processing parameters."""
        pass
    
    @abstractmethod
    def get_processing_stats(self) -> Dict[str, Any]:
        """Get processing statistics."""
        pass
    
    @abstractmethod
    def start_processing(self) -> None:
        """Start the processing pipeline."""
        pass
    
    @abstractmethod
    def stop_processing(self) -> None:
        """Stop the processing pipeline."""
        pass


class IRealtimeService(ABC):
    """Interface for realtime service coordination."""
    
    @abstractmethod
    async def handle_frame(self, frame_data: bytes) -> Dict[str, Any]:
        """Handle incoming frame data."""
        pass
    
    @abstractmethod
    async def get_current_metrics(self) -> Dict[str, Any]:
        """Get current analysis metrics."""
        pass
    
    @abstractmethod
    def configure_analysis(self, config: Dict[str, Any]) -> None:
        """Configure analysis parameters."""
        pass
    
    @abstractmethod
    def start_session(self) -> str:
        """Start a new analysis session."""
        pass
    
    @abstractmethod
    def end_session(self, session_id: str) -> Dict[str, Any]:
        """End an analysis session and return summary."""
        pass


# Type aliases for better readability
FrameData = np.ndarray
KeypointData = List[Dict[str, Any]]
MetricsData = Dict[str, Any]
ConfigData = Dict[str, Any]