"""
Interface definitions for the Ambient gait analysis system.

This module defines the abstract interfaces that all concrete implementations
must follow, ensuring consistency and enabling dependency injection.

@Theodore Mui
Monday, July 28, 2025 12:30:00 AM
"""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

# Forward declarations for Frame types and data models
try:
    from ambient.core.frame import Frame, FrameSequence
    from ambient.core.data_models import (
        GaitFeatures, GaitMetrics, ClassificationResult, 
        ConditionPrediction, AnalysisResult
    )
except ImportError:
    # Handle circular import during initial setup
    Frame = Any
    FrameSequence = Any
    GaitFeatures = Any
    GaitMetrics = Any
    ClassificationResult = Any
    ConditionPrediction = Any
    AnalysisResult = Any


class IConfigurationManager(ABC):
    """Interface for configuration management."""

    @abstractmethod
    def get_config_value(
        self, key: str, default: Any = None, target_type: Optional[type] = None
    ) -> Any:
        """
        Get a configuration value by key with support for nested keys and type conversion.

        Args:
            key: Configuration key to retrieve (supports dot notation for nested keys)
            default: Default value if key not found
            target_type: Optional explicit type to convert the value to

        Returns:
            Configuration value or default, converted to appropriate type
        """
        pass

    @abstractmethod
    def get_videos_directory(self) -> Path:
        """Get the videos directory path."""
        pass

    @abstractmethod
    def get_openpose_directory(self) -> Path:
        """Get the OpenPose outputs directory path."""
        pass

    @abstractmethod
    def get_gemini_api_key(self) -> str:
        """Get the Gemini API key."""
        pass

    @abstractmethod
    def is_gemini_configured(self) -> bool:
        """Check if Gemini is properly configured."""
        pass

    @abstractmethod
    def validate_configuration(self) -> bool:
        """Validate all configuration settings."""
        pass


class IFileManager(ABC):
    """Interface for file management operations."""

    @abstractmethod
    def upload_video(self, video_path: str) -> Optional[Any]:
        """Upload a video file and return a reference."""
        pass

    @abstractmethod
    def upload_csv(self, csv_path: str) -> Optional[Any]:
        """Upload a CSV file and return a reference."""
        pass

    @abstractmethod
    def get_cached_reference(self, file_path: str) -> Optional[Any]:
        """Get a cached file reference if available."""
        pass

    @abstractmethod
    def clear_cache(self) -> None:
        """Clear the file cache."""
        pass


class IAnalyzer(ABC):
    """Interface for analysis operations."""

    @abstractmethod
    def analyze_video(self, video_path: str, csv_paths: List[str]) -> tuple:
        """
        Analyze a video with associated CSV files.

        Returns:
            Tuple of (raw_response, generated_text) where:
            - raw_response: The complete response object from the AI model
            - generated_text: The extracted analysis text
        """
        pass

    @abstractmethod
    def get_analysis_prompt(self) -> str:
        """Get the analysis prompt template."""
        pass


class IOutputManager(ABC):
    """Interface for output management."""

    @abstractmethod
    def save_raw_response(self, record_id: str, response: Any) -> Path:
        """Save the raw response to a file."""
        pass

    @abstractmethod
    def save_analysis_text(self, record_id: str, text: str) -> Path:
        """Save the analysis text to a file."""
        pass

    @abstractmethod
    def get_output_directories(self) -> Dict[str, Path]:
        """Get the output directory paths."""
        pass


class IPoseEstimator(ABC):
    """Interface for pose estimation operations."""

    @abstractmethod
    def estimate_pose(self, frame: Frame) -> Dict[str, Any]:
        """
        Estimate pose from a single frame.

        Args:
            frame: Frame object containing image data

        Returns:
            Dictionary containing pose estimation results with keypoints,
            confidence scores, and metadata
        """
        pass

    @abstractmethod
    def estimate_pose_sequence(self, sequence: FrameSequence) -> List[Dict[str, Any]]:
        """
        Estimate poses from a sequence of frames.

        Args:
            sequence: FrameSequence object containing multiple frames

        Returns:
            List of pose estimation results, one per frame
        """
        pass

    @abstractmethod
    def get_estimator_name(self) -> str:
        """Get the name of this pose estimator."""
        pass

    @abstractmethod
    def get_keypoint_format(self) -> str:
        """Get the keypoint format used by this estimator (e.g., 'COCO', 'BODY_25')."""
        pass

    @abstractmethod
    def is_available(self) -> bool:
        """Check if this estimator is available and properly configured."""
        pass

    # Legacy methods for backward compatibility with existing GAVD processing
    @abstractmethod
    def estimate_image_keypoints(
        self,
        image_path: str,
        model: str = "BODY_25",
        bbox: Optional[Dict[str, Union[int, float]]] = None,
    ) -> List[Dict[str, Union[float, int]]]:
        """
        Legacy method for estimating keypoints from image path.
        
        This method maintains backward compatibility with existing GAVD processing
        while internally using the new Frame-based approach.
        
        Args:
            image_path: Path to the input image frame
            model: Pose model to use
            bbox: Optional bounding box
            
        Returns:
            List of keypoints as dicts with keys: x, y, confidence
        """
        pass

    @abstractmethod
    def estimate_video_keypoints(
        self,
        video_path: Union[str, Path],
        model: str = "BODY_25",
    ) -> List[List[Dict[str, Union[float, int]]]]:
        """
        Legacy method for estimating keypoints from video path.
        
        This method maintains backward compatibility with existing GAVD processing
        while internally using the new Frame-based approach.
        
        Args:
            video_path: Path to the input video file
            model: Pose model to use
            
        Returns:
            List of pose estimation results, one per frame
        """
        pass


class IVideoProcessor(ABC):
    """Interface for video processing operations."""

    @abstractmethod
    def load_video(self, video_path: Union[str, Path]) -> FrameSequence:
        """
        Load video as a FrameSequence.

        Args:
            video_path: Path to video file

        Returns:
            FrameSequence object containing video frames
        """
        pass

    @abstractmethod
    def extract_frame(self, video_path: Union[str, Path], frame_index: int) -> Frame:
        """
        Extract a single frame from video.

        Args:
            video_path: Path to video file
            frame_index: Frame index to extract (0-based)

        Returns:
            Frame object containing the extracted frame
        """
        pass

    @abstractmethod
    def get_video_info(self, video_path: Union[str, Path]) -> Dict[str, Any]:
        """
        Get video metadata information.

        Args:
            video_path: Path to video file

        Returns:
            Dictionary containing video metadata (duration, fps, resolution, etc.)
        """
        pass


class IGaitAnalyzer(ABC):
    """Interface for gait analysis operations."""

    @abstractmethod
    def analyze_gait_sequence(
        self, 
        pose_sequence: List[Dict[str, Any]], 
        metadata: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Analyze gait patterns from pose sequence.

        Args:
            pose_sequence: List of pose estimation results
            metadata: Optional metadata about the sequence

        Returns:
            Dictionary containing gait analysis results
        """
        pass

    @abstractmethod
    def extract_gait_features(self, pose_sequence: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Extract gait features from pose sequence.

        Args:
            pose_sequence: List of pose estimation results

        Returns:
            Dictionary containing extracted gait features
        """
        pass

    @abstractmethod
    def detect_gait_cycles(self, pose_sequence: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Detect gait cycles in pose sequence.

        Args:
            pose_sequence: List of pose estimation results

        Returns:
            List of detected gait cycles with timing information
        """
        pass


class IClassifier(ABC):
    """Interface for classification operations."""

    @abstractmethod
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
        pass

    @abstractmethod
    def get_classification_confidence(self, result: Dict[str, Any]) -> float:
        """
        Get confidence score for classification result.

        Args:
            result: Classification result

        Returns:
            Confidence score between 0 and 1
        """
        pass

    @abstractmethod
    def explain_classification(self, result: Dict[str, Any]) -> str:
        """
        Generate explanation for classification result.

        Args:
            result: Classification result

        Returns:
            Human-readable explanation of the classification
        """
        pass
