"""
Base pose estimator class and common utilities.

This module provides the base class for all pose estimators and shared
data structures.

Author: AlexPose Team
"""

from typing import Optional, Dict, Any, List, Union
from pathlib import Path
from dataclasses import dataclass


@dataclass
class Keypoint:
    """Keypoint data structure."""
    x: float
    y: float
    confidence: float
    id: int = 0


class PoseEstimator:
    """
    Base class for pose estimators.
    
    This class defines the interface that all pose estimators must implement
    for backward compatibility with legacy code.
    """
    
    def is_available(self) -> bool:
        """
        Check if the pose estimator is available and properly configured.
        
        Returns:
            True if estimator is available, False otherwise
        """
        return True  # Default implementation assumes availability
    
    def estimate_image_keypoints(
        self,
        image_path: str,
        model: str = "BODY_25",
        bbox: Optional[Dict[str, float]] = None
    ) -> List[Dict[str, Any]]:
        """
        Estimate keypoints from a single image.
        
        Args:
            image_path: Path to image file
            model: Model name/type
            bbox: Optional bounding box for region of interest
            
        Returns:
            List of keypoint dictionaries
        """
        raise NotImplementedError
    
    def estimate_video_keypoints(
        self,
        video_path: Union[str, Path],
        model: str = "BODY_25"
    ) -> Union[Dict[str, Any], List[List[Dict[str, Any]]]]:
        """
        Estimate keypoints for all frames in a video.
        
        Args:
            video_path: Path to video file
            model: Model name/type
            
        Returns:
            Dictionary with 'frames', 'video_width', 'video_height' keys
            or List of frame keypoints for backward compatibility
        """
        raise NotImplementedError
    
    def cache_fingerprint(self) -> str:
        """
        Get a unique fingerprint for caching purposes.
        
        Returns:
            Unique identifier string
        """
        return self.__class__.__name__
    
    def supports_video_batch(self) -> bool:
        """
        Check if this estimator supports video batch processing.
        
        Returns:
            True if batch processing is supported
        """
        return False
