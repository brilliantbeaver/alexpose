"""
MediaPipe Model Management Module

This module handles MediaPipe model downloads, caching, and landmarker creation.
It follows the Single Responsibility Principle by focusing solely on model
management operations.

Author: AlexPose Team
"""

import urllib.request
from pathlib import Path
from typing import Optional

try:
    import mediapipe as mp
    from mediapipe.tasks import python
    from mediapipe.tasks.python import vision
    MEDIAPIPE_AVAILABLE = True
except ImportError:
    MEDIAPIPE_AVAILABLE = False
    mp = None
    python = None
    vision = None


class MediaPipeModelManager:
    """
    Manages MediaPipe model downloads and caching.
    
    This class follows the Single Responsibility Principle by focusing
    solely on model management operations.
    """
    
    DEFAULT_MODEL_URL = (
        "https://storage.googleapis.com/mediapipe-models/"
        "pose_landmarker/pose_landmarker_full/float16/1/pose_landmarker_full.task"
    )
    
    def __init__(self, models_dir: Optional[Path] = None):
        """
        Initialize the model manager.
        
        Args:
            models_dir: Directory to store models. Defaults to data/models
        """
        if models_dir is None:
            # Default to data/models relative to project root
            self.models_dir = Path.cwd() / "data" / "models"
        else:
            self.models_dir = Path(models_dir)
        
        self.models_dir.mkdir(parents=True, exist_ok=True)
    
    def get_model_path(self, model_name: str = "pose_landmarker_full.task") -> Path:
        """Get the path to a model file."""
        return self.models_dir / model_name
    
    def is_model_downloaded(self, model_name: str = "pose_landmarker_full.task") -> bool:
        """Check if a model is already downloaded."""
        return self.get_model_path(model_name).exists()
    
    def download_model(
        self, 
        model_url: Optional[str] = None,
        model_name: str = "pose_landmarker_full.task",
        force: bool = False
    ) -> Optional[str]:
        """
        Download a MediaPipe model if not already present.
        
        Args:
            model_url: URL to download from. Defaults to MediaPipe full model
            model_name: Name to save the model as
            force: Force re-download even if model exists
            
        Returns:
            Path to the downloaded model, or None if download failed
        """
        model_path = self.get_model_path(model_name)
        
        # Check if already downloaded
        if model_path.exists() and not force:
            print(f"[OK] Model already exists: {model_path}")
            return str(model_path)
        
        # Use default URL if not provided
        if model_url is None:
            model_url = self.DEFAULT_MODEL_URL
        
        print(f"📥 Downloading MediaPipe pose landmarker model...")
        print(f"   URL: {model_url}")
        print(f"   Destination: {model_path}")
        
        try:
            print("⏳ Downloading... (this may take a moment)")
            urllib.request.urlretrieve(model_url, model_path)
            
            # Verify download
            if model_path.exists():
                size_mb = model_path.stat().st_size / (1024 * 1024)
                print(f"[OK] Model downloaded successfully!")
                print(f"[CHART] Size: {size_mb:.1f} MB")
                return str(model_path)
            else:
                print(f"[ERROR] Download completed but file not found")
                return None
                
        except Exception as e:
            print(f"[ERROR] Download failed: {e}")
            # Clean up partial download
            if model_path.exists():
                model_path.unlink()
            return None
    
    def ensure_model_available(
        self,
        model_name: str = "pose_landmarker_full.task",
        model_url: Optional[str] = None
    ) -> Optional[str]:
        """
        Ensure a model is available, downloading if necessary.
        
        Args:
            model_name: Name of the model
            model_url: URL to download from if needed
            
        Returns:
            Path to the model, or None if unavailable
        """
        if self.is_model_downloaded(model_name):
            return str(self.get_model_path(model_name))
        return self.download_model(model_url, model_name)


class PoseLandmarkerFactory:
    """
    Factory for creating MediaPipe Pose Landmarker instances.
    
    This class follows the Factory Pattern and Single Responsibility Principle
    by focusing on landmarker creation with various configurations.
    """
    
    @staticmethod
    def create_landmarker(
        model_path: str,
        num_poses: int = 1,
        min_pose_detection_confidence: float = 0.5,
        min_pose_presence_confidence: float = 0.5,
        min_tracking_confidence: float = 0.5,
        output_segmentation_masks: bool = False
    ):
        """
        Create a MediaPipe Pose Landmarker with specified configuration.
        
        Args:
            model_path: Path to the pose landmarker model file
            num_poses: Maximum number of poses to detect
            min_pose_detection_confidence: Minimum confidence for pose detection
            min_pose_presence_confidence: Minimum confidence for pose presence
            min_tracking_confidence: Minimum confidence for pose tracking
            output_segmentation_masks: Whether to output segmentation masks
            
        Returns:
            Configured PoseLandmarker instance, or None if creation failed
            
        Raises:
            ImportError: If MediaPipe is not available
        """
        if not MEDIAPIPE_AVAILABLE:
            raise ImportError(
                "MediaPipe is not available. Install it with: pip install mediapipe"
            )
        
        try:
            # Import warning suppression
            from ambient.pose.suppress_warnings import suppress_stderr_fd
            
            # Create base options
            base_options = python.BaseOptions(model_asset_path=model_path)
            
            # Create pose landmarker options
            options = vision.PoseLandmarkerOptions(
                base_options=base_options,
                running_mode=vision.RunningMode.IMAGE,
                num_poses=num_poses,
                min_pose_detection_confidence=min_pose_detection_confidence,
                min_pose_presence_confidence=min_pose_presence_confidence,
                min_tracking_confidence=min_tracking_confidence,
                output_segmentation_masks=output_segmentation_masks
            )
            
            # Create the landmarker with warning suppression
            with suppress_stderr_fd():
                landmarker = vision.PoseLandmarker.create_from_options(options)
            
            print(f"[OK] Pose Landmarker created from {model_path}")
            return landmarker
            
        except Exception as e:
            print(f"[ERROR] Failed to create Pose Landmarker: {e}")
            return None
