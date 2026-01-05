"""
Pose estimator factory and integration for GAVD processing.

Provides a unified interface for different pose estimation frameworks.
"""

from typing import Optional, Dict, Any, List
from pathlib import Path
from loguru import logger
from dataclasses import dataclass
import os

# Suppress TensorFlow Lite verbose warnings (keep errors)
os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '2')  # 0=all, 1=info, 2=warning, 3=error

# Check MediaPipe availability
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


@dataclass
class Keypoint:
    """Keypoint data structure."""
    x: float
    y: float
    confidence: float
    id: int = 0


class PoseEstimator:
    """Base class for pose estimators."""
    
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
        video_path: Path,
        model: str = "BODY_25"
    ) -> Dict[str, Any]:
        """
        Estimate keypoints for all frames in a video.
        
        Args:
            video_path: Path to video file
            model: Model name/type
            
        Returns:
            Dictionary with:
                - 'frames': List of frame keypoints (each frame is a list of keypoint dicts)
                - 'video_width': Actual video width used for keypoint coordinates
                - 'video_height': Actual video height used for keypoint coordinates
        """
        raise NotImplementedError
    
    def cache_fingerprint(self) -> str:
        """
        Get a unique fingerprint for caching purposes.
        
        Returns:
            Unique identifier string
        """
        return self.__class__.__name__


class MediaPipeEstimator(PoseEstimator):
    """MediaPipe pose estimator implementation using tasks API."""
    
    def __init__(
        self,
        model_path: Optional[str] = None,
        default_model: str = "BODY_25",
        min_pose_detection_confidence: float = 0.5,
        min_pose_presence_confidence: float = 0.5,
        min_tracking_confidence: float = 0.5
    ):
        """
        Initialize MediaPipe estimator with tasks API.
        
        Args:
            model_path: Path to MediaPipe pose landmarker model file (.task)
            default_model: Default model name (for compatibility)
            min_pose_detection_confidence: Minimum confidence for pose detection
            min_pose_presence_confidence: Minimum confidence for pose presence
            min_tracking_confidence: Minimum confidence for tracking
        """
        if not MEDIAPIPE_AVAILABLE:
            raise ImportError(
                "MediaPipe is not installed. Install with: pip install mediapipe"
            )
        
        # Store configuration
        self.default_model = default_model
        self.min_pose_detection_confidence = min_pose_detection_confidence
        self.min_pose_presence_confidence = min_pose_presence_confidence
        self.min_tracking_confidence = min_tracking_confidence
        
        # Resolve model path
        if model_path is None:
            # Use default model path
            model_path = Path("data/models/pose_landmarker_lite.task")
        
        self.model_path = Path(model_path).resolve()
        
        # Check if model file exists
        if not self.model_path.exists():
            raise FileNotFoundError(
                f"MediaPipe model file not found: {self.model_path}\n"
                f"Download from: https://developers.google.com/mediapipe/solutions/vision/pose_landmarker/index#models"
            )
        
        logger.info(f"MediaPipe pose estimator initialized with model: {self.model_path}")
    
    def is_available(self) -> bool:
        """
        Check if MediaPipe estimator is available.
        
        Returns:
            True if MediaPipe is installed and model file exists
        """
        return MEDIAPIPE_AVAILABLE and self.model_path.exists()
    
    def _get_image_landmarker(self):
        """Create and return a MediaPipe PoseLandmarker for image processing."""
        base_options = python.BaseOptions(model_asset_path=str(self.model_path))
        options = vision.PoseLandmarkerOptions(
            base_options=base_options,
            running_mode=vision.RunningMode.IMAGE,
            min_pose_detection_confidence=self.min_pose_detection_confidence,
            min_pose_presence_confidence=self.min_pose_presence_confidence,
            min_tracking_confidence=self.min_tracking_confidence
        )
        return vision.PoseLandmarker.create_from_options(options)
    
    def _get_video_landmarker(self):
        """Create and return a MediaPipe PoseLandmarker for video processing."""
        base_options = python.BaseOptions(model_asset_path=str(self.model_path))
        options = vision.PoseLandmarkerOptions(
            base_options=base_options,
            running_mode=vision.RunningMode.VIDEO,
            min_pose_detection_confidence=self.min_pose_detection_confidence,
            min_pose_presence_confidence=self.min_pose_presence_confidence,
            min_tracking_confidence=self.min_tracking_confidence
        )
        return vision.PoseLandmarker.create_from_options(options)
    
    def _parse_mediapipe_landmarks(
        self,
        result,
        image_width: int = 1,
        image_height: int = 1
    ) -> List[Dict[str, Any]]:
        """
        Parse MediaPipe landmarks into keypoint format.
        
        Args:
            result: MediaPipe PoseLandmarkerResult
            image_width: Image width for coordinate conversion
            image_height: Image height for coordinate conversion
            
        Returns:
            List of keypoint dictionaries
        """
        if not result.pose_landmarks or len(result.pose_landmarks) == 0:
            return []
        
        # Get first person's landmarks
        landmarks = result.pose_landmarks[0]
        
        keypoints = []
        for idx, landmark in enumerate(landmarks):
            keypoints.append({
                "x": landmark.x * image_width,
                "y": landmark.y * image_height,
                "confidence": landmark.visibility,
                "id": idx
            })
        
        return keypoints
    
    def estimate_image_keypoints(
        self,
        image_path: str,
        model: str = "BODY_25",
        bbox: Optional[Dict[str, float]] = None
    ) -> List[Dict[str, Any]]:
        """
        Estimate keypoints using MediaPipe tasks API.
        
        Args:
            image_path: Path to image file
            model: Model name (for compatibility, not used)
            bbox: Optional bounding box for region of interest
            
        Returns:
            List of keypoint dictionaries
        """
        import cv2
        import numpy as np
        
        # Check if image file exists
        if not Path(image_path).exists():
            raise FileNotFoundError(f"Image file not found: {image_path}")
        
        # Read image
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Failed to read image: {image_path}")
        
        # Get image dimensions
        image_height, image_width = image.shape[:2]
        
        # Apply bounding box crop if provided
        if bbox:
            left = int(bbox.get("left", 0))
            top = int(bbox.get("top", 0))
            width = int(bbox.get("width", image_width))
            height = int(bbox.get("height", image_height))
            
            # Ensure bounds are valid
            left = max(0, min(left, image_width - 1))
            top = max(0, min(top, image_height - 1))
            right = min(left + width, image_width)
            bottom = min(top + height, image_height)
            
            image = image[top:bottom, left:right]
            image_height, image_width = image.shape[:2]
        
        # Convert BGR to RGB
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Ensure image is contiguous in memory (required by MediaPipe)
        if not image_rgb.flags['C_CONTIGUOUS']:
            image_rgb = np.ascontiguousarray(image_rgb)
        
        # Create MediaPipe Image object with explicit format
        mp_image = mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=image_rgb
        )
        
        # Create landmarker and detect
        landmarker = self._get_image_landmarker()
        try:
            result = landmarker.detect(mp_image)
            
            if not result.pose_landmarks or len(result.pose_landmarks) == 0:
                logger.warning(f"No pose detected in image: {image_path}")
                return []
            
            # Parse landmarks with explicit dimensions
            keypoints = self._parse_mediapipe_landmarks(result, image_width, image_height)
            
            # Adjust coordinates if bbox was applied
            if bbox:
                left = int(bbox.get("left", 0))
                top = int(bbox.get("top", 0))
                for kp in keypoints:
                    kp["x"] += left
                    kp["y"] += top
            
            return keypoints
            
        finally:
            landmarker.close()
    
    def estimate_video_keypoints(
        self,
        video_path: Path,
        model: str = "BODY_25"
    ) -> Dict[str, Any]:
        """
        Estimate keypoints for all frames in a video.
        
        Args:
            video_path: Path to video file
            model: Model name (for compatibility, not used)
            
        Returns:
            Dictionary with:
                - 'frames': List of frame keypoints (each frame is a list of keypoint dicts)
                - 'video_width': Actual video width used for keypoint coordinates
                - 'video_height': Actual video height used for keypoint coordinates
        """
        import cv2
        import numpy as np
        
        # Check if video file exists
        if not Path(video_path).exists():
            raise FileNotFoundError(f"Video file not found: {video_path}")
        
        # Open video
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise ValueError(f"Failed to open video: {video_path}")
        
        try:
            # Get video properties
            fps = cap.get(cv2.CAP_PROP_FPS)
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            
            logger.info(f"Processing video: {frame_count} frames at {fps} fps ({width}x{height})")
            logger.debug("Note: MediaPipe warnings about 'feedback manager' and 'NORM_RECT' are expected and can be ignored")
            
            # Create landmarker for video
            landmarker = self._get_video_landmarker()
            
            try:
                all_keypoints = []
                frame_idx = 0
                
                while True:
                    ret, frame = cap.read()
                    if not ret:
                        break
                    
                    # Convert BGR to RGB
                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    
                    # Ensure frame is contiguous in memory (required by MediaPipe)
                    if not frame_rgb.flags['C_CONTIGUOUS']:
                        frame_rgb = np.ascontiguousarray(frame_rgb)
                    
                    # Create MediaPipe Image object with explicit format
                    mp_image = mp.Image(
                        image_format=mp.ImageFormat.SRGB,
                        data=frame_rgb
                    )
                    
                    # Calculate timestamp in milliseconds
                    timestamp_ms = int((frame_idx / fps) * 1000)
                    
                    # Detect pose
                    result = landmarker.detect_for_video(mp_image, timestamp_ms)
                    
                    # Parse landmarks with explicit dimensions
                    keypoints = self._parse_mediapipe_landmarks(result, width, height)
                    all_keypoints.append(keypoints)
                    
                    frame_idx += 1
                
                logger.info(f"Processed {frame_idx} frames")
                
                # Return keypoints with video dimensions for proper scaling
                return {
                    'frames': all_keypoints,
                    'video_width': width,
                    'video_height': height
                }
                
            finally:
                landmarker.close()
                
        finally:
            cap.release()
    
    def supports_video_batch(self) -> bool:
        """Check if this estimator supports video batch processing."""
        return True
    
    def cache_fingerprint(self) -> str:
        """Get cache fingerprint."""
        return "mediapipe_tasks_v1"


class OpenPoseEstimator(PoseEstimator):
    """OpenPose estimator placeholder - not yet implemented."""
    
    def __init__(self):
        """Initialize OpenPose estimator."""
        logger.warning("OpenPose estimator is not yet implemented")
        raise NotImplementedError("OpenPose estimator not yet implemented")
    
    def is_available(self) -> bool:
        """
        Check if OpenPose estimator is available.
        
        Returns:
            False - OpenPose is not yet implemented
        """
        return False
    
    def estimate_image_keypoints(
        self,
        image_path: str,
        model: str = "BODY_25",
        bbox: Optional[Dict[str, float]] = None
    ) -> List[Dict[str, Any]]:
        """Estimate keypoints using OpenPose."""
        raise NotImplementedError("OpenPose estimator not yet implemented")
    
    def cache_fingerprint(self) -> str:
        """Get cache fingerprint."""
        return "openpose_v1"


def get_pose_estimator(estimator_type: str = "mediapipe") -> Optional[PoseEstimator]:
    """
    Factory function to get a pose estimator instance.
    
    Args:
        estimator_type: Type of estimator (mediapipe, openpose, etc.)
        
    Returns:
        PoseEstimator instance or None if not available
    """
    estimator_type = estimator_type.lower()
    
    if estimator_type == "mediapipe":
        if not MEDIAPIPE_AVAILABLE:
            logger.error("MediaPipe is not installed. Install with: pip install mediapipe")
            return None
        
        try:
            # Try to create MediaPipe estimator with default model path
            model_path = Path("data/models/pose_landmarker_lite.task")
            
            # If default model doesn't exist, try to use built-in model
            if not model_path.exists():
                logger.warning(
                    f"Default model not found at {model_path}. "
                    f"Download from: https://developers.google.com/mediapipe/solutions/vision/pose_landmarker/index#models"
                )
                # Try alternative paths
                alt_paths = [
                    Path("pose_landmarker_lite.task"),
                    Path("models/pose_landmarker_lite.task"),
                    Path("../data/models/pose_landmarker_lite.task")
                ]
                
                for alt_path in alt_paths:
                    if alt_path.exists():
                        model_path = alt_path
                        logger.info(f"Using model from: {model_path}")
                        break
                else:
                    logger.error(
                        "No MediaPipe model file found. Please download pose_landmarker_lite.task "
                        "from https://developers.google.com/mediapipe/solutions/vision/pose_landmarker/index#models"
                    )
                    return None
            
            return MediaPipeEstimator(model_path=str(model_path))
            
        except Exception as e:
            logger.error(f"Failed to create MediaPipe estimator: {str(e)}")
            return None
    
    elif estimator_type == "openpose":
        logger.warning("OpenPose estimator not yet implemented")
        return None
    
    elif estimator_type == "ultralytics":
        logger.warning("Ultralytics estimator not yet implemented")
        return None
    
    elif estimator_type == "alphapose":
        logger.warning("AlphaPose estimator not yet implemented")
        return None
    
    else:
        logger.error(f"Unknown estimator type: {estimator_type}")
        return None
