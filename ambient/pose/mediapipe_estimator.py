"""
MediaPipe pose estimator implementation.

This module provides pose estimation using MediaPipe models
with support for both the new Frame-based API and legacy compatibility.

Author: AlexPose Team
"""

import os
import sys
import io
import contextlib
from pathlib import Path
from typing import Dict, List, Optional, Union, Any
import numpy as np
from loguru import logger

# Import warning suppression utilities
from ambient.pose.suppress_warnings import suppress_stderr_fd

# Check MediaPipe availability - wrap import to suppress any C++ init warnings
try:
    with suppress_stderr_fd():
        import mediapipe as mp
        from mediapipe.tasks import python
        from mediapipe.tasks.python import vision
    MEDIAPIPE_AVAILABLE = True
except ImportError:
    MEDIAPIPE_AVAILABLE = False
    mp = None
    python = None
    vision = None

# Import Frame classes with fallback for development
try:
    from ambient.core.frame import Frame, FrameSequence, FrameError
    from ambient.core.interfaces import IPoseEstimator
    FRAME_SUPPORT = True
except ImportError:
    # Fallback for development/testing
    Frame = Any
    FrameSequence = Any
    FrameError = Exception
    IPoseEstimator = object
    FRAME_SUPPORT = False

# Import base estimator for backward compatibility
from ambient.pose.base_estimator import PoseEstimator as BasePoseEstimator
PoseEstimator = BasePoseEstimator

try:
    import cv2
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False


Keypoint = Dict[str, Union[float, int]]


class MediaPipeEstimator(PoseEstimator, IPoseEstimator):
    """
    Pose estimator using MediaPipe models.
    
    This estimator uses MediaPipe's tasks API to provide efficient pose estimation
    with support for both image and video processing. It supports the new Frame-based
    API while maintaining backward compatibility.
    """
    
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
        
        if not CV2_AVAILABLE:
            raise ImportError("OpenCV is required for MediaPipe estimator")
        
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
        # Suppress C++ warnings during landmarker initialization
        with suppress_stderr_fd():
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
        # Suppress C++ warnings during landmarker initialization
        with suppress_stderr_fd():
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
    
    # New Frame-based methods
    def estimate_pose(self, frame: Frame) -> Dict[str, Any]:
        """
        Estimate pose from a single Frame object.
        
        Args:
            frame: Frame object containing image data
            
        Returns:
            Dictionary containing pose estimation results
        """
        if not FRAME_SUPPORT:
            raise RuntimeError("Frame support not available - Frame classes not imported")
        
        try:
            # Load frame data
            frame_data = frame.load()
            
            # Get frame dimensions
            image_height, image_width = frame_data.shape[:2]
            
            # Convert to RGB if needed (MediaPipe expects RGB)
            if frame.format == "BGR":
                frame_data = cv2.cvtColor(frame_data, cv2.COLOR_BGR2RGB)
            
            # Ensure image is contiguous in memory (required by MediaPipe)
            if not frame_data.flags['C_CONTIGUOUS']:
                frame_data = np.ascontiguousarray(frame_data)
            
            # Create MediaPipe Image object with explicit format
            mp_image = mp.Image(
                image_format=mp.ImageFormat.SRGB,
                data=frame_data
            )
            
            # Create landmarker and detect
            landmarker = self._get_image_landmarker()
            try:
                # Suppress MediaPipe internal warnings during detection
                with suppress_stderr_fd():
                    result = landmarker.detect(mp_image)
                
                # Parse landmarks with explicit dimensions
                keypoints = self._parse_mediapipe_landmarks(result, image_width, image_height)
                
                # Convert to new format
                pose_result = {
                    "keypoints": keypoints,
                    "estimator": self.get_estimator_name(),
                    "format": self.get_keypoint_format(),
                    "frame_metadata": frame.metadata,
                    "confidence_scores": [kp.get("confidence", 0.0) for kp in keypoints],
                    "num_keypoints": len(keypoints),
                    "processing_metadata": {
                        "model_path": str(self.model_path),
                        "frame_shape": frame_data.shape,
                        "frame_format": frame.format,
                        "min_detection_confidence": self.min_pose_detection_confidence
                    }
                }
                
                return pose_result
                
            finally:
                landmarker.close()
                
        except Exception as e:
            logger.error(f"MediaPipe estimation failed for frame: {e}")
            # Return empty result with error information
            return {
                "keypoints": [],
                "estimator": self.get_estimator_name(),
                "format": self.get_keypoint_format(),
                "error": str(e),
                "frame_metadata": frame.metadata,
                "confidence_scores": [],
                "num_keypoints": 0
            }
    
    def estimate_pose_sequence(self, sequence: FrameSequence) -> List[Dict[str, Any]]:
        """
        Estimate poses from a FrameSequence.
        
        Args:
            sequence: FrameSequence object containing multiple frames
            
        Returns:
            List of pose estimation results, one per frame
        """
        if not FRAME_SUPPORT:
            raise RuntimeError("Frame support not available - Frame classes not imported")
        
        results = []
        
        # Create landmarker for video
        landmarker = self._get_video_landmarker()
        
        try:
            # Process frames
            for frame_idx, frame in enumerate(sequence.frames):
                try:
                    # Load frame data
                    frame_data = frame.load()
                    
                    # Get frame dimensions
                    image_height, image_width = frame_data.shape[:2]
                    
                    # Convert to RGB if needed
                    if frame.format == "BGR":
                        frame_data = cv2.cvtColor(frame_data, cv2.COLOR_BGR2RGB)
                    
                    # Ensure frame is contiguous in memory
                    if not frame_data.flags['C_CONTIGUOUS']:
                        frame_data = np.ascontiguousarray(frame_data)
                    
                    # Create MediaPipe Image object
                    mp_image = mp.Image(
                        image_format=mp.ImageFormat.SRGB,
                        data=frame_data
                    )
                    
                    # Calculate timestamp in milliseconds
                    fps = sequence.metadata.get('fps', 30.0)
                    timestamp_ms = int((frame_idx / fps) * 1000)
                    
                    # Detect pose - suppress internal MediaPipe warnings
                    with suppress_stderr_fd():
                        result = landmarker.detect_for_video(mp_image, timestamp_ms)
                    
                    # Parse landmarks
                    keypoints = self._parse_mediapipe_landmarks(result, image_width, image_height)
                    
                    pose_result = {
                        "keypoints": keypoints,
                        "estimator": self.get_estimator_name(),
                        "format": self.get_keypoint_format(),
                        "sequence_index": frame_idx,
                        "frame_metadata": frame.metadata,
                        "confidence_scores": [kp.get("confidence", 0.0) for kp in keypoints],
                        "num_keypoints": len(keypoints),
                        "processing_metadata": {
                            "model_path": str(self.model_path),
                            "timestamp_ms": timestamp_ms
                        }
                    }
                    
                    results.append(pose_result)
                    
                except Exception as e:
                    logger.warning(f"Failed to process frame {frame_idx}: {e}")
                    results.append({
                        "keypoints": [],
                        "estimator": self.get_estimator_name(),
                        "format": self.get_keypoint_format(),
                        "error": str(e),
                        "sequence_index": frame_idx,
                        "confidence_scores": [],
                        "num_keypoints": 0
                    })
        
        finally:
            landmarker.close()
        
        return results
    
    def get_estimator_name(self) -> str:
        """Get the name of this pose estimator."""
        return "MediaPipe"
    
    def get_keypoint_format(self) -> str:
        """Get the keypoint format used by this estimator."""
        return "MEDIAPIPE_33"  # MediaPipe uses 33 landmarks
    
    # Legacy methods for backward compatibility
    def estimate_image_keypoints(
        self,
        image_path: str,
        model: str = "BODY_25",
        bbox: Optional[Dict[str, float]] = None
    ) -> List[Keypoint]:
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
            # Suppress MediaPipe internal warnings during detection
            with suppress_stderr_fd():
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
            logger.debug("MediaPipe internal warnings (feedback manager, NORM_RECT) are suppressed during detection")
            
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
                    
                    # Detect pose - suppress internal MediaPipe warnings
                    with suppress_stderr_fd():
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
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the loaded model."""
        return {
            "model_path": str(self.model_path),
            "model_exists": self.model_path.exists(),
            "default_model": self.default_model,
            "min_pose_detection_confidence": self.min_pose_detection_confidence,
            "min_pose_presence_confidence": self.min_pose_presence_confidence,
            "min_tracking_confidence": self.min_tracking_confidence,
            "keypoint_format": self.get_keypoint_format(),
            "available": self.is_available()
        }
