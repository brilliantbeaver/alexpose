"""
Keypoint Extractor Module

This module provides the SequenceKeypointExtractor class for extracting
pose keypoints from video sequences using MediaPipe.

The extractor handles:
- Single image keypoint extraction
- Video frame keypoint extraction
- Batch sequence processing
- Model management and caching
- Warning suppression for clean output

Author: AlexPose Team
"""

import sys
import os
from pathlib import Path
from typing import List, Optional
import pandas as pd
import cv2
import numpy as np

# Import data structures
from ambient.pose.keypoint_data import (
    Keypoint,
    KeypointSet,
    KeypointFormat,
    MEDIAPIPE_33_NAMES,
)

# Import model management
from ambient.pose.model_management import (
    MediaPipeModelManager,
    PoseLandmarkerFactory,
)

# Import utilities
from ambient.pose.utils import suppress_stderr_fd
from ambient.utils.log_config import get_logger

logger = get_logger(__name__)
           
# MediaPipe pose landmark names
POSE_LANDMARK_NAMES = MEDIAPIPE_33_NAMES

try:
    import mediapipe as mp
    MEDIAPIPE_AVAILABLE = True
except ImportError:
    MEDIAPIPE_AVAILABLE = False
    mp = None


class SequenceKeypointExtractor:
    """
    Extracts pose keypoints from video sequences.
    
    This class handles the extraction of pose keypoints from video frames,
    supporting both single frames and batch processing of sequences.
    It follows the Single Responsibility Principle by focusing on
    keypoint extraction from video data.
    
    Automatically detects Windows threading issues and switches to process
    isolation when needed to prevent WinError 1 problems.
    
    Attributes:
        model_manager: MediaPipeModelManager instance for model handling
        landmarker_factory: PoseLandmarkerFactory for creating landmarkers
        suppress_warnings: Whether to suppress MediaPipe warnings
        
    Example:
        >>> extractor = SequenceKeypointExtractor()
        >>> image = cv2.imread("person.jpg")
        >>> image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        >>> keypoints = extractor.extract_from_image(image_rgb)
        >>> print(f"Detected {len(keypoints)} keypoints")
    """
    
    def __init__(
        self,
        model_manager: Optional[MediaPipeModelManager] = None,
        landmarker_factory: Optional[PoseLandmarkerFactory] = None,
        suppress_warnings: bool = True,
        use_process_isolation: Optional[bool] = None
    ):
        """
        Initialize the sequence keypoint extractor.
        
        Args:
            model_manager: Model manager instance. Creates default if None
            landmarker_factory: Landmarker factory instance. Uses default if None
            suppress_warnings: Whether to suppress MediaPipe warnings during extraction
            use_process_isolation: Force process isolation (None = auto-detect on Windows)
        """
        self.model_manager = model_manager or MediaPipeModelManager()
        self.landmarker_factory = landmarker_factory or PoseLandmarkerFactory()
        self.suppress_warnings = suppress_warnings
        self._landmarker = None
        self._model_path = None
        self._frame_count = 0  # Track frames processed for memory management
        self._max_frames_before_reset = 100  # Reset landmarker every 100 frames to prevent memory leaks
        
        # Process isolation for Windows threading issues
        self._use_process_isolation = use_process_isolation
        self._process_extractor = None
        self._threading_failures = 0  # Track consecutive threading failures
        self._max_threading_failures = 3  # Switch to process isolation after 3 failures
    
    def _should_use_process_isolation(self) -> bool:
        """
        Determine if process isolation should be used.
        
        Returns:
            True if process isolation should be used
        """
        if self._use_process_isolation is not None:
            return self._use_process_isolation
        
        # Auto-detect: use process isolation on Windows if threading failures occur
        if os.name == 'nt':  # Windows
            return self._threading_failures >= self._max_threading_failures
        
        return False
    
    def _get_process_extractor(self):
        """Get or create process-isolated extractor."""
        if self._process_extractor is None:
            from ambient.pose.process_isolated_extractor import ProcessIsolatedSequenceExtractor
            
            model_path = self.model_manager.ensure_model_available()
            self._process_extractor = ProcessIsolatedSequenceExtractor(
                model_path=model_path,
                num_workers=1,  # Single worker for Windows stability
                worker_timeout=30.0
            )
            
            # Better logging based on why process isolation is being used
            if self._use_process_isolation is True:
                logger.info(f"Using process isolation (configured for Windows optimization)")
            else:
                logger.info(f"Switched to process isolation due to threading issues")
        
        return self._process_extractor
    def _ensure_landmarker(self, model_path: Optional[str] = None):
        """Ensure a landmarker is available using singleton pattern to prevent memory leaks."""
        if model_path is None:
            # Ensure model is downloaded
            model_path = self.model_manager.ensure_model_available()
            if model_path is None:
                raise RuntimeError("Failed to download or locate pose model")
        
        # Use singleton pattern to prevent MediaPipe memory leaks on Windows
        from ambient.pose.mediapipe_singleton import get_mediapipe_singleton
        
        singleton = get_mediapipe_singleton()
        
        try:
            landmarker = singleton.get_landmarker(model_path, self.landmarker_factory)
            self._threading_failures = 0  # Reset failure counter on success
            return landmarker
        except Exception as e:
            # Check if this is a threading-related error
            error_str = str(e)
            if "WinError 1" in error_str or "Incorrect function" in error_str:
                self._threading_failures += 1
                logger.warning(f"Threading failure #{self._threading_failures}: {e}")
                
                # If too many failures, switch to process isolation
                if self._threading_failures >= self._max_threading_failures:
                    logger.warning(f"Too many threading failures, switching to process isolation")
                    raise RuntimeError("Threading failures detected, switching to process isolation")
            
            # If singleton fails, try to reset and retry once
            logger.warning(f"Singleton landmarker failed, resetting: {e}")
            singleton.reset_landmarker()
            
            try:
                landmarker = singleton.get_landmarker(model_path, self.landmarker_factory)
                self._threading_failures = 0  # Reset failure counter on success
                return landmarker
            except Exception as retry_e:
                self._threading_failures += 1
                raise RuntimeError(f"Failed to create landmarker after reset: {retry_e}") from e
    
    def reset_landmarker(self):
        """Reset the landmarker state using singleton pattern."""
        from ambient.pose.mediapipe_singleton import get_mediapipe_singleton
        
        singleton = get_mediapipe_singleton()
        singleton.reset_landmarker()
        
        # Also reset local state
        self._landmarker = None
        self._model_path = None
        self._frame_count = 0
        self._threading_failures = 0  # Reset failure counter
    
    def cleanup(self):
        """Clean up resources including process isolation."""
        # Reset singleton state
        self.reset_landmarker()
        
        # Clean up process extractor if it exists
        if self._process_extractor is not None:
            try:
                self._process_extractor.stop()
            except Exception as e:
                logger.warning(f"Error stopping process extractor: {e}")
            finally:
                self._process_extractor = None
    
    def __del__(self):
        """Destructor - ensure cleanup."""
        try:
            self.cleanup()
        except Exception:
            pass
    
    def extract_from_image(
        self,
        image: np.ndarray,
        model_path: Optional[str] = None
    ) -> KeypointSet:
        """
        Extract pose keypoints from a single image using singleton pattern to prevent memory leaks.
        
        Automatically switches to process isolation if threading issues are detected on Windows.
        
        Args:
            image: RGB image array (height, width, 3)
            model_path: Optional path to model file
            
        Returns:
            KeypointSet object with:
            - keypoints: List of Keypoint objects with full metadata
            - format: KeypointFormat.MEDIAPIPE_33
            - frame_width, frame_height: Image dimensions
            
        Raises:
            ImportError: If MediaPipe is not available
            RuntimeError: If landmarker creation fails
            
        Example:
            >>> extractor = SequenceKeypointExtractor()
            >>> image = cv2.imread("person.jpg")
            >>> image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            >>> keypoints = extractor.extract_from_image(image_rgb)
        """
        if not MEDIAPIPE_AVAILABLE:
            raise ImportError("MediaPipe is required for pose extraction")
        
        # Check if we should use process isolation FIRST (before any MediaPipe operations)
        if self._should_use_process_isolation():
            # Log why we're using process isolation
            # if self._use_process_isolation is True:
            #     logger.info(f"Using process isolation for MediaPipe (Windows optimization)")
            # else:
            #     logger.info(f"Using process isolation due to {self._threading_failures} threading failures")
            
            try:
                process_extractor = self._get_process_extractor()
                result = process_extractor.extract_from_image(image)
                if result is not None:
                    return result
                else:
                    # Fallback to empty result
                    height, width = image.shape[:2]
                    return KeypointSet(
                        keypoints=[],
                        format=KeypointFormat.CUSTOM,
                        frame_width=width,
                        frame_height=height
                    )
            except Exception as e:
                logger.error(f"Process isolation failed: {e}")
                # Continue with singleton approach as fallback
        
        # Use singleton pattern (original approach)
        try:
            # Get landmarker using singleton pattern
            landmarker = self._ensure_landmarker(model_path)
            
            # Increment frame counter in singleton for memory management
            from ambient.pose.mediapipe_singleton import get_mediapipe_singleton
            singleton = get_mediapipe_singleton()
            singleton.increment_frame_count()
            
            # Convert to MediaPipe image format
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image)
            
            # Detect pose with file descriptor level warning suppression
            if self.suppress_warnings:
                with suppress_stderr_fd():
                    detection_result = landmarker.detect(mp_image)
            else:
                detection_result = landmarker.detect(mp_image)
            
            # Extract keypoints
            height, width = image.shape[:2]
            
            if not detection_result.pose_landmarks:
                # Return empty result with CUSTOM format (no keypoints detected)
                return KeypointSet(
                    keypoints=[],
                    format=KeypointFormat.CUSTOM,
                    frame_width=width,
                    frame_height=height
                )
            
            pose_landmarks = detection_result.pose_landmarks[0]
            
            # Return KeypointSet structure
            return KeypointSet.from_mediapipe(
                landmarks=pose_landmarks,
                frame_width=width,
                frame_height=height,
                landmark_names=POSE_LANDMARK_NAMES
            )
            
        except RuntimeError as e:
            # Check if this is a threading issue that should trigger process isolation
            if "Threading failures detected" in str(e) or "WinError 1" in str(e):
                logger.info(f"Switching to process isolation due to threading issues")
                try:
                    process_extractor = self._get_process_extractor()
                    result = process_extractor.extract_from_image(image)
                    if result is not None:
                        return result
                except Exception as proc_e:
                    logger.error(f"Process isolation also failed: {proc_e}")
            
            # If MediaPipe fails, reset the singleton and try once more
            logger.warning(f"MediaPipe detection failed, resetting singleton: {e}")
            from ambient.pose.mediapipe_singleton import get_mediapipe_singleton
            singleton = get_mediapipe_singleton()
            singleton.reset_landmarker()
            
            # Try once more with fresh landmarker
            try:
                landmarker = self._ensure_landmarker(model_path)
                mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image)
                
                if self.suppress_warnings:
                    with suppress_stderr_fd():
                        detection_result = landmarker.detect(mp_image)
                else:
                    detection_result = landmarker.detect(mp_image)
                
                height, width = image.shape[:2]
                
                if not detection_result.pose_landmarks:
                    return KeypointSet(
                        keypoints=[],
                        format=KeypointFormat.CUSTOM,
                        frame_width=width,
                        frame_height=height
                    )
                
                pose_landmarks = detection_result.pose_landmarks[0]
                
                return KeypointSet.from_mediapipe(
                    landmarks=pose_landmarks,
                    frame_width=width,
                    frame_height=height,
                    landmark_names=POSE_LANDMARK_NAMES
                )
                
            except Exception as retry_e:
                # If retry also fails, return empty result
                logger.error(f"MediaPipe retry also failed: {retry_e}")
                height, width = image.shape[:2]
                return KeypointSet(
                    keypoints=[],
                    format=KeypointFormat.CUSTOM,
                    frame_width=width,
                    frame_height=height
                )
        
        except Exception as e:
            # Handle any other unexpected errors
            logger.error(f"Unexpected error in keypoint extraction: {e}")
            height, width = image.shape[:2]
            return KeypointSet(
                keypoints=[],
                format=KeypointFormat.CUSTOM,
                frame_width=width,
                frame_height=height
            )
    
    def extract_from_frame_file(
        self,
        image_path: str,
        model_path: Optional[str] = None
    ) -> KeypointSet:
        """
        Extract pose keypoints from an image file.
        
        Args:
            image_path: Path to image file
            model_path: Optional path to model file
            
        Returns:
            KeypointSet object
            
        Raises:
            ValueError: If image file cannot be read
            
        Example:
            >>> extractor = SequenceKeypointExtractor()
            >>> keypoints = extractor.extract_from_frame_file("person.jpg")
        """
        # Read image
        image = cv2.imread(str(image_path))
        if image is None:
            raise ValueError(f"Failed to read image from {image_path}")
        
        # Convert BGR to RGB
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        return self.extract_from_image(image_rgb, model_path)
    
    def extract_from_video_frame(
        self,
        video_path: Path,
        frame_number: int,
        model_path: Optional[str] = None
    ) -> Optional[KeypointSet]:
        """
        Extract pose keypoints from a specific video frame.
        
        Uses Windows-safe FFmpeg extraction with proper error handling and
        automatic fallback to OpenCV when needed.
        
        Args:
            video_path: Path to video file
            frame_number: Frame number to extract (1-based)
            model_path: Optional path to model file
            
        Returns:
            KeypointSet object, or None if extraction failed
            
        Example:
            >>> extractor = SequenceKeypointExtractor()
            >>> keypoints = extractor.extract_from_video_frame(
            ...     Path("video.mp4"), frame_number=10
            ... )
        """
        try:
            # Import here to avoid circular imports
            from ambient.pose.windows_ffmpeg_handler import WindowsVideoFrameExtractor
            
            # Create a fresh extractor instance for each call to avoid state issues
            frame_extractor = WindowsVideoFrameExtractor(
                prefer_ffmpeg=True,
                ffmpeg_timeout=30,
                verbose=False  # Suppress debug logs for cleaner output
            )
            
            # Extract frame (returns BGR format)
            frame = frame_extractor.extract_frame(video_path, frame_number)
            
            if frame is None:
                logger.warning(f"Failed to extract frame {frame_number} from {video_path}")
                return None
            
            # Validate frame data
            if frame.size == 0 or frame.shape[0] == 0 or frame.shape[1] == 0:
                logger.warning(f"Invalid frame data for frame {frame_number}")
                return None
            
            # Convert BGR to RGB for MediaPipe with error handling
            try:
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            except Exception as e:
                logger.error(f"Color conversion failed for frame {frame_number}: {e}")
                return None
            
            # Extract keypoints using MediaPipe with retry logic
            max_retries = 2
            for attempt in range(max_retries):
                try:
                    result_kp = self.extract_from_image(frame_rgb, model_path)
                    
                    # Add frame number as timestamp for tracking
                    result_kp.timestamp = float(frame_number)
                    
                    return result_kp
                    
                except Exception as e:
                    if attempt < max_retries - 1:
                        logger.warning(f"MediaPipe extraction failed for frame {frame_number}, attempt {attempt + 1}: {e}")
                        # Reset landmarker state for retry
                        self._landmarker = None
                        continue
                    else:
                        logger.error(f"MediaPipe extraction failed for frame {frame_number} after {max_retries} attempts: {e}")
                        return None
                    
        except Exception as e:
            logger.error(f"Frame extraction failed for frame {frame_number}: {e}")
            # Try fallback to OpenCV as last resort
            try:
                return self._extract_from_video_frame_opencv(video_path, frame_number, model_path)
            except Exception as fallback_e:
                logger.error(f"OpenCV fallback also failed for frame {frame_number}: {fallback_e}")
                return None
    
    def _extract_from_video_frame_opencv(
        self,
        video_path: Path,
        frame_number: int,
        model_path: Optional[str] = None
    ) -> Optional[KeypointSet]:
        """
        Fallback method using OpenCV for frame extraction.
        
        This may have issues on Windows with certain codecs.
        """
        # Open video
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            logger.error(f"Cannot open video: {video_path}")
            return None
        
        try:
            # Seek to frame (convert to 0-based)
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number - 1)
            ret, frame = cap.read()
            
            if not ret:
                logger.error(f"Could not read frame {frame_number}")
                return None
            
            # Convert BGR to RGB
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Extract keypoints
            result = self.extract_from_image(frame_rgb, model_path)
            
            # Add frame number as timestamp
            result.timestamp = float(frame_number)
            
            return result
            
        finally:
            cap.release()
    
    def extract_from_sequence(
        self,
        sequence_data: pd.DataFrame,
        video_base_path: Path,
        model_path: Optional[str] = None,
        verbose: bool = False
    ) -> List[KeypointSet]:
        """
        Extract pose keypoints from a sequence of video frames.
        
        This method processes a DataFrame containing frame information
        and extracts keypoints from each frame in the sequence.
        
        Args:
            sequence_data: DataFrame with columns 'frame_num' and 'url'
            video_base_path: Base path where videos are stored
            model_path: Optional path to model file
            verbose: Whether to print progress information
            
        Returns:
            List of KeypointSet objects
            
        Example:
            >>> import pandas as pd
            >>> extractor = SequenceKeypointExtractor()
            >>> sequence_data = pd.DataFrame({
            ...     'frame_num': [1, 2, 3],
            ...     'url': ['http://youtube.com/watch?v=abc'] * 3,
            ...     'seq': ['seq1'] * 3
            ... })
            >>> keypoints = extractor.extract_from_sequence(
            ...     sequence_data, Path("data/youtube")
            ... )
        """
        if sequence_data.empty:
            logger.error("Empty sequence data provided")
            return []
        
        # Ensure landmarker is available
        self._ensure_landmarker(model_path)
        
        sequence_id = sequence_data['seq'].iloc[0] if 'seq' in sequence_data.columns else "unknown"
        num_frames = len(sequence_data)
        
        if verbose:
            logger.info(f"Processing sequence: {sequence_id}")
            logger.info(f"Number of frames: {num_frames}")
        
        keypoints_array = []
        
        for fnum in range(num_frames):
            frame_row = sequence_data.iloc[fnum]
            actual_frame_num = int(frame_row['frame_num'])
            
            try:
                if verbose and fnum % 10 == 0:  # Log every 10th frame to reduce noise
                    # logger.bind(plain=True).debug(f"\tframe {actual_frame_num} ({fnum+1}/{num_frames})")
                    print(f"\tframe {actual_frame_num} ({fnum+1}/{num_frames})")
                
                # Get video path
                from ambient.utils.youtube_cache import extract_video_id
                url = frame_row['url']
                video_id = extract_video_id(url)
                video_path = video_base_path / f"{video_id}.mp4"
                
                if not video_path.exists():
                    logger.error(f"Video not found: {video_path}")
                    return []
                
                # Extract keypoints from this frame
                keypoints = self.extract_from_video_frame(
                    video_path,
                    actual_frame_num,
                    model_path
                )
                
                if keypoints is None:
                    logger.error(f"Failed to extract keypoints from frame {actual_frame_num}")
                    return []
                
                keypoints_array.append(keypoints)
            except Exception as e:
                logger.warning(f"Frame {fnum} of {sequence_id} failed to be processed")
        
        if verbose:
            logger.info("Sequence processing complete")
        
        return keypoints_array
