"""
Pose Keypoint Utilities

Comprehensive utilities for pose keypoint extraction, processing, and analysis.
This module provides:
- Bounding box operations
- MediaPipe pose estimation integration
- Keypoint extraction from video sequences
- Model management and downloading
- Batch processing capabilities

The module follows SOLID principles with clear separation of concerns:
- BoundingBoxProcessor: Handles bounding box calculations
- KeypointGenerator: Generates synthetic keypoints
- PoseKeypointExtractor: Extracts keypoints from bounding boxes
- MediaPipeModelManager: Manages MediaPipe model downloads
- PoseLandmarkerFactory: Creates MediaPipe pose landmarkers
- SequenceKeypointExtractor: Extracts keypoints from video sequences
- KeypointVisualizer: Visualizes pose detection results

Note: This module now uses the new Keypoint and KeypointSet data structures
from ambient.pose.keypoint_data for improved type safety and extensibility.
"""

import os
import sys
import tempfile
import urllib.request
import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import warnings
import contextlib

# Suppress all warnings at the Python level
warnings.filterwarnings('ignore')

import cv2
import numpy as np
import pandas as pd

# Import new data structures
from ambient.pose.keypoint_data import (
    Keypoint,
    KeypointSet,
    KeypointFormat,
    KeypointSchema,
    MEDIAPIPE_33_NAMES,
    get_schema,
)

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


# ============================================================================
# Low-level stderr suppression utilities
# ============================================================================

@contextlib.contextmanager
def suppress_stderr_fd():
    """
    Suppress stderr at the file descriptor level.
    
    This is the most aggressive form of stderr suppression and will catch
    C++ level output from TensorFlow Lite, MediaPipe, and GLOG that cannot
    be suppressed through Python's warnings module or sys.stderr redirection.
    
    Works on both Unix and Windows systems, and is compatible with Jupyter notebooks.
    
    Example:
        with suppress_stderr_fd():
            # C++ warnings are completely suppressed
            import mediapipe as mp
            landmarker = mp.solutions.pose.Pose()
    """
    import io
    
    # Check if we're in a Jupyter notebook or if stderr doesn't have a file descriptor
    try:
        stderr_fd = sys.stderr.fileno()
    except (AttributeError, io.UnsupportedOperation):
        # We're in Jupyter or another environment without real file descriptors
        # Fall back to Python-level suppression only
        old_stderr = sys.stderr
        try:
            sys.stderr = open(os.devnull, 'w')
            yield
        finally:
            sys.stderr.close()
            sys.stderr = old_stderr
        return
    
    # Normal file descriptor-based suppression for regular Python environments
    # Duplicate the stderr file descriptor to restore it later
    with os.fdopen(os.dup(stderr_fd), 'wb') as copied:
        # Flush any pending output
        sys.stderr.flush()
        
        try:
            # Determine the null device for this platform
            null_file = 'NUL' if sys.platform == 'win32' else '/dev/null'
            
            # Redirect stderr to the null device
            with open(null_file, 'wb') as devnull:
                os.dup2(devnull.fileno(), stderr_fd)
            
            yield
            
        finally:
            # Restore the original stderr
            sys.stderr.flush()
            os.dup2(copied.fileno(), stderr_fd)


class BoundingBoxProcessor:
    """
    Handles bounding box calculations and center point extraction.

    This class follows the Single Responsibility Principle by focusing
    solely on bounding box operations.
    """

    @staticmethod
    def calculate_center(bbox: Dict[str, Union[int, float]]) -> Tuple[float, float]:
        """
        Calculate the center point of a bounding box.

        Args:
            bbox (Dict[str, Union[int, float]]): Bounding box with 'left', 'top', 'width', 'height'

        Returns:
            Tuple[float, float]: (center_x, center_y) coordinates

        Raises:
            ValueError: If bbox is None or missing required keys
        """
        if not bbox or not isinstance(bbox, dict):
            raise ValueError("Bounding box must be a non-empty dictionary")

        left = bbox.get("left", 0)
        top = bbox.get("top", 0)
        width = bbox.get("width", 0)
        height = bbox.get("height", 0)

        center_x = left + width / 2
        center_y = top + height / 2

        return center_x, center_y


class KeypointGenerator:
    """
    Generates pose keypoints using various strategies.

    This class follows the Open/Closed Principle by allowing different
    keypoint generation strategies while keeping the interface stable.
    """

    @staticmethod
    def create_keypoint(
        x: float, y: float, confidence: float = 0.8
    ) -> Dict[str, Union[float, int]]:
        """
        Create a single keypoint with specified coordinates and confidence.

        Args:
            x (float): X coordinate
            y (float): Y coordinate
            confidence (float): Confidence score (0.0 to 1.0)

        Returns:
            Dict[str, Union[float, int]]: Keypoint dictionary
        """
        return {
            "x": x,
            "y": y,
            "confidence": max(0.0, min(1.0, confidence)),  # Clamp to valid range
        }

    @staticmethod
    def generate_grid_keypoints(
        center_x: float,
        center_y: float,
        num_keypoints: int = 25,
        grid_spacing: float = 5.0,
        confidence: float = 0.8,
    ) -> List[Dict[str, Union[float, int]]]:
        """
        Generate keypoints in a grid pattern around a center point.

        Args:
            center_x (float): Center X coordinate
            center_y (float): Center Y coordinate
            num_keypoints (int): Number of keypoints to generate
            grid_spacing (float): Spacing between keypoints in the grid
            confidence (float): Confidence score for all keypoints

        Returns:
            List[Dict[str, Union[float, int]]]: List of keypoint dictionaries
        """
        keypoints = []
        grid_size = int(num_keypoints**0.5)  # Calculate grid dimensions

        for i in range(num_keypoints):
            # Calculate grid position
            grid_x = i % grid_size
            grid_y = i // grid_size

            # Calculate offset from center
            offset_x = (grid_x - grid_size // 2) * grid_spacing
            offset_y = (grid_y - grid_size // 2) * grid_spacing

            # Create keypoint
            keypoint = KeypointGenerator.create_keypoint(
                center_x + offset_x, center_y + offset_y, confidence
            )
            keypoints.append(keypoint)

        return keypoints


class PoseKeypointExtractor:
    """
    Extracts pose keypoints from bounding box data using configurable strategies.

    This class follows the Dependency Inversion Principle by depending on
    abstractions (BoundingBoxProcessor and KeypointGenerator) rather than
    concrete implementations.
    """

    def __init__(
        self,
        bbox_processor: Optional[BoundingBoxProcessor] = None,
        keypoint_generator: Optional[KeypointGenerator] = None,
    ):
        """
        Initialize the pose keypoint extractor.

        Args:
            bbox_processor (Optional[BoundingBoxProcessor]): Bounding box processor
            keypoint_generator (Optional[KeypointGenerator]): Keypoint generator
        """
        self.bbox_processor = bbox_processor or BoundingBoxProcessor()
        self.keypoint_generator = keypoint_generator or KeypointGenerator()

    def extract_from_bbox(
        self,
        bbox: Dict[str, Union[int, float]],
        num_keypoints: int = 25,
        grid_spacing: float = 5.0,
        confidence: float = 0.8,
    ) -> List[Dict[str, Union[float, int]]]:
        """
        Extract pose keypoints from bounding box data.

        Args:
            bbox (Dict[str, Union[int, float]]): Bounding box dictionary
            num_keypoints (int): Number of keypoints to generate
            grid_spacing (float): Spacing between keypoints in grid
            confidence (float): Confidence score for keypoints

        Returns:
            List[Dict[str, Union[float, int]]]: List of keypoint dictionaries

        Raises:
            ValueError: If bbox is invalid or None
        """
        if not bbox or not isinstance(bbox, dict):
            raise ValueError("Bounding box must be a non-empty dictionary")

        # Calculate center using the bbox processor
        center_x, center_y = self.bbox_processor.calculate_center(bbox)

        # Generate keypoints using the keypoint generator
        keypoints = self.keypoint_generator.generate_grid_keypoints(
            center_x, center_y, num_keypoints, grid_spacing, confidence
        )

        return keypoints


# MediaPipe pose landmark names (33 landmarks for BLAZEPOSE_33 format)
# Now imported from keypoint_data module
POSE_LANDMARK_NAMES = MEDIAPIPE_33_NAMES


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
            
            # Create the landmarker
            landmarker = vision.PoseLandmarker.create_from_options(options)
            print(f"[OK] Pose Landmarker created from {model_path}")
            return landmarker
            
        except Exception as e:
            print(f"[ERROR] Failed to create Pose Landmarker: {e}")
            return None


class SequenceKeypointExtractor:
    """
    Extracts pose keypoints from video sequences.
    
    This class handles the extraction of pose keypoints from video frames,
    supporting both single frames and batch processing of sequences.
    It follows the Single Responsibility Principle by focusing on
    keypoint extraction from video data.
    """
    
    def __init__(
        self,
        model_manager: Optional[MediaPipeModelManager] = None,
        landmarker_factory: Optional[PoseLandmarkerFactory] = None,
        suppress_warnings: bool = True
    ):
        """
        Initialize the sequence keypoint extractor.
        
        Args:
            model_manager: Model manager instance. Creates default if None
            landmarker_factory: Landmarker factory instance. Uses default if None
            suppress_warnings: Whether to suppress MediaPipe warnings during extraction
        """
        self.model_manager = model_manager or MediaPipeModelManager()
        self.landmarker_factory = landmarker_factory or PoseLandmarkerFactory()
        self.suppress_warnings = suppress_warnings
        self._landmarker = None
        self._model_path = None
    
    def _ensure_landmarker(self, model_path: Optional[str] = None):
        """Ensure a landmarker is available, creating if necessary."""
        if model_path is None:
            # Ensure model is downloaded
            model_path = self.model_manager.ensure_model_available()
            if model_path is None:
                raise RuntimeError("Failed to download or locate pose model")
        
        # Create landmarker if needed or if model path changed
        if self._landmarker is None or self._model_path != model_path:
            if self.suppress_warnings:
                # Use file descriptor level suppression for C++ warnings
                with suppress_stderr_fd():
                    self._landmarker = self.landmarker_factory.create_landmarker(model_path)
            else:
                self._landmarker = self.landmarker_factory.create_landmarker(model_path)
            
            self._model_path = model_path
            
            if self._landmarker is None:
                raise RuntimeError(f"Failed to create landmarker from {model_path}")
        
        return self._landmarker
    
    def extract_from_image(
        self,
        image: np.ndarray,
        model_path: Optional[str] = None
    ) -> KeypointSet:
        """
        Extract pose keypoints from a single image.
        
        Args:
            image: RGB image array (height, width, 3)
            model_path: Optional path to model file
            
        Returns:
            KeypointSet object with:
            - keypoints: List of Keypoint objects with full metadata
            - format: KeypointFormat.MEDIAPIPE_33
            - frame_width, frame_height: Image dimensions
        """
        if not MEDIAPIPE_AVAILABLE:
            raise ImportError("MediaPipe is required for pose extraction")
        
        # Ensure landmarker is available
        landmarker = self._ensure_landmarker(model_path)
        
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
            # Return empty result
            return KeypointSet(
                keypoints=[],
                format=KeypointFormat.MEDIAPIPE_33,
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
        
        Args:
            video_path: Path to video file
            frame_number: Frame number to extract (1-based)
            model_path: Optional path to model file
            
        Returns:
            KeypointSet object, or None if extraction failed
        """
        # Open video
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            print(f"[ERROR] Cannot open video: {video_path}")
            return None
        
        try:
            # Seek to frame (convert to 0-based)
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number - 1)
            ret, frame = cap.read()
            
            if not ret:
                print(f"[ERROR] Could not read frame {frame_number}")
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
        verbose: bool = True
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
        """
        if sequence_data.empty:
            print("[ERROR] Empty sequence data provided")
            return []
        
        # Ensure landmarker is available
        self._ensure_landmarker(model_path)
        
        sequence_id = sequence_data['seq'].iloc[0] if 'seq' in sequence_data.columns else "unknown"
        num_frames = len(sequence_data)
        
        if verbose:
            print(f"Processing sequence: {sequence_id}")
            print(f"Number of frames: {num_frames}")
        
        keypoints_array = []
        
        try:
            for fnum in range(num_frames):
                frame_row = sequence_data.iloc[fnum]
                actual_frame_num = int(frame_row['frame_num'])
                
                if verbose:
                    print(f"{actual_frame_num}", end=" ", flush=True)
                
                # Get video path
                from ambient.utils.youtube_cache import extract_video_id
                url = frame_row['url']
                video_id = extract_video_id(url)
                video_path = video_base_path / f"{video_id}.mp4"
                
                if not video_path.exists():
                    print(f"\n[ERROR] Video not found: {video_path}")
                    return []
                
                # Extract keypoints from this frame
                keypoints = self.extract_from_video_frame(
                    video_path,
                    actual_frame_num,
                    model_path
                )
                
                if keypoints is None:
                    print(f"\n[ERROR] Failed to extract keypoints from frame {actual_frame_num}")
                    return []
                
                keypoints_array.append(keypoints)
            
            if verbose:
                print("\n[OK] Sequence processing complete")
            
            return keypoints_array
            
        except Exception as e:
            print(f"\n[ERROR] Sequence processing failed: {e}")
            return []


class KeypointVisualizer:
    """
    Visualizes pose keypoints on images.
    
    This class provides visualization capabilities for pose detection results,
    following the Single Responsibility Principle by focusing on visualization.
    """
    
    @staticmethod
    def draw_keypoints(
        image: np.ndarray,
        keypoint_set: KeypointSet,
        confidence_threshold: float = 0.5,
        color: Tuple[int, int, int] = (255, 0, 0),
        radius: int = 5
    ) -> np.ndarray:
        """
        Draw keypoints on an image.
        
        Args:
            image: RGB image array
            keypoint_set: KeypointSet object
            confidence_threshold: Minimum confidence to draw
            color: RGB color tuple
            radius: Circle radius for keypoints
            
        Returns:
            Image with keypoints drawn
        """
        annotated = image.copy()
        
        for kp in keypoint_set.keypoints:
            if kp.confidence > confidence_threshold:
                x, y = int(kp.x), int(kp.y)
                cv2.circle(annotated, (x, y), radius, color, -1)
        
        return annotated
    
    @staticmethod
    def draw_skeleton(
        image: np.ndarray,
        keypoint_set: KeypointSet,
        confidence_threshold: float = 0.5,
        keypoint_color: Tuple[int, int, int] = (255, 0, 0),
        line_color: Tuple[int, int, int] = (0, 255, 0),
        radius: int = 5,
        thickness: int = 2
    ) -> np.ndarray:
        """
        Draw keypoints and skeleton connections on an image.
        
        Args:
            image: RGB image array
            keypoint_set: KeypointSet object
            confidence_threshold: Minimum confidence to draw
            keypoint_color: RGB color tuple for keypoints
            line_color: RGB color tuple for skeleton lines
            radius: Circle radius for keypoints
            thickness: Line thickness for skeleton
            
        Returns:
            Image with keypoints and skeleton drawn
        """
        annotated = image.copy()
        
        # Get schema for connections
        schema = get_schema(keypoint_set.format)
        
        # Draw skeleton connections first (so they appear behind keypoints)
        if schema and schema.connections:
            for start_idx, end_idx in schema.connections:
                if start_idx < len(keypoint_set) and end_idx < len(keypoint_set):
                    kp_start = keypoint_set[start_idx]
                    kp_end = keypoint_set[end_idx]
                    
                    # Only draw if both keypoints are confident
                    if kp_start.confidence > confidence_threshold and kp_end.confidence > confidence_threshold:
                        pt1 = (int(kp_start.x), int(kp_start.y))
                        pt2 = (int(kp_end.x), int(kp_end.y))
                        cv2.line(annotated, pt1, pt2, line_color, thickness)
        
        # Draw keypoints on top
        for kp in keypoint_set.keypoints:
            if kp.confidence > confidence_threshold:
                x, y = int(kp.x), int(kp.y)
                cv2.circle(annotated, (x, y), radius, keypoint_color, -1)
        
        return annotated
    
    @staticmethod
    def get_summary_stats(keypoint_set: KeypointSet) -> Dict[str, Union[int, float]]:
        """
        Get summary statistics for a keypoint set.
        
        Args:
            keypoint_set: KeypointSet object
            
        Returns:
            Dictionary with statistics
        """
        if not keypoint_set.keypoints:
            return {
                'total_landmarks': 0,
                'visible_landmarks': 0,
                'reliable_landmarks': 0,
                'avg_confidence': 0.0,
                'avg_visibility': 0.0,
                'detection_quality': 0.0
            }
        
        return {
            'total_landmarks': len(keypoint_set),
            'visible_landmarks': len(keypoint_set.visible_keypoints),
            'reliable_landmarks': len(keypoint_set.reliable_keypoints),
            'avg_confidence': keypoint_set.avg_confidence,
            'avg_visibility': keypoint_set.avg_visibility,
            'detection_quality': keypoint_set.detection_quality,
            'format': keypoint_set.format.value
        }


# Convenience functions for backward compatibility and ease of use

def ensure_model_downloaded(project_root: Path) -> Optional[str]:
    """
    Convenience function to ensure MediaPipe model is downloaded.
    
    Args:
        project_root: Project root directory
        
    Returns:
        Path to model file, or None if download failed
    """
    models_dir = project_root / "data" / "models"
    manager = MediaPipeModelManager(models_dir)
    return manager.ensure_model_available()


def create_pose_landmarker(model_path: str):
    """
    Convenience function to create a pose landmarker.
    
    Args:
        model_path: Path to model file
        
    Returns:
        PoseLandmarker instance or None
    """
    factory = PoseLandmarkerFactory()
    return factory.create_landmarker(model_path)
