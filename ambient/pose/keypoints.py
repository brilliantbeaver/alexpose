"""
Pose Keypoint Utilities

Comprehensive utilities for pose keypoint extraction, processing, and analysis.
This module provides:
- Bounding box operations
- Keypoint generation
- Keypoint extraction from bounding boxes
- Visualization utilities

The module follows SOLID principles with clear separation of concerns:
- BoundingBoxProcessor: Handles bounding box calculations
- KeypointGenerator: Generates synthetic keypoints
- PoseKeypointExtractor: Extracts keypoints from bounding boxes
- KeypointVisualizer: Visualizes pose detection results

For model management, see ambient.pose.model_management
For video sequence extraction, see ambient.pose.keypoint_extractor

Note: This module uses the Keypoint and KeypointSet data structures
from ambient.pose.keypoint_data for improved type safety and extensibility.
"""

import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import pandas as pd

# Suppress all warnings at the Python level
warnings.filterwarnings('ignore')

import cv2
import numpy as np

# Import data structures
from ambient.pose.keypoint_data import (
    Keypoint,
    KeypointSet,
    KeypointFormat,
    KeypointSchema,
    MEDIAPIPE_33_NAMES,
    get_schema,
)

# Import model management (for backward compatibility)
from ambient.pose.model_management import (
    MediaPipeModelManager,
    PoseLandmarkerFactory,
    MEDIAPIPE_AVAILABLE,
)

# MediaPipe pose landmark names (33 landmarks for BLAZEPOSE_33 format)
POSE_LANDMARK_NAMES = MEDIAPIPE_33_NAMES


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
    Extracts pose keypoints from bounding box regions in images.

    This class uses MediaPipe pose estimation to extract real pose keypoints
    from image regions defined by bounding boxes, rather than generating
    synthetic placeholder keypoints.
    """

    def __init__(
        self,
        sequence_extractor: Optional["SequenceKeypointExtractor"] = None,
        model_path: Optional[str] = None,
    ):
        """
        Initialize the pose keypoint extractor.

        Args:
            sequence_extractor: SequenceKeypointExtractor instance for pose detection
            model_path: Optional path to MediaPipe model file
        """
        if sequence_extractor is None:
            # Import here to avoid circular dependency at module level
            from ambient.pose.keypoint_extractor import SequenceKeypointExtractor
            sequence_extractor = SequenceKeypointExtractor()
        self.sequence_extractor = sequence_extractor
        self.model_path = model_path

    def extract_from_bbox(
        self,
        image: np.ndarray,
        bbox: Dict[str, Union[int, float]],
        model_path: Optional[str] = None,
    ) -> KeypointSet:
        """
        Extract pose keypoints from a bounding box region in an image.

        This method crops the image to the bounding box, runs pose estimation
        on the cropped region, and transforms the coordinates back to the
        original image space.

        Args:
            image: RGB image array (height, width, 3)
            bbox: Bounding box dictionary with 'left', 'top', 'width', 'height'
            model_path: Optional path to model file (uses instance default if None)

        Returns:
            KeypointSet object with keypoints in original image coordinates

        Raises:
            ValueError: If bbox is invalid or None
        """
        if not bbox or not isinstance(bbox, dict):
            raise ValueError("Bounding box must be a non-empty dictionary")

        # Extract bbox coordinates
        left = int(bbox.get("left", 0))
        top = int(bbox.get("top", 0))
        width = int(bbox.get("width", 0))
        height = int(bbox.get("height", 0))

        # Validate bbox is within image bounds
        img_height, img_width = image.shape[:2]
        left = max(0, min(left, img_width - 1))
        top = max(0, min(top, img_height - 1))
        right = min(left + width, img_width)
        bottom = min(top + height, img_height)

        # Crop image to bounding box
        cropped_image = image[top:bottom, left:right]

        if cropped_image.size == 0:
            # Return empty keypoint set if crop is invalid
            return KeypointSet(
                keypoints=[],
                format=KeypointFormat.MEDIAPIPE_33,
                frame_width=img_width,
                frame_height=img_height
            )

        # Extract keypoints from cropped region
        use_model_path = model_path or self.model_path
        keypoint_set = self.sequence_extractor.extract_from_image(
            cropped_image, use_model_path
        )

        # Transform keypoint coordinates back to original image space
        for keypoint in keypoint_set.keypoints:
            keypoint.x += left
            keypoint.y += top

        # Update frame dimensions to match original image
        keypoint_set.frame_width = img_width
        keypoint_set.frame_height = img_height

        return keypoint_set


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


def get_keypoints(
    project_root: Path,
    sequence_data: pd.DataFrame,
    model_path: Optional[str] = None,
    verbose: bool = True
) -> List[KeypointSet]:
    """
    Convenience function to extract keypoints from a sequence.
    
    This is a simplified interface that wraps SequenceKeypointExtractor
    for backward compatibility and ease of use.
    
    Args:
        project_root: Project root directory
        sequence_data: DataFrame with columns 'frame_num', 'url', 'seq'
        model_path: Optional path to model file
        verbose: Whether to print progress information
        
    Returns:
        List of KeypointSet objects
        
    Raises:
        TypeError: If sequence_data is not a DataFrame
        
    Example:
        >>> import pandas as pd
        >>> from pathlib import Path
        >>> sequence_df = pd.DataFrame({
        ...     'frame_num': [1, 2, 3],
        ...     'url': ['http://youtube.com/watch?v=abc'] * 3,
        ...     'seq': ['seq1'] * 3
        ... })
        >>> keypoints = get_keypoints(Path.cwd(), sequence_df)
    """
    # Validate input type
    if not isinstance(sequence_data, pd.DataFrame):
        raise TypeError(
            f"sequence_data must be pd.DataFrame, got {type(sequence_data).__name__}"
        )
    
    # Import here to avoid circular dependency
    from ambient.pose.keypoint_extractor import SequenceKeypointExtractor
    
    # Create extractor
    extractor = SequenceKeypointExtractor()
    
    # Extract keypoints
    video_base_path = project_root / "data" / "youtube"
    return extractor.extract_from_sequence(
        sequence_data,
        video_base_path,
        model_path,
        verbose
    )
