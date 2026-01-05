"""
Pose Estimation Utilities for GAVD Data Processing

Utilities for bounding box handling and simple placeholder keypoint generation.
Kept as a lightweight dependency to avoid circular imports.
"""

from typing import Dict, List, Optional, Tuple, Union


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

        # Placeholder generator (kept for fallback compatibility)
        keypoints = self.keypoint_generator.generate_grid_keypoints(
            center_x, center_y, num_keypoints, grid_spacing, confidence
        )

        return keypoints
