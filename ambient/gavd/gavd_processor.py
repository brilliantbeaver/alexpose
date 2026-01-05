"""
GAVD Data Loader Module

This module provides a comprehensive data loading system for GAVD (Gait Abnormality Video Dataset)
data with emphasis on modularity, extensibility, and maintainability.

Key Features:
- Flexible CSV data loading with dictionary field parsing
- Sequence-based data organization
- Configurable data validation and preprocessing
- Extensible data transformation pipeline
- Comprehensive error handling and logging

Design Principles:
- Single Responsibility Principle: Each class has one clear purpose
- Open/Closed Principle: Extensible through inheritance and composition
- Dependency Inversion: Depends on abstractions, not concrete implementations
- Interface Segregation: Small, focused interfaces
- Liskov Substitution: Subclasses can replace base classes

Author: Theodore Mui
Date: 2025-07-26
"""

from loguru import logger as loguru_logger
import os
import sys
import subprocess
import tempfile
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Union

import pandas as pd

# Add the theodore directory to the Python path for imports
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from ambient.gavd.keypoints import BoundingBoxProcessor, KeypointGenerator
from ambient.gavd.pose_estimators import PoseEstimator
from ambient.utils.youtube_cache import extract_video_id
from ambient.utils.csv_parser import parse_csv_with_dicts
from ambient.utils.youtube_cache import cache_youtube_videos_from_rows


class DataValidator(ABC):
    """
    Abstract base class for data validation strategies.

    This follows the Strategy pattern, allowing different validation
    approaches to be plugged in without changing the main loader.
    """

    @abstractmethod
    def validate(self, data: Any) -> bool:
        """
        Validate the given data.

        Args:
            data: Data to validate

        Returns:
            bool: True if valid, False otherwise
        """
        pass

    @abstractmethod
    def get_error_message(self) -> str:
        """
        Get error message for failed validation.

        Returns:
            str: Error message
        """
        pass


class GAVDDataValidator(DataValidator):
    """
    Validator specifically for GAVD data format.

    Ensures data meets GAVD-specific requirements and constraints.
    """

    def __init__(self):
        self.last_error = ""
        self.required_columns = ["seq", "frame_num", "bbox"]
        self.optional_columns = ["vid_info", "person_id", "confidence"]

    def validate(self, data: pd.DataFrame) -> bool:
        """
        Validate GAVD DataFrame format and content.

        Args:
            data (pd.DataFrame): DataFrame to validate

        Returns:
            bool: True if valid, False otherwise
        """
        if data is None or data.empty:
            self.last_error = "DataFrame is None or empty"
            return False

        # Check required columns
        missing_columns = [
            col for col in self.required_columns if col not in data.columns
        ]
        if missing_columns:
            self.last_error = f"Missing required columns: {missing_columns}"
            return False

        # Check for valid sequence IDs
        if data["seq"].isna().any():
            self.last_error = "Found null values in 'seq' column"
            return False

        # Check for valid frame numbers
        if "frame_num" in data.columns and data["frame_num"].isna().any():
            self.last_error = "Found null values in 'frame_num' column"
            return False

        return True

    def get_error_message(self) -> str:
        """Get the last validation error message."""
        return self.last_error


class DataTransformer(ABC):
    """
    Abstract base class for data transformation strategies.

    Allows different transformation approaches to be applied
    to the loaded data without modifying the main loader.
    """

    @abstractmethod
    def transform(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Transform the given DataFrame.

        Args:
            data (pd.DataFrame): Input DataFrame

        Returns:
            pd.DataFrame: Transformed DataFrame
        """
        pass


class GAVDDataTransformer(DataTransformer):
    """
    Transformer for GAVD-specific data preprocessing.

    Handles common transformations like frame number conversion,
    data type casting, and column normalization.
    """

    def __init__(
        self,
        convert_frame_num: bool = True,
        normalize_bbox: bool = False,
        add_metadata: bool = True,
    ):
        """
        Initialize the transformer with configuration options.

        Args:
            convert_frame_num (bool): Whether to convert frame_num to numeric
            normalize_bbox (bool): Whether to normalize bounding box coordinates
            add_metadata (bool): Whether to add processing metadata
        """
        self.convert_frame_num = convert_frame_num
        self.normalize_bbox = normalize_bbox
        self.add_metadata = add_metadata

    def transform(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Apply transformations to GAVD DataFrame.

        Args:
            data (pd.DataFrame): Input GAVD DataFrame

        Returns:
            pd.DataFrame: Transformed DataFrame
        """
        df = data.copy()

        # Convert frame_num to numeric type for proper sorting
        if self.convert_frame_num and "frame_num" in df.columns:
            df["frame_num"] = pd.to_numeric(df["frame_num"], errors="coerce")

        # Normalize bounding box coordinates if requested
        if self.normalize_bbox and "bbox" in df.columns:
            df = self._normalize_bbox_coordinates(df)

        # Add processing metadata if requested
        if self.add_metadata:
            df = self._add_processing_metadata(df)

        return df

    def _normalize_bbox_coordinates(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Normalize bounding box coordinates to 0-1 range.

        Args:
            df (pd.DataFrame): DataFrame with bbox column

        Returns:
            pd.DataFrame: DataFrame with normalized bbox coordinates
        """
        # This is a placeholder for bbox normalization logic
        # Implementation would depend on specific requirements
        return df

    def _add_processing_metadata(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add processing metadata to DataFrame.

        Args:
            df (pd.DataFrame): Input DataFrame

        Returns:
            pd.DataFrame: DataFrame with added metadata
        """
        df["_processed_at"] = pd.Timestamp.now()
        df["_source_file"] = getattr(df, "_source_file", "unknown")
        return df


class GAVDDataLoader:
    """
    Comprehensive GAVD data loading and organization system.

    This class follows the Single Responsibility Principle by focusing
    solely on data loading and organization, while using composition
    to delegate validation and transformation responsibilities.

    Features:
    - Configurable validation strategies
    - Pluggable transformation pipelines
    - Comprehensive error handling
    - Flexible data organization options
    - Extensible design for future enhancements
    """

    def __init__(
        self,
        validator: Optional[DataValidator] = None,
        transformer: Optional[DataTransformer] = None,
        logger: Optional[Any] = None,
    ):
        """
        Initialize the GAVD data loader with optional components.

        Args:
            validator (Optional[DataValidator]): Data validation strategy
            transformer (Optional[DataTransformer]): Data transformation strategy
            logger (Optional[Any]): Logger for debugging and monitoring
        """
        self.validator = validator or GAVDDataValidator()
        self.transformer = transformer or GAVDDataTransformer()
        self.logger = logger or self._setup_default_logger()

        # Default configuration
        self.default_dict_fields = ["bbox", "vid_info"]
        self.default_verbose = False

    def _setup_default_logger(self):
        """Return Loguru's singleton logger."""
        return loguru_logger
    
    def _download_youtube_videos(
        self,
        rows: List[Dict[str, Any]],
        youtube_dir: str = "data/youtube",
        cookies_path: Optional[Union[Path, str]] = project_root / "config/yt_cookies.txt",
    ) -> None:
        """
        Download all unique Youtube videos.
        """
        # Normalize cookies path (supports None/str/Path)
        if cookies_path is None:
            cookies_arg = None
        else:
            candidate = Path(cookies_path)
            if not candidate.exists():
                self.logger.warning(
                    f"YouTube cookies file not found at {candidate}"
                )
                cookies_arg = None
            else:
                cookies_arg = candidate

        summary = cache_youtube_videos_from_rows(
            rows,
            output_dir=youtube_dir,
            cookies_file=cookies_arg,
        )
        loguru_logger.info(f"YouTube cache attempted: {summary['attempted']}")
        loguru_logger.info(f"YouTube cache skipped: {summary['skipped']}")
        loguru_logger.info(f"YouTube cache downloaded: {summary['downloaded']}")
        if summary["failed"] > 0:
            loguru_logger.warning(f"YouTube cache failed: {summary['failed']}")
        else:
            loguru_logger.info(f"YouTube cache failed: {summary['failed']}")

    def load_gavd_data(
        self,
        csv_file_path: str,
        dict_fields: Optional[List[str]] = None,
        convert_frame_num: bool = True,
        verbose: bool = False,
    ) -> pd.DataFrame:
        """
        Load GAVD CSV data with comprehensive validation and transformation.

        This method implements a complete data loading pipeline:
        1. File existence and accessibility validation
        2. CSV parsing with dictionary field handling
        3. Data validation using configured validator
        4. Data transformation using configured transformer
        5. Comprehensive error handling and logging

        Args:
            csv_file_path (str): Path to GAVD CSV file
            dict_fields (Optional[List[str]]): Fields to parse as dictionaries
            convert_frame_num (bool): Whether to convert frame_num to numeric type
            verbose (bool): Whether to print detailed information

        Returns:
            pd.DataFrame: Loaded and processed GAVD data

        Raises:
            FileNotFoundError: If CSV file doesn't exist or is inaccessible
            ValueError: If CSV file is empty, invalid, or fails validation
            RuntimeError: If data transformation fails
        """
        # Validate file existence
        self._validate_file_path(csv_file_path)

        # Loguru: call directly; sinks control output level
        self.logger.info(f"Loading GAVD data from: {csv_file_path}")

        # Use default dict fields if not specified
        if dict_fields is None:
            dict_fields = self.default_dict_fields

        # Parse CSV with dictionary fields
        try:
            rows = parse_csv_with_dicts(csv_file_path, dict_fields=dict_fields)
        except Exception as e:
            raise ValueError(f"Failed to parse CSV file {csv_file_path}: {str(e)}")

        if not rows:
            raise ValueError(f"CSV file is empty: {csv_file_path}")

        # Download all unique Youtube videos
        self._download_youtube_videos(rows)

        # Convert to DataFrame
        df = pd.DataFrame(rows)

        # Store source file information for metadata
        df._source_file = csv_file_path

        # Validate data using configured validator
        if not self.validator.validate(df):
            raise ValueError(
                f"Data validation failed: {self.validator.get_error_message()}"
            )

        # Transform data using configured transformer
        try:
            df = self.transformer.transform(df)
        except Exception as e:
            raise RuntimeError(f"Data transformation failed: {str(e)}")

        # Log loading statistics
        # Loguru: call directly; sinks control output level
        self._log_loading_statistics(df)

        return df

    def _validate_file_path(self, file_path: str) -> None:
        """
        Validate file path existence and accessibility.

        Args:
            file_path (str): Path to validate

        Raises:
            FileNotFoundError: If file doesn't exist or is inaccessible
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"GAVD CSV file not found: {file_path}")

        if not os.path.isfile(file_path):
            raise FileNotFoundError(f"Path is not a file: {file_path}")

        if not os.access(file_path, os.R_OK):
            raise FileNotFoundError(f"File is not readable: {file_path}")

    def _log_loading_statistics(self, df: pd.DataFrame) -> None:
        """
        Log comprehensive loading statistics.

        Args:
            df (pd.DataFrame): Loaded DataFrame
        """
        self.logger.info(
            f"Loaded {len(df)} rows with {df['seq'].nunique()} unique sequences"
        )
        self.logger.info(f"Columns: {list(df.columns)}")

        if "frame_num" in df.columns:
            frame_range = f"{df['frame_num'].min()}-{df['frame_num'].max()}"
            self.logger.info(f"Frame range: {frame_range}")

    def organize_by_sequence(
        self, df: pd.DataFrame, sort_by_frame: bool = True, verbose: bool = False
    ) -> Dict[str, pd.DataFrame]:
        """
        Organize GAVD data by sequence ID with configurable sorting.

        This method provides flexible sequence organization with:
        - Optional frame-based sorting
        - Comprehensive sequence metadata
        - Configurable verbosity for debugging
        - Efficient data grouping

        Args:
            df (pd.DataFrame): GAVD data DataFrame
            sort_by_frame (bool): Whether to sort sequences by frame number
            verbose (bool): Whether to print detailed information

        Returns:
            Dict[str, pd.DataFrame]: Dictionary with sequence ID as key and
                                   sequence data as value
        """
        sequences = {}

        # Get unique sequence IDs
        unique_sequences = df["seq"].unique()

        if verbose:
            self.logger.info(f"Organizing {len(unique_sequences)} sequences")

        for seq_id in unique_sequences:
            # Extract sequence data
            seq_data = df[df["seq"] == seq_id].copy()

            # Sort by frame number if requested and available
            if sort_by_frame and "frame_num" in seq_data.columns:
                seq_data = seq_data.sort_values("frame_num")

            # Store sequence data
            sequences[seq_id] = seq_data

            # Log sequence information if verbose
            if verbose:
                self._log_sequence_info(seq_id, seq_data)

        return sequences

    def _log_sequence_info(self, seq_id: str, seq_data: pd.DataFrame) -> None:
        """
        Log detailed information about a sequence.

        Args:
            seq_id (str): Sequence identifier
            seq_data (pd.DataFrame): Sequence data
        """
        frame_range = "N/A"
        if "frame_num" in seq_data.columns:
            frame_range = f"{seq_data['frame_num'].min()}-{seq_data['frame_num'].max()}"

        self.logger.debug(
            f"Sequence {seq_id}: {len(seq_data)} frames, frames {frame_range}"
        )

    def get_sequence_statistics(
        self, sequences: Dict[str, pd.DataFrame]
    ) -> Dict[str, Any]:
        """
        Generate comprehensive statistics for organized sequences.

        Args:
            sequences (Dict[str, pd.DataFrame]): Organized sequence data

        Returns:
            Dict[str, Any]: Statistics including count, frame ranges, etc.
        """
        stats = {
            "total_sequences": len(sequences),
            "total_frames": sum(len(seq) for seq in sequences.values()),
            "sequence_frame_counts": {},
            "frame_ranges": {},
            "avg_frames_per_sequence": 0,
        }

        if sequences:
            for seq_id, seq_data in sequences.items():
                frame_count = len(seq_data)
                stats["sequence_frame_counts"][seq_id] = frame_count

                if "frame_num" in seq_data.columns:
                    frame_range = {
                        "min": int(seq_data["frame_num"].min()),
                        "max": int(seq_data["frame_num"].max()),
                    }
                    stats["frame_ranges"][seq_id] = frame_range

            stats["avg_frames_per_sequence"] = stats["total_frames"] / len(sequences)

        return stats

    def filter_sequences(
        self,
        sequences: Dict[str, pd.DataFrame],
        min_frames: Optional[int] = None,
        max_frames: Optional[int] = None,
        frame_range_filter: Optional[Callable[[pd.DataFrame], bool]] = None,
    ) -> Dict[str, pd.DataFrame]:
        """
        Filter sequences based on various criteria.

        Args:
            sequences (Dict[str, pd.DataFrame]): Input sequences
            min_frames (Optional[int]): Minimum number of frames required
            max_frames (Optional[int]): Maximum number of frames allowed
            frame_range_filter (Optional[Callable]): Custom filter function

        Returns:
            Dict[str, pd.DataFrame]: Filtered sequences
        """
        filtered_sequences = {}

        for seq_id, seq_data in sequences.items():
            frame_count = len(seq_data)

            # Apply frame count filters
            if min_frames is not None and frame_count < min_frames:
                continue
            if max_frames is not None and frame_count > max_frames:
                continue

            # Apply custom filter if provided
            if frame_range_filter is not None and not frame_range_filter(seq_data):
                continue

            filtered_sequences[seq_id] = seq_data

        return filtered_sequences


# Factory function for easy instantiation
def create_gavd_loader(
    validator: Optional[DataValidator] = None,
    transformer: Optional[DataTransformer] = None,
    logger: Optional[Any] = None,
) -> GAVDDataLoader:
    """
    Factory function to create a GAVDDataLoader with optional components.

    Args:
        validator (Optional[DataValidator]): Custom validator
        transformer (Optional[DataTransformer]): Custom transformer
        logger (Optional[Any]): Custom logger (Loguru or compatible)

    Returns:
        GAVDDataLoader: Configured loader instance
    """
    return GAVDDataLoader(validator=validator, transformer=transformer, logger=logger)


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
        self.bbox_processor = bbox_processor or self._create_default_bbox_processor()
        self.keypoint_generator = (
            keypoint_generator or self._create_default_keypoint_generator()
        )

    def _create_default_bbox_processor(self):
        """Create default bounding box processor."""
        from ambient.gavd.keypoints import BoundingBoxProcessor

        return BoundingBoxProcessor()

    def _create_default_keypoint_generator(self):
        """Create default keypoint generator."""
        from ambient.gavd.keypoints import KeypointGenerator

        return KeypointGenerator()

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


class PoseDataConverter:
    """
    Converts GAVD data to pose estimation format.

    This class follows the Single Responsibility Principle by focusing
    solely on data format conversion.
    """

    def __init__(
        self,
        keypoint_extractor: Optional[PoseKeypointExtractor] = None,
        estimator: Optional[PoseEstimator] = None,
        video_cache_dir: Optional[Union[str, Path]] = "data/youtube",
    ):
        """
        Initialize the pose data converter.

        Args:
            keypoint_extractor (Optional[PoseKeypointExtractor]): Keypoint extractor
        """
        self.keypoint_extractor = keypoint_extractor or PoseKeypointExtractor()
        # Optional pluggable estimator (e.g., OpenPose, MediaPipe)
        self.estimator: Optional[PoseEstimator] = estimator
        self.video_cache_dir: Path = self._resolve_cache_dir(video_cache_dir)

    def _resolve_cache_dir(self, p: Optional[Union[str, Path]]) -> Path:
        base = Path(p or "data/youtube")
        if not base.is_absolute():
            # Resolve relative to project root
            root = Path(__file__).parents[2]
            return (root / base).resolve()
        return base.resolve()

    def _resolve_cached_video_path(self, url: str) -> Optional[Path]:
        """Given a YouTube URL, locate a cached local video file.

        Uses the ID-based filename scheme: <VIDEO_ID>.<ext> under video_cache_dir.
        Accepts common video extensions and requires non-zero size.
        """
        if not isinstance(url, str) or not url:
            return None
        video_id = extract_video_id(url)
        if not video_id:
            return None
        candidates = []
        for ext in (".mp4", ".webm", ".mkv", ".mov"):
            candidate = self.video_cache_dir / f"{video_id}{ext}"
            if candidate.exists() and candidate.is_file():
                try:
                    if candidate.stat().st_size > 0:
                        candidates.append(candidate)
                except OSError:
                    continue
        if not candidates:
            return None
        # Prefer mp4 if present, else first
        for c in candidates:
            if c.suffix.lower() == ".mp4":
                return c
        return candidates[0]

    def _calculate_frame_offset(self, seq_data: pd.DataFrame) -> int:
        """Calculate frame offset for a sequence.
        
        If frame numbers start at a high value (e.g., 1757), this might indicate
        that annotations are for a video segment, not the full video.
        However, if frame numbers are absolute in the full video, offset should be 0.
        
        Args:
            seq_data: Sequence DataFrame with frame_num column
            
        Returns:
            Frame offset to apply (default: 0, meaning use frame_num as-is)
        """
        if "frame_num" not in seq_data.columns or len(seq_data) == 0:
            return 0
        
        min_frame = int(seq_data["frame_num"].min())
        # If frame numbers start at 1 or close to it, no offset needed
        # If they start much higher (e.g., 1757), they're likely absolute frame numbers
        # in the full video, so we use them as-is (offset = 0)
        # Only apply offset if we detect a pattern suggesting relative numbering
        if min_frame <= 10:
            # Frame numbers start near 0/1, might be relative - but GAVD uses absolute
            return 0
        else:
            # Frame numbers are absolute in full video - use as-is
            return 0

    def _scale_bbox_coordinates(
        self, 
        bbox: Dict[str, Union[int, float]], 
        annotation_width: int, 
        annotation_height: int,
        video_width: int, 
        video_height: int
    ) -> Dict[str, Union[int, float]]:
        """Scale bounding box coordinates if video resolution differs from annotation.
        
        Args:
            bbox: Bounding box dict with 'left', 'top', 'width', 'height'
            annotation_width: Width from vid_info
            annotation_height: Height from vid_info
            video_width: Actual video width
            video_height: Actual video height
            
        Returns:
            Scaled bounding box dict
        """
        if not isinstance(bbox, dict):
            return bbox
        
        # If resolutions match, no scaling needed
        if annotation_width == video_width and annotation_height == video_height:
            return bbox
        
        # Calculate scaling factors
        scale_x = video_width / annotation_width if annotation_width > 0 else 1.0
        scale_y = video_height / annotation_height if annotation_height > 0 else 1.0
        
        # Scale coordinates
        scaled_bbox = {
            "left": bbox.get("left", 0) * scale_x,
            "top": bbox.get("top", 0) * scale_y,
            "width": bbox.get("width", 0) * scale_x,
            "height": bbox.get("height", 0) * scale_y,
        }
        
        return scaled_bbox

    def _extract_frame_image(self, video_path: Path, frame_index: int) -> Path:
        """Extract a single frame as an image using ffmpeg and return temp image path.

        Uses frame index selection (0-based). Caller is responsible for cleanup.
        
        Note: frame_index should be 0-based. If you have a 1-based frame_num from CSV,
        convert it to 0-based before calling this method (frame_index = frame_num - 1).
        """
        temp_dir = tempfile.mkdtemp(prefix="gavd_frame_")
        out_path = Path(temp_dir) / f"frame_{frame_index:06d}.jpg"
        # Build ffmpeg command: select frame by index (0-based)
        # Use 'select' filter with exact frame number matching
        # Note: comma must be escaped in ffmpeg filter expressions
        select_expr = f"select=eq(n\\,{frame_index})"
        cmd = [
            "ffmpeg",
            "-v",
            "error",
            "-i",
            str(video_path),
            "-vf",
            select_expr,
            "-frames:v",
            "1",
            str(out_path),
            "-y",
        ]
        completed = subprocess.run(cmd, capture_output=True, text=True)
        if completed.returncode != 0 or not out_path.exists() or out_path.stat().st_size == 0:
            raise RuntimeError(
                f"ffmpeg failed to extract frame {frame_index} from {video_path}: {completed.stderr}"
            )
        return out_path

    def convert_sequence_to_pose_format(
        self,
        seq_data: pd.DataFrame,
        include_metadata: bool = True,
        person_id: int = 0,
        num_keypoints: int = 25,
        grid_spacing: float = 5.0,
        confidence: float = 0.8,
        model_pose: str = "BODY_25",
        image_path_field: str = "image_path",
    ) -> List[Dict[str, Any]]:
        """
        Convert a GAVD sequence to pose estimation format.

        Args:
            seq_data (pd.DataFrame): Single sequence data
            include_metadata (bool): Whether to include GAVD metadata
            person_id (int): Person ID for the pose data
            num_keypoints (int): Number of keypoints to generate
            grid_spacing (float): Spacing between keypoints in grid
            confidence (float): Confidence score for keypoints

        Returns:
            List[Dict[str, Any]]: Pose data in OpenPose-like format
        """
        pose_frames: List[Dict[str, Any]] = []

        # If estimator is available and URLs present, use cached videos for frames
        urls_in_seq = set()
        if "url" in seq_data.columns:
            urls_in_seq = set(u for u in seq_data["url"].dropna().tolist() if isinstance(u, str) and u.strip())

        # Map URLs to cached local files; if URL is missing, skip only rows for that URL
        url_to_video: Dict[str, Path] = {}
        missing_urls: set[str] = set()
        if self.estimator is not None and urls_in_seq:
            for url in urls_in_seq:
                vp = self._resolve_cached_video_path(url)
                if vp is None:
                    missing_urls.add(url)
                else:
                    url_to_video[url] = vp

        has_urls = bool(urls_in_seq)

        for _, row in seq_data.iterrows():
            bbox = row["bbox"]
            if not isinstance(bbox, dict):
                continue

            pose_keypoints: List[Dict[str, Union[float, int]]]

            if self.estimator is not None and has_urls:
                # Determine frame index (0-based).
                # GAVD CSV uses 1-based frame numbers (starting from 1), so we convert to 0-based.
                # However, if frame_num values are absolute frame numbers in the video (e.g., 1757),
                # we use them directly after converting to 0-based.
                frame_num_val = int(row.get("frame_num", 0))
                # Convert 1-based frame_num to 0-based frame_index
                # This handles both cases: frame_num starting at 1, or absolute frame numbers
                frame_index = frame_num_val - 1 if frame_num_val >= 1 else frame_num_val
                url_val = row.get("url")
                # If URL missing or not cached, skip this row
                if not isinstance(url_val, str) or not url_val.strip():
                    continue
                if url_val in missing_urls:
                    continue
                video_path = url_to_video.get(url_val)
                if video_path is None:
                    # URL missing for this row; skip row
                    continue
                try:
                    # Batch approach: compute once per video per sequence for efficiency
                    estimator_fingerprint = (
                        self.estimator.cache_fingerprint() if hasattr(self.estimator, "cache_fingerprint") else "est"
                    )
                    cache_key = f"{video_path}::{model_pose}::{estimator_fingerprint}"
                    if not hasattr(self, "_video_kp_cache"):
                        self._video_kp_cache = {}
                    if cache_key not in self._video_kp_cache:
                        # Estimate all frames at once
                        try:
                            loguru_logger.debug(f"Processing video keypoints: {video_path}::{model_pose}")
                            video_result = self.estimator.estimate_video_keypoints(
                                video_path, model=model_pose
                            )
                        except AttributeError:
                            # Estimator doesn't support video mode; fall back per-frame
                            video_result = None
                        self._video_kp_cache[cache_key] = video_result
                    video_result = self._video_kp_cache.get(cache_key)

                    # Handle both old (list) and new (dict) return formats
                    all_frames = None
                    source_video_width = None
                    source_video_height = None
                    
                    if isinstance(video_result, dict):
                        all_frames = video_result.get('frames', [])
                        source_video_width = video_result.get('video_width')
                        source_video_height = video_result.get('video_height')
                    elif isinstance(video_result, list):
                        all_frames = video_result
                    
                    if all_frames is not None and 0 <= frame_index < len(all_frames):
                        pose_keypoints = all_frames[frame_index]
                        # Store source video dimensions with keypoints for proper scaling
                        if source_video_width and source_video_height:
                            for kp in pose_keypoints:
                                kp['source_width'] = source_video_width
                                kp['source_height'] = source_video_height
                    else:
                        # Fallback to ffmpeg + single-frame estimation
                        img_path = self._extract_frame_image(video_path, frame_index)
                        try:
                            loguru_logger.debug(f"Processing image keypoints: {img_path}::{frame_index}")
                            pose_keypoints = self.estimator.estimate_image_keypoints(
                                image_path=str(img_path), model=model_pose, bbox=bbox
                            )
                        finally:
                            try:
                                Path(img_path).unlink(missing_ok=True)  # type: ignore[arg-type]
                            except Exception:
                                pass
                except Exception:
                    # Fallback to placeholder on any failure
                    pose_keypoints = self.keypoint_extractor.extract_from_bbox(
                        bbox, num_keypoints, grid_spacing, confidence
                    )
            else:
                # Fallback to placeholder generator
                pose_keypoints = self.keypoint_extractor.extract_from_bbox(
                    bbox, num_keypoints, grid_spacing, confidence
                )

            frame_data = {
                "frame": row.get("frame_num"),
                "person_id": person_id,
                "pose_keypoints_2d": pose_keypoints,
            }

            if include_metadata:
                frame_data["gavd_metadata"] = {
                    "seq": row.get("seq"),
                    "gait_pat": row.get("gait_pat", "Unknown"),
                    "cam_view": row.get("cam_view", "Unknown"),
                    "gait_event": row.get("gait_event", "Unknown"),
                    "dataset": row.get("dataset", "Unknown"),
                    "bbox": bbox,
                    "vid_info": row.get("vid_info", {}),
                    "url": row.get("url"),
                }

            pose_frames.append(frame_data)

        return pose_frames


class GAVDProcessor:
    """
    High-level processor for GAVD data with pose estimation capabilities.

    This class combines all the utilities into a cohesive interface
    following the Facade pattern and SOLID principles.

    Features:
    - Comprehensive GAVD data processing pipeline
    - Pose estimation integration
    - Configurable processing parameters
    - Flexible output formats
    - Error handling and logging
    """

    def __init__(
        self,
        data_loader: Optional[GAVDDataLoader] = None,
        data_converter: Optional[PoseDataConverter] = None,
    ):
        """
        Initialize the GAVD processor with optional components.

        Args:
            data_loader (Optional[GAVDDataLoader]): Data loader component
            data_converter (Optional[PoseDataConverter]): Data converter component
        """
        self.data_loader = data_loader or GAVDDataLoader()
        self.data_converter = data_converter or PoseDataConverter()

    def process_gavd_file(
        self,
        csv_file_path: str,
        max_sequences: Optional[int] = None,
        include_metadata: bool = True,
        verbose: bool = False,
    ) -> Dict[str, Any]:
        """
        Process a GAVD CSV file and convert to pose format.

        This method implements a complete processing pipeline:
        1. Load and validate GAVD data
        2. Organize data by sequence
        3. Convert sequences to pose format
        4. Generate comprehensive output with statistics

        Args:
            csv_file_path (str): Path to GAVD CSV file
            max_sequences (Optional[int]): Maximum number of sequences to process
            include_metadata (bool): Whether to include GAVD metadata
            verbose (bool): Whether to print detailed information

        Returns:
            Dict[str, Any]: Processed data with sequences and pose information
                - total_sequences: Number of processed sequences
                - sequences: Dictionary of processed sequence data
                - summary: Processing statistics
        """
        # Load GAVD data
        df = self.data_loader.load_gavd_data(csv_file_path, verbose=verbose)

        # Organize by sequence
        sequences = self.data_loader.organize_by_sequence(df, verbose=verbose)

        # Process sequences
        processed_sequences = {}
        sequence_count = 0

        for seq_id, seq_data in sequences.items():
            if max_sequences and sequence_count >= max_sequences:
                break

            # Convert to pose format
            pose_data = self.data_converter.convert_sequence_to_pose_format(
                seq_data, include_metadata=include_metadata
            )

            processed_sequences[seq_id] = {
                "original_data": seq_data,
                "pose_data": pose_data,
                "frame_count": len(pose_data),
            }

            sequence_count += 1

        return {
            "total_sequences": len(processed_sequences),
            "sequences": processed_sequences,
            "summary": {
                "total_frames": sum(
                    seq["frame_count"] for seq in processed_sequences.values()
                ),
                "average_frames_per_sequence": (
                    sum(seq["frame_count"] for seq in processed_sequences.values())
                    / len(processed_sequences)
                    if processed_sequences
                    else 0
                ),
            },
        }

    def process_sequences_with_filtering(
        self,
        csv_file_path: str,
        min_frames: Optional[int] = None,
        max_frames: Optional[int] = None,
        include_metadata: bool = True,
        verbose: bool = False,
    ) -> Dict[str, Any]:
        """
        Process GAVD data with sequence filtering capabilities.

        Args:
            csv_file_path (str): Path to GAVD CSV file
            min_frames (Optional[int]): Minimum frames required per sequence
            max_frames (Optional[int]): Maximum frames allowed per sequence
            include_metadata (bool): Whether to include GAVD metadata
            verbose (bool): Whether to print detailed information

        Returns:
            Dict[str, Any]: Filtered and processed data
        """
        # Load and organize data
        df = self.data_loader.load_gavd_data(csv_file_path, verbose=verbose)
        sequences = self.data_loader.organize_by_sequence(df, verbose=verbose)

        # Apply filtering
        filtered_sequences = self.data_loader.filter_sequences(
            sequences, min_frames=min_frames, max_frames=max_frames
        )

        # Process filtered sequences
        processed_sequences = {}

        for seq_id, seq_data in filtered_sequences.items():
            pose_data = self.data_converter.convert_sequence_to_pose_format(
                seq_data, include_metadata=include_metadata
            )

            processed_sequences[seq_id] = {
                "original_data": seq_data,
                "pose_data": pose_data,
                "frame_count": len(pose_data),
            }

        return {
            "total_sequences": len(processed_sequences),
            "sequences": processed_sequences,
            "filtering_stats": {
                "original_sequences": len(sequences),
                "filtered_sequences": len(filtered_sequences),
                "total_frames": sum(
                    seq["frame_count"] for seq in processed_sequences.values()
                ),
                "average_frames_per_sequence": (
                    sum(seq["frame_count"] for seq in processed_sequences.values())
                    / len(processed_sequences)
                    if processed_sequences
                    else 0
                ),
            },
        }

    def get_processing_statistics(
        self, processed_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Generate comprehensive statistics for processed data.

        Args:
            processed_data (Dict[str, Any]): Output from process_gavd_file

        Returns:
            Dict[str, Any]: Detailed processing statistics
        """
        sequences = processed_data.get("sequences", {})

        if not sequences:
            return {
                "total_sequences": 0,
                "total_frames": 0,
                "average_frames_per_sequence": 0,
                "sequence_frame_distribution": {},
                "processing_efficiency": 0.0,
            }

        frame_counts = [seq["frame_count"] for seq in sequences.values()]

        stats = {
            "total_sequences": len(sequences),
            "total_frames": sum(frame_counts),
            "average_frames_per_sequence": sum(frame_counts) / len(frame_counts),
            "min_frames_per_sequence": min(frame_counts),
            "max_frames_per_sequence": max(frame_counts),
            "sequence_frame_distribution": {
                "short_sequences": len([f for f in frame_counts if f < 10]),
                "medium_sequences": len([f for f in frame_counts if 10 <= f <= 50]),
                "long_sequences": len([f for f in frame_counts if f > 50]),
            },
            "processing_efficiency": len(sequences) / max(len(sequences), 1),
        }

        return stats


# Factory function for GAVDProcessor
def create_gavd_processor(
    data_loader: Optional[GAVDDataLoader] = None,
    data_converter: Optional[PoseDataConverter] = None,
) -> GAVDProcessor:
    """
    Factory function to create a GAVDProcessor with optional components.

    Args:
        data_loader (Optional[GAVDDataLoader]): Custom data loader
        data_converter (Optional[PoseDataConverter]): Custom data converter

    Returns:
        GAVDProcessor: Configured processor instance
    """
    return GAVDProcessor(data_loader=data_loader, data_converter=data_converter)
