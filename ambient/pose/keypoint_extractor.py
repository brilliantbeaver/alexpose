"""
Keypoint Extractor Module

Provides pose keypoint extraction from images and video sequences using MediaPipe.

Key features:
- Single image/frame extraction
- Batch sequence processing
- Automatic process isolation on Windows for threading issues
- Robust error handling and recovery

Author: AlexPose Team
"""

import os
from pathlib import Path
from typing import List, Optional, Tuple
import pandas as pd
import cv2
import numpy as np

from ambient.pose.keypoint_data import (
    KeypointSet,
    KeypointFormat,
    MEDIAPIPE_33_NAMES,
)
from ambient.pose.model_management import (
    MediaPipeModelManager,
    PoseLandmarkerFactory,
)
from ambient.pose.utils import suppress_stderr_fd
from ambient.utils.log_config import get_logger
from ambient.utils.youtube_cache import extract_video_id

logger = get_logger(__name__)

try:
    import mediapipe as mp

    MEDIAPIPE_AVAILABLE = True
except ImportError:
    MEDIAPIPE_AVAILABLE = False
    mp = None


class SequenceKeypointExtractor:
    """
    Extracts pose keypoints from images and video sequences using MediaPipe.

    Automatically handles Windows threading issues by switching to process isolation
    when needed. Provides robust error handling and recovery mechanisms.

    Example:
        >>> extractor = SequenceKeypointExtractor()
        >>> image = cv2.imread("person.jpg")
        >>> image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        >>> keypoints = extractor.extract_from_image(image_rgb)
    """

    def __init__(
        self,
        model_manager: Optional[MediaPipeModelManager] = None,
        landmarker_factory: Optional[PoseLandmarkerFactory] = None,
        suppress_warnings: bool = True,
    ):
        """
        Initialize the keypoint extractor.

        Args:
            model_manager: Model manager (creates default if None)
            landmarker_factory: Landmarker factory (uses default if None)
            suppress_warnings: Suppress MediaPipe warnings
        """
        self.model_manager = model_manager or MediaPipeModelManager()
        self.landmarker_factory = landmarker_factory or PoseLandmarkerFactory()
        self.suppress_warnings = suppress_warnings

    # ========================================================================
    # MediaPipe Landmarker Management
    # ========================================================================

    def _get_landmarker(self, model_path: Optional[str] = None):
        """Get MediaPipe landmarker using singleton pattern."""
        if model_path is None:
            model_path = self.model_manager.ensure_model_available()
            if model_path is None:
                raise RuntimeError("Failed to download or locate pose model")

        from ambient.pose.mediapipe_singleton import get_mediapipe_singleton

        singleton = get_mediapipe_singleton()

        try:
            landmarker = singleton.get_landmarker(model_path, self.landmarker_factory)
            return landmarker
        except Exception as e:
            return self._handle_landmarker_error(e, model_path, singleton)

    def _handle_landmarker_error(self, error: Exception, model_path: str, singleton):
        """Handle landmarker creation errors with retry logic."""
        # Try reset and retry
        logger.warning(f"Landmarker failed, resetting: {error}")
        singleton.reset_landmarker()

        try:
            landmarker = singleton.get_landmarker(model_path, self.landmarker_factory)
            return landmarker
        except Exception as retry_error:
            raise RuntimeError(
                f"Failed to create landmarker after reset: {retry_error}"
            ) from error

    def reset_landmarker(self):
        """Reset landmarker state."""
        from ambient.pose.mediapipe_singleton import get_mediapipe_singleton

        get_mediapipe_singleton().reset_landmarker()

    def cleanup(self):
        """Clean up all resources."""
        self.reset_landmarker()

    def __del__(self):
        """Ensure cleanup on deletion."""
        try:
            self.cleanup()
        except Exception:
            pass

    # ========================================================================
    # Core Extraction Methods
    # ========================================================================

    def extract_from_image(
        self, image: np.ndarray, model_path: Optional[str] = None
    ) -> KeypointSet:
        """
        Extract pose keypoints from a single RGB image.

        Args:
            image: RGB image array (height, width, 3)
            model_path: Optional path to model file

        Returns:
            KeypointSet with detected keypoints (empty if none detected)

        Example:
            >>> image_rgb = cv2.cvtColor(cv2.imread("person.jpg"), cv2.COLOR_BGR2RGB)
            >>> keypoints = extractor.extract_from_image(image_rgb)
        """
        if not MEDIAPIPE_AVAILABLE:
            raise ImportError("MediaPipe is required for pose extraction")

        # Use MediaPipe singleton
        return self._extract_with_mediapipe(image, model_path)

    def _extract_with_mediapipe(
        self, image: np.ndarray, model_path: Optional[str] = None
    ) -> KeypointSet:
        """Extract keypoints using MediaPipe singleton."""
        height, width = image.shape[:2]

        try:
            landmarker = self._get_landmarker(model_path)

            # Increment frame counter for memory management
            from ambient.pose.mediapipe_singleton import get_mediapipe_singleton

            get_mediapipe_singleton().increment_frame_count()

            # Detect pose
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image)
            detection_result = self._detect_pose(landmarker, mp_image)

            # Convert to KeypointSet
            return self._create_keypoint_set(detection_result, width, height)

        except RuntimeError as e:
            # Retry once with reset
            return self._retry_extraction_with_reset(image, model_path, width, height)

        except Exception as e:
            logger.error(f"Unexpected error in keypoint extraction: {e}")
            return self._empty_keypoint_set(width, height)

    def _detect_pose(self, landmarker, mp_image):
        """Detect pose with optional warning suppression."""
        if self.suppress_warnings:
            with suppress_stderr_fd():
                return landmarker.detect(mp_image)
        return landmarker.detect(mp_image)

    def _create_keypoint_set(
        self, detection_result, width: int, height: int
    ) -> KeypointSet:
        """Create KeypointSet from MediaPipe detection result."""
        if not detection_result.pose_landmarks:
            return self._empty_keypoint_set(width, height)

        return KeypointSet.from_mediapipe(
            landmarks=detection_result.pose_landmarks[0],
            frame_width=width,
            frame_height=height,
            landmark_names=MEDIAPIPE_33_NAMES,
        )

    def _empty_keypoint_set(self, width: int, height: int) -> KeypointSet:
        """Create empty KeypointSet."""
        return KeypointSet(
            keypoints=[],
            format=KeypointFormat.CUSTOM,
            frame_width=width,
            frame_height=height,
        )

    def _retry_extraction_with_reset(
        self, image: np.ndarray, model_path: Optional[str], width: int, height: int
    ) -> KeypointSet:
        """Retry extraction after resetting landmarker."""
        logger.warning("Retrying extraction after reset")

        from ambient.pose.mediapipe_singleton import get_mediapipe_singleton

        get_mediapipe_singleton().reset_landmarker()

        try:
            landmarker = self._get_landmarker(model_path)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image)
            detection_result = self._detect_pose(landmarker, mp_image)
            return self._create_keypoint_set(detection_result, width, height)
        except Exception as e:
            logger.error(f"Retry also failed: {e}")
            return self._empty_keypoint_set(width, height)

    def extract_from_frame_file(
        self, image_path: str, model_path: Optional[str] = None
    ) -> KeypointSet:
        """
        Extract pose keypoints from an image file.

        Args:
            image_path: Path to image file
            model_path: Optional path to model file

        Returns:
            KeypointSet object
        """
        image = cv2.imread(str(image_path))
        if image is None:
            raise ValueError(f"Failed to read image from {image_path}")

        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        return self.extract_from_image(image_rgb, model_path)

    def extract_from_video_frame(
        self, video_path: Path, frame_number: int, model_path: Optional[str] = None
    ) -> Optional[KeypointSet]:
        """
        Extract pose keypoints from a specific video frame.

        Uses FFmpeg with OpenCV fallback for robust frame extraction.

        Args:
            video_path: Path to video file
            frame_number: Frame number to extract (1-based)
            model_path: Optional path to model file

        Returns:
            KeypointSet or None if extraction failed
        """
        # Extract frame from video
        frame = self._extract_video_frame(video_path, frame_number)
        if frame is None:
            return None

        # Convert to RGB and extract keypoints
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        keypoints = self._extract_with_retry(frame_rgb, model_path, max_retries=2)

        if keypoints is not None:
            keypoints.timestamp = float(frame_number)

        return keypoints

    def _extract_video_frame(
        self, video_path: Path, frame_number: int
    ) -> Optional[np.ndarray]:
        """Extract frame from video using FFmpeg with OpenCV fallback."""
        try:
            from ambient.pose.windows_ffmpeg_handler import WindowsVideoFrameExtractor

            frame_extractor = WindowsVideoFrameExtractor(
                prefer_ffmpeg=True, ffmpeg_timeout=30, verbose=False
            )

            frame = frame_extractor.extract_frame(video_path, frame_number)

            if frame is None or frame.size == 0:
                logger.warning(
                    f"Failed to extract frame {frame_number} from {video_path}"
                )
                return None

            return frame

        except Exception as e:
            logger.error(f"Frame extraction failed for frame {frame_number}: {e}")
            return self._extract_frame_opencv_fallback(video_path, frame_number)

    def _extract_frame_opencv_fallback(
        self, video_path: Path, frame_number: int
    ) -> Optional[np.ndarray]:
        """Fallback frame extraction using OpenCV."""
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            logger.error(f"Cannot open video: {video_path}")
            return None

        try:
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number - 1)
            ret, frame = cap.read()
            return frame if ret else None
        finally:
            cap.release()

    def _extract_with_retry(
        self, image: np.ndarray, model_path: Optional[str], max_retries: int = 2
    ) -> Optional[KeypointSet]:
        """Extract keypoints with retry logic."""
        for attempt in range(max_retries):
            try:
                return self.extract_from_image(image, model_path)
            except Exception as e:
                if attempt < max_retries - 1:
                    logger.warning(f"Extraction failed, attempt {attempt + 1}: {e}")
                    self.reset_landmarker()
                else:
                    logger.error(f"Extraction failed after {max_retries} attempts: {e}")
                    return None

    # ========================================================================
    # Sequence Processing
    # ========================================================================

    def extract_from_sequence(
        self,
        sequence_data: pd.DataFrame,
        video_base_path: Path,
        model_path: Optional[str] = None,
        verbose: bool = False,
        filter_empty: bool = False,
        min_keypoints: int = 25,
    ) -> List[KeypointSet]:
        """
        Extract pose keypoints from a sequence of video frames.

        Processes each frame robustly - individual failures don't stop processing.

        Args:
            sequence_data: DataFrame with 'frame_num' and 'url' columns
            video_base_path: Base path where videos are stored
            model_path: Optional path to model file
            verbose: Print progress information
            filter_empty: Filter out frames with too few keypoints
            min_keypoints: Minimum keypoints required when filtering (default 25)

        Returns:
            List of KeypointSet objects (may contain None for failed frames)
            If filter_empty=True, only returns frames with >= min_keypoints
            Returns empty list if validation fails or no frames could be processed
        """
        # Validate input
        if not self._validate_sequence_input(sequence_data, video_base_path):
            logger.error("Sequence validation failed - returning empty array")
            return []

        # Initialize
        try:
            self._get_landmarker(model_path)
        except Exception as e:
            logger.error(f"Failed to initialize landmarker: {e}")
            return []

        # Process frames
        keypoints_array = self._process_all_frames(
            sequence_data, video_base_path, model_path, verbose
        )

        # Check if we got any results at all
        if not keypoints_array:
            logger.warning("No frames were processed - extraction completely failed")
            return []

        # Apply filtering if requested
        if filter_empty:
            filtered = self._filter_keypoints(keypoints_array, min_keypoints, verbose)
            if not filtered:
                logger.warning(
                    f"All {len(keypoints_array)} frames were filtered out "
                    f"(min_keypoints={min_keypoints}). Consider lowering min_keypoints threshold."
                )
            return filtered

        return keypoints_array

    def _validate_sequence_input(
        self, sequence_data: pd.DataFrame, video_base_path: Path
    ) -> bool:
        """Validate sequence data and paths with detailed logging."""
        is_valid, message = self.validate_sequence_data_verbose(
            sequence_data, video_base_path
        )
        if not is_valid:
            logger.error(f"Sequence validation failed: {message}")
            logger.error(f"  DataFrame shape: {sequence_data.shape if not sequence_data.empty else 'empty'}")
            logger.error(f"  Video base path: {video_base_path}")
            if not sequence_data.empty and 'url' in sequence_data.columns:
                sample_url = sequence_data['url'].iloc[0] if len(sequence_data) > 0 else 'N/A'
                logger.error(f"  Sample URL: {sample_url}")
        return is_valid

    def _process_all_frames(
        self,
        sequence_data: pd.DataFrame,
        video_base_path: Path,
        model_path: Optional[str],
        verbose: bool,
    ) -> List[Optional[KeypointSet]]:
        """Process all frames in sequence."""
        sequence_id = sequence_data.get("seq", pd.Series(["unknown"])).iloc[0]
        num_frames = len(sequence_data)

        if verbose:
            logger.info(f"Processing sequence: {sequence_id} ({num_frames} frames)")

        keypoints_array = []
        failed_frames = []
        video_cache = {}

        for idx in range(num_frames):
            frame_row = sequence_data.iloc[idx]
            keypoints = self._process_frame(
                frame_row,
                idx,
                video_base_path,
                video_cache,
                model_path,
                verbose,
                num_frames,
            )

            keypoints_array.append(keypoints)
            if keypoints is None:
                failed_frames.append(idx)

        # Log summary
        success_count = sum(1 for kp in keypoints_array if kp is not None)
        if verbose or failed_frames:
            logger.info(f"Processed {success_count}/{num_frames} frames successfully")
            if failed_frames and len(failed_frames) <= 10:
                logger.warning(f"Failed frames: {failed_frames}")

        return keypoints_array

    def _process_frame(
        self,
        frame_row: pd.Series,
        idx: int,
        video_base_path: Path,
        video_cache: dict,
        model_path: Optional[str],
        verbose: bool,
        total_frames: int,
    ) -> Optional[KeypointSet]:
        """Process a single frame from sequence."""
        try:
            # Get frame number
            frame_num = int(frame_row["frame_num"])

            # Progress logging
            if verbose and idx % 10 == 0:
                print(f"\tframe {frame_num} ({idx+1}/{total_frames})")

            # Get video path
            url = frame_row.get("url")
            if not url or pd.isna(url):
                if idx == 0:  # Only log once to avoid spam
                    logger.warning(f"Missing or invalid URL in frame data")
                return None

            video_path = self._get_cached_video_path(url, video_base_path, video_cache)
            if video_path is None:
                if idx == 0:  # Only log once
                    logger.warning(f"Video file not found for URL: {url}")
                return None

            # Extract keypoints
            keypoints = self.extract_from_video_frame(video_path, frame_num, model_path)
            
            # Log if extraction returned None or empty keypoints
            if keypoints is None:
                if idx == 0:
                    logger.warning(f"Keypoint extraction returned None for frame {frame_num}")
            elif len(keypoints.keypoints) == 0:
                if idx == 0:
                    logger.warning(f"Keypoint extraction returned 0 keypoints for frame {frame_num}")
            
            return keypoints

        except Exception as e:
            if idx < 5:  # Only log first few errors to avoid spam
                logger.warning(f"Error processing frame {idx} (frame_num={frame_row.get('frame_num', 'unknown')}): {e}")
            return None

    def _get_cached_video_path(
        self, url: str, video_base_path: Path, cache: dict
    ) -> Optional[Path]:
        """Get video path from URL with caching."""
        if url not in cache:
            try:
                video_id = extract_video_id(url)
                video_path = video_base_path / f"{video_id}.mp4"
                cache[url] = video_path if video_path.exists() else None
            except Exception as e:
                logger.warning(f"Failed to resolve video path: {e}")
                cache[url] = None

        return cache[url]

    def _filter_keypoints(
        self,
        keypoints_array: List[Optional[KeypointSet]],
        min_keypoints: int,
        verbose: bool,
    ) -> List[KeypointSet]:
        """Filter keypoints by minimum count."""
        original_count = len(keypoints_array)

        filtered = [
            kp
            for kp in keypoints_array
            if kp is not None and len(kp.keypoints) >= min_keypoints
        ]

        removed = original_count - len(filtered)
        if verbose or removed > 0:
            logger.info(
                f"Filtered to {len(filtered)}/{original_count} frames "
                f"(min {min_keypoints} keypoints, removed {removed})"
            )

        return filtered

    # ========================================================================
    # Statistics and Validation
    # ========================================================================

    def get_extraction_statistics(
        self, keypoints_array: List[Optional[KeypointSet]]
    ) -> dict:
        """
        Get detailed statistics about keypoint extraction results.

        Returns:
            Dictionary with: total, none, empty, valid, full, partial, poor,
            avg_keypoints, success_rate
        """
        stats = {
            "total": len(keypoints_array),
            "none": 0,
            "empty": 0,
            "valid": 0,
            "full": 0,
            "partial": 0,
            "poor": 0,
            "avg_keypoints": 0.0,
            "success_rate": 0.0,
        }

        if not keypoints_array:
            return stats

        keypoint_counts = []

        for kp_set in keypoints_array:
            if kp_set is None:
                stats["none"] += 1
            else:
                num_kp = len(kp_set.keypoints)

                if num_kp == 0:
                    stats["empty"] += 1
                else:
                    stats["valid"] += 1
                    keypoint_counts.append(num_kp)

                    if num_kp == 33:
                        stats["full"] += 1
                    elif num_kp >= 20:
                        stats["partial"] += 1
                    else:
                        stats["poor"] += 1

        if keypoint_counts:
            stats["avg_keypoints"] = sum(keypoint_counts) / len(keypoint_counts)

        if stats["total"] > 0:
            stats["success_rate"] = (stats["valid"] / stats["total"]) * 100

        return stats

    def print_extraction_statistics(
        self,
        keypoints_array: List[Optional[KeypointSet]],
        sequence_name: str = "Sequence",
    ) -> None:
        """Print detailed statistics about keypoint extraction results."""
        stats = self.get_extraction_statistics(keypoints_array)

        print(f"\n{'='*70}")
        print(f"Keypoint Statistics: {sequence_name}")
        print(f"{'='*70}")
        print(f"Total frames: {stats['total']}")
        
        # Handle empty array case
        if stats['total'] == 0:
            print("  ⚠️  No frames to analyze")
            print(f"{'='*70}\n")
            return
        
        print(f"  ✅ Valid detections: {stats['valid']} ({stats['success_rate']:.1f}%)")
        print(
            f"  ⚠️  Empty detections: {stats['empty']} ({stats['empty']/stats['total']*100:.1f}%)"
        )
        print(
            f"  ❌ Failed extractions: {stats['none']} ({stats['none']/stats['total']*100:.1f}%)"
        )

        if stats["valid"] > 0:
            print(f"\nKeypoint counts (valid frames only):")
            print(f"  Average: {stats['avg_keypoints']:.1f}")

            print(f"\nQuality breakdown:")
            print(f"  🟢 Full (33 keypoints): {stats['full']}")
            print(f"  🟡 Partial (20-32): {stats['partial']}")
            print(f"  🔴 Poor (<20): {stats['poor']}")

        print(f"{'='*70}\n")

    def validate_sequence_data_verbose(
        self, sequence_data: pd.DataFrame, video_base_path: Path
    ) -> Tuple[bool, str]:
        """
        Validate sequence data and provide detailed diagnostic message.

        Returns:
            (is_valid, diagnostic_message)
        """
        if sequence_data.empty:
            return False, "DataFrame is empty"

        required_cols = ["frame_num", "url"]
        missing_cols = [
            col for col in required_cols if col not in sequence_data.columns
        ]
        if missing_cols:
            available = list(sequence_data.columns)
            return False, f"Missing columns: {missing_cols}. Available: {available}"

        if not video_base_path.exists():
            return False, f"Video base path does not exist: {video_base_path}"

        # Check if at least one video file exists
        sample_url = sequence_data["url"].iloc[0]
        try:
            video_id = extract_video_id(sample_url)
            video_path = video_base_path / f"{video_id}.mp4"
            if not video_path.exists():
                return False, f"Sample video file not found: {video_path}"
        except Exception as e:
            return False, f"Error validating video path: {e}"

        return True, "Validation passed"
