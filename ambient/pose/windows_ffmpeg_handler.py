"""
Windows-specific FFmpeg handler for reliable video frame extraction.

This module implements Windows-safe FFmpeg operations with proper error handling,
resource management, and subprocess isolation to prevent file sharing conflicts.
"""

import os
import subprocess
import tempfile
import time
import uuid
from contextlib import contextmanager
from pathlib import Path
from typing import Optional, Generator, Dict, Any
import cv2
import numpy as np

from ambient.utils.log_config import get_logger

logger = get_logger(__name__)


class WindowsFFmpegError(Exception):
    """Base exception for Windows FFmpeg operations."""
    pass


class FFmpegNotFoundError(WindowsFFmpegError):
    """FFmpeg executable not found."""
    pass


class FFmpegExtractionError(WindowsFFmpegError):
    """FFmpeg frame extraction failed."""
    pass


class WindowsTempFileManager:
    """
    Manages temporary files for Windows-safe subprocess operations.
    
    This class handles the Windows-specific file sharing restrictions that
    prevent external processes from accessing files created with Python's
    NamedTemporaryFile when delete=True is used.
    """
    
    def __init__(self, prefix: str = "alexpose", suffix: str = ".jpg", verbose: bool = False):
        """
        Initialize the temp file manager.
        
        Args:
            prefix: Prefix for temporary filenames
            suffix: File extension suffix
            verbose: Enable verbose debug logging
        """
        self.prefix = prefix
        self.suffix = suffix
        self.verbose = verbose
        self.temp_dir = Path(tempfile.gettempdir())
        self.temp_dir.mkdir(exist_ok=True)
    
    @contextmanager
    def create_temp_file(self) -> Generator[Path, None, None]:
        """
        Create a temporary file that can be safely accessed by external processes.
        
        Yields:
            Path to temporary file
            
        Raises:
            OSError: If temporary file creation fails
        """
        # Generate unique filename using UUID to prevent conflicts
        unique_id = uuid.uuid4().hex
        temp_filename = f"{self.prefix}_{unique_id}{self.suffix}"
        temp_path = self.temp_dir / temp_filename
        
        try:
            # logger.debug(f"Created temporary file: {temp_path}")
            yield temp_path
        finally:
            # Robust cleanup with retry logic for Windows file locking
            self._cleanup_temp_file(temp_path)
    
    def _cleanup_temp_file(self, temp_path: Path, max_retries: int = 3) -> None:
        """
        Clean up temporary file with retry logic for Windows.
        
        Args:
            temp_path: Path to temporary file
            max_retries: Maximum number of cleanup attempts
        """
        if not temp_path.exists():
            return
        
        for attempt in range(max_retries):
            try:
                temp_path.unlink()
                # logger.debug(f"Successfully cleaned up temporary file: {temp_path}")
                return
            except (OSError, PermissionError) as e:
                if attempt < max_retries - 1:
                    # Wait briefly and retry (Windows may still have file handle open)
                    wait_time = 0.1 * (2 ** attempt)  # Exponential backoff
                    logger.debug(f"Cleanup attempt {attempt + 1} failed, retrying in {wait_time}s: {e}")
                    time.sleep(wait_time)
                else:
                    # Final attempt failed, log but don't raise
                    logger.warning(f"Failed to cleanup temporary file after {max_retries} attempts: {temp_path}")
                    # File will be cleaned up by OS eventually


class WindowsFFmpegExtractor:
    """
    Windows-safe FFmpeg frame extractor with proper error handling.
    
    This class encapsulates all FFmpeg operations with Windows-specific
    optimizations and error recovery mechanisms.
    """
    
    def __init__(self, timeout: int = 30, verbose: bool = False):
        """
        Initialize the FFmpeg extractor.
        
        Args:
            timeout: Timeout in seconds for FFmpeg operations
            verbose: Enable verbose debug logging
        """
        self.timeout = timeout
        self.verbose = verbose
        self.temp_manager = WindowsTempFileManager(verbose=verbose)
        self._ffmpeg_available = None
    
    def is_ffmpeg_available(self) -> bool:
        """
        Check if FFmpeg is available and working.
        
        Returns:
            True if FFmpeg is available, False otherwise
        """
        if self._ffmpeg_available is not None:
            return self._ffmpeg_available
        
        try:
            result = subprocess.run(
                ['ffmpeg', '-version'],
                capture_output=True,
                text=True,
                timeout=5,
                creationflags=subprocess.CREATE_NO_WINDOW if os.name == 'nt' else 0
            )
            self._ffmpeg_available = result.returncode == 0
            if not self._ffmpeg_available:
                logger.warning("FFmpeg is installed but not working properly")
        except (subprocess.TimeoutExpired, FileNotFoundError, OSError) as e:
            logger.warning(f"FFmpeg not available: {e}")
            self._ffmpeg_available = False
        
        return self._ffmpeg_available
    
    def extract_frame(
        self,
        video_path: Path,
        frame_number: int,
        output_format: str = "jpg"
    ) -> Optional[np.ndarray]:
        """
        Extract a specific frame from video using FFmpeg.
        
        Args:
            video_path: Path to video file
            frame_number: Frame number to extract (1-based)
            output_format: Output image format (jpg, png)
            
        Returns:
            Extracted frame as numpy array (BGR format), or None if failed
            
        Raises:
            FFmpegNotFoundError: If FFmpeg is not available
            FFmpegExtractionError: If frame extraction fails
        """
        if not self.is_ffmpeg_available():
            raise FFmpegNotFoundError("FFmpeg is not available or not working")
        
        if not video_path.exists():
            raise FFmpegExtractionError(f"Video file does not exist: {video_path}")
        
        # Use temp file manager for Windows-safe operation
        with self.temp_manager.create_temp_file() as temp_image_path:
            try:
                # Build FFmpeg command for precise frame extraction
                cmd = self._build_ffmpeg_command(
                    video_path, frame_number, temp_image_path, output_format
                )
                
                # Execute FFmpeg with proper Windows handling
                result = self._execute_ffmpeg_command(cmd)
                
                # Validate and read the extracted frame
                return self._read_extracted_frame(temp_image_path)
                
            except subprocess.TimeoutExpired:
                raise FFmpegExtractionError(
                    f"FFmpeg extraction timed out after {self.timeout} seconds for frame {frame_number}"
                )
            except Exception as e:
                raise FFmpegExtractionError(f"FFmpeg extraction failed: {e}") from e
    
    def _build_ffmpeg_command(
        self,
        video_path: Path,
        frame_number: int,
        output_path: Path,
        output_format: str
    ) -> list[str]:
        """
        Build FFmpeg command for frame extraction.
        
        Args:
            video_path: Input video path
            frame_number: Frame number (1-based)
            output_path: Output image path
            output_format: Output format
            
        Returns:
            FFmpeg command as list of strings
        """
        # Convert 1-based frame number to 0-based for FFmpeg
        frame_index = frame_number - 1 if frame_number >= 1 else 0
        
        cmd = [
            'ffmpeg',
            '-i', str(video_path),
            '-vf', f'select=eq(n\\,{frame_index})',  # Precise frame selection
            '-vframes', '1',  # Extract only one frame
            '-y',  # Overwrite output file if exists
            '-loglevel', 'error',  # Suppress verbose output
            str(output_path)  # Remove explicit format - let FFmpeg infer from extension
        ]
        
        # logger.debug(f"FFmpeg command: {' '.join(cmd)}")
        return cmd
    
    def _execute_ffmpeg_command(self, cmd: list[str]) -> subprocess.CompletedProcess:
        """
        Execute FFmpeg command with Windows-specific settings.
        
        Args:
            cmd: FFmpeg command as list
            
        Returns:
            Completed process result
            
        Raises:
            subprocess.TimeoutExpired: If command times out
            FFmpegExtractionError: If command fails
        """
        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=self.timeout,
                creationflags=subprocess.CREATE_NO_WINDOW if os.name == 'nt' else 0,
                # Ensure proper encoding for Windows
                encoding='utf-8',
                errors='replace'
            )
            
            if result.returncode != 0:
                error_msg = result.stderr.strip() if result.stderr else "Unknown FFmpeg error"
                raise FFmpegExtractionError(
                    f"FFmpeg command failed (exit code {result.returncode}): {error_msg}"
                )
            
            return result
            
        except subprocess.TimeoutExpired:
            logger.warning(f"FFmpeg command timed out after {self.timeout} seconds")
            raise
        except Exception as e:
            raise FFmpegExtractionError(f"Failed to execute FFmpeg command: {e}") from e
    
    def _read_extracted_frame(self, image_path: Path) -> np.ndarray:
        """
        Read and validate extracted frame image.
        
        Args:
            image_path: Path to extracted image
            
        Returns:
            Frame as numpy array (BGR format)
            
        Raises:
            FFmpegExtractionError: If frame reading fails
        """
        # Verify file was created
        if not image_path.exists():
            raise FFmpegExtractionError("FFmpeg did not create output file")
        
        # Check file size
        file_size = image_path.stat().st_size
        if file_size == 0:
            raise FFmpegExtractionError("FFmpeg created empty output file")
        
        # Read image with OpenCV
        frame = cv2.imread(str(image_path))
        if frame is None:
            raise FFmpegExtractionError(f"Failed to read extracted frame from {image_path}")
        
        # Validate frame dimensions
        if frame.shape[0] == 0 or frame.shape[1] == 0:
            raise FFmpegExtractionError("Extracted frame has invalid dimensions")
        
        # logger.debug(f"Successfully extracted frame: {frame.shape} from {image_path} ({file_size} bytes)")
        return frame


class WindowsVideoFrameExtractor:
    """
    High-level video frame extractor with FFmpeg and OpenCV fallback.
    
    This class provides a unified interface for frame extraction with
    automatic fallback from FFmpeg to OpenCV when needed.
    """
    
    def __init__(self, prefer_ffmpeg: bool = True, ffmpeg_timeout: int = 30, verbose: bool = False):
        """
        Initialize the video frame extractor.
        
        Args:
            prefer_ffmpeg: Whether to prefer FFmpeg over OpenCV
            verbose: Enable verbose debug logging
        """
        self.prefer_ffmpeg = prefer_ffmpeg
        self.verbose = verbose
        self.ffmpeg_extractor = WindowsFFmpegExtractor(timeout=ffmpeg_timeout, verbose=verbose)
        self._extraction_stats = {
            'ffmpeg_success': 0,
            'ffmpeg_failures': 0,
            'opencv_success': 0,
            'opencv_failures': 0
        }
    
    def extract_frame(
        self,
        video_path: Path,
        frame_number: int
    ) -> Optional[np.ndarray]:
        """
        Extract frame with automatic fallback strategy.
        
        Args:
            video_path: Path to video file
            frame_number: Frame number to extract (1-based)
            
        Returns:
            Extracted frame as numpy array (BGR format), or None if failed
        """
        if self.prefer_ffmpeg:
            # Try FFmpeg first
            frame = self._try_ffmpeg_extraction(video_path, frame_number)
            if frame is not None:
                return frame
            
            # Fallback to OpenCV
            logger.debug(f"FFmpeg failed for frame {frame_number}, trying OpenCV fallback")
            return self._try_opencv_extraction(video_path, frame_number)
        else:
            # Try OpenCV first
            frame = self._try_opencv_extraction(video_path, frame_number)
            if frame is not None:
                return frame
            
            # Fallback to FFmpeg
            logger.debug(f"OpenCV failed for frame {frame_number}, trying FFmpeg fallback")
            return self._try_ffmpeg_extraction(video_path, frame_number)
    
    def _try_ffmpeg_extraction(self, video_path: Path, frame_number: int) -> Optional[np.ndarray]:
        """Try FFmpeg extraction with error handling."""
        try:
            frame = self.ffmpeg_extractor.extract_frame(video_path, frame_number)
            self._extraction_stats['ffmpeg_success'] += 1
            return frame
        except (FFmpegNotFoundError, FFmpegExtractionError) as e:
            self._extraction_stats['ffmpeg_failures'] += 1
            logger.debug(f"FFmpeg extraction failed for frame {frame_number}: {e}")
            return None
        except Exception as e:
            self._extraction_stats['ffmpeg_failures'] += 1
            logger.warning(f"Unexpected FFmpeg error for frame {frame_number}: {e}")
            return None
    
    def _try_opencv_extraction(self, video_path: Path, frame_number: int) -> Optional[np.ndarray]:
        """Try OpenCV extraction with error handling."""
        try:
            cap = cv2.VideoCapture(str(video_path))
            if not cap.isOpened():
                raise RuntimeError("Failed to open video file")
            
            # Convert 1-based to 0-based frame number
            frame_index = frame_number - 1 if frame_number >= 1 else 0
            
            # Try to seek to frame
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
            
            # Read frame
            ret, frame = cap.read()
            cap.release()
            
            if not ret or frame is None:
                raise RuntimeError("Failed to read frame")
            
            self._extraction_stats['opencv_success'] += 1
            return frame
            
        except Exception as e:
            self._extraction_stats['opencv_failures'] += 1
            logger.debug(f"OpenCV extraction failed for frame {frame_number}: {e}")
            return None
    
    def get_extraction_stats(self) -> Dict[str, Any]:
        """Get extraction statistics."""
        total_attempts = sum(self._extraction_stats.values())
        if total_attempts == 0:
            return self._extraction_stats
        
        stats = self._extraction_stats.copy()
        stats['total_attempts'] = total_attempts
        stats['ffmpeg_success_rate'] = stats['ffmpeg_success'] / total_attempts
        stats['opencv_success_rate'] = stats['opencv_success'] / total_attempts
        stats['overall_success_rate'] = (stats['ffmpeg_success'] + stats['opencv_success']) / total_attempts
        
        return stats