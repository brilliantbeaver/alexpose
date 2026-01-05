"""
Enhanced video processor for AlexPose.

This module provides comprehensive video processing capabilities including
YouTube support, frame extraction, and metadata handling using FFmpeg
with OpenCV fallback.

Author: AlexPose Team
"""

import subprocess
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from loguru import logger

from ambient.core.interfaces import IVideoProcessor
from ambient.core.frame import Frame, FrameSequence
from ambient.utils.video_utils import detect_ffmpeg, get_video_info_ffmpeg, get_video_info_opencv
from ambient.video.youtube_handler import YouTubeHandler

try:
    import cv2
    OPENCV_AVAILABLE = True
except ImportError:
    cv2 = None
    OPENCV_AVAILABLE = False


class VideoProcessor(IVideoProcessor):
    """
    Enhanced video processor with YouTube support and FFmpeg/OpenCV fallback.
    
    This processor can handle local video files and YouTube URLs, automatically
    downloading YouTube videos and extracting frames using the best available
    backend (FFmpeg preferred, OpenCV fallback).
    """
    
    def __init__(
        self,
        youtube_handler: Optional[YouTubeHandler] = None,
        prefer_ffmpeg: bool = True,
        cache_frames: bool = True
    ):
        """
        Initialize video processor.
        
        Args:
            youtube_handler: YouTube handler instance (creates default if None)
            prefer_ffmpeg: Whether to prefer FFmpeg over OpenCV
            cache_frames: Whether to cache extracted frames
        """
        self.youtube_handler = youtube_handler or YouTubeHandler()
        self.prefer_ffmpeg = prefer_ffmpeg
        self.cache_frames = cache_frames
        
        # Detect available backends
        self.ffmpeg_available = detect_ffmpeg()
        self.opencv_available = OPENCV_AVAILABLE
        
        # Determine backend to use
        if self.prefer_ffmpeg and self.ffmpeg_available:
            self.backend = "ffmpeg"
        elif self.opencv_available:
            self.backend = "opencv"
        else:
            raise RuntimeError("No video processing backend available (FFmpeg or OpenCV required)")
        
        logger.info(f"Video processor initialized with backend: {self.backend}")
        if self.ffmpeg_available:
            logger.info("FFmpeg available for video processing")
        if self.opencv_available:
            logger.info("OpenCV available for video processing")
    
    def load_video(self, video_path: Union[str, Path]) -> FrameSequence:
        """
        Load video as a FrameSequence.
        
        Args:
            video_path: Path to video file or YouTube URL
            
        Returns:
            FrameSequence object containing video frames
        """
        # Handle YouTube URLs
        if isinstance(video_path, str) and self._is_youtube_url(video_path):
            logger.info(f"Processing YouTube URL: {video_path}")
            local_path = self.youtube_handler.download_video(video_path)
            if not local_path:
                raise ValueError(f"Failed to download YouTube video: {video_path}")
            video_path = local_path
        
        video_path = Path(video_path)
        
        if not video_path.exists():
            raise FileNotFoundError(f"Video file not found: {video_path}")
        
        # Get video info
        video_info = self.get_video_info(video_path)
        
        # Create FrameSequence with lazy loading
        return FrameSequence.from_video(
            video_path=video_path,
            video_info=video_info,
            backend=self.backend
        )
    
    def extract_frame(self, video_path: Union[str, Path], frame_index: int) -> Frame:
        """
        Extract a single frame from video.
        
        Args:
            video_path: Path to video file or YouTube URL
            frame_index: Frame index to extract (0-based)
            
        Returns:
            Frame object containing the extracted frame
        """
        # Handle YouTube URLs
        if isinstance(video_path, str) and self._is_youtube_url(video_path):
            local_path = self.youtube_handler.download_video(video_path)
            if not local_path:
                raise ValueError(f"Failed to download YouTube video: {video_path}")
            video_path = local_path
        
        video_path = Path(video_path)
        
        if not video_path.exists():
            raise FileNotFoundError(f"Video file not found: {video_path}")
        
        # Extract frame using appropriate backend
        if self.backend == "ffmpeg":
            return self._extract_frame_ffmpeg(video_path, frame_index)
        else:
            return self._extract_frame_opencv(video_path, frame_index)
    
    def get_video_info(self, video_path: Union[str, Path]) -> Dict[str, Any]:
        """
        Get video metadata information.
        
        Args:
            video_path: Path to video file
            
        Returns:
            Dictionary containing video metadata
        """
        # Handle YouTube URLs
        if isinstance(video_path, str) and self._is_youtube_url(video_path):
            local_path = self.youtube_handler.download_video(video_path)
            if not local_path:
                raise ValueError(f"Failed to download YouTube video: {video_path}")
            video_path = local_path
        
        video_path = Path(video_path)
        
        if not video_path.exists():
            raise FileNotFoundError(f"Video file not found: {video_path}")
        
        # Get info using appropriate backend
        if self.backend == "ffmpeg":
            return get_video_info_ffmpeg(video_path)
        else:
            return get_video_info_opencv(video_path)
    
    def _is_youtube_url(self, url: str) -> bool:
        """Check if URL is a YouTube URL."""
        return self.youtube_handler.is_youtube_url(url)
    
    def _extract_frame_ffmpeg(self, video_path: Path, frame_index: int) -> Frame:
        """Extract frame using FFmpeg."""
        try:
            import tempfile
            
            with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as temp_file:
                temp_path = temp_file.name
            
            # Use FFmpeg to extract specific frame
            cmd = [
                "ffmpeg",
                "-i", str(video_path),
                "-vf", f"select=eq(n\\,{frame_index})",
                "-vframes", "1",
                "-y",
                temp_path
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True)
            
            if result.returncode != 0:
                raise RuntimeError(f"FFmpeg failed: {result.stderr}")
            
            # Create Frame from extracted image
            frame = Frame.from_file(temp_path)
            
            # Clean up temporary file
            Path(temp_path).unlink(missing_ok=True)
            
            return frame
            
        except Exception as e:
            logger.error(f"FFmpeg frame extraction failed: {e}")
            # Fallback to OpenCV if available
            if self.opencv_available:
                logger.info("Falling back to OpenCV for frame extraction")
                return self._extract_frame_opencv(video_path, frame_index)
            else:
                raise
    
    def _extract_frame_opencv(self, video_path: Path, frame_index: int) -> Frame:
        """Extract frame using OpenCV."""
        if not self.opencv_available:
            raise RuntimeError("OpenCV not available for frame extraction")
        
        try:
            cap = cv2.VideoCapture(str(video_path))
            
            if not cap.isOpened():
                raise RuntimeError(f"Failed to open video: {video_path}")
            
            # Set frame position
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
            
            # Read frame
            ret, frame_data = cap.read()
            cap.release()
            
            if not ret:
                raise RuntimeError(f"Failed to read frame {frame_index}")
            
            # Create Frame from numpy array
            return Frame.from_array(frame_data)
            
        except Exception as e:
            logger.error(f"OpenCV frame extraction failed: {e}")
            raise
    
    def extract_frames_batch(
        self, 
        video_path: Union[str, Path], 
        frame_indices: List[int]
    ) -> List[Frame]:
        """
        Extract multiple frames efficiently.
        
        Args:
            video_path: Path to video file or YouTube URL
            frame_indices: List of frame indices to extract
            
        Returns:
            List of Frame objects
        """
        frames = []
        
        for frame_index in frame_indices:
            try:
                frame = self.extract_frame(video_path, frame_index)
                frames.append(frame)
            except Exception as e:
                logger.error(f"Failed to extract frame {frame_index}: {e}")
                # Create empty frame as placeholder
                frames.append(Frame.empty())
        
        return frames
    
    def get_frame_count(self, video_path: Union[str, Path]) -> int:
        """Get total number of frames in video."""
        video_info = self.get_video_info(video_path)
        return video_info.get("frame_count", 0)
    
    def get_fps(self, video_path: Union[str, Path]) -> float:
        """Get frames per second of video."""
        video_info = self.get_video_info(video_path)
        return video_info.get("fps", 30.0)
    
    def get_duration(self, video_path: Union[str, Path]) -> float:
        """Get duration of video in seconds."""
        video_info = self.get_video_info(video_path)
        return video_info.get("duration", 0.0)
    
    def is_valid_video(self, video_path: Union[str, Path]) -> bool:
        """Check if video file is valid and readable."""
        try:
            # Handle YouTube URLs
            if isinstance(video_path, str) and self._is_youtube_url(video_path):
                return self.youtube_handler.is_youtube_url(video_path)
            
            video_path = Path(video_path)
            
            if not video_path.exists():
                return False
            
            # Try to get video info
            self.get_video_info(video_path)
            return True
            
        except Exception:
            return False