"""
Video processing utilities for AlexPose.

This module provides video processing utilities with FFmpeg and OpenCV support,
including automatic fallback detection and frame extraction capabilities.

Author: AlexPose Team
"""

import os
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
from loguru import logger

try:
    import cv2
except ImportError:
    cv2 = None


class VideoProcessingError(Exception):
    """Exception raised for video processing errors."""
    pass


class VideoProcessor:
    """
    Video processing utility with FFmpeg and OpenCV support.
    
    Automatically detects available tools and provides fallback mechanisms
    for robust video processing across different environments.
    """
    
    def __init__(self):
        """Initialize video processor with capability detection."""
        self.ffmpeg_available = self._check_ffmpeg_availability()
        self.opencv_available = cv2 is not None
        
        if not self.ffmpeg_available and not self.opencv_available:
            raise VideoProcessingError(
                "Neither FFmpeg nor OpenCV is available. "
                "Please install at least one for video processing."
            )
        
        logger.info(f"Video processor initialized - FFmpeg: {self.ffmpeg_available}, OpenCV: {self.opencv_available}")
    
    def _check_ffmpeg_availability(self) -> bool:
        """Check if FFmpeg is available on the system."""
        try:
            result = subprocess.run(
                ["ffmpeg", "-version"],
                capture_output=True,
                text=True,
                timeout=10
            )
            return result.returncode == 0
        except (subprocess.CalledProcessError, FileNotFoundError, subprocess.TimeoutExpired):
            return False
    
    def get_video_info(self, video_path: Union[str, Path]) -> Dict[str, Union[int, float, str]]:
        """
        Get video information including duration, frame count, resolution, etc.
        
        Args:
            video_path: Path to video file
            
        Returns:
            Dictionary containing video information
        """
        video_path = Path(video_path)
        if not video_path.exists():
            raise VideoProcessingError(f"Video file not found: {video_path}")
        
        # Try FFmpeg first for more accurate information
        if self.ffmpeg_available:
            try:
                return self._get_video_info_ffmpeg(video_path)
            except Exception as e:
                logger.warning(f"FFmpeg failed to get video info: {e}")
        
        # Fallback to OpenCV
        if self.opencv_available:
            try:
                return self._get_video_info_opencv(video_path)
            except Exception as e:
                logger.warning(f"OpenCV failed to get video info: {e}")
        
        raise VideoProcessingError("Could not get video information with available tools")
    
    def _get_video_info_ffmpeg(self, video_path: Path) -> Dict[str, Union[int, float, str]]:
        """Get video info using FFmpeg."""
        cmd = [
            "ffprobe",
            "-v", "quiet",
            "-print_format", "json",
            "-show_format",
            "-show_streams",
            str(video_path)
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
        if result.returncode != 0:
            raise VideoProcessingError(f"FFprobe failed: {result.stderr}")
        
        import json
        probe_data = json.loads(result.stdout)
        
        # Find video stream
        video_stream = None
        for stream in probe_data.get("streams", []):
            if stream.get("codec_type") == "video":
                video_stream = stream
                break
        
        if not video_stream:
            raise VideoProcessingError("No video stream found")
        
        format_info = probe_data.get("format", {})
        
        return {
            "duration": float(format_info.get("duration", 0)),
            "frame_count": int(video_stream.get("nb_frames", 0)),
            "width": int(video_stream.get("width", 0)),
            "height": int(video_stream.get("height", 0)),
            "fps": eval(video_stream.get("r_frame_rate", "0/1")),  # Fraction to float
            "codec": video_stream.get("codec_name", "unknown"),
            "format": format_info.get("format_name", "unknown"),
            "size_bytes": int(format_info.get("size", 0))
        }
    
    def _get_video_info_opencv(self, video_path: Path) -> Dict[str, Union[int, float, str]]:
        """Get video info using OpenCV."""
        cap = cv2.VideoCapture(str(video_path))
        
        if not cap.isOpened():
            raise VideoProcessingError(f"Could not open video: {video_path}")
        
        try:
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            fps = cap.get(cv2.CAP_PROP_FPS)
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            
            duration = frame_count / fps if fps > 0 else 0
            
            return {
                "duration": duration,
                "frame_count": frame_count,
                "width": width,
                "height": height,
                "fps": fps,
                "codec": "unknown",  # OpenCV doesn't provide codec info easily
                "format": "unknown",
                "size_bytes": video_path.stat().st_size
            }
        finally:
            cap.release()
    
    def extract_frame(
        self,
        video_path: Union[str, Path],
        frame_index: int,
        output_path: Optional[Union[str, Path]] = None
    ) -> Union[np.ndarray, Path]:
        """
        Extract a single frame from video.
        
        Args:
            video_path: Path to video file
            frame_index: Frame index to extract (0-based)
            output_path: Optional output path for saving frame
            
        Returns:
            Numpy array if output_path is None, otherwise Path to saved frame
        """
        video_path = Path(video_path)
        if not video_path.exists():
            raise VideoProcessingError(f"Video file not found: {video_path}")
        
        # Try FFmpeg first
        if self.ffmpeg_available:
            try:
                return self._extract_frame_ffmpeg(video_path, frame_index, output_path)
            except Exception as e:
                logger.warning(f"FFmpeg frame extraction failed: {e}")
        
        # Fallback to OpenCV
        if self.opencv_available:
            try:
                return self._extract_frame_opencv(video_path, frame_index, output_path)
            except Exception as e:
                logger.warning(f"OpenCV frame extraction failed: {e}")
        
        raise VideoProcessingError("Could not extract frame with available tools")
    
    def _extract_frame_ffmpeg(
        self,
        video_path: Path,
        frame_index: int,
        output_path: Optional[Path]
    ) -> Union[np.ndarray, Path]:
        """Extract frame using FFmpeg."""
        if output_path is None:
            # Create temporary file
            temp_file = tempfile.NamedTemporaryFile(suffix=".jpg", delete=False)
            output_path = Path(temp_file.name)
            temp_file.close()
            cleanup_temp = True
        else:
            output_path = Path(output_path)
            cleanup_temp = False
        
        try:
            # FFmpeg command to extract specific frame
            cmd = [
                "ffmpeg",
                "-v", "error",
                "-i", str(video_path),
                "-vf", f"select=eq(n\\,{frame_index})",
                "-frames:v", "1",
                "-y",
                str(output_path)
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
            if result.returncode != 0:
                raise VideoProcessingError(f"FFmpeg error: {result.stderr}")
            
            if not output_path.exists() or output_path.stat().st_size == 0:
                raise VideoProcessingError("FFmpeg did not create output file")
            
            # If no output path was specified, load and return array
            if cleanup_temp:
                try:
                    if cv2 is not None:
                        frame = cv2.imread(str(output_path))
                        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    else:
                        from PIL import Image
                        with Image.open(output_path) as img:
                            frame = np.array(img.convert("RGB"))
                    return frame
                finally:
                    output_path.unlink()  # Clean up temp file
            else:
                return output_path
                
        except Exception as e:
            if cleanup_temp and output_path.exists():
                output_path.unlink()
            raise e
    
    def _extract_frame_opencv(
        self,
        video_path: Path,
        frame_index: int,
        output_path: Optional[Path]
    ) -> Union[np.ndarray, Path]:
        """Extract frame using OpenCV."""
        cap = cv2.VideoCapture(str(video_path))
        
        if not cap.isOpened():
            raise VideoProcessingError(f"Could not open video: {video_path}")
        
        try:
            # Set frame position
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
            
            ret, frame = cap.read()
            if not ret or frame is None:
                raise VideoProcessingError(f"Could not read frame {frame_index}")
            
            # Convert BGR to RGB
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Save if output path specified
            if output_path is not None:
                output_path = Path(output_path)
                output_path.parent.mkdir(parents=True, exist_ok=True)
                
                # Convert back to BGR for saving with OpenCV
                cv2.imwrite(str(output_path), frame)
                return output_path
            else:
                return frame_rgb
                
        finally:
            cap.release()
    
    def extract_frames_batch(
        self,
        video_path: Union[str, Path],
        frame_indices: List[int],
        output_dir: Optional[Union[str, Path]] = None
    ) -> Union[List[np.ndarray], List[Path]]:
        """
        Extract multiple frames from video.
        
        Args:
            video_path: Path to video file
            frame_indices: List of frame indices to extract
            output_dir: Optional output directory for saving frames
            
        Returns:
            List of numpy arrays or paths to saved frames
        """
        video_path = Path(video_path)
        if not video_path.exists():
            raise VideoProcessingError(f"Video file not found: {video_path}")
        
        if output_dir is not None:
            output_dir = Path(output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
        
        results = []
        
        # For efficiency, use OpenCV for batch extraction if available
        if self.opencv_available:
            try:
                results = self._extract_frames_batch_opencv(video_path, frame_indices, output_dir)
            except Exception as e:
                logger.warning(f"OpenCV batch extraction failed: {e}")
                results = []
        
        # Fallback to individual FFmpeg extractions
        if not results and self.ffmpeg_available:
            for i, frame_idx in enumerate(frame_indices):
                try:
                    if output_dir is not None:
                        output_path = output_dir / f"frame_{frame_idx:06d}.jpg"
                    else:
                        output_path = None
                    
                    result = self._extract_frame_ffmpeg(video_path, frame_idx, output_path)
                    results.append(result)
                except Exception as e:
                    logger.error(f"Failed to extract frame {frame_idx}: {e}")
                    raise
        
        if not results:
            raise VideoProcessingError("Could not extract frames with available tools")
        
        return results
    
    def _extract_frames_batch_opencv(
        self,
        video_path: Path,
        frame_indices: List[int],
        output_dir: Optional[Path]
    ) -> Union[List[np.ndarray], List[Path]]:
        """Extract multiple frames using OpenCV."""
        cap = cv2.VideoCapture(str(video_path))
        
        if not cap.isOpened():
            raise VideoProcessingError(f"Could not open video: {video_path}")
        
        results = []
        
        try:
            # Sort indices for efficient sequential reading
            sorted_indices = sorted(enumerate(frame_indices), key=lambda x: x[1])
            
            current_frame = 0
            for original_idx, frame_idx in sorted_indices:
                # Seek to frame if needed
                if frame_idx != current_frame:
                    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
                    current_frame = frame_idx
                
                ret, frame = cap.read()
                if not ret or frame is None:
                    raise VideoProcessingError(f"Could not read frame {frame_idx}")
                
                # Convert BGR to RGB
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                
                if output_dir is not None:
                    output_path = output_dir / f"frame_{frame_idx:06d}.jpg"
                    cv2.imwrite(str(output_path), frame)  # Save as BGR
                    results.append((original_idx, output_path))
                else:
                    results.append((original_idx, frame_rgb))
                
                current_frame += 1
            
            # Sort results back to original order
            results.sort(key=lambda x: x[0])
            return [result[1] for result in results]
            
        finally:
            cap.release()
    
    def convert_video(
        self,
        input_path: Union[str, Path],
        output_path: Union[str, Path],
        target_format: str = "mp4",
        target_resolution: Optional[Tuple[int, int]] = None,
        target_fps: Optional[float] = None
    ) -> Path:
        """
        Convert video to different format/resolution/fps.
        
        Args:
            input_path: Input video path
            output_path: Output video path
            target_format: Target format (mp4, avi, etc.)
            target_resolution: Target resolution as (width, height)
            target_fps: Target frame rate
            
        Returns:
            Path to converted video
        """
        input_path = Path(input_path)
        output_path = Path(output_path)
        
        if not input_path.exists():
            raise VideoProcessingError(f"Input video not found: {input_path}")
        
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        if self.ffmpeg_available:
            return self._convert_video_ffmpeg(
                input_path, output_path, target_format, target_resolution, target_fps
            )
        elif self.opencv_available:
            return self._convert_video_opencv(
                input_path, output_path, target_format, target_resolution, target_fps
            )
        else:
            raise VideoProcessingError("No video conversion tools available")
    
    def _convert_video_ffmpeg(
        self,
        input_path: Path,
        output_path: Path,
        target_format: str,
        target_resolution: Optional[Tuple[int, int]],
        target_fps: Optional[float]
    ) -> Path:
        """Convert video using FFmpeg."""
        cmd = ["ffmpeg", "-v", "error", "-i", str(input_path)]
        
        # Add resolution filter if specified
        filters = []
        if target_resolution:
            filters.append(f"scale={target_resolution[0]}:{target_resolution[1]}")
        
        if target_fps:
            filters.append(f"fps={target_fps}")
        
        if filters:
            cmd.extend(["-vf", ",".join(filters)])
        
        # Add codec based on format
        if target_format.lower() == "mp4":
            cmd.extend(["-c:v", "libx264", "-c:a", "aac"])
        elif target_format.lower() == "avi":
            cmd.extend(["-c:v", "libxvid", "-c:a", "mp3"])
        
        cmd.extend(["-y", str(output_path)])
        
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
        if result.returncode != 0:
            raise VideoProcessingError(f"FFmpeg conversion failed: {result.stderr}")
        
        return output_path
    
    def _convert_video_opencv(
        self,
        input_path: Path,
        output_path: Path,
        target_format: str,
        target_resolution: Optional[Tuple[int, int]],
        target_fps: Optional[float]
    ) -> Path:
        """Convert video using OpenCV (limited functionality)."""
        cap = cv2.VideoCapture(str(input_path))
        
        if not cap.isOpened():
            raise VideoProcessingError(f"Could not open input video: {input_path}")
        
        try:
            # Get original properties
            original_fps = cap.get(cv2.CAP_PROP_FPS)
            original_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            original_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            
            # Set target properties
            fps = target_fps if target_fps else original_fps
            width, height = target_resolution if target_resolution else (original_width, original_height)
            
            # Define codec
            fourcc = cv2.VideoWriter_fourcc(*'mp4v') if target_format.lower() == 'mp4' else cv2.VideoWriter_fourcc(*'XVID')
            
            # Create writer
            out = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))
            
            if not out.isOpened():
                raise VideoProcessingError("Could not create output video writer")
            
            try:
                while True:
                    ret, frame = cap.read()
                    if not ret:
                        break
                    
                    # Resize if needed
                    if (width, height) != (original_width, original_height):
                        frame = cv2.resize(frame, (width, height))
                    
                    out.write(frame)
                
                return output_path
                
            finally:
                out.release()
                
        finally:
            cap.release()


# Global video processor instance
_video_processor = None


def get_video_processor() -> VideoProcessor:
    """Get global video processor instance."""
    global _video_processor
    if _video_processor is None:
        _video_processor = VideoProcessor()
    return _video_processor


def is_ffmpeg_available() -> bool:
    """Check if FFmpeg is available."""
    return get_video_processor().ffmpeg_available


def is_opencv_available() -> bool:
    """Check if OpenCV is available."""
    return get_video_processor().opencv_available


def detect_ffmpeg() -> bool:
    """Detect if FFmpeg is available on the system."""
    return get_video_processor().ffmpeg_available


def get_video_info_ffmpeg(video_path: Union[str, Path]) -> Dict[str, Union[int, float, str]]:
    """Get video info using FFmpeg."""
    return get_video_processor()._get_video_info_ffmpeg(Path(video_path))


def get_video_info_opencv(video_path: Union[str, Path]) -> Dict[str, Union[int, float, str]]:
    """Get video info using OpenCV."""
    return get_video_processor()._get_video_info_opencv(Path(video_path))