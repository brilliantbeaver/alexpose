"""
Frame and FrameSequence data models for AlexPose.

This module provides flexible data models for representing video frames and sequences,
supporting multiple data sources and formats with lazy loading and memory management.

Key Features:
- Multiple data source support (files, videos, URLs, numpy arrays)
- Lazy loading for memory efficiency
- Format conversion (RGB, BGR, grayscale)
- Metadata support and error handling
- FFmpeg with OpenCV fallback for video processing

Author: AlexPose Team
"""

import os
import shutil
import subprocess
import tempfile
import threading
import time
import weakref
from collections import OrderedDict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import psutil
from loguru import logger

try:
    import cv2
except ImportError:
    cv2 = None

try:
    from PIL import Image
except ImportError:
    Image = None

try:
    import requests
except ImportError:
    requests = None


class FrameError(Exception):
    """Exception raised for Frame-related errors."""
    pass


class MemoryManager:
    """
    Global memory manager for Frame objects with LRU cache and automatic cleanup.
    
    This singleton class manages memory usage across all Frame objects,
    providing automatic cleanup when memory usage exceeds thresholds.
    """
    
    _instance = None
    _lock = threading.Lock()
    
    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._initialized = False
        return cls._instance
    
    def __init__(self):
        if self._initialized:
            return
        
        self._initialized = True
        self._loaded_frames = OrderedDict()  # LRU cache of loaded frames
        self._frame_registry = weakref.WeakSet()  # All frame instances
        self._memory_threshold_mb = 1024  # 1GB default threshold
        self._max_loaded_frames = 100  # Maximum frames to keep loaded
        self._cleanup_lock = threading.Lock()
        self._monitoring_enabled = True
        
        # Start memory monitoring thread
        self._monitor_thread = threading.Thread(target=self._memory_monitor, daemon=True)
        self._monitor_thread.start()
    
    def register_frame(self, frame: 'Frame') -> None:
        """Register a frame instance for memory management."""
        self._frame_registry.add(frame)
    
    def mark_frame_loaded(self, frame: 'Frame', data_size_mb: float) -> None:
        """Mark a frame as loaded and add to LRU cache."""
        frame_id = id(frame)
        
        with self._cleanup_lock:
            # Remove if already exists (to update position)
            if frame_id in self._loaded_frames:
                del self._loaded_frames[frame_id]
            
            # Add to end (most recently used)
            self._loaded_frames[frame_id] = {
                'frame': frame,
                'size_mb': data_size_mb,
                'load_time': time.time()
            }
            
            # Check if we need to cleanup
            self._check_memory_limits()
    
    def mark_frame_unloaded(self, frame: 'Frame') -> None:
        """Mark a frame as unloaded and remove from cache."""
        frame_id = id(frame)
        with self._cleanup_lock:
            if frame_id in self._loaded_frames:
                del self._loaded_frames[frame_id]
    
    def _check_memory_limits(self) -> None:
        """Check memory limits and cleanup if necessary."""
        # Check frame count limit
        while len(self._loaded_frames) > self._max_loaded_frames:
            self._unload_oldest_frame()
        
        # Check memory usage limit
        total_memory_mb = sum(info['size_mb'] for info in self._loaded_frames.values())
        if total_memory_mb > self._memory_threshold_mb:
            logger.info(f"Memory usage ({total_memory_mb:.1f}MB) exceeds threshold ({self._memory_threshold_mb}MB), cleaning up")
            self._cleanup_excess_memory()
    
    def _unload_oldest_frame(self) -> None:
        """Unload the least recently used frame."""
        if not self._loaded_frames:
            return
        
        # Get oldest frame (first in OrderedDict)
        frame_id, frame_info = next(iter(self._loaded_frames.items()))
        frame = frame_info['frame']
        
        try:
            if hasattr(frame, '_force_unload'):
                frame._force_unload()
                logger.debug(f"Automatically unloaded frame {frame_id} to free memory")
        except Exception as e:
            logger.warning(f"Failed to unload frame {frame_id}: {e}")
        
        del self._loaded_frames[frame_id]
    
    def _cleanup_excess_memory(self) -> None:
        """Cleanup frames until memory usage is under threshold."""
        target_memory = self._memory_threshold_mb * 0.8  # Clean up to 80% of threshold
        
        while self._loaded_frames:
            total_memory_mb = sum(info['size_mb'] for info in self._loaded_frames.values())
            if total_memory_mb <= target_memory:
                break
            self._unload_oldest_frame()
    
    def _memory_monitor(self) -> None:
        """Background thread to monitor system memory usage."""
        while self._monitoring_enabled:
            try:
                # Check system memory usage
                memory = psutil.virtual_memory()
                if memory.percent > 85:  # System memory usage > 85%
                    logger.warning(f"High system memory usage: {memory.percent:.1f}%")
                    with self._cleanup_lock:
                        self._cleanup_excess_memory()
                
                time.sleep(30)  # Check every 30 seconds
            except Exception as e:
                logger.error(f"Memory monitoring error: {e}")
                time.sleep(60)  # Wait longer on error
    
    def get_memory_stats(self) -> Dict[str, Any]:
        """Get current memory usage statistics."""
        with self._cleanup_lock:
            total_frames = len(self._loaded_frames)
            total_memory_mb = sum(info['size_mb'] for info in self._loaded_frames.values())
            
            return {
                'loaded_frames': total_frames,
                'total_memory_mb': total_memory_mb,
                'memory_threshold_mb': self._memory_threshold_mb,
                'max_loaded_frames': self._max_loaded_frames,
                'system_memory_percent': psutil.virtual_memory().percent
            }
    
    def set_memory_threshold(self, threshold_mb: int) -> None:
        """Set memory threshold for automatic cleanup."""
        self._memory_threshold_mb = threshold_mb
        logger.info(f"Memory threshold set to {threshold_mb}MB")
    
    def set_max_loaded_frames(self, max_frames: int) -> None:
        """Set maximum number of frames to keep loaded."""
        self._max_loaded_frames = max_frames
        logger.info(f"Maximum loaded frames set to {max_frames}")
    
    def force_cleanup(self) -> None:
        """Force cleanup of all loaded frames."""
        with self._cleanup_lock:
            frames_to_unload = list(self._loaded_frames.values())
            for frame_info in frames_to_unload:
                try:
                    frame = frame_info['frame']
                    if hasattr(frame, '_force_unload'):
                        frame._force_unload()
                except Exception as e:
                    logger.warning(f"Failed to force unload frame: {e}")
            
            self._loaded_frames.clear()
            logger.info("Forced cleanup of all loaded frames")
    
    def shutdown(self) -> None:
        """Shutdown memory manager and cleanup resources."""
        self._monitoring_enabled = False
        self.force_cleanup()


# Global memory manager instance
_memory_manager = MemoryManager()


class Frame:
    """
    Flexible frame data model supporting multiple data sources and formats.
    
    This class provides a unified interface for working with image/video frames
    from various sources while maintaining metadata and supporting advanced
    lazy loading and memory management.
    """
    
    def __init__(
        self,
        data: Optional[Union[np.ndarray, str, Path]] = None,
        source_type: str = "array",
        metadata: Optional[Dict[str, Any]] = None,
        format: str = "RGB",
        lazy_load: bool = False
    ):
        """
        Initialize a Frame object.
        
        Args:
            data: Frame data (numpy array, file path, or URL)
            source_type: Type of data source ("array", "file", "url", "video")
            metadata: Optional metadata dictionary
            format: Color format ("RGB", "BGR", "GRAY")
            lazy_load: Whether to defer loading until needed
        """
        self.data = data
        self.source_type = source_type
        self.metadata = metadata or {}
        self.format = format
        self.lazy_load = lazy_load
        self._loaded_data: Optional[np.ndarray] = None
        self._is_loaded = False
        self._data_size_mb = 0.0
        self._last_access_time = time.time()
        self._access_count = 0
        
        # Register with memory manager
        _memory_manager.register_frame(self)
        
        # Store original data reference for lazy loading
        if not lazy_load and isinstance(data, np.ndarray):
            self._loaded_data = data.copy()
            self._is_loaded = True
            self._access_count = 1  # Count the initial load
            self._calculate_memory_usage()
            _memory_manager.mark_frame_loaded(self, self._data_size_mb)
    
    @classmethod
    def from_file(
        cls, 
        file_path: Union[str, Path], 
        format: str = "RGB",
        lazy_load: bool = True
    ) -> "Frame":
        """
        Create a Frame from an image file.
        
        Args:
            file_path: Path to image file
            format: Desired color format
            lazy_load: Whether to defer loading
            
        Returns:
            Frame object
        """
        path = Path(file_path)
        if not path.exists():
            raise FrameError(f"File not found: {file_path}")
        
        metadata = {
            "source_path": str(path),
            "file_size": path.stat().st_size,
            "file_name": path.name
        }
        
        return cls(
            data=str(path),
            source_type="file",
            metadata=metadata,
            format=format,
            lazy_load=lazy_load
        )
    
    @classmethod
    def from_url(
        cls, 
        url: str, 
        format: str = "RGB",
        lazy_load: bool = True,
        timeout: int = 30
    ) -> "Frame":
        """
        Create a Frame from a URL.
        
        Args:
            url: URL to image resource
            format: Desired color format
            lazy_load: Whether to defer loading
            timeout: Request timeout in seconds
            
        Returns:
            Frame object
        """
        if not url.startswith(('http://', 'https://')):
            raise FrameError(f"Invalid URL format: {url}")
        
        metadata = {
            "source_url": url,
            "timeout": timeout
        }
        
        return cls(
            data=url,
            source_type="url",
            metadata=metadata,
            format=format,
            lazy_load=lazy_load
        )
    
    @classmethod
    def from_video(
        cls,
        video_path: Union[str, Path],
        frame_index: int,
        format: str = "RGB",
        lazy_load: bool = True
    ) -> "Frame":
        """
        Create a Frame from a video file at specific frame index.
        
        Args:
            video_path: Path to video file
            frame_index: Frame index to extract (0-based)
            format: Desired color format
            lazy_load: Whether to defer loading
            
        Returns:
            Frame object
        """
        path = Path(video_path)
        if not path.exists():
            raise FrameError(f"Video file not found: {video_path}")
        
        metadata = {
            "source_path": str(path),
            "frame_index": frame_index,
            "file_name": path.name
        }
        
        return cls(
            data={"video_path": str(path), "frame_index": frame_index},
            source_type="video",
            metadata=metadata,
            format=format,
            lazy_load=lazy_load
        )
    
    @classmethod
    def from_array(
        cls,
        array: np.ndarray,
        format: str = "RGB",
        metadata: Optional[Dict[str, Any]] = None
    ) -> "Frame":
        """
        Create a Frame from a numpy array.
        
        Args:
            array: Numpy array containing image data
            format: Color format of the array
            metadata: Optional metadata
            
        Returns:
            Frame object
        """
        if not isinstance(array, np.ndarray):
            raise FrameError("Data must be a numpy array")
        
        if len(array.shape) not in [2, 3]:
            raise FrameError("Array must be 2D (grayscale) or 3D (color)")
        
        frame_metadata = {
            "shape": array.shape,
            "dtype": str(array.dtype),
            "source": "array"
        }
        if metadata:
            frame_metadata.update(metadata)
        
        return cls(
            data=array,
            source_type="array",
            metadata=frame_metadata,
            format=format,
            lazy_load=False
        )
    
    @classmethod
    def empty(cls, shape: Tuple[int, ...] = (480, 640, 3), format: str = "RGB") -> "Frame":
        """
        Create an empty frame with specified shape.
        
        Args:
            shape: Frame shape (height, width, channels)
            format: Color format
            
        Returns:
            Empty Frame object
        """
        empty_data = np.zeros(shape, dtype=np.uint8)
        return cls(
            data=empty_data,
            source_type="array",
            format=format,
            lazy_load=False,
            metadata={"empty": True, "shape": shape}
        )
    
    def load(self) -> np.ndarray:
        """
        Load frame data into memory if not already loaded.
        
        Returns:
            Numpy array containing frame data
        """
        self._last_access_time = time.time()
        self._access_count += 1
        
        if self._is_loaded and self._loaded_data is not None:
            # Update LRU position
            _memory_manager.mark_frame_loaded(self, self._data_size_mb)
            return self._loaded_data
        
        try:
            if self.source_type == "array":
                self._loaded_data = self.data.copy() if isinstance(self.data, np.ndarray) else None
            elif self.source_type == "file":
                self._loaded_data = self._load_from_file(self.data)
            elif self.source_type == "video":
                self._loaded_data = self._load_from_video(self.data)
            elif self.source_type == "url":
                self._loaded_data = self._load_from_url(self.data)
            else:
                raise FrameError(f"Unsupported source type: {self.source_type}")
            
            if self._loaded_data is None:
                raise FrameError("Failed to load frame data")
            
            # Convert format if needed
            self._loaded_data = self._convert_format(self._loaded_data, self.format)
            self._is_loaded = True
            
            # Calculate memory usage and register with memory manager
            self._calculate_memory_usage()
            _memory_manager.mark_frame_loaded(self, self._data_size_mb)
            
            return self._loaded_data
            
        except Exception as e:
            logger.error(f"Failed to load frame: {e}")
            raise FrameError(f"Failed to load frame: {e}")
    
    def _calculate_memory_usage(self) -> None:
        """Calculate memory usage of loaded data."""
        if self._loaded_data is not None:
            # Calculate size in MB
            size_bytes = self._loaded_data.nbytes
            self._data_size_mb = size_bytes / (1024 * 1024)
        else:
            self._data_size_mb = 0.0
    
    def _load_from_file(self, file_path: str) -> Optional[np.ndarray]:
        """Load image data from file using available libraries."""
        path = Path(file_path)
        
        # Try OpenCV first if available
        if cv2 is not None:
            try:
                img = cv2.imread(str(path))
                if img is not None:
                    # OpenCV loads as BGR by default
                    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            except Exception as e:
                logger.warning(f"OpenCV failed to load {path}: {e}")
        
        # Try PIL as fallback
        if Image is not None:
            try:
                with Image.open(path) as img:
                    return np.array(img.convert("RGB"))
            except Exception as e:
                logger.warning(f"PIL failed to load {path}: {e}")
        
        raise FrameError(f"No suitable image library available to load {path}")
    
    def _load_from_video(self, video_data: Dict[str, Any]) -> Optional[np.ndarray]:
        """Load frame from video using FFmpeg with OpenCV fallback."""
        video_path = video_data["video_path"]
        frame_index = video_data["frame_index"]
        
        # Try FFmpeg first
        if self._is_ffmpeg_available():
            try:
                return self._extract_frame_ffmpeg(video_path, frame_index)
            except Exception as e:
                logger.warning(f"FFmpeg failed to extract frame: {e}")
        
        # Fallback to OpenCV
        if cv2 is not None:
            try:
                return self._extract_frame_opencv(video_path, frame_index)
            except Exception as e:
                logger.warning(f"OpenCV failed to extract frame: {e}")
        
        raise FrameError("No suitable video processing library available")
    
    def _load_from_url(self, url: str) -> Optional[np.ndarray]:
        """Load image data from URL."""
        if requests is None:
            raise FrameError("requests library not available for URL loading")
        
        try:
            timeout = self.metadata.get("timeout", 30)
            response = requests.get(url, timeout=timeout, stream=True)
            response.raise_for_status()
            
            # Check content type
            content_type = response.headers.get('content-type', '').lower()
            if not any(img_type in content_type for img_type in ['image/', 'jpeg', 'png', 'gif', 'bmp']):
                logger.warning(f"URL content type may not be an image: {content_type}")
            
            # Load image from response content
            if Image is not None:
                try:
                    from io import BytesIO
                    img = Image.open(BytesIO(response.content))
                    return np.array(img.convert("RGB"))
                except Exception as e:
                    logger.warning(f"PIL failed to load from URL {url}: {e}")
            
            # Fallback: save to temp file and use OpenCV
            if cv2 is not None:
                try:
                    with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as temp_file:
                        temp_file.write(response.content)
                        temp_path = temp_file.name
                    
                    try:
                        img = cv2.imread(temp_path)
                        if img is not None:
                            return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    finally:
                        try:
                            os.unlink(temp_path)
                        except OSError:
                            pass
                except Exception as e:
                    logger.warning(f"OpenCV failed to load from URL {url}: {e}")
            
            raise FrameError(f"No suitable image library available to load from URL: {url}")
            
        except requests.RequestException as e:
            raise FrameError(f"Failed to download image from URL {url}: {e}")
        except Exception as e:
            raise FrameError(f"Failed to load image from URL {url}: {e}")
    
    def _is_ffmpeg_available(self) -> bool:
        """Check if FFmpeg is available on the system."""
        try:
            subprocess.run(
                ["ffmpeg", "-version"],
                capture_output=True,
                check=True,
                timeout=5
            )
            return True
        except (subprocess.CalledProcessError, FileNotFoundError, subprocess.TimeoutExpired):
            return False
    
    def _extract_frame_ffmpeg(self, video_path: str, frame_index: int) -> np.ndarray:
        """Extract frame using FFmpeg."""
        with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as temp_file:
            temp_path = temp_file.name
        
        try:
            # FFmpeg command to extract specific frame
            cmd = [
                "ffmpeg",
                "-v", "error",
                "-i", video_path,
                "-vf", f"select=eq(n\\,{frame_index})",
                "-frames:v", "1",
                "-y",
                temp_path
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
            if result.returncode != 0:
                raise FrameError(f"FFmpeg error: {result.stderr}")
            
            # Load the extracted frame
            if not Path(temp_path).exists():
                raise FrameError("FFmpeg did not create output file")
            
            return self._load_from_file(temp_path)
            
        finally:
            # Clean up temporary file
            try:
                os.unlink(temp_path)
            except OSError:
                pass
    
    def _extract_frame_opencv(self, video_path: str, frame_index: int) -> np.ndarray:
        """Extract frame using OpenCV."""
        cap = cv2.VideoCapture(video_path)
        try:
            if not cap.isOpened():
                raise FrameError(f"Could not open video: {video_path}")
            
            # Set frame position
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
            
            ret, frame = cap.read()
            if not ret or frame is None:
                raise FrameError(f"Could not read frame {frame_index}")
            
            # Convert BGR to RGB
            return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
        finally:
            cap.release()
    
    def _convert_format(self, data: np.ndarray, target_format: str) -> np.ndarray:
        """Convert frame data to target format."""
        if target_format == self.format:
            return data
        
        if cv2 is None:
            logger.warning("OpenCV not available for format conversion")
            return data
        
        try:
            if self.format == "RGB" and target_format == "BGR":
                return cv2.cvtColor(data, cv2.COLOR_RGB2BGR)
            elif self.format == "BGR" and target_format == "RGB":
                return cv2.cvtColor(data, cv2.COLOR_BGR2RGB)
            elif target_format == "GRAY":
                if self.format == "RGB":
                    return cv2.cvtColor(data, cv2.COLOR_RGB2GRAY)
                elif self.format == "BGR":
                    return cv2.cvtColor(data, cv2.COLOR_BGR2GRAY)
            
            logger.warning(f"Unsupported format conversion: {self.format} -> {target_format}")
            return data
            
        except Exception as e:
            logger.error(f"Format conversion failed: {e}")
            return data
    
    @property
    def shape(self) -> Optional[Tuple[int, ...]]:
        """Get frame shape without loading if possible."""
        if self._is_loaded and self._loaded_data is not None:
            return self._loaded_data.shape
        
        # Try to get shape from metadata
        if "shape" in self.metadata:
            return tuple(self.metadata["shape"])
        
        return None
    
    @property
    def is_loaded(self) -> bool:
        """Check if frame data is loaded in memory."""
        return self._is_loaded
    
    @property
    def memory_usage_mb(self) -> float:
        """Get memory usage in MB."""
        return self._data_size_mb
    
    @property
    def access_stats(self) -> Dict[str, Any]:
        """Get access statistics for this frame."""
        return {
            'access_count': self._access_count,
            'last_access_time': self._last_access_time,
            'memory_usage_mb': self._data_size_mb,
            'is_loaded': self._is_loaded
        }
    
    def unload(self) -> None:
        """Unload frame data from memory to save space."""
        if self.lazy_load and self._is_loaded:
            self._loaded_data = None
            self._is_loaded = False
            _memory_manager.mark_frame_unloaded(self)
            logger.debug(f"Unloaded frame data, freed {self._data_size_mb:.2f}MB")
            self._data_size_mb = 0.0
    
    def _force_unload(self) -> None:
        """Force unload frame data (called by memory manager)."""
        if self._is_loaded:
            self._loaded_data = None
            self._is_loaded = False
            self._data_size_mb = 0.0
    
    def to_array(self) -> np.ndarray:
        """Get frame data as numpy array."""
        return self.load()
    
    def save(self, output_path: Union[str, Path], format: Optional[str] = None) -> None:
        """
        Save frame to file.
        
        Args:
            output_path: Output file path
            format: Optional format override
        """
        data = self.load()
        path = Path(output_path)
        
        if cv2 is not None:
            # Convert to BGR for OpenCV
            if self.format == "RGB":
                save_data = cv2.cvtColor(data, cv2.COLOR_RGB2BGR)
            else:
                save_data = data
            
            cv2.imwrite(str(path), save_data)
        elif Image is not None:
            # Use PIL
            if len(data.shape) == 3:
                img = Image.fromarray(data, mode="RGB")
            else:
                img = Image.fromarray(data, mode="L")
            img.save(path)
        else:
            raise FrameError("No suitable image library available for saving")
    
    def copy(self) -> "Frame":
        """Create a copy of this frame."""
        if self._is_loaded:
            return Frame.from_array(
                self._loaded_data.copy(),
                format=self.format,
                metadata=self.metadata.copy()
            )
        else:
            return Frame(
                data=self.data,
                source_type=self.source_type,
                metadata=self.metadata.copy(),
                format=self.format,
                lazy_load=self.lazy_load
            )
    
    def __del__(self):
        """Cleanup when frame is garbage collected."""
        try:
            if self._is_loaded:
                _memory_manager.mark_frame_unloaded(self)
        except Exception:
            pass  # Ignore errors during cleanup
    
    def resize(self, width: int, height: int) -> "Frame":
        """
        Resize frame to specified dimensions.
        
        Args:
            width: Target width
            height: Target height
            
        Returns:
            New Frame with resized data
        """
        data = self.load()
        
        if cv2 is not None:
            resized = cv2.resize(data, (width, height))
        elif Image is not None:
            img = Image.fromarray(data)
            resized = np.array(img.resize((width, height)))
        else:
            raise FrameError("No suitable image library available for resizing")
        
        metadata = self.metadata.copy()
        metadata.update({
            "original_shape": data.shape,
            "resized_to": (height, width, data.shape[2] if len(data.shape) == 3 else 1)
        })
        
        return Frame.from_array(resized, format=self.format, metadata=metadata)


class FrameSequence:
    """
    Container for sequences of frames with temporal information.
    
    This class manages collections of frames with support for lazy loading,
    temporal metadata, and efficient memory management including batch operations
    and automatic memory optimization.
    """
    
    def __init__(
        self,
        frames: Optional[List[Frame]] = None,
        metadata: Optional[Dict[str, Any]] = None,
        sequence_id: Optional[str] = None,
        lazy_load: bool = True
    ):
        """
        Initialize a FrameSequence.
        
        Args:
            frames: List of Frame objects
            metadata: Sequence-level metadata
            sequence_id: Optional sequence identifier
            lazy_load: Whether to use lazy loading for frames
        """
        self.frames = frames or []
        self.metadata = metadata or {}
        self.sequence_id = sequence_id
        self.lazy_load = lazy_load
        self._batch_size = 10  # Default batch size for operations
        self._preload_window = 3  # Number of frames to preload around current position
        self._current_position = 0
    
    @classmethod
    def from_video(
        cls,
        video_path: Union[str, Path],
        start_frame: int = 0,
        end_frame: Optional[int] = None,
        step: int = 1,
        format: str = "RGB",
        lazy_load: bool = True,
        video_info: Optional[Dict[str, Any]] = None,
        backend: str = "opencv"
    ) -> "FrameSequence":
        """
        Create a FrameSequence from a video file.
        
        Args:
            video_path: Path to video file
            start_frame: Starting frame index
            end_frame: Ending frame index (None for all frames)
            step: Frame step size
            format: Color format
            lazy_load: Whether to use lazy loading
            video_info: Optional video metadata
            backend: Backend to use ("ffmpeg" or "opencv")
            
        Returns:
            FrameSequence object
        """
        path = Path(video_path)
        if not path.exists():
            raise FrameError(f"Video file not found: {video_path}")
        
        # Get video info
        if video_info:
            total_frames = video_info.get("frame_count", 1000)
        else:
            total_frames = cls._get_video_frame_count(video_path)
        
        if end_frame is None:
            end_frame = total_frames
        
        # Create frames
        frames = []
        for frame_idx in range(start_frame, min(end_frame, total_frames), step):
            frame = Frame.from_video(
                video_path, frame_idx, format=format, lazy_load=lazy_load
            )
            frames.append(frame)
        
        metadata = {
            "source_video": str(path),
            "total_frames": total_frames,
            "start_frame": start_frame,
            "end_frame": end_frame,
            "step": step,
            "backend": backend
        }
        
        if video_info:
            metadata.update(video_info)
        
        return cls(frames=frames, metadata=metadata)
    
    @staticmethod
    def _get_video_frame_count(video_path: Union[str, Path]) -> int:
        """Get total frame count from video."""
        if cv2 is not None:
            try:
                cap = cv2.VideoCapture(str(video_path))
                if cap.isOpened():
                    count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                    cap.release()
                    return count
            except Exception:
                pass
        
        # Fallback: assume 30 FPS and get duration
        logger.warning("Could not determine video frame count, using estimate")
        return 1000  # Default estimate
    
    def __len__(self) -> int:
        """Get number of frames in sequence."""
        return len(self.frames)
    
    def __getitem__(self, index: Union[int, slice]) -> Union[Frame, "FrameSequence"]:
        """Get frame(s) by index with smart preloading."""
        if isinstance(index, int):
            self._current_position = index
            self._smart_preload(index)
            return self.frames[index]
        elif isinstance(index, slice):
            return FrameSequence(
                frames=self.frames[index],
                metadata=self.metadata.copy(),
                sequence_id=self.sequence_id,
                lazy_load=self.lazy_load
            )
        else:
            raise TypeError("Index must be int or slice")
    
    def _smart_preload(self, current_index: int) -> None:
        """Preload frames around current position for smooth access."""
        if not self.lazy_load:
            return
        
        start_idx = max(0, current_index - self._preload_window)
        end_idx = min(len(self.frames), current_index + self._preload_window + 1)
        
        # Preload frames in window
        for i in range(start_idx, end_idx):
            frame = self.frames[i]
            if not frame.is_loaded:
                try:
                    frame.load()
                except Exception as e:
                    logger.warning(f"Failed to preload frame {i}: {e}")
        
        # Unload frames outside window to save memory
        for i, frame in enumerate(self.frames):
            if i < start_idx or i >= end_idx:
                if frame.is_loaded and frame.lazy_load:
                    frame.unload()
    
    def append(self, frame: Frame) -> None:
        """Add a frame to the sequence."""
        self.frames.append(frame)
    
    def extend(self, frames: List[Frame]) -> None:
        """Add multiple frames to the sequence."""
        self.frames.extend(frames)
    
    def load_all(self) -> List[np.ndarray]:
        """Load all frames into memory."""
        return [frame.load() for frame in self.frames]
    
    def load_batch(self, start_idx: int, batch_size: Optional[int] = None) -> List[np.ndarray]:
        """
        Load a batch of frames into memory.
        
        Args:
            start_idx: Starting frame index
            batch_size: Number of frames to load (uses default if None)
            
        Returns:
            List of loaded frame arrays
        """
        if batch_size is None:
            batch_size = self._batch_size
        
        end_idx = min(start_idx + batch_size, len(self.frames))
        batch_data = []
        
        for i in range(start_idx, end_idx):
            try:
                data = self.frames[i].load()
                batch_data.append(data)
            except Exception as e:
                logger.warning(f"Failed to load frame {i} in batch: {e}")
                # Add placeholder for failed frame
                if batch_data:
                    # Use shape from previous frame
                    placeholder = np.zeros_like(batch_data[-1])
                else:
                    # Default placeholder shape
                    placeholder = np.zeros((480, 640, 3), dtype=np.uint8)
                batch_data.append(placeholder)
        
        return batch_data
    
    def process_in_batches(self, processor_func, batch_size: Optional[int] = None, 
                          unload_after_processing: bool = True) -> List[Any]:
        """
        Process frames in batches to manage memory usage.
        
        Args:
            processor_func: Function to process each batch of frames
            batch_size: Size of each batch
            unload_after_processing: Whether to unload frames after processing
            
        Returns:
            List of processing results
        """
        if batch_size is None:
            batch_size = self._batch_size
        
        results = []
        total_frames = len(self.frames)
        
        for start_idx in range(0, total_frames, batch_size):
            try:
                # Load batch
                batch_data = self.load_batch(start_idx, batch_size)
                
                # Process batch
                batch_result = processor_func(batch_data, start_idx)
                results.append(batch_result)
                
                # Unload frames if requested
                if unload_after_processing:
                    end_idx = min(start_idx + batch_size, total_frames)
                    for i in range(start_idx, end_idx):
                        if self.frames[i].lazy_load:
                            self.frames[i].unload()
                
                logger.debug(f"Processed batch {start_idx}-{start_idx + len(batch_data)}")
                
            except Exception as e:
                logger.error(f"Failed to process batch starting at {start_idx}: {e}")
                results.append(None)
        
        return results
    
    def unload_all(self) -> None:
        """Unload all frames from memory."""
        for frame in self.frames:
            frame.unload()
        logger.debug(f"Unloaded all frames in sequence {self.sequence_id}")
    
    def unload_range(self, start_idx: int, end_idx: int) -> None:
        """
        Unload a range of frames from memory.
        
        Args:
            start_idx: Starting frame index
            end_idx: Ending frame index (exclusive)
        """
        end_idx = min(end_idx, len(self.frames))
        for i in range(start_idx, end_idx):
            self.frames[i].unload()
        logger.debug(f"Unloaded frames {start_idx}-{end_idx} in sequence {self.sequence_id}")
    
    def get_memory_stats(self) -> Dict[str, Any]:
        """Get memory usage statistics for this sequence."""
        loaded_count = sum(1 for frame in self.frames if frame.is_loaded)
        total_memory_mb = sum(frame.memory_usage_mb for frame in self.frames)
        
        return {
            'sequence_id': self.sequence_id,
            'total_frames': len(self.frames),
            'loaded_frames': loaded_count,
            'total_memory_mb': total_memory_mb,
            'average_frame_size_mb': total_memory_mb / max(loaded_count, 1),
            'memory_efficiency': loaded_count / len(self.frames) if self.frames else 0
        }
    
    def optimize_memory(self, keep_recent: int = 5) -> None:
        """
        Optimize memory usage by keeping only recently accessed frames.
        
        Args:
            keep_recent: Number of most recently accessed frames to keep loaded
        """
        if not self.frames:
            return
        
        # Sort frames by last access time (most recent first)
        frame_access_times = [
            (i, frame.access_stats['last_access_time']) 
            for i, frame in enumerate(self.frames) 
            if frame.is_loaded
        ]
        frame_access_times.sort(key=lambda x: x[1], reverse=True)
        
        # Keep only the most recently accessed frames
        frames_to_keep = set(idx for idx, _ in frame_access_times[:keep_recent])
        
        unloaded_count = 0
        for i, frame in enumerate(self.frames):
            if frame.is_loaded and i not in frames_to_keep and frame.lazy_load:
                frame.unload()
                unloaded_count += 1
        
        logger.info(f"Memory optimization: unloaded {unloaded_count} frames, kept {len(frames_to_keep)} recent frames")
    
    def set_batch_size(self, batch_size: int) -> None:
        """Set the default batch size for batch operations."""
        self._batch_size = max(1, batch_size)
    
    def set_preload_window(self, window_size: int) -> None:
        """Set the preload window size for smart preloading."""
        self._preload_window = max(0, window_size)
    
    def get_frame_at_time(self, timestamp: float, fps: float = 30.0) -> Optional[Frame]:
        """
        Get frame at specific timestamp.
        
        Args:
            timestamp: Time in seconds
            fps: Frames per second
            
        Returns:
            Frame at timestamp or None if not found
        """
        frame_index = int(timestamp * fps)
        if 0 <= frame_index < len(self.frames):
            return self.frames[frame_index]
        return None
    
    def extract_features(self) -> Dict[str, Any]:
        """Extract basic features from the sequence."""
        if not self.frames:
            return {}
        
        # Load first frame to get dimensions
        first_frame = self.frames[0].load()
        
        features = {
            "sequence_length": len(self.frames),
            "frame_shape": first_frame.shape,
            "sequence_id": self.sequence_id,
            "metadata": self.metadata
        }
        
        return features


# Utility functions for memory management

def get_global_memory_stats() -> Dict[str, Any]:
    """Get global memory statistics for all Frame objects."""
    return _memory_manager.get_memory_stats()


def set_global_memory_threshold(threshold_mb: int) -> None:
    """Set global memory threshold for automatic cleanup."""
    _memory_manager.set_memory_threshold(threshold_mb)


def set_max_loaded_frames(max_frames: int) -> None:
    """Set maximum number of frames to keep loaded globally."""
    _memory_manager.set_max_loaded_frames(max_frames)


def force_global_cleanup() -> None:
    """Force cleanup of all loaded frames globally."""
    _memory_manager.force_cleanup()


def shutdown_memory_manager() -> None:
    """Shutdown the global memory manager."""
    _memory_manager.shutdown()


class MemoryOptimizedFrameSequence(FrameSequence):
    """
    Memory-optimized version of FrameSequence with automatic memory management.
    
    This class automatically manages memory by unloading frames that haven't
    been accessed recently and provides streaming-like access to large sequences.
    """
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._auto_optimize = True
        self._optimization_interval = 50  # Optimize every N accesses
        self._access_counter = 0
    
    def __getitem__(self, index: Union[int, slice]) -> Union[Frame, "FrameSequence"]:
        """Get frame(s) with automatic memory optimization."""
        result = super().__getitem__(index)
        
        if self._auto_optimize:
            self._access_counter += 1
            if self._access_counter >= self._optimization_interval:
                self.optimize_memory()
                self._access_counter = 0
        
        return result
    
    def enable_auto_optimization(self, enabled: bool = True, interval: int = 50) -> None:
        """
        Enable or disable automatic memory optimization.
        
        Args:
            enabled: Whether to enable auto-optimization
            interval: Number of accesses between optimizations
        """
        self._auto_optimize = enabled
        self._optimization_interval = max(1, interval)
        logger.info(f"Auto-optimization {'enabled' if enabled else 'disabled'} with interval {interval}")


# Context manager for temporary memory management settings
class TemporaryMemorySettings:
    """Context manager for temporarily changing memory management settings."""
    
    def __init__(self, threshold_mb: Optional[int] = None, max_frames: Optional[int] = None):
        self.new_threshold = threshold_mb
        self.new_max_frames = max_frames
        self.old_threshold = None
        self.old_max_frames = None
    
    def __enter__(self):
        # Save current settings
        stats = _memory_manager.get_memory_stats()
        self.old_threshold = stats['memory_threshold_mb']
        self.old_max_frames = stats['max_loaded_frames']
        
        # Apply new settings
        if self.new_threshold is not None:
            _memory_manager.set_memory_threshold(self.new_threshold)
        if self.new_max_frames is not None:
            _memory_manager.set_max_loaded_frames(self.new_max_frames)
        
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        # Restore old settings
        _memory_manager.set_memory_threshold(self.old_threshold)
        _memory_manager.set_max_loaded_frames(self.old_max_frames)