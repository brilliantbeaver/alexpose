"""
MediaPipe Singleton Manager for Windows Memory Leak Prevention.

This module implements a singleton pattern for MediaPipe landmarkers to prevent
the memory leaks and threading issues that occur on Windows when multiple
landmarker instances are created.
"""

import threading
import gc
from typing import Optional, Dict, Any
from pathlib import Path

from ambient.utils.log_config import get_logger

logger = get_logger(__name__)


class MediaPipeLandmarkerSingleton:
    """
    Singleton manager for MediaPipe landmarkers to prevent Windows memory leaks.
    
    This class ensures only one landmarker instance exists at a time and provides
    proper cleanup mechanisms to prevent the threading and memory issues that
    plague MediaPipe on Windows.
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
        
        self._landmarker = None
        self._model_path = None
        self._frame_count = 0
        self._max_frames_before_reset = 500  # Reset every 500 frames to prevent memory leaks
        self._creation_lock = threading.Lock()
        self._initialized = True
        
        logger.debug("MediaPipe singleton initialized")
    
    def get_landmarker(self, model_path: str, landmarker_factory) -> Any:
        """
        Get or create a MediaPipe landmarker instance.
        
        Args:
            model_path: Path to the MediaPipe model
            landmarker_factory: Factory for creating landmarkers
            
        Returns:
            MediaPipe landmarker instance
        """
        with self._creation_lock:
            # Check if we need to create or recreate the landmarker
            should_recreate = (
                self._landmarker is None or
                self._model_path != model_path or
                (self._max_frames_before_reset > 0 and 
                 self._frame_count >= self._max_frames_before_reset)
            )
            
            if should_recreate:
                reason = "initial creation"
                if self._landmarker is not None:
                    if self._model_path != model_path:
                        reason = "model path changed"
                    elif self._frame_count >= self._max_frames_before_reset:
                        reason = f"frame limit reached ({self._frame_count}/{self._max_frames_before_reset})"
                
                logger.debug(f"Creating new landmarker: {reason}")
                
                # Clean up existing landmarker
                if self._landmarker is not None:
                    self._cleanup_landmarker()
                
                # Create new landmarker
                try:
                    self._landmarker = landmarker_factory.create_landmarker(model_path)
                    self._model_path = model_path
                    self._frame_count = 0
                    
                    if self._landmarker is None:
                        raise RuntimeError(f"Failed to create landmarker from {model_path}")
                    
                    logger.debug("New landmarker created successfully")
                    
                except Exception as e:
                    logger.error(f"Failed to create landmarker: {e}")
                    self._landmarker = None
                    self._model_path = None
                    raise
            
            return self._landmarker
    
    def increment_frame_count(self):
        """Increment the frame counter for memory management."""
        self._frame_count += 1
    
    def set_max_frames_before_reset(self, max_frames: int):
        """
        Set the maximum number of frames before forcing a landmarker reset.
        
        Args:
            max_frames: Maximum frames to process before reset (default: 500)
                       Set to 0 to disable automatic resets
        """
        if max_frames < 0:
            raise ValueError("max_frames must be >= 0")
        self._max_frames_before_reset = max_frames
        logger.info(f"Max frames before reset set to: {max_frames}")
    
    def reset_frame_count(self):
        """Reset the frame counter without recreating the landmarker."""
        self._frame_count = 0
    
    def reset_landmarker(self):
        """Force reset the landmarker instance."""
        with self._creation_lock:
            logger.debug("Force resetting landmarker")
            self._cleanup_landmarker()
            self._landmarker = None
            self._model_path = None
            self._frame_count = 0
    
    def _cleanup_landmarker(self):
        """Clean up the current landmarker instance."""
        if self._landmarker is not None:
            try:
                # Set to None first to prevent further use
                landmarker = self._landmarker
                self._landmarker = None
                
                # Delete the reference
                del landmarker
                
                # Force garbage collection to clean up MediaPipe resources
                gc.collect()
                
                logger.debug("Landmarker cleaned up successfully")
                
            except Exception as e:
                logger.warning(f"Error during landmarker cleanup: {e}")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get statistics about the singleton state."""
        return {
            'has_landmarker': self._landmarker is not None,
            'model_path': self._model_path,
            'frame_count': self._frame_count,
            'max_frames_before_reset': self._max_frames_before_reset
        }


def get_mediapipe_singleton() -> MediaPipeLandmarkerSingleton:
    """Get the MediaPipe landmarker singleton instance."""
    return MediaPipeLandmarkerSingleton()