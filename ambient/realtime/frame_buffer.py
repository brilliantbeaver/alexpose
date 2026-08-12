"""
Frame buffer implementation for realtime processing.

This module provides efficient frame storage and retrieval for realtime
pose estimation with circular buffer management and memory optimization.
"""

import time
from collections import deque
from typing import Dict, List, Optional, Any
import numpy as np
from loguru import logger

from .interfaces import IFrameBuffer, RealtimeFrame


class FrameBuffer(IFrameBuffer):
    """
    Circular frame buffer for efficient realtime frame management.
    
    This implementation follows the Single Responsibility Principle by
    focusing solely on frame storage and retrieval operations.
    """
    
    def __init__(
        self,
        max_size: int = 30,
        max_memory_mb: int = 100,
        auto_cleanup: bool = True
    ):
        """
        Initialize frame buffer.
        
        Args:
            max_size: Maximum number of frames to store
            max_memory_mb: Maximum memory usage in MB
            auto_cleanup: Whether to automatically cleanup old frames
        """
        self.max_size = max_size
        self.max_memory_bytes = max_memory_mb * 1024 * 1024
        self.auto_cleanup = auto_cleanup
        
        self._frames: deque[RealtimeFrame] = deque(maxlen=max_size)
        self._current_memory = 0
        self._frame_counter = 0
        self._stats = {
            'frames_added': 0,
            'frames_dropped': 0,
            'memory_cleanups': 0,
            'average_frame_size': 0
        }
        
        logger.info(
            f"FrameBuffer initialized: max_size={max_size}, "
            f"max_memory={max_memory_mb}MB"
        )
    
    def add_frame(self, frame: RealtimeFrame) -> None:
        """
        Add a frame to the buffer.
        
        Args:
            frame: Frame to add to buffer
        """
        try:
            # Calculate frame size
            frame_size = frame.data.nbytes if hasattr(frame.data, 'nbytes') else 0
            
            # Check memory constraints
            if self.auto_cleanup and self._should_cleanup_memory(frame_size):
                self._cleanup_old_frames()
            
            # Add frame (deque automatically handles max_size)
            if len(self._frames) == self.max_size:
                # Remove oldest frame from memory tracking
                old_frame = self._frames[0]
                old_size = old_frame.data.nbytes if hasattr(old_frame.data, 'nbytes') else 0
                self._current_memory -= old_size
                self._stats['frames_dropped'] += 1
            
            self._frames.append(frame)
            self._current_memory += frame_size
            self._frame_counter += 1
            self._stats['frames_added'] += 1
            
            # Update average frame size
            if self._stats['frames_added'] > 0:
                self._stats['average_frame_size'] = (
                    self._current_memory / len(self._frames)
                )
            
        except Exception as e:
            logger.error(f"Failed to add frame to buffer: {e}")
            raise
    
    def get_latest_frame(self) -> Optional[RealtimeFrame]:
        """
        Get the most recent frame.
        
        Returns:
            Latest frame or None if buffer is empty
        """
        try:
            return self._frames[-1] if self._frames else None
        except IndexError:
            return None
    
    def get_frame_sequence(self, count: int) -> List[RealtimeFrame]:
        """
        Get a sequence of recent frames.
        
        Args:
            count: Number of frames to retrieve
            
        Returns:
            List of recent frames (newest first)
        """
        try:
            if not self._frames:
                return []
            
            # Get the most recent 'count' frames
            start_idx = max(0, len(self._frames) - count)
            return list(self._frames)[start_idx:]
            
        except Exception as e:
            logger.error(f"Failed to get frame sequence: {e}")
            return []
    
    def clear(self) -> None:
        """Clear all frames from buffer."""
        try:
            self._frames.clear()
            self._current_memory = 0
            logger.debug("Frame buffer cleared")
            
        except Exception as e:
            logger.error(f"Failed to clear frame buffer: {e}")
    
    def get_buffer_stats(self) -> Dict[str, Any]:
        """
        Get buffer statistics.
        
        Returns:
            Dictionary containing buffer statistics
        """
        return {
            'current_size': len(self._frames),
            'max_size': self.max_size,
            'current_memory_mb': self._current_memory / (1024 * 1024),
            'max_memory_mb': self.max_memory_bytes / (1024 * 1024),
            'memory_usage_percent': (
                (self._current_memory / self.max_memory_bytes) * 100
                if self.max_memory_bytes > 0 else 0
            ),
            'frames_added': self._stats['frames_added'],
            'frames_dropped': self._stats['frames_dropped'],
            'memory_cleanups': self._stats['memory_cleanups'],
            'average_frame_size_kb': self._stats['average_frame_size'] / 1024,
            'frame_counter': self._frame_counter
        }
    
    def _should_cleanup_memory(self, new_frame_size: int) -> bool:
        """
        Check if memory cleanup is needed.
        
        Args:
            new_frame_size: Size of frame being added
            
        Returns:
            True if cleanup is needed
        """
        projected_memory = self._current_memory + new_frame_size
        return projected_memory > self.max_memory_bytes
    
    def _cleanup_old_frames(self) -> None:
        """Remove old frames to free memory."""
        try:
            initial_size = len(self._frames)
            target_memory = self.max_memory_bytes * 0.8  # Clean to 80% capacity
            
            while (
                self._frames and 
                self._current_memory > target_memory
            ):
                old_frame = self._frames.popleft()
                old_size = old_frame.data.nbytes if hasattr(old_frame.data, 'nbytes') else 0
                self._current_memory -= old_size
                self._stats['frames_dropped'] += 1
            
            cleaned_count = initial_size - len(self._frames)
            if cleaned_count > 0:
                self._stats['memory_cleanups'] += 1
                logger.debug(
                    f"Cleaned {cleaned_count} frames, "
                    f"memory: {self._current_memory / (1024*1024):.1f}MB"
                )
                
        except Exception as e:
            logger.error(f"Failed to cleanup old frames: {e}")
    
    def get_frame_by_id(self, frame_id: int) -> Optional[RealtimeFrame]:
        """
        Get frame by ID.
        
        Args:
            frame_id: Frame ID to search for
            
        Returns:
            Frame with matching ID or None
        """
        try:
            for frame in reversed(self._frames):
                if frame.frame_id == frame_id:
                    return frame
            return None
            
        except Exception as e:
            logger.error(f"Failed to get frame by ID {frame_id}: {e}")
            return None
    
    def get_frames_in_time_range(
        self, 
        start_time: float, 
        end_time: float
    ) -> List[RealtimeFrame]:
        """
        Get frames within a time range.
        
        Args:
            start_time: Start timestamp
            end_time: End timestamp
            
        Returns:
            List of frames within time range
        """
        try:
            return [
                frame for frame in self._frames
                if start_time <= frame.timestamp <= end_time
            ]
            
        except Exception as e:
            logger.error(f"Failed to get frames in time range: {e}")
            return []
    
    def is_full(self) -> bool:
        """Check if buffer is at maximum capacity."""
        return len(self._frames) >= self.max_size
    
    def is_empty(self) -> bool:
        """Check if buffer is empty."""
        return len(self._frames) == 0
    
    def get_memory_usage_mb(self) -> float:
        """Get current memory usage in MB."""
        return self._current_memory / (1024 * 1024)