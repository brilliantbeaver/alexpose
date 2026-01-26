"""
Stream processor for coordinating realtime pose estimation and gait analysis.

This module provides the main coordination logic for processing webcam
streams with pose estimation and gait analysis in realtime.
"""

import asyncio
import time
import base64
from typing import Dict, List, Optional, Any
import numpy as np
import cv2
from loguru import logger

from .interfaces import (
    IStreamProcessor,
    RealtimeFrame,
    ProcessingMode
)
from .frame_buffer import FrameBuffer
from .pose_estimator import RealtimePoseEstimator
from .gait_analyzer import RealtimeGaitAnalyzer
from .pose_tracker import PoseTracker


class StreamProcessor(IStreamProcessor):
    """
    Main stream processor for realtime gait analysis.
    
    This class coordinates frame processing, pose estimation, and gait analysis
    following the Single Responsibility Principle by delegating specific tasks
    to specialized components.
    """
    
    def __init__(
        self,
        model_path: Optional[str] = None,
        processing_mode: ProcessingMode = ProcessingMode.BALANCED,
        buffer_size: int = 30,
        enable_tracking: bool = True
    ):
        """
        Initialize stream processor.
        
        Args:
            model_path: Path to pose estimation model
            processing_mode: Processing mode for performance/accuracy tradeoff
            buffer_size: Size of frame buffer
            enable_tracking: Whether to enable pose tracking
        """
        self.processing_mode = processing_mode
        self.enable_tracking = enable_tracking
        
        # Initialize components
        self.frame_buffer = FrameBuffer(max_size=buffer_size)
        self.pose_estimator = RealtimePoseEstimator(
            model_path=model_path,
            processing_mode=processing_mode
        )
        self.gait_analyzer = RealtimeGaitAnalyzer()
        
        if enable_tracking:
            self.pose_tracker = PoseTracker()
        else:
            self.pose_tracker = None
        
        # Processing state
        self._is_processing = False
        self._frame_counter = 0
        self._session_start_time = None
        
        # Statistics
        self._processing_stats = {
            'frames_received': 0,
            'frames_processed': 0,
            'frames_failed': 0,
            'average_processing_time_ms': 0.0,
            'total_processing_time_ms': 0.0,
            'poses_detected': 0,
            'gait_analyses_completed': 0,
            'session_duration_seconds': 0.0
        }
        
        logger.info(
            f"StreamProcessor initialized: mode={processing_mode.value}, "
            f"buffer_size={buffer_size}, tracking={enable_tracking}"
        )
    
    async def process_frame(self, frame_data: bytes) -> Dict[str, Any]:
        """
        Process a single frame from the stream.
        
        Args:
            frame_data: Raw frame data (base64 encoded image)
            
        Returns:
            Processing result with pose data and metrics
        """
        if not self._is_processing:
            return {'error': 'Processor not started'}
        
        start_time = time.time()
        
        try:
            # Decode frame data
            frame = await self._decode_frame(frame_data)
            if frame is None:
                return {'error': 'Failed to decode frame'}
            
            # Add to buffer
            self.frame_buffer.add_frame(frame)
            self._processing_stats['frames_received'] += 1
            
            # Estimate pose
            pose_result = self.pose_estimator.estimate_pose(frame)
            
            # Apply pose tracking if enabled
            if self.pose_tracker and pose_result.keypoints:
                recent_poses = [
                    self.pose_estimator.estimate_pose(f) 
                    for f in self.frame_buffer.get_frame_sequence(5)[:-1]
                ]
                pose_result = self.pose_tracker.track_pose(pose_result, recent_poses)
            
            # Update gait analysis
            gait_metrics = None
            if pose_result.keypoints:
                self._processing_stats['poses_detected'] += 1
                gait_metrics = self.gait_analyzer.update_with_pose(pose_result)
                if gait_metrics:
                    self._processing_stats['gait_analyses_completed'] += 1
            
            # Update processing statistics
            processing_time_ms = (time.time() - start_time) * 1000
            self._update_processing_stats(processing_time_ms)
            
            # Prepare response
            response = {
                'success': True,
                'frame_id': frame.frame_id,
                'timestamp': frame.timestamp,
                'pose': {
                    'keypoints': pose_result.keypoints,
                    'confidence_scores': pose_result.confidence_scores,
                    'processing_time_ms': pose_result.processing_time_ms,
                    'frame_id': frame.frame_id,
                    'timestamp': frame.timestamp,
                    'estimator_info': pose_result.estimator_info
                },
                'processing_time_ms': processing_time_ms
            }
            
            # Add gait metrics if available
            if gait_metrics:
                response['gait_metrics'] = {
                    'cadence': gait_metrics.cadence,
                    'step_length': gait_metrics.step_length,
                    'stride_length': gait_metrics.stride_length,
                    'walking_speed': gait_metrics.walking_speed,
                    'symmetry_index': gait_metrics.symmetry_index,
                    'stability_score': gait_metrics.stability_score,
                    'confidence': gait_metrics.confidence,
                    'timestamp': gait_metrics.timestamp
                }
            
            return response
            
        except Exception as e:
            logger.error(f"Frame processing failed: {e}")
            self._processing_stats['frames_failed'] += 1
            
            processing_time_ms = (time.time() - start_time) * 1000
            self._update_processing_stats(processing_time_ms)
            
            return {
                'success': False,
                'error': str(e),
                'processing_time_ms': processing_time_ms
            }
    
    def set_processing_parameters(self, params: Dict[str, Any]) -> None:
        """
        Set processing parameters.
        
        Args:
            params: Dictionary of parameters to update
        """
        try:
            # Update processing mode
            if 'processing_mode' in params:
                mode_str = params['processing_mode']
                if mode_str in [m.value for m in ProcessingMode]:
                    new_mode = ProcessingMode(mode_str)
                    self.processing_mode = new_mode
                    self.pose_estimator.set_processing_mode(new_mode)
                    logger.info(f"Processing mode updated to: {mode_str}")
            
            # Update confidence thresholds
            if 'confidence_threshold' in params:
                # This would require updating the gait analyzer
                pass
            
            # Update buffer size
            if 'buffer_size' in params:
                new_size = int(params['buffer_size'])
                if new_size > 0:
                    # Create new buffer with updated size
                    old_frames = self.frame_buffer.get_frame_sequence(new_size)
                    self.frame_buffer = FrameBuffer(max_size=new_size)
                    for frame in old_frames:
                        self.frame_buffer.add_frame(frame)
                    logger.info(f"Buffer size updated to: {new_size}")
            
        except Exception as e:
            logger.error(f"Failed to update processing parameters: {e}")
    
    def get_processing_stats(self) -> Dict[str, Any]:
        """
        Get processing statistics.
        
        Returns:
            Dictionary containing processing statistics
        """
        # Update session duration
        if self._session_start_time:
            self._processing_stats['session_duration_seconds'] = (
                time.time() - self._session_start_time
            )
        
        # Combine with component statistics
        stats = self._processing_stats.copy()
        stats.update({
            'pose_estimator_stats': self.pose_estimator.get_performance_stats(),
            'frame_buffer_stats': self.frame_buffer.get_buffer_stats(),
            'tracking_enabled': self.enable_tracking
        })
        
        if self.pose_tracker:
            stats['tracking_confidence'] = self.pose_tracker.get_tracking_confidence()
        
        return stats
    
    def start_processing(self) -> None:
        """Start the processing pipeline."""
        if self._is_processing:
            logger.warning("Processing already started")
            return
        
        self._is_processing = True
        self._session_start_time = time.time()
        self._frame_counter = 0
        
        # Reset component states
        self.frame_buffer.clear()
        self.gait_analyzer.reset_analysis()
        if self.pose_tracker:
            self.pose_tracker.reset_tracking()
        
        logger.info("Stream processing started")
    
    def stop_processing(self) -> None:
        """Stop the processing pipeline."""
        if not self._is_processing:
            logger.warning("Processing not started")
            return
        
        self._is_processing = False
        
        # Log final statistics
        final_stats = self.get_processing_stats()
        logger.info(
            f"Stream processing stopped. "
            f"Processed {final_stats['frames_processed']} frames in "
            f"{final_stats['session_duration_seconds']:.1f} seconds"
        )
    
    async def _decode_frame(self, frame_data: bytes) -> Optional[RealtimeFrame]:
        """
        Decode frame data from bytes.
        
        Args:
            frame_data: Raw frame data
            
        Returns:
            Decoded RealtimeFrame or None if failed
        """
        try:
            # Assume frame_data is base64 encoded image
            if isinstance(frame_data, str):
                # Remove data URL prefix if present
                if frame_data.startswith('data:image'):
                    frame_data = frame_data.split(',')[1]
                
                # Decode base64
                image_bytes = base64.b64decode(frame_data)
            else:
                image_bytes = frame_data
            
            # Convert to numpy array
            nparr = np.frombuffer(image_bytes, np.uint8)
            
            # Decode image
            image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            
            if image is None:
                logger.error("Failed to decode image data")
                return None
            
            # Create RealtimeFrame
            self._frame_counter += 1
            frame = RealtimeFrame(
                data=image,
                timestamp=time.time(),
                frame_id=self._frame_counter,
                metadata={
                    'width': image.shape[1],
                    'height': image.shape[0],
                    'channels': image.shape[2] if len(image.shape) > 2 else 1,
                    'dtype': str(image.dtype)
                }
            )
            
            return frame
            
        except Exception as e:
            logger.error(f"Failed to decode frame: {e}")
            return None
    
    def _update_processing_stats(self, processing_time_ms: float) -> None:
        """Update processing statistics."""
        stats = self._processing_stats
        
        stats['frames_processed'] += 1
        stats['total_processing_time_ms'] += processing_time_ms
        
        # Update average processing time
        if stats['frames_processed'] > 0:
            stats['average_processing_time_ms'] = (
                stats['total_processing_time_ms'] / stats['frames_processed']
            )
    
    def is_processing(self) -> bool:
        """Check if processor is currently processing."""
        return self._is_processing
    
    def get_current_frame(self) -> Optional[RealtimeFrame]:
        """Get the current frame from buffer."""
        return self.frame_buffer.get_latest_frame()
    
    def get_recent_poses(self, count: int = 10) -> List[Dict[str, Any]]:
        """
        Get recent pose results.
        
        Args:
            count: Number of recent poses to return
            
        Returns:
            List of recent pose results
        """
        recent_frames = self.frame_buffer.get_frame_sequence(count)
        poses = []
        
        for frame in recent_frames:
            try:
                pose_result = self.pose_estimator.estimate_pose(frame)
                poses.append({
                    'frame_id': frame.frame_id,
                    'timestamp': frame.timestamp,
                    'keypoints': pose_result.keypoints,
                    'confidence_scores': pose_result.confidence_scores
                })
            except Exception as e:
                logger.error(f"Failed to get pose for frame {frame.frame_id}: {e}")
        
        return poses