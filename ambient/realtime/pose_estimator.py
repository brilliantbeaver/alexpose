"""
Realtime pose estimator optimized for low-latency processing.

This module provides a specialized pose estimator designed for realtime
applications with performance optimizations and adaptive quality control.
"""

import time
from typing import Dict, List, Optional, Any
import numpy as np
import cv2
from loguru import logger

from ambient.pose.mediapipe_estimator import MediaPipeEstimator, MEDIAPIPE_AVAILABLE
from ambient.pose.suppress_warnings import suppress_stderr_fd
from .interfaces import (
    IRealtimePoseEstimator, 
    RealtimeFrame, 
    RealtimePoseResult, 
    ProcessingMode
)

if MEDIAPIPE_AVAILABLE:
    import mediapipe as mp
    from mediapipe.tasks import python
    from mediapipe.tasks.python import vision


class RealtimePoseEstimator(IRealtimePoseEstimator):
    """
    Realtime pose estimator with performance optimizations.
    
    This class extends the base MediaPipe estimator with realtime-specific
    optimizations including adaptive quality, frame skipping, and
    performance monitoring.
    """
    
    def __init__(
        self,
        model_path: Optional[str] = None,
        processing_mode: ProcessingMode = ProcessingMode.BALANCED,
        target_fps: int = 30,
        max_processing_time_ms: float = 33.0  # ~30 FPS
    ):
        """
        Initialize realtime pose estimator.
        
        Args:
            model_path: Path to MediaPipe model file
            processing_mode: Processing mode for performance/accuracy tradeoff
            target_fps: Target processing frame rate
            max_processing_time_ms: Maximum processing time per frame
        """
        if not MEDIAPIPE_AVAILABLE:
            raise ImportError("MediaPipe is required for realtime pose estimation")
        
        self.processing_mode = processing_mode
        self.target_fps = target_fps
        self.max_processing_time_ms = max_processing_time_ms
        
        # Initialize base estimator
        self._base_estimator = MediaPipeEstimator(model_path=model_path)
        
        # Performance tracking
        self._performance_stats = {
            'frames_processed': 0,
            'frames_skipped': 0,
            'average_processing_time_ms': 0.0,
            'max_processing_time_ms': 0.0,
            'min_processing_time_ms': float('inf'),
            'total_processing_time_ms': 0.0,
            'fps': 0.0,
            'last_fps_update': time.time()
        }
        
        # Adaptive quality parameters
        self._quality_params = self._get_quality_params(processing_mode)
        
        # Frame skipping logic
        self._frame_skip_counter = 0
        self._frame_skip_interval = 1  # Process every N frames
        
        # Create persistent landmarker for VIDEO mode
        self._landmarker = None
        self._create_landmarker()
        
        logger.info(
            f"RealtimePoseEstimator initialized: mode={processing_mode.value}, "
            f"target_fps={target_fps}"
        )
    
    def _create_landmarker(self):
        """Create MediaPipe landmarker for VIDEO mode."""
        try:
            with suppress_stderr_fd():
                base_options = python.BaseOptions(
                    model_asset_path=str(self._base_estimator.model_path)
                )
                options = vision.PoseLandmarkerOptions(
                    base_options=base_options,
                    running_mode=vision.RunningMode.VIDEO,  # Changed from IMAGE to VIDEO
                    num_poses=1,
                    min_pose_detection_confidence=self._quality_params['min_detection_confidence'],
                    min_pose_presence_confidence=self._quality_params['min_detection_confidence'],
                    min_tracking_confidence=self._quality_params['min_tracking_confidence']
                )
                self._landmarker = vision.PoseLandmarker.create_from_options(options)
            logger.info("MediaPipe landmarker created in VIDEO mode")
        except Exception as e:
            logger.error(f"Failed to create landmarker: {e}")
            raise
    
    def estimate_pose(self, frame: RealtimeFrame) -> RealtimePoseResult:
        """
        Estimate pose from a single frame with realtime optimizations.
        
        Args:
            frame: Input frame for pose estimation
            
        Returns:
            Pose estimation result with timing information
        """
        start_time = time.time()
        
        try:
            # Check if we should skip this frame for performance
            if self._should_skip_frame():
                return self._create_empty_result(frame, "frame_skipped")
            
            # Preprocess frame for optimal performance
            processed_frame = self._preprocess_frame(frame)
            
            # Perform pose estimation
            pose_result = self._estimate_pose_internal(processed_frame)
            
            # Post-process results
            result = self._postprocess_result(pose_result, frame)
            
            # Update performance statistics
            processing_time_ms = (time.time() - start_time) * 1000
            self._update_performance_stats(processing_time_ms)
            
            result.processing_time_ms = processing_time_ms
            
            return result
            
        except Exception as e:
            logger.error(f"Pose estimation failed: {e}")
            processing_time_ms = (time.time() - start_time) * 1000
            return self._create_error_result(frame, str(e), processing_time_ms)
    
    def set_processing_mode(self, mode: ProcessingMode) -> None:
        """
        Set processing mode for performance optimization.
        
        Args:
            mode: New processing mode
        """
        self.processing_mode = mode
        self._quality_params = self._get_quality_params(mode)
        
        # Process every frame for real-time tracking - no skipping
        self._frame_skip_interval = 1  # Always process every frame
        
        logger.info(f"Processing mode changed to: {mode.value}")
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """
        Get performance statistics.
        
        Returns:
            Dictionary containing performance metrics
        """
        # Update FPS calculation
        current_time = time.time()
        time_diff = current_time - self._performance_stats['last_fps_update']
        
        if time_diff >= 1.0:  # Update FPS every second
            frames_in_period = self._performance_stats['frames_processed']
            self._performance_stats['fps'] = frames_in_period / time_diff
            self._performance_stats['last_fps_update'] = current_time
        
        return self._performance_stats.copy()
    
    def is_ready(self) -> bool:
        """
        Check if estimator is ready for processing.
        
        Returns:
            True if estimator is ready
        """
        return self._base_estimator.is_available()
    
    def _get_quality_params(self, mode: ProcessingMode) -> Dict[str, Any]:
        """Get quality parameters for processing mode."""
        if mode == ProcessingMode.FAST:
            return {
                'min_detection_confidence': 0.3,
                'min_tracking_confidence': 0.3,
                'resize_factor': 0.5,  # Resize to 50% for speed
                'blur_kernel': None
            }
        elif mode == ProcessingMode.BALANCED:
            return {
                'min_detection_confidence': 0.4,  # Lowered from 0.5 for better detection
                'min_tracking_confidence': 0.4,  # Lowered from 0.5 for smoother tracking
                'resize_factor': 1.0,  # Changed from 0.75 - no resize for better quality
                'blur_kernel': None
            }
        else:  # ACCURATE
            return {
                'min_detection_confidence': 0.5,  # Lowered from 0.7 for better detection
                'min_tracking_confidence': 0.5,  # Lowered from 0.7 for smoother tracking
                'resize_factor': 1.0,  # Full resolution
                'blur_kernel': None  # Removed blur for lower latency
            }
    
    def _should_skip_frame(self) -> bool:
        """Determine if current frame should be skipped."""
        self._frame_skip_counter += 1
        
        # Skip based on interval
        if self._frame_skip_counter % self._frame_skip_interval != 0:
            self._performance_stats['frames_skipped'] += 1
            return True
        
        # Don't skip frames based on processing time - let MediaPipe handle it
        # This ensures we always try to process frames for real-time tracking
        
        return False
    
    def _preprocess_frame(self, frame: RealtimeFrame) -> np.ndarray:
        """
        Preprocess frame for optimal pose estimation.
        
        Args:
            frame: Input frame
            
        Returns:
            Preprocessed frame data
        """
        # Use frame data directly without copying for speed
        image = frame.data
        
        # Resize for performance if needed
        resize_factor = self._quality_params['resize_factor']
        if resize_factor != 1.0:
            height, width = image.shape[:2]
            new_height = int(height * resize_factor)
            new_width = int(width * resize_factor)
            image = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_LINEAR)
        
        # Apply blur for noise reduction if specified
        blur_kernel = self._quality_params['blur_kernel']
        if blur_kernel:
            image = cv2.GaussianBlur(image, blur_kernel, 0)
        
        # Ensure RGB format
        if len(image.shape) == 3 and image.shape[2] == 3:
            # Assume BGR and convert to RGB
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        return image
    
    def _estimate_pose_internal(self, image: np.ndarray) -> Dict[str, Any]:
        """
        Internal pose estimation using MediaPipe.
        
        Args:
            image: Preprocessed image
            
        Returns:
            Raw pose estimation result
        """
        # Create MediaPipe Image
        mp_image = mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=image
        )
        
        try:
            # Perform detection using VIDEO mode with timestamp
            # VIDEO mode requires timestamp in milliseconds
            timestamp_ms = int(time.time() * 1000)
            
            with suppress_stderr_fd():
                result = self._landmarker.detect_for_video(mp_image, timestamp_ms)
            
            return {
                'landmarks': result.pose_landmarks,
                'world_landmarks': result.pose_world_landmarks,
                'segmentation_masks': result.segmentation_masks
            }
            
        except Exception as e:
            logger.error(f"MediaPipe detection failed: {e}")
            return {
                'landmarks': [],
                'world_landmarks': [],
                'segmentation_masks': None
            }
    
    def _postprocess_result(
        self, 
        pose_result: Dict[str, Any], 
        original_frame: RealtimeFrame
    ) -> RealtimePoseResult:
        """
        Post-process pose estimation result.
        
        Args:
            pose_result: Raw pose result from MediaPipe
            original_frame: Original input frame
            
        Returns:
            Processed realtime pose result
        """
        keypoints = []
        confidence_scores = []
        
        if pose_result['landmarks'] and len(pose_result['landmarks']) > 0:
            landmarks = pose_result['landmarks'][0]
            
            # MediaPipe returns normalized coordinates (0-1 range)
            # Scale to original frame dimensions
            frame_height, frame_width = original_frame.data.shape[:2]
            
            for idx, landmark in enumerate(landmarks):
                # Convert normalized coordinates to pixel coordinates
                keypoint = {
                    'x': landmark.x * frame_width,
                    'y': landmark.y * frame_height,
                    'z': landmark.z if hasattr(landmark, 'z') else 0.0,
                    'confidence': landmark.visibility,
                    'id': idx,
                    'name': self._get_landmark_name(idx)
                }
                keypoints.append(keypoint)
                confidence_scores.append(landmark.visibility)
        else:
            logger.warning("No landmarks detected in pose result")
        
        return RealtimePoseResult(
            keypoints=keypoints,
            confidence_scores=confidence_scores,
            processing_time_ms=0.0,  # Will be set by caller
            frame_id=original_frame.frame_id,
            timestamp=original_frame.timestamp,
            estimator_info={
                'estimator': 'MediaPipe',
                'processing_mode': self.processing_mode.value,
                'quality_params': self._quality_params,
                'num_keypoints': len(keypoints)
            }
        )
    
    def _get_landmark_name(self, idx: int) -> str:
        """Get landmark name from index based on MediaPipe Pose model."""
        landmark_names = [
            'nose', 'left_eye_inner', 'left_eye', 'left_eye_outer',
            'right_eye_inner', 'right_eye', 'right_eye_outer',
            'left_ear', 'right_ear', 'mouth_left', 'mouth_right',
            'left_shoulder', 'right_shoulder', 'left_elbow', 'right_elbow',
            'left_wrist', 'right_wrist', 'left_pinky', 'right_pinky',
            'left_index', 'right_index', 'left_thumb', 'right_thumb',
            'left_hip', 'right_hip', 'left_knee', 'right_knee',
            'left_ankle', 'right_ankle', 'left_heel', 'right_heel',
            'left_foot_index', 'right_foot_index'
        ]
        return landmark_names[idx] if idx < len(landmark_names) else f'landmark_{idx}'
    
    def _create_empty_result(
        self, 
        frame: RealtimeFrame, 
        reason: str
    ) -> RealtimePoseResult:
        """Create empty result for skipped frames."""
        return RealtimePoseResult(
            keypoints=[],
            confidence_scores=[],
            processing_time_ms=0.0,
            frame_id=frame.frame_id,
            timestamp=frame.timestamp,
            estimator_info={
                'estimator': 'MediaPipe',
                'processing_mode': self.processing_mode.value,
                'skipped': True,
                'skip_reason': reason
            }
        )
    
    def _create_error_result(
        self, 
        frame: RealtimeFrame, 
        error: str, 
        processing_time_ms: float
    ) -> RealtimePoseResult:
        """Create error result for failed processing."""
        return RealtimePoseResult(
            keypoints=[],
            confidence_scores=[],
            processing_time_ms=processing_time_ms,
            frame_id=frame.frame_id,
            timestamp=frame.timestamp,
            estimator_info={
                'estimator': 'MediaPipe',
                'processing_mode': self.processing_mode.value,
                'error': error
            }
        )
    
    def _update_performance_stats(self, processing_time_ms: float) -> None:
        """Update performance statistics."""
        stats = self._performance_stats
        
        stats['frames_processed'] += 1
        stats['total_processing_time_ms'] += processing_time_ms
        
        # Update timing statistics
        stats['max_processing_time_ms'] = max(
            stats['max_processing_time_ms'], 
            processing_time_ms
        )
        stats['min_processing_time_ms'] = min(
            stats['min_processing_time_ms'], 
            processing_time_ms
        )
        
        # Update average
        if stats['frames_processed'] > 0:
            stats['average_processing_time_ms'] = (
                stats['total_processing_time_ms'] / stats['frames_processed']
            )