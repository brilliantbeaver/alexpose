"""
Pose tracker for maintaining consistency across frames.

This module provides pose tracking functionality to smooth pose estimates
across frames and handle temporary detection failures.
"""

import time
import math
from typing import Dict, List, Optional, Any, Tuple
import numpy as np
from loguru import logger

from .interfaces import IPoseTracker, RealtimePoseResult


class PoseTracker(IPoseTracker):
    """
    Pose tracker for maintaining temporal consistency.
    
    This tracker uses simple motion prediction and confidence-based
    smoothing to maintain consistent pose estimates across frames.
    """
    
    def __init__(
        self,
        smoothing_factor: float = 0.7,
        max_tracking_distance: float = 50.0,
        confidence_threshold: float = 0.3,
        max_missing_frames: int = 5
    ):
        """
        Initialize pose tracker.
        
        Args:
            smoothing_factor: Factor for temporal smoothing (0-1)
            max_tracking_distance: Maximum distance for keypoint tracking
            confidence_threshold: Minimum confidence for tracking
            max_missing_frames: Maximum frames to track without detection
        """
        self.smoothing_factor = smoothing_factor
        self.max_tracking_distance = max_tracking_distance
        self.confidence_threshold = confidence_threshold
        self.max_missing_frames = max_missing_frames
        
        # Tracking state
        self._previous_pose: Optional[RealtimePoseResult] = None
        self._tracking_confidence = 0.0
        self._missing_frame_count = 0
        self._keypoint_velocities: Dict[int, Tuple[float, float]] = {}
        
        logger.info(
            f"PoseTracker initialized: smoothing={smoothing_factor}, "
            f"max_distance={max_tracking_distance}"
        )
    
    def track_pose(
        self,
        current_pose: RealtimePoseResult,
        previous_poses: List[RealtimePoseResult]
    ) -> RealtimePoseResult:
        """
        Track pose across frames for consistency.
        
        Args:
            current_pose: Current pose estimation result
            previous_poses: List of previous pose results
            
        Returns:
            Tracked and smoothed pose result
        """
        try:
            # If no keypoints detected, try to predict from previous poses
            if not current_pose.keypoints:
                return self._handle_missing_pose(current_pose, previous_poses)
            
            # If this is the first pose, just return it
            if self._previous_pose is None:
                self._previous_pose = current_pose
                self._tracking_confidence = np.mean(current_pose.confidence_scores)
                self._missing_frame_count = 0
                return current_pose
            
            # Track keypoints from previous frame
            tracked_pose = self._track_keypoints(current_pose, self._previous_pose)
            
            # Apply temporal smoothing
            smoothed_pose = self._apply_smoothing(tracked_pose, self._previous_pose)
            
            # Update tracking state
            self._update_tracking_state(smoothed_pose)
            
            return smoothed_pose
            
        except Exception as e:
            logger.error(f"Pose tracking failed: {e}")
            return current_pose
    
    def get_tracking_confidence(self) -> float:
        """
        Get current tracking confidence.
        
        Returns:
            Tracking confidence score (0-1)
        """
        return self._tracking_confidence
    
    def reset_tracking(self) -> None:
        """Reset tracking state."""
        self._previous_pose = None
        self._tracking_confidence = 0.0
        self._missing_frame_count = 0
        self._keypoint_velocities.clear()
        
        logger.debug("Pose tracking state reset")
    
    def _track_keypoints(
        self,
        current_pose: RealtimePoseResult,
        previous_pose: RealtimePoseResult
    ) -> RealtimePoseResult:
        """
        Track individual keypoints between frames.
        
        Args:
            current_pose: Current pose result
            previous_pose: Previous pose result
            
        Returns:
            Pose with tracked keypoints
        """
        if not previous_pose.keypoints:
            return current_pose
        
        # Create lookup for previous keypoints
        prev_keypoints = {kp['id']: kp for kp in previous_pose.keypoints}
        
        tracked_keypoints = []
        tracked_confidences = []
        
        for current_kp in current_pose.keypoints:
            kp_id = current_kp['id']
            
            # Find corresponding previous keypoint
            prev_kp = prev_keypoints.get(kp_id)
            
            if prev_kp is None:
                # No previous keypoint, use current as-is
                tracked_keypoints.append(current_kp.copy())
                tracked_confidences.append(current_kp.get('confidence', 0.0))
                continue
            
            # Calculate distance moved
            dx = current_kp['x'] - prev_kp['x']
            dy = current_kp['y'] - prev_kp['y']
            distance = math.sqrt(dx*dx + dy*dy)
            
            # Check if movement is reasonable
            if distance > self.max_tracking_distance:
                # Large movement, reduce confidence or use prediction
                if current_kp.get('confidence', 0) < self.confidence_threshold:
                    # Low confidence and large movement, use prediction
                    predicted_kp = self._predict_keypoint_position(kp_id, prev_kp)
                    if predicted_kp:
                        tracked_keypoints.append(predicted_kp)
                        tracked_confidences.append(predicted_kp.get('confidence', 0.0))
                        continue
            
            # Use current keypoint but update velocity
            self._update_keypoint_velocity(kp_id, dx, dy, 
                                         current_pose.timestamp - previous_pose.timestamp)
            
            tracked_keypoints.append(current_kp.copy())
            tracked_confidences.append(current_kp.get('confidence', 0.0))
        
        # Create tracked pose result
        tracked_pose = RealtimePoseResult(
            keypoints=tracked_keypoints,
            confidence_scores=tracked_confidences,
            processing_time_ms=current_pose.processing_time_ms,
            frame_id=current_pose.frame_id,
            timestamp=current_pose.timestamp,
            estimator_info=current_pose.estimator_info.copy()
        )
        
        # Add tracking info
        tracked_pose.estimator_info['tracking_applied'] = True
        tracked_pose.estimator_info['tracking_confidence'] = self._tracking_confidence
        
        return tracked_pose
    
    def _apply_smoothing(
        self,
        current_pose: RealtimePoseResult,
        previous_pose: RealtimePoseResult
    ) -> RealtimePoseResult:
        """
        Apply temporal smoothing to pose keypoints.
        
        Args:
            current_pose: Current pose result
            previous_pose: Previous pose result
            
        Returns:
            Smoothed pose result
        """
        if not previous_pose.keypoints:
            return current_pose
        
        # Create lookup for previous keypoints
        prev_keypoints = {kp['id']: kp for kp in previous_pose.keypoints}
        
        smoothed_keypoints = []
        smoothed_confidences = []
        
        for current_kp in current_pose.keypoints:
            kp_id = current_kp['id']
            prev_kp = prev_keypoints.get(kp_id)
            
            if prev_kp is None:
                # No previous keypoint, use current as-is
                smoothed_keypoints.append(current_kp.copy())
                smoothed_confidences.append(current_kp.get('confidence', 0.0))
                continue
            
            # Apply exponential smoothing
            alpha = self.smoothing_factor
            
            smoothed_kp = current_kp.copy()
            smoothed_kp['x'] = alpha * prev_kp['x'] + (1 - alpha) * current_kp['x']
            smoothed_kp['y'] = alpha * prev_kp['y'] + (1 - alpha) * current_kp['y']
            
            # Smooth confidence as well
            prev_conf = prev_kp.get('confidence', 0.0)
            curr_conf = current_kp.get('confidence', 0.0)
            smoothed_conf = alpha * prev_conf + (1 - alpha) * curr_conf
            smoothed_kp['confidence'] = smoothed_conf
            
            smoothed_keypoints.append(smoothed_kp)
            smoothed_confidences.append(smoothed_conf)
        
        # Create smoothed pose result
        smoothed_pose = RealtimePoseResult(
            keypoints=smoothed_keypoints,
            confidence_scores=smoothed_confidences,
            processing_time_ms=current_pose.processing_time_ms,
            frame_id=current_pose.frame_id,
            timestamp=current_pose.timestamp,
            estimator_info=current_pose.estimator_info.copy()
        )
        
        # Add smoothing info
        smoothed_pose.estimator_info['smoothing_applied'] = True
        smoothed_pose.estimator_info['smoothing_factor'] = self.smoothing_factor
        
        return smoothed_pose
    
    def _handle_missing_pose(
        self,
        current_pose: RealtimePoseResult,
        previous_poses: List[RealtimePoseResult]
    ) -> RealtimePoseResult:
        """
        Handle case where no pose is detected in current frame.
        
        Args:
            current_pose: Current pose result (empty)
            previous_poses: List of previous pose results
            
        Returns:
            Predicted pose result or empty result
        """
        self._missing_frame_count += 1
        
        # If too many missing frames, give up tracking
        if self._missing_frame_count > self.max_missing_frames:
            self._tracking_confidence = 0.0
            return current_pose
        
        # Try to predict pose from previous poses
        if self._previous_pose and self._previous_pose.keypoints:
            predicted_keypoints = []
            predicted_confidences = []
            
            for prev_kp in self._previous_pose.keypoints:
                predicted_kp = self._predict_keypoint_position(
                    prev_kp['id'], 
                    prev_kp
                )
                
                if predicted_kp:
                    # Reduce confidence for predicted keypoints
                    predicted_kp['confidence'] *= 0.5
                    predicted_keypoints.append(predicted_kp)
                    predicted_confidences.append(predicted_kp['confidence'])
            
            if predicted_keypoints:
                # Create predicted pose result
                predicted_pose = RealtimePoseResult(
                    keypoints=predicted_keypoints,
                    confidence_scores=predicted_confidences,
                    processing_time_ms=current_pose.processing_time_ms,
                    frame_id=current_pose.frame_id,
                    timestamp=current_pose.timestamp,
                    estimator_info=current_pose.estimator_info.copy()
                )
                
                # Add prediction info
                predicted_pose.estimator_info['pose_predicted'] = True
                predicted_pose.estimator_info['missing_frame_count'] = self._missing_frame_count
                
                # Reduce tracking confidence
                self._tracking_confidence *= 0.8
                
                return predicted_pose
        
        # No prediction possible, return empty result
        self._tracking_confidence *= 0.5
        return current_pose
    
    def _predict_keypoint_position(
        self,
        keypoint_id: int,
        previous_keypoint: Dict[str, Any]
    ) -> Optional[Dict[str, Any]]:
        """
        Predict keypoint position based on velocity.
        
        Args:
            keypoint_id: ID of keypoint to predict
            previous_keypoint: Previous keypoint data
            
        Returns:
            Predicted keypoint or None
        """
        velocity = self._keypoint_velocities.get(keypoint_id)
        
        if velocity is None:
            # No velocity data, just use previous position
            return previous_keypoint.copy()
        
        # Predict position based on velocity
        predicted_kp = previous_keypoint.copy()
        predicted_kp['x'] += velocity[0]
        predicted_kp['y'] += velocity[1]
        
        return predicted_kp
    
    def _update_keypoint_velocity(
        self,
        keypoint_id: int,
        dx: float,
        dy: float,
        dt: float
    ) -> None:
        """
        Update velocity for a keypoint.
        
        Args:
            keypoint_id: ID of keypoint
            dx: Change in x position
            dy: Change in y position
            dt: Time difference
        """
        if dt <= 0:
            return
        
        # Calculate velocity
        vx = dx / dt
        vy = dy / dt
        
        # Apply smoothing to velocity
        if keypoint_id in self._keypoint_velocities:
            prev_vx, prev_vy = self._keypoint_velocities[keypoint_id]
            alpha = 0.3  # Velocity smoothing factor
            vx = alpha * prev_vx + (1 - alpha) * vx
            vy = alpha * prev_vy + (1 - alpha) * vy
        
        self._keypoint_velocities[keypoint_id] = (vx, vy)
    
    def _update_tracking_state(self, pose: RealtimePoseResult) -> None:
        """
        Update internal tracking state.
        
        Args:
            pose: Current pose result
        """
        self._previous_pose = pose
        self._missing_frame_count = 0
        
        # Update tracking confidence based on pose quality
        if pose.confidence_scores:
            pose_confidence = np.mean(pose.confidence_scores)
            # Exponential moving average of confidence
            alpha = 0.1
            self._tracking_confidence = (
                alpha * self._tracking_confidence + 
                (1 - alpha) * pose_confidence
            )
        else:
            self._tracking_confidence *= 0.9  # Slight decay if no keypoints