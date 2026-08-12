"""
Realtime gait analyzer for live pose data analysis.

This module provides lightweight gait analysis optimized for realtime
processing with minimal computational overhead and immediate feedback.
"""

import time
import math
from typing import Dict, List, Optional, Any, Tuple
import numpy as np
from collections import deque
from loguru import logger

from .interfaces import (
    IRealtimeGaitAnalyzer,
    RealtimePoseResult,
    RealtimeGaitMetrics
)


class RealtimeGaitAnalyzer(IRealtimeGaitAnalyzer):
    """
    Lightweight gait analyzer for realtime pose analysis.
    
    This analyzer focuses on essential gait metrics that can be computed
    efficiently from a sliding window of pose data.
    """
    
    def __init__(
        self,
        window_size: int = 60,  # ~2 seconds at 30 FPS
        min_poses_for_analysis: int = 10,
        confidence_threshold: float = 0.5
    ):
        """
        Initialize realtime gait analyzer.
        
        Args:
            window_size: Number of poses to keep in sliding window
            min_poses_for_analysis: Minimum poses needed for analysis
            confidence_threshold: Minimum confidence for pose keypoints
        """
        self.window_size = window_size
        self.min_poses_for_analysis = min_poses_for_analysis
        self.confidence_threshold = confidence_threshold
        
        # Sliding window of poses
        self._pose_window: deque[RealtimePoseResult] = deque(maxlen=window_size)
        
        # Analysis state
        self._last_analysis_time = 0.0
        self._analysis_interval = 0.1  # Analyze every 100ms
        
        # Gait cycle detection state
        self._gait_cycles = []
        self._last_heel_strike_time = None
        self._step_times = deque(maxlen=10)
        
        # Movement tracking
        self._previous_positions = {}
        self._velocity_history = deque(maxlen=30)
        
        logger.info(
            f"RealtimeGaitAnalyzer initialized: window_size={window_size}, "
            f"min_poses={min_poses_for_analysis}"
        )
    
    def analyze_pose_sequence(
        self, 
        poses: List[RealtimePoseResult]
    ) -> RealtimeGaitMetrics:
        """
        Analyze a sequence of poses for gait metrics.
        
        Args:
            poses: List of pose results to analyze
            
        Returns:
            Gait metrics computed from pose sequence
        """
        if len(poses) < self.min_poses_for_analysis:
            return self._create_empty_metrics()
        
        try:
            # Extract key measurements
            measurements = self._extract_measurements(poses)
            
            # Compute gait metrics
            metrics = self._compute_gait_metrics(measurements)
            
            return metrics
            
        except Exception as e:
            logger.error(f"Gait analysis failed: {e}")
            return self._create_empty_metrics()
    
    def update_with_pose(self, pose: RealtimePoseResult) -> Optional[RealtimeGaitMetrics]:
        """
        Update analysis with a new pose, return metrics if available.
        
        Args:
            pose: New pose result to add
            
        Returns:
            Gait metrics if analysis is ready, None otherwise
        """
        # Add pose to sliding window
        self._pose_window.append(pose)
        
        # Check if we should perform analysis
        current_time = time.time()
        if (
            current_time - self._last_analysis_time < self._analysis_interval or
            len(self._pose_window) < self.min_poses_for_analysis
        ):
            return None
        
        self._last_analysis_time = current_time
        
        # Perform analysis on current window
        return self.analyze_pose_sequence(list(self._pose_window))
    
    def reset_analysis(self) -> None:
        """Reset analysis state."""
        self._pose_window.clear()
        self._gait_cycles.clear()
        self._step_times.clear()
        self._velocity_history.clear()
        self._previous_positions.clear()
        self._last_heel_strike_time = None
        self._last_analysis_time = 0.0
        
        logger.debug("Gait analysis state reset")
    
    def get_required_pose_count(self) -> int:
        """Get minimum number of poses needed for analysis."""
        return self.min_poses_for_analysis
    
    def _extract_measurements(self, poses: List[RealtimePoseResult]) -> Dict[str, Any]:
        """
        Extract key measurements from pose sequence.
        
        Args:
            poses: List of pose results
            
        Returns:
            Dictionary of extracted measurements
        """
        measurements = {
            'timestamps': [],
            'left_ankle_positions': [],
            'right_ankle_positions': [],
            'left_knee_positions': [],
            'right_knee_positions': [],
            'hip_positions': [],
            'shoulder_positions': [],
            'head_positions': [],
            'pose_confidences': []
        }
        
        for pose in poses:
            if not pose.keypoints:
                continue
            
            measurements['timestamps'].append(pose.timestamp)
            measurements['pose_confidences'].append(
                np.mean(pose.confidence_scores) if pose.confidence_scores else 0.0
            )
            
            # Extract key joint positions (MediaPipe 33-point model)
            keypoints = {kp['id']: kp for kp in pose.keypoints}
            
            # Ankle positions (left: 29, right: 30)
            left_ankle = keypoints.get(29, {})
            right_ankle = keypoints.get(30, {})
            measurements['left_ankle_positions'].append(
                (left_ankle.get('x', 0), left_ankle.get('y', 0))
            )
            measurements['right_ankle_positions'].append(
                (right_ankle.get('x', 0), right_ankle.get('y', 0))
            )
            
            # Knee positions (left: 25, right: 26)
            left_knee = keypoints.get(25, {})
            right_knee = keypoints.get(26, {})
            measurements['left_knee_positions'].append(
                (left_knee.get('x', 0), left_knee.get('y', 0))
            )
            measurements['right_knee_positions'].append(
                (right_knee.get('x', 0), right_knee.get('y', 0))
            )
            
            # Hip positions (left: 23, right: 24)
            left_hip = keypoints.get(23, {})
            right_hip = keypoints.get(24, {})
            hip_center_x = (left_hip.get('x', 0) + right_hip.get('x', 0)) / 2
            hip_center_y = (left_hip.get('y', 0) + right_hip.get('y', 0)) / 2
            measurements['hip_positions'].append((hip_center_x, hip_center_y))
            
            # Shoulder positions (left: 11, right: 12)
            left_shoulder = keypoints.get(11, {})
            right_shoulder = keypoints.get(12, {})
            shoulder_center_x = (left_shoulder.get('x', 0) + right_shoulder.get('x', 0)) / 2
            shoulder_center_y = (left_shoulder.get('y', 0) + right_shoulder.get('y', 0)) / 2
            measurements['shoulder_positions'].append((shoulder_center_x, shoulder_center_y))
            
            # Head position (nose: 0)
            nose = keypoints.get(0, {})
            measurements['head_positions'].append(
                (nose.get('x', 0), nose.get('y', 0))
            )
        
        return measurements
    
    def _compute_gait_metrics(self, measurements: Dict[str, Any]) -> RealtimeGaitMetrics:
        """
        Compute gait metrics from measurements.
        
        Args:
            measurements: Extracted measurements
            
        Returns:
            Computed gait metrics
        """
        try:
            # Basic validation
            if not measurements['timestamps']:
                return self._create_empty_metrics()
            
            timestamps = np.array(measurements['timestamps'])
            time_span = timestamps[-1] - timestamps[0]
            
            if time_span <= 0:
                return self._create_empty_metrics()
            
            # Compute walking speed
            walking_speed = self._compute_walking_speed(measurements)
            
            # Compute cadence (steps per minute)
            cadence = self._compute_cadence(measurements)
            
            # Compute step and stride lengths
            step_length, stride_length = self._compute_step_stride_lengths(measurements)
            
            # Compute symmetry index
            symmetry_index = self._compute_symmetry_index(measurements)
            
            # Compute stability score
            stability_score = self._compute_stability_score(measurements)
            
            # Overall confidence based on pose quality
            confidence = np.mean(measurements['pose_confidences'])
            
            return RealtimeGaitMetrics(
                cadence=cadence,
                step_length=step_length,
                stride_length=stride_length,
                walking_speed=walking_speed,
                symmetry_index=symmetry_index,
                stability_score=stability_score,
                confidence=confidence,
                timestamp=timestamps[-1]
            )
            
        except Exception as e:
            logger.error(f"Failed to compute gait metrics: {e}")
            return self._create_empty_metrics()
    
    def _compute_walking_speed(self, measurements: Dict[str, Any]) -> Optional[float]:
        """Compute walking speed from hip movement."""
        try:
            hip_positions = measurements['hip_positions']
            timestamps = measurements['timestamps']
            
            if len(hip_positions) < 2:
                return None
            
            # Calculate total distance traveled by hip center
            total_distance = 0.0
            for i in range(1, len(hip_positions)):
                dx = hip_positions[i][0] - hip_positions[i-1][0]
                dy = hip_positions[i][1] - hip_positions[i-1][1]
                distance = math.sqrt(dx*dx + dy*dy)
                total_distance += distance
            
            # Calculate time span
            time_span = timestamps[-1] - timestamps[0]
            
            if time_span > 0:
                # Normalize to relative units (0-5 scale)
                # Typical walking speed in pixels/sec ranges from 10-100 depending on distance
                # We normalize to a 0-5 scale where 1.0-1.5 is typical walking
                speed_pixels_per_second = total_distance / time_span
                # Normalize: assume 50 pixels/sec is "normal" walking (1.0)
                normalized_speed = speed_pixels_per_second / 50.0
                return max(0.0, min(5.0, normalized_speed))  # Clamp to 0-5 range
            
            return None
            
        except Exception as e:
            logger.error(f"Failed to compute walking speed: {e}")
            return None
    
    def _compute_cadence(self, measurements: Dict[str, Any]) -> Optional[float]:
        """Compute cadence from ankle movement patterns."""
        try:
            left_ankle = measurements['left_ankle_positions']
            right_ankle = measurements['right_ankle_positions']
            timestamps = measurements['timestamps']
            
            if len(left_ankle) < 10:  # Need sufficient data
                return None
            
            # Detect heel strikes by finding local minima in ankle y-coordinates
            # (assuming y increases downward)
            left_y = [pos[1] for pos in left_ankle]
            right_y = [pos[1] for pos in right_ankle]
            
            # Simple peak detection for heel strikes
            left_strikes = self._detect_heel_strikes(left_y, timestamps)
            right_strikes = self._detect_heel_strikes(right_y, timestamps)
            
            # Combine and sort all strikes
            all_strikes = sorted(left_strikes + right_strikes)
            
            if len(all_strikes) < 2:
                return None
            
            # Calculate average time between strikes
            strike_intervals = []
            for i in range(1, len(all_strikes)):
                interval = all_strikes[i] - all_strikes[i-1]
                strike_intervals.append(interval)
            
            if strike_intervals:
                avg_interval = np.mean(strike_intervals)
                # Convert to steps per minute (each strike is a step)
                cadence = 60.0 / avg_interval if avg_interval > 0 else None
                return cadence
            
            return None
            
        except Exception as e:
            logger.error(f"Failed to compute cadence: {e}")
            return None
    
    def _detect_heel_strikes(self, y_positions: List[float], timestamps: List[float]) -> List[float]:
        """Detect heel strikes from ankle y-positions."""
        strikes = []
        
        if len(y_positions) < 5:
            return strikes
        
        # Simple local minima detection
        for i in range(2, len(y_positions) - 2):
            if (y_positions[i] < y_positions[i-1] and 
                y_positions[i] < y_positions[i+1] and
                y_positions[i] < y_positions[i-2] and
                y_positions[i] < y_positions[i+2]):
                strikes.append(timestamps[i])
        
        return strikes
    
    def _compute_step_stride_lengths(
        self, 
        measurements: Dict[str, Any]
    ) -> Tuple[Optional[float], Optional[float]]:
        """Compute step and stride lengths."""
        try:
            left_ankle = measurements['left_ankle_positions']
            right_ankle = measurements['right_ankle_positions']
            hip_positions = measurements['hip_positions']
            
            if len(left_ankle) < 10 or len(hip_positions) < 2:
                return None, None
            
            # Calculate body height estimate (hip to head distance) for normalization
            # Use first and last hip positions to estimate body scale
            first_hip_y = hip_positions[0][1]
            # Estimate head at ~1.5x hip height above hip (rough approximation)
            estimated_body_height = abs(first_hip_y * 0.5)  # Rough body height in pixels
            
            if estimated_body_height < 10:  # Sanity check
                estimated_body_height = 100  # Default fallback
            
            # Find maximum separation between ankles (approximate step length)
            max_separation = 0.0
            for i in range(len(left_ankle)):
                if i < len(right_ankle):
                    dx = left_ankle[i][0] - right_ankle[i][0]
                    dy = left_ankle[i][1] - right_ankle[i][1]
                    separation = math.sqrt(dx*dx + dy*dy)
                    max_separation = max(max_separation, separation)
            
            # Normalize by body height to get relative units
            # Typical step length is ~0.4-0.5 of body height
            # Normalize so that 50-70 pixels (typical step) = 0.5-0.7 relative units
            step_length = (max_separation / estimated_body_height) * 0.5 if max_separation > 0 else None
            stride_length = (max_separation / estimated_body_height) * 1.0 if max_separation > 0 else None
            
            # Clamp to reasonable ranges
            if step_length is not None:
                step_length = max(0.0, min(2.0, step_length))
            if stride_length is not None:
                stride_length = max(0.0, min(4.0, stride_length))
            
            return step_length, stride_length
            
        except Exception as e:
            logger.error(f"Failed to compute step/stride lengths: {e}")
            return None, None
    
    def _compute_symmetry_index(self, measurements: Dict[str, Any]) -> Optional[float]:
        """Compute left-right symmetry index."""
        try:
            left_ankle = measurements['left_ankle_positions']
            right_ankle = measurements['right_ankle_positions']
            
            if len(left_ankle) < 5 or len(right_ankle) < 5:
                return None
            
            # Calculate movement variance for each ankle
            left_x_var = np.var([pos[0] for pos in left_ankle])
            left_y_var = np.var([pos[1] for pos in left_ankle])
            right_x_var = np.var([pos[0] for pos in right_ankle])
            right_y_var = np.var([pos[1] for pos in right_ankle])
            
            left_movement = left_x_var + left_y_var
            right_movement = right_x_var + right_y_var
            
            if left_movement + right_movement == 0:
                return 1.0  # Perfect symmetry (no movement)
            
            # Symmetry index: 1.0 = perfect symmetry, 0.0 = completely asymmetric
            symmetry = 1.0 - abs(left_movement - right_movement) / (left_movement + right_movement)
            return max(0.0, min(1.0, symmetry))
            
        except Exception as e:
            logger.error(f"Failed to compute symmetry index: {e}")
            return None
    
    def _compute_stability_score(self, measurements: Dict[str, Any]) -> Optional[float]:
        """Compute stability score based on trunk movement."""
        try:
            shoulder_positions = measurements['shoulder_positions']
            hip_positions = measurements['hip_positions']
            
            if len(shoulder_positions) < 5 or len(hip_positions) < 5:
                return None
            
            # Calculate trunk sway (shoulder movement relative to hip movement)
            shoulder_x_var = np.var([pos[0] for pos in shoulder_positions])
            hip_x_var = np.var([pos[0] for pos in hip_positions])
            
            if hip_x_var == 0:
                return 1.0 if shoulder_x_var == 0 else 0.0
            
            # Stability score: lower shoulder variance relative to hip = more stable
            stability = 1.0 / (1.0 + (shoulder_x_var / hip_x_var))
            return max(0.0, min(1.0, stability))
            
        except Exception as e:
            logger.error(f"Failed to compute stability score: {e}")
            return None
    
    def _create_empty_metrics(self) -> RealtimeGaitMetrics:
        """Create empty metrics for cases with insufficient data."""
        return RealtimeGaitMetrics(
            cadence=None,
            step_length=None,
            stride_length=None,
            walking_speed=None,
            symmetry_index=None,
            stability_score=None,
            confidence=0.0,
            timestamp=time.time()
        )