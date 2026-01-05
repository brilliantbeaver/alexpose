"""
Temporal analysis module for gait analysis.

This module provides enhanced gait cycle detection and temporal analysis
capabilities for identifying gait patterns and abnormalities.

Author: AlexPose Team
"""

import numpy as np
from typing import Any, Dict, List, Optional, Tuple
from loguru import logger

from ambient.core.frame import FrameSequence


class TemporalAnalyzer:
    """
    Enhanced temporal analyzer for gait cycle detection and analysis.
    
    This class provides sophisticated gait cycle detection using multiple
    methods and temporal feature extraction for gait analysis.
    """
    
    def __init__(
        self,
        fps: float = 30.0,
        min_cycle_duration: float = 0.8,  # seconds
        max_cycle_duration: float = 2.5,  # seconds
        detection_method: str = "heel_strike"
    ):
        """
        Initialize temporal analyzer.
        
        Args:
            fps: Frames per second of the video
            min_cycle_duration: Minimum gait cycle duration in seconds
            max_cycle_duration: Maximum gait cycle duration in seconds
            detection_method: Method for cycle detection ("heel_strike", "toe_off", "combined")
        """
        self.fps = fps
        self.min_cycle_duration = min_cycle_duration
        self.max_cycle_duration = max_cycle_duration
        self.detection_method = detection_method
        
        # Convert durations to frames
        self.min_cycle_frames = int(min_cycle_duration * fps)
        self.max_cycle_frames = int(max_cycle_duration * fps)
        
        logger.info(f"Temporal analyzer initialized with {detection_method} detection")
    
    def detect_gait_cycles(self, pose_sequence: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Detect gait cycles in pose sequence.
        
        Args:
            pose_sequence: List of pose estimation results
            
        Returns:
            List of detected gait cycles with timing information
        """
        if not pose_sequence:
            return []
        
        # Convert poses to array format
        keypoints_array = self._poses_to_array(pose_sequence)
        if keypoints_array is None:
            return []
        
        cycles = []
        
        try:
            if self.detection_method == "heel_strike":
                cycles = self._detect_cycles_heel_strike(keypoints_array)
            elif self.detection_method == "toe_off":
                cycles = self._detect_cycles_toe_off(keypoints_array)
            elif self.detection_method == "combined":
                cycles = self._detect_cycles_combined(keypoints_array)
            else:
                logger.warning(f"Unknown detection method: {self.detection_method}")
                cycles = self._detect_cycles_heel_strike(keypoints_array)  # Default
            
            # Add metadata to cycles
            for i, cycle in enumerate(cycles):
                cycle["cycle_id"] = i
                cycle["duration_seconds"] = cycle["duration_frames"] / self.fps
                cycle["detection_method"] = self.detection_method
            
        except Exception as e:
            logger.error(f"Gait cycle detection failed: {e}")
        
        return cycles
    
    def _poses_to_array(self, pose_sequence: List[Dict[str, Any]]) -> Optional[np.ndarray]:
        """Convert pose sequence to numpy array."""
        if not pose_sequence:
            return None
        
        # Get keypoints from first valid pose
        keypoints_data = None
        for pose in pose_sequence:
            if pose.get("keypoints"):
                keypoints_data = pose["keypoints"]
                break
        
        if not keypoints_data:
            return None
        
        num_keypoints = len(keypoints_data)
        num_frames = len(pose_sequence)
        
        # Create array: [frames, keypoints, (x, y, confidence)]
        keypoints_array = np.zeros((num_frames, num_keypoints, 3))
        
        for frame_idx, pose in enumerate(pose_sequence):
            keypoints = pose.get("keypoints", [])
            for kp_idx, kp in enumerate(keypoints):
                if kp_idx < num_keypoints:
                    keypoints_array[frame_idx, kp_idx, 0] = kp.get("x", 0)
                    keypoints_array[frame_idx, kp_idx, 1] = kp.get("y", 0)
                    keypoints_array[frame_idx, kp_idx, 2] = kp.get("confidence", 0)
        
        return keypoints_array
    
    def _detect_cycles_heel_strike(self, keypoints: np.ndarray) -> List[Dict[str, Any]]:
        """Detect gait cycles using heel strike events."""
        cycles = []
        
        # Assume COCO format for now - can be extended for other formats
        left_ankle_idx = 15  # COCO left ankle
        right_ankle_idx = 16  # COCO right ankle
        
        if keypoints.shape[1] <= max(left_ankle_idx, right_ankle_idx):
            logger.warning("Insufficient keypoints for heel strike detection")
            return cycles
        
        # Extract ankle positions
        left_ankle = keypoints[:, left_ankle_idx, :2]
        right_ankle = keypoints[:, right_ankle_idx, :2]
        
        # Detect heel strikes for both feet
        left_strikes = self._detect_heel_strikes(left_ankle)
        right_strikes = self._detect_heel_strikes(right_ankle)
        
        # Combine and sort strikes
        all_strikes = []
        for frame in left_strikes:
            all_strikes.append({"frame": frame, "foot": "left"})
        for frame in right_strikes:
            all_strikes.append({"frame": frame, "foot": "right"})
        
        all_strikes.sort(key=lambda x: x["frame"])
        
        # Create cycles between consecutive strikes of the same foot
        left_strikes_only = [s["frame"] for s in all_strikes if s["foot"] == "left"]
        right_strikes_only = [s["frame"] for s in all_strikes if s["foot"] == "right"]
        
        # Process left foot cycles
        for i in range(len(left_strikes_only) - 1):
            start_frame = left_strikes_only[i]
            end_frame = left_strikes_only[i + 1]
            duration = end_frame - start_frame
            
            if self.min_cycle_frames <= duration <= self.max_cycle_frames:
                cycles.append({
                    "start_frame": start_frame,
                    "end_frame": end_frame,
                    "duration_frames": duration,
                    "foot": "left",
                    "type": "heel_strike_cycle"
                })
        
        # Process right foot cycles
        for i in range(len(right_strikes_only) - 1):
            start_frame = right_strikes_only[i]
            end_frame = right_strikes_only[i + 1]
            duration = end_frame - start_frame
            
            if self.min_cycle_frames <= duration <= self.max_cycle_frames:
                cycles.append({
                    "start_frame": start_frame,
                    "end_frame": end_frame,
                    "duration_frames": duration,
                    "foot": "right",
                    "type": "heel_strike_cycle"
                })
        
        # Sort cycles by start frame
        cycles.sort(key=lambda x: x["start_frame"])
        
        return cycles
    
    def _detect_heel_strikes(self, ankle_positions: np.ndarray) -> List[int]:
        """Detect heel strike events from ankle positions."""
        if len(ankle_positions) < 10:
            return []
        
        # Calculate vertical velocity (y-direction)
        y_positions = ankle_positions[:, 1]
        y_velocity = np.diff(y_positions)
        
        # Smooth the velocity signal
        window_size = min(5, len(y_velocity) // 4)
        if window_size >= 3:
            y_velocity_smooth = self._smooth_signal(y_velocity, window_size)
        else:
            y_velocity_smooth = y_velocity
        
        # Find local minima in y-position (heel strikes occur at lowest points)
        heel_strikes = []
        
        # Use a simple peak detection algorithm
        for i in range(1, len(y_positions) - 1):
            if (y_positions[i] < y_positions[i-1] and 
                y_positions[i] < y_positions[i+1] and
                i > self.min_cycle_frames):  # Ensure minimum distance between strikes
                
                # Check if this is significantly lower than surrounding points
                local_window = max(5, self.min_cycle_frames // 4)
                start_idx = max(0, i - local_window)
                end_idx = min(len(y_positions), i + local_window)
                
                local_min = np.min(y_positions[start_idx:end_idx])
                if y_positions[i] <= local_min + 2:  # Within 2 pixels of local minimum
                    # Ensure minimum distance from previous heel strike
                    if not heel_strikes or (i - heel_strikes[-1]) >= self.min_cycle_frames:
                        heel_strikes.append(i)
        
        return heel_strikes
    
    def _detect_cycles_toe_off(self, keypoints: np.ndarray) -> List[Dict[str, Any]]:
        """Detect gait cycles using toe-off events."""
        # Similar to heel strike but looking for toe-off events
        # This is a simplified implementation
        return self._detect_cycles_heel_strike(keypoints)  # Fallback for now
    
    def _detect_cycles_combined(self, keypoints: np.ndarray) -> List[Dict[str, Any]]:
        """Detect gait cycles using combined heel strike and toe-off events."""
        # Combine both methods for more robust detection
        heel_cycles = self._detect_cycles_heel_strike(keypoints)
        toe_cycles = self._detect_cycles_toe_off(keypoints)
        
        # For now, just return heel strike cycles
        # TODO: Implement proper combination logic
        return heel_cycles
    
    def _smooth_signal(self, signal: np.ndarray, window_size: int) -> np.ndarray:
        """Smooth signal using moving average."""
        if window_size < 3:
            return signal
        
        # Ensure odd window size
        if window_size % 2 == 0:
            window_size += 1
        
        half_window = window_size // 2
        smoothed = np.zeros_like(signal)
        
        for i in range(len(signal)):
            start_idx = max(0, i - half_window)
            end_idx = min(len(signal), i + half_window + 1)
            smoothed[i] = np.mean(signal[start_idx:end_idx])
        
        return smoothed
    
    def analyze_cycle_timing(self, cycles: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Analyze timing characteristics of detected gait cycles.
        
        Args:
            cycles: List of detected gait cycles
            
        Returns:
            Dictionary containing timing analysis results
        """
        if not cycles:
            return {}
        
        analysis = {}
        
        # Separate left and right foot cycles
        left_cycles = [c for c in cycles if c.get("foot") == "left"]
        right_cycles = [c for c in cycles if c.get("foot") == "right"]
        
        # Analyze cycle durations
        all_durations = [c["duration_seconds"] for c in cycles]
        if all_durations:
            analysis["cycle_duration_mean"] = np.mean(all_durations)
            analysis["cycle_duration_std"] = np.std(all_durations)
            analysis["cycle_duration_cv"] = np.std(all_durations) / np.mean(all_durations)
            analysis["cycle_duration_range"] = np.max(all_durations) - np.min(all_durations)
        
        # Analyze left foot cycles
        if left_cycles:
            left_durations = [c["duration_seconds"] for c in left_cycles]
            analysis["left_cycle_duration_mean"] = np.mean(left_durations)
            analysis["left_cycle_duration_std"] = np.std(left_durations)
            analysis["left_cycle_count"] = len(left_cycles)
        
        # Analyze right foot cycles
        if right_cycles:
            right_durations = [c["duration_seconds"] for c in right_cycles]
            analysis["right_cycle_duration_mean"] = np.mean(right_durations)
            analysis["right_cycle_duration_std"] = np.std(right_durations)
            analysis["right_cycle_count"] = len(right_cycles)
        
        # Calculate asymmetry
        if left_cycles and right_cycles:
            left_mean = analysis["left_cycle_duration_mean"]
            right_mean = analysis["right_cycle_duration_mean"]
            analysis["cycle_duration_asymmetry"] = abs(left_mean - right_mean) / ((left_mean + right_mean) / 2)
        
        # Calculate cadence (steps per minute)
        if cycles:
            total_time = max(c["end_frame"] for c in cycles) / self.fps
            total_steps = len(cycles)
            analysis["cadence_steps_per_minute"] = (total_steps / total_time) * 60
            analysis["cadence_cycles_per_minute"] = analysis["cadence_steps_per_minute"] / 2  # Each cycle is 2 steps
        
        # Analyze cycle regularity
        if len(cycles) >= 3:
            intervals = []
            for i in range(1, len(cycles)):
                interval = cycles[i]["start_frame"] - cycles[i-1]["start_frame"]
                intervals.append(interval / self.fps)
            
            if intervals:
                analysis["step_interval_mean"] = np.mean(intervals)
                analysis["step_interval_std"] = np.std(intervals)
                analysis["step_regularity_cv"] = np.std(intervals) / np.mean(intervals)
        
        return analysis
    
    def extract_phase_features(self, cycles: List[Dict[str, Any]], keypoints: np.ndarray) -> Dict[str, Any]:
        """
        Extract features for different phases of gait cycles.
        
        Args:
            cycles: List of detected gait cycles
            keypoints: Keypoints array
            
        Returns:
            Dictionary containing phase-based features
        """
        if not cycles or keypoints is None:
            return {}
        
        features = {}
        
        # Analyze each cycle's phases
        stance_durations = []
        swing_durations = []
        
        for cycle in cycles:
            start_frame = cycle["start_frame"]
            end_frame = cycle["end_frame"]
            
            if end_frame <= start_frame or end_frame >= keypoints.shape[0]:
                continue
            
            cycle_keypoints = keypoints[start_frame:end_frame]
            
            # Estimate stance and swing phases (simplified)
            # Stance phase: when foot is on ground (lower y-position)
            # Swing phase: when foot is in air (higher y-position)
            
            foot = cycle.get("foot", "left")
            ankle_idx = 15 if foot == "left" else 16  # COCO format
            
            if ankle_idx < cycle_keypoints.shape[1]:
                ankle_y = cycle_keypoints[:, ankle_idx, 1]
                
                # Find stance phase (lower 60% of ankle positions)
                y_threshold = np.percentile(ankle_y, 60)
                stance_frames = np.sum(ankle_y <= y_threshold)
                swing_frames = len(ankle_y) - stance_frames
                
                stance_duration = stance_frames / self.fps
                swing_duration = swing_frames / self.fps
                
                stance_durations.append(stance_duration)
                swing_durations.append(swing_duration)
        
        # Calculate phase statistics
        if stance_durations:
            features["stance_duration_mean"] = np.mean(stance_durations)
            features["stance_duration_std"] = np.std(stance_durations)
            features["stance_percentage_mean"] = np.mean([s / (s + w) for s, w in zip(stance_durations, swing_durations)])
        
        if swing_durations:
            features["swing_duration_mean"] = np.mean(swing_durations)
            features["swing_duration_std"] = np.std(swing_durations)
            features["swing_percentage_mean"] = np.mean([w / (s + w) for s, w in zip(stance_durations, swing_durations)])
        
        # Calculate stance/swing ratio
        if stance_durations and swing_durations:
            ratios = [s / w for s, w in zip(stance_durations, swing_durations)]
            features["stance_swing_ratio_mean"] = np.mean(ratios)
            features["stance_swing_ratio_std"] = np.std(ratios)
        
        return features
    
    def detect_gait_events(self, pose_sequence: List[Dict[str, Any]]) -> Dict[str, List[int]]:
        """
        Detect specific gait events (heel strike, toe off, etc.).
        
        Args:
            pose_sequence: List of pose estimation results
            
        Returns:
            Dictionary mapping event types to frame indices
        """
        events = {
            "left_heel_strike": [],
            "right_heel_strike": [],
            "left_toe_off": [],
            "right_toe_off": []
        }
        
        keypoints_array = self._poses_to_array(pose_sequence)
        if keypoints_array is None:
            return events
        
        # Detect heel strikes
        left_ankle_idx = 15  # COCO format
        right_ankle_idx = 16
        
        if keypoints_array.shape[1] > max(left_ankle_idx, right_ankle_idx):
            left_ankle = keypoints_array[:, left_ankle_idx, :2]
            right_ankle = keypoints_array[:, right_ankle_idx, :2]
            
            events["left_heel_strike"] = self._detect_heel_strikes(left_ankle)
            events["right_heel_strike"] = self._detect_heel_strikes(right_ankle)
            
            # Toe-off detection (simplified - opposite of heel strike)
            events["left_toe_off"] = self._detect_toe_offs(left_ankle)
            events["right_toe_off"] = self._detect_toe_offs(right_ankle)
        
        return events
    
    def _detect_toe_offs(self, ankle_positions: np.ndarray) -> List[int]:
        """Detect toe-off events from ankle positions."""
        if len(ankle_positions) < 10:
            return []
        
        # Toe-off occurs at local maxima in y-position
        y_positions = ankle_positions[:, 1]
        toe_offs = []
        
        for i in range(1, len(y_positions) - 1):
            if (y_positions[i] > y_positions[i-1] and 
                y_positions[i] > y_positions[i+1]):
                
                # Check if this is significantly higher than surrounding points
                local_window = max(5, self.min_cycle_frames // 4)
                start_idx = max(0, i - local_window)
                end_idx = min(len(y_positions), i + local_window)
                
                local_max = np.max(y_positions[start_idx:end_idx])
                if y_positions[i] >= local_max - 2:  # Within 2 pixels of local maximum
                    # Ensure minimum distance from previous toe-off
                    if not toe_offs or (i - toe_offs[-1]) >= self.min_cycle_frames:
                        toe_offs.append(i)
        
        return toe_offs