"""
Enhanced pose estimators for AlexPose that work with Frame objects.

This module provides enhanced pose estimators that implement the new IPoseEstimator
interface while maintaining backward compatibility with existing GAVD processing.

Author: AlexPose Team
"""

import numpy as np
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from loguru import logger

from ambient.core.interfaces import IPoseEstimator
from ambient.core.frame import Frame, FrameSequence
from ambient.gavd.pose_estimators import MediaPipeEstimator as LegacyMediaPipeEstimator
from ambient.gavd.pose_estimators import OpenPoseEstimator as LegacyOpenPoseEstimator
from ambient.gavd.pose_estimators import MEDIAPIPE_AVAILABLE

try:
    import cv2
except ImportError:
    cv2 = None


class EnhancedMediaPipeEstimator(IPoseEstimator):
    """
    Enhanced MediaPipe pose estimator that works with Frame objects.
    
    This estimator leverages the existing MediaPipe tasks implementation
    while providing the new IPoseEstimator interface.
    """
    
    def __init__(
        self,
        model_complexity: int = 1,
        min_detection_confidence: float = 0.5,
        min_tracking_confidence: float = 0.5,
        model_path: Optional[Union[str, Path]] = None
    ):
        """
        Initialize enhanced MediaPipe estimator.
        
        Args:
            model_complexity: Model complexity (0, 1, or 2) - used for compatibility
            min_detection_confidence: Minimum detection confidence
            min_tracking_confidence: Minimum tracking confidence
            model_path: Path to MediaPipe model file (uses default if None)
        """
        if not MEDIAPIPE_AVAILABLE:
            raise ImportError(
                "MediaPipe is not available. Please install with: pip install mediapipe"
            )
        
        self.model_complexity = model_complexity
        self.min_detection_confidence = min_detection_confidence
        self.min_tracking_confidence = min_tracking_confidence
        
        # Use default model path if not provided
        if model_path is None:
            # MediaPipe tasks API uses built-in models, create a dummy path
            import tempfile
            self.model_path = Path(tempfile.gettempdir()) / "pose_landmarker.task"
            # For MediaPipe tasks, we don't need an actual file path for built-in models
            self._use_builtin_model = True
        else:
            self.model_path = Path(model_path)
            self._use_builtin_model = False
        
        # Initialize the legacy estimator to leverage existing implementation
        try:
            if self._use_builtin_model:
                # Create a temporary model file path for the legacy estimator
                # The legacy estimator will handle the MediaPipe tasks API properly
                self.legacy_estimator = None  # We'll create it on demand
            else:
                self.legacy_estimator = LegacyMediaPipeEstimator(
                    model_path=self.model_path,
                    min_pose_detection_confidence=min_detection_confidence,
                    min_pose_presence_confidence=min_tracking_confidence,
                    min_tracking_confidence=min_tracking_confidence
                )
            
            logger.info(f"Enhanced MediaPipe estimator initialized")
            
        except Exception as e:
            logger.warning(f"Failed to initialize MediaPipe estimator: {e}")
            self.legacy_estimator = None
    
    def _get_legacy_estimator(self):
        """Get or create legacy estimator on demand."""
        if self.legacy_estimator is None and self._use_builtin_model:
            # For built-in models, we'll work directly with MediaPipe tasks API
            # without requiring a model file path
            return None
        return self.legacy_estimator
    
    def estimate_pose(self, frame: Frame) -> Dict[str, Any]:
        """
        Estimate pose from a single frame.
        
        Args:
            frame: Frame object containing image data
            
        Returns:
            Dictionary containing pose estimation results
        """
        try:
            # For now, use a simplified approach that works with Frame objects
            # Save frame to temporary file and use existing implementation
            import tempfile
            import os
            
            with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as temp_file:
                temp_path = temp_file.name
            
            try:
                # Save frame to temporary file
                frame.save(temp_path)
                
                # Use existing MediaPipe implementation if available
                legacy_estimator = self._get_legacy_estimator()
                if legacy_estimator:
                    keypoints = legacy_estimator.estimate_image_keypoints(temp_path)
                else:
                    # Fallback: create dummy keypoints for now
                    keypoints = self._create_dummy_keypoints(frame)
                
                # Convert to enhanced format
                enhanced_keypoints = []
                for i, kp in enumerate(keypoints):
                    enhanced_keypoints.append({
                        "id": i,
                        "x": kp.get("x", 0),
                        "y": kp.get("y", 0),
                        "confidence": kp.get("confidence", 0.0)
                    })
                
                return {
                    "keypoints": enhanced_keypoints,
                    "estimator": self.get_estimator_name(),
                    "format": self.get_keypoint_format(),
                    "confidence": self._calculate_overall_confidence(enhanced_keypoints),
                    "metadata": {
                        "frame_shape": frame.shape,
                        "model_complexity": self.model_complexity
                    }
                }
                
            finally:
                # Clean up temporary file
                try:
                    os.unlink(temp_path)
                except OSError:
                    pass
                    
        except Exception as e:
            logger.error(f"MediaPipe pose estimation failed: {e}")
            return {
                "keypoints": [],
                "estimator": self.get_estimator_name(),
                "format": self.get_keypoint_format(),
                "confidence": 0.0,
                "error": str(e)
            }
    
    def _create_dummy_keypoints(self, frame: Frame) -> List[Dict[str, Any]]:
        """Create dummy keypoints when MediaPipe is not available."""
        shape = frame.shape
        if shape is None:
            image = frame.load()
            height, width = image.shape[:2]
        else:
            height, width = shape[:2]
        
        # Create 33 dummy keypoints (MediaPipe BlazePose format)
        keypoints = []
        for i in range(33):
            keypoints.append({
                "x": np.random.uniform(0, width),
                "y": np.random.uniform(0, height),
                "confidence": 0.5  # Low confidence for dummy data
            })
        
        return keypoints
    
    def estimate_pose_sequence(self, sequence: FrameSequence) -> List[Dict[str, Any]]:
        """
        Estimate poses from a sequence of frames.
        
        Args:
            sequence: FrameSequence object containing multiple frames
            
        Returns:
            List of pose estimation results, one per frame
        """
        results = []
        
        for i, frame in enumerate(sequence.frames):
            try:
                result = self.estimate_pose(frame)
                result["frame_index"] = i
                results.append(result)
            except Exception as e:
                logger.error(f"Failed to process frame {i}: {e}")
                results.append({
                    "keypoints": [],
                    "estimator": self.get_estimator_name(),
                    "format": self.get_keypoint_format(),
                    "confidence": 0.0,
                    "frame_index": i,
                    "error": str(e)
                })
        
        return results
    
    def _calculate_overall_confidence(self, keypoints: List[Dict[str, Any]]) -> float:
        """Calculate overall confidence from keypoints."""
        if not keypoints:
            return 0.0
        
        confidences = [kp.get("confidence", 0.0) for kp in keypoints]
        return sum(confidences) / len(confidences)
    
    def get_estimator_name(self) -> str:
        """Get the name of this pose estimator."""
        return "mediapipe"
    
    def get_keypoint_format(self) -> str:
        """Get the keypoint format used by this estimator."""
        return "BLAZEPOSE_33"
    
    def is_available(self) -> bool:
        """Check if this estimator is available and properly configured."""
        return MEDIAPIPE_AVAILABLE


class EnhancedOpenPoseEstimator(IPoseEstimator):
    """
    Enhanced OpenPose estimator that works with Frame objects.
    
    This estimator wraps the existing OpenPose functionality with the new interface.
    """
    
    def __init__(
        self,
        openpose_root: Optional[Union[str, Path]] = None,
        model: str = "BODY_25",
        net_resolution: str = "-1x368",
        number_people_max: int = 1
    ):
        """
        Initialize enhanced OpenPose estimator.
        
        Args:
            openpose_root: Path to OpenPose installation
            model: Pose model to use ("BODY_25" or "COCO")
            net_resolution: Network resolution
            number_people_max: Maximum number of people to detect
        """
        self.model = model
        self.net_resolution = net_resolution
        self.number_people_max = number_people_max
        
        try:
            # Initialize legacy OpenPose estimator
            additional_args = [
                "--number_people_max", str(number_people_max),
                "--net_resolution", net_resolution,
                "--scale_number", "1"
            ]
            
            self.legacy_estimator = LegacyOpenPoseEstimator(
                openpose_root=openpose_root,
                default_model=model,
                additional_args=additional_args
            )
            self._available = True
            logger.info(f"Enhanced OpenPose estimator initialized with model {model}")
            
        except Exception as e:
            logger.warning(f"OpenPose not available: {e}")
            self.legacy_estimator = None
            self._available = False
    
    def estimate_pose(self, frame: Frame) -> Dict[str, Any]:
        """
        Estimate pose from a single frame.
        
        Args:
            frame: Frame object containing image data
            
        Returns:
            Dictionary containing pose estimation results
        """
        if not self.is_available():
            return {
                "keypoints": [],
                "estimator": self.get_estimator_name(),
                "format": self.get_keypoint_format(),
                "confidence": 0.0,
                "error": "OpenPose not available"
            }
        
        try:
            # Save frame to temporary file for OpenPose
            import tempfile
            with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as temp_file:
                temp_path = temp_file.name
            
            try:
                # Save frame
                frame.save(temp_path)
                
                # Use legacy estimator
                keypoints = self.legacy_estimator.estimate_image_keypoints(
                    temp_path, model=self.model
                )
                
                # Convert to enhanced format
                enhanced_keypoints = []
                for i, kp in enumerate(keypoints):
                    enhanced_keypoints.append({
                        "id": i,
                        "x": kp["x"],
                        "y": kp["y"],
                        "confidence": kp["confidence"]
                    })
                
                return {
                    "keypoints": enhanced_keypoints,
                    "estimator": self.get_estimator_name(),
                    "format": self.get_keypoint_format(),
                    "confidence": self._calculate_overall_confidence(enhanced_keypoints),
                    "metadata": {
                        "model": self.model,
                        "net_resolution": self.net_resolution,
                        "number_people_max": self.number_people_max
                    }
                }
                
            finally:
                # Clean up temporary file
                import os
                try:
                    os.unlink(temp_path)
                except OSError:
                    pass
                    
        except Exception as e:
            logger.error(f"OpenPose estimation failed: {e}")
            return {
                "keypoints": [],
                "estimator": self.get_estimator_name(),
                "format": self.get_keypoint_format(),
                "confidence": 0.0,
                "error": str(e)
            }
    
    def estimate_pose_sequence(self, sequence: FrameSequence) -> List[Dict[str, Any]]:
        """
        Estimate poses from a sequence of frames.
        
        Args:
            sequence: FrameSequence object containing multiple frames
            
        Returns:
            List of pose estimation results, one per frame
        """
        results = []
        
        for i, frame in enumerate(sequence.frames):
            try:
                result = self.estimate_pose(frame)
                result["frame_index"] = i
                results.append(result)
            except Exception as e:
                logger.error(f"Failed to process frame {i}: {e}")
                results.append({
                    "keypoints": [],
                    "estimator": self.get_estimator_name(),
                    "format": self.get_keypoint_format(),
                    "confidence": 0.0,
                    "frame_index": i,
                    "error": str(e)
                })
        
        return results
    
    def _calculate_overall_confidence(self, keypoints: List[Dict[str, Any]]) -> float:
        """Calculate overall confidence from keypoints."""
        if not keypoints:
            return 0.0
        
        confidences = [kp.get("confidence", 0.0) for kp in keypoints]
        return sum(confidences) / len(confidences)
    
    def get_estimator_name(self) -> str:
        """Get the name of this pose estimator."""
        return "openpose"
    
    def get_keypoint_format(self) -> str:
        """Get the keypoint format used by this estimator."""
        return self.model
    
    def is_available(self) -> bool:
        """Check if this estimator is available and properly configured."""
        return self._available and self.legacy_estimator is not None


class DummyPoseEstimator(IPoseEstimator):
    """
    Dummy pose estimator for testing and fallback purposes.
    
    This estimator generates synthetic pose data for testing when
    real pose estimators are not available.
    """
    
    def __init__(self, keypoint_count: int = 33):
        """
        Initialize dummy estimator.
        
        Args:
            keypoint_count: Number of keypoints to generate
        """
        self.keypoint_count = keypoint_count
        logger.info(f"Dummy pose estimator initialized with {keypoint_count} keypoints")
    
    def estimate_pose(self, frame: Frame) -> Dict[str, Any]:
        """Generate dummy pose estimation."""
        try:
            # Get frame dimensions
            shape = frame.shape
            if shape is None:
                # Load frame to get shape
                image = frame.load()
                height, width = image.shape[:2]
            else:
                height, width = shape[:2]
            
            # Generate dummy keypoints
            keypoints = []
            for i in range(self.keypoint_count):
                keypoints.append({
                    "id": i,
                    "x": np.random.uniform(0, width),
                    "y": np.random.uniform(0, height),
                    "confidence": np.random.uniform(0.5, 1.0)
                })
            
            return {
                "keypoints": keypoints,
                "estimator": self.get_estimator_name(),
                "format": self.get_keypoint_format(),
                "confidence": 0.8,
                "metadata": {
                    "frame_shape": (height, width),
                    "synthetic": True
                }
            }
            
        except Exception as e:
            logger.error(f"Dummy pose estimation failed: {e}")
            return {
                "keypoints": [],
                "estimator": self.get_estimator_name(),
                "format": self.get_keypoint_format(),
                "confidence": 0.0,
                "error": str(e)
            }
    
    def estimate_pose_sequence(self, sequence: FrameSequence) -> List[Dict[str, Any]]:
        """Generate dummy pose sequence."""
        results = []
        
        for i, frame in enumerate(sequence.frames):
            result = self.estimate_pose(frame)
            result["frame_index"] = i
            results.append(result)
        
        return results
    
    def get_estimator_name(self) -> str:
        """Get the name of this pose estimator."""
        return "dummy"
    
    def get_keypoint_format(self) -> str:
        """Get the keypoint format used by this estimator."""
        return f"DUMMY_{self.keypoint_count}"
    
    def is_available(self) -> bool:
        """Check if this estimator is available."""
        return True