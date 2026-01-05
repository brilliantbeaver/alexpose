"""
Pose estimator factory for creating and managing different pose estimation backends.

This module provides a factory pattern for creating pose estimators with
automatic discovery and registration capabilities.

Author: AlexPose Team
"""

from pathlib import Path
from typing import Dict, Type, Optional, Any, List
from loguru import logger

from ambient.core.interfaces import IPoseEstimator
from ambient.gavd.pose_estimators import OpenPoseEstimator, MediaPipeEstimator
from ambient.pose.ultralytics_estimator import UltralyticsEstimator
from ambient.pose.alphapose_estimator import AlphaPoseEstimator


class PoseEstimatorFactory:
    """
    Factory for creating pose estimator instances.
    
    This factory supports automatic registration of pose estimators and
    provides a unified interface for creating estimators with different
    configurations.
    """
    
    def __init__(self):
        """Initialize the pose estimator factory."""
        self._estimators: Dict[str, Type[IPoseEstimator]] = {}
        self._register_default_estimators()
        
        logger.info("Pose estimator factory initialized")
    
    def _register_default_estimators(self):
        """Register default pose estimators."""
        # Register OpenPose estimator
        self.register_estimator("openpose", OpenPoseEstimator)
        
        # Register MediaPipe estimator
        self.register_estimator("mediapipe", MediaPipeEstimator)
        
        # Register Ultralytics YOLO pose estimators
        # YOLOv8-pose - 2023 release with COCO 17 keypoints
        self.register_estimator("yolov8-pose", UltralyticsEstimator)
        
        # YOLO11-pose - 2024 release with improved accuracy
        self.register_estimator("yolov11-pose", UltralyticsEstimator)
        
        # Register AlphaPose estimator
        self.register_estimator("alphapose", AlphaPoseEstimator)
        
        logger.info("Default pose estimators registered")
    
    def register_estimator(self, name: str, estimator_class: Type[IPoseEstimator]):
        """
        Register a pose estimator class.
        
        Args:
            name: Name to register the estimator under
            estimator_class: Class that implements IPoseEstimator
        """
        # Check if class implements IPoseEstimator interface (lenient check)
        # Some estimators may use alternative base classes (e.g., GAVD PoseEstimator)
        try:
            if not issubclass(estimator_class, IPoseEstimator):
                logger.warning(
                    f"Estimator class {estimator_class.__name__} does not implement IPoseEstimator interface. "
                    f"Registering anyway for compatibility."
                )
        except TypeError:
            # Handle cases where issubclass check fails
            logger.warning(f"Could not verify interface for {estimator_class.__name__}")
        
        self._estimators[name.lower()] = estimator_class
        logger.info(f"Registered pose estimator: {name}")
    
    def create_estimator(
        self, 
        estimator_type: str, 
        config: Optional[Dict[str, Any]] = None
    ) -> IPoseEstimator:
        """
        Create a pose estimator instance.
        
        Args:
            estimator_type: Type of estimator to create
            config: Configuration parameters for the estimator
            
        Returns:
            Configured pose estimator instance
            
        Raises:
            ValueError: If estimator type is not registered
            RuntimeError: If estimator creation fails
        """
        estimator_type = estimator_type.lower()
        config = config or {}
        
        if estimator_type not in self._estimators:
            available = ", ".join(self._estimators.keys())
            raise ValueError(
                f"Unknown estimator type: {estimator_type}. "
                f"Available types: {available}"
            )
        
        estimator_class = self._estimators[estimator_type]
        
        try:
            # Create estimator with configuration
            if estimator_type == "openpose":
                return self._create_openpose_estimator(config)
            elif estimator_type == "mediapipe":
                return self._create_mediapipe_estimator(config)
            elif estimator_type in ["yolov8-pose", "yolov11-pose"]:
                return self._create_ultralytics_estimator(config, estimator_type)
            elif estimator_type == "alphapose":
                return self._create_alphapose_estimator(config)
            else:
                # Generic creation for custom estimators
                return estimator_class(**config)
                
        except Exception as e:
            logger.error(f"Failed to create {estimator_type} estimator: {e}")
            raise RuntimeError(f"Failed to create {estimator_type} estimator: {e}")
    
    def _create_openpose_estimator(self, config: Dict[str, Any]) -> OpenPoseEstimator:
        """Create OpenPose estimator with configuration."""
        # OpenPose is not yet implemented, so just create with no args
        return OpenPoseEstimator()
    
    def _create_mediapipe_estimator(self, config: Dict[str, Any]) -> MediaPipeEstimator:
        """Create MediaPipe estimator with configuration."""
        # Extract MediaPipe-specific configuration
        model_path = config.get("model_path")
        if not model_path:
            # Use default model path
            model_path = Path("data/models/pose_landmarker_lite.task")
        
        default_model = config.get("default_model", "BODY_25")
        min_pose_detection_confidence = config.get("min_pose_detection_confidence", 0.5)
        min_pose_presence_confidence = config.get("min_pose_presence_confidence", 0.5)
        min_tracking_confidence = config.get("min_tracking_confidence", 0.5)
        
        return MediaPipeEstimator(
            model_path=model_path,
            default_model=default_model,
            min_pose_detection_confidence=min_pose_detection_confidence,
            min_pose_presence_confidence=min_pose_presence_confidence,
            min_tracking_confidence=min_tracking_confidence
        )
    
    def _create_ultralytics_estimator(self, config: Dict[str, Any], estimator_type: str) -> UltralyticsEstimator:
        """Create Ultralytics YOLO estimator with configuration."""
        # Determine model name based on estimator type
        # Use data/models/ directory for better organization
        if estimator_type == "yolov11-pose":
            default_model = "data/models/yolo11n-pose.pt"
        else:  # yolov8-pose
            default_model = "data/models/yolov8n-pose.pt"
        
        # Extract Ultralytics-specific configuration
        model_name = config.get("model_name", default_model)
        device = config.get("device", "auto")
        confidence_threshold = config.get("confidence_threshold", 0.25)
        iou_threshold = config.get("iou_threshold", 0.7)
        max_detections = config.get("max_detections", 1)
        
        return UltralyticsEstimator(
            model_name=model_name,
            device=device,
            confidence_threshold=confidence_threshold,
            iou_threshold=iou_threshold,
            max_detections=max_detections
        )
    
    def _create_alphapose_estimator(self, config: Dict[str, Any]) -> AlphaPoseEstimator:
        """Create AlphaPose estimator with configuration."""
        # Extract AlphaPose-specific configuration
        alphapose_root = config.get("alphapose_root")
        config_file = config.get("config_file")
        checkpoint = config.get("checkpoint")
        detector = config.get("detector", "yolo")
        confidence_threshold = config.get("confidence_threshold", 0.3)
        nms_threshold = config.get("nms_threshold", 0.6)
        
        return AlphaPoseEstimator(
            alphapose_root=alphapose_root,
            config_file=config_file,
            checkpoint=checkpoint,
            detector=detector,
            confidence_threshold=confidence_threshold,
            nms_threshold=nms_threshold
        )
    
    def list_available_estimators(self) -> List[str]:
        """
        Get list of available estimator types.
        
        Returns:
            List of registered estimator names
        """
        return list(self._estimators.keys())
    
    def get_estimator_info(self, estimator_type: str) -> Dict[str, Any]:
        """
        Get information about a specific estimator type.
        
        Args:
            estimator_type: Type of estimator to get info for
            
        Returns:
            Dictionary containing estimator information
        """
        estimator_type = estimator_type.lower()
        
        if estimator_type not in self._estimators:
            raise ValueError(f"Unknown estimator type: {estimator_type}")
        
        estimator_class = self._estimators[estimator_type]
        
        info = {
            "name": estimator_type,
            "class": estimator_class.__name__,
            "module": estimator_class.__module__,
            "available": False
        }
        
        # Check if estimator is available
        try:
            # Create a temporary instance to check availability
            temp_config = self._get_default_config(estimator_type)
            temp_estimator = self.create_estimator(estimator_type, temp_config)
            info["available"] = temp_estimator.is_available()
        except Exception as e:
            info["error"] = str(e)
        
        return info
    
    def _get_default_config(self, estimator_type: str) -> Dict[str, Any]:
        """Get default configuration for an estimator type."""
        if estimator_type == "openpose":
            return {
                "openpose_root": None,  # Will use environment variable
                "default_model": "BODY_25"
            }
        elif estimator_type == "mediapipe":
            return {
                "model_path": Path("data/models/pose_landmarker_lite.task"),
                "default_model": "BODY_25"
            }
        elif estimator_type == "yolov8-pose":
            return {
                "model_name": "data/models/yolov8n-pose.pt",
                "device": "auto"
            }
        elif estimator_type == "yolov11-pose":
            return {
                "model_name": "data/models/yolo11n-pose.pt",
                "device": "auto"
            }
        elif estimator_type == "alphapose":
            return {
                "alphapose_root": None,  # Must be configured
                "detector": "yolo"
            }
        else:
            return {}
    
    def get_available_estimators(self) -> List[str]:
        """
        Get list of estimators that are currently available (properly configured).
        
        Returns:
            List of available estimator names
        """
        available = []
        
        for estimator_type in self._estimators.keys():
            try:
                info = self.get_estimator_info(estimator_type)
                if info.get("available", False):
                    available.append(estimator_type)
            except Exception:
                continue
        
        return available
    
    def create_best_available_estimator(
        self, 
        preferred_order: Optional[List[str]] = None
    ) -> Optional[IPoseEstimator]:
        """
        Create the best available pose estimator.
        
        Args:
            preferred_order: Preferred order of estimators to try
            
        Returns:
            Best available pose estimator or None if none available
        """
        if preferred_order is None:
            preferred_order = ["mediapipe", "openpose"]  # Default preference
        
        # Try preferred estimators first
        for estimator_type in preferred_order:
            if estimator_type in self._estimators:
                try:
                    estimator = self.create_estimator(estimator_type)
                    if estimator.is_available():
                        logger.info(f"Created best available estimator: {estimator_type}")
                        return estimator
                except Exception as e:
                    logger.warning(f"Failed to create {estimator_type} estimator: {e}")
        
        # Try any remaining available estimators
        available = self.get_available_estimators()
        for estimator_type in available:
            if estimator_type not in preferred_order:
                try:
                    estimator = self.create_estimator(estimator_type)
                    logger.info(f"Created fallback estimator: {estimator_type}")
                    return estimator
                except Exception as e:
                    logger.warning(f"Failed to create {estimator_type} estimator: {e}")
        
        logger.error("No pose estimators are available")
        return None


# Global factory instance
_factory_instance = None


def get_pose_estimator_factory() -> PoseEstimatorFactory:
    """
    Get the global pose estimator factory instance.
    
    Returns:
        Global PoseEstimatorFactory instance
    """
    global _factory_instance
    if _factory_instance is None:
        _factory_instance = PoseEstimatorFactory()
    return _factory_instance


def create_pose_estimator(
    estimator_type: str, 
    config: Optional[Dict[str, Any]] = None
) -> IPoseEstimator:
    """
    Convenience function to create a pose estimator.
    
    Args:
        estimator_type: Type of estimator to create
        config: Configuration parameters
        
    Returns:
        Configured pose estimator instance
    """
    factory = get_pose_estimator_factory()
    return factory.create_estimator(estimator_type, config)


def get_available_pose_estimators() -> List[str]:
    """
    Convenience function to get available pose estimators.
    
    Returns:
        List of available estimator names
    """
    factory = get_pose_estimator_factory()
    return factory.get_available_estimators()


def create_best_pose_estimator(
    preferred_order: Optional[List[str]] = None
) -> Optional[IPoseEstimator]:
    """
    Convenience function to create the best available pose estimator.
    
    Args:
        preferred_order: Preferred order of estimators to try
        
    Returns:
        Best available pose estimator or None if none available
    """
    factory = get_pose_estimator_factory()
    return factory.create_best_available_estimator(preferred_order)