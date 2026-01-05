"""
Ultralytics YOLO pose estimator implementation.

This module provides pose estimation using Ultralytics YOLO models
with support for both the new Frame-based API and legacy compatibility.

Author: AlexPose Team
"""

import tempfile
import os
from pathlib import Path
from typing import Dict, List, Optional, Union, Any
import numpy as np
from loguru import logger

# Import Frame classes with fallback for development
try:
    from ambient.core.frame import Frame, FrameSequence, FrameError
    from ambient.core.interfaces import IPoseEstimator
    FRAME_SUPPORT = True
except ImportError:
    # Fallback for development/testing
    Frame = Any
    FrameSequence = Any
    FrameError = Exception
    IPoseEstimator = object
    FRAME_SUPPORT = False

try:
    from ultralytics import YOLO
    import cv2
    ULTRALYTICS_AVAILABLE = True
except ImportError:
    ULTRALYTICS_AVAILABLE = False

# Import model utilities for path resolution
try:
    from ambient.utils.model_utils import resolve_yolo_model_path
    MODEL_UTILS_AVAILABLE = True
except ImportError:
    MODEL_UTILS_AVAILABLE = False
    logger.warning("Model utilities not available - using direct model paths")


Keypoint = Dict[str, Union[float, int]]


class UltralyticsEstimator(IPoseEstimator):
    """
    Pose estimator using Ultralytics YOLO models.
    
    This estimator uses YOLO pose detection models to extract human keypoints
    with support for both image and video processing. It supports the new
    Frame-based API while maintaining backward compatibility.
    """
    
    def __init__(
        self,
        model_name: str = "yolov8n-pose.pt",
        device: str = "auto",
        confidence_threshold: float = 0.25,
        iou_threshold: float = 0.7,
        max_detections: int = 1
    ):
        """
        Initialize Ultralytics pose estimator.
        
        Args:
            model_name: YOLO model name or path (e.g., "yolov8n-pose.pt" or "data/models/yolov8n-pose.pt")
            device: Device to run inference on ('cpu', 'cuda', 'auto')
            confidence_threshold: Confidence threshold for detections
            iou_threshold: IoU threshold for NMS
            max_detections: Maximum number of people to detect
        """
        if not ULTRALYTICS_AVAILABLE:
            raise ImportError(
                "Ultralytics is not installed. Please install it with: "
                "pip install ultralytics"
            )
        
        # Store original model name
        self.model_name = model_name
        
        # Resolve model path with intelligent fallback
        if MODEL_UTILS_AVAILABLE:
            resolved_model_path = resolve_yolo_model_path(model_name)
            logger.debug(f"Resolved model path: {model_name} -> {resolved_model_path}")
        else:
            resolved_model_path = model_name
            logger.debug(f"Using model path directly: {model_name}")
        
        self.resolved_model_path = resolved_model_path
        self.device = device
        self.confidence_threshold = confidence_threshold
        self.iou_threshold = iou_threshold
        self.max_detections = max_detections
        
        # Initialize model
        try:
            self.model = YOLO(resolved_model_path)
            if device != "auto":
                self.model.to(device)
            logger.info(f"Ultralytics YOLO model loaded: {model_name} (from {resolved_model_path})")
        except Exception as e:
            logger.error(f"Failed to load YOLO model {model_name} (resolved: {resolved_model_path}): {e}")
            raise RuntimeError(f"Failed to load YOLO model: {e}")
    
    # New Frame-based methods
    def estimate_pose(self, frame: Frame) -> Dict[str, Any]:
        """
        Estimate pose from a single Frame object.
        
        Args:
            frame: Frame object containing image data
            
        Returns:
            Dictionary containing pose estimation results
        """
        if not FRAME_SUPPORT:
            raise RuntimeError("Frame support not available - Frame classes not imported")
        
        try:
            # Load frame data
            frame_data = frame.load()
            
            # Convert to BGR if needed (YOLO expects BGR)
            if frame.format == "RGB":
                import cv2
                frame_data = cv2.cvtColor(frame_data, cv2.COLOR_RGB2BGR)
            
            # Run YOLO inference
            results = self.model(
                frame_data,
                conf=self.confidence_threshold,
                iou=self.iou_threshold,
                max_det=self.max_detections,
                verbose=False
            )
            
            # Parse results
            keypoints = self._parse_yolo_results(results)
            
            # Convert to new format
            result = {
                "keypoints": keypoints,
                "estimator": self.get_estimator_name(),
                "format": self.get_keypoint_format(),
                "frame_metadata": frame.metadata,
                "confidence_scores": [kp.get("confidence", 0.0) for kp in keypoints],
                "num_keypoints": len(keypoints),
                "processing_metadata": {
                    "model_name": self.model_name,
                    "device": str(self.model.device),
                    "frame_shape": frame_data.shape,
                    "frame_format": frame.format,
                    "confidence_threshold": self.confidence_threshold
                }
            }
            
            return result
            
        except Exception as e:
            logger.error(f"Ultralytics estimation failed for frame: {e}")
            # Return empty result with error information
            return {
                "keypoints": [],
                "estimator": self.get_estimator_name(),
                "format": self.get_keypoint_format(),
                "error": str(e),
                "frame_metadata": frame.metadata,
                "confidence_scores": [],
                "num_keypoints": 0
            }
    
    def estimate_pose_sequence(self, sequence: FrameSequence) -> List[Dict[str, Any]]:
        """
        Estimate poses from a FrameSequence.
        
        Args:
            sequence: FrameSequence object containing multiple frames
            
        Returns:
            List of pose estimation results, one per frame
        """
        if not FRAME_SUPPORT:
            raise RuntimeError("Frame support not available - Frame classes not imported")
        
        results = []
        
        # Process frames in batches for memory efficiency
        def process_batch(batch_frames, start_idx):
            batch_results = []
            
            # Prepare batch data
            batch_images = []
            valid_indices = []
            
            for i, frame_data in enumerate(batch_frames):
                frame_idx = start_idx + i
                frame = sequence.frames[frame_idx]
                
                try:
                    # Convert to BGR if needed
                    if frame.format == "RGB":
                        import cv2
                        frame_data = cv2.cvtColor(frame_data, cv2.COLOR_RGB2BGR)
                    
                    batch_images.append(frame_data)
                    valid_indices.append(frame_idx)
                    
                except Exception as e:
                    logger.warning(f"Failed to prepare frame {frame_idx}: {e}")
            
            # Run batch inference
            if batch_images:
                try:
                    batch_results_raw = self.model(
                        batch_images,
                        conf=self.confidence_threshold,
                        iou=self.iou_threshold,
                        max_det=self.max_detections,
                        verbose=False
                    )
                    
                    # Process each result
                    for i, (result_raw, frame_idx) in enumerate(zip(batch_results_raw, valid_indices)):
                        frame = sequence.frames[frame_idx]
                        
                        try:
                            keypoints = self._parse_yolo_results([result_raw])
                            
                            pose_result = {
                                "keypoints": keypoints,
                                "estimator": self.get_estimator_name(),
                                "format": self.get_keypoint_format(),
                                "sequence_index": frame_idx,
                                "frame_metadata": frame.metadata,
                                "confidence_scores": [kp.get("confidence", 0.0) for kp in keypoints],
                                "num_keypoints": len(keypoints),
                                "processing_metadata": {
                                    "model_name": self.model_name,
                                    "batch_index": i
                                }
                            }
                            
                            batch_results.append(pose_result)
                            
                        except Exception as e:
                            logger.warning(f"Failed to process result for frame {frame_idx}: {e}")
                            # Add empty result for failed frame
                            batch_results.append({
                                "keypoints": [],
                                "estimator": self.get_estimator_name(),
                                "format": self.get_keypoint_format(),
                                "error": str(e),
                                "sequence_index": frame_idx,
                                "confidence_scores": [],
                                "num_keypoints": 0
                            })
                
                except Exception as e:
                    logger.error(f"Batch inference failed: {e}")
                    # Add empty results for all frames in batch
                    for frame_idx in valid_indices:
                        batch_results.append({
                            "keypoints": [],
                            "estimator": self.get_estimator_name(),
                            "format": self.get_keypoint_format(),
                            "error": str(e),
                            "sequence_index": frame_idx,
                            "confidence_scores": [],
                            "num_keypoints": 0
                        })
            
            return batch_results
        
        # Process sequence in batches
        batch_results = sequence.process_in_batches(
            process_batch, 
            batch_size=8,  # Process 8 frames at a time
            unload_after_processing=True
        )
        
        # Flatten batch results
        for batch in batch_results:
            if batch:
                results.extend(batch)
        
        return results
    
    def get_estimator_name(self) -> str:
        """Get the name of this pose estimator."""
        return "Ultralytics"
    
    def get_keypoint_format(self) -> str:
        """Get the keypoint format used by this estimator."""
        return "COCO_17"  # YOLO pose uses COCO format
    
    def is_available(self) -> bool:
        """Check if this estimator is available and properly configured."""
        try:
            return ULTRALYTICS_AVAILABLE and self.model is not None
        except Exception:
            return False
    
    # Legacy methods for backward compatibility
    def estimate_image_keypoints(
        self,
        image_path: str,
        model: str = "BODY_25",
        bbox: Optional[Dict[str, Union[int, float]]] = None,
    ) -> List[Keypoint]:
        """
        Estimate 2D keypoints from an image using Ultralytics YOLO.
        
        Args:
            image_path: Path to the input image frame.
            model: Pose model to use (ignored for YOLO, kept for interface compatibility).
            bbox: Optional bounding box to crop the image before processing.
        
        Returns:
            A list of keypoints as dicts with keys: x, y, confidence.
        """
        image_path = Path(image_path)
        if not image_path.exists():
            raise FileNotFoundError(f"Image not found: {image_path}")
        
        # Load image
        image = cv2.imread(str(image_path))
        if image is None:
            raise ValueError(f"Could not load image: {image_path}")
        
        # Apply bounding box crop if provided
        if bbox and all(k in bbox for k in ["left", "top", "width", "height"]):
            left = int(bbox["left"])
            top = int(bbox["top"])
            width = int(bbox["width"])
            height = int(bbox["height"])
            
            # Ensure bounds are within image
            left = max(0, min(left, image.shape[1] - 1))
            top = max(0, min(top, image.shape[0] - 1))
            right = min(left + width, image.shape[1])
            bottom = min(top + height, image.shape[0])
            
            image = image[top:bottom, left:right]
        
        # Run YOLO inference
        results = self.model(
            image,
            conf=self.confidence_threshold,
            iou=self.iou_threshold,
            max_det=self.max_detections,
            verbose=False
        )
        
        # Parse results
        return self._parse_yolo_results(results)
    
    def estimate_video_keypoints(
        self,
        video_path: Union[str, Path],
        model: str = "BODY_25",
    ) -> List[List[Keypoint]]:
        """
        Estimate keypoints for all frames of a video using Ultralytics YOLO.
        
        Args:
            video_path: Path to the input video file.
            model: Pose model to use (ignored for YOLO, kept for interface compatibility).
        
        Returns:
            A list where index i corresponds to frame index i. Missing frames
            are represented by empty lists.
        """
        video_path = Path(video_path)
        if not video_path.exists():
            raise FileNotFoundError(f"Video not found: {video_path}")
        
        # Use YOLO's built-in video processing
        results = self.model(
            str(video_path),
            conf=self.confidence_threshold,
            iou=self.iou_threshold,
            max_det=self.max_detections,
            verbose=False,
            stream=True  # Process video frame by frame
        )
        
        frame_results = []
        for result in results:
            keypoints = self._parse_yolo_results([result])
            frame_results.append(keypoints)
        
        return frame_results
    
    def _parse_yolo_results(self, results) -> List[Keypoint]:
        """Parse YOLO results into keypoint format."""
        if not results or len(results) == 0:
            return []
        
        result = results[0]  # Take first result
        
        # Check if keypoints are available
        if not hasattr(result, 'keypoints') or result.keypoints is None:
            return []
        
        keypoints_data = result.keypoints.data
        
        if keypoints_data is None or len(keypoints_data) == 0:
            return []
        
        # Take first person's keypoints
        person_keypoints = keypoints_data[0]  # Shape: [17, 3] for COCO format
        
        keypoints = []
        for i in range(len(person_keypoints)):
            kp = person_keypoints[i]
            if len(kp) >= 3:  # x, y, confidence
                keypoints.append({
                    "x": float(kp[0]),
                    "y": float(kp[1]),
                    "confidence": float(kp[2])
                })
        
        return keypoints
    
    def supports_video_batch(self) -> bool:
        """Ultralytics YOLO supports video batch processing."""
        return True
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the loaded model."""
        info = {
            "model_name": self.model_name,
            "resolved_path": getattr(self, 'resolved_model_path', self.model_name),
            "device": str(self.model.device) if self.model else "unknown",
            "confidence_threshold": self.confidence_threshold,
            "iou_threshold": self.iou_threshold,
            "max_detections": self.max_detections,
            "keypoint_format": self.get_keypoint_format(),
            "available": self.is_available()
        }
        
        if self.model:
            try:
                # Get model metadata if available
                info["model_type"] = "YOLOv8-pose"
                info["input_size"] = getattr(self.model.model, 'imgsz', 640)
            except Exception:
                pass
        
        return info