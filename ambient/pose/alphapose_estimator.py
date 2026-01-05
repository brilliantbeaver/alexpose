"""
AlphaPose estimator implementation.

This module provides pose estimation using AlphaPose models
with support for both the new Frame-based API and legacy compatibility.

Author: AlexPose Team
"""

import json
import subprocess
import tempfile
import shutil
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
    import cv2
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False


Keypoint = Dict[str, Union[float, int]]


class AlphaPoseEstimator(IPoseEstimator):
    """
    Pose estimator using AlphaPose models.
    
    This estimator interfaces with AlphaPose through command-line execution
    to provide high-accuracy pose estimation. It supports the new Frame-based
    API while maintaining backward compatibility.
    
    Note: This requires AlphaPose to be installed and properly configured.
    """
    
    def __init__(
        self,
        alphapose_root: Optional[Union[str, Path]] = None,
        config_file: Optional[str] = None,
        checkpoint: Optional[str] = None,
        detector: str = "yolo",
        confidence_threshold: float = 0.3,
        nms_threshold: float = 0.6
    ):
        """
        Initialize AlphaPose estimator.
        
        Args:
            alphapose_root: Path to AlphaPose installation
            config_file: Path to AlphaPose config file
            checkpoint: Path to model checkpoint
            detector: Person detector to use ('yolo', 'tracker')
            confidence_threshold: Confidence threshold for pose detection
            nms_threshold: NMS threshold for person detection
        """
        if not CV2_AVAILABLE:
            raise ImportError("OpenCV is required for AlphaPose estimator")
        
        # Set up paths
        self.alphapose_root = Path(alphapose_root or "/path/to/AlphaPose").expanduser()
        self.config_file = config_file
        self.checkpoint = checkpoint
        self.detector = detector
        self.confidence_threshold = confidence_threshold
        self.nms_threshold = nms_threshold
        
        # Validate installation
        self._validate_installation()
        
        logger.info(f"AlphaPose estimator initialized with {detector} detector")
    
    def _validate_installation(self):
        """Validate AlphaPose installation."""
        if not self.alphapose_root.exists():
            raise FileNotFoundError(
                f"AlphaPose root directory not found: {self.alphapose_root}. "
                "Please install AlphaPose and set the correct path."
            )
        
        # Check for main script
        main_script = self.alphapose_root / "scripts" / "demo_inference.py"
        if not main_script.exists():
            # Try alternative locations
            main_script = self.alphapose_root / "demo_inference.py"
            if not main_script.exists():
                raise FileNotFoundError(
                    f"AlphaPose demo script not found. Expected at: "
                    f"{self.alphapose_root}/scripts/demo_inference.py"
                )
        
        self.main_script = main_script
    
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
            
            # Create temporary file for AlphaPose processing
            with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as temp_file:
                temp_path = temp_file.name
            
            try:
                # Save frame to temporary file
                frame.save(temp_path)
                
                # Use legacy method to process the temporary file
                keypoints = self.estimate_image_keypoints(temp_path)
                
                # Convert to new format
                result = {
                    "keypoints": keypoints,
                    "estimator": self.get_estimator_name(),
                    "format": self.get_keypoint_format(),
                    "frame_metadata": frame.metadata,
                    "confidence_scores": [kp.get("confidence", 0.0) for kp in keypoints],
                    "num_keypoints": len(keypoints),
                    "processing_metadata": {
                        "detector": self.detector,
                        "frame_shape": frame_data.shape,
                        "frame_format": frame.format,
                        "confidence_threshold": self.confidence_threshold
                    }
                }
                
                return result
                
            except Exception as save_error:
                logger.error(f"Failed to save frame to temporary file: {save_error}")
                raise FrameError(f"Frame processing failed: {save_error}")
            finally:
                # Clean up temporary file
                try:
                    Path(temp_path).unlink()
                except OSError:
                    pass
                    
        except Exception as e:
            logger.error(f"AlphaPose estimation failed for frame: {e}")
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
        
        # For AlphaPose, it's more efficient to save frames to a temporary directory
        # and process them as a batch
        with tempfile.TemporaryDirectory(prefix="alphapose_frames_") as temp_dir:
            temp_dir_path = Path(temp_dir)
            
            # Save all frames to temporary directory
            frame_paths = []
            for i, frame in enumerate(sequence.frames):
                try:
                    frame_path = temp_dir_path / f"frame_{i:06d}.jpg"
                    frame.save(frame_path)
                    frame_paths.append((i, frame_path))
                except Exception as e:
                    logger.warning(f"Failed to save frame {i}: {e}")
            
            if frame_paths:
                # Process batch with AlphaPose
                try:
                    batch_results = self._process_image_batch([fp[1] for fp in frame_paths])
                    
                    # Map results back to frames
                    for (frame_idx, _), keypoints in zip(frame_paths, batch_results):
                        frame = sequence.frames[frame_idx]
                        
                        result = {
                            "keypoints": keypoints,
                            "estimator": self.get_estimator_name(),
                            "format": self.get_keypoint_format(),
                            "sequence_index": frame_idx,
                            "frame_metadata": frame.metadata,
                            "confidence_scores": [kp.get("confidence", 0.0) for kp in keypoints],
                            "num_keypoints": len(keypoints),
                            "processing_metadata": {
                                "detector": self.detector,
                                "batch_processing": True
                            }
                        }
                        
                        results.append(result)
                
                except Exception as e:
                    logger.error(f"Batch processing failed: {e}")
                    # Add empty results for all frames
                    for frame_idx, _ in frame_paths:
                        results.append({
                            "keypoints": [],
                            "estimator": self.get_estimator_name(),
                            "format": self.get_keypoint_format(),
                            "error": str(e),
                            "sequence_index": frame_idx,
                            "confidence_scores": [],
                            "num_keypoints": 0
                        })
        
        # Sort results by sequence index
        results.sort(key=lambda x: x.get("sequence_index", 0))
        
        return results
    
    def get_estimator_name(self) -> str:
        """Get the name of this pose estimator."""
        return "AlphaPose"
    
    def get_keypoint_format(self) -> str:
        """Get the keypoint format used by this estimator."""
        return "COCO_17"  # AlphaPose typically uses COCO format
    
    def is_available(self) -> bool:
        """Check if this estimator is available and properly configured."""
        try:
            return (
                self.alphapose_root.exists() and
                self.main_script.exists() and
                CV2_AVAILABLE
            )
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
        Estimate 2D keypoints from an image using AlphaPose.
        
        Args:
            image_path: Path to the input image frame.
            model: Pose model to use (ignored for AlphaPose, kept for interface compatibility).
            bbox: Optional bounding box to crop the image before processing.
        
        Returns:
            A list of keypoints as dicts with keys: x, y, confidence.
        """
        image_path = Path(image_path)
        if not image_path.exists():
            raise FileNotFoundError(f"Image not found: {image_path}")
        
        # Process single image
        results = self._process_image_batch([image_path])
        return results[0] if results else []
    
    def estimate_video_keypoints(
        self,
        video_path: Union[str, Path],
        model: str = "BODY_25",
    ) -> List[List[Keypoint]]:
        """
        Estimate keypoints for all frames of a video using AlphaPose.
        
        Args:
            video_path: Path to the input video file.
            model: Pose model to use (ignored for AlphaPose, kept for interface compatibility).
        
        Returns:
            A list where index i corresponds to frame index i. Missing frames
            are represented by empty lists.
        """
        video_path = Path(video_path)
        if not video_path.exists():
            raise FileNotFoundError(f"Video not found: {video_path}")
        
        with tempfile.TemporaryDirectory(prefix="alphapose_video_") as temp_dir:
            temp_dir_path = Path(temp_dir)
            output_dir = temp_dir_path / "output"
            
            # Build AlphaPose command for video
            cmd = [
                "python", str(self.main_script),
                "--indir", str(video_path.parent),
                "--outdir", str(output_dir),
                "--detector", self.detector,
                "--pose_track"
            ]
            
            # Add optional parameters
            if self.config_file:
                cmd.extend(["--cfg", self.config_file])
            if self.checkpoint:
                cmd.extend(["--checkpoint", self.checkpoint])
            
            cmd.extend([
                "--pose_flow",
                "--format", "coco",
                "--min_box_area", "0",
                "--detbatch", "1"
            ])
            
            # Execute AlphaPose
            try:
                result = subprocess.run(
                    cmd,
                    cwd=str(self.alphapose_root),
                    capture_output=True,
                    text=True,
                    check=False
                )
                
                if result.returncode != 0:
                    logger.error(f"AlphaPose failed: {result.stderr}")
                    return []
                
                # Parse output JSON
                json_files = list(output_dir.glob("*.json"))
                if not json_files:
                    logger.warning("No AlphaPose output files found")
                    return []
                
                return self._parse_alphapose_json(json_files[0])
                
            except Exception as e:
                logger.error(f"AlphaPose video processing failed: {e}")
                return []
    
    def _process_image_batch(self, image_paths: List[Path]) -> List[List[Keypoint]]:
        """Process a batch of images with AlphaPose."""
        if not image_paths:
            return []
        
        with tempfile.TemporaryDirectory(prefix="alphapose_batch_") as temp_dir:
            temp_dir_path = Path(temp_dir)
            input_dir = temp_dir_path / "input"
            output_dir = temp_dir_path / "output"
            input_dir.mkdir()
            
            # Copy images to input directory
            copied_paths = []
            for i, img_path in enumerate(image_paths):
                dest_path = input_dir / f"image_{i:06d}{img_path.suffix}"
                shutil.copy2(img_path, dest_path)
                copied_paths.append(dest_path)
            
            # Build AlphaPose command
            cmd = [
                "python", str(self.main_script),
                "--indir", str(input_dir),
                "--outdir", str(output_dir),
                "--detector", self.detector,
                "--format", "coco",
                "--min_box_area", "0",
                "--detbatch", "1"
            ]
            
            # Add optional parameters
            if self.config_file:
                cmd.extend(["--cfg", self.config_file])
            if self.checkpoint:
                cmd.extend(["--checkpoint", self.checkpoint])
            
            # Execute AlphaPose
            try:
                result = subprocess.run(
                    cmd,
                    cwd=str(self.alphapose_root),
                    capture_output=True,
                    text=True,
                    check=False
                )
                
                if result.returncode != 0:
                    logger.error(f"AlphaPose failed: {result.stderr}")
                    return [[] for _ in image_paths]
                
                # Parse output JSON
                json_files = list(output_dir.glob("*.json"))
                if not json_files:
                    logger.warning("No AlphaPose output files found")
                    return [[] for _ in image_paths]
                
                # Parse results and map back to input order
                all_results = self._parse_alphapose_json(json_files[0])
                
                # Map results back to original image order
                results = []
                for i, _ in enumerate(image_paths):
                    if i < len(all_results):
                        results.append(all_results[i])
                    else:
                        results.append([])
                
                return results
                
            except Exception as e:
                logger.error(f"AlphaPose batch processing failed: {e}")
                return [[] for _ in image_paths]
    
    def _parse_alphapose_json(self, json_path: Path) -> List[List[Keypoint]]:
        """Parse AlphaPose JSON output."""
        try:
            with open(json_path, 'r') as f:
                data = json.load(f)
            
            # Group by image_id (frame)
            frame_results = {}
            
            for item in data:
                image_id = item.get('image_id', 0)
                keypoints_flat = item.get('keypoints', [])
                score = item.get('score', 0.0)
                
                # Convert flat keypoints to structured format
                keypoints = []
                for i in range(0, len(keypoints_flat), 3):
                    if i + 2 < len(keypoints_flat):
                        keypoints.append({
                            "x": float(keypoints_flat[i]),
                            "y": float(keypoints_flat[i + 1]),
                            "confidence": float(keypoints_flat[i + 2])
                        })
                
                if image_id not in frame_results:
                    frame_results[image_id] = []
                
                # Only keep the highest scoring person per frame
                if not frame_results[image_id] or score > frame_results[image_id][0].get('score', 0):
                    frame_results[image_id] = keypoints
            
            # Convert to list format
            max_frame = max(frame_results.keys()) if frame_results else 0
            results = []
            
            for i in range(max_frame + 1):
                if i in frame_results:
                    results.append(frame_results[i])
                else:
                    results.append([])
            
            return results
            
        except Exception as e:
            logger.error(f"Failed to parse AlphaPose JSON: {e}")
            return []
    
    def supports_video_batch(self) -> bool:
        """AlphaPose supports video batch processing."""
        return True
    
    def get_accuracy_metrics(self) -> Dict[str, Any]:
        """Get accuracy metrics for this estimator."""
        return {
            "estimator": self.get_estimator_name(),
            "detector": self.detector,
            "confidence_threshold": self.confidence_threshold,
            "nms_threshold": self.nms_threshold,
            "keypoint_format": self.get_keypoint_format(),
            "typical_accuracy": "High (90%+ on COCO dataset)",
            "strengths": [
                "High accuracy pose estimation",
                "Good performance on challenging poses",
                "Robust person detection",
                "Multi-person support"
            ],
            "limitations": [
                "Requires external installation",
                "Slower than real-time methods",
                "Command-line interface dependency"
            ]
        }