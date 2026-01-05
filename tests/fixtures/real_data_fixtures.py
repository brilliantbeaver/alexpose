"""
Real data fixtures for comprehensive testing with minimal mocking.
Enhanced for property-based testing infrastructure.
"""

import pytest
import numpy as np
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
import json
import csv
import tempfile
import shutil
from dataclasses import dataclass
from unittest.mock import Mock

try:
    import cv2
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False

try:
    from ambient.core.frame import Frame, FrameSequence
    from ambient.core.data_models import GaitFeatures, GaitMetrics, Keypoint
    AMBIENT_AVAILABLE = True
except ImportError:
    AMBIENT_AVAILABLE = False


@dataclass
class TestDataset:
    """Container for test dataset information."""
    name: str
    path: Path
    data_type: str
    metadata: Dict[str, Any]
    is_synthetic: bool = True


class RealDataManager:
    """Enhanced manager for real and synthetic test data."""
    
    def __init__(self, base_dir: Optional[Path] = None):
        self.base_dir = base_dir or Path("data/test_data/real")
        self.base_dir.mkdir(parents=True, exist_ok=True)
        self.synthetic_dir = self.base_dir / "synthetic"
        self.synthetic_dir.mkdir(exist_ok=True)
        self.datasets: Dict[str, TestDataset] = {}
        self._initialize_datasets()
    
    def _initialize_datasets(self):
        """Initialize available datasets."""
        # Video datasets
        video_types = ["normal_walking", "abnormal_gait", "multiple_subjects", "parkinsonian_gait", "hemiplegic_gait"]
        for video_type in video_types:
            real_path = self.base_dir / f"{video_type}.mp4"
            synthetic_path = self.synthetic_dir / f"{video_type}.mp4"
            
            if real_path.exists():
                self.datasets[video_type] = TestDataset(
                    name=video_type,
                    path=real_path,
                    data_type="video",
                    metadata=self._get_video_metadata(real_path),
                    is_synthetic=False
                )
            elif synthetic_path.exists() or CV2_AVAILABLE:
                if not synthetic_path.exists() and CV2_AVAILABLE:
                    self._create_synthetic_video(synthetic_path, video_type)
                self.datasets[video_type] = TestDataset(
                    name=video_type,
                    path=synthetic_path,
                    data_type="video",
                    metadata=self._get_video_metadata(synthetic_path),
                    is_synthetic=True
                )
        
        # Keypoint datasets
        keypoint_types = ["mediapipe_landmarks", "openpose_keypoints", "pose_sequence"]
        for kp_type in keypoint_types:
            real_path = self.base_dir / f"{kp_type}.json"
            synthetic_path = self.synthetic_dir / f"{kp_type}.json"
            
            if real_path.exists():
                self.datasets[kp_type] = TestDataset(
                    name=kp_type,
                    path=real_path,
                    data_type="keypoints",
                    metadata={"source": "real_data"},
                    is_synthetic=False
                )
            else:
                if not synthetic_path.exists():
                    self._create_synthetic_keypoints(synthetic_path, kp_type)
                self.datasets[kp_type] = TestDataset(
                    name=kp_type,
                    path=synthetic_path,
                    data_type="keypoints",
                    metadata={"source": "synthetic"},
                    is_synthetic=True
                )
        
        # Gait feature datasets
        gait_types = ["normal_gait_features", "abnormal_gait_features", "parkinsonian_features", "hemiplegic_features"]
        for gait_type in gait_types:
            real_path = self.base_dir / f"{gait_type}.json"
            synthetic_path = self.synthetic_dir / f"{gait_type}.json"
            
            if real_path.exists():
                self.datasets[gait_type] = TestDataset(
                    name=gait_type,
                    path=real_path,
                    data_type="gait_features",
                    metadata={"source": "real_data"},
                    is_synthetic=False
                )
            else:
                if not synthetic_path.exists():
                    self._create_synthetic_gait_features(synthetic_path, gait_type)
                self.datasets[gait_type] = TestDataset(
                    name=gait_type,
                    path=synthetic_path,
                    data_type="gait_features",
                    metadata={"source": "synthetic"},
                    is_synthetic=True
                )
    
    def get_sample_videos(self) -> Dict[str, Path]:
        """Get available sample videos."""
        videos = {}
        for name, dataset in self.datasets.items():
            if dataset.data_type == "video":
                videos[name] = dataset.path
        return videos
    
    def get_sample_keypoints(self) -> Dict[str, List[Dict[str, Any]]]:
        """Get available sample keypoints."""
        keypoints = {}
        for name, dataset in self.datasets.items():
            if dataset.data_type == "keypoints":
                with open(dataset.path, 'r') as f:
                    keypoints[name] = json.load(f)
        return keypoints
    
    def get_sample_gait_features(self) -> Dict[str, Dict[str, Any]]:
        """Get available sample gait features."""
        features = {}
        for name, dataset in self.datasets.items():
            if dataset.data_type == "gait_features":
                with open(dataset.path, 'r') as f:
                    features[name] = json.load(f)
        return features
    
    def get_gavd_test_subset(self) -> Dict[str, Any]:
        """Get GAVD test subset data."""
        gavd_file = self.base_dir / "gavd_test_subset.json"
        synthetic_gavd = self.synthetic_dir / "gavd_test_subset.json"
        
        # First try to load real GAVD data from CSV files
        real_gavd_data = self._load_real_gavd_data()
        if real_gavd_data:
            return real_gavd_data
        
        # Fall back to cached JSON files
        if gavd_file.exists():
            with open(gavd_file, 'r') as f:
                return json.load(f)
        elif synthetic_gavd.exists():
            with open(synthetic_gavd, 'r') as f:
                return json.load(f)
        else:
            # Create synthetic GAVD subset
            synthetic_data = self._create_synthetic_gavd_subset()
            with open(synthetic_gavd, 'w') as f:
                json.dump(synthetic_data, f, indent=2)
            return synthetic_data
    
    def create_property_test_data(
        self,
        data_type: str,
        count: int = 10,
        variation: str = "mixed"
    ) -> List[Any]:
        """Create data specifically for property-based testing."""
        if data_type == "keypoints":
            return [self._generate_property_keypoints(variation) for _ in range(count)]
        elif data_type == "gait_features":
            return [self._generate_property_gait_features(variation) for _ in range(count)]
        elif data_type == "classification_results":
            return [self._generate_property_classification_result(variation) for _ in range(count)]
        else:
            raise ValueError(f"Unknown data type for property testing: {data_type}")
    
    def _get_video_metadata(self, video_path: Path) -> Dict[str, Any]:
        """Get video metadata."""
        if not video_path.exists() or not CV2_AVAILABLE:
            return {"error": "Video not accessible"}
        
        try:
            cap = cv2.VideoCapture(str(video_path))
            metadata = {
                "frame_count": int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
                "fps": cap.get(cv2.CAP_PROP_FPS),
                "width": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
                "height": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
                "duration": int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) / cap.get(cv2.CAP_PROP_FPS)
            }
            cap.release()
            return metadata
        except Exception as e:
            return {"error": str(e)}
    
    def _create_synthetic_video(self, video_path: Path, video_type: str):
        """Create synthetic video with enhanced realism."""
        if not CV2_AVAILABLE:
            return
        
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(str(video_path), fourcc, 30.0, (640, 480))
        
        if video_type == "normal_walking":
            self._create_normal_walking_video(out)
        elif video_type == "abnormal_gait":
            self._create_abnormal_gait_video(out)
        elif video_type == "multiple_subjects":
            self._create_multiple_subjects_video(out)
        elif video_type == "parkinsonian_gait":
            self._create_parkinsonian_gait_video(out)
        elif video_type == "hemiplegic_gait":
            self._create_hemiplegic_gait_video(out)
        
        out.release()
    
    def _create_synthetic_keypoints(self, keypoints_path: Path, keypoint_type: str):
        """Create synthetic keypoints data."""
        if keypoint_type == "mediapipe_landmarks":
            keypoints = self._generate_mediapipe_landmarks()
        elif keypoint_type == "openpose_keypoints":
            keypoints = self._generate_openpose_keypoints()
        elif keypoint_type == "pose_sequence":
            keypoints = [self._generate_mediapipe_landmarks() for _ in range(30)]
        else:
            keypoints = self._generate_mediapipe_landmarks()
        
        with open(keypoints_path, 'w') as f:
            json.dump(keypoints, f, indent=2)
    
    def _create_synthetic_gait_features(self, features_path: Path, gait_type: str):
        """Create synthetic gait features data."""
        if gait_type == "normal_gait_features":
            features = self._generate_normal_gait_features()
        elif gait_type == "abnormal_gait_features":
            features = self._generate_abnormal_gait_features()
        elif gait_type == "parkinsonian_features":
            features = self._generate_parkinsonian_features()
        elif gait_type == "hemiplegic_features":
            features = self._generate_hemiplegic_features()
        else:
            features = self._generate_normal_gait_features()
        
        with open(features_path, 'w') as f:
            json.dump(features, f, indent=2)
    
    def _generate_property_keypoints(self, variation: str) -> List[Dict[str, Any]]:
        """Generate keypoints for property testing."""
        if variation == "normal":
            return self._generate_mediapipe_landmarks(confidence_range=(0.7, 1.0))
        elif variation == "low_confidence":
            return self._generate_mediapipe_landmarks(confidence_range=(0.1, 0.5))
        elif variation == "mixed":
            return self._generate_mediapipe_landmarks(confidence_range=(0.1, 1.0))
        else:
            return self._generate_mediapipe_landmarks()
    
    def _generate_property_gait_features(self, variation: str) -> Dict[str, Any]:
        """Generate gait features for property testing."""
        if variation == "normal":
            return self._generate_normal_gait_features()
        elif variation == "abnormal":
            return self._generate_abnormal_gait_features()
        elif variation == "edge_case":
            return self._generate_edge_case_gait_features()
        else:
            # Mixed - randomly choose
            return np.random.choice([
                self._generate_normal_gait_features(),
                self._generate_abnormal_gait_features()
            ])
    
    def _generate_property_classification_result(self, variation: str) -> Dict[str, Any]:
        """Generate classification results for property testing."""
        if variation == "normal":
            return {
                "is_normal": True,
                "confidence": 0.8 + np.random.random() * 0.2,
                "explanation": "Normal gait pattern detected",
                "conditions": []
            }
        elif variation == "abnormal":
            return {
                "is_normal": False,
                "confidence": 0.7 + np.random.random() * 0.3,
                "explanation": "Abnormal gait pattern detected",
                "conditions": [
                    {
                        "name": np.random.choice(["parkinson", "hemiplegia", "neuropathy"]),
                        "confidence": 0.6 + np.random.random() * 0.4
                    }
                ]
            }
        else:
            # Mixed
            is_normal = np.random.choice([True, False])
            return self._generate_property_classification_result("normal" if is_normal else "abnormal")
    
    def _create_normal_walking_video(self, video_writer):
        """Create normal walking pattern video with enhanced realism."""
        for frame_num in range(90):  # 3 seconds at 30fps
            frame = np.zeros((480, 640, 3), dtype=np.uint8)
            
            # Add background texture
            frame[:, :] = [20, 25, 30]
            
            # Person walking from left to right
            person_x = int(50 + (frame_num * 6))
            person_y = 240
            
            # Normal symmetric gait with slight natural variation
            base_frequency = 0.3
            leg_offset = int(20 * np.sin(frame_num * base_frequency) + 2 * np.sin(frame_num * base_frequency * 3))
            arm_offset = int(15 * np.sin(frame_num * base_frequency + np.pi))  # Arms opposite to legs
            
            # Draw more detailed person
            # Head
            cv2.circle(frame, (person_x, person_y - 60), 15, (255, 255, 255), -1)
            # Body
            cv2.rectangle(frame, (person_x - 10, person_y - 45), (person_x + 10, person_y + 10), (255, 255, 255), -1)
            # Arms
            cv2.line(frame, (person_x - 10, person_y - 20), (person_x - 25 + arm_offset, person_y + 5), (255, 255, 255), 3)
            cv2.line(frame, (person_x + 10, person_y - 20), (person_x + 25 - arm_offset, person_y + 5), (255, 255, 255), 3)
            # Legs
            cv2.line(frame, (person_x, person_y + 10), (person_x - 15 + leg_offset, person_y + 50), (255, 255, 255), 4)
            cv2.line(frame, (person_x, person_y + 10), (person_x + 15 - leg_offset, person_y + 50), (255, 255, 255), 4)
            
            video_writer.write(frame)
    
    def _create_abnormal_gait_video(self, video_writer):
        """Create abnormal gait pattern video."""
        for frame_num in range(90):
            frame = np.zeros((480, 640, 3), dtype=np.uint8)
            frame[:, :] = [20, 25, 30]
            
            person_x = int(50 + (frame_num * 4))  # Slower movement
            person_y = 240
            
            # Abnormal asymmetric gait
            leg_offset = int(30 * np.sin(frame_num * 0.2) + 10 * np.sin(frame_num * 0.7))
            
            # Draw person with limp
            cv2.circle(frame, (person_x, person_y - 60), 15, (255, 255, 255), -1)
            cv2.rectangle(frame, (person_x - 10, person_y - 45), (person_x + 10, person_y + 10), (255, 255, 255), -1)
            cv2.line(frame, (person_x, person_y + 10), (person_x - 20 + leg_offset, person_y + 45), (255, 255, 255), 3)  # Shorter left leg
            cv2.line(frame, (person_x, person_y + 10), (person_x + 15 - leg_offset//2, person_y + 50), (255, 255, 255), 3)  # Normal right leg
            
            video_writer.write(frame)
    
    def _create_multiple_subjects_video(self, video_writer):
        """Create video with multiple subjects."""
        for frame_num in range(120):  # 4 seconds
            frame = np.zeros((480, 640, 3), dtype=np.uint8)
            frame[:, :] = [20, 25, 30]
            
            # Two people walking
            for person_id in range(2):
                person_x = int(50 + (frame_num * 4) + person_id * 100)
                person_y = 200 + person_id * 80
                
                leg_offset = int(20 * np.sin(frame_num * 0.3 + person_id * np.pi))
                
                cv2.circle(frame, (person_x, person_y - 60), 12, (255, 255, 255), -1)
                cv2.rectangle(frame, (person_x - 8, person_y - 45), (person_x + 8, person_y + 10), (255, 255, 255), -1)
                cv2.line(frame, (person_x, person_y + 10), (person_x - 12 + leg_offset, person_y + 40), (255, 255, 255), 2)
                cv2.line(frame, (person_x, person_y + 10), (person_x + 12 - leg_offset, person_y + 40), (255, 255, 255), 2)
            
            video_writer.write(frame)
    
    def _create_parkinsonian_gait_video(self, video_writer):
        """Create Parkinsonian gait pattern video."""
        for frame_num in range(120):  # 4 seconds - slower movement
            frame = np.zeros((480, 640, 3), dtype=np.uint8)
            frame[:, :] = [20, 25, 30]
            
            # Slower, shuffling movement
            person_x = int(50 + (frame_num * 3))  # Slower progression
            person_y = 240
            
            # Reduced arm swing, shuffling steps
            leg_offset = int(8 * np.sin(frame_num * 0.4))  # Smaller, slower steps
            arm_offset = int(5 * np.sin(frame_num * 0.4))  # Reduced arm swing
            
            # Stooped posture
            cv2.circle(frame, (person_x, person_y - 50), 12, (255, 255, 255), -1)  # Head lower
            cv2.rectangle(frame, (person_x - 8, person_y - 35), (person_x + 8, person_y + 15), (255, 255, 255), -1)  # Stooped body
            cv2.line(frame, (person_x - 8, person_y - 15), (person_x - 15 + arm_offset, person_y + 10), (255, 255, 255), 2)  # Reduced arm swing
            cv2.line(frame, (person_x + 8, person_y - 15), (person_x + 15 - arm_offset, person_y + 10), (255, 255, 255), 2)
            cv2.line(frame, (person_x, person_y + 15), (person_x - 8 + leg_offset, person_y + 45), (255, 255, 255), 3)  # Shuffling steps
            cv2.line(frame, (person_x, person_y + 15), (person_x + 8 - leg_offset, person_y + 45), (255, 255, 255), 3)
            
            video_writer.write(frame)
    
    def _create_hemiplegic_gait_video(self, video_writer):
        """Create hemiplegic gait pattern video."""
        for frame_num in range(100):
            frame = np.zeros((480, 640, 3), dtype=np.uint8)
            frame[:, :] = [20, 25, 30]
            
            person_x = int(50 + (frame_num * 4))
            person_y = 240
            
            # Asymmetric gait - one side affected
            normal_leg_offset = int(20 * np.sin(frame_num * 0.3))
            affected_leg_offset = int(8 * np.sin(frame_num * 0.15))  # Slower, reduced movement
            
            cv2.circle(frame, (person_x, person_y - 60), 15, (255, 255, 255), -1)
            cv2.rectangle(frame, (person_x - 10, person_y - 45), (person_x + 10, person_y + 10), (255, 255, 255), -1)
            # Normal left side
            cv2.line(frame, (person_x, person_y + 10), (person_x - 15 + normal_leg_offset, person_y + 50), (255, 255, 255), 4)
            # Affected right side
            cv2.line(frame, (person_x, person_y + 10), (person_x + 10 - affected_leg_offset, person_y + 45), (255, 255, 255), 3)
            
            video_writer.write(frame)
    
    def _generate_mediapipe_landmarks(self, confidence_range: Tuple[float, float] = (0.7, 1.0)) -> List[Dict[str, Any]]:
        """Generate realistic MediaPipe landmarks."""
        keypoints = []
        
        # Generate landmarks with anatomical relationships
        center_x, center_y = 320, 240
        
        for i in range(33):  # MediaPipe pose landmarks
            # Add anatomical variation based on landmark type
            if i < 11:  # Face/head landmarks
                x_offset = np.random.normal(0, 20)
                y_offset = np.random.normal(-100, 15)
            elif i < 23:  # Upper body landmarks
                x_offset = np.random.normal(0, 40)
                y_offset = np.random.normal(-20, 30)
            else:  # Lower body landmarks
                x_offset = np.random.normal(0, 30)
                y_offset = np.random.normal(60, 40)
            
            keypoint = {
                "x": center_x + x_offset,
                "y": center_y + y_offset,
                "confidence": np.random.uniform(confidence_range[0], confidence_range[1]),
                "visibility": np.random.uniform(0.8, 1.0)
            }
            keypoints.append(keypoint)
        
        return keypoints
    
    def _generate_openpose_keypoints(self) -> List[Dict[str, Any]]:
        """Generate OpenPose-style keypoints."""
        keypoints = []
        center_x, center_y = 320, 240
        
        for i in range(25):  # OpenPose body keypoints
            x_offset = np.random.normal(0, 50)
            y_offset = np.random.normal(0, 60)
            
            keypoint = {
                "x": center_x + x_offset,
                "y": center_y + y_offset,
                "confidence": np.random.uniform(0.6, 1.0)
            }
            keypoints.append(keypoint)
        
        return keypoints
    
    def _generate_normal_gait_features(self) -> Dict[str, Any]:
        """Generate normal gait features with realistic variation."""
        return {
            "temporal_features": {
                "stride_time": np.random.normal(1.2, 0.1),
                "cadence": np.random.normal(110, 10),
                "stance_phase_duration": np.random.normal(0.65, 0.03),
                "swing_phase_duration": np.random.normal(0.35, 0.03),
                "double_support_time": np.random.normal(0.12, 0.02)
            },
            "spatial_features": {
                "stride_length": np.random.normal(1.4, 0.1),
                "step_width": np.random.normal(0.15, 0.02),
                "step_length_left": np.random.normal(0.7, 0.05),
                "step_length_right": np.random.normal(0.7, 0.05)
            },
            "symmetry_features": {
                "left_right_symmetry": np.random.normal(0.95, 0.03),
                "temporal_symmetry": np.random.normal(0.92, 0.04),
                "spatial_symmetry": np.random.normal(0.94, 0.03)
            }
        }
    
    def _generate_abnormal_gait_features(self) -> Dict[str, Any]:
        """Generate abnormal gait features."""
        return {
            "temporal_features": {
                "stride_time": np.random.normal(1.6, 0.2),
                "cadence": np.random.normal(85, 15),
                "stance_phase_duration": np.random.normal(0.75, 0.05),
                "swing_phase_duration": np.random.normal(0.25, 0.05),
                "double_support_time": np.random.normal(0.20, 0.05)
            },
            "spatial_features": {
                "stride_length": np.random.normal(1.0, 0.15),
                "step_width": np.random.normal(0.25, 0.05),
                "step_length_left": np.random.normal(0.45, 0.1),
                "step_length_right": np.random.normal(0.55, 0.1)
            },
            "symmetry_features": {
                "left_right_symmetry": np.random.normal(0.75, 0.1),
                "temporal_symmetry": np.random.normal(0.65, 0.1),
                "spatial_symmetry": np.random.normal(0.70, 0.1)
            }
        }
    
    def _generate_edge_case_gait_features(self) -> Dict[str, Any]:
        """Generate edge case gait features for boundary testing."""
        return {
            "temporal_features": {
                "stride_time": np.random.choice([0.5, 3.0]),  # Boundary values
                "cadence": np.random.choice([50, 200]),
                "stance_phase_duration": np.random.choice([0.4, 0.8]),
                "swing_phase_duration": np.random.choice([0.2, 0.6]),
                "double_support_time": np.random.choice([0.05, 0.3])
            },
            "spatial_features": {
                "stride_length": np.random.choice([0.5, 3.0]),
                "step_width": np.random.choice([0.05, 0.5]),
                "step_length_left": np.random.uniform(0.2, 1.5),
                "step_length_right": np.random.uniform(0.2, 1.5)
            },
            "symmetry_features": {
                "left_right_symmetry": np.random.choice([0.5, 1.0]),
                "temporal_symmetry": np.random.choice([0.5, 1.0]),
                "spatial_symmetry": np.random.choice([0.5, 1.0])
            }
        }
    
    def _generate_parkinsonian_features(self) -> Dict[str, Any]:
        """Generate Parkinsonian gait features."""
        return {
            "temporal_features": {
                "stride_time": np.random.normal(1.8, 0.3),  # Slower
                "cadence": np.random.normal(75, 12),  # Reduced
                "stance_phase_duration": np.random.normal(0.78, 0.05),  # Increased
                "swing_phase_duration": np.random.normal(0.22, 0.05),  # Reduced
                "double_support_time": np.random.normal(0.25, 0.05)  # Increased
            },
            "spatial_features": {
                "stride_length": np.random.normal(0.8, 0.15),  # Shuffling
                "step_width": np.random.normal(0.12, 0.03),  # Narrow base
                "step_length_left": np.random.normal(0.4, 0.08),
                "step_length_right": np.random.normal(0.4, 0.08)
            },
            "symmetry_features": {
                "left_right_symmetry": np.random.normal(0.85, 0.08),
                "temporal_symmetry": np.random.normal(0.80, 0.1),
                "spatial_symmetry": np.random.normal(0.82, 0.09)
            }
        }
    
    def _generate_hemiplegic_features(self) -> Dict[str, Any]:
        """Generate hemiplegic gait features."""
        return {
            "temporal_features": {
                "stride_time": np.random.normal(1.7, 0.25),
                "cadence": np.random.normal(80, 15),
                "stance_phase_duration": np.random.normal(0.72, 0.08),
                "swing_phase_duration": np.random.normal(0.28, 0.08),
                "double_support_time": np.random.normal(0.18, 0.05)
            },
            "spatial_features": {
                "stride_length": np.random.normal(0.9, 0.2),
                "step_width": np.random.normal(0.22, 0.05),  # Wide base for stability
                "step_length_left": np.random.normal(0.5, 0.1),  # Normal side
                "step_length_right": np.random.normal(0.3, 0.1)  # Affected side
            },
            "symmetry_features": {
                "left_right_symmetry": np.random.normal(0.65, 0.12),  # Highly asymmetric
                "temporal_symmetry": np.random.normal(0.60, 0.15),
                "spatial_symmetry": np.random.normal(0.58, 0.12)
            }
        }
    
    def _load_real_gavd_data(self) -> Optional[Dict[str, Any]]:
        """Load real GAVD data from CSV files."""
        try:
            import pandas as pd
            
            # Try to load the main GAVD CSV file
            gavd_csv_path = Path("data/GAVD_Clinical_Annotations_1.csv")
            if not gavd_csv_path.exists():
                # Try the simple version
                gavd_csv_path = Path("data/GAVD_Clinical_Annotations_1_simple.csv")
                if not gavd_csv_path.exists():
                    return None
            
            # Load the CSV data
            df = pd.read_csv(gavd_csv_path, low_memory=False)
            
            # Process the data into test-friendly format
            return self._process_gavd_dataframe(df)
            
        except ImportError:
            # pandas not available
            return None
        except Exception as e:
            print(f"Warning: Failed to load real GAVD data: {e}")
            return None
    
    def _process_gavd_dataframe(self, df) -> Dict[str, Any]:
        """Process GAVD dataframe into test-friendly format."""
        # Get unique videos and their metadata
        unique_videos = df.groupby(['id', 'url']).agg({
            'gait_pat': 'first',
            'dataset': 'first',
            'vid_info': 'first',
            'seq': 'first'
        }).reset_index()
        
        # Categorize gait patterns
        normal_patterns = ['normal', 'style', 'exercise']
        abnormal_patterns = ['parkinsons', 'abnormal', 'cerebral palsy', 'myopathic', 
                           'antalgic', 'inebriated', 'stroke', 'prosthetic']
        
        normal_subjects = []
        abnormal_subjects = []
        
        for _, row in unique_videos.iterrows():
            gait_pattern = row['gait_pat']
            video_id = row['id']
            youtube_url = row['url']
            
            # Parse video info if available
            vid_info = {}
            try:
                if pd.notna(row['vid_info']) and row['vid_info']:
                    vid_info = eval(row['vid_info']) if isinstance(row['vid_info'], str) else row['vid_info']
            except:
                vid_info = {}
            
            subject_data = {
                "subject_id": video_id,
                "gait_pattern": gait_pattern,
                "youtube_url": youtube_url,
                "sequence_id": row['seq'],
                "video_metadata": vid_info,
                "source": "real_gavd_data"
            }
            
            if gait_pattern in normal_patterns:
                normal_subjects.append(subject_data)
            elif gait_pattern in abnormal_patterns:
                abnormal_subjects.append(subject_data)
        
        # Limit to reasonable test subset sizes
        normal_subjects = normal_subjects[:10]  # Max 10 normal subjects
        abnormal_subjects = abnormal_subjects[:15]  # Max 15 abnormal subjects
        
        return {
            "normal_subjects": normal_subjects,
            "abnormal_subjects": abnormal_subjects,
            "metadata": {
                "source": "real_gavd_csv",
                "total_subjects": len(normal_subjects) + len(abnormal_subjects),
                "normal_count": len(normal_subjects),
                "abnormal_count": len(abnormal_subjects),
                "gait_patterns": {
                    "normal": list(set(s["gait_pattern"] for s in normal_subjects)),
                    "abnormal": list(set(s["gait_pattern"] for s in abnormal_subjects))
                },
                "synthetic": False
            }
        }

    def get_gavd_video_urls(self) -> Dict[str, List[str]]:
        """Get YouTube URLs from GAVD data for testing."""
        gavd_data = self.get_gavd_test_subset()
        
        normal_urls = [s.get("youtube_url") for s in gavd_data.get("normal_subjects", []) 
                      if s.get("youtube_url")]
        abnormal_urls = [s.get("youtube_url") for s in gavd_data.get("abnormal_subjects", []) 
                        if s.get("youtube_url")]
        
        return {
            "normal_gait_urls": normal_urls[:3],  # Limit to 3 for testing
            "abnormal_gait_urls": abnormal_urls[:5]  # Limit to 5 for testing
        }
        """Create enhanced synthetic GAVD subset."""
        return {
            "normal_subjects": [
                {
                    "subject_id": f"N{i:03d}",
                    "age": np.random.randint(20, 60),
                    "gender": np.random.choice(["M", "F"]),
                    "height": np.random.randint(150, 190),
                    "weight": np.random.randint(50, 90),
                    "gait_features": self._generate_normal_gait_features()
                }
                for i in range(1, 11)  # 10 normal subjects
            ],
            "abnormal_subjects": [
                {
                    "subject_id": f"A{i:03d}",
                    "age": np.random.randint(40, 80),
                    "gender": np.random.choice(["M", "F"]),
                    "height": np.random.randint(150, 185),
                    "weight": np.random.randint(55, 95),
                    "condition": np.random.choice(["parkinson", "hemiplegia", "neuropathy", "arthritis"]),
                    "gait_features": self._generate_abnormal_gait_features()
                }
                for i in range(1, 6)  # 5 abnormal subjects
            ],
            "metadata": {
                "created_date": "2024-01-01",
                "version": "1.0",
                "description": "Synthetic GAVD subset for testing",
                "total_subjects": 15
            }
        }


# Global data manager instance
real_data_manager = RealDataManager()


class RealDataGenerator:
    """Generate realistic test data for comprehensive testing."""
    
    @staticmethod
    def create_realistic_pose_sequence(
        duration: float = 5.0,
        frame_rate: float = 30.0,
        gait_pattern: str = "normal"
    ) -> List[List[Dict[str, float]]]:
        """Generate realistic pose keypoint sequence."""
        num_frames = int(duration * frame_rate)
        sequence = []
        
        for frame_idx in range(num_frames):
            # Simulate walking motion with realistic joint movements
            time_progress = frame_idx / num_frames
            
            # Base pose (standing position)
            keypoints = []
            
            # Generate 33 MediaPipe-style landmarks
            for landmark_idx in range(33):
                # Simulate realistic coordinate ranges
                base_x = 320 + np.sin(time_progress * 4 * np.pi) * 50  # Walking motion
                base_y = 240
                
                # Add landmark-specific offsets
                if landmark_idx < 11:  # Face landmarks
                    x = base_x + np.random.normal(0, 5)
                    y = base_y - 100 + np.random.normal(0, 3)
                elif landmark_idx < 23:  # Upper body
                    x = base_x + np.random.normal(0, 10)
                    y = base_y - 50 + np.random.normal(0, 5)
                else:  # Lower body - add gait-specific motion
                    if gait_pattern == "abnormal":
                        # Asymmetric gait pattern
                        leg_offset = 30 * np.sin(time_progress * 2 * np.pi + landmark_idx)
                    else:
                        # Normal symmetric gait
                        leg_offset = 20 * np.sin(time_progress * 3 * np.pi)
                    
                    x = base_x + leg_offset + np.random.normal(0, 8)
                    y = base_y + 50 + np.random.normal(0, 10)
                
                # Realistic confidence scores
                confidence = 0.7 + 0.3 * np.random.random()
                
                keypoints.append({
                    "x": float(x),
                    "y": float(y),
                    "confidence": float(confidence)
                })
            
            sequence.append(keypoints)
        
        return sequence
    
    @staticmethod
    def create_realistic_gait_features(gait_type: str = "normal") -> Dict[str, Any]:
        """Generate realistic gait features based on gait type."""
        if gait_type == "normal":
            return {
                "temporal_features": {
                    "stride_time": np.random.normal(1.2, 0.1),
                    "cadence": np.random.normal(110, 10),
                    "stance_phase_duration": np.random.normal(0.65, 0.05),
                    "swing_phase_duration": np.random.normal(0.35, 0.03)
                },
                "spatial_features": {
                    "stride_length": np.random.normal(1.4, 0.15),
                    "step_width": np.random.normal(0.15, 0.03),
                    "step_length_left": np.random.normal(0.7, 0.08),
                    "step_length_right": np.random.normal(0.7, 0.08)
                },
                "symmetry_features": {
                    "left_right_symmetry": np.random.normal(0.95, 0.03),
                    "temporal_symmetry": np.random.normal(0.92, 0.05),
                    "spatial_symmetry": np.random.normal(0.94, 0.04)
                }
            }
        elif gait_type == "abnormal":
            return {
                "temporal_features": {
                    "stride_time": np.random.normal(1.5, 0.2),
                    "cadence": np.random.normal(85, 15),
                    "stance_phase_duration": np.random.normal(0.75, 0.08),
                    "swing_phase_duration": np.random.normal(0.25, 0.05)
                },
                "spatial_features": {
                    "stride_length": np.random.normal(1.1, 0.2),
                    "step_width": np.random.normal(0.25, 0.08),
                    "step_length_left": np.random.normal(0.6, 0.12),
                    "step_length_right": np.random.normal(0.5, 0.15)
                },
                "symmetry_features": {
                    "left_right_symmetry": np.random.normal(0.75, 0.1),
                    "temporal_symmetry": np.random.normal(0.70, 0.12),
                    "spatial_symmetry": np.random.normal(0.72, 0.1)
                }
            }
        else:
            raise ValueError(f"Unknown gait type: {gait_type}")
    
    @staticmethod
    def create_gavd_test_data(num_samples: int = 50) -> List[Dict[str, Any]]:
        """Create realistic GAVD-style test data."""
        conditions = [
            "normal", "parkinson", "hemiplegia", "diplegia", 
            "neuropathy", "myopathy", "huntington"
        ]
        
        test_data = []
        for i in range(num_samples):
            condition = np.random.choice(conditions)
            is_normal = condition == "normal"
            
            # Generate features based on condition
            if is_normal:
                features = RealDataGenerator.create_realistic_gait_features("normal")
            else:
                features = RealDataGenerator.create_realistic_gait_features("abnormal")
            
            sample = {
                "sample_id": f"test_sample_{i:03d}",
                "condition": condition,
                "is_normal": is_normal,
                "features": features,
                "metadata": {
                    "age": np.random.randint(20, 80),
                    "gender": np.random.choice(["M", "F"]),
                    "height": np.random.normal(170, 10),
                    "weight": np.random.normal(70, 15)
                }
            }
            test_data.append(sample)
        
        return test_data


@pytest.fixture(scope="session")
def realistic_pose_sequences():
    """Provide realistic pose sequences for testing."""
    return {
        "normal_walking": RealDataGenerator.create_realistic_pose_sequence(
            duration=3.0, frame_rate=30.0, gait_pattern="normal"
        ),
        "abnormal_gait": RealDataGenerator.create_realistic_pose_sequence(
            duration=3.0, frame_rate=30.0, gait_pattern="abnormal"
        ),
        "short_sequence": RealDataGenerator.create_realistic_pose_sequence(
            duration=1.0, frame_rate=30.0, gait_pattern="normal"
        )
    }


@pytest.fixture(scope="session")
def realistic_gait_features():
    """Provide realistic gait features for testing."""
    return {
        "normal": RealDataGenerator.create_realistic_gait_features("normal"),
        "abnormal": RealDataGenerator.create_realistic_gait_features("abnormal")
    }


@pytest.fixture(scope="session")
def gavd_test_dataset():
    """Provide GAVD-style test dataset."""
    return RealDataGenerator.create_gavd_test_data(num_samples=100)


@pytest.fixture
def real_frame_sequence(tmp_path):
    """Create a real FrameSequence with actual image data."""
    if not AMBIENT_AVAILABLE or not CV2_AVAILABLE:
        pytest.skip("Required packages not available")
    
    frames = []
    sequence_id = "real_test_sequence"
    
    # Create 10 frames with realistic image data
    for i in range(10):
        # Create realistic image with person-like shape
        image = np.zeros((480, 640, 3), dtype=np.uint8)
        
        # Add background
        image[:, :] = [50, 50, 50]
        
        # Add person silhouette (simplified)
        person_x = 320 + int(i * 5)  # Moving person
        cv2.circle(image, (person_x, 200), 20, (255, 255, 255), -1)  # Head
        cv2.rectangle(image, (person_x - 15, 220), (person_x + 15, 300), (255, 255, 255), -1)  # Body
        
        # Save frame to file
        frame_path = tmp_path / f"frame_{i:03d}.jpg"
        cv2.imwrite(str(frame_path), image)
        
        # Create Frame object
        frame = Frame(
            frame_id=f"{sequence_id}_frame_{i:03d}",
            sequence_id=sequence_id,
            frame_number=i,
            timestamp=i * 0.033,
            file_path=frame_path,
            width=640,
            height=480,
            channels=3,
            format='RGB'
        )
        frames.append(frame)
    
    return FrameSequence(
        sequence_id=sequence_id,
        frames=frames,
        metadata={
            "fps": 30.0,
            "duration": 0.33,
            "total_frames": 10,
            "source": "synthetic_test_data"
        }
    )


@pytest.fixture
def real_video_metadata():
    """Provide realistic video metadata for testing."""
    return {
        "normal_gait_video": {
            "duration": 5.2,
            "frame_rate": 30.0,
            "width": 1920,
            "height": 1080,
            "total_frames": 156,
            "format": "mp4",
            "file_size": 2048576,  # 2MB
            "codec": "h264"
        },
        "abnormal_gait_video": {
            "duration": 8.7,
            "frame_rate": 25.0,
            "width": 1280,
            "height": 720,
            "total_frames": 218,
            "format": "avi",
            "file_size": 5242880,  # 5MB
            "codec": "xvid"
        }
    }


@pytest.fixture
def classification_test_cases():
    """Provide test cases for classification testing."""
    return [
        {
            "case_id": "normal_001",
            "features": RealDataGenerator.create_realistic_gait_features("normal"),
            "expected_classification": "normal",
            "expected_confidence_range": (0.7, 1.0)
        },
        {
            "case_id": "abnormal_001",
            "features": RealDataGenerator.create_realistic_gait_features("abnormal"),
            "expected_classification": "abnormal",
            "expected_confidence_range": (0.6, 1.0)
        },
        {
            "case_id": "borderline_001",
            "features": {
                "temporal_features": {
                    "stride_time": 1.35,  # Borderline value
                    "cadence": 95,
                    "stance_phase_duration": 0.68,
                    "swing_phase_duration": 0.32
                },
                "spatial_features": {
                    "stride_length": 1.25,
                    "step_width": 0.18,
                    "step_length_left": 0.65,
                    "step_length_right": 0.60
                },
                "symmetry_features": {
                    "left_right_symmetry": 0.85,  # Reduced symmetry
                    "temporal_symmetry": 0.82,
                    "spatial_symmetry": 0.80
                }
            },
            "expected_classification": "uncertain",
            "expected_confidence_range": (0.4, 0.7)
        }
    ]


@pytest.fixture
def performance_test_data():
    """Provide data for performance testing."""
    return {
        "small_video": {
            "duration": 2.0,
            "frame_rate": 30.0,
            "resolution": (640, 480),
            "expected_processing_time": 10.0  # seconds
        },
        "medium_video": {
            "duration": 10.0,
            "frame_rate": 30.0,
            "resolution": (1280, 720),
            "expected_processing_time": 45.0  # seconds
        },
        "large_video": {
            "duration": 30.0,
            "frame_rate": 30.0,
            "resolution": (1920, 1080),
            "expected_processing_time": 120.0  # seconds
        }
    }


@pytest.fixture
def error_test_cases():
    """Provide test cases for error handling validation."""
    return {
        "invalid_video_formats": [
            "test.txt", "test.pdf", "test.docx", "test.exe"
        ],
        "corrupted_video_data": b"not a video file content",
        "invalid_pose_data": [
            [],  # Empty keypoints
            [{"x": "invalid", "y": 100, "confidence": 0.8}],  # Invalid x coordinate
            [{"x": 100, "y": 100, "confidence": 1.5}],  # Invalid confidence
            [{"x": 100, "y": 100}],  # Missing confidence
        ],
        "invalid_gait_features": [
            {},  # Empty features
            {"temporal_features": {}},  # Missing required features
            {"temporal_features": {"stride_time": -1.0}},  # Invalid values
        ]
    }


# ============================================================================
# Property Testing Fixtures
# ============================================================================

@pytest.fixture(scope="session")
def sample_gait_videos():
    """Provide sample gait videos for testing."""
    return real_data_manager.get_sample_videos()


@pytest.fixture(scope="session")
def gavd_test_subset():
    """Provide GAVD test subset data."""
    return real_data_manager.get_gavd_test_subset()


@pytest.fixture(scope="session")
def gavd_video_urls():
    """Provide GAVD YouTube video URLs for testing."""
    return real_data_manager.get_gavd_video_urls()


@pytest.fixture
def real_pose_keypoints():
    """Provide real pose keypoints data."""
    keypoints_data = real_data_manager.get_sample_keypoints()
    return keypoints_data.get("mediapipe_landmarks", [])


@pytest.fixture
def real_gait_features():
    """Provide real gait features data."""
    features_data = real_data_manager.get_sample_gait_features()
    return features_data.get("normal_gait_features", {})


@pytest.fixture
def property_test_keypoints():
    """Provide keypoints specifically for property testing."""
    return real_data_manager.create_property_test_data("keypoints", count=20, variation="mixed")


@pytest.fixture
def property_test_gait_features():
    """Provide gait features specifically for property testing."""
    return real_data_manager.create_property_test_data("gait_features", count=15, variation="mixed")


@pytest.fixture
def property_test_classification_results():
    """Provide classification results specifically for property testing."""
    return real_data_manager.create_property_test_data("classification_results", count=10, variation="mixed")


@pytest.fixture
def enhanced_real_data_dir():
    """Get the enhanced real data directory."""
    return real_data_manager.base_dir


@pytest.fixture
def property_test_data_manager():
    """Provide the property test data manager."""
    return real_data_manager


def get_real_data_manager() -> RealDataManager:
    """Get the global real data manager instance."""
    return real_data_manager