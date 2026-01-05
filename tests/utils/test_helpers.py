"""
Test utility functions and helpers for comprehensive testing.
"""

import time
import psutil
import numpy as np
from pathlib import Path
from typing import Dict, List, Any, Optional, Union, Callable
from contextlib import contextmanager
import json
import tempfile
import shutil

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


class DataValidator:
    """Validate test data integrity and format compliance."""
    
    @staticmethod
    def validate_keypoints(keypoints: List[Dict[str, Any]]) -> bool:
        """Validate keypoint data format and values."""
        if not isinstance(keypoints, list):
            return False
        
        for kp in keypoints:
            if not isinstance(kp, dict):
                return False
            
            # Check required fields
            required_fields = ["x", "y", "confidence"]
            if not all(field in kp for field in required_fields):
                return False
            
            # Check value types and ranges
            if not isinstance(kp["x"], (int, float)):
                return False
            if not isinstance(kp["y"], (int, float)):
                return False
            if not isinstance(kp["confidence"], (int, float)):
                return False
            
            # Check confidence range
            if not 0.0 <= kp["confidence"] <= 1.0:
                return False
        
        return True
    
    @staticmethod
    def validate_gait_features(features: Dict[str, Any]) -> bool:
        """Validate gait features structure and values."""
        if not isinstance(features, dict):
            return False
        
        # Check required feature categories
        required_categories = ["temporal_features", "spatial_features", "symmetry_features"]
        if not all(category in features for category in required_categories):
            return False
        
        # Validate temporal features
        temporal = features["temporal_features"]
        temporal_required = ["stride_time", "cadence"]
        if not all(field in temporal for field in temporal_required):
            return False
        
        # Check value ranges
        if not 0.5 <= temporal["stride_time"] <= 3.0:
            return False
        if not 50 <= temporal["cadence"] <= 200:
            return False
        
        # Validate spatial features
        spatial = features["spatial_features"]
        spatial_required = ["stride_length", "step_width"]
        if not all(field in spatial for field in spatial_required):
            return False
        
        if not 0.5 <= spatial["stride_length"] <= 3.0:
            return False
        if not 0.05 <= spatial["step_width"] <= 0.5:
            return False
        
        # Validate symmetry features
        symmetry = features["symmetry_features"]
        if "left_right_symmetry" in symmetry:
            if not 0.0 <= symmetry["left_right_symmetry"] <= 1.0:
                return False
        
        return True
    
    @staticmethod
    def validate_frame_sequence(sequence: 'FrameSequence') -> bool:
        """Validate FrameSequence structure and data."""
        if not AMBIENT_AVAILABLE:
            return True  # Skip validation if ambient not available
        
        if not isinstance(sequence, FrameSequence):
            return False
        
        if not sequence.frames:
            return False
        
        # Check frame ordering
        frame_numbers = [f.frame_number for f in sequence.frames if f.frame_number is not None]
        if frame_numbers and frame_numbers != sorted(frame_numbers):
            return False
        
        # Check timestamp ordering
        timestamps = [f.timestamp for f in sequence.frames if f.timestamp is not None]
        if timestamps and timestamps != sorted(timestamps):
            return False
        
        return True


class PerformanceProfiler:
    """Profile performance metrics during test execution."""
    
    def __init__(self):
        self.metrics = {}
        self.process = psutil.Process()
        self.baseline_metrics = {}
    
    @contextmanager
    def profile(self, operation_name: str):
        """Context manager for profiling operations."""
        start_time = time.time()
        start_memory = self.process.memory_info().rss
        start_cpu = self.process.cpu_percent()
        
        # Monitor peak memory usage
        peak_memory = start_memory
        monitoring = True
        
        def monitor_memory():
            nonlocal peak_memory, monitoring
            while monitoring:
                current_memory = self.process.memory_info().rss
                peak_memory = max(peak_memory, current_memory)
                time.sleep(0.1)
        
        import threading
        monitor_thread = threading.Thread(target=monitor_memory, daemon=True)
        monitor_thread.start()
        
        try:
            yield
        finally:
            monitoring = False
            monitor_thread.join(timeout=1.0)
            
            end_time = time.time()
            end_memory = self.process.memory_info().rss
            end_cpu = self.process.cpu_percent()
            
            self.metrics[operation_name] = {
                "execution_time": end_time - start_time,
                "memory_delta": end_memory - start_memory,
                "peak_memory": peak_memory,
                "memory_delta_mb": (end_memory - start_memory) / (1024 * 1024),
                "peak_memory_mb": peak_memory / (1024 * 1024),
                "cpu_usage": end_cpu - start_cpu,
                "timestamp": time.time()
            }
    
    def get_metrics(self, operation_name: str) -> Dict[str, float]:
        """Get performance metrics for an operation."""
        return self.metrics.get(operation_name, {})
    
    def set_baseline(self, operation_name: str, metrics: Dict[str, float]):
        """Set baseline metrics for regression testing."""
        self.baseline_metrics[operation_name] = metrics
    
    def check_regression(
        self, 
        operation_name: str, 
        tolerance_percent: float = 10.0
    ) -> Dict[str, Any]:
        """Check for performance regression against baseline."""
        current = self.get_metrics(operation_name)
        baseline = self.baseline_metrics.get(operation_name)
        
        if not baseline or not current:
            return {"regression_detected": False, "reason": "No baseline or current metrics"}
        
        regressions = []
        
        # Check execution time regression
        time_change = ((current["execution_time"] - baseline["execution_time"]) 
                      / baseline["execution_time"] * 100)
        if time_change > tolerance_percent:
            regressions.append(f"Execution time increased by {time_change:.1f}%")
        
        # Check memory regression
        memory_change = ((current["peak_memory_mb"] - baseline["peak_memory_mb"]) 
                        / baseline["peak_memory_mb"] * 100)
        if memory_change > tolerance_percent:
            regressions.append(f"Peak memory usage increased by {memory_change:.1f}%")
        
        return {
            "regression_detected": len(regressions) > 0,
            "regressions": regressions,
            "time_change_percent": time_change,
            "memory_change_percent": memory_change,
            "current_metrics": current,
            "baseline_metrics": baseline
        }
    
    def assert_performance_bounds(
        self,
        operation_name: str,
        max_time: Optional[float] = None,
        max_memory_mb: Optional[float] = None,
        max_cpu_percent: Optional[float] = None
    ):
        """Assert that performance metrics are within bounds."""
        metrics = self.get_metrics(operation_name)
        
        if not metrics:
            raise AssertionError(f"No metrics found for operation '{operation_name}'")
        
        if max_time and metrics.get("execution_time", 0) > max_time:
            raise AssertionError(
                f"Operation '{operation_name}' took {metrics['execution_time']:.2f}s, "
                f"expected < {max_time}s"
            )
        
        if max_memory_mb and metrics.get("peak_memory_mb", 0) > max_memory_mb:
            raise AssertionError(
                f"Operation '{operation_name}' used {metrics['peak_memory_mb']:.2f}MB peak, "
                f"expected < {max_memory_mb}MB"
            )
        
        if max_cpu_percent and metrics.get("cpu_usage", 0) > max_cpu_percent:
            raise AssertionError(
                f"Operation '{operation_name}' used {metrics['cpu_usage']:.1f}% CPU, "
                f"expected < {max_cpu_percent}%"
            )
    
    def generate_performance_report(self) -> Dict[str, Any]:
        """Generate comprehensive performance report."""
        if not self.metrics:
            return {"error": "No performance metrics collected"}
        
        report = {
            "summary": {
                "total_operations": len(self.metrics),
                "total_time": sum(m["execution_time"] for m in self.metrics.values()),
                "peak_memory_mb": max(m["peak_memory_mb"] for m in self.metrics.values()),
                "average_cpu": sum(m["cpu_usage"] for m in self.metrics.values()) / len(self.metrics)
            },
            "operations": {}
        }
        
        for op_name, metrics in self.metrics.items():
            report["operations"][op_name] = {
                "execution_time_ms": metrics["execution_time"] * 1000,
                "peak_memory_mb": metrics["peak_memory_mb"],
                "memory_delta_mb": metrics["memory_delta_mb"],
                "cpu_usage_percent": metrics["cpu_usage"]
            }
        
        return report


class FileManager:
    """Manage test files and temporary directories."""
    
    def __init__(self, base_dir: Optional[Path] = None):
        self.base_dir = base_dir or Path(tempfile.gettempdir()) / "alexpose_tests"
        self.base_dir.mkdir(exist_ok=True)
        self.created_files = []
        self.created_dirs = []
    
    def create_test_video(
        self,
        filename: str,
        duration: float = 2.0,
        fps: float = 30.0,
        resolution: tuple = (640, 480)
    ) -> Path:
        """Create a test video file with realistic content."""
        if not CV2_AVAILABLE:
            raise RuntimeError("OpenCV not available for video creation")
        
        video_path = self.base_dir / filename
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(str(video_path), fourcc, fps, resolution)
        
        num_frames = int(duration * fps)
        for frame_idx in range(num_frames):
            # Create frame with moving elements
            frame = np.zeros((resolution[1], resolution[0], 3), dtype=np.uint8)
            
            # Add background
            frame[:, :] = [30, 30, 30]
            
            # Add moving person simulation
            person_x = int(50 + (frame_idx * 5) % (resolution[0] - 100))
            person_y = resolution[1] // 2
            
            # Draw simple person
            cv2.circle(frame, (person_x, person_y - 40), 15, (255, 255, 255), -1)  # Head
            cv2.rectangle(frame, (person_x - 10, person_y - 25), (person_x + 10, person_y + 20), (255, 255, 255), -1)  # Body
            
            # Add leg motion
            leg_offset = int(20 * np.sin(frame_idx * 0.3))
            cv2.line(frame, (person_x, person_y + 20), (person_x - 10 + leg_offset, person_y + 50), (255, 255, 255), 3)
            cv2.line(frame, (person_x, person_y + 20), (person_x + 10 - leg_offset, person_y + 50), (255, 255, 255), 3)
            
            out.write(frame)
        
        out.release()
        self.created_files.append(video_path)
        return video_path
    
    def create_test_image(
        self,
        filename: str,
        resolution: tuple = (640, 480),
        content_type: str = "person"
    ) -> Path:
        """Create a test image file."""
        if not CV2_AVAILABLE:
            raise RuntimeError("OpenCV not available for image creation")
        
        image_path = self.base_dir / filename
        image = np.zeros((resolution[1], resolution[0], 3), dtype=np.uint8)
        
        if content_type == "person":
            # Draw simple person silhouette
            center_x, center_y = resolution[0] // 2, resolution[1] // 2
            cv2.circle(image, (center_x, center_y - 60), 20, (255, 255, 255), -1)  # Head
            cv2.rectangle(image, (center_x - 15, center_y - 40), (center_x + 15, center_y + 30), (255, 255, 255), -1)  # Body
            cv2.line(image, (center_x, center_y + 30), (center_x - 20, center_y + 80), (255, 255, 255), 5)  # Left leg
            cv2.line(image, (center_x, center_y + 30), (center_x + 20, center_y + 80), (255, 255, 255), 5)  # Right leg
        elif content_type == "noise":
            # Random noise
            image = np.random.randint(0, 255, (resolution[1], resolution[0], 3), dtype=np.uint8)
        
        cv2.imwrite(str(image_path), image)
        self.created_files.append(image_path)
        return image_path
    
    def create_test_json(self, filename: str, data: Dict[str, Any]) -> Path:
        """Create a test JSON file."""
        json_path = self.base_dir / filename
        with open(json_path, 'w') as f:
            json.dump(data, f, indent=2)
        self.created_files.append(json_path)
        return json_path
    
    def cleanup(self):
        """Clean up all created test files and directories."""
        for file_path in self.created_files:
            if file_path.exists():
                file_path.unlink()
        
        for dir_path in self.created_dirs:
            if dir_path.exists():
                shutil.rmtree(dir_path)
        
        self.created_files.clear()
        self.created_dirs.clear()


class MockDataGenerator:
    """Generate mock data for testing when real data is not available."""
    
    @staticmethod
    def generate_mock_pose_estimation_result(num_keypoints: int = 33) -> List[Dict[str, float]]:
        """Generate mock pose estimation result."""
        keypoints = []
        for i in range(num_keypoints):
            keypoints.append({
                "x": float(100 + i * 10 + np.random.normal(0, 5)),
                "y": float(200 + i * 5 + np.random.normal(0, 3)),
                "confidence": float(0.7 + 0.3 * np.random.random())
            })
        return keypoints
    
    @staticmethod
    def generate_mock_classification_result(is_normal: bool = True) -> Dict[str, Any]:
        """Generate mock classification result."""
        if is_normal:
            return {
                "is_normal": True,
                "confidence": 0.85 + 0.15 * np.random.random(),
                "explanation": "Normal gait pattern detected with symmetric stride and appropriate cadence.",
                "conditions": [],
                "feature_importance": {
                    "stride_symmetry": 0.3,
                    "cadence": 0.25,
                    "step_length": 0.2,
                    "balance": 0.25
                }
            }
        else:
            conditions = ["parkinson", "hemiplegia", "neuropathy"]
            condition = np.random.choice(conditions)
            return {
                "is_normal": False,
                "confidence": 0.7 + 0.3 * np.random.random(),
                "explanation": f"Abnormal gait pattern consistent with {condition}.",
                "conditions": [
                    {
                        "name": condition,
                        "confidence": 0.75 + 0.25 * np.random.random(),
                        "severity": np.random.choice(["mild", "moderate", "severe"])
                    }
                ],
                "feature_importance": {
                    "stride_asymmetry": 0.4,
                    "cadence_irregularity": 0.3,
                    "balance_issues": 0.3
                }
            }


def assert_keypoints_valid(keypoints: List[Dict[str, Any]], expected_count: Optional[int] = None):
    """Assert that keypoints are valid and properly formatted."""
    assert isinstance(keypoints, list), "Keypoints must be a list"
    
    if expected_count is not None:
        assert len(keypoints) == expected_count, f"Expected {expected_count} keypoints, got {len(keypoints)}"
    
    for i, kp in enumerate(keypoints):
        assert isinstance(kp, dict), f"Keypoint {i} must be a dictionary"
        assert "x" in kp, f"Keypoint {i} missing 'x' coordinate"
        assert "y" in kp, f"Keypoint {i} missing 'y' coordinate"
        assert "confidence" in kp, f"Keypoint {i} missing 'confidence' score"
        
        assert isinstance(kp["x"], (int, float)), f"Keypoint {i} 'x' must be numeric"
        assert isinstance(kp["y"], (int, float)), f"Keypoint {i} 'y' must be numeric"
        assert isinstance(kp["confidence"], (int, float)), f"Keypoint {i} 'confidence' must be numeric"
        
        assert 0.0 <= kp["confidence"] <= 1.0, f"Keypoint {i} confidence {kp['confidence']} not in [0,1]"


def assert_gait_features_valid(features: Dict[str, Any]):
    """Assert that gait features are valid and properly formatted."""
    assert isinstance(features, dict), "Features must be a dictionary"
    
    # Check required categories
    required_categories = ["temporal_features", "spatial_features"]
    for category in required_categories:
        assert category in features, f"Missing required category: {category}"
        assert isinstance(features[category], dict), f"{category} must be a dictionary"
    
    # Validate temporal features
    temporal = features["temporal_features"]
    if "stride_time" in temporal:
        assert 0.5 <= temporal["stride_time"] <= 3.0, f"Invalid stride_time: {temporal['stride_time']}"
    if "cadence" in temporal:
        assert 50 <= temporal["cadence"] <= 200, f"Invalid cadence: {temporal['cadence']}"
    
    # Validate spatial features
    spatial = features["spatial_features"]
    if "stride_length" in spatial:
        assert 0.5 <= spatial["stride_length"] <= 3.0, f"Invalid stride_length: {spatial['stride_length']}"
    if "step_width" in spatial:
        assert 0.05 <= spatial["step_width"] <= 0.5, f"Invalid step_width: {spatial['step_width']}"


def assert_classification_result_valid(result: Dict[str, Any]):
    """Assert that classification result is valid and properly formatted."""
    assert isinstance(result, dict), "Classification result must be a dictionary"
    
    # Check required fields
    required_fields = ["is_normal", "confidence"]
    for field in required_fields:
        assert field in result, f"Missing required field: {field}"
    
    assert isinstance(result["is_normal"], bool), "is_normal must be boolean"
    assert isinstance(result["confidence"], (int, float)), "confidence must be numeric"
    assert 0.0 <= result["confidence"] <= 1.0, f"Invalid confidence: {result['confidence']}"
    
    # If abnormal, should have condition information
    if not result["is_normal"]:
        if "conditions" in result:
            assert isinstance(result["conditions"], list), "conditions must be a list"


def create_temporary_test_environment() -> Path:
    """Create a temporary test environment with necessary directories."""
    temp_dir = Path(tempfile.mkdtemp(prefix="alexpose_test_"))
    
    # Create standard directory structure
    (temp_dir / "data").mkdir()
    (temp_dir / "logs").mkdir()
    (temp_dir / "config").mkdir()
    (temp_dir / "models").mkdir()
    
    return temp_dir


def cleanup_test_environment(test_dir: Path):
    """Clean up temporary test environment."""
    if test_dir.exists() and test_dir.is_dir():
        shutil.rmtree(test_dir)


@contextmanager
def temporary_test_environment():
    """Context manager for temporary test environment."""
    test_dir = create_temporary_test_environment()
    try:
        yield test_dir
    finally:
        cleanup_test_environment(test_dir)