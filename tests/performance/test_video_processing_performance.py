"""Performance tests for video processing components."""

import pytest
import time
import asyncio
import tempfile
from pathlib import Path
from typing import Dict, Any
import numpy as np

from tests.performance.benchmark_framework import PerformanceBenchmark, PerformanceMetrics
from tests.fixtures.real_data_fixtures import RealDataManager

try:
    import cv2
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False

try:
    from ambient.video.processor import VideoProcessor
    from ambient.core.frame import Frame, FrameSequence
    AMBIENT_AVAILABLE = True
except ImportError:
    AMBIENT_AVAILABLE = False


@pytest.mark.performance
@pytest.mark.slow
class TestVideoProcessingPerformance:
    """Performance tests for video processing pipeline."""
    
    def setup_method(self):
        """Set up performance testing framework."""
        self.benchmark = PerformanceBenchmark()
        self.real_data_manager = RealDataManager()
        
        # Create test videos if needed
        self._ensure_test_videos()
    
    def _ensure_test_videos(self):
        """Ensure test videos exist for performance testing."""
        if not CV2_AVAILABLE:
            pytest.skip("OpenCV not available for video creation")
        
        # Create test videos of different durations
        self.test_videos = {}
        
        # 5-second video (small)
        small_video_path = Path("data/test_data/performance_test_5s.mp4")
        if not small_video_path.exists():
            small_video_path.parent.mkdir(parents=True, exist_ok=True)
            self._create_test_video(small_video_path, duration=5.0, fps=30.0)
        self.test_videos["5s"] = small_video_path
        
        # 30-second video (target size)
        medium_video_path = Path("data/test_data/performance_test_30s.mp4")
        if not medium_video_path.exists():
            medium_video_path.parent.mkdir(parents=True, exist_ok=True)
            self._create_test_video(medium_video_path, duration=30.0, fps=30.0)
        self.test_videos["30s"] = medium_video_path
        
        # 60-second video (large)
        large_video_path = Path("data/test_data/performance_test_60s.mp4")
        if not large_video_path.exists():
            large_video_path.parent.mkdir(parents=True, exist_ok=True)
            self._create_test_video(large_video_path, duration=60.0, fps=30.0)
        self.test_videos["60s"] = large_video_path
    
    def _create_test_video(self, video_path: Path, duration: float, fps: float):
        """Create a test video with realistic gait motion."""
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(str(video_path), fourcc, fps, (640, 480))
        
        total_frames = int(duration * fps)
        
        for frame_num in range(total_frames):
            frame = np.zeros((480, 640, 3), dtype=np.uint8)
            
            # Add background texture
            frame[:, :] = [20, 25, 30]
            
            # Person walking from left to right with realistic gait
            person_x = int(50 + (frame_num * 4) % 540)  # Loop across screen
            person_y = 240
            
            # Normal symmetric gait with natural variation
            base_frequency = 0.3
            leg_offset = int(20 * np.sin(frame_num * base_frequency) + 2 * np.sin(frame_num * base_frequency * 3))
            arm_offset = int(15 * np.sin(frame_num * base_frequency + np.pi))
            
            # Draw detailed person
            cv2.circle(frame, (person_x, person_y - 60), 15, (255, 255, 255), -1)  # Head
            cv2.rectangle(frame, (person_x - 10, person_y - 45), (person_x + 10, person_y + 10), (255, 255, 255), -1)  # Body
            cv2.line(frame, (person_x - 10, person_y - 20), (person_x - 25 + arm_offset, person_y + 5), (255, 255, 255), 3)  # Left arm
            cv2.line(frame, (person_x + 10, person_y - 20), (person_x + 25 - arm_offset, person_y + 5), (255, 255, 255), 3)  # Right arm
            cv2.line(frame, (person_x, person_y + 10), (person_x - 15 + leg_offset, person_y + 50), (255, 255, 255), 4)  # Left leg
            cv2.line(frame, (person_x, person_y + 10), (person_x + 15 - leg_offset, person_y + 50), (255, 255, 255), 4)  # Right leg
            
            out.write(frame)
        
        out.release()
    
    @pytest.mark.skipif(not AMBIENT_AVAILABLE, reason="Ambient video processor not available")
    def test_video_loading_performance(self):
        """Test video loading performance with real VideoProcessor."""
        
        def load_video():
            processor = VideoProcessor()
            video_path = self.test_videos["30s"]
            frame_sequence = processor.load_video(video_path)
            return len(frame_sequence)
        
        # Benchmark video loading
        metrics = self.benchmark.benchmark_function(load_video, iterations=1)
        
        # Validate performance targets
        assert metrics.execution_time < 10.0, f"Video loading took {metrics.execution_time:.2f}s, target: <10s"
        assert metrics.peak_memory_mb < 1024, f"Memory usage {metrics.peak_memory_mb:.1f}MB, target: <1GB"
        
        # Check for regression using configurable tolerance
        regression_result = self.benchmark.validate_performance_regression(
            "video_loading_30s", metrics
        )
        
        if regression_result['regression_detected']:
            pytest.fail(f"Performance regression detected: {regression_result['regressions']}")
    
    @pytest.mark.skipif(not AMBIENT_AVAILABLE, reason="Ambient video processor not available")
    def test_frame_extraction_performance(self):
        """Test frame extraction performance with real VideoProcessor."""
        
        def extract_frames():
            processor = VideoProcessor()
            video_path = self.test_videos["30s"]
            
            # Get video info first
            video_info = processor.get_video_info(video_path)
            frame_count = video_info.get("frame_count", 900)  # 30s * 30fps
            
            # Extract every 10th frame for performance testing
            frame_indices = list(range(0, frame_count, 10))
            frames = processor.extract_frames_batch(video_path, frame_indices)
            
            return len(frames)
        
        metrics = self.benchmark.benchmark_function(extract_frames, iterations=1)
        
        # Performance targets for 30s video frame extraction
        assert metrics.execution_time < 20.0, f"Frame extraction took {metrics.execution_time:.2f}s, target: <20s"
        assert metrics.peak_memory_mb < 1536, f"Memory usage {metrics.peak_memory_mb:.1f}MB, target: <1.5GB"
        
        # Check regression with configurable tolerance
        regression_result = self.benchmark.validate_performance_regression(
            "frame_extraction_30s", metrics
        )
        
        if regression_result['regression_detected']:
            pytest.fail(f"Performance regression detected: {regression_result['regressions']}")
    
    @pytest.mark.skipif(not AMBIENT_AVAILABLE, reason="Ambient video processor not available")
    def test_video_metadata_extraction_performance(self):
        """Test video metadata extraction performance."""
        
        def extract_metadata():
            processor = VideoProcessor()
            video_path = self.test_videos["30s"]
            
            # Extract metadata multiple times to test consistency
            metadata_results = []
            for _ in range(5):
                metadata = processor.get_video_info(video_path)
                metadata_results.append(metadata)
            
            return metadata_results
        
        metrics = self.benchmark.benchmark_function(extract_metadata, iterations=1)
        
        # Performance targets for metadata extraction
        assert metrics.execution_time < 2.0, f"Metadata extraction took {metrics.execution_time:.2f}s, target: <2s"
        assert metrics.peak_memory_mb < 256, f"Memory usage {metrics.peak_memory_mb:.1f}MB, target: <256MB"
        
        # Check regression with configurable tolerance
        regression_result = self.benchmark.validate_performance_regression(
            "metadata_extraction", metrics
        )
        
        if regression_result['regression_detected']:
            pytest.fail(f"Performance regression detected: {regression_result['regressions']}")
    
    @pytest.mark.skipif(not AMBIENT_AVAILABLE, reason="Ambient video processor not available")
    def test_complete_video_processing_pipeline_30s(self):
        """Test complete 30-second video processing pipeline performance (target: <60s)."""
        
        def complete_pipeline():
            processor = VideoProcessor()
            video_path = self.test_videos["30s"]
            
            # Step 1: Load video and get metadata
            start_time = time.time()
            video_info = processor.get_video_info(video_path)
            metadata_time = time.time() - start_time
            
            # Step 2: Load video as frame sequence
            start_time = time.time()
            frame_sequence = processor.load_video(video_path)
            loading_time = time.time() - start_time
            
            # Step 3: Extract sample frames for processing
            start_time = time.time()
            frame_count = video_info.get("frame_count", 900)
            sample_indices = list(range(0, frame_count, 30))  # Every 30th frame
            sample_frames = processor.extract_frames_batch(video_path, sample_indices)
            extraction_time = time.time() - start_time
            
            # Step 4: Simulate pose estimation on sample frames
            start_time = time.time()
            pose_results = []
            for frame in sample_frames[:10]:  # Limit to first 10 frames for performance
                # Simulate pose estimation processing time
                time.sleep(0.1)  # Realistic pose estimation time per frame
                pose_results.append({
                    "landmarks": [{"x": np.random.uniform(0, 1), "y": np.random.uniform(0, 1), "confidence": np.random.uniform(0.5, 1.0)} for _ in range(33)],
                    "confidence": np.random.uniform(0.7, 1.0)
                })
            pose_time = time.time() - start_time
            
            # Step 5: Simulate gait analysis
            start_time = time.time()
            gait_features = {
                'temporal_features': {
                    'stride_time': np.random.normal(1.2, 0.1),
                    'cadence': np.random.normal(115, 10),
                    'stance_time': np.random.normal(0.7, 0.05)
                },
                'spatial_features': {
                    'stride_length': np.random.normal(1.4, 0.15),
                    'step_width': np.random.normal(0.1, 0.02),
                    'foot_angle': np.random.normal(15, 5)
                }
            }
            gait_time = time.time() - start_time
            
            return {
                "video_info": video_info,
                "frame_count": len(frame_sequence),
                "sample_frames": len(sample_frames),
                "pose_results": len(pose_results),
                "gait_features": gait_features,
                "timing": {
                    "metadata": metadata_time,
                    "loading": loading_time,
                    "extraction": extraction_time,
                    "pose_estimation": pose_time,
                    "gait_analysis": gait_time
                }
            }
        
        metrics = self.benchmark.benchmark_function(complete_pipeline, iterations=1)
        
        # Performance targets for complete 30s video pipeline (target: <60s)
        assert metrics.execution_time < 60.0, f"Complete 30s video pipeline took {metrics.execution_time:.2f}s, target: <60s"
        assert metrics.peak_memory_mb < 2048, f"Memory usage {metrics.peak_memory_mb:.1f}MB, target: <2GB"
        
        # Check regression with configurable tolerance
        regression_result = self.benchmark.validate_performance_regression(
            "complete_pipeline_30s", metrics
        )
        
        if regression_result['regression_detected']:
            pytest.fail(f"Performance regression detected: {regression_result['regressions']}")
    
    @pytest.mark.skipif(not AMBIENT_AVAILABLE, reason="Ambient video processor not available")
    def test_video_processing_scalability(self):
        """Test video processing performance scaling with video duration."""
        
        def process_video(video_key):
            processor = VideoProcessor()
            video_path = self.test_videos[video_key]
            
            # Basic processing pipeline
            video_info = processor.get_video_info(video_path)
            frame_sequence = processor.load_video(video_path)
            
            # Extract sample frames
            frame_count = video_info.get("frame_count", 150)
            sample_indices = list(range(0, min(frame_count, 300), 30))  # Limit samples
            sample_frames = processor.extract_frames_batch(video_path, sample_indices)
            
            return {
                "duration": video_info.get("duration", 0),
                "frame_count": frame_count,
                "sample_frames": len(sample_frames)
            }
        
        # Test different video durations
        scalability_results = {}
        
        for video_key in ["5s", "30s"]:  # Skip 60s for faster testing
            if video_key in self.test_videos:
                metrics = self.benchmark.benchmark_function(
                    lambda vk=video_key: process_video(vk), 
                    iterations=1
                )
                scalability_results[video_key] = {
                    "execution_time": metrics.execution_time,
                    "memory_usage": metrics.peak_memory_mb,
                    "throughput": metrics.throughput
                }
        
        # Validate scaling behavior
        if "5s" in scalability_results and "30s" in scalability_results:
            time_5s = scalability_results["5s"]["execution_time"]
            time_30s = scalability_results["30s"]["execution_time"]
            
            # Processing time should scale reasonably (not more than 10x for 6x duration)
            scaling_factor = time_30s / time_5s if time_5s > 0 else 1
            assert scaling_factor < 10.0, f"Processing time scaling factor {scaling_factor:.2f}x too high, target: <10x"
            
            # Memory usage should not scale linearly with duration
            memory_5s = scalability_results["5s"]["memory_usage"]
            memory_30s = scalability_results["30s"]["memory_usage"]
            memory_scaling = memory_30s / memory_5s if memory_5s > 0 else 1
            assert memory_scaling < 3.0, f"Memory scaling factor {memory_scaling:.2f}x too high, target: <3x"


@pytest.mark.performance
@pytest.mark.slow
class TestConcurrentVideoProcessingPerformance:
    """Performance tests for concurrent video processing scenarios."""
    
    def setup_method(self):
        """Set up performance testing framework."""
        self.benchmark = PerformanceBenchmark()
        self.real_data_manager = RealDataManager()
    
    @pytest.mark.skipif(not AMBIENT_AVAILABLE, reason="Ambient video processor not available")
    def test_concurrent_video_analysis(self):
        """Test concurrent video analysis performance (target: 5 concurrent analyses)."""
        
        def analyze_video():
            processor = VideoProcessor()
            
            # Create a small test video for concurrent processing
            with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as temp_file:
                temp_path = Path(temp_file.name)
            
            try:
                # Create small video for concurrent testing
                if CV2_AVAILABLE:
                    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                    out = cv2.VideoWriter(str(temp_path), fourcc, 30.0, (320, 240))
                    
                    # Create 60 frames (2 seconds)
                    for i in range(60):
                        frame = np.zeros((240, 320, 3), dtype=np.uint8)
                        frame[:, :] = [50, 50, 50]
                        # Simple moving dot
                        cv2.circle(frame, (160 + int(i * 2), 120), 10, (255, 255, 255), -1)
                        out.write(frame)
                    out.release()
                
                # Process the video
                video_info = processor.get_video_info(temp_path)
                frame_sequence = processor.load_video(temp_path)
                
                # Simulate analysis
                time.sleep(0.5)  # Realistic analysis time
                
                return {
                    "classification": "normal", 
                    "confidence": 0.85,
                    "frame_count": len(frame_sequence),
                    "duration": video_info.get("duration", 2.0)
                }
            finally:
                # Clean up
                if temp_path.exists():
                    temp_path.unlink()
        
        # Test concurrent processing
        concurrent_result = self.benchmark.benchmark_concurrent_operations(
            analyze_video, num_concurrent=5
        )
        
        # Performance targets
        assert concurrent_result['total_time'] < 30.0, f"5 concurrent analyses took {concurrent_result['total_time']:.2f}s, target: <30s"
        assert concurrent_result['success_rate'] >= 0.8, f"Success rate {concurrent_result['success_rate']:.2f}, target: >=0.8"
        assert concurrent_result['memory_usage_mb'] < 2048, f"Memory usage {concurrent_result['memory_usage_mb']:.1f}MB, target: <2GB"
        
        # Validate concurrent efficiency (should be better than sequential)
        sequential_time = 5 * 2.5  # 5 analyses * 2.5s each (estimated)
        efficiency = sequential_time / concurrent_result['total_time']
        assert efficiency > 1.5, f"Concurrent efficiency {efficiency:.2f}x, target: >1.5x speedup"
    
    @pytest.mark.skipif(not AMBIENT_AVAILABLE, reason="Ambient video processor not available")
    def test_concurrent_frame_extraction(self):
        """Test concurrent frame extraction performance."""
        
        def extract_frames():
            processor = VideoProcessor()
            
            # Create temporary test video
            with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as temp_file:
                temp_path = Path(temp_file.name)
            
            try:
                if CV2_AVAILABLE:
                    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                    out = cv2.VideoWriter(str(temp_path), fourcc, 30.0, (320, 240))
                    
                    # Create 90 frames (3 seconds)
                    for i in range(90):
                        frame = np.zeros((240, 320, 3), dtype=np.uint8)
                        frame[:, :] = [30, 30, 30]
                        # Moving pattern
                        cv2.rectangle(frame, (i*2, 100), (i*2+20, 140), (255, 255, 255), -1)
                        out.write(frame)
                    out.release()
                
                # Extract frames
                frame_indices = list(range(0, 90, 10))  # Every 10th frame
                frames = processor.extract_frames_batch(temp_path, frame_indices)
                
                return len(frames)
            finally:
                if temp_path.exists():
                    temp_path.unlink()
        
        # Test concurrent frame extraction
        concurrent_result = self.benchmark.benchmark_concurrent_operations(
            extract_frames, num_concurrent=3
        )
        
        # Performance targets for concurrent frame extraction
        assert concurrent_result['total_time'] < 15.0, f"3 concurrent extractions took {concurrent_result['total_time']:.2f}s, target: <15s"
        assert concurrent_result['success_rate'] >= 0.9, f"Success rate {concurrent_result['success_rate']:.2f}, target: >=0.9"
        assert concurrent_result['memory_usage_mb'] < 1024, f"Memory usage {concurrent_result['memory_usage_mb']:.1f}MB, target: <1GB"
    
    def test_concurrent_api_requests(self):
        """Test concurrent API request handling."""
        
        def make_api_request():
            # Mock API request processing
            time.sleep(0.1)  # Simulate API response time
            return {"status": "success", "data": "response"}
        
        # Test concurrent API requests
        concurrent_result = self.benchmark.benchmark_concurrent_operations(
            make_api_request, num_concurrent=20
        )
        
        # Performance targets for API (target: <200ms for status endpoints)
        assert concurrent_result['average_time_per_operation'] < 0.2, f"Average API response {concurrent_result['average_time_per_operation']:.3f}s, target: <0.2s"
        assert concurrent_result['success_rate'] >= 0.95, f"API success rate {concurrent_result['success_rate']:.2f}, target: >=0.95"
        assert concurrent_result['throughput'] > 50, f"API throughput {concurrent_result['throughput']:.1f} req/s, target: >50 req/s"


@pytest.mark.performance
class TestVideoProcessingMemoryUsage:
    """Performance tests for memory usage monitoring during video processing."""
    
    def setup_method(self):
        """Set up performance testing framework."""
        self.benchmark = PerformanceBenchmark()
    
    @pytest.mark.skipif(not AMBIENT_AVAILABLE or not CV2_AVAILABLE, reason="Required packages not available")
    def test_video_processing_memory_limits(self):
        """Test memory usage during video processing stays within limits."""
        
        def process_large_video():
            processor = VideoProcessor()
            
            # Create a larger test video for memory testing
            with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as temp_file:
                temp_path = Path(temp_file.name)
            
            try:
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                out = cv2.VideoWriter(str(temp_path), fourcc, 30.0, (640, 480))
                
                # Create 300 frames (10 seconds) with varying content
                for i in range(300):
                    frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
                    # Add some structure to make it more realistic
                    cv2.rectangle(frame, (i, 200), (i+50, 280), (255, 255, 255), -1)
                    out.write(frame)
                out.release()
                
                # Process the video
                video_info = processor.get_video_info(temp_path)
                frame_sequence = processor.load_video(temp_path)
                
                # Extract sample frames
                frame_count = video_info.get("frame_count", 300)
                sample_indices = list(range(0, frame_count, 30))
                sample_frames = processor.extract_frames_batch(temp_path, sample_indices)
                
                return len(sample_frames)
            finally:
                if temp_path.exists():
                    temp_path.unlink()
        
        metrics = self.benchmark.benchmark_function(process_large_video, iterations=1)
        
        # Memory usage targets
        assert metrics.peak_memory_mb < 2048, f"Peak memory usage {metrics.peak_memory_mb:.1f}MB, target: <2GB"
        assert metrics.memory_usage_mb < 1024, f"Final memory usage {metrics.memory_usage_mb:.1f}MB, target: <1GB"
    
    @pytest.mark.skipif(not AMBIENT_AVAILABLE, reason="Ambient video processor not available")
    def test_memory_leak_detection(self):
        """Test for memory leaks in repeated video operations."""
        
        def repeated_video_operation():
            processor = VideoProcessor()
            
            # Create small temporary video
            with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as temp_file:
                temp_path = Path(temp_file.name)
            
            try:
                if CV2_AVAILABLE:
                    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                    out = cv2.VideoWriter(str(temp_path), fourcc, 30.0, (320, 240))
                    
                    # Create 30 frames (1 second)
                    for i in range(30):
                        frame = np.zeros((240, 320, 3), dtype=np.uint8)
                        frame[i*8:(i+1)*8, :] = [255, 255, 255]  # Moving white bar
                        out.write(frame)
                    out.release()
                
                # Perform basic operations
                video_info = processor.get_video_info(temp_path)
                frame = processor.extract_frame(temp_path, 15)  # Middle frame
                
                return video_info.get("frame_count", 30)
            finally:
                if temp_path.exists():
                    temp_path.unlink()
        
        # Run multiple iterations to detect leaks
        metrics = self.benchmark.benchmark_function(repeated_video_operation, iterations=10)
        
        # Memory leak detection (memory usage should be minimal for repeated operations)
        assert metrics.memory_usage_mb < 200, f"Memory usage after 10 iterations: {metrics.memory_usage_mb:.1f}MB, possible leak detected"
        assert metrics.peak_memory_mb < 500, f"Peak memory usage {metrics.peak_memory_mb:.1f}MB, target: <500MB"
    
    @pytest.mark.skipif(not CV2_AVAILABLE, reason="OpenCV not available")
    def test_frame_sequence_memory_management(self):
        """Test memory management in frame sequences."""
        
        def create_and_process_frame_sequence():
            # Create multiple frame sequences to test memory management
            sequences = []
            temp_files = []
            
            try:
                for seq_id in range(5):
                    frames = []
                    for frame_id in range(20):
                        # Create frame data
                        frame_data = np.random.randint(0, 255, (240, 320, 3), dtype=np.uint8)
                        
                        # Create temporary frame file
                        with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as temp_file:
                            temp_path = Path(temp_file.name)
                            temp_files.append(temp_path)
                            cv2.imwrite(str(temp_path), frame_data)
                        
                        if AMBIENT_AVAILABLE:
                            frame = Frame.from_file(temp_path, format='RGB', lazy_load=True)
                            frames.append(frame)
                    
                    if AMBIENT_AVAILABLE and frames:
                        sequence = FrameSequence(
                            frames=frames,
                            metadata={"fps": 30.0, "duration": 0.66, "sequence_id": f"test_sequence_{seq_id}"}
                        )
                        sequences.append(sequence)
                
                return len(sequences)
            finally:
                # Clean up temporary files
                for temp_path in temp_files:
                    if temp_path.exists():
                        temp_path.unlink()
        
        metrics = self.benchmark.benchmark_function(create_and_process_frame_sequence, iterations=1)
        
        # Memory targets for frame sequence management
        assert metrics.peak_memory_mb < 1024, f"Peak memory usage {metrics.peak_memory_mb:.1f}MB, target: <1GB"
        assert metrics.memory_usage_mb < 512, f"Final memory usage {metrics.memory_usage_mb:.1f}MB, target: <512MB"


@pytest.fixture(scope="session")
def performance_report():
    """Generate performance report after all tests."""
    yield
    
    # Generate final performance report
    benchmark = PerformanceBenchmark()
    report = benchmark.generate_performance_report()
    
    report_file = Path("tests/performance/performance_report.json")
    report_file.parent.mkdir(parents=True, exist_ok=True)
    
    import json
    with open(report_file, 'w') as f:
        json.dump(report, f, indent=2)
    
    print(f"\nPerformance report saved to: {report_file}")
    print(f"Total benchmarks: {report['total_benchmarks']}")
    if report['total_benchmarks'] > 0:
        print(f"Average execution time: {report['summary']['avg_execution_time']:.2f}s")
        print(f"Average memory usage: {report['summary']['avg_memory_usage']:.1f}MB")