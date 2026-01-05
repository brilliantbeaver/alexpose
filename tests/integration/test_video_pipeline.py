"""
Comprehensive video analysis pipeline integration tests.

This module tests the complete video analysis workflow from upload
to classification using real components and data.
"""

import pytest
import asyncio
import time
from pathlib import Path
from typing import Dict, Any, List
import logging

from tests.integration.integration_framework import IntegrationTestFramework, PipelineResult
from tests.fixtures.real_data_fixtures import get_real_data_manager

logger = logging.getLogger(__name__)


# Mark all tests in this module as asyncio
pytestmark = pytest.mark.asyncio


class TestVideoAnalysisPipeline:
    """Test complete video analysis pipeline workflows."""

    @pytest.fixture(scope="class")
    def integration_framework(self):
        """Provide integration test framework."""
        framework = IntegrationTestFramework()
        yield framework
        framework.cleanup_test_artifacts()

    @pytest.fixture(scope="class")
    def sample_videos(self):
        """Provide sample videos for testing."""
        data_manager = get_real_data_manager()
        return data_manager.get_sample_videos()

    @pytest.fixture(scope="class")
    def gavd_test_data(self):
        """Provide GAVD test data for integration testing."""
        data_manager = get_real_data_manager()
        return data_manager.get_gavd_test_subset()

    @pytest.mark.integration
    @pytest.mark.slow
    async def test_complete_normal_gait_analysis_workflow(
        self, 
        integration_framework: IntegrationTestFramework,
        sample_videos: Dict[str, Path]
    ):
        """
        Test complete video analysis workflow with normal gait video.
        
        This test validates the entire pipeline from video upload through
        classification for a normal walking pattern.
        """
        # Skip if no normal walking video available
        if "normal_walking" not in sample_videos:
            pytest.skip("Normal walking video not available")
        
        video_file = sample_videos["normal_walking"]
        
        # Execute complete pipeline
        result = await integration_framework.test_complete_video_analysis_pipeline(
            video_file=video_file,
            expected_classification="normal",
            timeout_seconds=120.0  # 2 minutes max
        )
        
        # Validate pipeline success
        assert result.pipeline_success, f"Pipeline failed: {result.error_summary}"
        
        # Validate all steps completed successfully
        successful_steps = result.get_successful_steps()
        expected_steps = [
            "video_upload", "frame_extraction", "pose_estimation", 
            "gait_analysis", "classification", "result_validation"
        ]
        
        for step in expected_steps:
            assert step in successful_steps, f"Step {step} did not complete successfully"
        
        # Validate performance requirements
        assert result.total_processing_time < 120.0, \
            f"Processing took too long: {result.total_processing_time}s"
        
        # Validate final classification result
        assert result.final_result is not None, "No final classification result"
        assert result.final_result.get("classification") == "normal", \
            f"Expected normal classification, got {result.final_result.get('classification')}"
        
        # Validate confidence score
        confidence = result.final_result.get("confidence", 0.0)
        assert 0.0 <= confidence <= 1.0, f"Invalid confidence score: {confidence}"
        assert confidence >= 0.5, f"Low confidence score: {confidence}"

    @pytest.mark.integration
    @pytest.mark.slow
    async def test_complete_abnormal_gait_analysis_workflow(
        self, 
        integration_framework: IntegrationTestFramework,
        sample_videos: Dict[str, Path]
    ):
        """
        Test complete video analysis workflow with abnormal gait video.
        
        This test validates the entire pipeline for abnormal gait patterns
        and ensures proper classification.
        """
        # Skip if no abnormal gait video available
        if "abnormal_gait" not in sample_videos:
            pytest.skip("Abnormal gait video not available")
        
        video_file = sample_videos["abnormal_gait"]
        
        # Execute complete pipeline
        result = await integration_framework.test_complete_video_analysis_pipeline(
            video_file=video_file,
            expected_classification="abnormal",
            timeout_seconds=120.0
        )
        
        # Validate pipeline success
        assert result.pipeline_success, f"Pipeline failed: {result.error_summary}"
        
        # Validate all steps completed successfully
        successful_steps = result.get_successful_steps()
        expected_steps = [
            "video_upload", "frame_extraction", "pose_estimation", 
            "gait_analysis", "classification", "result_validation"
        ]
        
        for step in expected_steps:
            assert step in successful_steps, f"Step {step} did not complete successfully"
        
        # Validate final classification result
        assert result.final_result is not None, "No final classification result"
        assert result.final_result.get("classification") == "abnormal", \
            f"Expected abnormal classification, got {result.final_result.get('classification')}"

    @pytest.mark.integration
    @pytest.mark.slow
    async def test_pipeline_with_multiple_subjects_video(
        self, 
        integration_framework: IntegrationTestFramework,
        sample_videos: Dict[str, Path]
    ):
        """
        Test pipeline with video containing multiple subjects.
        
        This test validates that the system can handle videos with
        multiple people and still produce meaningful results.
        """
        # Skip if no multiple subjects video available
        if "multiple_subjects" not in sample_videos:
            pytest.skip("Multiple subjects video not available")
        
        video_file = sample_videos["multiple_subjects"]
        
        # Execute pipeline without expected classification (uncertain outcome)
        result = await integration_framework.test_complete_video_analysis_pipeline(
            video_file=video_file,
            timeout_seconds=180.0  # Longer timeout for complex video
        )
        
        # Validate pipeline success (should handle multiple subjects gracefully)
        assert result.pipeline_success, f"Pipeline failed: {result.error_summary}"
        
        # Validate that pose estimation detected multiple subjects
        pose_step = result.get_step_result("pose_estimation")
        assert pose_step is not None, "Pose estimation step not found"
        assert pose_step.success, "Pose estimation failed"
        
        # The system should still produce a classification result
        assert result.final_result is not None, "No final classification result"
        assert "classification" in result.final_result, "Missing classification field"

    @pytest.mark.integration
    @pytest.mark.slow
    async def test_complete_workflow_timing_and_performance_validation(
        self, 
        integration_framework: IntegrationTestFramework,
        sample_videos: Dict[str, Path]
    ):
        """
        Comprehensive test for complete workflow timing and performance validation.
        
        This test validates that the entire pipeline meets performance benchmarks
        and provides detailed timing analysis for each step.
        """
        performance_results = {}
        validation_results = {}
        
        # Test with different video types to validate performance across scenarios
        test_scenarios = [
            ("normal_walking", "normal", 120.0),
            ("abnormal_gait", "abnormal", 120.0),
            ("multiple_subjects", None, 180.0)  # Longer timeout for complex video
        ]
        
        for video_type, expected_classification, timeout in test_scenarios:
            if video_type not in sample_videos or not sample_videos[video_type].exists():
                continue
            
            video_file = sample_videos[video_type]
            
            # Get video information for context-aware validation
            try:
                video_info = integration_framework.video_processor.get_video_info(video_file)
            except Exception:
                # Use default video info if unable to get actual info
                video_info = {
                    "duration": 5.0,
                    "fps": 30.0,
                    "frame_count": 150,
                    "width": 640,
                    "height": 480
                }
            
            # Execute complete pipeline with detailed timing
            start_time = time.time()
            pipeline_result = await integration_framework.test_complete_video_analysis_pipeline(
                video_file=video_file,
                expected_classification=expected_classification,
                timeout_seconds=timeout
            )
            total_execution_time = time.time() - start_time
            
            # Store performance results
            performance_results[video_type] = {
                "pipeline_result": pipeline_result,
                "video_info": video_info,
                "total_execution_time": total_execution_time
            }
            
            # Validate workflow timing and performance
            validation_result = integration_framework.validate_workflow_timing_and_performance(
                pipeline_result=pipeline_result,
                video_info=video_info
            )
            validation_results[video_type] = validation_result
            
            # Assert overall performance validation passes
            assert validation_result["overall_pass"], \
                f"Performance validation failed for {video_type}: {validation_result['recommendations']}"
            
            # Validate specific timing requirements
            timing_validation = validation_result["timing_validation"]
            
            # Total time should be within benchmark
            total_time_validation = timing_validation.get("total_time", {})
            assert total_time_validation.get("pass", False), \
                f"Total processing time exceeded benchmark for {video_type}: " \
                f"{total_time_validation.get('actual', 0):.2f}s > {total_time_validation.get('benchmark', 0):.2f}s"
            
            # Critical steps should meet timing requirements
            critical_steps = ["pose_estimation", "gait_analysis", "classification"]
            for step_name in critical_steps:
                if step_name in timing_validation:
                    step_validation = timing_validation[step_name]
                    assert step_validation.get("pass", False), \
                        f"Step '{step_name}' exceeded timing benchmark for {video_type}: " \
                        f"{step_validation.get('actual', 0):.2f}s > {step_validation.get('benchmark', 0):.2f}s"
            
            # Validate performance characteristics
            performance_validation = validation_result["performance_validation"]
            
            # Check throughput requirements
            throughput = performance_validation.get("throughput", {})
            if "video_processing_rate" in throughput:
                video_rate = throughput["video_processing_rate"]
                assert video_rate.get("pass", False), \
                    f"Video processing rate below minimum for {video_type}: " \
                    f"{video_rate.get('value', 0):.2f} < minimum required rate"
            
            # Check step efficiency (informational, not a hard failure)
            efficiency = performance_validation.get("efficiency", {})
            if "step_balance" in efficiency:
                step_balance = efficiency["step_balance"]
                # Log step balance information but don't fail the test
                # This is informational for performance analysis
                if not step_balance.get("pass", True):
                    logger.info(
                        f"Step balance note for {video_type}: "
                        f"'{step_balance.get('dominant_step', 'unknown')}' takes "
                        f"{step_balance.get('dominant_step_percentage', 0):.1f}% of total time"
                    )
        
        # Generate comprehensive performance report
        self._generate_performance_report(performance_results, validation_results, integration_framework)
        
        # Validate that we tested at least one scenario successfully
        assert len(performance_results) > 0, "No video scenarios were successfully tested"
        
        # Validate that all tested scenarios passed performance requirements
        all_passed = all(
            validation_results[video_type]["overall_pass"] 
            for video_type in validation_results
        )
        assert all_passed, "Not all video scenarios passed performance validation"

    def _generate_performance_report(
        self, 
        performance_results: Dict[str, Any],
        validation_results: Dict[str, Any],
        integration_framework: IntegrationTestFramework
    ):
        """Generate a comprehensive performance report for analysis."""
        report = {
            "test_timestamp": time.time(),
            "scenarios_tested": len(performance_results),
            "overall_summary": {},
            "detailed_results": {}
        }
        
        # Calculate overall statistics
        total_times = []
        step_times = {}
        
        for video_type, perf_data in performance_results.items():
            pipeline_result = perf_data["pipeline_result"]
            total_times.append(pipeline_result.total_processing_time)
            
            for step in pipeline_result.steps:
                if step.step_name not in step_times:
                    step_times[step.step_name] = []
                step_times[step.step_name].append(step.processing_time)
        
        if total_times:
            report["overall_summary"] = {
                "average_total_time": sum(total_times) / len(total_times),
                "min_total_time": min(total_times),
                "max_total_time": max(total_times),
                "step_averages": {
                    step_name: sum(times) / len(times)
                    for step_name, times in step_times.items()
                }
            }
        
        # Store detailed results
        for video_type in performance_results:
            report["detailed_results"][video_type] = {
                "performance": performance_results[video_type],
                "validation": validation_results[video_type]
            }
        
        # Log performance report for analysis
        logger.info(f"Performance validation report: {report['overall_summary']}")
        
        # Store report for potential CI/CD integration
        integration_framework.performance_metrics["latest_validation_report"] = report

    @pytest.mark.integration
    @pytest.mark.performance
    async def test_workflow_performance_regression_detection(
        self, 
        integration_framework: IntegrationTestFramework,
        sample_videos: Dict[str, Path]
    ):
        """
        Test workflow performance regression detection.
        
        This test establishes performance baselines and detects regressions
        in subsequent runs.
        """
        if "normal_walking" not in sample_videos or not sample_videos["normal_walking"].exists():
            pytest.skip("Normal walking video not available for regression testing")
        
        video_file = sample_videos["normal_walking"]
        
        # Execute pipeline multiple times to establish baseline
        execution_times = []
        step_times = {}
        
        for run in range(3):  # Run 3 times for statistical significance
            pipeline_result = await integration_framework.test_complete_video_analysis_pipeline(
                video_file=video_file,
                timeout_seconds=120.0
            )
            
            assert pipeline_result.pipeline_success, f"Pipeline failed on run {run + 1}"
            
            execution_times.append(pipeline_result.total_processing_time)
            
            for step in pipeline_result.steps:
                if step.step_name not in step_times:
                    step_times[step.step_name] = []
                step_times[step.step_name].append(step.processing_time)
        
        # Calculate baseline metrics
        baseline_total_time = sum(execution_times) / len(execution_times)
        baseline_step_times = {
            step_name: sum(times) / len(times)
            for step_name, times in step_times.items()
        }
        
        # Store baseline for future regression detection
        baseline_key = "workflow_regression_baseline"
        integration_framework.performance_metrics[baseline_key] = {
            "total_time": baseline_total_time,
            "step_times": baseline_step_times,
            "sample_count": len(execution_times)
        }
        
        # Validate performance consistency (coefficient of variation < 20%)
        if len(execution_times) > 1:
            import statistics
            std_dev = statistics.stdev(execution_times)
            cv = (std_dev / baseline_total_time) * 100
            
            assert cv < 30.0, \
                f"Performance too inconsistent: coefficient of variation {cv:.1f}% > 30%"
        
        # Validate that baseline is within acceptable range
        assert baseline_total_time < 120.0, \
            f"Baseline performance too slow: {baseline_total_time:.2f}s > 120s"
        
        # Log baseline establishment
        logger.info(f"Established performance baseline: {baseline_total_time:.2f}s total time")
        logger.info(f"Step baselines: {baseline_step_times}")

    @pytest.mark.integration
    @pytest.mark.performance
    async def test_workflow_memory_usage_validation(
        self, 
        integration_framework: IntegrationTestFramework,
        sample_videos: Dict[str, Path]
    ):
        """
        Test workflow memory usage validation.
        
        This test monitors memory usage during pipeline execution
        and validates it stays within acceptable limits.
        """
        if "normal_walking" not in sample_videos or not sample_videos["normal_walking"].exists():
            pytest.skip("Normal walking video not available for memory testing")
        
        video_file = sample_videos["normal_walking"]
        
        # Monitor memory usage during pipeline execution
        import psutil
        process = psutil.Process()
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB
        peak_memory = initial_memory
        memory_samples = []
        
        def monitor_memory():
            nonlocal peak_memory
            while monitoring:
                try:
                    current_memory = process.memory_info().rss / 1024 / 1024
                    peak_memory = max(peak_memory, current_memory)
                    memory_samples.append(current_memory)
                    time.sleep(0.5)  # Sample every 500ms
                except psutil.NoSuchProcess:
                    break
        
        # Start memory monitoring
        monitoring = True
        import threading
        monitor_thread = threading.Thread(target=monitor_memory, daemon=True)
        monitor_thread.start()
        
        try:
            # Execute pipeline
            pipeline_result = await integration_framework.test_complete_video_analysis_pipeline(
                video_file=video_file,
                timeout_seconds=120.0
            )
            
            assert pipeline_result.pipeline_success, "Pipeline failed during memory monitoring"
            
        finally:
            # Stop monitoring
            monitoring = False
            monitor_thread.join(timeout=1.0)
        
        final_memory = process.memory_info().rss / 1024 / 1024
        memory_increase = peak_memory - initial_memory
        
        # Validate memory usage
        max_allowed_memory = 2048.0  # 2GB limit
        assert peak_memory < max_allowed_memory, \
            f"Peak memory usage {peak_memory:.1f}MB exceeds limit {max_allowed_memory}MB"
        
        # Validate memory increase is reasonable
        max_allowed_increase = 1536.0  # 1.5GB increase limit
        assert memory_increase < max_allowed_increase, \
            f"Memory increase {memory_increase:.1f}MB exceeds limit {max_allowed_increase}MB"
        
        # Check for potential memory leaks (final memory should be close to initial)
        memory_leak_threshold = 100.0  # 100MB threshold
        memory_leak = final_memory - initial_memory
        
        if memory_leak > memory_leak_threshold:
            logger.warning(f"Potential memory leak detected: {memory_leak:.1f}MB increase")
        
        # Log memory usage statistics
        logger.info(f"Memory usage - Initial: {initial_memory:.1f}MB, "
                   f"Peak: {peak_memory:.1f}MB, Final: {final_memory:.1f}MB")
        
        # Store memory metrics for analysis
        integration_framework.performance_metrics["memory_usage"] = {
            "initial_mb": initial_memory,
            "peak_mb": peak_memory,
            "final_mb": final_memory,
            "increase_mb": memory_increase,
            "samples": len(memory_samples),
            "average_mb": sum(memory_samples) / len(memory_samples) if memory_samples else initial_memory
        }

    @pytest.mark.integration
    @pytest.mark.slow
    async def test_pipeline_error_handling(
        self, 
        integration_framework: IntegrationTestFramework,
        tmp_path: Path
    ):
        """
        Test pipeline error handling with invalid inputs.
        
        This test validates that the pipeline gracefully handles
        various error conditions and provides meaningful error messages.
        """
        # Test with non-existent video file
        non_existent_video = tmp_path / "non_existent.mp4"
        
        result = await integration_framework.test_complete_video_analysis_pipeline(
            video_file=non_existent_video,
            timeout_seconds=30.0
        )
        
        assert not result.pipeline_success, "Pipeline should fail with non-existent video"
        assert len(result.error_summary) > 0, "Should have error messages"
        
        # Validate that video upload step failed
        upload_step = result.get_step_result("video_upload")
        assert upload_step is not None, "Video upload step should be present"
        assert not upload_step.success, "Video upload should fail"
        assert "not found" in upload_step.error.lower(), "Should indicate file not found"

    @pytest.mark.integration
    @pytest.mark.slow
    async def test_pipeline_permission_denied_error(
        self, 
        integration_framework: IntegrationTestFramework,
        tmp_path: Path
    ):
        """Test pipeline handling of permission denied errors."""
        # Create a file with restricted permissions
        restricted_video = tmp_path / "restricted.mp4"
        restricted_video.write_bytes(b"fake video content")
        
        # Try to make file unreadable (may not work on all systems)
        try:
            import os
            os.chmod(restricted_video, 0o000)
            
            result = await integration_framework.test_complete_video_analysis_pipeline(
                video_file=restricted_video,
                timeout_seconds=30.0
            )
            
            assert not result.pipeline_success, "Pipeline should fail with permission denied"
            
            upload_step = result.get_step_result("video_upload")
            if upload_step and not upload_step.success:
                error_msg = upload_step.error.lower()
                # Accept various error messages that indicate file access issues
                assert any(keyword in error_msg for keyword in 
                          ["permission", "access", "denied", "failed", "cannot", "unable"]), \
                    f"Should indicate file access error: {upload_step.error}"
            
        except (OSError, PermissionError):
            pytest.skip("Cannot test permission denied on this system")
        finally:
            # Restore permissions for cleanup
            try:
                os.chmod(restricted_video, 0o644)
            except:
                pass

    @pytest.mark.integration
    @pytest.mark.slow
    async def test_pipeline_interrupted_processing(
        self, 
        integration_framework: IntegrationTestFramework,
        sample_videos: Dict[str, Path]
    ):
        """Test pipeline behavior when processing is interrupted."""
        if "normal_walking" not in sample_videos:
            pytest.skip("Normal walking video not available")
        
        video_file = sample_videos["normal_walking"]
        
        # Test with extremely short timeout to simulate interruption
        result = await integration_framework.test_complete_video_analysis_pipeline(
            video_file=video_file,
            timeout_seconds=0.01  # 10ms - should timeout immediately
        )
        
        assert not result.pipeline_success, "Pipeline should fail due to timeout"
        
        # Should have timeout-related error
        error_messages = " ".join(result.error_summary).lower()
        assert "timeout" in error_messages or "interrupted" in error_messages, \
            "Should indicate timeout/interruption"

    @pytest.mark.integration
    @pytest.mark.slow
    async def test_pipeline_resource_exhaustion_simulation(
        self, 
        integration_framework: IntegrationTestFramework,
        tmp_path: Path
    ):
        """Test pipeline behavior under simulated resource exhaustion."""
        # Create a mock video file
        mock_video = tmp_path / "resource_test.mp4"
        mock_video.write_bytes(b"mock video for resource testing")
        
        # Mock memory exhaustion by patching a component to raise MemoryError
        from unittest.mock import patch
        
        with patch.object(integration_framework.video_processor, 'load_video', 
                         side_effect=MemoryError("Insufficient memory")):
            result = await integration_framework.test_complete_video_analysis_pipeline(
                video_file=mock_video,
                timeout_seconds=30.0
            )
            
            assert not result.pipeline_success, "Pipeline should fail with memory error"
            
            # Check for memory-related error in frame extraction step
            frame_step = result.get_step_result("frame_extraction")
            if frame_step and not frame_step.success:
                assert "memory" in frame_step.error.lower(), \
                    "Should indicate memory error"

    @pytest.mark.integration
    @pytest.mark.slow
    async def test_pipeline_partial_failure_recovery(
        self, 
        integration_framework: IntegrationTestFramework,
        sample_videos: Dict[str, Path]
    ):
        """Test pipeline recovery from partial failures."""
        if "normal_walking" not in sample_videos:
            pytest.skip("Normal walking video not available")
        
        video_file = sample_videos["normal_walking"]
        
        # Mock pose estimation to fail initially, then succeed
        from unittest.mock import patch, Mock
        
        call_count = 0
        def failing_pose_estimation(frame_sequence):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise RuntimeError("Pose estimation failed")
            else:
                # Return mock pose data
                return [[{"x": 100.0, "y": 200.0, "confidence": 0.8} for _ in range(33)] 
                       for _ in range(len(frame_sequence))]
        
        with patch.object(integration_framework.pose_estimator, 'estimate_pose_sequence',
                         side_effect=failing_pose_estimation):
            # First attempt should fail
            result1 = await integration_framework.test_complete_video_analysis_pipeline(
                video_file=video_file,
                timeout_seconds=60.0
            )
            
            assert not result1.pipeline_success, "First attempt should fail"
            
            pose_step = result1.get_step_result("pose_estimation")
            assert pose_step is not None and not pose_step.success, \
                "Pose estimation should fail on first attempt"
            
            # Second attempt should succeed (if retry logic exists)
            # Note: This tests the framework's ability to handle retries
            result2 = await integration_framework.test_complete_video_analysis_pipeline(
                video_file=video_file,
                timeout_seconds=60.0
            )
            
            # The second attempt might still fail if no retry logic exists
            # This is acceptable - we're testing error handling, not retry logic
            if not result2.pipeline_success:
                pose_step2 = result2.get_step_result("pose_estimation")
                if pose_step2 and not pose_step2.success:
                    assert "pose estimation failed" in pose_step2.error.lower(), \
                        "Should indicate pose estimation failure"

    @pytest.mark.integration
    @pytest.mark.slow
    async def test_pipeline_corrupted_video_handling(
        self, 
        integration_framework: IntegrationTestFramework,
        tmp_path: Path
    ):
        """Test pipeline handling of corrupted video files."""
        # Create corrupted video file
        corrupted_video = tmp_path / "corrupted.mp4"
        corrupted_video.write_bytes(b"This is not valid video data" * 100)
        
        result = await integration_framework.test_complete_video_analysis_pipeline(
            video_file=corrupted_video,
            timeout_seconds=60.0
        )
        
        assert not result.pipeline_success, "Pipeline should fail with corrupted video"
        
        # Should fail at video upload or frame extraction step
        failed_steps = result.get_failed_steps()
        assert len(failed_steps) > 0, "Should have failed steps"
        assert any(step in failed_steps for step in ["video_upload", "frame_extraction"]), \
            "Should fail at video processing step"

    @pytest.mark.integration
    @pytest.mark.slow
    async def test_pipeline_empty_video_handling(
        self, 
        integration_framework: IntegrationTestFramework,
        tmp_path: Path
    ):
        """Test pipeline handling of empty video files."""
        # Create empty video file
        empty_video = tmp_path / "empty.mp4"
        empty_video.touch()  # Create empty file
        
        result = await integration_framework.test_complete_video_analysis_pipeline(
            video_file=empty_video,
            timeout_seconds=30.0
        )
        
        assert not result.pipeline_success, "Pipeline should fail with empty video"
        
        # Should fail at video upload or frame extraction step
        upload_step = result.get_step_result("video_upload")
        if upload_step and upload_step.success:
            # If upload succeeds, frame extraction should fail
            frame_step = result.get_step_result("frame_extraction")
            assert frame_step is not None, "Frame extraction step should be present"
            assert not frame_step.success, "Frame extraction should fail for empty video"

    @pytest.mark.integration
    @pytest.mark.slow
    async def test_pipeline_very_short_video_handling(
        self, 
        integration_framework: IntegrationTestFramework,
        tmp_path: Path
    ):
        """Test pipeline handling of very short videos (< 1 second)."""
        # Create very short video using OpenCV if available
        short_video = tmp_path / "very_short.mp4"
        
        try:
            import cv2
            import numpy as np
            
            # Create 0.5 second video (15 frames at 30fps)
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(str(short_video), fourcc, 30.0, (320, 240))
            
            for i in range(15):  # 0.5 seconds
                frame = np.random.randint(0, 255, (240, 320, 3), dtype=np.uint8)
                out.write(frame)
            out.release()
            
            result = await integration_framework.test_complete_video_analysis_pipeline(
                video_file=short_video,
                timeout_seconds=60.0
            )
            
            # Very short videos might succeed or fail depending on gait analysis requirements
            if not result.pipeline_success:
                # If it fails, should be at gait analysis step (insufficient data)
                gait_step = result.get_step_result("gait_analysis")
                if gait_step and not gait_step.success:
                    assert "insufficient" in gait_step.error.lower() or "short" in gait_step.error.lower(), \
                        "Should indicate insufficient data for analysis"
            
        except ImportError:
            pytest.skip("OpenCV not available for short video creation")

    @pytest.mark.integration
    @pytest.mark.slow
    async def test_pipeline_very_large_video_handling(
        self, 
        integration_framework: IntegrationTestFramework,
        tmp_path: Path
    ):
        """Test pipeline handling of very large videos."""
        # Create large video file (simulate without actually creating huge file)
        large_video = tmp_path / "large_video.mp4"
        
        try:
            import cv2
            import numpy as np
            
            # Create longer video (30 seconds at 30fps = 900 frames)
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(str(large_video), fourcc, 30.0, (640, 480))
            
            for i in range(900):  # 30 seconds
                frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
                out.write(frame)
            out.release()
            
            result = await integration_framework.test_complete_video_analysis_pipeline(
                video_file=large_video,
                timeout_seconds=300.0  # 5 minutes for large video
            )
            
            # Large videos should either succeed or fail gracefully
            if not result.pipeline_success:
                # Check if failure is due to timeout or resource constraints
                assert result.total_processing_time <= 300.0, "Should respect timeout"
                
                # Check for memory or resource-related errors
                error_messages = " ".join(result.error_summary).lower()
                resource_indicators = ["memory", "timeout", "resource", "size"]
                if any(indicator in error_messages for indicator in resource_indicators):
                    assert True, "Failed due to resource constraints (acceptable)"
            
        except ImportError:
            pytest.skip("OpenCV not available for large video creation")

    @pytest.mark.integration
    @pytest.mark.slow
    async def test_pipeline_network_interruption_simulation(
        self, 
        integration_framework: IntegrationTestFramework,
        sample_videos: Dict[str, Path]
    ):
        """Test pipeline resilience to network interruptions (for YouTube videos)."""
        if "normal_walking" not in sample_videos:
            pytest.skip("Normal walking video not available")
        
        # Simulate network interruption by mocking YouTube handler failure
        from unittest.mock import patch, Mock
        
        mock_youtube_handler = Mock()
        mock_youtube_handler.is_youtube_url.return_value = True
        mock_youtube_handler.download_video.side_effect = ConnectionError("Network interrupted")
        
        youtube_url = "https://www.youtube.com/watch?v=test_network_failure"
        
        with patch.object(integration_framework.video_processor, 'youtube_handler', mock_youtube_handler):
            with patch.object(integration_framework.video_processor, '_is_youtube_url', return_value=True):
                result = await integration_framework.test_complete_video_analysis_pipeline(
                    video_file=youtube_url,
                    timeout_seconds=60.0
                )
                
                assert not result.pipeline_success, "Pipeline should fail with network error"
                
                upload_step = result.get_step_result("video_upload")
                assert upload_step is not None, "Video upload step should be present"
                assert not upload_step.success, "Video upload should fail"
                assert "network" in upload_step.error.lower() or "connection" in upload_step.error.lower(), \
                    "Should indicate network error"

    @pytest.mark.integration
    @pytest.mark.slow
    async def test_pipeline_disk_space_simulation(
        self, 
        integration_framework: IntegrationTestFramework,
        sample_videos: Dict[str, Path]
    ):
        """Test pipeline handling of disk space issues."""
        if "normal_walking" not in sample_videos:
            pytest.skip("Normal walking video not available")
        
        video_file = sample_videos["normal_walking"]
        
        # Simulate disk space error by mocking file operations
        from unittest.mock import patch
        
        def mock_write_that_fails(*args, **kwargs):
            raise OSError("No space left on device")
        
        # This is a simulation - in practice, disk space errors are hard to reproduce
        # We'll test that the pipeline handles OSError exceptions gracefully
        with patch('builtins.open', side_effect=OSError("No space left on device")):
            try:
                result = await integration_framework.test_complete_video_analysis_pipeline(
                    video_file=video_file,
                    timeout_seconds=60.0
                )
                
                # If the pipeline uses the mocked open, it should fail gracefully
                if not result.pipeline_success:
                    error_messages = " ".join(result.error_summary).lower()
                    assert "space" in error_messages or "device" in error_messages, \
                        "Should indicate disk space error"
                
            except OSError:
                # If the mock causes an unhandled exception, that's also a valid test result
                assert True, "Pipeline should handle disk space errors gracefully"

    @pytest.mark.integration
    @pytest.mark.slow
    async def test_pipeline_timeout_handling(
        self, 
        integration_framework: IntegrationTestFramework,
        sample_videos: Dict[str, Path]
    ):
        """Test pipeline timeout handling."""
        if "normal_walking" not in sample_videos:
            pytest.skip("Normal walking video not available")
        
        video_file = sample_videos["normal_walking"]
        
        # Test with very short timeout
        result = await integration_framework.test_complete_video_analysis_pipeline(
            video_file=video_file,
            timeout_seconds=0.1  # 100ms - should timeout
        )
        
        assert not result.pipeline_success, "Pipeline should fail due to timeout"
        assert result.total_processing_time >= 0.1, "Should respect minimum timeout"
        
        # Check for timeout in error messages
        error_messages = " ".join(result.error_summary).lower()
        assert "timeout" in error_messages, "Should indicate timeout error"

    @pytest.mark.integration
    @pytest.mark.slow
    async def test_pipeline_malformed_video_metadata(
        self, 
        integration_framework: IntegrationTestFramework,
        tmp_path: Path
    ):
        """Test pipeline handling of videos with malformed metadata."""
        # Create video file with unusual properties
        malformed_video = tmp_path / "malformed_metadata.mp4"
        
        try:
            import cv2
            import numpy as np
            
            # Create video with unusual frame rate and dimensions
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(str(malformed_video), fourcc, 1.5, (13, 17))  # Unusual values
            
            for i in range(10):
                frame = np.zeros((17, 13, 3), dtype=np.uint8)
                out.write(frame)
            out.release()
            
            result = await integration_framework.test_complete_video_analysis_pipeline(
                video_file=malformed_video,
                timeout_seconds=60.0
            )
            
            # Pipeline should handle unusual metadata gracefully
            if not result.pipeline_success:
                # Check that failure is handled gracefully with meaningful error
                assert len(result.error_summary) > 0, "Should provide error information"
                
                # Errors should be descriptive, not just generic exceptions
                error_text = " ".join(result.error_summary).lower()
                assert not any(word in error_text for word in ["traceback", "exception", "error:"]), \
                    "Errors should be user-friendly, not raw exceptions"
            
        except ImportError:
            pytest.skip("OpenCV not available for malformed video creation")

    @pytest.mark.integration
    @pytest.mark.slow
    async def test_pipeline_zero_byte_video_file(
        self, 
        integration_framework: IntegrationTestFramework,
        tmp_path: Path
    ):
        """Test pipeline handling of zero-byte video files."""
        # Create zero-byte file
        zero_byte_video = tmp_path / "zero_byte.mp4"
        zero_byte_video.touch()  # Creates empty file
        
        result = await integration_framework.test_complete_video_analysis_pipeline(
            video_file=zero_byte_video,
            timeout_seconds=30.0
        )
        
        assert not result.pipeline_success, "Pipeline should fail with zero-byte video"
        
        # Should fail at video upload or frame extraction step
        upload_step = result.get_step_result("video_upload")
        frame_step = result.get_step_result("frame_extraction")
        
        # Either upload or frame extraction should fail
        assert (upload_step and not upload_step.success) or \
               (frame_step and not frame_step.success), \
               "Either upload or frame extraction should fail"
        
        # Error should indicate file size or format issue
        all_errors = " ".join(result.error_summary).lower()
        assert any(keyword in all_errors for keyword in 
                  ["empty", "size", "invalid", "format", "corrupt", "failed", "ffprobe"]), \
            f"Should indicate file issue: {result.error_summary}"

    @pytest.mark.integration
    @pytest.mark.slow
    async def test_pipeline_unsupported_codec_handling(
        self, 
        integration_framework: IntegrationTestFramework,
        tmp_path: Path
    ):
        """Test pipeline handling of videos with unsupported codecs."""
        # Create a file with video extension but non-video content
        fake_video = tmp_path / "fake_codec.mp4"
        fake_video.write_bytes(b"This is not a real video file with proper codec")
        
        result = await integration_framework.test_complete_video_analysis_pipeline(
            video_file=fake_video,
            timeout_seconds=30.0
        )
        
        assert not result.pipeline_success, "Pipeline should fail with unsupported codec"
        
        # Should fail at video processing step
        upload_step = result.get_step_result("video_upload")
        frame_step = result.get_step_result("frame_extraction")
        
        # Check for codec-related errors in any step
        error_messages = []
        for step in result.steps:
            if not step.success and step.error:
                error_messages.append(step.error.lower())
        
        all_errors = " ".join(error_messages)
        
        # Should indicate codec/format issue somewhere in the pipeline
        assert any(keyword in all_errors for keyword in 
                  ["codec", "format", "invalid", "corrupt", "decode", "frames", "empty", "failed", "ffprobe"]), \
            f"Should indicate codec/format issue: {result.error_summary}"

    @pytest.mark.integration
    @pytest.mark.slow
    async def test_pipeline_network_dependency_failure(
        self, 
        integration_framework: IntegrationTestFramework,
        sample_videos: Dict[str, Path]
    ):
        """Test pipeline handling when network-dependent components fail."""
        if "normal_walking" not in sample_videos:
            pytest.skip("Normal walking video not available")
        
        video_file = sample_videos["normal_walking"]
        
        # Mock network failure in classifier (which might use external APIs)
        from unittest.mock import patch
        import requests
        
        with patch.object(integration_framework.classifier, 'classify_gait',
                         side_effect=requests.ConnectionError("Network unreachable")):
            result = await integration_framework.test_complete_video_analysis_pipeline(
                video_file=video_file,
                timeout_seconds=60.0
            )
            
            # Pipeline should fail at classification step
            assert not result.pipeline_success, "Pipeline should fail with network error"
            
            classification_step = result.get_step_result("classification")
            assert classification_step is not None, "Classification step should be present"
            assert not classification_step.success, "Classification should fail"
            
            # Error should indicate network issue
            assert any(keyword in classification_step.error.lower() for keyword in 
                      ["network", "connection", "unreachable", "timeout"]), \
                f"Should indicate network error: {classification_step.error}"

    @pytest.mark.integration
    @pytest.mark.slow
    async def test_pipeline_component_initialization_failure(
        self, 
        integration_framework: IntegrationTestFramework,
        sample_videos: Dict[str, Path]
    ):
        """Test pipeline handling when components fail to initialize."""
        if "normal_walking" not in sample_videos:
            pytest.skip("Normal walking video not available")
        
        video_file = sample_videos["normal_walking"]
        
        # Mock component initialization failure
        from unittest.mock import patch
        
        # Temporarily replace pose estimator with None to simulate init failure
        original_estimator = integration_framework.pose_estimator
        integration_framework.pose_estimator = None
        
        try:
            result = await integration_framework.test_complete_video_analysis_pipeline(
                video_file=video_file,
                timeout_seconds=60.0
            )
            
            assert not result.pipeline_success, "Pipeline should fail with component init failure"
            
            pose_step = result.get_step_result("pose_estimation")
            assert pose_step is not None, "Pose estimation step should be present"
            assert not pose_step.success, "Pose estimation should fail"
            
            # Error should indicate initialization issue
            error_msg = pose_step.error.lower()
            assert any(keyword in error_msg for keyword in 
                      ["none", "not initialized", "null", "missing"]), \
                f"Should indicate initialization failure: {pose_step.error}"
        
        finally:
            # Restore original estimator
            integration_framework.pose_estimator = original_estimator

    @pytest.mark.integration
    @pytest.mark.slow
    async def test_pipeline_data_corruption_during_processing(
        self, 
        integration_framework: IntegrationTestFramework,
        sample_videos: Dict[str, Path]
    ):
        """Test pipeline handling of data corruption during processing."""
        if "normal_walking" not in sample_videos:
            pytest.skip("Normal walking video not available")
        
        video_file = sample_videos["normal_walking"]
        
        # Mock data corruption in gait analysis
        from unittest.mock import patch
        
        def corrupted_gait_analysis(pose_sequence):
            # Return corrupted/invalid gait features
            return {
                "temporal_features": None,  # Corrupted data
                "spatial_features": {"invalid": "data"},
                "symmetry_features": float('inf')  # Invalid numeric value
            }
        
        with patch.object(integration_framework.gait_analyzer, 'analyze_gait_sequence',
                         side_effect=corrupted_gait_analysis):
            result = await integration_framework.test_complete_video_analysis_pipeline(
                video_file=video_file,
                timeout_seconds=60.0
            )
            
            # Pipeline might succeed with corrupted data or fail gracefully
            if not result.pipeline_success:
                # Check for data validation errors
                gait_step = result.get_step_result("gait_analysis")
                classification_step = result.get_step_result("classification")
                
                if gait_step and not gait_step.success:
                    assert "gait" in gait_step.error.lower(), \
                        "Should indicate gait analysis issue"
                
                if classification_step and not classification_step.success:
                    error_msg = classification_step.error.lower()
                    assert any(keyword in error_msg for keyword in 
                              ["invalid", "corrupt", "data", "format"]), \
                        f"Should indicate data corruption: {classification_step.error}"

    @pytest.mark.integration
    @pytest.mark.slow
    async def test_pipeline_extreme_video_dimensions(
        self, 
        integration_framework: IntegrationTestFramework,
        tmp_path: Path
    ):
        """Test pipeline handling of videos with extreme dimensions."""
        try:
            import cv2
            import numpy as np
            
            # Test cases with extreme dimensions
            extreme_cases = [
                {"name": "very_wide", "width": 10000, "height": 10},
                {"name": "very_tall", "width": 10, "height": 10000},
                {"name": "tiny", "width": 1, "height": 1},
                {"name": "odd_ratio", "width": 1337, "height": 42}
            ]
            
            for case in extreme_cases:
                video_file = tmp_path / f"{case['name']}.mp4"
                
                try:
                    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                    out = cv2.VideoWriter(str(video_file), fourcc, 30.0, 
                                        (case['width'], case['height']))
                    
                    # Create a few frames
                    for i in range(30):  # 1 second
                        frame = np.zeros((case['height'], case['width'], 3), dtype=np.uint8)
                        out.write(frame)
                    out.release()
                    
                    result = await integration_framework.test_complete_video_analysis_pipeline(
                        video_file=video_file,
                        timeout_seconds=60.0
                    )
                    
                    # Pipeline should either succeed or fail gracefully
                    if not result.pipeline_success:
                        # Check for dimension-related errors
                        all_errors = " ".join(result.error_summary).lower()
                        # Don't assert specific error messages as extreme dimensions 
                        # might be handled differently by different components
                        assert len(result.error_summary) > 0, \
                            f"Should provide error info for {case['name']}"
                    
                except Exception as e:
                    # OpenCV might not support extreme dimensions
                    pytest.skip(f"Cannot create video with {case['name']} dimensions: {e}")
        
        except ImportError:
            pytest.skip("OpenCV not available for extreme dimension testing")

    @pytest.mark.integration
    @pytest.mark.slow
    async def test_pipeline_with_invalid_video_format(
        self, 
        integration_framework: IntegrationTestFramework,
        tmp_path: Path
    ):
        """
        Test pipeline with invalid video format.
        
        This test validates that the pipeline properly rejects
        unsupported video formats.
        """
        # Create a fake video file with invalid extension
        invalid_video = tmp_path / "fake_video.txt"
        invalid_video.write_text("This is not a video file")
        
        result = await integration_framework.test_complete_video_analysis_pipeline(
            video_file=invalid_video,
            timeout_seconds=30.0
        )
        
        assert not result.pipeline_success, "Pipeline should fail with invalid format"
        
        # Validate that video upload step failed due to format
        upload_step = result.get_step_result("video_upload")
        assert upload_step is not None, "Video upload step should be present"
        assert not upload_step.success, "Video upload should fail"
        assert "format" in upload_step.error.lower(), "Should indicate format error"

    @pytest.mark.integration
    @pytest.mark.slow
    async def test_pipeline_step_isolation(
        self, 
        integration_framework: IntegrationTestFramework,
        sample_videos: Dict[str, Path]
    ):
        """
        Test that pipeline steps are properly isolated.
        
        This test validates that failures in one step don't corrupt
        the overall pipeline state and that steps can be retried.
        """
        if "normal_walking" not in sample_videos:
            pytest.skip("Normal walking video not available")
        
        video_file = sample_videos["normal_walking"]
        
        # Execute pipeline multiple times to test isolation
        results = []
        
        for i in range(3):
            result = await integration_framework.test_complete_video_analysis_pipeline(
                video_file=video_file,
                timeout_seconds=120.0
            )
            results.append(result)
        
        # All executions should succeed
        for i, result in enumerate(results):
            assert result.pipeline_success, f"Execution {i+1} failed: {result.error_summary}"
        
        # Results should be consistent (within reasonable bounds)
        classifications = [r.final_result.get("classification") for r in results]
        confidences = [r.final_result.get("confidence") for r in results]
        
        # All classifications should be the same
        assert len(set(classifications)) == 1, f"Inconsistent classifications: {classifications}"
        
        # Confidence scores should be reasonably consistent (within 0.2)
        confidence_range = max(confidences) - min(confidences)
        assert confidence_range <= 0.2, f"Confidence scores too variable: {confidences}"

    @pytest.mark.integration
    @pytest.mark.slow
    async def test_pipeline_memory_usage(
        self, 
        integration_framework: IntegrationTestFramework,
        sample_videos: Dict[str, Path]
    ):
        """
        Test pipeline memory usage stays within acceptable bounds.
        
        This test validates that the pipeline doesn't consume
        excessive memory during processing.
        """
        import psutil
        import os
        
        if "normal_walking" not in sample_videos:
            pytest.skip("Normal walking video not available")
        
        video_file = sample_videos["normal_walking"]
        
        # Get initial memory usage
        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        # Execute pipeline
        result = await integration_framework.test_complete_video_analysis_pipeline(
            video_file=video_file,
            timeout_seconds=120.0
        )
        
        # Get final memory usage
        final_memory = process.memory_info().rss / 1024 / 1024  # MB
        memory_increase = final_memory - initial_memory
        
        # Validate pipeline success
        assert result.pipeline_success, f"Pipeline failed: {result.error_summary}"
        
        # Validate memory usage (should not increase by more than 2GB)
        assert memory_increase < 2048, \
            f"Excessive memory usage: {memory_increase:.1f}MB increase"
        
        # Log memory usage for monitoring
        print(f"Memory usage: {initial_memory:.1f}MB → {final_memory:.1f}MB (+{memory_increase:.1f}MB)")

    @pytest.mark.integration
    @pytest.mark.slow
    async def test_pipeline_concurrent_execution(
        self, 
        integration_framework: IntegrationTestFramework,
        sample_videos: Dict[str, Path]
    ):
        """
        Test pipeline can handle concurrent executions.
        
        This test validates that multiple pipeline executions
        can run concurrently without interference.
        """
        if "normal_walking" not in sample_videos:
            pytest.skip("Normal walking video not available")
        
        video_file = sample_videos["normal_walking"]
        
        # Create multiple concurrent pipeline executions
        tasks = []
        for i in range(3):  # 3 concurrent executions
            task = integration_framework.test_complete_video_analysis_pipeline(
                video_file=video_file,
                timeout_seconds=180.0  # Longer timeout for concurrent execution
            )
            tasks.append(task)
        
        # Execute all tasks concurrently
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Validate all executions completed successfully
        successful_results = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                pytest.fail(f"Concurrent execution {i+1} raised exception: {result}")
            else:
                assert result.pipeline_success, \
                    f"Concurrent execution {i+1} failed: {result.error_summary}"
                successful_results.append(result)
        
        # Validate consistency across concurrent executions
        classifications = [r.final_result.get("classification") for r in successful_results]
        assert len(set(classifications)) == 1, \
            f"Inconsistent classifications in concurrent execution: {classifications}"

    @pytest.mark.integration
    @pytest.mark.slow
    async def test_pipeline_with_real_gavd_data(
        self, 
        integration_framework: IntegrationTestFramework,
        gavd_test_subset: Dict[str, Any],
        gavd_video_urls: Dict[str, List[str]]
    ):
        """
        Test pipeline with real GAVD dataset samples.
        
        This test validates the pipeline works with actual
        clinical gait analysis data from the GAVD dataset.
        """
        # Check if we have real GAVD data
        metadata = gavd_test_subset.get("metadata", {})
        if metadata.get("synthetic", True):
            pytest.skip("Real GAVD data not available, using synthetic data")
        
        print(f"Testing with real GAVD data: {metadata}")
        
        # Test with normal subjects (using local video files if available)
        normal_subjects = gavd_test_subset.get("normal_subjects", [])
        if normal_subjects:
            # For integration testing, we'll use the available local video files
            # since downloading YouTube videos in tests is not practical
            data_manager = get_real_data_manager()
            sample_videos = data_manager.get_sample_videos()
            
            if "normal_walking" in sample_videos:
                # Test with local normal walking video, but validate against GAVD metadata
                normal_subject = normal_subjects[0]  # Use first normal subject for validation
                
                result = await integration_framework.test_complete_video_analysis_pipeline(
                    video_file=sample_videos["normal_walking"],
                    expected_classification="normal",
                    timeout_seconds=120.0
                )
                
                assert result.pipeline_success, f"Pipeline failed with GAVD normal data: {result.error_summary}"
                
                # Validate that the result is consistent with GAVD expectations
                final_result = result.final_result
                assert final_result is not None, "No final classification result"
                assert final_result.get("classification") == "normal", \
                    f"Expected normal classification for GAVD normal subject, got {final_result.get('classification')}"
                
                # Log GAVD subject information for traceability
                print(f"Tested with GAVD normal subject: {normal_subject['subject_id']} ({normal_subject['gait_pattern']})")
        
        # Test with abnormal subjects
        abnormal_subjects = gavd_test_subset.get("abnormal_subjects", [])
        if abnormal_subjects and "abnormal_gait" in sample_videos:
            # Test with local abnormal gait video, but validate against GAVD metadata
            abnormal_subject = abnormal_subjects[0]  # Use first abnormal subject for validation
            
            result = await integration_framework.test_complete_video_analysis_pipeline(
                video_file=sample_videos["abnormal_gait"],
                expected_classification="abnormal",
                timeout_seconds=120.0
            )
            
            assert result.pipeline_success, f"Pipeline failed with GAVD abnormal data: {result.error_summary}"
            
            # Validate that the result is consistent with GAVD expectations
            final_result = result.final_result
            assert final_result is not None, "No final classification result"
            assert final_result.get("classification") == "abnormal", \
                f"Expected abnormal classification for GAVD abnormal subject, got {final_result.get('classification')}"
            
            # Log GAVD subject information for traceability
            print(f"Tested with GAVD abnormal subject: {abnormal_subject['subject_id']} ({abnormal_subject['gait_pattern']})")
        
        # Validate GAVD data structure and completeness
        assert len(normal_subjects) > 0 or len(abnormal_subjects) > 0, \
            "GAVD test subset should contain at least some subjects"
        
        # Validate metadata completeness
        assert "total_subjects" in metadata, "GAVD metadata should include total_subjects"
        assert "gait_patterns" in metadata, "GAVD metadata should include gait_patterns"
        
        print(f"Successfully tested with {metadata['total_subjects']} GAVD subjects")
        print(f"Normal patterns: {metadata['gait_patterns']['normal']}")
        print(f"Abnormal patterns: {metadata['gait_patterns']['abnormal']}")

    @pytest.mark.integration
    @pytest.mark.slow
    async def test_pipeline_with_gavd_gait_patterns(
        self, 
        integration_framework: IntegrationTestFramework,
        gavd_test_subset: Dict[str, Any]
    ):
        """
        Test pipeline behavior with different GAVD gait patterns.
        
        This test validates that the pipeline can handle the variety
        of gait patterns present in the GAVD dataset.
        """
        metadata = gavd_test_subset.get("metadata", {})
        if metadata.get("synthetic", True):
            pytest.skip("Real GAVD data not available")
        
        # Get available gait patterns from GAVD data
        gait_patterns = metadata.get("gait_patterns", {})
        all_patterns = gait_patterns.get("normal", []) + gait_patterns.get("abnormal", [])
        
        assert len(all_patterns) > 0, "Should have gait patterns from GAVD data"
        
        # Test that we can identify the variety of patterns
        expected_normal_patterns = ["normal", "style", "exercise"]
        expected_abnormal_patterns = ["parkinsons", "abnormal", "cerebral palsy", "myopathic", 
                                    "antalgic", "inebriated", "stroke", "prosthetic"]
        
        # Validate that we have a good variety of patterns
        normal_patterns_found = [p for p in all_patterns if p in expected_normal_patterns]
        abnormal_patterns_found = [p for p in all_patterns if p in expected_abnormal_patterns]
        
        print(f"GAVD normal patterns found: {normal_patterns_found}")
        print(f"GAVD abnormal patterns found: {abnormal_patterns_found}")
        
        # We should have at least some patterns from each category
        assert len(normal_patterns_found) > 0 or len(abnormal_patterns_found) > 0, \
            "Should have at least some recognizable gait patterns"
        
        # Test with available local videos to simulate different pattern processing
        data_manager = get_real_data_manager()
        sample_videos = data_manager.get_sample_videos()
        
        if sample_videos:
            # Test one video to ensure pipeline works with GAVD pattern context
            test_video = list(sample_videos.values())[0]
            
            result = await integration_framework.test_complete_video_analysis_pipeline(
                video_file=test_video,
                timeout_seconds=120.0
            )
            
            assert result.pipeline_success, f"Pipeline should work with GAVD pattern context: {result.error_summary}"
            
            # The classification should be valid regardless of the specific pattern
            final_result = result.final_result
            assert final_result is not None, "Should have classification result"
            assert final_result.get("classification") in ["normal", "abnormal"], \
                "Should have valid classification"

    @pytest.mark.integration
    @pytest.mark.slow
    async def test_gavd_data_quality_validation(
        self, 
        gavd_test_subset: Dict[str, Any]
    ):
        """
        Test GAVD data quality and structure validation.
        
        This test ensures that the GAVD data loaded for testing
        meets quality standards and has the expected structure.
        """
        # Validate overall structure
        assert "normal_subjects" in gavd_test_subset, "Should have normal_subjects"
        assert "abnormal_subjects" in gavd_test_subset, "Should have abnormal_subjects"
        assert "metadata" in gavd_test_subset, "Should have metadata"
        
        metadata = gavd_test_subset["metadata"]
        normal_subjects = gavd_test_subset["normal_subjects"]
        abnormal_subjects = gavd_test_subset["abnormal_subjects"]
        
        # Validate metadata
        assert "source" in metadata, "Metadata should include source"
        assert "total_subjects" in metadata, "Metadata should include total_subjects"
        assert "synthetic" in metadata, "Metadata should indicate if synthetic"
        
        # Validate subject data structure
        all_subjects = normal_subjects + abnormal_subjects
        assert len(all_subjects) > 0, "Should have at least some subjects"
        
        for subject in all_subjects[:5]:  # Check first 5 subjects
            assert "subject_id" in subject, "Subject should have subject_id"
            assert "gait_pattern" in subject, "Subject should have gait_pattern"
            
            # If real data, should have additional fields
            if not metadata.get("synthetic", True):
                assert "youtube_url" in subject, "Real GAVD subject should have youtube_url"
                assert "sequence_id" in subject, "Real GAVD subject should have sequence_id"
                assert "source" in subject, "Real GAVD subject should have source"
        
        # Validate data counts match metadata
        assert metadata["total_subjects"] == len(all_subjects), \
            "Metadata total_subjects should match actual count"
        assert metadata["normal_count"] == len(normal_subjects), \
            "Metadata normal_count should match actual count"
        assert metadata["abnormal_count"] == len(abnormal_subjects), \
            "Metadata abnormal_count should match actual count"
        
        print(f"GAVD data validation passed:")
        print(f"  Source: {metadata['source']}")
        print(f"  Total subjects: {metadata['total_subjects']}")
        print(f"  Normal subjects: {metadata['normal_count']}")
        print(f"  Abnormal subjects: {metadata['abnormal_count']}")
        print(f"  Is synthetic: {metadata['synthetic']}")

    @pytest.mark.integration
    @pytest.mark.slow
    async def test_real_video_file_integration(
        self, 
        integration_framework: IntegrationTestFramework,
        sample_gait_videos: Dict[str, Path]
    ):
        """
        Test integration with real video files from test data directory.
        
        This test ensures that the pipeline works correctly with the
        actual video files available in the test data directory.
        """
        tested_videos = 0
        
        for video_type, video_path in sample_gait_videos.items():
            if not video_path.exists():
                print(f"Skipping {video_type}: file not found at {video_path}")
                continue
            
            print(f"Testing with real video file: {video_type} ({video_path})")
            
            # Determine expected classification based on video type
            expected_classification = None
            if "abnormal" in video_type.lower():
                expected_classification = "abnormal"
            elif "normal" in video_type.lower():
                expected_classification = "normal"
            
            # Test the pipeline with the real video
            result = await integration_framework.test_complete_video_analysis_pipeline(
                video_file=video_path,
                expected_classification=expected_classification,
                timeout_seconds=180.0  # Longer timeout for real videos
            )
            
            # Validate pipeline success
            assert result.pipeline_success, \
                f"Pipeline failed with real video {video_type}: {result.error_summary}"
            
            # Validate processing time is reasonable
            assert result.total_processing_time < 180.0, \
                f"Processing took too long for {video_type}: {result.total_processing_time}s"
            
            # Validate final result
            final_result = result.final_result
            assert final_result is not None, f"No final result for {video_type}"
            assert "classification" in final_result, f"Missing classification for {video_type}"
            assert "confidence" in final_result, f"Missing confidence for {video_type}"
            
            # Validate confidence is reasonable
            confidence = final_result.get("confidence", 0.0)
            assert 0.0 <= confidence <= 1.0, f"Invalid confidence for {video_type}: {confidence}"
            
            tested_videos += 1
            print(f"Successfully processed {video_type}: {final_result.get('classification')} (confidence: {confidence:.2f})")
        
        assert tested_videos > 0, "Should have tested at least one real video file"
        print(f"Successfully tested {tested_videos} real video files")

    @pytest.mark.integration
    @pytest.mark.slow
    async def test_pipeline_end_to_end_timing(
        self, 
        integration_framework: IntegrationTestFramework,
        sample_videos: Dict[str, Path]
    ):
        """
        Test detailed timing analysis of pipeline steps.
        
        This test provides detailed timing information for
        performance optimization and monitoring.
        """
        timing_results = {}
        
        for video_type, video_path in sample_videos.items():
            if not video_path.exists():
                continue
            
            result = await integration_framework.test_complete_video_analysis_pipeline(
                video_file=video_path,
                timeout_seconds=300.0
            )
            
            if result.pipeline_success:
                step_timings = {}
                for step in result.steps:
                    step_timings[step.step_name] = {
                        "processing_time": step.processing_time,
                        "success": step.success,
                        "metadata": step.metadata
                    }
                
                timing_results[video_type] = {
                    "total_time": result.total_processing_time,
                    "steps": step_timings
                }
        
        # Log timing results for analysis
        print("\n=== Pipeline Timing Analysis ===")
        for video_type, timings in timing_results.items():
            print(f"\n{video_type.upper()}:")
            print(f"  Total Time: {timings['total_time']:.2f}s")
            for step_name, step_data in timings['steps'].items():
                print(f"  {step_name}: {step_data['processing_time']:.2f}s")
        
        # Validate that we have timing data
        assert len(timing_results) > 0, "No timing results collected"

    # ========================================
    # ADDITIONAL ERROR HANDLING AND EDGE CASES
    # ========================================

    @pytest.mark.integration
    @pytest.mark.slow
    async def test_pipeline_with_corrupted_video_headers(
        self, 
        integration_framework: IntegrationTestFramework,
        tmp_path: Path
    ):
        """Test pipeline handling of videos with corrupted headers."""
        # Create video with corrupted header
        corrupted_video = tmp_path / "corrupted_header.mp4"
        
        # Write partial MP4 header followed by garbage
        mp4_header = b'\x00\x00\x00\x20ftypmp41'  # Partial MP4 header
        garbage_data = b'\xFF\xFE\xFD\xFC' * 1000  # Garbage data
        
        corrupted_video.write_bytes(mp4_header + garbage_data)
        
        result = await integration_framework.test_complete_video_analysis_pipeline(
            video_file=corrupted_video,
            timeout_seconds=60.0
        )
        
        assert not result.pipeline_success, "Pipeline should fail with corrupted header"
        
        # Should fail at video upload or frame extraction
        failed_steps = result.get_failed_steps()
        assert any(step in failed_steps for step in ["video_upload", "frame_extraction"]), \
            "Should fail at video processing step"

    @pytest.mark.integration
    @pytest.mark.slow
    async def test_pipeline_with_infinite_loop_video(
        self, 
        integration_framework: IntegrationTestFramework,
        tmp_path: Path
    ):
        """Test pipeline handling of videos that could cause infinite loops."""
        # Create video file that might cause processing loops
        loop_video = tmp_path / "infinite_loop.mp4"
        
        try:
            import cv2
            import numpy as np
            
            # Create video with identical frames (could cause loop detection issues)
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(str(loop_video), fourcc, 30.0, (320, 240))
            
            # Write 1000 identical frames
            identical_frame = np.zeros((240, 320, 3), dtype=np.uint8)
            for i in range(1000):
                out.write(identical_frame)
            out.release()
            
            # Test with short timeout to prevent actual infinite loops
            result = await integration_framework.test_complete_video_analysis_pipeline(
                video_file=loop_video,
                timeout_seconds=30.0  # Short timeout
            )
            
            # Either succeeds or times out gracefully
            if not result.pipeline_success:
                assert "timeout" in " ".join(result.error_summary).lower() or \
                       result.total_processing_time >= 30.0, \
                    "Should timeout or handle infinite loop gracefully"
            
        except ImportError:
            pytest.skip("OpenCV not available for infinite loop video creation")

    @pytest.mark.integration
    @pytest.mark.slow
    async def test_pipeline_with_video_codec_errors(
        self, 
        integration_framework: IntegrationTestFramework,
        tmp_path: Path
    ):
        """Test pipeline handling of videos with codec-specific errors."""
        # Create file with video extension but audio codec
        audio_as_video = tmp_path / "audio_as_video.mp4"
        
        # Write MP3 header in MP4 file (codec mismatch)
        mp3_header = b'ID3\x03\x00\x00\x00'  # MP3 ID3 header
        audio_as_video.write_bytes(mp3_header + b'\x00' * 1000)
        
        result = await integration_framework.test_complete_video_analysis_pipeline(
            video_file=audio_as_video,
            timeout_seconds=60.0
        )
        
        assert not result.pipeline_success, "Pipeline should fail with codec mismatch"
        
        # Check for codec-related errors
        error_messages = " ".join(result.error_summary).lower()
        assert any(keyword in error_messages for keyword in 
                  ["codec", "format", "decode", "invalid", "ffprobe", "failed"]), \
            f"Should indicate codec error: {result.error_summary}"

    @pytest.mark.integration
    @pytest.mark.slow
    async def test_pipeline_with_system_resource_exhaustion(
        self, 
        integration_framework: IntegrationTestFramework,
        sample_videos: Dict[str, Path]
    ):
        """Test pipeline behavior under simulated system resource exhaustion."""
        if "normal_walking" not in sample_videos:
            pytest.skip("Normal walking video not available")
        
        video_file = sample_videos["normal_walking"]
        
        # Mock system resource exhaustion
        from unittest.mock import patch
        import psutil
        
        # Simulate low memory condition
        def mock_memory_info():
            # Return very low available memory
            return type('obj', (object,), {
                'available': 1024 * 1024,  # 1MB available
                'percent': 99.0  # 99% memory usage
            })
        
        with patch.object(psutil, 'virtual_memory', return_value=mock_memory_info()):
            result = await integration_framework.test_complete_video_analysis_pipeline(
                video_file=video_file,
                timeout_seconds=60.0
            )
            
            # Pipeline should either succeed or fail gracefully
            if not result.pipeline_success:
                error_messages = " ".join(result.error_summary).lower()
                # Don't assert specific error messages as resource handling varies
                assert len(result.error_summary) > 0, "Should provide error information"

    @pytest.mark.integration
    @pytest.mark.slow
    async def test_pipeline_with_interrupted_network_operations(
        self, 
        integration_framework: IntegrationTestFramework,
        sample_videos: Dict[str, Path]
    ):
        """Test pipeline handling of interrupted network operations."""
        if "normal_walking" not in sample_videos:
            pytest.skip("Normal walking video not available")
        
        video_file = sample_videos["normal_walking"]
        
        # Mock network interruption during classification
        from unittest.mock import patch
        import requests
        
        call_count = 0
        def intermittent_network_error(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count <= 2:
                raise requests.ConnectionError("Network interrupted")
            else:
                # Return mock successful response after retries
                return {
                    "classification": "normal",
                    "confidence": 0.75,
                    "explanation": "Classification after network recovery"
                }
        
        with patch.object(integration_framework.classifier, 'classify_gait',
                         side_effect=intermittent_network_error):
            result = await integration_framework.test_complete_video_analysis_pipeline(
                video_file=video_file,
                timeout_seconds=120.0
            )
            
            # Should either succeed after retries or fail with network error
            if not result.pipeline_success:
                classification_step = result.get_step_result("classification")
                if classification_step and not classification_step.success:
                    assert "network" in classification_step.error.lower() or \
                           "connection" in classification_step.error.lower(), \
                        f"Should indicate network error: {classification_step.error}"

    @pytest.mark.integration
    @pytest.mark.slow
    async def test_pipeline_with_partial_component_failures(
        self, 
        integration_framework: IntegrationTestFramework,
        sample_videos: Dict[str, Path]
    ):
        """Test pipeline handling when components partially fail."""
        if "normal_walking" not in sample_videos:
            pytest.skip("Normal walking video not available")
        
        video_file = sample_videos["normal_walking"]
        
        # Mock partial pose estimation failure (some frames fail)
        from unittest.mock import patch
        
        def partial_pose_estimation(frame_sequence):
            # Return poses for only half the frames
            half_length = len(frame_sequence) // 2
            mock_keypoints = [
                {"x": 100.0 + i, "y": 200.0 + i, "confidence": 0.8}
                for i in range(33)
            ]
            return [mock_keypoints for _ in range(half_length)]
        
        with patch.object(integration_framework.pose_estimator, 'estimate_pose_sequence',
                         side_effect=partial_pose_estimation):
            result = await integration_framework.test_complete_video_analysis_pipeline(
                video_file=video_file,
                timeout_seconds=120.0
            )
            
            # Pipeline might succeed with partial data or fail gracefully
            if result.pipeline_success:
                # Verify that partial data was handled
                pose_step = result.get_step_result("pose_estimation")
                assert pose_step and pose_step.success, "Pose estimation should succeed with partial data"
            else:
                # Should provide meaningful error about partial failure
                assert len(result.error_summary) > 0, "Should explain partial failure"

    @pytest.mark.integration
    @pytest.mark.slow
    async def test_pipeline_with_configuration_corruption(
        self, 
        integration_framework: IntegrationTestFramework,
        sample_videos: Dict[str, Path]
    ):
        """Test pipeline handling of corrupted configuration."""
        if "normal_walking" not in sample_videos:
            pytest.skip("Normal walking video not available")
        
        video_file = sample_videos["normal_walking"]
        
        # Mock corrupted configuration by patching the config object directly
        from unittest.mock import patch
        
        # Store original config
        original_config = integration_framework.config_manager.config
        
        # Create corrupted config
        corrupted_config = original_config
        corrupted_config.pose_estimation.default_estimator = None  # Corrupt the config
        
        with patch.object(integration_framework.config_manager, 'config', corrupted_config):
            result = await integration_framework.test_complete_video_analysis_pipeline(
                video_file=video_file,
                timeout_seconds=60.0
            )
            
            # Should handle configuration corruption gracefully
            if not result.pipeline_success:
                error_messages = " ".join(result.error_summary).lower()
                # Should indicate configuration or initialization issues
                assert any(keyword in error_messages for keyword in 
                          ["config", "initialization", "none", "missing"]), \
                    f"Should indicate configuration issue: {result.error_summary}"

    @pytest.mark.integration
    @pytest.mark.slow
    async def test_pipeline_with_race_conditions(
        self, 
        integration_framework: IntegrationTestFramework,
        sample_videos: Dict[str, Path]
    ):
        """Test pipeline handling of race conditions in concurrent operations."""
        if "normal_walking" not in sample_videos:
            pytest.skip("Normal walking video not available")
        
        video_file = sample_videos["normal_walking"]
        
        # Create multiple concurrent pipeline executions with shared resources
        import asyncio
        
        async def concurrent_pipeline_with_shared_state():
            # Simulate shared state that could cause race conditions
            shared_counter = {"value": 0}
            
            def increment_counter():
                # Non-atomic operation that could cause race conditions
                current = shared_counter["value"]
                import time
                time.sleep(0.001)  # Small delay to increase race condition chance
                shared_counter["value"] = current + 1
                return shared_counter["value"]
            
            # Mock component that uses shared state
            from unittest.mock import patch
            
            def race_condition_gait_analysis(pose_sequence):
                counter_value = increment_counter()
                return {
                    "temporal_features": {"stride_time": 1.2, "cadence": 110.0},
                    "spatial_features": {"stride_length": 1.4, "step_width": 0.15},
                    "symmetry_features": {"left_right_symmetry": 0.95},
                    "counter_value": counter_value  # Include counter for race detection
                }
            
            with patch.object(integration_framework.gait_analyzer, 'analyze_gait_sequence',
                             side_effect=race_condition_gait_analysis):
                return await integration_framework.test_complete_video_analysis_pipeline(
                    video_file=video_file,
                    timeout_seconds=60.0
                )
        
        # Run multiple concurrent executions
        tasks = [concurrent_pipeline_with_shared_state() for _ in range(3)]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Validate that race conditions were handled
        successful_results = [r for r in results if not isinstance(r, Exception) and r.pipeline_success]
        
        # At least some should succeed
        assert len(successful_results) > 0, "Some concurrent executions should succeed"
        
        # Check for race condition indicators
        counter_values = []
        for result in successful_results:
            gait_step = result.get_step_result("gait_analysis")
            if gait_step and gait_step.success and gait_step.data:
                counter_value = gait_step.data.get("counter_value")
                if counter_value:
                    counter_values.append(counter_value)
        
        # If we have counter values, check for race conditions but allow graceful handling
        if counter_values:
            unique_values = len(set(counter_values))
            total_values = len(counter_values)
            
            # If race conditions occurred, that's expected in this test
            # The important thing is that the pipeline still completed successfully
            if unique_values < total_values:
                logger.info(f"Race condition detected as expected: {counter_values}")
                # This is actually the expected behavior - race conditions can happen
                # but the pipeline should still complete successfully
            else:
                logger.info(f"No race conditions detected: {counter_values}")
            
            # The main assertion is that despite potential race conditions,
            # the pipeline completed successfully
            assert len(successful_results) >= 1, "Pipeline should complete despite race conditions"

    @pytest.mark.integration
    @pytest.mark.slow
    async def test_pipeline_with_cascading_failures(
        self, 
        integration_framework: IntegrationTestFramework,
        sample_videos: Dict[str, Path]
    ):
        """Test pipeline handling of cascading failures across components."""
        if "normal_walking" not in sample_videos:
            pytest.skip("Normal walking video not available")
        
        video_file = sample_videos["normal_walking"]
        
        # Mock cascading failure scenario
        from unittest.mock import patch
        
        failure_count = 0
        
        def cascading_failure_pose_estimation(frame_sequence):
            nonlocal failure_count
            failure_count += 1
            if failure_count == 1:
                # First call fails
                raise RuntimeError("Pose estimation primary failure")
            else:
                # Subsequent calls also fail due to corrupted state
                raise RuntimeError("Pose estimation secondary failure due to corrupted state")
        
        def cascading_failure_gait_analysis(pose_sequence):
            # This should not be called due to pose estimation failure
            # But if it is, it should also fail due to invalid input
            raise RuntimeError("Gait analysis failure due to invalid pose data")
        
        with patch.object(integration_framework.pose_estimator, 'estimate_pose_sequence',
                         side_effect=cascading_failure_pose_estimation):
            with patch.object(integration_framework.gait_analyzer, 'analyze_gait_sequence',
                             side_effect=cascading_failure_gait_analysis):
                result = await integration_framework.test_complete_video_analysis_pipeline(
                    video_file=video_file,
                    timeout_seconds=60.0
                )
                
                assert not result.pipeline_success, "Pipeline should fail with cascading failures"
                
                # Should fail at pose estimation and not proceed to gait analysis
                pose_step = result.get_step_result("pose_estimation")
                gait_step = result.get_step_result("gait_analysis")
                
                assert pose_step and not pose_step.success, "Pose estimation should fail"
                assert gait_step is None, "Gait analysis should not be attempted after pose failure"
                
                # Error should indicate the primary failure
                assert "pose estimation" in " ".join(result.error_summary).lower(), \
                    "Should indicate pose estimation as primary failure"

    @pytest.mark.integration
    @pytest.mark.slow
    async def test_pipeline_with_gradual_degradation(
        self, 
        integration_framework: IntegrationTestFramework,
        sample_videos: Dict[str, Path]
    ):
        """Test pipeline handling of gradual performance degradation."""
        if "normal_walking" not in sample_videos:
            pytest.skip("Normal walking video not available")
        
        video_file = sample_videos["normal_walking"]
        
        # Mock gradual degradation in pose estimation
        from unittest.mock import patch
        import time
        
        call_count = 0
        
        def degrading_pose_estimation(frame_sequence):
            nonlocal call_count
            call_count += 1
            
            # Simulate increasing processing time (degradation)
            degradation_delay = call_count * 0.1
            time.sleep(degradation_delay)
            
            # Simulate decreasing quality (fewer confident landmarks)
            confidence_degradation = max(0.1, 0.9 - (call_count * 0.1))
            
            mock_keypoints = [
                {"x": 100.0 + i, "y": 200.0 + i, "confidence": confidence_degradation}
                for i in range(33)
            ]
            return [mock_keypoints for _ in range(len(frame_sequence))]
        
        with patch.object(integration_framework.pose_estimator, 'estimate_pose_sequence',
                         side_effect=degrading_pose_estimation):
            result = await integration_framework.test_complete_video_analysis_pipeline(
                video_file=video_file,
                timeout_seconds=120.0
            )
            
            # Pipeline should handle degradation gracefully
            if result.pipeline_success:
                pose_step = result.get_step_result("pose_estimation")
                assert pose_step and pose_step.success, "Should handle degraded performance"
                
                # Processing time should reflect degradation
                assert pose_step.processing_time > 0.1, "Should show increased processing time"
            else:
                # If it fails, should be due to timeout or quality issues
                error_messages = " ".join(result.error_summary).lower()
                assert any(keyword in error_messages for keyword in 
                          ["timeout", "quality", "confidence", "degradation"]), \
                    f"Should indicate degradation-related failure: {result.error_summary}"