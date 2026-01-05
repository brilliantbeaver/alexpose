"""Tests for configurable performance regression tolerance system."""

import pytest
import tempfile
import json
from pathlib import Path
from unittest.mock import patch, MagicMock
import os

from tests.performance.benchmark_framework import PerformanceBenchmark, PerformanceMetrics
from tests.performance.performance_config import PerformanceConfig, get_performance_config, configure_test_tolerance


class TestConfigurableTolerance:
    """Test suite for configurable performance regression tolerance."""
    
    def setup_method(self):
        """Set up test environment."""
        self.temp_dir = Path(tempfile.mkdtemp())
        self.config_file = self.temp_dir / "test_performance_config.json"
        self.baseline_file = self.temp_dir / "test_baselines.json"
    
    def teardown_method(self):
        """Clean up test environment."""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    @pytest.mark.fast
    def test_performance_config_creation(self):
        """Test that performance configuration can be created and loaded."""
        config = PerformanceConfig(self.config_file)
        
        # Should create default configuration
        assert self.config_file.exists()
        
        # Check default values
        assert config.tolerance_config.execution_time_tolerance == 10.0
        assert config.tolerance_config.memory_usage_tolerance == 15.0
        assert config.tolerance_config.throughput_tolerance == 10.0
        
        # Check that test-specific tolerances are set up
        assert len(config.tolerance_config.test_specific_tolerances) > 0
        assert "video_loading_30s" in config.tolerance_config.test_specific_tolerances
    
    @pytest.mark.fast
    def test_tolerance_retrieval(self):
        """Test that tolerances can be retrieved correctly."""
        config = PerformanceConfig(self.config_file)
        
        # Test default tolerance
        default_tolerance = config.get_tolerance("unknown_test", "execution_time")
        assert default_tolerance == 10.0  # Default value
        
        # Test test-specific tolerance
        video_tolerance = config.get_tolerance("video_loading_30s", "execution_time")
        assert video_tolerance == 15.0  # Specific value from config
        
        # Test memory tolerance
        memory_tolerance = config.get_tolerance("video_loading_30s", "memory_usage")
        assert memory_tolerance == 20.0  # Specific memory tolerance
    
    @pytest.mark.fast
    def test_ci_environment_multiplier(self):
        """Test that CI environment applies tolerance multiplier."""
        config = PerformanceConfig(self.config_file)
        
        # Test local environment (default)
        local_tolerance = config.get_tolerance("video_loading_30s", "execution_time")
        
        # Mock CI environment
        with patch.dict(os.environ, {'CI': 'true'}):
            ci_tolerance = config.get_tolerance("video_loading_30s", "execution_time")
            
            # CI tolerance should be higher (multiplied by ci_multiplier)
            expected_ci_tolerance = 15.0 * 1.5  # base tolerance * ci_multiplier
            assert ci_tolerance == expected_ci_tolerance
            assert ci_tolerance > local_tolerance
    
    @pytest.mark.fast
    def test_tolerance_configuration_update(self):
        """Test that tolerance configuration can be updated."""
        config = PerformanceConfig(self.config_file)
        
        # Set new tolerance for a test
        config.set_test_tolerance("new_test", execution_time=25.0, memory_usage=30.0)
        
        # Verify the tolerance was set
        assert config.get_tolerance("new_test", "execution_time") == 25.0
        assert config.get_tolerance("new_test", "memory_usage") == 30.0
        
        # Verify configuration was saved
        assert self.config_file.exists()
        
        # Create new config instance to verify persistence
        new_config = PerformanceConfig(self.config_file)
        assert new_config.get_tolerance("new_test", "execution_time") == 25.0
        assert new_config.get_tolerance("new_test", "memory_usage") == 30.0
    
    @pytest.mark.fast
    def test_benchmark_framework_integration(self):
        """Test that benchmark framework integrates with configurable tolerance."""
        benchmark = PerformanceBenchmark(
            baseline_file=self.baseline_file,
            use_config=True
        )
        
        # Create test metrics
        baseline_metrics = PerformanceMetrics(
            execution_time=1.0,
            memory_usage_mb=100.0,
            cpu_usage_percent=50.0,
            peak_memory_mb=150.0,
            throughput=10.0
        )
        
        # Establish baseline
        benchmark.establish_baseline("test_configurable", baseline_metrics)
        
        # Create current metrics with slight regression
        current_metrics = PerformanceMetrics(
            execution_time=1.12,  # 12% increase
            memory_usage_mb=110.0,  # 10% increase
            cpu_usage_percent=55.0,
            peak_memory_mb=165.0,  # 10% increase
            throughput=9.0  # 10% decrease
        )
        
        # Test regression detection with default tolerance (10%)
        result = benchmark.validate_performance_regression("test_configurable", current_metrics)
        
        # Should detect regression (12% > 10% default tolerance)
        assert result['regression_detected'] is True
        assert len(result['regressions']) > 0
        assert 'tolerances_used' in result
    
    @pytest.mark.fast
    def test_benchmark_framework_custom_tolerance(self):
        """Test that benchmark framework respects custom tolerance values."""
        benchmark = PerformanceBenchmark(
            baseline_file=self.baseline_file,
            use_config=True
        )
        
        # Configure higher tolerance for this test
        benchmark.configure_test_tolerance("test_custom", execution_time=20.0, memory_usage=25.0, throughput=20.0)
        
        # Create test metrics
        baseline_metrics = PerformanceMetrics(
            execution_time=1.0,
            memory_usage_mb=100.0,
            cpu_usage_percent=50.0,
            peak_memory_mb=150.0,
            throughput=10.0
        )
        
        # Establish baseline
        benchmark.establish_baseline("test_custom", baseline_metrics)
        
        # Create current metrics with moderate regression
        current_metrics = PerformanceMetrics(
            execution_time=1.15,  # 15% increase
            memory_usage_mb=115.0,  # 15% increase
            cpu_usage_percent=55.0,
            peak_memory_mb=172.5,  # 15% increase
            throughput=8.5  # 15% decrease
        )
        
        # Test regression detection with custom tolerance
        result = benchmark.validate_performance_regression("test_custom", current_metrics)
        
        # Should NOT detect regression (15% < 20% custom tolerance)
        assert result['regression_detected'] is False
        assert result['tolerances_used']['time_tolerance'] == 20.0
        assert result['tolerances_used']['memory_tolerance'] == 25.0
    
    @pytest.mark.fast
    def test_benchmark_framework_without_config(self):
        """Test that benchmark framework works without configuration system."""
        benchmark = PerformanceBenchmark(
            baseline_file=self.baseline_file,
            use_config=False
        )
        
        # Create test metrics
        baseline_metrics = PerformanceMetrics(
            execution_time=1.0,
            memory_usage_mb=100.0,
            cpu_usage_percent=50.0,
            peak_memory_mb=150.0,
            throughput=10.0
        )
        
        # Establish baseline
        benchmark.establish_baseline("test_no_config", baseline_metrics)
        
        # Create current metrics with regression
        current_metrics = PerformanceMetrics(
            execution_time=1.12,  # 12% increase
            memory_usage_mb=110.0,
            cpu_usage_percent=55.0,
            peak_memory_mb=165.0,
            throughput=9.0
        )
        
        # Test regression detection with explicit tolerance
        result = benchmark.validate_performance_regression(
            "test_no_config", 
            current_metrics, 
            tolerance_percent=15.0
        )
        
        # Should NOT detect regression (12% < 15% explicit tolerance)
        assert result['regression_detected'] is False
        assert result['tolerances_used']['time_tolerance'] == 15.0
    
    @pytest.mark.fast
    def test_tolerance_info_retrieval(self):
        """Test that tolerance information can be retrieved from benchmark."""
        benchmark = PerformanceBenchmark(
            baseline_file=self.baseline_file,
            use_config=True
        )
        
        # Get tolerance info for a configured test
        tolerance_info = benchmark.get_tolerance_info("video_loading_30s")
        
        assert 'execution_time_tolerance' in tolerance_info
        assert 'memory_usage_tolerance' in tolerance_info
        assert 'throughput_tolerance' in tolerance_info
        
        # Should use configured values
        assert tolerance_info['execution_time_tolerance'] == 15.0
        assert tolerance_info['memory_usage_tolerance'] == 20.0
    
    @pytest.mark.fast
    def test_performance_report_includes_tolerance_config(self):
        """Test that performance report includes tolerance configuration."""
        benchmark = PerformanceBenchmark(
            baseline_file=self.baseline_file,
            use_config=True
        )
        
        # Add some baseline metrics
        test_metrics = PerformanceMetrics(
            execution_time=1.0,
            memory_usage_mb=100.0,
            cpu_usage_percent=50.0,
            peak_memory_mb=150.0,
            throughput=10.0
        )
        benchmark.establish_baseline("test_report", test_metrics)
        
        # Generate report
        report = benchmark.generate_performance_report()
        
        # Should include tolerance configuration
        assert 'tolerance_configuration' in report
        assert 'default_tolerances' in report['tolerance_configuration']
        assert 'test_specific_tolerances' in report['tolerance_configuration']
        assert 'environment_multipliers' in report['tolerance_configuration']
    
    @pytest.mark.fast
    def test_convenience_functions(self):
        """Test convenience functions for tolerance configuration."""
        # Test configure_test_tolerance function
        with patch('tests.performance.performance_config._performance_config', None):
            # This should create a new config instance
            configure_test_tolerance("convenience_test", execution_time=30.0)
            
            # Verify the configuration was set
            from tests.performance.performance_config import get_test_tolerance
            tolerance = get_test_tolerance("convenience_test", "execution_time")
            assert tolerance == 30.0
    
    @pytest.mark.fast
    def test_invalid_metric_type_handling(self):
        """Test handling of invalid metric types."""
        config = PerformanceConfig(self.config_file)
        
        # Should fall back to default for unknown metric type
        tolerance = config.get_tolerance("test", "unknown_metric")
        assert tolerance == 10.0  # Default fallback
    
    @pytest.mark.fast
    def test_config_file_corruption_handling(self):
        """Test handling of corrupted configuration files."""
        # Create corrupted config file
        with open(self.config_file, 'w') as f:
            f.write("invalid json content")
        
        # Should handle corruption gracefully and create default config
        config = PerformanceConfig(self.config_file)
        
        # Should have default values
        assert config.tolerance_config.execution_time_tolerance == 10.0
        
        # Should have recreated the config file with valid JSON
        assert self.config_file.exists()
        with open(self.config_file, 'r') as f:
            data = json.load(f)  # Should not raise exception
            assert 'tolerance' in data


@pytest.mark.integration
class TestConfigurableToleranceIntegration:
    """Integration tests for configurable tolerance system."""
    
    def setup_method(self):
        """Set up integration test environment."""
        self.temp_dir = Path(tempfile.mkdtemp())
        self.config_file = self.temp_dir / "integration_config.json"
        self.baseline_file = self.temp_dir / "integration_baselines.json"
    
    def teardown_method(self):
        """Clean up integration test environment."""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    @pytest.mark.slow
    def test_end_to_end_tolerance_configuration(self):
        """Test complete end-to-end tolerance configuration workflow."""
        
        # Step 1: Create benchmark with configuration
        benchmark = PerformanceBenchmark(
            baseline_file=self.baseline_file,
            use_config=True
        )
        
        # Step 2: Configure custom tolerance for a test
        benchmark.configure_test_tolerance(
            "integration_test",
            execution_time=40.0,  # Higher tolerance to account for timing variations
            memory_usage=30.0,
            throughput=30.0  # Higher tolerance for throughput variations
        )
        
        # Step 3: Run a mock performance test
        def mock_performance_operation():
            """Mock operation that takes some time and memory."""
            import time
            time.sleep(0.1)  # Simulate work
            return "completed"
        
        # Benchmark the operation
        metrics = benchmark.benchmark_function(mock_performance_operation, iterations=3)
        
        # Step 4: Establish baseline
        benchmark.establish_baseline("integration_test", metrics)
        
        # Step 5: Run again with slightly different performance
        def slower_mock_operation():
            """Slightly slower mock operation."""
            import time
            time.sleep(0.12)  # 20% slower
            return "completed"
        
        new_metrics = benchmark.benchmark_function(slower_mock_operation, iterations=3)
        
        # Step 6: Check regression with configured tolerance
        result = benchmark.validate_performance_regression("integration_test", new_metrics)
        
        # Should not detect regression due to high configured tolerance (40%)
        assert result['regression_detected'] is False
        assert result['tolerances_used']['time_tolerance'] == 40.0
        
        # Step 7: Generate comprehensive report
        report = benchmark.generate_performance_report()
        
        # Verify report contains all expected information
        assert 'tolerance_configuration' in report
        assert 'integration_test' in report['tolerance_configuration']['test_specific_tolerances']
        assert report['tolerance_configuration']['test_specific_tolerances']['integration_test']['execution_time_tolerance'] == 40.0