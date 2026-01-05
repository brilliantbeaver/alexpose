"""Integration test for API response time testing functionality."""

import pytest
from tests.performance.test_api_response_time import APIResponseTimeTester, ResponseTimeMetrics


@pytest.mark.performance
@pytest.mark.integration
class TestAPIResponseTimeIntegration:
    """Integration tests for API response time testing."""
    
    def test_api_response_time_tester_initialization(self):
        """Test that APIResponseTimeTester initializes correctly."""
        tester = APIResponseTimeTester()
        
        assert tester.base_url == "http://localhost:8000"
        assert tester.benchmark is not None
        assert tester.session is None
    
    def test_response_time_metrics_calculation(self):
        """Test ResponseTimeMetrics calculations."""
        response_times = [0.1, 0.15, 0.12, 0.18, 0.09, 0.2, 0.11, 0.16, 0.13, 0.14]
        
        metrics = ResponseTimeMetrics(
            endpoint="/test",
            method="GET",
            response_times=response_times,
            success_count=10,
            error_count=0
        )
        
        # Test calculations
        assert abs(metrics.average_response_time - 0.138) < 0.01
        assert metrics.success_rate == 1.0
        assert metrics.p95_response_time <= 0.2
        assert metrics.p99_response_time <= 0.2
    
    def test_mock_api_request_functionality(self):
        """Test that mock API requests work correctly."""
        tester = APIResponseTimeTester()
        
        # Test different endpoints
        endpoints = ["/health", "/status", "/api/v1/health", "/api/v1/status"]
        
        for endpoint in endpoints:
            response = tester.mock_api_request(endpoint)
            
            assert isinstance(response, dict)
            assert "status" in response or "message" in response
            
            # Test response time measurement
            response_time = tester.measure_single_request(endpoint)
            assert response_time > 0
            assert response_time < 1.0  # Should be reasonable
    
    def test_multiple_requests_measurement(self):
        """Test measuring multiple requests."""
        tester = APIResponseTimeTester()
        
        metrics = tester.measure_multiple_requests("/health", num_requests=10)
        
        assert metrics.endpoint == "/health"
        assert metrics.method == "GET"
        assert len(metrics.response_times) == 10
        assert metrics.success_count == 10
        assert metrics.error_count == 0
        assert metrics.success_rate == 1.0
        assert metrics.average_response_time > 0
    
    def test_concurrent_requests_measurement(self):
        """Test measuring concurrent requests."""
        tester = APIResponseTimeTester()
        
        metrics = tester.measure_concurrent_requests(
            "/health", 
            num_concurrent=5, 
            total_requests=20
        )
        
        assert metrics.endpoint == "/health"
        assert len(metrics.response_times) == 20
        assert metrics.success_count == 20
        assert metrics.error_count == 0
        assert metrics.success_rate == 1.0
    
    def test_api_response_time_target_validation(self):
        """Test that API response time targets are properly validated."""
        tester = APIResponseTimeTester()
        
        # Test status endpoints meet target
        status_endpoints = ["/health", "/status"]
        target_time = 0.2  # 200ms
        
        for endpoint in status_endpoints:
            metrics = tester.measure_multiple_requests(endpoint, num_requests=20)
            
            # Should meet target
            assert metrics.average_response_time < target_time, \
                f"{endpoint} average response time: {metrics.average_response_time:.3f}s, target: <{target_time}s"
            
            # Should have good success rate
            assert metrics.success_rate >= 0.95, \
                f"{endpoint} success rate: {metrics.success_rate:.2f}, target: >=0.95"
    
    def test_performance_framework_integration(self):
        """Test integration with performance benchmarking framework."""
        tester = APIResponseTimeTester()
        
        # Test that benchmark framework is properly integrated
        assert hasattr(tester, 'benchmark')
        assert hasattr(tester.benchmark, 'validate_performance_regression')
        assert hasattr(tester.benchmark, 'establish_baseline')
        
        # Test regression detection functionality
        from tests.performance.benchmark_framework import PerformanceMetrics
        
        test_metrics = PerformanceMetrics(
            execution_time=0.1,
            memory_usage_mb=0.0,
            cpu_usage_percent=0.0,
            peak_memory_mb=0.0,
            throughput=10.0
        )
        
        regression_result = tester.benchmark.validate_performance_regression(
            "test_api_integration", 
            test_metrics
        )
        
        # Should establish baseline on first run
        assert regression_result.get('baseline_established', False) or not regression_result['regression_detected']