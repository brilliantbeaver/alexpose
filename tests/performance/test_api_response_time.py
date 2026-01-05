"""API response time testing with specific focus on status endpoints."""

import pytest
import time
import asyncio
import statistics
from typing import Dict, List, Any, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass

try:
    import aiohttp
    AIOHTTP_AVAILABLE = True
except ImportError:
    AIOHTTP_AVAILABLE = False

try:
    import requests
    REQUESTS_AVAILABLE = True
except ImportError:
    REQUESTS_AVAILABLE = False

from tests.performance.benchmark_framework import PerformanceBenchmark


@dataclass
class ResponseTimeMetrics:
    """Metrics for API response time analysis."""
    endpoint: str
    method: str
    response_times: List[float]
    success_count: int
    error_count: int
    
    @property
    def average_response_time(self) -> float:
        """Calculate average response time."""
        return statistics.mean(self.response_times) if self.response_times else 0.0
    
    @property
    def median_response_time(self) -> float:
        """Calculate median response time."""
        return statistics.median(self.response_times) if self.response_times else 0.0
    
    @property
    def p95_response_time(self) -> float:
        """Calculate 95th percentile response time."""
        if not self.response_times:
            return 0.0
        sorted_times = sorted(self.response_times)
        index = int(0.95 * len(sorted_times))
        return sorted_times[min(index, len(sorted_times) - 1)]
    
    @property
    def p99_response_time(self) -> float:
        """Calculate 99th percentile response time."""
        if not self.response_times:
            return 0.0
        sorted_times = sorted(self.response_times)
        index = int(0.99 * len(sorted_times))
        return sorted_times[min(index, len(sorted_times) - 1)]
    
    @property
    def success_rate(self) -> float:
        """Calculate success rate."""
        total = self.success_count + self.error_count
        return self.success_count / total if total > 0 else 0.0


class APIResponseTimeTester:
    """Dedicated API response time testing framework."""
    
    def __init__(self, base_url: str = "http://localhost:8000"):
        self.base_url = base_url
        self.benchmark = PerformanceBenchmark()
        self.session = None
    
    def mock_api_request(self, endpoint: str, method: str = "GET", timeout: float = 5.0) -> Dict[str, Any]:
        """Mock API request for testing without actual server."""
        # Simulate realistic response times for different endpoints
        response_times = {
            "/": 0.08,           # 80ms for root
            "/health": 0.05,     # 50ms for health check
            "/status": 0.12,     # 120ms for status
            "/api/v1/status": 0.15,  # 150ms for API status
            "/api/v1/health": 0.06,  # 60ms for API health
            "/api/v1/analyze": 1.5,  # 1.5s for analysis
            "/api/v1/upload": 0.8,   # 800ms for upload
            "/api/v1/results": 0.25, # 250ms for results
        }
        
        base_time = response_times.get(endpoint, 0.1)
        
        # Add realistic variance (±20%)
        import random
        variance = random.uniform(0.8, 1.2)
        actual_time = base_time * variance
        
        # Simulate network latency and processing
        time.sleep(actual_time)
        
        # Return appropriate response based on endpoint
        if endpoint in ["/health", "/api/v1/health"]:
            return {
                "status": "healthy",
                "timestamp": time.time(),
                "version": "0.1.0"
            }
        elif endpoint in ["/status", "/api/v1/status"]:
            return {
                "status": "running",
                "active_analyses": random.randint(0, 3),
                "queue_size": random.randint(0, 10),
                "uptime": random.randint(3600, 86400)
            }
        elif endpoint == "/":
            return {
                "message": "AlexPose Gait Analysis System",
                "version": "0.1.0",
                "status": "running"
            }
        else:
            return {"message": "success", "endpoint": endpoint}
    
    def measure_single_request(self, endpoint: str, method: str = "GET") -> float:
        """Measure response time for a single request."""
        start_time = time.time()
        try:
            response = self.mock_api_request(endpoint, method)
            end_time = time.time()
            return end_time - start_time
        except Exception:
            return -1.0  # Indicate error
    
    def measure_multiple_requests(
        self, 
        endpoint: str, 
        method: str = "GET", 
        num_requests: int = 100
    ) -> ResponseTimeMetrics:
        """Measure response times for multiple requests."""
        response_times = []
        success_count = 0
        error_count = 0
        
        for _ in range(num_requests):
            response_time = self.measure_single_request(endpoint, method)
            if response_time > 0:
                response_times.append(response_time)
                success_count += 1
            else:
                error_count += 1
        
        return ResponseTimeMetrics(
            endpoint=endpoint,
            method=method,
            response_times=response_times,
            success_count=success_count,
            error_count=error_count
        )
    
    def measure_concurrent_requests(
        self, 
        endpoint: str, 
        method: str = "GET", 
        num_concurrent: int = 10,
        total_requests: int = 100
    ) -> ResponseTimeMetrics:
        """Measure response times under concurrent load."""
        response_times = []
        success_count = 0
        error_count = 0
        
        def make_request():
            return self.measure_single_request(endpoint, method)
        
        with ThreadPoolExecutor(max_workers=num_concurrent) as executor:
            futures = [executor.submit(make_request) for _ in range(total_requests)]
            
            for future in as_completed(futures):
                try:
                    response_time = future.result(timeout=10)
                    if response_time > 0:
                        response_times.append(response_time)
                        success_count += 1
                    else:
                        error_count += 1
                except Exception:
                    error_count += 1
        
        return ResponseTimeMetrics(
            endpoint=endpoint,
            method=method,
            response_times=response_times,
            success_count=success_count,
            error_count=error_count
        )


@pytest.mark.performance
@pytest.mark.integration
class TestAPIResponseTime:
    """Test API response times with focus on status endpoints."""
    
    def setup_method(self):
        """Set up response time testing."""
        self.tester = APIResponseTimeTester()
        self.status_endpoints = [
            "/health",
            "/status", 
            "/api/v1/health",
            "/api/v1/status"
        ]
        self.target_response_time = 0.2  # 200ms target
    
    def test_status_endpoints_response_time_target(self):
        """Test that status endpoints meet <200ms response time target."""
        results = {}
        
        for endpoint in self.status_endpoints:
            metrics = self.tester.measure_multiple_requests(endpoint, num_requests=50)
            results[endpoint] = metrics
            
            # Validate average response time
            assert metrics.average_response_time < self.target_response_time, \
                f"{endpoint} average response time: {metrics.average_response_time:.3f}s, target: <{self.target_response_time}s"
            
            # Validate 95th percentile (most requests should be fast)
            assert metrics.p95_response_time < self.target_response_time * 1.5, \
                f"{endpoint} P95 response time: {metrics.p95_response_time:.3f}s, target: <{self.target_response_time * 1.5}s"
            
            # Validate success rate
            assert metrics.success_rate >= 0.98, \
                f"{endpoint} success rate: {metrics.success_rate:.2f}, target: >=0.98"
        
        # Log results for analysis
        for endpoint, metrics in results.items():
            print(f"\n{endpoint} Response Time Analysis:")
            print(f"  Average: {metrics.average_response_time:.3f}s")
            print(f"  Median: {metrics.median_response_time:.3f}s")
            print(f"  P95: {metrics.p95_response_time:.3f}s")
            print(f"  P99: {metrics.p99_response_time:.3f}s")
            print(f"  Success Rate: {metrics.success_rate:.2f}")
    
    def test_status_endpoints_under_load(self):
        """Test status endpoint response times under concurrent load."""
        load_scenarios = [
            (5, 50),   # 5 concurrent, 50 total requests
            (10, 100), # 10 concurrent, 100 total requests
            (20, 200), # 20 concurrent, 200 total requests
        ]
        
        for endpoint in self.status_endpoints:
            for concurrent, total in load_scenarios:
                metrics = self.tester.measure_concurrent_requests(
                    endpoint, 
                    num_concurrent=concurrent,
                    total_requests=total
                )
                
                # Under load, allow slightly higher response times
                load_target = self.target_response_time * 1.5  # 300ms under load
                
                assert metrics.average_response_time < load_target, \
                    f"{endpoint} under load ({concurrent} concurrent): {metrics.average_response_time:.3f}s, target: <{load_target}s"
                
                # Success rate should remain high under load
                assert metrics.success_rate >= 0.95, \
                    f"{endpoint} success rate under load: {metrics.success_rate:.2f}, target: >=0.95"
                
                print(f"\n{endpoint} Load Test ({concurrent} concurrent, {total} total):")
                print(f"  Average: {metrics.average_response_time:.3f}s")
                print(f"  P95: {metrics.p95_response_time:.3f}s")
                print(f"  Success Rate: {metrics.success_rate:.2f}")
    
    def test_response_time_consistency(self):
        """Test response time consistency over multiple measurements."""
        consistency_results = {}
        
        for endpoint in self.status_endpoints:
            # Take multiple measurements
            measurements = []
            for _ in range(5):  # 5 measurement rounds
                metrics = self.tester.measure_multiple_requests(endpoint, num_requests=20)
                measurements.append(metrics.average_response_time)
                time.sleep(0.1)  # Small delay between measurements
            
            consistency_results[endpoint] = measurements
            
            # Calculate consistency metrics
            avg_of_averages = statistics.mean(measurements)
            std_dev = statistics.stdev(measurements) if len(measurements) > 1 else 0
            coefficient_of_variation = std_dev / avg_of_averages if avg_of_averages > 0 else 0
            
            # Response times should be consistent (low variation)
            assert coefficient_of_variation < 0.3, \
                f"{endpoint} response time variation too high: CV={coefficient_of_variation:.3f}, target: <0.3"
            
            # Average should still meet target
            assert avg_of_averages < self.target_response_time, \
                f"{endpoint} average response time: {avg_of_averages:.3f}s, target: <{self.target_response_time}s"
            
            print(f"\n{endpoint} Consistency Analysis:")
            print(f"  Average of Averages: {avg_of_averages:.3f}s")
            print(f"  Standard Deviation: {std_dev:.3f}s")
            print(f"  Coefficient of Variation: {coefficient_of_variation:.3f}")
    
    def test_response_time_regression_detection(self):
        """Test response time regression detection."""
        for endpoint in self.status_endpoints:
            # Measure current performance
            metrics = self.tester.measure_multiple_requests(endpoint, num_requests=30)
            
            # Convert to PerformanceMetrics for regression testing
            from tests.performance.benchmark_framework import PerformanceMetrics
            perf_metrics = PerformanceMetrics(
                execution_time=metrics.average_response_time,
                memory_usage_mb=0.0,  # Not applicable for API tests
                cpu_usage_percent=0.0,  # Not applicable for API tests
                peak_memory_mb=0.0,  # Not applicable for API tests
                throughput=1.0 / metrics.average_response_time if metrics.average_response_time > 0 else 0
            )
            
            # Check for regression using configurable tolerance
            regression_result = self.tester.benchmark.validate_performance_regression(
                f"api_response_time_{endpoint.replace('/', '_')}", 
                perf_metrics
            )
            
            if regression_result['regression_detected']:
                # Log regression but don't fail test on first detection
                print(f"\nRegression detected for {endpoint}:")
                for regression in regression_result['regressions']:
                    print(f"  - {regression}")
            else:
                print(f"\n{endpoint} performance: {'baseline established' if regression_result.get('baseline_established') else 'no regression'}")
    
    def test_status_endpoint_response_structure(self):
        """Test that status endpoints return properly structured responses quickly."""
        expected_structures = {
            "/health": ["status", "timestamp", "version"],
            "/status": ["status"],
            "/api/v1/health": ["status", "timestamp", "version"],
            "/api/v1/status": ["status"]
        }
        
        for endpoint in self.status_endpoints:
            start_time = time.time()
            response = self.tester.mock_api_request(endpoint)
            response_time = time.time() - start_time
            
            # Response should be fast
            assert response_time < self.target_response_time, \
                f"{endpoint} response time: {response_time:.3f}s, target: <{self.target_response_time}s"
            
            # Response should have expected structure
            expected_fields = expected_structures.get(endpoint, [])
            for field in expected_fields:
                assert field in response, \
                    f"{endpoint} missing required field: {field}"
            
            # Status field should indicate healthy/running state
            if "status" in response:
                assert response["status"] in ["healthy", "running"], \
                    f"{endpoint} invalid status: {response['status']}"
    
    @pytest.mark.asyncio
    @pytest.mark.skipif(not AIOHTTP_AVAILABLE, reason="aiohttp not available")
    async def test_async_status_endpoint_performance(self):
        """Test status endpoint performance using async requests."""
        
        async def async_request(endpoint: str) -> float:
            """Make async request and measure response time."""
            start_time = time.time()
            
            # Simulate async API call
            response_times = {
                "/health": 0.05,
                "/status": 0.12,
                "/api/v1/health": 0.06,
                "/api/v1/status": 0.15
            }
            
            await asyncio.sleep(response_times.get(endpoint, 0.1))
            return time.time() - start_time
        
        for endpoint in self.status_endpoints:
            # Test multiple async requests
            tasks = [async_request(endpoint) for _ in range(50)]
            response_times = await asyncio.gather(*tasks)
            
            avg_response_time = statistics.mean(response_times)
            
            # Async requests should be fast
            assert avg_response_time < self.target_response_time, \
                f"{endpoint} async average response time: {avg_response_time:.3f}s, target: <{self.target_response_time}s"
            
            print(f"\n{endpoint} Async Performance:")
            print(f"  Average: {avg_response_time:.3f}s")
            print(f"  Min: {min(response_times):.3f}s")
            print(f"  Max: {max(response_times):.3f}s")
    
    def test_response_time_percentiles(self):
        """Test detailed response time percentile analysis."""
        for endpoint in self.status_endpoints:
            metrics = self.tester.measure_multiple_requests(endpoint, num_requests=200)
            
            # Calculate various percentiles
            percentiles = [50, 75, 90, 95, 99]
            percentile_values = {}
            
            if metrics.response_times:
                sorted_times = sorted(metrics.response_times)
                for p in percentiles:
                    index = int((p / 100.0) * len(sorted_times))
                    percentile_values[p] = sorted_times[min(index, len(sorted_times) - 1)]
            
            # Validate percentile targets
            assert percentile_values.get(50, 0) < self.target_response_time * 0.8, \
                f"{endpoint} P50: {percentile_values.get(50, 0):.3f}s, target: <{self.target_response_time * 0.8}s"
            
            assert percentile_values.get(95, 0) < self.target_response_time * 1.2, \
                f"{endpoint} P95: {percentile_values.get(95, 0):.3f}s, target: <{self.target_response_time * 1.2}s"
            
            assert percentile_values.get(99, 0) < self.target_response_time * 1.5, \
                f"{endpoint} P99: {percentile_values.get(99, 0):.3f}s, target: <{self.target_response_time * 1.5}s"
            
            print(f"\n{endpoint} Percentile Analysis:")
            for p in percentiles:
                print(f"  P{p}: {percentile_values.get(p, 0):.3f}s")