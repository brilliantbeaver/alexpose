"""API load testing and performance validation."""

import pytest
import time
import asyncio
import statistics
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, List, Any

try:
    import aiohttp
    AIOHTTP_AVAILABLE = True
except ImportError:
    AIOHTTP_AVAILABLE = False

from tests.performance.benchmark_framework import PerformanceBenchmark


@pytest.mark.performance
@pytest.mark.integration
class TestAPILoadPerformance:
    """Test API performance under various load conditions."""
    
    def setup_method(self):
        """Set up performance testing framework."""
        self.benchmark = PerformanceBenchmark()
        self.base_url = "http://localhost:8000"  # Adjust based on test environment
    
    def mock_api_request(self, endpoint: str, method: str = "GET", data: Dict = None) -> Dict[str, Any]:
        """Mock API request for testing without actual server."""
        # Simulate API response time based on endpoint
        response_times = {
            "/health": 0.05,      # 50ms for health check
            "/status": 0.1,       # 100ms for status
            "/upload": 0.5,       # 500ms for upload
            "/analyze": 2.0,      # 2s for analysis
            "/results": 0.2       # 200ms for results
        }
        
        base_time = response_times.get(endpoint, 0.1)
        
        # Add some variance
        import random
        actual_time = base_time * random.uniform(0.8, 1.2)
        time.sleep(actual_time)
        
        # Simulate different response types
        if endpoint == "/health":
            return {"status": "healthy", "timestamp": time.time()}
        elif endpoint == "/status":
            return {"status": "running", "active_analyses": random.randint(0, 5)}
        elif endpoint == "/upload":
            return {"upload_id": f"upload_{random.randint(1000, 9999)}", "status": "uploaded"}
        elif endpoint == "/analyze":
            return {"analysis_id": f"analysis_{random.randint(1000, 9999)}", "status": "completed"}
        elif endpoint == "/results":
            return {"results": {"classification": "normal", "confidence": 0.85}}
        else:
            return {"message": "success"}
    
    def test_health_endpoint_performance(self):
        """Test health endpoint performance (target: <200ms)."""
        
        def health_check():
            return self.mock_api_request("/health")
        
        # Test single request performance
        metrics = self.benchmark.benchmark_function(health_check, iterations=10)
        
        # Health endpoint should be very fast
        avg_time_per_request = metrics.execution_time / 10
        assert avg_time_per_request < 0.2, \
            f"Health endpoint average response time: {avg_time_per_request:.3f}s, target: <0.2s"
        
        # Check for regression using configurable tolerance
        regression_result = self.benchmark.validate_performance_regression(
            "health_endpoint", metrics
        )
        
        if regression_result['regression_detected'] and not regression_result.get('baseline_established', False):
            pytest.fail(f"Health endpoint regression: {regression_result['regressions']}")
        elif regression_result.get('baseline_established', False):
            print(f"Baseline established for health endpoint: {metrics.execution_time:.3f}s")
    
    def test_status_endpoint_response_time_detailed(self):
        """Detailed response time testing for status endpoints (target: <200ms)."""
        status_endpoints = ["/health", "/status", "/api/v1/health", "/api/v1/status"]
        
        for endpoint in status_endpoints:
            # Test multiple iterations for statistical significance
            response_times = []
            
            for _ in range(50):  # 50 requests for better statistics
                start_time = time.time()
                response = self.mock_api_request(endpoint)
                end_time = time.time()
                response_times.append(end_time - start_time)
            
            # Calculate statistics
            avg_time = statistics.mean(response_times)
            median_time = statistics.median(response_times)
            p95_time = sorted(response_times)[int(0.95 * len(response_times))]
            p99_time = sorted(response_times)[int(0.99 * len(response_times))]
            
            # Validate response time targets
            assert avg_time < 0.2, \
                f"{endpoint} average response time: {avg_time:.3f}s, target: <0.2s"
            
            assert p95_time < 0.25, \
                f"{endpoint} P95 response time: {p95_time:.3f}s, target: <0.25s"
            
            assert p99_time < 0.3, \
                f"{endpoint} P99 response time: {p99_time:.3f}s, target: <0.3s"
            
            # Log detailed metrics
            print(f"\n{endpoint} Response Time Metrics:")
            print(f"  Average: {avg_time:.3f}s")
            print(f"  Median: {median_time:.3f}s")
            print(f"  P95: {p95_time:.3f}s")
            print(f"  P99: {p99_time:.3f}s")
            print(f"  Min: {min(response_times):.3f}s")
            print(f"  Max: {max(response_times):.3f}s")
    
    def test_concurrent_api_requests(self):
        """Test API performance under concurrent load."""
        
        def make_concurrent_requests(num_requests: int, endpoint: str):
            """Make concurrent requests to an endpoint."""
            start_time = time.time()
            
            with ThreadPoolExecutor(max_workers=min(num_requests, 20)) as executor:
                futures = [
                    executor.submit(self.mock_api_request, endpoint) 
                    for _ in range(num_requests)
                ]
                
                results = []
                response_times = []
                
                for future in as_completed(futures):
                    request_start = time.time()
                    try:
                        result = future.result()
                        results.append(result)
                        response_times.append(time.time() - request_start)
                    except Exception as e:
                        results.append({"error": str(e)})
            
            total_time = time.time() - start_time
            
            return {
                'total_requests': num_requests,
                'successful_requests': len([r for r in results if 'error' not in r]),
                'total_time': total_time,
                'average_response_time': statistics.mean(response_times) if response_times else 0,
                'median_response_time': statistics.median(response_times) if response_times else 0,
                'max_response_time': max(response_times) if response_times else 0,
                'throughput': num_requests / total_time
            }
        
        # Test different load levels
        load_tests = [
            (10, "/status"),
            (50, "/status"),
            (100, "/health"),
            (20, "/results")
        ]
        
        for num_requests, endpoint in load_tests:
            result = make_concurrent_requests(num_requests, endpoint)
            
            # Validate performance targets
            success_rate = result['successful_requests'] / result['total_requests']
            assert success_rate >= 0.95, \
                f"{endpoint} success rate: {success_rate:.2f} with {num_requests} requests, target: >=0.95"
            
            # Response time targets vary by endpoint
            if endpoint in ["/health", "/status"]:
                assert result['average_response_time'] < 0.2, \
                    f"{endpoint} avg response time: {result['average_response_time']:.3f}s, target: <0.2s"
            elif endpoint == "/results":
                assert result['average_response_time'] < 0.5, \
                    f"{endpoint} avg response time: {result['average_response_time']:.3f}s, target: <0.5s"
            
            # Throughput should be reasonable
            if endpoint in ["/health", "/status"]:
                assert result['throughput'] > 20, \
                    f"{endpoint} throughput: {result['throughput']:.1f} req/s, target: >20 req/s"
    
    @pytest.mark.asyncio
    @pytest.mark.skipif(not AIOHTTP_AVAILABLE, reason="aiohttp not available")
    async def test_async_api_load(self):
        """Test API load using async requests."""
        
        async def async_api_request(session, endpoint: str):
            """Make async API request (mocked)."""
            # Simulate async API call
            response_times = {
                "/health": 0.05,
                "/status": 0.1,
                "/results": 0.2
            }
            
            await asyncio.sleep(response_times.get(endpoint, 0.1))
            
            return {
                "endpoint": endpoint,
                "status": "success",
                "timestamp": time.time()
            }
        
        async def run_async_load_test(num_requests: int, endpoint: str):
            """Run async load test."""
            start_time = time.time()
            
            # Create session (mocked)
            session = None  # Would be aiohttp.ClientSession() in real test
            
            # Create tasks
            tasks = [
                async_api_request(session, endpoint) 
                for _ in range(num_requests)
            ]
            
            # Execute all requests concurrently
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            total_time = time.time() - start_time
            
            successful_results = [r for r in results if not isinstance(r, Exception)]
            
            return {
                'total_requests': num_requests,
                'successful_requests': len(successful_results),
                'total_time': total_time,
                'throughput': num_requests / total_time,
                'success_rate': len(successful_results) / num_requests
            }
        
        # Test async load
        result = await run_async_load_test(100, "/health")
        
        # Async should handle high concurrency well
        assert result['success_rate'] >= 0.95, \
            f"Async success rate: {result['success_rate']:.2f}, target: >=0.95"
        
        assert result['throughput'] > 50, \
            f"Async throughput: {result['throughput']:.1f} req/s, target: >50 req/s"
        
        assert result['total_time'] < 5.0, \
            f"100 async requests took {result['total_time']:.2f}s, target: <5s"
    
    def test_sustained_load_performance(self):
        """Test API performance under sustained load."""
        
        def sustained_load_test(duration_seconds: int, requests_per_second: int):
            """Run sustained load test."""
            start_time = time.time()
            end_time = start_time + duration_seconds
            
            total_requests = 0
            successful_requests = 0
            response_times = []
            
            while time.time() < end_time:
                batch_start = time.time()
                
                # Make batch of requests
                with ThreadPoolExecutor(max_workers=10) as executor:
                    futures = [
                        executor.submit(self.mock_api_request, "/status") 
                        for _ in range(requests_per_second)
                    ]
                    
                    for future in as_completed(futures):
                        request_start = time.time()
                        try:
                            result = future.result()
                            successful_requests += 1
                            response_times.append(time.time() - request_start)
                        except Exception:
                            pass
                        total_requests += 1
                
                # Wait for next batch
                batch_duration = time.time() - batch_start
                if batch_duration < 1.0:
                    time.sleep(1.0 - batch_duration)
            
            actual_duration = time.time() - start_time
            
            return {
                'duration': actual_duration,
                'total_requests': total_requests,
                'successful_requests': successful_requests,
                'success_rate': successful_requests / total_requests if total_requests > 0 else 0,
                'average_response_time': statistics.mean(response_times) if response_times else 0,
                'throughput': successful_requests / actual_duration
            }
        
        # Test sustained load (30 seconds at 10 req/s)
        result = sustained_load_test(duration_seconds=10, requests_per_second=5)  # Reduced for testing
        
        # Validate sustained performance
        assert result['success_rate'] >= 0.9, \
            f"Sustained load success rate: {result['success_rate']:.2f}, target: >=0.9"
        
        assert result['average_response_time'] < 0.3, \
            f"Sustained load avg response time: {result['average_response_time']:.3f}s, target: <0.3s"
        
        assert result['throughput'] > 3, \
            f"Sustained throughput: {result['throughput']:.1f} req/s, target: >3 req/s"
    
    def test_api_error_handling_under_load(self):
        """Test API error handling under load conditions."""
        
        def error_prone_request():
            """Request that may fail randomly."""
            import random
            
            if random.random() < 0.1:  # 10% failure rate
                raise Exception("Simulated API error")
            
            return self.mock_api_request("/status")
        
        # Test error handling under load
        concurrent_result = self.benchmark.benchmark_concurrent_operations(
            error_prone_request, num_concurrent=20
        )
        
        # Should handle errors gracefully
        assert concurrent_result['success_rate'] >= 0.8, \
            f"Error handling success rate: {concurrent_result['success_rate']:.2f}, target: >=0.8"
        
        # Should complete in reasonable time despite errors
        assert concurrent_result['total_time'] < 10.0, \
            f"Error handling test took {concurrent_result['total_time']:.2f}s, target: <10s"
    
    def test_api_rate_limiting_behavior(self):
        """Test API behavior under rate limiting conditions."""
        
        def rate_limited_request():
            """Request that simulates rate limiting."""
            import random
            
            # Simulate rate limiting (slower responses when overloaded)
            base_time = 0.1
            if random.random() < 0.3:  # 30% chance of rate limiting
                base_time = 1.0  # Slower response when rate limited
            
            time.sleep(base_time)
            return {"status": "success", "rate_limited": base_time > 0.5}
        
        # Test rate limiting behavior
        start_time = time.time()
        
        with ThreadPoolExecutor(max_workers=30) as executor:
            futures = [executor.submit(rate_limited_request) for _ in range(50)]
            results = []
            
            for future in as_completed(futures):
                try:
                    result = future.result()
                    results.append(result)
                except Exception as e:
                    results.append({"error": str(e)})
        
        total_time = time.time() - start_time
        
        # Analyze rate limiting behavior
        successful_results = [r for r in results if 'error' not in r]
        rate_limited_count = len([r for r in successful_results if r.get('rate_limited', False)])
        
        # Should handle rate limiting gracefully
        success_rate = len(successful_results) / len(results)
        assert success_rate >= 0.9, \
            f"Rate limiting success rate: {success_rate:.2f}, target: >=0.9"
        
        # Should complete even with rate limiting
        assert total_time < 30.0, \
            f"Rate limiting test took {total_time:.2f}s, target: <30s"
        
        # Some requests should be rate limited (validates test setup)
        rate_limited_percentage = rate_limited_count / len(successful_results) if successful_results else 0
        assert 0.1 <= rate_limited_percentage <= 0.5, \
            f"Rate limited percentage: {rate_limited_percentage:.2f}, expected: 0.1-0.5"

    def test_concurrent_upload_load_testing(self):
        """Test load testing for concurrent uploads (target: handle 10 concurrent uploads)."""
        
        def simulate_video_upload(upload_id: int, file_size_mb: float = 50.0):
            """Simulate video upload with realistic processing time."""
            import random
            
            # Simulate upload processing time based on file size
            # Base time: 1 second per 10MB + network overhead
            base_upload_time = (file_size_mb / 10.0) + random.uniform(0.5, 1.5)
            
            # Add some variance for network conditions
            network_factor = random.uniform(0.8, 1.3)
            actual_upload_time = base_upload_time * network_factor
            
            # Simulate potential upload failures (2% failure rate for more realistic testing)
            if random.random() < 0.02:
                time.sleep(actual_upload_time * 0.3)  # Partial upload before failure
                raise Exception(f"Upload {upload_id} failed: Network timeout")
            
            time.sleep(actual_upload_time)
            
            return {
                "upload_id": f"upload_{upload_id}_{random.randint(1000, 9999)}",
                "status": "uploaded",
                "file_size_mb": file_size_mb,
                "upload_time": actual_upload_time,
                "processing_time": random.uniform(2.0, 5.0)  # Additional processing time
            }
        
        # Test different concurrent upload scenarios
        upload_scenarios = [
            {"concurrent_uploads": 5, "file_size_mb": 30.0, "description": "5 concurrent small uploads"},
            {"concurrent_uploads": 10, "file_size_mb": 50.0, "description": "10 concurrent medium uploads"},
            {"concurrent_uploads": 15, "file_size_mb": 25.0, "description": "15 concurrent small uploads"},
            {"concurrent_uploads": 8, "file_size_mb": 100.0, "description": "8 concurrent large uploads"}
        ]
        
        for scenario in upload_scenarios:
            print(f"\nTesting scenario: {scenario['description']}")
            
            start_time = time.time()
            
            with ThreadPoolExecutor(max_workers=scenario['concurrent_uploads']) as executor:
                futures = [
                    executor.submit(simulate_video_upload, i, scenario['file_size_mb'])
                    for i in range(scenario['concurrent_uploads'])
                ]
                
                results = []
                upload_times = []
                
                for future in as_completed(futures):
                    try:
                        result = future.result(timeout=120)  # 2 minute timeout per upload
                        results.append(result)
                        upload_times.append(result['upload_time'])
                    except Exception as e:
                        results.append({"error": str(e)})
            
            total_time = time.time() - start_time
            successful_uploads = [r for r in results if 'error' not in r]
            failed_uploads = [r for r in results if 'error' in r]
            
            # Calculate metrics
            success_rate = len(successful_uploads) / len(results)
            avg_upload_time = statistics.mean(upload_times) if upload_times else 0
            max_upload_time = max(upload_times) if upload_times else 0
            throughput = len(successful_uploads) / total_time if total_time > 0 else 0
            
            # Validate performance targets (adjust success rate for large uploads)
            expected_success_rate = 0.85 if scenario['file_size_mb'] >= 100.0 else 0.9
            assert success_rate >= expected_success_rate, \
                f"{scenario['description']} success rate: {success_rate:.2f}, target: >={expected_success_rate}"
            
            # Upload time should be reasonable for file size
            expected_max_time = (scenario['file_size_mb'] / 5.0) + 10.0  # 5MB/s + 10s overhead
            assert avg_upload_time <= expected_max_time, \
                f"{scenario['description']} avg upload time: {avg_upload_time:.1f}s, target: <={expected_max_time:.1f}s"
            
            # Total time should be reasonable for concurrent uploads
            expected_total_time = expected_max_time * 1.5  # Allow 50% overhead for concurrency
            assert total_time <= expected_total_time, \
                f"{scenario['description']} total time: {total_time:.1f}s, target: <={expected_total_time:.1f}s"
            
            # Throughput should be reasonable
            min_throughput = 0.5  # At least 0.5 uploads per second
            assert throughput >= min_throughput, \
                f"{scenario['description']} throughput: {throughput:.2f} uploads/s, target: >={min_throughput}"
            
            # Log detailed metrics
            print(f"  Success rate: {success_rate:.2%}")
            print(f"  Successful uploads: {len(successful_uploads)}/{len(results)}")
            print(f"  Failed uploads: {len(failed_uploads)}")
            print(f"  Average upload time: {avg_upload_time:.1f}s")
            print(f"  Maximum upload time: {max_upload_time:.1f}s")
            print(f"  Total execution time: {total_time:.1f}s")
            print(f"  Throughput: {throughput:.2f} uploads/s")
            
            if failed_uploads:
                print(f"  Failed upload errors: {[r['error'] for r in failed_uploads[:3]]}")
    
    def test_upload_stress_testing(self):
        """Test upload system under stress conditions."""
        
        def stress_upload_simulation(upload_id: int):
            """Simulate upload under stress conditions."""
            import random
            
            # Simulate various stress conditions
            stress_conditions = [
                {"name": "network_congestion", "delay_factor": 2.0, "failure_rate": 0.1},
                {"name": "server_overload", "delay_factor": 3.0, "failure_rate": 0.15},
                {"name": "normal_load", "delay_factor": 1.0, "failure_rate": 0.05},
                {"name": "high_bandwidth", "delay_factor": 0.5, "failure_rate": 0.02}
            ]
            
            condition = random.choice(stress_conditions)
            
            # Base upload time
            base_time = random.uniform(2.0, 8.0)
            actual_time = base_time * condition["delay_factor"]
            
            # Check for failure
            if random.random() < condition["failure_rate"]:
                time.sleep(actual_time * 0.4)  # Partial processing before failure
                raise Exception(f"Upload {upload_id} failed under {condition['name']}")
            
            time.sleep(actual_time)
            
            return {
                "upload_id": f"stress_upload_{upload_id}",
                "status": "uploaded",
                "condition": condition["name"],
                "upload_time": actual_time,
                "stress_factor": condition["delay_factor"]
            }
        
        # Run stress test with high concurrency
        num_uploads = 25
        max_workers = 20
        
        start_time = time.time()
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = [
                executor.submit(stress_upload_simulation, i)
                for i in range(num_uploads)
            ]
            
            results = []
            for future in as_completed(futures):
                try:
                    result = future.result(timeout=60)  # 1 minute timeout
                    results.append(result)
                except Exception as e:
                    results.append({"error": str(e)})
        
        total_time = time.time() - start_time
        
        # Analyze stress test results
        successful_uploads = [r for r in results if 'error' not in r]
        failed_uploads = [r for r in results if 'error' in r]
        
        # Group by stress condition
        condition_stats = {}
        for result in successful_uploads:
            condition = result['condition']
            if condition not in condition_stats:
                condition_stats[condition] = []
            condition_stats[condition].append(result['upload_time'])
        
        success_rate = len(successful_uploads) / len(results)
        throughput = len(successful_uploads) / total_time
        
        # Validate stress test performance
        assert success_rate >= 0.75, \
            f"Stress test success rate: {success_rate:.2f}, target: >=0.75"
        
        assert total_time <= 120.0, \
            f"Stress test total time: {total_time:.1f}s, target: <=120s"
        
        assert throughput >= 0.2, \
            f"Stress test throughput: {throughput:.2f} uploads/s, target: >=0.2"
        
        # Log stress test results
        print(f"\nStress Test Results:")
        print(f"  Total uploads: {num_uploads}")
        print(f"  Successful: {len(successful_uploads)}")
        print(f"  Failed: {len(failed_uploads)}")
        print(f"  Success rate: {success_rate:.2%}")
        print(f"  Total time: {total_time:.1f}s")
        print(f"  Throughput: {throughput:.2f} uploads/s")
        
        # Log condition-specific stats
        for condition, times in condition_stats.items():
            avg_time = statistics.mean(times)
            print(f"  {condition}: {len(times)} uploads, avg time: {avg_time:.1f}s")
    
    def test_upload_memory_usage_under_load(self):
        """Test memory usage during concurrent uploads."""
        import psutil
        
        def memory_intensive_upload(upload_id: int, file_size_mb: float = 100.0):
            """Simulate memory-intensive upload processing."""
            import random
            
            # Simulate memory allocation for file processing
            # Create data structure to simulate file buffer
            buffer_size = int(file_size_mb * 1024 * 1024 * 0.1)  # 10% of file size in memory
            buffer = bytearray(buffer_size)
            
            # Simulate processing time
            processing_time = random.uniform(3.0, 8.0)
            time.sleep(processing_time)
            
            # Simulate some processing on the buffer
            for i in range(0, min(1000, len(buffer)), 100):
                buffer[i] = random.randint(0, 255)
            
            result = {
                "upload_id": f"memory_upload_{upload_id}",
                "status": "uploaded",
                "file_size_mb": file_size_mb,
                "buffer_size_mb": buffer_size / (1024 * 1024),
                "processing_time": processing_time
            }
            
            # Clean up buffer
            del buffer
            
            return result
        
        # Monitor memory usage
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
                    time.sleep(0.5)
                except psutil.NoSuchProcess:
                    break
        
        # Start memory monitoring
        monitoring = True
        monitor_thread = threading.Thread(target=monitor_memory, daemon=True)
        monitor_thread.start()
        
        try:
            # Run concurrent uploads with memory monitoring
            num_uploads = 10
            file_size_mb = 75.0
            
            start_time = time.time()
            
            with ThreadPoolExecutor(max_workers=8) as executor:
                futures = [
                    executor.submit(memory_intensive_upload, i, file_size_mb)
                    for i in range(num_uploads)
                ]
                
                results = []
                for future in as_completed(futures):
                    try:
                        result = future.result(timeout=30)
                        results.append(result)
                    except Exception as e:
                        results.append({"error": str(e)})
            
            total_time = time.time() - start_time
            
        finally:
            # Stop memory monitoring
            monitoring = False
            monitor_thread.join(timeout=2.0)
        
        final_memory = process.memory_info().rss / 1024 / 1024
        memory_increase = peak_memory - initial_memory
        avg_memory = statistics.mean(memory_samples) if memory_samples else initial_memory
        
        # Analyze results
        successful_uploads = [r for r in results if 'error' not in r]
        success_rate = len(successful_uploads) / len(results)
        
        # Validate memory usage
        max_allowed_memory_increase = 1000.0  # 1GB increase maximum
        assert memory_increase <= max_allowed_memory_increase, \
            f"Memory increase: {memory_increase:.1f}MB, target: <={max_allowed_memory_increase}MB"
        
        assert peak_memory <= 2048.0, \
            f"Peak memory usage: {peak_memory:.1f}MB, target: <=2048MB"
        
        assert success_rate >= 0.9, \
            f"Memory test success rate: {success_rate:.2f}, target: >=0.9"
        
        # Log memory usage results
        print(f"\nMemory Usage Test Results:")
        print(f"  Initial memory: {initial_memory:.1f}MB")
        print(f"  Peak memory: {peak_memory:.1f}MB")
        print(f"  Final memory: {final_memory:.1f}MB")
        print(f"  Memory increase: {memory_increase:.1f}MB")
        print(f"  Average memory: {avg_memory:.1f}MB")
        print(f"  Successful uploads: {len(successful_uploads)}/{len(results)}")
        print(f"  Total time: {total_time:.1f}s")
    
    def test_upload_bandwidth_simulation(self):
        """Test upload performance under different bandwidth conditions."""
        
        def bandwidth_limited_upload(upload_id: int, bandwidth_mbps: float, file_size_mb: float):
            """Simulate upload with bandwidth limitations."""
            import random
            
            # Calculate theoretical upload time based on bandwidth
            theoretical_time = (file_size_mb * 8) / bandwidth_mbps  # Convert MB to Mbits
            
            # Add realistic overhead (protocol overhead, network variance)
            overhead_factor = random.uniform(1.2, 1.8)
            actual_upload_time = theoretical_time * overhead_factor
            
            # Add some network jitter
            jitter = random.uniform(-0.1, 0.1) * actual_upload_time
            final_upload_time = max(0.1, actual_upload_time + jitter)
            
            time.sleep(final_upload_time)
            
            return {
                "upload_id": f"bandwidth_upload_{upload_id}",
                "status": "uploaded",
                "file_size_mb": file_size_mb,
                "bandwidth_mbps": bandwidth_mbps,
                "theoretical_time": theoretical_time,
                "actual_upload_time": final_upload_time,
                "efficiency": theoretical_time / final_upload_time
            }
        
        # Test different bandwidth scenarios
        bandwidth_scenarios = [
            {"bandwidth_mbps": 100.0, "file_size_mb": 50.0, "concurrent": 5, "description": "High bandwidth"},
            {"bandwidth_mbps": 50.0, "file_size_mb": 50.0, "concurrent": 8, "description": "Medium bandwidth"},
            {"bandwidth_mbps": 10.0, "file_size_mb": 30.0, "concurrent": 3, "description": "Low bandwidth"},
            {"bandwidth_mbps": 5.0, "file_size_mb": 20.0, "concurrent": 2, "description": "Very low bandwidth"}
        ]
        
        for scenario in bandwidth_scenarios:
            print(f"\nTesting bandwidth scenario: {scenario['description']}")
            
            start_time = time.time()
            
            with ThreadPoolExecutor(max_workers=scenario['concurrent']) as executor:
                futures = [
                    executor.submit(
                        bandwidth_limited_upload, 
                        i, 
                        scenario['bandwidth_mbps'], 
                        scenario['file_size_mb']
                    )
                    for i in range(scenario['concurrent'])
                ]
                
                results = []
                for future in as_completed(futures):
                    try:
                        result = future.result(timeout=300)  # 5 minute timeout
                        results.append(result)
                    except Exception as e:
                        results.append({"error": str(e)})
            
            total_time = time.time() - start_time
            
            # Analyze bandwidth test results
            successful_uploads = [r for r in results if 'error' not in r]
            success_rate = len(successful_uploads) / len(results)
            
            if successful_uploads:
                avg_upload_time = statistics.mean([r['actual_upload_time'] for r in successful_uploads])
                avg_efficiency = statistics.mean([r['efficiency'] for r in successful_uploads])
                total_data_mb = sum([r['file_size_mb'] for r in successful_uploads])
                effective_throughput = total_data_mb / total_time if total_time > 0 else 0
            else:
                avg_upload_time = 0
                avg_efficiency = 0
                effective_throughput = 0
            
            # Validate bandwidth performance
            assert success_rate >= 0.95, \
                f"{scenario['description']} success rate: {success_rate:.2f}, target: >=0.95"
            
            # Efficiency should be reasonable (at least 50% of theoretical)
            assert avg_efficiency >= 0.5, \
                f"{scenario['description']} efficiency: {avg_efficiency:.2f}, target: >=0.5"
            
            # Log bandwidth test results
            print(f"  Success rate: {success_rate:.2%}")
            print(f"  Average upload time: {avg_upload_time:.1f}s")
            print(f"  Average efficiency: {avg_efficiency:.2%}")
            print(f"  Effective throughput: {effective_throughput:.1f} MB/s")
            print(f"  Total time: {total_time:.1f}s")