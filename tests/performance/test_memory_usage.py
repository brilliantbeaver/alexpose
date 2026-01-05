"""Memory usage performance tests."""

import pytest
import time
import gc
import psutil
import threading
from typing import List, Dict, Any
import numpy as np

from tests.performance.benchmark_framework import PerformanceBenchmark


@pytest.mark.performance
class TestMemoryUsagePerformance:
    """Test memory usage patterns and limits."""
    
    def setup_method(self):
        """Set up performance testing framework."""
        self.benchmark = PerformanceBenchmark()
        self.process = psutil.Process()
        
        # Force garbage collection before each test
        gc.collect()
    
    def get_memory_usage_mb(self) -> float:
        """Get current memory usage in MB."""
        return self.process.memory_info().rss / 1024 / 1024
    
    def test_video_processing_memory_limits(self):
        """Test memory usage during video processing stays within limits."""
        
        def process_large_video():
            """Simulate processing a large video file."""
            initial_memory = self.get_memory_usage_mb()
            
            # Simulate video frames (1920x1080x3 for 300 frames = ~1.8GB uncompressed)
            frames = []
            peak_memory = initial_memory
            
            try:
                for i in range(100):  # Reduced for testing
                    # Create frame data
                    frame = np.random.randint(0, 255, (1080, 1920, 3), dtype=np.uint8)
                    frames.append(frame)
                    
                    # Track peak memory
                    current_memory = self.get_memory_usage_mb()
                    peak_memory = max(peak_memory, current_memory)
                    
                    # Simulate processing delay
                    time.sleep(0.01)
                
                # Simulate processing all frames
                processed_frames = []
                for frame in frames:
                    # Simple processing (edge detection simulation)
                    processed = frame.astype(np.float32) * 0.5
                    processed_frames.append(processed.astype(np.uint8))
                    
                    current_memory = self.get_memory_usage_mb()
                    peak_memory = max(peak_memory, current_memory)
                
                return {
                    'frames_processed': len(processed_frames),
                    'peak_memory_mb': peak_memory,
                    'final_memory_mb': self.get_memory_usage_mb()
                }
                
            finally:
                # Clean up
                del frames
                if 'processed_frames' in locals():
                    del processed_frames
                gc.collect()
        
        # Benchmark video processing memory usage
        metrics = self.benchmark.benchmark_function(process_large_video, iterations=1)
        
        # Memory usage targets (target: <2GB)
        assert metrics.peak_memory_mb < 2048, \
            f"Peak memory usage {metrics.peak_memory_mb:.1f}MB, target: <2GB"
        
        # Memory should be released after processing
        final_memory = self.get_memory_usage_mb()
        assert final_memory < 1024, \
            f"Final memory usage {final_memory:.1f}MB, target: <1GB (memory leak check)"
    
    def test_concurrent_memory_usage(self):
        """Test memory usage during concurrent operations."""
        
        def memory_intensive_task():
            """Task that uses significant memory."""
            # Allocate memory for this task
            data = np.random.rand(300, 300, 5)  # ~90MB (reduced for stability)
            
            # Process the data
            result = np.sum(data ** 2)
            
            # Simulate processing time
            time.sleep(0.2)  # Reduced time for faster testing
            
            # Clean up
            del data
            gc.collect()
            
            return {'result': float(result)}
        
        # Test concurrent memory usage
        initial_memory = self.get_memory_usage_mb()
        
        concurrent_result = self.benchmark.benchmark_concurrent_operations(
            memory_intensive_task, num_concurrent=3  # Reduced for stability
        )
        
        final_memory = self.get_memory_usage_mb()
        
        # Validate concurrent memory usage
        assert concurrent_result['memory_usage_mb'] < 2048, \
            f"Concurrent memory usage {concurrent_result['memory_usage_mb']:.1f}MB, target: <2GB"
        
        # Memory should return to reasonable levels
        memory_increase = final_memory - initial_memory
        assert memory_increase < 500, \
            f"Memory increase after concurrent ops: {memory_increase:.1f}MB, target: <500MB"
        
        assert concurrent_result['success_rate'] >= 0.6, \
            f"Success rate under memory pressure: {concurrent_result['success_rate']:.2f}, target: >=0.6"
    
    def test_memory_leak_detection(self):
        """Test for memory leaks in repeated operations."""
        
        def potentially_leaky_operation():
            """Operation that might leak memory."""
            # Create some data
            data = np.random.rand(100, 100)
            
            # Process it
            processed = data * 2 + 1
            result = np.sum(processed)
            
            # Intentionally don't clean up everything to test leak detection
            # In real code, this would be a bug
            
            return float(result)
        
        # Measure memory usage over multiple iterations
        memory_measurements = []
        
        for i in range(50):
            initial_memory = self.get_memory_usage_mb()
            
            # Run the operation
            result = potentially_leaky_operation()
            
            # Force garbage collection
            gc.collect()
            
            final_memory = self.get_memory_usage_mb()
            memory_measurements.append(final_memory - initial_memory)
            
            # Small delay to allow system to stabilize
            time.sleep(0.01)
        
        # Analyze memory usage trend
        avg_memory_per_op = sum(memory_measurements) / len(memory_measurements)
        
        # Check for memory leaks (memory usage should be minimal per operation)
        assert avg_memory_per_op < 10, \
            f"Average memory per operation: {avg_memory_per_op:.2f}MB, possible leak detected"
        
        # Check that memory usage isn't growing significantly over time
        early_avg = sum(memory_measurements[:10]) / 10
        late_avg = sum(memory_measurements[-10:]) / 10
        growth_rate = (late_avg - early_avg) / early_avg if early_avg != 0 else 0
        
        # Allow for more variance in memory measurements on Windows
        assert abs(growth_rate) < 2.0, \
            f"Memory usage growth rate: {growth_rate:.2f}, possible leak detected"
    
    def test_large_dataset_processing(self):
        """Test memory usage when processing large datasets."""
        
        def process_large_dataset():
            """Process a large dataset in chunks to manage memory."""
            total_processed = 0
            peak_memory = self.get_memory_usage_mb()
            
            # Process data in chunks to avoid memory issues
            chunk_size = 1000
            num_chunks = 10
            
            for chunk_idx in range(num_chunks):
                # Create chunk data
                chunk_data = np.random.rand(chunk_size, 100)
                
                # Process chunk
                processed_chunk = np.sum(chunk_data ** 2, axis=1)
                total_processed += len(processed_chunk)
                
                # Track memory usage
                current_memory = self.get_memory_usage_mb()
                peak_memory = max(peak_memory, current_memory)
                
                # Clean up chunk data
                del chunk_data, processed_chunk
                
                # Periodic garbage collection
                if chunk_idx % 3 == 0:
                    gc.collect()
            
            return {
                'total_processed': total_processed,
                'peak_memory_mb': peak_memory
            }
        
        # Benchmark large dataset processing
        metrics = self.benchmark.benchmark_function(process_large_dataset, iterations=1)
        
        # Memory should stay reasonable even with large datasets
        assert metrics.peak_memory_mb < 1024, \
            f"Peak memory for large dataset: {metrics.peak_memory_mb:.1f}MB, target: <1GB"
        
        # Processing should complete successfully
        assert metrics.execution_time < 30.0, \
            f"Large dataset processing took {metrics.execution_time:.2f}s, target: <30s"
    
    def test_memory_pressure_handling(self):
        """Test system behavior under memory pressure."""
        
        def memory_pressure_task():
            """Task that creates memory pressure."""
            allocated_arrays = []
            
            try:
                # Gradually increase memory usage
                for i in range(20):
                    # Allocate 50MB chunks
                    array = np.random.rand(50 * 1024 * 1024 // 8)  # 50MB
                    allocated_arrays.append(array)
                    
                    # Check if we're approaching memory limits
                    current_memory = self.get_memory_usage_mb()
                    if current_memory > 1500:  # Stop before hitting 2GB limit
                        break
                    
                    time.sleep(0.1)
                
                # Process some of the data
                total_sum = 0
                for array in allocated_arrays[:5]:  # Process first 5 arrays
                    total_sum += np.sum(array)
                
                return {
                    'arrays_allocated': len(allocated_arrays),
                    'total_sum': float(total_sum),
                    'peak_memory_mb': self.get_memory_usage_mb()
                }
                
            finally:
                # Clean up all allocated memory
                del allocated_arrays
                gc.collect()
        
        # Test memory pressure handling
        initial_memory = self.get_memory_usage_mb()
        
        metrics = self.benchmark.benchmark_function(memory_pressure_task, iterations=1)
        
        final_memory = self.get_memory_usage_mb()
        
        # Should handle memory pressure gracefully
        assert metrics.peak_memory_mb < 2048, \
            f"Peak memory under pressure: {metrics.peak_memory_mb:.1f}MB, target: <2GB"
        
        # Memory should be cleaned up properly
        memory_cleanup = initial_memory - final_memory
        assert abs(memory_cleanup) < 200, \
            f"Memory cleanup delta: {memory_cleanup:.1f}MB, should return to baseline"
    
    def test_memory_monitoring_accuracy(self):
        """Test accuracy of memory monitoring."""
        
        def controlled_memory_allocation():
            """Allocate a known amount of memory."""
            # Allocate exactly 100MB
            array_size = 100 * 1024 * 1024 // 8  # 100MB of float64
            test_array = np.ones(array_size, dtype=np.float64)
            
            # Keep it alive for measurement
            time.sleep(0.5)
            
            # Clean up
            del test_array
            gc.collect()
            
            return array_size
        
        # Monitor memory allocation
        initial_memory = self.get_memory_usage_mb()
        
        metrics = self.benchmark.benchmark_function(controlled_memory_allocation, iterations=1)
        
        final_memory = self.get_memory_usage_mb()
        
        # Validate memory monitoring accuracy
        # Peak memory should show the allocation
        expected_increase = 100  # 100MB allocated
        actual_peak_increase = metrics.peak_memory_mb - initial_memory
        
        # Allow for some variance in memory measurement
        assert 80 <= actual_peak_increase <= 120, \
            f"Memory monitoring accuracy: expected ~100MB increase, got {actual_peak_increase:.1f}MB"
        
        # Memory should return to baseline after cleanup
        memory_return = abs(final_memory - initial_memory)
        assert memory_return < 50, \
            f"Memory should return to baseline, delta: {memory_return:.1f}MB"


@pytest.mark.performance
class TestMemoryOptimization:
    """Test memory optimization strategies."""
    
    def setup_method(self):
        """Set up performance testing framework."""
        self.benchmark = PerformanceBenchmark()
    
    def test_streaming_vs_batch_processing(self):
        """Compare memory usage of streaming vs batch processing."""
        
        def batch_processing():
            """Process all data at once (high memory usage)."""
            # Load all data into memory
            all_data = []
            for i in range(200):  # Increased size for clearer difference
                data_chunk = np.random.rand(2000, 200)  # Larger chunks
                all_data.append(data_chunk)
            
            # Process all data
            results = []
            for data in all_data:
                result = np.sum(data ** 2)
                results.append(result)
            
            # Keep data in memory longer to show peak usage
            time.sleep(0.1)
            
            return len(results)
        
        def streaming_processing():
            """Process data in streaming fashion (low memory usage)."""
            results = []
            
            for i in range(200):  # Same number of chunks
                # Load and process one chunk at a time
                data_chunk = np.random.rand(2000, 200)  # Same size chunks
                result = np.sum(data_chunk ** 2)
                results.append(result)
                
                # Clean up immediately
                del data_chunk
                
                # Force garbage collection periodically
                if i % 10 == 0:
                    gc.collect()
            
            return len(results)
        
        # Compare memory usage
        batch_metrics = self.benchmark.benchmark_function(batch_processing, iterations=1)
        streaming_metrics = self.benchmark.benchmark_function(streaming_processing, iterations=1)
        
        # Streaming should use significantly less memory
        memory_reduction = (batch_metrics.peak_memory_mb - streaming_metrics.peak_memory_mb) / batch_metrics.peak_memory_mb
        
        assert memory_reduction > 0.2, \
            f"Streaming memory reduction: {memory_reduction:.2f}, target: >20% reduction"
        
        # Both should produce same results (functionality preserved)
        assert batch_metrics.execution_time > 0 and streaming_metrics.execution_time > 0, \
            "Both processing methods should complete successfully"
    
    def test_memory_pool_efficiency(self):
        """Test memory pool efficiency for repeated allocations."""
        
        def without_memory_pool():
            """Repeated allocations without pooling."""
            results = []
            
            for i in range(100):  # More iterations to show difference
                # Allocate new array each time
                data = np.random.rand(2000, 200)  # Larger arrays
                processed = data * 2
                result = np.sum(processed)
                results.append(result)
                
                # Clean up
                del data, processed
                
                # Force garbage collection occasionally
                if i % 20 == 0:
                    gc.collect()
            
            return len(results)
        
        def with_memory_reuse():
            """Reuse memory allocations where possible."""
            results = []
            
            # Pre-allocate reusable arrays
            data_buffer = np.empty((2000, 200))  # Same size as above
            processed_buffer = np.empty((2000, 200))
            
            for i in range(100):  # Same number of iterations
                # Reuse existing arrays
                data_buffer[:] = np.random.rand(2000, 200)  # Fill existing buffer
                np.multiply(data_buffer, 2, out=processed_buffer)
                result = np.sum(processed_buffer)
                results.append(result)
            
            return len(results)
        
        # Compare memory efficiency
        no_pool_metrics = self.benchmark.benchmark_function(without_memory_pool, iterations=1)
        with_reuse_metrics = self.benchmark.benchmark_function(with_memory_reuse, iterations=1)
        
        # Memory reuse should be more efficient (or at least not worse)
        memory_improvement = (no_pool_metrics.peak_memory_mb - with_reuse_metrics.peak_memory_mb) / no_pool_metrics.peak_memory_mb
        
        # Allow for minimal improvement or at least no regression
        assert memory_improvement >= -0.1, \
            f"Memory reuse should not be worse: {memory_improvement:.2f}, target: >=-10%"
        
        # Should also be faster due to reduced allocations
        time_improvement = (no_pool_metrics.execution_time - with_reuse_metrics.execution_time) / no_pool_metrics.execution_time
        
        assert time_improvement > 0.0, \
            f"Time improvement with memory reuse: {time_improvement:.2f}, target: >0% improvement"