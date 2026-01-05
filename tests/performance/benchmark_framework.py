"""Performance benchmarking framework for AlexPose system."""

import time
import psutil
import threading
from typing import Dict, List, Callable, Any, Optional
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
from pathlib import Path
import logging

logger = logging.getLogger(__name__)

@dataclass
class PerformanceMetrics:
    """Performance metrics for test execution."""
    execution_time: float
    memory_usage_mb: float
    cpu_usage_percent: float
    peak_memory_mb: float
    throughput: float = 0.0  # operations per second
    
    def to_dict(self) -> Dict[str, float]:
        """Convert metrics to dictionary for serialization."""
        return {
            'execution_time': self.execution_time,
            'memory_usage_mb': self.memory_usage_mb,
            'cpu_usage_percent': self.cpu_usage_percent,
            'peak_memory_mb': self.peak_memory_mb,
            'throughput': self.throughput
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, float]) -> 'PerformanceMetrics':
        """Create metrics from dictionary."""
        return cls(**data)

class PerformanceBenchmark:
    """Performance benchmarking and regression testing."""
    
    def __init__(self, baseline_file: Optional[Path] = None, use_config: bool = True):
        self.baseline_file = baseline_file or Path("tests/performance/baselines.json")
        self.baseline_metrics: Dict[str, PerformanceMetrics] = {}
        self.current_metrics: Dict[str, PerformanceMetrics] = {}
        self.use_config = use_config
        
        # Import performance config if requested
        if self.use_config:
            try:
                from .performance_config import get_performance_config
                self.performance_config = get_performance_config()
            except ImportError:
                logger.warning("Performance config not available, using default tolerances")
                self.performance_config = None
                self.use_config = False
        else:
            self.performance_config = None
        
        self._load_baselines()
    
    def _load_baselines(self):
        """Load baseline metrics from file."""
        if self.baseline_file.exists():
            try:
                with open(self.baseline_file, 'r') as f:
                    data = json.load(f)
                    for name, metrics_dict in data.items():
                        self.baseline_metrics[name] = PerformanceMetrics.from_dict(metrics_dict)
                logger.info(f"Loaded {len(self.baseline_metrics)} baseline metrics")
            except Exception as e:
                logger.warning(f"Failed to load baselines: {e}")
    
    def _save_baselines(self):
        """Save baseline metrics to file."""
        try:
            self.baseline_file.parent.mkdir(parents=True, exist_ok=True)
            data = {name: metrics.to_dict() for name, metrics in self.baseline_metrics.items()}
            with open(self.baseline_file, 'w') as f:
                json.dump(data, f, indent=2)
            logger.info(f"Saved {len(self.baseline_metrics)} baseline metrics")
        except Exception as e:
            logger.error(f"Failed to save baselines: {e}")
    
    def benchmark_function(
        self, 
        func: Callable, 
        *args, 
        iterations: int = 1,
        **kwargs
    ) -> PerformanceMetrics:
        """Benchmark a function's performance."""
        
        # Memory monitoring setup
        process = psutil.Process()
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB
        peak_memory = initial_memory
        
        def monitor_memory():
            nonlocal peak_memory
            while monitoring:
                try:
                    current_memory = process.memory_info().rss / 1024 / 1024
                    peak_memory = max(peak_memory, current_memory)
                    time.sleep(0.1)
                except psutil.NoSuchProcess:
                    break
        
        # Start monitoring
        monitoring = True
        monitor_thread = threading.Thread(target=monitor_memory, daemon=True)
        monitor_thread.start()
        
        # Execute benchmark
        start_time = time.time()
        cpu_start = process.cpu_percent()
        
        try:
            for _ in range(iterations):
                func(*args, **kwargs)
        finally:
            # Stop monitoring
            monitoring = False
            monitor_thread.join(timeout=1.0)
        
        end_time = time.time()
        cpu_end = process.cpu_percent()
        
        final_memory = process.memory_info().rss / 1024 / 1024
        execution_time = end_time - start_time
        
        return PerformanceMetrics(
            execution_time=execution_time,
            memory_usage_mb=final_memory - initial_memory,
            cpu_usage_percent=(cpu_end - cpu_start) / iterations if iterations > 0 else 0,
            peak_memory_mb=peak_memory,
            throughput=iterations / execution_time if execution_time > 0 else 0
        )
    
    def benchmark_concurrent_operations(
        self, 
        func: Callable, 
        num_concurrent: int = 5,
        *args,
        **kwargs
    ) -> Dict[str, Any]:
        """Test system performance under concurrent load."""
        
        def execute_operation(operation_id: int):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                logger.error(f"Concurrent operation {operation_id} failed: {e}")
                return None
        
        start_time = time.time()
        process = psutil.Process()
        initial_memory = process.memory_info().rss / 1024 / 1024
        peak_memory = initial_memory
        
        # Monitor memory during concurrent execution
        monitoring = True
        def monitor_memory():
            nonlocal peak_memory
            while monitoring:
                try:
                    current_memory = process.memory_info().rss / 1024 / 1024
                    peak_memory = max(peak_memory, current_memory)
                    time.sleep(0.1)
                except psutil.NoSuchProcess:
                    break
        
        monitor_thread = threading.Thread(target=monitor_memory, daemon=True)
        monitor_thread.start()
        
        try:
            with ThreadPoolExecutor(max_workers=num_concurrent) as executor:
                futures = [executor.submit(execute_operation, i) for i in range(num_concurrent)]
                results = []
                for future in as_completed(futures):
                    try:
                        result = future.result(timeout=30)  # 30 second timeout
                        results.append(result)
                    except Exception as e:
                        logger.error(f"Concurrent operation failed: {e}")
                        results.append(None)
        finally:
            monitoring = False
            monitor_thread.join(timeout=1.0)
        
        total_time = time.time() - start_time
        final_memory = process.memory_info().rss / 1024 / 1024
        
        successful_results = [r for r in results if r is not None]
        
        return {
            'concurrent_operations': num_concurrent,
            'total_time': total_time,
            'average_time_per_operation': total_time / num_concurrent,
            'successful_operations': len(successful_results),
            'failed_operations': len(results) - len(successful_results),
            'throughput': num_concurrent / total_time if total_time > 0 else 0,
            'memory_usage_mb': final_memory - initial_memory,
            'peak_memory_mb': peak_memory,
            'success_rate': len(successful_results) / num_concurrent if num_concurrent > 0 else 0
        }
    
    def establish_baseline(self, test_name: str, metrics: PerformanceMetrics):
        """Establish baseline metrics for a test."""
        self.baseline_metrics[test_name] = metrics
        self._save_baselines()
        logger.info(f"Established baseline for {test_name}: {metrics.execution_time:.2f}s")
    
    def validate_performance_regression(
        self, 
        test_name: str, 
        current_metrics: PerformanceMetrics,
        tolerance_percent: Optional[float] = None,
        memory_tolerance_percent: Optional[float] = None,
        throughput_tolerance_percent: Optional[float] = None
    ) -> Dict[str, Any]:
        """Validate that performance hasn't regressed with configurable tolerance."""
        
        if test_name not in self.baseline_metrics:
            # First run - establish baseline
            self.establish_baseline(test_name, current_metrics)
            return {
                'regression_detected': False,
                'baseline_established': True,
                'message': f"Baseline established for {test_name}",
                'baseline_metrics': current_metrics,
                'current_metrics': current_metrics
            }
        
        baseline = self.baseline_metrics[test_name]
        
        # Get tolerances from configuration or use provided values
        if self.use_config and self.performance_config:
            time_tolerance = tolerance_percent or self.performance_config.get_tolerance(test_name, "execution_time")
            memory_tolerance = memory_tolerance_percent or self.performance_config.get_tolerance(test_name, "memory_usage")
            throughput_tolerance = throughput_tolerance_percent or self.performance_config.get_tolerance(test_name, "throughput")
        else:
            # Fallback to provided values or defaults
            time_tolerance = tolerance_percent or 10.0
            memory_tolerance = memory_tolerance_percent or 15.0
            throughput_tolerance = throughput_tolerance_percent or 10.0
        
        # Check for regressions
        time_regression = (current_metrics.execution_time - baseline.execution_time) / baseline.execution_time * 100
        
        # Handle memory regression calculation with zero baseline
        if baseline.peak_memory_mb > 0:
            memory_regression = (current_metrics.peak_memory_mb - baseline.peak_memory_mb) / baseline.peak_memory_mb * 100
        else:
            memory_regression = 0.0  # No meaningful memory comparison if baseline is zero
        
        # Handle throughput regression calculation
        if baseline.throughput > 0:
            throughput_regression = (baseline.throughput - current_metrics.throughput) / baseline.throughput * 100
        else:
            throughput_regression = 0.0
        
        regressions = []
        if time_regression > time_tolerance:
            regressions.append(f"Execution time increased by {time_regression:.1f}% (tolerance: {time_tolerance:.1f}%)")
        
        if memory_regression > memory_tolerance:
            regressions.append(f"Memory usage increased by {memory_regression:.1f}% (tolerance: {memory_tolerance:.1f}%)")
        
        if throughput_regression > throughput_tolerance:
            regressions.append(f"Throughput decreased by {throughput_regression:.1f}% (tolerance: {throughput_tolerance:.1f}%)")
        
        # Check for improvements (negative regression is good)
        improvements = []
        if time_regression < -time_tolerance:
            improvements.append(f"Execution time improved by {abs(time_regression):.1f}%")
        
        if memory_regression < -memory_tolerance:
            improvements.append(f"Memory usage improved by {abs(memory_regression):.1f}%")
        
        if throughput_regression < -throughput_tolerance:
            improvements.append(f"Throughput improved by {abs(throughput_regression):.1f}%")
        
        return {
            'regression_detected': len(regressions) > 0,
            'regressions': regressions,
            'improvements': improvements,
            'time_change_percent': time_regression,
            'memory_change_percent': memory_regression,
            'throughput_change_percent': throughput_regression,
            'tolerances_used': {
                'time_tolerance': time_tolerance,
                'memory_tolerance': memory_tolerance,
                'throughput_tolerance': throughput_tolerance
            },
            'baseline_metrics': baseline,
            'current_metrics': current_metrics
        }
    
    def get_tolerance_info(self, test_name: str) -> Dict[str, float]:
        """Get tolerance information for a specific test."""
        if self.use_config and self.performance_config:
            return {
                'execution_time_tolerance': self.performance_config.get_tolerance(test_name, "execution_time"),
                'memory_usage_tolerance': self.performance_config.get_tolerance(test_name, "memory_usage"),
                'throughput_tolerance': self.performance_config.get_tolerance(test_name, "throughput")
            }
        else:
            return {
                'execution_time_tolerance': 10.0,
                'memory_usage_tolerance': 15.0,
                'throughput_tolerance': 10.0
            }
    
    def configure_test_tolerance(self, test_name: str, **tolerances):
        """Configure tolerance for a specific test."""
        if self.use_config and self.performance_config:
            self.performance_config.set_test_tolerance(test_name, **tolerances)
            logger.info(f"Updated tolerance configuration for {test_name}: {tolerances}")
        else:
            logger.warning("Performance configuration not available, cannot set test tolerance")
    
    def generate_performance_report(self) -> Dict[str, Any]:
        """Generate comprehensive performance report."""
        report = {
            'timestamp': time.time(),
            'total_benchmarks': len(self.baseline_metrics),
            'baselines': {name: metrics.to_dict() for name, metrics in self.baseline_metrics.items()},
            'summary': {
                'avg_execution_time': sum(m.execution_time for m in self.baseline_metrics.values()) / len(self.baseline_metrics) if self.baseline_metrics else 0,
                'avg_memory_usage': sum(m.memory_usage_mb for m in self.baseline_metrics.values()) / len(self.baseline_metrics) if self.baseline_metrics else 0,
                'avg_throughput': sum(m.throughput for m in self.baseline_metrics.values()) / len(self.baseline_metrics) if self.baseline_metrics else 0
            }
        }
        
        # Add tolerance configuration information if available
        if self.use_config and self.performance_config:
            report['tolerance_configuration'] = self.performance_config.get_all_tolerances()
        
        return report
    
    def integrate_with_reporter(self, test_name: str, metrics: PerformanceMetrics):
        """Integrate with performance reporter for trend analysis."""
        try:
            from .performance_reporter import PerformanceReporter
            
            # Create reporter instance
            reporter = PerformanceReporter()
            
            # Record metrics for trend analysis
            metrics_dict = {
                'execution_time': metrics.execution_time,
                'memory_usage_mb': metrics.memory_usage_mb,
                'cpu_usage_percent': metrics.cpu_usage_percent,
                'peak_memory_mb': metrics.peak_memory_mb,
                'throughput': metrics.throughput
            }
            
            reporter.record_performance_data(test_name, metrics_dict)
            logger.info(f"Performance data recorded for trend analysis: {test_name}")
            
        except ImportError:
            logger.warning("Performance reporter not available for trend analysis")
        except Exception as e:
            logger.error(f"Failed to integrate with performance reporter: {e}")