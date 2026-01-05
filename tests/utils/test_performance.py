"""
Test performance monitoring and reporting utilities.

This module provides utilities for monitoring test execution performance,
establishing baselines, and detecting performance regressions.
"""

import time
import psutil
import pytest
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
from pathlib import Path
import json
import statistics


@dataclass
class PerformanceMetrics:
    """Container for test performance metrics."""
    test_name: str
    execution_time: float
    memory_usage_mb: float
    cpu_usage_percent: float
    timestamp: float = field(default_factory=time.time)
    metadata: Dict[str, Any] = field(default_factory=dict)


class PerformanceMonitor:
    """Monitor and track test performance metrics."""
    
    def __init__(self, baseline_file: Optional[Path] = None):
        """Initialize performance monitor.
        
        Args:
            baseline_file: Path to baseline performance data file
        """
        self.baseline_file = baseline_file or Path("tests/performance_baselines.json")
        self.current_metrics: List[PerformanceMetrics] = []
        self.baselines: Dict[str, Dict[str, float]] = {}
        self._load_baselines()
    
    def _load_baselines(self) -> None:
        """Load performance baselines from file."""
        if self.baseline_file.exists():
            try:
                with open(self.baseline_file, 'r') as f:
                    self.baselines = json.load(f)
            except (json.JSONDecodeError, IOError):
                self.baselines = {}
    
    def save_baselines(self) -> None:
        """Save current baselines to file."""
        self.baseline_file.parent.mkdir(parents=True, exist_ok=True)
        with open(self.baseline_file, 'w') as f:
            json.dump(self.baselines, f, indent=2)
    
    def start_monitoring(self, test_name: str) -> 'PerformanceContext':
        """Start monitoring a test's performance.
        
        Args:
            test_name: Name of the test being monitored
            
        Returns:
            Performance monitoring context manager
        """
        return PerformanceContext(self, test_name)
    
    def record_metrics(self, metrics: PerformanceMetrics) -> None:
        """Record performance metrics for a test.
        
        Args:
            metrics: Performance metrics to record
        """
        self.current_metrics.append(metrics)
    
    def establish_baseline(self, test_name: str, metrics: PerformanceMetrics) -> None:
        """Establish performance baseline for a test.
        
        Args:
            test_name: Name of the test
            metrics: Performance metrics to use as baseline
        """
        self.baselines[test_name] = {
            'execution_time': metrics.execution_time,
            'memory_usage_mb': metrics.memory_usage_mb,
            'cpu_usage_percent': metrics.cpu_usage_percent
        }
    
    def check_regression(self, test_name: str, metrics: PerformanceMetrics, 
                        tolerance: float = 0.2) -> Dict[str, Any]:
        """Check for performance regression against baseline.
        
        Args:
            test_name: Name of the test
            metrics: Current performance metrics
            tolerance: Acceptable performance degradation (0.2 = 20%)
            
        Returns:
            Dictionary with regression analysis results
        """
        if test_name not in self.baselines:
            return {
                'has_baseline': False,
                'regression_detected': False,
                'message': f'No baseline found for {test_name}'
            }
        
        baseline = self.baselines[test_name]
        regressions = []
        
        # Check execution time regression
        time_regression = (metrics.execution_time - baseline['execution_time']) / baseline['execution_time']
        if time_regression > tolerance:
            regressions.append(f'Execution time increased by {time_regression:.1%}')
        
        # Check memory usage regression
        memory_regression = (metrics.memory_usage_mb - baseline['memory_usage_mb']) / baseline['memory_usage_mb']
        if memory_regression > tolerance:
            regressions.append(f'Memory usage increased by {memory_regression:.1%}')
        
        return {
            'has_baseline': True,
            'regression_detected': len(regressions) > 0,
            'regressions': regressions,
            'time_regression': time_regression,
            'memory_regression': memory_regression,
            'message': '; '.join(regressions) if regressions else 'No regression detected'
        }
    
    def generate_report(self) -> Dict[str, Any]:
        """Generate performance report for all recorded metrics.
        
        Returns:
            Dictionary containing performance report
        """
        if not self.current_metrics:
            return {'message': 'No performance metrics recorded'}
        
        # Group metrics by test name
        test_groups = {}
        for metric in self.current_metrics:
            if metric.test_name not in test_groups:
                test_groups[metric.test_name] = []
            test_groups[metric.test_name].append(metric)
        
        report = {
            'total_tests': len(test_groups),
            'total_executions': len(self.current_metrics),
            'tests': {}
        }
        
        for test_name, metrics in test_groups.items():
            execution_times = [m.execution_time for m in metrics]
            memory_usage = [m.memory_usage_mb for m in metrics]
            
            report['tests'][test_name] = {
                'executions': len(metrics),
                'avg_execution_time': statistics.mean(execution_times),
                'min_execution_time': min(execution_times),
                'max_execution_time': max(execution_times),
                'avg_memory_usage': statistics.mean(memory_usage),
                'max_memory_usage': max(memory_usage)
            }
            
            # Add regression analysis if baseline exists
            if metrics:
                latest_metric = metrics[-1]
                regression_info = self.check_regression(test_name, latest_metric)
                report['tests'][test_name]['regression_analysis'] = regression_info
        
        return report


class PerformanceContext:
    """Context manager for monitoring test performance."""
    
    def __init__(self, monitor: PerformanceMonitor, test_name: str):
        """Initialize performance context.
        
        Args:
            monitor: Performance monitor instance
            test_name: Name of the test being monitored
        """
        self.monitor = monitor
        self.test_name = test_name
        self.start_time: Optional[float] = None
        self.start_memory: Optional[float] = None
        self.process = psutil.Process()
    
    def __enter__(self) -> 'PerformanceContext':
        """Start performance monitoring."""
        self.start_time = time.time()
        self.start_memory = self.process.memory_info().rss / 1024 / 1024  # MB
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """Stop performance monitoring and record metrics."""
        if self.start_time is None:
            return
        
        execution_time = time.time() - self.start_time
        current_memory = self.process.memory_info().rss / 1024 / 1024  # MB
        memory_usage = current_memory - (self.start_memory or 0)
        cpu_usage = self.process.cpu_percent()
        
        metrics = PerformanceMetrics(
            test_name=self.test_name,
            execution_time=execution_time,
            memory_usage_mb=memory_usage,
            cpu_usage_percent=cpu_usage
        )
        
        self.monitor.record_metrics(metrics)


# Global performance monitor instance
_performance_monitor = PerformanceMonitor()


def get_performance_monitor() -> PerformanceMonitor:
    """Get the global performance monitor instance."""
    return _performance_monitor


@pytest.fixture
def performance_monitor():
    """Pytest fixture for performance monitoring."""
    return get_performance_monitor()


def monitor_performance(test_name: str):
    """Decorator for monitoring test performance.
    
    Args:
        test_name: Name of the test to monitor
        
    Usage:
        @monitor_performance("test_video_processing")
        def test_video_processing():
            # Test implementation
            pass
    """
    def decorator(func):
        def wrapper(*args, **kwargs):
            with _performance_monitor.start_monitoring(test_name):
                return func(*args, **kwargs)
        return wrapper
    return decorator


# Performance assertion helpers
def assert_execution_time_under(seconds: float, test_name: str = None):
    """Assert that the last recorded test execution was under the specified time.
    
    Args:
        seconds: Maximum allowed execution time
        test_name: Optional test name to check specific test
    """
    monitor = get_performance_monitor()
    if not monitor.current_metrics:
        pytest.fail("No performance metrics recorded")
    
    if test_name:
        matching_metrics = [m for m in monitor.current_metrics if m.test_name == test_name]
        if not matching_metrics:
            pytest.fail(f"No performance metrics found for test: {test_name}")
        latest_metric = matching_metrics[-1]
    else:
        latest_metric = monitor.current_metrics[-1]
    
    if latest_metric.execution_time > seconds:
        pytest.fail(
            f"Test {latest_metric.test_name} took {latest_metric.execution_time:.2f}s, "
            f"expected under {seconds}s"
        )


def assert_memory_usage_under(mb: float, test_name: str = None):
    """Assert that the last recorded test memory usage was under the specified amount.
    
    Args:
        mb: Maximum allowed memory usage in MB
        test_name: Optional test name to check specific test
    """
    monitor = get_performance_monitor()
    if not monitor.current_metrics:
        pytest.fail("No performance metrics recorded")
    
    if test_name:
        matching_metrics = [m for m in monitor.current_metrics if m.test_name == test_name]
        if not matching_metrics:
            pytest.fail(f"No performance metrics found for test: {test_name}")
        latest_metric = matching_metrics[-1]
    else:
        latest_metric = monitor.current_metrics[-1]
    
    if latest_metric.memory_usage_mb > mb:
        pytest.fail(
            f"Test {latest_metric.test_name} used {latest_metric.memory_usage_mb:.2f}MB, "
            f"expected under {mb}MB"
        )


def assert_no_performance_regression(test_name: str, tolerance: float = 0.2):
    """Assert that there's no performance regression for the specified test.
    
    Args:
        test_name: Name of the test to check
        tolerance: Acceptable performance degradation (0.2 = 20%)
    """
    monitor = get_performance_monitor()
    matching_metrics = [m for m in monitor.current_metrics if m.test_name == test_name]
    
    if not matching_metrics:
        pytest.fail(f"No performance metrics found for test: {test_name}")
    
    latest_metric = matching_metrics[-1]
    regression_info = monitor.check_regression(test_name, latest_metric, tolerance)
    
    if regression_info['regression_detected']:
        pytest.fail(f"Performance regression detected for {test_name}: {regression_info['message']}")