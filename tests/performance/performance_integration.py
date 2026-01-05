"""Integration module for connecting performance reporting with existing test framework."""

import pytest
import time
from pathlib import Path
from typing import Dict, Any, Optional
import logging

from .benchmark_framework import PerformanceBenchmark, PerformanceMetrics
from .performance_reporter import PerformanceReporter
from .performance_dashboard import PerformanceDashboard

logger = logging.getLogger(__name__)


class PerformanceTestIntegration:
    """Integration layer for performance testing with reporting and trend analysis."""
    
    def __init__(self, 
                 enable_reporting: bool = True,
                 enable_dashboards: bool = True,
                 auto_generate_reports: bool = False):
        self.enable_reporting = enable_reporting
        self.enable_dashboards = enable_dashboards
        self.auto_generate_reports = auto_generate_reports
        
        # Initialize components
        self.benchmark = PerformanceBenchmark()
        
        if self.enable_reporting:
            self.reporter = PerformanceReporter()
        else:
            self.reporter = None
        
        if self.enable_dashboards and self.reporter:
            self.dashboard = PerformanceDashboard(self.reporter)
        else:
            self.dashboard = None
    
    def run_performance_test(self, 
                           test_name: str,
                           test_function,
                           *args,
                           iterations: int = 1,
                           record_results: bool = True,
                           check_regression: bool = True,
                           **kwargs) -> Dict[str, Any]:
        """Run performance test with integrated reporting and trend analysis."""
        
        # Execute benchmark
        metrics = self.benchmark.benchmark_function(
            test_function, *args, iterations=iterations, **kwargs
        )
        
        # Record results for trend analysis
        if record_results and self.reporter:
            self._record_performance_metrics(test_name, metrics)
        
        # Check for regression
        regression_result = None
        if check_regression:
            regression_result = self.benchmark.validate_performance_regression(
                test_name, metrics
            )
        
        # Prepare result
        result = {
            'test_name': test_name,
            'metrics': metrics,
            'regression_analysis': regression_result,
            'timestamp': time.time()
        }
        
        # Log results
        self._log_performance_results(result)
        
        return result
    
    def run_concurrent_performance_test(self,
                                      test_name: str,
                                      test_function,
                                      num_concurrent: int = 5,
                                      record_results: bool = True,
                                      *args,
                                      **kwargs) -> Dict[str, Any]:
        """Run concurrent performance test with integrated reporting."""
        
        # Execute concurrent benchmark
        concurrent_result = self.benchmark.benchmark_concurrent_operations(
            test_function, num_concurrent, *args, **kwargs
        )
        
        # Convert to PerformanceMetrics for consistency
        metrics = PerformanceMetrics(
            execution_time=concurrent_result['total_time'],
            memory_usage_mb=concurrent_result.get('memory_usage_mb', 0),
            cpu_usage_percent=0,  # Not tracked in concurrent tests
            peak_memory_mb=concurrent_result.get('peak_memory_mb', 0),
            throughput=concurrent_result['throughput']
        )
        
        # Record results
        if record_results and self.reporter:
            self._record_performance_metrics(f"{test_name}_concurrent_{num_concurrent}", metrics)
        
        # Check regression
        regression_result = self.benchmark.validate_performance_regression(
            f"{test_name}_concurrent_{num_concurrent}", metrics
        )
        
        result = {
            'test_name': f"{test_name}_concurrent_{num_concurrent}",
            'concurrent_details': concurrent_result,
            'metrics': metrics,
            'regression_analysis': regression_result,
            'timestamp': time.time()
        }
        
        self._log_performance_results(result)
        
        return result
    
    def _record_performance_metrics(self, test_name: str, metrics: PerformanceMetrics):
        """Record performance metrics for trend analysis."""
        if not self.reporter:
            return
        
        metrics_dict = {
            'execution_time': metrics.execution_time,
            'memory_usage_mb': metrics.memory_usage_mb,
            'cpu_usage_percent': metrics.cpu_usage_percent,
            'peak_memory_mb': metrics.peak_memory_mb,
            'throughput': metrics.throughput
        }
        
        self.reporter.record_performance_data(test_name, metrics_dict)
    
    def _log_performance_results(self, result: Dict[str, Any]):
        """Log performance test results."""
        test_name = result['test_name']
        metrics = result['metrics']
        regression = result.get('regression_analysis')
        
        logger.info(f"Performance test completed: {test_name}")
        logger.info(f"  Execution time: {metrics.execution_time:.3f}s")
        logger.info(f"  Memory usage: {metrics.memory_usage_mb:.1f}MB")
        logger.info(f"  Peak memory: {metrics.peak_memory_mb:.1f}MB")
        logger.info(f"  Throughput: {metrics.throughput:.2f} ops/s")
        
        if regression:
            if regression['regression_detected']:
                logger.warning(f"  ⚠️  Regression detected: {regression['regressions']}")
            elif regression.get('improvements'):
                logger.info(f"  ✅ Improvements detected: {regression['improvements']}")
            else:
                logger.info("  📊 Performance within expected range")
    
    def generate_test_session_report(self) -> Optional[Path]:
        """Generate performance report for current test session."""
        if not self.reporter:
            return None
        
        try:
            report = self.reporter.generate_performance_report(
                report_period="session",
                include_trends=True
            )
            
            logger.info(f"Test session performance report generated")
            logger.info(f"  Total tests: {report.total_tests}")
            logger.info(f"  Regressions: {len(report.regression_alerts)}")
            logger.info(f"  Improvements: {len(report.improvement_highlights)}")
            
            return self.reporter.reports_dir / "latest_session_report.json"
        
        except Exception as e:
            logger.error(f"Failed to generate test session report: {e}")
            return None
    
    def generate_test_session_dashboard(self) -> Optional[Path]:
        """Generate performance dashboard for current test session."""
        if not self.dashboard:
            return None
        
        try:
            dashboard_path = self.dashboard.generate_dashboard(
                dashboard_type="comprehensive",
                days_back=1  # Focus on recent data
            )
            
            logger.info(f"Test session dashboard generated: {dashboard_path}")
            return dashboard_path
        
        except Exception as e:
            logger.error(f"Failed to generate test session dashboard: {e}")
            return None


# Pytest fixtures for performance testing integration
@pytest.fixture(scope="session")
def performance_integration():
    """Provide performance testing integration for test session."""
    integration = PerformanceTestIntegration(
        enable_reporting=True,
        enable_dashboards=True,
        auto_generate_reports=False
    )
    
    yield integration
    
    # Generate session report and dashboard at end of test session
    if integration.auto_generate_reports:
        integration.generate_test_session_report()
        integration.generate_test_session_dashboard()


@pytest.fixture
def performance_test(performance_integration):
    """Provide performance test helper for individual tests."""
    def run_test(test_name: str, test_function, *args, **kwargs):
        return performance_integration.run_performance_test(
            test_name, test_function, *args, **kwargs
        )
    
    return run_test


@pytest.fixture
def concurrent_performance_test(performance_integration):
    """Provide concurrent performance test helper."""
    def run_test(test_name: str, test_function, num_concurrent: int = 5, *args, **kwargs):
        return performance_integration.run_concurrent_performance_test(
            test_name, test_function, num_concurrent, *args, **kwargs
        )
    
    return run_test


# Pytest hooks for automatic performance reporting
def pytest_sessionstart(session):
    """Initialize performance monitoring at start of test session."""
    if hasattr(session.config, 'performance_integration'):
        logger.info("Performance monitoring initialized for test session")


def pytest_sessionfinish(session, exitstatus):
    """Generate performance reports at end of test session."""
    # Check if performance reporting is enabled
    performance_reporting = getattr(session.config.option, 'performance_reporting', False)
    
    if performance_reporting:
        try:
            integration = PerformanceTestIntegration(
                enable_reporting=True,
                enable_dashboards=True
            )
            
            # Generate session report
            report_path = integration.generate_test_session_report()
            if report_path:
                print(f"\n📊 Performance report: {report_path}")
            
            # Generate session dashboard
            dashboard_path = integration.generate_test_session_dashboard()
            if dashboard_path:
                print(f"📈 Performance dashboard: {dashboard_path}")
        
        except Exception as e:
            logger.error(f"Failed to generate session performance reports: {e}")


def pytest_addoption(parser):
    """Add performance reporting options to pytest."""
    parser.addoption(
        "--performance-reporting",
        action="store_true",
        default=False,
        help="Enable automatic performance reporting"
    )
    
    parser.addoption(
        "--performance-dashboard",
        action="store_true", 
        default=False,
        help="Generate performance dashboard after test session"
    )


# Decorators for easy performance testing
def performance_test(test_name: Optional[str] = None,
                    iterations: int = 1,
                    record_results: bool = True,
                    check_regression: bool = True):
    """Decorator for performance testing with integrated reporting."""
    
    def decorator(func):
        def wrapper(*args, **kwargs):
            # Get or create performance integration
            integration = PerformanceTestIntegration()
            
            # Use function name if test_name not provided
            name = test_name or func.__name__
            
            # Run performance test
            result = integration.run_performance_test(
                name, func, *args,
                iterations=iterations,
                record_results=record_results,
                check_regression=check_regression,
                **kwargs
            )
            
            return result
        
        return wrapper
    return decorator


def concurrent_performance_test(test_name: Optional[str] = None,
                               num_concurrent: int = 5,
                               record_results: bool = True):
    """Decorator for concurrent performance testing."""
    
    def decorator(func):
        def wrapper(*args, **kwargs):
            integration = PerformanceTestIntegration()
            name = test_name or func.__name__
            
            result = integration.run_concurrent_performance_test(
                name, func, num_concurrent, record_results, *args, **kwargs
            )
            
            return result
        
        return wrapper
    return decorator


# Example usage functions
def example_performance_test_usage():
    """Example of how to use the performance testing integration."""
    
    # Method 1: Using the integration directly
    integration = PerformanceTestIntegration()
    
    def sample_function():
        time.sleep(0.1)
        return "completed"
    
    result = integration.run_performance_test(
        "sample_test", sample_function, iterations=5
    )
    
    print(f"Test completed: {result['test_name']}")
    print(f"Execution time: {result['metrics'].execution_time:.3f}s")
    
    # Method 2: Using decorators
    @performance_test("decorated_test", iterations=3)
    def decorated_sample_function():
        time.sleep(0.05)
        return "decorated completed"
    
    decorated_result = decorated_sample_function()
    
    # Method 3: Using pytest fixtures (in test files)
    """
    def test_my_performance(performance_test):
        def my_test_function():
            # Your test logic here
            time.sleep(0.1)
            return "test result"
        
        result = performance_test("my_test", my_test_function)
        assert result['metrics'].execution_time < 1.0
    """


if __name__ == "__main__":
    # Run example usage
    example_performance_test_usage()