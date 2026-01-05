"""Tests for performance reporting and trend analysis functionality."""

import pytest
import json
import time
import tempfile
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, Any

from tests.performance.performance_reporter import PerformanceReporter, PerformanceTrend, PerformanceReport
from tests.performance.performance_dashboard import PerformanceDashboard
from tests.performance.performance_integration import PerformanceTestIntegration


@pytest.mark.performance
@pytest.mark.slow
class TestPerformanceReporter:
    """Test performance reporting functionality."""
    
    def setup_method(self):
        """Set up test environment."""
        # Use temporary directory for testing
        self.temp_dir = Path(tempfile.mkdtemp())
        self.reporter = PerformanceReporter(
            reports_dir=self.temp_dir / "reports",
            history_file=self.temp_dir / "history.json",
            max_history_days=30
        )
    
    def teardown_method(self):
        """Clean up test environment."""
        import shutil
        if self.temp_dir.exists():
            shutil.rmtree(self.temp_dir)
    
    def test_record_performance_data(self):
        """Test recording performance data."""
        test_name = "test_function"
        metrics = {
            'execution_time': 1.5,
            'memory_usage_mb': 256.0,
            'throughput': 10.0
        }
        
        # Record data
        self.reporter.record_performance_data(test_name, metrics)
        
        # Verify data was recorded
        assert test_name in self.reporter.performance_history
        assert len(self.reporter.performance_history[test_name]) == 1
        
        recorded_entry = self.reporter.performance_history[test_name][0]
        assert recorded_entry['metrics'] == metrics
        assert 'timestamp' in recorded_entry
        assert 'datetime' in recorded_entry
    
    def test_analyze_performance_trends_insufficient_data(self):
        """Test trend analysis with insufficient data."""
        # Record only 2 data points (need at least 3)
        test_name = "insufficient_data_test"
        for i in range(2):
            metrics = {'execution_time': 1.0 + i * 0.1}
            self.reporter.record_performance_data(test_name, metrics)
            time.sleep(0.01)  # Small delay to ensure different timestamps
        
        trends = self.reporter.analyze_performance_trends(test_name)
        
        # Should not generate trends with insufficient data
        assert len(trends) == 0
    
    def test_analyze_performance_trends_stable(self):
        """Test trend analysis with stable performance."""
        test_name = "stable_test"
        
        # Record stable performance data
        for i in range(5):
            metrics = {'execution_time': 1.0}  # Constant execution time
            self.reporter.record_performance_data(test_name, metrics)
            time.sleep(0.01)
        
        trends = self.reporter.analyze_performance_trends(test_name)
        
        # Should detect stable trend
        trend_key = f"{test_name}_execution_time"
        assert trend_key in trends
        
        trend = trends[trend_key]
        assert trend.trend_direction == 'stable'
        assert trend.trend_strength < 0.3  # Weak trend for stable data
        assert not trend.regression_detected
    
    def test_analyze_performance_trends_regression(self):
        """Test trend analysis with performance regression."""
        test_name = "regression_test"
        
        # Record increasing execution times (regression) - more realistic increase
        base_time = time.time() - 3600  # 1 hour ago
        for i in range(5):
            # Create more realistic timestamps (spread over time)
            timestamp = base_time + i * 600  # 10 minutes apart
            metrics = {'execution_time': 1.0 + i * 0.1}  # Gradual increase: 1.0, 1.1, 1.2, 1.3, 1.4
            
            # Manually add to history with specific timestamp
            if test_name not in self.reporter.performance_history:
                self.reporter.performance_history[test_name] = []
            
            self.reporter.performance_history[test_name].append({
                'timestamp': timestamp,
                'datetime': datetime.fromtimestamp(timestamp).isoformat(),
                'metrics': metrics
            })
        
        trends = self.reporter.analyze_performance_trends(test_name)
        
        trend_key = f"{test_name}_execution_time"
        assert trend_key in trends
        
        trend = trends[trend_key]
        assert trend.trend_direction == 'increasing'
        assert trend.trend_strength > 0.5  # Strong trend
        assert trend.regression_detected  # Should detect regression
    
    def test_analyze_performance_trends_improvement(self):
        """Test trend analysis with performance improvement."""
        test_name = "improvement_test"
        
        # Record decreasing execution times (improvement) with realistic timestamps
        base_time = time.time() - 3600  # 1 hour ago
        for i in range(5):
            timestamp = base_time + i * 600  # 10 minutes apart
            metrics = {'execution_time': 2.0 - i * 0.2}  # Decreasing times: 2.0, 1.8, 1.6, 1.4, 1.2
            
            # Manually add to history with specific timestamp
            if test_name not in self.reporter.performance_history:
                self.reporter.performance_history[test_name] = []
            
            self.reporter.performance_history[test_name].append({
                'timestamp': timestamp,
                'datetime': datetime.fromtimestamp(timestamp).isoformat(),
                'metrics': metrics
            })
        
        trends = self.reporter.analyze_performance_trends(test_name)
        
        trend_key = f"{test_name}_execution_time"
        assert trend_key in trends
        
        trend = trends[trend_key]
        assert trend.trend_direction == 'decreasing'
        assert trend.trend_strength > 0.5
        assert not trend.regression_detected  # Improvement, not regression
    
    def test_generate_performance_report(self):
        """Test performance report generation."""
        # Record some test data
        test_names = ["test_1", "test_2", "test_3"]
        for test_name in test_names:
            for i in range(3):
                metrics = {
                    'execution_time': 1.0 + i * 0.1,
                    'memory_usage_mb': 100.0 + i * 10,
                    'throughput': 5.0 - i * 0.5
                }
                self.reporter.record_performance_data(test_name, metrics)
                time.sleep(0.01)
        
        # Generate report
        report = self.reporter.generate_performance_report(report_period="weekly")
        
        # Verify report structure
        assert isinstance(report, PerformanceReport)
        assert report.report_period == "weekly"
        assert report.total_tests == len(test_names)
        assert len(report.performance_summary) > 0
        assert isinstance(report.trend_analysis, dict)
        assert isinstance(report.regression_alerts, list)
        assert isinstance(report.improvement_highlights, list)
        assert isinstance(report.recommendations, list)
    
    def test_export_performance_data_json(self):
        """Test exporting performance data in JSON format."""
        test_name = "export_test"
        
        # Record test data
        for i in range(3):
            metrics = {'execution_time': 1.0 + i * 0.1}
            self.reporter.record_performance_data(test_name, metrics)
            time.sleep(0.01)
        
        # Export data
        exported_data = self.reporter.export_performance_data(
            format="json",
            test_name=test_name,
            days_back=1
        )
        
        # Verify export
        data = json.loads(exported_data)
        assert test_name in data
        assert len(data[test_name]) == 3
        
        for entry in data[test_name]:
            assert 'timestamp' in entry
            assert 'metrics' in entry
            assert 'execution_time' in entry['metrics']
    
    def test_export_performance_data_csv(self):
        """Test exporting performance data in CSV format."""
        test_name = "csv_export_test"
        
        # Record test data
        metrics = {'execution_time': 1.5, 'memory_usage_mb': 256.0}
        self.reporter.record_performance_data(test_name, metrics)
        
        # Export data
        exported_data = self.reporter.export_performance_data(
            format="csv",
            test_name=test_name,
            days_back=1
        )
        
        # Verify CSV format
        lines = exported_data.strip().split('\n')
        assert len(lines) >= 2  # Header + at least one data row
        
        header = lines[0]
        assert 'test_name' in header
        assert 'metric_name' in header
        assert 'metric_value' in header
        
        # Check data rows
        data_lines = lines[1:]
        assert len(data_lines) == 2  # Two metrics recorded
    
    def test_get_performance_dashboard_data(self):
        """Test getting dashboard data."""
        # Record test data with trends
        test_name = "dashboard_test"
        for i in range(5):
            metrics = {
                'execution_time': 1.0 + i * 0.2,  # Increasing trend
                'memory_usage_mb': 100.0,
                'throughput': 10.0 - i * 0.5  # Decreasing trend
            }
            self.reporter.record_performance_data(test_name, metrics)
            time.sleep(0.01)
        
        # Get dashboard data
        dashboard_data = self.reporter.get_performance_dashboard_data()
        
        # Verify dashboard data structure
        assert 'timestamp' in dashboard_data
        assert 'summary' in dashboard_data
        assert 'trend_charts' in dashboard_data
        assert 'regression_alerts' in dashboard_data
        assert 'top_performing_tests' in dashboard_data
        assert 'slowest_tests' in dashboard_data
        
        # Verify summary data
        summary = dashboard_data['summary']
        assert summary['total_tests_tracked'] == 1
        assert summary['total_trends_analyzed'] > 0
    
    def test_cleanup_old_history(self):
        """Test cleanup of old performance history."""
        test_name = "cleanup_test"
        
        # Record old data (simulate data from 100 days ago)
        old_timestamp = time.time() - (100 * 24 * 60 * 60)
        old_entry = {
            'timestamp': old_timestamp,
            'datetime': datetime.fromtimestamp(old_timestamp).isoformat(),
            'metrics': {'execution_time': 1.0}
        }
        
        # Manually add old data
        self.reporter.performance_history[test_name] = [old_entry]
        
        # Record recent data
        self.reporter.record_performance_data(test_name, {'execution_time': 1.5})
        
        # Trigger cleanup (by saving history)
        self.reporter._save_performance_history()
        
        # Verify old data was cleaned up
        remaining_entries = self.reporter.performance_history[test_name]
        assert len(remaining_entries) == 1  # Only recent entry should remain
        assert remaining_entries[0]['metrics']['execution_time'] == 1.5


@pytest.mark.performance
class TestPerformanceDashboard:
    """Test performance dashboard generation."""
    
    def setup_method(self):
        """Set up test environment."""
        self.temp_dir = Path(tempfile.mkdtemp())
        self.reporter = PerformanceReporter(
            reports_dir=self.temp_dir / "reports",
            history_file=self.temp_dir / "history.json"
        )
        self.dashboard = PerformanceDashboard(self.reporter)
        self.dashboard.dashboard_dir = self.temp_dir / "dashboards"
    
    def teardown_method(self):
        """Clean up test environment."""
        import shutil
        if self.temp_dir.exists():
            shutil.rmtree(self.temp_dir)
    
    def test_generate_comprehensive_dashboard(self):
        """Test comprehensive dashboard generation."""
        # Record some test data
        for i in range(3):
            metrics = {
                'execution_time': 1.0 + i * 0.1,
                'memory_usage_mb': 100.0 + i * 10
            }
            self.reporter.record_performance_data("dashboard_test", metrics)
            time.sleep(0.01)
        
        # Generate dashboard
        dashboard_path = self.dashboard.generate_dashboard("comprehensive", days_back=1)
        
        # Verify dashboard was created
        assert dashboard_path.exists()
        assert dashboard_path.suffix == '.html'
        
        # Verify content contains expected elements
        content = dashboard_path.read_text(encoding='utf-8')
        assert 'AlexPose Performance Dashboard' in content
        assert 'Performance Trends' in content
        assert 'chart.js' in content.lower()
    
    def test_generate_summary_dashboard(self):
        """Test summary dashboard generation."""
        # Record test data
        self.reporter.record_performance_data("summary_test", {'execution_time': 1.0})
        
        # Generate summary dashboard
        dashboard_path = self.dashboard.generate_dashboard("summary", days_back=1)
        
        # Verify dashboard
        assert dashboard_path.exists()
        
        content = dashboard_path.read_text(encoding='utf-8')
        assert 'Performance Summary' in content
        assert 'Tests Monitored' in content
    
    def test_generate_all_dashboards(self):
        """Test generating all dashboard types."""
        # Record test data
        self.reporter.record_performance_data("all_dashboards_test", {'execution_time': 1.0})
        
        # Generate all dashboards
        dashboards = self.dashboard.generate_all_dashboards(days_back=1)
        
        # Verify all dashboard types were generated
        expected_types = ['comprehensive', 'summary', 'trends']
        assert len(dashboards) == len(expected_types)
        
        for dashboard_type in expected_types:
            assert dashboard_type in dashboards
            assert dashboards[dashboard_type].exists()


@pytest.mark.performance
class TestPerformanceIntegration:
    """Test performance testing integration."""
    
    def setup_method(self):
        """Set up test environment."""
        self.temp_dir = Path(tempfile.mkdtemp())
        
        # Create integration with temporary directory
        self.integration = PerformanceTestIntegration(
            enable_reporting=True,
            enable_dashboards=True
        )
        
        # Override paths for testing
        self.integration.reporter.reports_dir = self.temp_dir / "reports"
        self.integration.reporter.history_file = self.temp_dir / "history.json"
        self.integration.dashboard.dashboard_dir = self.temp_dir / "dashboards"
    
    def teardown_method(self):
        """Clean up test environment."""
        import shutil
        if self.temp_dir.exists():
            shutil.rmtree(self.temp_dir)
    
    def test_run_performance_test(self):
        """Test running performance test with integration."""
        def sample_test_function():
            time.sleep(0.1)
            return "test completed"
        
        # Run performance test
        result = self.integration.run_performance_test(
            "integration_test",
            sample_test_function,
            iterations=1
        )
        
        # Verify result structure
        assert 'test_name' in result
        assert 'metrics' in result
        assert 'regression_analysis' in result
        assert 'timestamp' in result
        
        assert result['test_name'] == "integration_test"
        assert result['metrics'].execution_time > 0.05  # Should be at least 0.1s
        
        # Verify data was recorded
        assert "integration_test" in self.integration.reporter.performance_history
    
    def test_run_concurrent_performance_test(self):
        """Test running concurrent performance test."""
        def concurrent_test_function():
            time.sleep(0.05)
            return "concurrent test completed"
        
        # Run concurrent performance test
        result = self.integration.run_concurrent_performance_test(
            "concurrent_test",
            concurrent_test_function,
            num_concurrent=3
        )
        
        # Verify result
        assert 'test_name' in result
        assert 'concurrent_details' in result
        assert 'metrics' in result
        
        concurrent_details = result['concurrent_details']
        assert concurrent_details['concurrent_operations'] == 3
        assert concurrent_details['success_rate'] > 0
    
    def test_generate_test_session_report(self):
        """Test generating test session report."""
        # Run some performance tests
        def test_func():
            time.sleep(0.01)
        
        self.integration.run_performance_test("session_test_1", test_func)
        self.integration.run_performance_test("session_test_2", test_func)
        
        # Generate session report
        report_path = self.integration.generate_test_session_report()
        
        # Verify report was generated
        assert report_path is not None
        assert report_path.exists()
    
    def test_generate_test_session_dashboard(self):
        """Test generating test session dashboard."""
        # Run performance test
        def test_func():
            time.sleep(0.01)
        
        self.integration.run_performance_test("dashboard_session_test", test_func)
        
        # Generate session dashboard
        dashboard_path = self.integration.generate_test_session_dashboard()
        
        # Verify dashboard was generated
        assert dashboard_path is not None
        assert dashboard_path.exists()
        assert dashboard_path.suffix == '.html'


@pytest.mark.performance
class TestPerformanceTrendAnalysis:
    """Test specific trend analysis functionality."""
    
    def setup_method(self):
        """Set up test environment."""
        self.temp_dir = Path(tempfile.mkdtemp())
        self.reporter = PerformanceReporter(
            reports_dir=self.temp_dir / "reports",
            history_file=self.temp_dir / "history.json"
        )
    
    def teardown_method(self):
        """Clean up test environment."""
        import shutil
        if self.temp_dir.exists():
            shutil.rmtree(self.temp_dir)
    
    def test_trend_calculation_linear_increase(self):
        """Test trend calculation for linear increase."""
        test_name = "linear_increase_test"
        
        # Create linear increasing data
        base_time = time.time() - 3600  # 1 hour ago
        for i in range(10):
            timestamp = base_time + i * 360  # Every 6 minutes
            metrics = {'execution_time': 1.0 + i * 0.1}  # Linear increase
            
            # Manually add to history with specific timestamp
            if test_name not in self.reporter.performance_history:
                self.reporter.performance_history[test_name] = []
            
            self.reporter.performance_history[test_name].append({
                'timestamp': timestamp,
                'datetime': datetime.fromtimestamp(timestamp).isoformat(),
                'metrics': metrics
            })
        
        # Analyze trends
        trends = self.reporter.analyze_performance_trends(test_name, days_back=1)
        
        # Verify linear trend detection
        trend_key = f"{test_name}_execution_time"
        assert trend_key in trends
        
        trend = trends[trend_key]
        assert trend.trend_direction == 'increasing'
        assert trend.trend_strength > 0.8  # Should be strong linear correlation
        assert trend.regression_detected
    
    def test_trend_calculation_noisy_data(self):
        """Test trend calculation with noisy data."""
        test_name = "noisy_data_test"
        
        # Create noisy data around a stable mean
        import random
        base_time = time.time() - 3600
        
        for i in range(15):
            timestamp = base_time + i * 240  # Every 4 minutes
            # Add random noise around 1.0
            execution_time = 1.0 + random.uniform(-0.2, 0.2)
            metrics = {'execution_time': execution_time}
            
            if test_name not in self.reporter.performance_history:
                self.reporter.performance_history[test_name] = []
            
            self.reporter.performance_history[test_name].append({
                'timestamp': timestamp,
                'datetime': datetime.fromtimestamp(timestamp).isoformat(),
                'metrics': metrics
            })
        
        # Analyze trends
        trends = self.reporter.analyze_performance_trends(test_name, days_back=1)
        
        # Verify noisy data handling
        trend_key = f"{test_name}_execution_time"
        assert trend_key in trends
        
        trend = trends[trend_key]
        # Should detect stable or weak trend due to noise
        assert trend.trend_strength < 0.5 or trend.trend_direction == 'stable'
        assert not trend.regression_detected  # Noise shouldn't trigger regression
    
    def test_multiple_metrics_trend_analysis(self):
        """Test trend analysis with multiple metrics."""
        test_name = "multi_metric_test"
        
        # Record data with different trends for different metrics
        base_time = time.time() - 1800  # 30 minutes ago
        for i in range(8):
            timestamp = base_time + i * 225  # Every 3.75 minutes
            metrics = {
                'execution_time': 1.0 + i * 0.05,  # Slight increase
                'memory_usage_mb': 200.0 - i * 5,  # Decrease (improvement)
                'throughput': 10.0 + i * 0.2  # Increase (improvement)
            }
            
            if test_name not in self.reporter.performance_history:
                self.reporter.performance_history[test_name] = []
            
            self.reporter.performance_history[test_name].append({
                'timestamp': timestamp,
                'datetime': datetime.fromtimestamp(timestamp).isoformat(),
                'metrics': metrics
            })
        
        # Analyze trends
        trends = self.reporter.analyze_performance_trends(test_name, days_back=1)
        
        # Should have trends for all three metrics
        expected_trends = [
            f"{test_name}_execution_time",
            f"{test_name}_memory_usage_mb", 
            f"{test_name}_throughput"
        ]
        
        for expected_trend in expected_trends:
            assert expected_trend in trends
        
        # Verify trend directions
        exec_time_trend = trends[f"{test_name}_execution_time"]
        memory_trend = trends[f"{test_name}_memory_usage_mb"]
        throughput_trend = trends[f"{test_name}_throughput"]
        
        assert exec_time_trend.trend_direction == 'increasing'
        assert memory_trend.trend_direction == 'decreasing'
        assert throughput_trend.trend_direction == 'increasing'
        
        # Memory decrease should not be flagged as regression (it's improvement)
        assert not memory_trend.regression_detected
        # Throughput increase should not be flagged as regression (it's improvement)
        assert not throughput_trend.regression_detected


if __name__ == "__main__":
    pytest.main([__file__, "-v"])