"""Performance reporting and trend analysis for AlexPose system."""

import json
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, asdict
import statistics
import logging

logger = logging.getLogger(__name__)

@dataclass
class PerformanceTrend:
    """Performance trend data for a specific metric."""
    metric_name: str
    test_name: str
    values: List[float]
    timestamps: List[float]
    trend_direction: str  # 'improving', 'degrading', 'stable'
    trend_strength: float  # 0.0 to 1.0
    regression_detected: bool
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'PerformanceTrend':
        """Create from dictionary."""
        return cls(**data)

@dataclass
class PerformanceReport:
    """Comprehensive performance report."""
    timestamp: float
    report_period: str
    total_tests: int
    performance_summary: Dict[str, Any]
    trend_analysis: Dict[str, PerformanceTrend]
    regression_alerts: List[Dict[str, Any]]
    improvement_highlights: List[Dict[str, Any]]
    recommendations: List[str]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        data = asdict(self)
        # Convert PerformanceTrend objects to dictionaries
        data['trend_analysis'] = {
            name: trend.to_dict() for name, trend in self.trend_analysis.items()
        }
        return data
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'PerformanceReport':
        """Create from dictionary."""
        # Convert trend analysis dictionaries back to PerformanceTrend objects
        trend_analysis = {
            name: PerformanceTrend.from_dict(trend_data) 
            for name, trend_data in data.get('trend_analysis', {}).items()
        }
        data['trend_analysis'] = trend_analysis
        return cls(**data)

class PerformanceReporter:
    """Advanced performance reporting and trend analysis system."""
    
    def __init__(self, 
                 reports_dir: Path = Path("tests/performance/reports"),
                 history_file: Path = Path("tests/performance/performance_history.json"),
                 max_history_days: int = 90):
        self.reports_dir = reports_dir
        self.history_file = history_file
        self.max_history_days = max_history_days
        
        # Ensure directories exist
        self.reports_dir.mkdir(parents=True, exist_ok=True)
        self.history_file.parent.mkdir(parents=True, exist_ok=True)
        
        # Load historical data
        self.performance_history = self._load_performance_history()
    
    def _load_performance_history(self) -> Dict[str, List[Dict[str, Any]]]:
        """Load historical performance data."""
        if self.history_file.exists():
            try:
                with open(self.history_file, 'r') as f:
                    return json.load(f)
            except Exception as e:
                logger.warning(f"Failed to load performance history: {e}")
        
        return {}
    
    def _save_performance_history(self):
        """Save performance history to file."""
        try:
            # Clean old data before saving
            self._cleanup_old_history()
            
            with open(self.history_file, 'w') as f:
                json.dump(self.performance_history, f, indent=2)
            logger.info(f"Saved performance history with {len(self.performance_history)} test entries")
        except Exception as e:
            logger.error(f"Failed to save performance history: {e}")
    
    def _cleanup_old_history(self):
        """Remove performance data older than max_history_days."""
        cutoff_time = time.time() - (self.max_history_days * 24 * 60 * 60)
        
        for test_name in list(self.performance_history.keys()):
            # Filter out old entries
            self.performance_history[test_name] = [
                entry for entry in self.performance_history[test_name]
                if entry.get('timestamp', 0) > cutoff_time
            ]
            
            # Remove empty test entries
            if not self.performance_history[test_name]:
                del self.performance_history[test_name]
    
    def record_performance_data(self, test_name: str, metrics: Dict[str, float]):
        """Record performance data for trend analysis."""
        timestamp = time.time()
        
        # Initialize test history if needed
        if test_name not in self.performance_history:
            self.performance_history[test_name] = []
        
        # Add new performance data
        performance_entry = {
            'timestamp': timestamp,
            'datetime': datetime.fromtimestamp(timestamp).isoformat(),
            'metrics': metrics
        }
        
        self.performance_history[test_name].append(performance_entry)
        
        # Save updated history
        self._save_performance_history()
        
        logger.info(f"Recorded performance data for {test_name}: {metrics}")
    
    def analyze_performance_trends(self, 
                                 test_name: Optional[str] = None,
                                 days_back: int = 30) -> Dict[str, PerformanceTrend]:
        """Analyze performance trends for tests."""
        cutoff_time = time.time() - (days_back * 24 * 60 * 60)
        trends = {}
        
        # Determine which tests to analyze
        tests_to_analyze = [test_name] if test_name else list(self.performance_history.keys())
        
        for test in tests_to_analyze:
            if test not in self.performance_history:
                continue
            
            # Get recent data
            recent_data = [
                entry for entry in self.performance_history[test]
                if entry.get('timestamp', 0) > cutoff_time
            ]
            
            if len(recent_data) < 3:  # Need at least 3 data points for trend analysis
                continue
            
            # Analyze trends for each metric
            for metric_name in recent_data[0].get('metrics', {}).keys():
                trend = self._calculate_trend(test, metric_name, recent_data)
                if trend:
                    trends[f"{test}_{metric_name}"] = trend
        
        return trends
    
    def _calculate_trend(self, test_name: str, metric_name: str, data: List[Dict[str, Any]]) -> Optional[PerformanceTrend]:
        """Calculate trend for a specific metric."""
        try:
            # Extract values and timestamps
            values = []
            timestamps = []
            
            for entry in data:
                if metric_name in entry.get('metrics', {}):
                    values.append(entry['metrics'][metric_name])
                    timestamps.append(entry['timestamp'])
            
            if len(values) < 3:
                return None
            
            # Calculate trend using linear regression
            trend_direction, trend_strength = self._calculate_linear_trend(values, timestamps)
            
            # Detect regression based on metric type
            regression_detected = self._detect_regression(metric_name, values, trend_direction, trend_strength)
            
            return PerformanceTrend(
                metric_name=metric_name,
                test_name=test_name,
                values=values,
                timestamps=timestamps,
                trend_direction=trend_direction,
                trend_strength=trend_strength,
                regression_detected=regression_detected
            )
        
        except Exception as e:
            logger.error(f"Failed to calculate trend for {test_name}.{metric_name}: {e}")
            return None
    
    def _calculate_linear_trend(self, values: List[float], timestamps: List[float]) -> Tuple[str, float]:
        """Calculate linear trend direction and strength."""
        if len(values) < 2:
            return 'stable', 0.0
        
        # Simple linear regression
        n = len(values)
        sum_x = sum(timestamps)
        sum_y = sum(values)
        sum_xy = sum(x * y for x, y in zip(timestamps, values))
        sum_x2 = sum(x * x for x in timestamps)
        
        # Calculate slope
        denominator = n * sum_x2 - sum_x * sum_x
        if denominator == 0:
            return 'stable', 0.0
        
        slope = (n * sum_xy - sum_x * sum_y) / denominator
        
        # Calculate correlation coefficient for trend strength
        mean_x = sum_x / n
        mean_y = sum_y / n
        
        numerator = sum((x - mean_x) * (y - mean_y) for x, y in zip(timestamps, values))
        denominator_x = sum((x - mean_x) ** 2 for x in timestamps)
        denominator_y = sum((y - mean_y) ** 2 for y in values)
        
        if denominator_x == 0 or denominator_y == 0:
            correlation = 0.0
        else:
            correlation = numerator / (denominator_x * denominator_y) ** 0.5
        
        # Determine trend direction based on correlation and value change
        # Use correlation strength to determine if trend is significant
        strength = abs(correlation)
        
        if strength < 0.3:  # Weak correlation indicates stable trend
            direction = 'stable'
        else:
            # For strong correlations, check the direction of value change
            first_value = values[0]
            last_value = values[-1]
            value_change_percent = (last_value - first_value) / first_value if first_value != 0 else 0
            
            # Consider trend significant if change is > 5%
            if abs(value_change_percent) < 0.05:
                direction = 'stable'
            elif value_change_percent > 0:
                direction = 'increasing'
            else:
                direction = 'decreasing'
        
        return direction, strength
    
    def _detect_regression(self, metric_name: str, values: List[float], 
                          trend_direction: str, trend_strength: float) -> bool:
        """Detect if there's a performance regression."""
        if len(values) < 3 or trend_strength < 0.3:  # Weak trend
            return False
        
        # For performance metrics, increasing values are usually bad
        performance_metrics = ['execution_time', 'memory_usage_mb', 'peak_memory_mb']
        
        # For throughput metrics, decreasing values are bad
        throughput_metrics = ['throughput', 'operations_per_second', 'requests_per_second']
        
        if any(perf_metric in metric_name.lower() for perf_metric in performance_metrics):
            # Regression if values are increasing significantly
            return trend_direction == 'increasing' and trend_strength > 0.5
        
        elif any(throughput_metric in metric_name.lower() for throughput_metric in throughput_metrics):
            # Regression if throughput is decreasing significantly
            return trend_direction == 'decreasing' and trend_strength > 0.5
        
        # For unknown metrics, use conservative approach
        return False
    
    def generate_performance_report(self, 
                                  report_period: str = "weekly",
                                  include_trends: bool = True) -> PerformanceReport:
        """Generate comprehensive performance report."""
        timestamp = time.time()
        
        # Determine analysis period
        if report_period == "daily":
            days_back = 1
        elif report_period == "weekly":
            days_back = 7
        elif report_period == "monthly":
            days_back = 30
        else:
            days_back = 7  # Default to weekly
        
        # Analyze trends
        trends = self.analyze_performance_trends(days_back=days_back) if include_trends else {}
        
        # Generate performance summary
        performance_summary = self._generate_performance_summary(days_back)
        
        # Identify regressions and improvements
        regression_alerts = []
        improvement_highlights = []
        
        for trend_key, trend in trends.items():
            if trend.regression_detected:
                regression_alerts.append({
                    'test_name': trend.test_name,
                    'metric': trend.metric_name,
                    'trend_direction': trend.trend_direction,
                    'trend_strength': trend.trend_strength,
                    'latest_value': trend.values[-1] if trend.values else 0,
                    'severity': 'high' if trend.trend_strength > 0.7 else 'medium'
                })
            
            # Check for improvements (opposite of regression logic)
            elif self._detect_improvement(trend.metric_name, trend.trend_direction, trend.trend_strength):
                improvement_highlights.append({
                    'test_name': trend.test_name,
                    'metric': trend.metric_name,
                    'trend_direction': trend.trend_direction,
                    'trend_strength': trend.trend_strength,
                    'improvement_percent': self._calculate_improvement_percent(trend.values)
                })
        
        # Generate recommendations
        recommendations = self._generate_recommendations(regression_alerts, improvement_highlights, trends)
        
        report = PerformanceReport(
            timestamp=timestamp,
            report_period=report_period,
            total_tests=len(self.performance_history),
            performance_summary=performance_summary,
            trend_analysis=trends,
            regression_alerts=regression_alerts,
            improvement_highlights=improvement_highlights,
            recommendations=recommendations
        )
        
        # Save report
        self._save_report(report)
        
        return report
    
    def _generate_performance_summary(self, days_back: int) -> Dict[str, Any]:
        """Generate performance summary statistics."""
        cutoff_time = time.time() - (days_back * 24 * 60 * 60)
        
        summary = {
            'period_days': days_back,
            'total_test_runs': 0,
            'avg_execution_time': 0.0,
            'avg_memory_usage': 0.0,
            'avg_throughput': 0.0,
            'test_statistics': {}
        }
        
        all_execution_times = []
        all_memory_usage = []
        all_throughput = []
        
        for test_name, history in self.performance_history.items():
            recent_runs = [
                entry for entry in history
                if entry.get('timestamp', 0) > cutoff_time
            ]
            
            if not recent_runs:
                continue
            
            summary['total_test_runs'] += len(recent_runs)
            
            # Collect metrics for overall averages
            for run in recent_runs:
                metrics = run.get('metrics', {})
                if 'execution_time' in metrics:
                    all_execution_times.append(metrics['execution_time'])
                if 'memory_usage_mb' in metrics or 'peak_memory_mb' in metrics:
                    memory_val = metrics.get('memory_usage_mb', metrics.get('peak_memory_mb', 0))
                    all_memory_usage.append(memory_val)
                if 'throughput' in metrics:
                    all_throughput.append(metrics['throughput'])
            
            # Test-specific statistics
            test_execution_times = [
                run['metrics'].get('execution_time', 0) for run in recent_runs
                if 'execution_time' in run.get('metrics', {})
            ]
            
            if test_execution_times:
                summary['test_statistics'][test_name] = {
                    'runs': len(recent_runs),
                    'avg_execution_time': statistics.mean(test_execution_times),
                    'min_execution_time': min(test_execution_times),
                    'max_execution_time': max(test_execution_times),
                    'std_execution_time': statistics.stdev(test_execution_times) if len(test_execution_times) > 1 else 0
                }
        
        # Calculate overall averages
        if all_execution_times:
            summary['avg_execution_time'] = statistics.mean(all_execution_times)
        if all_memory_usage:
            summary['avg_memory_usage'] = statistics.mean(all_memory_usage)
        if all_throughput:
            summary['avg_throughput'] = statistics.mean(all_throughput)
        
        return summary
    
    def _detect_improvement(self, metric_name: str, trend_direction: str, trend_strength: float) -> bool:
        """Detect if there's a performance improvement."""
        if trend_strength < 0.3:  # Weak trend
            return False
        
        # For performance metrics, decreasing values are good
        performance_metrics = ['execution_time', 'memory_usage_mb', 'peak_memory_mb']
        
        # For throughput metrics, increasing values are good
        throughput_metrics = ['throughput', 'operations_per_second', 'requests_per_second']
        
        if any(perf_metric in metric_name.lower() for perf_metric in performance_metrics):
            return trend_direction == 'decreasing' and trend_strength > 0.5
        
        elif any(throughput_metric in metric_name.lower() for throughput_metric in throughput_metrics):
            return trend_direction == 'increasing' and trend_strength > 0.5
        
        return False
    
    def _calculate_improvement_percent(self, values: List[float]) -> float:
        """Calculate improvement percentage from first to last value."""
        if len(values) < 2:
            return 0.0
        
        first_val = values[0]
        last_val = values[-1]
        
        if first_val == 0:
            return 0.0
        
        return abs((first_val - last_val) / first_val * 100)
    
    def _generate_recommendations(self, 
                                regression_alerts: List[Dict[str, Any]],
                                improvement_highlights: List[Dict[str, Any]],
                                trends: Dict[str, PerformanceTrend]) -> List[str]:
        """Generate actionable recommendations based on analysis."""
        recommendations = []
        
        # Regression-based recommendations
        if regression_alerts:
            high_severity_regressions = [alert for alert in regression_alerts if alert['severity'] == 'high']
            
            if high_severity_regressions:
                recommendations.append(
                    f"URGENT: {len(high_severity_regressions)} high-severity performance regressions detected. "
                    "Immediate investigation recommended."
                )
            
            memory_regressions = [alert for alert in regression_alerts if 'memory' in alert['metric']]
            if memory_regressions:
                recommendations.append(
                    "Memory usage regressions detected. Check for memory leaks or inefficient data structures."
                )
            
            time_regressions = [alert for alert in regression_alerts if 'time' in alert['metric']]
            if time_regressions:
                recommendations.append(
                    "Execution time regressions detected. Profile code for performance bottlenecks."
                )
        
        # Improvement-based recommendations
        if improvement_highlights:
            recommendations.append(
                f"Great news! {len(improvement_highlights)} performance improvements detected. "
                "Consider documenting optimization techniques for future reference."
            )
        
        # Trend-based recommendations
        stable_trends = [trend for trend in trends.values() if trend.trend_direction == 'stable']
        if len(stable_trends) > len(trends) * 0.8:  # 80% stable
            recommendations.append(
                "Performance is generally stable. Consider establishing tighter performance targets."
            )
        
        # Data quality recommendations
        sparse_data_tests = []
        for test_name in self.performance_history:
            recent_count = len([
                entry for entry in self.performance_history[test_name]
                if entry.get('timestamp', 0) > time.time() - (7 * 24 * 60 * 60)
            ])
            if recent_count < 3:
                sparse_data_tests.append(test_name)
        
        if sparse_data_tests:
            recommendations.append(
                f"Insufficient performance data for {len(sparse_data_tests)} tests. "
                "Consider running performance tests more frequently."
            )
        
        # Default recommendation if no specific issues
        if not recommendations:
            recommendations.append(
                "Performance monitoring is active. Continue regular performance testing to maintain system health."
            )
        
        return recommendations
    
    def _save_report(self, report: PerformanceReport):
        """Save performance report to file."""
        timestamp_str = datetime.fromtimestamp(report.timestamp).strftime("%Y%m%d_%H%M%S")
        report_filename = f"performance_report_{report.report_period}_{timestamp_str}.json"
        report_path = self.reports_dir / report_filename
        
        try:
            with open(report_path, 'w') as f:
                json.dump(report.to_dict(), f, indent=2)
            
            # Also save as latest report
            latest_path = self.reports_dir / f"latest_{report.report_period}_report.json"
            with open(latest_path, 'w') as f:
                json.dump(report.to_dict(), f, indent=2)
            
            logger.info(f"Performance report saved to {report_path}")
        
        except Exception as e:
            logger.error(f"Failed to save performance report: {e}")
    
    def get_performance_dashboard_data(self) -> Dict[str, Any]:
        """Get data for performance dashboard visualization."""
        recent_trends = self.analyze_performance_trends(days_back=30)
        
        dashboard_data = {
            'timestamp': time.time(),
            'summary': {
                'total_tests_tracked': len(self.performance_history),
                'total_trends_analyzed': len(recent_trends),
                'regressions_detected': len([t for t in recent_trends.values() if t.regression_detected]),
                'data_points_total': sum(len(history) for history in self.performance_history.values())
            },
            'trend_charts': {},
            'regression_alerts': [],
            'top_performing_tests': [],
            'slowest_tests': []
        }
        
        # Prepare trend chart data
        for trend_key, trend in recent_trends.items():
            if len(trend.values) >= 5:  # Only include trends with sufficient data
                dashboard_data['trend_charts'][trend_key] = {
                    'test_name': trend.test_name,
                    'metric_name': trend.metric_name,
                    'timestamps': [datetime.fromtimestamp(ts).isoformat() for ts in trend.timestamps],
                    'values': trend.values,
                    'trend_direction': trend.trend_direction,
                    'trend_strength': trend.trend_strength
                }
        
        # Collect regression alerts
        for trend in recent_trends.values():
            if trend.regression_detected:
                dashboard_data['regression_alerts'].append({
                    'test_name': trend.test_name,
                    'metric': trend.metric_name,
                    'latest_value': trend.values[-1] if trend.values else 0,
                    'trend_strength': trend.trend_strength
                })
        
        # Find top performing and slowest tests
        latest_performance = {}
        for test_name, history in self.performance_history.items():
            if history:
                latest_run = max(history, key=lambda x: x.get('timestamp', 0))
                execution_time = latest_run.get('metrics', {}).get('execution_time', 0)
                if execution_time > 0:
                    latest_performance[test_name] = execution_time
        
        if latest_performance:
            # Top performing (fastest) tests
            fastest_tests = sorted(latest_performance.items(), key=lambda x: x[1])[:5]
            dashboard_data['top_performing_tests'] = [
                {'test_name': name, 'execution_time': time} for name, time in fastest_tests
            ]
            
            # Slowest tests
            slowest_tests = sorted(latest_performance.items(), key=lambda x: x[1], reverse=True)[:5]
            dashboard_data['slowest_tests'] = [
                {'test_name': name, 'execution_time': time} for name, time in slowest_tests
            ]
        
        return dashboard_data
    
    def export_performance_data(self, 
                               format: str = "json",
                               test_name: Optional[str] = None,
                               days_back: int = 30) -> str:
        """Export performance data in various formats."""
        cutoff_time = time.time() - (days_back * 24 * 60 * 60)
        
        # Filter data
        if test_name:
            if test_name in self.performance_history:
                filtered_data = {test_name: self.performance_history[test_name]}
            else:
                filtered_data = {}
        else:
            filtered_data = self.performance_history
        
        # Apply time filter
        export_data = {}
        for test, history in filtered_data.items():
            recent_history = [
                entry for entry in history
                if entry.get('timestamp', 0) > cutoff_time
            ]
            if recent_history:
                export_data[test] = recent_history
        
        # Export in requested format
        if format.lower() == "json":
            return json.dumps(export_data, indent=2)
        elif format.lower() == "csv":
            return self._export_to_csv(export_data)
        else:
            raise ValueError(f"Unsupported export format: {format}")
    
    def _export_to_csv(self, data: Dict[str, List[Dict[str, Any]]]) -> str:
        """Export performance data to CSV format."""
        import csv
        import io
        
        output = io.StringIO()
        writer = csv.writer(output)
        
        # Write header
        writer.writerow(['test_name', 'timestamp', 'datetime', 'metric_name', 'metric_value'])
        
        # Write data
        for test_name, history in data.items():
            for entry in history:
                timestamp = entry.get('timestamp', 0)
                datetime_str = entry.get('datetime', '')
                
                for metric_name, metric_value in entry.get('metrics', {}).items():
                    writer.writerow([test_name, timestamp, datetime_str, metric_name, metric_value])
        
        return output.getvalue()