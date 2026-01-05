#!/usr/bin/env python3
"""Command-line interface for performance reporting and trend analysis."""

import argparse
import json
import sys
from pathlib import Path
from typing import Optional

from .performance_reporter import PerformanceReporter
from .performance_dashboard import PerformanceDashboard


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="AlexPose Performance Reporting and Analysis Tool",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Generate weekly performance report
  python -m tests.performance.performance_cli report --period weekly

  # Generate comprehensive dashboard
  python -m tests.performance.performance_cli dashboard --type comprehensive

  # Analyze trends for specific test
  python -m tests.performance.performance_cli trends --test video_processing_30s

  # Export performance data to CSV
  python -m tests.performance.performance_cli export --format csv --days 30

  # Record performance data (for integration with test runs)
  python -m tests.performance.performance_cli record --test my_test --execution-time 1.5 --memory 256
        """
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Report command
    report_parser = subparsers.add_parser('report', help='Generate performance report')
    report_parser.add_argument('--period', choices=['daily', 'weekly', 'monthly'], 
                              default='weekly', help='Report period')
    report_parser.add_argument('--output', type=str, help='Output file path')
    report_parser.add_argument('--format', choices=['json', 'text'], default='json',
                              help='Output format')
    
    # Dashboard command
    dashboard_parser = subparsers.add_parser('dashboard', help='Generate performance dashboard')
    dashboard_parser.add_argument('--type', choices=['comprehensive', 'summary', 'trends'],
                                 default='comprehensive', help='Dashboard type')
    dashboard_parser.add_argument('--days', type=int, default=30, help='Days of data to include')
    dashboard_parser.add_argument('--open', action='store_true', help='Open dashboard in browser')
    
    # Trends command
    trends_parser = subparsers.add_parser('trends', help='Analyze performance trends')
    trends_parser.add_argument('--test', type=str, help='Specific test to analyze')
    trends_parser.add_argument('--days', type=int, default=30, help='Days of data to analyze')
    trends_parser.add_argument('--format', choices=['json', 'text'], default='text',
                              help='Output format')
    
    # Export command
    export_parser = subparsers.add_parser('export', help='Export performance data')
    export_parser.add_argument('--format', choices=['json', 'csv'], default='json',
                              help='Export format')
    export_parser.add_argument('--test', type=str, help='Specific test to export')
    export_parser.add_argument('--days', type=int, default=30, help='Days of data to export')
    export_parser.add_argument('--output', type=str, help='Output file path')
    
    # Record command (for integration with test runs)
    record_parser = subparsers.add_parser('record', help='Record performance data')
    record_parser.add_argument('--test', type=str, required=True, help='Test name')
    record_parser.add_argument('--execution-time', type=float, help='Execution time in seconds')
    record_parser.add_argument('--memory', type=float, help='Memory usage in MB')
    record_parser.add_argument('--throughput', type=float, help='Throughput (ops/sec)')
    record_parser.add_argument('--cpu', type=float, help='CPU usage percentage')
    
    # Status command
    status_parser = subparsers.add_parser('status', help='Show performance monitoring status')
    status_parser.add_argument('--verbose', action='store_true', help='Verbose output')
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return 1
    
    try:
        reporter = PerformanceReporter()
        
        if args.command == 'report':
            return handle_report_command(reporter, args)
        elif args.command == 'dashboard':
            return handle_dashboard_command(reporter, args)
        elif args.command == 'trends':
            return handle_trends_command(reporter, args)
        elif args.command == 'export':
            return handle_export_command(reporter, args)
        elif args.command == 'record':
            return handle_record_command(reporter, args)
        elif args.command == 'status':
            return handle_status_command(reporter, args)
        else:
            print(f"Unknown command: {args.command}")
            return 1
    
    except Exception as e:
        print(f"Error: {e}")
        return 1


def handle_report_command(reporter: PerformanceReporter, args) -> int:
    """Handle report generation command."""
    print(f"Generating {args.period} performance report...")
    
    report = reporter.generate_performance_report(report_period=args.period)
    
    if args.format == 'json':
        output_data = json.dumps(report.to_dict(), indent=2)
    else:
        output_data = format_report_as_text(report)
    
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w') as f:
            f.write(output_data)
        print(f"Report saved to: {output_path}")
    else:
        print(output_data)
    
    return 0


def handle_dashboard_command(reporter: PerformanceReporter, args) -> int:
    """Handle dashboard generation command."""
    print(f"Generating {args.type} performance dashboard...")
    
    dashboard = PerformanceDashboard(reporter)
    dashboard_path = dashboard.generate_dashboard(args.type, args.days)
    
    print(f"Dashboard generated: {dashboard_path}")
    
    if args.open:
        try:
            import webbrowser
            webbrowser.open(f"file://{dashboard_path.absolute()}")
            print("Dashboard opened in browser")
        except Exception as e:
            print(f"Failed to open dashboard in browser: {e}")
    
    return 0


def handle_trends_command(reporter: PerformanceReporter, args) -> int:
    """Handle trends analysis command."""
    print(f"Analyzing performance trends...")
    
    trends = reporter.analyze_performance_trends(
        test_name=args.test,
        days_back=args.days
    )
    
    if not trends:
        print("No trend data available for the specified criteria.")
        return 0
    
    if args.format == 'json':
        output_data = json.dumps(
            {name: trend.to_dict() for name, trend in trends.items()},
            indent=2
        )
        print(output_data)
    else:
        print(format_trends_as_text(trends))
    
    return 0


def handle_export_command(reporter: PerformanceReporter, args) -> int:
    """Handle data export command."""
    print(f"Exporting performance data in {args.format} format...")
    
    export_data = reporter.export_performance_data(
        format=args.format,
        test_name=args.test,
        days_back=args.days
    )
    
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w') as f:
            f.write(export_data)
        print(f"Data exported to: {output_path}")
    else:
        print(export_data)
    
    return 0


def handle_record_command(reporter: PerformanceReporter, args) -> int:
    """Handle performance data recording command."""
    metrics = {}
    
    if args.execution_time is not None:
        metrics['execution_time'] = args.execution_time
    if args.memory is not None:
        metrics['memory_usage_mb'] = args.memory
    if args.throughput is not None:
        metrics['throughput'] = args.throughput
    if args.cpu is not None:
        metrics['cpu_usage_percent'] = args.cpu
    
    if not metrics:
        print("No metrics provided. Use --execution-time, --memory, --throughput, or --cpu")
        return 1
    
    reporter.record_performance_data(args.test, metrics)
    print(f"Performance data recorded for test: {args.test}")
    print(f"Metrics: {metrics}")
    
    return 0


def handle_status_command(reporter: PerformanceReporter, args) -> int:
    """Handle status command."""
    dashboard_data = reporter.get_performance_dashboard_data()
    summary = dashboard_data.get('summary', {})
    
    print("📊 Performance Monitoring Status")
    print("=" * 40)
    print(f"Tests Tracked: {summary.get('total_tests_tracked', 0)}")
    print(f"Trends Analyzed: {summary.get('total_trends_analyzed', 0)}")
    print(f"Regressions Detected: {summary.get('regressions_detected', 0)}")
    print(f"Total Data Points: {summary.get('data_points_total', 0)}")
    
    if args.verbose:
        print("\n📈 Recent Activity:")
        
        # Show recent regression alerts
        alerts = dashboard_data.get('regression_alerts', [])
        if alerts:
            print(f"\n⚠️  Active Regression Alerts ({len(alerts)}):")
            for alert in alerts[:5]:  # Show first 5
                print(f"  - {alert['test_name']}: {alert['metric']} (strength: {alert['trend_strength']:.2f})")
        
        # Show top performing tests
        top_tests = dashboard_data.get('top_performing_tests', [])
        if top_tests:
            print(f"\n⚡ Top Performing Tests:")
            for test in top_tests[:3]:  # Show top 3
                print(f"  - {test['test_name']}: {test['execution_time']:.3f}s")
        
        # Show slowest tests
        slow_tests = dashboard_data.get('slowest_tests', [])
        if slow_tests:
            print(f"\n🐌 Slowest Tests:")
            for test in slow_tests[:3]:  # Show top 3
                print(f"  - {test['test_name']}: {test['execution_time']:.3f}s")
    
    print(f"\n📁 Data Location: {reporter.history_file}")
    print(f"📁 Reports Location: {reporter.reports_dir}")
    
    return 0


def format_report_as_text(report) -> str:
    """Format performance report as human-readable text."""
    from datetime import datetime
    
    text = []
    text.append("🚀 AlexPose Performance Report")
    text.append("=" * 50)
    text.append(f"Generated: {datetime.fromtimestamp(report.timestamp).strftime('%Y-%m-%d %H:%M:%S')}")
    text.append(f"Period: {report.report_period}")
    text.append(f"Total Tests: {report.total_tests}")
    text.append("")
    
    # Performance Summary
    text.append("📊 Performance Summary")
    text.append("-" * 30)
    summary = report.performance_summary
    text.append(f"Period: {summary.get('period_days', 0)} days")
    text.append(f"Total Test Runs: {summary.get('total_test_runs', 0)}")
    text.append(f"Average Execution Time: {summary.get('avg_execution_time', 0):.3f}s")
    text.append(f"Average Memory Usage: {summary.get('avg_memory_usage', 0):.1f}MB")
    text.append(f"Average Throughput: {summary.get('avg_throughput', 0):.2f} ops/s")
    text.append("")
    
    # Regression Alerts
    if report.regression_alerts:
        text.append("⚠️  Regression Alerts")
        text.append("-" * 30)
        for alert in report.regression_alerts:
            severity = alert.get('severity', 'medium').upper()
            text.append(f"[{severity}] {alert['test_name']}")
            text.append(f"  Metric: {alert['metric']}")
            text.append(f"  Trend: {alert['trend_direction']} (strength: {alert['trend_strength']:.2f})")
            text.append(f"  Latest Value: {alert['latest_value']:.3f}")
            text.append("")
    
    # Improvements
    if report.improvement_highlights:
        text.append("✅ Performance Improvements")
        text.append("-" * 30)
        for improvement in report.improvement_highlights:
            text.append(f"✓ {improvement['test_name']}")
            text.append(f"  Metric: {improvement['metric']}")
            text.append(f"  Improvement: {improvement['improvement_percent']:.1f}%")
            text.append("")
    
    # Recommendations
    if report.recommendations:
        text.append("💡 Recommendations")
        text.append("-" * 30)
        for i, rec in enumerate(report.recommendations, 1):
            text.append(f"{i}. {rec}")
        text.append("")
    
    return "\n".join(text)


def format_trends_as_text(trends) -> str:
    """Format trends analysis as human-readable text."""
    text = []
    text.append("📈 Performance Trends Analysis")
    text.append("=" * 50)
    
    for trend_name, trend in trends.items():
        text.append(f"\n🔍 {trend.test_name} - {trend.metric_name}")
        text.append("-" * 40)
        text.append(f"Trend Direction: {trend.trend_direction}")
        text.append(f"Trend Strength: {trend.trend_strength:.3f}")
        text.append(f"Data Points: {len(trend.values)}")
        text.append(f"Regression Detected: {'Yes' if trend.regression_detected else 'No'}")
        
        if trend.values:
            text.append(f"Latest Value: {trend.values[-1]:.3f}")
            text.append(f"Min Value: {min(trend.values):.3f}")
            text.append(f"Max Value: {max(trend.values):.3f}")
            
            if len(trend.values) > 1:
                import statistics
                text.append(f"Average: {statistics.mean(trend.values):.3f}")
                if len(trend.values) > 2:
                    text.append(f"Std Dev: {statistics.stdev(trend.values):.3f}")
    
    return "\n".join(text)


if __name__ == '__main__':
    sys.exit(main())