#!/usr/bin/env python3
"""Demo script showing performance reporting and trend analysis functionality."""

import time
import random
from pathlib import Path

from tests.performance.performance_reporter import PerformanceReporter
from tests.performance.performance_dashboard import PerformanceDashboard
from tests.performance.performance_integration import PerformanceTestIntegration


def simulate_performance_data():
    """Simulate realistic performance data over time."""
    print("🔄 Simulating performance data collection...")
    
    reporter = PerformanceReporter()
    
    # Simulate different test scenarios
    test_scenarios = {
        'video_processing_30s': {
            'base_time': 45.0,
            'trend': 'stable',
            'noise': 5.0
        },
        'api_response_time': {
            'base_time': 0.15,
            'trend': 'improving',
            'noise': 0.02
        },
        'memory_usage_test': {
            'base_time': 512.0,
            'trend': 'degrading',
            'noise': 50.0
        },
        'concurrent_analysis': {
            'base_time': 12.0,
            'trend': 'stable',
            'noise': 2.0
        }
    }
    
    # Simulate data collection over the past 14 days
    base_timestamp = time.time() - (14 * 24 * 60 * 60)  # 14 days ago
    
    for day in range(14):
        for test_name, scenario in test_scenarios.items():
            # Generate 2-4 data points per day per test
            daily_runs = random.randint(2, 4)
            
            for run in range(daily_runs):
                # Calculate timestamp for this run
                timestamp = base_timestamp + (day * 24 * 60 * 60) + (run * 6 * 60 * 60)  # Spread throughout day
                
                # Generate performance metrics based on scenario
                execution_time = generate_metric_value(
                    scenario['base_time'], 
                    scenario['trend'], 
                    scenario['noise'], 
                    day, 14
                )
                
                memory_usage = generate_metric_value(
                    scenario['base_time'] * 2,  # Memory roughly 2x execution time
                    scenario['trend'],
                    scenario['noise'] * 2,
                    day, 14
                )
                
                throughput = generate_metric_value(
                    10.0,  # Base throughput
                    'improving' if scenario['trend'] == 'degrading' else scenario['trend'],
                    2.0,
                    day, 14
                )
                
                metrics = {
                    'execution_time': max(0.01, execution_time),  # Minimum 0.01s
                    'memory_usage_mb': max(10.0, memory_usage),   # Minimum 10MB
                    'peak_memory_mb': max(10.0, memory_usage * 1.2),
                    'cpu_usage_percent': random.uniform(10, 80),
                    'throughput': max(0.1, throughput)
                }
                
                # Manually add to history with specific timestamp
                if test_name not in reporter.performance_history:
                    reporter.performance_history[test_name] = []
                
                reporter.performance_history[test_name].append({
                    'timestamp': timestamp,
                    'datetime': time.strftime('%Y-%m-%dT%H:%M:%S', time.localtime(timestamp)),
                    'metrics': metrics
                })
    
    # Save the simulated data
    reporter._save_performance_history()
    print(f"✅ Generated performance data for {len(test_scenarios)} tests over 14 days")
    
    return reporter


def generate_metric_value(base_value, trend, noise, day, total_days):
    """Generate a metric value based on trend and noise."""
    # Apply trend
    if trend == 'improving':
        trend_factor = 1.0 - (day / total_days) * 0.3  # 30% improvement over time
    elif trend == 'degrading':
        trend_factor = 1.0 + (day / total_days) * 0.4  # 40% degradation over time
    else:  # stable
        trend_factor = 1.0
    
    # Apply noise
    noise_factor = random.uniform(-noise, noise) / base_value
    
    return base_value * trend_factor * (1.0 + noise_factor)


def demo_trend_analysis(reporter):
    """Demonstrate trend analysis functionality."""
    print("\n📈 Analyzing Performance Trends...")
    
    trends = reporter.analyze_performance_trends(days_back=14)
    
    print(f"Found {len(trends)} performance trends:")
    
    for trend_name, trend in trends.items():
        print(f"\n🔍 {trend.test_name} - {trend.metric_name}")
        print(f"   Direction: {trend.trend_direction}")
        print(f"   Strength: {trend.trend_strength:.3f}")
        print(f"   Regression: {'Yes' if trend.regression_detected else 'No'}")
        print(f"   Data Points: {len(trend.values)}")
        
        if trend.values:
            print(f"   Latest Value: {trend.values[-1]:.3f}")
            print(f"   Value Range: {min(trend.values):.3f} - {max(trend.values):.3f}")


def demo_performance_report(reporter):
    """Demonstrate performance report generation."""
    print("\n📊 Generating Performance Report...")
    
    report = reporter.generate_performance_report(report_period="weekly")
    
    print(f"Report Period: {report.report_period}")
    print(f"Total Tests: {report.total_tests}")
    print(f"Regression Alerts: {len(report.regression_alerts)}")
    print(f"Improvements: {len(report.improvement_highlights)}")
    
    # Show regression alerts
    if report.regression_alerts:
        print("\n⚠️  Regression Alerts:")
        for alert in report.regression_alerts:
            print(f"   - {alert['test_name']}: {alert['metric']} ({alert.get('severity', 'medium')} severity)")
    
    # Show improvements
    if report.improvement_highlights:
        print("\n✅ Performance Improvements:")
        for improvement in report.improvement_highlights:
            print(f"   - {improvement['test_name']}: {improvement['metric']} ({improvement['improvement_percent']:.1f}% better)")
    
    # Show recommendations
    if report.recommendations:
        print("\n💡 Recommendations:")
        for i, rec in enumerate(report.recommendations, 1):
            print(f"   {i}. {rec}")


def demo_dashboard_generation(reporter):
    """Demonstrate dashboard generation."""
    print("\n🎨 Generating Performance Dashboards...")
    
    dashboard = PerformanceDashboard(reporter)
    
    # Generate all dashboard types
    dashboards = dashboard.generate_all_dashboards(days_back=14)
    
    print("Generated dashboards:")
    for dashboard_type, path in dashboards.items():
        print(f"   - {dashboard_type.title()}: {path}")
    
    return dashboards


def demo_integration_testing():
    """Demonstrate performance testing integration."""
    print("\n🔧 Testing Performance Integration...")
    
    integration = PerformanceTestIntegration(
        enable_reporting=True,
        enable_dashboards=True
    )
    
    # Define sample test functions
    def fast_operation():
        time.sleep(0.05)
        return "fast completed"
    
    def slow_operation():
        time.sleep(0.5)
        return "slow completed"
    
    def variable_operation():
        time.sleep(random.uniform(0.1, 0.3))
        return "variable completed"
    
    # Run performance tests
    test_functions = [
        ("fast_test", fast_operation),
        ("slow_test", slow_operation),
        ("variable_test", variable_operation)
    ]
    
    results = []
    for test_name, test_func in test_functions:
        print(f"   Running {test_name}...")
        result = integration.run_performance_test(test_name, test_func, iterations=3)
        results.append(result)
        
        metrics = result['metrics']
        print(f"     Execution time: {metrics.execution_time:.3f}s")
        print(f"     Memory usage: {metrics.memory_usage_mb:.1f}MB")
        
        if result['regression_analysis']['regression_detected']:
            print(f"     ⚠️  Regression detected!")
    
    # Test concurrent performance
    print("   Running concurrent test...")
    concurrent_result = integration.run_concurrent_performance_test(
        "concurrent_demo", fast_operation, num_concurrent=5
    )
    
    concurrent_details = concurrent_result['concurrent_details']
    print(f"     Concurrent operations: {concurrent_details['concurrent_operations']}")
    print(f"     Success rate: {concurrent_details['success_rate']:.2f}")
    print(f"     Total time: {concurrent_details['total_time']:.3f}s")
    
    return results


def demo_cli_usage():
    """Demonstrate CLI usage examples."""
    print("\n💻 CLI Usage Examples:")
    print("   # Generate weekly performance report")
    print("   python -m tests.performance.performance_cli report --period weekly")
    print()
    print("   # Generate comprehensive dashboard")
    print("   python -m tests.performance.performance_cli dashboard --type comprehensive --open")
    print()
    print("   # Analyze trends for specific test")
    print("   python -m tests.performance.performance_cli trends --test video_processing_30s --days 14")
    print()
    print("   # Export performance data to CSV")
    print("   python -m tests.performance.performance_cli export --format csv --days 30 --output perf_data.csv")
    print()
    print("   # Check monitoring status")
    print("   python -m tests.performance.performance_cli status --verbose")


def main():
    """Main demo function."""
    print("🚀 AlexPose Performance Reporting & Trend Analysis Demo")
    print("=" * 60)
    
    try:
        # Step 1: Simulate performance data
        reporter = simulate_performance_data()
        
        # Step 2: Demonstrate trend analysis
        demo_trend_analysis(reporter)
        
        # Step 3: Generate performance report
        demo_performance_report(reporter)
        
        # Step 4: Generate dashboards
        dashboards = demo_dashboard_generation(reporter)
        
        # Step 5: Demonstrate integration testing
        demo_integration_testing()
        
        # Step 6: Show CLI usage examples
        demo_cli_usage()
        
        print("\n🎉 Demo completed successfully!")
        print("\nGenerated files:")
        print(f"   - Performance history: {reporter.history_file}")
        print(f"   - Reports directory: {reporter.reports_dir}")
        
        if dashboards:
            print("   - Dashboards:")
            for dashboard_type, path in dashboards.items():
                print(f"     * {dashboard_type}: {path}")
        
        print("\nTo view dashboards, open the HTML files in your browser.")
        print("To run CLI commands, use: python -m tests.performance.performance_cli --help")
        
    except Exception as e:
        print(f"❌ Demo failed: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())