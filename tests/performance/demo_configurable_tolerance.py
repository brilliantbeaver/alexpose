#!/usr/bin/env python3
"""
Demonstration of configurable performance regression tolerance system.

This script shows how to use the configurable tolerance system for performance
regression detection in the AlexPose testing framework.
"""

import time
import tempfile
from pathlib import Path
from tests.performance.benchmark_framework import PerformanceBenchmark, PerformanceMetrics
from tests.performance.performance_config import get_performance_config


def demo_basic_usage():
    """Demonstrate basic usage of configurable tolerance."""
    print("=== Basic Configurable Tolerance Demo ===\n")
    
    # Create temporary files for this demo
    temp_dir = Path(tempfile.mkdtemp())
    baseline_file = temp_dir / "demo_baselines.json"
    
    try:
        # Create benchmark with configurable tolerance enabled
        benchmark = PerformanceBenchmark(baseline_file=baseline_file, use_config=True)
        
        # Define a mock operation to benchmark
        def mock_operation():
            """Mock operation that takes some time."""
            time.sleep(0.05)  # 50ms
            return "completed"
        
        print("1. Running initial benchmark to establish baseline...")
        initial_metrics = benchmark.benchmark_function(mock_operation, iterations=3)
        print(f"   Execution time: {initial_metrics.execution_time:.3f}s")
        print(f"   Memory usage: {initial_metrics.memory_usage_mb:.1f}MB")
        
        # Establish baseline
        benchmark.establish_baseline("demo_test", initial_metrics)
        print("   ✓ Baseline established")
        
        print("\n2. Getting tolerance configuration for this test...")
        tolerance_info = benchmark.get_tolerance_info("demo_test")
        print(f"   Execution time tolerance: {tolerance_info['execution_time_tolerance']:.1f}%")
        print(f"   Memory usage tolerance: {tolerance_info['memory_usage_tolerance']:.1f}%")
        
        print("\n3. Running test with slight performance regression...")
        
        def slower_operation():
            """Slightly slower operation."""
            time.sleep(0.055)  # 10% slower
            return "completed"
        
        new_metrics = benchmark.benchmark_function(slower_operation, iterations=3)
        print(f"   New execution time: {new_metrics.execution_time:.3f}s")
        
        # Check for regression
        result = benchmark.validate_performance_regression("demo_test", new_metrics)
        
        if result['regression_detected']:
            print("   ❌ Performance regression detected:")
            for regression in result['regressions']:
                print(f"      - {regression}")
        else:
            print("   ✅ No performance regression detected")
            print(f"   Time change: {result['time_change_percent']:.1f}%")
        
        print(f"\n   Tolerances used:")
        for metric, tolerance in result['tolerances_used'].items():
            print(f"      {metric}: {tolerance:.1f}%")
        
    finally:
        # Cleanup
        import shutil
        shutil.rmtree(temp_dir, ignore_errors=True)


def demo_custom_tolerance():
    """Demonstrate setting custom tolerance for specific tests."""
    print("\n=== Custom Tolerance Configuration Demo ===\n")
    
    temp_dir = Path(tempfile.mkdtemp())
    baseline_file = temp_dir / "demo_custom_baselines.json"
    
    try:
        benchmark = PerformanceBenchmark(baseline_file=baseline_file, use_config=True)
        
        print("1. Configuring custom tolerance for a specific test...")
        benchmark.configure_test_tolerance(
            "custom_demo_test",
            execution_time=25.0,  # 25% tolerance
            memory_usage=30.0,    # 30% tolerance
            throughput=20.0       # 20% tolerance
        )
        print("   ✓ Custom tolerance configured")
        
        # Show the configured tolerance
        tolerance_info = benchmark.get_tolerance_info("custom_demo_test")
        print(f"   Custom execution time tolerance: {tolerance_info['execution_time_tolerance']:.1f}%")
        print(f"   Custom memory usage tolerance: {tolerance_info['memory_usage_tolerance']:.1f}%")
        
        def baseline_operation():
            time.sleep(0.04)
            return "baseline"
        
        print("\n2. Establishing baseline...")
        baseline_metrics = benchmark.benchmark_function(baseline_operation, iterations=3)
        benchmark.establish_baseline("custom_demo_test", baseline_metrics)
        print(f"   Baseline execution time: {baseline_metrics.execution_time:.3f}s")
        
        def regressed_operation():
            time.sleep(0.05)  # 25% slower
            return "regressed"
        
        print("\n3. Testing with 25% performance regression...")
        regressed_metrics = benchmark.benchmark_function(regressed_operation, iterations=3)
        print(f"   New execution time: {regressed_metrics.execution_time:.3f}s")
        
        result = benchmark.validate_performance_regression("custom_demo_test", regressed_metrics)
        
        if result['regression_detected']:
            print("   ❌ Performance regression detected:")
            for regression in result['regressions']:
                print(f"      - {regression}")
        else:
            print("   ✅ No regression detected (within custom tolerance)")
            print(f"   Time change: {result['time_change_percent']:.1f}%")
            print(f"   Custom tolerance: {result['tolerances_used']['time_tolerance']:.1f}%")
        
    finally:
        import shutil
        shutil.rmtree(temp_dir, ignore_errors=True)


def demo_environment_awareness():
    """Demonstrate environment-aware tolerance adjustment."""
    print("\n=== Environment-Aware Tolerance Demo ===\n")
    
    config = get_performance_config()
    
    print("1. Current environment detection:")
    all_tolerances = config.get_all_tolerances()
    current_env = all_tolerances['current_environment']
    print(f"   Environment: {current_env}")
    print(f"   CI multiplier: {all_tolerances['environment_multipliers']['ci']}x")
    print(f"   Local multiplier: {all_tolerances['environment_multipliers']['local']}x")
    
    print("\n2. Tolerance comparison for 'video_loading_30s' test:")
    base_tolerance = config.get_tolerance("video_loading_30s", "execution_time")
    print(f"   Current environment tolerance: {base_tolerance:.1f}%")
    
    # Show what it would be in CI
    import os
    original_ci = os.environ.get('CI')
    try:
        os.environ['CI'] = 'true'
        # Create new config to pick up environment change
        from tests.performance.performance_config import PerformanceConfig
        ci_config = PerformanceConfig()
        ci_tolerance = ci_config.get_tolerance("video_loading_30s", "execution_time")
        print(f"   CI environment tolerance: {ci_tolerance:.1f}%")
    finally:
        if original_ci is None:
            os.environ.pop('CI', None)
        else:
            os.environ['CI'] = original_ci
    
    print("\n3. Test-specific tolerances:")
    for test_name, tolerances in all_tolerances['test_specific_tolerances'].items():
        if 'execution_time_tolerance' in tolerances:
            print(f"   {test_name}: {tolerances['execution_time_tolerance']:.1f}%")


def demo_cli_integration():
    """Demonstrate CLI tool integration."""
    print("\n=== CLI Tool Integration Demo ===\n")
    
    print("The tolerance_manager.py CLI tool provides easy management:")
    print("   python tests/performance/tolerance_manager.py list")
    print("   python tests/performance/tolerance_manager.py set-test my_test --execution-time 20.0")
    print("   python tests/performance/tolerance_manager.py validate")
    print("   python tests/performance/tolerance_manager.py export backup.json")
    
    print("\nFor a complete list of CLI commands, run:")
    print("   python tests/performance/tolerance_manager.py --help")


def main():
    """Run all demonstrations."""
    print("Performance Regression Detection with Configurable Tolerance")
    print("=" * 60)
    
    demo_basic_usage()
    demo_custom_tolerance()
    demo_environment_awareness()
    demo_cli_integration()
    
    print("\n" + "=" * 60)
    print("Demo completed! The configurable tolerance system is ready to use.")
    print("\nKey benefits:")
    print("• Test-specific tolerance configuration")
    print("• Environment-aware tolerance adjustment (CI vs local)")
    print("• Backward compatibility with existing tests")
    print("• CLI tool for easy management")
    print("• Comprehensive reporting and validation")


if __name__ == "__main__":
    main()