#!/usr/bin/env python3
"""
Demonstration of the completed debugging tools and systems.

This script shows how all the debugging components work together
to provide comprehensive test failure analysis and debugging support.
"""

import tempfile
import json
from pathlib import Path
from datetime import datetime

# Import debugging components
from tests.debugging.artifact_collector import ArtifactCollector
from tests.debugging.pattern_matcher import PatternMatcher
from tests.debugging.test_monitor import TestMonitor, TestStatus
from tests.debugging.failure_reporter import FailureReporter
from tests.utils.debugging_helpers import debug_test_failure, analyze_test_environment


def demo_artifact_collection():
    """Demonstrate artifact collection capabilities."""
    print("🔍 ARTIFACT COLLECTION DEMO")
    print("=" * 50)
    
    with tempfile.TemporaryDirectory() as temp_dir:
        collector = ArtifactCollector(artifacts_dir=Path(temp_dir) / "artifacts")
        
        # Simulate a test failure
        test_name = "test_demo_failure"
        error_info = {
            "error_message": "AssertionError: Expected confidence in [0.0, 1.0], got 1.5",
            "traceback": "Traceback (most recent call last):\n  File test.py, line 42\n    assert 0.0 <= confidence <= 1.0",
            "timestamp": datetime.now().isoformat()
        }
        
        # Collect artifacts
        artifacts = collector.collect_artifacts(
            test_name=test_name,
            error_info=error_info,
            capture_logs=True,
            capture_files=True,
            capture_environment=True
        )
        
        print(f"✅ Collected artifacts for: {artifacts.test_name}")
        print(f"   - System state captured: {artifacts.system_state is not None}")
        print(f"   - Environment snapshot: {len(artifacts.environment_snapshot)} items")
        print(f"   - Log files collected: {len(artifacts.log_files)}")
        print(f"   - File artifacts: {len(artifacts.file_artifacts)}")
        print(f"   - Error traceback: {artifacts.error_traceback is not None}")
        
        # Show artifact summary
        summary = collector.get_artifact_summary()
        print(f"   - Total artifact sets: {summary['total_artifact_sets']}")
        print()


def demo_pattern_matching():
    """Demonstrate pattern matching and historical analysis."""
    print("🔍 PATTERN MATCHING DEMO")
    print("=" * 50)
    
    with tempfile.TemporaryDirectory() as temp_dir:
        matcher = PatternMatcher(db_path=Path(temp_dir) / "patterns.db")
        
        # Add some sample failures
        sample_failures = [
            {
                "test_name": "test_confidence_validation",
                "error_message": "AssertionError: confidence not in [0.0, 1.0] range",
                "stack_trace": "assert 0.0 <= confidence <= 1.0"
            },
            {
                "test_name": "test_stride_time_validation", 
                "error_message": "AssertionError: stride_time not in [0.8, 2.0] range",
                "stack_trace": "assert 0.8 <= stride_time <= 2.0"
            },
            {
                "test_name": "test_type_error",
                "error_message": "TypeError: unsupported operand type(s) for +: 'int' and 'str'",
                "stack_trace": "result = 5 + 'hello'"
            },
            {
                "test_name": "test_confidence_validation",
                "error_message": "AssertionError: confidence not in [0.0, 1.0] range", 
                "stack_trace": "assert 0.0 <= confidence <= 1.0"
            }
        ]
        
        # Process failures and collect matches
        all_matches = []
        for failure in sample_failures:
            matches = matcher.add_failure(**failure)
            all_matches.extend(matches)
        
        print(f"✅ Processed {len(sample_failures)} failures")
        print(f"   - Pattern matches found: {len(all_matches)}")
        
        # Show historical analysis
        analysis = matcher.get_historical_analysis(days=1)
        print(f"   - Total failures: {analysis.total_failures}")
        print(f"   - Unique patterns: {analysis.unique_patterns}")
        print(f"   - New patterns: {len(analysis.new_patterns)}")
        
        # Show pattern report
        report = matcher.get_pattern_report()
        print(f"   - Total patterns in system: {report['total_patterns']}")
        print(f"   - Top patterns: {len(report['top_patterns'])}")
        print()


def demo_test_monitoring():
    """Demonstrate test monitoring and alerting."""
    print("🔍 TEST MONITORING DEMO")
    print("=" * 50)
    
    with tempfile.TemporaryDirectory() as temp_dir:
        monitor = TestMonitor(db_path=Path(temp_dir) / "monitor.db")
        monitor.start_monitoring()
        
        try:
            # Simulate monitoring several tests
            test_scenarios = [
                ("test_successful", TestStatus.PASSED, None),
                ("test_failed_assertion", TestStatus.FAILED, "AssertionError: Test failed"),
                ("test_timeout", TestStatus.FAILED, "TimeoutError: Test exceeded time limit"),
                ("test_skipped", TestStatus.SKIPPED, None),
                ("test_another_failure", TestStatus.FAILED, "ValueError: Invalid input")
            ]
            
            for test_name, status, error in test_scenarios:
                monitor.start_test(test_name)
                monitor.end_test(test_name, status, error)
            
            # Get monitoring metrics
            metrics = monitor.get_metrics()
            print(f"✅ Monitored {metrics.total_tests} tests")
            print(f"   - Passed: {metrics.passed_tests}")
            print(f"   - Failed: {metrics.failed_tests}")
            print(f"   - Skipped: {metrics.skipped_tests}")
            print(f"   - Failure rate: {metrics.failure_rate:.1%}")
            print(f"   - Average duration: {metrics.average_duration:.3f}s")
            
            # Check for alerts
            alerts = monitor.get_alerts()
            print(f"   - Alerts generated: {len(alerts)}")
            
        finally:
            monitor.stop_monitoring()
        
        print()


def demo_failure_reporting():
    """Demonstrate failure reporting and tracking."""
    print("🔍 FAILURE REPORTING DEMO")
    print("=" * 50)
    
    with tempfile.TemporaryDirectory() as temp_dir:
        reporter = FailureReporter(db_path=Path(temp_dir) / "reports.db")
        
        # Report several failures
        failure_scenarios = [
            {
                "test_name": "test_gait_analysis",
                "error_message": "AssertionError: Gait features validation failed",
                "stack_trace": "assert validate_gait_features(features)",
                "environment_info": {"python_version": "3.9", "os": "linux"}
            },
            {
                "test_name": "test_pose_estimation", 
                "error_message": "ValueError: Invalid pose landmarks detected",
                "stack_trace": "raise ValueError('Invalid landmarks')",
                "environment_info": {"python_version": "3.9", "mediapipe_version": "0.8.9"}
            },
            {
                "test_name": "test_gait_analysis",  # Duplicate - should be deduplicated
                "error_message": "AssertionError: Gait features validation failed",
                "stack_trace": "assert validate_gait_features(features)",
                "environment_info": {"python_version": "3.9", "os": "linux"}
            }
        ]
        
        report_ids = []
        for scenario in failure_scenarios:
            report_id = reporter.report_failure(**scenario)
            report_ids.append(report_id)
        
        print(f"✅ Processed {len(failure_scenarios)} failure reports")
        print(f"   - Unique report IDs: {len(set(report_ids))}")  # Should be 2 due to deduplication
        
        # Show failure trends
        trends = reporter.get_failure_trends(days=1)
        print(f"   - Total failures: {trends.total_failures}")
        print(f"   - Unique failures: {trends.unique_failures}")
        print(f"   - New failures: {trends.new_failures}")
        print(f"   - Resolution rate: {trends.resolution_rate:.1%}")
        
        # Get reports
        reports = reporter.get_reports(limit=10)
        print(f"   - Active reports: {len(reports)}")
        
        # Show a sample report
        if reports:
            sample_report = reports[0]
            print(f"   - Sample report: {sample_report.report_id}")
            print(f"     * Test: {sample_report.test_name}")
            print(f"     * Status: {sample_report.status.value}")
            print(f"     * Priority: {sample_report.priority.value}")
            print(f"     * Occurrences: {sample_report.occurrence_count}")
        
        print()


def demo_debugging_helpers():
    """Demonstrate debugging helpers and utilities."""
    print("🔍 DEBUGGING HELPERS DEMO")
    print("=" * 50)
    
    # Analyze test environment
    env_analysis = analyze_test_environment()
    print(f"✅ Environment analysis completed")
    print(f"   - System health: {env_analysis['system_health']}")
    print(f"   - Issues found: {len(env_analysis['issues'])}")
    print(f"   - Recommendations: {len(env_analysis['recommendations'])}")
    
    if env_analysis['issues']:
        print(f"   - Sample issue: {env_analysis['issues'][0]}")
    if env_analysis['recommendations']:
        print(f"   - Sample recommendation: {env_analysis['recommendations'][0]}")
    
    # Create debug context for a sample failure
    debug_context = debug_test_failure(
        test_name="test_sample_debug",
        error_message="AssertionError: Expected value 10, got 5",
        failing_input={"value": 5, "expected": 10}
    )
    
    print(f"✅ Debug context created")
    print(f"   - Test: {debug_context.test_name}")
    print(f"   - System state captured: {debug_context.system_state is not None}")
    print(f"   - Reproduction steps: {len(debug_context.reproduction_steps)}")
    print(f"   - Error info: {debug_context.error_info is not None}")
    
    print()


def demo_integration_workflow():
    """Demonstrate complete integration workflow."""
    print("🔍 INTEGRATION WORKFLOW DEMO")
    print("=" * 50)
    
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        
        # Initialize all components
        collector = ArtifactCollector(artifacts_dir=temp_path / "artifacts")
        matcher = PatternMatcher(db_path=temp_path / "patterns.db")
        monitor = TestMonitor(db_path=temp_path / "monitor.db")
        reporter = FailureReporter(db_path=temp_path / "reports.db")
        
        monitor.start_monitoring()
        
        try:
            # Simulate complete failure handling workflow
            test_name = "test_integration_workflow"
            error_message = "AssertionError: Integration test failed"
            stack_trace = "Traceback: assert integration_result == expected"
            environment_info = {"test_type": "integration", "components": ["video", "pose", "gait"]}
            
            print("1. Starting test monitoring...")
            monitor.start_test(test_name)
            
            print("2. Collecting artifacts...")
            artifacts = collector.collect_artifacts(
                test_name=test_name,
                error_info={
                    "error_message": error_message,
                    "traceback": stack_trace,
                    "timestamp": datetime.now().isoformat()
                }
            )
            
            print("3. Matching failure patterns...")
            matches = matcher.add_failure(
                test_name=test_name,
                error_message=error_message,
                stack_trace=stack_trace,
                environment=environment_info
            )
            
            print("4. Reporting failure...")
            report_id = reporter.report_failure(
                test_name=test_name,
                error_message=error_message,
                stack_trace=stack_trace,
                environment_info=environment_info
            )
            
            print("5. Ending test monitoring...")
            monitor.end_test(
                test_name=test_name,
                status=TestStatus.FAILED,
                error_message=error_message,
                artifacts=artifacts.file_artifacts
            )
            
            print("6. Creating debug context...")
            debug_context = debug_test_failure(
                test_name=test_name,
                error_message=error_message
            )
            
            # Generate summary reports
            print("7. Generating reports...")
            
            # Export monitoring report
            monitor_report_file = temp_path / "monitor_report.json"
            monitor.export_report(monitor_report_file)
            
            # Export pattern report
            pattern_report_file = temp_path / "pattern_report.json"
            pattern_report_file.write_text(
                json.dumps(matcher.get_pattern_report(), indent=2)
            )
            
            # Export failure summary
            failure_summary_file = temp_path / "failure_summary.json"
            reporter.export_report_summary(failure_summary_file)
            
            print(f"✅ Complete workflow executed successfully!")
            print(f"   - Artifacts collected: {len(artifacts.file_artifacts)} files")
            print(f"   - Pattern matches: {len(matches)}")
            print(f"   - Failure report: {report_id}")
            print(f"   - Debug context: {len(debug_context.reproduction_steps)} steps")
            print(f"   - Reports generated: 3 files")
            
        finally:
            monitor.stop_monitoring()
        
        print()


def main():
    """Run all debugging tools demonstrations."""
    print("🚀 DEBUGGING TOOLS DEMONSTRATION")
    print("=" * 60)
    print("This demo shows the completed Task 4.3 debugging tools:")
    print("- Artifact Collection")
    print("- Pattern Matching & Historical Analysis") 
    print("- Test Monitoring & Alerting")
    print("- Failure Reporting & Tracking")
    print("- Debugging Helpers & Utilities")
    print("- Complete Integration Workflow")
    print("=" * 60)
    print()
    
    try:
        demo_artifact_collection()
        demo_pattern_matching()
        demo_test_monitoring()
        demo_failure_reporting()
        demo_debugging_helpers()
        demo_integration_workflow()
        
        print("🎉 ALL DEBUGGING TOOLS DEMONSTRATIONS COMPLETED SUCCESSFULLY!")
        print()
        print("Task 4.3: Test Failure Analysis and Debugging Tools - COMPLETED ✅")
        print()
        print("Key Features Implemented:")
        print("✅ Comprehensive artifact collection (logs, system state, environment)")
        print("✅ Pattern matching with historical analysis and trend detection")
        print("✅ Real-time test monitoring with automated alerting")
        print("✅ Failure reporting with deduplication and tracking")
        print("✅ Debugging helpers with minimal reproduction generation")
        print("✅ Complete integration between all debugging components")
        print()
        print("The debugging infrastructure is now ready for production use!")
        
    except Exception as e:
        print(f"❌ Demo failed with error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()