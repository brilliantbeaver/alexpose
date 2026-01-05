#!/usr/bin/env python3
"""
Test script for log rotation and retention policies.

This script tests the enhanced log rotation and retention functionality
added to the AlexPose logging system.
"""

import sys
import tempfile
import shutil
from pathlib import Path
from datetime import datetime, timedelta

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from ambient.utils.logging import (
    get_rotation_retention_policies,
    validate_rotation_retention_config,
    calculate_estimated_disk_usage,
    setup_advanced_log_rotation,
    cleanup_old_logs,
    get_log_statistics,
    monitor_log_disk_usage,
    setup_logging,
    create_component_logger
)


def test_rotation_retention_policies():
    """Test environment-specific rotation and retention policies."""
    print("Testing rotation and retention policies...")
    
    # Test different environments
    environments = ["development", "production", "staging", "testing", "unknown"]
    
    for env in environments:
        policies = get_rotation_retention_policies(env)
        print(f"\n{env.upper()} Environment Policies:")
        for log_type, policy in policies.items():
            print(f"  {log_type}: rotation={policy['rotation']}, retention={policy['retention']}")
        
        # Validate each policy
        for log_type, policy in policies.items():
            validation = validate_rotation_retention_config(
                policy['rotation'], 
                policy['retention']
            )
            if not validation['valid']:
                print(f"  WARNING: Invalid policy for {log_type}: {validation['errors']}")
    
    print("✓ Rotation and retention policies test completed")


def test_disk_usage_calculation():
    """Test disk usage estimation."""
    print("\nTesting disk usage calculation...")
    
    for env in ["development", "production", "staging"]:
        policies = get_rotation_retention_policies(env)
        usage = calculate_estimated_disk_usage(policies, daily_log_volume_mb=50)
        
        print(f"\n{env.upper()} Estimated Disk Usage:")
        print(f"  Total: {usage['total_estimated_gb']:.2f} GB")
        print(f"  Recommendations: {usage['recommendations']}")
        
        for log_type, breakdown in usage['breakdown'].items():
            print(f"  {log_type}: {breakdown['estimated_total_mb']:.1f} MB "
                  f"({breakdown['retention_days']} days retention)")
    
    print("✓ Disk usage calculation test completed")


def test_validation():
    """Test configuration validation."""
    print("\nTesting configuration validation...")
    
    test_cases = [
        ("1 day", "30 days", True),
        ("12 hours", "7 days", True),
        ("100 MB", "10", True),
        ("invalid", "30 days", False),
        ("1 day", "invalid", False),
        ("1000 MB", "365 days", True),  # Should have warnings
    ]
    
    for rotation, retention, should_be_valid in test_cases:
        validation = validate_rotation_retention_config(rotation, retention)
        print(f"  {rotation} / {retention}: valid={validation['valid']}, "
              f"errors={len(validation['errors'])}, warnings={len(validation['warnings'])}")
        
        if validation['valid'] != should_be_valid:
            print(f"    UNEXPECTED: Expected valid={should_be_valid}, got {validation['valid']}")
    
    print("✓ Configuration validation test completed")


def test_logging_with_rotation():
    """Test actual logging with rotation policies."""
    print("\nTesting logging with rotation policies...")
    
    # Create temporary directory for testing
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_log_dir = Path(temp_dir) / "test_logs"
        
        # Store original handlers to restore later
        from loguru import logger
        original_handlers = list(logger._core.handlers.keys())
        
        try:
            # Test development environment
            setup_logging(
                log_level="DEBUG",
                log_dir=str(temp_log_dir),
                environment="development"
            )
            
            # Create component logger and generate some logs
            test_logger = create_component_logger("test_component")
            
            for i in range(10):
                test_logger.info(f"Test log message {i}", operation="test_logging", 
                               iteration=i, test_data={"value": i * 10})
                test_logger.debug(f"Debug message {i}", operation="test_debug")
                if i % 3 == 0:
                    test_logger.warning(f"Warning message {i}", operation="test_warning")
            
            # Check that log files were created
            log_files = list(temp_log_dir.glob("*.log"))
            print(f"  Created {len(log_files)} log files:")
            for log_file in log_files:
                size_kb = log_file.stat().st_size / 1024
                print(f"    {log_file.name}: {size_kb:.1f} KB")
            
            # Test log statistics
            stats = get_log_statistics(str(temp_log_dir))
            print(f"  Log statistics: {stats['total_files']} files, "
                  f"{stats['total_size_mb']:.2f} MB total")
        
        finally:
            # Clean up logger handlers to release file locks
            current_handlers = list(logger._core.handlers.keys())
            for handler_id in current_handlers:
                if handler_id not in original_handlers:
                    logger.remove(handler_id)
    
    print("✓ Logging with rotation test completed")


def test_log_management():
    """Test log management utilities."""
    print("\nTesting log management utilities...")
    
    # Create temporary directory with some test log files
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_log_dir = Path(temp_dir) / "test_logs"
        temp_log_dir.mkdir()
        
        # Create some fake log files with different ages
        now = datetime.now()
        test_files = [
            ("alexpose_2024-01-01.log", now - timedelta(days=45)),
            ("alexpose_2024-01-15.log", now - timedelta(days=30)),
            ("errors_2024-01-01.log", now - timedelta(days=45)),
            ("debug_2024-01-20.log", now - timedelta(days=5)),
            ("performance_2024-01-25.log", now - timedelta(days=1)),
        ]
        
        for filename, file_date in test_files:
            file_path = temp_log_dir / filename
            file_path.write_text(f"Test log content for {filename}\n" * 100)
            
            # Set file modification time using os.utime
            timestamp = file_date.timestamp()
            import os
            os.utime(file_path, (timestamp, timestamp))
        
        # Test log statistics
        stats = get_log_statistics(str(temp_log_dir))
        print(f"  Created test logs: {stats['total_files']} files, "
              f"{stats['total_size_mb']:.2f} MB")
        print(f"  Log types: {list(stats['by_type'].keys())}")
        
        # Test cleanup (dry run)
        cleanup_result = cleanup_old_logs(str(temp_log_dir), dry_run=True)
        print(f"  Cleanup simulation: {cleanup_result['files_deleted']} files would be deleted, "
              f"{cleanup_result['space_freed_mb']:.2f} MB would be freed")
        
        # Test disk usage monitoring
        monitoring = monitor_log_disk_usage(str(temp_log_dir), threshold_gb=0.001)  # Low threshold for testing
        print(f"  Disk monitoring: {monitoring['logs_size_gb']:.3f} GB used, "
              f"threshold exceeded: {monitoring['threshold_exceeded']}")
    
    print("✓ Log management utilities test completed")


def main():
    """Run all log rotation and retention tests."""
    print("AlexPose Log Rotation and Retention Policy Tests")
    print("=" * 50)
    
    try:
        test_rotation_retention_policies()
        test_disk_usage_calculation()
        test_validation()
        test_logging_with_rotation()
        test_log_management()
        
        print("\n" + "=" * 50)
        print("✓ All log rotation and retention tests completed successfully!")
        
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    sys.exit(main())