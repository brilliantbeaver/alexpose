#!/usr/bin/env python3
"""
Example demonstrating the enhanced log rotation and retention policies.

This example shows how to use the new log rotation and retention features
in the AlexPose logging system.
"""

import sys
import time
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from ambient.utils.logging import (
    setup_logging,
    get_rotation_retention_policies,
    calculate_estimated_disk_usage,
    cleanup_old_logs,
    get_log_statistics,
    monitor_log_disk_usage,
    create_component_logger,
    configure_development_logging,
    configure_production_logging
)


def demonstrate_environment_policies():
    """Demonstrate different environment policies."""
    print("Environment-Specific Log Rotation Policies")
    print("=" * 50)
    
    environments = ["development", "production", "staging", "testing"]
    
    for env in environments:
        print(f"\n{env.upper()} Environment:")
        policies = get_rotation_retention_policies(env)
        
        for log_type, policy in policies.items():
            print(f"  {log_type:12}: rotation={policy['rotation']:8} | "
                  f"retention={policy['retention']:8} | "
                  f"compression={policy['compression']}")
        
        # Calculate estimated disk usage
        usage = calculate_estimated_disk_usage(policies, daily_log_volume_mb=100)
        print(f"  Estimated disk usage: {usage['total_estimated_gb']:.1f} GB")
        
        if usage['recommendations']:
            print(f"  Recommendations: {', '.join(usage['recommendations'])}")


def demonstrate_logging_setup():
    """Demonstrate setting up logging with rotation policies."""
    print("\n\nSetting Up Logging with Advanced Rotation")
    print("=" * 50)
    
    # Configure for development environment
    print("\nConfiguring development logging...")
    configure_development_logging()
    
    # Create component loggers
    video_logger = create_component_logger("video_processor")
    pose_logger = create_component_logger("pose_estimator")
    gait_logger = create_component_logger("gait_analyzer")
    
    # Generate various types of log messages
    print("Generating sample log messages...")
    
    for i in range(5):
        video_logger.info(f"Processing video frame {i}", operation="process_frame",
                         frame_number=i, processing_time=0.1 + i * 0.05)
        
        pose_logger.debug(f"Pose estimation for frame {i}", operation="estimate_pose",
                         keypoints_detected=17, confidence=0.85 + i * 0.02)
        
        if i % 2 == 0:
            gait_logger.warning(f"Low confidence in gait analysis {i}", 
                              operation="analyze_gait", confidence=0.6)
        
        # Simulate some processing time
        time.sleep(0.1)
    
    print("✓ Sample logs generated")


def demonstrate_log_management():
    """Demonstrate log management utilities."""
    print("\n\nLog Management and Monitoring")
    print("=" * 50)
    
    log_dir = "logs/dev"  # Development log directory
    
    # Get log statistics
    print("\nCurrent log statistics:")
    stats = get_log_statistics(log_dir)
    
    if "error" in stats:
        print(f"  {stats['error']}")
        return
    
    print(f"  Total files: {stats['total_files']}")
    print(f"  Total size: {stats['total_size_mb']:.2f} MB")
    
    if stats['by_type']:
        print("  By log type:")
        for log_type, info in stats['by_type'].items():
            print(f"    {log_type}: {info['count']} files, {info['size_mb']:.2f} MB")
    
    # Monitor disk usage
    print("\nDisk usage monitoring:")
    monitoring = monitor_log_disk_usage(log_dir, threshold_gb=0.1)  # Low threshold for demo
    
    if "error" in monitoring:
        print(f"  {monitoring['error']}")
    else:
        print(f"  Log directory size: {monitoring['logs_size_gb']:.3f} GB")
        print(f"  Disk free space: {monitoring['disk_free_gb']:.1f} GB")
        print(f"  Threshold exceeded: {monitoring['threshold_exceeded']}")
        
        if monitoring['recommendations']:
            print("  Recommendations:")
            for rec in monitoring['recommendations']:
                print(f"    - {rec}")
    
    # Demonstrate cleanup (dry run)
    print("\nLog cleanup simulation:")
    cleanup_result = cleanup_old_logs(log_dir, dry_run=True)
    
    if "error" in cleanup_result:
        print(f"  {cleanup_result['error']}")
    else:
        print(f"  Files that would be deleted: {cleanup_result['files_deleted']}")
        print(f"  Space that would be freed: {cleanup_result['space_freed_mb']:.2f} MB")
        
        if cleanup_result['deleted_files']:
            print("  Files marked for deletion:")
            for file_info in cleanup_result['deleted_files'][:3]:  # Show first 3
                print(f"    - {Path(file_info['file']).name} "
                      f"(age: {file_info['age_days']} days, "
                      f"size: {file_info['size_mb']:.2f} MB)")


def demonstrate_production_vs_development():
    """Compare production vs development logging configurations."""
    print("\n\nProduction vs Development Comparison")
    print("=" * 50)
    
    dev_policies = get_rotation_retention_policies("development")
    prod_policies = get_rotation_retention_policies("production")
    
    print(f"{'Log Type':<15} {'Dev Rotation':<12} {'Prod Rotation':<12} "
          f"{'Dev Retention':<12} {'Prod Retention':<12}")
    print("-" * 75)
    
    all_log_types = set(dev_policies.keys()) | set(prod_policies.keys())
    
    for log_type in sorted(all_log_types):
        dev_policy = dev_policies.get(log_type, {})
        prod_policy = prod_policies.get(log_type, {})
        
        dev_rotation = dev_policy.get('rotation', 'N/A')
        prod_rotation = prod_policy.get('rotation', 'N/A')
        dev_retention = dev_policy.get('retention', 'N/A')
        prod_retention = prod_policy.get('retention', 'N/A')
        
        print(f"{log_type:<15} {dev_rotation:<12} {prod_rotation:<12} "
              f"{dev_retention:<12} {prod_retention:<12}")
    
    # Compare disk usage estimates
    dev_usage = calculate_estimated_disk_usage(dev_policies)
    prod_usage = calculate_estimated_disk_usage(prod_policies)
    
    print(f"\nEstimated disk usage:")
    print(f"  Development: {dev_usage['total_estimated_gb']:.1f} GB")
    print(f"  Production:  {prod_usage['total_estimated_gb']:.1f} GB")


def main():
    """Run the log rotation and retention demonstration."""
    print("AlexPose Enhanced Log Rotation and Retention Demo")
    print("=" * 60)
    
    try:
        demonstrate_environment_policies()
        demonstrate_logging_setup()
        demonstrate_log_management()
        demonstrate_production_vs_development()
        
        print("\n" + "=" * 60)
        print("✓ Log rotation and retention demonstration completed!")
        print("\nKey Features Demonstrated:")
        print("  • Environment-specific rotation and retention policies")
        print("  • Automatic disk usage estimation and monitoring")
        print("  • Log cleanup utilities with dry-run capability")
        print("  • Comprehensive log statistics and management")
        print("  • Production vs development policy comparison")
        
    except Exception as e:
        print(f"\n❌ Demo failed with error: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    sys.exit(main())