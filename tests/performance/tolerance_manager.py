#!/usr/bin/env python3
"""Command-line tool for managing performance regression tolerance configuration."""

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, Any

from performance_config import PerformanceConfig, get_performance_config


def list_tolerances(config: PerformanceConfig):
    """List all tolerance configurations."""
    all_tolerances = config.get_all_tolerances()
    
    print("=== Performance Tolerance Configuration ===\n")
    
    print("Default Tolerances:")
    for metric, value in all_tolerances['default_tolerances'].items():
        print(f"  {metric}: {value}%")
    
    print(f"\nEnvironment: {all_tolerances['current_environment']}")
    print("Environment Multipliers:")
    for env, multiplier in all_tolerances['environment_multipliers'].items():
        print(f"  {env}: {multiplier}x")
    
    print("\nTest-Specific Tolerances:")
    if all_tolerances['test_specific_tolerances']:
        for test_name, tolerances in all_tolerances['test_specific_tolerances'].items():
            print(f"  {test_name}:")
            for metric, value in tolerances.items():
                metric_display = metric.replace('_tolerance', '')
                print(f"    {metric_display}: {value}%")
    else:
        print("  None configured")


def set_tolerance(config: PerformanceConfig, test_name: str, tolerances: Dict[str, float]):
    """Set tolerance for a specific test."""
    config.set_test_tolerance(test_name, **tolerances)
    print(f"Updated tolerance configuration for '{test_name}':")
    for metric, value in tolerances.items():
        print(f"  {metric}: {value}%")


def set_default_tolerance(config: PerformanceConfig, metric: str, value: float):
    """Set default tolerance for a metric type."""
    if metric == "execution_time":
        config.tolerance_config.execution_time_tolerance = value
    elif metric == "memory_usage":
        config.tolerance_config.memory_usage_tolerance = value
    elif metric == "throughput":
        config.tolerance_config.throughput_tolerance = value
    else:
        print(f"Error: Unknown metric type '{metric}'")
        print("Valid metric types: execution_time, memory_usage, throughput")
        return
    
    config.save_config()
    print(f"Updated default {metric} tolerance to {value}%")


def remove_test_tolerance(config: PerformanceConfig, test_name: str):
    """Remove test-specific tolerance configuration."""
    if test_name in config.tolerance_config.test_specific_tolerances:
        del config.tolerance_config.test_specific_tolerances[test_name]
        config.save_config()
        print(f"Removed tolerance configuration for '{test_name}'")
    else:
        print(f"No tolerance configuration found for '{test_name}'")


def export_config(config: PerformanceConfig, output_file: Path):
    """Export configuration to a file."""
    all_tolerances = config.get_all_tolerances()
    
    with open(output_file, 'w') as f:
        json.dump(all_tolerances, f, indent=2)
    
    print(f"Configuration exported to {output_file}")


def import_config(config: PerformanceConfig, input_file: Path):
    """Import configuration from a file."""
    if not input_file.exists():
        print(f"Error: File {input_file} does not exist")
        return
    
    try:
        with open(input_file, 'r') as f:
            data = json.load(f)
        
        # Update configuration
        if 'default_tolerances' in data:
            defaults = data['default_tolerances']
            config.tolerance_config.execution_time_tolerance = defaults.get('execution_time', 10.0)
            config.tolerance_config.memory_usage_tolerance = defaults.get('memory_usage', 15.0)
            config.tolerance_config.throughput_tolerance = defaults.get('throughput', 10.0)
        
        if 'environment_multipliers' in data:
            multipliers = data['environment_multipliers']
            config.tolerance_config.ci_multiplier = multipliers.get('ci', 1.5)
            config.tolerance_config.local_multiplier = multipliers.get('local', 1.0)
        
        if 'test_specific_tolerances' in data:
            config.tolerance_config.test_specific_tolerances = data['test_specific_tolerances']
        
        config.save_config()
        print(f"Configuration imported from {input_file}")
        
    except Exception as e:
        print(f"Error importing configuration: {e}")


def validate_config(config: PerformanceConfig):
    """Validate the current configuration."""
    all_tolerances = config.get_all_tolerances()
    issues = []
    
    # Check for reasonable default values
    defaults = all_tolerances['default_tolerances']
    if defaults['execution_time'] < 1.0 or defaults['execution_time'] > 100.0:
        issues.append(f"Default execution_time tolerance ({defaults['execution_time']}%) seems unreasonable")
    
    if defaults['memory_usage'] < 1.0 or defaults['memory_usage'] > 100.0:
        issues.append(f"Default memory_usage tolerance ({defaults['memory_usage']}%) seems unreasonable")
    
    # Check test-specific tolerances
    for test_name, tolerances in all_tolerances['test_specific_tolerances'].items():
        for metric, value in tolerances.items():
            if value < 0.1 or value > 200.0:
                issues.append(f"Test '{test_name}' {metric} ({value}%) seems unreasonable")
    
    if issues:
        print("Configuration validation issues found:")
        for issue in issues:
            print(f"  - {issue}")
        return False
    else:
        print("Configuration validation passed - all tolerances are reasonable")
        return True


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Manage performance regression tolerance configuration",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # List all tolerance configurations
  python tolerance_manager.py list
  
  # Set tolerance for a specific test
  python tolerance_manager.py set-test video_loading_30s --execution-time 20.0 --memory-usage 25.0
  
  # Set default tolerance
  python tolerance_manager.py set-default execution_time 12.0
  
  # Remove test-specific configuration
  python tolerance_manager.py remove-test video_loading_30s
  
  # Export configuration
  python tolerance_manager.py export config_backup.json
  
  # Import configuration
  python tolerance_manager.py import config_backup.json
  
  # Validate configuration
  python tolerance_manager.py validate
        """
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # List command
    subparsers.add_parser('list', help='List all tolerance configurations')
    
    # Set test tolerance command
    set_test_parser = subparsers.add_parser('set-test', help='Set tolerance for a specific test')
    set_test_parser.add_argument('test_name', help='Name of the test')
    set_test_parser.add_argument('--execution-time', type=float, help='Execution time tolerance (%)')
    set_test_parser.add_argument('--memory-usage', type=float, help='Memory usage tolerance (%)')
    set_test_parser.add_argument('--throughput', type=float, help='Throughput tolerance (%)')
    
    # Set default tolerance command
    set_default_parser = subparsers.add_parser('set-default', help='Set default tolerance')
    set_default_parser.add_argument('metric', choices=['execution_time', 'memory_usage', 'throughput'],
                                   help='Metric type')
    set_default_parser.add_argument('value', type=float, help='Tolerance value (%)')
    
    # Remove test tolerance command
    remove_parser = subparsers.add_parser('remove-test', help='Remove test-specific tolerance')
    remove_parser.add_argument('test_name', help='Name of the test')
    
    # Export command
    export_parser = subparsers.add_parser('export', help='Export configuration to file')
    export_parser.add_argument('output_file', type=Path, help='Output file path')
    
    # Import command
    import_parser = subparsers.add_parser('import', help='Import configuration from file')
    import_parser.add_argument('input_file', type=Path, help='Input file path')
    
    # Validate command
    subparsers.add_parser('validate', help='Validate current configuration')
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return
    
    # Get configuration
    config = get_performance_config()
    
    # Execute command
    if args.command == 'list':
        list_tolerances(config)
    
    elif args.command == 'set-test':
        tolerances = {}
        if args.execution_time is not None:
            tolerances['execution_time'] = args.execution_time
        if args.memory_usage is not None:
            tolerances['memory_usage'] = args.memory_usage
        if args.throughput is not None:
            tolerances['throughput'] = args.throughput
        
        if not tolerances:
            print("Error: At least one tolerance value must be specified")
            return
        
        set_tolerance(config, args.test_name, tolerances)
    
    elif args.command == 'set-default':
        set_default_tolerance(config, args.metric, args.value)
    
    elif args.command == 'remove-test':
        remove_test_tolerance(config, args.test_name)
    
    elif args.command == 'export':
        export_config(config, args.output_file)
    
    elif args.command == 'import':
        import_config(config, args.input_file)
    
    elif args.command == 'validate':
        validate_config(config)


if __name__ == '__main__':
    main()