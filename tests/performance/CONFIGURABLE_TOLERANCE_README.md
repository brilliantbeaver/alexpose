# Configurable Performance Regression Tolerance

This document describes the configurable tolerance system for performance regression detection in the AlexPose testing framework.

## Overview

The configurable tolerance system allows you to set different tolerance levels for performance regression detection on a per-test basis, with support for different environments (CI vs local) and different metric types (execution time, memory usage, throughput).

## Features

- **Test-specific tolerances**: Configure different tolerance levels for different tests
- **Metric-specific tolerances**: Set different tolerances for execution time, memory usage, and throughput
- **Environment awareness**: Automatically apply different tolerances in CI vs local environments
- **Default fallbacks**: Sensible defaults for tests without specific configuration
- **CLI management**: Command-line tool for managing tolerance configurations
- **Backward compatibility**: Works with existing tests that specify explicit tolerances

## Configuration Structure

The tolerance configuration is stored in `tests/performance/performance_config.json`:

```json
{
  "tolerance": {
    "execution_time_tolerance": 10.0,
    "memory_usage_tolerance": 15.0,
    "throughput_tolerance": 10.0,
    "test_specific_tolerances": {
      "video_loading_30s": {
        "execution_time_tolerance": 15.0,
        "memory_usage_tolerance": 20.0
      },
      "api_response_time_health": {
        "execution_time_tolerance": 10.0,
        "memory_usage_tolerance": 10.0
      }
    },
    "ci_multiplier": 1.5,
    "local_multiplier": 1.0
  }
}
```

## Usage in Tests

### Basic Usage

```python
from tests.performance.benchmark_framework import PerformanceBenchmark

# Create benchmark with configurable tolerance enabled (default)
benchmark = PerformanceBenchmark(use_config=True)

# Run performance test
metrics = benchmark.benchmark_function(my_function)

# Check for regression using configured tolerances
result = benchmark.validate_performance_regression("my_test", metrics)

if result['regression_detected']:
    pytest.fail(f"Performance regression: {result['regressions']}")
```

### Configuring Test-Specific Tolerances

```python
# Configure tolerance for a specific test
benchmark.configure_test_tolerance(
    "my_test",
    execution_time=25.0,  # 25% tolerance for execution time
    memory_usage=30.0,    # 30% tolerance for memory usage
    throughput=20.0       # 20% tolerance for throughput
)
```

### Getting Tolerance Information

```python
# Get tolerance info for a test
tolerance_info = benchmark.get_tolerance_info("my_test")
print(f"Execution time tolerance: {tolerance_info['execution_time_tolerance']}%")
```

### Backward Compatibility

```python
# Still works with explicit tolerance (overrides configuration)
result = benchmark.validate_performance_regression(
    "my_test", 
    metrics, 
    tolerance_percent=20.0  # Explicit tolerance overrides config
)
```

## Command-Line Management

Use the `tolerance_manager.py` tool to manage tolerance configurations:

### List Current Configuration

```bash
python tests/performance/tolerance_manager.py list
```

### Set Test-Specific Tolerance

```bash
# Set tolerance for a specific test
python tests/performance/tolerance_manager.py set-test video_loading_30s \
    --execution-time 20.0 --memory-usage 25.0

# Set only execution time tolerance
python tests/performance/tolerance_manager.py set-test api_test \
    --execution-time 15.0
```

### Set Default Tolerances

```bash
# Set default execution time tolerance
python tests/performance/tolerance_manager.py set-default execution_time 12.0

# Set default memory usage tolerance
python tests/performance/tolerance_manager.py set-default memory_usage 18.0
```

### Remove Test Configuration

```bash
python tests/performance/tolerance_manager.py remove-test video_loading_30s
```

### Export/Import Configuration

```bash
# Export current configuration
python tests/performance/tolerance_manager.py export backup.json

# Import configuration
python tests/performance/tolerance_manager.py import backup.json
```

### Validate Configuration

```bash
python tests/performance/tolerance_manager.py validate
```

## Environment-Specific Behavior

The system automatically detects the environment and applies appropriate multipliers:

- **Local environment**: Uses base tolerance values
- **CI environment**: Multiplies tolerances by `ci_multiplier` (default: 1.5x)

CI environment is detected by checking for these environment variables:
- `CI`
- `GITHUB_ACTIONS`
- `JENKINS_URL`
- `TRAVIS`
- `CIRCLECI`

## Default Test Configurations

The system comes with sensible defaults for common test types:

| Test Type | Execution Time | Memory Usage | Notes |
|-----------|----------------|--------------|-------|
| Video processing | 15-30% | 20-30% | Higher tolerance due to complexity |
| API endpoints | 10-15% | 10-15% | Stricter tolerance for responsiveness |
| Concurrent analysis | 20% | 25% | Moderate tolerance for concurrency |
| Memory tests | 15% | 10% | Stricter memory tolerance |

## Integration with Existing Tests

The configurable tolerance system is designed to work seamlessly with existing performance tests:

1. **Automatic migration**: Existing tests continue to work without changes
2. **Gradual adoption**: Tests can be migrated to use configurable tolerances one at a time
3. **Override capability**: Explicit tolerance parameters still work and override configuration

## Best Practices

### Setting Tolerances

1. **Start conservative**: Begin with lower tolerances and adjust based on actual variance
2. **Consider test complexity**: More complex operations typically need higher tolerances
3. **Account for environment**: CI environments are typically less stable than local
4. **Monitor trends**: Use performance reports to identify tests that need tolerance adjustments

### Test Organization

1. **Group similar tests**: Use consistent naming patterns for related tests
2. **Document rationale**: Include comments explaining why specific tolerances were chosen
3. **Regular review**: Periodically review and adjust tolerances based on historical data

### CI/CD Integration

1. **Environment detection**: Ensure CI environment variables are properly set
2. **Baseline management**: Establish baselines in stable environments
3. **Failure analysis**: Use detailed regression reports to diagnose performance issues

## Troubleshooting

### Common Issues

1. **Configuration not loading**: Check file permissions and JSON syntax
2. **Tolerances too strict**: Review historical performance data to set appropriate levels
3. **Environment detection**: Verify CI environment variables are set correctly

### Debugging

```python
# Check if configuration is loaded
benchmark = PerformanceBenchmark(use_config=True)
print(f"Config loaded: {benchmark.use_config}")
print(f"Config available: {benchmark.performance_config is not None}")

# Get detailed tolerance information
tolerance_info = benchmark.get_tolerance_info("my_test")
print(f"Tolerance info: {tolerance_info}")

# Generate comprehensive report
report = benchmark.generate_performance_report()
print(f"Tolerance config: {report.get('tolerance_configuration', 'Not available')}")
```

## API Reference

### PerformanceConfig Class

- `get_tolerance(test_name, metric_type)`: Get tolerance for specific test and metric
- `set_test_tolerance(test_name, **tolerances)`: Set test-specific tolerances
- `get_all_tolerances()`: Get complete tolerance configuration
- `save_config()`: Save configuration to file

### PerformanceBenchmark Class

- `validate_performance_regression(test_name, metrics, tolerance_percent=None, ...)`: Validate regression with configurable tolerance
- `configure_test_tolerance(test_name, **tolerances)`: Configure test tolerance
- `get_tolerance_info(test_name)`: Get tolerance information for test

### Convenience Functions

- `get_performance_config()`: Get global configuration instance
- `configure_test_tolerance(test_name, **tolerances)`: Configure test tolerance
- `get_test_tolerance(test_name, metric_type)`: Get test tolerance

## Migration Guide

### From Hardcoded Tolerances

**Before:**
```python
result = benchmark.validate_performance_regression(
    "my_test", metrics, tolerance_percent=15.0
)
```

**After:**
```python
# Option 1: Use configuration (recommended)
result = benchmark.validate_performance_regression("my_test", metrics)

# Option 2: Configure once, then use
benchmark.configure_test_tolerance("my_test", execution_time=15.0)
result = benchmark.validate_performance_regression("my_test", metrics)
```

### Setting Up New Tests

1. Run the test once to establish baseline
2. Monitor performance variance over several runs
3. Set appropriate tolerances based on observed variance
4. Document the rationale for chosen tolerances

## Future Enhancements

Potential future improvements to the tolerance system:

1. **Adaptive tolerances**: Automatically adjust tolerances based on historical variance
2. **Time-based tolerances**: Different tolerances for different times of day/week
3. **Hardware-specific tolerances**: Different tolerances for different hardware configurations
4. **Statistical analysis**: Use statistical methods to determine appropriate tolerances
5. **Integration with monitoring**: Connect with performance monitoring systems