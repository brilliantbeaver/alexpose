# Test Debugging Guide

## Overview

This guide provides comprehensive strategies for debugging test failures in the AlexPose testing framework. Effective debugging is crucial for maintaining test reliability and developer productivity.

## Test Failure Analysis Framework

### Automated Failure Analysis

The system includes an automated failure analyzer that categorizes and provides debugging suggestions:

```python
from tests.property.failure_analyzer import TestFailureAnalyzer

analyzer = TestFailureAnalyzer()
analysis = analyzer.analyze_failure(test_result)

print("Failure Category:", analysis['failure_category'])
print("Root Causes:", analysis['root_cause_analysis'])
print("Debugging Suggestions:", analysis['debugging_suggestions'])
```

### Failure Categories

The analyzer categorizes failures into these types:

1. **Assertion Failures**: Test expectations not met
2. **Timeout Failures**: Tests exceeding time limits
3. **Memory Failures**: Out of memory or memory leaks
4. **Dependency Failures**: Missing imports or external dependencies
5. **Property Test Failures**: Hypothesis found falsifying examples
6. **Configuration Failures**: Environment or setup issues

## Debugging Strategies by Test Type

### Unit Test Debugging

#### Common Issues and Solutions

**1. Assertion Failures**

```python
# Problem: Unexpected assertion failure
def test_stride_calculation():
    result = calculate_stride_time([1.0, 2.0, 3.0])
    assert result == 2.0  # Fails with result = 1.0

# Debugging approach:
def test_stride_calculation_debug():
    timestamps = [1.0, 2.0, 3.0]
    result = calculate_stride_time(timestamps)
    
    # Add debugging output
    print(f"Input: {timestamps}")
    print(f"Expected: 2.0, Got: {result}")
    
    # Check intermediate calculations
    intervals = [timestamps[i+1] - timestamps[i] for i in range(len(timestamps)-1)]
    print(f"Intervals: {intervals}")
    print(f"Average interval: {sum(intervals) / len(intervals)}")
    
    assert result == 2.0
```

**2. Floating Point Precision Issues**

```python
# Problem: Floating point comparison failures
def test_with_float_precision():
    result = complex_calculation()
    # assert result == 1.0  # May fail due to precision
    
    # Solution: Use approximate comparison
    assert abs(result - 1.0) < 1e-6
    
    # Or use pytest.approx
    import pytest
    assert result == pytest.approx(1.0, rel=1e-6)
```

**3. Test Data Issues**

```python
# Problem: Test fails with certain data
def test_with_data_validation():
    test_data = load_test_data()
    
    # Add data validation
    assert test_data is not None, "Test data failed to load"
    assert len(test_data) > 0, "Test data is empty"
    assert all(isinstance(item, dict) for item in test_data), "Invalid data format"
    
    # Proceed with test
    result = process_data(test_data)
    assert result is not None
```

### Property Test Debugging

#### Understanding Hypothesis Failures

When Hypothesis finds a falsifying example:

```
FAILED test_property.py::test_stride_time_positive
>   assert result > 0
E   assert -0.5 > 0

Falsifying example: test_stride_time_positive(
    timestamps=[2.0, 1.0, 3.0]  # Non-monotonic sequence
)

You can reproduce this example by temporarily adding @reproduce_failure('6.82.4', b'AXic...') as a decorator on your test
```

#### Debugging Property Test Failures

**1. Reproduce the Exact Failure**

```python
from hypothesis import reproduce_failure

@reproduce_failure('6.82.4', b'AXic...')  # From Hypothesis output
@given(st.lists(st.floats()))
def test_stride_time_positive(self, timestamps):
    """Reproduce specific failure for debugging."""
    result = calculate_stride_time(timestamps)
    
    # Add debugging for the specific case
    print(f"Failing input: {timestamps}")
    print(f"Result: {result}")
    
    assert result > 0
```

**2. Add Assumptions to Filter Invalid Inputs**

```python
from hypothesis import assume

@given(st.lists(st.floats(), min_size=2))
def test_stride_time_with_assumptions(self, timestamps):
    # Filter out invalid inputs
    assume(all(t >= 0 for t in timestamps))  # Only positive timestamps
    assume(len(set(timestamps)) == len(timestamps))  # No duplicates
    assume(timestamps == sorted(timestamps))  # Only sorted sequences
    
    result = calculate_stride_time(timestamps)
    assert result > 0
```

**3. Use Example Decorator for Edge Cases**

```python
from hypothesis import example

@given(st.floats())
@example(0.0)  # Always test zero
@example(float('inf'))  # Always test infinity
@example(float('nan'))  # Always test NaN
def test_with_edge_cases(self, value):
    if math.isnan(value) or math.isinf(value):
        with pytest.raises(ValueError):
            process_value(value)
    else:
        result = process_value(value)
        assert isinstance(result, float)
```

**4. Simplify Complex Strategies**

```python
# Complex strategy that might generate invalid data
@st.composite
def complex_gait_data(draw):
    # Simplified version for debugging
    return {
        'stride_time': draw(st.floats(min_value=0.5, max_value=3.0)),
        'cadence': draw(st.floats(min_value=60, max_value=180))
    }

# Debug with simpler strategy first
@st.composite
def simple_gait_data(draw):
    stride_time = draw(st.floats(min_value=1.0, max_value=2.0))
    return {
        'stride_time': stride_time,
        'cadence': 60.0 / stride_time  # Ensure consistency
    }
```

### Integration Test Debugging

#### Common Integration Test Issues

**1. External Service Dependencies**

```python
import pytest
from unittest.mock import patch

@pytest.mark.integration
def test_with_external_service():
    # Check if external service is available
    try:
        response = requests.get("http://external-service/health", timeout=5)
        service_available = response.status_code == 200
    except requests.RequestException:
        service_available = False
    
    if not service_available:
        pytest.skip("External service not available")
    
    # Proceed with integration test
    result = call_external_service()
    assert result is not None
```

**2. Database Connection Issues**

```python
@pytest.mark.integration
def test_database_operations():
    # Verify database connection
    try:
        db = get_database_connection()
        db.execute("SELECT 1")
    except Exception as e:
        pytest.fail(f"Database connection failed: {e}")
    
    # Test database operations
    result = perform_database_operation()
    assert result is not None
```

**3. File System Dependencies**

```python
import tempfile
import shutil

@pytest.mark.integration
def test_file_operations():
    # Use temporary directory for testing
    with tempfile.TemporaryDirectory() as temp_dir:
        test_file = Path(temp_dir) / "test_data.json"
        
        # Ensure we can write to the directory
        try:
            test_file.write_text('{"test": true}')
        except PermissionError:
            pytest.skip("Cannot write to test directory")
        
        # Proceed with file operations test
        result = process_file(test_file)
        assert result is not None
```

### Performance Test Debugging

#### Performance Test Failures

**1. Performance Regression Detection**

```python
def test_performance_with_debugging():
    benchmark = PerformanceBenchmark()
    
    def operation_to_test():
        return expensive_operation()
    
    # Measure performance
    metrics = benchmark.benchmark_function(operation_to_test, iterations=5)
    
    # Debug performance issues
    print(f"Execution time: {metrics.execution_time:.2f}s")
    print(f"Memory usage: {metrics.memory_usage_mb:.1f}MB")
    print(f"Peak memory: {metrics.peak_memory_mb:.1f}MB")
    
    # Check for regression
    regression_result = benchmark.validate_performance_regression(
        "operation_test", metrics, tolerance_percent=20.0
    )
    
    if regression_result['regression_detected']:
        print("Performance regression details:")
        for regression in regression_result['regressions']:
            print(f"  - {regression}")
        
        # Provide debugging information
        baseline = regression_result['baseline_metrics']
        current = regression_result['current_metrics']
        print(f"Baseline: {baseline.execution_time:.2f}s")
        print(f"Current: {current.execution_time:.2f}s")
        
        pytest.fail("Performance regression detected")
```

**2. Memory Leak Detection**

```python
import gc
import psutil

def test_memory_leak_debugging():
    process = psutil.Process()
    
    # Measure initial memory
    initial_memory = process.memory_info().rss / 1024 / 1024
    
    # Run operation multiple times
    for i in range(10):
        result = potentially_leaky_operation()
        
        # Force garbage collection
        gc.collect()
        
        # Check memory after each iteration
        current_memory = process.memory_info().rss / 1024 / 1024
        memory_increase = current_memory - initial_memory
        
        print(f"Iteration {i}: Memory increase: {memory_increase:.1f}MB")
        
        # Fail if memory keeps growing
        if memory_increase > 100:  # 100MB threshold
            pytest.fail(f"Potential memory leak detected: {memory_increase:.1f}MB increase")
```

## Debugging Tools and Techniques

### Using pytest Debugging Features

**1. Verbose Output**

```bash
# Run tests with verbose output
pytest -v test_file.py

# Show local variables on failure
pytest -l test_file.py

# Drop into debugger on failure
pytest --pdb test_file.py

# Stop on first failure
pytest -x test_file.py
```

**2. Custom Debugging Fixtures**

```python
import pytest

@pytest.fixture
def debug_info():
    """Provide debugging information for tests."""
    import sys
    import platform
    
    return {
        'python_version': sys.version,
        'platform': platform.platform(),
        'memory_available': psutil.virtual_memory().available / 1024 / 1024
    }

def test_with_debug_info(debug_info):
    print(f"Running on: {debug_info['platform']}")
    print(f"Python: {debug_info['python_version']}")
    print(f"Available memory: {debug_info['memory_available']:.1f}MB")
    
    # Your test logic here
```

### Logging for Test Debugging

**1. Structured Test Logging**

```python
import logging
import pytest

# Configure logging for tests
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)

def test_with_logging():
    logger.info("Starting test execution")
    
    try:
        result = complex_operation()
        logger.debug(f"Operation result: {result}")
        
        assert result is not None
        logger.info("Test passed successfully")
        
    except Exception as e:
        logger.error(f"Test failed with error: {e}")
        raise
```

**2. Conditional Debug Output**

```python
import os

DEBUG_TESTS = os.getenv('DEBUG_TESTS', 'false').lower() == 'true'

def debug_print(*args, **kwargs):
    """Print debug information only when DEBUG_TESTS is enabled."""
    if DEBUG_TESTS:
        print(*args, **kwargs)

def test_with_conditional_debug():
    debug_print("Debug mode enabled")
    
    result = operation_under_test()
    debug_print(f"Operation result: {result}")
    
    assert result is not None
```

### Memory Debugging

**1. Memory Profiling**

```python
import tracemalloc

def test_with_memory_profiling():
    # Start memory tracing
    tracemalloc.start()
    
    # Take snapshot before operation
    snapshot1 = tracemalloc.take_snapshot()
    
    # Run operation
    result = memory_intensive_operation()
    
    # Take snapshot after operation
    snapshot2 = tracemalloc.take_snapshot()
    
    # Compare snapshots
    top_stats = snapshot2.compare_to(snapshot1, 'lineno')
    
    print("Top 10 memory allocations:")
    for stat in top_stats[:10]:
        print(stat)
    
    # Stop tracing
    tracemalloc.stop()
    
    assert result is not None
```

**2. Memory Leak Detection**

```python
import weakref
import gc

def test_memory_leak_detection():
    """Test for memory leaks using weak references."""
    objects_created = []
    
    def create_object():
        obj = SomeClass()
        # Keep weak reference to track if object is garbage collected
        objects_created.append(weakref.ref(obj))
        return obj
    
    # Create and use objects
    for _ in range(10):
        obj = create_object()
        process_object(obj)
        del obj
    
    # Force garbage collection
    gc.collect()
    
    # Check if objects were properly garbage collected
    alive_objects = [ref for ref in objects_created if ref() is not None]
    
    if alive_objects:
        pytest.fail(f"Memory leak detected: {len(alive_objects)} objects not garbage collected")
```

### Test Environment Debugging

**1. Environment Validation**

```python
import os
import sys
import pytest

def validate_test_environment():
    """Validate test environment setup."""
    issues = []
    
    # Check Python version
    if sys.version_info < (3, 11):
        issues.append(f"Python version {sys.version} is too old")
    
    # Check required environment variables
    required_vars = ['OPENAI_API_KEY', 'GOOGLE_API_KEY']
    for var in required_vars:
        if not os.getenv(var):
            issues.append(f"Missing environment variable: {var}")
    
    # Check available memory
    available_memory = psutil.virtual_memory().available / 1024 / 1024 / 1024
    if available_memory < 2:  # Less than 2GB
        issues.append(f"Low available memory: {available_memory:.1f}GB")
    
    return issues

@pytest.fixture(scope="session", autouse=True)
def check_test_environment():
    """Automatically check test environment before running tests."""
    issues = validate_test_environment()
    if issues:
        pytest.skip(f"Test environment issues: {'; '.join(issues)}")
```

**2. Configuration Debugging**

```python
def test_configuration_debugging():
    """Debug configuration issues."""
    from ambient.core.config import get_config
    
    config = get_config()
    
    # Print configuration for debugging
    print("Current configuration:")
    for key, value in config.items():
        # Mask sensitive values
        if 'key' in key.lower() or 'password' in key.lower():
            print(f"  {key}: {'*' * len(str(value))}")
        else:
            print(f"  {key}: {value}")
    
    # Validate critical configuration
    assert config.get('pose_estimator') is not None, "Pose estimator not configured"
    assert config.get('llm_model') is not None, "LLM model not configured"
```

## Debugging Flaky Tests

### Identifying Flaky Tests

**1. Statistical Analysis**

```python
import subprocess
import json

def run_test_multiple_times(test_name, iterations=10):
    """Run a test multiple times to detect flakiness."""
    results = []
    
    for i in range(iterations):
        result = subprocess.run([
            'pytest', test_name, '--tb=no', '-q'
        ], capture_output=True, text=True)
        
        results.append({
            'iteration': i + 1,
            'passed': result.returncode == 0,
            'output': result.stdout + result.stderr
        })
    
    # Analyze results
    pass_count = sum(1 for r in results if r['passed'])
    pass_rate = pass_count / iterations
    
    print(f"Test: {test_name}")
    print(f"Pass rate: {pass_rate:.2%} ({pass_count}/{iterations})")
    
    if pass_rate < 1.0:
        print("FLAKY TEST DETECTED!")
        print("Failed iterations:")
        for r in results:
            if not r['passed']:
                print(f"  Iteration {r['iteration']}: {r['output']}")
    
    return pass_rate
```

**2. Flakiness Root Causes**

Common causes of flaky tests:

- **Race Conditions**: Timing-dependent behavior
- **External Dependencies**: Network, file system, databases
- **Random Data**: Non-deterministic test data generation
- **Resource Contention**: Memory, CPU, or I/O limitations
- **State Pollution**: Tests affecting each other

### Fixing Flaky Tests

**1. Race Condition Fixes**

```python
import time
import threading

# Problem: Race condition in async test
def test_async_operation_flaky():
    start_async_operation()
    time.sleep(0.1)  # Unreliable timing
    assert operation_completed()

# Solution: Proper synchronization
def test_async_operation_fixed():
    event = threading.Event()
    
    def completion_callback():
        event.set()
    
    start_async_operation(callback=completion_callback)
    
    # Wait for completion with timeout
    assert event.wait(timeout=5.0), "Operation did not complete within 5 seconds"
    assert operation_completed()
```

**2. External Dependency Isolation**

```python
from unittest.mock import patch, MagicMock

# Problem: Flaky due to external API
def test_external_api_flaky():
    response = call_external_api()
    assert response['status'] == 'success'

# Solution: Mock external dependencies
@patch('module.call_external_api')
def test_external_api_fixed(mock_api):
    mock_api.return_value = {'status': 'success', 'data': 'test'}
    
    response = call_external_api()
    assert response['status'] == 'success'
    mock_api.assert_called_once()
```

**3. Deterministic Random Data**

```python
import random
import numpy as np

# Problem: Non-deterministic random data
def test_with_random_data_flaky():
    data = [random.random() for _ in range(10)]
    result = process_data(data)
    assert result > 0.5  # May fail randomly

# Solution: Fixed seed for reproducibility
def test_with_random_data_fixed():
    random.seed(42)
    np.random.seed(42)
    
    data = [random.random() for _ in range(10)]
    result = process_data(data)
    assert result > 0.5  # Deterministic result
```

## Advanced Debugging Techniques

### Test Isolation

**1. Fixture Isolation**

```python
import pytest
import tempfile
import shutil

@pytest.fixture
def isolated_environment():
    """Provide isolated environment for each test."""
    # Create temporary directory
    temp_dir = tempfile.mkdtemp()
    
    # Change to temporary directory
    original_cwd = os.getcwd()
    os.chdir(temp_dir)
    
    yield temp_dir
    
    # Cleanup
    os.chdir(original_cwd)
    shutil.rmtree(temp_dir)

def test_with_isolation(isolated_environment):
    """Test runs in isolated environment."""
    # Test logic here - won't affect other tests
    pass
```

**2. Database Transaction Rollback**

```python
@pytest.fixture
def db_transaction():
    """Provide database transaction that rolls back after test."""
    connection = get_database_connection()
    transaction = connection.begin()
    
    yield connection
    
    # Rollback transaction to undo all changes
    transaction.rollback()
    connection.close()

def test_database_operations(db_transaction):
    """Test database operations with automatic rollback."""
    # Database changes will be rolled back automatically
    db_transaction.execute("INSERT INTO test_table VALUES (1, 'test')")
    result = db_transaction.execute("SELECT * FROM test_table WHERE id = 1")
    assert result.fetchone() is not None
```

### Debugging Test Infrastructure

**1. Test Runner Debugging**

```python
# Custom pytest plugin for debugging
class DebugPlugin:
    def pytest_runtest_setup(self, item):
        print(f"Setting up test: {item.name}")
    
    def pytest_runtest_call(self, item):
        print(f"Running test: {item.name}")
    
    def pytest_runtest_teardown(self, item):
        print(f"Tearing down test: {item.name}")
    
    def pytest_runtest_logreport(self, report):
        if report.failed:
            print(f"Test failed: {report.nodeid}")
            print(f"Failure reason: {report.longrepr}")

# Register plugin
pytest.main(['-p', 'no:cacheprovider', '--tb=short'], plugins=[DebugPlugin()])
```

**2. Fixture Dependency Debugging**

```python
@pytest.fixture
def debug_fixture_order():
    """Debug fixture execution order."""
    print("Fixture setup order:")
    
    import inspect
    frame = inspect.currentframe()
    while frame:
        if 'pytest' in str(frame.f_code.co_filename):
            break
        print(f"  {frame.f_code.co_name} in {frame.f_code.co_filename}")
        frame = frame.f_back
    
    yield
    
    print("Fixture teardown")
```

## Best Practices for Test Debugging

### Do's

✅ **Use systematic debugging approaches** - don't guess randomly
✅ **Reproduce failures consistently** before attempting fixes
✅ **Add logging and debug output** to understand test behavior
✅ **Use appropriate debugging tools** for different failure types
✅ **Isolate tests properly** to prevent interference
✅ **Fix flaky tests immediately** - don't ignore them
✅ **Document debugging steps** for complex issues
✅ **Use version control** to track debugging changes

### Don'ts

❌ **Don't ignore intermittent failures** - they indicate real problems
❌ **Don't add random delays** to fix timing issues
❌ **Don't disable failing tests** without understanding the cause
❌ **Don't make tests overly complex** to work around issues
❌ **Don't rely on external services** without proper mocking
❌ **Don't skip environment validation** in CI/CD
❌ **Don't leave debug code** in production tests
❌ **Don't assume tests are correct** - they can have bugs too

## Debugging Checklist

When debugging test failures, work through this checklist:

### Initial Analysis
- [ ] What type of test is failing? (unit, integration, property, performance)
- [ ] Is the failure consistent or intermittent?
- [ ] What changed since the test last passed?
- [ ] Are there any error messages or stack traces?

### Environment Check
- [ ] Is the test environment properly configured?
- [ ] Are all dependencies available and correct versions?
- [ ] Is there sufficient memory and disk space?
- [ ] Are required services running?

### Test-Specific Debugging
- [ ] Can the failure be reproduced locally?
- [ ] Are test assumptions still valid?
- [ ] Is test data correct and available?
- [ ] Are mocks and fixtures working correctly?

### Code Analysis
- [ ] Has the implementation changed?
- [ ] Are there any obvious bugs in the test or implementation?
- [ ] Do error messages provide useful information?
- [ ] Are there any resource leaks or cleanup issues?

### Resolution
- [ ] Fix identified issues
- [ ] Verify fix resolves the problem
- [ ] Run related tests to ensure no regressions
- [ ] Document the issue and resolution if complex

Remember: Effective debugging is a skill that improves with practice. Take time to understand failures thoroughly rather than applying quick fixes that might mask underlying issues.