# Developer Testing Guidelines

## Overview

This document provides comprehensive guidelines for developers working with the AlexPose testing framework. Following these guidelines ensures consistent, high-quality testing practices across the project.

## Testing Philosophy

### Core Principles

1. **Test-Driven Development (TDD)**: Write tests before implementation when possible
2. **Real Data Priority**: Use real data over mocks (target: >70% real data usage)
3. **Property-Based Testing**: Validate universal properties across all inputs
4. **Fast Feedback**: Maintain rapid development cycles with fast test execution
5. **Comprehensive Coverage**: Achieve component-specific coverage targets

### Testing Pyramid

Our testing strategy follows this distribution:

- **Property-Based Tests (30%)**: Universal correctness validation
- **Unit Tests (50%)**: Focused component testing
- **Integration Tests (15%)**: End-to-end workflow validation
- **End-to-End Tests (5%)**: Complete system validation

## Test Categories and Markers

### Available Test Markers

Use pytest markers to categorize your tests:

```python
import pytest

@pytest.mark.fast          # < 1 second execution
@pytest.mark.slow          # 1-30 seconds execution
@pytest.mark.performance   # 30+ seconds execution
@pytest.mark.integration   # Requires external resources
@pytest.mark.property      # Property-based tests with Hypothesis
@pytest.mark.hardware      # Requires specific hardware (GPU, camera)
```

### Test Execution Commands

```bash
# Development workflow - fast tests only
pytest -v -m "not slow and not performance"

# Pre-commit - all tests except performance
pytest -v -m "not performance"

# CI/CD pipeline - comprehensive testing
pytest -v -m "not performance" --cov=ambient --cov=server

# Property tests with specific profile
pytest -v -m property --hypothesis-profile=ci

# Performance tests
pytest -v -m performance
```

## Writing Effective Tests

### Unit Tests

#### Best Practices

1. **One Assertion Per Test**: Each test should verify one specific behavior
2. **Descriptive Names**: Test names should clearly describe what is being tested
3. **Arrange-Act-Assert**: Structure tests with clear setup, execution, and verification
4. **Independent Tests**: Tests should not depend on each other

#### Example

```python
import pytest
from ambient.core.frame import FrameSequence

class TestFrameSequence:
    """Test frame sequence functionality."""
    
    def test_frame_sequence_creation_with_valid_data(self):
        """Test that FrameSequence can be created with valid frame data."""
        # Arrange
        frame_data = [
            {"timestamp": 0.0, "data": b"frame1"},
            {"timestamp": 0.033, "data": b"frame2"}
        ]
        
        # Act
        sequence = FrameSequence(frame_data)
        
        # Assert
        assert len(sequence) == 2
        assert sequence[0]["timestamp"] == 0.0
    
    def test_frame_sequence_rejects_invalid_timestamp(self):
        """Test that FrameSequence rejects frames with invalid timestamps."""
        # Arrange
        invalid_frame_data = [
            {"timestamp": -1.0, "data": b"frame1"}  # Negative timestamp
        ]
        
        # Act & Assert
        with pytest.raises(ValueError, match="Invalid timestamp"):
            FrameSequence(invalid_frame_data)
```

### Property-Based Tests

#### When to Use Property Tests

- Testing universal properties that should hold for all inputs
- Validating mathematical relationships
- Testing serialization/deserialization round trips
- Verifying invariants after transformations

#### Writing Property Tests

```python
from hypothesis import given, strategies as st
import pytest

@pytest.mark.property
class TestGaitFeatureProperties:
    """Property-based tests for gait feature extraction."""
    
    @given(st.floats(min_value=0.1, max_value=10.0))
    def test_stride_time_always_positive(self, stride_time):
        """
        Feature: testing-enhancement, Property 1: Stride time validation
        For any valid stride time input, the calculated features should maintain positive values.
        **Validates: Requirements 2.1**
        """
        # Arrange
        from ambient.analysis.temporal_analyzer import TemporalAnalyzer
        analyzer = TemporalAnalyzer()
        
        # Act
        features = analyzer.calculate_stride_features(stride_time)
        
        # Assert
        assert features["stride_time"] > 0
        assert features["cadence"] > 0
    
    @given(st.lists(st.floats(min_value=0, max_value=1), min_size=33, max_size=33))
    def test_pose_landmarks_round_trip(self, landmark_confidences):
        """
        Feature: testing-enhancement, Property 2: Pose landmark serialization
        For any valid pose landmarks, serialization then deserialization should preserve data.
        **Validates: Requirements 1.3**
        """
        # Arrange
        from ambient.core.data_models import PoseLandmarks
        landmarks = PoseLandmarks.from_confidences(landmark_confidences)
        
        # Act
        serialized = landmarks.to_json()
        deserialized = PoseLandmarks.from_json(serialized)
        
        # Assert
        assert deserialized.confidences == landmarks.confidences
```

#### Hypothesis Configuration

Configure Hypothesis profiles in `pytest.ini`:

```ini
[tool:pytest]
hypothesis_profiles = {
    "dev": {"max_examples": 10, "deadline": 1000},
    "ci": {"max_examples": 100, "deadline": 5000},
    "thorough": {"max_examples": 1000, "deadline": 10000}
}
```

### Integration Tests

#### Best Practices

1. **Real Data Usage**: Use actual video files and datasets when possible
2. **End-to-End Workflows**: Test complete pipelines from input to output
3. **Error Handling**: Include failure scenarios and edge cases
4. **Resource Cleanup**: Ensure proper cleanup of test resources

#### Example

```python
import pytest
from pathlib import Path

@pytest.mark.integration
@pytest.mark.slow
class TestVideoProcessingPipeline:
    """Integration tests for complete video processing pipeline."""
    
    def test_complete_gait_analysis_workflow(self, sample_gait_videos):
        """Test complete workflow from video upload to classification."""
        # Arrange
        video_file = sample_gait_videos["normal_walking"]
        
        # Act
        from tests.integration.integration_framework import IntegrationTestFramework
        framework = IntegrationTestFramework()
        result = framework.test_complete_video_analysis_pipeline(
            video_file=video_file,
            expected_classification="normal"
        )
        
        # Assert
        assert result['pipeline_success'] is True
        assert result['classification'] == "normal"
        assert result['processing_time'] < 120.0  # 2 minutes max
```

## Test Data Management

### Real Data Fixtures

Use the `RealDataManager` for consistent test data:

```python
import pytest
from tests.fixtures.real_data_fixtures import RealDataManager

@pytest.fixture(scope="session")
def real_data_manager():
    """Provide real data manager for tests."""
    return RealDataManager()

@pytest.fixture
def sample_gait_videos(real_data_manager):
    """Provide real gait video samples."""
    return real_data_manager.get_sample_videos()

def test_with_real_video(sample_gait_videos):
    """Example test using real video data."""
    video_path = sample_gait_videos["normal_walking"]
    assert video_path.exists()
```

### Synthetic Data Generation

For property-based testing, create realistic synthetic data:

```python
from hypothesis import strategies as st
import numpy as np

# Custom strategy for pose landmarks
@st.composite
def pose_landmarks_strategy(draw):
    """Generate realistic pose landmarks."""
    landmarks = []
    for i in range(33):  # MediaPipe has 33 landmarks
        x = draw(st.floats(min_value=0, max_value=1920))
        y = draw(st.floats(min_value=0, max_value=1080))
        confidence = draw(st.floats(min_value=0.5, max_value=1.0))
        landmarks.append({"x": x, "y": y, "confidence": confidence})
    return landmarks
```

## Performance Testing

### Writing Performance Tests

```python
import pytest
from tests.performance.benchmark_framework import PerformanceBenchmark

@pytest.mark.performance
class TestVideoProcessingPerformance:
    """Performance tests for video processing."""
    
    def setup_method(self):
        """Set up performance testing framework."""
        self.benchmark = PerformanceBenchmark()
    
    def test_video_processing_performance(self):
        """Test video processing meets performance targets."""
        def process_video():
            # Your video processing code here
            pass
        
        # Benchmark the function
        metrics = self.benchmark.benchmark_function(process_video, iterations=1)
        
        # Validate performance targets
        assert metrics.execution_time < 60.0, f"Processing took {metrics.execution_time:.2f}s, target: <60s"
        assert metrics.peak_memory_mb < 2048, f"Memory usage {metrics.peak_memory_mb:.1f}MB, target: <2GB"
        
        # Check for regression
        regression_result = self.benchmark.validate_performance_regression(
            "video_processing", metrics, tolerance_percent=10.0
        )
        
        if regression_result['regression_detected']:
            pytest.fail(f"Performance regression detected: {regression_result['regressions']}")
```

## Coverage Requirements

### Component-Specific Targets

| Component | Minimum Coverage | Target Coverage |
|-----------|------------------|-----------------|
| Core (Frame, Config, Interfaces) | 90% | 95% |
| Domain (Pose, Gait, Classification) | 85% | 90% |
| Integration (API, Video, Storage) | 75% | 85% |
| Overall System | 80% | 85% |

### Measuring Coverage

```bash
# Run tests with coverage
pytest --cov=ambient --cov=server --cov-report=html --cov-report=term-missing

# Generate detailed coverage analysis
python tests/coverage/coverage_analyzer.py

# Check coverage thresholds
python tests/quality/quality_gates.py
```

## Debugging Test Failures

### Common Issues and Solutions

#### 1. Flaky Tests

**Problem**: Tests that pass/fail inconsistently

**Solutions**:
- Use fixed seeds for random data generation
- Add proper setup/teardown for test isolation
- Increase timeouts for async operations
- Use `pytest-rerunfailures` for temporary mitigation

```python
# Fix random seed for reproducible tests
import random
import numpy as np

def setup_method(self):
    random.seed(42)
    np.random.seed(42)
```

#### 2. Slow Tests

**Problem**: Tests taking too long to execute

**Solutions**:
- Use smaller test datasets
- Mock expensive operations
- Optimize test fixtures
- Mark as `@pytest.mark.slow` if necessary

#### 3. Memory Issues

**Problem**: Tests consuming too much memory

**Solutions**:
- Clean up large objects explicitly
- Use generators instead of lists for large datasets
- Monitor memory usage with performance tests

```python
def test_with_memory_cleanup(self):
    large_data = create_large_dataset()
    try:
        # Test logic here
        pass
    finally:
        del large_data
        gc.collect()
```

### Test Failure Analysis

Use the built-in failure analyzer:

```python
from tests.property.failure_analyzer import TestFailureAnalyzer

analyzer = TestFailureAnalyzer()
analysis = analyzer.analyze_failure(test_result)
print(analysis['debugging_suggestions'])
```

## Continuous Integration

### Pre-commit Hooks

Set up pre-commit hooks to run tests automatically:

```yaml
# .pre-commit-config.yaml
repos:
  - repo: local
    hooks:
      - id: fast-tests
        name: Run fast tests
        entry: pytest -m "fast" --tb=short
        language: system
        pass_filenames: false
```

### CI/CD Integration

Use the provided CI scripts:

```bash
# Run CI test suite
./scripts/ci-test-runner.sh --category all --quality-gates --coverage-report

# Generate coverage reports
./scripts/coverage-reporter.sh --threshold 85 --validate
```

## Best Practices Summary

### Do's

✅ **Write tests first** when possible (TDD approach)
✅ **Use descriptive test names** that explain the behavior being tested
✅ **Prefer real data** over mocks when feasible
✅ **Write property tests** for universal behaviors
✅ **Keep tests independent** and isolated
✅ **Use appropriate test markers** for categorization
✅ **Clean up resources** in test teardown
✅ **Monitor test performance** and optimize slow tests
✅ **Maintain high coverage** on critical components
✅ **Document complex test scenarios**

### Don'ts

❌ **Don't write tests that depend on each other**
❌ **Don't ignore flaky tests** - fix them immediately
❌ **Don't mock everything** - use real data when possible
❌ **Don't write overly complex tests** - keep them simple and focused
❌ **Don't skip edge cases** - they often reveal bugs
❌ **Don't forget to test error conditions**
❌ **Don't leave TODO comments** in test code
❌ **Don't commit failing tests** to main branch
❌ **Don't ignore performance regressions**
❌ **Don't write tests without assertions**

## Getting Help

### Resources

- **Testing Strategy**: `docs/specs/testing-strategy.md`
- **Property Testing Guide**: `docs/testing/property-testing-guide.md`
- **Debugging Guide**: `docs/testing/debugging-guide.md`
- **Test Data Management**: `docs/testing/test-data-management.md`

### Common Commands

```bash
# Run specific test categories
pytest -m fast                    # Fast tests only
pytest -m "slow and not performance"  # Slow tests, no performance
pytest -m integration             # Integration tests only
pytest -m property --hypothesis-profile=dev  # Property tests

# Coverage analysis
pytest --cov=ambient --cov-report=html
python tests/coverage/coverage_analyzer.py

# Quality gates
python tests/quality/quality_gates.py --no-fail

# Performance testing
pytest -m performance -v
```

### Support

For questions or issues with testing:

1. Check existing test examples in the codebase
2. Review this documentation and related guides
3. Run the quality gates to identify specific issues
4. Use the failure analyzer for debugging test failures

Remember: Good tests are an investment in code quality and developer productivity. Take the time to write them well!