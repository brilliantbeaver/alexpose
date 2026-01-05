# Property-Based Testing Guide

## Overview

Property-based testing (PBT) is a powerful testing methodology that validates software correctness by testing universal properties across many generated inputs. This guide provides comprehensive information on implementing property-based tests in the AlexPose system.

## What is Property-Based Testing?

Property-based testing differs from traditional example-based testing by:

- **Universal Validation**: Tests properties that should hold for all valid inputs
- **Automated Input Generation**: Uses libraries like Hypothesis to generate test data
- **Comprehensive Coverage**: Explores edge cases that developers might miss
- **Formal Specifications**: Encodes requirements as executable properties

### Example Comparison

**Traditional Test**:
```python
def test_stride_time_calculation():
    # Test with specific example
    assert calculate_stride_time([1.0, 2.0, 3.0]) == 2.0
```

**Property-Based Test**:
```python
@given(st.lists(st.floats(min_value=0.1, max_value=10.0), min_size=2))
def test_stride_time_always_positive(timestamps):
    # Test property that holds for all valid inputs
    result = calculate_stride_time(timestamps)
    assert result > 0  # Property: stride time is always positive
```

## Core Concepts

### Properties

A property is a characteristic that should hold true across all valid executions of a system. Properties serve as the bridge between human-readable specifications and machine-verifiable correctness guarantees.

#### Common Property Patterns

1. **Invariants**: Properties preserved after transformations
2. **Round Trip**: Combining operation with its inverse returns original value
3. **Idempotence**: Doing operation twice equals doing it once
4. **Metamorphic**: Relationships between different inputs/outputs
5. **Model-Based**: Comparing optimized vs. reference implementation
6. **Error Conditions**: Invalid inputs properly signal errors

### Hypothesis Strategies

Hypothesis provides strategies for generating test data:

```python
from hypothesis import strategies as st

# Basic strategies
st.integers(min_value=0, max_value=100)
st.floats(min_value=0.0, max_value=1.0)
st.text(min_size=1, max_size=50)
st.lists(st.integers(), min_size=0, max_size=10)

# Composite strategies
st.tuples(st.floats(), st.floats())  # Pairs of floats
st.dictionaries(st.text(), st.integers())  # String -> int mappings
```

## AlexPose Property Testing Framework

### Property Test Registry

The system uses a registry to manage all correctness properties:

```python
from tests.property.property_registry import PropertyTestRegistry
from tests.property.base_property import PropertyTestInterface

class MyProperty(PropertyTestInterface):
    def get_strategy(self):
        return st.integers(min_value=1, max_value=100)
    
    def validate_property(self, test_data):
        # Property validation logic
        return True
    
    def get_requirement_mapping(self):
        return ["1.1", "1.2"]

# Register the property
registry = PropertyTestRegistry()
registry.register_property("my_property", MyProperty())
```

### Custom Strategies for Domain Objects

Create domain-specific strategies for realistic test data:

```python
# Pose landmark strategy
@st.composite
def pose_landmarks_strategy(draw):
    """Generate realistic MediaPipe pose landmarks."""
    landmarks = []
    for i in range(33):  # MediaPipe has 33 landmarks
        x = draw(st.floats(min_value=0, max_value=1920))
        y = draw(st.floats(min_value=0, max_value=1080))
        confidence = draw(st.floats(min_value=0.5, max_value=1.0))
        landmarks.append({
            'x': x, 'y': y, 'confidence': confidence,
            'visibility': draw(st.floats(min_value=0.0, max_value=1.0))
        })
    return landmarks

# Gait feature strategy
@st.composite
def gait_features_strategy(draw):
    """Generate realistic gait features."""
    return {
        'stride_time': draw(st.floats(min_value=0.8, max_value=2.0)),
        'cadence': draw(st.floats(min_value=60, max_value=180)),
        'stride_length': draw(st.floats(min_value=0.8, max_value=2.5)),
        'step_width': draw(st.floats(min_value=0.05, max_value=0.3)),
        'foot_angle': draw(st.floats(min_value=-30, max_value=30))
    }

# Video frame sequence strategy
@st.composite
def frame_sequence_strategy(draw):
    """Generate realistic frame sequences."""
    num_frames = draw(st.integers(min_value=30, max_value=300))  # 1-10 seconds at 30fps
    frame_rate = draw(st.floats(min_value=15.0, max_value=60.0))
    
    frames = []
    for i in range(num_frames):
        timestamp = i / frame_rate
        frame_data = {
            'timestamp': timestamp,
            'width': draw(st.integers(min_value=640, max_value=1920)),
            'height': draw(st.integers(min_value=480, max_value=1080)),
            'channels': 3
        }
        frames.append(frame_data)
    
    return frames
```

## Writing Property Tests

### Basic Property Test Structure

```python
import pytest
from hypothesis import given, strategies as st

@pytest.mark.property
class TestGaitAnalysisProperties:
    """Property-based tests for gait analysis."""
    
    @given(gait_features_strategy())
    def test_gait_classification_consistency(self, features):
        """
        Feature: testing-enhancement, Property 1: Classification consistency
        For any valid gait features, classification should be deterministic.
        **Validates: Requirements 3.1**
        """
        from ambient.classification.llm_classifier import LLMClassifier
        
        classifier = LLMClassifier()
        
        # Run classification multiple times
        result1 = classifier.classify_gait(features)
        result2 = classifier.classify_gait(features)
        
        # Property: same input should produce same output
        assert result1['classification'] == result2['classification']
        assert abs(result1['confidence'] - result2['confidence']) < 0.01
```

### Round Trip Properties

Essential for serialization, parsing, and data transformation:

```python
@given(pose_landmarks_strategy())
def test_pose_landmarks_serialization_round_trip(self, landmarks):
    """
    Feature: testing-enhancement, Property 2: Pose landmarks round trip
    For any valid pose landmarks, serialization then deserialization preserves data.
    **Validates: Requirements 1.3**
    """
    from ambient.core.data_models import PoseLandmarks
    
    # Create landmarks object
    pose = PoseLandmarks(landmarks)
    
    # Round trip: serialize then deserialize
    serialized = pose.to_json()
    deserialized = PoseLandmarks.from_json(serialized)
    
    # Property: round trip preserves data
    assert len(deserialized.landmarks) == len(pose.landmarks)
    for orig, deser in zip(pose.landmarks, deserialized.landmarks):
        assert abs(orig['x'] - deser['x']) < 1e-6
        assert abs(orig['y'] - deser['y']) < 1e-6
        assert abs(orig['confidence'] - deser['confidence']) < 1e-6
```

### Invariant Properties

Properties that remain constant despite transformations:

```python
@given(frame_sequence_strategy())
def test_frame_sequence_length_invariant(self, frames):
    """
    Feature: testing-enhancement, Property 3: Frame sequence length invariant
    For any frame sequence, filtering operations preserve relative ordering.
    **Validates: Requirements 2.1**
    """
    from ambient.core.frame import FrameSequence
    
    sequence = FrameSequence(frames)
    original_length = len(sequence)
    
    # Apply filtering (should preserve order, may reduce length)
    filtered = sequence.filter_by_quality(min_quality=0.5)
    
    # Invariant: filtered sequence length <= original length
    assert len(filtered) <= original_length
    
    # Invariant: timestamps remain in ascending order
    timestamps = [frame['timestamp'] for frame in filtered.frames]
    assert timestamps == sorted(timestamps)
```

### Metamorphic Properties

Test relationships between different inputs/outputs:

```python
@given(st.floats(min_value=0.1, max_value=5.0), st.floats(min_value=0.1, max_value=5.0))
def test_stride_time_comparison_property(self, time1, time2):
    """
    Feature: testing-enhancement, Property 4: Stride time comparison
    For any two stride times, the longer time should produce lower cadence.
    **Validates: Requirements 2.2**
    """
    from ambient.analysis.temporal_analyzer import TemporalAnalyzer
    
    analyzer = TemporalAnalyzer()
    
    cadence1 = analyzer.calculate_cadence(time1)
    cadence2 = analyzer.calculate_cadence(time2)
    
    # Metamorphic property: longer stride time → lower cadence
    if time1 > time2:
        assert cadence1 < cadence2
    elif time1 < time2:
        assert cadence1 > cadence2
    else:
        assert abs(cadence1 - cadence2) < 1e-6
```

### Error Condition Properties

Test that invalid inputs are properly handled:

```python
@given(st.lists(st.floats(), max_size=1))  # Lists with 0 or 1 elements
def test_insufficient_data_error_handling(self, insufficient_data):
    """
    Feature: testing-enhancement, Property 5: Error handling for insufficient data
    For any input with insufficient data points, appropriate errors should be raised.
    **Validates: Requirements 4.1**
    """
    from ambient.analysis.gait_analyzer import GaitAnalyzer
    
    analyzer = GaitAnalyzer()
    
    # Property: insufficient data should raise appropriate error
    with pytest.raises((ValueError, InsufficientDataError)):
        analyzer.extract_features(insufficient_data)
```

## Advanced Techniques

### Stateful Testing

Test sequences of operations:

```python
from hypothesis.stateful import RuleBasedStateMachine, rule, invariant

class GaitAnalysisStateMachine(RuleBasedStateMachine):
    """Stateful testing for gait analysis workflow."""
    
    def __init__(self):
        super().__init__()
        self.analyzer = GaitAnalyzer()
        self.features = []
    
    @rule(new_features=gait_features_strategy())
    def add_features(self, new_features):
        """Add new gait features to the analysis."""
        self.features.append(new_features)
        self.analyzer.add_sample(new_features)
    
    @rule()
    def analyze_current_features(self):
        """Analyze current feature set."""
        if self.features:
            result = self.analyzer.get_analysis()
            assert result is not None
    
    @invariant()
    def features_count_consistent(self):
        """Invariant: analyzer should track same number of features."""
        assert len(self.analyzer.samples) == len(self.features)

# Run stateful test
TestGaitAnalysis = GaitAnalysisStateMachine.TestCase
```

### Composite Strategies

Build complex test data from simpler components:

```python
@st.composite
def complete_gait_session_strategy(draw):
    """Generate a complete gait analysis session."""
    # Session metadata
    session_id = draw(st.text(min_size=5, max_size=20))
    subject_id = draw(st.integers(min_value=1, max_value=1000))
    
    # Video information
    video_duration = draw(st.floats(min_value=10.0, max_value=300.0))
    frame_rate = draw(st.floats(min_value=15.0, max_value=60.0))
    
    # Generate frame sequence
    num_frames = int(video_duration * frame_rate)
    frames = draw(st.lists(
        frame_data_strategy(),
        min_size=num_frames,
        max_size=num_frames
    ))
    
    # Generate pose landmarks for each frame
    pose_sequences = draw(st.lists(
        pose_landmarks_strategy(),
        min_size=num_frames,
        max_size=num_frames
    ))
    
    return {
        'session_id': session_id,
        'subject_id': subject_id,
        'video_duration': video_duration,
        'frame_rate': frame_rate,
        'frames': frames,
        'pose_sequences': pose_sequences
    }
```

## Configuration and Profiles

### Hypothesis Profiles

Configure different execution profiles:

```python
# In pytest.ini or conftest.py
from hypothesis import settings, Verbosity

# Development profile - fast feedback
settings.register_profile("dev", max_examples=10, deadline=1000)

# CI profile - thorough but time-bounded
settings.register_profile("ci", max_examples=100, deadline=5000)

# Thorough profile - comprehensive testing
settings.register_profile("thorough", max_examples=1000, deadline=10000, verbosity=Verbosity.verbose)

# Load profile based on environment
import os
settings.load_profile(os.getenv("HYPOTHESIS_PROFILE", "dev"))
```

### Custom Settings

Apply settings to specific tests:

```python
from hypothesis import settings

@settings(max_examples=500, deadline=None)
@given(complex_data_strategy())
def test_complex_property(self, data):
    """Test that requires many examples and no time limit."""
    # Complex property validation
    pass
```

## Debugging Property Test Failures

### Understanding Failure Output

When a property test fails, Hypothesis provides:

1. **Falsifying Example**: The specific input that caused failure
2. **Shrinking Process**: Simplified version of the failing input
3. **Reproduction**: Exact steps to reproduce the failure

### Example Failure Analysis

```
FAILED test_property.py::test_stride_time_positive - AssertionError
>   assert result > 0
E   assert -0.5 > 0

Falsifying example: test_stride_time_positive(
    timestamps=[1.0, 0.5, 2.0]  # Non-monotonic timestamps
)
```

### Debugging Strategies

1. **Add Assumptions**: Filter out invalid inputs
```python
from hypothesis import assume

@given(st.lists(st.floats(), min_size=2))
def test_with_assumptions(self, timestamps):
    assume(all(t >= 0 for t in timestamps))  # Only positive timestamps
    assume(timestamps == sorted(timestamps))  # Only sorted timestamps
    # Test logic here
```

2. **Use Examples Decorator**: Test specific known cases
```python
from hypothesis import example

@given(st.floats())
@example(0.0)  # Always test zero
@example(float('inf'))  # Always test infinity
def test_with_examples(self, value):
    # Test logic here
```

3. **Reproduce Failures**: Use the exact failing example
```python
def test_reproduce_failure():
    """Reproduce specific failure for debugging."""
    failing_input = [1.0, 0.5, 2.0]  # From Hypothesis output
    result = calculate_stride_time(failing_input)
    # Debug the specific failure
```

## Integration with Requirements

### Requirement Traceability

Each property test must reference the requirements it validates:

```python
@given(pose_landmarks_strategy())
def test_pose_detection_accuracy(self, landmarks):
    """
    Feature: testing-enhancement, Property 6: Pose detection accuracy
    For any valid pose landmarks, detection confidence should meet minimum thresholds.
    **Validates: Requirements 2.3, 2.4**
    """
    # Property validation logic
    pass
```

### Coverage Validation

Ensure all testable requirements have corresponding properties:

```python
# Use the property registry to validate coverage
from tests.property.property_registry import PropertyTestRegistry

registry = PropertyTestRegistry()
coverage = registry.validate_coverage()

# Check that all requirements are covered
required_requirements = ["1.1", "1.2", "2.1", "2.2"]  # From requirements doc
covered_requirements = set()
for prop_name, requirements in coverage.items():
    covered_requirements.update(requirements)

missing_coverage = set(required_requirements) - covered_requirements
assert not missing_coverage, f"Requirements without property coverage: {missing_coverage}"
```

## Performance Considerations

### Execution Time

Property tests can be slower than unit tests due to:
- Multiple example generation
- Complex input creation
- Comprehensive validation

### Optimization Strategies

1. **Profile-Based Execution**: Use appropriate profiles for different contexts
2. **Efficient Strategies**: Design strategies that generate valid data quickly
3. **Assumption Placement**: Place assumptions early to avoid wasted generation
4. **Parallel Execution**: Run property tests in parallel when possible

```python
# Efficient strategy design
@st.composite
def efficient_pose_strategy(draw):
    """Optimized pose landmark generation."""
    # Pre-calculate valid ranges
    valid_x_range = (0, 1920)
    valid_y_range = (0, 1080)
    valid_conf_range = (0.5, 1.0)
    
    # Generate all landmarks at once
    landmarks = []
    for _ in range(33):
        landmarks.append({
            'x': draw(st.floats(*valid_x_range)),
            'y': draw(st.floats(*valid_y_range)),
            'confidence': draw(st.floats(*valid_conf_range))
        })
    
    return landmarks
```

## Best Practices

### Do's

✅ **Write universal properties** that hold for all valid inputs
✅ **Use descriptive property names** that explain what is being tested
✅ **Include requirement traceability** in property documentation
✅ **Design efficient strategies** for realistic test data
✅ **Test round-trip properties** for serialization/parsing
✅ **Validate invariants** after transformations
✅ **Use appropriate profiles** for different execution contexts
✅ **Debug failures systematically** using Hypothesis output
✅ **Combine with unit tests** for comprehensive coverage

### Don'ts

❌ **Don't test implementation details** - focus on observable behavior
❌ **Don't ignore failing properties** - they often reveal real bugs
❌ **Don't make properties too specific** - they should be universal
❌ **Don't generate invalid data** without proper assumptions
❌ **Don't write slow strategies** without considering performance
❌ **Don't skip error condition testing** - invalid inputs matter
❌ **Don't forget to register properties** in the property registry
❌ **Don't write properties without requirement mapping**

## Common Patterns

### Parser Testing

```python
@given(st.text())
def test_config_parser_round_trip(self, config_text):
    """Configuration parsing should be reversible."""
    assume(is_valid_config_format(config_text))
    
    parsed = parse_config(config_text)
    regenerated = generate_config(parsed)
    reparsed = parse_config(regenerated)
    
    assert parsed == reparsed
```

### Mathematical Properties

```python
@given(st.floats(min_value=0.1, max_value=10.0))
def test_cadence_stride_relationship(self, stride_time):
    """Cadence should be inversely related to stride time."""
    cadence = 60.0 / stride_time  # steps per minute
    calculated_stride = 60.0 / cadence
    
    assert abs(stride_time - calculated_stride) < 1e-6
```

### State Machine Properties

```python
@given(st.lists(st.sampled_from(['start', 'pause', 'resume', 'stop'])))
def test_analysis_state_machine(self, operations):
    """Analysis state machine should handle operation sequences correctly."""
    analyzer = GaitAnalyzer()
    
    for operation in operations:
        if operation == 'start' and analyzer.state == 'idle':
            analyzer.start()
        elif operation == 'pause' and analyzer.state == 'running':
            analyzer.pause()
        # ... handle other operations
    
    # Property: state should always be valid
    assert analyzer.state in ['idle', 'running', 'paused', 'stopped']
```

## Conclusion

Property-based testing provides powerful validation of system correctness by testing universal properties across comprehensive input spaces. When combined with traditional unit tests and integration tests, property-based tests form a robust foundation for ensuring software quality in the AlexPose system.

The key to successful property-based testing is identifying the right properties to test and designing efficient strategies for generating realistic test data. Focus on properties that capture the essential behavior of your system and provide clear requirement traceability for comprehensive coverage validation.