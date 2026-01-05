"""
Base classes and interfaces for property-based testing.

This module provides the foundation for implementing property-based tests
with consistent structure, error handling, and reporting.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Callable, Union
from dataclasses import dataclass
import time
import traceback
from enum import Enum
import pytest
from hypothesis import given, strategies as st, settings, Verbosity
from hypothesis.errors import Unsatisfiable

from .property_registry import PropertyCategory, PropertyPriority


class PropertyTestResult(Enum):
    """Result status of a property test."""
    PASSED = "passed"
    FAILED = "failed"
    SKIPPED = "skipped"
    ERROR = "error"
    TIMEOUT = "timeout"


@dataclass
class PropertyTestExecution:
    """Execution details of a property test."""
    property_name: str
    result: PropertyTestResult
    execution_time: float
    examples_tested: int
    error_message: Optional[str] = None
    failure_details: Optional[Dict[str, Any]] = None
    counterexample: Optional[Any] = None
    traceback: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert execution details to dictionary."""
        return {
            "property_name": self.property_name,
            "result": self.result.value,
            "execution_time": self.execution_time,
            "examples_tested": self.examples_tested,
            "error_message": self.error_message,
            "failure_details": self.failure_details,
            "counterexample": str(self.counterexample) if self.counterexample else None,
            "traceback": self.traceback
        }


class PropertyTestInterface(ABC):
    """Abstract interface for property-based tests."""
    
    def __init__(
        self,
        name: str,
        description: str,
        category: PropertyCategory,
        priority: PropertyPriority = PropertyPriority.MEDIUM,
        requirements: Optional[List[str]] = None,
        expected_examples: int = 100,
        timeout_seconds: int = 30
    ):
        self.name = name
        self.description = description
        self.category = category
        self.priority = priority
        self.requirements = requirements or []
        self.expected_examples = expected_examples
        self.timeout_seconds = timeout_seconds
    
    @abstractmethod
    def generate_test_data(self) -> st.SearchStrategy:
        """Generate test data strategy for this property."""
        pass
    
    @abstractmethod
    def test_property(self, data: Any) -> bool:
        """Test the property with given data. Return True if property holds."""
        pass
    
    def setup(self) -> None:
        """Setup before running the property test."""
        pass
    
    def teardown(self) -> None:
        """Cleanup after running the property test."""
        pass
    
    def validate_preconditions(self, data: Any) -> bool:
        """Validate that data meets preconditions for the test."""
        return True
    
    def handle_failure(self, data: Any, exception: Exception) -> Dict[str, Any]:
        """Handle test failure and provide debugging information."""
        return {
            "input_data": str(data),
            "exception_type": type(exception).__name__,
            "exception_message": str(exception),
            "property_name": self.name,
            "category": self.category.value
        }
    
    def execute(self) -> PropertyTestExecution:
        """Execute the property test and return execution details."""
        start_time = time.time()
        examples_tested = 0
        
        try:
            self.setup()
            
            # Configure Hypothesis settings
            test_settings = settings(
                max_examples=self.expected_examples,
                deadline=self.timeout_seconds * 1000,  # Convert to milliseconds
                verbosity=Verbosity.normal,
                suppress_health_check=[],
                report_multiple_bugs=False
            )
            
            # Create the property test function
            @given(self.generate_test_data())
            @test_settings
            def property_test(data):
                nonlocal examples_tested
                examples_tested += 1
                
                # Validate preconditions
                if not self.validate_preconditions(data):
                    pytest.skip(f"Preconditions not met for data: {data}")
                
                # Test the property
                try:
                    result = self.test_property(data)
                    if not result:
                        failure_details = self.handle_failure(data, Exception("Property assertion failed"))
                        raise AssertionError(f"Property failed: {failure_details}")
                except Exception as e:
                    failure_details = self.handle_failure(data, e)
                    raise AssertionError(f"Property test error: {failure_details}") from e
            
            # Execute the test
            property_test()
            
            execution_time = time.time() - start_time
            
            return PropertyTestExecution(
                property_name=self.name,
                result=PropertyTestResult.PASSED,
                execution_time=execution_time,
                examples_tested=examples_tested
            )
            
        except Unsatisfiable as e:
            execution_time = time.time() - start_time
            return PropertyTestExecution(
                property_name=self.name,
                result=PropertyTestResult.SKIPPED,
                execution_time=execution_time,
                examples_tested=examples_tested,
                error_message=f"Could not generate satisfiable test data: {str(e)}"
            )
            
        except AssertionError as e:
            execution_time = time.time() - start_time
            return PropertyTestExecution(
                property_name=self.name,
                result=PropertyTestResult.FAILED,
                execution_time=execution_time,
                examples_tested=examples_tested,
                error_message=str(e),
                traceback=traceback.format_exc()
            )
            
        except Exception as e:
            execution_time = time.time() - start_time
            return PropertyTestExecution(
                property_name=self.name,
                result=PropertyTestResult.ERROR,
                execution_time=execution_time,
                examples_tested=examples_tested,
                error_message=str(e),
                traceback=traceback.format_exc()
            )
            
        finally:
            try:
                self.teardown()
            except Exception:
                pass  # Don't let teardown errors mask test results


class VideoProcessingProperty(PropertyTestInterface):
    """Base class for video processing property tests."""
    
    def __init__(self, name: str, description: str, **kwargs):
        super().__init__(
            name=name,
            description=description,
            category=PropertyCategory.VIDEO_PROCESSING,
            **kwargs
        )


class PoseEstimationProperty(PropertyTestInterface):
    """Base class for pose estimation property tests."""
    
    def __init__(self, name: str, description: str, **kwargs):
        super().__init__(
            name=name,
            description=description,
            category=PropertyCategory.POSE_ESTIMATION,
            **kwargs
        )


class GaitAnalysisProperty(PropertyTestInterface):
    """Base class for gait analysis property tests."""
    
    def __init__(self, name: str, description: str, **kwargs):
        super().__init__(
            name=name,
            description=description,
            category=PropertyCategory.GAIT_ANALYSIS,
            **kwargs
        )


class ClassificationProperty(PropertyTestInterface):
    """Base class for classification property tests."""
    
    def __init__(self, name: str, description: str, **kwargs):
        super().__init__(
            name=name,
            description=description,
            category=PropertyCategory.CLASSIFICATION,
            **kwargs
        )


class PropertyTestRunner:
    """Runner for executing property tests with comprehensive reporting."""
    
    def __init__(self):
        self.executions: List[PropertyTestExecution] = []
    
    def run_property(self, property_test: PropertyTestInterface) -> PropertyTestExecution:
        """Run a single property test."""
        execution = property_test.execute()
        self.executions.append(execution)
        return execution
    
    def run_properties(self, properties: List[PropertyTestInterface]) -> List[PropertyTestExecution]:
        """Run multiple property tests."""
        executions = []
        for prop in properties:
            execution = self.run_property(prop)
            executions.append(execution)
        return executions
    
    def generate_report(self) -> Dict[str, Any]:
        """Generate comprehensive test execution report."""
        if not self.executions:
            return {"error": "No property tests executed"}
        
        total_tests = len(self.executions)
        passed = len([e for e in self.executions if e.result == PropertyTestResult.PASSED])
        failed = len([e for e in self.executions if e.result == PropertyTestResult.FAILED])
        errors = len([e for e in self.executions if e.result == PropertyTestResult.ERROR])
        skipped = len([e for e in self.executions if e.result == PropertyTestResult.SKIPPED])
        
        total_time = sum(e.execution_time for e in self.executions)
        total_examples = sum(e.examples_tested for e in self.executions)
        
        report = {
            "summary": {
                "total_tests": total_tests,
                "passed": passed,
                "failed": failed,
                "errors": errors,
                "skipped": skipped,
                "success_rate": (passed / total_tests * 100) if total_tests > 0 else 0,
                "total_execution_time": total_time,
                "total_examples_tested": total_examples,
                "average_time_per_test": total_time / total_tests if total_tests > 0 else 0
            },
            "executions": [e.to_dict() for e in self.executions],
            "failures": [
                e.to_dict() for e in self.executions 
                if e.result in [PropertyTestResult.FAILED, PropertyTestResult.ERROR]
            ],
            "performance_metrics": {
                "fastest_test": min(self.executions, key=lambda e: e.execution_time).property_name,
                "slowest_test": max(self.executions, key=lambda e: e.execution_time).property_name,
                "most_examples": max(self.executions, key=lambda e: e.examples_tested).property_name,
                "least_examples": min(self.executions, key=lambda e: e.examples_tested).property_name
            }
        }
        
        return report
    
    def clear_results(self):
        """Clear all execution results."""
        self.executions.clear()


def create_property_test(
    name: str,
    description: str,
    category: PropertyCategory,
    data_strategy: st.SearchStrategy,
    test_function: Callable[[Any], bool],
    priority: PropertyPriority = PropertyPriority.MEDIUM,
    requirements: Optional[List[str]] = None,
    expected_examples: int = 100,
    timeout_seconds: int = 30
) -> PropertyTestInterface:
    """Factory function to create property tests from functions."""
    
    class FunctionalPropertyTest(PropertyTestInterface):
        def generate_test_data(self) -> st.SearchStrategy:
            return data_strategy
        
        def test_property(self, data: Any) -> bool:
            return test_function(data)
    
    return FunctionalPropertyTest(
        name=name,
        description=description,
        category=category,
        priority=priority,
        requirements=requirements,
        expected_examples=expected_examples,
        timeout_seconds=timeout_seconds
    )


def property_test_decorator(
    name: str,
    description: str,
    category: PropertyCategory,
    data_strategy: st.SearchStrategy,
    priority: PropertyPriority = PropertyPriority.MEDIUM,
    requirements: Optional[List[str]] = None,
    expected_examples: int = 100,
    timeout_seconds: int = 30
):
    """Decorator for creating property tests from functions."""
    def decorator(test_function: Callable[[Any], bool]) -> PropertyTestInterface:
        return create_property_test(
            name=name,
            description=description,
            category=category,
            data_strategy=data_strategy,
            test_function=test_function,
            priority=priority,
            requirements=requirements,
            expected_examples=expected_examples,
            timeout_seconds=timeout_seconds
        )
    return decorator


# Utility functions for common property test patterns

def assert_within_bounds(value: Union[int, float], min_val: Union[int, float], max_val: Union[int, float], name: str = "value"):
    """Assert that a value is within specified bounds."""
    if not (min_val <= value <= max_val):
        raise AssertionError(f"{name} {value} not within bounds [{min_val}, {max_val}]")


def assert_list_length(lst: List[Any], expected_length: int, name: str = "list"):
    """Assert that a list has the expected length."""
    if len(lst) != expected_length:
        raise AssertionError(f"{name} length {len(lst)} != expected {expected_length}")


def assert_dict_has_keys(d: Dict[str, Any], required_keys: List[str], name: str = "dictionary"):
    """Assert that a dictionary has all required keys."""
    missing_keys = [key for key in required_keys if key not in d]
    if missing_keys:
        raise AssertionError(f"{name} missing required keys: {missing_keys}")


def assert_all_values_in_range(values: List[Union[int, float]], min_val: Union[int, float], max_val: Union[int, float], name: str = "values"):
    """Assert that all values in a list are within specified range."""
    out_of_range = [v for v in values if not (min_val <= v <= max_val)]
    if out_of_range:
        raise AssertionError(f"{name} contains out-of-range values: {out_of_range}")