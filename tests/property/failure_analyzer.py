"""
Property test failure analysis and debugging support.

This module provides comprehensive failure analysis for property-based tests,
helping developers understand why tests fail and how to fix them.
"""

import json
import traceback
from typing import Any, Dict, List, Optional, Tuple, Union
from dataclasses import dataclass, asdict
from pathlib import Path
import re
from datetime import datetime
from enum import Enum

try:
    from hypothesis.errors import Flaky, Unsatisfiable, InvalidArgument
    HYPOTHESIS_AVAILABLE = True
except ImportError:
    HYPOTHESIS_AVAILABLE = False


class FailureCategory(Enum):
    """Categories of property test failures."""
    ASSERTION_ERROR = "assertion_error"
    INVALID_INPUT = "invalid_input"
    BOUNDARY_VIOLATION = "boundary_violation"
    TYPE_ERROR = "type_error"
    TIMEOUT = "timeout"
    RESOURCE_ERROR = "resource_error"
    CONFIGURATION_ERROR = "configuration_error"
    HYPOTHESIS_ERROR = "hypothesis_error"
    UNKNOWN = "unknown"


@dataclass
class FailurePattern:
    """Pattern for identifying common failure types."""
    category: FailureCategory
    pattern: str
    description: str
    suggested_fix: str
    confidence: float = 1.0


@dataclass
class FailureAnalysis:
    """Analysis of a property test failure."""
    property_name: str
    failure_category: FailureCategory
    error_message: str
    counterexample: Optional[str]
    root_cause: str
    suggested_fixes: List[str]
    related_patterns: List[FailurePattern]
    confidence: float
    analysis_timestamp: str
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert analysis to dictionary."""
        return asdict(self)


class PropertyFailureAnalyzer:
    """Analyzer for property test failures with pattern recognition."""
    
    def __init__(self):
        self.failure_patterns = self._initialize_patterns()
        self.failure_history: List[FailureAnalysis] = []
    
    def analyze_failure(
        self,
        property_name: str,
        error_message: str,
        exception_type: str,
        counterexample: Optional[Any] = None,
        traceback_str: Optional[str] = None
    ) -> FailureAnalysis:
        """Analyze a property test failure and provide debugging information."""
        
        # Categorize the failure
        category = self._categorize_failure(error_message, exception_type, traceback_str)
        
        # Find matching patterns
        matching_patterns = self._find_matching_patterns(error_message, exception_type, category)
        
        # Determine root cause
        root_cause = self._determine_root_cause(
            error_message, exception_type, counterexample, matching_patterns
        )
        
        # Generate suggested fixes
        suggested_fixes = self._generate_suggested_fixes(
            category, matching_patterns, counterexample, property_name
        )
        
        # Calculate confidence
        confidence = self._calculate_confidence(matching_patterns, category)
        
        analysis = FailureAnalysis(
            property_name=property_name,
            failure_category=category,
            error_message=error_message,
            counterexample=str(counterexample) if counterexample else None,
            root_cause=root_cause,
            suggested_fixes=suggested_fixes,
            related_patterns=matching_patterns,
            confidence=confidence,
            analysis_timestamp=datetime.now().isoformat()
        )
        
        self.failure_history.append(analysis)
        return analysis
    
    def _categorize_failure(
        self,
        error_message: str,
        exception_type: str,
        traceback_str: Optional[str] = None
    ) -> FailureCategory:
        """Categorize the type of failure."""
        error_lower = error_message.lower()
        
        # Hypothesis-specific errors
        if HYPOTHESIS_AVAILABLE:
            if exception_type in ['Flaky', 'Unsatisfiable', 'InvalidArgument']:
                return FailureCategory.HYPOTHESIS_ERROR
        
        # Assertion errors
        if exception_type == 'AssertionError':
            if any(keyword in error_lower for keyword in ['bounds', 'range', 'limit']):
                return FailureCategory.BOUNDARY_VIOLATION
            return FailureCategory.ASSERTION_ERROR
        
        # Type errors
        if exception_type == 'TypeError':
            return FailureCategory.TYPE_ERROR
        
        # Timeout errors
        if exception_type in ['TimeoutError', 'DeadlineExceeded'] or 'timeout' in error_lower:
            return FailureCategory.TIMEOUT
        
        # Resource errors
        if exception_type in ['MemoryError', 'OSError', 'IOError'] or any(
            keyword in error_lower for keyword in ['memory', 'disk', 'file not found']
        ):
            return FailureCategory.RESOURCE_ERROR
        
        # Configuration errors
        if any(keyword in error_lower for keyword in ['config', 'setting', 'parameter']):
            return FailureCategory.CONFIGURATION_ERROR
        
        # Invalid input
        if any(keyword in error_lower for keyword in ['invalid', 'malformed', 'corrupt']):
            return FailureCategory.INVALID_INPUT
        
        return FailureCategory.UNKNOWN
    
    def _find_matching_patterns(
        self,
        error_message: str,
        exception_type: str,
        category: FailureCategory
    ) -> List[FailurePattern]:
        """Find patterns that match the failure."""
        matching_patterns = []
        
        for pattern in self.failure_patterns:
            if pattern.category == category:
                if re.search(pattern.pattern, error_message, re.IGNORECASE):
                    matching_patterns.append(pattern)
        
        return matching_patterns
    
    def _determine_root_cause(
        self,
        error_message: str,
        exception_type: str,
        counterexample: Optional[Any],
        patterns: List[FailurePattern]
    ) -> str:
        """Determine the root cause of the failure."""
        if patterns:
            # Use the highest confidence pattern
            best_pattern = max(patterns, key=lambda p: p.confidence)
            return best_pattern.description
        
        # Fallback to generic analysis
        if exception_type == 'AssertionError':
            if counterexample:
                return f"Property assertion failed for input: {counterexample}"
            return "Property assertion failed - the expected condition was not met"
        
        if exception_type == 'TypeError':
            return "Type mismatch - incorrect data type provided to function"
        
        if 'timeout' in error_message.lower():
            return "Test execution exceeded time limit"
        
        return f"Unexpected error of type {exception_type}: {error_message}"
    
    def _generate_suggested_fixes(
        self,
        category: FailureCategory,
        patterns: List[FailurePattern],
        counterexample: Optional[Any],
        property_name: str
    ) -> List[str]:
        """Generate suggested fixes for the failure."""
        fixes = []
        
        # Use pattern-specific fixes
        for pattern in patterns:
            fixes.append(pattern.suggested_fix)
        
        # Add category-specific generic fixes
        if category == FailureCategory.BOUNDARY_VIOLATION:
            fixes.extend([
                "Check if input values are within expected ranges",
                "Verify boundary conditions in your property test",
                "Consider adjusting the data generation strategy to avoid edge cases"
            ])
        
        elif category == FailureCategory.TYPE_ERROR:
            fixes.extend([
                "Verify that input data types match function expectations",
                "Add type validation in your property test",
                "Check if data generation strategy produces correct types"
            ])
        
        elif category == FailureCategory.TIMEOUT:
            fixes.extend([
                "Reduce the number of test examples or increase timeout",
                "Optimize the property test implementation for better performance",
                "Check for infinite loops or blocking operations"
            ])
        
        elif category == FailureCategory.HYPOTHESIS_ERROR:
            fixes.extend([
                "Review your data generation strategy for satisfiability",
                "Consider using assume() to filter invalid inputs",
                "Adjust Hypothesis settings (max_examples, deadline)"
            ])
        
        elif category == FailureCategory.INVALID_INPUT:
            fixes.extend([
                "Add input validation to handle edge cases",
                "Use assume() to filter out invalid inputs",
                "Improve data generation strategy to avoid invalid cases"
            ])
        
        # Add counterexample-specific fixes
        if counterexample:
            fixes.append(f"Debug with the failing input: {counterexample}")
            fixes.append("Create a unit test with this specific input for easier debugging")
        
        # Remove duplicates while preserving order
        seen = set()
        unique_fixes = []
        for fix in fixes:
            if fix not in seen:
                seen.add(fix)
                unique_fixes.append(fix)
        
        return unique_fixes
    
    def _calculate_confidence(
        self,
        patterns: List[FailurePattern],
        category: FailureCategory
    ) -> float:
        """Calculate confidence in the analysis."""
        if not patterns:
            return 0.5  # Low confidence without matching patterns
        
        # Average confidence of matching patterns
        avg_pattern_confidence = sum(p.confidence for p in patterns) / len(patterns)
        
        # Boost confidence for well-known categories
        category_boost = {
            FailureCategory.ASSERTION_ERROR: 0.1,
            FailureCategory.TYPE_ERROR: 0.1,
            FailureCategory.BOUNDARY_VIOLATION: 0.15,
            FailureCategory.HYPOTHESIS_ERROR: 0.2,
        }.get(category, 0.0)
        
        return min(1.0, avg_pattern_confidence + category_boost)
    
    def _initialize_patterns(self) -> List[FailurePattern]:
        """Initialize common failure patterns."""
        return [
            # Boundary violation patterns
            FailurePattern(
                category=FailureCategory.BOUNDARY_VIOLATION,
                pattern=r"confidence.*not.*\[0.*1\]",
                description="Confidence score outside valid range [0.0, 1.0]",
                suggested_fix="Ensure confidence scores are clamped to [0.0, 1.0] range",
                confidence=0.9
            ),
            FailurePattern(
                category=FailureCategory.BOUNDARY_VIOLATION,
                pattern=r"stride_time.*not.*\[0\.8.*2\.0\]",
                description="Stride time outside physiological range",
                suggested_fix="Validate stride time is within realistic human gait range [0.8, 2.0] seconds",
                confidence=0.85
            ),
            FailurePattern(
                category=FailureCategory.BOUNDARY_VIOLATION,
                pattern=r"cadence.*not.*\[60.*180\]",
                description="Cadence outside physiological range",
                suggested_fix="Ensure cadence is within realistic range [60, 180] steps/minute",
                confidence=0.85
            ),
            
            # Type error patterns
            FailurePattern(
                category=FailureCategory.TYPE_ERROR,
                pattern=r"'NoneType'.*has no attribute",
                description="Attempting to access attribute on None value",
                suggested_fix="Add null checks before accessing object attributes",
                confidence=0.9
            ),
            FailurePattern(
                category=FailureCategory.TYPE_ERROR,
                pattern=r"unsupported operand type.*for.*'float'.*'str'",
                description="Attempting arithmetic operation between incompatible types",
                suggested_fix="Ensure numeric operations use compatible data types",
                confidence=0.85
            ),
            
            # Assertion error patterns
            FailurePattern(
                category=FailureCategory.ASSERTION_ERROR,
                pattern=r"length.*!=.*expected.*33",
                description="MediaPipe landmark count mismatch",
                suggested_fix="Verify MediaPipe returns exactly 33 landmarks for valid poses",
                confidence=0.9
            ),
            FailurePattern(
                category=FailureCategory.ASSERTION_ERROR,
                pattern=r"missing required keys",
                description="Required dictionary keys are missing",
                suggested_fix="Ensure all required keys are present in data structures",
                confidence=0.85
            ),
            
            # Hypothesis error patterns
            FailurePattern(
                category=FailureCategory.HYPOTHESIS_ERROR,
                pattern=r"Unable to satisfy assumptions",
                description="Hypothesis cannot generate data satisfying assume() conditions",
                suggested_fix="Relax assume() conditions or improve data generation strategy",
                confidence=0.9
            ),
            FailurePattern(
                category=FailureCategory.HYPOTHESIS_ERROR,
                pattern=r"Flaky test",
                description="Test results are inconsistent across runs",
                suggested_fix="Check for non-deterministic behavior or race conditions",
                confidence=0.8
            ),
            
            # Resource error patterns
            FailurePattern(
                category=FailureCategory.RESOURCE_ERROR,
                pattern=r"out of memory",
                description="Insufficient memory for test execution",
                suggested_fix="Reduce test data size or optimize memory usage",
                confidence=0.9
            ),
            FailurePattern(
                category=FailureCategory.RESOURCE_ERROR,
                pattern=r"file not found",
                description="Required test file is missing",
                suggested_fix="Ensure test data files exist or create them in test setup",
                confidence=0.85
            ),
            
            # Configuration error patterns
            FailurePattern(
                category=FailureCategory.CONFIGURATION_ERROR,
                pattern=r"API key.*not.*found",
                description="Missing API key configuration",
                suggested_fix="Set required API keys in test environment or use mocks",
                confidence=0.9
            ),
            FailurePattern(
                category=FailureCategory.CONFIGURATION_ERROR,
                pattern=r"invalid.*configuration",
                description="Configuration validation failed",
                suggested_fix="Check configuration file format and required fields",
                confidence=0.8
            )
        ]
    
    def get_failure_statistics(self) -> Dict[str, Any]:
        """Get statistics about failure patterns."""
        if not self.failure_history:
            return {"message": "No failures recorded"}
        
        total_failures = len(self.failure_history)
        category_counts = {}
        
        for analysis in self.failure_history:
            category = analysis.failure_category.value
            category_counts[category] = category_counts.get(category, 0) + 1
        
        most_common_category = max(category_counts.items(), key=lambda x: x[1])
        
        return {
            "total_failures": total_failures,
            "category_distribution": category_counts,
            "most_common_category": most_common_category[0],
            "most_common_count": most_common_category[1],
            "average_confidence": sum(a.confidence for a in self.failure_history) / total_failures
        }
    
    def export_failure_report(self, output_path: Optional[Path] = None) -> str:
        """Export comprehensive failure analysis report."""
        if not self.failure_history:
            return json.dumps({"message": "No failures to report"}, indent=2)
        
        report = {
            "summary": self.get_failure_statistics(),
            "analyses": [analysis.to_dict() for analysis in self.failure_history],
            "patterns": [
                {
                    "category": pattern.category.value,
                    "pattern": pattern.pattern,
                    "description": pattern.description,
                    "suggested_fix": pattern.suggested_fix,
                    "confidence": pattern.confidence
                }
                for pattern in self.failure_patterns
            ],
            "generated_at": datetime.now().isoformat()
        }
        
        report_json = json.dumps(report, indent=2)
        
        if output_path:
            output_path.write_text(report_json)
        
        return report_json
    
    def clear_history(self):
        """Clear failure analysis history."""
        self.failure_history.clear()


# Global analyzer instance
failure_analyzer = PropertyFailureAnalyzer()


def analyze_property_failure(
    property_name: str,
    error_message: str,
    exception_type: str,
    counterexample: Optional[Any] = None,
    traceback_str: Optional[str] = None
) -> FailureAnalysis:
    """Analyze a property test failure using the global analyzer."""
    return failure_analyzer.analyze_failure(
        property_name=property_name,
        error_message=error_message,
        exception_type=exception_type,
        counterexample=counterexample,
        traceback_str=traceback_str
    )


def get_failure_analyzer() -> PropertyFailureAnalyzer:
    """Get the global failure analyzer instance."""
    return failure_analyzer