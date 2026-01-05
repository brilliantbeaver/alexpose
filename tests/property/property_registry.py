"""
Property test registry system for managing all correctness properties.

This module provides a centralized registry for all property-based tests,
enabling requirement traceability and systematic validation coverage.
"""

from typing import Dict, List, Any, Optional, Callable, Set
from dataclasses import dataclass, field
from enum import Enum
import inspect
from pathlib import Path
import yaml
import json


class PropertyCategory(Enum):
    """Categories of property tests."""
    VIDEO_PROCESSING = "video_processing"
    POSE_ESTIMATION = "pose_estimation"
    GAIT_ANALYSIS = "gait_analysis"
    CLASSIFICATION = "classification"
    INTEGRATION = "integration"
    PERFORMANCE = "performance"


class PropertyPriority(Enum):
    """Priority levels for property tests."""
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


@dataclass
class PropertyTestDefinition:
    """Definition of a property test with metadata."""
    name: str
    description: str
    category: PropertyCategory
    priority: PropertyPriority
    requirements: List[str] = field(default_factory=list)
    test_function: Optional[Callable] = None
    module_path: Optional[str] = None
    enabled: bool = True
    tags: Set[str] = field(default_factory=set)
    expected_examples: int = 100
    timeout_seconds: int = 30
    
    def __post_init__(self):
        """Validate property definition after initialization."""
        if not self.name:
            raise ValueError("Property name cannot be empty")
        if not self.description:
            raise ValueError("Property description cannot be empty")
        if not isinstance(self.category, PropertyCategory):
            raise ValueError(f"Invalid category: {self.category}")
        if not isinstance(self.priority, PropertyPriority):
            raise ValueError(f"Invalid priority: {self.priority}")


class PropertyTestRegistry:
    """Registry for managing all property-based tests."""
    
    def __init__(self):
        self._properties: Dict[str, PropertyTestDefinition] = {}
        self._categories: Dict[PropertyCategory, List[str]] = {
            category: [] for category in PropertyCategory
        }
        self._requirements_map: Dict[str, List[str]] = {}
        self._load_predefined_properties()
    
    def register_property(
        self,
        name: str,
        description: str,
        category: PropertyCategory,
        priority: PropertyPriority = PropertyPriority.MEDIUM,
        requirements: Optional[List[str]] = None,
        test_function: Optional[Callable] = None,
        tags: Optional[Set[str]] = None,
        expected_examples: int = 100,
        timeout_seconds: int = 30
    ) -> PropertyTestDefinition:
        """Register a new property test."""
        if name in self._properties:
            raise ValueError(f"Property '{name}' already registered")
        
        property_def = PropertyTestDefinition(
            name=name,
            description=description,
            category=category,
            priority=priority,
            requirements=requirements or [],
            test_function=test_function,
            tags=tags or set(),
            expected_examples=expected_examples,
            timeout_seconds=timeout_seconds
        )
        
        # Extract module path if test function provided
        if test_function:
            property_def.module_path = f"{test_function.__module__}.{test_function.__name__}"
        
        self._properties[name] = property_def
        self._categories[category].append(name)
        
        # Update requirements mapping
        for req in property_def.requirements:
            if req not in self._requirements_map:
                self._requirements_map[req] = []
            self._requirements_map[req].append(name)
        
        return property_def
    
    def get_property(self, name: str) -> Optional[PropertyTestDefinition]:
        """Get a property test definition by name."""
        return self._properties.get(name)
    
    def get_properties_by_category(self, category: PropertyCategory) -> List[PropertyTestDefinition]:
        """Get all properties in a specific category."""
        property_names = self._categories.get(category, [])
        return [self._properties[name] for name in property_names if name in self._properties]
    
    def get_properties_by_priority(self, priority: PropertyPriority) -> List[PropertyTestDefinition]:
        """Get all properties with a specific priority."""
        return [prop for prop in self._properties.values() if prop.priority == priority]
    
    def get_properties_by_requirement(self, requirement: str) -> List[PropertyTestDefinition]:
        """Get all properties that validate a specific requirement."""
        property_names = self._requirements_map.get(requirement, [])
        return [self._properties[name] for name in property_names if name in self._properties]
    
    def get_all_properties(self) -> List[PropertyTestDefinition]:
        """Get all registered properties."""
        return list(self._properties.values())
    
    def get_enabled_properties(self) -> List[PropertyTestDefinition]:
        """Get all enabled properties."""
        return [prop for prop in self._properties.values() if prop.enabled]
    
    def enable_property(self, name: str) -> bool:
        """Enable a property test."""
        if name in self._properties:
            self._properties[name].enabled = True
            return True
        return False
    
    def disable_property(self, name: str) -> bool:
        """Disable a property test."""
        if name in self._properties:
            self._properties[name].enabled = False
            return True
        return False
    
    def validate_coverage(self) -> Dict[str, Any]:
        """Validate that all requirements are covered by properties."""
        # Load requirements from specification files
        requirements = self._load_requirements()
        
        coverage_report = {
            "total_requirements": len(requirements),
            "covered_requirements": 0,
            "uncovered_requirements": [],
            "coverage_percentage": 0.0,
            "property_count": len(self._properties),
            "enabled_property_count": len(self.get_enabled_properties()),
            "coverage_by_category": {},
            "coverage_by_priority": {}
        }
        
        # Check coverage for each requirement
        for req_id, req_desc in requirements.items():
            if req_id in self._requirements_map:
                coverage_report["covered_requirements"] += 1
            else:
                coverage_report["uncovered_requirements"].append({
                    "id": req_id,
                    "description": req_desc
                })
        
        # Calculate coverage percentage
        if requirements:
            coverage_report["coverage_percentage"] = (
                coverage_report["covered_requirements"] / len(requirements) * 100
            )
        
        # Coverage by category
        for category in PropertyCategory:
            category_props = self.get_properties_by_category(category)
            coverage_report["coverage_by_category"][category.value] = {
                "total": len(category_props),
                "enabled": len([p for p in category_props if p.enabled])
            }
        
        # Coverage by priority
        for priority in PropertyPriority:
            priority_props = self.get_properties_by_priority(priority)
            coverage_report["coverage_by_priority"][priority.value] = {
                "total": len(priority_props),
                "enabled": len([p for p in priority_props if p.enabled])
            }
        
        return coverage_report
    
    def generate_test_execution_plan(
        self,
        categories: Optional[List[PropertyCategory]] = None,
        priorities: Optional[List[PropertyPriority]] = None,
        enabled_only: bool = True
    ) -> List[PropertyTestDefinition]:
        """Generate an execution plan for property tests."""
        properties = self.get_all_properties()
        
        # Filter by enabled status
        if enabled_only:
            properties = [p for p in properties if p.enabled]
        
        # Filter by categories
        if categories:
            properties = [p for p in properties if p.category in categories]
        
        # Filter by priorities
        if priorities:
            properties = [p for p in properties if p.priority in priorities]
        
        # Sort by priority (critical first) and then by category
        priority_order = {
            PropertyPriority.CRITICAL: 0,
            PropertyPriority.HIGH: 1,
            PropertyPriority.MEDIUM: 2,
            PropertyPriority.LOW: 3
        }
        
        properties.sort(key=lambda p: (priority_order[p.priority], p.category.value, p.name))
        
        return properties
    
    def export_registry(self, format: str = "json") -> str:
        """Export registry to JSON or YAML format."""
        registry_data = {
            "properties": {},
            "categories": {},
            "requirements_coverage": self._requirements_map,
            "metadata": {
                "total_properties": len(self._properties),
                "enabled_properties": len(self.get_enabled_properties()),
                "categories": [cat.value for cat in PropertyCategory],
                "priorities": [pri.value for pri in PropertyPriority]
            }
        }
        
        # Export properties
        for name, prop in self._properties.items():
            registry_data["properties"][name] = {
                "description": prop.description,
                "category": prop.category.value,
                "priority": prop.priority.value,
                "requirements": prop.requirements,
                "module_path": prop.module_path,
                "enabled": prop.enabled,
                "tags": list(prop.tags),
                "expected_examples": prop.expected_examples,
                "timeout_seconds": prop.timeout_seconds
            }
        
        # Export categories
        for category, property_names in self._categories.items():
            registry_data["categories"][category.value] = property_names
        
        if format.lower() == "yaml":
            return yaml.dump(registry_data, default_flow_style=False, sort_keys=True)
        else:
            return json.dumps(registry_data, indent=2, sort_keys=True)
    
    def _load_predefined_properties(self):
        """Load predefined property definitions."""
        # Video Processing Properties (1-4)
        self.register_property(
            name="video_format_validation",
            description="For any file extension, the system should correctly identify valid video formats",
            category=PropertyCategory.VIDEO_PROCESSING,
            priority=PropertyPriority.CRITICAL,
            requirements=["REQ-1.1", "REQ-1.2"],
            tags={"format_validation", "input_validation"}
        )
        
        self.register_property(
            name="frame_extraction_consistency",
            description="Frame extraction should produce consistent frame counts for identical videos",
            category=PropertyCategory.VIDEO_PROCESSING,
            priority=PropertyPriority.HIGH,
            requirements=["REQ-1.3", "REQ-1.4"],
            tags={"frame_extraction", "consistency"}
        )
        
        self.register_property(
            name="frame_metadata_preservation",
            description="Frame metadata (timestamp, dimensions) should be accurately preserved during processing",
            category=PropertyCategory.VIDEO_PROCESSING,
            priority=PropertyPriority.HIGH,
            requirements=["REQ-1.5"],
            tags={"metadata", "preservation"}
        )
        
        self.register_property(
            name="video_processing_memory_bounds",
            description="Video processing should stay within defined memory limits",
            category=PropertyCategory.VIDEO_PROCESSING,
            priority=PropertyPriority.MEDIUM,
            requirements=["REQ-1.6"],
            tags={"memory", "performance", "bounds"}
        )
        
        # Pose Estimation Properties (5-8)
        self.register_property(
            name="mediapipe_landmark_count_consistency",
            description="MediaPipe should return exactly 33 pose landmarks for valid input frames",
            category=PropertyCategory.POSE_ESTIMATION,
            priority=PropertyPriority.CRITICAL,
            requirements=["REQ-2.1"],
            tags={"mediapipe", "landmark_count", "consistency"}
        )
        
        self.register_property(
            name="keypoint_confidence_validation",
            description="All keypoint confidence scores should be within valid range [0.0, 1.0]",
            category=PropertyCategory.POSE_ESTIMATION,
            priority=PropertyPriority.HIGH,
            requirements=["REQ-2.2"],
            tags={"confidence", "validation", "bounds"}
        )
        
        self.register_property(
            name="pose_estimation_determinism",
            description="Identical input frames should produce consistent pose estimations",
            category=PropertyCategory.POSE_ESTIMATION,
            priority=PropertyPriority.HIGH,
            requirements=["REQ-2.3"],
            tags={"determinism", "consistency"}
        )
        
        self.register_property(
            name="pose_estimator_factory_consistency",
            description="Pose estimator factory should consistently create valid estimator instances",
            category=PropertyCategory.POSE_ESTIMATION,
            priority=PropertyPriority.MEDIUM,
            requirements=["REQ-2.4"],
            tags={"factory", "consistency", "instantiation"}
        )
        
        # Gait Analysis Properties (9-14)
        self.register_property(
            name="gait_feature_extraction_completeness",
            description="Gait feature extraction should produce all required feature categories",
            category=PropertyCategory.GAIT_ANALYSIS,
            priority=PropertyPriority.CRITICAL,
            requirements=["REQ-3.1"],
            tags={"feature_extraction", "completeness"}
        )
        
        self.register_property(
            name="temporal_feature_validity",
            description="Temporal gait features should be within physiologically realistic ranges",
            category=PropertyCategory.GAIT_ANALYSIS,
            priority=PropertyPriority.HIGH,
            requirements=["REQ-3.2"],
            tags={"temporal_features", "physiological_bounds"}
        )
        
        self.register_property(
            name="spatial_feature_consistency",
            description="Spatial gait features should satisfy geometric constraints and relationships",
            category=PropertyCategory.GAIT_ANALYSIS,
            priority=PropertyPriority.HIGH,
            requirements=["REQ-3.3"],
            tags={"spatial_features", "geometric_constraints"}
        )
        
        self.register_property(
            name="symmetry_analysis_bounds",
            description="Gait symmetry indices should be within valid range [0.0, 1.0]",
            category=PropertyCategory.GAIT_ANALYSIS,
            priority=PropertyPriority.MEDIUM,
            requirements=["REQ-3.4"],
            tags={"symmetry", "bounds", "validation"}
        )
        
        self.register_property(
            name="gait_cycle_detection_accuracy",
            description="Gait cycle detection should identify valid cycle boundaries",
            category=PropertyCategory.GAIT_ANALYSIS,
            priority=PropertyPriority.HIGH,
            requirements=["REQ-3.5"],
            tags={"gait_cycle", "detection", "accuracy"}
        )
        
        self.register_property(
            name="feature_extraction_robustness",
            description="Feature extraction should handle edge cases and invalid inputs gracefully",
            category=PropertyCategory.GAIT_ANALYSIS,
            priority=PropertyPriority.MEDIUM,
            requirements=["REQ-3.6"],
            tags={"robustness", "error_handling", "edge_cases"}
        )
        
        # Classification Properties (15-18)
        self.register_property(
            name="binary_classification_completeness",
            description="Binary classification should always produce normal/abnormal output with confidence",
            category=PropertyCategory.CLASSIFICATION,
            priority=PropertyPriority.CRITICAL,
            requirements=["REQ-4.1"],
            tags={"binary_classification", "completeness"}
        )
        
        self.register_property(
            name="classification_confidence_bounds",
            description="Classification confidence scores should be within valid range [0.0, 1.0]",
            category=PropertyCategory.CLASSIFICATION,
            priority=PropertyPriority.HIGH,
            requirements=["REQ-4.2"],
            tags={"confidence", "bounds", "validation"}
        )
        
        self.register_property(
            name="llm_response_consistency",
            description="LLM responses should be consistent for identical inputs with same parameters",
            category=PropertyCategory.CLASSIFICATION,
            priority=PropertyPriority.MEDIUM,
            requirements=["REQ-4.3"],
            tags={"llm", "consistency", "determinism"}
        )
        
        self.register_property(
            name="classification_explanation_completeness",
            description="Classification results should include complete explanations and supporting evidence",
            category=PropertyCategory.CLASSIFICATION,
            priority=PropertyPriority.MEDIUM,
            requirements=["REQ-4.4"],
            tags={"explanation", "completeness", "evidence"}
        )
    
    def _load_requirements(self) -> Dict[str, str]:
        """Load requirements from specification files."""
        requirements = {}
        
        # Try to load from specification files
        spec_files = [
            ".kiro/specs/testing-enhancement/requirements.md",
            "docs/requirements.md",
            "requirements.yaml"
        ]
        
        for spec_file in spec_files:
            spec_path = Path(spec_file)
            if spec_path.exists():
                try:
                    if spec_file.endswith('.yaml') or spec_file.endswith('.yml'):
                        with open(spec_path, 'r') as f:
                            data = yaml.safe_load(f)
                            if isinstance(data, dict) and 'requirements' in data:
                                requirements.update(data['requirements'])
                    else:
                        # Parse markdown for requirement IDs
                        with open(spec_path, 'r') as f:
                            content = f.read()
                            # Simple regex to find REQ-X.Y patterns
                            import re
                            req_pattern = r'REQ-(\d+\.\d+)'
                            matches = re.findall(req_pattern, content)
                            for match in matches:
                                req_id = f"REQ-{match}"
                                if req_id not in requirements:
                                    requirements[req_id] = f"Requirement {req_id} from {spec_file}"
                except Exception:
                    continue  # Skip files that can't be parsed
        
        # Add default requirements if none found
        if not requirements:
            requirements = {
                "REQ-1.1": "Video format validation",
                "REQ-1.2": "Input validation for video files",
                "REQ-1.3": "Frame extraction consistency",
                "REQ-1.4": "Frame count accuracy",
                "REQ-1.5": "Metadata preservation",
                "REQ-1.6": "Memory usage bounds",
                "REQ-2.1": "MediaPipe landmark consistency",
                "REQ-2.2": "Keypoint confidence validation",
                "REQ-2.3": "Pose estimation determinism",
                "REQ-2.4": "Estimator factory consistency",
                "REQ-3.1": "Gait feature completeness",
                "REQ-3.2": "Temporal feature validity",
                "REQ-3.3": "Spatial feature consistency",
                "REQ-3.4": "Symmetry bounds validation",
                "REQ-3.5": "Gait cycle detection",
                "REQ-3.6": "Feature extraction robustness",
                "REQ-4.1": "Binary classification completeness",
                "REQ-4.2": "Classification confidence bounds",
                "REQ-4.3": "LLM response consistency",
                "REQ-4.4": "Classification explanation completeness"
            }
        
        return requirements


# Global registry instance
property_registry = PropertyTestRegistry()


def register_property_test(
    name: str,
    description: str,
    category: PropertyCategory,
    priority: PropertyPriority = PropertyPriority.MEDIUM,
    requirements: Optional[List[str]] = None,
    tags: Optional[Set[str]] = None,
    expected_examples: int = 100,
    timeout_seconds: int = 30
):
    """Decorator for registering property tests."""
    def decorator(test_function: Callable) -> Callable:
        property_registry.register_property(
            name=name,
            description=description,
            category=category,
            priority=priority,
            requirements=requirements,
            test_function=test_function,
            tags=tags,
            expected_examples=expected_examples,
            timeout_seconds=timeout_seconds
        )
        return test_function
    return decorator


def get_registry() -> PropertyTestRegistry:
    """Get the global property test registry."""
    return property_registry