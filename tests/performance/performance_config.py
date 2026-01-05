"""Configuration system for performance regression detection."""

import os
import json
from typing import Dict, Any, Optional
from pathlib import Path
from dataclasses import dataclass, asdict
import logging

logger = logging.getLogger(__name__)

@dataclass
class ToleranceConfig:
    """Configuration for performance regression tolerance."""
    
    # Default tolerance percentages
    execution_time_tolerance: float = 10.0
    memory_usage_tolerance: float = 15.0
    throughput_tolerance: float = 10.0
    
    # Test-specific tolerances
    test_specific_tolerances: Dict[str, Dict[str, float]] = None
    
    # Environment-specific tolerances
    ci_multiplier: float = 1.5  # CI environments are less stable
    local_multiplier: float = 1.0
    
    def __post_init__(self):
        if self.test_specific_tolerances is None:
            self.test_specific_tolerances = {}
    
    def get_tolerance(self, test_name: str, metric_type: str = "execution_time") -> float:
        """Get tolerance for a specific test and metric type."""
        
        # Ensure metric_type has _tolerance suffix
        if not metric_type.endswith('_tolerance'):
            metric_key = f"{metric_type}_tolerance"
        else:
            metric_key = metric_type
        
        # Check for test-specific tolerance first
        if test_name in self.test_specific_tolerances:
            test_config = self.test_specific_tolerances[test_name]
            if metric_key in test_config:
                base_tolerance = test_config[metric_key]
            else:
                base_tolerance = getattr(self, metric_key, 10.0)
        else:
            base_tolerance = getattr(self, metric_key, 10.0)
        
        # Apply environment multiplier
        multiplier = self.ci_multiplier if self._is_ci_environment() else self.local_multiplier
        return base_tolerance * multiplier
    
    def _is_ci_environment(self) -> bool:
        """Check if running in CI environment."""
        ci_indicators = ['CI', 'GITHUB_ACTIONS', 'JENKINS_URL', 'TRAVIS', 'CIRCLECI']
        return any(os.getenv(indicator) for indicator in ci_indicators)
    
    def set_test_tolerance(self, test_name: str, **tolerances):
        """Set specific tolerances for a test."""
        if test_name not in self.test_specific_tolerances:
            self.test_specific_tolerances[test_name] = {}
        
        for metric_type, tolerance in tolerances.items():
            if not metric_type.endswith('_tolerance'):
                metric_type = f"{metric_type}_tolerance"
            self.test_specific_tolerances[test_name][metric_type] = tolerance

class PerformanceConfig:
    """Central configuration manager for performance testing."""
    
    def __init__(self, config_file: Optional[Path] = None):
        self.config_file = config_file or Path("tests/performance/performance_config.json")
        self.tolerance_config = ToleranceConfig()
        self._load_config()
    
    def _load_config(self):
        """Load configuration from file."""
        if self.config_file.exists():
            try:
                with open(self.config_file, 'r') as f:
                    data = json.load(f)
                
                # Load tolerance configuration
                if 'tolerance' in data:
                    tolerance_data = data['tolerance']
                    self.tolerance_config = ToleranceConfig(
                        execution_time_tolerance=tolerance_data.get('execution_time_tolerance', 10.0),
                        memory_usage_tolerance=tolerance_data.get('memory_usage_tolerance', 15.0),
                        throughput_tolerance=tolerance_data.get('throughput_tolerance', 10.0),
                        test_specific_tolerances=tolerance_data.get('test_specific_tolerances', {}),
                        ci_multiplier=tolerance_data.get('ci_multiplier', 1.5),
                        local_multiplier=tolerance_data.get('local_multiplier', 1.0)
                    )
                
                logger.info(f"Loaded performance configuration from {self.config_file}")
                
            except Exception as e:
                logger.warning(f"Failed to load performance config: {e}, using defaults")
                self._create_default_config()
        else:
            self._create_default_config()
    
    def _create_default_config(self):
        """Create default configuration with common test tolerances."""
        
        # Set up default test-specific tolerances based on existing usage
        default_tolerances = {
            # Video processing tests - higher tolerance due to complexity
            "video_loading_30s": {
                "execution_time_tolerance": 15.0,
                "memory_usage_tolerance": 20.0
            },
            "frame_extraction_30s": {
                "execution_time_tolerance": 25.0,  # Frame extraction can be variable
                "memory_usage_tolerance": 30.0
            },
            "complete_pipeline_30s": {
                "execution_time_tolerance": 30.0,  # End-to-end pipeline has more variance
                "memory_usage_tolerance": 25.0
            },
            "metadata_extraction": {
                "execution_time_tolerance": 20.0,
                "memory_usage_tolerance": 15.0
            },
            
            # API response time tests - stricter tolerance
            "api_response_time_health": {
                "execution_time_tolerance": 10.0,
                "memory_usage_tolerance": 10.0
            },
            "api_response_time_status": {
                "execution_time_tolerance": 10.0,
                "memory_usage_tolerance": 10.0
            },
            "health_endpoint": {
                "execution_time_tolerance": 15.0,
                "memory_usage_tolerance": 15.0
            },
            
            # Concurrent analysis tests - moderate tolerance
            "concurrent_analysis_5": {
                "execution_time_tolerance": 20.0,
                "memory_usage_tolerance": 25.0,
                "throughput_tolerance": 15.0
            },
            
            # Memory usage tests - stricter memory tolerance
            "memory_usage_standard": {
                "execution_time_tolerance": 15.0,
                "memory_usage_tolerance": 10.0
            }
        }
        
        self.tolerance_config.test_specific_tolerances = default_tolerances
        self.save_config()
    
    def save_config(self):
        """Save current configuration to file."""
        try:
            self.config_file.parent.mkdir(parents=True, exist_ok=True)
            
            config_data = {
                'tolerance': {
                    'execution_time_tolerance': self.tolerance_config.execution_time_tolerance,
                    'memory_usage_tolerance': self.tolerance_config.memory_usage_tolerance,
                    'throughput_tolerance': self.tolerance_config.throughput_tolerance,
                    'test_specific_tolerances': self.tolerance_config.test_specific_tolerances,
                    'ci_multiplier': self.tolerance_config.ci_multiplier,
                    'local_multiplier': self.tolerance_config.local_multiplier
                }
            }
            
            with open(self.config_file, 'w') as f:
                json.dump(config_data, f, indent=2)
            
            logger.info(f"Saved performance configuration to {self.config_file}")
            
        except Exception as e:
            logger.error(f"Failed to save performance config: {e}")
    
    def get_tolerance(self, test_name: str, metric_type: str = "execution_time") -> float:
        """Get tolerance for a specific test and metric type."""
        return self.tolerance_config.get_tolerance(test_name, metric_type)
    
    def set_test_tolerance(self, test_name: str, **tolerances):
        """Set specific tolerances for a test and save configuration."""
        self.tolerance_config.set_test_tolerance(test_name, **tolerances)
        self.save_config()
    
    def get_all_tolerances(self) -> Dict[str, Any]:
        """Get all tolerance configurations."""
        return {
            'default_tolerances': {
                'execution_time': self.tolerance_config.execution_time_tolerance,
                'memory_usage': self.tolerance_config.memory_usage_tolerance,
                'throughput': self.tolerance_config.throughput_tolerance
            },
            'environment_multipliers': {
                'ci': self.tolerance_config.ci_multiplier,
                'local': self.tolerance_config.local_multiplier
            },
            'test_specific_tolerances': self.tolerance_config.test_specific_tolerances,
            'current_environment': 'ci' if self.tolerance_config._is_ci_environment() else 'local'
        }

# Global configuration instance
_performance_config = None

def get_performance_config() -> PerformanceConfig:
    """Get the global performance configuration instance."""
    global _performance_config
    if _performance_config is None:
        _performance_config = PerformanceConfig()
    return _performance_config

def configure_test_tolerance(test_name: str, **tolerances):
    """Convenience function to configure test tolerance."""
    config = get_performance_config()
    config.set_test_tolerance(test_name, **tolerances)

def get_test_tolerance(test_name: str, metric_type: str = "execution_time") -> float:
    """Convenience function to get test tolerance."""
    config = get_performance_config()
    return config.get_tolerance(test_name, metric_type)