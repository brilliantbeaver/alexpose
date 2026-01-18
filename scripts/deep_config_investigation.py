"""Deep investigation of configuration loading issue."""

import sys
from pathlib import Path
import yaml

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

print("=" * 80)
print("DEEP CONFIGURATION INVESTIGATION")
print("=" * 80)

# Step 1: Check raw YAML files
print("\n1. RAW YAML FILES")
print("-" * 80)

main_config_path = Path("config/alexpose.yaml")
dev_config_path = Path("config/development.yaml")

with open(main_config_path) as f:
    main_config = yaml.safe_load(f)

with open(dev_config_path) as f:
    dev_config = yaml.safe_load(f)

print(f"Main config estimators: {list(main_config['pose_estimation']['estimators'].keys())}")
print(f"Dev config estimators: {list(dev_config['pose_estimation']['estimators'].keys())}")

# Step 2: Simulate deep merge
print("\n2. SIMULATING DEEP MERGE")
print("-" * 80)

def deep_merge(base, override):
    """Simulate the deep merge logic."""
    for key, value in override.items():
        if key in base and isinstance(base[key], dict) and isinstance(value, dict):
            deep_merge(base[key], value)
        else:
            base[key] = value

# Create a copy for testing
import copy
merged = copy.deepcopy(main_config)
deep_merge(merged, dev_config)

print(f"After merge estimators: {list(merged['pose_estimation']['estimators'].keys())}")
print(f"Default estimator: {merged['pose_estimation']['default_estimator']}")

# Check if default is in estimators
default_in_estimators = merged['pose_estimation']['default_estimator'] in merged['pose_estimation']['estimators']
print(f"Default estimator in merged estimators: {default_in_estimators}")

# Step 3: Check what ConfigurationManager actually loads
print("\n3. CONFIGURATION MANAGER LOADING")
print("-" * 80)

from ambient.core.config import ConfigurationManager

config_mgr = ConfigurationManager(config_dir="config", environment="development")

print(f"Loaded estimators: {list(config_mgr.config.pose_estimation.estimators.keys())}")
print(f"Default estimator: {config_mgr.config.pose_estimation.default_estimator}")
print(f"Default in estimators: {config_mgr.config.pose_estimation.default_estimator in config_mgr.config.pose_estimation.estimators}")

# Step 4: Check enabled status
print("\n4. ESTIMATOR ENABLED STATUS")
print("-" * 80)

for name, est_config in config_mgr.config.pose_estimation.estimators.items():
    print(f"  {name}: enabled={est_config.enabled}")

# Step 5: Check raw config
print("\n5. RAW CONFIG INSPECTION")
print("-" * 80)

print(f"Raw config has pose_estimation: {'pose_estimation' in config_mgr._raw_config}")
if 'pose_estimation' in config_mgr._raw_config:
    pe_raw = config_mgr._raw_config['pose_estimation']
    print(f"Raw estimators: {list(pe_raw.get('estimators', {}).keys())}")
    print(f"Raw default_estimator: {pe_raw.get('default_estimator')}")

# Step 6: Validate and capture errors
print("\n6. VALIDATION ERRORS")
print("-" * 80)

# Temporarily capture validation
validation_errors = []
validation_warnings = []

# Call internal validation methods directly
config_mgr._validate_pose_estimation(validation_errors, validation_warnings)

print(f"Errors: {validation_errors}")
print(f"Warnings: {validation_warnings}")

print("\n" + "=" * 80)
print("INVESTIGATION COMPLETE")
print("=" * 80)
