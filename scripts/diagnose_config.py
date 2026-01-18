"""Diagnose configuration loading issues."""

import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from ambient.core.config import ConfigurationManager

# Create config manager
config_mgr = ConfigurationManager(config_dir="config", environment="development")

# Check pose estimation config
pe_config = config_mgr.config.pose_estimation

print("=" * 60)
print("POSE ESTIMATION CONFIGURATION")
print("=" * 60)
print(f"Default estimator: {pe_config.default_estimator}")
print(f"Confidence threshold: {pe_config.confidence_threshold}")
print(f"\nConfigured estimators: {list(pe_config.estimators.keys())}")
print(f"\nEnabled estimators:")
for name, est_config in pe_config.estimators.items():
    print(f"  - {name}: enabled={est_config.enabled}")

print(f"\nDefault estimator in dict: {pe_config.default_estimator in pe_config.estimators}")

# Check background tasks
bt_config = config_mgr.config.background_tasks
print("\n" + "=" * 60)
print("BACKGROUND TASKS CONFIGURATION")
print("=" * 60)
print(f"Enabled: {bt_config.enabled}")
print(f"Broker URL: {bt_config.celery.broker_url}")
