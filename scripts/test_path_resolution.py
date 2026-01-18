"""Test configuration path resolution from different directories."""

import sys
from pathlib import Path

print("=" * 80)
print("PATH RESOLUTION TEST")
print("=" * 80)

# Simulate running from different directories
test_cases = [
    ("project_root", Path("C:/Users/alexm/dev/alexpose")),
    ("server_dir", Path("C:/Users/alexm/dev/alexpose/server")),
    ("frontend_dir", Path("C:/Users/alexm/dev/alexpose/frontend")),
]

for name, cwd in test_cases:
    print(f"\n{name.upper()}: {cwd}")
    print("-" * 80)
    
    # Simulate the logic from server/main.py
    current_dir = cwd
    if current_dir.name == "server":
        config_dir = current_dir.parent / "config"
    else:
        config_dir = current_dir / "config"
    
    print(f"  current_dir.name: {current_dir.name}")
    print(f"  Resolved config_dir: {config_dir}")
    print(f"  Config exists: {config_dir.exists()}")
    
    if config_dir.exists():
        alexpose_yaml = config_dir / "alexpose.yaml"
        dev_yaml = config_dir / "development.yaml"
        print(f"  alexpose.yaml exists: {alexpose_yaml.exists()}")
        print(f"  development.yaml exists: {dev_yaml.exists()}")
    else:
        print(f"  ❌ CONFIG DIRECTORY NOT FOUND!")
        print(f"  This will cause configuration to fail!")

print("\n" + "=" * 80)
print("CONCLUSION")
print("=" * 80)
print("When running from 'frontend' directory:")
print("  - Config dir resolves to: frontend/config (WRONG!)")
print("  - Should resolve to: ../config or alexpose/config")
print("  - This causes empty estimators dict!")
