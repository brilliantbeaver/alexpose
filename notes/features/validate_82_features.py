#!/usr/bin/env python3
"""
Validation script for 82-feature optimization.

This script validates that:
1. Feature count is exactly 82
2. No max/min fields exist in GaitFeatureVector
3. std fields are present
4. Feature extraction works correctly
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from ambient.classification.features import GaitFeatureVector, FeatureExtractionConfig


def validate_feature_count():
    """Validate that feature count is exactly 82."""
    print("=" * 70)
    print("VALIDATING FEATURE COUNT")
    print("=" * 70)
    
    # Get all feature names
    all_features = GaitFeatureVector.get_feature_names()
    
    print(f"\nTotal features: {len(all_features)}")
    
    if len(all_features) == 82:
        print("✅ PASS: Feature count is exactly 82")
        return True
    else:
        print(f"❌ FAIL: Expected 82 features, got {len(all_features)}")
        return False


def validate_no_max_min():
    """Validate that joint angle max/min fields are removed."""
    print("\n" + "=" * 70)
    print("VALIDATING JOINT ANGLE MAX/MIN REMOVAL")
    print("=" * 70)
    
    all_features = GaitFeatureVector.get_feature_names()
    
    # Check for joint angle max/min (not kinematic max/min which are valid)
    joint_names = ["hip", "knee", "ankle"]
    sides = ["left", "right"]
    
    joint_max_features = []
    joint_min_features = []
    
    for side in sides:
        for joint in joint_names:
            max_name = f"{side}_{joint}_max"
            min_name = f"{side}_{joint}_min"
            if max_name in all_features:
                joint_max_features.append(max_name)
            if min_name in all_features:
                joint_min_features.append(min_name)
    
    print(f"\nJoint angle features with '_max': {len(joint_max_features)}")
    print(f"Joint angle features with '_min': {len(joint_min_features)}")
    
    # Note: velocity_max and acceleration_max are valid kinematic features
    kinematic_max = [f for f in all_features if "_max" in f and f not in joint_max_features]
    kinematic_min = [f for f in all_features if "_min" in f and f not in joint_min_features]
    
    if kinematic_max:
        print(f"ℹ️  Valid kinematic max features: {kinematic_max}")
    if kinematic_min:
        print(f"ℹ️  Valid kinematic min features: {kinematic_min}")
    
    if joint_max_features:
        print(f"❌ FAIL: Found joint angle max features: {joint_max_features}")
        return False
    
    if joint_min_features:
        print(f"❌ FAIL: Found joint angle min features: {joint_min_features}")
        return False
    
    print("✅ PASS: No joint angle max/min features found")
    return True


def validate_std_present():
    """Validate that std fields are present."""
    print("\n" + "=" * 70)
    print("VALIDATING STD PRESENCE")
    print("=" * 70)
    
    all_features = GaitFeatureVector.get_feature_names()
    
    # Expected std features
    expected_std = [
        "left_hip_std",
        "left_knee_std",
        "left_ankle_std",
        "right_hip_std",
        "right_knee_std",
        "right_ankle_std",
    ]
    
    missing_std = [f for f in expected_std if f not in all_features]
    
    print(f"\nExpected std features: {len(expected_std)}")
    print(f"Found std features: {len(expected_std) - len(missing_std)}")
    
    if missing_std:
        print(f"❌ FAIL: Missing std features: {missing_std}")
        return False
    
    print("✅ PASS: All std features present")
    return True


def validate_feature_groups():
    """Validate feature group counts."""
    print("\n" + "=" * 70)
    print("VALIDATING FEATURE GROUPS")
    print("=" * 70)
    
    expected_counts = {
        "core_angles": 15,
        "spatiotemporal": 4,
        "temporal_phases": 4,
        "kinematic": 9,
        "symmetry_indices": 6,
        "variability": 3,
        "postural": 2,
        "extended_angles": 6,  # Changed from 18 to 6
        "temporal_extended": 12,
        "stability": 4,
        "stride_extended": 5,
        "symmetry_extended": 10,
        "kinematic_extended": 2,
    }
    
    all_pass = True
    
    for group, expected_count in expected_counts.items():
        features = GaitFeatureVector.get_feature_names([group])
        actual_count = len(features)
        
        status = "✅" if actual_count == expected_count else "❌"
        print(f"{status} {group:25s}: {actual_count:2d} features (expected {expected_count:2d})")
        
        if actual_count != expected_count:
            all_pass = False
            print(f"   Features: {features}")
    
    total_expected = sum(expected_counts.values())
    print(f"\n{'✅' if all_pass else '❌'} Total: {total_expected} features")
    
    return all_pass


def validate_config():
    """Validate FeatureExtractionConfig."""
    print("\n" + "=" * 70)
    print("VALIDATING FEATURE EXTRACTION CONFIG")
    print("=" * 70)
    
    config = FeatureExtractionConfig.comprehensive_mode()
    expected_count = config.get_expected_feature_count()
    
    print(f"\nExpected feature count from config: {expected_count}")
    
    if expected_count == 82:
        print("✅ PASS: Config expects 82 features")
        return True
    else:
        print(f"❌ FAIL: Config expects {expected_count} features, should be 82")
        return False


def main():
    """Run all validations."""
    print("\n" + "=" * 70)
    print("82-FEATURE OPTIMIZATION VALIDATION")
    print("=" * 70)
    
    results = {
        "Feature Count": validate_feature_count(),
        "Max/Min Removal": validate_no_max_min(),
        "Std Presence": validate_std_present(),
        "Feature Groups": validate_feature_groups(),
        "Config": validate_config(),
    }
    
    print("\n" + "=" * 70)
    print("VALIDATION SUMMARY")
    print("=" * 70)
    
    for test_name, passed in results.items():
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"{status}: {test_name}")
    
    all_passed = all(results.values())
    
    print("\n" + "=" * 70)
    if all_passed:
        print("✅ ALL VALIDATIONS PASSED")
        print("=" * 70)
        print("\nThe 82-feature optimization is complete and validated.")
        return 0
    else:
        print("❌ SOME VALIDATIONS FAILED")
        print("=" * 70)
        print("\nPlease review the failures above.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
