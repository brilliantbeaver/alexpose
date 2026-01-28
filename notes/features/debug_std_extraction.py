#!/usr/bin/env python3
"""
Debug script to trace std extraction issue.

This will help us understand if:
1. std values are in features_dict
2. std values are being extracted correctly
3. Where the 0.00 values are coming from
"""

import sys
from pathlib import Path

project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

# Create a mock analysis_results with known std values
mock_analysis_results = {
    "features": {
        "left_hip_mean": 173.19,
        "left_hip_range": 18.19,
        "left_hip_std": 5.5,  # NON-ZERO VALUE
        "left_knee_mean": 167.92,
        "left_knee_range": 51.77,
        "left_knee_std": 12.3,  # NON-ZERO VALUE
        "left_ankle_mean": 147.31,
        "left_ankle_range": 62.75,
        "left_ankle_std": 15.8,  # NON-ZERO VALUE
        "right_hip_mean": 171.89,
        "right_hip_range": 20.00,
        "right_hip_std": 6.2,  # NON-ZERO VALUE
        "right_knee_mean": 167.12,
        "right_knee_range": 38.30,
        "right_knee_std": 10.5,  # NON-ZERO VALUE
        "right_ankle_mean": 148.73,
        "right_ankle_range": 52.02,
        "right_ankle_std": 14.1,  # NON-ZERO VALUE
    },
    "timing_analysis": {},
    "phase_features": {},
    "symmetry_analysis": {},
}

print("=" * 70)
print("DEBUGGING STD EXTRACTION")
print("=" * 70)

print("\n1. Mock features_dict contains:")
for key, value in mock_analysis_results["features"].items():
    if "std" in key:
        print(f"   {key}: {value}")

# Import and test
from ambient.classification.features import GaitFeatureVector

print("\n2. Creating GaitFeatureVector from mock data...")
feature_vector = GaitFeatureVector.from_analysis_results(
    mock_analysis_results,
    sample_id="test_001",
    condition_label="test"
)

print("\n3. Extracted std values in GaitFeatureVector:")
std_features = [
    "left_hip_std",
    "left_knee_std",
    "left_ankle_std",
    "right_hip_std",
    "right_knee_std",
    "right_ankle_std",
]

for feature_name in std_features:
    value = getattr(feature_vector, feature_name)
    expected = mock_analysis_results["features"].get(feature_name, 0.0)
    status = "✅" if value == expected else "❌"
    print(f"   {status} {feature_name}: {value} (expected {expected})")

print("\n4. Checking mean/range for comparison:")
comparison_features = [
    "left_hip_mean",
    "left_hip_range",
    "right_hip_mean",
    "right_hip_range",
]

for feature_name in comparison_features:
    value = getattr(feature_vector, feature_name)
    expected = mock_analysis_results["features"].get(feature_name, 0.0)
    status = "✅" if abs(value - expected) < 0.01 else "❌"
    print(f"   {status} {feature_name}: {value} (expected {expected})")

print("\n" + "=" * 70)
print("CONCLUSION:")
print("=" * 70)

all_std_correct = all(
    getattr(feature_vector, name) == mock_analysis_results["features"].get(name, 0.0)
    for name in std_features
)

if all_std_correct:
    print("✅ std extraction is working correctly!")
    print("   The issue must be that std values are NOT in features_dict")
    print("   Check FeatureExtractor._extract_joint_angle_features()")
else:
    print("❌ std extraction is BROKEN!")
    print("   The values ARE in features_dict but not being extracted")
    print("   Check GaitFeatureVector.from_analysis_results()")
