#!/usr/bin/env python3
"""
Test script to verify comprehensive feature extraction.

This script tests the enhanced feature extraction pipeline to ensure
all 60+ features are being properly extracted and included in the
GaitFeatureVector.

Usage:
    python scripts/test_comprehensive_features.py

Author: AlexPose Team
Date: January 27, 2026
"""

import sys
from pathlib import Path
import numpy as np

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from ambient.analysis.gait_analyzer import EnhancedGaitAnalyzer
from ambient.classification.features import GaitFeatureVector
from ambient.classification.enhanced_features_fix import (
    extract_missing_features,
    diagnose_feature_extraction_gaps,
    get_comprehensive_feature_mapping
)


def create_mock_pose_sequence(num_frames: int = 100) -> list:
    """Create a mock pose sequence for testing."""
    pose_sequence = []
    
    for frame_idx in range(num_frames):
        # Create mock keypoints (33 keypoints for MediaPipe format)
        keypoints = []
        for kp_idx in range(33):
            # Add some realistic movement patterns
            x = 320 + np.sin(frame_idx * 0.1 + kp_idx) * 50
            y = 240 + np.cos(frame_idx * 0.1 + kp_idx) * 30
            confidence = 0.8 + np.random.random() * 0.2
            
            keypoints.append({
                "x": x,
                "y": y,
                "confidence": confidence
            })
        
        pose_sequence.append({
            "frame_number": frame_idx,
            "keypoints": keypoints,
            "timestamp": frame_idx / 30.0  # 30 FPS
        })
    
    return pose_sequence


def test_feature_extraction():
    """Test the comprehensive feature extraction pipeline."""
    print("=" * 70)
    print("COMPREHENSIVE FEATURE EXTRACTION TEST")
    print("=" * 70)
    
    # Create mock data
    print("1. Creating mock pose sequence...")
    pose_sequence = create_mock_pose_sequence(150)  # 5 seconds at 30 FPS
    print(f"   Created {len(pose_sequence)} frames")
    
    # Initialize analyzer
    print("\n2. Initializing EnhancedGaitAnalyzer...")
    analyzer = EnhancedGaitAnalyzer(
        keypoint_format="COCO_17",
        fps=30.0
    )
    
    # Run analysis
    print("\n3. Running comprehensive gait analysis...")
    analysis_results = analyzer.analyze_gait_sequence(pose_sequence)
    
    # Check analysis results
    print("\n4. Analysis Results Summary:")
    components = ["features", "timing_analysis", "phase_features", "symmetry_analysis"]
    for component in components:
        data = analysis_results.get(component, {})
        if data:
            non_zero = len([k for k, v in data.items() if isinstance(v, (int, float)) and v != 0.0])
            print(f"   {component:20s}: {len(data):3d} features ({non_zero:3d} non-zero)")
        else:
            print(f"   {component:20s}: MISSING")
    
    # Test legacy feature extraction (should be 15 features)
    print("\n5. Testing Legacy Feature Extraction...")
    try:
        # This would normally use joint angles, but we'll test with mock data
        legacy_features = GaitFeatureVector(
            sample_id="test_legacy",
            condition_label="test"
        )
        legacy_array = legacy_features.to_array(feature_groups=["core_angles"])
        print(f"   Legacy features: {len(legacy_array)} (expected: 15)")
    except Exception as e:
        print(f"   Legacy test failed: {e}")
    
    # Test comprehensive feature extraction
    print("\n6. Testing Comprehensive Feature Extraction...")
    try:
        comprehensive_features = GaitFeatureVector.from_analysis_results(
            analysis_results,
            sample_id="test_comprehensive",
            condition_label="test"
        )
        
        if comprehensive_features:
            # Test different feature group combinations
            all_features = comprehensive_features.to_array()
            core_only = comprehensive_features.to_array(feature_groups=["core_angles"])
            
            print(f"   All features: {len(all_features)} (expected: 60+)")
            print(f"   Core only: {len(core_only)} (expected: 15)")
            
            # Show feature group breakdown
            print("\n   Feature Group Breakdown:")
            feature_groups = comprehensive_features.get_feature_groups()
            total_features = 0
            for group_name, feature_names in feature_groups.items():
                group_array = comprehensive_features.to_array(feature_groups=[group_name])
                non_zero = np.count_nonzero(group_array)
                print(f"     {group_name:20s}: {len(feature_names):3d} features ({non_zero:3d} non-zero)")
                total_features += len(feature_names)
            
            print(f"\n   Total available features: {total_features}")
            
            # Test validation
            is_valid, issues = comprehensive_features.validate(check_all_groups=True)
            print(f"\n   Validation: {'PASSED' if is_valid else 'FAILED'}")
            if issues:
                print("   Issues found:")
                for issue in issues[:5]:  # Show first 5 issues
                    print(f"     - {issue}")
                if len(issues) > 5:
                    print(f"     ... and {len(issues) - 5} more")
        
        else:
            print("   ERROR: Failed to create comprehensive features")
    
    except Exception as e:
        print(f"   Comprehensive test failed: {e}")
        import traceback
        traceback.print_exc()
    
    # Diagnose feature gaps
    print("\n7. Diagnosing Feature Extraction Gaps...")
    try:
        diagnostic_report = diagnose_feature_extraction_gaps(analysis_results)
        
        print("   Analysis Components:")
        for component, info in diagnostic_report["analysis_components"].items():
            if "issue" in info:
                print(f"     {component:20s}: {info['issue']}")
            else:
                print(f"     {component:20s}: {info['feature_count']:3d} features ({info['non_zero_features']:3d} non-zero)")
        
        gaps = diagnostic_report["feature_gaps"]
        print(f"\n   Feature Mapping:")
        print(f"     Available features: {gaps['total_available']}")
        print(f"     Mappable features:  {gaps['total_mappable']}")
        print(f"     Unmapped features:  {len(gaps['unmapped_features'])}")
        print(f"     Missing features:   {len(gaps['missing_features'])}")
        
        if diagnostic_report["recommendations"]:
            print("\n   Recommendations:")
            for rec in diagnostic_report["recommendations"]:
                print(f"     - {rec}")
    
    except Exception as e:
        print(f"   Diagnostic failed: {e}")
    
    # Test missing features extraction
    print("\n8. Testing Missing Features Extraction...")
    try:
        missing_features = extract_missing_features(analysis_results)
        non_zero_missing = {k: v for k, v in missing_features.items() if v != 0.0 and v != "unknown"}
        
        print(f"   Missing features found: {len(missing_features)}")
        print(f"   Non-zero missing features: {len(non_zero_missing)}")
        
        if non_zero_missing:
            print("   Sample non-zero missing features:")
            for i, (key, value) in enumerate(list(non_zero_missing.items())[:5]):
                print(f"     {key:30s}: {value}")
            if len(non_zero_missing) > 5:
                print(f"     ... and {len(non_zero_missing) - 5} more")
    
    except Exception as e:
        print(f"   Missing features test failed: {e}")
    
    print("\n" + "=" * 70)
    print("TEST COMPLETE")
    print("=" * 70)


def test_feature_comparison():
    """Compare current vs expected feature counts."""
    print("\nFEATURE COUNT COMPARISON:")
    print("-" * 40)
    
    # Expected feature counts based on analysis
    expected_counts = {
        "core_angles": 15,           # Original features
        "spatiotemporal": 4,         # Walking speed, cadence, stride length, step width
        "temporal_phases": 4,        # Stance %, swing %, double support %, ratio
        "symmetry_indices": 6,       # SI for stride, stance, swing, hip, knee, ankle
        "kinematic": 9,              # Velocity, acceleration, jerk stats
        "variability": 3,            # CV for stride time, step length, velocity
        "postural": 2,               # Trunk lean, pelvic tilt
        "extended_angles": 18,       # Std, max, min for 6 joints
        "temporal_extended": 12,     # Extended temporal features
        "stability": 4,              # COM movement, stability, sway
        "stride_extended": 5,        # Extended stride features
        "symmetry_extended": 10,     # Extended symmetry features
        "kinematic_extended": 2,     # Extended kinematic features
    }
    
    total_expected = sum(expected_counts.values())
    print(f"Expected total features: {total_expected}")
    
    # Test actual counts
    try:
        feature_groups = GaitFeatureVector.get_feature_groups()
        total_actual = 0
        
        print("\nActual vs Expected:")
        for group_name, expected_count in expected_counts.items():
            actual_features = feature_groups.get(group_name, [])
            actual_count = len(actual_features)
            status = "✓" if actual_count == expected_count else "✗"
            print(f"  {group_name:20s}: {actual_count:3d} / {expected_count:3d} {status}")
            total_actual += actual_count
        
        print(f"\nTotal: {total_actual} / {total_expected} {'✓' if total_actual == total_expected else '✗'}")
        
        if total_actual != total_expected:
            print(f"\nDifference: {total_actual - total_expected} features")
    
    except Exception as e:
        print(f"Feature comparison failed: {e}")


if __name__ == "__main__":
    test_feature_extraction()
    test_feature_comparison()