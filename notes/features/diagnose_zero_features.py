"""
Diagnostic script to investigate why extended joint angle statistics are 0.00

This script helps identify where in the pipeline features are being lost.
"""

import json
from ambient.analysis.gait_analyzer import EnhancedGaitAnalyzer
from ambient.classification.features import GaitFeatureVector


def diagnose_feature_extraction(pose_sequence, sample_id="test"):
    """
    Diagnose feature extraction pipeline to find where features are lost.
    
    Args:
        pose_sequence: List of pose estimation results
        sample_id: Sample identifier
    """
    print("=" * 80)
    print("FEATURE EXTRACTION DIAGNOSTIC")
    print("=" * 80)
    print()
    
    # Initialize analyzer with comprehensive features
    print("1. Initializing EnhancedGaitAnalyzer...")
    analyzer = EnhancedGaitAnalyzer(
        comprehensive_features=True,
        feature_extraction_config={
            "include_joint_statistics": True,
            "extract_extended_features": True,
            "include_stability_features": True,
            "include_advanced_temporal": True,
            "confidence_threshold": 0.3
        }
    )
    print(f"   ✓ Analyzer initialized")
    print(f"   - include_joint_statistics: {analyzer.feature_extractor.include_joint_statistics}")
    print(f"   - extract_extended_features: {analyzer.feature_extractor.extract_extended_features}")
    print()
    
    # Analyze sequence
    print("2. Analyzing gait sequence...")
    results = analyzer.analyze_gait_sequence(pose_sequence)
    print(f"   ✓ Analysis complete")
    print()
    
    # Check features_dict
    print("3. Checking features_dict from FeatureExtractor...")
    features_dict = results.get("features", {})
    print(f"   Total features in dict: {len(features_dict)}")
    
    # Check for extended joint angle features
    extended_angle_keys = [
        "left_hip_std", "left_hip_max", "left_hip_min",
        "left_knee_std", "left_knee_max", "left_knee_min",
        "left_ankle_std", "left_ankle_max", "left_ankle_min",
        "right_hip_std", "right_hip_max", "right_hip_min",
        "right_knee_std", "right_knee_max", "right_knee_min",
        "right_ankle_std", "right_ankle_max", "right_ankle_min"
    ]
    
    print("\n   Extended joint angle features in features_dict:")
    found_count = 0
    for key in extended_angle_keys:
        value = features_dict.get(key, "NOT FOUND")
        if value != "NOT FOUND":
            found_count += 1
            print(f"   ✓ {key}: {value}")
        else:
            print(f"   ✗ {key}: NOT FOUND")
    
    print(f"\n   Found {found_count}/{len(extended_angle_keys)} extended angle features")
    print()
    
    # Check timing_analysis
    print("4. Checking timing_analysis from TemporalAnalyzer...")
    timing_analysis = results.get("timing_analysis", {})
    print(f"   Total features in timing_analysis: {len(timing_analysis)}")
    
    critical_timing_keys = ["cycle_count", "left_cycle_count", "right_cycle_count"]
    print("\n   Critical timing features:")
    for key in critical_timing_keys:
        value = timing_analysis.get(key, "NOT FOUND")
        if value != "NOT FOUND":
            print(f"   ✓ {key}: {value}")
        else:
            print(f"   ✗ {key}: NOT FOUND")
    print()
    
    # Check phase_features
    print("5. Checking phase_features from TemporalAnalyzer...")
    phase_features = results.get("phase_features", {})
    print(f"   Total features in phase_features: {len(phase_features)}")
    
    if "phase_asymmetry" in phase_features:
        print(f"   ✓ phase_asymmetry: {phase_features['phase_asymmetry']}")
    else:
        print(f"   ✗ phase_asymmetry: NOT FOUND")
    print()
    
    # Check symmetry_analysis
    print("6. Checking symmetry_analysis from SymmetryAnalyzer...")
    symmetry_analysis = results.get("symmetry_analysis", {})
    print(f"   Total features in symmetry_analysis: {len(symmetry_analysis)}")
    
    symmetry_score_keys = [
        "positional_symmetry_score",
        "movement_symmetry_score",
        "temporal_symmetry_score"
    ]
    print("\n   Symmetry component scores:")
    for key in symmetry_score_keys:
        value = symmetry_analysis.get(key, "NOT FOUND")
        if value != "NOT FOUND":
            print(f"   ✓ {key}: {value}")
        else:
            print(f"   ✗ {key}: NOT FOUND")
    print()
    
    # Extract GaitFeatureVector
    print("7. Extracting GaitFeatureVector...")
    features = GaitFeatureVector.from_analysis_results(
        results,
        sample_id=sample_id,
        condition_label="test"
    )
    
    if features is None:
        print("   ✗ Feature extraction returned None!")
        return
    
    print(f"   ✓ Feature vector created")
    print()
    
    # Check final feature values
    print("8. Checking final feature values in GaitFeatureVector...")
    
    print("\n   Extended joint angle statistics:")
    extended_angle_features = {
        "left_hip_std": features.left_hip_std,
        "left_hip_max": features.left_hip_max,
        "left_hip_min": features.left_hip_min,
        "left_knee_std": features.left_knee_std,
        "left_knee_max": features.left_knee_max,
        "left_knee_min": features.left_knee_min,
        "left_ankle_std": features.left_ankle_std,
        "left_ankle_max": features.left_ankle_max,
        "left_ankle_min": features.left_ankle_min,
        "right_hip_std": features.right_hip_std,
        "right_hip_max": features.right_hip_max,
        "right_hip_min": features.right_hip_min,
        "right_knee_std": features.right_knee_std,
        "right_knee_max": features.right_knee_max,
        "right_knee_min": features.right_knee_min,
        "right_ankle_std": features.right_ankle_std,
        "right_ankle_max": features.right_ankle_max,
        "right_ankle_min": features.right_ankle_min,
    }
    
    zero_count = 0
    for key, value in extended_angle_features.items():
        if value == 0.0:
            zero_count += 1
            print(f"   ✗ {key}: {value} (ZERO)")
        else:
            print(f"   ✓ {key}: {value}")
    
    print(f"\n   {zero_count}/{len(extended_angle_features)} features are zero")
    print()
    
    print("\n   Other critical features:")
    print(f"   cycle_count: {features.cycle_count}")
    print(f"   phase_asymmetry: {features.phase_asymmetry}")
    print(f"   positional_symmetry_score: {features.positional_symmetry_score}")
    print(f"   movement_symmetry_score: {features.movement_symmetry_score}")
    print(f"   temporal_symmetry_score: {features.temporal_symmetry_score}")
    print()
    
    # Summary
    print("=" * 80)
    print("DIAGNOSTIC SUMMARY")
    print("=" * 80)
    
    if found_count == 0:
        print("❌ ISSUE: Extended angle features NOT FOUND in features_dict")
        print("   → FeatureExtractor is not creating these features")
        print("   → Check if include_joint_statistics is actually being used")
    elif zero_count == len(extended_angle_features):
        print("❌ ISSUE: Features found in features_dict but all zero in GaitFeatureVector")
        print("   → Problem in GaitFeatureVector.from_analysis_results()")
        print("   → Check safe_extract() calls and feature mapping")
    elif zero_count > 0:
        print(f"⚠️  PARTIAL ISSUE: {zero_count} features are zero")
        print("   → Some features are being extracted correctly")
        print("   → Others may have calculation issues")
    else:
        print("✅ SUCCESS: All extended angle features have non-zero values")
    
    print()
    
    return results, features


if __name__ == "__main__":
    print("This is a diagnostic module. Import and call diagnose_feature_extraction()")
    print("Example:")
    print("  from diagnose_zero_features import diagnose_feature_extraction")
    print("  results, features = diagnose_feature_extraction(pose_sequence)")
