"""
Debug script to investigate why extended joint angle statistics are 0.00

This will print out exactly what's in features_dict after FeatureExtractor runs.
"""

import json
from ambient.analysis.gait_analyzer import EnhancedGaitAnalyzer
from ambient.classification.features import GaitFeatureVector


def debug_extended_features(pose_sequence, sample_id="debug"):
    """
    Debug extended feature extraction.
    
    Args:
        pose_sequence: List of pose estimation results
        sample_id: Sample identifier
    """
    print("=" * 80)
    print("EXTENDED FEATURES DEBUG")
    print("=" * 80)
    print()
    
    # Initialize analyzer
    print("1. Initializing EnhancedGaitAnalyzer with comprehensive features...")
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
    
    print(f"   ✓ include_joint_statistics: {analyzer.feature_extractor.include_joint_statistics}")
    print(f"   ✓ extract_extended_features: {analyzer.feature_extractor.extract_extended_features}")
    print(f"   ✓ confidence_threshold: {analyzer.feature_extractor.confidence_threshold}")
    print()
    
    # Analyze
    print("2. Running analysis...")
    results = analyzer.analyze_gait_sequence(pose_sequence)
    print("   ✓ Analysis complete")
    print()
    
    # Check features_dict
    print("3. Checking features_dict...")
    features_dict = results.get("features", {})
    print(f"   Total keys in features_dict: {len(features_dict)}")
    print()
    
    # Look for extended angle features
    extended_keys = [
        "left_hip_std", "left_hip_max", "left_hip_min",
        "left_knee_std", "left_knee_max", "left_knee_min",
        "left_ankle_std", "left_ankle_max", "left_ankle_min",
        "right_hip_std", "right_hip_max", "right_hip_min",
        "right_knee_std", "right_knee_max", "right_knee_min",
        "right_ankle_std", "right_ankle_max", "right_ankle_min"
    ]
    
    print("4. Extended joint angle features in features_dict:")
    found = []
    missing = []
    
    for key in extended_keys:
        if key in features_dict:
            value = features_dict[key]
            found.append(key)
            print(f"   ✓ {key}: {value}")
        else:
            missing.append(key)
            print(f"   ✗ {key}: NOT IN DICT")
    
    print()
    print(f"   Found: {len(found)}/{len(extended_keys)}")
    print(f"   Missing: {len(missing)}/{len(extended_keys)}")
    print()
    
    # If missing, check what angle-related keys ARE in the dict
    if missing:
        print("5. What angle-related keys ARE in features_dict?")
        angle_keys = [k for k in features_dict.keys() if any(joint in k for joint in ["hip", "knee", "ankle"])]
        print(f"   Found {len(angle_keys)} angle-related keys:")
        for key in sorted(angle_keys):
            print(f"     - {key}: {features_dict[key]}")
        print()
    
    # Check phase_features
    print("6. Checking phase_features...")
    phase_features = results.get("phase_features", {})
    if "phase_asymmetry" in phase_features:
        print(f"   ✓ phase_asymmetry: {phase_features['phase_asymmetry']}")
    else:
        print(f"   ✗ phase_asymmetry: NOT IN DICT")
    
    # Print all phase_features keys
    print(f"   All phase_features keys: {list(phase_features.keys())}")
    print()
    
    # Extract GaitFeatureVector
    print("7. Extracting GaitFeatureVector...")
    features = GaitFeatureVector.from_analysis_results(results, sample_id=sample_id)
    
    if features is None:
        print("   ✗ Feature extraction returned None!")
        return
    
    print("   ✓ Feature vector created")
    print()
    
    # Check final values
    print("8. Final feature values:")
    print(f"   left_hip_std: {features.left_hip_std}")
    print(f"   left_knee_std: {features.left_knee_std}")
    print(f"   left_ankle_std: {features.left_ankle_std}")
    print(f"   phase_asymmetry: {features.phase_asymmetry}")
    print()
    
    # Diagnosis
    print("=" * 80)
    print("DIAGNOSIS")
    print("=" * 80)
    
    if len(missing) == len(extended_keys):
        print("❌ CRITICAL: Extended angle features are NOT being created by FeatureExtractor")
        print()
        print("Possible causes:")
        print("  1. include_joint_statistics is not actually True when extract_features runs")
        print("  2. Angle arrays are empty (no frames pass confidence threshold)")
        print("  3. Exception is being caught and logged")
        print()
        print("Check logs for:")
        print("  - 'Joint angle extraction failed' warnings")
        print("  - 'No valid angle values for...' warnings")
        
    elif len(missing) > 0:
        print(f"⚠️  PARTIAL: {len(found)} features found, {len(missing)} missing")
        print(f"   Missing: {missing}")
        
    else:
        print("✅ SUCCESS: All extended angle features are in features_dict")
        if features.left_hip_std == 0.0:
            print("   But values are 0.0 - check if angle arrays are empty")
    
    print()
    return results, features


if __name__ == "__main__":
    print("This is a debug module. Import and call debug_extended_features()")
    print("Example:")
    print("  from debug_extended_features import debug_extended_features")
    print("  results, features = debug_extended_features(pose_sequence)")
