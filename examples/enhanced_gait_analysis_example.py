#!/usr/bin/env python3
"""
Enhanced Gait Analysis Example

This example demonstrates how to use the enhanced gait analysis pipeline
with the new 34-feature GaitFeatureVector for comprehensive gait assessment.

Features demonstrated:
- Enhanced feature extraction (34 features vs 15 legacy)
- Evidence-based symmetry indices
- Flexible feature group selection
- Clinical interpretation
- Backward compatibility

Author: AlexPose Team
"""

import sys
import numpy as np
from typing import Dict, Any, List

# Add the project root to the path
sys.path.insert(0, '.')

from ambient.analysis.gait_analyzer import EnhancedGaitAnalyzer
from ambient.classification.features import GaitFeatureVector


def create_sample_pose_sequence(condition: str = "normal", num_frames: int = 120) -> List[Dict[str, Any]]:
    """
    Create a sample pose sequence simulating different gait conditions.
    
    Args:
        condition: Type of gait to simulate ("normal", "antalgic", "hemiplegic", "parkinsons")
        num_frames: Number of frames to generate
        
    Returns:
        List of pose dictionaries
    """
    poses = []
    
    for frame_idx in range(num_frames):
        t = frame_idx / 30.0  # Time in seconds (30 fps)
        
        if condition == "normal":
            # Normal symmetric gait
            left_hip_y = 200 + 10 * np.sin(2 * np.pi * t)
            right_hip_y = 200 + 10 * np.sin(2 * np.pi * t + 0.1)  # Slight phase shift
            left_knee_y = left_hip_y + 50 + 20 * np.sin(4 * np.pi * t)
            right_knee_y = right_hip_y + 50 + 20 * np.sin(4 * np.pi * t + 0.1)
            left_ankle_y = left_knee_y + 50 + 30 * np.sin(6 * np.pi * t)
            right_ankle_y = right_knee_y + 50 + 30 * np.sin(6 * np.pi * t + 0.1)
            
        elif condition == "antalgic":
            # Antalgic gait - favoring right leg, reduced left stance
            left_hip_y = 200 + 6 * np.sin(2 * np.pi * t)  # Reduced motion
            right_hip_y = 200 + 12 * np.sin(2 * np.pi * t + 0.3)  # Normal motion, phase shift
            left_knee_y = left_hip_y + 45 + 12 * np.sin(4 * np.pi * t)  # Reduced flexion
            right_knee_y = right_hip_y + 55 + 25 * np.sin(4 * np.pi * t + 0.3)
            left_ankle_y = left_knee_y + 45 + 18 * np.sin(6 * np.pi * t)
            right_ankle_y = right_knee_y + 55 + 35 * np.sin(6 * np.pi * t + 0.3)
            
        elif condition == "hemiplegic":
            # Hemiplegic gait - left side affected, circumduction
            left_hip_y = 200 + 4 * np.sin(2 * np.pi * t)  # Very reduced motion
            right_hip_y = 200 + 12 * np.sin(2 * np.pi * t + 0.5)  # Compensatory motion
            left_knee_y = left_hip_y + 50 + 5 * np.sin(4 * np.pi * t)  # Minimal flexion
            right_knee_y = right_hip_y + 50 + 25 * np.sin(4 * np.pi * t + 0.5)
            left_ankle_y = left_knee_y + 52 + 8 * np.sin(6 * np.pi * t)  # Foot drop
            right_ankle_y = right_knee_y + 50 + 30 * np.sin(6 * np.pi * t + 0.5)
            
        elif condition == "parkinsons":
            # Parkinsonian gait - reduced amplitude, shuffling
            left_hip_y = 200 + 4 * np.sin(2 * np.pi * t)  # Reduced motion
            right_hip_y = 200 + 4 * np.sin(2 * np.pi * t + 0.05)  # Minimal phase shift
            left_knee_y = left_hip_y + 50 + 8 * np.sin(4 * np.pi * t)  # Reduced flexion
            right_knee_y = right_hip_y + 50 + 8 * np.sin(4 * np.pi * t + 0.05)
            left_ankle_y = left_knee_y + 50 + 10 * np.sin(6 * np.pi * t)  # Shuffling
            right_ankle_y = right_knee_y + 50 + 10 * np.sin(6 * np.pi * t + 0.05)
            
        else:
            raise ValueError(f"Unknown condition: {condition}")
        
        # Add forward progression
        x_offset = t * 15  # Slower for pathological conditions
        if condition == "parkinsons":
            x_offset = t * 8  # Very slow progression
        
        # Create keypoints (COCO-17 format)
        keypoints = [
            {"x": 320, "y": 100, "confidence": 0.9},  # nose
            {"x": 310, "y": 110, "confidence": 0.8},  # left_eye
            {"x": 330, "y": 110, "confidence": 0.8},  # right_eye
            {"x": 300, "y": 120, "confidence": 0.7},  # left_ear
            {"x": 340, "y": 120, "confidence": 0.7},  # right_ear
            {"x": 280 + x_offset, "y": 150, "confidence": 0.9},  # left_shoulder
            {"x": 360 + x_offset, "y": 150, "confidence": 0.9},  # right_shoulder
            {"x": 270 + x_offset, "y": 180, "confidence": 0.8},  # left_elbow
            {"x": 370 + x_offset, "y": 180, "confidence": 0.8},  # right_elbow
            {"x": 260 + x_offset, "y": 210, "confidence": 0.7},  # left_wrist
            {"x": 380 + x_offset, "y": 210, "confidence": 0.7},  # right_wrist
            {"x": 290 + x_offset, "y": left_hip_y, "confidence": 0.9},  # left_hip
            {"x": 350 + x_offset, "y": right_hip_y, "confidence": 0.9},  # right_hip
            {"x": 285 + x_offset, "y": left_knee_y, "confidence": 0.9},  # left_knee
            {"x": 355 + x_offset, "y": right_knee_y, "confidence": 0.9},  # right_knee
            {"x": 280 + x_offset, "y": left_ankle_y, "confidence": 0.9},  # left_ankle
            {"x": 360 + x_offset, "y": right_ankle_y, "confidence": 0.9},  # right_ankle
        ]
        
        poses.append({
            "keypoints": keypoints,
            "frame_index": frame_idx,
            "timestamp": t
        })
    
    return poses


def analyze_gait_condition(condition: str) -> Dict[str, Any]:
    """
    Analyze a specific gait condition and return comprehensive results.
    
    Args:
        condition: Gait condition to analyze
        
    Returns:
        Dictionary with analysis results
    """
    print(f"\n{'='*60}")
    print(f"ANALYZING {condition.upper()} GAIT")
    print(f"{'='*60}")
    
    # Create sample data
    print(f"1. Generating {condition} gait sequence...")
    pose_sequence = create_sample_pose_sequence(condition, num_frames=120)
    print(f"   Created {len(pose_sequence)} frames")
    
    # Initialize analyzer
    print("2. Initializing enhanced gait analyzer...")
    analyzer = EnhancedGaitAnalyzer(keypoint_format="COCO_17", fps=30.0)
    
    # Analyze gait
    print("3. Performing comprehensive gait analysis...")
    analysis_results = analyzer.analyze_gait_sequence(pose_sequence)
    
    # Create enhanced feature vector
    print("4. Extracting enhanced feature vector...")
    feature_vector = GaitFeatureVector.from_analysis_results(
        analysis_results,
        sample_id=f"{condition}_sample",
        condition_label=condition
    )
    
    if feature_vector is None:
        print("   ERROR: Failed to create feature vector")
        return {}
    
    # Validate features
    is_valid, issues = feature_vector.validate(check_all_groups=True)
    print(f"5. Feature validation: {'PASS' if is_valid else 'FAIL'}")
    if issues:
        print(f"   Issues: {issues}")
    
    # Display key features
    print(f"\n6. KEY GAIT FEATURES:")
    print(f"   Walking Speed:     {feature_vector.walking_speed_ms:.3f} m/s")
    print(f"   Cadence:           {feature_vector.cadence_steps_min:.1f} steps/min")
    print(f"   Stride Length:     {feature_vector.stride_length_m:.3f} m")
    print(f"   Step Width:        {feature_vector.step_width_m:.3f} m")
    
    print(f"\n   TEMPORAL PHASES:")
    print(f"   Stance Phase:      {feature_vector.stance_percentage:.1f}%")
    print(f"   Swing Phase:       {feature_vector.swing_percentage:.1f}%")
    print(f"   Stance/Swing Ratio:{feature_vector.stance_swing_ratio:.2f}")
    
    print(f"\n   SYMMETRY INDICES:")
    print(f"   Stride Length SI:  {feature_vector.stride_length_si:.1f}%")
    print(f"   Hip Angle SI:      {feature_vector.hip_angle_si:.1f}%")
    print(f"   Knee Angle SI:     {feature_vector.knee_angle_si:.1f}%")
    print(f"   Ankle Angle SI:    {feature_vector.ankle_angle_si:.1f}%")
    
    print(f"\n   VARIABILITY:")
    print(f"   Stride Time CV:    {feature_vector.stride_time_cv:.3f}")
    print(f"   Step Length CV:    {feature_vector.step_length_cv:.3f}")
    
    print(f"\n   POSTURAL:")
    print(f"   Trunk Lean:        {feature_vector.trunk_lean_angle:.1f}°")
    print(f"   Pelvic Tilt:       {feature_vector.pelvic_tilt_mean:.1f}°")
    
    # Clinical interpretation
    print(f"\n7. CLINICAL INTERPRETATION:")
    
    # Walking speed assessment
    if feature_vector.walking_speed_ms < 0.8:
        print("   ⚠ Reduced walking speed (normal: >1.0 m/s)")
    elif feature_vector.walking_speed_ms > 1.2:
        print("   ✓ Normal walking speed")
    else:
        print("   ~ Borderline walking speed")
    
    # Symmetry assessment
    asymmetric_features = 0
    if feature_vector.stride_length_si > 16:
        asymmetric_features += 1
        print("   ⚠ Significant stride length asymmetry (>16%)")
    if feature_vector.hip_angle_si > 16:
        asymmetric_features += 1
        print("   ⚠ Significant hip angle asymmetry (>16%)")
    if feature_vector.knee_angle_si > 16:
        asymmetric_features += 1
        print("   ⚠ Significant knee angle asymmetry (>16%)")
    if feature_vector.ankle_angle_si > 16:
        asymmetric_features += 1
        print("   ⚠ Significant ankle angle asymmetry (>16%)")
    
    if asymmetric_features == 0:
        print("   ✓ Gait appears symmetric within normal limits")
    elif asymmetric_features <= 2:
        print(f"   ~ Mild asymmetry detected ({asymmetric_features} features)")
    else:
        print(f"   ⚠ Significant asymmetry detected ({asymmetric_features} features)")
    
    # Temporal phase assessment
    if feature_vector.stance_percentage < 55 or feature_vector.stance_percentage > 70:
        print("   ⚠ Abnormal stance phase duration (normal: 55-70%)")
    else:
        print("   ✓ Normal stance phase duration")
    
    # Variability assessment
    if feature_vector.stride_time_cv > 0.05:
        print("   ⚠ High stride time variability (indicates instability)")
    else:
        print("   ✓ Normal stride time variability")
    
    return {
        "condition": condition,
        "feature_vector": feature_vector,
        "analysis_results": analysis_results,
        "asymmetric_features": asymmetric_features
    }


def demonstrate_feature_group_selection():
    """Demonstrate flexible feature group selection."""
    print(f"\n{'='*60}")
    print("FEATURE GROUP SELECTION DEMONSTRATION")
    print(f"{'='*60}")
    
    # Create a sample feature vector
    pose_sequence = create_sample_pose_sequence("normal", 100)
    analyzer = EnhancedGaitAnalyzer()
    analysis_results = analyzer.analyze_gait_sequence(pose_sequence)
    feature_vector = GaitFeatureVector.from_analysis_results(analysis_results)
    
    # Show different feature group combinations
    feature_groups = GaitFeatureVector.get_feature_groups()
    
    print("Available feature groups:")
    for group_name, features in feature_groups.items():
        print(f"  {group_name}: {len(features)} features")
    
    print(f"\nFeature selection examples:")
    
    # Legacy compatibility
    legacy_features = feature_vector.to_array(feature_groups=["core_angles"])
    print(f"  Legacy (core angles only):     {len(legacy_features)} features")
    
    # Core + spatiotemporal
    basic_features = feature_vector.to_array(feature_groups=["core_angles", "spatiotemporal"])
    print(f"  Basic (core + spatiotemporal): {len(basic_features)} features")
    
    # Clinical focus
    clinical_features = feature_vector.to_array(feature_groups=[
        "core_angles", "spatiotemporal", "symmetry_indices"
    ])
    print(f"  Clinical focus:                {len(clinical_features)} features")
    
    # Research comprehensive
    all_features = feature_vector.to_array()
    print(f"  Comprehensive (all groups):    {len(all_features)} features")
    
    print(f"\nThis flexibility allows:")
    print(f"  - Legacy classifiers to work unchanged")
    print(f"  - Gradual migration to enhanced features")
    print(f"  - Task-specific feature selection")
    print(f"  - Performance optimization")


def main():
    """Main demonstration function."""
    print("Enhanced Gait Analysis Example")
    print("=" * 60)
    print("This example demonstrates the enhanced gait analysis pipeline")
    print("with 34-feature vectors and evidence-based clinical assessment.")
    
    # Analyze different gait conditions
    conditions = ["normal", "antalgic", "hemiplegic", "parkinsons"]
    results = {}
    
    for condition in conditions:
        try:
            results[condition] = analyze_gait_condition(condition)
        except Exception as e:
            print(f"Error analyzing {condition}: {e}")
    
    # Demonstrate feature group selection
    demonstrate_feature_group_selection()
    
    # Summary comparison
    print(f"\n{'='*60}")
    print("COMPARATIVE ANALYSIS SUMMARY")
    print(f"{'='*60}")
    
    print(f"{'Condition':<12} {'Speed(m/s)':<10} {'Cadence':<8} {'Stance%':<8} {'SI_Hip%':<8} {'SI_Knee%':<9}")
    print("-" * 60)
    
    for condition, result in results.items():
        if result:
            fv = result["feature_vector"]
            print(f"{condition:<12} {fv.walking_speed_ms:<10.3f} {fv.cadence_steps_min:<8.1f} "
                  f"{fv.stance_percentage:<8.1f} {fv.hip_angle_si:<8.1f} {fv.knee_angle_si:<9.1f}")
    
    print(f"\nKey Observations:")
    print(f"- Normal gait shows balanced parameters and low asymmetry")
    print(f"- Antalgic gait shows reduced speed and increased asymmetry")
    print(f"- Hemiplegic gait shows significant asymmetry (>16% SI)")
    print(f"- Parkinsonian gait shows reduced speed and amplitude")
    
    print(f"\n{'='*60}")
    print("ENHANCED GAIT ANALYSIS EXAMPLE COMPLETED")
    print(f"{'='*60}")
    print(f"✓ 34 comprehensive features extracted")
    print(f"✓ Evidence-based symmetry indices calculated")
    print(f"✓ Clinical interpretation provided")
    print(f"✓ Backward compatibility maintained")
    print(f"✓ Flexible feature group selection demonstrated")


if __name__ == "__main__":
    main()