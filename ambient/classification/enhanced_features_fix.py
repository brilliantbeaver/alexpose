"""
Enhanced Feature Extraction Fix

This module provides the missing feature extraction logic to bridge the gap
between what's calculated by the analysis components and what's included
in the GaitFeatureVector.

The issue: EnhancedGaitAnalyzer calculates 60+ features, but GaitFeatureVector
only extracts 34 of them due to incomplete feature mapping.

Author: AlexPose Team
Date: January 27, 2026
"""

import numpy as np
from typing import Dict, Any, Optional
from loguru import logger


def extract_missing_features(analysis_results: Dict[str, Any]) -> Dict[str, Any]:
    """
    Extract the missing features that are calculated but not included in GaitFeatureVector.
    
    This function identifies and extracts the features that are available in the
    analysis results but not being used in the current feature vector creation.
    
    Args:
        analysis_results: Dictionary from EnhancedGaitAnalyzer.analyze_gait_sequence()
        
    Returns:
        Dictionary containing the missing features with proper naming
    """
    missing_features = {}
    
    # Extract components
    features_dict = analysis_results.get("features", {})
    timing_analysis = analysis_results.get("timing_analysis", {})
    phase_features = analysis_results.get("phase_features", {})
    symmetry_analysis = analysis_results.get("symmetry_analysis", {})
    
    # Helper function
    def safe_extract(source_dict, key, default=0.0):
        value = source_dict.get(key, default)
        return default if value is None or np.isnan(value) else float(value)
    
    # ========== MISSING JOINT ANGLE FEATURES ==========
    # Standard deviation, max, min for each joint
    joint_angle_features = [
        "left_hip", "left_knee", "left_ankle", 
        "right_hip", "right_knee", "right_ankle"
    ]
    
    for joint in joint_angle_features:
        missing_features[f"{joint}_std"] = safe_extract(features_dict, f"{joint}_std")
        missing_features[f"{joint}_max"] = safe_extract(features_dict, f"{joint}_max")
        missing_features[f"{joint}_min"] = safe_extract(features_dict, f"{joint}_min")
    
    # ========== MISSING TEMPORAL FEATURES ==========
    missing_features["sequence_length"] = safe_extract(features_dict, "sequence_length")
    missing_features["duration_seconds"] = safe_extract(features_dict, "duration_seconds")
    missing_features["dominant_frequency"] = safe_extract(features_dict, "dominant_frequency")
    missing_features["fps"] = safe_extract(features_dict, "fps", 30.0)
    
    # ========== MISSING STABILITY FEATURES ==========
    missing_features["com_movement_mean"] = safe_extract(features_dict, "com_movement_mean")
    missing_features["com_movement_std"] = safe_extract(features_dict, "com_movement_std")
    missing_features["com_stability_index"] = safe_extract(features_dict, "com_stability_index")
    missing_features["postural_sway_area"] = safe_extract(features_dict, "postural_sway_area")
    
    # ========== MISSING STRIDE FEATURES ==========
    missing_features["step_width_std"] = safe_extract(features_dict, "step_width_std")
    missing_features["step_width_range"] = safe_extract(features_dict, "step_width_range")
    missing_features["left_ankle_total_distance"] = safe_extract(features_dict, "left_ankle_total_distance")
    missing_features["right_ankle_total_distance"] = safe_extract(features_dict, "right_ankle_total_distance")
    missing_features["ankle_distance_asymmetry"] = safe_extract(features_dict, "ankle_distance_asymmetry")
    
    # ========== MISSING SYMMETRY FEATURES ==========
    # Individual joint symmetry indices
    symmetry_joints = ["shoulder", "elbow", "wrist", "hip", "knee", "ankle"]
    for joint in symmetry_joints:
        missing_features[f"{joint}_symmetry_index"] = safe_extract(features_dict, f"{joint}_symmetry_index")
    
    # ========== MISSING ADVANCED TEMPORAL FEATURES ==========
    # From TemporalAnalyzer
    missing_features["cycle_count"] = safe_extract(timing_analysis, "cycle_count")
    missing_features["left_cycle_duration_mean"] = safe_extract(timing_analysis, "left_cycle_duration_mean")
    missing_features["right_cycle_duration_mean"] = safe_extract(timing_analysis, "right_cycle_duration_mean")
    missing_features["cycle_duration_asymmetry"] = safe_extract(timing_analysis, "cycle_duration_asymmetry")
    missing_features["step_regularity_cv"] = safe_extract(timing_analysis, "step_regularity_cv")
    
    # Enhanced phase features
    missing_features["double_support_duration_mean"] = safe_extract(phase_features, "double_support_duration_mean")
    missing_features["stance_duration_mean"] = safe_extract(phase_features, "stance_duration_mean")
    missing_features["swing_duration_mean"] = safe_extract(phase_features, "swing_duration_mean")
    missing_features["phase_asymmetry"] = safe_extract(phase_features, "phase_asymmetry")
    
    # ========== MISSING ADVANCED SYMMETRY FEATURES ==========
    # From SymmetryAnalyzer
    missing_features["overall_symmetry_index"] = safe_extract(symmetry_analysis, "overall_symmetry_index")
    missing_features["symmetry_classification"] = symmetry_analysis.get("symmetry_classification", "unknown")
    missing_features["positional_symmetry_score"] = safe_extract(symmetry_analysis, "positional_symmetry_score")
    missing_features["movement_symmetry_score"] = safe_extract(symmetry_analysis, "movement_symmetry_score")
    missing_features["temporal_symmetry_score"] = safe_extract(symmetry_analysis, "temporal_symmetry_score")
    
    # ========== MISSING KINEMATIC FEATURES ==========
    # Enhanced velocity and acceleration features
    missing_features["walking_speed_pixels_per_sec"] = safe_extract(features_dict, "walking_speed_pixels_per_sec")
    missing_features["estimated_stride_length_pixels"] = safe_extract(features_dict, "estimated_stride_length_pixels")
    
    # Log summary of extracted features
    non_zero_features = {k: v for k, v in missing_features.items() if v != 0.0 and v != "unknown"}
    logger.info(f"Extracted {len(missing_features)} missing features, {len(non_zero_features)} have non-zero values")
    
    return missing_features


def get_comprehensive_feature_mapping() -> Dict[str, str]:
    """
    Get mapping of analysis result keys to GaitFeatureVector field names.
    
    This helps identify which features from the analysis results should map
    to which fields in the GaitFeatureVector.
    
    Returns:
        Dictionary mapping analysis result keys to feature vector field names
    """
    return {
        # Core joint angles (already mapped)
        "left_hip_mean": "left_hip_mean",
        "left_knee_mean": "left_knee_mean", 
        "left_ankle_mean": "left_ankle_mean",
        "right_hip_mean": "right_hip_mean",
        "right_knee_mean": "right_knee_mean",
        "right_ankle_mean": "right_ankle_mean",
        "left_hip_range": "left_hip_range",
        "left_knee_range": "left_knee_range",
        "left_ankle_range": "left_ankle_range", 
        "right_hip_range": "right_hip_range",
        "right_knee_range": "right_knee_range",
        "right_ankle_range": "right_ankle_range",
        
        # Kinematic features (already mapped)
        "velocity_mean": "velocity_mean",
        "velocity_std": "velocity_std",
        "velocity_max": "velocity_max",
        "velocity_min": "velocity_min",
        "acceleration_mean": "acceleration_mean",
        "acceleration_std": "acceleration_std",
        "acceleration_max": "acceleration_max",
        "jerk_mean": "jerk_mean",
        "jerk_std": "jerk_std",
        
        # MISSING MAPPINGS - These should be added to GaitFeatureVector
        "left_hip_std": "left_hip_std",
        "left_hip_max": "left_hip_max", 
        "left_hip_min": "left_hip_min",
        "left_knee_std": "left_knee_std",
        "left_knee_max": "left_knee_max",
        "left_knee_min": "left_knee_min",
        "left_ankle_std": "left_ankle_std",
        "left_ankle_max": "left_ankle_max",
        "left_ankle_min": "left_ankle_min",
        "right_hip_std": "right_hip_std",
        "right_hip_max": "right_hip_max",
        "right_hip_min": "right_hip_min",
        "right_knee_std": "right_knee_std",
        "right_knee_max": "right_knee_max",
        "right_knee_min": "right_knee_min",
        "right_ankle_std": "right_ankle_std",
        "right_ankle_max": "right_ankle_max",
        "right_ankle_min": "right_ankle_min",
        
        # Temporal features
        "sequence_length": "sequence_length",
        "duration_seconds": "duration_seconds",
        "dominant_frequency": "dominant_frequency",
        "fps": "fps",
        
        # Stability features
        "com_movement_mean": "com_movement_mean",
        "com_movement_std": "com_movement_std", 
        "com_stability_index": "com_stability_index",
        "postural_sway_area": "postural_sway_area",
        
        # Enhanced stride features
        "step_width_std": "step_width_std",
        "step_width_range": "step_width_range",
        "left_ankle_total_distance": "left_ankle_total_distance",
        "right_ankle_total_distance": "right_ankle_total_distance",
        "ankle_distance_asymmetry": "ankle_distance_asymmetry",
        
        # Individual joint symmetry
        "shoulder_symmetry_index": "shoulder_symmetry_index",
        "elbow_symmetry_index": "elbow_symmetry_index",
        "wrist_symmetry_index": "wrist_symmetry_index",
        "hip_symmetry_index": "hip_symmetry_index",
        "knee_symmetry_index": "knee_symmetry_index",
        "ankle_symmetry_index": "ankle_symmetry_index",
        
        # Advanced temporal features
        "cycle_count": "cycle_count",
        "left_cycle_duration_mean": "left_cycle_duration_mean",
        "right_cycle_duration_mean": "right_cycle_duration_mean",
        "cycle_duration_asymmetry": "cycle_duration_asymmetry",
        "step_regularity_cv": "step_regularity_cv",
        "double_support_duration_mean": "double_support_duration_mean",
        "stance_duration_mean": "stance_duration_mean",
        "swing_duration_mean": "swing_duration_mean",
        "phase_asymmetry": "phase_asymmetry",
        
        # Advanced symmetry features
        "overall_symmetry_index": "overall_symmetry_index",
        "symmetry_classification": "symmetry_classification",
        "positional_symmetry_score": "positional_symmetry_score",
        "movement_symmetry_score": "movement_symmetry_score",
        "temporal_symmetry_score": "temporal_symmetry_score",
        
        # Enhanced kinematic features
        "walking_speed_pixels_per_sec": "walking_speed_pixels_per_sec",
        "estimated_stride_length_pixels": "estimated_stride_length_pixels",
    }


def diagnose_feature_extraction_gaps(analysis_results: Dict[str, Any]) -> Dict[str, Any]:
    """
    Diagnose what features are available vs what's being extracted.
    
    Args:
        analysis_results: Dictionary from EnhancedGaitAnalyzer
        
    Returns:
        Diagnostic report showing available vs extracted features
    """
    report = {
        "timestamp": np.datetime64('now').astype(str),
        "analysis_components": {},
        "feature_gaps": {},
        "recommendations": []
    }
    
    # Analyze each component
    components = ["features", "timing_analysis", "phase_features", "symmetry_analysis"]
    
    for component in components:
        component_data = analysis_results.get(component, {})
        if component_data:
            report["analysis_components"][component] = {
                "available_features": list(component_data.keys()),
                "feature_count": len(component_data),
                "non_zero_features": len([k for k, v in component_data.items() 
                                        if isinstance(v, (int, float)) and v != 0.0])
            }
        else:
            report["analysis_components"][component] = {
                "available_features": [],
                "feature_count": 0,
                "non_zero_features": 0,
                "issue": "Component missing or empty"
            }
    
    # Identify gaps
    comprehensive_mapping = get_comprehensive_feature_mapping()
    available_features = set()
    
    for component_data in [analysis_results.get(c, {}) for c in components]:
        available_features.update(component_data.keys())
    
    mapped_features = set(comprehensive_mapping.keys())
    
    report["feature_gaps"] = {
        "total_available": len(available_features),
        "total_mappable": len(mapped_features),
        "unmapped_features": list(available_features - mapped_features),
        "missing_features": list(mapped_features - available_features)
    }
    
    # Generate recommendations
    if report["feature_gaps"]["unmapped_features"]:
        report["recommendations"].append(
            f"Add {len(report['feature_gaps']['unmapped_features'])} unmapped features to GaitFeatureVector"
        )
    
    if report["feature_gaps"]["missing_features"]:
        report["recommendations"].append(
            f"Fix {len(report['feature_gaps']['missing_features'])} missing features in analysis pipeline"
        )
    
    # Check for zero-value features
    zero_features = []
    for component in components:
        component_data = analysis_results.get(component, {})
        for key, value in component_data.items():
            if isinstance(value, (int, float)) and value == 0.0:
                zero_features.append(f"{component}.{key}")
    
    if len(zero_features) > 10:  # Threshold for concern
        report["recommendations"].append(
            f"Investigate {len(zero_features)} features with zero values - may indicate calculation issues"
        )
    
    return report


# Example usage and testing
if __name__ == "__main__":
    # This would be used to test the feature extraction
    print("Enhanced Feature Extraction Fix - Ready for Integration")
    print("Use extract_missing_features() to get the missing features")
    print("Use diagnose_feature_extraction_gaps() to analyze the pipeline")