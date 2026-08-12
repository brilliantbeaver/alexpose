"""
Helper functions for Jupyter notebooks.

This module provides convenience functions for extracting enhanced gait features
in notebook environments.

Author: AlexPose Team
"""

from typing import List, Dict, Any, Optional
from loguru import logger

from ambient.analysis.gait_analyzer import EnhancedGaitAnalyzer
from ambient.classification.features import GaitFeatureVector
from ambient.pose.keypoint_data import KeypointData


def extract_enhanced_features_from_keypoints(
    keypoints_array: List[KeypointData],
    sample_id: str = "",
    condition_label: str = "",
    keypoint_format: str = "BLAZEPOSE_33",
    fps: float = 30.0
) -> Optional[GaitFeatureVector]:
    """
    Extract enhanced gait features (34 features) from keypoints array.
    
    This is a convenience function for notebooks that wraps the EnhancedGaitAnalyzer
    to extract comprehensive features including spatiotemporal, temporal phases,
    symmetry indices, variability, and postural features.
    
    Args:
        keypoints_array: List of KeypointData objects from pose estimation
        sample_id: Identifier for this sample
        condition_label: Ground truth condition label (e.g., "normal", "parkinsons")
        keypoint_format: Format of keypoints (default: "BLAZEPOSE_33")
        fps: Frames per second of the video (default: 30.0)
        
    Returns:
        GaitFeatureVector with 34 comprehensive features, or None if extraction fails
        
    Example:
        >>> from ambient.utils.notebook_helpers import extract_enhanced_features_from_keypoints
        >>> 
        >>> # After extracting keypoints
        >>> keypoints_array = extractor.extract_from_sequence(...)
        >>> 
        >>> # Extract enhanced features
        >>> features = extract_enhanced_features_from_keypoints(
        ...     keypoints_array,
        ...     sample_id="normal_001",
        ...     condition_label="normal"
        ... )
        >>> 
        >>> # Get all 34 features
        >>> X = features.to_array()
        >>> print(f"Extracted {len(X)} features")
    """
    try:
        # Convert keypoints to pose sequence format expected by analyzer
        pose_sequence = []
        for kp_data in keypoints_array:
            frame_dict = {
                "keypoints": [
                    {
                        "x": kp.x,
                        "y": kp.y,
                        "confidence": kp.confidence
                    }
                    for kp in kp_data.keypoints
                ],
                "frame_index": getattr(kp_data, 'frame_index', 0),
                "timestamp": getattr(kp_data, 'timestamp', 0.0)
            }
            pose_sequence.append(frame_dict)
        
        # Use EnhancedGaitAnalyzer to get comprehensive analysis
        analyzer = EnhancedGaitAnalyzer(
            keypoint_format=keypoint_format,
            fps=fps
        )
        
        analysis_results = analyzer.analyze_gait_sequence(pose_sequence)
        
        # Check for errors
        if "error" in analysis_results:
            logger.error(f"Analysis failed: {analysis_results['error']}")
            return None
        
        # Extract features using from_analysis_results
        feature_vector = GaitFeatureVector.from_analysis_results(
            analysis_results,
            sample_id=sample_id,
            condition_label=condition_label
        )
        
        return feature_vector
        
    except Exception as e:
        logger.error(f"Failed to extract enhanced features: {e}")
        return None


def print_feature_comparison(
    legacy_features: GaitFeatureVector,
    enhanced_features: GaitFeatureVector
):
    """
    Print a comparison between legacy (15) and enhanced (34) features.
    
    Args:
        legacy_features: Feature vector from from_joint_angles()
        enhanced_features: Feature vector from from_analysis_results()
    """
    print("=" * 80)
    print("FEATURE COMPARISON: Legacy (15) vs Enhanced (34)")
    print("=" * 80)
    
    # Core angles (both have these)
    print("\n📊 CORE JOINT ANGLES (15 features) - Both methods")
    print("-" * 80)
    core_names = GaitFeatureVector.get_feature_names(["core_angles"])
    
    print(f"{'Feature':<25} {'Legacy':>12} {'Enhanced':>12} {'Difference':>12}")
    print("-" * 80)
    for name in core_names:
        legacy_val = getattr(legacy_features, name)
        enhanced_val = getattr(enhanced_features, name)
        diff = enhanced_val - legacy_val
        print(f"{name:<25} {legacy_val:>12.2f} {enhanced_val:>12.2f} {diff:>12.2f}")
    
    # New features (only enhanced has these)
    print("\n✨ NEW FEATURES (19 features) - Enhanced only")
    print("-" * 80)
    
    # Spatiotemporal
    print("\n  Spatiotemporal Parameters (4 features):")
    spatio_names = GaitFeatureVector.get_feature_names(["spatiotemporal"])
    for name in spatio_names:
        val = getattr(enhanced_features, name)
        print(f"    {name:<30} {val:>10.2f}")
    
    # Temporal phases
    print("\n  Temporal Phase Features (4 features):")
    temporal_names = GaitFeatureVector.get_feature_names(["temporal_phases"])
    for name in temporal_names:
        val = getattr(enhanced_features, name)
        print(f"    {name:<30} {val:>10.2f}")
    
    # Symmetry indices
    print("\n  Symmetry Indices (6 features):")
    symmetry_names = GaitFeatureVector.get_feature_names(["symmetry_indices"])
    for name in symmetry_names:
        val = getattr(enhanced_features, name)
        status = "✓ Normal" if val < 12 else "⚠ Mild" if val < 16 else "✗ Pathological"
        print(f"    {name:<30} {val:>10.2f}%  {status}")
    
    # Variability
    print("\n  Variability Metrics (3 features):")
    variability_names = GaitFeatureVector.get_feature_names(["variability"])
    for name in variability_names:
        val = getattr(enhanced_features, name)
        print(f"    {name:<30} {val:>10.3f}")
    
    # Postural
    print("\n  Postural Features (2 features):")
    postural_names = GaitFeatureVector.get_feature_names(["postural"])
    for name in postural_names:
        val = getattr(enhanced_features, name)
        print(f"    {name:<30} {val:>10.2f}°")
    
    print("\n" + "=" * 80)
    print(f"Total: Legacy = 15 features, Enhanced = 34 features (+19 new)")
    print("=" * 80)


def print_feature_summary(feature_vector: GaitFeatureVector, show_all: bool = True):
    """
    Print a formatted summary of the feature vector.
    
    Args:
        feature_vector: GaitFeatureVector to summarize
        show_all: If True, shows all 34 features. If False, shows only core 15.
    """
    print(feature_vector.get_feature_summary(include_all_groups=show_all))
