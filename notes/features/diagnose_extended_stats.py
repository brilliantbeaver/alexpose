#!/usr/bin/env python3
"""
Diagnostic script to trace extended joint angle statistics extraction.

This script helps identify why extended statistics (std, max, min) are showing 0.00
despite having non-zero mean and range values.
"""

import sys
from pathlib import Path
from loguru import logger

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from ambient.analysis.gait_analyzer import EnhancedGaitAnalyzer
from ambient.classification.features import GaitFeatureVector


def diagnose_feature_extraction(sample_id: str = "cljo32213001u3n6lel97up5f"):
    """
    Diagnose feature extraction for a specific sample.
    
    This function:
    1. Loads the pose sequence
    2. Runs feature extraction with detailed logging
    3. Checks what's in features_dict
    4. Checks what makes it to GaitFeatureVector
    5. Reports any discrepancies
    """
    
    logger.info(f"Diagnosing feature extraction for sample: {sample_id}")
    
    # TODO: Load your pose sequence here
    # For now, we'll create a mock sequence to demonstrate the diagnostic flow
    logger.warning("Using mock pose sequence - replace with actual data loading")
    
    # Mock pose sequence (replace with actual data loading)
    pose_sequence = []
    
    if not pose_sequence:
        logger.error("No pose sequence loaded - cannot diagnose")
        logger.info("To use this script:")
        logger.info("1. Load your actual pose sequence data")
        logger.info("2. Pass it to the analyzer")
        logger.info("3. Check the output")
        return
    
    # Initialize analyzer with comprehensive features
    logger.info("Initializing EnhancedGaitAnalyzer with comprehensive_features=True")
    analyzer = EnhancedGaitAnalyzer(
        keypoint_format="COCO_17",
        fps=30.0,
        comprehensive_features=True
    )
    
    # Check if include_joint_statistics is set correctly
    logger.info(f"FeatureExtractor.include_joint_statistics = {analyzer.feature_extractor.include_joint_statistics}")
    
    # Run analysis
    logger.info("Running gait sequence analysis...")
    analysis_results = analyzer.analyze_gait_sequence(pose_sequence)
    
    # Check what's in features_dict
    features_dict = analysis_results.get("features", {})
    logger.info(f"Features dict contains {len(features_dict)} features")
    
    # Check for extended angle statistics
    extended_angle_keys = [
        "left_hip_std", "left_hip_max", "left_hip_min",
        "left_knee_std", "left_knee_max", "left_knee_min",
        "left_ankle_std", "left_ankle_max", "left_ankle_min",
        "right_hip_std", "right_hip_max", "right_hip_min",
        "right_knee_std", "right_knee_max", "right_knee_min",
        "right_ankle_std", "right_ankle_max", "right_ankle_min",
    ]
    
    logger.info("\n" + "="*70)
    logger.info("EXTENDED ANGLE STATISTICS IN features_dict:")
    logger.info("="*70)
    
    for key in extended_angle_keys:
        value = features_dict.get(key, "NOT FOUND")
        if value == "NOT FOUND":
            logger.error(f"  {key:30s}: NOT IN DICT")
        elif value == 0.0:
            logger.warning(f"  {key:30s}: {value:8.2f} (ZERO)")
        else:
            logger.success(f"  {key:30s}: {value:8.2f} (OK)")
    
    # Check mean and range values for comparison
    logger.info("\n" + "="*70)
    logger.info("MEAN AND RANGE VALUES (for comparison):")
    logger.info("="*70)
    
    comparison_keys = [
        "left_hip_mean", "left_hip_range",
        "left_knee_mean", "left_knee_range",
        "left_ankle_mean", "left_ankle_range",
        "right_hip_mean", "right_hip_range",
        "right_knee_mean", "right_knee_range",
        "right_ankle_mean", "right_ankle_range",
    ]
    
    for key in comparison_keys:
        value = features_dict.get(key, "NOT FOUND")
        if value == "NOT FOUND":
            logger.error(f"  {key:30s}: NOT IN DICT")
        elif value == 0.0:
            logger.warning(f"  {key:30s}: {value:8.2f}")
        else:
            logger.success(f"  {key:30s}: {value:8.2f}")
    
    # Create GaitFeatureVector
    logger.info("\n" + "="*70)
    logger.info("Creating GaitFeatureVector...")
    logger.info("="*70)
    
    feature_vector = GaitFeatureVector.from_analysis_results(
        analysis_results,
        sample_id=sample_id,
        condition_label="normal",
        feature_extraction_mode="comprehensive"
    )
    
    if feature_vector is None:
        logger.error("Failed to create GaitFeatureVector")
        return
    
    # Check what made it to the feature vector
    logger.info("\n" + "="*70)
    logger.info("EXTENDED ANGLE STATISTICS IN GaitFeatureVector:")
    logger.info("="*70)
    
    for key in extended_angle_keys:
        value = getattr(feature_vector, key, "NOT FOUND")
        if value == "NOT FOUND":
            logger.error(f"  {key:30s}: NOT IN VECTOR")
        elif value == 0.0:
            logger.warning(f"  {key:30s}: {value:8.2f} (ZERO)")
        else:
            logger.success(f"  {key:30s}: {value:8.2f} (OK)")
    
    # Summary
    logger.info("\n" + "="*70)
    logger.info("DIAGNOSIS SUMMARY:")
    logger.info("="*70)
    
    # Count zeros in features_dict
    dict_zeros = sum(1 for key in extended_angle_keys if features_dict.get(key, 0.0) == 0.0)
    dict_missing = sum(1 for key in extended_angle_keys if key not in features_dict)
    
    # Count zeros in feature_vector
    vector_zeros = sum(1 for key in extended_angle_keys if getattr(feature_vector, key, 0.0) == 0.0)
    
    logger.info(f"Extended angle statistics in features_dict:")
    logger.info(f"  - Missing from dict: {dict_missing}/18")
    logger.info(f"  - Zero values: {dict_zeros}/18")
    logger.info(f"  - Non-zero values: {18 - dict_zeros - dict_missing}/18")
    
    logger.info(f"\nExtended angle statistics in GaitFeatureVector:")
    logger.info(f"  - Zero values: {vector_zeros}/18")
    logger.info(f"  - Non-zero values: {18 - vector_zeros}/18")
    
    if dict_missing > 0:
        logger.error("\n❌ PROBLEM: Extended statistics are NOT being created in features_dict")
        logger.error("   This means FeatureExtractor._extract_joint_angle_features() is not creating them")
        logger.error("   Check if include_joint_statistics=True when FeatureExtractor is initialized")
    elif dict_zeros > 0:
        logger.warning("\n⚠️  PROBLEM: Extended statistics ARE in features_dict but have zero values")
        logger.warning("   This means the angle arrays might be empty or the calculation is failing")
        logger.warning("   Check the angle calculation logic in FeatureExtractor")
    elif vector_zeros > 0:
        logger.error("\n❌ PROBLEM: Extended statistics are in features_dict but not making it to GaitFeatureVector")
        logger.error("   This means GaitFeatureVector.from_analysis_results() is not extracting them correctly")
        logger.error("   Check the safe_extract() calls in from_analysis_results()")
    else:
        logger.success("\n✅ SUCCESS: All extended statistics are present and non-zero!")


if __name__ == "__main__":
    # Configure logging
    logger.remove()
    logger.add(
        sys.stderr,
        format="<green>{time:HH:mm:ss}</green> | <level>{level:8}</level> | <level>{message}</level>",
        level="DEBUG"
    )
    
    diagnose_feature_extraction()
