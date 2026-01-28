#!/usr/bin/env python3
"""
Comprehensive Feature Extraction Diagnostic Tool

This script performs a deep investigation of the feature extraction pipeline
to identify any issues with feature calculation, extraction, or transfer.

Author: Kiro AI Assistant
Date: January 27, 2026
"""

import sys
import numpy as np
from pathlib import Path
from dataclasses import asdict
from typing import Dict, Any, List

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from ambient.analysis.gait_analyzer import EnhancedGaitAnalyzer
from ambient.analysis.feature_extractor import FeatureExtractor
from ambient.analysis.temporal_analyzer import TemporalAnalyzer
from ambient.analysis.symmetry_analyzer import SymmetryAnalyzer
from ambient.classification.features import GaitFeatureVector


def create_diagnostic_pose_sequence(num_frames=70):
    """Create a diagnostic pose sequence with known characteristics."""
    pose_sequence = []
    
    for frame in range(num_frames):
        keypoints = []
        
        # COCO_17 format: 17 keypoints
        for kp_idx in range(17):
            # Create realistic walking motion with known patterns
            base_x = 200 + np.sin(frame * 0.15) * 30
            base_y = 300 + np.cos(frame * 0.1) * 20
            
            # Add keypoint-specific offsets
            if kp_idx in [11, 12]:  # Hips
                y_offset = -50
                x_offset = 0
            elif kp_idx in [13, 14]:  # Knees
                y_offset = 0
                x_offset = np.sin(frame * 0.2) * 15
                base_x += x_offset
            elif kp_idx in [15, 16]:  # Ankles
                y_offset = 50
                x_offset = np.sin(frame * 0.25) * 25
                base_x += x_offset
            else:
                y_offset = np.random.uniform(-10, 10)
                x_offset = 0
            
            x = base_x + x_offset + np.random.uniform(-2, 2)
            y = base_y + y_offset + np.random.uniform(-2, 2)
            
            # Vary confidence to test threshold handling
            if frame < 5 or frame > 65:
                confidence = np.random.uniform(0.2, 0.5)  # Low confidence at edges
            elif kp_idx in [15, 16] and frame % 10 == 0:
                confidence = np.random.uniform(0.1, 0.3)  # Occasional low confidence
            else:
                confidence = np.random.uniform(0.7, 0.95)  # Good confidence
            
            keypoints.append({
                "x": float(x),
                "y": float(y),
                "confidence": float(confidence)
            })
        
        pose_sequence.append({"keypoints": keypoints})
    
    return pose_sequence


def diagnose_feature_extractor(pose_sequence: List[Dict[str, Any]]):
    """Diagnose the FeatureExtractor component."""
    print("\n" + "="*80)
    print("DIAGNOSING FEATURE EXTRACTOR")
    print("="*80)
    
    # Test with different confidence thresholds
    thresholds = [0.0, 0.3, 0.5]
    
    for threshold in thresholds:
        print(f"\n📊 Testing with confidence_threshold={threshold}")
        
        extractor = FeatureExtractor(
            keypoint_format="COCO_17",
            fps=30.0,
            confidence_threshold=threshold,
            include_joint_statistics=True,
            include_stability_features=True,
            include_advanced_temporal=True
        )
        
        features = extractor.extract_features(pose_sequence)
        
        # Analyze extracted features
        total_features = len(features)
        zero_features = [k for k, v in features.items() if v == 0.0]
        non_zero_features = total_features - len(zero_features)
        
        print(f"   Total features extracted: {total_features}")
        print(f"   Non-zero features: {non_zero_features}")
        print(f"   Zero-value features: {len(zero_features)}")
        
        # Check specific feature categories
        joint_stats = [k for k in features.keys() if any(x in k for x in ['_std', '_max', '_min'])]
        joint_stats_zero = [k for k in joint_stats if features[k] == 0.0]
        
        temporal_features = [k for k in features.keys() if any(x in k for x in ['sequence_length', 'duration', 'frequency', 'cadence'])]
        temporal_zero = [k for k in temporal_features if features[k] == 0.0]
        
        kinematic_features = [k for k in features.keys() if any(x in k for x in ['velocity', 'acceleration', 'jerk'])]
        kinematic_zero = [k for k in kinematic_features if features[k] == 0.0]
        
        print(f"\n   Feature Category Analysis:")
        print(f"   - Joint statistics: {len(joint_stats)} total, {len(joint_stats_zero)} zero")
        print(f"   - Temporal features: {len(temporal_features)} total, {len(temporal_zero)} zero")
        print(f"   - Kinematic features: {len(kinematic_features)} total, {len(kinematic_zero)} zero")
        
        if joint_stats_zero:
            print(f"   ⚠️  Zero joint statistics: {joint_stats_zero[:5]}")
        
        # Sample some feature values
        print(f"\n   Sample feature values:")
        sample_keys = list(features.keys())[:10]
        for key in sample_keys:
            print(f"   - {key}: {features[key]:.4f}")


def diagnose_temporal_analyzer(pose_sequence: List[Dict[str, Any]]):
    """Diagnose the TemporalAnalyzer component."""
    print("\n" + "="*80)
    print("DIAGNOSING TEMPORAL ANALYZER")
    print("="*80)
    
    # Test with different minimum cycle durations
    min_durations = [0.8, 0.5, 0.3]
    
    for min_duration in min_durations:
        print(f"\n📊 Testing with min_cycle_duration={min_duration}s")
        
        analyzer = TemporalAnalyzer(
            fps=30.0,
            min_cycle_duration=min_duration,
            max_cycle_duration=2.5
        )
        
        cycles = analyzer.detect_gait_cycles(pose_sequence)
        
        print(f"   Detected cycles: {len(cycles)}")
        
        if cycles:
            timing_analysis = analyzer.analyze_cycle_timing(cycles)
            
            print(f"   Timing analysis keys: {list(timing_analysis.keys())}")
            print(f"   Cycle count: {timing_analysis.get('cycle_count', 0)}")
            print(f"   Left cycle duration: {timing_analysis.get('left_cycle_duration_mean', 0):.3f}s")
            print(f"   Right cycle duration: {timing_analysis.get('right_cycle_duration_mean', 0):.3f}s")
            
            # Check for zero values
            zero_timing = [k for k, v in timing_analysis.items() if v == 0.0]
            if zero_timing:
                print(f"   ⚠️  Zero timing features: {zero_timing}")
        else:
            print(f"   ⚠️  No cycles detected - all temporal features will be 0.00")


def diagnose_symmetry_analyzer(pose_sequence: List[Dict[str, Any]]):
    """Diagnose the SymmetryAnalyzer component."""
    print("\n" + "="*80)
    print("DIAGNOSING SYMMETRY ANALYZER")
    print("="*80)
    
    # Test with different confidence thresholds
    thresholds = [0.5, 0.3, 0.1]
    
    for threshold in thresholds:
        print(f"\n📊 Testing with confidence_threshold={threshold}")
        
        analyzer = SymmetryAnalyzer(
            keypoint_format="COCO_17",
            confidence_threshold=threshold
        )
        
        symmetry_results = analyzer.analyze_symmetry(pose_sequence)
        
        total_features = len(symmetry_results)
        zero_features = [k for k, v in symmetry_results.items() if v == 0.0]
        non_zero_features = total_features - len(zero_features)
        
        print(f"   Total symmetry features: {total_features}")
        print(f"   Non-zero features: {non_zero_features}")
        print(f"   Zero-value features: {len(zero_features)}")
        
        # Check specific symmetry categories
        symmetry_indices = [k for k in symmetry_results.keys() if 'symmetry_index' in k]
        symmetry_scores = [k for k in symmetry_results.keys() if 'symmetry_score' in k]
        
        print(f"\n   Symmetry indices: {len(symmetry_indices)}")
        print(f"   Symmetry scores: {len(symmetry_scores)}")
        
        if zero_features:
            print(f"   ⚠️  Zero symmetry features: {zero_features[:5]}")


def diagnose_feature_vector_creation(analysis_results: Dict[str, Any]):
    """Diagnose the GaitFeatureVector creation from analysis results."""
    print("\n" + "="*80)
    print("DIAGNOSING FEATURE VECTOR CREATION")
    print("="*80)
    
    # Check what's in analysis_results
    print(f"\n📊 Analysis Results Structure:")
    print(f"   Keys: {list(analysis_results.keys())}")
    
    if "features" in analysis_results:
        features_dict = analysis_results["features"]
        print(f"   Features dict size: {len(features_dict)}")
        print(f"   Sample features: {list(features_dict.keys())[:10]}")
    
    if "timing_analysis" in analysis_results:
        timing = analysis_results["timing_analysis"]
        print(f"   Timing analysis keys: {list(timing.keys())}")
    
    if "symmetry_analysis" in analysis_results:
        symmetry = analysis_results["symmetry_analysis"]
        print(f"   Symmetry analysis keys: {list(symmetry.keys())}")
    
    # Create feature vector
    print(f"\n📊 Creating GaitFeatureVector:")
    
    feature_vector = GaitFeatureVector.from_analysis_results(
        analysis_results,
        sample_id="diagnostic_test",
        condition_label="test",
        feature_extraction_mode="comprehensive"
    )
    
    if feature_vector:
        # Convert to dict to analyze
        fv_dict = asdict(feature_vector)
        
        # Remove metadata fields
        metadata_fields = ['sample_id', 'condition_label', '_feature_groups_enabled']
        for field in metadata_fields:
            fv_dict.pop(field, None)
        
        total_features = len(fv_dict)
        zero_features = [k for k, v in fv_dict.items() if v == 0.0]
        non_zero_features = total_features - len(zero_features)
        
        print(f"   Total features in vector: {total_features}")
        print(f"   Non-zero features: {non_zero_features}")
        print(f"   Zero-value features: {len(zero_features)}")
        print(f"   Zero percentage: {len(zero_features)/total_features*100:.1f}%")
        
        # Categorize zero features
        joint_stats_zero = [k for k in zero_features if any(x in k for x in ['_std', '_max', '_min'])]
        temporal_zero = [k for k in zero_features if any(x in k for x in ['cycle_', 'phase_', 'duration'])]
        symmetry_zero = [k for k in zero_features if 'symmetry' in k]
        
        print(f"\n   Zero Features by Category:")
        print(f"   - Joint statistics: {len(joint_stats_zero)}")
        print(f"   - Temporal features: {len(temporal_zero)}")
        print(f"   - Symmetry features: {len(symmetry_zero)}")
        
        if joint_stats_zero:
            print(f"   ⚠️  Zero joint stats: {joint_stats_zero}")
        if temporal_zero:
            print(f"   ⚠️  Zero temporal: {temporal_zero[:5]}")
        if symmetry_zero:
            print(f"   ⚠️  Zero symmetry: {symmetry_zero[:5]}")
        
        # Check if features from analysis_results made it to feature_vector
        print(f"\n📊 Feature Transfer Verification:")
        
        # Check joint angle stats
        if "features" in analysis_results:
            features_dict = analysis_results["features"]
            
            # Check if left_hip_std exists in features_dict
            if "left_hip_std" in features_dict:
                source_value = features_dict["left_hip_std"]
                vector_value = feature_vector.left_hip_std
                print(f"   left_hip_std: source={source_value:.4f}, vector={vector_value:.4f}")
                if source_value != vector_value:
                    print(f"   ⚠️  VALUE MISMATCH!")
            else:
                print(f"   ⚠️  left_hip_std NOT in features_dict")
        
        return feature_vector
    else:
        print(f"   ❌ Feature vector creation FAILED!")
        return None


def diagnose_complete_pipeline():
    """Run complete pipeline diagnosis."""
    print("\n" + "🔬"*40)
    print("COMPLETE FEATURE EXTRACTION PIPELINE DIAGNOSIS")
    print("🔬"*40)
    
    # Create diagnostic data
    print(f"\n📊 Creating diagnostic pose sequence...")
    pose_sequence = create_diagnostic_pose_sequence(num_frames=70)
    
    # Calculate confidence statistics
    all_confidences = []
    for pose in pose_sequence:
        for kp in pose["keypoints"]:
            all_confidences.append(kp["confidence"])
    
    print(f"   Frames: {len(pose_sequence)}")
    print(f"   Keypoints per frame: {len(pose_sequence[0]['keypoints'])}")
    print(f"   Confidence range: {min(all_confidences):.3f} - {max(all_confidences):.3f}")
    print(f"   Mean confidence: {np.mean(all_confidences):.3f}")
    print(f"   Confidences < 0.3: {sum(1 for c in all_confidences if c < 0.3)} ({sum(1 for c in all_confidences if c < 0.3)/len(all_confidences)*100:.1f}%)")
    
    # Diagnose each component
    diagnose_feature_extractor(pose_sequence)
    diagnose_temporal_analyzer(pose_sequence)
    diagnose_symmetry_analyzer(pose_sequence)
    
    # Test complete pipeline
    print("\n" + "="*80)
    print("TESTING COMPLETE PIPELINE (EnhancedGaitAnalyzer)")
    print("="*80)
    
    analyzer = EnhancedGaitAnalyzer(
        keypoint_format="COCO_17",
        fps=30.0,
        comprehensive_features=True
    )
    
    analysis_results = analyzer.analyze_gait_sequence(pose_sequence)
    
    # Diagnose feature vector creation
    feature_vector = diagnose_feature_vector_creation(analysis_results)
    
    # Final summary
    print("\n" + "="*80)
    print("DIAGNOSIS SUMMARY")
    print("="*80)
    
    if feature_vector:
        fv_dict = asdict(feature_vector)
        metadata_fields = ['sample_id', 'condition_label', '_feature_groups_enabled']
        for field in metadata_fields:
            fv_dict.pop(field, None)
        
        zero_count = sum(1 for v in fv_dict.values() if v == 0.0)
        total_count = len(fv_dict)
        
        print(f"\n✅ Pipeline completed successfully")
        print(f"   Total features: {total_count}")
        print(f"   Non-zero features: {total_count - zero_count}")
        print(f"   Zero features: {zero_count}")
        print(f"   Zero percentage: {zero_count/total_count*100:.1f}%")
        
        if zero_count / total_count > 0.3:
            print(f"\n⚠️  WARNING: >30% features are zero - investigation needed!")
            print(f"   This suggests issues with:")
            print(f"   1. Confidence thresholds too strict")
            print(f"   2. Temporal analysis not detecting cycles")
            print(f"   3. Feature transfer from analysis_results to feature_vector")
        elif zero_count / total_count > 0.1:
            print(f"\n✅ ACCEPTABLE: 10-30% zero features is normal for short sequences")
        else:
            print(f"\n🎉 EXCELLENT: <10% zero features - pipeline working well!")
    else:
        print(f"\n❌ Pipeline FAILED - feature vector creation returned None")
    
    return feature_vector


def main():
    """Run the diagnostic tool."""
    try:
        feature_vector = diagnose_complete_pipeline()
        return feature_vector is not None
    except Exception as e:
        print(f"\n❌ Diagnostic failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
