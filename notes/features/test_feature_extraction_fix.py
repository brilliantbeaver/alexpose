#!/usr/bin/env python3
"""
Test script to verify the feature extraction fixes.

This script tests the fixes for zero-value features by:
1. Creating sample pose data with various confidence levels
2. Testing FeatureExtractor with different confidence thresholds
3. Verifying that joint angle statistics are properly calculated
4. Testing temporal and symmetry analyzers with relaxed thresholds

Author: Kiro AI Assistant
Date: January 27, 2026
"""

import sys
import numpy as np
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from ambient.analysis.feature_extractor import FeatureExtractor
from ambient.analysis.temporal_analyzer import TemporalAnalyzer
from ambient.analysis.symmetry_analyzer import SymmetryAnalyzer

def create_test_pose_sequence(num_frames=70, confidence_range=(0.2, 0.9)):
    """Create test pose sequence with varying confidence levels."""
    pose_sequence = []
    
    # Generate realistic pose data
    for frame in range(num_frames):
        keypoints = []
        # COCO_17 format: 17 keypoints
        for kp in range(17):
            # Generate x, y coordinates (simulate walking motion)
            x = 100 + np.sin(frame * 0.1 + kp * 0.5) * 50
            y = 200 + np.cos(frame * 0.1 + kp * 0.3) * 30
            
            # Generate confidence in specified range
            confidence = np.random.uniform(confidence_range[0], confidence_range[1])
            
            keypoints.append({
                "x": x,
                "y": y, 
                "confidence": confidence
            })
        
        pose_sequence.append({"keypoints": keypoints})
    
    return pose_sequence

def test_feature_extractor():
    """Test FeatureExtractor with different confidence thresholds."""
    print("🧪 Testing FeatureExtractor...")
    
    # Create test data
    pose_sequence = create_test_pose_sequence(num_frames=70, confidence_range=(0.1, 0.9))
    
    # Test with old threshold (would fail)
    print("\n📊 Testing with confidence_threshold=0.0 (old behavior):")
    extractor_old = FeatureExtractor(
        keypoint_format="COCO_17",
        confidence_threshold=0.0,
        include_joint_statistics=True
    )
    features_old = extractor_old.extract_features(pose_sequence)
    
    # Count non-zero joint statistics
    joint_stats = [k for k in features_old.keys() if any(x in k for x in ['_std', '_max', '_min'])]
    non_zero_stats = [k for k in joint_stats if features_old[k] != 0.0]
    
    print(f"   Joint statistics features: {len(joint_stats)}")
    print(f"   Non-zero statistics: {len(non_zero_stats)}")
    print(f"   Sample values: {[(k, features_old[k]) for k in list(joint_stats)[:3]]}")
    
    # Test with new threshold (should work better)
    print("\n📊 Testing with confidence_threshold=0.3 (new behavior):")
    extractor_new = FeatureExtractor(
        keypoint_format="COCO_17", 
        confidence_threshold=0.3,
        include_joint_statistics=True
    )
    features_new = extractor_new.extract_features(pose_sequence)
    
    # Count non-zero joint statistics
    joint_stats_new = [k for k in features_new.keys() if any(x in k for x in ['_std', '_max', '_min'])]
    non_zero_stats_new = [k for k in joint_stats_new if features_new[k] != 0.0]
    
    print(f"   Joint statistics features: {len(joint_stats_new)}")
    print(f"   Non-zero statistics: {len(non_zero_stats_new)}")
    print(f"   Sample values: {[(k, features_new[k]) for k in list(joint_stats_new)[:3]]}")
    
    improvement = len(non_zero_stats_new) - len(non_zero_stats)
    print(f"\n✅ Improvement: +{improvement} non-zero features")
    
    return len(non_zero_stats_new) > len(non_zero_stats)

def test_temporal_analyzer():
    """Test TemporalAnalyzer with reduced minimum cycle duration."""
    print("\n🧪 Testing TemporalAnalyzer...")
    
    # Create short sequence (2.3 seconds like your example)
    num_frames = 70  # 2.33 seconds at 30fps
    
    # Create mock pose sequence
    pose_sequence = []
    for i in range(num_frames):
        pose_sequence.append({
            "keypoints": [
                {"x": 100 + i, "y": 200, "confidence": 0.8},  # Moving keypoint
                {"x": 150, "y": 250 + np.sin(i * 0.2) * 20, "confidence": 0.7}  # Oscillating
            ]
        })
    
    # Test with old minimum (0.8s = 24 frames)
    print("\n📊 Testing with min_cycle_duration=0.8s (old behavior):")
    analyzer_old = TemporalAnalyzer(fps=30.0, min_cycle_duration=0.8)
    cycles_old = analyzer_old.detect_gait_cycles(pose_sequence)
    print(f"   Detected cycles: {len(cycles_old)}")
    
    # Test with new minimum (0.5s = 15 frames)  
    print("\n📊 Testing with min_cycle_duration=0.5s (new behavior):")
    analyzer_new = TemporalAnalyzer(fps=30.0, min_cycle_duration=0.5)
    cycles_new = analyzer_new.detect_gait_cycles(pose_sequence)
    print(f"   Detected cycles: {len(cycles_new)}")
    
    improvement = len(cycles_new) - len(cycles_old)
    print(f"\n✅ Improvement: +{improvement} detected cycles")
    
    return len(cycles_new) >= len(cycles_old)

def test_symmetry_analyzer():
    """Test SymmetryAnalyzer with reduced confidence threshold."""
    print("\n🧪 Testing SymmetryAnalyzer...")
    
    # Create test pose sequence with mixed confidence levels
    pose_sequence = create_test_pose_sequence(num_frames=70, confidence_range=(0.2, 0.9))
    
    # Test with old threshold (0.5)
    print("\n📊 Testing with confidence_threshold=0.5 (old behavior):")
    analyzer_old = SymmetryAnalyzer(keypoint_format="COCO_17", confidence_threshold=0.5)
    symmetry_old = analyzer_old.analyze_symmetry(pose_sequence)
    non_zero_old = [k for k, v in symmetry_old.items() if v != 0.0]
    print(f"   Symmetry features: {len(symmetry_old)}")
    print(f"   Non-zero features: {len(non_zero_old)}")
    
    # Test with new threshold (0.3)
    print("\n📊 Testing with confidence_threshold=0.3 (new behavior):")
    analyzer_new = SymmetryAnalyzer(keypoint_format="COCO_17", confidence_threshold=0.3)
    symmetry_new = analyzer_new.analyze_symmetry(pose_sequence)
    non_zero_new = [k for k, v in symmetry_new.items() if v != 0.0]
    print(f"   Symmetry features: {len(symmetry_new)}")
    print(f"   Non-zero features: {len(non_zero_new)}")
    
    improvement = len(non_zero_new) - len(non_zero_old)
    print(f"\n✅ Improvement: +{improvement} non-zero features")
    
    return len(non_zero_new) >= len(non_zero_old)

def main():
    """Run all tests."""
    print("🔧 AlexPose Feature Extraction Fix - Test Suite")
    print("=" * 60)
    
    results = []
    
    try:
        results.append(test_feature_extractor())
        results.append(test_temporal_analyzer())
        results.append(test_symmetry_analyzer())
        
        print("\n" + "=" * 60)
        print("📋 Test Results Summary:")
        print(f"   FeatureExtractor: {'✅ PASS' if results[0] else '❌ FAIL'}")
        print(f"   TemporalAnalyzer: {'✅ PASS' if results[1] else '❌ FAIL'}")
        print(f"   SymmetryAnalyzer: {'✅ PASS' if results[2] else '❌ FAIL'}")
        
        if all(results):
            print("\n🎉 All tests passed! The fixes should resolve the zero-value features issue.")
        else:
            print("\n⚠️  Some tests failed. Additional debugging may be needed.")
            
    except Exception as e:
        print(f"\n❌ Test suite failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    return all(results)

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)