#!/usr/bin/env python3
"""
Real-world test scenario to demonstrate the zero-value features fix.

This script simulates the actual issue you encountered where many features
were showing as 0.00 due to strict confidence thresholds and other issues.

Author: Kiro AI Assistant
Date: January 27, 2026
"""

import sys
import numpy as np
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from ambient.analysis.gait_analyzer import EnhancedGaitAnalyzer
from ambient.classification.features import GaitFeatureVector
from dataclasses import asdict

def create_realistic_pose_sequence():
    """Create a realistic pose sequence that mimics your actual data."""
    pose_sequence = []
    
    # Simulate 70 frames (2.33 seconds at 30fps) like your example
    for frame in range(70):
        keypoints = []
        
        # COCO_17 format: 17 keypoints
        for kp_idx in range(17):
            # Simulate realistic walking motion
            base_x = 200 + np.sin(frame * 0.15) * 30  # Side-to-side sway
            base_y = 300 + np.cos(frame * 0.1) * 20   # Up-down motion
            
            # Add keypoint-specific offsets
            if kp_idx in [11, 12]:  # Hips
                y_offset = -50
            elif kp_idx in [13, 14]:  # Knees  
                y_offset = 0
                x_offset = np.sin(frame * 0.2) * 15  # Knee movement
                base_x += x_offset
            elif kp_idx in [15, 16]:  # Ankles
                y_offset = 50
                x_offset = np.sin(frame * 0.25) * 25  # Ankle movement
                base_x += x_offset
            else:
                y_offset = np.random.uniform(-10, 10)
            
            x = base_x + np.random.uniform(-5, 5)
            y = base_y + y_offset + np.random.uniform(-5, 5)
            
            # Simulate realistic confidence issues:
            # - Some keypoints have very low confidence (causing 0.00 features)
            # - Some frames have missing keypoints (confidence = 0)
            if frame < 10 or frame > 60:  # Start/end frames often have issues
                confidence = np.random.uniform(0.0, 0.4)  # Low confidence
            elif kp_idx in [15, 16] and frame % 8 == 0:  # Ankle occlusion
                confidence = 0.0  # Missing keypoint
            else:
                confidence = np.random.uniform(0.6, 0.95)  # Good confidence
            
            keypoints.append({
                "x": x,
                "y": y,
                "confidence": confidence
            })
        
        pose_sequence.append({"keypoints": keypoints})
    
    return pose_sequence

def test_before_and_after_fix():
    """Test the complete pipeline before and after the fix."""
    print("🧪 Real-World Scenario Test")
    print("=" * 50)
    
    # Create realistic test data
    pose_sequence = create_realistic_pose_sequence()
    print(f"📊 Created pose sequence with {len(pose_sequence)} frames")
    
    # Analyze confidence distribution
    all_confidences = []
    zero_count = 0
    low_count = 0
    
    for pose in pose_sequence:
        for kp in pose["keypoints"]:
            conf = kp["confidence"]
            all_confidences.append(conf)
            if conf == 0.0:
                zero_count += 1
            elif conf < 0.3:
                low_count += 1
    
    print(f"📈 Confidence distribution:")
    print(f"   Total keypoints: {len(all_confidences)}")
    print(f"   Zero confidence: {zero_count} ({zero_count/len(all_confidences)*100:.1f}%)")
    print(f"   Low confidence (<0.3): {low_count} ({low_count/len(all_confidences)*100:.1f}%)")
    print(f"   Mean confidence: {np.mean(all_confidences):.3f}")
    
    # Test with old configuration (simulating the issue)
    print("\n🔧 Testing with OLD configuration (strict thresholds):")
    analyzer_old = EnhancedGaitAnalyzer(
        keypoint_format="COCO_17",
        fps=30.0,
        comprehensive_features=True,
        feature_extraction_config={
            "confidence_threshold": 0.0,  # Old: any confidence > 0
            "include_joint_statistics": True
        }
    )
    
    # Manually set temporal analyzer to old settings
    analyzer_old.temporal_analyzer.min_cycle_duration = 0.8
    analyzer_old.temporal_analyzer.min_cycle_frames = int(0.8 * 30)
    
    # Manually set symmetry analyzer to old settings  
    analyzer_old.symmetry_analyzer.confidence_threshold = 0.5
    
    results_old = analyzer_old.analyze_gait_sequence(pose_sequence)
    
    # Extract features using old method
    features_old = GaitFeatureVector.from_analysis_results(
        results_old, 
        sample_id="test_old", 
        condition_label="normal",
        feature_extraction_mode="comprehensive"
    )
    
    if features_old:
        old_dict = asdict(features_old)
        zero_features_old = [k for k, v in old_dict.items() if v == 0.0]
        non_zero_old = len(old_dict) - len(zero_features_old)
        
        print(f"   Total features: {len(old_dict)}")
        print(f"   Zero-value features: {len(zero_features_old)}")
        print(f"   Non-zero features: {non_zero_old}")
        print(f"   Zero-value percentage: {len(zero_features_old)/len(old_dict)*100:.1f}%")
        
        # Show some problematic features
        joint_stats_zero = [k for k in zero_features_old if any(x in k for x in ['_std', '_max', '_min'])]
        temporal_zero = [k for k in zero_features_old if any(x in k for x in ['cycle_', 'phase_', 'stance_time_si'])]
        symmetry_zero = [k for k in zero_features_old if 'symmetry' in k and k in zero_features_old]
        
        print(f"   Joint statistics (std/max/min) = 0.00: {len(joint_stats_zero)}")
        print(f"   Temporal features = 0.00: {len(temporal_zero)}")
        print(f"   Symmetry features = 0.00: {len(symmetry_zero)}")
    else:
        print("   ❌ Feature extraction failed!")
        return False
    
    # Test with new configuration (after the fix)
    print("\n🔧 Testing with NEW configuration (relaxed thresholds):")
    analyzer_new = EnhancedGaitAnalyzer(
        keypoint_format="COCO_17",
        fps=30.0,
        comprehensive_features=True,
        feature_extraction_config={
            "confidence_threshold": 0.3,  # New: reasonable threshold
            "include_joint_statistics": True
        }
    )
    
    # New settings are already applied via the fixes
    results_new = analyzer_new.analyze_gait_sequence(pose_sequence)
    
    # Extract features using new method
    features_new = GaitFeatureVector.from_analysis_results(
        results_new,
        sample_id="test_new", 
        condition_label="normal",
        feature_extraction_mode="comprehensive"
    )
    
    if features_new:
        new_dict = asdict(features_new)
        zero_features_new = [k for k, v in new_dict.items() if v == 0.0]
        non_zero_new = len(new_dict) - len(zero_features_new)
        
        print(f"   Total features: {len(new_dict)}")
        print(f"   Zero-value features: {len(zero_features_new)}")
        print(f"   Non-zero features: {non_zero_new}")
        print(f"   Zero-value percentage: {len(zero_features_new)/len(new_dict)*100:.1f}%")
        
        # Show improvements
        joint_stats_zero_new = [k for k in zero_features_new if any(x in k for x in ['_std', '_max', '_min'])]
        temporal_zero_new = [k for k in zero_features_new if any(x in k for x in ['cycle_', 'phase_', 'stance_time_si'])]
        symmetry_zero_new = [k for k in zero_features_new if 'symmetry' in k and k in zero_features_new]
        
        print(f"   Joint statistics (std/max/min) = 0.00: {len(joint_stats_zero_new)}")
        print(f"   Temporal features = 0.00: {len(temporal_zero_new)}")
        print(f"   Symmetry features = 0.00: {len(symmetry_zero_new)}")
    else:
        print("   ❌ Feature extraction failed!")
        return False
    
    # Calculate improvements
    print("\n📈 IMPROVEMENT SUMMARY:")
    print("=" * 50)
    
    total_improvement = non_zero_new - non_zero_old
    zero_reduction = len(zero_features_old) - len(zero_features_new)
    
    print(f"✅ Non-zero features: {non_zero_old} → {non_zero_new} (+{total_improvement})")
    print(f"✅ Zero-value features: {len(zero_features_old)} → {len(zero_features_new)} (-{zero_reduction})")
    print(f"✅ Zero-value percentage: {len(zero_features_old)/len(old_dict)*100:.1f}% → {len(zero_features_new)/len(new_dict)*100:.1f}%")
    
    # Specific improvements
    joint_improvement = len(joint_stats_zero) - len(joint_stats_zero_new)
    temporal_improvement = len(temporal_zero) - len(temporal_zero_new)
    symmetry_improvement = len(symmetry_zero) - len(symmetry_zero_new)
    
    print(f"✅ Joint statistics fixed: {joint_improvement}")
    print(f"✅ Temporal features fixed: {temporal_improvement}")
    print(f"✅ Symmetry features fixed: {symmetry_improvement}")
    
    if total_improvement > 0:
        print(f"\n🎉 SUCCESS! The fixes resolved {total_improvement} zero-value features!")
        print("   Your feature vector should now have proper values instead of 0.00")
        return True
    else:
        print(f"\n⚠️  Limited improvement detected. May need additional debugging.")
        return False

def main():
    """Run the real-world test scenario."""
    try:
        success = test_before_and_after_fix()
        return success
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)