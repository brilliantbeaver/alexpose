#!/usr/bin/env python3
"""
Enhanced Gait Analysis Example

This example demonstrates the enhanced gait analysis capabilities including:
- Feature extraction with FeatureExtractor
- Temporal analysis with TemporalAnalyzer  
- Symmetry analysis with SymmetryAnalyzer
- LLM-based classification with LLMClassifier
- Integration with Frame objects

Author: AlexPose Team
"""

import sys
import os
from pathlib import Path

# Add the project root to the Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import numpy as np
from typing import Dict, Any, List

from ambient.analysis.gait_analyzer import EnhancedGaitAnalyzer
from ambient.classification.llm_classifier import LLMClassifier
from ambient.core.frame import Frame, FrameSequence
from ambient.core.data_models import Keypoint


def create_sample_pose_data() -> List[Dict[str, Any]]:
    """Create sample pose data for testing."""
    pose_sequence = []
    
    # Generate 100 frames of synthetic gait data
    for frame_idx in range(100):
        # Simulate walking motion with some asymmetry
        t = frame_idx / 30.0  # Time in seconds (30 FPS)
        
        # Basic walking pattern with left-right alternation
        left_phase = np.sin(2 * np.pi * 1.2 * t)  # 1.2 Hz step frequency
        right_phase = np.sin(2 * np.pi * 1.2 * t + np.pi)  # 180 degrees out of phase
        
        # Add some asymmetry to make it interesting
        asymmetry_factor = 0.8 if frame_idx % 40 < 20 else 1.0
        
        keypoints = [
            {"x": 320, "y": 100, "confidence": 0.9},  # nose
            {"x": 310, "y": 110, "confidence": 0.8},  # left_eye
            {"x": 330, "y": 110, "confidence": 0.8},  # right_eye
            {"x": 300, "y": 120, "confidence": 0.7},  # left_ear
            {"x": 340, "y": 120, "confidence": 0.7},  # right_ear
            {"x": 280, "y": 200, "confidence": 0.9},  # left_shoulder
            {"x": 360, "y": 200, "confidence": 0.9},  # right_shoulder
            {"x": 260, "y": 250, "confidence": 0.8},  # left_elbow
            {"x": 380, "y": 250, "confidence": 0.8},  # right_elbow
            {"x": 240, "y": 300, "confidence": 0.7},  # left_wrist
            {"x": 400, "y": 300, "confidence": 0.7},  # right_wrist
            {"x": 300, "y": 350, "confidence": 0.9},  # left_hip
            {"x": 340, "y": 350, "confidence": 0.9},  # right_hip
            # Left knee with walking motion
            {"x": 290 + 10 * left_phase, "y": 450 + 20 * abs(left_phase), "confidence": 0.9},
            # Right knee with walking motion and asymmetry
            {"x": 350 + 10 * right_phase * asymmetry_factor, "y": 450 + 20 * abs(right_phase), "confidence": 0.9},
            # Left ankle
            {"x": 285 + 15 * left_phase, "y": 550 + 30 * abs(left_phase), "confidence": 0.8},
            # Right ankle with asymmetry
            {"x": 355 + 15 * right_phase * asymmetry_factor, "y": 550 + 30 * abs(right_phase), "confidence": 0.8}
        ]
        
        pose_data = {
            "keypoints": keypoints,
            "timestamp": t,
            "frame_id": frame_idx
        }
        
        pose_sequence.append(pose_data)
    
    return pose_sequence


def create_sample_frame_sequence() -> FrameSequence:
    """Create sample Frame sequence for testing."""
    frames = []
    pose_data = create_sample_pose_data()
    
    for i, pose in enumerate(pose_data):
        # Create a simple synthetic image (100x100 RGB)
        synthetic_image = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        
        # Create Frame object with metadata containing pose data
        metadata = {
            "frame_id": i,
            "timestamp": pose["timestamp"],
            "pose_data": pose,
            "keypoints": pose["keypoints"]
        }
        
        frame = Frame(
            data=synthetic_image,
            source_type="array",
            metadata=metadata,
            format="RGB"
        )
        
        frames.append(frame)
    
    return FrameSequence(frames)


def demonstrate_enhanced_analysis():
    """Demonstrate enhanced gait analysis capabilities."""
    print("Enhanced Gait Analysis Demonstration")
    print("=" * 50)
    
    # Create sample data
    print("\n1. Creating sample gait data...")
    pose_sequence = create_sample_pose_data()
    frame_sequence = create_sample_frame_sequence()
    
    print(f"   - Generated {len(pose_sequence)} frames of pose data")
    print(f"   - Created FrameSequence with {len(frame_sequence.frames)} frames")
    
    # Initialize enhanced gait analyzer
    print("\n2. Initializing Enhanced Gait Analyzer...")
    analyzer = EnhancedGaitAnalyzer(
        keypoint_format="COCO_17",
        fps=30.0
    )
    
    # Perform comprehensive analysis
    print("\n3. Performing comprehensive gait analysis...")
    
    # Test with pose sequence
    print("   a) Analyzing pose sequence...")
    results_pose = analyzer.analyze_gait_sequence(
        pose_sequence,
        metadata={"source": "synthetic_data", "subject_id": "test_001"}
    )
    
    # Test with Frame sequence
    print("   b) Analyzing Frame sequence...")
    results_frame = analyzer.analyze_frame_sequence(
        frame_sequence,
        metadata={"source": "synthetic_frames", "subject_id": "test_001"}
    )
    
    # Display results
    print("\n4. Analysis Results:")
    print("-" * 30)
    
    def display_results(results: Dict[str, Any], title: str):
        print(f"\n{title}:")
        
        # Sequence info
        seq_info = results.get("sequence_info", {})
        print(f"  Duration: {seq_info.get('duration_seconds', 'N/A'):.2f} seconds")
        print(f"  Frames: {seq_info.get('num_frames', 'N/A')}")
        
        # Features summary
        features = results.get("features", {})
        if features:
            print(f"  Velocity mean: {features.get('velocity_mean', 'N/A'):.3f}")
            print(f"  Velocity std: {features.get('velocity_std', 'N/A'):.3f}")
            print(f"  Acceleration mean: {features.get('acceleration_mean', 'N/A'):.3f}")
        
        # Gait cycles
        cycles = results.get("gait_cycles", [])
        print(f"  Detected gait cycles: {len(cycles)}")
        
        # Timing analysis
        timing = results.get("timing_analysis", {})
        if timing:
            print(f"  Cadence: {timing.get('cadence_steps_per_minute', 'N/A'):.1f} steps/min")
            print(f"  Cycle duration mean: {timing.get('cycle_duration_mean', 'N/A'):.3f} sec")
        
        # Symmetry analysis
        symmetry = results.get("symmetry_analysis", {})
        if symmetry:
            print(f"  Overall symmetry index: {symmetry.get('overall_symmetry_index', 'N/A'):.3f}")
            print(f"  Symmetry classification: {symmetry.get('symmetry_classification', 'N/A')}")
        
        # Summary assessment
        summary = results.get("summary", {})
        if summary and "overall_assessment" in summary:
            overall = summary["overall_assessment"]
            print(f"  Overall assessment: {overall.get('overall_level', 'N/A')}")
            print(f"  Confidence: {overall.get('confidence', 'N/A')}")
    
    display_results(results_pose, "Pose Sequence Analysis")
    display_results(results_frame, "Frame Sequence Analysis")
    
    # Test LLM Classification (if available)
    print("\n5. Testing LLM Classification...")
    try:
        classifier = LLMClassifier(default_model="gpt-4o-mini")
        available_models = classifier.get_available_models()
        
        if available_models:
            print(f"   Available models: {', '.join(available_models)}")
            
            # Classify the gait data
            print("   Performing classification...")
            classification_result = classifier.classify_gait(results_pose)
            
            print(f"   Classification: {classification_result.classification}")
            print(f"   Confidence: {classification_result.confidence:.3f}")
            print(f"   Model used: {classification_result.model_used}")
            print(f"   Processing time: {classification_result.processing_time:.3f}s")
            print(f"   Explanation: {classification_result.explanation[:200]}...")
            
        else:
            print("   No LLM models available (API keys not configured)")
            
    except Exception as e:
        print(f"   LLM classification failed: {e}")
        print("   This is expected if API keys are not configured")
    
    print("\n6. Testing individual analysis components...")
    
    # Test feature extraction
    print("   a) Feature extraction...")
    features = analyzer.extract_gait_features(pose_sequence)
    print(f"      Extracted {len(features)} features")
    
    # Test cycle detection
    print("   b) Gait cycle detection...")
    cycles = analyzer.detect_gait_cycles(pose_sequence)
    print(f"      Detected {len(cycles)} gait cycles")
    
    # Test Frame-based methods
    print("   c) Frame-based feature extraction...")
    frame_features = analyzer.extract_features_from_frames(frame_sequence)
    print(f"      Extracted {len(frame_features)} features from frames")
    
    print("\n✅ Enhanced gait analysis demonstration completed successfully!")
    print("\nKey capabilities demonstrated:")
    print("- ✅ Enhanced feature extraction (kinematic, temporal, symmetry)")
    print("- ✅ Advanced gait cycle detection")
    print("- ✅ Comprehensive symmetry analysis")
    print("- ✅ Frame object support")
    print("- ✅ LLM-based classification (if API keys available)")
    print("- ✅ Backward compatibility with existing pose data")


if __name__ == "__main__":
    demonstrate_enhanced_analysis()