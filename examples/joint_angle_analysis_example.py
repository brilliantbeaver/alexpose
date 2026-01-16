"""
Joint Angle Analysis Example

This example demonstrates how to use the joint angle calculation module
to analyze gait patterns from pose estimation data.

Author: AlexPose Team
"""

import numpy as np
import matplotlib.pyplot as plt
from ambient.pose.joint_angles import get_joint_angles


def create_sample_gait_sequence(num_frames=60):
    """
    Create a sample gait sequence for demonstration.
    
    This simulates a person walking with realistic joint movements.
    """
    keypoints_sequence = []
    
    for frame_idx in range(num_frames):
        # Create 33 keypoints (MediaPipe format)
        keypoints = []
        for i in range(33):
            keypoints.append({
                "x": 500.0,
                "y": 400.0,
                "confidence": 0.5
            })
        
        # Simulate gait cycle (2 seconds = 2 complete cycles at 30 FPS)
        phase = (frame_idx / num_frames) * 4 * np.pi
        
        # Left leg
        left_hip_y = 400.0
        left_knee_y = 600.0 + 50 * np.sin(phase)
        left_ankle_y = 800.0 + 100 * np.sin(phase)
        left_foot_y = 850.0 + 100 * np.sin(phase)
        
        keypoints[11] = {"x": 500.0, "y": 200.0, "confidence": 0.95}  # Left shoulder
        keypoints[23] = {"x": 500.0, "y": left_hip_y, "confidence": 0.95}  # Left hip
        keypoints[25] = {"x": 490.0, "y": left_knee_y, "confidence": 0.95}  # Left knee
        keypoints[27] = {"x": 480.0, "y": left_ankle_y, "confidence": 0.95}  # Left ankle
        keypoints[31] = {"x": 470.0, "y": left_foot_y, "confidence": 0.90}  # Left foot
        
        # Right leg (opposite phase)
        right_phase = phase + np.pi
        right_hip_y = 400.0
        right_knee_y = 600.0 + 50 * np.sin(right_phase)
        right_ankle_y = 800.0 + 100 * np.sin(right_phase)
        right_foot_y = 850.0 + 100 * np.sin(right_phase)
        
        keypoints[12] = {"x": 600.0, "y": 200.0, "confidence": 0.95}  # Right shoulder
        keypoints[24] = {"x": 600.0, "y": right_hip_y, "confidence": 0.95}  # Right hip
        keypoints[26] = {"x": 610.0, "y": right_knee_y, "confidence": 0.95}  # Right knee
        keypoints[28] = {"x": 620.0, "y": right_ankle_y, "confidence": 0.95}  # Right ankle
        keypoints[32] = {"x": 630.0, "y": right_foot_y, "confidence": 0.90}  # Right foot
        
        keypoints_sequence.append(keypoints)
    
    return keypoints_sequence


def example_basic_usage():
    """Example 1: Basic joint angle calculation."""
    print("=" * 60)
    print("Example 1: Basic Joint Angle Calculation")
    print("=" * 60)
    
    # Create sample data
    keypoints_sequence = create_sample_gait_sequence(num_frames=60)
    
    # Calculate joint angles
    angles = get_joint_angles(
        keypoints_sequence,
        keypoint_format="BLAZEPOSE_33",
        fps=30.0,
        confidence_threshold=0.3
    )
    
    print(f"\nProcessed {len(angles.frames)} frames")
    print(f"Keypoint format: {angles.keypoint_format}")
    print(f"FPS: {angles.fps}")
    
    # Show angles for first frame
    first_frame = angles.frames[0]
    print(f"\nFrame 0 angles:")
    for joint_name, angle_obj in first_frame.angles.items():
        print(f"  {joint_name}: {angle_obj.angle_degrees:.1f}° "
              f"(confidence: {angle_obj.confidence:.2f})")


def example_time_series_analysis():
    """Example 2: Time series analysis of joint angles."""
    print("\n" + "=" * 60)
    print("Example 2: Time Series Analysis")
    print("=" * 60)
    
    keypoints_sequence = create_sample_gait_sequence(num_frames=60)
    angles = get_joint_angles(keypoints_sequence)
    
    # Get angle series
    left_knee_series = angles.get_joint_angle_series("left_knee")
    right_knee_series = angles.get_joint_angle_series("right_knee")
    
    print(f"\nLeft knee angles: {len(left_knee_series)} frames")
    print(f"Right knee angles: {len(right_knee_series)} frames")
    
    # Calculate statistics
    left_stats = angles.get_statistics("left_knee")
    right_stats = angles.get_statistics("right_knee")
    
    print(f"\nLeft Knee Statistics:")
    print(f"  Mean: {left_stats['mean']:.1f}°")
    print(f"  Std Dev: {left_stats['std']:.1f}°")
    print(f"  Range: {left_stats['range']:.1f}°")
    print(f"  Min: {left_stats['min']:.1f}°")
    print(f"  Max: {left_stats['max']:.1f}°")
    
    print(f"\nRight Knee Statistics:")
    print(f"  Mean: {right_stats['mean']:.1f}°")
    print(f"  Std Dev: {right_stats['std']:.1f}°")
    print(f"  Range: {right_stats['range']:.1f}°")


def example_bilateral_comparison():
    """Example 3: Bilateral (left-right) comparison."""
    print("\n" + "=" * 60)
    print("Example 3: Bilateral Comparison")
    print("=" * 60)
    
    keypoints_sequence = create_sample_gait_sequence(num_frames=60)
    angles = get_joint_angles(keypoints_sequence)
    
    # Compare left and right sides
    joints = ["hip", "knee", "ankle"]
    
    print("\nBilateral Symmetry Analysis:")
    for joint in joints:
        left_name = f"left_{joint}"
        right_name = f"right_{joint}"
        
        left_stats = angles.get_statistics(left_name)
        right_stats = angles.get_statistics(right_name)
        
        if left_stats["valid_count"] > 0 and right_stats["valid_count"] > 0:
            asymmetry = abs(left_stats["mean"] - right_stats["mean"])
            
            print(f"\n{joint.upper()}:")
            print(f"  Left mean: {left_stats['mean']:.1f}°")
            print(f"  Right mean: {right_stats['mean']:.1f}°")
            print(f"  Asymmetry: {asymmetry:.1f}°")
            
            if asymmetry < 5:
                print(f"  Status: ✓ Symmetric")
            elif asymmetry < 10:
                print(f"  Status: ⚠ Mild asymmetry")
            else:
                print(f"  Status: ⚠ Significant asymmetry")


def example_visualization():
    """Example 4: Visualizing joint angles over time."""
    print("\n" + "=" * 60)
    print("Example 4: Visualization")
    print("=" * 60)
    
    keypoints_sequence = create_sample_gait_sequence(num_frames=60)
    angles = get_joint_angles(keypoints_sequence)
    
    # Get angle series
    left_knee = angles.get_joint_angle_series("left_knee")
    right_knee = angles.get_joint_angle_series("right_knee")
    left_hip = angles.get_joint_angle_series("left_hip")
    right_hip = angles.get_joint_angle_series("right_hip")
    
    # Create time axis
    time = np.arange(len(left_knee)) / 30.0  # Convert frames to seconds
    
    # Create plot
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
    
    # Plot knee angles
    ax1.plot(time, left_knee, 'b-', label='Left Knee', linewidth=2)
    ax1.plot(time, right_knee, 'r-', label='Right Knee', linewidth=2)
    ax1.set_xlabel('Time (seconds)')
    ax1.set_ylabel('Angle (degrees)')
    ax1.set_title('Knee Angles During Gait')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot hip angles
    ax2.plot(time, left_hip, 'b-', label='Left Hip', linewidth=2)
    ax2.plot(time, right_hip, 'r-', label='Right Hip', linewidth=2)
    ax2.set_xlabel('Time (seconds)')
    ax2.set_ylabel('Angle (degrees)')
    ax2.set_title('Hip Angles During Gait')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('joint_angles_visualization.png', dpi=150)
    print("\nVisualization saved to: joint_angles_visualization.png")
    
    # Optionally show the plot
    # plt.show()


def example_clinical_assessment():
    """Example 5: Clinical assessment based on joint angles."""
    print("\n" + "=" * 60)
    print("Example 5: Clinical Assessment")
    print("=" * 60)
    
    keypoints_sequence = create_sample_gait_sequence(num_frames=60)
    angles = get_joint_angles(keypoints_sequence)
    
    print("\nClinical Gait Assessment:")
    
    # 1. Range of Motion Assessment
    print("\n1. Range of Motion:")
    for joint in ["left_knee", "right_knee"]:
        stats = angles.get_statistics(joint)
        if stats["valid_count"] > 0:
            normal_rom = (0, 140)  # Normal knee ROM
            
            print(f"\n  {joint.replace('_', ' ').title()}:")
            print(f"    ROM: {stats['range']:.1f}°")
            
            if stats['range'] < 100:
                print(f"    ⚠ Reduced range of motion")
            elif stats['max'] > 150:
                print(f"    ⚠ Excessive flexion")
            else:
                print(f"    ✓ Normal range of motion")
    
    # 2. Bilateral Symmetry
    print("\n2. Bilateral Symmetry:")
    left_knee_stats = angles.get_statistics("left_knee")
    right_knee_stats = angles.get_statistics("right_knee")
    
    if left_knee_stats["valid_count"] > 0 and right_knee_stats["valid_count"] > 0:
        asymmetry = abs(left_knee_stats["mean"] - right_knee_stats["mean"])
        print(f"  Knee asymmetry: {asymmetry:.1f}°")
        
        if asymmetry < 5:
            print(f"  ✓ Symmetric gait pattern")
        else:
            print(f"  ⚠ Asymmetric gait pattern detected")
    
    # 3. Movement Smoothness
    print("\n3. Movement Smoothness:")
    left_knee_series = angles.get_joint_angle_series("left_knee")
    valid_angles = left_knee_series[~np.isnan(left_knee_series)]
    
    if len(valid_angles) > 5:
        changes = np.abs(np.diff(valid_angles))
        max_change = np.max(changes)
        mean_change = np.mean(changes)
        
        print(f"  Max frame-to-frame change: {max_change:.1f}°")
        print(f"  Mean frame-to-frame change: {mean_change:.1f}°")
        
        if max_change > 30:
            print(f"  ⚠ Jerky movement detected")
        elif mean_change < 2:
            print(f"  ⚠ Reduced movement variability")
        else:
            print(f"  ✓ Normal movement smoothness")


def main():
    """Run all examples."""
    print("\n" + "=" * 60)
    print("Joint Angle Analysis Examples")
    print("AlexPose Gait Analysis System")
    print("=" * 60)
    
    # Run examples
    example_basic_usage()
    example_time_series_analysis()
    example_bilateral_comparison()
    example_visualization()
    example_clinical_assessment()
    
    print("\n" + "=" * 60)
    print("Examples completed successfully!")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()
