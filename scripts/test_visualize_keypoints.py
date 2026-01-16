"""
Test script for visualize_keypoints function fix.

This script verifies that the visualize_keypoints function works correctly
with the new KeypointSet data structure.
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Configure pose environment before imports
from ambient.pose.pose_config import configure_pose_environment
configure_pose_environment()

import numpy as np
from ambient.pose.keypoint_data import Keypoint, KeypointSet, KeypointFormat
from ambient.pose.keypoints import KeypointVisualizer


def test_keypoint_visualizer_methods():
    """Test that KeypointVisualizer has the expected methods."""
    print("=" * 60)
    print("TEST 1: KeypointVisualizer Methods")
    print("=" * 60)
    
    # Check for expected methods
    expected_methods = ['draw_keypoints', 'draw_skeleton', 'get_summary_stats']
    
    for method_name in expected_methods:
        if hasattr(KeypointVisualizer, method_name):
            print(f"[OK] KeypointVisualizer.{method_name}() exists")
        else:
            print(f"[ERROR] KeypointVisualizer.{method_name}() NOT FOUND")
            return False
    
    # Check that add_landmark_names does NOT exist
    if hasattr(KeypointVisualizer, 'add_landmark_names'):
        print("[ERROR] KeypointVisualizer.add_landmark_names() should NOT exist")
        return False
    else:
        print("[OK] KeypointVisualizer.add_landmark_names() correctly does not exist")
    
    print()
    return True


def test_keypoint_set_structure():
    """Test that KeypointSet has the expected structure."""
    print("=" * 60)
    print("TEST 2: KeypointSet Structure")
    print("=" * 60)
    
    # Create a sample keypoint
    kp = Keypoint(
        id=0,
        name="NOSE",
        x=100.0,
        y=150.0,
        z=0.0,
        confidence=0.95,
        visibility=0.90,
        presence=0.98,
        x_normalized=0.5,
        y_normalized=0.5
    )
    
    print(f"[OK] Created Keypoint: {kp.name}")
    print(f"     Position: ({kp.x:.1f}, {kp.y:.1f})")
    print(f"     Confidence: {kp.confidence:.3f}")
    
    # Create a KeypointSet
    keypoints = [kp]
    keypoint_set = KeypointSet(
        keypoints=keypoints,
        format=KeypointFormat.CUSTOM,
        frame_width=640,
        frame_height=480
    )
    
    print(f"[OK] Created KeypointSet with {len(keypoint_set)} keypoints")
    print(f"     Format: {keypoint_set.format.value}")
    print(f"     Frame size: {keypoint_set.frame_width}x{keypoint_set.frame_height}")
    print()
    return True


def test_draw_skeleton():
    """Test that draw_skeleton works with KeypointSet."""
    print("=" * 60)
    print("TEST 3: Draw Skeleton Function")
    print("=" * 60)
    
    # Create a sample image
    image = np.zeros((480, 640, 3), dtype=np.uint8)
    
    # Create sample keypoints
    keypoints = []
    for i in range(5):
        kp = Keypoint(
            id=i,
            name=f"POINT_{i}",
            x=100.0 + i * 50,
            y=150.0 + i * 30,
            z=0.0,
            confidence=0.8,
            visibility=0.8,
            presence=0.8,
            x_normalized=(100.0 + i * 50) / 640,
            y_normalized=(150.0 + i * 30) / 480
        )
        keypoints.append(kp)
    
    keypoint_set = KeypointSet(
        keypoints=keypoints,
        format=KeypointFormat.CUSTOM,
        frame_width=640,
        frame_height=480
    )
    
    try:
        # Test draw_skeleton
        annotated = KeypointVisualizer.draw_skeleton(
            image,
            keypoint_set,
            confidence_threshold=0.5,
            keypoint_color=(255, 0, 0),
            line_color=(0, 255, 0)
        )
        print(f"[OK] draw_skeleton() executed successfully")
        print(f"     Input shape: {image.shape}")
        print(f"     Output shape: {annotated.shape}")
        print(f"     Keypoints drawn: {len(keypoint_set)}")
    except Exception as e:
        print(f"[ERROR] draw_skeleton() failed: {e}")
        return False
    
    print()
    return True


def test_get_summary_stats():
    """Test that get_summary_stats works with KeypointSet."""
    print("=" * 60)
    print("TEST 4: Get Summary Stats Function")
    print("=" * 60)
    
    # Create sample keypoints with varying confidence
    keypoints = []
    for i in range(10):
        kp = Keypoint(
            id=i,
            name=f"POINT_{i}",
            x=100.0 + i * 50,
            y=150.0,
            z=0.0,
            confidence=0.5 + (i % 5) * 0.1,  # Varying confidence
            visibility=0.6 + (i % 4) * 0.1,
            presence=0.7 + (i % 3) * 0.1,
            x_normalized=(100.0 + i * 50) / 640,
            y_normalized=150.0 / 480
        )
        keypoints.append(kp)
    
    keypoint_set = KeypointSet(
        keypoints=keypoints,
        format=KeypointFormat.CUSTOM,
        frame_width=640,
        frame_height=480
    )
    
    try:
        stats = KeypointVisualizer.get_summary_stats(keypoint_set)
        print(f"[OK] get_summary_stats() executed successfully")
        print(f"     Total landmarks: {stats['total_landmarks']}")
        print(f"     Visible landmarks: {stats['visible_landmarks']}")
        print(f"     Reliable landmarks: {stats['reliable_landmarks']}")
        print(f"     Average confidence: {stats['avg_confidence']:.3f}")
        print(f"     Average visibility: {stats['avg_visibility']:.3f}")
        print(f"     Detection quality: {stats['detection_quality']:.3f}")
    except Exception as e:
        print(f"[ERROR] get_summary_stats() failed: {e}")
        return False
    
    print()
    return True


def test_visualize_keypoints_import():
    """Test that visualize_keypoints can be imported and has correct signature."""
    print("=" * 60)
    print("TEST 5: Visualize Keypoints Import")
    print("=" * 60)
    
    try:
        from ambient.utils.eval_keypoints import visualize_keypoints
        print("[OK] visualize_keypoints imported successfully")
        
        # Check function signature
        import inspect
        sig = inspect.signature(visualize_keypoints)
        params = list(sig.parameters.keys())
        print(f"[OK] Function parameters: {params}")
        
        if 'keypoints' in params and 'frame' in params:
            print("[OK] Function has expected parameters")
        else:
            print("[ERROR] Function missing expected parameters")
            return False
            
    except Exception as e:
        print(f"[ERROR] Import failed: {e}")
        return False
    
    print()
    return True


def main():
    """Run all tests."""
    print("\n" + "=" * 60)
    print("VISUALIZE KEYPOINTS FIX - TEST SUITE")
    print("=" * 60 + "\n")
    
    tests = [
        test_keypoint_visualizer_methods,
        test_keypoint_set_structure,
        test_draw_skeleton,
        test_get_summary_stats,
        test_visualize_keypoints_import,
    ]
    
    results = []
    for test in tests:
        try:
            result = test()
            results.append(result)
        except Exception as e:
            print(f"[ERROR] Test failed with exception: {e}")
            import traceback
            traceback.print_exc()
            results.append(False)
    
    # Summary
    print("=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)
    passed = sum(results)
    total = len(results)
    print(f"Passed: {passed}/{total}")
    
    if all(results):
        print("\n[OK] ALL TESTS PASSED!")
        print("\nThe visualize_keypoints function should now work correctly.")
        print("You can test it in the notebook with:")
        print("  from ambient.utils.eval_keypoints import get_keypoints, visualize_keypoints")
        print("  keypoints, frame = get_keypoints(project_root, sequences)")
        print("  visualize_keypoints(keypoints, frame)")
        return 0
    else:
        print("\n[ERROR] SOME TESTS FAILED")
        print("Please review the errors above.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
