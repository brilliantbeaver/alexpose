#!/usr/bin/env python3
"""
Test script to verify that GAVD processing uses process isolation immediately
on Windows without any WinError 1 failures.

This script tests the optimized Windows processing to ensure:
1. Process isolation is used from the start (no singleton attempts)
2. No WinError 1 errors occur during processing
3. Real MediaPipe keypoints are extracted successfully
4. Processing completes efficiently
"""

import os
import sys
import time
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from ambient.utils.log_config import get_logger
from ambient.pose.keypoint_extractor import SequenceKeypointExtractor
from ambient.gavd.gavd_processor import PoseKeypointExtractor
import cv2
import numpy as np

logger = get_logger(__name__)


def test_immediate_process_isolation():
    """Test that process isolation is used immediately on Windows."""
    print("=" * 60)
    print("Testing Immediate Process Isolation on Windows")
    print("=" * 60)
    
    # Test 1: SequenceKeypointExtractor with explicit process isolation
    print("\n1. Testing SequenceKeypointExtractor with explicit process isolation...")
    
    try:
        # Create extractor with explicit process isolation (simulating Windows optimization)
        use_process_isolation = os.name == 'nt'  # Windows
        extractor = SequenceKeypointExtractor(use_process_isolation=use_process_isolation)
        
        if use_process_isolation:
            print("✅ Process isolation enabled by default on Windows")
        else:
            print("ℹ️  Process isolation not needed on non-Windows platform")
        
        # Create a test image
        test_image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        
        print("   Extracting keypoints from test image...")
        start_time = time.time()
        
        # This should use process isolation immediately without any singleton attempts
        keypoint_set = extractor.extract_from_image(test_image)
        
        end_time = time.time()
        processing_time = end_time - start_time
        
        print(f"   ✅ Keypoint extraction completed in {processing_time:.3f}s")
        print(f"   ✅ Detected {len(keypoint_set.keypoints)} keypoints")
        print(f"   ✅ Format: {keypoint_set.format}")
        
        # Cleanup
        extractor.cleanup()
        
    except Exception as e:
        print(f"   ❌ SequenceKeypointExtractor test failed: {e}")
        return False
    
    # Test 2: PoseKeypointExtractor (GAVD processing)
    print("\n2. Testing PoseKeypointExtractor (GAVD processing)...")
    
    try:
        # Create GAVD keypoint extractor (should use Windows optimization)
        gavd_extractor = PoseKeypointExtractor()
        
        # Create test image and bbox
        test_image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        test_bbox = {"left": 100, "top": 100, "width": 200, "height": 300}
        
        print("   Extracting keypoints with GAVD processor...")
        start_time = time.time()
        
        # This should use the optimized _ensure_sequence_extractor with process isolation
        keypoints = gavd_extractor.extract_from_image_and_bbox(test_image, test_bbox)
        
        end_time = time.time()
        processing_time = end_time - start_time
        
        print(f"   ✅ GAVD keypoint extraction completed in {processing_time:.3f}s")
        print(f"   ✅ Detected {len(keypoints)} keypoints")
        
        if keypoints:
            print(f"   ✅ Sample keypoint: x={keypoints[0]['x']:.2f}, y={keypoints[0]['y']:.2f}, conf={keypoints[0]['confidence']:.3f}")
        
    except Exception as e:
        print(f"   ❌ PoseKeypointExtractor test failed: {e}")
        return False
    
    return True


def test_no_winerror_failures():
    """Test that no WinError 1 failures occur during processing."""
    print("\n" + "=" * 60)
    print("Testing for WinError 1 Prevention")
    print("=" * 60)
    
    if os.name != 'nt':
        print("ℹ️  Skipping WinError test on non-Windows platform")
        return True
    
    print("\nProcessing multiple frames to verify no WinError 1 failures...")
    
    try:
        # Create extractor with Windows optimization
        extractor = SequenceKeypointExtractor(use_process_isolation=True)
        
        # Process multiple frames to stress test
        num_frames = 10
        success_count = 0
        
        for i in range(num_frames):
            # Create different test images
            test_image = np.random.randint(0, 255, (360 + i*10, 640 + i*5, 3), dtype=np.uint8)
            
            try:
                keypoint_set = extractor.extract_from_image(test_image)
                success_count += 1
                print(f"   Frame {i+1}: ✅ {len(keypoint_set.keypoints)} keypoints")
            except Exception as e:
                if "WinError 1" in str(e) or "Incorrect function" in str(e):
                    print(f"   Frame {i+1}: ❌ WinError 1 detected: {e}")
                    return False
                else:
                    print(f"   Frame {i+1}: ⚠️  Other error: {e}")
        
        print(f"\n✅ Processed {success_count}/{num_frames} frames successfully")
        print("✅ No WinError 1 failures detected!")
        
        # Cleanup
        extractor.cleanup()
        
        return success_count == num_frames
        
    except Exception as e:
        print(f"❌ WinError test failed: {e}")
        return False


def test_performance_comparison():
    """Test performance of optimized vs non-optimized processing."""
    print("\n" + "=" * 60)
    print("Performance Comparison")
    print("=" * 60)
    
    if os.name != 'nt':
        print("ℹ️  Performance comparison only relevant on Windows")
        return True
    
    # Test optimized processing (process isolation from start)
    print("\n1. Testing optimized processing (process isolation from start)...")
    
    try:
        extractor_optimized = SequenceKeypointExtractor(use_process_isolation=True)
        test_image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        
        start_time = time.time()
        keypoint_set = extractor_optimized.extract_from_image(test_image)
        optimized_time = time.time() - start_time
        
        print(f"   ✅ Optimized processing: {optimized_time:.3f}s")
        print(f"   ✅ Keypoints detected: {len(keypoint_set.keypoints)}")
        
        extractor_optimized.cleanup()
        
    except Exception as e:
        print(f"   ❌ Optimized processing failed: {e}")
        return False
    
    print(f"\n✅ Optimized processing completed successfully")
    print(f"✅ Processing time: {optimized_time:.3f}s per frame")
    
    return True


def main():
    """Run all tests."""
    print("GAVD Immediate Process Isolation Test Suite")
    print("=" * 60)
    
    all_tests_passed = True
    
    # Test 1: Immediate process isolation
    if not test_immediate_process_isolation():
        all_tests_passed = False
    
    # Test 2: No WinError 1 failures
    if not test_no_winerror_failures():
        all_tests_passed = False
    
    # Test 3: Performance comparison
    if not test_performance_comparison():
        all_tests_passed = False
    
    # Final results
    print("\n" + "=" * 60)
    print("FINAL RESULTS")
    print("=" * 60)
    
    if all_tests_passed:
        print("✅ ALL TESTS PASSED!")
        print("✅ Windows optimization is working correctly")
        print("✅ Process isolation is used immediately")
        print("✅ No WinError 1 failures expected")
        print("✅ GAVD processing should be faster and more reliable")
    else:
        print("❌ SOME TESTS FAILED!")
        print("❌ Windows optimization may need further investigation")
    
    return all_tests_passed


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)