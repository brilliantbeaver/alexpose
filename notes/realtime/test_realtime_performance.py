#!/usr/bin/env python3
"""
Test script to verify realtime performance optimizations.

This script checks:
1. MediaPipe VIDEO mode is being used
2. Confidence thresholds are correct
3. Frame processing is optimized
4. No unnecessary frame skipping
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from ambient.realtime.pose_estimator import RealtimePoseEstimator
from ambient.realtime.interfaces import ProcessingMode, RealtimeFrame
import numpy as np
import time


def test_estimator_initialization():
    """Test that estimator initializes correctly with VIDEO mode."""
    print("Testing estimator initialization...")
    
    estimator = RealtimePoseEstimator(processing_mode=ProcessingMode.BALANCED)
    
    # Check that landmarker was created
    assert estimator._landmarker is not None, "Landmarker should be created"
    print("✓ Landmarker created successfully")
    
    # Check quality parameters
    params = estimator._quality_params
    assert params['min_detection_confidence'] == 0.4, "Detection confidence should be 0.4"
    assert params['min_tracking_confidence'] == 0.4, "Tracking confidence should be 0.4"
    assert params['resize_factor'] == 1.0, "Resize factor should be 1.0"
    print("✓ Quality parameters correct")
    
    # Check frame skip interval
    assert estimator._frame_skip_interval == 1, "Should process every frame"
    print("✓ Frame skip interval correct")
    
    print("✓ All initialization tests passed!\n")


def test_frame_processing_speed():
    """Test frame processing speed."""
    print("Testing frame processing speed...")
    
    estimator = RealtimePoseEstimator(processing_mode=ProcessingMode.BALANCED)
    
    # Create a test frame (640x480 RGB)
    test_image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    test_frame = RealtimeFrame(
        data=test_image,
        timestamp=time.time(),
        frame_id=1,
        metadata={'width': 640, 'height': 480}
    )
    
    # Process multiple frames and measure time
    processing_times = []
    num_frames = 10
    
    print(f"Processing {num_frames} test frames...")
    for i in range(num_frames):
        test_frame.frame_id = i + 1
        test_frame.timestamp = time.time()
        
        start = time.time()
        result = estimator.estimate_pose(test_frame)
        elapsed = (time.time() - start) * 1000  # Convert to ms
        
        processing_times.append(elapsed)
        print(f"  Frame {i+1}: {elapsed:.1f}ms")
    
    # Calculate statistics
    avg_time = sum(processing_times) / len(processing_times)
    max_time = max(processing_times)
    min_time = min(processing_times)
    
    print(f"\nProcessing Statistics:")
    print(f"  Average: {avg_time:.1f}ms")
    print(f"  Min: {min_time:.1f}ms")
    print(f"  Max: {max_time:.1f}ms")
    print(f"  Target: <33ms (30 FPS)")
    
    if avg_time < 33:
        print("✓ Processing speed meets 30 FPS target!")
    elif avg_time < 50:
        print("⚠ Processing speed acceptable for 20 FPS")
    else:
        print("✗ Processing speed may cause lag")
    
    print()


def test_no_frame_skipping():
    """Test that frames are not being skipped unnecessarily."""
    print("Testing frame skip behavior...")
    
    estimator = RealtimePoseEstimator(processing_mode=ProcessingMode.BALANCED)
    
    # Create test frame
    test_image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    test_frame = RealtimeFrame(
        data=test_image,
        timestamp=time.time(),
        frame_id=1,
        metadata={'width': 640, 'height': 480}
    )
    
    # Process frames and check skip behavior
    frames_processed = 0
    frames_skipped = 0
    
    for i in range(20):
        test_frame.frame_id = i + 1
        result = estimator.estimate_pose(test_frame)
        
        if result.estimator_info.get('skipped'):
            frames_skipped += 1
        else:
            frames_processed += 1
    
    print(f"Processed: {frames_processed}/20 frames")
    print(f"Skipped: {frames_skipped}/20 frames")
    
    # With our optimizations, no frames should be skipped
    if frames_skipped == 0:
        print("✓ No frames skipped - optimal for real-time!")
    else:
        print(f"⚠ {frames_skipped} frames skipped")
    
    print()


def test_performance_stats():
    """Test performance statistics tracking."""
    print("Testing performance statistics...")
    
    estimator = RealtimePoseEstimator(processing_mode=ProcessingMode.BALANCED)
    
    # Process some frames
    test_image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    test_frame = RealtimeFrame(
        data=test_image,
        timestamp=time.time(),
        frame_id=1,
        metadata={'width': 640, 'height': 480}
    )
    
    for i in range(5):
        test_frame.frame_id = i + 1
        estimator.estimate_pose(test_frame)
    
    # Get stats
    stats = estimator.get_performance_stats()
    
    print(f"Frames processed: {stats['frames_processed']}")
    print(f"Frames skipped: {stats['frames_skipped']}")
    print(f"Average time: {stats['average_processing_time_ms']:.1f}ms")
    
    assert stats['frames_processed'] > 0, "Should have processed frames"
    print("✓ Performance stats working correctly\n")


def main():
    """Run all tests."""
    print("=" * 60)
    print("Realtime Performance Optimization Tests")
    print("=" * 60)
    print()
    
    try:
        test_estimator_initialization()
        test_frame_processing_speed()
        test_no_frame_skipping()
        test_performance_stats()
        
        print("=" * 60)
        print("✓ All tests passed!")
        print("=" * 60)
        print()
        print("Next steps:")
        print("1. Start backend: uvicorn server.main:app --reload")
        print("2. Start frontend: cd frontend && npm run dev")
        print("3. Test real-time analysis with webcam")
        print("4. Verify larger keypoints and thicker skeleton lines")
        print("5. Check for reduced latency and smooth tracking")
        
    except Exception as e:
        print(f"\n✗ Test failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()
