#!/usr/bin/env python3
"""
Test script to verify MediaPipe memory leak fix on Windows.

This script tests the singleton pattern implementation to ensure MediaPipe
landmarkers don't cause memory leaks and WinError 1 issues on Windows.
"""

import sys
import time
import gc
import psutil
import os
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from ambient.pose.keypoint_extractor import SequenceKeypointExtractor
from ambient.pose.mediapipe_singleton import get_mediapipe_singleton


def get_memory_usage():
    """Get current memory usage in MB."""
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / 1024 / 1024


def test_singleton_pattern():
    """Test that the singleton pattern works correctly."""
    print("🔄 Testing MediaPipe Singleton Pattern")
    print("=" * 50)
    
    # Get singleton instances
    singleton1 = get_mediapipe_singleton()
    singleton2 = get_mediapipe_singleton()
    
    # Verify they are the same instance
    if singleton1 is singleton2:
        print("✅ Singleton pattern working: Same instance returned")
    else:
        print("❌ Singleton pattern failed: Different instances returned")
        return False
    
    # Test stats
    stats = singleton1.get_stats()
    print(f"📊 Initial stats: {stats}")
    
    return True


def test_memory_leak_prevention():
    """Test that memory leaks are prevented with singleton pattern."""
    print("\n🧠 Testing Memory Leak Prevention")
    print("=" * 50)
    
    # Find a test video
    test_video = find_test_video()
    if not test_video:
        return False
    
    print(f"📹 Using test video: {test_video}")
    
    # Get initial memory usage
    initial_memory = get_memory_usage()
    print(f"🔍 Initial memory usage: {initial_memory:.1f} MB")
    
    # Create extractor
    extractor = SequenceKeypointExtractor()
    
    # Process many frames to test for memory leaks
    num_frames = 100
    frame_numbers = list(range(1, num_frames + 1))
    
    print(f"\n🎯 Processing {num_frames} frames to test memory stability...")
    
    successful_extractions = 0
    winerror_count = 0
    memory_readings = []
    
    start_time = time.time()
    
    for i, frame_num in enumerate(frame_numbers):
        if i % 10 == 0:  # Progress update every 10 frames
            current_memory = get_memory_usage()
            memory_readings.append(current_memory)
            print(f"   Frame {frame_num}: Memory {current_memory:.1f} MB")
        
        try:
            keypoints = extractor.extract_from_video_frame(test_video, frame_num)
            
            if keypoints is not None and len(keypoints.keypoints) > 0:
                successful_extractions += 1
            
        except Exception as e:
            error_str = str(e)
            if "WinError 1" in error_str or "Incorrect function" in error_str:
                winerror_count += 1
                print(f"   ❌ WinError 1 at frame {frame_num}")
    
    end_time = time.time()
    processing_time = end_time - start_time
    
    # Final memory check
    final_memory = get_memory_usage()
    memory_increase = final_memory - initial_memory
    
    # Force garbage collection
    gc.collect()
    gc_memory = get_memory_usage()
    
    print(f"\n📊 Memory Leak Test Results")
    print("=" * 50)
    print(f"✅ Successful extractions: {successful_extractions}/{num_frames}")
    print(f"🚨 WinError 1 occurrences: {winerror_count}")
    print(f"⏱️  Processing time: {processing_time:.2f}s")
    print(f"⚡ Average time per frame: {processing_time/num_frames:.3f}s")
    print(f"🧠 Initial memory: {initial_memory:.1f} MB")
    print(f"🧠 Final memory: {final_memory:.1f} MB")
    print(f"🧠 Memory after GC: {gc_memory:.1f} MB")
    print(f"📈 Memory increase: {memory_increase:.1f} MB")
    
    # Analyze memory trend
    if len(memory_readings) > 1:
        memory_trend = memory_readings[-1] - memory_readings[0]
        print(f"📊 Memory trend: {memory_trend:.1f} MB over {len(memory_readings)} readings")
    
    # Determine success
    success_criteria = [
        winerror_count == 0,  # No WinError 1 occurrences
        memory_increase < 100,  # Memory increase less than 100MB
        successful_extractions >= num_frames * 0.9  # At least 90% success rate
    ]
    
    if all(success_criteria):
        print("\n🎉 EXCELLENT: Memory leak prevention working!")
        print("   ✅ No WinError 1 occurrences")
        print("   ✅ Memory usage stable")
        print("   ✅ High success rate")
        return True
    else:
        print("\n⚠️  NEEDS IMPROVEMENT:")
        if winerror_count > 0:
            print(f"   ❌ {winerror_count} WinError 1 occurrences")
        if memory_increase >= 100:
            print(f"   ❌ High memory increase: {memory_increase:.1f} MB")
        if successful_extractions < num_frames * 0.9:
            print(f"   ❌ Low success rate: {successful_extractions/num_frames*100:.1f}%")
        return False


def test_singleton_reset():
    """Test singleton reset functionality."""
    print("\n🔄 Testing Singleton Reset Functionality")
    print("=" * 50)
    
    singleton = get_mediapipe_singleton()
    
    # Get initial stats
    initial_stats = singleton.get_stats()
    print(f"📊 Initial stats: {initial_stats}")
    
    # Process a few frames to change state
    test_video = find_test_video()
    if test_video:
        extractor = SequenceKeypointExtractor()
        
        print("🎯 Processing frames to change singleton state...")
        for frame_num in [1, 2, 3]:
            try:
                extractor.extract_from_video_frame(test_video, frame_num)
            except Exception:
                pass
        
        # Check stats after processing
        after_stats = singleton.get_stats()
        print(f"📊 After processing: {after_stats}")
        
        # Reset singleton
        print("🔄 Resetting singleton...")
        singleton.reset_landmarker()
        
        # Check stats after reset
        reset_stats = singleton.get_stats()
        print(f"📊 After reset: {reset_stats}")
        
        # Verify reset worked
        if reset_stats['frame_count'] == 0 and not reset_stats['has_landmarker']:
            print("✅ Singleton reset working correctly")
            return True
        else:
            print("❌ Singleton reset failed")
            return False
    
    return True


def test_concurrent_access():
    """Test concurrent access to singleton (threading safety)."""
    print("\n🔀 Testing Concurrent Access (Threading Safety)")
    print("=" * 50)
    
    import threading
    import queue
    
    test_video = find_test_video()
    if not test_video:
        return False
    
    results_queue = queue.Queue()
    num_threads = 3
    frames_per_thread = 5
    
    def worker_thread(thread_id):
        """Worker thread function."""
        try:
            extractor = SequenceKeypointExtractor()
            thread_results = []
            
            for i in range(frames_per_thread):
                frame_num = thread_id * frames_per_thread + i + 1
                try:
                    keypoints = extractor.extract_from_video_frame(test_video, frame_num)
                    success = keypoints is not None and len(keypoints.keypoints) > 0
                    thread_results.append((frame_num, success, None))
                except Exception as e:
                    thread_results.append((frame_num, False, str(e)))
            
            results_queue.put((thread_id, thread_results))
            
        except Exception as e:
            results_queue.put((thread_id, f"Thread error: {e}"))
    
    # Start threads
    threads = []
    for i in range(num_threads):
        thread = threading.Thread(target=worker_thread, args=(i,))
        threads.append(thread)
        thread.start()
    
    # Wait for all threads to complete
    for thread in threads:
        thread.join()
    
    # Collect results
    total_successes = 0
    total_attempts = 0
    winerror_count = 0
    
    print(f"🔍 Results from {num_threads} concurrent threads:")
    
    while not results_queue.empty():
        thread_id, thread_results = results_queue.get()
        
        if isinstance(thread_results, str):
            print(f"   Thread {thread_id}: {thread_results}")
            continue
        
        thread_successes = 0
        for frame_num, success, error in thread_results:
            total_attempts += 1
            if success:
                total_successes += 1
                thread_successes += 1
            elif error and ("WinError 1" in error or "Incorrect function" in error):
                winerror_count += 1
        
        print(f"   Thread {thread_id}: {thread_successes}/{frames_per_thread} successful")
    
    success_rate = total_successes / total_attempts if total_attempts > 0 else 0
    
    print(f"\n📊 Concurrent Access Results:")
    print(f"✅ Total successes: {total_successes}/{total_attempts}")
    print(f"📈 Success rate: {success_rate*100:.1f}%")
    print(f"🚨 WinError 1 occurrences: {winerror_count}")
    
    if winerror_count == 0 and success_rate >= 0.8:
        print("✅ Concurrent access working correctly")
        return True
    else:
        print("⚠️  Concurrent access has issues")
        return False


def find_test_video():
    """Find a test video file for testing."""
    possible_paths = [
        Path("data/gavd"),
        Path("data/youtube"),
        Path("test_artifacts"),
        Path("examples"),
    ]
    
    for base_path in possible_paths:
        if base_path.exists():
            for video_file in base_path.rglob("*.mp4"):
                if video_file.stat().st_size > 1024:  # At least 1KB
                    return video_file
    
    print("❌ No test video found. Please ensure a video file exists in:")
    for path in possible_paths:
        print(f"   - {path}")
    return None


def main():
    """Run all MediaPipe memory leak fix tests."""
    print("🚀 AlexPose MediaPipe Memory Leak Fix Test Suite")
    print("=" * 60)
    print("🎯 Testing singleton pattern and Windows stability")
    print("🧠 Includes memory leak prevention and threading safety")
    print()
    
    # Test 1: Singleton pattern
    test1_success = test_singleton_pattern()
    
    # Test 2: Memory leak prevention
    test2_success = test_memory_leak_prevention()
    
    # Test 3: Singleton reset
    test3_success = test_singleton_reset()
    
    # Test 4: Concurrent access
    test4_success = test_concurrent_access()
    
    # Final results
    print(f"\n🏁 Final Test Results")
    print("=" * 60)
    print(f"🔄 Singleton pattern test: {'✅ PASS' if test1_success else '❌ FAIL'}")
    print(f"🧠 Memory leak prevention: {'✅ PASS' if test2_success else '❌ FAIL'}")
    print(f"🔄 Singleton reset test: {'✅ PASS' if test3_success else '❌ FAIL'}")
    print(f"🔀 Concurrent access test: {'✅ PASS' if test4_success else '❌ FAIL'}")
    
    all_passed = test1_success and test2_success and test3_success and test4_success
    
    if all_passed:
        print(f"\n🎉 ALL TESTS PASSED!")
        print("   ✅ MediaPipe singleton pattern working correctly")
        print("   ✅ Memory leaks prevented on Windows")
        print("   ✅ No more WinError 1 issues expected")
        print("   ✅ Threading safety implemented")
        print("   ✅ GAVD processing should be stable")
        return True
    else:
        print(f"\n⚠️  SOME TESTS FAILED")
        print("   Please review the error messages above.")
        print("   Additional fixes may be needed for complete stability.")
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)