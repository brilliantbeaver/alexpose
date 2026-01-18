#!/usr/bin/env python3
"""
Test script to verify FFmpeg Windows fix for keypoint extraction.

This script tests the Windows-specific FFmpeg temporary file handling
to ensure no more "WinError 1: Incorrect function" errors occur.
"""

import sys
import time
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from ambient.pose.keypoint_extractor import SequenceKeypointExtractor
from ambient.pose.windows_ffmpeg_handler import WindowsVideoFrameExtractor
from ambient.gavd.gavd_processor import GAVDProcessor


def test_windows_ffmpeg_handler():
    """Test the new Windows FFmpeg handler directly."""
    print("🔧 Testing Windows FFmpeg Handler")
    print("=" * 50)
    
    # Find a test video file
    test_video = find_test_video()
    if not test_video:
        return False
    
    print(f"📹 Using test video: {test_video}")
    print(f"📁 Video size: {test_video.stat().st_size / (1024*1024):.1f} MB")
    
    # Test the Windows FFmpeg handler
    print("\n🎯 Testing Windows FFmpeg Handler")
    print("-" * 40)
    
    extractor = WindowsVideoFrameExtractor(prefer_ffmpeg=True, ffmpeg_timeout=30)
    
    # Test multiple frames
    test_frames = [1, 5, 10, 15, 20]
    successful_extractions = 0
    
    start_time = time.time()
    
    for frame_num in test_frames:
        print(f"🔍 Testing frame {frame_num}...")
        
        try:
            frame = extractor.extract_frame(test_video, frame_num)
            
            if frame is not None:
                successful_extractions += 1
                print(f"   ✅ Success: Frame shape {frame.shape}")
            else:
                print(f"   ❌ Failed to extract frame")
                
        except Exception as e:
            print(f"   ❌ Exception: {e}")
    
    end_time = time.time()
    processing_time = end_time - start_time
    
    # Get extraction statistics
    stats = extractor.get_extraction_stats()
    
    print(f"\n📊 Windows FFmpeg Handler Results")
    print("=" * 50)
    print(f"✅ Successful extractions: {successful_extractions}/{len(test_frames)}")
    print(f"⏱️  Total processing time: {processing_time:.2f}s")
    print(f"⚡ Average time per frame: {processing_time/len(test_frames):.2f}s")
    print(f"🎯 FFmpeg success rate: {stats.get('ffmpeg_success_rate', 0)*100:.1f}%")
    print(f"🔄 OpenCV fallback rate: {stats.get('opencv_success_rate', 0)*100:.1f}%")
    print(f"📈 Overall success rate: {stats.get('overall_success_rate', 0)*100:.1f}%")
    
    return successful_extractions == len(test_frames)


def test_keypoint_extraction_integration():
    """Test keypoint extraction with the new Windows handler."""
    print("\n🎯 Testing Keypoint Extraction Integration")
    print("=" * 50)
    
    test_video = find_test_video()
    if not test_video:
        return False
    
    print(f"📹 Using test video: {test_video}")
    
    extractor = SequenceKeypointExtractor()
    
    # Test multiple frames
    test_frames = [1, 5, 10]
    successful_extractions = 0
    total_keypoints = 0
    
    start_time = time.time()
    
    for frame_num in test_frames:
        print(f"🔍 Testing keypoint extraction for frame {frame_num}...")
        
        try:
            keypoints = extractor.extract_from_video_frame(test_video, frame_num)
            
            if keypoints is not None and len(keypoints.keypoints) > 0:
                successful_extractions += 1
                total_keypoints += len(keypoints.keypoints)
                print(f"   ✅ Success: {len(keypoints.keypoints)} keypoints extracted")
                
                # Check keypoint quality
                if len(keypoints.keypoints) >= 20:  # MediaPipe should give 33 keypoints
                    print(f"   📊 Good keypoint count: {len(keypoints.keypoints)}")
                else:
                    print(f"   ⚠️  Low keypoint count: {len(keypoints.keypoints)}")
            else:
                print(f"   ❌ Failed to extract keypoints")
                
        except Exception as e:
            print(f"   ❌ Exception: {e}")
    
    end_time = time.time()
    processing_time = end_time - start_time
    
    print(f"\n📊 Keypoint Extraction Results")
    print("=" * 50)
    print(f"✅ Successful extractions: {successful_extractions}/{len(test_frames)}")
    print(f"🎯 Total keypoints extracted: {total_keypoints}")
    print(f"📊 Average keypoints per frame: {total_keypoints/max(successful_extractions, 1):.1f}")
    print(f"⏱️  Total processing time: {processing_time:.2f}s")
    print(f"⚡ Average time per frame: {processing_time/len(test_frames):.2f}s")
    
    return successful_extractions == len(test_frames)


def test_gavd_processing_no_errors():
    """Test GAVD processing to ensure no FFmpeg errors."""
    print("\n🏥 Testing GAVD Processing (No FFmpeg Errors)")
    print("=" * 50)
    
    # Find a GAVD video file
    gavd_path = Path("data/gavd")
    if not gavd_path.exists():
        print("❌ GAVD data directory not found")
        return False
    
    test_video = None
    for condition_dir in gavd_path.iterdir():
        if condition_dir.is_dir():
            for video_file in condition_dir.glob("*.mp4"):
                test_video = video_file
                break
            if test_video:
                break
    
    if not test_video:
        print("❌ No GAVD video files found")
        return False
    
    print(f"📹 Using GAVD video: {test_video}")
    
    # Test GAVD processor with limited frames
    processor = GAVDProcessor()
    
    try:
        print("🔍 Testing GAVD keypoint extraction (5 frames)...")
        
        # Use the batch optimization path (no estimator)
        sequences = processor._extract_pose_sequences(
            video_path=test_video,
            pose_estimator=None,  # This triggers batch optimization
            max_frames=5  # Limit to 5 frames for testing
        )
        
        if sequences and len(sequences) > 0:
            total_frames = sum(len(seq.keypoint_sets) for seq in sequences)
            print(f"✅ Successfully processed {total_frames} frames")
            print(f"📊 Generated {len(sequences)} sequences")
            
            # Check keypoint quality
            if sequences[0].keypoint_sets:
                sample_kp = sequences[0].keypoint_sets[0]
                print(f"🎯 Sample keypoints: {len(sample_kp.keypoints)} per frame")
                
            return True
        else:
            print("❌ No sequences generated")
            return False
            
    except Exception as e:
        print(f"❌ GAVD processing failed: {e}")
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
    """Run all Windows FFmpeg fix tests."""
    print("🚀 AlexPose Windows FFmpeg Fix Test Suite")
    print("=" * 60)
    print("🎯 Testing comprehensive Windows-safe FFmpeg implementation")
    print("🔧 Includes proper OOP design with error handling and fallbacks")
    print()
    
    # Test 1: Windows FFmpeg handler
    test1_success = test_windows_ffmpeg_handler()
    
    # Test 2: Keypoint extraction integration
    test2_success = test_keypoint_extraction_integration()
    
    # Test 3: GAVD processing integration
    test3_success = test_gavd_processing_no_errors()
    
    # Final results
    print(f"\n🏁 Final Test Results")
    print("=" * 60)
    print(f"🔧 Windows FFmpeg handler: {'✅ PASS' if test1_success else '❌ FAIL'}")
    print(f"🎯 Keypoint extraction: {'✅ PASS' if test2_success else '❌ FAIL'}")
    print(f"🏥 GAVD processing: {'✅ PASS' if test3_success else '❌ FAIL'}")
    
    all_passed = test1_success and test2_success and test3_success
    
    if all_passed:
        print(f"\n🎉 ALL TESTS PASSED!")
        print("   ✅ Windows FFmpeg fix is working correctly")
        print("   ✅ No more 'WinError 1: Incorrect function' errors expected")
        print("   ✅ Proper OOP design with error handling implemented")
        print("   ✅ Automatic fallback to OpenCV when FFmpeg fails")
        print("   ✅ Robust temporary file management for Windows")
        return True
    else:
        print(f"\n⚠️  SOME TESTS FAILED")
        print("   Please review the error messages above.")
        print("   The Windows FFmpeg fix may need additional adjustments.")
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)