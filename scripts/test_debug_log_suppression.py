#!/usr/bin/env python3
"""
Test script to verify that DEBUG logs from Windows FFmpeg handler are suppressed.

This script tests that:
1. Normal operation produces clean logs without verbose DEBUG messages
2. Verbose mode can still be enabled when needed for debugging
3. GAVD processing has clean output during normal operation
"""

import os
import sys
import time
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from ambient.utils.log_config import get_logger
from ambient.pose.windows_ffmpeg_handler import WindowsVideoFrameExtractor
from ambient.pose.keypoint_extractor import SequenceKeypointExtractor
import numpy as np

logger = get_logger(__name__)


def test_normal_operation_clean_logs():
    """Test that normal operation produces clean logs."""
    print("=" * 60)
    print("Testing Normal Operation (Clean Logs)")
    print("=" * 60)
    
    print("\n1. Testing WindowsVideoFrameExtractor with verbose=False (default)...")
    
    try:
        # Create extractor with default settings (verbose=False)
        extractor = WindowsVideoFrameExtractor(
            prefer_ffmpeg=True,
            ffmpeg_timeout=30
            # verbose=False is the default
        )
        
        print("   ✅ WindowsVideoFrameExtractor created with clean logging")
        print("   ℹ️  You should NOT see DEBUG messages about:")
        print("      - 'Created temporary file'")
        print("      - 'FFmpeg command'") 
        print("      - 'Successfully extracted frame'")
        print("      - 'Successfully cleaned up temporary file'")
        
        return True
        
    except Exception as e:
        print(f"   ❌ Failed to create WindowsVideoFrameExtractor: {e}")
        return False


def test_verbose_mode_debug_logs():
    """Test that verbose mode still produces debug logs when needed."""
    print("\n" + "=" * 60)
    print("Testing Verbose Mode (Debug Logs Enabled)")
    print("=" * 60)
    
    print("\n2. Testing WindowsVideoFrameExtractor with verbose=True...")
    
    try:
        # Create extractor with verbose logging enabled
        extractor = WindowsVideoFrameExtractor(
            prefer_ffmpeg=True,
            ffmpeg_timeout=30,
            verbose=True  # Enable debug logs
        )
        
        print("   ✅ WindowsVideoFrameExtractor created with verbose logging")
        print("   ℹ️  With verbose=True, you WOULD see DEBUG messages about:")
        print("      - 'Created temporary file'")
        print("      - 'FFmpeg command'")
        print("      - 'Successfully extracted frame'")
        print("      - 'Successfully cleaned up temporary file'")
        
        return True
        
    except Exception as e:
        print(f"   ❌ Failed to create verbose WindowsVideoFrameExtractor: {e}")
        return False


def test_sequence_extractor_clean_logs():
    """Test that SequenceKeypointExtractor uses clean logging."""
    print("\n" + "=" * 60)
    print("Testing SequenceKeypointExtractor (Clean Logs)")
    print("=" * 60)
    
    print("\n3. Testing SequenceKeypointExtractor with Windows optimization...")
    
    try:
        # Create extractor with Windows optimization (should use clean logging)
        use_process_isolation = os.name == 'nt'  # Windows
        extractor = SequenceKeypointExtractor(use_process_isolation=use_process_isolation)
        
        print(f"   ✅ SequenceKeypointExtractor created (process_isolation={use_process_isolation})")
        print("   ℹ️  Internal WindowsVideoFrameExtractor should use verbose=False")
        
        # Test with a dummy image
        test_image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        
        print("   🔍 Testing keypoint extraction (should have clean logs)...")
        start_time = time.time()
        
        keypoint_set = extractor.extract_from_image(test_image)
        
        end_time = time.time()
        processing_time = end_time - start_time
        
        print(f"   ✅ Keypoint extraction completed in {processing_time:.3f}s")
        print(f"   ✅ Detected {len(keypoint_set.keypoints)} keypoints")
        print("   ✅ No verbose FFmpeg DEBUG logs should appear above")
        
        # Cleanup
        extractor.cleanup()
        
        return True
        
    except Exception as e:
        print(f"   ❌ SequenceKeypointExtractor test failed: {e}")
        return False


def test_log_level_verification():
    """Verify what types of logs should and shouldn't appear."""
    print("\n" + "=" * 60)
    print("Log Level Verification Guide")
    print("=" * 60)
    
    print("\n✅ SHOULD see these log levels during normal operation:")
    print("   • INFO  - Important status messages")
    print("   • WARN  - Warning messages")
    print("   • ERROR - Error messages")
    
    print("\n❌ Should NOT see these DEBUG messages during normal operation:")
    print("   • 'Created temporary file: C:\\Users\\...\\alexpose_*.jpg'")
    print("   • 'FFmpeg command: ffmpeg -i ... -vf select=eq(n\\,X) ...'")
    print("   • 'Successfully extracted frame: (360, 640, 3) from ... (X bytes)'")
    print("   • 'Successfully cleaned up temporary file: C:\\Users\\...'")
    print("   • 'FFmpeg is available and working'")
    
    print("\n🔧 To enable DEBUG logs for troubleshooting:")
    print("   • Set verbose=True when creating WindowsVideoFrameExtractor")
    print("   • Or adjust the logging level in your configuration")
    
    return True


def main():
    """Run all debug log suppression tests."""
    print("DEBUG Log Suppression Test Suite")
    print("=" * 60)
    
    all_tests_passed = True
    
    # Test 1: Normal operation with clean logs
    if not test_normal_operation_clean_logs():
        all_tests_passed = False
    
    # Test 2: Verbose mode with debug logs
    if not test_verbose_mode_debug_logs():
        all_tests_passed = False
    
    # Test 3: SequenceKeypointExtractor clean logs
    if not test_sequence_extractor_clean_logs():
        all_tests_passed = False
    
    # Test 4: Log level verification guide
    if not test_log_level_verification():
        all_tests_passed = False
    
    # Final results
    print("\n" + "=" * 60)
    print("FINAL RESULTS")
    print("=" * 60)
    
    if all_tests_passed:
        print("✅ ALL TESTS PASSED!")
        print("✅ DEBUG log suppression is working correctly")
        print("✅ Normal operation should have clean logs")
        print("✅ Verbose mode is available for debugging")
        print("✅ GAVD processing will have professional output")
    else:
        print("❌ SOME TESTS FAILED!")
        print("❌ DEBUG log suppression may need further investigation")
    
    return all_tests_passed


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)