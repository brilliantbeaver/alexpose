#!/usr/bin/env python3
"""
Test script to validate GAVD data integration functionality.
This script tests the enhanced real data manager and GAVD data loading.
"""

import sys
from pathlib import Path

# Add the project root to the Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

def test_gavd_data_loading():
    """Test GAVD data loading functionality."""
    print("Testing GAVD data loading...")
    
    try:
        from tests.fixtures.real_data_fixtures import get_real_data_manager
        
        # Get the real data manager
        data_manager = get_real_data_manager()
        print(f"✓ Real data manager initialized")
        
        # Test GAVD data loading
        gavd_data = data_manager.get_gavd_test_subset()
        print(f"✓ GAVD data loaded successfully")
        
        # Validate data structure
        assert "normal_subjects" in gavd_data, "Missing normal_subjects"
        assert "abnormal_subjects" in gavd_data, "Missing abnormal_subjects"
        assert "metadata" in gavd_data, "Missing metadata"
        print(f"✓ GAVD data structure validated")
        
        # Print metadata
        metadata = gavd_data["metadata"]
        print(f"\nGAVD Data Summary:")
        print(f"  Source: {metadata.get('source', 'unknown')}")
        print(f"  Total subjects: {metadata.get('total_subjects', 0)}")
        print(f"  Normal subjects: {metadata.get('normal_count', 0)}")
        print(f"  Abnormal subjects: {metadata.get('abnormal_count', 0)}")
        print(f"  Is synthetic: {metadata.get('synthetic', True)}")
        
        if not metadata.get('synthetic', True):
            print(f"  Normal patterns: {metadata.get('gait_patterns', {}).get('normal', [])}")
            print(f"  Abnormal patterns: {metadata.get('gait_patterns', {}).get('abnormal', [])}")
        
        # Test video URLs if available
        try:
            video_urls = data_manager.get_gavd_video_urls()
            print(f"\nGAVD Video URLs:")
            print(f"  Normal gait URLs: {len(video_urls.get('normal_gait_urls', []))}")
            print(f"  Abnormal gait URLs: {len(video_urls.get('abnormal_gait_urls', []))}")
            
            # Show first few URLs as examples
            normal_urls = video_urls.get('normal_gait_urls', [])
            if normal_urls:
                print(f"  Example normal URL: {normal_urls[0]}")
            
            abnormal_urls = video_urls.get('abnormal_gait_urls', [])
            if abnormal_urls:
                print(f"  Example abnormal URL: {abnormal_urls[0]}")
                
        except Exception as e:
            print(f"  Warning: Could not load video URLs: {e}")
        
        print(f"\n✓ GAVD data integration test passed!")
        return True
        
    except Exception as e:
        print(f"✗ GAVD data integration test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_sample_videos():
    """Test sample video loading functionality."""
    print("\nTesting sample video loading...")
    
    try:
        from tests.fixtures.real_data_fixtures import get_real_data_manager
        
        data_manager = get_real_data_manager()
        sample_videos = data_manager.get_sample_videos()
        
        print(f"Sample videos found:")
        for video_type, video_path in sample_videos.items():
            exists = video_path.exists() if hasattr(video_path, 'exists') else Path(video_path).exists()
            status = "✓" if exists else "✗"
            print(f"  {status} {video_type}: {video_path}")
        
        available_videos = [v for v, p in sample_videos.items() 
                          if (hasattr(p, 'exists') and p.exists()) or Path(p).exists()]
        
        print(f"\n✓ Found {len(available_videos)} available sample videos")
        return True
        
    except Exception as e:
        print(f"✗ Sample video test failed: {e}")
        return False

def main():
    """Run all integration tests."""
    print("=" * 60)
    print("GAVD Integration Test Suite")
    print("=" * 60)
    
    tests_passed = 0
    total_tests = 2
    
    # Test GAVD data loading
    if test_gavd_data_loading():
        tests_passed += 1
    
    # Test sample videos
    if test_sample_videos():
        tests_passed += 1
    
    print("\n" + "=" * 60)
    print(f"Test Results: {tests_passed}/{total_tests} tests passed")
    
    if tests_passed == total_tests:
        print("✓ All tests passed! GAVD integration is working correctly.")
        return 0
    else:
        print("✗ Some tests failed. Please check the output above.")
        return 1

if __name__ == "__main__":
    sys.exit(main())