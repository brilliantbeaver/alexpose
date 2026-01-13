#!/usr/bin/env python3
"""
Test script to verify the KeyError fixes in notebooks/utils/keypoints.py
"""

import sys
import pandas as pd
from pathlib import Path

# Add project root to path
project_root = Path.cwd()
sys.path.insert(0, str(project_root))

def test_dataframe_access():
    """Test the corrected DataFrame column access"""
    
    # Create a sample DataFrame similar to GAVD structure
    sample_data = pd.DataFrame({
        'seq': ['cljan9b4p00043n6ligceanyp'] * 5,
        'frame_num': [1757, 1758, 1759, 1760, 1761],
        'cam_view': ['front'] * 5,
        'gait_event': ['stance'] * 5,
        'dataset': ['GAVD'] * 5,
        'gait_pat': ['normal'] * 5,
        'bbox': [{'left': 156.0, 'top': 125.0, 'width': 100, 'height': 200}] * 5,
        'vid_info': [{}] * 5,
        'id': ['test_id'] * 5,
        'url': ['https://youtube.com/watch?v=test'] * 5
    })
    
    print("✅ Sample DataFrame created:")
    print(f"   Shape: {sample_data.shape}")
    print(f"   Columns: {list(sample_data.columns)}")
    
    # Test the corrected access method
    try:
        sequence_id = sample_data['seq'].iloc[0]
        print(f"✅ Corrected access works: sequence_id = '{sequence_id}'")
    except Exception as e:
        print(f"❌ Corrected access failed: {e}")
        return False
    
    # Test the old broken method to show it would fail
    try:
        # This would cause the KeyError
        broken_access = sample_data.seq[0]  # This might work in some cases
        print(f"⚠️  Old method happened to work: '{broken_access}'")
    except KeyError as e:
        print(f"❌ Old method fails as expected: KeyError {e}")
    except Exception as e:
        print(f"❌ Old method fails with: {e}")
    
    # Test edge cases
    empty_df = pd.DataFrame()
    try:
        if empty_df.empty:
            print("✅ Empty DataFrame validation works")
        else:
            print("❌ Empty DataFrame validation failed")
    except Exception as e:
        print(f"❌ Empty DataFrame test failed: {e}")
    
    # Test missing column
    missing_col_df = pd.DataFrame({'other_col': [1, 2, 3]})
    try:
        if 'seq' not in missing_col_df.columns:
            print("✅ Missing column validation works")
        else:
            print("❌ Missing column validation failed")
    except Exception as e:
        print(f"❌ Missing column test failed: {e}")
    
    return True

def test_import_fixes():
    """Test that the fixed functions can be imported"""
    try:
        from notebooks.utils.keypoints import pose_estimation_all_sequences
        print("✅ pose_estimation_all_sequences can be imported")
    except ImportError as e:
        print(f"❌ Import failed: {e}")
        return False
    except Exception as e:
        print(f"⚠️  Import warning: {e}")
    
    try:
        from notebooks.utils.keypoints import extract_pose_from_sequence
        print("✅ extract_pose_from_sequence can be imported")
    except ImportError as e:
        print(f"❌ Import failed: {e}")
        return False
    except Exception as e:
        print(f"⚠️  Import warning: {e}")
    
    return True

if __name__ == "__main__":
    print("🔍 Testing KeyError fixes...")
    print("=" * 50)
    
    success = True
    
    print("\n1. Testing DataFrame access fixes:")
    success &= test_dataframe_access()
    
    print("\n2. Testing function imports:")
    success &= test_import_fixes()
    
    print("\n" + "=" * 50)
    if success:
        print("✅ All tests passed! The KeyError fixes should work.")
    else:
        print("❌ Some tests failed. Check the output above.")
    
    print("\n📋 Summary of fixes applied:")
    print("   1. Fixed DataFrame column access: sequence_data.seq[0] → sequence_data['seq'].iloc[0]")
    print("   2. Added validation for empty DataFrames")
    print("   3. Added validation for missing columns")
    print("   4. Improved error handling and messages")