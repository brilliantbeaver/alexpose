"""
Check if GAVD pose data includes source dimensions (new format).

This script checks if pose data has been processed with the new code
that includes source video dimensions with each keypoint.
"""

import sys
import json
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

def check_pose_data_version(dataset_id: str):
    """
    Check if pose data for a dataset includes source dimensions.
    
    Args:
        dataset_id: Dataset ID to check
    """
    print("=" * 80)
    print("POSE DATA VERSION CHECKER")
    print("=" * 80)
    
    # Find pose data file
    results_dir = Path("data/storage/gavd_results")
    pose_data_file = results_dir / f"{dataset_id}_pose_data.json"
    
    if not pose_data_file.exists():
        print(f"\n❌ Pose data file not found: {pose_data_file}")
        print(f"\nThis dataset may not have been processed yet.")
        print(f"Run: python -m ambient.cli process-gavd {dataset_id}")
        return False
    
    print(f"\n✓ Found pose data file: {pose_data_file}")
    
    # Load pose data
    try:
        with open(pose_data_file, 'r') as f:
            pose_data = json.load(f)
    except Exception as e:
        print(f"\n❌ Failed to load pose data: {e}")
        return False
    
    print(f"✓ Loaded pose data")
    
    # Check format
    if not pose_data:
        print(f"\n⚠️  Pose data is empty")
        return False
    
    # Get first sequence
    first_seq_id = list(pose_data.keys())[0]
    first_seq = pose_data[first_seq_id]
    
    if not first_seq:
        print(f"\n⚠️  First sequence has no frames")
        return False
    
    # Get first frame
    first_frame_num = list(first_seq.keys())[0]
    first_frame = first_seq[first_frame_num]
    
    print(f"\nChecking first frame: Sequence={first_seq_id}, Frame={first_frame_num}")
    
    # Check if it's the new format (dict with metadata)
    if isinstance(first_frame, dict):
        has_keypoints = 'keypoints' in first_frame
        has_source_width = 'source_width' in first_frame
        has_source_height = 'source_height' in first_frame
        
        print(f"  - Has 'keypoints' field: {has_keypoints}")
        print(f"  - Has 'source_width' field: {has_source_width}")
        print(f"  - Has 'source_height' field: {has_source_height}")
        
        if has_keypoints and has_source_width and has_source_height:
            source_width = first_frame['source_width']
            source_height = first_frame['source_height']
            keypoints = first_frame['keypoints']
            
            print(f"\n✅ NEW FORMAT DETECTED!")
            print(f"  - Source dimensions: {source_width}x{source_height}")
            print(f"  - Number of keypoints: {len(keypoints)}")
            
            # Check if keypoints also have source dims (redundant but good to verify)
            if keypoints and len(keypoints) > 0:
                first_kp = keypoints[0]
                if isinstance(first_kp, dict):
                    kp_has_source = 'source_width' in first_kp and 'source_height' in first_kp
                    print(f"  - Keypoints have source dims: {kp_has_source}")
                    if kp_has_source:
                        print(f"    (Keypoint source: {first_kp['source_width']}x{first_kp['source_height']})")
            
            print(f"\n✅ This dataset has been processed with the NEW code.")
            print(f"✅ Pose overlays should display correctly.")
            return True
        else:
            print(f"\n⚠️  OLD FORMAT DETECTED (dict without source dimensions)")
            print(f"\n❌ This dataset needs to be REPROCESSED.")
            print(f"\nRun: python -m ambient.cli process-gavd {dataset_id}")
            return False
    
    elif isinstance(first_frame, list):
        print(f"\n⚠️  OLD FORMAT DETECTED (list of keypoints)")
        print(f"  - Number of keypoints: {len(first_frame)}")
        
        # Check if keypoints have source dims
        if first_frame and len(first_frame) > 0:
            first_kp = first_frame[0]
            if isinstance(first_kp, dict):
                has_source = 'source_width' in first_kp and 'source_height' in first_kp
                print(f"  - Keypoints have source dims: {has_source}")
                
                if has_source:
                    print(f"\n✅ Keypoints have source dimensions!")
                    print(f"  - Source: {first_kp['source_width']}x{first_kp['source_height']}")
                    print(f"\n⚠️  But frame-level metadata is missing.")
                    print(f"   Consider reprocessing for best results.")
                    return True
        
        print(f"\n❌ This dataset needs to be REPROCESSED.")
        print(f"\nRun: python -m ambient.cli process-gavd {dataset_id}")
        return False
    
    else:
        print(f"\n❌ UNKNOWN FORMAT: {type(first_frame)}")
        return False

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python check_pose_data_version.py <dataset_id>")
        print("\nExample:")
        print("  python check_pose_data_version.py abc123-def456-ghi789")
        sys.exit(1)
    
    dataset_id = sys.argv[1]
    success = check_pose_data_version(dataset_id)
    
    print("\n" + "=" * 80)
    if success:
        print("✅ RESULT: Data is up-to-date")
    else:
        print("❌ RESULT: Data needs reprocessing")
    print("=" * 80)
    
    sys.exit(0 if success else 1)
