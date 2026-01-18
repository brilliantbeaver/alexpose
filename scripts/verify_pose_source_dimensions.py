"""
Verify that pose data includes source dimensions after the fix.

This script checks if newly processed GAVD data includes source_width
and source_height in keypoints, which is critical for proper overlay scaling.
"""

import sys
import json
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

def verify_pose_source_dimensions(dataset_id: str):
    """
    Verify that pose data includes source dimensions.
    
    Args:
        dataset_id: Dataset ID to check
    """
    print("=" * 80)
    print("POSE SOURCE DIMENSIONS VERIFICATION")
    print("=" * 80)
    
    # Find pose data file
    results_dir = Path("data/storage/gavd_results")
    pose_data_file = results_dir / f"{dataset_id}_pose_data.json"
    
    if not pose_data_file.exists():
        print(f"\n❌ Pose data file not found: {pose_data_file}")
        print(f"\nRun: python -m ambient.cli process-gavd {dataset_id}")
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
    
    if not pose_data:
        print(f"\n⚠️  Pose data is empty")
        return False
    
    # Check multiple sequences and frames
    sequences_checked = 0
    frames_checked = 0
    frames_with_source_dims = 0
    frames_without_source_dims = 0
    
    for seq_id, seq_frames in pose_data.items():
        if not seq_frames:
            continue
            
        sequences_checked += 1
        
        for frame_num, frame_data in seq_frames.items():
            frames_checked += 1
            
            # Check frame-level source dimensions
            has_frame_level_dims = False
            if isinstance(frame_data, dict):
                if 'source_width' in frame_data and 'source_height' in frame_data:
                    has_frame_level_dims = True
                    if frame_data['source_width'] and frame_data['source_height']:
                        frames_with_source_dims += 1
                    else:
                        frames_without_source_dims += 1
                        
                # Check keypoint-level source dimensions
                keypoints = frame_data.get('keypoints', [])
                if keypoints and len(keypoints) > 0:
                    first_kp = keypoints[0]
                    if isinstance(first_kp, dict):
                        has_kp_dims = 'source_width' in first_kp and 'source_height' in first_kp
                        
                        if has_kp_dims:
                            if not has_frame_level_dims:
                                frames_with_source_dims += 1
                            
                            # Show details for first frame
                            if frames_checked == 1:
                                print(f"\n✓ First frame analysis:")
                                print(f"  - Sequence: {seq_id}")
                                print(f"  - Frame: {frame_num}")
                                print(f"  - Keypoints: {len(keypoints)}")
                                print(f"  - Frame-level dims: {has_frame_level_dims}")
                                print(f"  - Keypoint-level dims: {has_kp_dims}")
                                if has_kp_dims:
                                    print(f"  - Source dimensions: {first_kp['source_width']}x{first_kp['source_height']}")
                                    print(f"  - First keypoint: x={first_kp['x']:.1f}, y={first_kp['y']:.1f}, conf={first_kp['confidence']:.2f}")
                        else:
                            if not has_frame_level_dims:
                                frames_without_source_dims += 1
            
            # Only check first 10 frames per sequence for speed
            if frames_checked >= sequences_checked * 10:
                break
        
        # Only check first 3 sequences for speed
        if sequences_checked >= 3:
            break
    
    print(f"\n" + "=" * 80)
    print(f"VERIFICATION RESULTS")
    print(f"=" * 80)
    print(f"Sequences checked: {sequences_checked}")
    print(f"Frames checked: {frames_checked}")
    print(f"Frames WITH source dimensions: {frames_with_source_dims}")
    print(f"Frames WITHOUT source dimensions: {frames_without_source_dims}")
    
    if frames_without_source_dims == 0 and frames_with_source_dims > 0:
        print(f"\n✅ SUCCESS: All checked frames have source dimensions!")
        print(f"✅ Pose overlays should display correctly.")
        return True
    elif frames_with_source_dims > 0:
        print(f"\n⚠️  PARTIAL: Some frames have source dimensions, some don't.")
        print(f"   This might indicate mixed old/new data.")
        return False
    else:
        print(f"\n❌ FAILURE: No frames have source dimensions!")
        print(f"❌ Dataset needs to be reprocessed with updated code.")
        print(f"\nRun: python -m ambient.cli process-gavd {dataset_id}")
        return False

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python verify_pose_source_dimensions.py <dataset_id>")
        print("\nExample:")
        print("  python verify_pose_source_dimensions.py cljar9bqg00c43n6lmh1qhydd")
        sys.exit(1)
    
    dataset_id = sys.argv[1]
    success = verify_pose_source_dimensions(dataset_id)
    
    sys.exit(0 if success else 1)
