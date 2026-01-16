"""
Test script for pose_estimation_for_frames() function
Verifies the KeypointSet compatibility fix
"""

from pathlib import Path
import sys

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from ambient.utils.eval_keypoints import pose_estimation_for_frames
from ambient.data.gavd_loader import GAVDLoader

def test_pose_estimation_for_frames():
    """Test pose_estimation_for_frames with multiple frames"""
    
    print("=" * 60)
    print("TEST: pose_estimation_for_frames() - KeypointSet Fix")
    print("=" * 60)
    
    # Load GAVD data
    print("\n[1/3] Loading GAVD data...")
    loader = GAVDLoader(project_root)
    sequences = loader.load_sequences()
    
    if not sequences:
        print("[ERROR] No sequences found")
        return False
    
    # Get first sequence
    first_seq_id = list(sequences.keys())[0]
    sequence_data = sequences[first_seq_id]
    print(f"[OK] Loaded sequence: {first_seq_id}")
    print(f"     Total frames: {len(sequence_data)}")
    
    # Test with specific frame indices
    print("\n[2/3] Testing pose estimation on frames [0, 100, 200]...")
    try:
        results = pose_estimation_for_frames(
            project_root=project_root,
            sequence_data=sequence_data,
            frame_indices=[0, 100, 200],
            show_each=False  # Don't show visualizations in test
        )
        
        print(f"\n[OK] Function executed successfully!")
        print(f"     Processed {len(results)} frames")
        
        # Verify results structure
        print("\n[3/3] Verifying results structure...")
        for i, result in enumerate(results):
            print(f"\n   Frame {i+1}:")
            print(f"     - frame_index: {result['frame_index']}")
            print(f"     - frame_num: {result['frame_num']}")
            print(f"     - landmarks_count: {result['landmarks_count']}")
            print(f"     - avg_confidence: {result['avg_confidence']:.3f}")
            
            # Verify all expected keys exist
            assert 'frame_index' in result
            assert 'frame_num' in result
            assert 'landmarks_count' in result
            assert 'avg_confidence' in result
            assert 'metadata' in result
            
            # Verify types
            assert isinstance(result['landmarks_count'], int)
            assert isinstance(result['avg_confidence'], (int, float))
        
        print("\n" + "=" * 60)
        print("[SUCCESS] All tests passed!")
        print("=" * 60)
        return True
        
    except Exception as e:
        print(f"\n[ERROR] Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_pose_estimation_for_frames()
    sys.exit(0 if success else 1)
