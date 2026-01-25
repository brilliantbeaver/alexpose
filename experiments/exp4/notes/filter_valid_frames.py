"""
Helper functions to filter and handle frames with empty keypoints.
Add these to your notebook.
"""

def filter_valid_keypoints(keypoints_array, min_keypoints=25):
    """
    Filter keypoint array to only include frames with valid detections.
    
    Args:
        keypoints_array: List of KeypointSet objects (may contain None or empty)
        min_keypoints: Minimum number of keypoints required (default 25 out of 33)
    
    Returns:
        tuple: (valid_keypoints, valid_indices, stats)
    """
    valid_keypoints = []
    valid_indices = []
    
    stats = {
        'total': len(keypoints_array),
        'none': 0,
        'empty': 0,
        'low_quality': 0,
        'valid': 0
    }
    
    for i, kp_set in enumerate(keypoints_array):
        if kp_set is None:
            stats['none'] += 1
            continue
        
        num_kp = len(kp_set.keypoints)
        
        if num_kp == 0:
            stats['empty'] += 1
            continue
        
        if num_kp < min_keypoints:
            stats['low_quality'] += 1
            continue
        
        # Valid frame
        valid_keypoints.append(kp_set)
        valid_indices.append(i)
        stats['valid'] += 1
    
    return valid_keypoints, valid_indices, stats


def print_keypoint_stats(keypoints_array, sequence_name="Sequence"):
    """
    Print detailed statistics about keypoint extraction results.
    """
    print(f"\n{'='*70}")
    print(f"Keypoint Statistics: {sequence_name}")
    print(f"{'='*70}")
    
    if not keypoints_array:
        print("❌ Empty array - no frames processed")
        return
    
    # Analyze
    none_count = sum(1 for kp in keypoints_array if kp is None)
    empty_count = sum(1 for kp in keypoints_array if kp is not None and len(kp.keypoints) == 0)
    valid_count = sum(1 for kp in keypoints_array if kp is not None and len(kp.keypoints) > 0)
    
    print(f"Total frames: {len(keypoints_array)}")
    print(f"  ✅ Valid detections: {valid_count} ({valid_count/len(keypoints_array)*100:.1f}%)")
    print(f"  ⚠️  Empty detections: {empty_count} ({empty_count/len(keypoints_array)*100:.1f}%)")
    print(f"  ❌ Failed extractions: {none_count} ({none_count/len(keypoints_array)*100:.1f}%)")
    
    if valid_count > 0:
        # Keypoint count distribution
        kp_counts = [len(kp.keypoints) for kp in keypoints_array if kp is not None and len(kp.keypoints) > 0]
        avg_kp = sum(kp_counts) / len(kp_counts)
        min_kp = min(kp_counts)
        max_kp = max(kp_counts)
        
        print(f"\nKeypoint counts (valid frames only):")
        print(f"  Average: {avg_kp:.1f}")
        print(f"  Range: {min_kp} - {max_kp}")
        
        # Quality assessment
        full_detections = sum(1 for c in kp_counts if c == 33)
        partial_detections = sum(1 for c in kp_counts if 20 <= c < 33)
        poor_detections = sum(1 for c in kp_counts if c < 20)
        
        print(f"\nQuality breakdown:")
        print(f"  🟢 Full (33 keypoints): {full_detections}")
        print(f"  🟡 Partial (20-32): {partial_detections}")
        print(f"  🔴 Poor (<20): {poor_detections}")
    
    print(f"{'='*70}\n")


def extract_with_filtering(df, video_base_path, extractor, min_keypoints=25, verbose=True):
    """
    Extract keypoints and automatically filter to valid frames.
    
    Returns:
        tuple: (valid_keypoints, valid_df, stats)
    """
    print(f"🔄 Extracting keypoints from {len(df)} frames...")
    
    # Extract all frames
    all_keypoints = extractor.extract_from_sequence(
        sequence_data=df,
        video_base_path=video_base_path,
        verbose=verbose
    )
    
    if not all_keypoints:
        print("❌ No keypoints extracted!")
        return [], df.iloc[0:0], {}
    
    # Print stats
    print_keypoint_stats(all_keypoints, sequence_name=df['seq'].iloc[0] if 'seq' in df.columns else "Unknown")
    
    # Filter to valid frames
    valid_keypoints, valid_indices, stats = filter_valid_keypoints(all_keypoints, min_keypoints)
    
    # Create filtered DataFrame
    valid_df = df.iloc[valid_indices].reset_index(drop=True)
    
    print(f"✅ Filtered to {len(valid_keypoints)} valid frames (min {min_keypoints} keypoints)")
    print(f"   Removed: {stats['none'] + stats['empty'] + stats['low_quality']} frames")
    
    return valid_keypoints, valid_df, stats


# Example usage for notebook:
"""
# In your notebook, replace:
#   normal_keypoints_array = extractor.extract_from_sequence(...)
# 
# With:
#   normal_keypoints_array, normal_df_filtered, stats = extract_with_filtering(
#       normal_df, 
#       video_base_path, 
#       extractor,
#       min_keypoints=25,  # Require at least 25 out of 33 keypoints
#       verbose=True
#   )
#
# This will:
# 1. Extract keypoints from all frames
# 2. Show detailed statistics
# 3. Filter to only frames with good detections
# 4. Return filtered keypoints AND filtered DataFrame (aligned)
"""
