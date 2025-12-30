# Refactored MediaPipe Test Function with Parameters
# Allows testing different sequences and frames

def test_mediapipe_pose_detection(sequence_id=None, frame_index=None, frame_num=None, 
                                 use_bbox=True, show_visualization=True, 
                                 confidence_threshold=0.3, title_prefix="MediaPipe Pose"):
    """
    Flexible MediaPipe pose detection function with parameters
    
    Args:
        sequence_id (str, optional): Specific sequence ID to test. If None, uses first available sequence
        frame_index (int, optional): 0-based index within the sequence. If None, uses frame 10
        frame_num (int, optional): Specific frame number from video. Overrides frame_index if provided
        use_bbox (bool): Whether to show bounding box on visualization
        show_visualization (bool): Whether to display the 3-panel visualization
        confidence_threshold (float): Minimum confidence for drawing connections (default: 0.3)
        title_prefix (str): Prefix for the visualization title
    
    Returns:
        tuple: (keypoints, frame_rgb, metadata) or None if failed
            - keypoints: List of detected pose landmarks
            - frame_rgb: RGB image array
            - metadata: Dict with sequence info, frame info, etc.
    """
    
    if not pose_landmarker:
        print("❌ Pose landmarker not available")
        return None
    
    try:
        # Handle sequence selection
        if sequence_id is None:
            # Use first available sequence
            sequence_id = list(sequences.keys())[0]
            print(f"🔄 No sequence specified, using first available: {sequence_id}")
        elif sequence_id not in sequences:
            print(f"❌ Sequence '{sequence_id}' not found. Available sequences:")
            for seq_id in sequences.keys():
                print(f"   - {seq_id}")
            return None
        
        sequence_data = sequences[sequence_id]
        
        # Handle frame selection
        if frame_num is not None:
            # Find the row with the specific frame number
            matching_rows = sequence_data[sequence_data['frame_num'] == frame_num]
            if matching_rows.empty:
                print(f"❌ Frame {frame_num} not found in sequence {sequence_id}")
                print(f"   Available frame range: {sequence_data['frame_num'].min()} - {sequence_data['frame_num'].max()}")
                return None
            frame_row = matching_rows.iloc[0]
            actual_frame_index = matching_rows.index[0] - sequence_data.index[0]  # Relative index within sequence
        else:
            # Use frame index (default to 10 if not specified)
            if frame_index is None:
                frame_index = 10
                print(f"🔄 No frame specified, using frame index {frame_index}")
            
            if frame_index >= len(sequence_data):
                print(f"❌ Frame index {frame_index} out of range. Sequence has {len(sequence_data)} frames.")
                return None
            
            frame_row = sequence_data.iloc[frame_index]
            actual_frame_index = frame_index
        
        # Extract frame information
        actual_frame_num = int(frame_row['frame_num'])
        url = frame_row['url']
        bbox = frame_row.get('bbox', {}) if use_bbox else {}
        
        print(f"🎯 Processing sequence: {sequence_id}")
        print(f"   Frame index: {actual_frame_index} | Frame number: {actual_frame_num}")
        
        # Get video path
        video_id = extract_video_id(url)
        video_path = project_root / "data" / "youtube" / f"{video_id}.mp4"
        
        if not video_path.exists():
            print(f"❌ Video not found: {video_path}")
            return None
        
        # Extract frame from video
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            print(f"❌ Cannot open video: {video_path}")
            return None
            
        cap.set(cv2.CAP_PROP_POS_FRAMES, actual_frame_num - 1)  # Convert to 0-based
        ret, frame = cap.read()
        cap.release()
        
        if not ret:
            print(f"❌ Could not read frame {actual_frame_num}")
            return None
        
        # Convert to RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        print(f"📹 Frame extracted: {frame_rgb.shape}")
        
        # Use FULL frame for pose detection (critical for success)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
        
        # Detect pose on full frame
        print("🔍 Running pose detection on full frame...")
        detection_result = pose_landmarker.detect(mp_image)
        
        # Extract keypoints from full frame
        keypoints = []
        if detection_result.pose_landmarks:
            pose_landmarks = detection_result.pose_landmarks[0]
            height, width = frame_rgb.shape[:2]
            
            for i, landmark in enumerate(pose_landmarks):
                keypoint = {
                    'id': i,
                    'name': POSE_LANDMARK_NAMES[i],
                    'x': landmark.x * width,  # Full frame coordinates
                    'y': landmark.y * height,  # Full frame coordinates
                    'z': landmark.z,
                    'visibility': landmark.visibility,
                    'presence': landmark.presence,
                    'confidence': landmark.visibility,
                    'x_normalized': landmark.x,
                    'y_normalized': landmark.y
                }
                keypoints.append(keypoint)
            
            print(f"✅ Detected {len(keypoints)} landmarks")
        else:
            print("⚠️ No pose detected")
            keypoints = []
        
        # Create metadata
        metadata = {
            'sequence_id': sequence_id,
            'frame_index': actual_frame_index,
            'frame_num': actual_frame_num,
            'video_id': video_id,
            'video_path': str(video_path),
            'frame_shape': frame_rgb.shape,
            'bbox': bbox,
            'landmarks_detected': len(keypoints),
            'confidence_threshold': confidence_threshold
        }
        
        # Visualize if requested
        if show_visualization:
            if keypoints:
                title = f"{title_prefix} - Seq: {sequence_id[:8]}... | Frame: {actual_frame_num}"
                visualize_pose_with_skeleton_NO_EMOJI(frame_rgb, keypoints, bbox, title)
            else:
                # Show original image even if no pose detected
                plt.figure(figsize=(10, 8))
                plt.imshow(frame_rgb)
                plt.title(f"No Pose Detected - Frame {actual_frame_num}")
                plt.axis('off')
                plt.show()
        
        return keypoints, frame_rgb, metadata
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return None


def test_multiple_frames(sequence_id=None, frame_indices=None, frame_nums=None, 
                        max_frames=5, show_each=True):
    """
    Test multiple frames from a sequence
    
    Args:
        sequence_id (str, optional): Sequence to test. If None, uses first available
        frame_indices (list, optional): List of frame indices to test
        frame_nums (list, optional): List of specific frame numbers to test
        max_frames (int): Maximum number of frames to test if using auto-selection
        show_each (bool): Whether to show visualization for each frame
    
    Returns:
        list: List of results from each frame test
    """
    
    if sequence_id is None:
        sequence_id = list(sequences.keys())[0]
    
    if sequence_id not in sequences:
        print(f"❌ Sequence '{sequence_id}' not found")
        return []
    
    sequence_data = sequences[sequence_id]
    results = []
    
    # Determine which frames to test
    if frame_nums is not None:
        # Test specific frame numbers
        test_frames = [(None, fn) for fn in frame_nums]  # (frame_index, frame_num)
        print(f"🎯 Testing {len(frame_nums)} specific frame numbers: {frame_nums}")
    elif frame_indices is not None:
        # Test specific frame indices
        test_frames = [(fi, None) for fi in frame_indices]
        print(f"🎯 Testing {len(frame_indices)} specific frame indices: {frame_indices}")
    else:
        # Auto-select frames evenly distributed across sequence
        total_frames = len(sequence_data)
        if total_frames <= max_frames:
            indices = list(range(total_frames))
        else:
            # Evenly distribute frames
            step = total_frames // max_frames
            indices = [i * step for i in range(max_frames)]
        
        test_frames = [(fi, None) for fi in indices]
        print(f"🎯 Auto-selected {len(indices)} frames from sequence (total: {total_frames})")
    
    # Test each frame
    for i, (frame_idx, frame_num) in enumerate(test_frames):
        print(f"\n--- Testing frame {i+1}/{len(test_frames)} ---")
        
        result = test_mediapipe_pose_detection(
            sequence_id=sequence_id,
            frame_index=frame_idx,
            frame_num=frame_num,
            show_visualization=show_each,
            title_prefix=f"Frame {i+1}/{len(test_frames)}"
        )
        
        if result:
            keypoints, frame_rgb, metadata = result
            results.append({
                'frame_index': metadata['frame_index'],
                'frame_num': metadata['frame_num'],
                'landmarks_count': len(keypoints),
                'avg_confidence': np.mean([kp['confidence'] for kp in keypoints]) if keypoints else 0,
                'metadata': metadata
            })
        else:
            results.append({
                'frame_index': frame_idx,
                'frame_num': frame_num,
                'landmarks_count': 0,
                'avg_confidence': 0,
                'metadata': None
            })
    
    # Summary
    print(f"\n📊 SUMMARY - Tested {len(results)} frames from sequence {sequence_id}")
    successful = sum(1 for r in results if r['landmarks_count'] > 0)
    print(f"   ✅ Successful detections: {successful}/{len(results)}")
    
    if successful > 0:
        avg_landmarks = np.mean([r['landmarks_count'] for r in results if r['landmarks_count'] > 0])
        avg_confidence = np.mean([r['avg_confidence'] for r in results if r['avg_confidence'] > 0])
        print(f"   📈 Average landmarks detected: {avg_landmarks:.1f}")
        print(f"   📈 Average confidence: {avg_confidence:.3f}")
    
    return results


def explore_all_sequences(max_frames_per_seq=3, show_visualizations=True):
    """
    Explore pose detection across all available sequences
    
    Args:
        max_frames_per_seq (int): Maximum frames to test per sequence
        show_visualizations (bool): Whether to show visualizations
    
    Returns:
        dict: Results organized by sequence
    """
    
    all_results = {}
    
    print(f"🔍 Exploring pose detection across {len(sequences)} sequences")
    print(f"   Testing up to {max_frames_per_seq} frames per sequence")
    
    for i, seq_id in enumerate(sequences.keys()):
        print(f"\n{'='*60}")
        print(f"SEQUENCE {i+1}/{len(sequences)}: {seq_id}")
        print(f"{'='*60}")
        
        results = test_multiple_frames(
            sequence_id=seq_id,
            max_frames=max_frames_per_seq,
            show_each=show_visualizations
        )
        
        all_results[seq_id] = results
    
    # Overall summary
    print(f"\n{'='*60}")
    print("OVERALL SUMMARY")
    print(f"{'='*60}")
    
    total_tests = sum(len(results) for results in all_results.values())
    total_successful = sum(sum(1 for r in results if r['landmarks_count'] > 0) 
                          for results in all_results.values())
    
    print(f"📊 Total tests: {total_tests}")
    print(f"✅ Successful detections: {total_successful}/{total_tests} ({100*total_successful/total_tests:.1f}%)")
    
    # Per-sequence summary
    for seq_id, results in all_results.items():
        successful = sum(1 for r in results if r['landmarks_count'] > 0)
        print(f"   {seq_id[:20]}...: {successful}/{len(results)} successful")
    
    return all_results


print("✅ Refactored MediaPipe test functions ready!")
print("📋 Available functions:")
print("   - test_mediapipe_pose_detection(): Test single frame with parameters")
print("   - test_multiple_frames(): Test multiple frames from a sequence") 
print("   - explore_all_sequences(): Test across all sequences")