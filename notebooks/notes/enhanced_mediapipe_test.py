# Enhanced MediaPipe Test Function with Proper Stick Figure Visualization

enhanced_test_function = '''
def test_new_mediapipe_with_skeleton():
    """
    Test pose detection with GAVD video data and draw proper stick figure skeleton
    Based on MediaPipe's official pose landmark connections
    """
    
    if not pose_landmarker:
        print("❌ Pose landmarker not available")
        return None
    
    try:
        # Get first sequence and frame
        first_seq_id = list(sequences.keys())[0]
        frame_row = sequences[first_seq_id].iloc[10]
        frame_num = int(frame_row['frame_num'])
        url = frame_row['url']
        bbox = frame_row.get('bbox', {})
        
        print(f"🎯 Processing frame {frame_num} from sequence {first_seq_id}")
        
        # Get video path
        video_id = extract_video_id(url)
        video_path = project_root / "data" / "youtube" / f"{video_id}.mp4"
        
        if not video_path.exists():
            print(f"❌ Video not found: {video_path}")
            return None
        
        # Extract frame
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            print(f"❌ Cannot open video: {video_path}")
            return None
            
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num - 1)
        ret, frame = cap.read()
        cap.release()
        
        if not ret:
            print(f"❌ Could not read frame {frame_num}")
            return None
        
        # Convert to RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        print(f"📹 Frame extracted: {frame_rgb.shape}")
        
        # Apply bounding box crop if available
        cropped_image = frame_rgb
        crop_offset = (0, 0)
        
        if bbox and isinstance(bbox, dict):
            left = max(0, int(bbox.get('left', 0)))
            top = max(0, int(bbox.get('top', 0)))
            width = int(bbox.get('width', frame_rgb.shape[1]))
            height = int(bbox.get('height', frame_rgb.shape[0]))
            
            right = min(frame_rgb.shape[1], left + width)
            bottom = min(frame_rgb.shape[0], top + height)
            
            if width > 0 and height > 0:
                cropped_image = frame_rgb[top:bottom, left:right]
                crop_offset = (left, top)
                print(f"📏 Applied bbox crop: ({left},{top}) to ({right},{bottom})")
        
        # Convert to MediaPipe Image
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=cropped_image)
        
        # Detect pose
        print("🔍 Running pose detection...")
        detection_result = pose_landmarker.detect(mp_image)
        
        # Extract keypoints
        keypoints = []
        if detection_result.pose_landmarks:
            pose_landmarks = detection_result.pose_landmarks[0]
            height, width = cropped_image.shape[:2]
            
            for i, landmark in enumerate(pose_landmarks):
                keypoint = {
                    'id': i,
                    'name': POSE_LANDMARK_NAMES[i],
                    'x': landmark.x * width + crop_offset[0],  # Adjust for crop
                    'y': landmark.y * height + crop_offset[1],  # Adjust for crop
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
        
        # Visualize with proper stick figure
        if keypoints:
            visualize_pose_with_skeleton(frame_rgb, keypoints, bbox, 
                                       f"MediaPipe Pose - Frame {frame_num}")
        else:
            # Show original image even if no pose detected
            plt.figure(figsize=(10, 8))
            plt.imshow(frame_rgb)
            plt.title(f"No Pose Detected - Frame {frame_num}")
            plt.axis('off')
            plt.show()
        
        return keypoints, frame_rgb
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return None


def visualize_pose_with_skeleton(image, keypoints, bbox=None, title="Pose Detection"):
    """
    Visualize pose with proper stick figure skeleton using MediaPipe connections
    
    Args:
        image: Original RGB image
        keypoints: List of keypoint dictionaries
        bbox: Optional bounding box
        title: Plot title
    """
    
    # MediaPipe Pose Connections (Official)
    # Based on MediaPipe's pose_connections.py and pose landmark model
    POSE_CONNECTIONS = [
        # Face connections
        (0, 1),   # nose to left_eye_inner
        (1, 2),   # left_eye_inner to left_eye
        (2, 3),   # left_eye to left_eye_outer
        (3, 7),   # left_eye_outer to left_ear
        (0, 4),   # nose to right_eye_inner
        (4, 5),   # right_eye_inner to right_eye
        (5, 6),   # right_eye to right_eye_outer
        (6, 8),   # right_eye_outer to right_ear
        (9, 10),  # mouth_left to mouth_right
        
        # Upper body connections
        (11, 12), # left_shoulder to right_shoulder
        (11, 13), # left_shoulder to left_elbow
        (13, 15), # left_elbow to left_wrist
        (12, 14), # right_shoulder to right_elbow
        (14, 16), # right_elbow to right_wrist
        
        # Hand connections
        (15, 17), # left_wrist to left_pinky
        (15, 19), # left_wrist to left_index
        (15, 21), # left_wrist to left_thumb
        (17, 19), # left_pinky to left_index
        (16, 18), # right_wrist to right_pinky
        (16, 20), # right_wrist to right_index
        (16, 22), # right_wrist to right_thumb
        (18, 20), # right_pinky to right_index
        
        # Torso connections
        (11, 23), # left_shoulder to left_hip
        (12, 24), # right_shoulder to right_hip
        (23, 24), # left_hip to right_hip
        
        # Leg connections
        (23, 25), # left_hip to left_knee
        (25, 27), # left_knee to left_ankle
        (24, 26), # right_hip to right_knee
        (26, 28), # right_knee to right_ankle
        
        # Foot connections
        (27, 29), # left_ankle to left_heel
        (27, 31), # left_ankle to left_foot_index
        (29, 31), # left_heel to left_foot_index
        (28, 30), # right_ankle to right_heel
        (28, 32), # right_ankle to right_foot_index
        (30, 32), # right_heel to right_foot_index
    ]
    
    # Create figure with subplots
    fig, axes = plt.subplots(1, 3, figsize=(20, 7))
    
    # 1. Original image
    axes[0].imshow(image)
    axes[0].set_title('Original Image', fontsize=14)
    axes[0].axis('off')
    
    # Add bounding box to original if provided
    if bbox and isinstance(bbox, dict):
        left = bbox.get('left', 0)
        top = bbox.get('top', 0)
        width = bbox.get('width', 0)
        height = bbox.get('height', 0)
        
        if width > 0 and height > 0:
            from matplotlib.patches import Rectangle
            rect = Rectangle((left, top), width, height, 
                           linewidth=2, edgecolor='red', facecolor='none')
            axes[0].add_patch(rect)
    
    # 2. Keypoints only
    keypoints_image = image.copy()
    
    # Draw keypoints with different colors based on confidence
    high_conf_count = 0
    medium_conf_count = 0
    low_conf_count = 0
    
    for kp in keypoints:
        confidence = kp.get('confidence', 0)
        if confidence > 0.1:  # Only draw visible keypoints
            x, y = int(kp['x']), int(kp['y'])
            
            # Color and size based on confidence
            if confidence > 0.8:
                color = (0, 255, 0)      # Bright green - very high confidence
                radius = 6
                high_conf_count += 1
            elif confidence > 0.6:
                color = (50, 255, 50)    # Green - high confidence
                radius = 5
                high_conf_count += 1
            elif confidence > 0.4:
                color = (255, 255, 0)    # Yellow - medium confidence
                radius = 4
                medium_conf_count += 1
            elif confidence > 0.2:
                color = (255, 165, 0)    # Orange - low confidence
                radius = 3
                low_conf_count += 1
            else:
                color = (255, 100, 100)  # Light red - very low confidence
                radius = 2
                low_conf_count += 1
            
            # Draw keypoint
            cv2.circle(keypoints_image, (x, y), radius, color, -1)
            cv2.circle(keypoints_image, (x, y), radius + 1, (255, 255, 255), 1)  # White border
            
            # Add landmark number for key points
            if kp['id'] in [0, 11, 12, 15, 16, 23, 24, 27, 28]:  # Key landmarks
                cv2.putText(keypoints_image, str(kp['id']), (x + 8, y - 8),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
    
    axes[1].imshow(keypoints_image)
    axes[1].set_title(f'Keypoints Only\\n🟢 High: {high_conf_count} 🟡 Med: {medium_conf_count} 🟠 Low: {low_conf_count}', 
                     fontsize=12)
    axes[1].axis('off')
    
    # 3. Full skeleton with connections
    skeleton_image = image.copy()
    
    # Draw connections first (so they appear behind keypoints)
    connections_drawn = 0
    for start_idx, end_idx in POSE_CONNECTIONS:
        if start_idx < len(keypoints) and end_idx < len(keypoints):
            start_kp = keypoints[start_idx]
            end_kp = keypoints[end_idx]
            
            # Only draw connection if both keypoints are reasonably confident
            if (start_kp.get('confidence', 0) > 0.3 and 
                end_kp.get('confidence', 0) > 0.3):
                
                start_point = (int(start_kp['x']), int(start_kp['y']))
                end_point = (int(end_kp['x']), int(end_kp['y']))
                
                # Line color and thickness based on average confidence
                avg_conf = (start_kp.get('confidence', 0) + end_kp.get('confidence', 0)) / 2
                
                if avg_conf > 0.7:
                    color = (0, 255, 255)    # Cyan - high confidence connection
                    thickness = 3
                elif avg_conf > 0.5:
                    color = (100, 255, 255)  # Light cyan - medium confidence
                    thickness = 2
                else:
                    color = (150, 150, 255)  # Light blue - low confidence
                    thickness = 1
                
                cv2.line(skeleton_image, start_point, end_point, color, thickness)
                connections_drawn += 1
    
    # Draw keypoints on top of connections
    for kp in keypoints:
        confidence = kp.get('confidence', 0)
        if confidence > 0.2:
            x, y = int(kp['x']), int(kp['y'])
            
            # Keypoint color based on confidence
            if confidence > 0.7:
                color = (255, 0, 0)      # Red - high confidence
                radius = 5
            elif confidence > 0.5:
                color = (255, 100, 0)    # Orange-red - medium confidence
                radius = 4
            else:
                color = (255, 150, 150)  # Pink - low confidence
                radius = 3
            
            cv2.circle(skeleton_image, (x, y), radius, color, -1)
            cv2.circle(skeleton_image, (x, y), radius + 1, (255, 255, 255), 1)
    
    axes[2].imshow(skeleton_image)
    axes[2].set_title(f'Full Skeleton\\n{connections_drawn} connections drawn', fontsize=12)
    axes[2].axis('off')
    
    # Add overall title
    fig.suptitle(title, fontsize=16, y=0.98)
    plt.tight_layout()
    plt.show()
    
    # Print detailed statistics
    if keypoints:
        confidences = [kp.get('confidence', 0) for kp in keypoints]
        visible_landmarks = sum(1 for c in confidences if c > 0.5)
        avg_confidence = np.mean(confidences)
        
        print(f"\\n📊 Detailed Pose Analysis:")
        print(f"   🎯 Total landmarks: {len(keypoints)}")
        print(f"   👁️  Visible landmarks (>0.5): {visible_landmarks}")
        print(f"   🟢 High confidence (>0.7): {high_conf_count}")
        print(f"   🟡 Medium confidence (0.4-0.7): {medium_conf_count}")
        print(f"   🟠 Low confidence (0.2-0.4): {low_conf_count}")
        print(f"   📈 Average confidence: {avg_confidence:.3f}")
        print(f"   🔗 Skeleton connections drawn: {connections_drawn}")
        
        # Body part analysis
        face_landmarks = keypoints[0:11]
        upper_body_landmarks = keypoints[11:23]
        lower_body_landmarks = keypoints[23:33]
        
        face_conf = np.mean([kp['confidence'] for kp in face_landmarks])
        upper_conf = np.mean([kp['confidence'] for kp in upper_body_landmarks])
        lower_conf = np.mean([kp['confidence'] for kp in lower_body_landmarks])
        
        print(f"\\n🧍 Body Part Confidence:")
        print(f"   😊 Face (0-10): {face_conf:.3f}")
        print(f"   💪 Upper body (11-22): {upper_conf:.3f}")
        print(f"   🦵 Lower body (23-32): {lower_conf:.3f}")
    
    return fig

print("✅ Enhanced MediaPipe test function with skeleton visualization ready!")
'''

print("COPY THIS ENHANCED FUNCTION INTO YOUR JUPYTER NOTEBOOK:")
print("=" * 60)
print(enhanced_test_function)