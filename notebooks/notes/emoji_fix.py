# Fix for matplotlib font warnings - Replace emoji with text
# The warnings are caused by Unicode emoji characters not supported by DejaVu Sans font on Windows

# ORIGINAL PROBLEMATIC LINES:
# axes[1].set_title(f'Keypoints Only\n🟢 High: {high_conf_count} 🟡 Med: {medium_conf_count} 🟠 Low: {low_conf_count}')
# print(f"   🟢 High confidence (>0.7): {high_conf_count}")
# print(f"   🟡 Medium confidence (0.4-0.7): {medium_conf_count}")
# print(f"   🟠 Low confidence (0.2-0.4): {low_conf_count}")

# FIXED VERSIONS (replace emoji with simple text):

def visualize_pose_with_skeleton_NO_EMOJI(image, keypoints, bbox=None, title="Pose Detection"):
    """
    Visualize pose with proper stick figure skeleton using MediaPipe connections
    FIXED: Removed emoji characters to prevent matplotlib font warnings
    
    Args:
        image: Original RGB image
        keypoints: List of keypoint dictionaries
        bbox: Optional bounding box
        title: Plot title
    """
    
    # MediaPipe Pose Connections (Official)
    POSE_CONNECTIONS = [
        # Face connections
        (0, 1), (1, 2), (2, 3), (3, 7), (0, 4), (4, 5), (5, 6), (6, 8), (9, 10),
        # Upper body connections
        (11, 12), (11, 13), (13, 15), (12, 14), (14, 16),
        # Hand connections
        (15, 17), (15, 19), (15, 21), (17, 19), (16, 18), (16, 20), (16, 22), (18, 20),
        # Torso connections
        (11, 23), (12, 24), (23, 24),
        # Leg connections
        (23, 25), (25, 27), (24, 26), (26, 28),
        # Foot connections
        (27, 29), (27, 31), (29, 31), (28, 30), (28, 32), (30, 32),
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
    # FIXED: Removed emoji characters to prevent font warnings
    axes[1].set_title(f'Keypoints Only\nHigh: {high_conf_count} | Med: {medium_conf_count} | Low: {low_conf_count}', 
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
    axes[2].set_title(f'Full Skeleton\n{connections_drawn} connections drawn', fontsize=12)
    axes[2].axis('off')
    
    # Add overall title
    fig.suptitle(title, fontsize=16, y=0.98)
    plt.tight_layout()
    plt.show()
    
    # Print detailed statistics - FIXED: Removed emoji characters
    if keypoints:
        confidences = [kp.get('confidence', 0) for kp in keypoints]
        visible_landmarks = sum(1 for c in confidences if c > 0.5)
        avg_confidence = np.mean(confidences)
        
        print(f"\nDetailed Pose Analysis:")
        print(f"   Total landmarks: {len(keypoints)}")
        print(f"   Visible landmarks (>0.5): {visible_landmarks}")
        print(f"   High confidence (>0.7): {high_conf_count}")
        print(f"   Medium confidence (0.4-0.7): {medium_conf_count}")
        print(f"   Low confidence (0.2-0.4): {low_conf_count}")
        print(f"   Average confidence: {avg_confidence:.3f}")
        print(f"   Skeleton connections drawn: {connections_drawn}")
        
        # Body part analysis
        face_landmarks = keypoints[0:11]
        upper_body_landmarks = keypoints[11:23]
        lower_body_landmarks = keypoints[23:33]
        
        face_conf = np.mean([kp['confidence'] for kp in face_landmarks])
        upper_conf = np.mean([kp['confidence'] for kp in upper_body_landmarks])
        lower_conf = np.mean([kp['confidence'] for kp in lower_body_landmarks])
        
        print(f"\nBody Part Confidence:")
        print(f"   Face (0-10): {face_conf:.3f}")
        print(f"   Upper body (11-22): {upper_conf:.3f}")
        print(f"   Lower body (23-32): {lower_conf:.3f}")
    
    return fig

print("✅ Fixed visualization function without emoji characters!")
print("This will prevent matplotlib font warnings on Windows systems.")