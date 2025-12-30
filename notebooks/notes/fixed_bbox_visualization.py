#!/usr/bin/env python3
"""
Fixed Bounding Box Visualization

This script fixes the bounding box alignment issue in the MediaPipe pose detection visualization.

PROBLEM IDENTIFIED:
The bounding box coordinates in the CSV are for the original annotation resolution (e.g., 1280x720),
but the actual video being processed has a different resolution (e.g., 640x360).
The visualize_pose_with_skeleton function was drawing the bbox with original coordinates
without applying the necessary scaling.

SOLUTION:
1. Pass video resolution info to the visualization function
2. Apply the same scaling logic used in visualize_frame function
3. Scale bbox coordinates to match the actual video resolution

FIXES APPLIED:
- Modified visualize_pose_with_skeleton to accept vid_info and frame_shape parameters
- Added bbox coordinate scaling logic
- Updated test function to pass the necessary scaling information
"""

import cv2
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle


def visualize_pose_with_skeleton_fixed(image, keypoints, bbox=None, title="Pose Detection", 
                                     vid_info=None, frame_shape=None):
    """
    FIXED: Visualize pose with proper stick figure skeleton and correctly scaled bounding box
    
    Args:
        image: RGB image array
        keypoints: List of detected pose landmarks
        bbox: Bounding box dict with 'left', 'top', 'width', 'height' keys
        title: Title for the visualization
        vid_info: Video info dict with original annotation resolution
        frame_shape: Actual frame shape (height, width, channels)
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
    
    # FIXED: Add properly scaled bounding box to original
    if bbox and isinstance(bbox, dict):
        # Get actual frame dimensions
        if frame_shape is not None:
            actual_height, actual_width = frame_shape[:2]
        else:
            actual_height, actual_width = image.shape[:2]
        
        # Get annotation dimensions from vid_info
        if vid_info and isinstance(vid_info, dict):
            annotation_width = vid_info.get('width', actual_width)
            annotation_height = vid_info.get('height', actual_height)
        else:
            annotation_width = actual_width
            annotation_height = actual_height
        
        # Calculate scaling factors
        scale_x = actual_width / annotation_width if annotation_width > 0 else 1.0
        scale_y = actual_height / annotation_height if annotation_height > 0 else 1.0
        
        # Apply scaling to bbox coordinates
        left = bbox.get('left', 0) * scale_x
        top = bbox.get('top', 0) * scale_y
        width = bbox.get('width', 0) * scale_x
        height = bbox.get('height', 0) * scale_y
        
        # Debug output
        print(f"🔧 BBOX SCALING DEBUG:")
        print(f"   Original bbox: left={bbox.get('left', 0)}, top={bbox.get('top', 0)}, "
              f"width={bbox.get('width', 0)}, height={bbox.get('height', 0)}")
        print(f"   Annotation resolution: {annotation_width}x{annotation_height}")
        print(f"   Actual resolution: {actual_width}x{actual_height}")
        print(f"   Scale factors: x={scale_x:.3f}, y={scale_y:.3f}")
        print(f"   Scaled bbox: left={left:.1f}, top={top:.1f}, width={width:.1f}, height={height:.1f}")
        
        if width > 0 and height > 0:
            rect = Rectangle((left, top), width, height, 
                           linewidth=2, edgecolor='red', facecolor='none')
            axes[0].add_patch(rect)
            print(f"   ✅ Bounding box drawn with scaling applied")
        else:
            print(f"   ⚠️ Invalid bbox dimensions after scaling")
    else:
        print(f"   ℹ️ No bounding box to draw")
    
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
    axes[1].set_title(f'Keypoints Only\\nHigh: {high_conf_count} | Med: {medium_conf_count} | Low: {low_conf_count}', 
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
        
        print(f"\\nDetailed Pose Analysis:")
        print(f"   Total landmarks: {len(keypoints)}")
        print(f"   Visible landmarks (>0.5): {visible_landmarks}")
        print(f"   High confidence (>0.7): {high_conf_count}")
        print(f"   Medium confidence (0.4-0.7): {medium_conf_count}")
        print(f"   Low confidence (0.2-0.4): {low_conf_count}")
        print(f"   Average confidence: {avg_confidence:.3f}")
        print(f"   Skeleton connections drawn: {connections_drawn}")


def test_mediapipe_pose_detection_fixed(sequence_id=None, frame_index=None, frame_num=None, 
                                       use_bbox=True, show_visualization=True, 
                                       confidence_threshold=0.3, title_prefix="MediaPipe Pose FIXED"):
    """
    FIXED: MediaPipe pose detection function with proper bbox scaling
    
    This version fixes the bounding box alignment issue by:
    1. Extracting vid_info from the frame data
    2. Passing both vid_info and frame_shape to the visualization function
    3. Applying proper coordinate scaling in the visualization
    """
    
    # This would need to be imported from the actual notebook context
    # For now, this is a template showing the fix
    
    print("🔧 FIXED VERSION - This function template shows the required changes:")
    print("1. Extract vid_info from frame_row")
    print("2. Pass vid_info and frame_shape to visualization function")
    print("3. Apply scaling in visualize_pose_with_skeleton_fixed")
    
    # Example of the key changes needed:
    example_code = '''
    # In the test function, after extracting frame_row:
    vid_info = frame_row.get('vid_info', {})
    
    # When calling visualization:
    if show_visualization and keypoints:
        title = f"{title_prefix} - Seq: {sequence_id[:8]}... | Frame: {actual_frame_num}"
        visualize_pose_with_skeleton_fixed(frame_rgb, keypoints, bbox, title, 
                                         vid_info=vid_info, frame_shape=frame_rgb.shape)
    '''
    
    print("\\nRequired code changes:")
    print(example_code)


if __name__ == "__main__":
    print("🔧 Fixed Bounding Box Visualization")
    print("This script contains the corrected visualization function.")
    print("\\nKey fixes:")
    print("1. Added vid_info and frame_shape parameters to visualization function")
    print("2. Implemented proper bbox coordinate scaling")
    print("3. Added debug output to verify scaling calculations")
    print("\\nTo apply the fix:")
    print("1. Replace visualize_pose_with_skeleton with visualize_pose_with_skeleton_fixed")
    print("2. Update test function to pass vid_info and frame_shape parameters")
    print("3. Extract vid_info from frame_row in the test function")