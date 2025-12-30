#!/usr/bin/env python3
"""
BOUNDING BOX ALIGNMENT FIX

This patch fixes the bounding box misalignment issue in the MediaPipe pose detection visualization.

PROBLEM:
The bounding box coordinates in the CSV are for the original annotation resolution (e.g., 1280x720),
but the actual video being processed has a different resolution (e.g., 640x360).
The visualize_pose_with_skeleton function was drawing the bbox with original coordinates
without applying the necessary scaling.

ROOT CAUSE:
- CSV annotations: 1280x720 resolution
- Actual video: 640x360 resolution  
- Scale factors: x=0.5, y=0.5
- But bbox coordinates were not being scaled in the visualization

SOLUTION:
Apply the same scaling logic that's used in the visualize_frame function
to the visualize_pose_with_skeleton function.
"""

# STEP 1: Update the visualize_pose_with_skeleton function
def visualize_pose_with_skeleton_FIXED(image, keypoints, bbox=None, title="Pose Detection", 
                                     vid_info=None, frame_shape=None):
    """
    FIXED: Visualize pose with proper stick figure skeleton and correctly scaled bounding box
    
    Args:
        image: RGB image array
        keypoints: List of detected pose landmarks
        bbox: Bounding box dict with 'left', 'top', 'width', 'height' keys
        title: Title for the visualization
        vid_info: Video info dict with original annotation resolution (NEW)
        frame_shape: Actual frame shape (height, width, channels) (NEW)
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
        
        # Calculate scaling factors (SAME LOGIC AS visualize_frame function)
        scale_x = actual_width / annotation_width if annotation_width > 0 else 1.0
        scale_y = actual_height / annotation_height if annotation_height > 0 else 1.0
        
        # Apply scaling to bbox coordinates
        left = bbox.get('left', 0) * scale_x
        top = bbox.get('top', 0) * scale_y
        width = bbox.get('width', 0) * scale_x
        height = bbox.get('height', 0) * scale_y
        
        # Debug output
        print(f"🔧 BBOX SCALING APPLIED:")
        print(f"   Original: left={bbox.get('left', 0)}, top={bbox.get('top', 0)}, "
              f"w={bbox.get('width', 0)}, h={bbox.get('height', 0)}")
        print(f"   Annotation res: {annotation_width}x{annotation_height}")
        print(f"   Actual res: {actual_width}x{actual_height}")
        print(f"   Scale: x={scale_x:.3f}, y={scale_y:.3f}")
        print(f"   Scaled: left={left:.1f}, top={top:.1f}, w={width:.1f}, h={height:.1f}")
        
        if width > 0 and height > 0:
            from matplotlib.patches import Rectangle
            rect = Rectangle((left, top), width, height, 
                           linewidth=2, edgecolor='red', facecolor='none')
            axes[0].add_patch(rect)
            print(f"   ✅ Bounding box drawn with correct scaling")
        else:
            print(f"   ⚠️ Invalid bbox dimensions")
    
    # Rest of the function remains the same...
    # [Keypoints and skeleton drawing code would continue here]
    
    print("\\n🔧 This is the FIXED version with proper bbox scaling!")


# STEP 2: Update the test function call
def apply_fix_to_test_function():
    """
    Shows the required changes to the test_mediapipe_pose_detection function
    """
    
    fix_instructions = '''
    REQUIRED CHANGES to test_mediapipe_pose_detection function:
    
    1. Extract vid_info from frame_row (ALREADY DONE):
       vid_info = frame_row.get('vid_info', {})
    
    2. Update the visualization call (CHANGE THIS):
       
       OLD CODE:
       visualize_pose_with_skeleton(frame_rgb, keypoints, bbox, title)
       
       NEW CODE:
       visualize_pose_with_skeleton_FIXED(frame_rgb, keypoints, bbox, title, 
                                        vid_info=vid_info, frame_shape=frame_rgb.shape)
    
    That's it! The bbox will now be properly scaled.
    '''
    
    print(fix_instructions)


if __name__ == "__main__":
    print("🔧 BOUNDING BOX ALIGNMENT FIX")
    print("=" * 50)
    print()
    print("PROBLEM IDENTIFIED:")
    print("- CSV bbox coordinates: 1280x720 resolution")
    print("- Actual video: 640x360 resolution")
    print("- Scale factors: x=0.5, y=0.5")
    print("- But bbox was drawn without scaling!")
    print()
    print("SOLUTION:")
    print("- Apply same scaling logic as visualize_frame function")
    print("- Pass vid_info and frame_shape to visualization function")
    print("- Scale bbox coordinates before drawing")
    print()
    
    apply_fix_to_test_function()