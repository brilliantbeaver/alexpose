"""
Visualization Utilities for Pose Analysis

This module provides visualization functions for pose analysis, including:
- Frame visualization with bounding boxes
- Pose skeleton visualization
- Bounding box scaling and alignment

Originally located in notebooks/utils/, moved to ambient/utils/ for better
organization and to make it available as part of the core package.
"""

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional

from ambient.utils.youtube_cache import extract_video_id
from ambient.gavd import PoseDataConverter


def _scale_bbox_coordinates(
    bbox: Dict, 
    annotation_width: int, 
    annotation_height: int, 
    actual_width: int, 
    actual_height: int
) -> Tuple[int, int, int, int]:
    """
    Scale bounding box coordinates from annotation resolution to actual video resolution.
    
    Args:
        bbox: Dictionary containing bbox coordinates (left, top, width, height)
        annotation_width: Width used in the annotation
        annotation_height: Height used in the annotation
        actual_width: Actual video width
        actual_height: Actual video height
        
    Returns:
        Tuple of (left, top, width, height) scaled to actual video resolution
    """
    if annotation_width != actual_width or annotation_height != actual_height:
        scale_x = actual_width / annotation_width if annotation_width > 0 else 1.0
        scale_y = actual_height / annotation_height if annotation_height > 0 else 1.0
        left = int(bbox.get('left', 0) * scale_x)
        top = int(bbox.get('top', 0) * scale_y)
        width = int(bbox.get('width', 0) * scale_x)
        height = int(bbox.get('height', 0) * scale_y)
        print(f"Scaled bbox: annotation={annotation_width}x{annotation_height}, "
              f"video={actual_width}x{actual_height}, scale=({scale_x:.3f}, {scale_y:.3f})")
    else:
        left = int(bbox.get('left', 0))
        top = int(bbox.get('top', 0))
        width = int(bbox.get('width', 0))
        height = int(bbox.get('height', 0))
    
    return left, top, width, height


def draw_bounding_box(
    frame: np.ndarray,
    bbox: Optional[Dict],
    annotation_width: int,
    annotation_height: int,
    color: Tuple[int, int, int] = (255, 0, 0),
    thickness: int = 2,
    verbose: bool = True
) -> np.ndarray:
    """
    Draw a bounding box on a frame with proper scaling.
    
    Args:
        frame: RGB frame array to draw on
        bbox: Dictionary containing bbox coordinates (left, top, width, height)
        annotation_width: Width used in the annotation
        annotation_height: Height used in the annotation
        color: RGB color tuple for the bounding box (default: red)
        thickness: Line thickness for the bounding box
        verbose: Whether to print debug information
        
    Returns:
        Frame with bounding box drawn (modifies input frame in-place)
    """
    if not isinstance(bbox, dict):
        if verbose:
            print("No valid bounding box data provided")
        return frame
    
    actual_height, actual_width = frame.shape[:2]
    
    # Scale bbox coordinates
    left, top, width, height = _scale_bbox_coordinates(
        bbox, annotation_width, annotation_height, actual_width, actual_height
    )
    
    # Draw rectangle if dimensions are valid
    if width > 0 and height > 0:
        cv2.rectangle(frame, (left, top), (left + width, top + height), color, thickness)
        if verbose:
            print(f"Bbox drawn: left={left}, top={top}, width={width}, height={height}")
    elif verbose:
        print(f"Invalid bbox dimensions: width={width}, height={height}")
    
    return frame


def visualize_frame(
    project_root: str,
    sequences: Dict[str, pd.DataFrame],
    seq_id: str,
    frame_index: int = 0, 
    show_bbox: bool = True,
    bbox_color: Tuple[int, int, int] = (255, 0, 0),
    bbox_thickness: int = 2
):
    """
    Visualize a specific frame from a sequence with proper bbox alignment.
    
    FIXES APPLIED:
    1. Uses FFmpeg for precise frame extraction (more accurate than OpenCV seeking)
    2. Scales bbox coordinates if video resolution differs from annotation
    3. Validates frame number and video properties
    
    Args:
        project_root: Path to the project root directory
        sequences: Dictionary of sequences (from organize_by_sequence)
        seq_id: Sequence ID to visualize
        frame_index: Index of the frame within the sequence (0-based)
        show_bbox: Whether to draw the bounding box on the frame
        bbox_color: RGB color tuple for the bounding box (default: red)
        bbox_thickness: Line thickness for the bounding box
    """
    if seq_id not in sequences:
        print(f"Sequence {seq_id} not found. Available sequences: {list(sequences.keys())}")
        return
    
    seq_data = sequences[seq_id]
    
    if frame_index >= len(seq_data):
        print(f"Frame index {frame_index} out of range. Sequence has {len(seq_data)} frames.")
        return
    
    # Get the frame row
    frame_row = seq_data.iloc[frame_index]
    frame_num = int(frame_row['frame_num'])
    url = frame_row.get('url', '')
    bbox = frame_row.get('bbox', {})
    vid_info = frame_row.get('vid_info', {})
    
    # Extract video ID and find cached video
    video_id = extract_video_id(url) if url else None
    if not video_id:
        print(f"No URL found for frame {frame_index}")
        return
    
    # Look for cached video file
    video_cache_dir = project_root / "data" / "youtube"
    video_path = None
    for ext in ['.mp4', '.webm', '.mkv', '.mov']:
        candidate = video_cache_dir / f"{video_id}{ext}"
        if candidate.exists():
            video_path = candidate
            break
    
    if not video_path:
        print(f"Video not found in cache. Expected: {video_cache_dir / f'{video_id}.mp4'}")
        return
    
    # Get actual video properties
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"Failed to open video: {video_path}")
        return
    
    video_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    video_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    video_frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    
    # Validate frame number
    frame_index_0based = frame_num - 1  # Convert 1-based to 0-based
    if frame_index_0based < 0 or frame_index_0based >= video_frame_count:
        print(f"WARNING: Frame {frame_num} is out of video range (0-{video_frame_count-1})")
        print(f"Using frame index {frame_index_0based} (may cause misalignment)")
    
    # Extract frame using FFmpeg for precision (more accurate than OpenCV seeking)
    converter = PoseDataConverter(video_cache_dir=str(video_cache_dir))
    try:
        import tempfile
        import shutil
        temp_img_path = converter._extract_frame_image(video_path, frame_index_0based)
        frame = cv2.imread(str(temp_img_path))
        if frame is None:
            raise ValueError(f"Failed to read extracted frame from {temp_img_path}")
        # Clean up temp file
        temp_dir = temp_img_path.parent
        shutil.rmtree(temp_dir, ignore_errors=True)
    except Exception as e:
        print(f"FFmpeg extraction failed: {e}. Falling back to OpenCV...")
        # Fallback to OpenCV
        cap = cv2.VideoCapture(str(video_path))
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index_0based)
        ret, frame = cap.read()
        cap.release()
        if not ret:
            print(f"Failed to read frame {frame_num} from video")
            return
    
    # Convert BGR to RGB for matplotlib
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    actual_height, actual_width = frame_rgb.shape[:2]
    
    # Scale bbox coordinates if video resolution differs from annotation
    annotation_width = vid_info.get('width', actual_width)
    annotation_height = vid_info.get('height', actual_height)
    
    # Draw bounding box if requested
    if show_bbox:
        frame_rgb = draw_bounding_box(
            frame_rgb, 
            bbox, 
            annotation_width, 
            annotation_height,
            color=bbox_color,
            thickness=bbox_thickness
        )
    
    # Display the frame
    plt.figure(figsize=(12, 8))
    plt.imshow(frame_rgb)
    plt.title(f"Sequence: {seq_id}\nFrame: {frame_num} (index {frame_index})\n"
              f"Video: {actual_width}x{actual_height}, Annotation: {annotation_width}x{annotation_height}", 
              fontsize=12)
    plt.axis('off')
    plt.tight_layout()
    plt.show()
    
    print(f"Displayed frame {frame_num} from sequence {seq_id}")


def visualize_pose_with_skeleton(image, keypoints, bbox=None, title="Pose Detection", 
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
        print(f"[WRENCH] BBOX SCALING APPLIED:")
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
            print(f"   [OK] Bounding box drawn with correct scaling")
        else:
            print(f"   [WARNING] Invalid bbox dimensions")

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
