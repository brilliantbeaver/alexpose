"""
Evaluation Utilities for Keypoint Analysis

This module provides convenience functions for interactive notebook usage,
wrapping the core ambient.pose.keypoints module with notebook-friendly
interfaces and visualization capabilities for evaluation purposes.

Note: This module delegates to ambient.pose.keypoints for core functionality.
All core keypoint extraction logic lives in ambient.pose.keypoints.
This module only contains evaluation-specific wrapper functions for:
- Interactive visualization with matplotlib
- GAVD dataset-specific integration  
- User-friendly progress reporting
- Convenience functions for notebook exploration and evaluation

Originally located in notebooks/utils/, moved to ambient/utils/ for better
organization and to make it available as part of the core package.
"""

import json
import sys
from pathlib import Path

# CRITICAL: Configure pose estimation environment BEFORE any imports
from ambient.pose.pose_config import configure_pose_environment
configure_pose_environment()

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from typing import List, Dict, Optional, Tuple

import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# ============================================================================
# Re-export core functionality from ambient.pose.keypoints
# ============================================================================
# These are imported here for convenience so code can use:
#   from ambient.utils.eval_keypoints import get_keypoints, ensure_model_downloaded
# instead of:
#   from ambient.pose.keypoints import get_keypoints, ensure_model_downloaded
# ============================================================================
from ambient.pose.keypoints import (
    POSE_LANDMARK_NAMES,
    MediaPipeModelManager,
    PoseLandmarkerFactory,
    SequenceKeypointExtractor,
    KeypointVisualizer,
    ensure_model_downloaded,
    get_keypoints as _get_keypoints_core,  # Renamed to avoid conflict
    create_pose_landmarker,
)
from ambient.pose.keypoint_data import KeypointSet  # For type hints
from ambient.pose.joint_angles import get_joint_angles as calculate_angles
from ambient.utils.youtube_cache import extract_video_id
from ambient.utils.viz import visualize_pose_with_skeleton

print(f"MediaPipe detects {len(POSE_LANDMARK_NAMES)} landmarks")


# ============================================================================
# Evaluation-Specific Wrapper Functions
# ============================================================================
# The functions below are evaluation-specific wrappers that provide:
# - Interactive visualization with matplotlib
# - GAVD dataset-specific integration
# - User-friendly progress reporting
# - Convenience functions for notebook exploration
# ============================================================================


def get_keypoints(
    project_root: Path,
    sequence_data,  # Can be pd.DataFrame or Dict[str, pd.DataFrame]
    model_path: Optional[str] = None,
    verbose: bool = True
) -> Tuple[List, Optional[np.ndarray]]:
    """
    Extract keypoints from a video sequence (notebook-friendly wrapper).
    
    This is a convenience wrapper for interactive notebook use that provides
    backward compatibility with older notebook code by returning both keypoints
    and the first frame for visualization.
    
    Args:
        project_root: Project root directory
        sequence_data: DataFrame with sequence information OR dictionary of sequences
        model_path: Optional path to model file
        verbose: Whether to print progress
        
    Returns:
        Tuple of (keypoints, frame) where:
        - keypoints: List of KeypointSet objects
        - frame: First frame as numpy array (BGR) for visualization
    """
    # Use the core function to extract keypoints
    keypoints = _get_keypoints_core(
        project_root=project_root,
        sequence_data=sequence_data,
        model_path=model_path,
        verbose=verbose
    )
    
    # Extract the first frame for visualization
    frame = None
    try:
        # Handle both DataFrame and dict
        if isinstance(sequence_data, dict):
            first_key = list(sequence_data.keys())[0]
            seq_df = sequence_data[first_key]
        else:
            seq_df = sequence_data
        
        # Get first frame info
        first_row = seq_df.iloc[0]
        url = first_row['url']
        frame_num = int(first_row['frame_num'])
        
        # Get video path
        video_id = extract_video_id(url)
        video_path = project_root / "data" / "youtube" / f"{video_id}.mp4"
        
        if video_path.exists():
            # Extract first frame
            cap = cv2.VideoCapture(str(video_path))
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num - 1)
            ret, frame = cap.read()
            cap.release()
            
            if not ret:
                if verbose:
                    print("[WARNING] Could not read frame from video")
                frame = None
        else:
            if verbose:
                print(f"[WARNING] Video not found: {video_path}")
        
    except Exception as e:
        if verbose:
            print(f"[WARNING] Error extracting frame: {e}")
        frame = None
    
    return keypoints, frame


# ============================================================================
# Evaluation-Specific Wrapper Functions
# ============================================================================
# The functions below are evaluation-specific wrappers that provide:
# - Interactive visualization with matplotlib
# - GAVD dataset-specific integration
# - User-friendly progress reporting
# - Convenience functions for notebook exploration
# ============================================================================


# Evaluation-specific visualization function
def visualize_keypoints(keypoints: list, frame: np.ndarray):
    """
    Visualize keypoints on a frame with side-by-side comparison.
    
    This function is designed for interactive notebook use.
    
    Args:
        keypoints: List of KeypointSet objects (one per frame)
        frame: BGR image array from OpenCV
    """
    # Process results
    if keypoints and len(keypoints) > 0:
        # Get the first frame's keypoints (KeypointSet object)
        first_keypoint_set = keypoints[0]
        
        # Visualize
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 7))
        
        # Original frame
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        ax1.imshow(frame_rgb)
        ax1.set_title('Original Frame')
        ax1.axis('off')
        
        # Frame with pose - use draw_skeleton for better visualization
        annotated = KeypointVisualizer.draw_skeleton(
            frame_rgb, 
            first_keypoint_set,
            confidence_threshold=0.5,
            keypoint_color=(255, 0, 0),
            line_color=(0, 255, 0)
        )
        ax2.imshow(annotated)
        ax2.set_title(f'MediaPipe Detection ({len(first_keypoint_set)} landmarks)')
        ax2.axis('off')
        
        plt.tight_layout()
        plt.show()
        
        # Get and display statistics
        stats = KeypointVisualizer.get_summary_stats(first_keypoint_set)
        print(f"[OK] SUCCESS! Detected {stats['total_landmarks']} landmarks")
        print(f"[INFO] {stats['visible_landmarks']} landmarks are visible (confidence > 0.5)")
        print(f"[INFO] Average confidence: {stats['avg_confidence']:.3f}")
        print(f"[INFO] Detection quality: {stats['detection_quality']:.3f}")
        
        # Show some sample keypoints
        print("\n[INFO] Sample keypoints:")
        for i, kp in enumerate(first_keypoint_set.keypoints[:5]):  # Show first 5
            name = kp.name if kp.name else f"Point {kp.id}"
            print(f"  {name}: ({kp.x:.1f}, {kp.y:.1f}) confidence={kp.confidence:.3f}")
    else:
        print("[WARNING] No keypoints to visualize")


def extract_pose_from_sequence(
    project_root: Path, 
    sequence_data: pd.DataFrame, 
    frame_index: Optional[int] = None,
    frame_num: Optional[int] = None, 
    use_bbox: bool = True,
    show_visualization: bool = True, 
    confidence_threshold: float = 0.3,
    title_prefix: str = "MediaPipe Pose"
) -> Optional[Tuple[KeypointSet, np.ndarray, Dict]]:
    """
    Extract pose landmarks from GAVD dataset video frames using MediaPipe.
    
    This function provides a comprehensive testing interface for MediaPipe pose
    detection on GAVD dataset videos, designed for interactive notebook exploration.
    
    Workflow:
    1. Validates sequence and frame selection parameters
    2. Extracts specified frame from cached YouTube video
    3. Runs MediaPipe pose detection on full RGB frame
    4. Converts normalized landmarks to pixel coordinates
    5. Optionally visualizes results with skeleton connections
    6. Returns structured keypoint data with comprehensive metadata
    
    Args:
        project_root: Project root directory
        sequence_data: GAVD sequence DataFrame
        frame_index: 0-based index within the sequence (default: 10)
        frame_num: Specific frame number from video (overrides frame_index)
        use_bbox: Whether to include bounding box in visualization
        show_visualization: Whether to display pose visualization
        confidence_threshold: Minimum confidence for drawing connections
        title_prefix: Prefix text for visualization title
    
    Returns:
        Tuple of (keypoints, frame_rgb, metadata) or None if detection failed
        - keypoints: KeypointSet object with detected landmarks
        - frame_rgb: RGB image array
        - metadata: Dict with sequence info, frame numbers, detection stats
    """
    # Create extractor
    extractor = SequenceKeypointExtractor()
    
    try:
        # Validate DataFrame structure
        if sequence_data.empty:
            print("[ERROR] Empty sequence data provided")
            return None
            
        if 'seq' not in sequence_data.columns:
            print("[ERROR] Missing 'seq' column in sequence data")
            return None
            
        sequence_id = sequence_data['seq'].iloc[0]
        
        # Determine which frame to extract
        if frame_num is not None:
            # Find the row with the specific frame number
            matching_rows = sequence_data[sequence_data['frame_num'] == frame_num]
            if matching_rows.empty:
                print(f"[ERROR] Frame {frame_num} not found in sequence {sequence_id}")
                print(f"   Available frame range: {sequence_data['frame_num'].min()} - {sequence_data['frame_num'].max()}")
                return None
            frame_row = matching_rows.iloc[0]
            actual_frame_index = matching_rows.index[0] - sequence_data.index[0]
        else:
            # Use frame index (default to 10 if not specified)
            if frame_index is None:
                frame_index = 10
                print(f"🔄 No frame specified, using frame index {frame_index}")
            
            if frame_index >= len(sequence_data):
                print(f"[ERROR] Frame index {frame_index} out of range. Sequence has {len(sequence_data)} frames.")
                return None
            
            frame_row = sequence_data.iloc[frame_index]
            actual_frame_index = frame_index
        
        # Extract frame information
        actual_frame_num = int(frame_row['frame_num'])
        url = frame_row['url']
        bbox = frame_row.get('bbox', {}) if use_bbox else {}
        
        print(f"[TARGET] Processing sequence: {sequence_id}")
        print(f"   Frame index: {actual_frame_index} | Frame number: {actual_frame_num}")
        
        # Get video path
        video_id = extract_video_id(url)
        video_path = project_root / "data" / "youtube" / f"{video_id}.mp4"
        
        if not video_path.exists():
            print(f"[ERROR] Video not found: {video_path}")
            return None
        
        # Extract frame from video
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            print(f"[ERROR] Cannot open video: {video_path}")
            return None
            
        cap.set(cv2.CAP_PROP_POS_FRAMES, actual_frame_num - 1)
        ret, frame = cap.read()
        cap.release()
        
        if not ret:
            print(f"[ERROR] Could not read frame {actual_frame_num}")
            return None
        
        # Convert to RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        print(f"[VIDEO] Frame extracted: {frame_rgb.shape}")
        
        # Extract keypoints using the refactored extractor
        print("[SEARCH] Running pose detection on full frame...")
        keypoints = extractor.extract_from_image(frame_rgb)
        
        if keypoints:
            print(f"[OK] Detected {len(keypoints)} landmarks")
        else:
            print("[WARNING] No pose detected")
        
        # Create metadata
        metadata = {
            'sequence_id': sequence_id,
            'frame_index': actual_frame_index,
            'frame_num': actual_frame_num,
            'video_id': video_id,
            'video_path': str(video_path),
            'frame_shape': frame_rgb.shape,
            'bbox': bbox,
            'landmarks_detected': len(keypoints) if keypoints else 0,
            'confidence_threshold': confidence_threshold
        }
        
        # Visualize if requested
        if show_visualization:
            # Check if we have valid keypoints (KeypointSet with landmarks)
            if keypoints is not None and len(keypoints) > 0:
                title = f"{title_prefix} - Seq: {sequence_id[:8]}... | Frame: {actual_frame_num}"
                vid_info = frame_row.get('vid_info', {})
                
                # Convert KeypointSet to dict format for visualization
                # The KeypointSet.to_dict_list() method returns List[Dict]
                keypoints_dict = keypoints.to_dict_list()
                
                visualize_pose_with_skeleton(
                    frame_rgb, keypoints_dict, bbox, title, 
                    vid_info=vid_info, frame_shape=frame_rgb.shape
                )
            else:
                # Show original image even if no pose detected
                plt.figure(figsize=(10, 8))
                plt.imshow(frame_rgb)
                plt.title(f"No Pose Detected - Frame {actual_frame_num}")
                plt.axis('off')
                plt.show()
        
        return keypoints, frame_rgb, metadata
        
    except Exception as e:
        print(f"[ERROR] Extraction failed: {e}")
        import traceback
        traceback.print_exc()
        return None


def pose_estimation_for_frames(
    project_root: Path,
    sequence_data: pd.DataFrame,
    frame_indices: Optional[List[int]] = None,
    frame_nums: Optional[List[int]] = None, 
    max_frames: int = 5,
    show_each: bool = True
) -> List[Dict]:
    """
    Process pose estimation across multiple frames from a sequence.
    
    Args:
        project_root: Project root directory
        sequence_data: GAVD sequence DataFrame
        frame_indices: List of frame indices to test
        frame_nums: List of specific frame numbers to test
        max_frames: Maximum number of frames to test if using auto-selection
        show_each: Whether to show visualization for each frame
    
    Returns:
        List of results from each frame test
    """
    # Validate DataFrame structure
    if sequence_data.empty:
        print("[ERROR] Empty sequence data provided")
        return []
        
    if 'seq' not in sequence_data.columns:
        print("[ERROR] Missing 'seq' column in sequence data")
        return []
        
    sequence_id = sequence_data['seq'].iloc[0]
    results = []
    
    # Determine which frames to test
    if frame_nums is not None:
        test_frames = [(None, fn) for fn in frame_nums]
        print(f"[TARGET] Testing {len(frame_nums)} specific frame numbers: {frame_nums}")
    elif frame_indices is not None:
        test_frames = [(fi, None) for fi in frame_indices]
        print(f"[TARGET] Testing {len(frame_indices)} specific frame indices: {frame_indices}")
    else:
        # Auto-select frames evenly distributed across sequence
        total_frames = len(sequence_data)
        if total_frames <= max_frames:
            indices = list(range(total_frames))
        else:
            step = total_frames // max_frames
            indices = [i * step for i in range(max_frames)]
        
        test_frames = [(fi, None) for fi in indices]
        print(f"[TARGET] Auto-selected {len(indices)} frames from sequence (total: {total_frames})")
    
    # Test each frame
    for i, (frame_idx, frame_num) in enumerate(test_frames):
        print(f"\n--- Testing frame {i+1}/{len(test_frames)} ---")
        
        result = extract_pose_from_sequence(
            project_root=project_root,
            sequence_data=sequence_data,
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
                'avg_confidence': keypoints.avg_confidence if keypoints and len(keypoints) > 0 else 0.0,
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
    print(f"\n[CHART] SUMMARY - Tested {len(results)} frames from sequence {sequence_id}")
    successful = sum(1 for r in results if r['landmarks_count'] > 0)
    print(f"   [OK] Successful detections: {successful}/{len(results)}")
    
    if successful > 0:
        avg_landmarks = np.mean([r['landmarks_count'] for r in results if r['landmarks_count'] > 0])
        avg_confidence = np.mean([r['avg_confidence'] for r in results if r['avg_confidence'] > 0])
        print(f"   📈 Average landmarks detected: {avg_landmarks:.1f}")
        print(f"   📈 Average confidence: {avg_confidence:.3f}")
    
    return results


def pose_estimation_all_sequences(
    sequences: Dict[str, pd.DataFrame],
    max_frames_per_seq: int = 3,
    show_visualizations: bool = True
) -> Dict[str, List[Dict]]:
    """
    Explore pose detection across all available sequences.
    
    Args:
        sequences: Dictionary mapping sequence IDs to DataFrames
        max_frames_per_seq: Maximum frames to test per sequence
        show_visualizations: Whether to show visualizations
    
    Returns:
        Dictionary mapping sequence IDs to results
    """
    project_root = Path.cwd().parent
    all_results = {}
    
    print(f"[SEARCH] Exploring pose detection across {len(sequences)} sequences")
    print(f"   Testing up to {max_frames_per_seq} frames per sequence")
    
    for i, seq_id in enumerate(sequences.keys()):
        print(f"\n{'='*60}")
        print(f"SEQUENCE {i+1}/{len(sequences)}: {seq_id}")
        print(f"{'='*60}")
        
        results = pose_estimation_for_frames(
            project_root=project_root,
            sequence_data=sequences[seq_id],
            max_frames=max_frames_per_seq,
            show_each=show_visualizations
        )
        
        all_results[seq_id] = results
    
    # Overall summary
    print(f"\n{'='*60}")
    print("OVERALL SUMMARY")
    print(f"{'='*60}")
    
    total_tests = sum(len(results) for results in all_results.values())
    total_successful = sum(
        sum(1 for r in results if r['landmarks_count'] > 0) 
        for results in all_results.values()
    )
    
    print(f"[CHART] Total tests: {total_tests}")
    if total_tests > 0:
        print(f"[OK] Successful detections: {total_successful}/{total_tests} ({100*total_successful/total_tests:.1f}%)")
    
    # Per-sequence summary
    for seq_id, results in all_results.items():
        successful = sum(1 for r in results if r['landmarks_count'] > 0)
        print(f"   {seq_id[:20]}...: {successful}/{len(results)} successful")
    
    return all_results


#----------------------------------------------------------------------
# main - Example usage
#----------------------------------------------------------------------

if __name__ == "__main__":
    from ambient.gavd import GAVDDataLoader
    
    project_root = Path.cwd()
    print(f"==> project_root: {project_root}")

    # 1. Load GAVD gait sequences
    loader = GAVDDataLoader()
    ONE_SEQUENCE_PATH = project_root / "data" / "GAVD_Clinical_Annotations_1.1.csv"
    df = loader.load_gavd_data(ONE_SEQUENCE_PATH)
    sequences = loader.organize_by_sequence(df)
    sequence_id = list(sequences.keys())[1]
    sequence_data = sequences[sequence_id]
    print(f"\tnum_sequences: {len(sequences)}")

    # 2. Extract body keypoints (pose landmarks) using refactored code
    # Returns: List[KeypointSet] - one KeypointSet per frame
    # Each KeypointSet contains Keypoint objects with x, y, z, confidence, etc.
    keypoints_array = get_keypoints(
        project_root=project_root,
        sequence_data=sequence_data,
        verbose=True
    )
    print(f"==> # keypoints extracted: {len(keypoints_array)} frames")
    print(f"    Each frame has {len(keypoints_array[0])} keypoints")
    print(f"    Format: {keypoints_array[0].format.value}")
    print(f"    Avg confidence: {keypoints_array[0].avg_confidence:.3f}")

    # 3. Compute joint angles – For each frame, calculate hip, knee and ankle angles
    # The calculate_angles function (alias for get_joint_angles) now accepts:
    #   - List[KeypointSet] (new format) - RECOMMENDED
    #   - List[List[Dict]] (legacy format) - still supported for backward compatibility
    # 
    # Returns: JointAngleSequence object with:
    #   - .frames: List[FrameJointAngles] - angles for each frame
    #   - .get_statistics(joint_name): Get mean, std, min, max, range
    #   - .get_joint_angle_series(joint_name): Get time series as numpy array
    joint_angles = calculate_angles(
        keypoints_array=keypoints_array,  # List[KeypointSet]
        keypoint_format="BLAZEPOSE_33",
        fps=30.0,
        confidence_threshold=0.3
    )
    
    print(f"\n==> Joint angles calculated for {len(joint_angles.frames)} frames")
    print(f"    Joints detected: {list(joint_angles.frames[0].angles.keys())}")
    
    # Get statistics for left knee across all frames
    left_knee_stats = joint_angles.get_statistics('left_knee')
    print(f"\n--> Left knee angle statistics:")
    print(f"    Mean: {left_knee_stats['mean']:.2f}°")
    print(f"    Std: {left_knee_stats['std']:.2f}°")
    print(f"    Range: {left_knee_stats['range']:.2f}°")
    print(f"    Valid frames: {left_knee_stats['valid_count']}/{len(joint_angles.frames)}")
    
    # Show first frame's joint angles
    print("\n--> First frame joint angles:")
    print(json.dumps(joint_angles.frames[0].to_dict(), indent=2, default=str))
