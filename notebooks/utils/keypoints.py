import cv2
import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pathlib import Path
import tempfile
import urllib.request
from typing import List, Dict, Optional, Tuple

import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

from ambient.gavd.pose_estimators import MediaPipeEstimator, get_pose_estimator
from ambient.utils.youtube_cache import extract_video_id

from .viz import visualize_pose_with_skeleton

print("✅ Using MediaPipeEstimator from ambient package")

# MediaPipe pose landmark names (33 landmarks)
POSE_LANDMARK_NAMES = [
    'NOSE', 'LEFT_EYE_INNER', 'LEFT_EYE', 'LEFT_EYE_OUTER',
    'RIGHT_EYE_INNER', 'RIGHT_EYE', 'RIGHT_EYE_OUTER',
    'LEFT_EAR', 'RIGHT_EAR', 'MOUTH_LEFT', 'MOUTH_RIGHT',
    'LEFT_SHOULDER', 'RIGHT_SHOULDER', 'LEFT_ELBOW', 'RIGHT_ELBOW',
    'LEFT_WRIST', 'RIGHT_WRIST', 'LEFT_PINKY', 'RIGHT_PINKY',
    'LEFT_INDEX', 'RIGHT_INDEX', 'LEFT_THUMB', 'RIGHT_THUMB',
    'LEFT_HIP', 'RIGHT_HIP', 'LEFT_KNEE', 'RIGHT_KNEE',
    'LEFT_ANKLE', 'RIGHT_ANKLE', 'LEFT_HEEL', 'RIGHT_HEEL',
    'LEFT_FOOT_INDEX', 'RIGHT_FOOT_INDEX'
]

print(f"🎯 MediaPipe detects {len(POSE_LANDMARK_NAMES)} landmarks")

def ensure_model_downloaded(project_root: str):
    """Ensure MediaPipe pose model is downloaded"""
    model_dir = project_root / "data" / "models"
    model_dir.mkdir(parents=True, exist_ok=True)
    
    model_path = model_dir / "pose_landmarker_lite.task"
    
    if model_path.exists():
        print(f"✅ Model already exists: {model_path}")
        return str(model_path)
    
    print("📥 Downloading MediaPipe pose landmarker model...")
    model_url = "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/1/pose_landmarker_lite.task"
    
    try:
        print("⏳ Downloading... (this may take a moment)")
        urllib.request.urlretrieve(model_url, model_path)
        size_mb = model_path.stat().st_size / (1024 * 1024)
        print(f"✅ Model downloaded successfully!")
        print(f"📊 Size: {size_mb:.1f} MB")
        return str(model_path)
    except Exception as e:
        print(f"❌ Download failed: {e}")
        return None

# use the MediaPipeEstimator with our GAVD data
def get_keypoints(project_root: str, sequences: Dict[str, pd.DataFrame]):
    """Test pose detection using MediaPipeEstimator"""

    # Ensure model is downloaded and create estimator
    model_path = ensure_model_downloaded(project_root)
    if model_path:
        try:
            pose_estimator = MediaPipeEstimator(model_path=model_path)
            print(f"✅ MediaPipeEstimator created successfully")
            print(f"📍 Model available: {pose_estimator.is_available()}")
        except Exception as e:
            print(f"❌ Failed to create MediaPipeEstimator: {e}")
            pose_estimator = None
    else:
        print("❌ MediaPipeEstimator not available")
        return None
    
    try:
        # Get first sequence and frame
        first_seq_id = list(sequences.keys())[0]
        frame_row = sequences[first_seq_id].iloc[10]
        frame_num = int(frame_row['frame_num'])
        url = frame_row['url']
        
        # Get video path
        video_id = extract_video_id(url)
        video_path = project_root / "data" / "youtube" / f"{video_id}.mp4"
        
        if not video_path.exists():
            print(f"❌ Video not found: {video_path}")
            return None
        
        cap = cv2.VideoCapture(str(video_path))
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num - 1)
        ret, frame = cap.read()
        cap.release()
        
        if not ret:
            print(f"❌ Could not read frame {frame_num}")
            return None, None
        
        # Save frame to temporary file
        with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as tmp_file:
            temp_image_path = tmp_file.name
            cv2.imwrite(temp_image_path, frame)
        
        # Use MediaPipeEstimator to detect pose
        print("🔍 Running pose detection with MediaPipeEstimator...")
        keypoints = pose_estimator.estimate_image_keypoints(temp_image_path)
        
        # Clean up temporary file
        os.unlink(temp_image_path)
                
        return keypoints, frame
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return None, None


def visualize_keypoints(keypoints: list, frame: np.ndarray):
    print(f"keypoint: {type(keypoints)}, frame: {type(frame)}")
    # Process results
    if keypoints:
        # Add landmark names to keypoints
        for i, kp in enumerate(keypoints):
            if i < len(POSE_LANDMARK_NAMES):
                kp['name'] = POSE_LANDMARK_NAMES[i]
        
        # Visualize
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 7))
        
        # Original frame
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        ax1.imshow(frame_rgb)
        ax1.set_title('Original Frame')
        ax1.axis('off')
        
        # Frame with pose
        annotated = frame_rgb.copy()
        for kp in keypoints:
            if kp['confidence'] > 0.5:
                x, y = int(kp['x']), int(kp['y'])
                cv2.circle(annotated, (x, y), 5, (255, 0, 0), -1)
        
        ax2.imshow(annotated)
        ax2.set_title(f'MediaPipeEstimator Detection ({len(keypoints)} landmarks)')
        ax2.axis('off')
        
        plt.tight_layout()
        plt.show()
        
        print(f"✅ SUCCESS! Detected {len(keypoints)} landmarks")
        visible = sum(1 for kp in keypoints if kp['confidence'] > 0.5)
        print(f"👁️ {visible} landmarks are visible (confidence > 0.5)")
        
        # Show some sample keypoints
        print("\n📍 Sample keypoints:")
        for kp in keypoints[:5]:  # Show first 5
            name = kp.get('name', f"Point {kp['id']}")
            print(f"  {name}: ({kp['x']:.1f}, {kp['y']:.1f}) confidence={kp['confidence']:.3f}")

def create_pose_landmarker(model_path: str):
    """Create MediaPipe Pose Landmarker using the new API"""
    
    try:
        # Create base options
        base_options = python.BaseOptions(model_asset_path=model_path)
        
        # Create pose landmarker options
        options = vision.PoseLandmarkerOptions(
            base_options=base_options,
            running_mode=vision.RunningMode.IMAGE,
            num_poses=1,
            min_pose_detection_confidence=0.5,
            min_pose_presence_confidence=0.5,
            min_tracking_confidence=0.5,
            output_segmentation_masks=False
        )
        
        # Create the landmarker
        landmarker = vision.PoseLandmarker.create_from_options(options)
        print(f"✅ Pose Landmarker created from {model_path}")
        return landmarker
        
    except Exception as e:
        print(f"❌ Failed to create Pose Landmarker: {e}")
        return None


def extract_pose_from_sequence(project_root: str, sequence_data: pd.DataFrame, 
                                frame_index=None, frame_num=None, 
                                use_bbox=True, show_visualization=True, 
                                confidence_threshold=0.3, title_prefix="MediaPipe Pose"):
    """
    Extract pose landmarks from GAVD dataset video frames using MediaPipe pose estimation.
    
    This function provides a comprehensive testing interface for MediaPipe pose detection on GAVD dataset videos.
    It handles flexible frame selection (by index or frame number), extracts video frames, runs pose estimation
    on the full frame (critical for detection success), converts landmarks to pixel coordinates, and optionally
    displays visualization with skeleton overlay. The function is designed for interactive notebook exploration
    and debugging of pose detection pipeline.
    
    Workflow:
    1. Validates sequence and frame selection parameters
    2. Extracts specified frame from cached YouTube video
    3. Runs MediaPipe pose detection on full RGB frame
    4. Converts normalized landmarks to pixel coordinates
    5. Optionally visualizes results with skeleton connections
    6. Returns structured keypoint data with comprehensive metadata
    
    Args:
        sequence (pd.DataFrame): Specific GAVD sequence to analyze. If None, uses first available sequence
        frame_index (int, optional): 0-based index within the sequence. If None, defaults to frame 10
        frame_num (int, optional): Specific frame number from video. Overrides frame_index if provided
        use_bbox (bool): Whether to include bounding box information in visualization (default: True)
        show_visualization (bool): Whether to display the pose visualization with skeleton overlay (default: True)
        confidence_threshold (float): Minimum confidence threshold for drawing landmark connections (default: 0.3)
        title_prefix (str): Prefix text for the visualization title (default: "MediaPipe Pose")
    
    Returns:
        tuple: (keypoints, frame_rgb, metadata) or None if detection failed
            - keypoints (List[Dict]): List of 33 pose landmarks with pixel coordinates, confidence scores, and names
            - frame_rgb (np.ndarray): RGB image array of the processed frame
            - metadata (Dict): Comprehensive metadata including sequence info, frame details, detection stats
    
    Raises:
        Returns None on any failure (missing video, invalid frame, pose detection error)
        Prints detailed error messages and traceback for debugging
    
    Note:
        Requires pose_landmarker to be initialized and GAVD sequences data to be loaded.
        Uses full frame for pose detection which is critical for MediaPipe success.
    """
    
    MODEL_PATH = str(project_root / "data" / "models" / "pose_landmarker_full.task")
    if MODEL_PATH:
        pose_landmarker = create_pose_landmarker(MODEL_PATH)
    else:
        print("❌ Pose landmarker not available")
        return None
    
    try:
        # Validate DataFrame structure
        if sequence_data.empty:
            print("❌ Empty sequence data provided")
            return None
            
        if 'seq' not in sequence_data.columns:
            print("❌ Missing 'seq' column in sequence data")
            return None
            
        sequence_id = sequence_data['seq'].iloc[0]
        
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
                vid_info = frame_row.get('vid_info', {})
                visualize_pose_with_skeleton(frame_rgb, keypoints, bbox, title, 
                                        vid_info=vid_info, frame_shape=frame_rgb.shape)
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


def pose_estimation_for_frames(project_root: str, sequence_data: pd.DataFrame, frame_indices=None, frame_nums=None, 
                        max_frames=5, show_each=True):
    """
    Process pose estimation across multiple frames from a sequence
    
    Args:
        sequence_id (str, optional): Sequence to test. If None, uses first available
        frame_indices (list, optional): List of frame indices to test
        frame_nums (list, optional): List of specific frame numbers to test
        max_frames (int): Maximum number of frames to test if using auto-selection
        show_each (bool): Whether to show visualization for each frame
    
    Returns:
        list: List of results from each frame test
    """
    
    # Validate DataFrame structure
    if sequence_data.empty:
        print("❌ Empty sequence data provided")
        return []
        
    if 'seq' not in sequence_data.columns:
        print("❌ Missing 'seq' column in sequence data")
        return []
        
    sequence_id = sequence_data['seq'].iloc[0]
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


def pose_estimation_all_sequences(sequences: dict[str, pd.DataFrame], max_frames_per_seq=3, show_visualizations=True):
    """
    Explore pose detection across all available sequences
    
    Args:
        max_frames_per_seq (int): Maximum frames to test per sequence
        show_visualizations (bool): Whether to show visualizations
    
    Returns:
        dict: Results organized by sequence
    """
    
    project_root = Path.cwd().parent
    all_results = {}
    
    print(f"🔍 Exploring pose detection across {len(sequences)} sequences")
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
    total_successful = sum(sum(1 for r in results if r['landmarks_count'] > 0) 
                          for results in all_results.values())
    
    print(f"📊 Total tests: {total_tests}")
    print(f"✅ Successful detections: {total_successful}/{total_tests} ({100*total_successful/total_tests:.1f}%)")
    
    # Per-sequence summary
    for seq_id, results in all_results.items():
        successful = sum(1 for r in results if r['landmarks_count'] > 0)
        print(f"   {seq_id[:20]}...: {successful}/{len(results)} successful")
    
    return all_results

