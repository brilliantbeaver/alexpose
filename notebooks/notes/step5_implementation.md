# MediaPipe Pose Estimation Implementation for Jupyter Notebook

Copy and paste each of these code blocks into separate cells in your Jupyter notebook after the "Step 5" markdown cell.

## Cell 1: Setup and Imports

```python
# Import MediaPipe and related libraries
import tempfile
import numpy as np
import time
from typing import List, Dict, Tuple, Optional
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from IPython.display import display, HTML
%matplotlib inline

# Check if MediaPipe is available
try:
    import mediapipe as mp
    from mediapipe.tasks import python
    from mediapipe.tasks.python import vision
    print("✅ MediaPipe is available!")
    print(f"📦 MediaPipe version: {mp.__version__}")
    MEDIAPIPE_AVAILABLE = True
except ImportError as e:
    print("❌ MediaPipe is not installed. Please install it with: pip install mediapipe")
    print(f"Error: {e}")
    MEDIAPIPE_AVAILABLE = False

# Try to import the existing MediaPipeEstimator
try:
    from ambient.gavd.pose_estimators import MediaPipeEstimator
    print("✅ MediaPipeEstimator class imported successfully")
    ESTIMATOR_AVAILABLE = True
except ImportError as e:
    print(f"⚠️  MediaPipeEstimator not available: {e}")
    ESTIMATOR_AVAILABLE = False
```

## Cell 2: MediaPipe Native Setup

```python
# Initialize MediaPipe Pose (native API)
if MEDIAPIPE_AVAILABLE:
    mp_pose = mp.solutions.pose
    mp_drawing = mp.solutions.drawing_utils
    mp_drawing_styles = mp.solutions.drawing_styles
    
    # MediaPipe pose landmark names for reference
    POSE_LANDMARKS = [
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
    
    print(f"🎯 MediaPipe Pose detects {len(POSE_LANDMARKS)} landmarks:")
    for i, landmark in enumerate(POSE_LANDMARKS[:10]):  # Show first 10
        print(f"  {i:2d}: {landmark}")
    print("  ... (and 23 more landmarks)")
else:
    print("❌ MediaPipe not available - skipping native setup")
```

## Cell 3: Model Download

```python
# Download MediaPipe pose model if not available (for task-based API)
import urllib.request
from pathlib import Path

def download_mediapipe_model():
    """Download the MediaPipe pose landmarker model."""
    model_dir = project_root / "data" / "models"
    model_dir.mkdir(exist_ok=True)
    
    model_path = model_dir / "pose_landmarker_lite.task"
    
    if model_path.exists():
        print(f"✅ Model already exists at: {model_path}")
        return model_path
    
    print("📥 Downloading MediaPipe pose landmarker model...")
    model_url = "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/1/pose_landmarker_lite.task"
    
    try:
        print("⏳ This may take a moment...")
        urllib.request.urlretrieve(model_url, model_path)
        print(f"✅ Model downloaded successfully to: {model_path}")
        print(f"📊 Model size: {model_path.stat().st_size / (1024*1024):.1f} MB")
        return model_path
    except Exception as e:
        print(f"❌ Failed to download model: {e}")
        return None

# Download model if MediaPipe is available
if MEDIAPIPE_AVAILABLE:
    model_path = download_mediapipe_model()
else:
    model_path = None
    print("⏭️  Skipping model download - MediaPipe not available")
```

## Cell 4: Initialize Estimators

```python
# Initialize both MediaPipe estimators
pose_estimator = None
native_pose = None

# Try to initialize the existing MediaPipeEstimator class
if ESTIMATOR_AVAILABLE and MEDIAPIPE_AVAILABLE and model_path:
    try:
        pose_estimator = MediaPipeEstimator(
            model_path=model_path,
            min_pose_detection_confidence=0.5,
            min_pose_presence_confidence=0.5,
            min_tracking_confidence=0.5
        )
        print("✅ MediaPipeEstimator initialized successfully!")
    except Exception as e:
        print(f"⚠️  Failed to initialize MediaPipeEstimator: {e}")
        print("📝 Will use native MediaPipe API instead")
        pose_estimator = None

# Initialize native MediaPipe Pose as backup/alternative
if MEDIAPIPE_AVAILABLE:
    try:
        # This will be used for direct pose estimation
        print("✅ Native MediaPipe Pose API ready")
        native_pose_available = True
    except Exception as e:
        print(f"❌ Failed to setup native MediaPipe: {e}")
        native_pose_available = False
else:
    native_pose_available = False

# Summary of available methods
print("\n🔧 Available Pose Estimation Methods:")
if pose_estimator:
    print("  ✅ MediaPipeEstimator class (recommended)")
if native_pose_available:
    print("  ✅ Native MediaPipe API (fallback)")
if not pose_estimator and not native_pose_available:
    print("  ❌ No pose estimation methods available")
```

## Cell 5: Core Functions

```python
def extract_frame_from_video(video_path: Path, frame_number: int) -> Optional[np.ndarray]:
    """
    Extract a specific frame from a video file.
    
    Args:
        video_path: Path to the video file
        frame_number: Frame number to extract (1-based)
    
    Returns:
        Frame as numpy array in RGB format, or None if extraction fails
    """
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"❌ Error: Could not open video {video_path}")
        return None
    
    # Convert 1-based to 0-based frame number
    frame_index = frame_number - 1
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
    ret, frame = cap.read()
    cap.release()
    
    if not ret:
        print(f"❌ Error: Could not read frame {frame_number}")
        return None
    
    # Convert BGR to RGB
    return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)


def estimate_pose_native_mediapipe(image: np.ndarray, 
                                  min_detection_confidence: float = 0.5,
                                  min_tracking_confidence: float = 0.5) -> Tuple[Optional[any], np.ndarray]:
    """
    Estimate pose using native MediaPipe API on a single image.
    
    Args:
        image: Input image as numpy array (RGB format)
        min_detection_confidence: Minimum confidence for pose detection
        min_tracking_confidence: Minimum confidence for pose tracking
    
    Returns:
        Tuple of (pose_landmarks, annotated_image)
    """
    if not MEDIAPIPE_AVAILABLE:
        return None, image
    
    # Initialize MediaPipe Pose
    with mp_pose.Pose(
        static_image_mode=True,
        model_complexity=2,
        enable_segmentation=False,
        min_detection_confidence=min_detection_confidence,
        min_tracking_confidence=min_tracking_confidence
    ) as pose:
        
        # Process the image
        results = pose.process(image)
        
        # Create annotated image
        annotated_image = image.copy()
        
        if results.pose_landmarks:
            # Draw pose landmarks
            mp_drawing.draw_landmarks(
                annotated_image,
                results.pose_landmarks,
                mp_pose.POSE_CONNECTIONS,
                landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style()
            )
        
        return results.pose_landmarks, annotated_image


def extract_keypoints_from_landmarks(landmarks, image_width: int, image_height: int) -> List[Dict]:
    """
    Extract keypoints from MediaPipe landmarks.
    
    Args:
        landmarks: MediaPipe pose landmarks
        image_width: Width of the image
        image_height: Height of the image
    
    Returns:
        List of keypoint dictionaries with x, y, z, confidence, and landmark_name
    """
    if not landmarks:
        return []
    
    keypoints = []
    for i, landmark in enumerate(landmarks.landmark):
        keypoint = {
            'landmark_id': i,
            'landmark_name': POSE_LANDMARKS[i] if i < len(POSE_LANDMARKS) else f'LANDMARK_{i}',
            'x': landmark.x * image_width,  # Convert normalized to pixel coordinates
            'y': landmark.y * image_height,
            'z': landmark.z,  # Depth relative to hip midpoint
            'confidence': landmark.visibility,  # Use visibility as confidence
            'x_normalized': landmark.x,  # Keep normalized coordinates too
            'y_normalized': landmark.y,
        }
        keypoints.append(keypoint)
    
    return keypoints
```

## Cell 6: Main Pose Estimation Function

```python
def estimate_pose_for_frame(sequences: Dict[str, pd.DataFrame], seq_id: str, frame_index: int = 0, 
                           use_native: bool = False):
    """
    Estimate pose for a specific frame using MediaPipe.
    
    Args:
        sequences: Dictionary of sequences
        seq_id: Sequence ID
        frame_index: Frame index within the sequence (0-based)
        use_native: Whether to use native MediaPipe API instead of MediaPipeEstimator
    
    Returns:
        Tuple of (keypoints, frame_image, bbox_info)
    """
    # Check if any pose estimation method is available
    if not pose_estimator and not native_pose_available:
        print("❌ No pose estimation methods available")
        return None, None, None
    
    # Get frame information
    seq_data = sequences[seq_id]
    if frame_index >= len(seq_data):
        print(f"❌ Frame index {frame_index} out of range. Sequence has {len(seq_data)} frames.")
        return None, None, None
    
    frame_row = seq_data.iloc[frame_index]
    frame_num = int(frame_row['frame_num'])
    url = frame_row.get('url', '')
    bbox = frame_row.get('bbox', {})
    
    # Get video path
    video_id = extract_video_id(url) if url else None
    if not video_id:
        print(f"❌ No URL found for frame {frame_index}")
        return None, None, None
    
    video_cache_dir = project_root / "data" / "youtube"
    video_path = None
    for ext in ['.mp4', '.webm', '.mkv', '.mov']:
        candidate = video_cache_dir / f"{video_id}{ext}"
        if candidate.exists():
            video_path = candidate
            break
    
    if not video_path:
        print(f"❌ Video not found in cache. Expected: {video_cache_dir / f'{video_id}.mp4'}")
        return None, None, None
    
    try:
        print(f"🎯 Extracting pose from frame {frame_num} (index {frame_index})...")
        
        # Extract frame from video
        frame_image = extract_frame_from_video(video_path, frame_num)
        if frame_image is None:
            return None, None, None
        
        # Choose estimation method
        if pose_estimator and not use_native:
            # Use MediaPipeEstimator class
            print("🔧 Using MediaPipeEstimator class...")
            
            # Save frame to temporary file for the estimator
            temp_dir = Path(tempfile.mkdtemp(prefix="pose_estimation_"))
            temp_image_path = temp_dir / f"frame_{frame_num:06d}.jpg"
            
            # Convert RGB back to BGR for saving
            frame_bgr = cv2.cvtColor(frame_image, cv2.COLOR_RGB2BGR)
            cv2.imwrite(str(temp_image_path), frame_bgr)
            
            # Estimate pose keypoints
            keypoints = pose_estimator.estimate_image_keypoints(
                image_path=str(temp_image_path),
                bbox=bbox if isinstance(bbox, dict) else None
            )
            
            # Clean up temporary file
            import shutil
            shutil.rmtree(temp_dir, ignore_errors=True)
            
        elif native_pose_available:
            # Use native MediaPipe API
            print("🔧 Using native MediaPipe API...")
            
            # Apply bounding box crop if provided
            cropped_image = frame_image
            if bbox and isinstance(bbox, dict) and all(k in bbox for k in ['left', 'top', 'width', 'height']):
                left = max(0, int(bbox['left']))
                top = max(0, int(bbox['top']))
                right = min(frame_image.shape[1], left + int(bbox['width']))
                bottom = min(frame_image.shape[0], top + int(bbox['height']))
                cropped_image = frame_image[top:bottom, left:right]
                print(f"📏 Applied bbox crop: ({left}, {top}) to ({right}, {bottom})")
            
            # Estimate pose
            landmarks, annotated_image = estimate_pose_native_mediapipe(cropped_image)
            
            # Convert landmarks to keypoints format
            if landmarks:
                keypoints = extract_keypoints_from_landmarks(
                    landmarks, cropped_image.shape[1], cropped_image.shape[0]
                )
                
                # Adjust coordinates if we cropped the image
                if bbox and isinstance(bbox, dict):
                    left_offset = max(0, int(bbox.get('left', 0)))
                    top_offset = max(0, int(bbox.get('top', 0)))
                    for kp in keypoints:
                        kp['x'] += left_offset
                        kp['y'] += top_offset
            else:
                keypoints = []
        
        else:
            print("❌ No pose estimation method available")
            return None, None, None
        
        print(f"✅ Detected {len(keypoints)} keypoints")
        return keypoints, frame_image, bbox
        
    except Exception as e:
        print(f"❌ Error during pose estimation: {e}")
        import traceback
        traceback.print_exc()
        return None, None, None
```

## Cell 7: Visualization Function

```python
def visualize_pose_comprehensive(keypoints, frame_image, bbox=None, seq_id="", frame_index=0):
    """
    Comprehensive pose visualization that ensures proper display in Jupyter notebooks.
    
    Args:
        keypoints: List of keypoint dictionaries from MediaPipe
        frame_image: RGB frame image (numpy array)
        bbox: Optional bounding box dictionary
        seq_id: Sequence ID for title
        frame_index: Frame index for title
    """
    if keypoints is None or frame_image is None:
        print("❌ No keypoints or frame image to visualize")
        return
    
    print(f"🎨 Creating visualization for {len(keypoints)} keypoints...")
    
    # Ensure we're working with the right data types
    if not isinstance(frame_image, np.ndarray):
        print("❌ Frame image is not a numpy array")
        return
    
    # Create figure with explicit backend
    plt.ioff()  # Turn off interactive mode
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
    
    # Original image
    ax1.imshow(frame_image)
    ax1.set_title(f'Original Frame\\nSequence: {seq_id}\\nFrame Index: {frame_index}', fontsize=12)
    ax1.axis('off')
    
    # Add bounding box to original if provided
    if bbox and isinstance(bbox, dict):
        left = bbox.get('left', 0)
        top = bbox.get('top', 0)
        width = bbox.get('width', 0)
        height = bbox.get('height', 0)
        
        if width > 0 and height > 0:
            rect = patches.Rectangle((left, top), width, height, 
                                   linewidth=2, edgecolor='red', facecolor='none')
            ax1.add_patch(rect)
    
    # Pose estimation result
    vis_image = frame_image.copy()
    
    # MediaPipe pose connections for better visualization
    pose_connections = [
        # Face outline
        (0, 1), (1, 2), (2, 3), (3, 7),  # Left side
        (0, 4), (4, 5), (5, 6), (6, 8),  # Right side
        (9, 10),  # Mouth
        # Upper body
        (11, 12),  # Shoulders
        (11, 13), (13, 15),  # Left arm
        (12, 14), (14, 16),  # Right arm
        (15, 17), (15, 19), (15, 21),  # Left hand
        (16, 18), (16, 20), (16, 22),  # Right hand
        # Torso
        (11, 23), (12, 24), (23, 24),  # Torso
        # Lower body
        (23, 25), (25, 27),  # Left leg
        (24, 26), (26, 28),  # Right leg
        (27, 29), (27, 31),  # Left foot
        (28, 30), (28, 32),  # Right foot
    ]
    
    # Draw keypoints and connections
    confident_keypoints = 0
    
    for i, kp in enumerate(keypoints):
        confidence = kp.get('confidence', 0)
        if confidence > 0.3:  # Lower threshold for visibility
            x, y = int(kp['x']), int(kp['y'])
            
            # Color based on confidence
            if confidence > 0.7:
                color = (0, 255, 0)  # Green for high confidence
                confident_keypoints += 1
            elif confidence > 0.5:
                color = (255, 255, 0)  # Yellow for medium confidence
            else:
                color = (255, 165, 0)  # Orange for low confidence
            
            # Draw keypoint
            cv2.circle(vis_image, (x, y), 5, color, -1)
            cv2.circle(vis_image, (x, y), 6, (255, 255, 255), 1)  # White border
            
            # Add keypoint number for important landmarks
            if i in [0, 11, 12, 13, 14, 15, 16, 23, 24, 25, 26, 27, 28]:  # Key landmarks
                cv2.putText(vis_image, str(i), (x+8, y-8), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
    
    # Draw connections
    for start_idx, end_idx in pose_connections:
        if (start_idx < len(keypoints) and end_idx < len(keypoints)):
            start_kp = keypoints[start_idx]
            end_kp = keypoints[end_idx]
            
            if (start_kp.get('confidence', 0) > 0.3 and end_kp.get('confidence', 0) > 0.3):
                start_point = (int(start_kp['x']), int(start_kp['y']))
                end_point = (int(end_kp['x']), int(end_kp['y']))
                
                # Line thickness based on confidence
                avg_conf = (start_kp.get('confidence', 0) + end_kp.get('confidence', 0)) / 2
                thickness = 3 if avg_conf > 0.6 else 2
                
                cv2.line(vis_image, start_point, end_point, (0, 255, 255), thickness)
    
    # Display annotated image
    ax2.imshow(vis_image)
    ax2.set_title(f'MediaPipe Pose Estimation\\n{len(keypoints)} landmarks, {confident_keypoints} high confidence', 
                  fontsize=12)
    ax2.axis('off')
    
    plt.tight_layout()
    plt.show()
    
    # Print detailed statistics
    if keypoints:
        confidences = [kp.get('confidence', 0) for kp in keypoints]
        high_conf = sum(1 for c in confidences if c > 0.7)
        medium_conf = sum(1 for c in confidences if 0.5 < c <= 0.7)
        low_conf = sum(1 for c in confidences if 0.3 < c <= 0.5)
        
        print(f"\\n📊 Pose Detection Statistics:")
        print(f"   🟢 High confidence (>0.7): {high_conf}/{len(keypoints)}")
        print(f"   🟡 Medium confidence (0.5-0.7): {medium_conf}/{len(keypoints)}")
        print(f"   🟠 Low confidence (0.3-0.5): {low_conf}/{len(keypoints)}")
        print(f"   📈 Average confidence: {np.mean(confidences):.3f}")
        print(f"   📏 Image dimensions: {frame_image.shape[1]}x{frame_image.shape[0]}")
    
    return fig
```

## Cell 8: Test Pose Estimation

```python
# Test pose estimation on the first sequence
if pose_estimator or native_pose_available:
    first_seq_id = list(sequences.keys())[0]
    print(f"🧪 Testing pose estimation on sequence: {first_seq_id}")
    
    # Test on frame index 10 (should have good pose visibility)
    keypoints, frame_image, bbox = estimate_pose_for_frame(sequences, first_seq_id, frame_index=10)
    
    if keypoints and frame_image is not None:
        visualize_pose_comprehensive(keypoints, frame_image, bbox, first_seq_id, 10)
    else:
        print("❌ Failed to estimate pose for the frame")
        
        # Try with native API as fallback
        if native_pose_available:
            print("🔄 Trying with native MediaPipe API...")
            keypoints, frame_image, bbox = estimate_pose_for_frame(sequences, first_seq_id, frame_index=10, use_native=True)
            if keypoints and frame_image is not None:
                visualize_pose_comprehensive(keypoints, frame_image, bbox, first_seq_id, 10)
else:
    print("❌ No pose estimation methods available. Please check MediaPipe installation.")
```

## Instructions:

1. **Copy each cell above** and paste it into a new code cell in your Jupyter notebook
2. **Run the cells in order** - each cell depends on the previous ones
3. **Make sure MediaPipe is installed**: `pip install mediapipe`
4. **The visualization should appear** after running Cell 8

The implementation includes:
- ✅ **Dual approach**: Uses both MediaPipeEstimator class and native MediaPipe API
- ✅ **Robust visualization**: Ensures plots display properly in Jupyter
- ✅ **Error handling**: Graceful fallbacks if one method fails
- ✅ **Detailed output**: Statistics and confidence analysis
- ✅ **Visual feedback**: Color-coded keypoints and connections based on confidence