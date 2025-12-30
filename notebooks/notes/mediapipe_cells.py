# MediaPipe Pose Estimation Implementation for Jupyter Notebook
# This file contains the code cells to be added to the notebook

# Cell 1: Setup and Imports
setup_cell = '''
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
'''

# Cell 2: MediaPipe Native Setup
native_setup_cell = '''
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
'''

# Cell 3: Model Download
model_download_cell = '''
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
'''

# Cell 4: Initialize Estimators
initialize_cell = '''
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
print("\\n🔧 Available Pose Estimation Methods:")
if pose_estimator:
    print("  ✅ MediaPipeEstimator class (recommended)")
if native_pose_available:
    print("  ✅ Native MediaPipe API (fallback)")
if not pose_estimator and not native_pose_available:
    print("  ❌ No pose estimation methods available")
'''

# Cell 5: Core Functions
functions_cell = '''
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
'''

print("MediaPipe implementation cells created successfully!")
print("Copy and paste each cell into the Jupyter notebook after Step 5.")