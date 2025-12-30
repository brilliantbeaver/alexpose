# 🚀 **NEW MediaPipe Implementation (v0.10+)**

Based on the latest MediaPipe documentation, here's the **correct implementation** for the new API:

## 📋 **Cell 1: New MediaPipe Setup**

```python
# NEW MediaPipe API (v0.10+) - Correct Implementation
import numpy as np
import cv2
import matplotlib.pyplot as plt
from pathlib import Path
import tempfile
from typing import List, Dict, Optional, Tuple

%matplotlib inline

# Import NEW MediaPipe API
try:
    import mediapipe as mp
    from mediapipe.tasks import python
    from mediapipe.tasks.python import vision
    
    print("✅ NEW MediaPipe API imported successfully!")
    print(f"📦 MediaPipe version: {mp.__version__}")
    MEDIAPIPE_AVAILABLE = True
    
except ImportError as e:
    print(f"❌ MediaPipe import failed: {e}")
    print("💡 Install with: pip install mediapipe")
    MEDIAPIPE_AVAILABLE = False

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
```

## 📋 **Cell 2: Download Model**

```python
# Download the official MediaPipe pose model
import urllib.request

def download_pose_model():
    """Download the official MediaPipe pose landmarker model"""
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
        print(f"📍 Location: {model_path}")
        return str(model_path)
    except Exception as e:
        print(f"❌ Download failed: {e}")
        return None

# Download the model
if MEDIAPIPE_AVAILABLE:
    MODEL_PATH = download_pose_model()
else:
    MODEL_PATH = None
    print("⏭️ Skipping model download - MediaPipe not available")
```

## 📋 **Cell 3: Create Pose Landmarker**

```python
# Create the NEW MediaPipe Pose Landmarker
def create_pose_landmarker(model_path: str):
    """Create MediaPipe Pose Landmarker using the new API"""
    
    if not MEDIAPIPE_AVAILABLE or not model_path:
        print("❌ Cannot create landmarker - MediaPipe or model not available")
        return None
    
    try:
        # Create base options
        base_options = python.BaseOptions(model_asset_path=model_path)
        
        # Create pose landmarker options
        options = vision.PoseLandmarkerOptions(
            base_options=base_options,
            running_mode=vision.RunningMode.IMAGE,  # For single images
            num_poses=1,  # Detect up to 1 pose
            min_pose_detection_confidence=0.5,
            min_pose_presence_confidence=0.5,
            min_tracking_confidence=0.5,
            output_segmentation_masks=False  # Set to True if you want segmentation
        )
        
        # Create the landmarker
        landmarker = vision.PoseLandmarker.create_from_options(options)
        print("✅ Pose Landmarker created successfully!")
        return landmarker
        
    except Exception as e:
        print(f"❌ Failed to create Pose Landmarker: {e}")
        return None

# Create the landmarker
if MODEL_PATH:
    pose_landmarker = create_pose_landmarker(MODEL_PATH)
else:
    pose_landmarker = None
    print("⏭️ Skipping landmarker creation - model not available")
```

## 📋 **Cell 4: Frame Extraction Function**

```python
def extract_frame_from_video(video_path: Path, frame_number: int) -> Optional[np.ndarray]:
    """
    Extract a specific frame from video file.
    
    Args:
        video_path: Path to video file
        frame_number: Frame number to extract (1-based)
    
    Returns:
        RGB image array or None if failed
    """
    try:
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            print(f"❌ Cannot open video: {video_path}")
            return None
        
        # Get video properties
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        
        print(f"📹 Video: {total_frames} frames, {fps:.1f} FPS")
        
        # Convert to 0-based index
        frame_index = frame_number - 1
        
        if frame_index >= total_frames or frame_index < 0:
            print(f"❌ Frame {frame_number} out of range (1-{total_frames})")
            cap.release()
            return None
        
        # Seek and read frame
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
        ret, frame = cap.read()
        cap.release()
        
        if not ret:
            print(f"❌ Could not read frame {frame_number}")
            return None
        
        # Convert BGR to RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        print(f"✅ Extracted frame {frame_number}: {frame_rgb.shape}")
        return frame_rgb
        
    except Exception as e:
        print(f"❌ Frame extraction failed: {e}")
        return None

print("✅ Frame extraction function ready")
```

## 📋 **Cell 5: NEW Pose Detection Function**

```python
def detect_pose_new_api(image: np.ndarray, landmarker) -> Tuple[Optional[any], List[Dict]]:
    """
    Detect pose using NEW MediaPipe API
    
    Args:
        image: RGB image array
        landmarker: MediaPipe PoseLandmarker instance
    
    Returns:
        Tuple of (detection_result, keypoints_list)
    """
    if landmarker is None:
        print("❌ No pose landmarker available")
        return None, []
    
    try:
        # Convert numpy array to MediaPipe Image
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image)
        
        # Detect pose landmarks
        detection_result = landmarker.detect(mp_image)
        
        # Extract keypoints
        keypoints = []
        
        if detection_result.pose_landmarks:
            # Get first pose (we set num_poses=1)
            pose_landmarks = detection_result.pose_landmarks[0]
            
            height, width = image.shape[:2]
            
            for i, landmark in enumerate(pose_landmarks):
                keypoint = {
                    'id': i,
                    'name': POSE_LANDMARK_NAMES[i] if i < len(POSE_LANDMARK_NAMES) else f'LANDMARK_{i}',
                    'x': landmark.x * width,  # Convert normalized to pixel coordinates
                    'y': landmark.y * height,
                    'z': landmark.z,  # Depth (relative to hips)
                    'visibility': landmark.visibility,  # Likelihood of being visible
                    'presence': landmark.presence,  # Likelihood of being present
                    'confidence': landmark.visibility,  # Use visibility as confidence
                    'x_normalized': landmark.x,  # Keep normalized coordinates
                    'y_normalized': landmark.y,
                }
                keypoints.append(keypoint)
            
            print(f"✅ Detected pose with {len(keypoints)} landmarks")
        else:
            print("⚠️ No pose detected in image")
        
        return detection_result, keypoints
        
    except Exception as e:
        print(f"❌ Pose detection failed: {e}")
        import traceback
        traceback.print_exc()
        return None, []

print("✅ NEW pose detection function ready")
```

## 📋 **Cell 6: Visualization Function**

```python
def visualize_pose_new_api(image: np.ndarray, keypoints: List[Dict], 
                          title: str = "Pose Detection", show_connections: bool = True):
    """
    Visualize pose detection results using the new API
    
    Args:
        image: Original RGB image
        keypoints: List of keypoint dictionaries
        title: Plot title
        show_connections: Whether to draw skeleton connections
    """
    if not keypoints:
        print("❌ No keypoints to visualize")
        return
    
    # Create figure
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
    
    # Original image
    ax1.imshow(image)
    ax1.set_title('Original Image', fontsize=14)
    ax1.axis('off')
    
    # Create annotated image
    annotated = image.copy()
    
    # Define pose connections (MediaPipe standard connections)
    POSE_CONNECTIONS = [
        # Face
        (0, 1), (1, 2), (2, 3), (3, 7),  # Left eye to ear
        (0, 4), (4, 5), (5, 6), (6, 8),  # Right eye to ear
        (9, 10),  # Mouth
        # Arms
        (11, 12),  # Shoulders
        (11, 13), (13, 15),  # Left arm
        (12, 14), (14, 16),  # Right arm
        (15, 17), (15, 19), (15, 21),  # Left hand
        (16, 18), (16, 20), (16, 22),  # Right hand
        # Torso
        (11, 23), (12, 24), (23, 24),  # Shoulders to hips
        # Legs
        (23, 25), (25, 27),  # Left leg
        (24, 26), (26, 28),  # Right leg
        (27, 29), (27, 31),  # Left foot
        (28, 30), (28, 32),  # Right foot
    ]
    
    # Draw keypoints
    high_conf_count = 0
    for kp in keypoints:
        confidence = kp.get('confidence', 0)
        if confidence > 0.3:  # Only draw visible keypoints
            x, y = int(kp['x']), int(kp['y'])
            
            # Color based on confidence
            if confidence > 0.7:
                color = (0, 255, 0)  # Green - high confidence
                high_conf_count += 1
            elif confidence > 0.5:
                color = (255, 255, 0)  # Yellow - medium confidence
            else:
                color = (255, 165, 0)  # Orange - low confidence
            
            # Draw keypoint
            cv2.circle(annotated, (x, y), 6, color, -1)
            cv2.circle(annotated, (x, y), 7, (255, 255, 255), 1)  # White border
    
    # Draw connections
    if show_connections:
        for start_idx, end_idx in POSE_CONNECTIONS:
            if start_idx < len(keypoints) and end_idx < len(keypoints):
                start_kp = keypoints[start_idx]
                end_kp = keypoints[end_idx]
                
                if (start_kp.get('confidence', 0) > 0.3 and 
                    end_kp.get('confidence', 0) > 0.3):
                    
                    start_point = (int(start_kp['x']), int(start_kp['y']))
                    end_point = (int(end_kp['x']), int(end_kp['y']))
                    
                    cv2.line(annotated, start_point, end_point, (0, 255, 255), 3)
    
    # Show annotated image
    ax2.imshow(annotated)
    ax2.set_title(f'{title}\\n{len(keypoints)} landmarks, {high_conf_count} high confidence', 
                  fontsize=14)
    ax2.axis('off')
    
    plt.tight_layout()
    plt.show()
    
    # Print statistics
    if keypoints:
        confidences = [kp.get('confidence', 0) for kp in keypoints]
        avg_conf = np.mean(confidences)
        visible_count = sum(1 for c in confidences if c > 0.5)
        
        print(f"\\n📊 Pose Detection Statistics:")
        print(f"   🎯 Total landmarks: {len(keypoints)}")
        print(f"   👁️ Visible landmarks (>0.5): {visible_count}")
        print(f"   🟢 High confidence (>0.7): {high_conf_count}")
        print(f"   📈 Average confidence: {avg_conf:.3f}")
    
    return fig

print("✅ NEW visualization function ready")
```

## 📋 **Cell 7: Test with GAVD Data**

```python
# Test the NEW MediaPipe API with GAVD data
def test_new_mediapipe_with_gavd():
    """Test pose detection with GAVD video data using NEW API"""
    
    if not pose_landmarker:
        print("❌ Pose landmarker not available")
        return None
    
    try:
        # Get first sequence
        first_seq_id = list(sequences.keys())[0]
        seq_data = sequences[first_seq_id]
        
        print(f"🧪 Testing NEW MediaPipe API")
        print(f"📊 Sequence: {first_seq_id} ({len(seq_data)} frames)")
        
        # Get frame info (use frame 10)
        frame_row = seq_data.iloc[10]
        frame_num = int(frame_row['frame_num'])
        url = frame_row.get('url', '')
        bbox = frame_row.get('bbox', {})
        
        print(f"🎯 Processing frame {frame_num}")
        
        # Get video path
        video_id = extract_video_id(url) if url else None
        if not video_id:
            print("❌ No video ID found")
            return None
        
        video_cache_dir = project_root / "data" / "youtube"
        video_path = None
        for ext in ['.mp4', '.webm', '.mkv', '.mov']:
            candidate = video_cache_dir / f"{video_id}{ext}"
            if candidate.exists():
                video_path = candidate
                break
        
        if not video_path:
            print(f"❌ Video not found: {video_cache_dir / f'{video_id}.mp4'}")
            return None
        
        print(f"📹 Video found: {video_path.name}")
        
        # Extract frame
        frame_image = extract_frame_from_video(video_path, frame_num)
        if frame_image is None:
            return None
        
        # Apply bounding box crop if available
        cropped_image = frame_image
        crop_offset = (0, 0)
        
        if bbox and isinstance(bbox, dict):
            left = max(0, int(bbox.get('left', 0)))
            top = max(0, int(bbox.get('top', 0)))
            width = int(bbox.get('width', frame_image.shape[1]))
            height = int(bbox.get('height', frame_image.shape[0]))
            
            right = min(frame_image.shape[1], left + width)
            bottom = min(frame_image.shape[0], top + height)
            
            if width > 0 and height > 0:
                cropped_image = frame_image[top:bottom, left:right]
                crop_offset = (left, top)
                print(f"📏 Applied crop: ({left},{top}) to ({right},{bottom})")
        
        # Detect pose using NEW API
        print("🔍 Running pose detection...")
        detection_result, keypoints = detect_pose_new_api(cropped_image, pose_landmarker)
        
        # Adjust keypoint coordinates if we cropped
        if crop_offset != (0, 0):
            for kp in keypoints:
                kp['x'] += crop_offset[0]
                kp['y'] += crop_offset[1]
        
        # Visualize results
        if keypoints:
            visualize_pose_new_api(frame_image, keypoints, 
                                 f"NEW MediaPipe API - Frame {frame_num}")
        else:
            print("⚠️ No pose detected")
        
        return detection_result, keypoints, frame_image
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return None

# Run the test
print("🚀 Testing NEW MediaPipe API with GAVD data...")
test_results = test_new_mediapipe_with_gavd()

if test_results:
    print("\\n🎉 SUCCESS! NEW MediaPipe API is working correctly!")
else:
    print("\\n❌ Test failed - check error messages above")
```

---

## 🎯 **Key Changes from Old to New API:**

### **Old API (mp.solutions) - DEPRECATED:**
```python
# ❌ OLD - This doesn't work anymore
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils
```

### **New API (mp.tasks) - CURRENT:**
```python
# ✅ NEW - This is the correct approach
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# Create landmarker
base_options = python.BaseOptions(model_asset_path=model_path)
options = vision.PoseLandmarkerOptions(base_options=base_options)
landmarker = vision.PoseLandmarker.create_from_options(options)
```

## 🚀 **Instructions:**

1. **Copy each cell above** into your Jupyter notebook
2. **Run cells in order** (1 → 2 → 3 → 4 → 5 → 6 → 7)
3. **Cell 7 will test** with your GAVD video data
4. **You should see** pose detection visualization!

This implementation uses the **official new MediaPipe API** and should work perfectly with your MediaPipe installation!