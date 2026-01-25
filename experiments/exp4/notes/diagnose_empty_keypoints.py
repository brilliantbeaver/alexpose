"""
Diagnose why some frames have 0 keypoints.
"""
import sys
from pathlib import Path

project_root = Path.cwd().parent.parent
video_base_path = project_root / "data" / "youtube"
data_root = project_root / "experiments" / "exp3" / "data"

sys.path.insert(0, str(project_root))

from ambient.gavd import GAVDDataLoader
from ambient.pose.keypoint_extractor import SequenceKeypointExtractor
import cv2
import numpy as np

print("="*70)
print("DIAGNOSING EMPTY KEYPOINTS")
print("="*70)
print()

# Load the problematic sequence
target_seq = "cljo30lnz001q3n6lopfty7q5"
csv_file = data_root / "normal" / f"{target_seq}.csv"

gavd_loader = GAVDDataLoader()
normal_df = gavd_loader.load_gavd_data(csv_file)

print(f"Sequence: {target_seq}")
print(f"Total frames: {len(normal_df)}")
print()

# Extract keypoints
extractor = SequenceKeypointExtractor()
keypoints_array = extractor.extract_from_sequence(
    sequence_data=normal_df.head(10),  # First 10 frames
    video_base_path=video_base_path,
    verbose=False
)

print(f"Extracted: {len(keypoints_array)} frames")
print()

# Analyze results
print("Frame-by-frame analysis:")
print("-" * 70)

empty_frames = []
valid_frames = []

for i, kp_set in enumerate(keypoints_array):
    frame_num = normal_df.iloc[i]['frame_num']
    
    if kp_set is None:
        status = "❌ FAILED"
        num_kp = 0
        empty_frames.append(i)
    elif len(kp_set.keypoints) == 0:
        status = "⚠️  EMPTY"
        num_kp = 0
        empty_frames.append(i)
    else:
        status = "✅ OK"
        num_kp = len(kp_set.keypoints)
        valid_frames.append(i)
    
    print(f"Frame {i:2d} (video frame {frame_num:3d}): {status} - {num_kp} keypoints")

print("-" * 70)
print()

# Summary
print("Summary:")
print(f"  ✅ Valid frames: {len(valid_frames)}")
print(f"  ⚠️  Empty frames: {len(empty_frames)}")
print(f"  Success rate: {len(valid_frames)/len(keypoints_array)*100:.1f}%")
print()

# Investigate why frames are empty
if empty_frames:
    print("Why are frames empty?")
    print()
    
    # Check first empty frame
    first_empty = empty_frames[0]
    frame_num = normal_df.iloc[first_empty]['frame_num']
    url = normal_df.iloc[first_empty]['url']
    
    print(f"Investigating frame {first_empty} (video frame {frame_num}):")
    print()
    
    # Extract the actual frame image
    from ambient.utils.youtube_cache import extract_video_id
    video_id = extract_video_id(url)
    video_path = video_base_path / f"{video_id}.mp4"
    
    # Read frame
    cap = cv2.VideoCapture(str(video_path))
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num - 1)
    ret, frame = cap.read()
    cap.release()
    
    if ret:
        height, width = frame.shape[:2]
        print(f"  Frame dimensions: {width}x{height}")
        
        # Check if frame is mostly black/white
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        mean_brightness = np.mean(gray)
        std_brightness = np.std(gray)
        
        print(f"  Mean brightness: {mean_brightness:.1f} (0=black, 255=white)")
        print(f"  Std brightness: {std_brightness:.1f} (low=uniform)")
        
        # Check for person-sized regions
        if mean_brightness < 30:
            print(f"  ⚠️  Frame is very dark - person may not be visible")
        elif mean_brightness > 225:
            print(f"  ⚠️  Frame is very bright - person may be washed out")
        
        if std_brightness < 20:
            print(f"  ⚠️  Frame has low contrast - may be blank or uniform")
        
        # Check bbox if available
        if 'bbox' in normal_df.columns:
            bbox = normal_df.iloc[first_empty]['bbox']
            if bbox and isinstance(bbox, dict):
                print(f"  Bounding box: {bbox}")
                
                # Check if bbox is reasonable
                bbox_width = bbox.get('width', 0)
                bbox_height = bbox.get('height', 0)
                
                if bbox_width < 50 or bbox_height < 50:
                    print(f"  ⚠️  Bounding box is very small - person may be far away")
                
                # Check if person is in frame
                bbox_left = bbox.get('left', 0)
                bbox_top = bbox.get('top', 0)
                
                if bbox_left < 0 or bbox_top < 0:
                    print(f"  ⚠️  Bounding box is partially outside frame")
                
                if bbox_left + bbox_width > width or bbox_top + bbox_height > height:
                    print(f"  ⚠️  Bounding box extends beyond frame")
    else:
        print(f"  ❌ Could not read frame from video")
    
    print()
    print("Common reasons for empty keypoints:")
    print("  1. Person is not visible in frame (occluded, out of frame)")
    print("  2. Frame is too dark or too bright")
    print("  3. Person is too small or too far away")
    print("  4. Frame is blurry or low quality")
    print("  5. Multiple people in frame (MediaPipe picks strongest)")
    print("  6. Person is in unusual pose (lying down, bent over)")
    print()
    
    print("Solutions:")
    print("  1. Filter out frames with 0 keypoints before analysis")
    print("  2. Use confidence thresholds to filter low-quality detections")
    print("  3. Check video quality and lighting conditions")
    print("  4. Verify bounding boxes are correct")

print()
print("="*70)
