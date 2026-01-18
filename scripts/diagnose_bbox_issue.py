"""
Diagnostic script to understand the bounding box offset issue.

This script will:
1. Load a GAVD CSV file
2. Check the bbox and vid_info values
3. Download the video and check its actual resolution
4. Calculate what the scaling should be
5. Identify where the mismatch is occurring
"""

import sys
import json
from pathlib import Path
import cv2

sys.path.insert(0, str(Path(__file__).parent.parent))

from ambient.utils.csv_parser import parse_csv_with_dicts
from ambient.video.youtube_handler import YouTubeHandler

def diagnose_bbox_issue(csv_file: str, row_index: int = 0):
    """
    Diagnose bounding box scaling issues for a specific GAVD row.
    
    Args:
        csv_file: Path to GAVD CSV file
        row_index: Which row to analyze (default: first row)
    """
    print("=" * 80)
    print("BOUNDING BOX DIAGNOSTIC TOOL")
    print("=" * 80)
    
    # Load CSV
    print(f"\n1. Loading GAVD CSV: {csv_file}")
    rows = parse_csv_with_dicts(csv_file, dict_fields=['bbox', 'vid_info'])
    
    if not rows:
        print("ERROR: CSV file is empty!")
        return
    
    if row_index >= len(rows):
        print(f"ERROR: Row index {row_index} out of range (only {len(rows)} rows)")
        return
    
    row = rows[row_index]
    print(f"   ✓ Loaded {len(rows)} rows")
    print(f"   ✓ Analyzing row {row_index}")
    
    # Extract data
    print(f"\n2. Extracting data from row:")
    seq = row.get('seq', 'unknown')
    frame_num = row.get('frame_num', 'unknown')
    url = row.get('url', '')
    bbox = row.get('bbox', {})
    vid_info = row.get('vid_info', {})
    
    print(f"   - Sequence: {seq}")
    print(f"   - Frame: {frame_num}")
    print(f"   - URL: {url[:50]}...")
    print(f"   - Bounding Box: {json.dumps(bbox, indent=6)}")
    print(f"   - Vid Info: {json.dumps(vid_info, indent=6)}")
    
    # Check bbox coordinate space
    print(f"\n3. Analyzing bounding box coordinate space:")
    bbox_left = bbox.get('left', 0)
    bbox_top = bbox.get('top', 0)
    bbox_width = bbox.get('width', 0)
    bbox_height = bbox.get('height', 0)
    bbox_right = bbox_left + bbox_width
    bbox_bottom = bbox_top + bbox_height
    
    vid_info_width = vid_info.get('width', 0)
    vid_info_height = vid_info.get('height', 0)
    
    print(f"   - Bbox bounds: left={bbox_left}, top={bbox_top}, right={bbox_right}, bottom={bbox_bottom}")
    print(f"   - Vid_info dimensions: {vid_info_width}x{vid_info_height}")
    
    if bbox_right <= vid_info_width and bbox_bottom <= vid_info_height:
        print(f"   ✓ Bbox fits within vid_info dimensions")
        print(f"   ✓ Bbox is in vid_info coordinate space ({vid_info_width}x{vid_info_height})")
    else:
        print(f"   ⚠️  Bbox extends beyond vid_info dimensions!")
        print(f"   ⚠️  This suggests a coordinate space mismatch")
    
    # Download video and check actual resolution
    print(f"\n4. Downloading video to check actual resolution:")
    try:
        youtube_handler = YouTubeHandler()
        video_path = youtube_handler.get_or_download_video(url)
        
        if not video_path:
            print("   ❌ Failed to download video")
            return
        
        print(f"   ✓ Video downloaded: {video_path}")
        
        # Check video resolution
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            print("   ❌ Failed to open video")
            return
        
        actual_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        actual_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()
        
        print(f"   ✓ Actual video resolution: {actual_width}x{actual_height}")
        print(f"   ✓ FPS: {fps}")
        print(f"   ✓ Frame count: {frame_count}")
        
    except Exception as e:
        print(f"   ❌ Error checking video: {e}")
        return
    
    # Calculate scaling factors
    print(f"\n5. Calculating scaling factors:")
    print(f"   - Vid_info (annotation space): {vid_info_width}x{vid_info_height}")
    print(f"   - Actual video (display space): {actual_width}x{actual_height}")
    
    scale_x = actual_width / vid_info_width if vid_info_width > 0 else 1.0
    scale_y = actual_height / vid_info_height if vid_info_height > 0 else 1.0
    
    print(f"   - Scale factors: X={scale_x:.4f}, Y={scale_y:.4f}")
    
    if scale_x < 1.0 or scale_y < 1.0:
        print(f"   ⚠️  Video is SMALLER than annotation space!")
        print(f"   ⚠️  This will cause bbox to appear LARGER and OFFSET")
    elif scale_x > 1.0 or scale_y > 1.0:
        print(f"   ⚠️  Video is LARGER than annotation space!")
        print(f"   ⚠️  This will cause bbox to appear SMALLER")
    else:
        print(f"   ✓ Video matches annotation space exactly")
    
    # Calculate scaled bbox
    print(f"\n6. Calculating scaled bounding box:")
    scaled_left = bbox_left * scale_x
    scaled_top = bbox_top * scale_y
    scaled_width = bbox_width * scale_x
    scaled_height = bbox_height * scale_y
    scaled_right = scaled_left + scaled_width
    scaled_bottom = scaled_top + scaled_height
    
    print(f"   - Original bbox (vid_info space): [{bbox_left}, {bbox_top}, {bbox_right}, {bbox_bottom}]")
    print(f"   - Scaled bbox (video space): [{scaled_left:.1f}, {scaled_top:.1f}, {scaled_right:.1f}, {scaled_bottom:.1f}]")
    print(f"   - Scaled dimensions: {scaled_width:.1f}x{scaled_height:.1f}")
    
    if scaled_right > actual_width or scaled_bottom > actual_height:
        print(f"   ⚠️  Scaled bbox extends beyond video bounds!")
    else:
        print(f"   ✓ Scaled bbox fits within video bounds")
    
    # Summary
    print(f"\n" + "=" * 80)
    print("DIAGNOSIS SUMMARY")
    print("=" * 80)
    
    if abs(scale_x - 1.0) < 0.01 and abs(scale_y - 1.0) < 0.01:
        print("✅ No scaling issue detected - video matches annotation space")
    else:
        print(f"⚠️  SCALING MISMATCH DETECTED!")
        print(f"   - Annotation space: {vid_info_width}x{vid_info_height}")
        print(f"   - Actual video: {actual_width}x{actual_height}")
        print(f"   - Scale factors: {scale_x:.4f}x, {scale_y:.4f}y")
        print(f"\n   ROOT CAUSE:")
        if scale_x < 1.0:
            print(f"   - Video downloaded at LOWER resolution than annotations")
            print(f"   - Frontend IS scaling bbox correctly")
            print(f"   - BUT pose keypoints might NOT be scaled correctly")
        else:
            print(f"   - Video downloaded at HIGHER resolution than annotations")
            print(f"   - This should make bbox appear correct")
    
    print("=" * 80)

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python diagnose_bbox_issue.py <path_to_gavd_csv> [row_index]")
        print("\nExample:")
        print("  python diagnose_bbox_issue.py data/gavd/dataset.csv 0")
        sys.exit(1)
    
    csv_file = sys.argv[1]
    row_index = int(sys.argv[2]) if len(sys.argv) > 2 else 0
    
    diagnose_bbox_issue(csv_file, row_index)
