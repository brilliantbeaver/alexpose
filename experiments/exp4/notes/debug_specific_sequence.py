"""
Debug specific sequence: cljo30lnz001q3n6lopfty7q5
"""
import sys
from pathlib import Path

# Setup paths
project_root = Path.cwd().parent.parent
video_base_path = project_root / "data" / "youtube"
data_root = project_root / "experiments" / "exp3" / "data"

sys.path.insert(0, str(project_root))

print("="*70)
print("DEBUGGING SEQUENCE: cljo30lnz001q3n6lopfty7q5")
print("="*70)
print()

from ambient.gavd import GAVDDataLoader
from ambient.pose.keypoint_extractor import SequenceKeypointExtractor

# Find the CSV file
target_seq = "cljo30lnz001q3n6lopfty7q5"
csv_file = data_root / "normal" / f"{target_seq}.csv"

print(f"📄 CSV file: {csv_file}")
print(f"   Exists: {csv_file.exists()}")
print()

if not csv_file.exists():
    print("❌ CSV file not found!")
    sys.exit(1)

# Load data
gavd_loader = GAVDDataLoader()
normal_df = gavd_loader.load_gavd_data(csv_file)

print(f"✅ DataFrame loaded")
print(f"   Shape: {normal_df.shape}")
print(f"   Columns: {list(normal_df.columns)}")
print()

# Check required columns
required = ['frame_num', 'url']
has_required = all(col in normal_df.columns for col in required)
print(f"Required columns {required}: {'✅' if has_required else '❌'}")

if not has_required:
    missing = [col for col in required if col not in normal_df.columns]
    print(f"   Missing: {missing}")
    sys.exit(1)

print()

# Check unique URLs
unique_urls = normal_df['url'].unique()
print(f"📹 Unique video URLs: {len(unique_urls)}")
for url in unique_urls:
    print(f"   {url}")
print()

# Check video files exist
from ambient.utils.youtube_cache import extract_video_id

print("🔍 Checking video files:")
for url in unique_urls:
    try:
        video_id = extract_video_id(url)
        video_path = video_base_path / f"{video_id}.mp4"
        exists = video_path.exists()
        status = "✅" if exists else "❌"
        print(f"   {status} {video_id}.mp4 - {video_path}")
        
        if exists:
            # Check file size
            size_mb = video_path.stat().st_size / (1024 * 1024)
            print(f"      Size: {size_mb:.2f} MB")
    except Exception as e:
        print(f"   ❌ Error extracting video ID: {e}")

print()

# Check frame numbers
print(f"📊 Frame number range:")
print(f"   Min: {normal_df['frame_num'].min()}")
print(f"   Max: {normal_df['frame_num'].max()}")
print(f"   Count: {len(normal_df)}")
print()

# Show first few rows
print("📋 First 3 rows:")
print(normal_df[['seq', 'frame_num', 'url']].head(3))
print()

# Now try extraction with verbose diagnostics
print("="*70)
print("ATTEMPTING EXTRACTION")
print("="*70)
print()

extractor = SequenceKeypointExtractor()

# Use the new validation method
is_valid, message = extractor.validate_sequence_data_verbose(normal_df, video_base_path)
print(f"Validation: {'✅ PASS' if is_valid else '❌ FAIL'}")
print(f"Message: {message}")
print()

if not is_valid:
    print("❌ Validation failed - extraction will return empty array")
    sys.exit(1)

# Try with just first 3 frames
print("Extracting first 3 frames with verbose=True...")
print()

test_df = normal_df.head(3)
keypoints_array = extractor.extract_from_sequence(
    sequence_data=test_df,
    video_base_path=video_base_path,
    verbose=True
)

print()
print("="*70)
print("RESULTS")
print("="*70)
print(f"Returned array length: {len(keypoints_array)}")

if keypoints_array:
    successful = sum(1 for k in keypoints_array if k is not None)
    failed = len(keypoints_array) - successful
    print(f"✅ Successful frames: {successful}")
    print(f"❌ Failed frames: {failed}")
    
    if successful > 0:
        # Show first successful keypoint set
        first_success = next((k for k in keypoints_array if k is not None), None)
        if first_success:
            print(f"\n📍 First successful keypoint set:")
            print(f"   Format: {first_success.format}")
            print(f"   Keypoints: {len(first_success.keypoints)}")
            print(f"   Frame size: {first_success.frame_width}x{first_success.frame_height}")
else:
    print("❌ EMPTY ARRAY - No frames processed!")
    print("\nPossible causes:")
    print("1. Validation failed (check message above)")
    print("2. Landmarker initialization failed")
    print("3. All frames failed individual processing")
    print("\nCheck the logs above for ERROR messages")
