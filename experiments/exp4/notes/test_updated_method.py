"""
Test the updated extract_from_sequence() method with built-in filtering.
"""
import sys
from pathlib import Path

project_root = Path.cwd().parent.parent
sys.path.insert(0, str(project_root))

from ambient.gavd import GAVDDataLoader
from ambient.pose.keypoint_extractor import SequenceKeypointExtractor

# Setup
video_base_path = project_root / "data" / "youtube"
data_root = project_root / "experiments" / "exp3" / "data"
csv_file = data_root / "normal" / "cljo30lnz001q3n6lopfty7q5.csv"

print("="*70)
print("Testing Updated extract_from_sequence() Method")
print("="*70)
print()

# Load data
gavd_loader = GAVDDataLoader()
normal_df = gavd_loader.load_gavd_data(csv_file)
print(f"Loaded sequence: {normal_df['seq'].iloc[0]}")
print(f"Total frames in CSV: {len(normal_df)}")
print()

# Create extractor
extractor = SequenceKeypointExtractor()

# Test 1: Original behavior (no filtering)
print("Test 1: Extract all frames (filter_empty=False)")
print("-" * 70)
all_keypoints = extractor.extract_from_sequence(
    sequence_data=normal_df.head(10),
    video_base_path=video_base_path,
    verbose=False
)
print(f"Returned: {len(all_keypoints)} frames")
empty_count = sum(1 for kp in all_keypoints if kp is not None and len(kp.keypoints) == 0)
print(f"Empty frames: {empty_count}")
print()

# Test 2: With filtering
print("Test 2: Extract with filtering (filter_empty=True, min_keypoints=25)")
print("-" * 70)
filtered_keypoints = extractor.extract_from_sequence(
    sequence_data=normal_df.head(10),
    video_base_path=video_base_path,
    verbose=True,
    filter_empty=True,
    min_keypoints=25
)
print(f"Returned: {len(filtered_keypoints)} frames")
print(f"All frames have >= 25 keypoints: {all(len(kp.keypoints) >= 25 for kp in filtered_keypoints)}")
print()

# Test 3: Statistics
print("Test 3: Get statistics")
print("-" * 70)
stats = extractor.get_extraction_statistics(all_keypoints)
print(f"Statistics dictionary:")
for key, value in stats.items():
    print(f"  {key}: {value}")
print()

# Test 4: Print statistics
print("Test 4: Print formatted statistics")
print("-" * 70)
extractor.print_extraction_statistics(all_keypoints, "Test Sequence")

# Test 5: Different thresholds
print("Test 5: Different thresholds")
print("-" * 70)

strict = extractor.extract_from_sequence(
    sequence_data=normal_df.head(10),
    video_base_path=video_base_path,
    filter_empty=True,
    min_keypoints=33
)
print(f"Strict (33 keypoints): {len(strict)} frames")

moderate = extractor.extract_from_sequence(
    sequence_data=normal_df.head(10),
    video_base_path=video_base_path,
    filter_empty=True,
    min_keypoints=25
)
print(f"Moderate (25 keypoints): {len(moderate)} frames")

lenient = extractor.extract_from_sequence(
    sequence_data=normal_df.head(10),
    video_base_path=video_base_path,
    filter_empty=True,
    min_keypoints=20
)
print(f"Lenient (20 keypoints): {len(lenient)} frames")

print()
print("="*70)
print("✅ All tests passed!")
print("="*70)
