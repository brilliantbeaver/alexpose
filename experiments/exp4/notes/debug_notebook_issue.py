"""
Debug script that exactly replicates the notebook workflow.
"""
import sys
from pathlib import Path

# Setup paths EXACTLY like notebook
project_root = Path.cwd().parent.parent  # From exp4 directory
video_base_path = project_root / "data" / "youtube"
data_root = project_root / "experiments" / "exp3" / "data"

sys.path.insert(0, str(project_root))

print(f"Project root: {project_root}")
print(f"Data root: {data_root}")
print(f"Video path: {video_base_path}")
print(f"Data root exists: {data_root.exists()}")
print()

from ambient.gavd import GAVDDataLoader
from ambient.pose.keypoint_extractor import SequenceKeypointExtractor

# Get condition directories
import string
condition_paths = [
    p for p in data_root.iterdir() 
    if p.is_dir() and p.name[0] in string.ascii_letters
]

print(f"Found {len(condition_paths)} conditions:")
for path in sorted(condition_paths):
    csv_files = list(path.glob("*.csv"))
    print(f"  {path.name:15s}: {len(csv_files)} CSV files")
print()

if not condition_paths:
    print("❌ NO CONDITION DIRECTORIES FOUND!")
    print(f"   Check if {data_root} exists and has subdirectories")
    sys.exit(1)

# Use normal path like notebook
normal_path = [p for p in condition_paths if 'normal' in p.name.lower()]
if not normal_path:
    print("❌ NO 'normal' DIRECTORY FOUND!")
    sys.exit(1)

normal_path = normal_path[0]
normal_csv = list(normal_path.glob("*.csv"))[0]

print(f"Testing with: {normal_csv.name}")
print()

# Load data EXACTLY like notebook
gavd_loader = GAVDDataLoader()
normal_df = gavd_loader.load_gavd_data(normal_csv)

print(f"DataFrame shape: {normal_df.shape}")
print(f"Columns: {list(normal_df.columns)}")
print()

# Check required columns
required = ['frame_num', 'url']
has_required = all(col in normal_df.columns for col in required)
print(f"Has required columns {required}: {has_required}")

if not has_required:
    missing = [col for col in required if col not in normal_df.columns]
    print(f"❌ Missing: {missing}")
    sys.exit(1)

print()
print("Extracting keypoints (first 3 frames only)...")
print()

# Extract EXACTLY like notebook
extractor = SequenceKeypointExtractor()
normal_keypoints_array = extractor.extract_from_sequence(
    sequence_data=normal_df.head(3),  # Just 3 frames for speed
    video_base_path=video_base_path,
    verbose=True
)

print()
print(f"Result: {len(normal_keypoints_array)} keypoint sets")
if normal_keypoints_array:
    successful = sum(1 for k in normal_keypoints_array if k is not None)
    print(f"  ✅ Successful: {successful}")
    print(f"  ❌ Failed: {len(normal_keypoints_array) - successful}")
else:
    print("  ❌ EMPTY ARRAY RETURNED!")
