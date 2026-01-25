"""
Diagnostic script to identify why extract_from_sequence returns empty array.
"""
import sys
from pathlib import Path

# Setup paths
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

import pandas as pd
from ambient.gavd import GAVDDataLoader
from ambient.pose.keypoint_extractor import SequenceKeypointExtractor

# Paths
data_root = project_root / "experiments" / "exp3" / "data"
video_base_path = project_root / "data" / "youtube"

# Load a sample CSV
normal_path = data_root / "normal"
csv_files = list(normal_path.glob("*.csv"))
if not csv_files:
    print(f"❌ No CSV files found in {normal_path}")
    sys.exit(1)

sample_csv = csv_files[0]
print(f"📄 Loading: {sample_csv.name}\n")

# Load data
gavd_loader = GAVDDataLoader()
df = gavd_loader.load_gavd_data(sample_csv)

print(f"✅ DataFrame loaded: {df.shape}")
print(f"\n📊 Columns in DataFrame:")
for i, col in enumerate(df.columns, 1):
    print(f"   {i}. {col}")

print(f"\n🔍 First row sample:")
print(df.iloc[0].to_dict())

# Check for required columns
required_cols = ['frame_num', 'url']
missing_cols = [col for col in required_cols if col not in df.columns]

if missing_cols:
    print(f"\n❌ PROBLEM FOUND: Missing required columns: {missing_cols}")
    print(f"\n💡 Suggested fixes:")
    
    # Check for similar column names
    if 'frame_num' in missing_cols:
        frame_cols = [c for c in df.columns if 'frame' in c.lower()]
        if frame_cols:
            print(f"   - Found frame-related columns: {frame_cols}")
            print(f"   - Consider renaming '{frame_cols[0]}' to 'frame_num'")
    
    if 'url' in missing_cols:
        url_cols = [c for c in df.columns if 'url' in c.lower() or 'video' in c.lower()]
        if url_cols:
            print(f"   - Found URL-related columns: {url_cols}")
            print(f"   - Consider renaming '{url_cols[0]}' to 'url'")
else:
    print(f"\n✅ All required columns present: {required_cols}")
    
    # Try extraction
    print(f"\n🔄 Attempting keypoint extraction...")
    extractor = SequenceKeypointExtractor()
    
    # Take only first 5 frames for quick test
    test_df = df.head(5)
    
    try:
        keypoints = extractor.extract_from_sequence(
            sequence_data=test_df,
            video_base_path=video_base_path,
            verbose=True
        )
        
        if keypoints:
            print(f"\n✅ SUCCESS: Extracted {len(keypoints)} keypoint sets")
            print(f"   - Successful frames: {sum(1 for k in keypoints if k is not None)}")
            print(f"   - Failed frames: {sum(1 for k in keypoints if k is None)}")
        else:
            print(f"\n❌ FAILED: Returned empty array")
            print(f"   Check logs above for error messages")
    except Exception as e:
        print(f"\n❌ EXCEPTION: {e}")
        import traceback
        traceback.print_exc()

print("\n" + "="*60)
