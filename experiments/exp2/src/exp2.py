import sys
import pandas as pd
from typing import Dict, List
from pathlib import Path

from ambient.gavd import GaitDataProcessor, GAVDDataLoader, PoseDataConverter
from ambient.pose.pose_estimators import OpenPoseEstimator
from ambient.utils.csv_parser import parse_csv_with_dicts
from ambient.pose.keypoint_extractor import SequenceKeypointExtractor
from ambient.pose.joint_angles import get_joint_angles

# Set up paths
project_root = Path.cwd().parent.parent
csv_path = project_root / "data" / "gavd" / "parkinsons" / "cljan9b4p00043n6ligceanyp.csv"
video_base_path = project_root / "data" / "youtube"

print(f"csv_path: {csv_path}")

# Load data (this will auto-download YouTube videos to data/youtube)
gavd_loader = GAVDDataLoader()
df = gavd_loader.load_gavd_data(str(csv_path))
sequences = gavd_loader.organize_by_sequence(df)

print(type(sequences))
for seq in sequences:
    print(f"==> {type(seq)}")



# # Extract keypoints
# extractor = SequenceKeypointExtractor()
# keypoints_array = extractor.extract_from_sequence(
#     sequence_data,
#     video_base_path=video_base_path
# )

# joint_angles = get_joint_angles(
#     keypoints_array=keypoints_array,
#     keypoint_format="BLAZEPOSE_33",
#     fps=30.0,
#     confidence_threshold=0.3
# )

# for i in range(len(joint_angles.frames[0].angles)):
#     avg_joint_angle = joint_angles.get_statistics(joint_name=list(joint_angles.frames[0].angles.keys())[i])["mean"]
#     print(f"Average joint angle of {list(joint_angles.frames[0].angles.keys())[i]}: {avg_joint_angle}")
