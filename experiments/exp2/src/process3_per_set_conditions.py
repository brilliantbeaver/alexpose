##########################################################################################
# Load an process all GAVD CSV files across set of health condition sub-directories
#
# Date: Mon Jan 19 10:09:59 PST 2026
##########################################################################################

import string
import sys
import pandas as pd
from typing import Dict, List
from pathlib import Path

from ambient.gavd import GaitDataProcessor, GAVDDataLoader, PoseDataConverter
from ambient.pose.pose_estimators import OpenPoseEstimator
from ambient.pose.keypoint_extractor import SequenceKeypointExtractor
from ambient.pose.joint_angles import get_joint_angles
from ambient.utils.csv_parser import parse_csv_with_dicts
from ambient.utils.log_config import get_logger

logger = get_logger()
logger.remove()
logger.add(sys.stderr, level="WARNING")

# Set up paths
project_root = Path.cwd()
video_base_path = project_root / "data" / "youtube"
data_root = project_root / "experiments" / "exp2" / "data"

assert Path(video_base_path).exists(), "video_base_path does not exist"
assert Path(data_root).exists(), "data_root does not exist"

# ignore any "dot" files
condition_paths = [
    p for p in Path(data_root).iterdir() 
    if p.is_dir() and p.name[0] in string.ascii_letters
]

print(f"Examining {len(condition_paths)} number of conditions ...")

for condition_path in condition_paths:
    print(f"==> condition_path: {condition_path}")
    try:
        # Load data (this will auto-download YouTube videos to data/youtube)
        gavd_loader = GAVDDataLoader()

        for csv_path in Path(condition_path).glob("*.csv"):
            df = gavd_loader.load_gavd_data(str(csv_path))
            sequences = gavd_loader.organize_by_sequence(df)

            extractor = SequenceKeypointExtractor()
            for seq_id in sequences:
                sequence_df = sequences[seq_id]
                keypoints_array = extractor.extract_from_sequence(
                    sequence_df,
                    video_base_path=video_base_path
                )

                joint_angles = get_joint_angles(
                    keypoints_array=keypoints_array,
                    keypoint_format="BLAZEPOSE_33",
                    fps=30.0,
                    confidence_threshold=0.3
                )

                print(f"Average joint angles for {len(joint_angles.frames)} frames:")
                for i in range(len(joint_angles.frames[0].angles)):
                    avg_joint_angle = joint_angles.get_statistics(joint_name=list(joint_angles.frames[0].angles.keys())[i])["mean"]
                    print(f"\t<joint angle> of {list(joint_angles.frames[0].angles.keys())[i]}: {avg_joint_angle}")
    except Exception as e:
        logger.error(f"Cannot process: {condition_path}")