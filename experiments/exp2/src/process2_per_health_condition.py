##########################################################################################
# Load an process all GAVD CSV files in a health condition directory
#
# Date: Mon Jan 19 10:09:59 PST 2026
##########################################################################################

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

# Set up paths using utility function
from ambient.utils.path_utils import get_project_root

project_root = get_project_root()
condition_path = project_root / "experiments" / "exp2" / "data" / "parkinsons"
video_base_path = project_root / "data" / "youtube"

print(f"==> project_root: {project_root}")
print(f"--> condition_path: {condition_path}")

# Load data (this will auto-download YouTube videos to data/youtube)
gavd_loader = GAVDDataLoader()

assert Path(condition_path).exists(), "Condition path does not exist"

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

        print(f"==> {type(sequences[seq_id])}: {len(keypoints_array)}: {len(joint_angles.frames)}")

        for i in range(len(joint_angles.frames[0].angles)):
            avg_joint_angle = joint_angles.get_statistics(joint_name=list(joint_angles.frames[0].angles.keys())[i])["mean"]
            print(f"\t<joint angle> of {list(joint_angles.frames[0].angles.keys())[i]}: {avg_joint_angle}")
