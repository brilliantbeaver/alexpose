##########################################################################################
# Load an process all GAVD CSV files across set of health condition sub-directories
#
# Date: Mon Jan 19 10:09:59 PST 2026
##########################################################################################

# Suppress TensorFlow and MediaPipe logs BEFORE importing
import os
import sys
import contextlib

# Set environment variables to suppress logs
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Suppress TensorFlow logs
os.environ['GLOG_minloglevel'] = '3'      # Suppress MediaPipe/glog logs
os.environ['GLOG_logtostderr'] = '0'      # Don't log to stderr
os.environ['GLOG_stderrthreshold'] = '3'  # Only FATAL to stderr

# Additional suppression for absl logging (used by TensorFlow)
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_CPP_VMODULE'] = 'inference_feedback_manager=0,gl_context=0'

@contextlib.contextmanager
def suppress_stderr_fd():
    """Suppress stderr at file descriptor level to block C++ logs."""
    stderr_fd = sys.stderr.fileno()
    with open(os.devnull, 'w') as devnull:
        old_stderr = os.dup(stderr_fd)
        os.dup2(devnull.fileno(), stderr_fd)
        try:
            yield
        finally:
            os.dup2(old_stderr, stderr_fd)
            os.close(old_stderr)

import string
import pandas as pd
from typing import Dict, List
from pathlib import Path

# Import MediaPipe-dependent modules with stderr suppression
with suppress_stderr_fd():
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
    print(f"==> condition_name: {condition_path.name}")
    condition_name = condition_path.name
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

                # Validate before accessing frames
                if len(joint_angles.frames) == 0:
                    print(f"WARNING: No joint angle frames computed (keypoints: {len(keypoints_array)})")
                    continue
                
                if len(joint_angles.frames[0].angles) == 0:
                    print(f"WARNING: First frame has no joint angles (frames: {len(joint_angles.frames)})")
                    continue

                print(f"Average joint angles for {len(joint_angles.frames)} frames:")
                for joint_name in joint_angles.frames[0].angles.keys():
                    avg_joint_angle = joint_angles.get_statistics(joint_name=joint_name)["mean"]
                    print(f"\t<joint angle> of {joint_name}: {avg_joint_angle}")

            
    except Exception as e:
        logger.error(f"Cannot process: {condition_path}")