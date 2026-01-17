"""
Ambient Utils Package

This package contains utility functions for our "ambient" fall risk detection system.

@Theodore Mui
Monday, July 28, 2025 12:30:00 AM
"""

from .csv_parser import parse_csv_with_dicts, parse_csv_with_pandas, parse_openpose_csv

# Evaluation and visualization utilities (moved from notebooks/utils)
from .eval_keypoints import (
    visualize_keypoints,
    extract_pose_from_sequence,
    pose_estimation_for_frames,
    pose_estimation_all_sequences,
    ensure_model_downloaded,
    calculate_angles,
)
from .viz import (
    visualize_frame,
    visualize_pose_with_skeleton,
    draw_bounding_box,
)

__all__ = [
    # CSV parsing
    "parse_csv_with_dicts",
    "parse_csv_with_pandas",
    "parse_openpose_csv",
    # Evaluation utilities
    "visualize_keypoints",
    "extract_pose_from_sequence",
    "pose_estimation_for_frames",
    "pose_estimation_all_sequences",
    "get_keypoints",
    "ensure_model_downloaded",
    "calculate_angles",
    # Visualization utilities
    "visualize_frame",
    "visualize_pose_with_skeleton",
    "draw_bounding_box",
]

__version__ = "1.0.0"
__author__ = "Alex Mui and Theodore Mui"
