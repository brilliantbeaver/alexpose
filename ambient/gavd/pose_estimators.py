"""
Pluggable pose estimator implementations for extracting human keypoints.

This module provides an abstract `PoseEstimator` interface and a concrete
`OpenPoseEstimator` that shells out to CMU OpenPose via the binary located
under the directory specified by the `OPENPOSEPATH` environment variable.

Design goals:
- SOLID: clear abstractions; `PoseEstimator` defines the contract.
- DRY: shared parsing utilities; minimal duplication.
- Extensibility: additional estimators (e.g., MediaPipe, AlphaPose, YOLO)
  can implement the same interface and be dropped in without changing callers.
"""

from __future__ import annotations

import json
import os
import platform
import shutil
import subprocess
import tempfile
from abc import ABC, abstractmethod
from glob import glob
from pathlib import Path
from typing import Dict, List, Optional, Union
import re


Keypoint = Dict[str, Union[float, int]]


class PoseEstimator(ABC):
    """Abstract interface for pose estimation backends."""

    @abstractmethod
    def estimate_image_keypoints(
        self,
        image_path: str,
        model: str = "BODY_25",
        bbox: Optional[Dict[str, Union[int, float]]] = None,
    ) -> List[Keypoint]:
        """
        Estimate 2D keypoints from an image.

        Args:
            image_path: Path to the input image frame.
            model: Pose model to use. Supported: "BODY_25", "COCO".
            bbox: Optional bounding box; implementations may crop or filter.

        Returns:
            A list of keypoints as dicts with keys: x, y, confidence.
        """
        raise NotImplementedError

    @abstractmethod
    def estimate_video_keypoints(
        self,
        video_path: Union[str, Path],
        model: str = "BODY_25",
    ) -> List[List[Keypoint]]:
        """Estimate keypoints for all frames of a video.

        Args:
            video_path: Path to the input video file.
            model: Pose model to use. Supported: "BODY_25", "COCO".

        Returns:
            A list where index i corresponds to frame index i. Missing frames
            are represented by empty lists, so indexing by frame number is stable.
        """
        raise NotImplementedError

    def supports_video_batch(self) -> bool:
        """Whether this estimator supports video-batch processing (default True for OpenPose)."""
        return hasattr(self, "estimate_video_keypoints")


class OpenPoseEstimator(PoseEstimator):
    """
    Pose estimator that delegates to CMU OpenPose binary.

    It discovers the OpenPose binary from `OPENPOSEPATH` env var and invokes
    `OpenPoseDemo` to process the provided image. The resulting JSON is parsed
    into a normalized keypoint list. If multiple people are detected, the first
    person is used by default.
    """

    def __init__(
        self,
        openpose_root: Optional[Union[str, Path]] = None,
        default_model: str = "BODY_25",
        additional_args: Optional[List[str]] = None,
    ) -> None:
        self.openpose_root = Path(
            openpose_root or os.getenv("OPENPOSEPATH", "")
        ).expanduser()
        if not self.openpose_root:
            raise EnvironmentError(
                "OPENPOSEPATH is not set and openpose_root was not provided"
            )
        self.default_model = default_model
        # Default to a fast, GPU-friendly configuration if not provided
        default_args = [
            "--number_people_max",
            "1",
            "--scale_number",
            "1",
            "--net_resolution",
            "-1x368",
        ]
        self.additional_args = list(additional_args) if additional_args else default_args

        self._binary_path = self._resolve_openpose_demo()
        self._models_dir = self._resolve_models_dir()

    def _resolve_models_dir(self) -> Path:
        """Resolve and validate the OpenPose models directory.

        Raises a clear error if models are not present, with guidance to download.
        """
        models_dir = self.openpose_root / "models"
        if not models_dir.exists():
            raise FileNotFoundError(
                "OpenPose models directory not found. Expected at: "
                f"{models_dir}. You must download models by running the official "
                "script (Windows: scripts\\getModels.bat) or placing the 'models' "
                "folder in the OpenPose root."
            )
        # Quick sanity check for BODY_25
        body25_prototxt = models_dir / "pose" / "body_25" / "pose_deploy.prototxt"
        if not body25_prototxt.exists():
            raise FileNotFoundError(
                "OpenPose BODY_25 prototxt not found. Expected at: "
                f"{body25_prototxt}. Ensure models were fully downloaded."
            )
        return models_dir

    def _resolve_openpose_demo(self) -> Path:
        """Resolve the OpenPose demo binary path across platforms."""
        bin_dir = self.openpose_root / "bin"
        candidates = [
            bin_dir / "OpenPoseDemo.exe",  # Windows
            bin_dir / "OpenPoseDemo",  # Linux/macOS build
        ]
        for candidate in candidates:
            if candidate.exists():
                return candidate
        raise FileNotFoundError(
            f"OpenPose demo binary not found under {bin_dir}. "
            "Ensure OpenPose is installed and OPENPOSEPATH points to its root."
        )

    def estimate_image_keypoints(
        self,
        image_path: str,
        model: str = "BODY_25",
        bbox: Optional[Dict[str, Union[int, float]]] = None,
    ) -> List[Keypoint]:
        model = (model or self.default_model).upper()
        if model not in {"BODY_25", "COCO"}:
            raise ValueError("model must be 'BODY_25' or 'COCO'")

        # Prepare temp dirs for OpenPose I/O
        with tempfile.TemporaryDirectory(prefix="openpose_input_") as tmp_in, \
            tempfile.TemporaryDirectory(prefix="openpose_json_") as tmp_out:
            tmp_in_path = Path(tmp_in)
            tmp_out_path = Path(tmp_out)

            # Copy image into temp input dir (OpenPose expects --image_dir)
            src = Path(image_path)
            if not src.exists():
                raise FileNotFoundError(f"Image not found: {src}")
            dst = tmp_in_path / src.name
            shutil.copy2(src, dst)

            # Build command
            cmd = [
                str(self._binary_path),
                "--image_dir", str(tmp_in_path),
                "--write_json", str(tmp_out_path),
                "--display", "0",
                "--render_pose", "0",
                "--model_pose", model,
                "--model_folder", str(self._models_dir),
            ]
            if self.additional_args:
                cmd.extend(self.additional_args)

            # Execute OpenPose
            completed = subprocess.run(
                cmd,
                cwd=str(self.openpose_root),
                capture_output=True,
                text=True,
                check=False,
            )
            if completed.returncode != 0:
                raise RuntimeError(
                    f"OpenPose failed (exit {completed.returncode}):\n"
                    f"STDOUT: {completed.stdout}\nSTDERR: {completed.stderr}"
                )

            # Parse first JSON file generated
            json_files = sorted(glob(str(tmp_out_path / "*.json")))
            if not json_files:
                # No detections or output not generated
                return []

            with open(json_files[0], "r", encoding="utf-8") as f:
                data = json.load(f)

            return _parse_openpose_people(data)

    def estimate_video_keypoints(
        self,
        video_path: Union[str, Path],
        model: str = "BODY_25",
    ) -> List[List[Keypoint]]:
        """Estimate keypoints for all frames of a video via OpenPose.

        Returns a list where index i contains the list of keypoints for frame i
        (first person if multiple). Empty list for frames with no detections.
        """
        src = Path(video_path)
        if not src.exists():
            raise FileNotFoundError(f"Video not found: {src}")

        model = (model or self.default_model).upper()
        if model not in {"BODY_25", "COCO"}:
            raise ValueError("model must be 'BODY_25' or 'COCO'")

        with tempfile.TemporaryDirectory(prefix="openpose_video_json_") as tmp_out:
            tmp_out_path = Path(tmp_out)

            cmd = [
                str(self._binary_path),
                "--video", str(src),
                "--write_json", str(tmp_out_path),
                "--display", "0",
                "--render_pose", "0",
                "--model_pose", model,
                "--model_folder", str(self._models_dir),
            ]
            if self.additional_args:
                cmd.extend(self.additional_args)

            completed = subprocess.run(
                cmd,
                cwd=str(self.openpose_root),
                capture_output=True,
                text=True,
                check=False,
            )
            if completed.returncode != 0:
                raise RuntimeError(
                    f"OpenPose failed (exit {completed.returncode}):\n"
                    f"STDOUT: {completed.stdout}\nSTDERR: {completed.stderr}"
                )

            # Collect per-frame JSONs; filenames are zero-padded indices in practice
            json_files = sorted(glob(str(tmp_out_path / "*_keypoints.json")))
            # Build a mapping of explicit indices parsed from filenames
            idx_to_kps: Dict[int, List[Keypoint]] = {}
            max_idx = -1
            for jf in json_files:
                try:
                    with open(jf, "r", encoding="utf-8") as f:
                        data = json.load(f)
                    # Expect filenames like anyprefix_000000000123_keypoints.json
                    name = Path(jf).name
                    m = re.search(r"_(\d+)_keypoints\.json$", name)
                    idx = int(m.group(1)) if m else (max_idx + 1)
                    max_idx = max(max_idx, idx)
                    idx_to_kps[idx] = _parse_openpose_people(data)
                except Exception:
                    continue
            # Expand to dense list with empty lists for missing frames
            dense: List[List[Keypoint]] = []
            for i in range(max_idx + 1):
                dense.append(idx_to_kps.get(i, []))
            return dense


def _parse_openpose_people(payload: Dict[str, any]) -> List[Keypoint]:
    """Convert OpenPose 'people[].pose_keypoints_2d' to list of dict keypoints.

    If multiple people are present, the first entry is used.
    """
    people = payload.get("people", []) or []
    if not people:
        return []

    arr = people[0].get("pose_keypoints_2d", []) or []
    keypoints: List[Keypoint] = []
    # OpenPose packs as x1, y1, c1, x2, y2, c2, ...
    for i in range(0, len(arr), 3):
        try:
            x = float(arr[i])
            y = float(arr[i + 1])
            c = float(arr[i + 2])
        except (IndexError, ValueError, TypeError):
            break
        keypoints.append({"x": x, "y": y, "confidence": c})

    return keypoints


__all__ = [
    "PoseEstimator",
    "OpenPoseEstimator",
]


