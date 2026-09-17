"""Verify explicit full sources and original PTS before consuming supplied poses.

This milestone consumes complete-bout pose caches, not a hidden pose extractor.
RGB feature extraction is separately gated. No asset discovery is performed.
"""
from dataclasses import dataclass
import json
from pathlib import Path
import subprocess
import numpy as np
from .contracts import sha256_file


@dataclass
class TimedPose:
    times: np.ndarray
    coords: np.ndarray
    observed: np.ndarray
    confidence: np.ndarray
    frame_indices: np.ndarray
    width: int
    height: int

    def validate(self):
        n = len(self.times)
        if self.times.ndim != 1 or n < 1 or not np.isfinite(self.times).all() or np.any(np.diff(self.times) <= 0):
            raise ValueError("Missing, repeated or nonmonotone PTS; no silent repair")
        if self.coords.shape != (n, 33, 2) or self.observed.shape != (n, 33) or self.confidence.shape != (n, 33):
            raise ValueError("Pose shape must be [frames,33,2] with observation and confidence")
        if self.observed.dtype != np.bool_ or not np.issubdtype(self.frame_indices.dtype, np.integer) or self.frame_indices.shape != (n,) or np.any(np.diff(self.frame_indices) <= 0):
            raise ValueError("Require explicit bool observation and increasing original frame IDs")
        if not np.isfinite(self.coords[self.observed]).all() or not np.isfinite(self.confidence).all() or np.any((self.confidence < 0) | (self.confidence > 1)):
            raise ValueError("Observed coordinates/confidence invalid")
        if self.width <= 0 or self.height <= 0:
            raise ValueError("Explicit full-frame aspect ratio required")
        return self


def probe_pts(video, ffprobe="ffprobe"):
    path = Path(video["video_path"])
    if sha256_file(path) != video["sha256"]:
        raise ValueError("Full-video SHA256 changed")
    proc = subprocess.run([ffprobe, "-v", "error", "-select_streams", "v:0", "-show_frames", "-show_streams", "-show_entries", "frame=best_effort_timestamp_time:stream=width,height", "-of", "json", str(path)], capture_output=True, text=True, check=True)
    raw = json.loads(proc.stdout)
    frames = raw.get("frames", [])
    if not frames or any("best_effort_timestamp_time" not in f for f in frames):
        raise ValueError("Video lacks complete presentation timestamps")
    pts = np.array([float(f["best_effort_timestamp_time"]) for f in frames])
    if not np.isfinite(pts).all() or np.any(np.diff(pts) <= 0):
        raise ValueError("Video PTS invalid/nonmonotone; declare contiguous bouts upstream")
    stream = raw["streams"][0]
    return pts, int(stream["width"]), int(stream["height"])


def load_pose(entry, bout, *, video_timing, visibility_threshold):
    path = Path(entry["pose_path"])
    if sha256_file(path) != entry["sha256"]:
        raise ValueError("Pose cache SHA256 changed")
    with np.load(path, allow_pickle=False) as z:
        needed = {"times", "coords", "observed", "confidence", "frame_indices", "width", "height"}
        if not needed <= set(z.files):
            raise ValueError(f"Pose cache requires {sorted(needed)}")
        pose = TimedPose(z["times"], z["coords"], z["observed"], z["confidence"], z["frame_indices"], int(z["width"]), int(z["height"])).validate()
    pts, width, height = video_timing
    if (width, height) != (pose.width, pose.height) or pose.frame_indices.min() < 0 or pose.frame_indices.max() >= len(pts):
        raise ValueError("Pose frame index/aspect ratio differs from full video")
    if not np.allclose(pts[pose.frame_indices], pose.times, atol=1e-6, rtol=0):
        raise ValueError("Pose timestamp does not match original video presentation timestamp")
    expected = np.flatnonzero((pts >= bout["start_pts"]) & (pts < bout["end_pts_exclusive"]))
    if not np.array_equal(expected, pose.frame_indices):
        raise ValueError("Pose cache must retain every source frame in the COMPLETE bout, even failed detections")
    pose.observed = pose.observed & (pose.confidence >= visibility_threshold)
    return pose
