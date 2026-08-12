"""Data handling for the S-JEPA gait tutorials.

This module turns raw walking videos into skeleton sequences and then into the
fixed length windows the model trains on. It reuses the existing ``ambient`` pose
stack for the heavy lifting, so we never re-implement MediaPipe.

The flow is:

    mp4 file
      -> cv2 frame loop, sampled to target_fps
      -> ambient SequenceKeypointExtractor.extract_from_image  (33 landmarks)
      -> array of shape (T, 33, 3) with columns [x_pixels, y_pixels, visibility]
      -> normalise (root centre on the pelvis, scale by torso length)
      -> cache to a .npz file, one per video

There are no GAVD CSVs here. The GAVD pipeline needs per-frame bounding boxes and
YouTube URLs; our videos are already downloaded clips, so we just walk the folders
and run pose on whole frames.

Splitting is done by *source video id* (the YouTube id in manifest.csv), never by
clip, so clips cut from the same source never land on both sides of a split.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np


# Pelvis and shoulder landmark indices used for normalisation.
_LEFT_HIP, _RIGHT_HIP = 23, 24
_LEFT_SHOULDER, _RIGHT_SHOULDER = 11, 12

_CLIP_SUFFIX = re.compile(r"_clip-\d+$", re.IGNORECASE)


# ---------------------------------------------------------------------------
# Pose extraction from raw video
# ---------------------------------------------------------------------------

def load_video_sequence(
    video_path: str | Path,
    target_fps: int = 15,
    max_frames: Optional[int] = None,
    extractor=None,
    verbose: bool = False,
) -> np.ndarray:
    """Run MediaPipe over a video and return a (T, 33, 3) array.

    Columns are ``[x_pixels, y_pixels, visibility]``. Frames with no detection
    become rows of NaN so :func:`clean_sequence` can interpolate them later.

    ``extractor`` may be a pre-built ``SequenceKeypointExtractor`` (reuse one
    across many videos to keep the MediaPipe singleton warm). If ``None`` we make
    one on first use.
    """
    import cv2

    if extractor is None:
        from ambient.pose.keypoint_extractor import SequenceKeypointExtractor

        extractor = SequenceKeypointExtractor()

    video_path = Path(video_path)
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise IOError(f"Could not open video: {video_path}")

    src_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    stride = max(1, int(round(src_fps / float(target_fps))))

    rows: List[np.ndarray] = []
    frame_idx = 0
    kept = 0
    try:
        while True:
            ok, frame_bgr = cap.read()
            if not ok:
                break
            if frame_idx % stride == 0:
                frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
                kps = extractor.extract_from_image(frame_rgb)
                rows.append(_keypointset_to_row(kps))
                kept += 1
                if max_frames is not None and kept >= max_frames:
                    break
            frame_idx += 1
    finally:
        cap.release()

    if not rows:
        return np.empty((0, 33, 3), dtype=np.float32)

    seq = np.stack(rows, axis=0).astype(np.float32)
    if verbose:
        valid = np.isfinite(seq[:, :, 0]).all(axis=1).mean() if len(seq) else 0.0
        print(f"  {video_path.name}: {len(seq)} frames @ ~{target_fps}fps, "
              f"{valid:.0%} fully detected")
    return seq


def _keypointset_to_row(kps) -> np.ndarray:
    """Convert an ambient KeypointSet (33 keypoints) into one (33, 3) row.

    An empty detection yields a row of NaNs. Keypoints carry pixel x, pixel y and
    a confidence value that is MediaPipe's visibility.
    """
    row = np.full((33, 3), np.nan, dtype=np.float32)
    kp_list = getattr(kps, "keypoints", None) or []
    if len(kp_list) < 33:
        # Empty or partial detection: leave as NaN, cleaning handles it.
        if len(kp_list) == 0:
            return row
    for i, kp in enumerate(kp_list[:33]):
        row[i, 0] = float(getattr(kp, "x", np.nan))
        row[i, 1] = float(getattr(kp, "y", np.nan))
        conf = getattr(kp, "confidence", None)
        row[i, 2] = float(conf) if conf is not None else np.nan
    return row


# ---------------------------------------------------------------------------
# Cleaning and normalisation
# ---------------------------------------------------------------------------

def clean_sequence(seq: np.ndarray, min_valid_fraction: float = 0.3) -> Optional[np.ndarray]:
    """Interpolate short gaps and trim all-NaN ends.

    Returns the cleaned (T, 33, 3) array, or ``None`` if the video is too poor to
    use (fewer than ``min_valid_fraction`` fully detected frames).
    """
    if seq.shape[0] == 0:
        return None

    fully_detected = np.isfinite(seq[:, :, 0]).all(axis=1)
    if fully_detected.mean() < min_valid_fraction:
        return None

    # Trim leading and trailing frames that have no detection at all.
    any_detected = np.isfinite(seq[:, :, 0]).any(axis=1)
    if not any_detected.any():
        return None
    first, last = np.argmax(any_detected), len(any_detected) - np.argmax(any_detected[::-1])
    seq = seq[first:last].copy()

    # Per joint, per channel linear interpolation over time for the gaps.
    T = seq.shape[0]
    t_idx = np.arange(T)
    for v in range(33):
        for c in range(3):
            col = seq[:, v, c]
            good = np.isfinite(col)
            if good.sum() == 0:
                seq[:, v, c] = 0.0
            elif good.sum() < T:
                seq[:, v, c] = np.interp(t_idx, t_idx[good], col[good])
    return seq


def normalize_sequence(seq: np.ndarray) -> np.ndarray:
    """Root-centre on the pelvis and scale by torso length, per frame.

    The visibility channel is left untouched. This removes differences in where
    the walker is in the frame and how close the camera is, so the model sees
    shape and motion rather than absolute pixel position.
    """
    out = seq.copy().astype(np.float32)
    xy = out[:, :, :2]

    pelvis = (xy[:, _LEFT_HIP] + xy[:, _RIGHT_HIP]) / 2.0          # (T, 2)
    shoulder = (xy[:, _LEFT_SHOULDER] + xy[:, _RIGHT_SHOULDER]) / 2.0
    torso = np.linalg.norm(shoulder - pelvis, axis=1, keepdims=True)  # (T, 1)
    scale = np.clip(torso, 1e-3, None)                               # avoid /0

    xy = xy - pelvis[:, None, :]
    xy = xy / scale[:, None, :]
    out[:, :, :2] = xy
    return out


# ---------------------------------------------------------------------------
# Cache IO
# ---------------------------------------------------------------------------

def source_id_from_name(filename: str) -> str:
    """Strip a ``_clip-NN`` suffix to recover the shared source id."""
    stem = Path(filename).stem
    return _CLIP_SUFFIX.sub("", stem)


def save_sequence_npz(
    path: str | Path,
    keypoints_raw: np.ndarray,
    keypoints_norm: np.ndarray,
    fps: int,
    source_id: str,
    label: str,
    clip_name: str,
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        keypoints=keypoints_raw.astype(np.float32),
        keypoints_norm=keypoints_norm.astype(np.float32),
        fps=np.int32(fps),
        source_id=source_id,
        label=label,
        clip_name=clip_name,
    )


@dataclass
class SequenceRecord:
    """One cached video: metadata plus lazy access to the arrays."""

    path: Path
    label: str
    source_id: str
    clip_name: str
    n_frames: int

    def load_norm(self) -> np.ndarray:
        with np.load(self.path, allow_pickle=True) as z:
            return z["keypoints_norm"].astype(np.float32)

    def load_raw(self) -> np.ndarray:
        with np.load(self.path, allow_pickle=True) as z:
            return z["keypoints"].astype(np.float32)


def load_index(keypoints_dir: str | Path) -> List[SequenceRecord]:
    """Scan a folder of cached .npz files and return their records."""
    keypoints_dir = Path(keypoints_dir)
    records: List[SequenceRecord] = []
    for npz_path in sorted(keypoints_dir.glob("*.npz")):
        with np.load(npz_path, allow_pickle=True) as z:
            records.append(
                SequenceRecord(
                    path=npz_path,
                    label=str(z["label"]),
                    source_id=str(z["source_id"]),
                    clip_name=str(z["clip_name"]),
                    n_frames=int(z["keypoints_norm"].shape[0]),
                )
            )
    return records


# ---------------------------------------------------------------------------
# Windowing
# ---------------------------------------------------------------------------

def sliding_windows(seq: np.ndarray, window: int, stride: int) -> np.ndarray:
    """Cut a (T, 33, 3) sequence into (num_windows, window, 33, 3).

    If the sequence is shorter than one window it is padded by repeating the last
    frame, so every usable video yields at least one window.
    """
    T = seq.shape[0]
    if T == 0:
        return np.empty((0, window, seq.shape[1], seq.shape[2]), dtype=np.float32)
    if T < window:
        pad = np.repeat(seq[-1:], window - T, axis=0)
        seq = np.concatenate([seq, pad], axis=0)
        T = window
    starts = list(range(0, T - window + 1, stride))
    if not starts:
        starts = [0]
    return np.stack([seq[s:s + window] for s in starts], axis=0).astype(np.float32)


class SequenceWindowDataset:
    """A tiny torch Dataset of windows, built from a list of SequenceRecords.

    Kept import-light: torch is imported lazily inside ``__init__`` so that the
    non-training notebooks can import this module without torch installed.
    """

    def __init__(
        self,
        records: Sequence[SequenceRecord],
        window: int,
        stride: int,
        label_to_int: Optional[Dict[str, int]] = None,
    ):
        import torch  # noqa: F401  (validate availability, used by callers)

        self.windows: List[np.ndarray] = []
        self.labels: List[int] = []
        self.source_ids: List[str] = []
        self.label_to_int = label_to_int or {"normal": 0, "ms": 1, "pd": 2}

        for rec in records:
            seq = rec.load_norm()
            wins = sliding_windows(seq, window, stride)
            for w in wins:
                self.windows.append(w)
                self.labels.append(self.label_to_int[rec.label])
                self.source_ids.append(rec.source_id)

    def __len__(self) -> int:
        return len(self.windows)

    def __getitem__(self, i: int):
        import torch

        x = torch.from_numpy(self.windows[i])          # (window, 33, 3)
        y = int(self.labels[i])
        return x, y


# ---------------------------------------------------------------------------
# Leakage-safe grouped splitting
# ---------------------------------------------------------------------------

def grouped_train_test_split(
    records: Sequence[SequenceRecord],
    test_size: float = 0.3,
    seed: int = 42,
) -> Tuple[List[SequenceRecord], List[SequenceRecord]]:
    """Split records so that all clips of one source id stay together.

    Uses scikit-learn's GroupShuffleSplit on the source id, stratifying loosely by
    keeping the class balance close through the shuffle seed.
    """
    from sklearn.model_selection import GroupShuffleSplit

    groups = np.array([r.source_id for r in records])
    idx = np.arange(len(records))
    gss = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=seed)
    train_i, test_i = next(gss.split(idx, groups=groups))
    train = [records[i] for i in train_i]
    test = [records[i] for i in test_i]
    _assert_disjoint(train, test)
    return train, test


def grouped_kfold(
    records: Sequence[SequenceRecord],
    n_splits: int = 5,
    seed: int = 42,
):
    """Yield (train_records, test_records) folds grouped by source id.

    Uses StratifiedGroupKFold so class balance is respected while groups (source
    videos) never cross a fold boundary. ``n_splits`` is capped to the smallest
    class's group count so tiny classes do not break the split.
    """
    from sklearn.model_selection import StratifiedGroupKFold

    labels = np.array([r.label for r in records])
    groups = np.array([r.source_id for r in records])
    idx = np.arange(len(records))

    # Cap folds by the rarest class's number of distinct groups.
    per_class_groups = {
        c: len(set(g for g, l in zip(groups, labels) if l == c)) for c in set(labels)
    }
    max_folds = max(2, min(per_class_groups.values()))
    k = min(n_splits, max_folds)

    sgkf = StratifiedGroupKFold(n_splits=k, shuffle=True, random_state=seed)
    for train_i, test_i in sgkf.split(idx, labels, groups=groups):
        train = [records[i] for i in train_i]
        test = [records[i] for i in test_i]
        _assert_disjoint(train, test)
        yield train, test


def _assert_disjoint(train: Sequence[SequenceRecord], test: Sequence[SequenceRecord]) -> None:
    tr = {r.source_id for r in train}
    te = {r.source_id for r in test}
    overlap = tr & te
    if overlap:
        raise AssertionError(f"Source id leakage across split: {sorted(overlap)}")
