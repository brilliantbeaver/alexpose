"""Extract MediaPipe poses for the self-annotated augmentation-normal cohort.

Reuses notebook 02's exact extraction contract (crop -> MediaPipe VIDEO mode ->
full-frame-normalized (T,33,4) sequence -> np.savez_compressed with the identical
key set), but reads the *augmentation* GAVD CSVs written by annotate_normal_clips.py
and writes to a SEPARATE pose dir so the locked-96 canonical cohort is untouched.

The script loads ``gavd5-draft/.env`` (then the project-root ``.env``) without
overriding shell variables. It honours the same settings as notebooks 01-04:

``GAVD_MODE``
    Must be ``real``. This script never writes augmentation artifacts in smoke
    mode.
``GAVD_CACHE_DIR``
    Locates the MediaPipe model and runtime cache.
``GAVD_ARTIFACT_DIR``
    Locates the active artifact root. Outputs are written below
    ``<GAVD_ARTIFACT_DIR>/real``.

Output: <GAVD_ARTIFACT_DIR>/real/poses_augmented/normal/<sequence_id>.npz
        <GAVD_ARTIFACT_DIR>/real/augmented_pose_extraction_report.csv
"""
from __future__ import annotations

import ast
import hashlib
import os
from functools import lru_cache
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision
from dotenv import load_dotenv

TUTORIAL_DIR = Path(__file__).resolve().parent.parent
PROJECT_ROOT = TUTORIAL_DIR.parents[2]
load_dotenv(TUTORIAL_DIR / ".env", override=False)
load_dotenv(PROJECT_ROOT / ".env", override=False)

MODE = os.getenv("GAVD_MODE", "smoke").strip().lower()
if MODE not in {"smoke", "real"}:
    raise ValueError("GAVD_MODE must be smoke or real")

CACHE_DIR = Path(
    os.getenv("GAVD_CACHE_DIR", TUTORIAL_DIR / "work" / "cache")
).expanduser()
ARTIFACT_ROOT = Path(
    os.getenv("GAVD_ARTIFACT_DIR", TUTORIAL_DIR / "work" / "artifacts")
).expanduser()
ARTIFACT_DIR = ARTIFACT_ROOT / MODE
POSE_MODEL_PATH = CACHE_DIR / "models" / "pose_landmarker_lite.task"

AUG_CSV_DIR = TUTORIAL_DIR / "data-augmented" / "gavd" / "normal"
AUG_VIDEO_DIR = TUTORIAL_DIR / "data-augmented" / "youtube" / "normal"
OUT_POSE_DIR = ARTIFACT_DIR / "poses_augmented" / "normal"
REPORT_CSV = ARTIFACT_DIR / "augmented_pose_extraction_report.csv"

# match nb02 exactly
EXTRACTION_VERSION = "gavd3_pose_v2_video_mode"
VIS_THRESHOLD = 0.45
MIN_NEURO_OBSERVED = 0.45
MASK_KEYPOINTS = [11, 12, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32]


def safe_literal(value):
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return {}
    try:
        parsed = ast.literal_eval(str(value))
        return parsed if isinstance(parsed, dict) else {}
    except (ValueError, SyntaxError):
        return {}


@lru_cache(maxsize=1)
def pose_model_sha256():
    return hashlib.sha256(POSE_MODEL_PATH.read_bytes()).hexdigest()


def make_landmarker():
    options = vision.PoseLandmarkerOptions(
        base_options=mp_python.BaseOptions(
            model_asset_path=str(POSE_MODEL_PATH),
            delegate=mp_python.BaseOptions.Delegate.CPU,
        ),
        running_mode=vision.RunningMode.VIDEO,
        num_poses=1,
        min_pose_detection_confidence=VIS_THRESHOLD,
        min_pose_presence_confidence=VIS_THRESHOLD,
        min_tracking_confidence=VIS_THRESHOLD,
    )
    return vision.PoseLandmarker.create_from_options(options)


def scaled_crop_bounds(annotation_row, frame_shape, padding=0.15):
    """Verbatim nb02 logic: scale bbox by vid_info, pad, clamp."""
    frame_height, frame_width = frame_shape[:2]
    bbox = safe_literal(annotation_row.get("bbox"))
    source = safe_literal(annotation_row.get("vid_info"))
    source_width = float(source.get("width") or frame_width)
    source_height = float(source.get("height") or frame_height)
    scale_x = frame_width / source_width
    scale_y = frame_height / source_height
    left = float(bbox.get("left", 0.0)) * scale_x
    top = float(bbox.get("top", 0.0)) * scale_y
    width = float(bbox.get("width", source_width)) * scale_x
    height = float(bbox.get("height", source_height)) * scale_y
    left -= padding * width
    top -= padding * height
    width *= 1.0 + 2.0 * padding
    height *= 1.0 + 2.0 * padding
    x0 = max(0, int(np.floor(left)))
    y0 = max(0, int(np.floor(top)))
    x1 = min(frame_width, int(np.ceil(left + width)))
    y1 = min(frame_height, int(np.ceil(top + height)))
    if x1 <= x0 or y1 <= y0:
        return 0, 0, frame_width, frame_height
    return x0, y0, x1, y1


def detect_pose_in_crop(frame_bgr, annotation_row, landmarker, timestamp_ms):
    """Verbatim nb02: crop, detect, write full-frame-normalized (33,4) row."""
    height, width = frame_bgr.shape[:2]
    x0, y0, x1, y1 = scaled_crop_bounds(annotation_row, frame_bgr.shape)
    crop = frame_bgr[y0:y1, x0:x1]
    rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
    result = landmarker.detect_for_video(
        mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb), timestamp_ms
    )
    sequence_row = np.full((33, 4), np.nan, dtype=np.float32)
    sequence_row[:, 3] = 0.0
    if not result.pose_landmarks:
        return sequence_row, (x0, y0, x1, y1)
    crop_height, crop_width = crop.shape[:2]
    for index, landmark in enumerate(result.pose_landmarks[0]):
        sequence_row[index, 0] = (x0 + landmark.x * crop_width) / width
        sequence_row[index, 1] = (y0 + landmark.y * crop_height) / height
        sequence_row[index, 2] = landmark.z * crop_width / width
        sequence_row[index, 3] = float(landmark.visibility or 0.0)
    return sequence_row, (x0, y0, x1, y1)


def find_video(clip_id):
    for suf in (".mp4", ".mkv", ".webm", ".mov", ".m4v"):
        p = AUG_VIDEO_DIR / f"{clip_id}{suf}"
        if p.exists():
            return p
    raise FileNotFoundError(f"No augmentation video for {clip_id}")


def preflight() -> list[Path]:
    """Validate all inputs before creating a report or a single pose archive."""
    if MODE != "real":
        raise RuntimeError(
            "Augmented-pose extraction is real-data only. Set GAVD_MODE=real and restart."
        )
    if not POSE_MODEL_PATH.is_file():
        raise FileNotFoundError(
            f"Pose model not found at {POSE_MODEL_PATH}. Run notebook 02 with the same "
            "GAVD_CACHE_DIR first."
        )
    csvs = sorted(AUG_CSV_DIR.glob("*.csv"))
    if not csvs:
        raise FileNotFoundError(
            f"No augmentation CSVs found in {AUG_CSV_DIR}. "
            "Run notes/annotate_normal_clips.py first."
        )

    missing_clips = []
    for csv_path in csvs:
        first = pd.read_csv(csv_path, nrows=1)
        if first.empty or "id" not in first.columns:
            raise ValueError(f"Augmentation CSV {csv_path} has no usable id column")
        clip_id = str(first.iloc[0]["id"])
        try:
            find_video(clip_id)
        except FileNotFoundError:
            missing_clips.append(clip_id)
    if missing_clips:
        unique_missing = sorted(set(missing_clips))
        preview = ", ".join(unique_missing[:8])
        suffix = " ..." if len(unique_missing) > 8 else ""
        raise FileNotFoundError(
            f"Missing {len(unique_missing)} augmentation video clips in {AUG_VIDEO_DIR}: "
            f"{preview}{suffix}. No artifacts were written. Restore the matching clips or "
            "regenerate the CSV/video pair with notes/annotate_normal_clips.py."
        )
    return csvs


def extract_one(csv_path: Path):
    annotations = pd.read_csv(csv_path).sort_values("frame_num")
    annotations["frame_num"] = annotations["frame_num"].astype(int)
    by_frame = {int(r.frame_num): r for _, r in annotations.iterrows()}
    first = annotations.iloc[0]
    sequence_id = str(first["seq"])
    clip_id = str(first["id"])
    youtube_video_id = str(first.get("youtube_video_id", clip_id))
    video_path = find_video(clip_id)

    cap = cv2.VideoCapture(str(video_path))
    fps = float(cap.get(cv2.CAP_PROP_FPS))
    if not cap.isOpened() or fps <= 0:
        cap.release()
        raise RuntimeError(f"Could not decode {video_path}")
    first_frame = int(annotations["frame_num"].min())
    last_frame = int(annotations["frame_num"].max())
    cap.set(cv2.CAP_PROP_POS_FRAMES, first_frame - 1)
    pose_rows, frame_numbers, crop_bounds = [], [], []
    with make_landmarker() as landmarker:
        for frame_number in range(first_frame, last_frame + 1):
            ok, frame = cap.read()
            if not ok:
                break
            annotation = by_frame.get(frame_number)
            if annotation is None:
                continue
            ts = int(round((frame_number - first_frame) * 1000.0 / fps))
            row, bounds = detect_pose_in_crop(frame, annotation, landmarker, ts)
            pose_rows.append(row)
            frame_numbers.append(frame_number)
            crop_bounds.append(bounds)
    cap.release()
    sequence = np.stack(pose_rows)

    OUT_POSE_DIR.mkdir(parents=True, exist_ok=True)
    out = OUT_POSE_DIR / f"{sequence_id}.npz"
    np.savez_compressed(
        out,
        sequence=sequence,
        frame_numbers=np.asarray(frame_numbers, dtype=np.int32),
        crop_bounds=np.asarray(crop_bounds, dtype=np.int32),
        fps=np.asarray(fps, dtype=np.float32),
        sequence_id=np.asarray(sequence_id),
        video_id=np.asarray(youtube_video_id),          # grouping/leakage key
        condition=np.asarray("normal"),
        source_csv=np.asarray(str(csv_path)),
        source_video=np.asarray(str(video_path)),
        extraction_version=np.asarray(EXTRACTION_VERSION),
        pose_model=np.asarray(POSE_MODEL_PATH.name),
        pose_model_sha256=np.asarray(pose_model_sha256()),
        visibility_threshold=np.asarray(VIS_THRESHOLD, dtype=np.float32),
        # augmentation-cohort provenance (NOT in canonical npz; extra keys are safe)
        cohort=np.asarray("augmented_normal"),
        bbox_source=np.asarray("mediapipe_pose_auto"),
        clip_id=np.asarray(clip_id),
    )
    coverage = float((sequence[..., 3] > 0.0).any(axis=1).mean())
    neuro = float((sequence[:, MASK_KEYPOINTS, 3] >= VIS_THRESHOLD).mean())
    return dict(sequence_id=sequence_id, clip_id=clip_id,
                youtube_video_id=youtube_video_id, frames=len(sequence),
                pose_frame_coverage=coverage, neuro_observed=neuro, fps=fps,
                path=str(out))


def main():
    csvs = preflight()
    print(f"mode -> {MODE}")
    print(f"cache -> {CACHE_DIR}")
    print(f"artifacts -> {ARTIFACT_DIR}")
    print(f"Extracting {len(csvs)} augmentation sequences...")
    rows = []
    for i, c in enumerate(csvs, 1):
        try:
            r = extract_one(c)
            accepted = r["neuro_observed"] >= MIN_NEURO_OBSERVED
            status = "accepted" if accepted else "rejected_low_neurologic_coverage"
            rows.append({
                **r,
                "status": status,
                "accepted": accepted,
                "minimum_neuro_observed": MIN_NEURO_OBSERVED,
                "selection_reason": (
                    "meets_neurologic_coverage_threshold"
                    if accepted
                    else "neurologic_coverage_below_threshold"
                ),
            })
            print(f"  [{i:2d}/{len(csvs)}] {r['sequence_id']:34s} "
                  f"frames={r['frames']:3d} cov={r['pose_frame_coverage']:.2f} "
                  f"neuro={r['neuro_observed']:.2f} status={status}")
        except Exception as e:
            rows.append({"sequence_id": c.stem, "status": f"error: {e}"})
            print(f"  [{i:2d}/{len(csvs)}] {c.stem} ERROR: {e}")
    rep = pd.DataFrame(rows)
    rep.to_csv(REPORT_CSV, index=False)
    ok = rep[rep.status == "accepted"]
    print(f"\n==== DONE: {len(ok)}/{len(csvs)} accepted across "
          f"{ok['youtube_video_id'].nunique() if len(ok) else 0} videos ====")
    print(f"poses -> {OUT_POSE_DIR}")
    print(f"report -> {REPORT_CSV}")
    if len(ok):
        print(f"mean pose coverage: {ok.pose_frame_coverage.mean():.2f}, "
              f"mean neuro observed: {ok.neuro_observed.mean():.2f}")


if __name__ == "__main__":
    main()
