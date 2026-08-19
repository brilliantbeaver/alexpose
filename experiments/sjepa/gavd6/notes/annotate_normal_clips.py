"""Self-annotate the data-videos/normal clips into GAVD-schema per-sequence CSVs.

Per user instruction: for each clip in ``data-videos/normal/``, run MediaPipe Pose
on the full frame, select the single moving subject, derive a per-frame bounding
box on that subject, split the clip into gait sequences, and write one GAVD CSV
per sequence so that notebook 02 can extract poses from it unchanged.

Design (grounded in ``02_extract_and_watch_skeletons.ipynb``):
  * CSV schema matches GAVD exactly:
        seq, frame_num, cam_view, gait_event, dataset, gait_pat, bbox, vid_info, id, url
  * ``bbox`` is a stringified dict {'top','left','height','width'} in *source*
    pixels, and ``vid_info`` carries this clip's real width/height. nb02's
    ``scaled_crop_bounds`` scales bbox by vid_info -> our scale factor is 1.0, so
    the crop is applied exactly as annotated.
  * ``frame_num`` is 1-based (nb02 seeks CAP_PROP_POS_FRAMES = frame_num - 1).
  * Provenance: these bboxes are auto-detected from MediaPipe Pose, NOT human GAVD
    annotations. Recorded via gait_pat/dataset tags and the sidecar provenance CSV.

Zero new dependencies: mediapipe, cv2, numpy, scipy only (all in pyproject).
"""
from __future__ import annotations

import glob
import os
import re
from pathlib import Path

import cv2
import numpy as np
from scipy.signal import find_peaks

import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision

# ---------------------------------------------------------------------------
# Paths / constants (aligned with notebook 02)
# ---------------------------------------------------------------------------
TUTORIAL_DIR = Path(__file__).resolve().parent.parent          # .../experiments/sjepa/gavd6
CLIP_DIR = TUTORIAL_DIR / "data-videos" / "normal"
POSE_MODEL_PATH = TUTORIAL_DIR / "cache" / "models" / "pose_landmarker_lite.task"

# Where we write the synthesized GAVD CSVs (a *separate* augmentation cohort;
# the locked-96 cohort under the external GAVD vault is left untouched).
OUT_CSV_DIR = TUTORIAL_DIR / "data-augmented" / "gavd" / "normal"
OUT_VIDEO_DIR = TUTORIAL_DIR / "data-augmented" / "youtube" / "normal"
PROVENANCE_CSV = TUTORIAL_DIR / "data-augmented" / "augmented_normal_provenance.csv"

VIS_THRESHOLD = 0.45          # matches nb02 min_*_confidence / visibility_threshold
BBOX_PAD = 0.15               # matches nb02 scaled_crop_bounds padding
NUM_POSES = 3                 # detect up to 3 people, then select the subject
MASK_KEYPOINTS = [11, 12, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32]  # nb03 whitelist
ANKLES = [27, 28]
HIPS = [23, 24]

# Segmentation guardrails
MIN_SEQ_FRAMES = 24           # below this, 64-frame resize would be pure upsampling
CYCLES_PER_WINDOW = 2.0       # ~2 gait cycles per sequence window
WINDOW_OVERLAP = 0.5          # 50% overlap between windows
MIN_VALID_ANKLE_FRAC = 0.40   # need this fraction of frames with a visible ankle
MAX_WINDOWS_PER_CLIP = 8      # cap so one long clip can't dominate the cohort


# ---------------------------------------------------------------------------
# MediaPipe landmarker (IMAGE mode so we can score all candidate poses per frame)
# ---------------------------------------------------------------------------
def make_landmarker():
    if not POSE_MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Pose model not found at {POSE_MODEL_PATH}. Run notebook 02 first."
        )
    options = vision.PoseLandmarkerOptions(
        base_options=mp_python.BaseOptions(
            model_asset_path=str(POSE_MODEL_PATH),
            delegate=mp_python.BaseOptions.Delegate.CPU,
        ),
        running_mode=vision.RunningMode.VIDEO,
        num_poses=NUM_POSES,
        min_pose_detection_confidence=VIS_THRESHOLD,
        min_pose_presence_confidence=VIS_THRESHOLD,
        min_tracking_confidence=VIS_THRESHOLD,
    )
    return vision.PoseLandmarker.create_from_options(options)


def video_id_from_clip(path: Path) -> str:
    """Strip a trailing _clip-NN segment suffix to recover the YouTube id."""
    stem = path.stem
    return re.sub(r"_clip-\d+$", "", stem)


# ---------------------------------------------------------------------------
# Per-frame subject selection + landmark harvest
# ---------------------------------------------------------------------------
def _landmarks_xy_vis(pose_landmarks, w, h):
    """Return (33,2) pixel xy and (33,) visibility for one detected pose."""
    xy = np.full((33, 2), np.nan, dtype=np.float32)
    vis = np.zeros(33, dtype=np.float32)
    for i, lm in enumerate(pose_landmarks):
        xy[i, 0] = lm.x * w
        xy[i, 1] = lm.y * h
        vis[i] = float(lm.visibility or 0.0)
    return xy, vis


def _pose_score(xy, vis):
    """Score a candidate pose: bbox area (frac) * mean visibility of visible joints."""
    m = vis >= VIS_THRESHOLD
    if m.sum() < 4:
        return -1.0, None
    pts = xy[m]
    area = (pts[:, 0].max() - pts[:, 0].min()) * (pts[:, 1].max() - pts[:, 1].min())
    return area * float(vis[m].mean()), pts


def harvest_clip(clip_path: Path):
    """Run MediaPipe over the whole clip, selecting one subject per frame.

    Returns dict with per-frame xy (T,33,2), vis (T,33), selected-flag (T,),
    plus width/height/fps/frame_count.
    """
    cap = cv2.VideoCapture(str(clip_path))
    fps = float(cap.get(cv2.CAP_PROP_FPS)) or 30.0
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    xys, viss, ok_flags, prev_center = [], [], [], None
    idx = 0
    with make_landmarker() as lm:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            ts = int(round(idx * 1000.0 / fps))
            res = lm.detect_for_video(
                mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb), ts
            )
            best, best_xy, best_vis, best_center = -1.0, None, None, None
            for cand in (res.pose_landmarks or []):
                cxy, cvis = _landmarks_xy_vis(cand, w, h)
                score, pts = _pose_score(cxy, cvis)
                if score < 0:
                    continue
                center = np.array([pts[:, 0].mean(), pts[:, 1].mean()])
                # temporal continuity: prefer the subject nearest last frame's pick
                if prev_center is not None:
                    dist = np.linalg.norm(center - prev_center)
                    diag = float(np.hypot(w, h))
                    score = score * (1.0 + 0.5 * (1.0 - min(dist / diag, 1.0)))
                if score > best:
                    best, best_xy, best_vis, best_center = score, cxy, cvis, center
            if best_xy is None:
                xys.append(np.full((33, 2), np.nan, dtype=np.float32))
                viss.append(np.zeros(33, dtype=np.float32))
                ok_flags.append(False)
            else:
                xys.append(best_xy)
                viss.append(best_vis)
                ok_flags.append(True)
                prev_center = best_center
            idx += 1
    cap.release()
    return {
        "xy": np.stack(xys) if xys else np.zeros((0, 33, 2), np.float32),
        "vis": np.stack(viss) if viss else np.zeros((0, 33), np.float32),
        "ok": np.array(ok_flags, dtype=bool),
        "fps": fps,
        "w": w,
        "h": h,
        "n": idx,
    }


# ---------------------------------------------------------------------------
# Per-frame bbox from landmarks (padded, clamped) + temporal smoothing
# ---------------------------------------------------------------------------
def per_frame_bbox(xy, vis, w, h):
    """Return (T,4) [left, top, width, height] in pixels; NaN rows where no subject."""
    T = xy.shape[0]
    box = np.full((T, 4), np.nan, dtype=np.float32)
    for t in range(T):
        m = vis[t] >= VIS_THRESHOLD
        if m.sum() < 4:
            continue
        pts = xy[t][m]
        x0, y0 = pts[:, 0].min(), pts[:, 1].min()
        x1, y1 = pts[:, 0].max(), pts[:, 1].max()
        bw, bh = x1 - x0, y1 - y0
        x0 -= BBOX_PAD * bw
        y0 -= BBOX_PAD * bh
        bw *= 1 + 2 * BBOX_PAD
        bh *= 1 + 2 * BBOX_PAD
        box[t] = [x0, y0, bw, bh]
    return box


def smooth_and_fill(box, w, h, win=5):
    """Moving-median smoothing over valid rows + forward/back fill; clamp to frame."""
    T = box.shape[0]
    out = box.copy()
    # forward/backward fill NaN rows so cropping never fails
    for c in range(4):
        col = out[:, c]
        valid = np.where(~np.isnan(col))[0]
        if valid.size == 0:
            # no subject anywhere -> full frame
            out[:, 0], out[:, 1], out[:, 2], out[:, 3] = 0, 0, w, h
            return out
        col_filled = np.interp(np.arange(T), valid, col[valid])
        out[:, c] = col_filled
    # moving median
    half = win // 2
    med = out.copy()
    for t in range(T):
        lo, hi = max(0, t - half), min(T, t + half + 1)
        med[t] = np.median(out[lo:hi], axis=0)
    # clamp: left/top >= 0, box within frame
    med[:, 0] = np.clip(med[:, 0], 0, w - 1)
    med[:, 1] = np.clip(med[:, 1], 0, h - 1)
    med[:, 2] = np.clip(med[:, 2], 1, w - med[:, 0])
    med[:, 3] = np.clip(med[:, 3], 1, h - med[:, 1])
    return med


# ---------------------------------------------------------------------------
# Gait-cycle segmentation (scipy only) with whole-clip fallback
# ---------------------------------------------------------------------------
def estimate_stride_period(xy, vis, fps):
    """Dominant stride period (frames) via autocorrelation of ankle separation.

    Returns (period_frames, method) or (None, reason).
    """
    T = xy.shape[0]
    if T < MIN_SEQ_FRAMES:
        return None, "too_short"
    la, ra = xy[:, ANKLES[0], :], xy[:, ANKLES[1], :]
    va = (vis[:, ANKLES[0]] >= VIS_THRESHOLD) & (vis[:, ANKLES[1]] >= VIS_THRESHOLD)
    if va.mean() < MIN_VALID_ANKLE_FRAC:
        return None, "few_ankles"
    sep = np.linalg.norm(la - ra, axis=1)
    # interpolate missing, detrend, normalize
    idx = np.where(va)[0]
    sep = np.interp(np.arange(T), idx, sep[idx])
    sep = sep - sep.mean()
    if np.allclose(sep, 0):
        return None, "flat"
    ac = np.correlate(sep, sep, mode="full")[T - 1:]
    ac = ac / (ac[0] + 1e-9)
    # search physiologically plausible step period: 0.3s..1.5s -> stride ~ 2x step
    lo = max(int(0.3 * fps), 3)
    hi = min(int(1.6 * fps), T - 1)
    if hi <= lo:
        return None, "range"
    peaks, _ = find_peaks(ac[lo:hi], height=0.15)
    if peaks.size == 0:
        return None, "no_peak"
    step_period = lo + peaks[np.argmax(ac[lo:hi][peaks])]
    stride_period = 2 * step_period  # heel-strike to next ipsilateral heel-strike
    return float(stride_period), "autocorr_ankle_sep"


def segment_windows(T, period, fps):
    """Overlapping windows of ~CYCLES_PER_WINDOW strides. Whole-clip fallback."""
    if period is None:
        return [(0, T)]  # whole clip = one sequence
    win = int(round(CYCLES_PER_WINDOW * period))
    win = max(win, MIN_SEQ_FRAMES)
    if win >= T:
        return [(0, T)]
    step = max(int(round(win * (1 - WINDOW_OVERLAP))), 1)
    # widen the stride if the clip would exceed the per-clip window cap, so one
    # long clip contributes at most MAX_WINDOWS_PER_CLIP evenly-spaced sequences.
    approx = 1 + max(0, (T - win)) // step
    if approx > MAX_WINDOWS_PER_CLIP:
        step = max(int(np.ceil((T - win) / (MAX_WINDOWS_PER_CLIP - 1))), 1)
    windows = []
    start = 0
    while start < T:
        end = min(start + win, T)
        if end - start >= MIN_SEQ_FRAMES:
            windows.append((start, end))
        if end >= T or len(windows) >= MAX_WINDOWS_PER_CLIP:
            break
        start += step
    return windows or [(0, T)]


def infer_cam_view(xy, vis):
    """Rough camera-view label from mean shoulder->hip horizontal ordering."""
    # Use nose(0) vs hip-center x drift; fallback 'unknown'
    m = vis[:, HIPS].min(axis=1) >= VIS_THRESHOLD
    if m.sum() < 4:
        return "unknown"
    hip_c = xy[m][:, HIPS, 0].mean(axis=1)
    drift = hip_c[-1] - hip_c[0]
    if abs(drift) < 1e-3:
        return "front/back"
    return "left side" if drift < 0 else "right side"


# ---------------------------------------------------------------------------
# Main driver
# ---------------------------------------------------------------------------
def main():
    import pandas as pd

    OUT_CSV_DIR.mkdir(parents=True, exist_ok=True)
    OUT_VIDEO_DIR.mkdir(parents=True, exist_ok=True)
    PROVENANCE_CSV.parent.mkdir(parents=True, exist_ok=True)

    clips = sorted(CLIP_DIR.glob("*.mp4"))
    prov_rows = []
    seq_counter = {}

    for clip in clips:
        vid = video_id_from_clip(clip)
        print(f"\n=== {clip.name}  (video_id={vid}) ===")
        h = harvest_clip(clip)
        T, w, hh, fps = h["n"], h["w"], h["h"], h["fps"]
        ok_frac = float(h["ok"].mean()) if T else 0.0
        print(f"  {T} frames, {w}x{hh}, {fps:.1f} fps, subject detected in {ok_frac:.0%}")
        if T == 0 or ok_frac < 0.2:
            print("  SKIP: subject not reliably detected")
            prov_rows.append(dict(clip=clip.name, video_id=vid, status="skipped_low_detection",
                                  frames=T, subject_detect_frac=ok_frac))
            continue

        box = per_frame_bbox(h["xy"], h["vis"], w, hh)
        box = smooth_and_fill(box, w, hh)
        period, method = estimate_stride_period(h["xy"], h["vis"], fps)
        windows = segment_windows(T, period, fps)
        url = f"https://www.youtube.com/watch?v={vid}"
        cam = infer_cam_view(h["xy"], h["vis"])
        # Extraction id = the CLIP stem, so per-clip frame_num indexes the right file
        # even when several clips share one YouTube video_id. The true YouTube
        # video_id is kept separately for the leakage/grouping key.
        clip_id = clip.stem
        print(f"  stride_period={period} ({method}); {len(windows)} sequence window(s); cam={cam}")

        for (s, e) in windows:
            n = seq_counter.get(clip_id, 0)
            seq_counter[clip_id] = n + 1
            seq_id = f"aug-{clip_id}-w{n:02d}"
            rows = []
            for t in range(s, e):
                left, top, bw, bh = box[t]
                bbox = {"top": float(top), "left": float(left),
                        "height": float(bh), "width": float(bw)}
                vid_info = {"height": int(hh), "width": int(w), "mime_type": "video/mp4"}
                rows.append({
                    "seq": seq_id,
                    "frame_num": t + 1,                # 1-based (nb02 contract)
                    "cam_view": cam,
                    "gait_event": np.nan,
                    "dataset": "Abnormal Gait",        # matches existing normal rows
                    "gait_pat": "normal",
                    "bbox": str(bbox),
                    "vid_info": str(vid_info),
                    "id": clip_id,                     # resolves to <clip_id>.mp4
                    "url": url,
                    "youtube_video_id": vid,           # grouping / leakage key
                })
            df = pd.DataFrame(rows)
            out = OUT_CSV_DIR / f"{seq_id}.csv"
            df.to_csv(out, index=False)
            prov_rows.append(dict(
                clip=clip.name, clip_id=clip_id, youtube_video_id=vid,
                sequence_id=seq_id, status="written", frames=len(rows),
                first_frame=int(s + 1), last_frame=int(e),
                stride_period=period, seg_method=method,
                cam_view=cam, subject_detect_frac=ok_frac,
                bbox_source="mediapipe_pose_auto", pad=BBOX_PAD,
            ))
            print(f"    wrote {out.name}  ({len(rows)} frames)")

        # link the clip into the augmentation youtube dir under its CLIP id
        # (nb02 find_video looks for <id>.mp4, and we set id = clip_id above).
        link = OUT_VIDEO_DIR / f"{clip_id}.mp4"
        if not link.exists():
            try:
                os.symlink(clip.resolve(), link)
            except OSError:
                import shutil
                shutil.copy2(clip, link)

    prov = pd.DataFrame(prov_rows)
    prov.to_csv(PROVENANCE_CSV, index=False)
    written = prov[prov["status"] == "written"] if "status" in prov else prov
    print(f"\n==== DONE ====")
    n_vids = written["youtube_video_id"].nunique() if len(written) else 0
    print(f"CSVs written: {len(written)} sequences across {n_vids} videos")
    print(f"Provenance: {PROVENANCE_CSV}")


if __name__ == "__main__":
    main()
