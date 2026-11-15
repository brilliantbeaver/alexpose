"""Fresh GAVD ingestion with source-group splits and explicit pose provenance.

Video files stay under data/videos-gavd/<source_id>/. A source ID identifies a
recording, not a participant. Never infer person-disjoint evaluation from it.
"""
from __future__ import annotations

import ast
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
from typing import Mapping

import numpy as np
import pandas as pd

LABELS = ("normal", "parkinsons", "stroke", "cerebral palsy", "myopathic")
JOINTS = (11, 12, 13, 14, 15, 16, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32)
POSE_VERSION = "aaai_bbox_xy_v1"
CSV_COLUMNS = ("seq", "frame_num", "cam_view", "gait_event", "dataset", "gait_pat", "bbox", "vid_info", "id", "url")


def file_sha256(path: str | Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def build_manifest(repo_root: str | Path) -> pd.DataFrame:
    """Read every current selected CSV. No saved manifest or source CSV is read.

    The returned inventory has one row per annotated sequence, before downloads
    or pose exclusions. Reject changed schema, duplicate IDs and inconsistent
    labels instead of silently selecting a convenient subset.
    """
    root = Path(repo_root).resolve() / "data" / "gavd"
    records = []
    for label in LABELS:
        paths = sorted((root / label).glob("*.csv"))
        if not paths:
            raise FileNotFoundError(f"No current CSV files: {root / label}")
        for path in paths:
            table = pd.read_csv(path, keep_default_na=False, dtype=str)
            if tuple(table.columns) != CSV_COLUMNS or table.empty:
                raise ValueError(f"Empty CSV or unexpected schema: {path}")
            for key, value in (("seq", path.stem), ("gait_pat", label)):
                if not table[key].eq(value).all():
                    raise ValueError(f"Inconsistent {key}: {path}")
            if table["id"].nunique() != 1 or table["url"].nunique() != 1:
                raise ValueError(f"Multiple sources in sequence: {path}")
            source = str(table.iloc[0]["id"])
            if not re.fullmatch(r"[A-Za-z0-9_-]{11}", source):
                raise ValueError(f"Invalid source video ID: {source}")
            numeric = pd.to_numeric(table.frame_num, errors="raise").to_numpy(float)
            if np.any(numeric < 1) or not np.equal(numeric, np.floor(numeric)).all():
                raise ValueError(f"Frames must be positive one-based integers: {path}")
            if len(np.unique(numeric)) != len(numeric):
                raise ValueError(f"Duplicate source frame in {path}")
            records.append(dict(sequence_id=path.stem, label=label, source_id=source,
                                csv_path=str(path), csv_sha256=file_sha256(path),
                                n_frames=len(table), first_frame=int(numeric.min()),
                                last_frame=int(numeric.max()), url=str(table.iloc[0].url),
                                camera_views=",".join(sorted(set(table.cam_view)))))
    result = pd.DataFrame(records)
    if result.sequence_id.duplicated().any():
        raise ValueError("Sequence ID occurs in more than one label directory")
    result.attrs["manifest_sha256"] = manifest_digest(result)
    return result


def manifest_digest(manifest: pd.DataFrame) -> str:
    """Portable annotation identity, independent of absolute repository path."""
    cols = ["sequence_id", "label", "source_id", "csv_sha256"]
    values = manifest.sort_values("sequence_id")[cols].to_dict("records")
    return hashlib.sha256(json.dumps(values, sort_keys=True).encode()).hexdigest()


def assert_group_disjoint(manifest: pd.DataFrame, require_coverage: bool = True) -> None:
    """Validate source-level separation and optional five-class split coverage."""
    if not set(manifest.split).issubset({"train", "val", "test"}):
        raise ValueError("Unknown split label")
    if manifest.groupby("source_id").split.nunique().max() > 1:
        raise ValueError("Source video leaks between splits")
    if require_coverage:
        for split in ("train", "val", "test"):
            missing = set(LABELS) - set(manifest.loc[manifest.split.eq(split), "label"])
            if missing:
                raise ValueError(f"{split} lacks classes after selection/QC: {sorted(missing)}")


def grouped_split(manifest: pd.DataFrame, seed: int = 42,
                  fractions=(0.6, 0.2, 0.2), attempts: int = 2000) -> pd.DataFrame:
    """Choose a deterministic source assignment using only class counts.

    Minimize deviation from class-specific source and sequence proportions.
    A mixed-label source remains intact. This never uses pose values, model
    predictions or performance to choose a split. Freeze this table before QC.
    """
    f = np.asarray(fractions, float)
    if len(f) != 3 or np.any(f <= 0) or not np.isclose(f.sum(), 1):
        raise ValueError("Three positive train/val/test fractions must sum to one")
    if set(manifest.label) != set(LABELS):
        raise ValueError("Split requires all five prespecified labels")
    counts = pd.crosstab(manifest.source_id, manifest.label).reindex(columns=LABELS, fill_value=0)
    if ((counts > 0).sum(axis=0) < 3).any():
        raise ValueError("Every class needs at least three source groups for three partitions")
    groups, values = counts.index.to_numpy(), counts.to_numpy()
    n = len(groups)
    edges = np.rint(np.cumsum(f)[:-1] * n).astype(int)
    rng = np.random.default_rng(seed)
    best = None
    pure_groups = np.all((values > 0).sum(axis=1) == 1)
    for _ in range(attempts):
        if pure_groups:
            collected = [[], [], []]
            for k in range(len(LABELS)):
                group_indices = rng.permutation(np.flatnonzero(values[:, k]))
                nval = max(1, int(round(f[1] * len(group_indices))))
                ntest = max(1, int(round(f[2] * len(group_indices))))
                while nval + ntest >= len(group_indices):
                    if nval >= ntest and nval > 1:
                        nval -= 1
                    elif ntest > 1:
                        ntest -= 1
                    else:
                        raise ValueError("Insufficient source groups")
                ntrain = len(group_indices) - nval - ntest
                parts = np.split(group_indices, [ntrain, ntrain+nval])
                for j, part in enumerate(parts):
                    collected[j].extend(part)
            pieces = [np.asarray(part, dtype=int) for part in collected]
        else:
            perm = rng.permutation(n)
            pieces = np.split(perm, edges)
        observed = np.stack([values[p].sum(axis=0) for p in pieces])
        source_observed = np.stack([(values[p] > 0).sum(axis=0) for p in pieces])
        if np.any(observed == 0):
            continue
        cost = np.square(observed / observed.sum(axis=0) - f[:, None]).mean()
        cost += np.square(source_observed / source_observed.sum(axis=0) - f[:, None]).mean()
        if best is None or cost < best[0]:
            best = (float(cost), pieces)
    if best is None:
        raise ValueError("No split with all labels found. Use a registered grouped CV protocol.")
    assignment = {str(groups[i]): split for split, part in zip(("train", "val", "test"), best[1]) for i in part}
    out = manifest.copy()
    out["split"] = out.source_id.map(assignment)
    assert_group_disjoint(out)
    out.attrs.update(manifest.attrs)
    out.attrs["split_seed"] = seed
    out.attrs["split_sha256"] = hashlib.sha256(out.sort_values("sequence_id")[["sequence_id", "split"]].to_csv(index=False).encode()).hexdigest()
    return out


def find_video(repo_root: str | Path, source_id: str) -> Path | None:
    """Find one complete source file, never a partial download or audio stream."""
    if not re.fullmatch(r"[A-Za-z0-9_-]{11}", source_id):
        raise ValueError("Invalid source ID")
    root = Path(repo_root).resolve() / "data" / "videos-gavd"
    paths = [p for directory in (root / source_id, root)
             for ext in ("mp4", "mkv", "webm", "mov")
             for p in directory.glob(f"{source_id}.{ext}")]
    import cv2
    for path in paths:
        capture = cv2.VideoCapture(str(path))
        try:
            ok, _ = capture.read()
            if ok:
                return path
        finally:
            capture.release()
    return None


def download_sources(manifest: pd.DataFrame, repo_root: str | Path,
                     limit: int | None = None) -> pd.DataFrame:
    """Download each unique recording once through ambient's yt-dlp adapter.

    Checks that a complete output file exists, rather than trusting yt-dlp's
    status alone. A returned file is still subject to decode/pose QC. Failure
    records remain in the inventory. Credentials belong in the root .env.
    """
    from ambient.utils.youtube_cache import download_youtube_urls
    root = Path(repo_root).resolve()
    rows = []
    sources = manifest.drop_duplicates("source_id").sort_values("source_id")
    if limit is not None:
        if limit < 1:
            raise ValueError("limit must be positive or None")
        sources = sources.head(limit)
    for row in sources.itertuples(index=False):
        target = root / "data" / "videos-gavd" / row.source_id
        before = find_video(root, row.source_id)
        error = ""
        if before is None:
            try:
                response = download_youtube_urls([f"https://www.youtube.com/watch?v={row.source_id}"],
                                      output_dir=target, max_retries=2,
                                      format_selector="best[ext=mp4][height<=720]/best[height<=720]",
                                      max_concurrent_fragments=1)
                error = response.get("details", {}).get(f"https://www.youtube.com/watch?v={row.source_id}", {}).get("error", "")
            except Exception as exc:
                error = f"{type(exc).__name__}: {exc}"
        path = find_video(root, row.source_id)
        if path is None and not error:
            error = "No complete decodable source video after download; inspect the per-source log."
        rows.append(dict(source_id=row.source_id, status="present" if path else "failed",
                         checked_at_utc=datetime.now(timezone.utc).isoformat(),
                         video_path=str(path) if path else "", error=error,
                         video_sha256=file_sha256(path) if path else ""))
    report = pd.DataFrame(rows)
    output = root / "data" / "videos-gavd" / "download_report.csv"
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        previous = pd.read_csv(output, keep_default_na=False, dtype=str)
        merged = pd.concat([previous.loc[~previous.source_id.isin(report.source_id)], report], ignore_index=True)
    else:
        merged = report
    temporary = output.with_suffix(".tmp.csv")
    merged.sort_values("source_id").to_csv(temporary, index=False)
    temporary.replace(output)
    return report


def normalize_pose(raw_xy: np.ndarray, confidence: np.ndarray,
                   min_visibility: float = 0.5) -> np.ndarray:
    """Pelvis-center, divide by median visible torso length, keep missing NaN.

    Median sequence scale is robust to frame jitter but uses the whole clip.
    These representations support retrospective clip scoring. For causal
    deployment, estimate scale from the observed prefix only.
    """
    xy = np.asarray(raw_xy, np.float32).copy()
    xy[(~np.isfinite(confidence)) | (confidence < min_visibility)] = np.nan
    pelvis = (xy[:, 23] + xy[:, 24]) / 2
    shoulders = (xy[:, 11] + xy[:, 12]) / 2
    torso = np.linalg.norm(shoulders - pelvis, axis=1)
    good = torso[np.isfinite(torso) & (torso > 1e-6)]
    if len(good) == 0:
        return np.full_like(xy, np.nan)
    return (xy - pelvis[:, None]) / np.median(good)


def extract_sequence(row: Mapping, video_path: str | Path, cache_dir: str | Path,
                     *, frame_number_base: int, model_path: str | Path | None = None,
                     extractor=None, min_visibility: float = 0.5,
                     bbox_margin: float = 0.05) -> Path:
    """Use per-frame GAVD bounding boxes with ambient's MediaPipe extractor.

    A margin reduces cut-off limbs but is not a guarantee of person identity.
    Inspect a bbox overlay before batch runs. CSV bbox coordinates are scaled
    from annotated image dimensions into the downloaded video's dimensions.
    The explicit frame base records the caller's verified indexing convention.
    The repository's GAVD adapter uses one-based source frame numbers.
    """
    import cv2
    if frame_number_base not in (0, 1):
        raise ValueError("frame_number_base must be explicitly 0 or 1")
    if not 0 <= min_visibility <= 1 or not 0 <= bbox_margin <= 0.5:
        raise ValueError("Invalid confidence or bbox margin")
    row = dict(row)
    csv_path, video_path = Path(row["csv_path"]), Path(video_path)
    csv_hash = file_sha256(csv_path)
    if csv_hash != row["csv_sha256"]:
        raise ValueError("Annotations changed after manifest creation. Rebuild manifest and split.")
    if extractor is None:
        from ambient.pose.keypoint_extractor import SequenceKeypointExtractor
        extractor = SequenceKeypointExtractor()
        if model_path is None:
            model_path = extractor.model_manager.ensure_model_available()
            if model_path is None:
                raise FileNotFoundError("No pose model is available")
    model_hash = file_sha256(model_path) if model_path else "injected-test-extractor"
    context = dict(version=POSE_VERSION, sequence_id=row["sequence_id"], source_id=row["source_id"],
                   csv_sha256=csv_hash, video_sha256=file_sha256(video_path), model_sha256=model_hash,
                   frame_number_base=frame_number_base, min_visibility=min_visibility, bbox_margin=bbox_margin)
    out = Path(cache_dir) / f"{row['sequence_id']}.npz"
    if out.exists():
        previous = load_pose(out)
        if json.loads(str(previous["provenance"])) == context:
            return out
        raise ValueError(f"Stale pose cache: {out}. Archive or remove it explicitly before rerunning.")
    annotations = pd.read_csv(csv_path, keep_default_na=False, dtype={"seq": str, "id": str})
    annotations["frame_num"] = pd.to_numeric(annotations.frame_num, errors="raise")
    annotations = annotations.sort_values("frame_num")
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise IOError(f"Video cannot be decoded: {video_path}")
    fps = float(cap.get(cv2.CAP_PROP_FPS))
    if not np.isfinite(fps) or fps <= 0:
        cap.release()
        raise ValueError("Source FPS is unknown. Do not invent a default.")
    xy = np.full((len(annotations), 33, 2), np.nan, np.float32)
    confidence = np.zeros((len(annotations), 33), np.float32)
    decoded = np.zeros(len(annotations), bool)
    frame_numbers = annotations.frame_num.to_numpy(np.int64)
    try:
        previous_index = None
        for t, ann in enumerate(annotations.to_dict("records")):
            index = int(ann["frame_num"]) - frame_number_base
            if index < 0:
                raise ValueError("Frame base yields negative frame index")
            if previous_index is None or index != previous_index + 1:
                cap.set(cv2.CAP_PROP_POS_FRAMES, index)
            ok, bgr = cap.read()
            previous_index = index
            if not ok:
                previous_index = None
                continue
            decoded[t] = True
            box, info = ast.literal_eval(ann["bbox"]), ast.literal_eval(ann["vid_info"])
            height, width = bgr.shape[:2]
            sx, sy = width / float(info["width"]), height / float(info["height"])
            if min(float(box["width"]), float(box["height"])) <= 0:
                continue
            x0 = max(0, int(np.floor((box["left"] - bbox_margin * box["width"]) * sx)))
            y0 = max(0, int(np.floor((box["top"] - bbox_margin * box["height"]) * sy)))
            x1 = min(width, int(np.ceil((box["left"] + (1 + bbox_margin) * box["width"]) * sx)))
            y1 = min(height, int(np.ceil((box["top"] + (1 + bbox_margin) * box["height"]) * sy)))
            if x1 <= x0 or y1 <= y0:
                continue
            image = cv2.cvtColor(bgr[y0:y1, x0:x1], cv2.COLOR_BGR2RGB)
            result = extractor.extract_from_image(image, model_path=str(model_path) if model_path else None)
            for keypoint in getattr(result, "keypoints", []):
                j = int(keypoint.id)
                if 0 <= j < 33:
                    xy[t, j] = (keypoint.x + x0, keypoint.y + y0)
                    confidence[t, j] = keypoint.confidence
    finally:
        cap.release()
    normalized = normalize_pose(xy, confidence, min_visibility)
    out.parent.mkdir(parents=True, exist_ok=True)
    temporary = out.with_suffix(".tmp.npz")
    np.savez_compressed(temporary, raw_xy=xy, xy=normalized, confidence=confidence,
                        decoded=decoded, frame_numbers=frame_numbers, fps=np.float64(fps),
                        provenance=json.dumps(context, sort_keys=True))
    temporary.replace(out)
    return out


def load_pose(path: str | Path) -> dict:
    with np.load(path, allow_pickle=False) as values:
        return {key: values[key] for key in values.files}


def _resample(xy: np.ndarray, frame_numbers: np.ndarray, fps: float, target_fps: float, frame_number_base: int = 1):
    """Linear time interpolation only between adjacent valid source frames."""
    if len(frame_numbers) < 2:
        return np.empty((0, *xy.shape[1:]), np.float32), np.empty(0)
    if not np.isfinite(fps) or fps <= 0 or not np.isfinite(target_fps) or target_fps <= 0:
        raise ValueError("Sampling rates must be finite and positive")
    times = (frame_numbers.astype(float) - frame_number_base) / fps
    if np.any(np.diff(times) <= 0):
        raise ValueError("Frame numbers must be strictly increasing")
    grid = times[0] + np.arange(int(np.floor((times[-1] - times[0]) * target_fps + 1e-6)) + 1) / target_fps
    left = np.searchsorted(times, grid, side="right") - 1
    left = np.clip(left, 0, len(times) - 2)
    right = left + 1
    fraction = (grid - times[left]) / (times[right] - times[left])
    # Exact observed frames require only their own detection, not a neighbor.
    out = (1 - fraction[:, None, None]) * xy[left] + fraction[:, None, None] * xy[right]
    out[(frame_numbers[right] - frame_numbers[left]) != 1] = np.nan
    exact_left, exact_right = np.isclose(fraction, 0), np.isclose(fraction, 1)
    out[exact_left] = xy[left[exact_left]]
    out[exact_right] = xy[right[exact_right]]
    return out.astype(np.float32), grid


def normalize_window(raw_selected: np.ndarray) -> np.ndarray:
    """Center each frame and estimate scale from the first half only.

    Future frame coordinates cannot change the normalized observed prefix.
    This supports forecasting tasks whose observed prefix is at least half of
    the window. Shorter-prefix experiments must change and register this rule.
    """
    xy = np.asarray(raw_selected, np.float32)
    index = {joint: i for i,joint in enumerate(JOINTS)}
    pelvis = (xy[:,index[23]] + xy[:,index[24]]) / 2
    shoulder = (xy[:,index[11]] + xy[:,index[12]]) / 2
    torso = np.linalg.norm(shoulder[:len(xy)//2] - pelvis[:len(xy)//2], axis=-1)
    good = torso[np.isfinite(torso) & (torso > 1e-6)]
    if not len(good):
        return np.full_like(xy, np.nan)
    return (xy - pelvis[:,None]) / np.median(good)


def load_windows(manifest: pd.DataFrame, cache_dir: str | Path, frames: int = 32,
                 stride: int = 16, target_fps: float = 15,
                 require_coverage: bool = True, expected_frame_number_base: int = 1,
                 expected_min_visibility: float = 0.5,
                 expected_bbox_margin: float = 0.05,
                 patch_frames: int = 4, min_valid_token_fraction: float = 0.75,
                 expected_model_sha256: str | None = None) -> tuple[np.ndarray, pd.DataFrame]:
    """Return [N,frames,16,2] and matching window metadata from current poses.

    Frames are centered on their pelvis. Window scale uses only the first half
    of the window, preventing future torso information entering past context.
    Sources below target_fps are excluded, avoiding interpolation lookahead
    across the raw-time forecast boundary. Missing coordinates remain NaN.
    A token is valid only when every XY value
    in its joint-time patch is finite. Accept >=75% valid tokens overall and
    >=50% in each temporal half. The model must mask invalid attention keys
    and targets. No window crosses a sequence; source splits propagate intact.
    Missing/stale files are exclusions, reported in metadata.attrs['exclusions'].
    By default fail if QC removes a class from any split. Set coverage=False only
    for the explicitly limited notebook preview, never for reported experiments.
    """
    if frames < 2 or stride < 1 or target_fps <= 0 or patch_frames < 1 or frames % (2 * patch_frames):
        raise ValueError("Window must contain an even number of complete time patches")
    if not 0 < min_valid_token_fraction <= 1:
        raise ValueError("Token coverage must be in (0,1]")
    assert_group_disjoint(manifest, require_coverage=False)
    windows, metadata, exclusions = [], [], []
    pose_hashes, model_hashes = {}, set()
    for row in manifest.to_dict("records"):
        path = Path(cache_dir) / f"{row['sequence_id']}.npz"
        reason = ""
        if not path.exists():
            reason = "pose_missing"
        else:
            pose = load_pose(path)
            context = json.loads(str(pose["provenance"]))
            if context.get("csv_sha256") != row["csv_sha256"] or context.get("version") != POSE_VERSION or context.get("source_id") != row["source_id"]:
                reason = "pose_provenance_mismatch"
            elif context.get("frame_number_base") != expected_frame_number_base or context.get("min_visibility") != expected_min_visibility or context.get("bbox_margin") != expected_bbox_margin:
                reason = "pose_configuration_mismatch"
            elif expected_model_sha256 is not None and context.get("model_sha256") != expected_model_sha256:
                reason = "pose_model_mismatch"
            elif file_sha256(row["csv_path"]) != row["csv_sha256"]:
                reason = "annotation_changed"
            elif not np.isfinite(float(pose["fps"])) or float(pose["fps"]) <= 0:
                reason = "invalid_source_fps"
            elif float(pose["fps"]) < target_fps:
                reason = "source_fps_below_target"
        if reason:
            exclusions.append({**{k: row[k] for k in ("sequence_id", "label", "source_id", "split")}, "reason": reason})
            continue
        pose_hashes[row["sequence_id"]] = file_sha256(path)
        model_hashes.add(context.get("model_sha256", "unknown"))
        raw = pose["raw_xy"].copy()
        raw[(~np.isfinite(pose["confidence"])) | (pose["confidence"] < expected_min_visibility)] = np.nan
        xy, times = _resample(raw[:, JOINTS], pose["frame_numbers"], float(pose["fps"]), target_fps, expected_frame_number_base)
        kept = 0
        rejected = 0
        for start in range(0, len(xy) - frames + 1, stride):
            window = normalize_window(xy[start:start + frames])
            valid_tokens = np.isfinite(window.reshape(frames // patch_frames, patch_frames, len(JOINTS), 2)).all(axis=(1,3))
            half = len(valid_tokens) // 2
            coverage = float(valid_tokens.mean())
            prefix_coverage = float(valid_tokens[:half].mean())
            future_coverage = float(valid_tokens[half:].mean())
            if coverage < min_valid_token_fraction or min(prefix_coverage, future_coverage) < 0.5:
                rejected += 1
                continue
            windows.append(window)
            metadata.append({**{k: row[k] for k in ("sequence_id", "label", "source_id", "split")},
                             "window_id": f"{row['sequence_id']}:{start}", "start_seconds": float(times[start]),
                             "end_seconds": float(times[start + frames - 1]), "fps": target_fps,
                             "source_fps": float(pose["fps"]), "valid_token_fraction": coverage, "prefix_valid_fraction": prefix_coverage,
                             "future_valid_fraction": future_coverage})
            kept += 1
        if not kept or rejected:
            exclusions.append({**{k: row[k] for k in ("sequence_id", "label", "source_id", "split")},
                               "reason": ("sequence_too_short" if not kept and not rejected else "no_valid_windows") if not kept else "partial_window_qc",
                               "windows_kept": kept, "windows_rejected": rejected})
    meta = pd.DataFrame(metadata, columns=["sequence_id", "label", "source_id", "split", "window_id", "start_seconds", "end_seconds", "fps", "source_fps", "valid_token_fraction", "prefix_valid_fraction", "future_valid_fraction"])
    meta.attrs["exclusions"] = exclusions
    meta.attrs["manifest_sha256"] = manifest_digest(manifest)
    if len(model_hashes) > 1:
        raise ValueError("Pose files mix different model weights. Regenerate a consistent corpus.")
    corpus_context = dict(manifest_sha256=manifest_digest(manifest), pose_sha256=pose_hashes,
                          frames=frames, stride=stride, target_fps=target_fps,
                          normalization="prefix_half_median_torso_v1", joints=JOINTS,
                          patch_frames=patch_frames, min_valid_token_fraction=min_valid_token_fraction,
                          min_half_valid_fraction=0.5, missingness="nan_token_validity_v1",
                          upsampling_allowed=False)
    meta.attrs["corpus_sha256"] = hashlib.sha256(json.dumps(corpus_context,sort_keys=True).encode()).hexdigest()
    meta.attrs["pose_model_sha256"] = sorted(model_hashes)
    if require_coverage:
        assert_group_disjoint(meta)
    array = np.stack(windows) if windows else np.empty((0, frames, len(JOINTS), 2), np.float32)
    return array, meta
