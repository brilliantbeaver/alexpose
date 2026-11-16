"""Ambient pose extraction, explicit frame mappings, and causal window scaling."""
from __future__ import annotations
import hashlib
import json
from pathlib import Path
import numpy as np
import pandas as pd
from .data import JOINT_IDS, find_video, sha256_file

# Pin loaded adapter identity once, even if source files change during a process.
_ADAPTER_SHA256 = sha256_file(__file__)


def download_sources(table, repo_root):
    """Download missing IDs once, through Ambient, under data/videos-gavd/sources."""
    from ambient.utils.youtube_cache import download_youtube_urls
    missing = table.drop_duplicates('source_id')
    urls = [r.url for r in missing.itertuples() if find_video(repo_root, r.source_id, r.label) is None]
    return download_youtube_urls(urls, output_dir=Path(repo_root)/'data/videos-gavd/sources', filename_scheme='id')


def extract_sequence(csv_path, repo_root, model_path, cache_dir, frame_step=2, max_frames=None):
    """Extract requested one-based annotation frames from the complete source.

    Sequential OpenCV decoding does the one-based -> zero-based conversion once.
    Ambient's public RGB image API computes poses; no second frame conversion is
    passed to it. Frames are never renumbered or removed on pose failure.
    `timestamp_seconds` is the OpenCV presentation time; nominal time is separate.
    Full-image first-person detection still requires a person/reference review.
    """
    import cv2
    from ambient.pose.keypoint_extractor import SequenceKeypointExtractor
    import ambient.pose.keypoint_extractor as extractor_module
    csv_path, repo_root, model_path = Path(csv_path), Path(repo_root), Path(model_path)
    if frame_step < 1 or (max_frames is not None and max_frames < 1):
        raise ValueError('frame_step and max_frames must be positive')
    if not model_path.is_file():
        raise FileNotFoundError(f'Explicit MediaPipe model required: {model_path}')
    rows = pd.read_csv(csv_path, keep_default_na=False).sort_values('frame_num').iloc[::frame_step]
    if max_frames:
        rows = rows.iloc[:max_frames]
    if rows.empty or rows.id.nunique() != 1 or rows.seq.nunique() != 1:
        raise ValueError('Expected one nonempty source sequence')
    video = find_video(repo_root, str(rows.id.iloc[0]), str(rows.gait_pat.iloc[0]))
    if video is None:
        raise FileNotFoundError(f'Missing source video: {rows.id.iloc[0]}')
    identity = dict(schema=1, csv_sha256=sha256_file(csv_path), video_sha256=sha256_file(video),
                    model_sha256=sha256_file(model_path), extractor_sha256=sha256_file(extractor_module.__file__),
                    adapter_sha256=_ADAPTER_SHA256, frame_step=frame_step, max_frames=max_frames,
                    coordinate_units='pixels', pose_format='BLAZEPOSE_33', person_selection='full_image_first_detection')
    fingerprint = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
    cache = Path(cache_dir)/f'{rows.seq.iloc[0]}-{fingerprint[:20]}.npz'
    if cache.exists():
        with np.load(cache, allow_pickle=False) as saved:
            result = {k: saved[k] for k in saved.files}
        if str(result['fingerprint']) != fingerprint:
            raise ValueError('Pose cache fingerprint mismatch')
        return result
    requested = rows.frame_num.to_numpy(dtype=int)
    if np.any(requested < 1) or len(np.unique(requested)) != len(requested):
        raise ValueError('Frame indices must be unique one-based values')
    frame_to_output = {f-1:i for i,f in enumerate(requested)}
    xy = np.full((len(rows),33,2), np.nan, np.float32)
    confidence = np.zeros((len(rows),33), np.float32)
    times = np.full(len(rows), np.nan)
    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        raise RuntimeError(f'Cannot decode {video}')
    fps = cap.get(cv2.CAP_PROP_FPS)
    width, height = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    if not np.isfinite(fps) or fps <= 0:
        cap.release()
        raise ValueError('Video FPS unavailable')
    extractor = SequenceKeypointExtractor()
    try:
        for decoded_index in range(int(requested.max())):
            ok, frame = cap.read()
            if not ok:
                break
            if decoded_index not in frame_to_output:
                continue
            i = frame_to_output[decoded_index]
            times[i] = cap.get(cv2.CAP_PROP_POS_MSEC)/1000.0
            pose = extractor.extract_from_image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), str(model_path))
            if pose is not None:
                for kp in pose.keypoints:
                    if 0 <= kp.id < 33:
                        xy[i,kp.id] = [kp.x,kp.y]
                        confidence[i,kp.id] = kp.confidence
    finally:
        cap.release()
        extractor.cleanup()
    result = dict(xy=xy, confidence=confidence, frame_numbers=requested,
                  decoded_indexes=requested-1, timestamp_seconds=times,
                  nominal_seconds=(requested-1)/fps, fps=np.array(fps), sample_fps=np.array(fps/frame_step),
                  image_dimensions=np.array([width,height]), sequence_id=np.array(rows.seq.iloc[0]),
                  source_id=np.array(rows.id.iloc[0]), label=np.array(rows.gait_pat.iloc[0]),
                  fingerprint=np.array(fingerprint), provenance_json=np.array(json.dumps(identity, sort_keys=True)))
    cache.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(cache, **result)
    return result


def normalize_window(xy, confidence, context_frames=8, confidence_threshold=0.5):
    """Pelvis-centered 2D coordinates; scale uses the prefix only.

    Missing pelvis invalidates that frame. Missing observations become zero only
    after validity is retained. The target is relative articulation, not physical
    locomotion displacement. No interpolated point becomes observed evidence.
    """
    xy, confidence = np.asarray(xy, float), np.asarray(confidence, float)
    if xy.ndim != 3 or xy.shape[-1] != 2 or confidence.shape != xy.shape[:-1]:
        raise ValueError('Expected xy[T,J,2], confidence[T,J]')
    if xy.shape[1] == 33:
        xy, confidence = xy[:,JOINT_IDS], confidence[:,JOINT_IDS]
    if xy.shape[1] != len(JOINT_IDS) or not 1 <= context_frames <= len(xy):
        raise ValueError('Expected declared 16 joints and a nonempty context prefix')
    if not 0 <= confidence_threshold <= 1:
        raise ValueError('Confidence threshold must be in [0,1]')
    valid = np.isfinite(xy).all(-1) & np.isfinite(confidence) & (confidence >= confidence_threshold)
    j = {joint:i for i,joint in enumerate(JOINT_IDS)}
    hips_ok = valid[:,j[23]] & valid[:,j[24]]
    shoulders_ok = valid[:,j[11]] & valid[:,j[12]]
    pelvis = (xy[:,j[23]] + xy[:,j[24]])/2
    shoulders = (xy[:,j[11]] + xy[:,j[12]])/2
    lengths = np.linalg.norm(shoulders-pelvis, axis=-1)
    scale_ok = hips_ok[:context_frames] & shoulders_ok[:context_frames] & (lengths[:context_frames] > 1e-6)
    if not scale_ok.any():
        raise ValueError('No valid torso scale in context prefix')
    scale = np.median(lengths[:context_frames][scale_ok])
    valid &= hips_ok[:,None]
    normalized = (xy-pelvis[:,None])/scale
    return np.where(valid[...,None], normalized, 0).astype(np.float32), valid


def make_windows(pose, frames=16, stride=8, context_frames=8, confidence_threshold=0.5, min_valid_fraction=0.7):
    """Return disjointly identified windows; reject gaps and poor observations."""
    if frames < 2 or stride < 1 or not 0 <= min_valid_fraction <= 1:
        raise ValueError('Invalid window or quality settings')
    x, validity, metadata = [], [], []
    numbers = np.asarray(pose['frame_numbers'])
    for start in range(0, len(numbers)-frames+1, stride):
        stop = start+frames
        selected = numbers[start:stop]
        if not np.all(np.diff(selected) == np.diff(selected)[0]):
            continue
        try:
            values, valid = normalize_window(pose['xy'][start:stop], pose['confidence'][start:stop], context_frames, confidence_threshold)
        except ValueError:
            continue
        if valid.mean() < min_valid_fraction or not valid[:context_frames].any() or not valid[context_frames:].any():
            continue
        x.append(values)
        validity.append(valid)
        metadata.append(dict(sequence_id=str(pose['sequence_id']), source_id=str(pose['source_id']),
                             label=str(pose['label']), first_frame=int(selected[0]), last_frame=int(selected[-1]),
                             valid_fraction=float(valid.mean()), sample_fps=float(pose['fps'])/int(np.diff(selected)[0]),
                             pose_fingerprint=str(pose['fingerprint'])))
    return (np.stack(x) if x else np.empty((0,frames,len(JOINT_IDS),2),np.float32),
            np.stack(validity) if validity else np.empty((0,frames,len(JOINT_IDS)),bool), pd.DataFrame(metadata))


def prepare_windows(table, repo_root, model_path, cache_dir, frames=16, stride=8, context_frames=8,
                    frame_step=2, max_sequences=None, max_frames=None, confidence_threshold=0.5,
                    min_valid_fraction=0.7):
    """Prepare a fixed input cohort and return x, valid, window metadata, attrition.

    For a bounded pilot, pass an explicitly selected table and max_frames. Source
    membership must already be assigned with data.assign_splits; failures are
    reported and never cause replacement sources or split changes.
    """
    if 'split' not in table or table.groupby('source_id').split.nunique().max() != 1:
        raise ValueError('A source-disjoint partition table is required')
    selected = table.loc[table.eligible].copy()
    if max_sequences is not None:
        selected = selected.head(max_sequences)
    all_x, all_valid, all_meta, attrition = [], [], [], []
    for row in selected.itertuples():
        try:
            pose = extract_sequence(Path(repo_root)/row.csv_path, repo_root, model_path, cache_dir, frame_step, max_frames)
            x, valid, meta = make_windows(pose, frames, stride, context_frames, confidence_threshold, min_valid_fraction)
            reason = '' if len(x) else 'no_quality_eligible_window'
            if len(x):
                meta['split'] = row.split
                all_x.append(x); all_valid.append(valid); all_meta.append(meta)
        except (ValueError, RuntimeError, FileNotFoundError) as exc:
            reason, x = str(exc), []
        attrition.append(dict(sequence_id=row.sequence_id, source_id=row.source_id, label=row.label,
                              split=row.split, windows=len(x), exclusion_reason=reason))
    return (np.concatenate(all_x) if all_x else np.empty((0,frames,len(JOINT_IDS),2),np.float32),
            np.concatenate(all_valid) if all_valid else np.empty((0,frames,len(JOINT_IDS)),bool),
            pd.concat(all_meta, ignore_index=True) if all_meta else pd.DataFrame(), pd.DataFrame(attrition))


def video_preview(path, start_seconds=0.0, end_seconds=None):
    """Notebook player for the original video, positioned near the annotation."""
    from IPython.display import Video
    # IPython embeds bytes, so a notebook server does not need repository-wide access.
    # Embedding a long video enlarges the notebook. Clear outputs before sharing.
    video = Video(str(Path(path)), embed=True, html_attributes='controls preload="metadata"')
    if start_seconds or end_seconds is not None:
        from IPython.display import HTML
        html = video._repr_html_().replace('<video ', f'<video onloadedmetadata="this.currentTime={float(start_seconds)};" ')
        if end_seconds is not None:
            html = html.replace('<video ', f'<video ontimeupdate="if(this.currentTime>={float(end_seconds)})this.pause();" ')
        return HTML(html)
    return video
