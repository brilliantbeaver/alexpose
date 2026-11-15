"""Resumable local GAVD extraction with isolated MediaPipe worker processes."""
from __future__ import annotations
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
from multiprocessing import get_context
from pathlib import Path
import time
import os
import numpy as np
import pandas as pd
from .data import extract_sequence, find_video, load_pose, JOINTS

_EXTRACTOR = None
_MODEL_PATH = None


def _initialize(model_path):
    """Each process owns its singleton and native MediaPipe state."""
    global _EXTRACTOR, _MODEL_PATH
    os.environ.setdefault('OMP_NUM_THREADS', '1')
    from ambient.pose.keypoint_extractor import SequenceKeypointExtractor
    _EXTRACTOR = SequenceKeypointExtractor()
    _MODEL_PATH = model_path


def _extract_job(row, video_path, cache_dir, frame_number_base):
    start = time.monotonic()
    result = {key: row[key] for key in ('sequence_id', 'label', 'source_id')}
    result['checked_at_utc'] = datetime.now(timezone.utc).isoformat()
    result['n_annotation_frames'] = int(row['n_frames'])
    target = Path(cache_dir) / f"{row['sequence_id']}.npz"
    was_cached = target.exists()
    try:
        path = extract_sequence(row, video_path, cache_dir, frame_number_base=frame_number_base,
                                model_path=_MODEL_PATH, extractor=_EXTRACTOR)
        pose = load_pose(path)
        result.update(status='verified_cache' if was_cached else 'extracted', pose_path=str(path),
                      decoded_frames=int(pose['decoded'].sum()), fps=float(pose['fps']),
                      visible_joint_fraction=float((pose['confidence'][:, JOINTS] >= .5).mean()), error='')
    except Exception as exc:
        result.update(status='failed', pose_path='', decoded_frames=0, fps=None,
                      visible_joint_fraction=None, error=f'{type(exc).__name__}: {exc}')
    result['elapsed_seconds'] = time.monotonic() - start
    return result


def _write_report(path, records):
    report = pd.DataFrame(records).sort_values('sequence_id')
    temporary = path.with_suffix('.tmp.csv')
    report.to_csv(temporary, index=False)
    temporary.replace(path)
    return report


def batch_extract(manifest: pd.DataFrame, repo_root: str | Path, cache_dir: str | Path,
                  frame_number_base: int = 1, workers: int = 4,
                  report_path: str | Path | None = None) -> pd.DataFrame:
    """Extract every supplied sequence and atomically record successes/failures.

    Call with the complete frozen manifest for a reported experiment. Each spawn
    worker reuses its own ambient extractor. Existing poses are reused only after
    annotation, video, model and configuration provenance verification. Rerun after
    interruption; this function never loads an old manifest as its data source.
    """
    if workers < 1 or frame_number_base not in (0, 1):
        raise ValueError('workers must be positive and frame base must be 0 or 1')
    root, cache = Path(repo_root).resolve(), Path(cache_dir).resolve()
    cache.mkdir(parents=True, exist_ok=True)
    report_path = Path(report_path) if report_path is not None else cache.parent / 'pose_extraction_report.csv'
    report_path.parent.mkdir(parents=True, exist_ok=True)
    model_path = root / 'data/models/pose_landmarker_full.task'
    if not model_path.exists():
        from ambient.pose.model_management import MediaPipeModelManager
        available = MediaPipeModelManager(models_dir=root / 'data/models').ensure_model_available()
        if available is None:
            raise FileNotFoundError('The full MediaPipe pose model is unavailable')
        model_path = Path(available)
    source_paths = {source: find_video(root, source) for source in manifest.source_id.unique()}
    records, jobs = [], []
    for row in manifest.to_dict('records'):
        video = source_paths[row['source_id']]
        if video is None:
            records.append({**{k: row[k] for k in ('sequence_id','label','source_id')},
                            'checked_at_utc': datetime.now(timezone.utc).isoformat(),
                            'n_annotation_frames': row['n_frames'], 'status': 'video_missing_or_undecodable',
                            'pose_path': '', 'decoded_frames': 0, 'error': 'Source video did not decode',
                            'elapsed_seconds': 0.0})
        else:
            jobs.append((row, str(video)))
    if records:
        _write_report(report_path, records)
    print(f'Extracting {len(jobs)} sequences with {workers} workers; {len(records)} missing source sequences.', flush=True)
    with ProcessPoolExecutor(max_workers=workers, mp_context=get_context('spawn'),
                             initializer=_initialize, initargs=(str(model_path),)) as pool:
        futures = {pool.submit(_extract_job, row, video, str(cache), frame_number_base): row for row,video in jobs}
        for future in as_completed(futures):
            row = futures[future]
            try:
                result = future.result()
            except Exception as exc:
                result = {**{k: row[k] for k in ('sequence_id','label','source_id')},
                          'status':'worker_failed', 'error':f'{type(exc).__name__}: {exc}'}
            records.append(result)
            _write_report(report_path, records)
            print(f"{len(records)}/{len(manifest)} {result['sequence_id']} {result['status']}", flush=True)
    if not records:
        raise ValueError('Empty manifest')
    return _write_report(report_path, records)
