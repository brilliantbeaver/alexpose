"""Refresh all GAVD annotations, attempt bounded downloads, and extract available media.

All source media and download logs stay in data/videos-gavd. Existing annotation
and media hashes are verified by ambient-backed extraction. No model scores
select downloads, exclusions, or source groups.
"""
from pathlib import Path
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
import json
import os
import signal
import subprocess
import sys
import time

BASE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BASE))
from aaai_gait.settings import ROOT, Settings
from aaai_gait.data import find_video, file_sha256
from aaai_gait.workflow import write_json


def download_one(source_id):
    from ambient.utils.youtube_cache import download_youtube_urls
    target = ROOT / 'data/videos-gavd' / source_id
    response = download_youtube_urls(
        [f'https://www.youtube.com/watch?v={source_id}'], output_dir=target,
        max_retries=1, format_selector='best[ext=mp4][height<=720]/best[height<=720]',
        max_concurrent_fragments=1)
    path = find_video(ROOT, source_id)
    return {'source_id': source_id, 'status': 'downloaded' if path else 'failed',
            'video_path': str(path) if path else '',
            'error': '' if path else str(response.get('details', {}))[-1200:]}


def attempt(source, timeout, deadline, log_dir):
    record = {'source_id': source, 'checked_at_utc': datetime.now(timezone.utc).isoformat()}
    if time.monotonic() >= deadline:
        return {**record, 'status': 'budget_not_attempted', 'video_path': '', 'error': ''}
    log = log_dir / f'{source}.log'
    with log.open('w') as stream:
        proc = subprocess.Popen([sys.executable, str(Path(__file__).resolve()), '--source', source],
                                stdout=stream, stderr=subprocess.STDOUT, start_new_session=True)
        try:
            code = proc.wait(timeout=timeout)
            result_file = log_dir / f'{source}.json'
            if code == 0 and result_file.exists():
                return {**record, **json.loads(result_file.read_text())}
            return {**record, 'status': 'failed', 'video_path': '', 'error': f'Child exit {code}; see {log.name}'}
        except subprocess.TimeoutExpired:
            os.killpg(proc.pid, signal.SIGTERM)
            try:
                proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                os.killpg(proc.pid, signal.SIGKILL); proc.wait()
            return {**record, 'status': 'timeout', 'video_path': '', 'error': f'Acquisition cutoff {timeout}s'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', help=argparse.SUPPRESS)
    parser.add_argument('--download-missing', action='store_true')
    parser.add_argument('--download-budget-seconds', type=int, default=900)
    parser.add_argument('--source-timeout', type=int, default=45)
    parser.add_argument('--download-workers', type=int, default=4)
    parser.add_argument('--workers', type=int, default=4, help='Independent pose processes')
    parser.add_argument('--inventory-only', action='store_true')
    args = parser.parse_args()
    log_dir = ROOT / 'data/videos-gavd/acquisition_logs'
    log_dir.mkdir(parents=True, exist_ok=True)
    if args.source:
        # find_video validates the source ID before it can form an output path.
        find_video(ROOT, args.source)
        result = download_one(args.source)
        write_json(log_dir / f'{args.source}.json', result)
        return
    if min(args.source_timeout, args.download_workers, args.workers, args.download_budget_seconds) < 1:
        parser.error('All budgets and worker counts must be positive.')
    from aaai_gait.study_data import build_study_manifest, load_study_windows
    from aaai_gait.batch import batch_extract
    import pandas as pd
    settings = Settings()
    out = settings.artifact_dir / 'context_study/ingestion'
    out.mkdir(parents=True, exist_ok=True)
    manifest = build_study_manifest(ROOT)
    manifest.to_csv(out / 'fresh_manifest.csv', index=False)
    print(f'Fresh inventory: {len(manifest)} sequences; {manifest.source_id.nunique()} sources.', flush=True)
    records, missing = [], []
    # Normal sources get first acquisition opportunity, then a stable ID order.
    sources = manifest.assign(priority=manifest.label.ne('normal').astype(int)).sort_values(['priority','source_id']).drop_duplicates('source_id')
    for row in sources.itertuples(index=False):
        path = find_video(ROOT, row.source_id)
        if path is None:
            missing.append(row.source_id)
        else:
            records.append({'source_id': row.source_id, 'status': 'verified_present',
                            'video_path': str(path), 'video_sha256': file_sha256(path), 'error': ''})
    print(f'Decodable local sources: {len(records)}; missing: {len(missing)}.', flush=True)
    if args.download_missing and missing:
        deadline = time.monotonic() + args.download_budget_seconds
        with ThreadPoolExecutor(max_workers=args.download_workers) as pool:
            futures = {pool.submit(attempt, s, args.source_timeout, deadline, log_dir): s for s in missing}
            for future in as_completed(futures):
                result = future.result(); records.append(result)
                pd.DataFrame(records).to_csv(out / 'media_ledger.csv', index=False)
                if result['status'] != 'budget_not_attempted':
                    print(result['source_id'], result['status'], flush=True)
    else:
        records += [{'source_id':s,'status':'not_attempted','video_path':'','error':''} for s in missing]
    ledger = pd.DataFrame(records)
    ledger.to_csv(out / 'media_ledger.csv', index=False)
    write_json(out / 'acquisition_protocol.json', {
        'created_at_utc': datetime.now(timezone.utc).isoformat(),
        'settings':vars(args), 'ordering':'normal-first then source ID; no model scores',
        'media_status_counts':ledger.status.value_counts().to_dict()})
    if args.inventory_only:
        return
    batch_extract(manifest, ROOT, settings.pose_dir, settings.frame_number_base,
                  workers=args.workers, report_path=out / 'pose_extraction_report.csv')
    x, meta, exclusions = load_study_windows(manifest, settings.pose_dir,
        frames=settings.frames, stride=settings.stride, target_fps=settings.target_fps,
        expected_frame_number_base=settings.frame_number_base)
    meta.to_csv(out / 'retained_window_index.csv', index=False)
    if isinstance(exclusions, pd.DataFrame): exclusions.to_csv(out / 'exclusions.csv', index=False)
    else: write_json(out / 'exclusions.json', exclusions)
    summary = meta.groupby('label').agg(windows=('sequence_id','size'),sequences=('sequence_id','nunique'),sources=('source_id','nunique'))
    summary.to_csv(out / 'retained_by_label.csv')
    write_json(out / 'summary.json', {'windows':len(x),'sequences':meta.sequence_id.nunique(),
        'sources':meta.source_id.nunique(), 'normal_sources':meta.loc[meta.label.eq('normal'),'source_id'].nunique(),
        'annotation_sequences':len(manifest),'annotation_sources':manifest.source_id.nunique(),
        'cohort_status':'QC before source-fold assignment; no performance-based exclusions'})
    print(summary.to_string(), flush=True)


if __name__ == '__main__':
    main()
