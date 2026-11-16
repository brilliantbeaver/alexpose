"""Full-sequence 10 Hz observation study using shared Ambient pose inference.

The sampling clock is nominal source FPS. Native frames and measured decoder
times remain explicit. No interpolation, copied videos, or hidden pose repair is
performed. Each source is decoded once and resumable caches are identity checked.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import importlib.metadata
import json
from pathlib import Path
import time

import numpy as np
import pandas as pd

from .data import JOINT_IDS, assign_splits, build_inventory, save_inventory, sha256_file
from .extended_data import gate_person
from .pose import normalize_window
from .settings import configure

FRAMES, CONTEXT, STRIDE, TARGET_HZ = 24, 16, 8, 10.0
ADAPTER_SHA256 = sha256_file(__file__)


def write_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


def sampled_spans(frame_numbers, fps, hz=TARGET_HZ):
    """Nearest native frames on a fixed nominal grid, separately across gaps."""
    frames = np.asarray(frame_numbers, dtype=np.int64)
    if frames.ndim != 1 or not len(frames) or np.any(frames < 1) or np.any(np.diff(frames) <= 0):
        raise ValueError('Need sorted unique positive one-based annotation frames')
    if not np.isfinite(fps) or not np.isfinite(hz) or fps < hz or hz <= 0:
        raise ValueError('Source FPS must be finite and at least the target sampling rate')
    chunks = np.split(frames, np.flatnonzero(np.diff(frames) != 1) + 1)
    output = []
    for chunk in chunks:
        steps = np.arange(int(np.floor((chunk[-1] - chunk[0]) * hz / fps + 1e-9)) + 1)
        selected = chunk[0] + np.floor(steps * fps / hz + .5).astype(np.int64)
        if np.any(np.diff(selected) <= 0) or selected[-1] > chunk[-1]:
            raise ValueError('Invalid fixed-time native-frame sampling')
        output.append(selected)
    return output


def supported_window(xy, confidence, context=CONTEXT, minimum=.7):
    """Normalize on the prefix, and score only targets with prefix support."""
    x, valid = normalize_window(xy, confidence, context_frames=context)
    valid[context:] &= valid[:context].any(axis=0)[None, :]
    if valid[:context].mean() < minimum:
        return None, None, 'prefix_quality_below_0.7'
    if valid[context:].mean() < minimum:
        return None, None, 'future_quality_below_0.7'
    return np.where(valid[..., None], x, 0).astype(np.float32), valid, ''


def window_identity(source, numbers):
    return hashlib.sha256(json.dumps([str(source), list(map(int, numbers))], separators=(',', ':')).encode()).hexdigest()


def strict_gate(xy, confidence, bbox, info, width, height):
    conf, reason = gate_person(xy, confidence, bbox, info, width, height)
    if reason.startswith('unresolved'):
        conf = np.zeros_like(conf)
    return conf, reason


def _load_source_cache(path, expected):
    with np.load(path, allow_pickle=False) as archive:
        data = {key: archive[key].copy() for key in archive.files}
    identity = json.loads(str(data['provenance_json']))
    if identity != expected:
        raise ValueError(f'Cached source identity mismatch: {path}')
    n = len(data['frame_numbers'])
    if data['xy'].shape != (n, 33, 2) or data['confidence'].shape != (n, 33):
        raise ValueError(f'Invalid source pose shape: {path}')
    if list(map(int, data['frame_numbers'])) != expected['requested_frame_numbers']:
        raise ValueError(f'Cached native frame mismatch: {path}')
    if not np.array_equal(data['decoded_indexes'], data['frame_numbers'] - 1):
        raise ValueError(f'Cached frame origin mismatch: {path}')
    return data


def _extract_source(video, cache, expected, model):
    import cv2
    from ambient.pose.keypoint_extractor import SequenceKeypointExtractor
    if cache.exists():
        return _load_source_cache(cache, expected), 0, True
    requested = np.asarray(expected['requested_frame_numbers'], dtype=np.int64)
    xy = np.full((len(requested), 33, 2), np.nan, dtype=np.float32)
    confidence = np.zeros((len(requested), 33), dtype=np.float32)
    pts = np.full(len(requested), np.nan, dtype=np.float64)
    position = {int(number): i for i, number in enumerate(requested)}
    cap = cv2.VideoCapture(str(video))
    extractor = SequenceKeypointExtractor()
    model_calls = 0
    try:
        if not cap.isOpened():
            raise ValueError(f'Cannot decode source {video}')
        for decoded_index in range(int(requested[-1])):
            ok, frame = cap.read()
            if not ok:
                break
            number = decoded_index + 1
            if number not in position:
                continue
            i = position[number]
            pts[i] = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.
            pose = extractor.extract_from_image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), str(model))
            model_calls += 1
            if pose is not None:
                for point in pose.keypoints:
                    if 0 <= point.id < 33:
                        xy[i, point.id] = [point.x, point.y]
                        confidence[i, point.id] = point.confidence
    finally:
        cap.release()
        extractor.cleanup()
    data = dict(xy=xy, confidence=confidence, frame_numbers=requested,
                decoded_indexes=requested-1, timestamp_seconds=pts,
                nominal_seconds=(requested-1)/expected['fps'],
                provenance_json=np.array(json.dumps(expected, sort_keys=True)))
    temporary = cache.with_suffix('.tmp.npz')
    np.savez_compressed(temporary, **data)
    temporary.replace(cache)
    return data, model_calls, False


def prepare(output, max_sources=None, source_ids=None):
    """Create a new complete cohort or resume identity-matched source caches."""
    import cv2
    import ambient.pose.keypoint_extractor as extractor_module
    from ambient.pose import model_management, mediapipe_singleton
    project, repo = configure()
    output = Path(output).resolve()
    if (output/'preparation.json').exists():
        raise FileExistsError('Completed cohorts are immutable. Use a new output identity.')
    output.mkdir(parents=True, exist_ok=True)
    (output/'source_poses').mkdir(exist_ok=True)
    (output/'sequence_poses').mkdir(exist_ok=True)
    table, inventory = build_inventory(repo)
    table = assign_splits(table, seed=27)
    save_inventory(table, inventory, output)
    source_exclusion_path = project/'docs/source_exclusions.json'
    exclusions = json.loads(source_exclusion_path.read_text())
    selected = table[table.eligible & ~table.source_id.isin(exclusions['excluded_sources'])]
    sources = sorted(selected.source_id.unique())
    if source_ids is not None:
        unknown = set(source_ids)-set(sources)
        if unknown:
            raise ValueError(f'Ineligible or unknown sources: {unknown}')
        sources = sorted(source_ids)
    if max_sources is not None:
        sources = sources[:max_sources]
    selected = selected[selected.source_id.isin(sources)]
    selected.to_csv(output/'selected_sequences.csv', index=False)
    model = repo/'data/models/pose_landmarker_lite.task'
    common = dict(schema='restart-full10hz-source-v1', adapter_sha256=ADAPTER_SHA256,
        pose_model_sha256=sha256_file(model), extractor_sha256=sha256_file(extractor_module.__file__),
        extractor_support_sha256={str(Path(m.__file__).relative_to(repo)):sha256_file(m.__file__) for m in [model_management,mediapipe_singleton]},
        gate_sha256=sha256_file(Path(__file__).with_name('extended_data.py')),
        normalization_sha256=sha256_file(Path(__file__).with_name('pose.py')),
        source_exclusions_sha256=sha256_file(source_exclusion_path), pose_format='BLAZEPOSE_33',
        coordinate_units='pixels', sampling_hz=TARGET_HZ,
        sampling='nearest_native_frame_to_nominal_grid_with_contiguous_span_anchor',
        frame_origin=1, decoded_index_origin=0,
        person_selection='Ambient full-image first detection; later strict annotation pelvis gate',
        versions={name:importlib.metadata.version(name) for name in ['mediapipe','opencv-contrib-python','numpy','pandas']})
    request_identity = dict(common, selected_sequence_sha256=sha256_file(output/'selected_sequences.csv'),
        source_ids=sources, frames=FRAMES, context_frames=CONTEXT, stride=STRIDE,
        minimum_prefix_and_future_valid_fraction=.7, confidence_threshold=.5,
        unresolved_geometry_policy='invalidate_all_joints_for_frame')
    request_file = output/'request.json'
    if request_file.exists() and json.loads(request_file.read_text()) != request_identity:
        raise ValueError('Resume requested with changed data, code, or protocol identity')
    write_json(request_file, request_identity)
    started = time.monotonic()
    windows, validities, metadata, attrition, gates, source_log, duplicates = [], [], [], [], [], [], []
    seen_windows = {}
    total_calls = 0
    for source_i, source in enumerate(sources):
        source_started = time.monotonic()
        group = selected[selected.source_id==source].sort_values('sequence_id')
        video = repo/group.iloc[0].video_path
        video_hash = sha256_file(video)
        recorded_hash = str(group.iloc[0].video_sha256_recorded)
        if recorded_hash and video_hash != recorded_hash:
            raise ValueError(f'Current video differs from media manifest: {source}')
        cap = cv2.VideoCapture(str(video))
        fps = float(cap.get(cv2.CAP_PROP_FPS))
        width, height = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        cap.release()
        sequence_rows = {}
        requested = set()
        for row in group.itertuples():
            rows = pd.read_csv(repo/row.csv_path,keep_default_na=False).sort_values('frame_num')
            spans = sampled_spans(rows.frame_num.to_numpy(int), fps)
            sequence_rows[row.sequence_id]=(row,rows,spans)
            requested.update(int(f) for span in spans for f in span)
        expected = dict(common, source_id=source,video_path=str(video.relative_to(repo)),video_sha256=video_hash,
            fps=fps,image_dimensions=[width,height],requested_frame_numbers=sorted(requested),
            annotation_sha256={r.sequence_id:r.csv_sha256 for r in group.itertuples()})
        cache = output/'source_poses'/f'{source}.npz'
        raw, calls, reused = _extract_source(video, cache, expected, model)
        total_calls += calls
        raw_lookup = {int(number):i for i,number in enumerate(raw['frame_numbers'])}
        for sequence,(row,rows,spans) in sequence_rows.items():
            rows_by_number = rows.set_index('frame_num')
            numbers=np.concatenate(spans)
            raw_indices=np.array([raw_lookup[int(number)] for number in numbers])
            xy=raw['xy'][raw_indices].copy()
            confidence=raw['confidence'][raw_indices].copy()
            timestamps=raw['timestamp_seconds'][raw_indices].copy()
            reasons=[]
            for i,number in enumerate(numbers):
                annotation=rows_by_number.loc[int(number)]
                confidence[i],reason=strict_gate(xy[i],confidence[i],annotation.bbox,annotation.vid_info,width,height)
                reasons.append(reason)
                gates.append(dict(source_id=source,sequence_id=sequence,frame_number=int(number),reason=reason))
            sequence_identity=dict(source_cache_sha256=sha256_file(cache),source_id=source,sequence_id=sequence,
                csv_sha256=row.csv_sha256,adapter_sha256=ADAPTER_SHA256,source_exclusions_sha256=common['source_exclusions_sha256'],
                raw_source_provenance=expected)
            np.savez_compressed(output/'sequence_poses'/f'{sequence}.npz',xy=xy,confidence=confidence,
                raw_confidence=raw['confidence'][raw_indices],frame_numbers=numbers,decoded_indexes=numbers-1,
                timestamp_seconds=timestamps,nominal_seconds=(numbers-1)/fps,span_lengths=np.array(list(map(len,spans))),
                fps=np.array(fps),image_dimensions=np.array([width,height]),gate_reason=np.array(reasons),
                sequence_id=np.array(sequence),source_id=np.array(source),label=np.array(row.label),
                provenance_json=np.array(json.dumps(sequence_identity,sort_keys=True)))
            accepted=0
            attempted=0
            rejected=Counter()
            offset=0
            for span in spans:
                for local_start in range(0,len(span)-FRAMES+1,STRIDE):
                    attempted+=1
                    start=offset+local_start
                    end=start+FRAMES
                    selected_numbers=numbers[start:end]
                    selected_times=timestamps[start:end]
                    if not np.isfinite(selected_times).all() or np.any(np.diff(selected_times)<=0):
                        rejected['missing_or_nonmonotonic_decoded_timestamps']+=1
                        continue
                    try:
                        x,valid,reason=supported_window(xy[start:end],confidence[start:end])
                    except ValueError:
                        rejected['missing_prefix_torso_scale']+=1
                        continue
                    if reason:
                        rejected[reason]+=1
                        continue
                    identifier=window_identity(source,selected_numbers)
                    if identifier in seen_windows:
                        rejected['duplicate_source_frame_tuple']+=1
                        duplicates.append(dict(window_id=identifier,retained_sequence_id=seen_windows[identifier],duplicate_sequence_id=sequence,source_id=source))
                        continue
                    seen_windows[identifier]=sequence
                    accepted+=1
                    windows.append(x);validities.append(valid)
                    nominal=(selected_numbers-1)/fps
                    relative_targets=selected_times[CONTEXT:]-selected_times[CONTEXT-1]
                    metadata.append(dict(window_id=identifier,source_id=source,sequence_id=sequence,label=row.label,
                        split=row.split,original_split=row.split,first_frame=int(selected_numbers[0]),last_frame=int(selected_numbers[-1]),
                        frame_numbers=json.dumps(selected_numbers.tolist()),timestamp_seconds=json.dumps(selected_times.tolist()),
                        nominal_seconds=json.dumps(nominal.tolist()),future_horizon_seconds=json.dumps(relative_targets.tolist()),
                        sample_fps=TARGET_HZ,fps=fps,width=width,height=height,view=row.views,
                        prefix_valid_fraction=float(valid[:CONTEXT].mean()),future_valid_fraction=float(valid[CONTEXT:].mean()),
                        valid_fraction=float(valid.mean()),prefix_mean_confidence=float(confidence[start:start+CONTEXT,JOINT_IDS].mean()),
                        future_mean_confidence=float(confidence[start+CONTEXT:end,JOINT_IDS].mean()),
                        duration_seconds=float(selected_times[-1]-selected_times[0]),
                        final_horizon_seconds=float(relative_targets[-1]),native_frame_interval_min=int(np.diff(selected_numbers).min()),
                        native_frame_interval_max=int(np.diff(selected_numbers).max()),
                        maximum_grid_error_seconds=float(np.max(np.abs((selected_times-selected_times[0])-np.arange(FRAMES)/TARGET_HZ))),
                        unresolved_person_fraction=sum(r.startswith('unresolved') for r in reasons[start:end])/FRAMES))
                offset+=len(span)
            if not attempted:
                rejected['fewer_than_24_sampled_frames_in_contiguous_span']=1
            attrition.append(dict(sequence_id=sequence,source_id=source,label=row.label,split=row.split,
                annotated_frames=len(rows),sampled_frames=len(numbers),candidate_windows=attempted,retained_windows=accepted,
                rejection_counts=json.dumps(dict(rejected),sort_keys=True)))
        source_log.append(dict(source_id=source,sequences=len(group),requested_frames=len(requested),actual_model_calls=calls,
            reused_source_cache=reused,source_cache_sha256=sha256_file(cache),video_sha256=video_hash,
            seconds=time.monotonic()-source_started,retained_windows_so_far=len(windows)))
        write_json(output/'source_progress.json',dict(sources=source_log,completed_sources=len(source_log),total_sources=len(sources),
            elapsed_seconds=time.monotonic()-started,actual_model_calls=total_calls))
        print(json.dumps(source_log[-1]),flush=True)
    x=np.stack(windows) if windows else np.empty((0,FRAMES,len(JOINT_IDS),2),dtype=np.float32)
    valid=np.stack(validities) if validities else np.empty((0,FRAMES,len(JOINT_IDS)),dtype=bool)
    meta=pd.DataFrame(metadata)
    np.savez_compressed(output/'windows.npz',x=x,valid=valid)
    meta.to_csv(output/'windows.csv',index=False)
    pd.DataFrame(attrition).to_csv(output/'attrition.csv',index=False)
    pd.DataFrame(gates).to_csv(output/'person_gate_frames.csv',index=False)
    pd.DataFrame(duplicates,columns=['window_id','retained_sequence_id','duplicate_sequence_id','source_id']).to_csv(output/'duplicate_windows.csv',index=False)
    excluded=table[~table.eligible | table.source_id.isin(exclusions['excluded_sources'])].copy()
    excluded.loc[excluded.source_id.isin(exclusions['excluded_sources']),'exclusion_reason']='manual_source_identity_failure'
    excluded.to_csv(output/'excluded_sequences.csv',index=False)
    report=dict(identity='restart-full10hz-v1',mode='all-eligible-sequences-full-duration' if max_sources is None and source_ids is None else 'bounded-real-source-pilot',
        frames=FRAMES,context_frames=CONTEXT,stride=STRIDE,sample_fps=TARGET_HZ,n_windows=len(x),
        n_sequences=int(meta.sequence_id.nunique()) if len(meta) else 0,n_sources=int(meta.source_id.nunique()) if len(meta) else 0,
        selected_sequences=len(selected),selected_sources=len(sources),actual_model_calls_this_invocation=total_calls,
        source_cache_reuses=sum(s['reused_source_cache'] for s in source_log),seconds=time.monotonic()-started,
        data_sha256=sha256_file(output/'windows.npz'),metadata_sha256=sha256_file(output/'windows.csv'),
        attrition_sha256=sha256_file(output/'attrition.csv'),request_sha256=sha256_file(request_file),
        source_progress_sha256=sha256_file(output/'source_progress.json'),source_exclusions_sha256=common['source_exclusions_sha256'],
        adapter_sha256=ADAPTER_SHA256,by_label=meta.groupby('label').agg(windows=('window_id','count'),sequences=('sequence_id','nunique'),sources=('source_id','nunique')).reset_index().to_dict('records') if len(meta) else [],
        limitations=['Source IDs do not establish participant identity or diagnosis.','Existing source-disjoint test results have been seen; new analyses are exploratory.',
            'Sampling uses nominal FPS rather than presentation-time resampling; actual timestamps and grid errors are retained.',
            'Full-image first detection and annotation-box pelvis containment do not guarantee subject identity.',
            'Raw poses are native 2D image estimates; normalized outputs describe articulation rather than physical displacement.'])
    write_json(output/'preparation.json',report)
    print(json.dumps(report,indent=2),flush=True)
    return report


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',required=True)
    parser.add_argument('--max-sources',type=int)
    parser.add_argument('--source',action='append')
    arguments=parser.parse_args()
    prepare(arguments.output,arguments.max_sources,arguments.source)
