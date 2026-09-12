"""Expanded alignment/pose/cache stages using the original FI measurement routines."""
from pathlib import Path
import shutil

import numpy as np
import pandas as pd

from ..future_innovation.fi_contracts import read_json, write_json, write_once_json, sha256_file, save_npz, stable_key, runtime_versions
from ..future_innovation.fi_cache_reuse import read_parent, checked_file
from ..future_innovation.fi_cohort import build_candidates
from ..future_innovation.fi_feature_cache import load_window
from ..future_innovation.fi_nuisance_features import context_nuisance
from ..future_innovation.fi_token_regions import pool_context, pool_target, region_masks, union_box
from ..future_innovation.fi_video_pose import CropGeometry, decode_exact_window, extract_history, alignment_sheet
from ..future_innovation.fi_validity_audits import future_pixel_leakage_test
from ..future_innovation.fi_readiness import verify_readiness_tables
from .fi_scaling_cohort import read_study, PROTOCOL
from .fi_scaling_readiness import development_media, require_development_media, original_inputs


def seal(root, path, files, **metadata):
    write_once_json(Path(root) / path, dict(**metadata, artifacts={str(Path(p).relative_to(root)): sha256_file(p) for p in files}))


def verify_seal(root, path):
    receipt = read_json(Path(root) / path)
    if not receipt.get('artifacts'):
        raise ValueError('Empty stage receipt')
    for name, digest in receipt['artifacts'].items():
        checked_file(root, name, digest)
    return receipt


def verify_prepared_inputs(root):
    """Before encoding, check decoded frames/poses behind each per-window receipt."""
    receipt=verify_seal(root,'data/cohort-complete.json')
    for name in receipt['artifacts']:
        if name.startswith('data/poses/') and name.endswith('.json'):
            verify_seal(root,name)
    return receipt


def verify_data_identity(root):
    root=Path(root)
    frozen=read_json(root/'config/teacher-contract.json')
    current=read_json(root/'data/config/teacher-contract.json')
    # Runtime locations may differ; scientific teacher identity may not.
    for key,value in frozen.items():
        if key not in ('repository_path','checkpoint_path') and current.get(key)!=value:
            raise ValueError('Expanded teacher differs from the frozen parent teacher')
    run=read_json(root/'data/config/run-contract.json')
    for name,digest in run['config_sha256'].items():
        checked_file(root/'data','config/'+name,digest)


def initialize_data(root, parent, annotations, video_root, pose_model, teacher_root, checkpoint):
    """Bind supplied runtime locations to original model and annotation digests."""
    study, parent = read_study(root, parent, require_software=True)
    teacher = original_inputs(parent, annotations, pose_model, checkpoint)
    if not Path(video_root).is_dir():
        raise FileNotFoundError(f'Full-source video directory unavailable: {video_root}')
    teacher.update(repository_path=str(Path(teacher_root).resolve()), checkpoint_path=str(Path(checkpoint).resolve()))
    data = Path(root) / 'data'
    for directory in ('config','manifests','boxes','frames','poses','qc/alignment-overlays','teacher-cache','logs'):
        (data / directory).mkdir(parents=True, exist_ok=True)
    paths = dict(sequence_manifest=str((Path(root)/'config/full-sequences.csv').resolve()),
                 video_manifest=str((Path(root)/'config/full-videos.csv').resolve()),
                 annotations=[str(Path(p).resolve()) for p in annotations], youtube_dir=str(Path(video_root).resolve()), video_roots=[])
    write_once_json(data / 'config/teacher-contract.json', teacher)
    write_once_json(data / 'config/runtime-contract.json', runtime_versions())
    run = dict(protocol=PROTOCOL, cohort_size=None, minimum_sources=25, synthetic=study['synthetic'],
               source_cap=None, config_sha256={n:sha256_file(data/'config'/n) for n in ('teacher-contract.json','runtime-contract.json')},
               code_sha256=sha256_file(Path(root)/'config/software.json'),
               inputs_sha256={str(Path(p).resolve()):sha256_file(p) for p in [paths['sequence_manifest'],paths['video_manifest'],*annotations,pose_model]},
               input_paths=paths, pose_model=str(Path(pose_model).resolve()))
    write_once_json(data / 'config/run-contract.json', run)
    return data, run


def extract_one(row, data, pose_model):
    """Same direct-v3 inherited prefix processing; imputation never creates motion."""
    video, fps = decode_exact_window(row['video_path'], row['source_first_frame'])
    records = read_json(row['annotation_path'])
    with np.load(row['box_path'], allow_pickle=False) as payload:
        boxes = payload['source_boxes']
    geometry = CropGeometry(*video.shape[1:3])
    model_boxes, retention = geometry.boxes(boxes)
    for t in [*range(16), 19]:
        region_masks(union_box(model_boxes[2*t:2*t+2]), allow_empty_background=True)
    raw, history, scale = extract_history(video, records, boxes, fps, pose_model)
    if not np.any((history[1:,:,3] > 0) & (history[:-1,:,3] > 0)):
        raise ValueError('No valid observed joint transition')
    window = row['window_id']
    pose_path, frame_path = data/f'poses/{window}.npz', data/f'frames/{window}.npz'
    model_path, overlay = data/f'boxes/{window}-model.npz', data/f'qc/alignment-overlays/{window}.jpg'
    save_npz(pose_path, raw=raw, skeleton=history, body_scale=np.array(scale),
             source_frames=np.arange(row['source_first_frame'],row['source_first_frame']+32))
    save_npz(frame_path, video=video)
    save_npz(model_path, model_boxes=model_boxes, crop_retention=retention)
    alignment_sheet(video, raw, boxes, row['source_first_frame'], overlay)
    return dict(**row, decoded_fps=fps, source_width=video.shape[2], source_height=video.shape[1],
                horizon_seconds=8/fps, pose_path=str(pose_path), frame_path=str(frame_path),
                model_box_path=str(model_path), overlay_path=str(overlay),
                context_pose_coverage=float(history[...,3].mean()),
                minimum_person_crop_retention=float(retention[np.r_[0:32,38:40]].min()),
                eligibility_reason='required input checks passed', evidence_origin='new_processing')


def prepare(root, parent=None, **runtime):
    root = Path(root)
    _, parent = read_study(root, parent, require_software=True)
    if (root/'data/cohort-complete.json').exists():
        verify_prepared_inputs(root)
        return
    data, run = initialize_data(root, parent, **runtime)
    roster = pd.read_csv(root/'config/source-reservation.csv', dtype={'video_id': str})
    media = development_media(roster, run['input_paths']['video_manifest'], run['input_paths']['youtube_dir'])
    # Retryable operational inventory, not a scientific exclusion list. No
    # candidate contract may be sealed until all development media are available.
    media.to_csv(data/'logs/development-media.csv', index=False)
    require_development_media(media)
    build_candidates(data, **run['input_paths'])
    candidates = pd.read_csv(data/'manifests/candidates.csv')
    old_candidates = pd.read_csv(parent/'manifests/candidates.csv')
    # Expanding size must not silently change old windows or lose available inputs.
    compared = old_candidates.merge(candidates, on='sequence_id', suffixes=('_old','_new'), how='left')
    for column in ('window_id','video_id','source_first_frame','source_last_frame'):
        if not (compared[column+'_old'] == compared[column+'_new']).all():
            raise ValueError('Original candidate unavailable or alignment changed; repair inputs before continuing')
    _, old, _, _ = read_parent(parent)
    old_rows = {r['window_id']:r for r in old.to_dict('records')}
    dev = set(roster.loc[roster.role == 'development','video_id'])
    eligible, failures, receipts = [], [], []
    selected = candidates.loc[candidates.video_id.isin(dev)].to_dict('records')
    for position, row in enumerate(selected, 1):
        if position % 25 == 1 or position == len(selected):
            print(f'Pose preparation: checking window {position}/{len(selected)}; {len(eligible)} eligible so far', flush=True)
        receipt_path = data/f"poses/{row['window_id']}.json"
        if receipt_path.exists():
            item = verify_seal(root, str(receipt_path.relative_to(root)))
        elif row['window_id'] in old_rows:
            item = dict(status='eligible', row={**old_rows[row['window_id']], 'evidence_origin':'parent_cache'},
                        artifacts={'config/parent-snapshot.json':sha256_file(root/'config/parent-snapshot.json')})
            write_once_json(receipt_path, item)
        else:
            try:
                result = extract_one(row, data, run['pose_model'])
                paths = [Path(result[k]) for k in ('pose_path','frame_path','model_box_path','box_path','annotation_path','overlay_path')]
                seal(root, str(receipt_path.relative_to(root)), paths, status='eligible', row=result)
            except ValueError as error:
                # Deterministic decoding/pose eligibility failures are retained. Missing
                # resources and unexpected implementation errors propagate and block.
                seal(root, str(receipt_path.relative_to(root)), [Path(row['box_path']),Path(row['annotation_path'])],
                     status='ineligible', window_id=row['window_id'], sequence_id=row['sequence_id'], reason=str(error))
            item = read_json(receipt_path)
        receipts.append(receipt_path)
        if item['status'] == 'eligible': eligible.append(item['row'])
        else: failures.append({k:v for k,v in item.items() if k!='artifacts'})
    cohort = pd.DataFrame(eligible).drop(columns=['outer_fold'],errors='ignore').merge(
        roster[['video_id','source_group','outer_fold']], on='video_id', validate='many_to_one')
    if not len(cohort) or not cohort.window_id.is_unique:
        raise ValueError('Empty or duplicated expanded cohort')
    cohort.to_csv(data/'manifests/development-windows.csv',index=False)
    write_json(data/'manifests/pose-exclusions.json',failures)
    new = cohort[cohort.evidence_origin == 'new_processing']
    audits = new.sort_values('window_id',key=lambda c:c.map(lambda w:stable_key('scaling-audit',w))).head(3)
    if len(audits) != 3:
        raise ValueError('Expanded study requires three newly processed audit windows')
    audits[['window_id']].to_csv(data/'manifests/audit-windows.csv',index=False)
    seal(root,'data/cohort-complete.json', [*receipts,data/'manifests/development-windows.csv',
         data/'manifests/pose-exclusions.json',data/'manifests/audit-windows.csv',data/'config/candidates-contract.json',
         data/'manifests/candidates.csv',data/'manifests/exclusions.csv',data/'config/run-contract.json',
         data/'config/teacher-contract.json',data/'config/runtime-contract.json'],
         eligible_windows=len(cohort), eligible_sources=int(cohort.video_id.nunique()),
         confirmation_processed=False, failed_pose_windows=len(failures))


def encode_features(adapter, row, projection):
    video, boxes, model_boxes, skeleton = load_window(row)
    geometry_error = adapter.verify_geometry(video)
    context = pool_context(adapter.encode_past_context(video),model_boxes,allow_empty_background=True)
    person, background = pool_target(adapter.encode_full_target(video),model_boxes,allow_empty_background=True)
    nuisance, names, matching = context_nuisance(video,boxes,skeleton,row,allow_empty_background=True)
    support = np.mean([region_masks(union_box(model_boxes[t:t+2]),allow_empty_background=True)[1].mean() for t in range(0,32,2)])
    return dict(baseline=np.r_[context,nuisance,support].astype(np.float32),skeleton=skeleton,
                person=(person@projection).astype(np.float32),background=(background@projection).astype(np.float32),matching=matching), \
           [*names,'observed_background_token_fraction'], float(geometry_error)


def cache(root, parent=None, device='cuda', adapter=None):
    root = Path(root)
    _, parent = read_study(root,parent,require_software=True)
    if (root/'data/cache-complete.json').exists():
        load_expanded(root,parent,require_audit=False)
        return
    verify_prepared_inputs(root)
    verify_data_identity(root)
    data = root/'data'
    cohort = pd.read_csv(data/'manifests/development-windows.csv')
    projection = np.load(root/'config/projection-256.npy',allow_pickle=False)
    schema = read_json(root/'config/nuisance-schema.json')
    _, old, old_arrays, _ = read_parent(parent)
    old_index = dict(zip(old.window_id, range(len(old))))
    parent_index = pd.read_csv(parent/'manifests/cache-index.csv').set_index('window_id')
    binding = stable_key(sha256_file(root/'config/study.json'),sha256_file(root/'data/cohort-complete.json'))
    entries, files = [], []
    for position, row in enumerate(cohort.to_dict('records'), 1):
        if position % 25 == 1 or position == len(cohort):
            print(f'Teacher cache: checking window {position}/{len(cohort)}; {len(entries)} entries ready', flush=True)
        window = row['window_id']
        if row['evidence_origin'] == 'parent_cache':
            original = parent_index.loc[window]
            entries.append(dict(window_id=window,origin='parent_cache',path=f'teacher-cache/{window}.npz',sha256=original.sha256,binding=original.binding))
            continue
        path, receipt = data/f'teacher-cache/{window}.npz', data/f'teacher-cache/{window}.json'
        if receipt.exists():
            item = verify_seal(root,str(receipt.relative_to(root)))
            if item['binding'] != binding or item['window_id'] != window:
                raise ValueError('Interrupted cache identity changed')
        else:
            if adapter is None:
                from ..future_innovation.fi_vjepa_adapter import FrozenVJEPAAdapter
                adapter = FrozenVJEPAAdapter.from_run(data,device)
            values,names,geometry = encode_features(adapter,row,projection)
            if names != schema['columns']:
                raise ValueError('Expanded nuisance schema changed')
            save_npz(path,**values,window_id=np.array(window),binding=np.array(binding))
            seal(root,str(receipt.relative_to(root)),[path],window_id=window,binding=binding,geometry_max_abs=geometry)
        files.extend([path,receipt])
        entries.append(dict(window_id=window,origin='new_encoding',path=str(path.relative_to(root)),sha256=sha256_file(path),binding=binding))
    pd.DataFrame(entries).to_csv(data/'manifests/cache-index.csv',index=False)
    seal(root,'data/cache-complete.json',[*files,data/'manifests/cache-index.csv'],binding=binding,
         reused_windows=sum(e['origin']=='parent_cache' for e in entries),new_windows=sum(e['origin']=='new_encoding' for e in entries))
    load_expanded(root,parent,require_audit=False)


def load_expanded(root,parent=None,*,require_audit=True):
    root=Path(root)
    _,parent=read_study(root,parent)
    verify_seal(root,'data/cohort-complete.json')
    contract=verify_seal(root,'data/cache-complete.json')
    verify_data_identity(root)
    binding=stable_key(sha256_file(root/'config/study.json'),sha256_file(root/'data/cohort-complete.json'))
    if contract['binding']!=binding: raise ValueError('Expanded cache binding differs from frozen cohort/study')
    cohort=pd.read_csv(root/'data/manifests/development-windows.csv')
    roster=pd.read_csv(root/'config/source-reservation.csv').set_index('video_id')
    if not cohort.window_id.is_unique or any(roster.loc[v,'role']!='development' for v in cohort.video_id):
        raise ValueError('Invalid development/confirmation cohort')
    for key in ('outer_fold','source_group'):
        if not np.array_equal(cohort[key],roster.loc[cohort.video_id,key]):
            raise ValueError('Frozen source grouping changed')
    index=pd.read_csv(root/'data/manifests/cache-index.csv')
    if not index.window_id.is_unique or set(index.window_id)!=set(cohort.window_id):
        raise ValueError('Missing/duplicate expanded cache')
    _,old,old_arrays,_=read_parent(parent)
    old_positions=dict(zip(old.window_id,range(len(old))))
    old_rows=old.set_index('window_id')
    arrays={k:[] for k in old_arrays}
    index=index.set_index('window_id')
    width=read_json(root/'config/nuisance-schema.json')['context_embedding_columns']+len(read_json(root/'config/nuisance-schema.json')['columns'])
    shapes={'baseline':(width,),'person':(256,),'background':(256,),'skeleton':(32,33,4),'matching':(11,)}
    for window in cohort.window_id:
        item=index.loc[window]
        base=parent if item.origin=='parent_cache' else root
        path=checked_file(base,item.path,item.sha256)
        if item.origin not in ('parent_cache','new_encoding') or (item.origin=='new_encoding' and item.binding!=contract['binding']):
            raise ValueError('Wrong expanded cache origin/binding')
        record=cohort.set_index('window_id').loc[window]
        expected_origin='parent_cache' if record.evidence_origin=='parent_cache' else 'new_encoding'
        if item.origin!=expected_origin:
            raise ValueError('Cohort/cache origin mismatch')
        if item.origin=='parent_cache':
            if window not in old_positions or any(record[k]!=old_rows.loc[window,k] for k in ('video_id','sequence_id','source_first_frame','source_last_frame','decoded_fps')):
                raise ValueError('Reused cache window/source metadata changed')
        with np.load(path,allow_pickle=False) as payload:
            if set(payload.files)!={*arrays,'window_id','binding'} or str(payload['window_id'])!=window or str(payload['binding'])!=item.binding:
                raise ValueError('Wrong expanded cache payload identity')
            for k in arrays:
                v=payload[k]
                if v.shape!=shapes[k] or not np.isfinite(v).all(): raise ValueError('Wrong expanded cache shape/value')
                if item.origin=='parent_cache' and (window not in old_positions or not np.array_equal(v,old_arrays[k][old_positions[window]])):
                    raise ValueError('Parent cache relabeled')
                arrays[k].append(v)
    arrays={k:np.stack(v) for k,v in arrays.items()}
    if require_audit:
        receipt=verify_seal(root,'data/audit-complete.json')
        stable,leak=verify_readiness_tables(root/'data',pd.read_csv(root/'data/manifests/audit-windows.csv'),3)
        if receipt['cache_sha256']!=sha256_file(root/'data/cache-complete.json') or not stable.passed.all() or not leak.passed.all():
            raise ValueError('Expanded teacher audit failed or changed')
    return cohort,arrays


def audit_teacher(root,parent=None,device='cuda',adapter=None):
    root=Path(root)
    read_study(root,parent,require_software=True)
    if (root/'data/audit-complete.json').exists():
        load_expanded(root,parent)
        return
    cohort,arrays=load_expanded(root,parent,require_audit=False)
    verify_prepared_inputs(root)
    if adapter is None:
        from ..future_innovation.fi_vjepa_adapter import FrozenVJEPAAdapter
        adapter=FrozenVJEPAAdapter.from_run(root/'data',device)
    projection=np.load(root/'config/projection-256.npy',allow_pickle=False)
    planned=pd.read_csv(root/'data/manifests/audit-windows.csv')
    rows=cohort.set_index('window_id')
    stability,leakage=[],[]
    for window in planned.window_id:
        video,_,boxes,_=load_window(rows.loc[window])
        leak=future_pixel_leakage_test(adapter,video,np.random.default_rng(int(stable_key('leakage',window)[:16],16)))
        original=adapter.encode_full_target(video)
        repeat=adapter.encode_full_target(video)
        target=pool_target(original,boxes,allow_empty_background=True)[0]@projection
        i=int(np.flatnonzero(cohort.window_id.to_numpy()==window)[0])
        error=float(np.max(np.abs(original-repeat)))
        cache_error=float(np.max(np.abs(target-arrays['person'][i])))
        stability.append(dict(window_id=window,context_max_abs=leak['repeat_max_abs'],target_max_abs=error,cache_max_abs=cache_error,
                              passed=bool(leak['stable'] and error<=1e-6 and cache_error<=1e-6)))
        leakage.append(dict(window_id=window,**leak))
    for name,values in (('teacher-stability',stability),('causal-leakage',leakage)):
        pd.DataFrame(values).to_csv(root/f'data/qc/{name}.csv',index=False)
    stable,leak=verify_readiness_tables(root/'data',planned,3)
    if not stable.passed.all() or not leak.passed.all():
        raise ValueError('New teacher repeatability/prefix-isolation audit failed')
    seal(root,'data/audit-complete.json',[root/'data/qc/teacher-stability.csv',root/'data/qc/causal-leakage.csv'],
         cache_sha256=sha256_file(root/'data/cache-complete.json'),new_teacher_evidence=True)
