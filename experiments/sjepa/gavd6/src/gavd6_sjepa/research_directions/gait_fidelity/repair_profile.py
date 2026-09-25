"""Measured preparation admission and metadata-only expansion readiness.

The timing sample contains already-open development windows only. It never uses
protected confirmation references, outcome scores, or assumptions that an absent
exposure record proves independence. Auxiliary datasets remain outside this fixed
experiment unless a separate scientifically reviewed protocol is implemented.
"""
from __future__ import annotations

from copy import deepcopy
from datetime import datetime, timezone
import math
from pathlib import Path
import time

import numpy as np
import pandas as pd

from .common import atomic_json, digest, read_json, sha256, utc_now
from .repair_cohort import PANEL, _check_deadline, _fixture, _identity, _parent_plan, _publish, _read_plan

BENCHMARK_SCHEMA = 'gf-repair-preparation-benchmark-v1'
BENCHMARK_PLAN_SCHEMA = 'gf-repair-preparation-benchmark-plan-v1'


def _bound_metadata(config, path):
    path = Path(path).resolve()
    expected = config['repair'].get('binding_files', {}).get(str(path))
    if expected is None:
        raise PermissionError(f'Benchmark metadata was not bound by repair setup: {path}')
    if sha256(path) != expected:
        raise PermissionError(f'Bound benchmark metadata changed: {path}')
    return read_json(path), expected


def plan_benchmark(config):
    """Choose two actual retained development families by an immutable hash rank."""
    _check_deadline(config)
    parent, parent_hash = _bound_metadata(config, config['repair']['parent_config'])
    manifest_path = Path(config['repair']['parent_bundle'])/'manifest.json'
    metadata, manifest_hash = _bound_metadata(config, manifest_path)
    original = metadata['records']
    if any(r['split']=='confirmation' for r in original):
        raise PermissionError('A benchmark dependency must never contain confirmation targets')
    seen = {}
    for row in original:
        if row['split'] != 'development':
            continue
        if row['original_split'] != 'validation':
            raise PermissionError('Already-open development records must preserve original validation roles')
        family = row['source_family_id']
        signature = (row['canonical_person_id'],row['motion_hash'],float(row['start_s']),float(row['end_s']))
        if family in seen and seen[family][0] != signature:
            raise ValueError('A retained benchmark family has inconsistent identity or source timing')
        seen.setdefault(family,(signature,row))
    if _fixture(config):
        candidates = [deepcopy(value[1]) for value in seen.values()]
        for row in candidates:
            row.setdefault('source_dataset','software_fixture')
            row.setdefault('relative_path',row['motion_id'])
            row.setdefault('historical_motion_sha256',row['motion_hash'])
            row.setdefault('parent_audited_start_s',float(row['start_s']))
    else:
        frozen = _parent_plan(parent)
        candidates = []
        for row in frozen['records']:
            if row['split'] != 'development' or row['source_family_id'] not in seen:
                continue
            signature = (row['canonical_person_id'],row['motion_hash'],float(row['start_s']),float(row['end_s']))
            if signature != seen[row['source_family_id']][0]:
                raise PermissionError('Retained development records disagree with the original frozen source cohort')
            candidates.append({k:v for k,v in row.items() if k!='available'})
    ranked = sorted(candidates,key=lambda r:digest([731,'repair-preparation-benchmark',r['source_family_id']]))
    if len(ranked) < 2:
        raise ValueError('Preparation benchmarking needs two actual retained development source windows')
    chosen = ranked[:2]
    _, slim = _read_plan(config)
    result = dict(schema=BENCHMARK_PLAN_SCHEMA,fixture=_fixture(config),created_by='metadata_hash_selection',
        parent_config=str(Path(config['repair']['parent_config']).resolve()),parent_config_sha256=parent_hash,
        parent_manifest=str(manifest_path.resolve()),parent_manifest_sha256=manifest_hash,
        slim_cohort_identity=slim['identity'],panel=deepcopy(slim['panel']),
        source_family_ids=sorted(r['source_family_id'] for r in chosen),records=chosen,
        selection='first_two_hash_ranked_already_retained_development_windows',
        protected_confirmation_opened=False,model_outcomes_used=False,
        scope='Software timing fixture only' if _fixture(config) else 'Preparation throughput on already-open development; no new scientific evaluation')
    result['identity']=_identity(result)
    path=Path(config['repair']['benchmark_plan'])
    _publish(path,result)
    return result


def load_benchmark_cohort(config):
    """Source backend adapter with truthful development roles and exact windows."""
    authority=config.get('_repair_benchmark_config',config)
    _check_deadline(authority)
    path=Path(authority['repair']['benchmark_plan'])
    if not path.is_file():
        raise PermissionError('Freeze benchmark metadata before source preparation')
    plan=read_json(path)
    if plan.get('schema')!=BENCHMARK_PLAN_SCHEMA or plan.get('identity')!=_identity(plan):
        raise PermissionError('Benchmark metadata identity changed')
    # Reconstruct from bound parent metadata; a hand-edited plan cannot admit a
    # test person, a previously excluded family, or a moved interval.
    if plan_benchmark(authority)!=plan:
        raise PermissionError('Benchmark population changed')
    rows=plan['records']
    if len(rows)!=2 or any(r['split']!='development' or r['original_split']!='validation' for r in rows):
        raise PermissionError('Preparation benchmark may contain exactly two already-open development windows')
    missing=[r['raw_path'] for r in rows if not Path(r['raw_path']).is_file()]
    if missing:
        raise FileNotFoundError(f'Benchmark source motion unavailable: {missing[0]}')
    return pd.DataFrame(rows),[],plan['identity']


def _verify_benchmark_result(config, supplied):
    if not isinstance(supplied,dict):
        path=Path(supplied)
        supplied=dict(receipt=str(path),receipt_sha256=sha256(path))
    path=Path(supplied['receipt'])
    receipt=read_json(path)
    if supplied.get('receipt_sha256')!=sha256(path):
        raise PermissionError('Preparation benchmark receipt changed or lacks its completion hash')
    if receipt.get('schema')!=BENCHMARK_SCHEMA or receipt.get('identity')!=_identity(receipt):
        raise PermissionError('Preparation benchmark identity changed')
    if receipt['fixture']!=_fixture(config):
        raise PermissionError('Software fixture timing cannot admit source preparation')
    plan=read_json(config['repair']['benchmark_plan'])
    for key in ('parent_config','parent_manifest'):
        _bound_metadata(config,plan[key])
    _,slim=_read_plan(config)
    if (receipt['plan_sha256']!=sha256(config['repair']['benchmark_plan']) or
        receipt['plan_identity']!=plan['identity'] or plan['slim_cohort_identity']!=slim['identity'] or
        receipt['panel']!=slim['panel']):
        raise PermissionError('Benchmark timing does not bind the frozen confirmation panel')
    for name,expected in receipt['artifacts'].items():
        if sha256(name)!=expected:
            raise PermissionError(f'Preparation benchmark artifact changed: {name}')
    return receipt


def benchmark_preparation(config, output):
    """Time the full real pipeline with both estimators before opening test data."""
    _check_deadline(config)
    output=Path(output).resolve()
    if output.exists():
        raise FileExistsError('Preparation benchmark outputs are immutable; use a new attempt')
    started=time.monotonic()
    plan=plan_benchmark(config)
    rows=plan['records'];panel=plan['panel']
    if _fixture(config):
        from .data import save_dataset
        from .repair_cohort import _fixture_bundle
        output.mkdir(parents=True)
        data=_fixture_bundle(dict(panel=panel),rows,int(config.get('seed',731))+500009)
        # The generator is shared code, but the benchmark keeps the original
        # development identity/exposure metadata and never enters confirmation.
        by_family={r['source_family_id']:r for r in rows}
        for record in data.records:
            record['exposure']=by_family[record['source_family_id']]['exposure']
            record.pop('exposure_scope',None)
        data.provenance.update(confirmation_admitted=False,fixture=True,source_selection='repair_benchmark',
            identity=plan['identity'],hz=panel['hz'],samples=panel['samples'])
        bundle=save_dataset(data,output/'bundle',storage='npy')
        elapsed=time.monotonic()-started
        family_seconds={r['source_family_id']:elapsed/2 for r in rows}
    else:
        from .preparation import prepare
        parent=read_json(config['repair']['parent_config'])
        local=deepcopy(config)
        local['preparation']=deepcopy(parent['preparation'])
        local['preparation']['estimators']=[s for s in local['preparation']['estimators'] if s['family'] in PANEL['extractor_families']]
        local['data']=dict(local['data'],source_selection='repair_benchmark',partition='development',
            movement_levels_deg=panel['movement_levels_deg'],held_level_deg=15.,cameras=panel['cameras'],
            samples=panel['samples'],hz=panel['hz'],shard_index=0,num_shards=1,
            resume_preparation_paths=[])
        local['_repair_benchmark_config']=config
        bundle=prepare(local,output)
        elapsed=time.monotonic()-started
        family_seconds={}
        for path in sorted((output/'family-costs').glob('*.json')):
            cost=read_json(path)
            if cost.get('reused_chunk'):
                raise PermissionError('Cached preparation cannot substitute for end-to-end timing')
            family_seconds[cost['source_family_id']]=float(cost['gpu_seconds'])
    if Path(bundle).suffix=='.json':
        retained=[];bundle_path=None
    else:
        bundle_path=str(Path(bundle).resolve())
        retained=read_json(Path(bundle)/'manifest.json')['records']
    actual={r['source_family_id'] for r in retained}
    expected=set(plan['source_family_ids'])
    complete=actual==expected and set(family_seconds)==expected
    if retained and any(r['split']!='development' or r['original_split']!='validation' for r in retained):
        raise PermissionError('Benchmark preparation changed a development role')
    if any(not math.isfinite(v) or v<=0 for v in family_seconds.values()) or not math.isfinite(elapsed) or elapsed<=0:
        raise ValueError('Benchmark needs positive finite measured preparation timings')
    setup=max(0.,elapsed-sum(family_seconds.values()))
    artifacts={str(p.resolve()):sha256(p) for p in output.rglob('*') if p.is_file()}
    receipt=dict(schema=BENCHMARK_SCHEMA,created_utc=utc_now(),fixture=_fixture(config),
        status='complete' if complete else 'incomplete',plan=str(Path(config['repair']['benchmark_plan']).resolve()),
        plan_sha256=sha256(config['repair']['benchmark_plan']),plan_identity=plan['identity'],panel=panel,
        bundle=bundle_path,planned_windows=2,retained_windows=len(actual),
        missing_source_families=sorted(expected-actual),elapsed_seconds=elapsed,
        setup_seconds=setup,family_seconds=family_seconds,
        maximum_family_seconds=max(family_seconds.values()) if family_seconds else None,
        rendered_frames=2*3*2*2*2*panel['samples'],extractor_families=PANEL['extractor_families'],
        protected_confirmation_opened=False,model_outcomes_used=False,
        timing_evidence='software_fixture_not_HAIC_runtime' if _fixture(config) else 'measured_full_render_and_both_extractors',
        artifacts=artifacts)
    receipt['identity']=_identity(receipt)
    path=output/'benchmark-receipt.json';atomic_json(path,receipt)
    return dict(receipt,receipt=str(path),receipt_sha256=sha256(path))


def admit_confirmation(config, benchmark_result):
    """Conservative time/allocation gate; no target outcomes enter this decision."""
    from .scheduler import _state, verify_completed
    _check_deadline(config,enforce=False)
    benchmark=_verify_benchmark_result(config,benchmark_result)
    _,plan=_read_plan(config)
    now=datetime.now(timezone.utc)
    deadline=datetime.fromisoformat(config['repair']['deadline_utc'].replace('Z','+00:00'))
    remaining=(deadline-now).total_seconds()/3600
    workers=int(config['resources']['max_jobs'])
    if not 1<=workers<=8:
        raise ValueError('Use the frozen worker count in [1,8]')
    n=len(plan['records'])
    safety=float(config['repair'].get('preparation_safety_factor',1.5))
    queue=float(config['repair'].get('queue_allowance_hours',1.))
    evaluation=float(config['repair'].get('evaluation_allowance_hours',2.))
    if (not all(math.isfinite(v) for v in (safety,queue,evaluation)) or safety<1.5 or queue<1 or evaluation<2):
        raise ValueError('Preparation admission must retain safety >=1.5, queue >=1h and evaluation >=2h')
    state=_state(config['work'])
    completed=state.get('completed',{})
    shard_ids={f'confirmation-shard-{i:02d}':i for i in range(workers)}
    for phase in set(completed)&(set(shard_ids)|{'confirmation-evaluate'}):
        verify_completed(completed[phase])
    active={a.get('phase_id') for a in state['attempts']
            if a.get('status') in {'reserved','submitted','running','accounting_pending'}}
    remaining_shards={phase:index for phase,index in shard_ids.items() if phase not in completed}
    new_shards={phase:index for phase,index in remaining_shards.items() if phase not in active}
    counts={phase:sum(index%workers==shard for index in range(n)) for phase,shard in remaining_shards.items()}
    longest=max(counts.values(),default=0)
    new_windows=sum(counts[phase] for phase in new_shards)
    evaluation_remaining='confirmation-evaluate' not in completed
    evaluation_new=evaluation_remaining and 'confirmation-evaluate' not in active
    used=sum(float(a.get('allocated_gpu_hours',a['reserved_gpu_hours'])) for a in state['attempts'])
    cap=float(config['resources']['gpu_hours'])
    if not math.isfinite(used) or used<0 or not math.isfinite(cap) or cap<=0:
        raise ValueError('Allocation usage and cap must be finite and nonnegative')
    maximum=float(benchmark['maximum_family_seconds'] or 0.)
    setup=float(benchmark['setup_seconds'])
    # Active attempts are already charged at their full reservations in `used`;
    # completed shards are never charged again on a resumed coordinator.
    gpu=safety*(len(new_shards)*setup+new_windows*maximum)/3600
    wall=safety*(setup+longest*maximum)/3600 if remaining_shards else 0.
    phase_cap=float(config['resources'].get('prepare_wall_minutes',240))/60
    if not math.isfinite(phase_cap) or phase_cap<=0:
        raise ValueError('Preparation worker time limit must be finite and positive')
    grace=2/60
    evaluation_gpu=2. if evaluation_new else 0.
    projected_gpu=gpu+evaluation_gpu
    queue_needed=queue if new_shards or evaluation_new else 0.
    required=max(wall,phase_cap+grace if new_shards else 0.)+queue_needed+(evaluation+grace if evaluation_remaining else 0.)
    reasons=[]
    if benchmark['status']!='complete' or benchmark['retained_windows']!=2:
        reasons.append('benchmark_did_not_complete_both_declared_source_windows')
    if not _fixture(config) and remaining<=0 and (remaining_shards or evaluation_remaining):
        reasons.append('experiment_deadline_passed')
    if not _fixture(config) and required>remaining:
        reasons.append('insufficient_deadline_for_preparation_queue_and_evaluation_reserve')
    if used+projected_gpu>cap:
        reasons.append('insufficient_remaining_gpu_allocation')
    if wall>phase_cap:
        reasons.append('longest_projected_shard_exceeds_worker_time_limit')
    # Submit-time accounting reserves the full declared worker time limits.
    # Check those reservations as well as observed-cost projections.
    reservation=len(new_shards)*phase_cap+evaluation_gpu
    if used+reservation>cap:
        reasons.append('insufficient_cap_for_all_declared_worker_reservations')
    return dict(schema='gf-repair-confirmation-admission-v1',evaluated_utc=now.isoformat(),
        admitted=not reasons,status='ADMITTED' if not reasons else 'REFUSED',reasons=reasons,
        fixture=_fixture(config),benchmark_receipt_sha256=benchmark_result.get('receipt_sha256') if isinstance(benchmark_result,dict) else None,
        planned_people=len(plan['person_ids']),planned_windows=n,workers=workers,
        completed_preparation_shards=len(shard_ids)-len(remaining_shards),
        remaining_preparation_shards=len(remaining_shards),unreserved_preparation_shards=len(new_shards),
        projected_unreserved_windows=new_windows,evaluation_remaining=evaluation_remaining,
        windows_in_longest_shard=longest,measured_setup_seconds=setup,
        measured_maximum_family_seconds=maximum,safety_factor=safety,
        remaining_wall_hours=remaining,charged_or_reserved_gpu_hours=used,gpu_cap_hours=cap,
        projected_preparation_gpu_hours=gpu,projected_longest_shard_hours=wall,
        projected_evaluation_gpu_hours=evaluation_gpu,projected_additional_gpu_hours=projected_gpu,
        required_wall_hours=required,queue_allowance_hours=queue,evaluation_allowance_hours=evaluation,
        worker_reservation_gpu_hours=reservation,
        scientific_result_guaranteed=False,model_outcomes_used=False,
        limitation='Two development windows measure throughput, not all hardware or source variation; queue time remains uncertain')


def _csv(path):
    if not path.is_file(): return None
    return pd.read_csv(path,keep_default_na=False,dtype=str)


def audit_expansion(config):
    """Inventory CMU/GAVD metadata without claiming new participants or readiness."""
    parent=read_json(config['repair']['parent_config'])
    root=Path(config.get('asset_root',config.get('code_root',Path.cwd())))
    prep=parent.get('preparation',{})
    manifest=Path(prep.get('manifest_dir',root/'manifests/amass'))
    if (manifest/'amass').is_dir(): manifest=manifest/'amass'
    # CMU is deliberately absent from the approved eligible cohort because
    # folder aliases are unresolved. Its capacity audit needs the raw census.
    cmu_path=manifest/'amass_raw_inventory.csv'
    inventory=_csv(cmu_path)
    cmu=dict(ready=False,protocol_implemented=False,metadata=str(cmu_path),metadata_available=inventory is not None,
        verified_independent_people=None,scope='Potential held-source motion transfer, not additional verified participants')
    if inventory is not None and {'source_dataset','relative_path','subject_id_candidate','num_frames','mocap_framerate'}<=set(inventory):
        selected=inventory[inventory.source_dataset.str.casefold().eq('cmu')]
        minimum=(int(config['data']['samples'])-1)/float(config['data']['hz'])
        rates=pd.to_numeric(selected.mocap_framerate,errors='coerce')
        duration=(pd.to_numeric(selected.num_frames,errors='coerce')-1)/rates
        raw=Path(prep.get('amass_root',root/'data/amass/extracted'))
        cmu.update(motions=len(selected),candidate_folders=selected.subject_id_candidate.nunique(),
            minimum_duration_eligible_motions=int((duration.ge(minimum)&np.isfinite(duration)&rates.gt(0)).sum()),
            raw_files_present=sum((raw/p).is_file() for p in selected.relative_path),
            reason='No separate frozen CMU transfer protocol, verified person crosswalk, or action-label join was admitted')
    else:
        cmu['reason']='Required source inventory metadata is missing or incompatible'
    options=config.get('gavd',{})
    gavd_manifest=Path(options.get('manifest_dir',root/'manifests/gavd'))
    seq_path=gavd_manifest/'gavd_full_sequences.csv';sequences=_csv(seq_path)
    video_root=Path(options.get('video_root',root/'data/gavd_full/youtube/all'))
    gavd=dict(ready=False,protocol_implemented=False,metadata=str(seq_path),metadata_available=sequences is not None,
        video_root=str(video_root),verified_independent_people=None,
        scope='Recording-group label-information transfer; no paired dense motion reference')
    needed={'sequence_id','video_id','dataset_annotation','gait_pattern_annotation','first_frame','last_frame'}
    if sequences is not None and needed<=set(sequences):
        ids=set(sequences.video_id)
        label_groups={label:int(rows.video_id.nunique()) for label,rows in sequences.groupby('dataset_annotation',sort=True)}
        available=sum(any((video_root/f'{video}{suffix}').is_file() for suffix in ('.mp4','.mkv','.webm','.mov','.m4v')) for video in ids)
        native_length=pd.to_numeric(sequences.last_frame,errors='coerce')-pd.to_numeric(sequences.first_frame,errors='coerce')+1
        long_enough=sequences.loc[native_length.ge(int(config['data']['samples']))]
        gavd.update(sequences=len(sequences),recording_ids=len(ids),recording_groups_by_dataset_annotation=label_groups,
            videos_present=available,native_span_eligible_recording_ids=int(long_enough.video_id.nunique()),
            identity_ledger_available=bool(options.get('identity_csv') and Path(options['identity_csv']).is_file()),
            exposure_ledger_available=bool(options.get('reservation_csv') and Path(options['reservation_csv']).is_file()),
            compatible_pose_cache_verified=False,
            reason='Grouped binary transfer protocol and compatible body12/25Hz caches are not admitted; native frame count does not establish resampled support')
    else:
        gavd['reason']='Required GAVD sequence metadata is missing or incompatible'
    result=dict(schema='gf-repair-expansion-readiness-v1',created_utc=utc_now(),metadata_only=True,
        raw_motion_arrays_opened=False,media_decoded=False,downloads=0,
        cmu=cmu,gavd=gavd,experiment_scope_changed=False,
        recommendation='Keep both optional expansions outside the fixed repair experiment; missing evidence does not become verified independence')
    path=Path(config['work'])/'expansion-readiness.json';atomic_json(path,result)
    return result
