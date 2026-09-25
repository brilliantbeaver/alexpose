"""Isolated, receipt-bound readout repair and unopened-person confirmation.

The completed response study is a read-only dependency. This entry point never
changes its protocol, checkpoints, person roles, or historical result tables.
"""
from __future__ import annotations

import argparse
import copy
import csv
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import shlex
import subprocess
import sys
import time

import numpy as np

from .common import atomic_json, code_identity, digest, locked, read_json, sha256, utc_now, verify_code

KIND = 'gait_fidelity_readout_repair'
DEADLINE = '2026-09-25T15:00:00Z'
SEEDS = (17, 29, 43)
VARIANTS = ('jepa_delta_v1', 'jepa_endpoint_v1')
OBJECTIVES = ('scalar_low', 'dense_change')
ACTIVE = {'reserved', 'submitted', 'running', 'accounting_pending'}


def method(variant, objective):
    return f'R-repair-{variant}-{objective}'


def statistical_protocol(*, fixture=False):
    return dict(primary_candidate=method(VARIANTS[0], 'dense_change'),
                primary_comparator=method(VARIANTS[0], 'scalar_low'),
                primary_metric='waveform_error', primary_extractor='fixture-source' if fixture else 'vitpose',
                seeds=list(SEEDS), bootstrap_draws=20 if fixture else 2000, bootstrap_seed=731,
                independent_unit='canonical_person_id', primary_alpha=.05,
                primary_inference='paired_person_means_conditional_on_three_fixed_fits',
                secondary_inference='descriptive_unadjusted',
                response_rule='report_effect_and_interval; no noninferiority claim',
                meaningful_margin_deg=None, model_selection='none; all declared arms reported',
                development_and_confirmation='separate; never pooled',
                claim_scope='synthetic projected movement restoration; no clinical validity claim')


def _deadline(cfg, *, grace=0):
    if cfg['fixture']:
        return math.inf
    cutoff = datetime.fromisoformat(cfg['repair']['deadline_utc'].replace('Z', '+00:00'))
    remaining = (cutoff - datetime.now(timezone.utc)).total_seconds()
    if remaining <= grace:
        raise RuntimeError('Repair experiment cutoff reached; preserve partial artifacts and existing results')
    return remaining


def _bind(files, path, expected=None):
    path = Path(path).expanduser().resolve()
    actual = sha256(path)
    if expected is not None and actual != expected:
        raise RuntimeError(f'Dependency changed: {path}')
    files[str(path)] = actual
    return str(path)


def _read_study(work, files):
    from .scheduler import verify_completed
    work = Path(work).expanduser().resolve()
    cfg, plan, frozen, state = (read_json(work/name) for name in
                               ('config.json', 'plan.json', 'frozen.json', 'ledger.json'))
    if Path(cfg['work']).resolve() != work:
        raise ValueError('A bound study must remain at its recorded location')
    _bind(files, work/'config.json', frozen['config_sha256'])
    _bind(files, work/'plan.json', frozen['plan_sha256'])
    _bind(files, work/'frozen.json'); _bind(files, work/'ledger.json')
    verify_code(cfg['code_root'], frozen['code'])
    if any(a['status'] in ACTIVE for a in state['attempts']):
        raise RuntimeError('Finish unresolved source-study jobs before creating the repair extension')
    for phase in [p['phase_id'] for p in plan['phases']] + ['evaluation']:
        value = state['completed'].get(phase)
        if value is None:
            raise RuntimeError(f'Completed study is missing {phase}')
        verify_completed(value)
        receipt = read_json(value['receipt'])
        if (receipt.get('phase_id') != phase or receipt['result'] != value['result']
                or receipt['config_sha256'] != frozen['config_sha256']):
            raise RuntimeError('Completion result or configuration differs from its receipt')
        _bind(files, value['receipt'], value['sha256'])
    return cfg, plan, state


def _phase_result(plan, state, *, phase, variant=None, recipe=None, seed):
    matches = [p for p in plan['phases'] if p['phase'] == phase and p['seed'] == seed
               and (variant is None or p['recipe'].get('representation_variant') == variant)
               and (recipe is None or p['recipe']['recipe_id'] == recipe)]
    if len(matches) != 1:
        raise ValueError(f'Expected exactly one completed {phase}/{variant or recipe}/{seed}')
    return copy.deepcopy(state['completed'][matches[0]['phase_id']]['result'])


def build_plan():
    phases = []
    for variant in VARIANTS:
        for seed in SEEDS:
            calibration = f'calibrate-{variant}-{seed}'
            phases.append(dict(phase_id=calibration, kind='calibrate', variant=variant,
                               seed=seed, depends_on=[], allocation_minutes=30))
            for objective in OBJECTIVES:
                phases.append(dict(phase_id=f'fit-{variant}-{objective}-{seed}', kind='fit',
                                   variant=variant, seed=seed, objective=objective,
                                   method=method(variant, objective), depends_on=[calibration],
                                   allocation_minutes=60))
    plan = dict(schema='gf-repair-plan-v1', phases=phases,
                counts=dict(calibrations=6, new_readouts=12, new_pretraining=0,
                            retained_comparator_fits=15, evaluation_fits=27),
                variants=list(VARIANTS), objectives=list(OBJECTIVES), seeds=list(SEEDS))
    plan['identity'] = digest(plan)
    return plan


def initialize_repair(work, *, response_work, deadline_utc=DEADLINE, max_jobs=4, gpu_hours=48.):
    import torch
    from .config import repository_root
    from .data import load_dataset
    from .repair_cohort import plan_slim_cohort
    work = Path(work).expanduser().resolve(); response_work = Path(response_work).expanduser().resolve()
    if not 1 <= max_jobs <= 8 or not math.isfinite(gpu_hours) or gpu_hours <= 0:
        raise ValueError('Use 1–8 workers and a positive finite GPU allocation cap')
    cutoff = datetime.fromisoformat(deadline_utc.replace('Z', '+00:00'))
    if cutoff.tzinfo is None:
        raise ValueError('An explicit timezone is required for the deadline')
    deadline_utc = cutoff.astimezone(timezone.utc).isoformat().replace('+00:00', 'Z')
    if (work/'config.json').exists():
        cfg = load_repair(work)
        if (cfg['repair']['response_work'] != str(response_work) or cfg['repair']['deadline_utc'] != deadline_utc
                or cfg['resources']['max_jobs'] != max_jobs or cfg['resources']['gpu_hours'] != gpu_hours):
            raise ValueError('Existing repair protocol differs; create a new work directory')
        return cfg
    if work.exists() and any(work.iterdir()):
        raise FileExistsError('Repair work directory must be empty')
    files = {}
    response, response_plan, response_state = _read_study(response_work, files)
    if response.get('study_kind') != 'jepa_response_followup' or not response['followup'].get('include_base_readouts'):
        raise ValueError('Repair requires the completed 27-phase response readout-control study')
    parent_work = Path(response['followup']['parent_binding']['parent_work']).resolve()
    parent, parent_plan, parent_state = _read_study(parent_work, files)
    if any(work == p or work.is_relative_to(p) for p in (parent_work, response_work)):
        raise ValueError('Repair cannot live inside a completed source study')
    root = repository_root().resolve()
    if not response['fixture'] and any(root == Path(c['code_root']).resolve() for c in (response, parent)):
        raise ValueError('Use a new immutable release, distinct from both completed study releases')
    bundle_path = Path(response['followup']['parent_binding']['bundle']).resolve()
    manifest = read_json(bundle_path/'manifest.json')
    if any(r['split'] == 'confirmation' for r in manifest['records']):
        raise PermissionError('Confirmation data cannot enter the fitting dependency')
    _bind(files, bundle_path/'manifest.json', response['followup']['parent_binding']['manifest_sha256'])
    # The standard loader verifies array hashes. Record every dependency, so a
    # new invocation cannot consume replaced arrays under an old manifest.
    if manifest.get('storage', 'npz') == 'npy':
        for group in ('inputs', 'targets'):
            for item in manifest['arrays'][group].values():
                _bind(files, bundle_path/item['path'], item['sha256'])
    else:
        for group in ('inputs', 'targets'):
            _bind(files, bundle_path/f'{group}.npz', manifest[f'{group}_sha256'])
    upstream, comparisons = {}, []
    for variant in VARIANTS:
        for seed in SEEDS:
            result = _phase_result(response_plan, response_state, phase='pretrain', variant=variant, seed=seed)
            checkpoint = _bind(files, result['checkpoint'], result['checkpoint_sha256'])
            payload = torch.load(checkpoint, map_location='cpu', weights_only=True)
            upstream[f'{variant}/seed-{seed}'] = dict(checkpoint=checkpoint,
                checkpoint_sha256=sha256(checkpoint), signature_sha256=digest(payload['signature']))
            for objective in ('base', 'paired_change'):
                name = f'F-response-{variant}-graph_time-{objective}'
                value = _phase_result(response_plan, response_state, phase='readout', recipe=name, seed=seed)
                _bind(files, value['checkpoint'], value['checkpoint_sha256'])
                _bind(files, value['predictions'], value['predictions_sha256'])
                if value.get('prediction_indices'):
                    _bind(files, value['prediction_indices'], value['prediction_indices_sha256'])
                if value.get('seed') != seed:
                    raise ValueError('Completed response readout seed differs from its declared phase')
                comparisons.append({**value, 'method':name, 'seed':seed})
    for seed in SEEDS:
        value = _phase_result(parent_plan, parent_state, phase='end_to_end', recipe='P-direct-none-base', seed=seed)
        _bind(files, value['checkpoint'], value['checkpoint_sha256'])
        _bind(files, value['predictions'], value['predictions_sha256'])
        if value.get('prediction_indices'):
            _bind(files, value['prediction_indices'], value['prediction_indices_sha256'])
        if value.get('seed') != seed:
            raise ValueError('Completed direct readout seed differs from its declared phase')
        comparisons.append({**value, 'method':'P-direct-none-base', 'seed':seed})
    cfg = copy.deepcopy(response)
    cfg.pop('followup', None); cfg.pop('response', None)
    cfg.update(study_kind=KIND, experiment_set='repair', work=str(work), code_root=str(root),
               python=sys.executable, created_utc=utc_now(), seeds=list(SEEDS))
    cfg['repair'] = dict(schema='gf-repair-v1', response_work=str(response_work), parent_work=str(parent_work),
        parent_config=str(parent_work/'config.json'), parent_bundle=str(bundle_path),
        upstream_bindings=upstream, baseline_fits=comparisons, deadline_utc=deadline_utc,
        cohort_plan=str(work/'cohort/plan.json'), confirmation_lock=str(work/'confirmation/lock.json'),
        benchmark_plan=str(work/'benchmark/plan.json'), statistical_protocol=statistical_protocol(fixture=cfg['fixture']),
        binding_files=files, calibration_batches=2 if cfg['fixture'] else 32,
        queue_allowance_hours=1., evaluation_allowance_hours=2., preparation_safety_factor=1.5)
    cfg['resources'].update(max_jobs=max_jobs, gpu_hours=gpu_hours, max_attempts=2,
                            phase_wall_minutes=60, prepare_wall_minutes=240)
    _deadline(cfg)
    cohort = plan_slim_cohort(parent, windows_per_person=2, seed=731)
    cfg['repair']['cohort_identity'] = cohort['identity']
    work.mkdir(parents=True,exist_ok=True)
    atomic_json(work/'cohort/plan.json', cohort)
    atomic_json(work/'config.json', cfg); atomic_json(work/'plan.json', build_plan())
    atomic_json(work/'frozen.json', dict(schema='gf-repair-frozen-v1', code=code_identity(root),
        config_sha256=sha256(work/'config.json'), plan_sha256=sha256(work/'plan.json'),
        cohort_sha256=sha256(work/'cohort/plan.json')))
    (work/'session.env').write_text(''.join(f'export {k}={shlex.quote(str(v))}\n' for k,v in
        dict(GF_ROOT=root, GF_WORK=work, GF_PYTHON=sys.executable).items()))
    exposure_audit(cfg)
    return cfg


def load_repair(work, *, verify_dependencies=False):
    work = Path(work).expanduser().resolve(); cfg = read_json(work/'config.json')
    if cfg.get('study_kind') != KIND or Path(cfg['work']).resolve() != work:
        raise ValueError('This entry point requires an unmoved repair study')
    frozen = read_json(work/'frozen.json')
    for path, key in ((work/'config.json', 'config_sha256'), (work/'plan.json', 'plan_sha256'),
                      (Path(cfg['repair']['cohort_plan']), 'cohort_sha256')):
        if sha256(path) != frozen[key]:
            raise RuntimeError(f'Frozen repair protocol changed: {path}')
    verify_code(cfg['code_root'], frozen['code'])
    if verify_dependencies:
        for path, expected in cfg['repair']['binding_files'].items():
            if sha256(path) != expected:
                raise RuntimeError(f'Bound read-only dependency changed: {path}')
    return cfg


def exposure_audit(cfg):
    """Discover the configured ledger; absence of evidence remains unknown."""
    plan = read_json(cfg['repair']['cohort_plan'])
    people = plan.get('person_ids') or sorted({r['canonical_person_id'] for r in plan.get('records', [])})
    parent = read_json(cfg['repair']['parent_config'])
    candidates = list(dict.fromkeys(str(Path(v).expanduser()) for v in (
        parent.get('cohort', {}).get('reservation_csv'), parent.get('preparation', {}).get('reservation_csv'),
        str(Path(parent['asset_root'])/'outputs/synthetic-training-v2/full-01/inputs/person-reservations.csv')
            if parent.get('asset_root') else None) if v))
    available = [p for p in candidates if Path(p).is_file()]
    candidate = available[0] if available else (candidates[0] if candidates else None)
    rows = []
    for path in available:
        with Path(path).open(newline='') as stream:
            rows.extend(list(csv.DictReader(stream)))
    by_person = {}
    for row in rows:
        by_person.setdefault(row.get('canonical_person_id', ''), []).append(row)
    observed = read_json(Path(cfg['repair']['parent_bundle'])/'manifest.json')['records']
    observed_people = {r['canonical_person_id'] for r in observed}
    result = []
    for person in people:
        entries = by_person.get(person, [])
        verified = bool(entries) and all(r.get('exposure') == 'unexposed_verified'
            and str(r.get('reserved', '')).lower() == 'false' and r.get('original_split') == 'test'
            for r in entries)
        conflict = person in observed_people or any(
            r.get('exposure') not in {'unknown', 'unexposed_verified', ''}
            or str(r.get('reserved','')).strip().lower() in {'true','1','yes'}
            or r.get('original_split','') not in {'','test'} for r in entries)
        result.append(dict(canonical_person_id=person, status='known_exposure_or_conflict' if conflict
                           else 'documented_unexposed' if verified else 'review_required',
                           ledger_rows=len(entries), parent_data_overlap=person in observed_people))
    review_path=Path(cfg['work'])/'exposure-review-required.csv'
    if not review_path.exists():
        review_path.parent.mkdir(parents=True,exist_ok=True)
        with review_path.open('x',newline='') as stream:
            writer=csv.DictWriter(stream,fieldnames=['person_id','canonical_person_id','original_split','reserved','exposure'])
            writer.writeheader()
            for person in people:
                writer.writerow(dict(person_id=person,canonical_person_id=person,original_split='test',reserved='false',exposure='unknown'))
    receipt = dict(schema='gf-repair-exposure-audit-v1', people=result,
                   configured_ledger=candidate, configured_ledger_available=bool(candidate and Path(candidate).is_file()),
                   ledger_candidates=[dict(path=p,available=Path(p).is_file()) for p in candidates],
                   review_worksheet=str(review_path), worksheet_is_reviewed_evidence=False,
                   ready=bool(result) and all(r['status']=='documented_unexposed' for r in result),
                   claim='Checks recorded evidence only; no inferred innocence from absent logs')
    output = Path(cfg['work'])/'exposure-audit.json'; atomic_json(output, receipt)
    return receipt


def fit_manifest(cfg):
    from .scheduler import _state, verify_completed
    state = _state(cfg['work']); fits = []
    for p in build_plan()['phases']:
        if p['kind'] != 'fit': continue
        completed = state['completed'].get(p['phase_id'])
        if completed is None: raise RuntimeError('Complete all twelve readouts before freezing confirmation')
        verify_completed(completed); value = completed['result']
        fits.append(dict(method=p['method'], seed=p['seed'], checkpoint=value['checkpoint'],
                         sha256=value['checkpoint_sha256']))
    for value in cfg['repair']['baseline_fits']:
        fits.append(dict(method=value['method'], seed=value['seed'], checkpoint=value['checkpoint'],
                         sha256=value['checkpoint_sha256']))
    return fits


def execute_phase(cfg, phase_id, attempt_path):
    from .data import load_dataset
    from .scheduler import _state, verify_completed
    from .repair_training import calibrate_repair, train_repair
    _deadline(cfg, grace=120)
    work = Path(cfg['work']); attempt_path = Path(attempt_path).resolve()
    state = _state(work)
    attempts = [a for a in state['attempts'] if a['phase_id']==phase_id and Path(a['path']).resolve()==attempt_path]
    if len(attempts) != 1 or not attempt_path.is_relative_to(work/'attempts'):
        raise ValueError('Worker needs its own persisted allocation reservation')
    attempt_path.mkdir(parents=True, exist_ok=True)
    with locked(attempt_path/'worker.lock', nonblocking=True):
        if (attempt_path/'complete.json').exists(): raise FileExistsError('Completed attempts are immutable')
        load_repair(work)
        started = time.monotonic(); artifact_root = attempt_path/'result'
        if phase_id == 'benchmark':
            from .repair_profile import benchmark_preparation
            result = benchmark_preparation(cfg, artifact_root)
        elif phase_id == 'confirmation-evaluate':
            result = evaluate(cfg, split='confirmation')
            artifact_root = work/'confirmation/evaluation'
        elif phase_id.startswith('confirmation-shard-'):
            from .repair_cohort import prepare_slim
            result = prepare_slim(cfg, artifact_root, shard_index=int(phase_id.rsplit('-', 1)[1]),
                                  num_shards=cfg['resources']['max_jobs'])
        else:
            phase = next(p for p in build_plan()['phases'] if p['phase_id']==phase_id)
            for dependency in phase['depends_on']: verify_completed(state['completed'][dependency])
            binding = cfg['repair']['upstream_bindings'][f"{phase['variant']}/seed-{phase['seed']}"]
            if sha256(binding['checkpoint']) != binding['checkpoint_sha256']:
                raise RuntimeError('Upstream checkpoint changed')
            bundle = load_dataset(cfg['repair']['parent_bundle'])
            if phase['kind']=='calibrate':
                result = calibrate_repair(bundle, Path(binding['checkpoint']), cfg, artifact_root)
            else:
                cal = state['completed'][phase['depends_on'][0]]['result']['calibration_receipt']
                local = copy.deepcopy(cfg)
                previous = [a for a in state['attempts'] if a['phase_id']==phase_id and a['status']=='failed'
                            and (Path(a['path'])/'result/checkpoint.pt').is_file()]
                if previous: local['training']['resume_from']=str(Path(previous[-1]['path'])/'result/checkpoint.pt')
                result = train_repair(bundle, Path(binding['checkpoint']), local, artifact_root,
                                      objective=phase['objective'], seed=phase['seed'], calibration=Path(cal))
        load_repair(work)
        artifacts = {str(p.resolve()):sha256(p) for p in artifact_root.rglob('*') if p.is_file() and p.suffix!='.lock'}
        if phase_id == 'confirmation-evaluate':
            retained = list((work/'confirmation/predictions').glob('*')) + [
                work/'confirmation/lock.json', work/'confirmation/merged.json', work/'confirmation-complete.json']
            artifacts.update({str(p.resolve()):sha256(p) for p in retained if p.is_file()})
        atomic_json(attempt_path/'complete.json', dict(phase_id=phase_id, config_sha256=sha256(work/'config.json'),
            completed_utc=utc_now(), worker_seconds=time.monotonic()-started, result=result, artifacts=artifacts))
        return result


def evaluate(cfg, *, split='development'):
    from .data import load_dataset
    from .evaluation import load_predictions
    from .repair_training import predict_with_deadline
    from .repair_evaluation import evaluate_repair
    from .scheduler import _state, verify_completed
    from .repair_cohort import verify_slim_lock
    _deadline(cfg)
    work = Path(cfg['work']); state = _state(work); fits = fit_manifest(cfg)
    if split == 'confirmation':
        declaration = verify_slim_lock(cfg)
        bundle = load_dataset(work/'confirmation/bundle', allow_confirmation=True)
        output = work/'confirmation/evaluation'
    else:
        declaration = None
        bundle = load_dataset(cfg['repair']['parent_bundle']).subset('development')
        output = work/'development/evaluation'
    local = copy.deepcopy(cfg); receipts = []; arrays = []
    for fit in fits:
        _deadline(cfg)
        if sha256(fit['checkpoint']) != fit['sha256']: raise RuntimeError('Evaluation checkpoint changed')
        if split == 'confirmation':
            path = work/'confirmation/predictions'/f"{fit['method']}-{fit['seed']}.npy"
            path.parent.mkdir(parents=True, exist_ok=True)
            receipt_path = path.with_suffix('.json')
            if path.exists() and receipt_path.exists():
                receipt = read_json(receipt_path)
                if (sha256(path) != receipt['predictions_sha256'] or receipt['checkpoint_sha256'] != fit['sha256']
                        or receipt['record_identity'] != digest(bundle.records)):
                    raise RuntimeError('Existing confirmation predictions differ from their receipt')
            else:
                if path.exists():
                    # A killed process may finish the atomic array publication
                    # before publishing its provenance receipt. Keep that orphan
                    # for inspection and recompute from the locked checkpoint.
                    path.rename(path.with_name(path.name+f'.unpublished-{time.time_ns()}'))
                predict_with_deadline(bundle, Path(fit['checkpoint']), cfg, output=path)
                receipt = dict(method=fit['method'], seed=fit['seed'], split=split, predictions=str(path),
                    predictions_sha256=sha256(path), record_identity=digest(bundle.records), checkpoint_sha256=fit['sha256'])
                atomic_json(receipt_path, receipt)
            prediction = np.load(path, mmap_mode='r', allow_pickle=False)
        else:
            new = next((p for p in build_plan()['phases'] if p.get('method')==fit['method'] and p['seed']==fit['seed']), None)
            if new:
                completed = state['completed'][new['phase_id']]; verify_completed(completed); result=completed['result']
            else:
                result=next(v for v in cfg['repair']['baseline_fits'] if v['method']==fit['method'] and v['seed']==fit['seed'])
            prediction, indices = load_predictions(result)
            parent_records=read_json(Path(cfg['repair']['parent_bundle'])/'manifest.json')['records']
            expected=np.flatnonzero([r['split']=='development' for r in parent_records])
            if not np.array_equal(indices, expected): raise RuntimeError('Development prediction order changed')
            path=Path(result['predictions'])
            if path.suffix=='.npz':
                copied=work/'development/predictions'/f"{fit['method']}-{fit['seed']}.npy"
                copied.parent.mkdir(parents=True,exist_ok=True)
                if not copied.exists(): np.save(copied,prediction,allow_pickle=False)
                elif not np.array_equal(np.load(copied),prediction,equal_nan=True): raise RuntimeError('Fixture copy changed')
                path=copied
            receipt=dict(method=fit['method'],seed=fit['seed'],split=split,predictions=str(path),
                         predictions_sha256=sha256(path),record_identity=digest(bundle.records),checkpoint_sha256=fit['sha256'])
        receipts.append(receipt); arrays.append((fit['method'],fit['seed'],prediction))
    local['repair']['prediction_receipts']=receipts
    if output.exists():
        result=read_json(output/'complete.json')
        if (result['config_identity']!=digest(local) or result['record_identity']!=digest(bundle.records)
                or result['prediction_inputs']!=receipts):
            raise RuntimeError('Published evaluation does not bind the current fixed exports and configuration')
        for artifact,expected in result['artifacts'].items():
            if sha256(artifact)!=expected: raise RuntimeError('Published evaluation artifact changed')
        result.update(receipt=str(output/'complete.json'),sha256=sha256(output/'complete.json'))
    else:
        result = evaluate_repair(bundle, iter(arrays), local, output, split=split,
                                 lock=Path(cfg['repair']['confirmation_lock']) if declaration else None)
    atomic_json(work/f'{split}-complete.json', dict(result=result, artifacts={str(p.resolve()):sha256(p)
                for p in output.rglob('*') if p.is_file()}))
    return result


def status(cfg):
    from .scheduler import _state
    state = _state(cfg['work']); work=Path(cfg['work'])
    return dict(study_kind=KIND, work=str(work), fixture=cfg['fixture'], deadline_utc=cfg['repair']['deadline_utc'],
        completed_calibrations=sum(k.startswith('calibrate-') for k in state['completed']),
        completed_readouts=sum(k.startswith('fit-') for k in state['completed']), total_readouts=12,
        active=[dict(phase=a['phase_id'], job_id=a['job_id'], status=a.get('scheduler_state',a['status']))
                for a in state['attempts'] if a['status'] in ACTIVE],
        failed=[dict(phase=a['phase_id'],job_id=a['job_id'],state=a.get('scheduler_state'))
                for a in state['attempts'] if a['status']=='failed'],
        charged_or_reserved_gpu_hours=sum(a.get('allocated_gpu_hours',a['reserved_gpu_hours']) for a in state['attempts']),
        development_complete=(work/'development-complete.json').is_file(),
        confirmation_locked=Path(cfg['repair']['confirmation_lock']).is_file(),
        confirmation_complete=(work/'confirmation-complete.json').is_file(),
        coordinator=read_json(work/'control/coordinator.json') if (work/'control/coordinator.json').exists() else None,
        reports={split:str(work/split/'evaluation/report.md') for split in ('development','confirmation')
                 if (work/split/'evaluation/report.md').exists()},
        exposure_audit=read_json(work/'exposure-audit.json') if (work/'exposure-audit.json').exists() else None)


def verify(cfg):
    from .scheduler import _state, verify_completed
    load_repair(cfg['work'],verify_dependencies=True)
    state=_state(cfg['work'])
    for value in state['completed'].values(): verify_completed(value)
    count=0
    lock_path=Path(cfg['repair']['confirmation_lock'])
    if lock_path.exists():
        from .repair_cohort import verify_slim_lock
        verify_slim_lock(cfg,enforce_deadline=False)
    for split in ('development','confirmation'):
        path=Path(cfg['work'])/f'{split}-complete.json'
        if path.exists():
            for artifact,expected in read_json(path)['artifacts'].items():
                if sha256(artifact)!=expected: raise RuntimeError(f'Evaluation artifact changed: {artifact}')
                count+=1
    return dict(status='REPAIR_ARTIFACTS_VERIFIED', completed_worker_receipts=len(state['completed']),
                evaluation_artifacts=count, scientific_results=not cfg['fixture'])


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    sub=parser.add_subparsers(dest='command',required=True)
    for name in ('setup','status','preflight','exposure-audit','launch','run','worker','lock-confirmation','evaluate','verify','fixture','audit-expansion'):
        p=sub.add_parser(name);p.add_argument('--work',type=Path,required=True)
        if name=='setup':
            p.add_argument('--response-work',type=Path,required=True);p.add_argument('--deadline-utc',default=DEADLINE)
            p.add_argument('--max-jobs',type=int,default=4);p.add_argument('--gpu-hours',type=float,default=48.)
        if name in {'run','launch'}:
            p.add_argument('--stage',choices=('development','benchmark','confirmation'),default='development')
        if name=='run':
            p.add_argument('--local',action='store_true')
        if name=='worker': p.add_argument('--phase',required=True);p.add_argument('--attempt',type=Path,required=True)
        if name=='lock-confirmation':
            p.add_argument('--exposure-ledger',type=Path);p.add_argument('--reviewed-by',required=True);p.add_argument('--evidence',required=True)
        if name=='evaluate':p.add_argument('--split',choices=('development','confirmation'),default='development')
    args=parser.parse_args(argv)
    if args.command=='fixture':
        from .repair_fixture import run_fixture
        result=run_fixture(args.work)
    elif args.command=='setup':
        result=initialize_repair(args.work,response_work=args.response_work,deadline_utc=args.deadline_utc,
                                 max_jobs=args.max_jobs,gpu_hours=args.gpu_hours)
        result=dict(status='REPAIR_INITIALIZED',work=result['work'],plan=build_plan()['counts'])
    else:
        cfg=load_repair(args.work)
        if args.command=='status':result=status(cfg)
        elif args.command=='exposure-audit':result=exposure_audit(cfg)
        elif args.command=='verify':result=verify(cfg)
        elif args.command=='preflight':
            load_repair(args.work,verify_dependencies=True)
            result=dict(status='REPAIR_PREFLIGHT_COMPLETE',fixture=cfg['fixture'],exposure=exposure_audit(cfg),
                        deadline_seconds_remaining=_deadline(cfg) if not cfg['fixture'] else None)
        elif args.command=='worker':result=execute_phase(cfg,args.phase,args.attempt)
        elif args.command=='evaluate':
            if args.split=='confirmation' and not cfg['fixture']:
                raise ValueError('Use launch --stage confirmation for reserved and supervised source inference')
            result=evaluate(cfg,split=args.split)
        elif args.command=='lock-confirmation':
            from .repair_cohort import lock_slim_confirmation
            if not (Path(cfg['work'])/'development-complete.json').is_file():
                raise RuntimeError('Finish the declared development evaluation before confirmation lock')
            audit=exposure_audit(cfg)
            if any(p['status']=='known_exposure_or_conflict' for p in audit['people']):
                raise PermissionError('Recorded exposure conflicts with confirmation; inspect exposure-audit.json')
            result=lock_slim_confirmation(cfg,fit_manifest(cfg),exposure_ledger=args.exposure_ledger,
                reviewed_by=args.reviewed_by,evidence=args.evidence,output=Path(cfg['repair']['confirmation_lock']))
        elif args.command=='run':
            from .repair_execution import run
            result=run(cfg,stage=args.stage,local=args.local)
        elif args.command=='launch':
            from .repair_execution import launch
            result=launch(cfg,stage=args.stage)
        else:
            from .repair_profile import audit_expansion
            result=audit_expansion(cfg)
    print(json.dumps(result,indent=2,allow_nan=False))
    return 0


if __name__=='__main__':
    raise SystemExit(main())
