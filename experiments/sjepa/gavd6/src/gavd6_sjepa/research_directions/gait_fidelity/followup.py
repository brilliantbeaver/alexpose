"""Immutable, explicitly bound response follow-up to a completed Gait Fidelity run.

The parent supplies data and comparison artifacts as a read-only dependency.
It never appears as a fictitious preparation job in the child's ledger.
"""
from __future__ import annotations

import copy
from datetime import datetime, timezone, timedelta
import math
import os
from pathlib import Path
import shlex
import subprocess
import sys
import time

from .common import atomic_json, digest, read_json, sha256, utc_now, verify_code
from .spec import build_response_plan

KIND = 'jepa_response_followup'
DEADLINE = '2026-09-25T01:00:00Z'
BUDGETS = dict(profile=2., fits=36., diagnostics=4., recovery=6.)
BASELINES = ('M-paired_jepa-graph_time-paired_change',
             'M-coordinate-graph_time-paired_change', 'P-direct-none-paired_change',
             'I-initialized-none-paired_change', 'I-shuffled_jepa-graph_time-paired_change')
ACTIVE = {'reserved', 'submitted', 'running', 'accounting_pending'}


def is_followup(cfg):
    return cfg.get('study_kind') == KIND


def _time(value):
    result = datetime.fromisoformat(value.replace('Z', '+00:00'))
    if result.tzinfo is None:
        raise ValueError('Deadline requires an explicit UTC offset')
    return result.astimezone(timezone.utc)


def _bind_file(files, path, expected=None):
    path = Path(path).resolve()
    if not path.is_file():
        raise FileNotFoundError(f'Parent dependency is missing: {path}')
    actual = sha256(path)
    if expected is not None and actual != expected:
        raise RuntimeError(f'Parent dependency changed: {path}')
    files[str(path)] = actual


def inspect_parent(parent_work, *, deadline_utc=DEADLINE, include_base_readouts=False):
    """Verify completed evidence and record exact hashes without writing to the parent."""
    parent = Path(parent_work).expanduser().resolve()
    cfg = read_json(parent/'config.json')
    if cfg.get('study_kind') == KIND or Path(cfg['work']).resolve() != parent:
        raise ValueError('The parent must be an original, unmoved Gait Fidelity study')
    if not cfg['fixture'] and datetime.now(timezone.utc) >= _time(deadline_utc):
        raise RuntimeError('The registered follow-up deadline has passed; no new source run was created')
    if cfg.get('seeds') != [17, 29, 43]:
        raise ValueError('The follow-up requires a completed three-seed parent')
    frozen, plan, state = (read_json(parent/name) for name in ('frozen.json', 'plan.json', 'ledger.json'))
    verify_code(cfg['code_root'], frozen['code'])
    if any(a['status'] in ACTIVE for a in state['attempts']):
        raise RuntimeError('Parent workers are unresolved; finish the core before creating a follow-up')
    required = {p['phase_id'] for p in plan['phases']} | {'prepare', 'evaluation'}
    if not cfg['fixture']:
        required.add('profile')
    if not required.issubset(state['completed']):
        raise RuntimeError('The complete parent matrix and its evaluation receipt are required')
    if any('receipt' not in state['completed'][key] for key in required-{'prepare'}):
        raise RuntimeError('Every required parent model, profile and evaluation needs its completion receipt')
    files = {}
    for name in ('config.json', 'plan.json', 'frozen.json', 'ledger.json'):
        expected = frozen.get(name.replace('.json', '_sha256')) if name in {'config.json', 'plan.json'} else None
        _bind_file(files, parent/name, expected)
    completed = state['completed']
    # Bind all completed model/evaluation receipts and their published artifacts.
    # Preparation videos are not needed for training; their source receipts remain bound.
    for phase_id, value in completed.items():
        if 'receipt' not in value:
            continue
        _bind_file(files, value['receipt'], value['sha256'])
        receipt = read_json(value['receipt'])
        if receipt.get('phase_id') != phase_id or receipt.get('result') != value.get('result'):
            raise RuntimeError('Parent ledger phase/result differs from its hashed completion receipt')
        if receipt['config_sha256'] != frozen['config_sha256']:
            raise RuntimeError('Parent completion belongs to a different configuration')
        if phase_id.startswith(('prepare', 'confirmation', 'gavd-')):
            continue
        for path, expected in receipt['artifacts'].items():
            _bind_file(files, path, expected)
    bundle = Path(completed['prepare']['result']['bundle']).resolve()
    manifest = read_json(bundle/'manifest.json')
    if any(row['split'] == 'confirmation' for row in manifest['records']):
        raise PermissionError('Confirmation arrays cannot be a follow-up dependency')
    _bind_file(files, bundle/'manifest.json', completed['prepare'].get('bundle_manifest_sha256'))
    if manifest.get('storage', 'npz') == 'npy':
        for group in ('inputs', 'targets'):
            for item in manifest['arrays'][group].values():
                path = bundle/item['path']
                if path.resolve().parent != bundle:
                    raise ValueError('Dataset array escapes its parent bundle')
                _bind_file(files, path, item['sha256'])
    else:
        for group in ('inputs', 'targets'):
            _bind_file(files, bundle/f'{group}.npz', manifest[f'{group}_sha256'])
    for directory in ('admissions',):
        for path in (parent/directory).rglob('*'):
            if path.is_file() and path.suffix in {'.json', '.csv', '.py'}:
                _bind_file(files, path)
    if (parent/'data/admission.json').is_file():
        _bind_file(files, parent/'data/admission.json')
    if cfg['fixture']:
        updates = {k: cfg['training'][k] for k in ('pretraining_updates', 'readout_updates', 'end_to_end_updates')}
    else:
        selected = completed['profile']['result']['budget']['selected']
        if not selected:
            raise RuntimeError('Parent profiling did not admit training')
        updates = selected['updates']
    for phase in plan['phases']:
        expected = updates[{'pretrain':'pretraining_updates','readout':'readout_updates','end_to_end':'end_to_end_updates'}[phase['phase']]]
        if completed[phase['phase_id']]['result'].get('updates') != expected:
            raise RuntimeError('A parent phase differs from its selected actual update budget')
    baselines = BASELINES + (tuple(m.replace('-paired_change', '-base') for m in BASELINES)
                            if include_base_readouts else ())
    baseline_phases = {p['phase_id']: p for p in plan['phases']
                       if p['phase'] != 'pretrain' and p['recipe']['recipe_id'] in baselines}
    if {(p['recipe']['recipe_id'], p['seed']) for p in baseline_phases.values()} != {
            (method, seed) for method in baselines for seed in (17, 29, 43)}:
        raise ValueError('Parent lacks the registered comparison methods across all three seeds')
    result = dict(schema='gait-fidelity-parent-binding-v1', parent_work=str(parent),
                  parent_code_root=cfg['code_root'], parent_python=cfg['python'],
                  parent_frozen=frozen, parent_completed=copy.deepcopy(completed),
                  baseline_phases=baseline_phases, bundle=str(bundle),
                  manifest_sha256=sha256(bundle/'manifest.json'), files=files,
                  actual_updates=updates, evidence_status=manifest['evidence_status'])
    result['identity'] = digest(result)
    return cfg, result


def initialize_followup(work, *, parent_work, deadline_utc=None, include_base_readouts=None):
    from .config import repository_root, load_config
    work = Path(work).expanduser().resolve()
    parent = Path(parent_work).expanduser().resolve()
    if work == parent or work.is_relative_to(parent):
        raise ValueError('A child directory must not be inside its immutable parent')
    if (work/'config.json').exists():
        cfg = load_config(work)
        if not is_followup(cfg) or cfg['followup']['parent_binding']['parent_work'] != str(parent):
            raise ValueError('Existing directory belongs to another study or parent')
        if deadline_utc is not None and _time(deadline_utc) != _time(cfg['followup']['deadline_utc']):
            raise ValueError('A saved follow-up deadline cannot be changed; use a new child directory')
        if include_base_readouts is not None and include_base_readouts != cfg['followup'].get('include_base_readouts', False):
            raise ValueError('A saved follow-up matrix cannot be changed; use a new child directory')
        verify_parent(cfg)
        return cfg
    if work.exists() and any(work.iterdir()):
        raise FileExistsError('A new follow-up needs an empty child directory')
    deadline_utc = _time(deadline_utc or DEADLINE).isoformat().replace('+00:00', 'Z')
    include_base_readouts = bool(include_base_readouts)
    cfg, binding = inspect_parent(parent, deadline_utc=deadline_utc, include_base_readouts=include_base_readouts)
    plan = build_response_plan(include_base_readouts=include_base_readouts)
    code_root = repository_root().resolve()
    if not cfg['fixture'] and code_root == Path(cfg['code_root']).resolve():
        raise ValueError('Use an immutable NEW release for the follow-up; do not edit or reuse the frozen parent checkout')
    cfg = copy.deepcopy(cfg)
    cfg.update(study_kind=KIND, created_utc=utc_now(), work=str(work), code_root=str(code_root),
               python=sys.executable, experiment_set=plan['experiment_set'], source_bundle=None,
               seeds=[17, 29, 43])
    cfg['training'].update(binding['actual_updates'])
    cfg['data']['source_selection'] = 'prepared_parent_dependency'
    cfg['followup'] = dict(schema='gait-fidelity-response-followup-v1', parent_binding=binding,
                           deadline_utc=deadline_utc, queue_allowance_hours=2.,
                           budgets_gpu_hours=BUDGETS.copy(), total_gpu_hours=48.,
                           admission=f"all_{len(plan['phases'])}_phases_at_parent_actual_updates; no_half_update_fallback",
                           core_priority='completed_parent_required', profile_updates=20)
    if include_base_readouts or deadline_utc != DEADLINE:
        cfg['followup'].update(schema='gait-fidelity-response-followup-v2',
                               include_base_readouts=include_base_readouts, evaluation_allowance_hours=2.)
    cfg['response'] = dict(protocol='paired-residual-coupling-v1', deadline_utc=deadline_utc)
    cfg['resources'].update(gpu_hours=0. if cfg['fixture'] else 48., phase_wall_minutes=120,
                             max_attempts=2, max_jobs=8)
    cfg['evaluation'].update(primary_candidate='F-response-jepa_delta_v1-graph_time-paired_change',
                              primary_comparator='F-response-jepa_endpoint_v1-graph_time-paired_change',
                              confirmation_admitted=False)
    work.mkdir(parents=True, exist_ok=True)
    atomic_json(work/'parent-binding.json', binding)
    atomic_json(work/'config.json', cfg)
    atomic_json(work/'plan.json', plan)
    (work/'session.env').write_text(''.join(f'export {k}={shlex.quote(str(v))}\n'
        for k, v in dict(GF_ROOT=code_root, GF_WORK=work, GF_PYTHON=sys.executable).items()))
    return cfg


def validate_followup(cfg):
    if not is_followup(cfg):
        return
    f = cfg['followup']
    if f['schema'] not in {'gait-fidelity-response-followup-v1', 'gait-fidelity-response-followup-v2'}:
        raise ValueError('Unknown follow-up protocol version')
    _time(f['deadline_utc'])
    if f['schema'].endswith('-v1') and f['deadline_utc'] != DEADLINE:
        raise ValueError('The original follow-up deadline is immutable')
    if f['budgets_gpu_hours'] != BUDGETS or f['total_gpu_hours'] != 48.:
        raise ValueError('The registered follow-up category allowances are immutable')
    extended = f.get('include_base_readouts', False)
    if type(extended) is not bool or (f['schema'].endswith('-v1') and extended):
        raise ValueError('Base readouts require an explicitly versioned protocol')
    expected_set = 'response_followup_readout_control' if extended else 'response_followup'
    if cfg['experiment_set'] != expected_set:
        raise ValueError('Follow-up matrix and declared readout controls differ')
    if f['schema'].endswith('-v2') and f.get('evaluation_allowance_hours') != 2.:
        raise ValueError('New launches reserve two CPU wall hours for final evaluation')
    if f['queue_allowance_hours'] != 2. or cfg['seeds'] != [17, 29, 43]:
        raise ValueError('Follow-up queue allowance and three-seed matrix must remain fixed')
    if any(cfg['training'][k] != v for k, v in f['parent_binding']['actual_updates'].items()):
        raise ValueError('Follow-up must inherit the actual parent update budget')
    if cfg['source_bundle'] is not None or cfg['data']['source_selection'] != 'prepared_parent_dependency':
        raise ValueError('Follow-up data is an explicit parent dependency, not a legacy source bundle')
    if cfg['resources']['phase_wall_minutes'] != 120 or cfg['resources']['gpu_hours'] != (0. if cfg['fixture'] else 48.):
        raise ValueError('Follow-up allocations require the registered two-hour phase and 48-hour total caps')
    if cfg['response']['deadline_utc'] != f['deadline_utc']:
        raise ValueError('Worker and coordinator deadlines differ')
    parent_path = Path(f['parent_binding']['parent_work'])/'config.json'
    if sha256(parent_path) != f['parent_binding']['files'].get(str(parent_path)):
        raise RuntimeError('The bound parent configuration changed')
    parent = read_json(parent_path)
    for key in ('model', 'measurement', 'seeds', 'fixture', 'mode', 'device', 'preparation', 'asset_root'):
        if cfg[key] != parent[key]:
            raise ValueError(f'Follow-up changed inherited scientific setting: {key}')
    expected_training = copy.deepcopy(parent['training'])
    expected_training.update(f['parent_binding']['actual_updates'])
    if cfg['training'] != expected_training:
        raise ValueError('Follow-up optimizer, masks, sampling and training settings must match the parent')
    expected_data = dict(parent['data'], source_selection='prepared_parent_dependency')
    if cfg['data'] != expected_data:
        raise ValueError('Follow-up data settings differ from the bound parent')
    expected_evaluation = dict(parent['evaluation'],
        primary_candidate='F-response-jepa_delta_v1-graph_time-paired_change',
        primary_comparator='F-response-jepa_endpoint_v1-graph_time-paired_change', confirmation_admitted=False)
    if cfg['evaluation'] != expected_evaluation:
        raise ValueError('Follow-up evaluation settings changed outside the registered primary contrast')


def verify_parent(cfg):
    binding = cfg['followup']['parent_binding']
    if digest({k: v for k, v in binding.items() if k != 'identity'}) != binding['identity']:
        raise RuntimeError('Parent binding identity changed')
    saved = read_json(Path(cfg['work'])/'parent-binding.json')
    if saved != binding:
        raise RuntimeError('Saved parent binding differs from child configuration')
    verify_code(binding['parent_code_root'], binding['parent_frozen']['code'])
    for path, expected in binding['files'].items():
        if not Path(path).is_file() or sha256(path) != expected:
            raise RuntimeError(f'Read-only parent dependency changed: {path}')
    if any(a['status'] in ACTIVE for a in read_json(Path(binding['parent_work'])/'ledger.json')['attempts']):
        raise RuntimeError('The parent has active work; core priority forbids child launch')
    return binding


def bundle_path(cfg):
    return Path(cfg['followup']['parent_binding']['bundle'])


def category(phase_id, previous_attempts=0):
    if previous_attempts:
        return 'recovery'
    return {'followup-profile': 'profile', 'followup-diagnostics': 'diagnostics'}.get(phase_id, 'fits')


def allocation_minutes(cfg, phase_id, previous_attempts=0, *, now=None):
    minutes = 240 if phase_id == 'followup-diagnostics' else 120
    if cfg['fixture']:
        return minutes
    current = now or datetime.now(timezone.utc)
    available = (_time(cfg['followup']['deadline_utc']) - current).total_seconds()/60
    if available < minutes + (cfg['followup']['queue_allowance_hours'] + cfg['followup'].get('evaluation_allowance_hours', 0.))*60:
        raise RuntimeError('Too little time remains for the full allocation and fixed queue allowance')
    return minutes


def budget_usage(state):
    values = {key: 0. for key in BUDGETS}
    for attempt in state['attempts']:
        charge = attempt.get('allocated_gpu_hours', attempt['reserved_gpu_hours'])
        if not math.isfinite(charge) or charge < 0:
            raise ValueError('Invalid allocation accounting')
        values[attempt['budget_category']] += charge
    return values


def response_timing_key(recipe, phase):
    suffix = ':base' if phase == 'readout' and recipe['readout_or_training_objective'] == 'base' else ''
    return f"{recipe['representation_variant']}:{phase}{suffix}"


def admit_matrix(cfg, timings, *, now=None):
    """Timing-only admission at inherited updates, with no result-dependent fallback."""
    phases = build_response_plan(include_base_readouts=cfg['followup'].get('include_base_readouts', False))['phases']
    predicted = {}
    for phase in phases:
        key = response_timing_key(phase['recipe'], phase['phase'])
        timing = timings[key]
        updates = cfg['training']['pretraining_updates' if phase['phase'] == 'pretrain' else 'readout_updates']
        seconds = float(timing['fixed_seconds']) + float(timing['optimization_seconds'])*updates/int(timing['profile_updates'])
        if not math.isfinite(seconds) or seconds <= 0:
            raise ValueError('Positive finite profile timing required for every phase')
        predicted[phase['phase_id']] = seconds*1.25
    projected = sum(predicted.values())/3600
    # A deterministic dependency-respecting schedule bounds wall time at the saved cap.
    finish, lanes = {}, [0.]*cfg['resources']['max_jobs']
    for phase in phases:
        lane = min(range(len(lanes)), key=lanes.__getitem__)
        prerequisite = max([finish.get(p, 0.) for p in phase['depends_on']] or [0.])
        start = max(lanes[lane], prerequisite)
        finish[phase['phase_id']] = lanes[lane] = start + predicted[phase['phase_id']]
    matrix_wall = max(lanes)/3600
    current = now or datetime.now(timezone.utc)
    available = (_time(cfg['followup']['deadline_utc']) - current).total_seconds()/3600
    evaluation_hours = cfg['followup'].get('evaluation_allowance_hours', 0.)
    needed_wall = matrix_wall + 4. + 6. + cfg['followup']['queue_allowance_hours'] + evaluation_hours
    admitted = projected <= 36. and max(predicted.values()) <= 119*60 and (cfg['fixture'] or needed_wall <= available)
    return dict(status='FOLLOWUP_MATRIX_ADMITTED' if admitted else 'FOLLOWUP_MATRIX_NOT_ADMITTED',
                admitted=admitted, actual_updates=cfg['followup']['parent_binding']['actual_updates'],
                projected_fit_gpu_hours=projected, projected_fit_wall_hours=matrix_wall,
                required_wall_hours=needed_wall, remaining_wall_hours=available,
                predicted_phase_seconds=predicted, timing_safety_factor=1.25,
                allocation_reservation_minutes=120, slurm_workload_minutes=119,
                deadline_utc=cfg['followup']['deadline_utc'], evaluated_utc=current.isoformat(),
                criterion=f'All {len(phases)} phases at inherited updates; timing only; 4h diagnostics, 6h recovery, 2h queue and {evaluation_hours:g}h evaluation held aside')


def execute_followup_phase(cfg, phase_id, attempt_path, *, started):
    """Execute only child work; parent files are read, checked, and never rewritten."""
    from .data import load_dataset
    from .scheduler import _state, verify_completed
    if not cfg['fixture'] and datetime.now(timezone.utc) >= _time(cfg['followup']['deadline_utc'])-timedelta(seconds=120):
        raise RuntimeError('Follow-up worker started inside the checkpoint deadline; no new workload was launched')
    verify_parent(cfg)
    work, attempt_path = Path(cfg['work']), Path(attempt_path)
    state = _state(work)
    if phase_id == 'followup-profile':
        from .config import preflight
        from .profiling import profile_response
        atomic_json(attempt_path/'preflight.json', preflight(cfg, gpu=not cfg['fixture']))
        bundle = load_dataset(bundle_path(cfg))
        folder = attempt_path/'profile'
        result = profile_response(bundle, cfg, folder, worker_setup_seconds=time.monotonic()-started)
    elif phase_id == 'followup-diagnostics':
        from .response_diagnostics import run_response_diagnostics
        verify_completed(state['completed']['followup-profile'])
        phases = read_json(work/'plan.json')['phases']
        include_child = all(p['phase_id'] in state['completed'] for p in phases)
        if state['completed']['followup-profile']['result']['admission']['admitted'] and not include_child:
            raise RuntimeError('Admitted child diagnostics require the complete declared optimization matrix')
        folder = attempt_path/'diagnostics'
        result = run_response_diagnostics(cfg, folder, include_child=include_child)
    else:
        from .training import train_phase
        phase = next((p for p in read_json(work/'plan.json')['phases'] if p['phase_id'] == phase_id), None)
        if phase is None:
            raise ValueError('Unregistered response follow-up phase')
        for dependency in phase['depends_on']:
            if dependency not in state['completed']:
                raise RuntimeError(f'Incomplete follow-up prerequisite: {dependency}')
            verify_completed(state['completed'][dependency])
        profile = state['completed']['followup-profile']
        verify_completed(profile)
        if not profile['result']['admission']['admitted']:
            raise RuntimeError('The complete declared optimization matrix was not admitted')
        local = copy.deepcopy(cfg)
        local['response']['calibration_receipt'] = profile['result']['calibration_receipt']
        bundle = load_dataset(bundle_path(cfg))
        upstream = (state['completed'][phase['depends_on'][0]]['result']['checkpoint']
                    if phase['phase'] == 'readout' else None)
        previous = [a for a in state['attempts'] if a['phase_id'] == phase_id and a['status'] == 'failed'
                    and (Path(a['path'])/'fit/checkpoint.pt').is_file()]
        if previous:
            local['training']['resume_from'] = str(Path(previous[-1]['path'])/'fit/checkpoint.pt')
        folder = attempt_path/'fit'
        result = dict(train_phase(bundle, phase['recipe'], phase['phase'], phase['seed'], local, folder, upstream))
    verify_parent(cfg)
    return result, folder


def _remaining_deadline(cfg, state, max_jobs):
    """Before allocating work, retain time for the rest of the admitted dependency graph."""
    if cfg['fixture']:
        return
    profile = state['completed'].get('followup-profile', {}).get('result')
    if not profile or not profile['admission']['admitted']:
        return
    phases = read_json(Path(cfg['work'])/'plan.json')['phases']
    pending = [p for p in phases if p['phase_id'] not in state['completed']]
    durations = profile['admission']['predicted_phase_seconds']
    finish, lanes = {}, [0.]*max_jobs
    for phase in pending:
        lane = min(range(max_jobs), key=lanes.__getitem__)
        prerequisite = max([finish.get(k, 0.) for k in phase['depends_on']] or [0.])
        finish[phase['phase_id']] = lanes[lane] = max(lanes[lane], prerequisite)+durations[phase['phase_id']]
    needed = max(lanes)/3600 + (0. if 'followup-diagnostics' in state['completed'] else 4.) + 2. + cfg['followup'].get('evaluation_allowance_hours', 0.)
    left = (_time(cfg['followup']['deadline_utc'])-datetime.now(timezone.utc)).total_seconds()/3600
    if needed > left:
        raise RuntimeError('Remaining complete matrix, diagnostics and queue allowance no longer fit before the deadline')


def deadline_cancellations(cfg, state, *, now=None):
    """Only cancel retained numeric child allocation IDs; never touch parent jobs."""
    from .scheduler import update_attempt
    current = now or datetime.now(timezone.utc)
    if cfg['fixture'] or current < _time(cfg['followup']['deadline_utc'])-timedelta(seconds=30):
        return []
    cancelled = []
    for attempt in state['attempts']:
        identifier = str(attempt.get('job_id') or '')
        if attempt['status'] not in ACTIVE or attempt.get('deadline_cancel_requested') or not identifier.isdigit():
            continue
        # Plain scancel cancels pending allocations too. --signal only signals
        # running steps and can leave a late pending job in the queue.
        result = subprocess.run(['scancel', identifier], capture_output=True, text=True)
        update_attempt(cfg['work'], attempt['id'], dict(deadline_cancel_requested=True,
            deadline_cancel_returncode=result.returncode, deadline_cancel_stderr=result.stderr))
        cancelled.append(identifier)
    return cancelled


def _finish_stopped(cfg, state, reason, *, refused=False):
    work = Path(cfg['work'])
    plan = read_json(work/'plan.json')
    result = dict(status='FOLLOWUP_MATRIX_NOT_ADMITTED' if refused else 'FOLLOWUP_INCOMPLETE',
                  reason=str(reason), deadline_utc=cfg['followup']['deadline_utc'],
                  completed_optimization_phases=sum(p['phase_id'] in state['completed'] for p in plan['phases']),
                  required_optimization_phases=len(plan['phases']), complete_matrix=False,
                  budget_categories=budget_usage(state), parent_unchanged=True,
                  diagnostics=state['completed'].get('followup-diagnostics',{}).get('result'))
    atomic_json(work/'followup-status.json', result)
    (work/'report.md').write_text('# JEPA response follow-up\n\n'
        f"Status: **{result['status']}**. {reason}\n\n"
        f"Completed {result['completed_optimization_phases']} of {len(plan['phases'])} optimization phases. "
        'This is an incomplete experiment and does not establish a comparative scientific result. '
        'The parent remains a read-only dependency. Retained attempts and checkpoints are available for inspection.\n')
    return result


def run_followup(cfg, *, local=False, max_jobs=8):
    from .scheduler import (_state, _reconcile, _submit, reserve, execute_worker,
                            complete_attempt, update_attempt)
    work = Path(cfg['work'])
    verify_parent(cfg)
    phases = read_json(work/'plan.json')['phases']
    stop_reason = None
    while True:
        if not local:
            _reconcile(cfg)
        state = _state(work)
        active = {a['phase_id'] for a in state['attempts'] if a['status'] in ACTIVE}
        if not local:
            deadline_cancellations(cfg, state)
            if datetime.now(timezone.utc) >= _time(cfg['followup']['deadline_utc'])-timedelta(seconds=120):
                stop_reason = 'Registered absolute deadline reached; no further workload admitted'
        if stop_reason:
            if not active:
                return _finish_stopped(cfg, state, stop_reason)
            time.sleep(min(20, cfg['resources']['poll_seconds']))
            continue
        profile = state['completed'].get('followup-profile', {}).get('result')
        if profile is None:
            ready = [] if 'followup-profile' in active else ['followup-profile']
        elif not profile['admission']['admitted']:
            if 'followup-diagnostics' in state['completed']:
                return _finish_stopped(cfg, state, 'Profile did not admit the complete matrix; baseline diagnostics retained', refused=True)
            ready = [] if 'followup-diagnostics' in active else ['followup-diagnostics']
        elif all(p['phase_id'] in state['completed'] for p in phases):
            if 'followup-diagnostics' in state['completed']:
                from .response_evaluation import evaluate_response_followup
                return evaluate_response_followup(cfg)
            ready = [] if 'followup-diagnostics' in active else ['followup-diagnostics']
        else:
            try:
                _remaining_deadline(cfg, state, max_jobs)
            except RuntimeError as exc:
                stop_reason = str(exc)
                continue
            ready = [p['phase_id'] for p in phases if p['phase_id'] not in state['completed']
                     and p['phase_id'] not in active and all(k in state['completed'] for k in p['depends_on'])]
        for phase_id in ready[:max(0, max_jobs-len(active))]:
            prior = sum(a['phase_id'] == phase_id for a in state['attempts'])
            if prior >= cfg['resources']['max_attempts']:
                raise RuntimeError(f'Follow-up phase exhausted its retained retries: {phase_id}')
            try:
                allocation_minutes(cfg, phase_id, prior)
            except RuntimeError as exc:
                stop_reason = str(exc)
                if not active:
                    return _finish_stopped(cfg, state, stop_reason, refused=bool(profile and not profile['admission']['admitted']))
                break
            if local:
                attempt = reserve(work, phase_id, 0., limit=0., max_jobs=max_jobs,
                                  budget_category=category(phase_id, prior), category_limits=BUDGETS)
                try:
                    execute_worker(cfg, phase_id, attempt['path'])
                    complete_attempt(work, attempt, dict(allocated_gpu_hours=0., scheduler_state='LOCAL_CPU'))
                except Exception:
                    update_attempt(work, attempt['id'], dict(status='failed', allocated_gpu_hours=0.))
                    raise
            else:
                _submit(cfg, phase_id)
        if not local:
            time.sleep(min(60, cfg['resources']['poll_seconds']))


def verify_followup(cfg):
    from .scheduler import _verify_frozen, _state, verify_completed
    _verify_frozen(cfg)
    binding = verify_parent(cfg)
    state = _state(cfg['work'])
    artifacts = 0
    for value in state['completed'].values():
        verify_completed(value)
        artifacts += len(read_json(value['receipt'])['artifacts'])
    from .response_evaluation import verify_response_followup
    metrics = verify_response_followup(cfg)
    return dict(status='FOLLOWUP_ARTIFACTS_VERIFIED', parent_unchanged=True,
                parent_files_checked=len(binding['files']), child_artifacts_checked=artifacts,
                completed_phases=len(state['completed']), evidence_status=binding['evidence_status'],
                metric_reconstruction=metrics, cuda_verified_here=False)
