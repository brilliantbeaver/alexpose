"""Bounded HAIC execution for the isolated repair extension.

Reservations precede submission; ambiguous submissions keep their reservation.
Only jobs owned by this work directory can be cancelled at its frozen cutoff.
"""
from __future__ import annotations

import importlib.util
import math
import os
from pathlib import Path
import re
import subprocess
import time
import uuid

from .common import atomic_json, locked, read_json, sha256, utc_now
from .repair import ACTIVE, _deadline, build_plan, evaluate, execute_phase, load_repair
from .scheduler import (_state, _reconcile, complete_attempt, reserve,
                        update_attempt, verify_completed)


def phase_plan(cfg, stage):
    if stage == 'development':
        return build_plan()['phases']
    if stage == 'benchmark':
        return [dict(phase_id='benchmark', depends_on=[], allocation_minutes=120)]
    if stage == 'confirmation':
        shards = [dict(phase_id=f'confirmation-shard-{i:02d}', depends_on=[],
                       allocation_minutes=cfg['resources']['prepare_wall_minutes'])
                  for i in range(cfg['resources']['max_jobs'])]
        return shards + [dict(phase_id='confirmation-evaluate',
                              depends_on=[s['phase_id'] for s in shards], allocation_minutes=120)]
    raise ValueError('Unknown repair stage')


def _submission(cfg, phase):
    """One GPU, one committed reservation; never retry an uncertain sbatch."""
    remaining = _deadline(cfg, grace=120)
    minutes = int(phase['allocation_minutes'])
    if remaining < (minutes + 2) * 60:
        raise RuntimeError(f"Insufficient time before cutoff for {phase['phase_id']} allocation")
    r = cfg['resources']; work = Path(cfg['work'])
    attempt = reserve(work, phase['phase_id'], minutes/60,
                      limit=r['gpu_hours'], max_jobs=r['max_jobs'])
    folder = Path(attempt['path']); folder.mkdir(parents=True)
    command = ['sbatch', '--parsable', f"--job-name={attempt['job_name']}",
        f"--account={r['account']}", f"--partition={r['partition']}",
        '--nodes=1', '--ntasks=1', '--export=ALL', '--no-requeue',
        f"--gres=gpu:{r['gpu']}", f'--time={minutes}',
        f"--cpus-per-task={r['cpu_workers']}", f"--mem={r['memory']}",
        f'--output={folder}/slurm-%j.out', f"--chdir={cfg['code_root']}",
        str(Path(cfg['code_root'])/'slurm/gait-fidelity/repair-worker.sbatch'),
        str(work), phase['phase_id'], str(folder)]
    receipt = dict(command=command, state='submitting', requested_utc=utc_now())
    atomic_json(folder/'submission.json', receipt)
    # Even a nonzero return can follow scheduler acceptance. The unique name
    # remains in the ledger and the ordinary scheduler reconciliation finds it.
    try:
        response = subprocess.run(command, capture_output=True, text=True, check=False)
    except OSError as error:
        atomic_json(folder/'submission.json', dict(receipt, error=str(error)))
        raise RuntimeError('Submission unresolved; reservation retained') from error
    receipt.update(stdout=response.stdout, stderr=response.stderr, returncode=response.returncode)
    atomic_json(folder/'submission.json', receipt)
    match = re.fullmatch(r'(\d+)(?:;[^\s;]+)?', response.stdout.strip())
    if response.returncode or match is None:
        raise RuntimeError('Submission unresolved; preserve ledger and reconcile its unique job name')
    update_attempt(work, attempt['id'], dict(job_id=match.group(1), status='submitted'))
    return match.group(1)


def cancel_owned(cfg):
    """No user-wide, name-prefix or partition cancellation is permitted."""
    cancelled, unresolved, errors = [], [], []
    for attempt in _state(cfg['work'])['attempts']:
        if attempt['status'] not in ACTIVE:
            continue
        job_id = str(attempt.get('job_id') or '')
        if not job_id.isdigit():
            unresolved.append(attempt['job_name'])
            continue
        result = subprocess.run(['scancel', job_id], capture_output=True, text=True, check=False)
        if result.returncode:
            errors.append(dict(job_id=job_id, error=result.stderr.strip()))
        else:
            cancelled.append(job_id)
        # Preserve the reservation until sacct reports the actual allocation.
    receipt = dict(reason='frozen_cutoff', cancelled_job_ids=cancelled,
                   unresolved_job_names=unresolved, cancellation_errors=errors, recorded_utc=utc_now())
    atomic_json(Path(cfg['work'])/'deadline-stop.json', receipt)
    return receipt


def _merge_confirmation(cfg):
    from .preparation import merge_datasets
    work = Path(cfg['work']); marker = work/'confirmation/merged.json'
    state = _state(work); paths = []
    for phase in phase_plan(cfg, 'confirmation')[:-1]:
        completed = state['completed'][phase['phase_id']]
        verify_completed(completed)
        value = completed['result']
        paths.append(value['bundle'] if value['bundle'] else
                     str(Path(completed['receipt']).parent/'result/repair-preparation-status.json'))
    if marker.exists():
        value = read_json(marker)
        if value['shard_paths'] != paths:
            raise RuntimeError('Merged confirmation source receipts changed')
        for file, expected in value['artifacts'].items():
            if sha256(file) != expected:
                raise RuntimeError('Merged confirmation artifact changed')
        return value
    destination=work/'confirmation/bundle'
    if destination.exists():
        destination.rename(destination.with_name(f'bundle-unpublished-{uuid.uuid4().hex[:12]}'))
    bundle = merge_datasets(paths, destination, allow_confirmation=True)
    value = dict(bundle=str(bundle), shard_paths=paths,
                 artifacts={str(p.resolve()):sha256(p) for p in Path(bundle).rglob('*') if p.is_file()})
    atomic_json(marker, value)
    return value


def _confirmation_admission(cfg):
    from .repair_cohort import verify_slim_lock
    from .repair_profile import admit_confirmation
    verify_slim_lock(cfg)
    completed = _state(cfg['work'])['completed'].get('benchmark')
    if completed is None:
        raise RuntimeError('Complete the real rendering/extraction benchmark before confirmation admission')
    verify_completed(completed)
    admission = admit_confirmation(cfg, completed['result'])
    atomic_json(Path(cfg['work'])/'confirmation/admission.json', admission)
    if not admission['admitted']:
        raise RuntimeError('Confirmation did not fit the measured remaining budget/deadline; inspect admission.json')
    return admission


def finished_stage(cfg, stage):
    """A completed stage is read-only, including after its execution cutoff."""
    work = Path(cfg['work']); state = _state(work)
    phases = phase_plan(cfg, stage)
    if not all(p['phase_id'] in state['completed'] for p in phases):
        return None
    for phase in phases: verify_completed(state['completed'][phase['phase_id']])
    if stage == 'development':
        if not (work/'development-complete.json').is_file(): return None
        from .repair import verify
        verify(cfg)
        return read_json(work/'development-complete.json')['result']
    return state['completed'][phases[-1]['phase_id']]['result']


def run(cfg, *, stage='development', local=False):
    if local != bool(cfg['fixture']):
        raise ValueError('Local execution is restricted to software fixtures')
    if not local and not os.environ.get('SLURM_JOB_ID'):
        raise RuntimeError('Submit a CPU coordinator through repair.sh launch')
    work = Path(cfg['work']); phases = phase_plan(cfg, stage)
    load_repair(work, verify_dependencies=True)
    with locked(work/'locks/controller.lock', nonblocking=True):
        finished = finished_stage(cfg, stage)
        if finished is not None: return finished
        try:
            _deadline(cfg, grace=120)
        except RuntimeError:
            if not local: cancel_owned(cfg)
            raise
        if stage == 'confirmation':
            _confirmation_admission(cfg)
        while True:
            try:
                _deadline(cfg, grace=120)
            except RuntimeError:
                if not local: cancel_owned(cfg)
                raise
            if not local:
                _reconcile(cfg)
            state = _state(work)
            if all(p['phase_id'] in state['completed'] for p in phases):
                for phase in phases: verify_completed(state['completed'][phase['phase_id']])
                break
            active = {a['phase_id'] for a in state['attempts'] if a['status'] in ACTIVE}
            ready = [p for p in phases if p['phase_id'] not in state['completed'] and p['phase_id'] not in active
                     and all(k in state['completed'] for k in p['depends_on'])]
            for phase in ready[:max(0, cfg['resources']['max_jobs']-len(active))]:
                previous = [a for a in state['attempts'] if a['phase_id'] == phase['phase_id']]
                if len(previous) >= cfg['resources']['max_attempts']:
                    raise RuntimeError(f"{phase['phase_id']} exhausted its retries; retained attempts require inspection")
                if phase['phase_id'] == 'confirmation-evaluate':
                    _merge_confirmation(cfg)
                if local:
                    attempt = reserve(work, phase['phase_id'], 0., limit=0., max_jobs=cfg['resources']['max_jobs'])
                    try:
                        execute_phase(cfg, phase['phase_id'], attempt['path'])
                        complete_attempt(work, attempt, dict(allocated_gpu_hours=0., scheduler_state='LOCAL_CPU'))
                    except Exception:
                        update_attempt(work, attempt['id'], dict(status='failed', allocated_gpu_hours=0.))
                        raise
                else:
                    _submission(cfg, phase)
            if not local:
                time.sleep(min(30, max(1, cfg['resources'].get('poll_seconds', 20))))
        if stage == 'development':
            if (work/'development-complete.json').exists():
                from .repair import verify
                verify(cfg)
                return read_json(work/'development-complete.json')['result']
            return evaluate(cfg, split='development')
        if stage == 'benchmark':
            return _state(work)['completed']['benchmark']['result']
        return _state(work)['completed']['confirmation-evaluate']['result']


def launch(cfg, *, stage='development'):
    """Submit a restartable CPU coordinator using the existing reconciliation rule."""
    if cfg['fixture']:
        raise ValueError('Fixtures run locally and cannot be submitted to HAIC')
    load_repair(cfg['work'], verify_dependencies=True)
    finished = finished_stage(cfg, stage)
    if finished is not None:
        return dict(state='stage_complete', stage=stage, duplicate_submitted=False, result=finished)
    remaining = _deadline(cfg, grace=180)
    if stage == 'confirmation': _confirmation_admission(cfg)
    # Reuse the established Slurm finished/ambiguous-state handling.
    path = Path(cfg['code_root'])/'slurm/gait-fidelity/submit.py'
    spec = importlib.util.spec_from_file_location('gf_original_submit', path)
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    work = Path(cfg['work']); control = work/'control'; logs = work/'logs'
    control.mkdir(exist_ok=True); logs.mkdir(exist_ok=True)
    receipt = control/'coordinator.json'
    with locked(control/'submit.lock', nonblocking=True):
        previous = read_json(receipt) if receipt.exists() else None
        if previous and previous.get('state') == 'submitting':
            raise RuntimeError(f"Unresolved coordinator {previous['job_name']}; no duplicate submitted")
        if previous and previous.get('job_id') and module.active_job(previous['job_id']):
            return dict(previous, requested_stage=stage, duplicate_submitted=False,
                        requested_stage_queued=previous['stage'] == stage)
        name = f'gf-repair-controller-{uuid.uuid4().hex[:12]}'
        log = logs/f'{name}-%j.out'; r = cfg['resources']
        command = ['sbatch', '--parsable', f'--job-name={name}', f"--account={r['account']}",
            f"--partition={r['partition']}", '--nodes=1', '--ntasks=1', '--cpus-per-task=2',
            f"--mem={r.get('controller_memory','64G')}", f'--time={math.ceil(remaining/60)+1}',
            '--export=ALL', '--no-requeue', f'--output={log}',
            str(Path(cfg['code_root'])/'slurm/gait-fidelity/repair-coordinator.sbatch'), str(work), stage]
        value = dict(state='submitting', stage=stage, job_name=name, command=command,
                     requested_utc=utc_now(), work=str(work), log_pattern=str(log))
        atomic_json(receipt, value)
        result = subprocess.run(command, capture_output=True, text=True, check=False)
        value.update(stdout=result.stdout, stderr=result.stderr, returncode=result.returncode)
        atomic_json(receipt, value)
        match = re.fullmatch(r'(\d+)(?:;[^\s;]+)?', result.stdout.strip())
        if result.returncode or match is None:
            raise RuntimeError('Coordinator submission unresolved; reconcile the recorded name before relaunching')
        value.update(state='submitted', job_id=match.group(1), log=str(log).replace('%j',match.group(1)))
        atomic_json(receipt, value); atomic_json(control/f'{name}.json', value)
        return value
