#!/usr/bin/env python3
"""A bounded, separate HAIC expansion of the paired synthetic development study.

The original scientific modules and pilot are read-only. One Slurm allocation
screens a larger panel, prepares its tracks once, fits two training budgets at
three seeds, and runs the calibration and preservation checks after every fit.
"""
from __future__ import annotations

import argparse
from contextlib import nullcontext
from datetime import datetime, timezone
import math
import os
from pathlib import Path
import re
import signal
import subprocess
import sys
import time
import uuid

ROOT = Path(__file__).resolve().parents[3]
SCRIPTS = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / 'src'))

import haic
from gavd6_sjepa.research_directions.synthetic_training_v2.config import ARMS, RunConfig
from gavd6_sjepa.research_directions.synthetic_training_v2.contracts import (
    TrackBundle, atomic_json, code_identity, digest, sha256_file,
)

read = haic.read
write_new = haic.write_new


def now():
    return datetime.now(timezone.utc).isoformat()


def implementation_files():
    """Pin orchestration as well as the scientific code's separate identity."""
    paths = [SCRIPTS / f'{name}.py' for name in (
        'full_experiment', 'expanded_inputs', 'expansion_results', 'check_results',
        'haic', 'haic_inputs', 'automated_inputs', 'review_motion', 'postrun_checks', 'run', 'prepare')]
    paths += list((SCRIPTS / 'diagnostics').glob('*.py'))
    paths += [ROOT / 'slurm/synthetic-training-v2' / name for name in
              ('full-experiment.sh', 'full-experiment.sbatch', 'common.sh', 'postrun-checks.sbatch')]
    return {str(path.relative_to(ROOT)): sha256_file(path) for path in sorted(paths)}


def assert_request_unchanged(request):
    if request['code_identity'] != code_identity(ROOT) or request['implementation'] != implementation_files():
        raise ValueError('Scientific or launcher code changed after submission. Preserve this checkout until the job ends.')
    protocol = ROOT / 'docs/studies/synthetic-training-v2/protocol.md'
    if sha256_file(protocol) != request['protocol_sha256']:
        raise ValueError('The frozen scientific protocol changed after submission.')
    for name, expected in request['pilot_files'].items():
        if sha256_file(name) != expected:
            raise ValueError(f'A retained pilot input changed: {name}')


def output_for(pilot, name):
    if not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9._-]{2,60}', name):
        raise ValueError('Use a name of 3–61 letters, digits, dots, underscores or hyphens.')
    output = Path(pilot).resolve().parent / name
    if output == Path(pilot).resolve():
        raise ValueError('The expansion must have a different name from the pilot.')
    return output


def estimate_limits(cfg, prepared, snapshot, prepare_job, train_people, dev_people, windows, updates):
    """Conservative projections from this pilot, never fixture throughput."""
    bundle = TrackBundle.load(cfg.bundle)
    train_windows = {(r['canonical_person_id'], r['window_id']) for r in bundle.records if r['split'] == 'train'}
    dev_windows = {(r['canonical_person_id'], r['window_id']) for r in bundle.records if r['split'] == 'development'}
    prior_clips = 3 * len(train_windows) + 4 * len(dev_windows)
    new_clips = (3 * train_people + 4 * dev_people) * windows
    if not prior_clips:
        raise ValueError('The pilot has no source windows for a preparation projection.')
    prep_seconds = float(snapshot[prepare_job['job_id']]['gpu_seconds'])
    if not math.isfinite(prep_seconds) or prep_seconds <= 0:
        raise ValueError('A positive measured pilot preparation allocation is required.')
    # Allocation overhead, estimator loading and full provenance scans are
    # included in the measured preparation allocation, not just rendering.
    preparation = max(600., 3 * prep_seconds * new_clips / prior_clips + 180.)
    per_budget = {}
    for count in updates:
        direct, jepa = 0., 0.
        for arm in ARMS:
            records = [read(cfg.root / 'fits' / f'{arm}-{seed}' / 'training.json') for seed in cfg.seeds]
            seconds = max(float(row['elapsed_seconds']) for row in records)
            if not math.isfinite(seconds) or seconds <= 0:
                raise ValueError(f'Invalid measured fitting time for {arm}.')
            old_steps = cfg.readout_updates if arm == 'initialized' else cfg.updates + cfg.readout_updates
            new_steps = count if arm == 'initialized' else 2 * count
            if arm in {'smoothnet', 'direct', 'static', 'initialized'}:
                direct += seconds * new_steps / old_steps
            else:
                jepa += seconds * new_steps / old_steps
        # Fixed minibatch size bounds training work; larger evaluation panels,
        # startup and checkpoints get both a multiplier and a fixed allowance.
        per_budget[str(count)] = max(300., 4 * max(direct, jepa) + 180.)
    return dict(preparation_seconds=preparation, fit_stage_seconds=per_budget,
                pilot_render_clips=prior_clips, planned_render_clips=new_clips,
                basis='3x measured preparation allocation scaled by clips; 4x measured fit time scaled by updates, plus startup allowances. Projections are not completion guarantees.')


def previous_expansion_costs(pilot, output):
    """Import every sibling attempt once, including failures; block concurrency."""
    jobs = []
    for path in sorted(pilot.parent.glob('*/request.json')):
        if path.parent == output:
            continue
        request = read(path)
        if request.get('kind') != 'stv2-full-experiment-v1' or request.get('pilot_work') != str(pilot):
            continue
        submission = read(path.parent / 'submission.json')
        if not submission.get('job_id'):
            raise ValueError(f'Unresolved earlier expansion submission: {path.parent}. Reconcile its unique job name before another allocation.')
        jobs.append(dict(job_id=submission['job_id'], output=str(path.parent)))
    snapshot = haic.scheduler_snapshot(jobs)
    entries = []
    for job in jobs:
        row = snapshot.get(job['job_id'])
        if row is None or row['state'] not in haic.TERMINAL:
            raise ValueError(f'Earlier expansion {job["job_id"]} is active or awaiting accounting. Use its status first.')
        resources = dict(item.split('=', 1) for item in row['allocation'].split(',') if '=' in item)
        typed = [int(value) for key, value in resources.items() if key.startswith(('gres/gpu:', 'gres/gpu/'))]
        count = int(resources['gres/gpu']) if 'gres/gpu' in resources else sum(typed)
        if count != 1 or (typed and sum(typed) != count):
            raise ValueError(f'Cannot certify one GPU for earlier expansion {job["job_id"]}: {row["allocation"]}')
        entries.append(dict(stage='previous_expansion_allocation', slurm_job=job['job_id'],
                            output=job['output'], gpu_seconds=row['gpu_seconds'] + 1,
                            state=row['state'], source='sacct allocation elapsed, including failed/partial attempts; one second rounding allowance'))
    return entries


def reservation_sources(pilot, preparation):
    """Carry ALL previous reservation knowledge, not only the four chosen people."""
    screen_path = Path(preparation['locomotion_audit']).parent / 'screen.json'
    screen = read(screen_path)
    for row in screen['reservation_sources']:
        if sha256_file(row['path']) != row['sha256']:
            raise ValueError(f'Historical reservation source changed: {row["path"]}. '
                             'Retain its protected identities before a new panel can be selected.')
    paths = [Path(row['path']) for row in screen['reservation_sources']]
    paths.append(Path(preparation['reservation_csv']))
    paths.extend(path for path in (pilot / 'inputs/review-drafts').glob('person-reservations*.csv'))
    # A disappeared known source must not turn a protected identity into unknown.
    sources = {str(path.expanduser().resolve(strict=True)): sha256_file(path)
               for path in dict.fromkeys(paths)}
    return screen_path, sources


def launch(args):
    from check_results import check_results
    pilot = Path(args.pilot_work).expanduser().resolve()
    output = output_for(pilot, args.name)
    if not math.isfinite(args.gpu_hours) or not 0 < args.gpu_hours <= 48:
        raise ValueError('--gpu-hours must be an explicit, positive additional allocation cap, at most 48.')
    if not (2 <= args.train_people <= 128 and 2 <= args.development_people <= 32
            and 2 <= args.windows_per_person <= 8 and 4 <= args.max_candidates <= 4096):
        raise ValueError('Invalid panel size: train 2–128, development 2–32, windows 2–8, candidates 4–4096.')
    if args.max_candidates < args.train_people + args.development_people:
        raise ValueError('max-candidates must cover the requested number of people.')
    updates = sorted(set(args.updates))
    if len(updates) != len(args.updates):
        raise ValueError('Training budgets must be unique.')
    # A dry run is read-only. Actual launches serialize across ALL names for
    # this pilot, so a second allocation cannot spend an unaccounted balance.
    lock = nullcontext() if args.dry_run else haic.stage_lock(pilot, 'full-expansion-manage')
    with lock:
        if (output / 'request.json').exists():
            print(f'Expansion already recorded: {output}. No duplicate job submitted; use status.')
            return output
        if output.exists():
            raise FileExistsError(f'Output already exists without a request: {output}. Use a new --name.')
        verification = check_results(pilot)
        state = haic.state_for(pilot)
        snapshot = haic.settled(state)
        entries = haic.ledger_entries(state, snapshot) + previous_expansion_costs(pilot, output)
        prior_hours = math.fsum(row['gpu_seconds'] for row in entries) / 3600
        # Slurm uses whole seconds. This exact hard cap is included in scope.
        allocation_seconds = math.ceil(args.gpu_hours * 3600)
        total_hours = prior_hours + allocation_seconds / 3600
        if total_hours > 48:
            raise PermissionError('Recorded prior costs plus this additional allocation exceed the 48 GPU-hour study ceiling.')
        cfg = RunConfig.load(state['source_config'])
        if set(cfg.arms) != set(ARMS):
            raise ValueError('Expansion needs a completed pilot containing all eight comparison arms.')
        for count in updates:
            RunConfig(run_id='expansion-validation', mode='source', device='cuda', bundle=cfg.bundle,
                      updates=count, readout_updates=count, seeds=tuple(args.seeds), authorized_gpu_hours=total_hours)
        if cfg.batch_size != 64:
            raise ValueError('This throughput projection requires the pilot batch size of 64.')
        prep_job = [job for job in state['jobs'] if job['phase'] == 'prepare'][-1]
        prepared = Path(prep_job['output'])
        preparation = read(prepared / 'preparation-provenance.json')['configuration']
        # Take the actual successful producer's configuration, not possibly
        # stale environment defaults or an edited preparation worksheet.
        preparation = dict(preparation)
        if preparation.get('review_mode') != 'automated_development':
            raise ValueError('This expansion is for the completed automated-development pilot.')
        screen_path, reservations = reservation_sources(pilot, preparation)
        from haic_inputs import check_inputs
        check_inputs(preparation, cfg.held_extractor)
        limits = estimate_limits(cfg, prepared, snapshot, prep_job, args.train_people,
                                 args.development_people, args.windows_per_person, updates)
        projected = limits['preparation_seconds'] + len(args.seeds) * sum(
            2 * value for value in limits['fit_stage_seconds'].values()) + 900
        if projected > allocation_seconds:
            raise PermissionError(f'Conservative projection is {projected / 3600:.2f} additional GPU hours; '
                                  f'--gpu-hours {args.gpu_hours:g} is too small. No job submitted.')
        request = dict(schema=1, kind='stv2-full-experiment-v1', created_utc=now(), pilot_work=str(pilot), output=str(output),
                       python=sys.executable, account=state['account'], partition=state['partition'],
                       train_people=args.train_people, development_people=args.development_people,
                       windows_per_person=args.windows_per_person, max_candidates=args.max_candidates,
                       updates=updates, seeds=list(args.seeds), held_extractor=cfg.held_extractor,
                       model=cfg.model, batch_size=64, prior_entries=entries,
                       allocation_seconds=allocation_seconds, authorized_gpu_hours=total_hours,
                       limits=limits, projection_including_allowance_seconds=projected,
                       pilot_bundle=cfg.bundle, pilot_verification=verification,
                       reservation_sources=list(reservations),
                       preparation=preparation, code_identity=code_identity(ROOT),
                       implementation=implementation_files(),
                       protocol_sha256=sha256_file(ROOT / 'docs/studies/synthetic-training-v2/protocol.md'),
                       pilot_files={**reservations, **{str(path): sha256_file(path) for path in (
                           Path(state['source_config']), cfg.root / 'identity.json',
                           Path(cfg.bundle) / 'manifest.json', prepared / 'preparation-provenance.json', screen_path)}},
                       evidence_status='automated-source-screen', confirmation_opened=False,
                       scope='Larger synthetic development panel; inherited frontal view and proxy landmarks. All pilot people excluded. No human review or real transfer claimed.')
        token = uuid.uuid4().hex[:12]
        command = sbatch_command(request, token)
        if args.dry_run:
            print(__import__('json').dumps(dict(request=request, command=command, submits_jobs=False), indent=2))
            return output
        output.mkdir(parents=True)
        write_new(output / 'request.json', request)
        write_new(output / 'submission.json', dict(status='submitting', token=token, command=command, request_digest=digest(request),
                                                  created_utc=now()))
        env = {k: v for k, v in os.environ.items() if not k.startswith('SBATCH_')}
        env.update(STV2_ROOT=str(ROOT), STV2_PYTHON=sys.executable)
        try:
            completed = subprocess.run(command, env=env, text=True, capture_output=True, check=True, timeout=90)
            job_id = completed.stdout.strip().split(';')[0]
            if not job_id.isdigit():
                raise ValueError(f'Unexpected sbatch response: {completed.stdout!r}')
        except Exception as error:
            detail = str(getattr(error, 'stderr', '') or '')
            atomic_json(output / 'submission.json', dict(status='uncertain', token=token, command=command, request_digest=digest(request),
                error=str(error), scheduler_stderr=detail, updated_utc=now(), action='Inspect squeue/sacct for the unique job name; repeating launch never resubmits this request.'))
            if detail:
                print(detail, file=sys.stderr)
            raise
        atomic_json(output / 'submission.json', dict(status='submitted', token=token, command=command, request_digest=digest(request),
                                                    job_id=job_id, updated_utc=now()))
        print(f'FULL_EXPERIMENT_SUBMITTED: {job_id}\nOutput: {output}\nLog: {output}/slurm-{job_id}.out')
        print(f'Additional hard allocation cap: {allocation_seconds / 3600:g} H100 hours; '
              f'recorded prior costs: {prior_hours:.4f} hours. No login-node controller is needed.')
        return output


def sbatch_command(request, token):
    seconds = request['allocation_seconds']
    hours, remainder = divmod(seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    output = Path(request['output'])
    return ['sbatch', '--parsable', '--no-requeue', '--export=ALL', f'--chdir={ROOT}',
            f'--account={request["account"]}', f'--partition={request["partition"]}',
            '--nodes=1', '--ntasks=1', '--gres=gpu:h100:1', '--cpus-per-task=8', '--mem=96G',
            f'--time={hours:02d}:{minutes:02d}:{seconds:02d}', f'--job-name=stv2-full-{token}',
            f'--output={output}/slurm-%j.out', str(ROOT / 'slurm/synthetic-training-v2/full-experiment.sbatch'),
            str(output / 'request.json'), digest(request)]


class Allocation:
    """A single allocation hard-capped by Slurm; CPU work also consumes its time."""
    def __init__(self, request):
        self.request = request
        self.started = time.monotonic()
        self.output = Path(request['output'])

    def elapsed(self):
        # Reserve startup/exit overhead in every scope; sacct supplies the final
        # authoritative total, never the sum of overlapping fit-stage receipts.
        return time.monotonic() - self.started + 90

    def remaining(self):
        return self.request['allocation_seconds'] - self.elapsed()

    def update(self, phase, **extra):
        atomic_json(self.output / 'status.json', dict(status='running', phase=phase, updated_utc=now(),
                    allocated_seconds_upper_estimate=self.elapsed(), **extra))
        print(f'{now()} {phase}', flush=True)

    def execute(self, phase, command):
        assert_request_unchanged(self.request)
        self.update(phase)
        remaining = self.remaining()
        if remaining <= 30:
            raise TimeoutError('Additional allocation cap exhausted; completed scopes remain retained.')
        # Each phase uses a fresh process so renderer/model memory is released.
        process = subprocess.Popen(command, cwd=ROOT, start_new_session=True)
        try:
            code = process.wait(timeout=remaining)
            if code:
                raise subprocess.CalledProcessError(code, command)
        except BaseException:
            if process.poll() is None:
                os.killpg(process.pid, signal.SIGTERM)
                try:
                    process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid, signal.SIGKILL)
                    process.wait()
            raise

    def scope(self, run_id, bundle, updates, seed, projected_seconds):
        if self.remaining() < 2 * projected_seconds + 60 and run_id != 'preparation-scope':
            raise PermissionError('Remaining allocation cannot cover both declared fitting-stage limits. '
                                  'Completed results are retained; no training budget is silently shortened.')
        entries = self.request['prior_entries'] + [dict(stage='current_allocation_before_scope',
            slurm_job=os.environ['SLURM_JOB_ID'], gpu_seconds=self.elapsed(),
            source='Elapsed allocation including CPU screening and analysis, plus 90 seconds startup/exit allowance; reconcile with sacct.')]
        ledger = self.output / 'config' / f'{run_id}-costs.json'
        write_new(ledger, dict(scope_authorized=True, entries=entries,
                             accounting='Cumulative snapshot. Do not sum snapshots or add overlapping fit receipts.'))
        cfg = RunConfig(run_id=run_id, output_root=str(self.output / 'runs'), mode='source',
            bundle=str(bundle), device='cuda', seed=seed, seeds=(seed,), updates=updates, readout_updates=updates,
            batch_size=self.request['batch_size'], model=self.request['model'],
            held_extractor=self.request['held_extractor'], authorized_gpu_hours=self.request['authorized_gpu_hours'],
            measured_gpu_hours=math.fsum(row['gpu_seconds'] for row in entries) / 3600,
            projected_gpu_hours=projected_seconds / 3600, cost_ledger=str(ledger))
        cfg.require_gpu_scope()
        path = self.output / 'config' / f'{run_id}.json'
        write_new(path, cfg.as_dict())
        return path


def run_request(path, expected_digest):
    request = read(path)
    output = Path(request['output'])
    if digest(request) != expected_digest:
        raise ValueError('Request changed after sbatch captured its digest; no experiment started.')
    if Path(path).resolve() != output / 'request.json' or request['python'] != sys.executable:
        raise ValueError('Use the immutable request and its saved interpreter.')
    if not os.environ.get('SLURM_JOB_ID'):
        raise PermissionError('The full experiment worker requires its submitted Slurm allocation.')
    submission = read(output / 'submission.json')
    if os.environ.get('SLURM_JOB_NAME') != f'stv2-full-{submission["token"]}':
        raise PermissionError('This is not the allocation submitted for this experiment.')
    if submission.get('job_id') and submission['job_id'] != os.environ['SLURM_JOB_ID']:
        raise PermissionError('Slurm job identity differs from the recorded submission.')
    with haic.stage_lock(output, 'full-worker'):
        if (output / 'status.json').exists():
            raise FileExistsError('This allocation attempt has already started. No automatic requeue or duplicate execution.')
        allocation = Allocation(request)
        try:
            assert_request_unchanged(request)
            allocation.execute('verify_allocated_environment', [sys.executable,
                str(ROOT / 'scripts/research_directions/synthetic_training/check_environment.py'), '--require-cuda'])
            # Verify the pilot again after queueing, before consuming its labels.
            allocation.execute('verify_pilot', [sys.executable, str(SCRIPTS / 'check_results.py'),
                                                '--work', request['pilot_work']])
            preparation_path = output / 'config/screening.json'
            write_new(preparation_path, request['preparation'])
            screened = output / 'inputs'
            allocation.execute('screen_new_people', [sys.executable, str(SCRIPTS / 'expanded_inputs.py'),
                '--config', str(preparation_path), '--output', str(screened),
                '--train-people', str(request['train_people']), '--development-people', str(request['development_people']),
                '--windows-per-person', str(request['windows_per_person']),
                '--max-candidates', str(request['max_candidates']), '--exclude-bundle', request['pilot_bundle'],
                *[item for path in request['reservation_sources'] for item in ('--reservation-csv', path)]])
            preparation = dict(request['preparation'], review_mode='automated_development',
                               locomotion_audit=str(screened / 'locomotion-audit.csv'),
                               reservation_csv=str(screened / 'person-reservations.csv'))
            from haic_inputs import check_inputs
            check_inputs(preparation, request['held_extractor'])
            paired = output / 'paired'
            limit = request['limits']['preparation_seconds']
            scope = allocation.scope('preparation-scope', paired / 'bundle', request['updates'][0],
                                     request['seeds'][0], limit)
            preparation['scope_config'] = str(scope)
            prepare_path = output / 'config/preparation.json'
            write_new(prepare_path, preparation)
            allocation.execute('prepare_expanded_tracks', [sys.executable, str(SCRIPTS / 'prepare.py'),
                                                          '--config', str(prepare_path), '--output', str(paired)])
            verify_panel(paired / 'bundle', request)
            configs, diagnostics = [], []
            # Each seed's short and long fits are paired, with useful results
            # emitted after the first run rather than held until the suite ends.
            for seed in request['seeds']:
                for updates in request['updates']:
                    run_id = f'updates-{updates:04d}-seed-{seed}'
                    scope = allocation.scope(run_id, paired / 'bundle', updates, seed,
                                             request['limits']['fit_stage_seconds'][str(updates)])
                    allocation.execute(f'fit_{run_id}', [sys.executable, str(SCRIPTS / 'run.py'),
                        '--config', str(scope), '--stage', 'all'])
                    diagnostic = output / 'diagnostics' / run_id
                    allocation.execute(f'analyze_{run_id}', [sys.executable, str(Path(__file__)),
                        '_analyze', '--config', str(scope), '--output', str(diagnostic)])
                    configs.append(str(scope)); diagnostics.append(str(diagnostic))
                    # Snapshot only completed analyses; report is explicitly
                    # partial until every declared budget/seed is present.
                    atomic_json(output / 'completed-scopes.json', dict(configs=configs, diagnostics=diagnostics,
                        expected_scopes=len(request['seeds']) * len(request['updates']), updated_utc=now()))
                    allocation.execute('refresh_partial_summary', [sys.executable, str(Path(__file__)),
                                       '_report', '--output', str(output)])
            assert_request_unchanged(request)
            atomic_json(output / 'status.json', dict(status='worker_complete', updated_utc=now(),
                completed_scopes=len(configs), output=str(output), report=str(output / 'report.md'),
                confirmation_opened=False, evidence_status='automated-source-screen',
                allocated_seconds_upper_estimate=allocation.elapsed(),
                scheduler='Final success additionally requires Slurm COMPLETED with exit 0:0.'))
            print(f'FULL_EXPERIMENT_WORKER_COMPLETE: {output}/report.md', flush=True)
        except BaseException as error:
            atomic_json(output / 'status.json', dict(status='failed', error=f'{type(error).__name__}: {error}',
                updated_utc=now(), allocated_seconds_upper_estimate=allocation.elapsed(),
                completed_results='See completed-scopes.json and runs/; no artifacts were overwritten.'))
            raise


def verify_panel(bundle_path, request):
    bundle = TrackBundle.load(bundle_path)
    bundle.validate(request['held_extractor'])
    old = TrackBundle.load(request['pilot_bundle'])
    old_people = {row['canonical_person_id'] for row in old.records}
    if old_people & {row['canonical_person_id'] for row in bundle.records}:
        raise ValueError('The expanded panel overlaps pilot people.')
    if bundle.evidence_status != 'automated-source-screen':
        raise ValueError('Automated preparation has an incorrect evidence label.')
    for split, wanted, variants, extractors in [('train', request['train_people'], 3, 2),
                                               ('development', request['development_people'], 4, 3)]:
        rows = [row for row in bundle.records if row['split'] == split]
        people = {row['canonical_person_id'] for row in rows}
        if len(people) != wanted or len(rows) != wanted * request['windows_per_person'] * variants * extractors:
            raise ValueError(f'The prepared {split} panel does not match the declared people/window/condition/extractor counts.')
        for person in people:
            if len({row['window_id'] for row in rows if row['canonical_person_id'] == person}) != request['windows_per_person']:
                raise ValueError(f'Window count differs for {person}.')
    atomic_json(Path(request['output']) / 'panel-verification.json', dict(
        status='expanded_panel_verified', train_people=request['train_people'],
        development_people=request['development_people'], physical_windows=(request['train_people'] + request['development_people']) * request['windows_per_person'],
        track_records=len(bundle.records), independent_development_people=request['development_people'],
        pilot_people_excluded=sorted(old_people), evidence_status=bundle.evidence_status))


def status(args):
    output = output_for(args.pilot_work, args.name)
    request = read(output / 'request.json')
    submission = read(output / 'submission.json')
    worker = read(output / 'status.json') if (output / 'status.json').exists() else dict(status='not_started')
    scheduler = None
    if submission.get('job_id'):
        scheduler = haic.scheduler_snapshot([dict(job_id=submission['job_id'])]).get(submission['job_id'])
    complete = (worker.get('status') == 'worker_complete' and scheduler is not None
                and scheduler['state'] == 'COMPLETED' and scheduler['exit_code'] == '0:0')
    result = dict(status='FULL_EXPERIMENT_COMPLETE' if complete else worker['status'],
                  output=str(output), submission=submission, worker=worker, scheduler=scheduler)
    if scheduler and scheduler['state'] in haic.TERMINAL:
        if scheduler['state'] != 'COMPLETED' or scheduler['exit_code'] != '0:0':
            result['status'] = 'allocation_failed'
        result['allocation_gpu_hours'] = scheduler['gpu_seconds'] / 3600
        result['total_recorded_gpu_hours'] = (math.fsum(row['gpu_seconds'] for row in request['prior_entries'])
                                               + scheduler['gpu_seconds']) / 3600
    if (output / 'completed-scopes.json').exists():
        result['completed_scopes'] = read(output / 'completed-scopes.json')
        result['report'] = str(output / 'report.md')
    print(__import__('json').dumps(result, indent=2))
    return result


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    commands = p.add_subparsers(dest='command', required=True)
    for name in ('launch', 'status'):
        sub = commands.add_parser(name)
        sub.add_argument('--pilot-work', type=Path, default=os.environ.get('STV2_WORK'))
        sub.add_argument('--name', default='full-01')
        if name == 'launch':
            sub.add_argument('--gpu-hours', required=True, type=float, help='Additional hard allocation cap, including CPU work while GPU is allocated.')
            sub.add_argument('--train-people', type=int, default=24)
            sub.add_argument('--development-people', type=int, default=8)
            sub.add_argument('--windows-per-person', type=int, default=4)
            sub.add_argument('--max-candidates', type=int, default=512)
            sub.add_argument('--updates', nargs='+', type=int, default=[200, 2000])
            sub.add_argument('--seeds', nargs='+', type=int, default=[17, 29, 43])
            sub.add_argument('--dry-run', action='store_true')
    worker = commands.add_parser('run')
    worker.add_argument('--request', type=Path, required=True)
    worker.add_argument('--expected-digest', required=True)
    analyze = commands.add_parser('_analyze', help=argparse.SUPPRESS)
    analyze.add_argument('--config', type=Path, required=True)
    analyze.add_argument('--output', type=Path, required=True)
    report = commands.add_parser('_report', help=argparse.SUPPRESS)
    report.add_argument('--output', type=Path, required=True)
    return p


def main(argv=None):
    p = parser(); args = p.parse_args(argv)
    if args.command in {'launch', 'status'} and args.pilot_work is None:
        p.error('Source the original pilot session.env or supply --pilot-work.')
    try:
        if args.command == 'launch': launch(args)
        elif args.command == 'status': status(args)
        elif args.command == 'run': run_request(args.request, args.expected_digest)
        elif args.command == '_analyze':
            from expansion_results import run_analysis
            run_analysis(args.config, args.output)
        elif args.command == '_report':
            from expansion_results import report_suite
            completed = read(args.output / 'completed-scopes.json')
            count = len(completed['configs'])
            summary = args.output / 'summaries' / f'completed-{count:02d}'
            report_suite(completed['configs'], completed['diagnostics'], summary)
            banner = (f'Completed **{count} of {completed["expected_scopes"]}** declared training scopes. '
                      'Final execution success also requires Slurm COMPLETED with exit 0:0.\n\n')
            temporary = args.output / 'report.md.tmp'
            temporary.write_text(banner + (summary / 'report.md').read_text())
            temporary.replace(args.output / 'report.md')
            atomic_json(args.output / 'latest-summary.json', dict(output=str(summary), completed_scopes=count,
                        expected_scopes=completed['expected_scopes']))
    except (OSError, ValueError, RuntimeError, PermissionError, AssertionError, KeyError, subprocess.SubprocessError) as error:
        print(f'FULL_EXPERIMENT: {type(error).__name__}: {error}', file=sys.stderr)
        return 1
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
