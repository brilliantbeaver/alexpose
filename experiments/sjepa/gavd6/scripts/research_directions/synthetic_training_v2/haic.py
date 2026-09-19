#!/usr/bin/env python3
"""Manage one HAIC development run without hand-written JSON or phase exports."""
from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import re
import shlex
import subprocess
import sys
import uuid

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / 'src'))
from gavd6_sjepa.research_directions.synthetic_training_v2.config import RunConfig, STAGES
from haic_inputs import preparation_config, create_drafts, check_inputs

TERMINAL = {'COMPLETED', 'FAILED', 'CANCELLED', 'TIMEOUT', 'OUT_OF_MEMORY',
            'NODE_FAIL', 'BOOT_FAIL', 'PREEMPTED', 'DEADLINE', 'REVOKED'}


def atomic_json(path, value):
    # Keep help/environment diagnostics usable before scientific packages import.
    from gavd6_sjepa.research_directions.synthetic_training_v2.contracts import atomic_json as write
    write(path, value)


def stage_lock(root, stage):
    from gavd6_sjepa.research_directions.temporal_gait.contracts import stage_lock as lock
    return lock(root, stage)


def read(path):
    return json.loads(Path(path).read_text())


def write_new(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x') as handle:
        json.dump(value, handle, indent=2, allow_nan=False)
        handle.write('\n')


def finite_hours(value):
    number = float(value)
    if not math.isfinite(number) or number < 0:
        raise ValueError('GPU hours must be finite and nonnegative.')
    return number


def environment_ok():
    command = [sys.executable, str(ROOT / 'scripts/research_directions/synthetic_training/check_environment.py')]
    return subprocess.run(command, check=False).returncode == 0


def initialize(args, env=os.environ):
    # A new name deliberately avoids adopting half-configured or already frozen runs.
    if not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9._-]{2,70}', args.name):
        raise ValueError('Use a run name containing letters, digits, dot, underscore, or hyphen.')
    work = ROOT / 'outputs/synthetic-training-v2' / args.name
    if work.exists():
        raise FileExistsError(f'Run directory already exists: {work}. Source its session.env, or choose a new name.')
    if not environment_ok():
        raise ValueError('No run files created. Select the verified Torch 2.6.0+cu124 interpreter with STV2_PYTHON '
                         '(usually /hai/scratch/$USER/envs/synthetic-training-cu124/bin/python), or follow the original environment repair.')
    budget = finite_hours(args.gpu_hours)
    if not 0 < budget <= 48:
        raise ValueError('Explicit authorized budget must be greater than zero and at most 48 GPU hours.')
    if args.prior_ledger:
        prior = read(args.prior_ledger)['entries']
        if not isinstance(prior, list):
            raise ValueError('Prior ledger entries must be a list.')
        prior = [dict(row) for row in prior]
        for row in prior:
            row['gpu_seconds'] = finite_hours(row['gpu_seconds'])
    else:
        prior = [dict(stage='prior_scope_costs', gpu_seconds=finite_hours(args.prior_gpu_hours) * 3600,
                      source='explicit user total; include earlier setup and failed allocations charged to this scope')]
    if math.fsum(row['gpu_seconds'] for row in prior) >= budget * 3600:
        raise ValueError('The prior costs leave no authorized GPU time.')
    settings = dict(name=args.name, work=str(work), python=sys.executable,
                    authorized_gpu_hours=budget, prior_entries=prior,
                    updates=args.updates, readout_updates=args.updates,
                    seeds=args.seeds, held_extractor='vitpose', batch_size=64,
                    account=env.get('ST_ACCOUNT', 'mind'), partition=env.get('ST_PARTITION', 'hai'),
                    jobs=[], source_config=None, pending_submission=None)
    # Validate recipe bounds before creating any run files.
    RunConfig(run_id='recipe-check', mode='source', device='cuda', bundle='future-bundle',
              updates=args.updates, readout_updates=args.updates, seeds=tuple(args.seeds),
              authorized_gpu_hours=budget)
    config = preparation_config(ROOT, work, env)
    work.mkdir(parents=True)
    write_new(work / 'config/preparation.json', config)
    write_new(work / 'control.json', settings)
    with (work / 'session.env').open('x') as handle:
        for key, value in dict(STV2_ROOT=str(ROOT), STV2_WORK=str(work), STV2_PYTHON=sys.executable).items():
            handle.write(f'export {key}={shlex.quote(value)}\n')
        handle.write(f'source {shlex.quote(str(ROOT / "slurm/synthetic-training-v2/study.env"))}\n')
    print(f'RUN_INITIALIZED: {work}')
    print(f'Source this after each login: source {shlex.quote(str(work / "session.env"))}')
    try:
        print('Review worksheets:', create_drafts(config, work))
    except (OSError, ValueError, KeyError) as error:
        print(f'Run saved, but worksheets need corrected manifest paths: {error}')
        print('Correct config/preparation.json, then run submit.sh inputs.')
    print('Complete inputs/locomotion-audit.csv and inputs/person-reservations.csv, then run submit.sh check.')
    return work


def state_for(work):
    path = Path(work) / 'control.json'
    if not path.is_file():
        raise FileNotFoundError(f'No managed run at {work}. Run init --name NAME first, then source its session.env.')
    state = read(path)
    if Path(state['work']).resolve() != Path(work).resolve():
        raise ValueError('Run directory moved; saved paths no longer match.')
    if state['python'] != sys.executable:
        raise ValueError(f'Source this run\'s session.env; saved interpreter is {state["python"]}.')
    return state


def save(state):
    atomic_json(Path(state['work']) / 'control.json', state)


def check(state):
    problems = []
    if not environment_ok():
        problems.append('Python environment failed. Use the original guide\'s environment repair; select the checked interpreter before init.')
    from gavd6_sjepa.research_directions.synthetic_training_v2.contracts import verify_preservation
    from gavd6_sjepa.research_directions.synthetic_training_v2.audit import reconstruct_pilot
    for label, function in (
        ('Historical evidence', lambda: (verify_preservation(ROOT), reconstruct_pilot(ROOT))),
        ('Source inputs', lambda: check_inputs(read(Path(state['work']) / 'config/preparation.json'), state['held_extractor'])),
    ):
        try:
            result = function()
            print(label + ': PASSED')
            if label == 'Source inputs':
                print(json.dumps(result, indent=2))
        except (OSError, ValueError, RuntimeError, KeyError) as error:
            problems.append(f'{label}: {error}')
    if problems:
        raise ValueError('\n\n'.join(problems))
    print('STV2_INPUTS_PASSED: ready for the allocated preparation check; GPU/render execution is still pending.')


def scheduler_snapshot(jobs):
    if not jobs:
        return {}
    identifiers = ','.join(row['job_id'] for row in jobs)
    output = subprocess.check_output(['sacct', '-D', '-n', '-P', '-j', identifiers,
        '--format=JobIDRaw,State,ExitCode,ElapsedRaw,AllocTRES'], text=True)
    result = {}
    for line in output.splitlines():
        fields = line.split('|')
        if len(fields) < 5 or not fields[0].isdigit():
            continue  # Do not also count .batch or .extern steps.
        identifier, status, exit_code, elapsed, tres = fields[:5]
        if identifier not in {row['job_id'] for row in jobs}:
            continue
        if identifier in result:
            raise RuntimeError(f'Multiple allocation records for job {identifier}. Reconcile requeue/restart costs before continuing; retain all sacct -D records.')
        seconds = float(elapsed)
        if not math.isfinite(seconds) or seconds < 0:
            raise ValueError(f'Invalid Slurm elapsed time for {identifier}.')
        result[identifier] = dict(state=status.split()[0].rstrip('+'), exit_code=exit_code,
                                  gpu_seconds=seconds, allocation=tres)
    return result


def settled(state):
    if state.get('pending_submission'):
        raise RuntimeError(f'An earlier sbatch response was uncertain: {state["pending_submission"]}. '
                           'Inspect squeue/sacct before any resubmission; retain control.json.')
    snapshot = scheduler_snapshot(state['jobs'])
    waiting = [row['job_id'] for row in state['jobs']
               if snapshot.get(row['job_id'], {}).get('state') not in TERMINAL]
    if waiting:
        raise RuntimeError('Jobs still active or accounting not yet available: ' + ', '.join(waiting) +
                           '. Run status. After a failure, cancel its pending dependent jobs before retrying.')
    return snapshot


def ledger_entries(state, snapshot):
    entries = [dict(row) for row in state['prior_entries']]
    for job in state['jobs']:
        row = snapshot[job['job_id']]
        if row['state'] not in TERMINAL:
            raise RuntimeError('Cannot freeze costs while an allocation is still active.')
        resources = dict(item.split('=', 1) for item in row['allocation'].split(',') if '=' in item)
        typed = [int(value) for key, value in resources.items() if key.startswith(('gres/gpu:', 'gres/gpu/'))]
        count = int(resources['gres/gpu']) if 'gres/gpu' in resources else sum(typed)
        if count not in (0, 1) or (typed and sum(typed) != count) or (job['gpu'] and row['gpu_seconds'] and count != 1):
            raise ValueError(f'Cannot verify one allocated GPU for job {job["job_id"]}: {row["allocation"]!r}')
        if not job['gpu'] and not count:
            continue
        # Slurm reports integer seconds. Round conservatively for any allocated job.
        seconds = row['gpu_seconds'] + (1 if count else 0)
        entries.append(dict(stage=job['stage'], slurm_job=job['job_id'], gpu_seconds=seconds,
                            status=row['state'], source='sacct top-level ElapsedRaw; +1 second rounding allowance'))
    return entries


def remaining(state, entries):
    return state['authorized_gpu_hours'] - math.fsum(float(row['gpu_seconds']) for row in entries) / 3600


def make_scope(state, name, bundle, entries, *, write=True):
    work = Path(state['work'])
    ledger = work / f'config/{name}-ledger.json'
    config_path = work / f'config/{name}.json'
    config = RunConfig(run_id=name, output_root=str(work), mode='source', device='cuda', bundle=str(bundle),
        updates=state['updates'], readout_updates=state['readout_updates'], seeds=tuple(state['seeds']),
        batch_size=state['batch_size'], held_extractor=state['held_extractor'],
        authorized_gpu_hours=state['authorized_gpu_hours'], projected_gpu_hours=.9,
        measured_gpu_hours=math.fsum(float(row['gpu_seconds']) for row in entries) / 3600,
        cost_ledger=str(ledger))
    if write:
        if config_path.exists() or ledger.exists() or config.root.exists():
            raise FileExistsError(f'Scope {name} already exists; retain it and use the managed retry command.')
        write_new(ledger, dict(scope_authorized=True, entries=entries))
        write_new(config_path, config.as_dict())
        config.require_gpu_scope()
    return config_path, config


def batch_command(state, stage, gpu, script, arguments, dependency=None, token='preview'):
    work = Path(state['work'])
    cmd = ['sbatch', '--parsable', '--no-requeue', '--export=ALL', f'--chdir={ROOT}',
           f'--account={state["account"]}', f'--partition={state["partition"]}',
           f'--job-name=stv2-{stage}-{token}', f'--output={work}/logs/{stage}-%j.out']
    if dependency:
        cmd.append(f'--dependency=afterok:{dependency}')
    cmd += ['--gres=gpu:h100:1', '--time=01:00:00'] if gpu else ['--time=00:20:00']
    return cmd + [str(ROOT / 'slurm/synthetic-training-v2' / script), *arguments]


def submit_one(state, stage, gpu, script, arguments, env, *, phase, scope, output='', dependency=None):
    token = uuid.uuid4().hex[:10]
    cmd = batch_command(state, stage, gpu, script, arguments, dependency, token)
    record = dict(phase=phase, stage=stage, gpu=gpu, scope=str(scope), output=str(output), token=token)
    state['pending_submission'] = dict(record=record, command=cmd)
    save(state)  # An ambiguous scheduler response must never cause an automatic duplicate.
    # Resource and array options come from these commands/scripts, not stale shell overrides.
    submission_env = {key: value for key, value in env.items() if not key.startswith('SBATCH_')}
    value = subprocess.check_output(cmd, env=submission_env, text=True).strip().split(';')[0]
    if not value.isdigit():
        raise RuntimeError(f'Uncertain sbatch response {value!r}; run status and inspect the recorded submission token.')
    state['jobs'].append(dict(record, job_id=value))
    state['pending_submission'] = None
    save(state)
    print(f'{stage}: job {value}; log {state["work"]}/logs/{stage}-{value}.out', flush=True)
    return value


def recover(state, args):
    pending = state.get('pending_submission')
    if not pending:
        raise ValueError('No uncertain submission is recorded.')
    if args.job_id:
        if not args.job_id.isdigit() or args.job_id in {job['job_id'] for job in state['jobs']}:
            raise ValueError('Supply the unique numeric job ID shown by Slurm.')
        output = subprocess.check_output(['sacct', '-n', '-P', '-j', args.job_id,
                                          '--format=JobIDRaw,JobName%100,User'], text=True)
        expected = f'stv2-{pending["record"]["stage"]}-{pending["record"]["token"]}'
        records = [line.split('|') for line in output.splitlines()]
        if not any(len(row) >= 3 and row[:3] == [args.job_id, expected, os.environ.get('USER')] for row in records):
            raise ValueError('Slurm job name/user do not match the recorded submission token, or accounting is not ready.')
        state['jobs'].append(dict(pending['record'], job_id=args.job_id))
    else:
        # Explicit human reconciliation only; a client failure does not prove rejection.
        state.setdefault('rejected_submissions', []).append(pending)
    state['pending_submission'] = None
    save(state)
    print('Submission history reconciled. Run status before retrying.')


def prepare(state, args):
    if state['source_config']:
        raise ValueError('Training has already been configured. Use a fresh run name for different paired data.')
    check(state)
    snapshot = settled(state)
    attempts = [job for job in state['jobs'] if job['phase'] == 'prepare']
    if attempts and not args.retry:
        raise ValueError('Preparation was already submitted. Run status; use prepare --retry only after a failed attempt.')
    if attempts and snapshot[attempts[-1]['job_id']]['state'] == 'COMPLETED':
        mode = read(Path(state['work']) / 'config/preparation.json').get('review_mode', 'human_audited')
        next_step = ('Run source --automated-screen.' if mode == 'automated_development'
                     else 'Inspect its overlays, then run source --overlays-reviewed.')
        raise ValueError('Preparation completed. ' + next_step)
    entries = ledger_entries(state, snapshot)
    if remaining(state, entries) < 1:
        raise PermissionError('Less than one GPU hour remains for the preparation allocation.')
    work = Path(state['work'])
    # Count frozen configs too: a scheduler refusal may leave a valid unused scope.
    index = len(list((work / 'config').glob('prepare-[0-9][0-9].json'))) + 1
    name = f'prepare-{index:02d}'
    output = work / f'paired-{index:02d}'
    scope, _ = make_scope(state, name, output / 'bundle', entries, write=not args.dry_run)
    config = read(work / 'config/preparation.json')
    config['scope_config'] = str(scope)
    frozen = work / f'config/preparation-{index:02d}.json'
    env = dict(os.environ, STV2_ROOT=str(ROOT), STV2_PYTHON=sys.executable,
               STV2_PREPARATION_CONFIG=str(frozen), STV2_PREPARATION_OUTPUT=str(output))
    if args.dry_run:
        print(json.dumps(dict(configuration=config, output=str(output),
              command=batch_command(state, 'prepare', True, 'prepare.sbatch', []), submits_jobs=False), indent=2))
        return
    write_new(frozen, config)
    (work / 'logs').mkdir(exist_ok=True)
    submit_one(state, 'prepare', True, 'prepare.sbatch', [], env, phase='prepare', scope=scope, output=output)


def source(state, args):
    snapshot = settled(state)
    jobs = [job for job in state['jobs'] if job['phase'] == 'source']
    if jobs and not args.retry:
        raise ValueError('Training was already submitted. Run status; use source --retry to resume after all old jobs are terminal.')
    preparations = [job for job in state['jobs'] if job['phase'] == 'prepare']
    if not preparations:
        raise ValueError('Run prepare and inspect the completed overlays first.')
    last = preparations[-1]
    accounting = snapshot[last['job_id']]
    if accounting['state'] != 'COMPLETED' or accounting['exit_code'] != '0:0':
        raise ValueError('The latest preparation allocation has not completed successfully.')
    prepared = Path(last['output'])
    if read(prepared / 'preparation-status.json')['status'] != 'source_prepared':
        raise ValueError('Preparation did not produce a successful source bundle.')
    automated_requested = getattr(args, 'automated_screen', False)
    if automated_requested and args.overlays_reviewed:
        raise ValueError('Choose automated screening or human overlay review, not both.')
    if not state['source_config'] and not args.overlays_reviewed and not automated_requested:
        raise ValueError(f'Inspect {prepared}/overlay-*.png for human-audited preparation, then use source --overlays-reviewed. '
                         'Algorithmically screened development preparation instead requires source --automated-screen.')
    from gavd6_sjepa.research_directions.synthetic_training_v2.contracts import TrackBundle, verify_preservation
    verify_preservation(ROOT)
    bundle = TrackBundle.load(prepared / 'bundle')
    bundle.validate(state['held_extractor'])
    mode = getattr(bundle, 'provenance', {}).get('configuration', {}).get('review_mode', 'human_audited')
    automated = mode == 'automated_development'
    if automated:
        if bundle.evidence_status != 'automated-source-screen':
            raise ValueError('Automated preparation requires automated-source-screen evidence status.')
        if not automated_requested and not (args.retry and state.get('automated_screen')):
            raise ValueError('Algorithmically screened inputs require source --automated-screen; they are not human-reviewed overlays.')
        if args.overlays_reviewed or state.get('overlays_reviewed'):
            raise ValueError('Automated source screening cannot be recorded as human overlay review.')
        if not bundle.records or any(
            row.get('review_mode') != 'automated_development'
            or row.get('locomotion_status') != 'algorithm_screened_locomotion'
            or (row.get('reserved') is not False and row.get('reserved') != 'unknown')
            or row.get('split') not in {'train', 'development'}
            for row in bundle.records
        ):
            raise ValueError('Automated source records lack compatible screening, reservation or development metadata.')
    else:
        if mode != 'human_audited' or bundle.evidence_status != 'source-run':
            raise ValueError('Human source preparation requires human_audited mode and source-run evidence status.')
        if automated_requested or state.get('automated_screen'):
            raise ValueError('The automated-screen flag requires an explicitly automated development bundle.')
    if not any(
        row['split'] == 'development' and row['extractor_family'] == state['held_extractor'] for row in bundle.records):
        raise ValueError('A real source bundle with the excluded-family development panel is required.')
    entries = ledger_entries(state, snapshot)
    if state['source_config']:
        config_path = Path(state['source_config'])
        config = RunConfig.load(config_path)
        from gavd6_sjepa.research_directions.synthetic_training_v2.workflow import initialize as bind, _receipt
        if (config.root / 'identity.json').exists():
            bind(config, ROOT)
        done = set()
        for stage in STAGES:
            if (config.root / f'receipts/{stage}.json').exists():
                _receipt(config, stage)
                done.add(stage)
        stages = [stage for stage in STAGES if stage not in done]
    else:
        config_path, config = make_scope(state, 'source-01', prepared / 'bundle', entries, write=False)
        stages = list(STAGES)
    if not stages:
        print(f'All stages already complete. Report: {config.root}/report.md')
        return
    needed = sum(stage in {'direct', 'jepa'} for stage in stages)
    if remaining(state, entries) < needed:
        raise PermissionError(f'Training needs {needed} one-hour GPU allocations; authorized remaining time is {remaining(state, entries):.3f} hours.')
    if args.dry_run:
        previous = None
        for stage in stages:
            print(shlex.join(batch_command(state, stage, stage in {'direct', 'jepa'}, 'stage.sbatch', [str(config_path), stage], previous)))
            previous = f'JOB_{stage}'
        print('DRY_RUN: no configuration changes or jobs submitted.')
        return
    if not state['source_config']:
        make_scope(state, 'source-01', prepared / 'bundle', entries)
        state['source_config'] = str(config_path)
        if automated:
            state['automated_screen'] = dict(preparation_job=last['job_id'], actor=os.environ.get('USER', 'unknown'),
                                             review_mode=mode, human_reviewed=False,
                                             evidence_status=bundle.evidence_status)
        else:
            state['overlays_reviewed'] = dict(preparation_job=last['job_id'], actor=os.environ.get('USER', 'unknown'))
        save(state)
    else:
        # Reconcile allocation overhead or killed attempts before reusing this immutable scope.
        for job in jobs:
            allocation = next((row for row in entries if str(row.get('slurm_job')) == job['job_id']), None)
            if allocation is None:
                continue
            costs = list((config.root / 'costs').glob('*.json'))
            recorded = math.fsum(float(row['gpu_seconds']) for row in map(read, costs)
                                 if str(row.get('slurm_job')) == job['job_id'])
            allocated = allocation['gpu_seconds']
            if allocated > recorded:
                write_new(config.root / f'costs/allocation-{job["job_id"]}-{uuid.uuid4().hex}.json',
                          dict(stage=job['stage'], slurm_job=job['job_id'], gpu_seconds=allocated-recorded,
                               status=snapshot[job['job_id']]['state'], source='unrecorded allocation seconds only'))
    env = dict(os.environ, STV2_ROOT=str(ROOT), STV2_PYTHON=sys.executable)
    previous = None
    for stage in stages:
        previous = submit_one(state, stage, stage in {'direct', 'jepa'}, 'stage.sbatch', [str(config_path), stage],
                              env, phase='source', scope=config_path, dependency=previous)


def status(state):
    snapshot = scheduler_snapshot(state['jobs'])
    print('Run:', state['work'])
    for job in state['jobs']:
        row = snapshot.get(job['job_id'], {})
        print(f'{job["stage"]:12} {job["job_id"]:12} {row.get("state", "ACCOUNTING_PENDING"):20} {row.get("exit_code", "")}; '
              f'log: {state["work"]}/logs/{job["stage"]}-{job["job_id"]}.out')
        if job['phase'] == 'prepare':
            print('  Paired data and overlays:', job['output'])
        elif job['phase'] == 'automated_inputs':
            print('  Automated screening records and playback:', job['output'])
    if state.get('pending_submission'):
        print('UNCERTAIN SUBMISSION:', json.dumps(state['pending_submission']))
    if all(snapshot.get(job['job_id'], {}).get('state') in TERMINAL for job in state['jobs']):
        print(f'Authorized hours remaining: {remaining(state, ledger_entries(state, snapshot)):.4f}')
    if state['source_config']:
        print('Results:', RunConfig.load(state['source_config']).root)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    init = commands.add_parser('init', help='Create a new run from existing ST_* paths; no GPU submission')
    init.add_argument('--name', required=True)
    init.add_argument('--gpu-hours', type=float, required=True)
    prior = init.add_mutually_exclusive_group(required=True)
    prior.add_argument('--prior-gpu-hours', type=float)
    prior.add_argument('--prior-ledger', type=Path)
    init.add_argument('--updates', type=int, default=200)
    init.add_argument('--seeds', type=int, nargs='+', default=[17])
    for name in ('check', 'inputs', 'status'):
        commands.add_parser(name)
    ledger = commands.add_parser('ledger', help='Export all settled allocation costs for the next run')
    ledger.add_argument('--output', type=Path, required=True)
    recovery = commands.add_parser('recover', help='Reconcile an uncertain sbatch response after inspecting Slurm')
    outcome = recovery.add_mutually_exclusive_group(required=True)
    outcome.add_argument('--job-id')
    outcome.add_argument('--not-submitted', action='store_true')
    for name in ('prepare', 'source'):
        command = commands.add_parser(name)
        command.add_argument('--dry-run', action='store_true')
        command.add_argument('--retry', action='store_true')
        if name == 'source':
            review = command.add_mutually_exclusive_group()
            review.add_argument('--overlays-reviewed', action='store_true')
            review.add_argument('--automated-screen', action='store_true',
                               help='Run explicitly algorithm-screened development data without claiming human review')
    args = parser.parse_args(argv)
    if args.command == 'init':
        initialize(args)
        return 0
    work = Path(os.environ['STV2_WORK']).resolve()
    state = state_for(work)
    if args.command == 'status':
        status(state)
    elif args.command == 'check':
        check(state)
    elif args.command == 'inputs':
        print('Review worksheets:', create_drafts(read(work / 'config/preparation.json'), work))
    elif args.command == 'ledger':
        with stage_lock(work, 'manage'):
            state = state_for(work)
            write_new(args.output, dict(scope_authorized=True, entries=ledger_entries(state, settled(state))))
        print('Saved cost ledger:', args.output)
    elif args.command == 'recover':
        with stage_lock(work, 'manage'):
            recover(state_for(work), args)
    else:
        # Dry-run must be read-only, including lock-directory creation.
        if args.dry_run:
            {'prepare': prepare, 'source': source}[args.command](state, args)
        else:
            with stage_lock(work, 'manage'):
                state = state_for(work)
                {'prepare': prepare, 'source': source}[args.command](state, args)
    return 0


if __name__ == '__main__':
    try:
        raise SystemExit(main())
    except (OSError, ValueError, RuntimeError, KeyError, subprocess.CalledProcessError) as error:
        print(f'STV2: {error}', file=sys.stderr)
        raise SystemExit(1)
