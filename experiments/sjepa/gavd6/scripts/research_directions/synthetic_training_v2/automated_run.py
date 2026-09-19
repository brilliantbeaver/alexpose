#!/usr/bin/env python3
"""Run the managed, machine-screened development comparison without manual CSV edits."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / 'src'))
import haic


class PhaseFailed(RuntimeError):
    """A recorded phase reached terminal job states with at least one failure."""


ACTIVE_STATES = {'PENDING', 'RUNNING', 'CONFIGURING', 'COMPLETING', 'SUSPENDED',
                 'RESIZING', 'REQUEUED', 'REQUEUE_FED', 'REQUEUE_HOLD',
                 'SIGNALING', 'STAGE_OUT', 'ACCOUNTING_PENDING'}


def read(path):
    return json.loads(Path(path).read_text())


def progress(work, phase, status, **details):
    value = dict(phase=phase, status=status, updated_utc=datetime.now(timezone.utc).isoformat(),
                 pid=os.getpid(), **details)
    haic.atomic_json(work / 'automation.json', value)
    print(f'{value["updated_utc"]} {phase}: {status}', flush=True)
    return value


def cancel_pending_source(state):
    """Cancel only recorded source dependents still PENDING in the live queue."""
    ids = {job['job_id'] for job in state['jobs'] if job['phase'] == 'source'}
    if not ids:
        return
    output = subprocess.check_output(['squeue', '--noheader', '--jobs=' + ','.join(sorted(ids)),
                                      '--format=%i|%T'], text=True)
    pending = [fields[0].strip() for line in output.splitlines()
               if len(fields := line.split('|')) == 2
               and fields[0].strip() in ids and fields[1].strip() == 'PENDING']
    if pending:
        subprocess.run(['scancel', *pending], check=True)
        print('Canceled this failed chain\'s pending dependent jobs:', ', '.join(pending), flush=True)


def wait_jobs(work, phase, *, poll_seconds=30):
    last = None
    failures = 0
    while True:
        state = haic.state_for(work)
        jobs = [job for job in state['jobs'] if job['phase'] == phase]
        if not jobs:
            raise ValueError(f'No recorded {phase} jobs.')
        # Older failed attempts remain in the ledger, but only the latest
        # attempt of each stage determines whether this phase succeeds.
        latest = {job['stage']: job for job in jobs}
        try:
            snapshot = haic.scheduler_snapshot(state['jobs'])
            failures = 0
        except (OSError, subprocess.CalledProcessError):
            failures += 1
            if failures >= 3:
                raise
            time.sleep(poll_seconds)
            continue
        statuses = {stage: dict(job_id=job['job_id'], **snapshot.get(job['job_id'], {'state': 'ACCOUNTING_PENDING'}))
                    for stage, job in latest.items()}
        unknown = {stage: row['state'] for stage, row in statuses.items()
                   if row['state'] not in haic.TERMINAL | ACTIVE_STATES}
        if unknown:
            raise ValueError(f'Unexpected scheduler states {unknown}; inspect accounting before resubmission.')
        visible = {stage: (row['job_id'], row['state'], row.get('exit_code')) for stage, row in statuses.items()}
        if visible != last:
            progress(work, phase, 'waiting', jobs=statuses)
            for stage, row in statuses.items():
                print(f"  {stage}: {row['job_id']} {row['state']} {row.get('exit_code', '')}", flush=True)
            last = visible
        failed = [row for row in statuses.values() if row['state'] in haic.TERMINAL
                  and (row['state'] != 'COMPLETED' or row.get('exit_code') != '0:0')]
        if failed and phase == 'source':
            cancel_pending_source(state)
        if all(row['state'] in haic.TERMINAL for row in statuses.values()):
            if failed:
                detail = '; '.join(f"{row['job_id']} {row['state']} {row.get('exit_code', '')}" for row in failed)
                raise PhaseFailed(f'{phase} failed: {detail}. Inspect {work}/logs. No retry was submitted automatically.')
            return latest
        time.sleep(poll_seconds)


def phase_succeeded(state, phase):
    jobs = [job for job in state['jobs'] if job['phase'] == phase]
    if not jobs:
        return False
    snapshot = haic.settled(state)
    latest = {job['stage']: job for job in jobs}
    return all(snapshot[job['job_id']]['state'] == 'COMPLETED'
               and snapshot[job['job_id']]['exit_code'] == '0:0' for job in latest.values())


def prepare_automated_inputs(work, *, retry, poll_seconds):
    with haic.stage_lock(work, 'manage'):
        state = haic.state_for(work)
        if state.get('pending_submission'):
            raise RuntimeError('Uncertain submission: use submit.sh status and README recovery; automatic duplication is forbidden.')
        attempts = [job for job in state['jobs'] if job['phase'] == 'automated_inputs']
        if not attempts and state['jobs']:
            raise ValueError('This run already has a different experiment. Preserve it; automation requires the unsubmitted run described in the README.')
        submit = not attempts
        if attempts:
            snapshot = haic.scheduler_snapshot(state['jobs'])
            previous = snapshot.get(attempts[-1]['job_id'], {})
            if previous.get('state') in haic.TERMINAL and (
                    previous['state'] != 'COMPLETED' or previous.get('exit_code') != '0:0'):
                if not retry:
                    raise ValueError('The CPU input attempt failed. Inspect its log; use launch --retry after correcting the cause.')
                if any(job['phase'] != 'automated_inputs' for job in state['jobs']):
                    raise ValueError('Do not change source inputs after preparation or training submission.')
                haic.settled(state)
                submit = True
        if submit:
            config = read(work / 'config/preparation.json')
            config['review_mode'] = 'automated_development'
            count = len(list((work / 'config').glob('automated-inputs-*.json'))) + 1
            snapshot_path = work / f'config/automated-inputs-{count:02d}.json'
            output = work / f'inputs/automated-{count:02d}'
            haic.write_new(snapshot_path, config)
            (work / 'logs').mkdir(exist_ok=True)
            env = dict(os.environ, STV2_ROOT=str(ROOT), STV2_PYTHON=sys.executable)
            haic.submit_one(state, 'automated_inputs', False, 'automated-inputs.sbatch',
                            [str(snapshot_path), str(output)], env,
                            phase='automated_inputs', scope=snapshot_path, output=output)
    latest = wait_jobs(work, 'automated_inputs', poll_seconds=poll_seconds)
    output = Path(latest['automated_inputs']['output'])
    with haic.stage_lock(work, 'manage'):
        state = haic.state_for(work)
        config_path = work / 'config/preparation.json'
        original = read(config_path)
        config = dict(original, review_mode='automated_development',
                      locomotion_audit=str(output / 'locomotion-audit.csv'),
                      reservation_csv=str(output / 'person-reservations.csv'))
        from haic_inputs import check_inputs
        result = check_inputs(config, state['held_extractor'])
        if config != original:
            if any(job['phase'] != 'automated_inputs' for job in state['jobs']):
                raise ValueError('The configured inputs differ after GPU submission; preserve the run and investigate.')
            backup = config_path.with_name('preparation.before-automation.json')
            if not backup.exists():
                haic.write_new(backup, original)
            haic.atomic_json(config_path, config)
        print('Automated source inputs validated:', json.dumps(result), flush=True)


def execute_phase(work, phase, *, retry, poll_seconds):
    state = haic.state_for(work)
    jobs = [job for job in state['jobs'] if job['phase'] == phase]
    if jobs:
        try:
            wait_jobs(work, phase, poll_seconds=poll_seconds)
            # A partially submitted source chain still needs explicit retry to
            # submit missing stages; a successful subset is not a full chain.
            from gavd6_sjepa.research_directions.synthetic_training_v2.config import STAGES
            if phase != 'source' or {job['stage'] for job in jobs} == set(STAGES):
                return
            if not retry:
                raise ValueError('Source chain is only partly submitted. Reconcile any uncertain submission; then use launch --retry.')
        except PhaseFailed:
            if not retry:
                raise
        if not retry:
            raise ValueError(f'{phase} already has unsuccessful history; an explicit retry is required.')
    command = [sys.executable, str(ROOT / 'scripts/research_directions/synthetic_training_v2/haic.py'), phase]
    if jobs:
        command.append('--retry')
    if phase == 'source':
        command.append('--automated-screen')
    subprocess.run(command, env=dict(os.environ, STV2_ROOT=str(ROOT), STV2_WORK=str(work),
                                    STV2_PYTHON=sys.executable), check=True)
    wait_jobs(work, phase, poll_seconds=poll_seconds)


def run(work, *, retry=False, poll_seconds=30):
    with haic.stage_lock(work, 'automation'):
        try:
            progress(work, 'checks', 'running')
            state = haic.state_for(work)
            if state.get('pending_submission'):
                raise ValueError('Uncertain scheduler submission: reconcile it before restarting automation.')
            if not haic.environment_ok():
                raise ValueError('The saved interpreter failed the required environment check; no new job submitted.')
            if not state['jobs']:
                if haic.remaining(state, state['prior_entries']) < 3:
                    raise ValueError('This first comparison needs room for three one-hour GPU allocations in the existing authorized budget.')
                from gavd6_sjepa.research_directions.synthetic_training_v2.contracts import verify_preservation
                from gavd6_sjepa.research_directions.synthetic_training_v2.audit import reconstruct_pilot
                verify_preservation(ROOT)
                reconstruct_pilot(ROOT)
            prepare_automated_inputs(work, retry=retry, poll_seconds=poll_seconds)
            execute_phase(work, 'prepare', retry=retry, poll_seconds=poll_seconds)
            execute_phase(work, 'source', retry=retry, poll_seconds=poll_seconds)
            from check_results import check_results
            result = check_results(work)
            progress(work, 'results', 'complete', results=result)
            print('AUTOMATED_EXPERIMENT_COMPLETE', flush=True)
        except Exception as error:
            progress(work, 'stopped', 'failed', error=str(error))
            raise


def controller_running(work):
    """Probe the shared run lock; a saved PID may be reused or belong to another host."""
    path = Path(work) / 'locks/process/automation.lock'
    try:
        stream = path.open('r')
    except FileNotFoundError:
        return False
    with stream:
        try:
            fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            return True
        fcntl.flock(stream, fcntl.LOCK_UN)
        return False


def launch(work, *, retry=False):
    haic.state_for(work)
    with haic.stage_lock(work, 'automation-launch'):
        saved = work / 'automation-process.json'
        if controller_running(work):
            print('Automation controller is already running. Use status; no duplicate controller started.')
            return
        summary = work / 'automation.json'
        if summary.exists() and read(summary).get('status') == 'complete':
            print('This experiment is complete. Use status to see the report path.')
            return
        command = [sys.executable, '-u', str(Path(__file__).resolve()), 'run', '--work', str(work)]
        if retry:
            command.append('--retry')
        with (work / 'automation.log').open('a') as log, open(os.devnull) as null:
            child = subprocess.Popen(command, stdin=null, stdout=log, stderr=subprocess.STDOUT,
                                     start_new_session=True, cwd=ROOT,
                                     env=dict(os.environ, STV2_WORK=str(work), STV2_ROOT=str(ROOT), STV2_PYTHON=sys.executable))
        haic.atomic_json(saved, dict(pid=child.pid, command=command))
        print(f'Controller started: PID {child.pid}\nLog: {work}/automation.log\n'
              'It will screen inputs, prepare, train, evaluate, and verify results. No prompts are required. '
              'A failed check stops the controller and is reported by status.')


def status(work):
    summary = work / 'automation.json'
    print(json.dumps(read(summary), indent=2) if summary.exists() else 'Automation has not started.')
    print('Controller running:', controller_running(work))
    haic.status(haic.state_for(work))
    log = work / 'automation.log'
    if log.exists():
        print('\nRecent controller output:')
        with log.open() as stream:
            from collections import deque
            print(''.join(deque(stream, maxlen=20)))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=('launch', 'run', 'status'))
    parser.add_argument('--work', type=Path, default=os.environ.get('STV2_WORK'))
    parser.add_argument('--retry', action='store_true')
    parser.add_argument('--poll-seconds', type=int, default=30)
    args = parser.parse_args(argv)
    if args.work is None:
        parser.error('Source the saved session.env or supply --work.')
    if args.poll_seconds < 10:
        parser.error('Use at least 10 seconds between scheduler checks.')
    work = args.work.expanduser().resolve()
    if args.command == 'launch':
        launch(work, retry=args.retry)
    elif args.command == 'run':
        run(work, retry=args.retry, poll_seconds=args.poll_seconds)
    else:
        status(work)
    return 0


if __name__ == '__main__':
    try:
        raise SystemExit(main())
    except (OSError, ValueError, RuntimeError, KeyError, subprocess.CalledProcessError) as error:
        print(f'AUTOMATED_RUN: {error}', file=sys.stderr)
        raise SystemExit(1)
