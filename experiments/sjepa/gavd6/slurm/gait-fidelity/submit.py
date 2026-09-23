#!/usr/bin/env python3
"""Submit one restartable CPU coordinator without duplicating uncertain jobs."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import fcntl
import json
import os
from pathlib import Path
import re
import subprocess
import uuid


def atomic_json(path: Path, value: dict) -> None:
    temporary = path.with_name(path.name + f'.{os.getpid()}.tmp')
    temporary.write_text(json.dumps(value, indent=2) + '\n')
    os.replace(temporary, path)


def active_job(job_id: str) -> bool:
    result = subprocess.run(['squeue', '--noheader', '--jobs', job_id, '--format=%i'],
                            capture_output=True, text=True, check=False)
    if result.returncode == 0:
        return bool(result.stdout.strip())
    # Some Slurm versions return an error for an already-finished job ID.
    accounted = subprocess.run(['sacct', '--noheader', '--allocations', '--parsable2',
                                '--jobs', job_id, '--format=JobIDRaw,State'],
                               capture_output=True, text=True, check=True)
    states = [line.split('|')[1].split()[0].rstrip('+') for line in accounted.stdout.splitlines()
              if line.split('|')[0] == job_id and len(line.split('|')) > 1]
    if not states:
        raise SystemExit(f'Cannot reconcile coordinator {job_id}; no duplicate was submitted. {result.stderr.strip()}')
    terminal = {'COMPLETED', 'CANCELLED', 'FAILED', 'TIMEOUT', 'NODE_FAIL', 'OUT_OF_MEMORY',
                'PREEMPTED', 'BOOT_FAIL', 'DEADLINE', 'REVOKED'}
    return not all(state in terminal for state in states)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--work', type=Path, required=True)
    parser.add_argument('--max-jobs', type=int, default=8, choices=range(1, 9))
    parser.add_argument('--account', help='Default: account saved in config.json.')
    parser.add_argument('--partition', help='Default: partition saved in config.json.')
    parser.add_argument('--controller-hours', type=int, default=72)
    stage=parser.add_mutually_exclusive_group()
    stage.add_argument('--prepare-only', action='store_true', help='Stop after shared data preparation and validation.')
    stage.add_argument('--gavd-only', action='store_true', help='Extract planned GAVD training/development recordings.')
    stage.add_argument('--gavd-confirmation', action='store_true', help='Extract locked GAVD confirmation recordings.')
    stage.add_argument('--confirmation-only', action='store_true', help='Prepare locked AMASS confirmation people.')
    args = parser.parse_args()
    if not 1 <= args.controller_hours <= 168:
        parser.error('--controller-hours must be between 1 and 168')
    work = args.work.expanduser().resolve()
    cfg = json.loads((work / 'config.json').read_text())
    resources=cfg.get('resources',{})
    stage_name=('prepare-only' if args.prepare_only else 'gavd-only' if args.gavd_only
                else 'gavd-confirmation' if args.gavd_confirmation else 'confirmation-only' if args.confirmation_only else 'full')
    if cfg.get('fixture'):
        parser.error('A software fixture uses run.sh fixture; do not submit it to HAIC.')
    code_root = Path(os.environ['GF_ROOT']).resolve()
    control, logs = work / 'control', work / 'logs'
    control.mkdir(parents=True, exist_ok=True)
    logs.mkdir(parents=True, exist_ok=True)
    receipt = control / 'coordinator.json'
    with (control / 'submit.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        previous = json.loads(receipt.read_text()) if receipt.exists() else None
        if previous and previous.get('state') == 'submitting':
            raise SystemExit(
                f"Unresolved coordinator submission {previous['job_name']}. "
                'Check squeue and sacct for that exact job name before changing the receipt; '
                'a duplicate was not submitted.')
        if previous and previous.get('job_id') and active_job(previous['job_id']):
            print(json.dumps(previous, indent=2))
            print('The coordinator is already queued or running. No duplicate submitted.')
            if previous.get('stage','prepare-only' if previous.get('prepare_only') else 'full') != stage_name:
                print(f'Requested stage {stage_name} was not queued. Wait for the active stage, then repeat this command.')
            return 0
        stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
        attempt = f'{stamp}-{uuid.uuid4().hex[:10]}'
        job_name = f'gf-controller-{attempt}'
        log = logs / f'coordinator-{attempt}-%j.out'
        command = [
            'sbatch', '--parsable', f'--job-name={job_name}', f"--account={args.account or resources.get('account','mind')}",
            f"--partition={args.partition or resources.get('partition','hai')}", '--nodes=1', '--ntasks=1', '--cpus-per-task=2',
            f"--mem={resources.get('controller_memory','64G')}", f'--time={args.controller_hours}:00:00', '--export=ALL', '--no-requeue',
            f'--output={log}', str(code_root / 'slurm/gait-fidelity/coordinator.sbatch'),
            str(work), str(args.max_jobs), stage_name,
        ]
        state = dict(state='submitting', job_name=job_name, attempt=attempt,
                     requested_utc=datetime.now(timezone.utc).isoformat(), command=command,
                     work=str(work), max_jobs=args.max_jobs, prepare_only=args.prepare_only, stage=stage_name,log_pattern=str(log))
        atomic_json(receipt, state)
        # If the process disappears after sbatch accepts a job, the unresolved receipt
        # deliberately prevents a second coordinator until scheduler reconciliation.
        try:
            result = subprocess.run(command, capture_output=True, text=True, check=False)
        except OSError as error:
            state.update(state='submission_failed', error=str(error))
            atomic_json(receipt, state)
            raise
        state.update(stdout=result.stdout, stderr=result.stderr, returncode=result.returncode)
        if result.returncode != 0:
            # A transport failure can happen after acceptance. Preserve the unique
            # name and block relaunch until the scheduler has been reconciled.
            atomic_json(receipt, state)
            raise SystemExit(f'Coordinator submission needs reconciliation for {job_name}: {result.stderr.strip()}')
        match = re.fullmatch(r'(\d+)(?:;[^\s;]+)?', result.stdout.strip())
        if match is None:
            atomic_json(receipt, state)
            raise SystemExit('Slurm returned an unrecognized receipt. Resolve its job name before relaunching.')
        state.update(state='submitted', job_id=match.group(1),
                     log=str(log).replace('%j', match.group(1)))
        atomic_json(receipt, state)
        atomic_json(control / f'coordinator-{attempt}.json', state)
        print(json.dumps(state, indent=2))
        print('Coordinator submitted. GPU workers start only after their checks pass.')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
