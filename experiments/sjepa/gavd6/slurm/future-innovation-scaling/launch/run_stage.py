"""Run one existing stage and retain operational evidence for notebook 23.

Every attempt gets separate logs. A process exit is not a scientific verdict.
Abruptly killed jobs may retain `running`; consult Slurm for their final state.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from uuid import uuid4

ROOT = Path(__file__).resolve().parents[3]
BINDINGS = ('config/study.json', 'manifests/learning-plan.json',
            'reports/complete.json', 'reports/learning-curve.json')


def binding(root):
    return {p: hashlib.sha256((root / p).read_bytes()).hexdigest() if (root / p).is_file() else None
            for p in BINDINGS}


def write_record(path, record):
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(record, indent=2) + '\n')
    temporary.replace(path)


def run_stage(root, stage, command, fold=None):
    root = Path(root).resolve()
    directory = root / 'logs/stages'
    if not directory.resolve().is_relative_to(root):
        raise ValueError('Stage logs must stay inside FI_RUN_ROOT')
    for relative in ('config/study.json', 'config/run-contract.json'):
        path = root / relative
        if path.exists() and json.loads(path.read_text()).get('protocol') != 'source-learning-curve-v1':
            raise ValueError('FI_RUN_ROOT must be a source learning-curve study, not a historical gate')
    directory.mkdir(parents=True, exist_ok=True)
    identity = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ') + '-' + uuid4().hex[:12]
    stem = f'{identity}-{stage}' + (f'-fold-{fold}' if fold is not None else '')
    path, log = directory / f'{stem}.json', directory / f'{stem}.log'
    record = dict(version='source-curve-stage-execution-v1', stage=stage, fold=fold,
                  run_root=str(root), command=command, status='running', returncode=None,
                  started_utc=datetime.now(timezone.utc).isoformat(),
                  slurm_job_id=os.environ.get('SLURM_JOB_ID'),
                  slurm_array_task_id=os.environ.get('SLURM_ARRAY_TASK_ID'),
                  log=str(log.relative_to(root)), binding_before=binding(root),
                  runner_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    start = time.monotonic()
    write_record(path, record)
    tail = ''
    process = None
    try:
        with log.open('x') as output:
            process = subprocess.Popen(command, cwd=ROOT, stdout=subprocess.PIPE,
                                       stderr=subprocess.STDOUT, text=True, bufsize=1,
                                       env={**os.environ, 'FI_RUN_ROOT': str(root)})
            for line in process.stdout:
                print(line, end='', flush=True)
                output.write(line)
                output.flush()
                tail = (tail + line)[-6000:]
            record['returncode'] = process.wait()
        record['binding_after'] = binding(root)
        if stage == 'verify' and record['returncode'] == 0:
            if (record['binding_before'] != record['binding_after']
                    or any(value is None for value in record['binding_after'].values())):
                raise ValueError('Verification requires unchanged study, plan and completed report bindings')
        record['status'] = 'passed' if record['returncode'] == 0 else 'failed'
    except (Exception, KeyboardInterrupt) as error:
        if process is not None and process.poll() is None:
            process.terminate()
            process.wait()
        record.update(status='failed', returncode=130 if isinstance(error, KeyboardInterrupt) else 2,
                      error=f'{type(error).__name__}: {error}')
        print(record['error'], file=sys.stderr)
    finally:
        if process is not None and process.stdout is not None:
            process.stdout.close()
        record.update(finished_utc=datetime.now(timezone.utc).isoformat(),
                      elapsed_seconds=time.monotonic() - start,
                      binding_after=binding(root))
        if record['status'] != 'passed':
            record['output_tail'] = tail
        write_record(path, record)
    return record['returncode']


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=['initialize', 'prepare', 'cache', 'audit-teacher', 'plan', 'run', 'report', 'verify'])
    parser.add_argument('--run-root', type=Path, default=os.environ.get('FI_RUN_ROOT'))
    parser.add_argument('--fold', type=int, choices=range(5))
    args, other = parser.parse_known_args()
    if args.run_root is None:
        parser.error('--run-root or FI_RUN_ROOT is required')
    if args.stage == 'initialize':
        if other or args.fold is not None:
            parser.error('initialize uses the existing environment variables only')
        command = [sys.executable, str(ROOT / 'slurm/future-innovation-scaling/launch/initialize.py'), 'initialize']
    else:
        command = [sys.executable, str(ROOT / 'scripts/research_directions/future_innovation/run_source_learning_curve.py'),
                   args.stage, '--run-root', str(args.run_root), *other]
        if args.fold is not None:
            command += ['--fold', str(args.fold)]
    return run_stage(args.run_root, args.stage, command, args.fold)


if __name__ == '__main__':
    raise SystemExit(main())
