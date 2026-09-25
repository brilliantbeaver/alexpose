#!/usr/bin/env python3
"""Supervise a follow-up process group independently of the Slurm coordinator."""
from __future__ import annotations

from datetime import datetime, timezone
import json
import os
from pathlib import Path
import signal
import subprocess
import sys


def supervise(command, *, deadline, receipt, grace_seconds=30.):
    cutoff = datetime.fromisoformat(deadline.replace('Z', '+00:00'))
    if cutoff.tzinfo is None:
        raise ValueError('Worker cutoff requires an explicit timezone')
    remaining = (cutoff-datetime.now(timezone.utc)).total_seconds()
    receipt = Path(receipt)

    def record(reason, **extra):
        receipt.parent.mkdir(parents=True, exist_ok=True)
        temporary = receipt.with_name(receipt.name+'.tmp')
        temporary.write_text(json.dumps(dict(status='DEADLINE_STOPPED', reason=reason,
            deadline_utc=deadline, recorded_utc=datetime.now(timezone.utc).isoformat(), **extra), indent=2)+'\n')
        temporary.replace(receipt)

    if remaining <= grace_seconds+5:
        record('Queued worker started too late; no workload was launched')
        return 124
    process = subprocess.Popen(command, start_new_session=True)
    group_stopped = False

    def stop(signum=None, frame=None):
        nonlocal group_stopped
        record('Deadline or allocation termination; child process group is stopping', child_pid=process.pid)
        try:
            os.killpg(process.pid, signal.SIGTERM)
        except ProcessLookupError:
            return
        seconds = max(0., min(grace_seconds, (cutoff-datetime.now(timezone.utc)).total_seconds()-5))
        try:
            process.wait(timeout=seconds)
        except subprocess.TimeoutExpired:
            pass
        finally:
            # The leader may exit on TERM while a feature-export descendant
            # ignores it. Reaping the leader is not proof that its group ended.
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            group_stopped = True
            process.wait()

    old = signal.signal(signal.SIGTERM, stop)
    try:
        try:
            # Grace and kill finish before the registered absolute cutoff.
            return process.wait(timeout=remaining-grace_seconds-5)
        except subprocess.TimeoutExpired:
            stop()
            return 124
    finally:
        if process.poll() is not None and not group_stopped:
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
        signal.signal(signal.SIGTERM, old)


def main():
    if len(sys.argv) != 4:
        raise SystemExit('Usage: worker.py WORK PHASE_ID ATTEMPT_DIRECTORY')
    work, phase, attempt = sys.argv[1:]
    cfg = json.loads((Path(work)/'config.json').read_text())
    command = [sys.executable, '-u', '-m', 'gavd6_sjepa.research_directions.gait_fidelity',
               'worker', '--work', work, '--phase-id', phase, '--attempt', attempt]
    if cfg.get('study_kind') != 'jepa_response_followup':
        os.execv(sys.executable, command)
    return supervise(command, deadline=cfg['followup']['deadline_utc'],
                     receipt=Path(attempt)/'deadline-stop.json')


if __name__ == '__main__':
    raise SystemExit(main())
