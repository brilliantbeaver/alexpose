#!/usr/bin/env python3
"""Enforce the repair cutoff even if its CPU coordinator disappears."""
from __future__ import annotations

import json
from pathlib import Path
import sys

from worker import supervise


def main():
    if len(sys.argv) != 4:
        raise SystemExit('Usage: repair-worker.py WORK PHASE ATTEMPT')
    work, phase, attempt = sys.argv[1:]
    cfg = json.loads((Path(work)/'config.json').read_text())
    if cfg.get('fixture') or cfg.get('study_kind') != 'gait_fidelity_readout_repair':
        raise ValueError('The HAIC repair supervisor requires a source repair configuration')
    command = [sys.executable, '-u', '-m', 'gavd6_sjepa.research_directions.gait_fidelity.repair',
               'worker', '--work', work, '--phase', phase, '--attempt', attempt]
    return supervise(command, deadline=cfg['repair']['deadline_utc'],
                     receipt=Path(attempt)/'deadline-stop.json')


if __name__ == '__main__':
    raise SystemExit(main())
