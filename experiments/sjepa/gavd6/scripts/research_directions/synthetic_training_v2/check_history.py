#!/usr/bin/env python3
"""Read-only diagnosis of a missing transfer versus changed historical evidence."""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / 'src'))
from gavd6_sjepa.research_directions.synthetic_training_v2.contracts import (
    inspect_preservation, preservation_failure,
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repo', type=Path, default=ROOT)
    parser.add_argument('--json', action='store_true', help='Include expected and actual hashes in JSON')
    parser.add_argument('--reconstruct', action='store_true', help='Also reconstruct pilot arithmetic after all hashes pass')
    args = parser.parse_args()
    report = inspect_preservation(args.repo)
    if report['status'] == 'pass' and args.reconstruct:
        from gavd6_sjepa.research_directions.synthetic_training_v2.audit import reconstruct_pilot
        pilot = reconstruct_pilot(args.repo)
        report['pilot'] = {'counts': pilot['counts'], 'checks': pilot['checks']}
    if args.json:
        print(json.dumps(report, indent=2))
    elif report['status'] != 'pass':
        print(preservation_failure(report))
    else:
        print(f"Preservation passed: {report['matched']}/{report['files_checked']} files match.")
        if args.reconstruct:
            print('Historical pilot counts:', report['pilot']['counts'])
            print('HISTORICAL_AUDIT_PASSED')
    return 0 if report['status'] == 'pass' else 1


if __name__ == '__main__':
    raise SystemExit(main())
