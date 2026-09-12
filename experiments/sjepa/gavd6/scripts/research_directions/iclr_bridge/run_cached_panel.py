"""Freeze, execute or read-only verify the separate cached accessibility panel."""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src"))
from gavd6_sjepa.research_directions.iclr_bridge.cached_panel import freeze_cached_panel, run_cached_panel, verify_cached_panel


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    subs = parser.add_subparsers(dest="command", required=True)
    freeze = subs.add_parser("freeze")
    freeze.add_argument("--source-run", required=True, type=Path)
    freeze.add_argument("--output-root", required=True, type=Path)
    freeze.add_argument("--protocol-document", type=Path)
    for name in ("run", "verify"):
        sub = subs.add_parser(name)
        sub.add_argument("--output-root", required=True, type=Path)
    args = parser.parse_args()
    if args.command == "freeze": result = freeze_cached_panel(args.source_run, args.output_root, args.protocol_document)
    elif args.command == "run": result = run_cached_panel(args.output_root)
    else: result = verify_cached_panel(args.output_root)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
