"""Inspect configuration and completed receipts without opening source media."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src"))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--run-root", type=Path)
    parser.add_argument("--phase", default="inventory",
                        choices=("inventory", "pilot", "develop", "confirm"))
    parser.add_argument("--verify-artifacts", action="store_true",
                        help="Verify frozen identity and every present stage receipt; no media decoding.")
    args = parser.parse_args(argv)
    from gavd6_sjepa.research_directions.temporal_gait.config import RunConfig
    cfg = RunConfig.from_env(config_path=args.config)
    cfg.validate(check_input_paths=False)
    if args.run_root is not None and args.run_root.expanduser().resolve() != cfg.root:
        raise ValueError("--run-root must agree with the resolved configuration")
    receipts = []
    folder = cfg.root / "receipts"
    # This scans only the explicitly selected output's receipts, never data roots.
    if folder.is_dir():
        for path in sorted(folder.glob("*.json")):
            payload = json.loads(path.read_text())
            if args.verify_artifacts:
                from gavd6_sjepa.research_directions.temporal_gait.contracts import verify_receipt
                verify_receipt(cfg, payload["stage"])
            receipts.append({"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                             "status": payload.get("status", "receipt_present")})
    if args.verify_artifacts:
        from gavd6_sjepa.research_directions.temporal_gait.contracts import verify_run
        verify_run(cfg)
    report = {"status": "present_artifacts_verified" if args.verify_artifacts else "configuration_valid", "mode": cfg.mode,
              "run_root": str(cfg.root), "phase": args.phase,
              "receipts": receipts, "media_opened": False,
              "note": "Present receipt validation is not proof of a complete task grid or a scientific pass."}
    print(json.dumps(report, indent=2))
    return report


if __name__ == "__main__":
    main()
