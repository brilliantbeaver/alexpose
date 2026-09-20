"""Audit the full-data registry; optionally run the five-fold experiment."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

EXP = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(EXP))

from sjepa.splits import load_full_registry, split_summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", action="store_true", help="train and evaluate all five folds")
    parser.add_argument("--smoke", action="store_true", help="tiny model and 4+2 updates; execution check only")
    parser.add_argument("--device", default=None)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--registry", type=Path, help="new version path, alongside reviewed exclusions.json")
    args = parser.parse_args()
    records, registry = load_full_registry(EXP, args.registry)
    import pandas as pd
    print(pd.DataFrame(split_summary(records, registry)).to_string(index=False))
    print("Registry SHA-256:", registry["registry_sha256"])
    print("All folds have disjoint train/validation/test sources; each usable clip is tested once.")
    if args.run:
        import torch
        from sjepa.config import get_config
        from sjepa.models import pick_device
        from sjepa.full_experiment import run_cross_validation, new_evaluation_dir
        if args.smoke:
            torch.set_num_threads(2)
        cfg = get_config(smoke=args.smoke)
        updates, more = (4, 2) if args.smoke else (800, 400)
        out = args.output_dir or new_evaluation_dir(EXP, registry, cfg)
        results = run_cross_validation(records, registry, cfg, args.device or pick_device(), updates, more, out)
        print(json.dumps({name: value["source_weighted"]["macro_f1"]
                          for name, value in results["metrics"].items()}, indent=2))
        print("Saved:", out)
        if args.smoke:
            print("SMOKE EXECUTION CHECK ONLY; not a trained-model performance result.")


if __name__ == "__main__":
    main()
