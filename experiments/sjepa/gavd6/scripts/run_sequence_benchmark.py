#!/usr/bin/env python3
"""Run the sequence-level Latent Laterality benchmark gate."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from gavd6_sjepa.latent_laterality import SequenceGaugeConfig
from gavd6_sjepa.sequence_benchmark import (
    make_manifest_examples,
    run_sequence_benchmark,
    run_synthetic_benchmark,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--synthetic-smoke", action="store_true")
    mode.add_argument("--gauge-manifest", type=Path)
    parser.add_argument("--tensor-root", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--identities", type=int, default=80)
    parser.add_argument("--frames", type=int, default=192)
    args = parser.parse_args()
    if args.gauge_manifest is not None and args.tensor_root is None:
        parser.error("--gauge-manifest requires --tensor-root")
    if args.identities < 12 or args.frames < 64:
        parser.error("Synthetic smoke needs at least 12 identities and 64 frames")
    return args


def main() -> int:
    args = parse_args()
    if args.synthetic_smoke:
        gates = run_synthetic_benchmark(
            args.output_dir,
            seed=args.seed,
            identities=args.identities,
            frames=args.frames,
        )
    else:
        config = SequenceGaugeConfig()
        examples = make_manifest_examples(
            args.gauge_manifest.resolve(),
            args.tensor_root.resolve(),
            config=config,
        )
        gates = run_sequence_benchmark(
            args.output_dir,
            seed=args.seed,
            config=config,
            examples=examples,
            synthetic_smoke=False,
        )
    print(json.dumps(gates, indent=2, sort_keys=True))
    return 0 if gates["ready_for_sg_jepa"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
