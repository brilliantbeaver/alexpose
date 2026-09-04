#!/usr/bin/env -S uv run --no-sync python
"""Run frozen AMASS-Core11 EMA-target probes on the canonical GAVD cohort."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys


PROJECT_DIR = Path(__file__).resolve().parents[4]
if str(PROJECT_DIR / "src") not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR / "src"))

from .jepa_model_architecture import resolve_pose_dir
from .gavd_core11_probe_evaluation import run_probe_experiment


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pose-dir", type=Path)
    parser.add_argument(
        "--checkpoint-dir",
        type=Path,
        default=Path("outputs/repaired-jepa-seed7-v2"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("work/artifacts/gavd_core11_frozen_probe"),
    )
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    project_dir = PROJECT_DIR
    pose_dir = args.pose_dir or resolve_pose_dir(project_dir)
    checkpoint_dir = args.checkpoint_dir
    output_dir = args.output_dir
    if not checkpoint_dir.is_absolute():
        checkpoint_dir = project_dir / checkpoint_dir
    if not output_dir.is_absolute():
        output_dir = project_dir / output_dir
    result = run_probe_experiment(
        pose_dir=pose_dir,
        checkpoint_dir=checkpoint_dir,
        output_dir=output_dir,
        device=args.device,
        batch_size=args.batch_size,
        seed=args.seed,
    )
    print(
        "Cohort:",
        len(result["sequence_table"]),
        "annotated sequences from",
        result["sequence_table"]["video_id"].nunique(),
        "source videos",
    )
    print("Evaluation: within-corpus, source-confounded descriptive nested ridge probes")
    print("All-96 zero/invalid-padded sensitivity:")
    print(result["summary"].to_string(index=False))
    print("Strict no-short-padding sensitivity:")
    print(result["strict_summary"].to_string(index=False))
    print(f"Saved: {result['output_dir']}")


if __name__ == "__main__":
    main()
