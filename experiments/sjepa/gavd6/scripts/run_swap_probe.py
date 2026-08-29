#!/usr/bin/env python3
"""Run the validation-only swap probe."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd
import torch

from gavd6_sjepa.amass_core11_jepa import (
    Core11WindowDataset,
    SyntheticCore11Dataset,
    build_window_index,
    configure_worker_tensor_sharing,
    core11_train_config,
    load_conversion_manifest,
)
from gavd6_sjepa.gait_parity_jepa import build_model, load_checkpoint
from gavd6_sjepa.swap_probe import (
    CorruptionConfig,
    run_probe,
    select_nonoverlapping_windows,
    sha256_file,
)


PROJECT_DIR = Path(__file__).resolve().parents[1]
DEFAULT_CHECKPOINT = (
    PROJECT_DIR
    / "outputs"
    / "repaired-jepa-seed7-v2"
    / "seed-7_standard_sjepa_best.pt"
)
REFERENCE_CHECKPOINT_SHA256 = "d12ddf0a8412bcae58ed167cc1ec560b978bffa590fd631ab23e519216b646bd"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Train lightweight semantic-switch/probe heads on AMASS train identities "
            "and evaluate five correction arms on validation identities."
        )
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=PROJECT_DIR / "manifests" / "amass_core11_conversion.csv",
    )
    parser.add_argument("--tensor-root", type=Path)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument(
        "--expected-checkpoint-sha256",
        default=REFERENCE_CHECKPOINT_SHA256,
        help="Required digest for the frozen reference checkpoint; use only the exact seed-7 artifact.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_DIR / "outputs" / "swap-probe-seed7",
    )
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--encoder-batch-size", type=int, default=256)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--max-train-windows", type=int, default=20_000)
    parser.add_argument("--max-validation-windows", type=int, default=2_800)
    parser.add_argument("--swapped-blocks", type=int, default=4)
    parser.add_argument("--noise-std", type=float, default=0.01)
    parser.add_argument("--occlusion-probability", type=float, default=0.10)
    parser.add_argument("--event-probability", type=float, default=0.80)
    parser.add_argument(
        "--sensor-reflection-probability", type=float, default=0.50
    )
    parser.add_argument(
        "--synthetic-smoke",
        action="store_true",
        help="Use small synthetic splits and an untrained smoke encoder to test plumbing.",
    )
    return parser.parse_args()


def _positive(name: str, value: int) -> int:
    if value < 1:
        raise ValueError(f"{name} must be positive")
    return value


def _synthetic_inputs(seed: int):
    sizes = {"train": 32, "validation": 16}
    datasets = {
        split: SyntheticCore11Dataset(split, size, seed + offset)
        for offset, (split, size) in enumerate(sizes.items(), start=1)
    }
    rows = {}
    for split, dataset in datasets.items():
        rows[split] = pd.DataFrame(
            {
                "window_id": [f"{split}:{index}" for index in range(len(dataset))],
                "identity": [f"synthetic::{split}::{index}" for index in range(len(dataset))],
                "tensor_relative_path": [f"synthetic/{split}/{index}" for index in range(len(dataset))],
                "start_frame": [0] * len(dataset),
                "split": [split] * len(dataset),
            }
        )
    model = build_model(core11_train_config("smoke"), "standard_sjepa", seed)
    return datasets, rows, model.target_encoder, {
        "synthetic_smoke": True,
        "checkpoint": None,
        "manifest": None,
        "tensor_root": None,
    }


def _real_inputs(args: argparse.Namespace):
    if args.tensor_root is None:
        raise ValueError("--tensor-root is required unless --synthetic-smoke is used")
    manifest_path = args.manifest.expanduser().resolve()
    tensor_root = args.tensor_root.expanduser().resolve()
    checkpoint = args.checkpoint.expanduser().resolve()
    if not manifest_path.is_file():
        raise FileNotFoundError(manifest_path)
    if not tensor_root.is_dir():
        raise FileNotFoundError(tensor_root)
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    checkpoint_sha256 = sha256_file(checkpoint)
    if checkpoint_sha256 != args.expected_checkpoint_sha256.lower():
        raise ValueError(
            "Checkpoint digest does not match the frozen seed-7 standard S-JEPA reference. "
            "Pass the exact artifact or explicitly set its expected digest."
        )

    manifest = load_conversion_manifest(manifest_path)
    index = build_window_index(manifest)
    rows = {
        "train": select_nonoverlapping_windows(
            index, "train", args.max_train_windows, args.seed
        ),
        "validation": select_nonoverlapping_windows(
            index, "validation", args.max_validation_windows, args.seed
        ),
    }
    train_identities = set(rows["train"]["identity"])
    validation_identities = set(rows["validation"]["identity"])
    if train_identities & validation_identities:
        raise ValueError("Train and validation identities overlap")
    selected_index = pd.concat((rows["train"], rows["validation"]), ignore_index=True)
    datasets = {
        split: Core11WindowDataset(
            manifest,
            selected_index,
            tensor_root,
            split=split,
            cache_sequences=2,
        )
        for split in ("train", "validation")
    }
    model, _, metadata = load_checkpoint(checkpoint)
    if model.variant != "standard_sjepa" or int(metadata.get("seed", -1)) != 7:
        raise ValueError("The bilateral-correction probe requires the frozen standard_sjepa checkpoint")
    return datasets, rows, model.target_encoder, {
        "synthetic_smoke": False,
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": checkpoint_sha256,
        "checkpoint_metadata": metadata,
        "manifest": str(manifest_path),
        "manifest_sha256": sha256_file(manifest_path),
        "tensor_root": str(tensor_root),
    }


def main() -> None:
    args = parse_args()
    for name in (
        "batch_size",
        "encoder_batch_size",
        "num_workers",
        "max_train_windows",
        "max_validation_windows",
    ):
        value = getattr(args, name)
        if name == "num_workers":
            if value < 0:
                raise ValueError("num_workers must be non-negative")
        else:
            _positive(name, value)
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but torch.cuda.is_available() is false")
    if args.num_workers:
        configure_worker_tensor_sharing()

    corruption = CorruptionConfig(
        swapped_blocks=args.swapped_blocks,
        noise_std=args.noise_std,
        occlusion_probability=args.occlusion_probability,
        sensor_reflection_probability=args.sensor_reflection_probability,
        event_probability=args.event_probability,
    )
    corruption.validate()
    if args.synthetic_smoke:
        datasets, rows, encoder, provenance = _synthetic_inputs(args.seed)
    else:
        datasets, rows, encoder, provenance = _real_inputs(args)
    provenance["command"] = " ".join(sys.argv)

    device = torch.device(args.device)
    encoder = encoder.to(device)
    print(
        "[swap-probe] starting: "
        f"device={device}, train_windows={len(rows['train'])}, "
        f"validation_windows={len(rows['validation'])}, output={args.output_dir.resolve()}",
        flush=True,
    )
    if not args.synthetic_smoke:
        print(
            "[swap-probe] inputs: "
            f"manifest={provenance['manifest']}, tensor_root={provenance['tensor_root']}, "
            f"checkpoint={provenance['checkpoint']}, "
            f"checkpoint_sha256={provenance['checkpoint_sha256']}",
            flush=True,
        )
    result = run_probe(
        train_dataset=datasets["train"],
        train_rows=rows["train"],
        validation_dataset=datasets["validation"],
        validation_rows=rows["validation"],
        encoder=encoder,
        device=device,
        output_dir=args.output_dir,
        corruption=corruption,
        seed=args.seed,
        batch_size=args.batch_size,
        encoder_batch_size=args.encoder_batch_size,
        num_workers=args.num_workers,
        provenance=provenance,
        progress=True,
    )
    print(result["edge_metrics"].to_string(index=False), flush=True)
    print(result["summary"].to_string(index=False), flush=True)
    print(f"Wrote complete probe artifacts to {args.output_dir.resolve()}", flush=True)


if __name__ == "__main__":
    main()
