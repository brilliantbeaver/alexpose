"""Minimal source-transfer screening trainer for Latent Laterality."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from gavd6_sjepa.shared_infrastructure.artifact_io_operations import atomic_write_json as _write_json
from gavd6_sjepa.research_directions.reflection_equivariance.amass_core11_training_pipeline import (
    Core11WindowDataset,
    build_window_index,
    core11_train_config,
    fit_variant,
    load_conversion_manifest,
)
from gavd6_sjepa.research_directions.reflection_equivariance.jepa_model_architecture import AMASS_VARIANTS, build_model


SOURCE_ROUTES = ("amass-only", "gavd-only", "amass-to-gavd")
SCREEN_VARIANTS = ("standard_sjepa", "reflection_equivariant")


def _datasets(manifest_path: Path, tensor_root: Path, *, gavd: bool):
    manifest = load_conversion_manifest(manifest_path)
    windows = build_window_index(manifest)
    train = Core11WindowDataset(
        manifest,
        windows,
        tensor_root,
        split="train",
        balance_source_groups=gavd,
    )
    validation = Core11WindowDataset(
        manifest, windows, tensor_root, split="validation"
    )
    return train, validation, {
        "sequences": int(len(manifest)),
        "train_windows": len(train),
        "validation_windows": len(validation),
        "train_groups": int(manifest.loc[manifest.split == "train", "identity"].nunique()),
        "validation_groups": int(
            manifest.loc[manifest.split == "validation", "identity"].nunique()
        ),
        "source_video_balanced": bool(gavd),
        "coordinate_frames": (
            sorted(manifest.coordinate_frame.astype(str).unique().tolist())
            if "coordinate_frame" in manifest
            else ["undeclared"]
        ),
    }


def run(args: argparse.Namespace) -> dict:
    output_dir = args.output_dir.resolve()
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"Output directory is not empty: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    config = core11_train_config(args.profile)
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")

    required_amass = args.route in {"amass-only", "amass-to-gavd"}
    required_gavd = args.route in {"gavd-only", "amass-to-gavd"}
    if required_amass and (args.amass_manifest is None or args.amass_tensor_root is None):
        raise ValueError("This route requires --amass-manifest and --amass-tensor-root")
    if required_gavd and (args.gavd_manifest is None or args.gavd_tensor_root is None):
        raise ValueError("This route requires --gavd-manifest and --gavd-tensor-root")

    model = build_model(config, args.variant, args.seed)
    stages = []
    data_summary = {}
    if required_amass:
        train, validation, summary = _datasets(
            args.amass_manifest.resolve(), args.amass_tensor_root.resolve(), gavd=False
        )
        stage_dir = output_dir / "stage-amass"
        model, _, _, result, _ = fit_variant(
            model,
            train,
            validation,
            device,
            seed=args.seed,
            output_dir=stage_dir,
            num_workers=args.num_workers,
            max_epochs=args.max_epochs,
            patience=args.patience,
        )
        stages.append(result)
        data_summary["amass"] = summary
    if required_gavd:
        train, validation, summary = _datasets(
            args.gavd_manifest.resolve(), args.gavd_tensor_root.resolve(), gavd=True
        )
        stage_dir = output_dir / "stage-gavd"
        model, _, _, result, _ = fit_variant(
            model,
            train,
            validation,
            device,
            seed=args.seed,
            output_dir=stage_dir,
            num_workers=args.num_workers,
            max_epochs=args.max_epochs,
            patience=args.patience,
        )
        stages.append(result)
        data_summary["gavd"] = summary
    payload = {
        "route": args.route,
        "variant": args.variant,
        "seed": args.seed,
        "profile": args.profile,
        "device": str(device),
        "test_split_evaluated": False,
        "condition_labels_in_jepa_loss": False,
        "data": data_summary,
        "stages": stages,
    }
    _write_json(output_dir / "route_result.json", payload)
    return payload


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--route", choices=SOURCE_ROUTES, required=True)
    parser.add_argument("--variant", choices=AMASS_VARIANTS, required=True)
    parser.add_argument("--amass-manifest", type=Path)
    parser.add_argument("--amass-tensor-root", type=Path)
    parser.add_argument("--gavd-manifest", type=Path)
    parser.add_argument("--gavd-tensor-root", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--profile", choices=("smoke", "full"), default="full")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--max-epochs", type=int)
    parser.add_argument("--patience", type=int)
    return parser.parse_args(argv)


def main() -> int:
    payload = run(parse_args())
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
