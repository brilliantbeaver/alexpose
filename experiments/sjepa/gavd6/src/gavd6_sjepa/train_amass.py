"""Run the validation-selected AMASS Core11 JEPA comparison."""

from __future__ import annotations

from dataclasses import asdict, replace
import json
import os
from pathlib import Path

import pandas as pd
import torch

from .amass_core11_jepa import (
    Core11WindowDataset,
    WINDOW_FRAMES,
    atomic_dataframe_to_csv,
    build_window_index,
    configure_worker_tensor_sharing,
    core11_train_config,
    evaluate_variant,
    fit_variant,
    load_conversion_manifest,
    make_synthetic_core11_datasets,
)
from .gait_parity_jepa import (
    VARIANTS,
    VICRegProjector,
    build_model,
    trainable_parameter_count,
)


PROJECT_DIR = Path(__file__).resolve().parents[2]


def env_flag(name: str, default: bool = False) -> bool:
    value = os.getenv(name)
    return default if value is None else value.strip().lower() in {"1", "true", "yes"}


def write_json(path: Path, payload: object) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    os.replace(temporary, path)


def _paths(run_root: Path | None, run_id: str) -> tuple[Path, Path, Path]:
    manifest = Path(
        os.getenv(
            "AMASS_CONVERSION_MANIFEST",
            (run_root / "manifests" / "amass_core11_conversion.csv")
            if run_root
            else (PROJECT_DIR / "manifests" / "amass_core11_conversion.csv"),
        )
    ).expanduser().resolve()
    tensors = Path(
        os.getenv(
            "AMASS_TENSOR_ROOT",
            (run_root / "core11") if run_root else (PROJECT_DIR / "core11"),
        )
    ).expanduser().resolve()
    output = Path(
        os.getenv(
            "AMASS_OUTPUT_DIR",
            PROJECT_DIR / "work" / "artifacts" / "amass_core11_jepa" / run_id,
        )
    ).expanduser().resolve()
    return manifest, tensors, output


def main() -> None:
    profile = os.getenv("AMASS_PROFILE", "smoke").strip().lower()
    config = core11_train_config(profile)
    epochs_text = os.getenv("AMASS_EPOCHS")
    patience_text = os.getenv("AMASS_PATIENCE")
    draws_text = os.getenv("AMASS_EVALUATION_DRAWS")
    if epochs_text:
        config = replace(config, epochs=int(epochs_text))
    if patience_text:
        config = replace(config, early_stopping_patience=int(patience_text))
    if draws_text:
        config = replace(config, evaluation_draws=int(draws_text))
    run_training = env_flag("AMASS_RUN_TRAINING")
    synthetic = env_flag("AMASS_SYNTHETIC_SMOKE")
    evaluate_test = env_flag("AMASS_EVALUATE_TEST", profile == "smoke")
    if synthetic and profile != "smoke":
        raise ValueError("AMASS_SYNTHETIC_SMOKE is only valid with AMASS_PROFILE=smoke")
    device_name = os.getenv("AMASS_DEVICE", "cuda").strip().lower()
    if device_name not in {"cpu", "cuda"}:
        raise ValueError("AMASS_DEVICE must be cpu or cuda")
    run_id = os.getenv("AMASS_RUN_ID", f"amass-core11-{profile}-repaired-v1")
    seeds = tuple(
        int(value)
        for value in os.getenv("AMASS_SEEDS", "7").split(",")
    )
    num_workers = int(os.getenv("AMASS_NUM_WORKERS", "4"))
    if num_workers < 0:
        raise ValueError("AMASS_NUM_WORKERS must be non-negative")
    sharing_strategy = (
        configure_worker_tensor_sharing() if num_workers else "single_process"
    )
    run_root_text = os.getenv("AMASS_RUN_ROOT")
    run_root = Path(run_root_text).expanduser().resolve() if run_root_text else None
    manifest_path, tensor_root, output_dir = _paths(run_root, run_id)

    if synthetic:
        datasets = make_synthetic_core11_datasets()
        index_summary = pd.DataFrame(
            {"windows": {split: len(dataset) for split, dataset in datasets.items()}}
        )
        window_index = None
    else:
        manifest = load_conversion_manifest(manifest_path)
        window_index = build_window_index(manifest)
        sequence_counts = manifest.groupby("split").size().rename("sequences")
        eligible_counts = (
            manifest.loc[manifest.canonical_frames >= WINDOW_FRAMES]
            .groupby("split")
            .size()
            .rename("eligible_sequences")
        )
        window_counts = window_index.groupby("split").size().rename("windows")
        index_summary = (
            pd.concat([sequence_counts, eligible_counts, window_counts], axis=1)
            .fillna(0)
            .astype(int)
        )
        datasets = None

    print(
        {
            "run_training": run_training,
            "profile": profile,
            "synthetic_smoke": synthetic,
            "evaluate_test": evaluate_test,
            "run_id": run_id,
            "device": device_name,
            "seeds": seeds,
            "num_workers": num_workers,
            "tensor_sharing_strategy": sharing_strategy,
            "epochs": config.epochs,
            "patience": config.early_stopping_patience,
            "manifest": None if synthetic else str(manifest_path),
            "tensor_root": None if synthetic else str(tensor_root),
            "output": str(output_dir),
        },
        flush=True,
    )
    print(index_summary.to_string(), flush=True)

    allocation_rows = []
    for variant in VARIANTS:
        model = build_model(config, variant, seeds[0])
        projector = VICRegProjector(model.config.embed_dim)
        allocation_rows.append(
            {
                "variant": variant,
                "embed_dim": model.config.embed_dim,
                "feedforward_dim": model.config.feedforward_dim,
                "trainable_parameters": trainable_parameter_count(model, projector),
            }
        )
    allocation = pd.DataFrame(allocation_rows)
    counts = allocation.trainable_parameters
    capacity_spread = float(counts.max() / counts.min() - 1)
    print(allocation.set_index("variant").to_string(), flush=True)
    print(f"Trainable-capacity spread: {capacity_spread:.2%}", flush=True)
    if capacity_spread > 0.05:
        raise RuntimeError("Trainable parameter counts differ by more than 5%; refusing to train")

    if not run_training:
        print("Dry configuration only. Set AMASS_RUN_TRAINING=1 to train.", flush=True)
        return
    if device_name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but torch.cuda.is_available() is false")
    if not synthetic:
        if not tensor_root.is_dir():
            raise FileNotFoundError(
                f"Core11 tensor root does not exist: {tensor_root}. "
                "Set AMASS_TENSOR_ROOT or AMASS_RUN_ROOT."
            )
        datasets = {
            split: Core11WindowDataset(
                manifest, window_index, tensor_root, split=split, cache_sequences=2
            )
            for split in ("train", "validation", "test")
        }
        for split, dataset in datasets.items():
            if not len(dataset):
                raise ValueError(f"{split} split has no windows")
        _ = datasets["train"][0]

    output_dir.mkdir(parents=True, exist_ok=True)
    atomic_dataframe_to_csv(allocation, output_dir / "capacity.csv")
    write_json(
        output_dir / "run_config.json",
        {
            "run_id": run_id,
            "profile": profile,
            "synthetic_smoke": synthetic,
            "evaluate_test": evaluate_test,
            "seeds": seeds,
            "num_workers": num_workers,
            "tensor_sharing_strategy": sharing_strategy,
            "variants": VARIANTS,
            "capacity_spread": capacity_spread,
            "manifest_path": None if synthetic else str(manifest_path),
            "tensor_root": None if synthetic else str(tensor_root),
            "train_config": asdict(config),
        },
    )

    device = torch.device(device_name)
    run_rows = []
    for seed in seeds:
        for variant in VARIANTS:
            if device.type == "cuda":
                torch.cuda.reset_peak_memory_stats(device)
            model = build_model(config, variant, seed)
            parameters = int(
                allocation.loc[
                    allocation.variant == variant, "trainable_parameters"
                ].iloc[0]
            )
            best_model, best_projector, _history, result, _ = fit_variant(
                model,
                datasets["train"],
                datasets["validation"],
                device,
                seed=seed,
                output_dir=output_dir,
                num_workers=num_workers,
            )
            stem = f"seed-{seed}_{variant}"
            test_metrics = None
            if evaluate_test:
                # Test is opt-in for full runs and occurs only after best-checkpoint reload.
                best_model = best_model.to(device)
                best_projector = best_projector.to(device)
                test_metrics, _ = evaluate_variant(
                    best_model,
                    best_projector,
                    datasets["test"],
                    device,
                    seed=seed,
                    split="test",
                    num_workers=num_workers,
                )
            peak_memory = (
                int(torch.cuda.max_memory_allocated(device))
                if device.type == "cuda"
                else None
            )
            row = {
                "seed": seed,
                "variant": variant,
                "trainable_parameters": parameters,
                "wall_seconds": result["wall_seconds"],
                "selected_epoch": result["selected_epoch"],
                "epochs_trained": result["epochs_trained"],
                "optimizer_updates": result["optimizer_updates"],
                "peak_cuda_bytes": peak_memory,
                "best_checkpoint": result["best_checkpoint"],
            }
            row.update(
                {f"validation_{name}": value for name, value in result["validation"].items()}
            )
            if test_metrics is not None:
                row.update({f"test_{name}": value for name, value in test_metrics.items()})
            run_rows.append(row)
            # Update the one allowed summary artifact after each completed
            # seed/variant, rather than waiting for the entire matrix to finish.
            atomic_dataframe_to_csv(pd.DataFrame(run_rows), output_dir / "summary.csv")
            message = (
                f"completed {stem}: selected epoch {result['selected_epoch']}, "
                f"validation KL {result['validation']['kl_divergence']:.4f}"
            )
            if test_metrics is not None:
                message += f", test KL {test_metrics['kl_divergence']:.4f}"
            print(message, flush=True)
            del best_model, best_projector, model
            if device.type == "cuda":
                torch.cuda.empty_cache()

    summary = pd.DataFrame(run_rows)
    print(summary.to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
