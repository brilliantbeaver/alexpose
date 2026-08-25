"""Run the AMASS Core11 JEPA training workflow without a Jupyter kernel.

The environment-variable contract matches
``notebooks/experiments/idea09_reflection_equivariance/08_amass_core11_training.ipynb``.
Training is deliberately opt-in: set ``AMASS_RUN_TRAINING=1``.
"""

from __future__ import annotations

from dataclasses import asdict
import json
import math
import os
from pathlib import Path
import platform
import sys

import pandas as pd
import torch


PROJECT_DIR = Path(__file__).resolve().parents[1]
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from amass_core11_jepa import (
    WINDOW_FRAMES,
    Core11WindowDataset,
    FixedBatchPlan,
    atomic_torch_save,
    build_window_index,
    checkpoint_payload,
    core11_train_config,
    load_conversion_manifest,
    optimizer_updates,
    sha256_file,
    train_streaming_variant,
    validate_archives,
    window_index_sha256,
)
from gait_parity_jepa import VARIANTS, build_model, commutation_report, parameter_count


def env_flag(name: str, default: bool = False) -> bool:
    value = os.getenv(name)
    return default if value is None else value.strip().lower() in {"1", "true", "yes"}


def write_json_atomic(path: Path, payload: object) -> None:
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def main() -> None:
    profile = os.getenv("AMASS_PROFILE", "smoke").strip().lower()
    config = core11_train_config(profile)
    run_training = env_flag("AMASS_RUN_TRAINING")
    device_name = os.getenv("AMASS_DEVICE", "cuda").strip().lower()
    if device_name not in {"cpu", "cuda"}:
        raise ValueError("AMASS_DEVICE must be cpu or cuda")
    run_id = os.getenv("AMASS_RUN_ID", f"amass-core11-{profile}-v1")
    seeds = tuple(
        int(value)
        for value in os.getenv("AMASS_SEEDS", "7" if profile == "smoke" else "7,19,31").split(",")
    )
    num_workers = int(os.getenv("AMASS_NUM_WORKERS", "4"))
    update_override_text = os.getenv("AMASS_UPDATES")
    update_override = (
        int(update_override_text)
        if update_override_text
        else (20 if profile == "smoke" else None)
    )
    validate_all_archives = env_flag("AMASS_VALIDATE_ALL", profile == "full")

    run_root_text = os.getenv("AMASS_RUN_ROOT")
    run_root = Path(run_root_text).expanduser().resolve() if run_root_text else None
    manifest_path = Path(
        os.getenv(
            "AMASS_CONVERSION_MANIFEST",
            (run_root / "manifests" / "amass_core11_conversion.csv")
            if run_root
            else (PROJECT_DIR / "manifests" / "amass_core11_conversion.csv"),
        )
    ).expanduser().resolve()
    tensor_root = Path(
        os.getenv(
            "AMASS_TENSOR_ROOT",
            (run_root / "core11") if run_root else (PROJECT_DIR / "core11"),
        )
    ).expanduser().resolve()
    output_dir = Path(
        os.getenv(
            "AMASS_OUTPUT_DIR",
            PROJECT_DIR / "work" / "artifacts" / "amass_core11_jepa" / run_id,
        )
    ).expanduser().resolve()

    print(
        {
            "run_training": run_training,
            "profile": profile,
            "run_id": run_id,
            "device": device_name,
            "seeds": seeds,
            "manifest": str(manifest_path),
            "tensor_root": str(tensor_root),
            "output": str(output_dir),
        },
        flush=True,
    )

    manifest = load_conversion_manifest(manifest_path)
    window_index = build_window_index(manifest)
    manifest_sha256 = sha256_file(manifest_path)
    window_index_sha256_value = window_index_sha256(window_index)
    code_sha256 = {
        name: sha256_file(PROJECT_DIR / name)
        for name in ("amass_core11_jepa.py", "gait_parity_jepa.py")
    }

    sequence_counts = manifest.groupby("split").size().rename("sequences")
    eligible_counts = (
        manifest.loc[manifest.canonical_frames >= WINDOW_FRAMES]
        .groupby("split")
        .size()
        .rename("eligible_sequences")
    )
    window_counts = window_index.groupby("split").size().rename("windows")
    index_summary = pd.concat([sequence_counts, eligible_counts, window_counts], axis=1).fillna(0).astype(int)
    print(index_summary.to_string(), flush=True)

    if len(manifest) == 8_854:
        assert int((manifest.canonical_frames < WINDOW_FRAMES).sum()) == 387
        assert len(window_index) == 93_691
        assert window_counts.to_dict() == {"test": 8_220, "train": 79_535, "validation": 5_936}

    print("manifest SHA-256    :", manifest_sha256, flush=True)
    print("window index SHA-256:", window_index_sha256_value, flush=True)
    print(
        "fingerprints         :",
        dict(sorted(manifest.groupby("gender").conversion_fingerprint.first().items())),
        flush=True,
    )

    if not run_training:
        updates = optimizer_updates(config, int(window_counts["train"]), update_override)
        print(
            f"Dry configuration only; updates per variant: {updates:,}. "
            "Set AMASS_RUN_TRAINING=1 to validate tensors and train.",
            flush=True,
        )
        return

    if device_name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but torch.cuda.is_available() is false")
    if not tensor_root.is_dir():
        raise FileNotFoundError(
            f"Core11 tensor root does not exist: {tensor_root}. "
            "Set AMASS_TENSOR_ROOT or AMASS_RUN_ROOT."
        )

    datasets = {
        split: Core11WindowDataset(manifest, window_index, tensor_root, split=split, cache_sequences=2)
        for split in ("train", "validation", "test")
    }
    updates = optimizer_updates(config, len(datasets["train"]), update_override)
    plan = FixedBatchPlan(len(datasets["train"]), config.batch_size, updates, seeds[0] + 17)
    planned_indices = {item for batch in plan for item in batch}
    planned_paths = {datasets["train"].windows[item]["tensor_relative_path"] for item in planned_indices}
    checked = validate_archives(
        manifest,
        tensor_root,
        None if validate_all_archives else planned_paths,
    )
    sample = datasets["train"][0]
    assert sample["coordinates"].shape == (64, 11, 3)
    assert sample["coordinates"].dtype == torch.float32
    assert sample["valid"].shape == (16, 11)
    assert sample["valid"].dtype == torch.bool
    print(f"Validated {checked:,} Core11 archives; updates per variant: {updates:,}", flush=True)

    allocation_rows = []
    for variant in VARIANTS:
        model = build_model(config, variant, seeds[0])
        allocation_rows.append(
            {
                "variant": variant,
                "trainable_parameters": parameter_count(model),
                "updates_per_seed": updates,
                "window_exposures_per_seed": updates * config.batch_size,
            }
        )
    print(pd.DataFrame(allocation_rows).set_index("variant").to_string(), flush=True)

    output_dir.mkdir(parents=True, exist_ok=True)
    index_path = output_dir / "window_index.csv"
    temporary_index = index_path.with_suffix(".csv.tmp")
    window_index.to_csv(temporary_index, index=False)
    os.replace(temporary_index, index_path)
    write_json_atomic(
        output_dir / "run_config.json",
        {
            "run_id": run_id,
            "profile": profile,
            "seeds": seeds,
            "variants": VARIANTS,
            "optimizer_updates_per_variant": updates,
            "manifest_path": str(manifest_path),
            "tensor_root": str(tensor_root),
            "manifest_sha256": manifest_sha256,
            "window_index_sha256": window_index_sha256_value,
            "code_sha256": code_sha256,
            "train_config": asdict(config),
        },
    )

    device = torch.device(device_name)
    runtime = {
        "python": platform.python_version(),
        "torch": str(torch.__version__),
        "device": str(device),
        "cuda": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
        "code_sha256": code_sha256,
    }
    audit_input = datasets["train"][0]["coordinates"].unsqueeze(0)
    run_rows = []

    for seed in seeds:
        for variant in VARIANTS:
            if device.type == "cuda":
                torch.cuda.reset_peak_memory_stats(device)
            model = build_model(config, variant, seed)
            model, projector, optimizer, history, wall_seconds = train_streaming_variant(
                model,
                datasets["train"],
                config,
                device,
                updates,
                seed,
                num_workers=num_workers,
            )
            if not history["total_loss"].map(math.isfinite).all():
                raise FloatingPointError(f"{variant} produced a non-finite loss")

            peak_memory = int(torch.cuda.max_memory_allocated(device)) if device.type == "cuda" else None
            stem = f"seed-{seed}_{variant}"
            history.to_csv(output_dir / f"{stem}_history.csv", index=False)
            payload = checkpoint_payload(
                model,
                projector,
                optimizer,
                variant=variant,
                seed=seed,
                config=config,
                manifest_sha256=manifest_sha256,
                window_sha256=window_index_sha256_value,
                updates=updates,
                runtime={**runtime, "wall_seconds": wall_seconds, "peak_cuda_bytes": peak_memory},
            )
            checkpoint_path = output_dir / f"{stem}.pt"
            atomic_torch_save(checkpoint_path, payload)

            reloaded = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
            fresh = build_model(config, variant, seed)
            fresh.load_state_dict(reloaded["model_state"])
            if reloaded["metadata"]["window_index_sha256"] != window_index_sha256_value:
                raise ValueError("Reloaded checkpoint has the wrong window index hash")

            geometry_max_abs = None
            if variant == "reflection_equivariant":
                reports = []
                for train_mode in (False, True):
                    report = commutation_report(
                        fresh.encoder,
                        audit_input,
                        train_mode=train_mode,
                        device=torch.device("cpu"),
                    )
                    reports.append(float(report.max_abs.max()))
                geometry_max_abs = max(reports)
                if geometry_max_abs > 5e-5:
                    raise AssertionError(f"Equivariance audit failed: {geometry_max_abs:.3e}")

            run_rows.append(
                {
                    "seed": seed,
                    "variant": variant,
                    "updates": updates,
                    "first_loss": float(history.total_loss.iloc[0]),
                    "last_loss": float(history.total_loss.iloc[-1]),
                    "wall_seconds": wall_seconds,
                    "peak_cuda_bytes": peak_memory,
                    "equivariance_max_abs": geometry_max_abs,
                    "checkpoint": str(checkpoint_path),
                }
            )
            print(f"completed {stem}: {history.total_loss.iloc[-1]:.4f}", flush=True)
            del model, projector, optimizer, fresh, reloaded
            if device.type == "cuda":
                torch.cuda.empty_cache()

    training_manifest = {
        "run_id": run_id,
        "manifest_sha256": manifest_sha256,
        "window_index_sha256": window_index_sha256_value,
        "runs": run_rows,
    }
    write_json_atomic(output_dir / "training_manifest.json", training_manifest)
    print(pd.DataFrame(run_rows).to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
