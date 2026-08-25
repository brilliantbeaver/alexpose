"""Build the single-notebook AMASS Core11 JEPA training workflow.

Run: python3 scripts/notebook_builders/idea09/build_amass_training.py
The generated notebook is dry by default; training remains an explicit opt-in.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
NOTEBOOK = ROOT / "notebooks" / "experiments" / "idea09_reflection_equivariance" / "08_amass_core11_training.ipynb"


def markdown(source: str) -> dict:
    source = source.strip("\n") + "\n"
    return {
        "cell_type": "markdown",
        "id": hashlib.sha256(("md:" + source).encode()).hexdigest()[:12],
        "metadata": {},
        "source": source.splitlines(keepends=True),
    }


def code(source: str) -> dict:
    source = source.strip("\n") + "\n"
    return {
        "cell_type": "code",
        "execution_count": None,
        "id": hashlib.sha256(("code:" + source).encode()).hexdigest()[:12],
        "metadata": {},
        "outputs": [],
        "source": source.splitlines(keepends=True),
    }


cells = [
    markdown(r'''# Train the three JEPA variants on AMASS Core11

This is the minimal AMASS counterpart to the CPU/GPU replication launchers. It uses the
converted manifest to build the frozen 64-frame/32-stride index, streams Core11 NPZ windows,
and trains the same three variants under one exposure-matched schedule:

1. `standard`
2. `paired_unconstrained`
3. `reflection_equivariant`

The notebook is dry by default. A smoke run is exactly 20 CUDA updates per variant. A full run
uses all training windows and the full GPU profile. Validation and test identities are indexed
but never used by the optimizer.

```bash
export AMASS_RUN_ROOT=/path/to/amass-run
export AMASS_RUN_TRAINING=1
export AMASS_PROFILE=smoke       # smoke first; then full
export AMASS_DEVICE=cuda
export AMASS_RUN_ID=amass-core11-smoke-v1
```
'''),
    markdown("## 1. Resolve paths and freeze the run configuration"),
    code(r'''from dataclasses import asdict
from pathlib import Path
import json
import math
import os
import platform
import sys

import pandas as pd
import torch
from torch.utils.data import DataLoader


def find_notebook_root(start=None):
    start = Path(start or Path.cwd()).expanduser().resolve()
    relative = Path("experiments") / "sjepa" / "gavd6"
    candidates = []
    override = os.getenv("GAVD6_ROOT")
    if override:
        candidates.append(Path(override).expanduser().resolve())
    for base in (start, *start.parents):
        candidates.extend((base, base / relative))
    for candidate in dict.fromkeys(candidates):
        if ((candidate / "amass_core11_jepa.py").is_file()
                and (candidate / "manifests" / "amass_core11_conversion.csv").is_file()):
            return candidate
    searched = "\n - ".join(str(path) for path in dict.fromkeys(candidates))
    raise FileNotFoundError(
        "Could not locate experiments/sjepa/gavd6. "
        "Set GAVD6_ROOT to that directory.\n"
        f"Searched:\n - {searched}"
    )


PROJECT_DIR = find_notebook_root()
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from amass_core11_jepa import *
from gait_parity_jepa import VARIANTS, build_model, commutation_report, parameter_count


def env_flag(name, default=False):
    value = os.getenv(name)
    return default if value is None else value.strip().lower() in {"1", "true", "yes"}


PROFILE = os.getenv("AMASS_PROFILE", "smoke").strip().lower()
CONFIG = core11_train_config(PROFILE)
RUN_TRAINING = env_flag("AMASS_RUN_TRAINING")
DEVICE_NAME = os.getenv("AMASS_DEVICE", "cuda").strip().lower()
if DEVICE_NAME not in {"cpu", "cuda"}:
    raise ValueError("AMASS_DEVICE must be cpu or cuda")
RUN_ID = os.getenv("AMASS_RUN_ID", f"amass-core11-{PROFILE}-v1")
SEEDS = tuple(int(value) for value in os.getenv(
    "AMASS_SEEDS", "7" if PROFILE == "smoke" else "7,19,31"
).split(","))
NUM_WORKERS = int(os.getenv("AMASS_NUM_WORKERS", "4"))
UPDATE_OVERRIDE = os.getenv("AMASS_UPDATES")
UPDATE_OVERRIDE = int(UPDATE_OVERRIDE) if UPDATE_OVERRIDE else (20 if PROFILE == "smoke" else None)
VALIDATE_ALL_ARCHIVES = env_flag("AMASS_VALIDATE_ALL", PROFILE == "full")

run_root_text = os.getenv("AMASS_RUN_ROOT")
RUN_ROOT = Path(run_root_text).expanduser().resolve() if run_root_text else None
MANIFEST_PATH = Path(os.getenv(
    "AMASS_CONVERSION_MANIFEST",
    (RUN_ROOT / "manifests" / "amass_core11_conversion.csv")
    if RUN_ROOT else (PROJECT_DIR / "manifests" / "amass_core11_conversion.csv"),
)).expanduser().resolve()
TENSOR_ROOT = Path(os.getenv(
    "AMASS_TENSOR_ROOT",
    (RUN_ROOT / "core11") if RUN_ROOT else (PROJECT_DIR / "core11"),
)).expanduser().resolve()
OUTPUT_DIR = Path(os.getenv(
    "AMASS_OUTPUT_DIR",
    PROJECT_DIR / "work" / "artifacts" / "amass_core11_jepa" / RUN_ID,
)).expanduser().resolve()

print({
    "run_training": RUN_TRAINING,
    "profile": PROFILE,
    "run_id": RUN_ID,
    "device": DEVICE_NAME,
    "seeds": SEEDS,
    "manifest": str(MANIFEST_PATH),
    "tensor_root": str(TENSOR_ROOT),
    "output": str(OUTPUT_DIR),
})
'''),
    markdown("## 2. Validate the manifest and build the deterministic window index"),
code(r'''MANIFEST = load_conversion_manifest(MANIFEST_PATH)
WINDOW_INDEX = build_window_index(MANIFEST)
MANIFEST_SHA256 = sha256_file(MANIFEST_PATH)
WINDOW_INDEX_SHA256 = window_index_sha256(WINDOW_INDEX)
CODE_SHA256 = {
    name: sha256_file(PROJECT_DIR / name)
    for name in ("amass_core11_jepa.py", "gait_parity_jepa.py")
}

sequence_counts = MANIFEST.groupby("split").size().rename("sequences")
eligible_counts = MANIFEST.loc[MANIFEST.canonical_frames >= WINDOW_FRAMES].groupby("split").size().rename("eligible_sequences")
window_counts = WINDOW_INDEX.groupby("split").size().rename("windows")
INDEX_SUMMARY = pd.concat([sequence_counts, eligible_counts, window_counts], axis=1).fillna(0).astype(int)
display(INDEX_SUMMARY)

if len(MANIFEST) == 8_854:
    assert int((MANIFEST.canonical_frames < WINDOW_FRAMES).sum()) == 387
    assert len(WINDOW_INDEX) == 93_691
    assert window_counts.to_dict() == {"test": 8_220, "train": 79_535, "validation": 5_936}

print("manifest SHA-256    :", MANIFEST_SHA256)
print("window index SHA-256:", WINDOW_INDEX_SHA256)
print("fingerprints         :", dict(sorted(MANIFEST.groupby("gender").conversion_fingerprint.first().items())))
'''),
    markdown(r'''## 3. Construct split-safe streaming datasets

Every sequence is checked against the conversion contract when first loaded. A full run scans all
archives before training; a smoke run checks only the sequences selected by its fixed 20-update plan.
Each worker retains at most two decompressed sequences. Spaces in AMASS filenames are handled by
`pathlib` and do not require shell quoting inside Python.
'''),
    code(r'''if RUN_TRAINING:
    if DEVICE_NAME == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but torch.cuda.is_available() is false")
    if not TENSOR_ROOT.is_dir():
        raise FileNotFoundError(
            f"Core11 tensor root does not exist: {TENSOR_ROOT}. "
            "Set AMASS_TENSOR_ROOT or AMASS_RUN_ROOT."
        )

    DATASETS = {
        split: Core11WindowDataset(
            MANIFEST, WINDOW_INDEX, TENSOR_ROOT, split=split, cache_sequences=2
        )
        for split in ("train", "validation", "test")
    }
    UPDATES = optimizer_updates(CONFIG, len(DATASETS["train"]), UPDATE_OVERRIDE)
    plan = FixedBatchPlan(len(DATASETS["train"]), CONFIG.batch_size, UPDATES, SEEDS[0] + 17)
    planned_indices = {item for batch in plan for item in batch}
    planned_paths = {
        DATASETS["train"].windows[item]["tensor_relative_path"] for item in planned_indices
    }
    checked = validate_archives(
        MANIFEST,
        TENSOR_ROOT,
        None if VALIDATE_ALL_ARCHIVES else planned_paths,
    )
    sample = DATASETS["train"][0]
    assert sample["coordinates"].shape == (64, 11, 3)
    assert sample["coordinates"].dtype == torch.float32
    assert sample["valid"].shape == (16, 11)
    assert sample["valid"].dtype == torch.bool
    print(f"Validated {checked:,} Core11 archives; updates per variant: {UPDATES:,}")
else:
    DATASETS = None
    UPDATES = optimizer_updates(CONFIG, int(window_counts["train"]), UPDATE_OVERRIDE)
    print("Dry configuration only — set AMASS_RUN_TRAINING=1 to validate tensors and train.")
'''),
    markdown(r'''## 4. Confirm the matched model allocation

All variants receive the same stable window order, target-mask stream, augmented-view stream, and
optimizer-update budget for each seed. Parameter counts are reported rather than claimed to be equal;
weight sharing is part of the architecture under test.
'''),
    code(r'''allocation_rows = []
for variant in VARIANTS:
    model = build_model(CONFIG, variant, SEEDS[0])
    allocation_rows.append({
        "variant": variant,
        "trainable_parameters": parameter_count(model),
        "updates_per_seed": UPDATES,
        "window_exposures_per_seed": UPDATES * CONFIG.batch_size,
    })
ALLOCATION = pd.DataFrame(allocation_rows).set_index("variant")
display(ALLOCATION)
'''),
    markdown(r'''## 5. Train, audit, and checkpoint all three variants

Each checkpoint includes model, EMA teacher, predictor, projector, optimizer, complete configuration,
both data hashes, both accepted conversion fingerprints, and runtime/GPU metadata. It is immediately
reloaded into a fresh model. The reflection-equivariant encoder is also checked layer by layer in both
evaluation and training modes.
'''),
    code(r'''def write_json_atomic(path, payload):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


RUN_ROWS = []
if RUN_TRAINING:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    index_path = OUTPUT_DIR / "window_index.csv"
    temporary_index = index_path.with_suffix(".csv.tmp")
    WINDOW_INDEX.to_csv(temporary_index, index=False)
    os.replace(temporary_index, index_path)
    write_json_atomic(OUTPUT_DIR / "run_config.json", {
        "run_id": RUN_ID,
        "profile": PROFILE,
        "seeds": SEEDS,
        "variants": VARIANTS,
        "optimizer_updates_per_variant": UPDATES,
        "manifest_path": str(MANIFEST_PATH),
        "tensor_root": str(TENSOR_ROOT),
        "manifest_sha256": MANIFEST_SHA256,
        "window_index_sha256": WINDOW_INDEX_SHA256,
        "code_sha256": CODE_SHA256,
        "train_config": asdict(CONFIG),
    })

    device = torch.device(DEVICE_NAME)
    runtime = {
        "python": platform.python_version(),
        "torch": str(torch.__version__),
        "device": str(device),
        "cuda": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
        "code_sha256": CODE_SHA256,
    }
    audit_input = DATASETS["train"][0]["coordinates"].unsqueeze(0)

    for seed in SEEDS:
        for variant in VARIANTS:
            if device.type == "cuda":
                torch.cuda.reset_peak_memory_stats(device)
            model = build_model(CONFIG, variant, seed)
            model, projector, optimizer, history, wall_seconds = train_streaming_variant(
                model,
                DATASETS["train"],
                CONFIG,
                device,
                UPDATES,
                seed,
                num_workers=NUM_WORKERS,
            )
            if not history["total_loss"].map(math.isfinite).all():
                raise FloatingPointError(f"{variant} produced a non-finite loss")

            peak_memory = (
                int(torch.cuda.max_memory_allocated(device)) if device.type == "cuda" else None
            )
            stem = f"seed-{seed}_{variant}"
            history.to_csv(OUTPUT_DIR / f"{stem}_history.csv", index=False)
            payload = checkpoint_payload(
                model,
                projector,
                optimizer,
                variant=variant,
                seed=seed,
                config=CONFIG,
                manifest_sha256=MANIFEST_SHA256,
                window_sha256=WINDOW_INDEX_SHA256,
                updates=UPDATES,
                runtime={**runtime, "wall_seconds": wall_seconds, "peak_cuda_bytes": peak_memory},
            )
            checkpoint_path = OUTPUT_DIR / f"{stem}.pt"
            atomic_torch_save(checkpoint_path, payload)

            reloaded = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
            fresh = build_model(CONFIG, variant, seed)
            fresh.load_state_dict(reloaded["model_state"])
            if reloaded["metadata"]["window_index_sha256"] != WINDOW_INDEX_SHA256:
                raise ValueError("Reloaded checkpoint has the wrong window index hash")

            geometry_max_abs = None
            if variant == "reflection_equivariant":
                reports = []
                for train_mode in (False, True):
                    report = commutation_report(
                        fresh.encoder, audit_input, train_mode=train_mode, device=torch.device("cpu")
                    )
                    reports.append(float(report.max_abs.max()))
                geometry_max_abs = max(reports)
                if geometry_max_abs > 5e-5:
                    raise AssertionError(f"Equivariance audit failed: {geometry_max_abs:.3e}")

            RUN_ROWS.append({
                "seed": seed,
                "variant": variant,
                "updates": UPDATES,
                "first_loss": float(history.total_loss.iloc[0]),
                "last_loss": float(history.total_loss.iloc[-1]),
                "wall_seconds": wall_seconds,
                "peak_cuda_bytes": peak_memory,
                "equivariance_max_abs": geometry_max_abs,
                "checkpoint": str(checkpoint_path),
            })
            print(f"completed {stem}: {history.total_loss.iloc[-1]:.4f}")
            del model, projector, optimizer, fresh, reloaded
            if device.type == "cuda":
                torch.cuda.empty_cache()

    TRAINING_MANIFEST = {
        "run_id": RUN_ID,
        "manifest_sha256": MANIFEST_SHA256,
        "window_index_sha256": WINDOW_INDEX_SHA256,
        "runs": RUN_ROWS,
    }
    write_json_atomic(OUTPUT_DIR / "training_manifest.json", TRAINING_MANIFEST)
    display(pd.DataFrame(RUN_ROWS))
else:
    print("Training skipped.")
'''),
    markdown(r'''## Output contract

A successful one-seed smoke run writes three checkpoints, three loss histories, the exact window
index, a run configuration, and `training_manifest.json`. Only after all three 20-update CUDA runs
reload cleanly and the equivariant geometry audit passes should `AMASS_PROFILE=full` be launched.

The validation and test splits remain untouched representation-evaluation inputs. This notebook does
not tune on them or claim downstream performance.
'''),
]


notebook = {
    "cells": cells,
    "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.11"},
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}

NOTEBOOK.write_text(json.dumps(notebook, indent=1) + "\n", encoding="utf-8")
print(f"Wrote {NOTEBOOK}")
