from __future__ import annotations

import hashlib
import json
import math
import os
import platform
import random
import tempfile
import time
import warnings
from collections import Counter
from pathlib import Path
from typing import Any, Callable, Iterator

import numpy as np
import sklearn
import torch

from .artifacts import checkpoint_path
from .config import ExperimentContext, canonical_json_digest, model_config
from .data import PreparedCohort
from .model import (
    SJEPAGait,
    VICRegProjector,
    anatomical_reflect_tensor,
    authorized_pool,
    cosine_ema,
    geometric_view,
    sjepa_cross_entropy,
    uniform_authorized_mask,
    valid_patches,
    vicreg_loss,
)
from .splitting import get_fold


ProgressCallback = Callable[[dict[str, Any]], None]

_IMPLEMENTATION_COMPATIBILITY_PATH = (
    Path(__file__).resolve().parent / "checkpoint_compatibility.json"
)


def implementation_digest() -> str:
    digest = hashlib.sha256()
    module_root = Path(__file__).resolve().parent
    for path in sorted(module_root.glob("*.py")):
        digest.update(path.name.encode("utf-8"))
        digest.update(path.read_bytes())
    return digest.hexdigest()


def _approved_implementation_compatibility() -> set[tuple[str, str]]:
    """Return narrowly reviewed (checkpoint, current-code) digest pairs."""
    try:
        payload = json.loads(_IMPLEMENTATION_COMPATIBILITY_PATH.read_text())
    except FileNotFoundError:
        return set()
    if payload.get("schema") != "neurips_laterality_checkpoint_compatibility/v1":
        raise RuntimeError("Unsupported checkpoint compatibility manifest")
    pairs = payload.get("compatible_pairs")
    if not isinstance(pairs, list):
        raise RuntimeError("Checkpoint compatibility manifest has no pair list")
    approved: set[tuple[str, str]] = set()
    for pair in pairs:
        if not isinstance(pair, dict) or set(pair) != {
            "checkpoint_implementation_digest",
            "current_implementation_digest",
            "reason",
        }:
            raise RuntimeError("Malformed checkpoint compatibility pair")
        checkpoint_digest = pair["checkpoint_implementation_digest"]
        current_digest = pair["current_implementation_digest"]
        if not all(
            isinstance(value, str) and len(value) == 64
            for value in (checkpoint_digest, current_digest)
        ):
            raise RuntimeError("Malformed implementation digest in compatibility pair")
        approved.add((checkpoint_digest, current_digest))
    return approved


def resolve_device() -> torch.device:
    requested = os.getenv("LATERALITY_DEVICE")
    if requested:
        return torch.device(requested)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def set_reproducible_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _emit_epoch_progress_without_rng_side_effects(
    callback: ProgressCallback,
    event: dict[str, Any],
    device: torch.device,
) -> None:
    """Call reporting code without allowing it to perturb later training views."""
    python_state = random.getstate()
    numpy_state = np.random.get_state()
    cpu_state = torch.random.get_rng_state()
    accelerator_state = None
    if device.type == "cuda":
        accelerator_state = torch.cuda.get_rng_state(device)
    elif device.type == "mps":
        accelerator_state = torch.mps.get_rng_state()
    try:
        callback(event)
    finally:
        random.setstate(python_state)
        np.random.set_state(numpy_state)
        torch.random.set_rng_state(cpu_state)
        if device.type == "cuda" and accelerator_state is not None:
            torch.cuda.set_rng_state(accelerator_state, device)
        elif device.type == "mps" and accelerator_state is not None:
            torch.mps.set_rng_state(accelerator_state)


def make_rng_streams(seed: int, fold: int) -> dict[str, np.random.Generator]:
    """Variant-independent streams keep the reflection comparison paired."""
    root = np.random.SeedSequence([int(seed), int(fold), 20260904])
    sampling_seed, mask_seed, reflection_seed = root.spawn(3)
    return {
        "sampling": np.random.default_rng(sampling_seed),
        "mask": np.random.default_rng(mask_seed),
        "reflection": np.random.default_rng(reflection_seed),
    }


def source_balanced_epoch_batches(
    table,
    train_sources: list[str],
    *,
    batch_size: int,
    updates_per_epoch: int,
    rng: np.random.Generator,
) -> Iterator[tuple[np.ndarray, list[str]]]:
    """Draw every source once, then source-uniform padding only if required."""
    source_to_rows = {
        source: table.index[table["video_id"].astype(str) == source].to_numpy(dtype=int)
        for source in train_sources
    }
    if any(len(rows) == 0 for rows in source_to_rows.values()):
        raise ValueError("A declared training source has no cohort rows")
    required = int(batch_size) * int(updates_per_epoch)
    if required < len(train_sources):
        raise ValueError("Update budget cannot cover every training source")
    ordered = list(rng.permutation(np.asarray(train_sources, dtype=object)))
    extra = required - len(ordered)
    if extra:
        ordered.extend(
            rng.choice(np.asarray(train_sources, dtype=object), size=extra, replace=True).tolist()
        )
    for start in range(0, required, batch_size):
        sources = list(map(str, ordered[start : start + batch_size]))
        indices = np.asarray(
            [rng.choice(source_to_rows[source]) for source in sources], dtype=int
        )
        yield indices, sources


def _atomic_torch_save(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    os.close(descriptor)
    try:
        torch.save(payload, temporary_name)
        os.replace(temporary_name, path)
    except Exception:
        Path(temporary_name).unlink(missing_ok=True)
        raise


def _expected_lineage(
    context: ExperimentContext,
    cohort: PreparedCohort,
    splits: dict[str, Any],
    fold: int,
    seed: int,
    variant: str,
    *,
    training_implementation_digest: str | None = None,
) -> dict[str, Any]:
    fold_payload = get_fold(splits, fold)
    return {
        "schema": "neurips_laterality_checkpoint/v2",
        "protocol_digest": context.protocol_digest,
        "context_digest": context.context_digest,
        "cohort_digest": cohort.cohort_digest,
        "split_digest": splits["split_digest"],
        "implementation_digest": (
            training_implementation_digest or implementation_digest()
        ),
        "runtime_versions": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "scikit_learn": sklearn.__version__,
            "torch": torch.__version__,
        },
        "profile": context.profile,
        "fold": int(fold),
        "seed": int(seed),
        "variant": str(variant),
        "model_config": model_config(context),
        "train_sources": sorted(map(str, fold_payload["train_sources"])),
        "forbidden_test_sources": sorted(map(str, fold_payload["test_sources"])),
        "labels_enter_objective": False,
        "checkpoint_selection": "fixed_predeclared_update_budget",
    }


def validate_checkpoint_lineage(
    checkpoint: dict[str, Any], expected: dict[str, Any]
) -> None:
    mismatches = [
        key for key, value in expected.items() if checkpoint.get(key) != value
    ]
    if mismatches:
        raise RuntimeError(f"Checkpoint lineage mismatch: {mismatches}")
    if checkpoint.get("lineage_digest") != canonical_json_digest(expected):
        raise RuntimeError("Checkpoint lineage digest mismatch")
    if set(checkpoint["train_sources"]) & set(checkpoint["forbidden_test_sources"]):
        raise RuntimeError("Checkpoint declares overlapping train and test sources")
    if checkpoint.get("objective_inputs") != ["coordinates", "validity"]:
        raise RuntimeError("Primary objective input contract is missing or label-contaminated")
    if checkpoint.get("sampling") != "source_uniform_then_sequence_uniform":
        raise RuntimeError("Checkpoint sampling contract mismatch")
    if checkpoint.get("rng_streams") != ["sampling", "mask", "reflection"]:
        raise RuntimeError("Checkpoint RNG-stream contract mismatch")
    if _state_digest(checkpoint.get("initial_target_state", {})) != checkpoint.get(
        "initial_state_digest"
    ):
        raise RuntimeError("Seed-matched initialization state digest mismatch")
    if _state_digest(checkpoint.get("model_state", {})) != checkpoint.get(
        "model_state_digest"
    ):
        raise RuntimeError("Learned model state digest mismatch")


def load_checkpoint(
    path: Path,
    context: ExperimentContext,
    cohort: PreparedCohort,
    splits: dict[str, Any],
    fold: int,
    seed: int,
    variant: str,
) -> dict[str, Any]:
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    expected = _expected_lineage(context, cohort, splits, fold, seed, variant)
    observed_implementation = checkpoint.get("implementation_digest")
    current_implementation = expected["implementation_digest"]
    if (
        observed_implementation != current_implementation
        and (observed_implementation, current_implementation)
        in _approved_implementation_compatibility()
    ):
        expected = {**expected, "implementation_digest": observed_implementation}
    validate_checkpoint_lineage(checkpoint, expected)
    return checkpoint


def _state_digest(state: dict[str, torch.Tensor]) -> str:
    digest = hashlib.sha256()
    for name, tensor in sorted(state.items()):
        digest.update(name.encode("utf-8"))
        digest.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def train_fold(
    context: ExperimentContext,
    cohort: PreparedCohort,
    splits: dict[str, Any],
    fold: int,
    seed: int,
    variant: str,
    *,
    reuse_valid: bool = True,
    progress_callback: ProgressCallback | None = None,
) -> dict[str, Any]:
    output_path = checkpoint_path(context.artifact_root, variant, fold, seed)
    if output_path.exists() and reuse_valid:
        return load_checkpoint(output_path, context, cohort, splits, fold, seed, variant)

    # Freeze this before any expensive work. If source files are edited while a job
    # is running, the checkpoint continues to identify the code that began the job.
    training_implementation_digest = implementation_digest()

    if variant not in context.protocol["training"]["variants"]:
        raise ValueError(f"Unknown variant {variant}")
    fold_payload = get_fold(splits, fold)
    train_sources = sorted(map(str, fold_payload["train_sources"]))
    test_sources = set(map(str, fold_payload["test_sources"]))
    if set(train_sources) & test_sources:
        raise AssertionError("Training sources overlap the outer test fold")
    train_rows = cohort.table.index[
        cohort.table["video_id"].astype(str).isin(train_sources)
    ].to_numpy(dtype=int)
    if set(cohort.table.loc[train_rows, "video_id"].astype(str)) != set(train_sources):
        raise AssertionError("Training row selection did not cover its declared sources")

    profile = context.profile_config
    training_config = context.protocol["training"]
    batch_size = int(profile["batch_size"])
    max_train_sources = max(len(item["train_sources"]) for item in splits["folds"])
    updates_per_epoch = int(math.ceil(max_train_sources / batch_size))
    epochs = int(profile["epochs"])
    total_updates = updates_per_epoch * epochs
    variant_config = context.protocol["training"]["variants"][variant]
    reflection_probability = float(variant_config["reflection_probability"])
    authorized = tuple(map(int, context.protocol["model"]["authorized_target_joints"]))
    segment_length = int(model_config(context)["segment_length"])
    mask_fraction = float(context.protocol["model"]["mask_fraction"])

    device = resolve_device()
    if progress_callback is not None:
        progress_callback(
            {
                "event": "job_started",
                "variant": str(variant),
                "fold": int(fold),
                "seed": int(seed),
                "epochs": epochs,
                "updates_per_epoch": updates_per_epoch,
                "total_optimizer_updates": total_updates,
                "train_sources": len(train_sources),
                "train_sequences": len(train_rows),
                "device": str(device),
            }
        )

    # Reset random state after the progress callback so display code cannot alter
    # initialization. The notebook callback itself performs no random operations.
    set_reproducible_seed(seed)
    model = SJEPAGait(**model_config(context)).to(device)
    projector = VICRegProjector(int(model_config(context)["embed_dim"])).to(device)
    initial_target_state = {
        key: tensor.detach().cpu().clone()
        for key, tensor in model.target_encoder.state_dict().items()
    }
    initial_state_digest = _state_digest(initial_target_state)
    trainable = [
        *model.view_encoder.parameters(),
        *model.predictor.parameters(),
        *projector.parameters(),
    ]
    if training_config["optimizer"] != "AdamW":
        raise ValueError("This protocol implementation supports only AdamW")
    optimizer = torch.optim.AdamW(
        trainable,
        lr=float(profile["learning_rate"]),
        betas=tuple(map(float, training_config["optimizer_betas"])),
        weight_decay=float(profile["weight_decay"]),
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(total_updates, 1), eta_min=0.0
    )
    rng_streams = make_rng_streams(seed, fold)
    sampling_rng = rng_streams["sampling"]
    mask_rng = rng_streams["mask"]
    reflection_rng = rng_streams["reflection"]
    history: list[dict[str, float | int]] = []
    draw_counts: Counter[str] = Counter()
    global_step = 0
    started = time.time()
    model.train()
    projector.train()
    for epoch in range(epochs):
        epoch_started = time.monotonic()
        epoch_losses: list[float] = []
        for row_indices, drawn_sources in source_balanced_epoch_batches(
            cohort.table,
            train_sources,
            batch_size=batch_size,
            updates_per_epoch=updates_per_epoch,
            rng=sampling_rng,
        ):
            if set(drawn_sources) & test_sources:
                raise AssertionError("Outer-test source reached an optimizer batch")
            draw_counts.update(drawn_sources)
            coordinates = torch.as_tensor(
                cohort.model_xyz[row_indices], dtype=torch.float32, device=device
            )
            valid = torch.as_tensor(
                cohort.model_valid[row_indices], dtype=torch.bool, device=device
            )
            if reflection_probability:
                reflected_rows = torch.as_tensor(
                    reflection_rng.random(len(coordinates)) < reflection_probability,
                    dtype=torch.bool,
                    device=device,
                )
                coordinates, valid = anatomical_reflect_tensor(
                    coordinates, valid, reflected_rows
                )
            patch_valid = valid_patches(valid, segment_length)
            target_mask = torch.as_tensor(
                uniform_authorized_mask(
                    patch_valid.detach().cpu().numpy(),
                    authorized,
                    mask_fraction,
                    mask_rng,
                ),
                dtype=torch.bool,
                device=device,
            )
            view_a = geometric_view(
                coordinates,
                valid,
                max_degrees=float(training_config["view_max_rotation_degrees"]),
                translate=float(training_config["view_max_translation"]),
            )
            view_b = geometric_view(
                coordinates,
                valid,
                max_degrees=float(training_config["view_max_rotation_degrees"]),
                translate=float(training_config["view_max_translation"]),
            )
            predicted, targets = model(
                view_a, coordinates, patch_valid, target_mask
            )
            jepa = sjepa_cross_entropy(
                predicted,
                targets,
                model.target_center,
                predictor_temperature=float(
                    training_config["predictor_temperature"]
                ),
                target_temperature=float(training_config["target_temperature"]),
            )
            tokens_a = model.view_encoder(view_a, patch_valid).reshape(
                len(view_a), model.view_encoder.segments, model.view_encoder.joints, -1
            )
            tokens_b = model.view_encoder(view_b, patch_valid).reshape(
                len(view_b), model.view_encoder.segments, model.view_encoder.joints, -1
            )
            pooled_a = authorized_pool(tokens_a, patch_valid, authorized)
            pooled_b = authorized_pool(tokens_b, patch_valid, authorized)
            regularizer = vicreg_loss(projector(pooled_a), projector(pooled_b))
            loss = jepa + float(profile["vicreg_weight"]) * regularizer
            if not torch.isfinite(loss):
                raise FloatingPointError(f"Non-finite training loss at update {global_step}")
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            if any(parameter.grad is not None for parameter in model.target_encoder.parameters()):
                raise AssertionError("Target encoder received gradients")
            torch.nn.utils.clip_grad_norm_(
                trainable, max_norm=float(training_config["gradient_clip_norm"])
            )
            optimizer.step()
            scheduler.step()
            model.update_target(
                cosine_ema(
                    global_step,
                    total_updates,
                    start=float(profile["ema_start"]),
                    end=float(training_config["ema_end"]),
                )
            )
            model.update_center(
                targets, beta=float(training_config["target_center_beta"])
            )
            epoch_losses.append(float(loss.detach().cpu()))
            global_step += 1
        epoch_summary: dict[str, float | int] = {
            "epoch": epoch + 1,
            "mean_total_loss": float(np.mean(epoch_losses)),
            "optimizer_updates": len(epoch_losses),
        }
        history.append(epoch_summary)
        if progress_callback is not None:
            _emit_epoch_progress_without_rng_side_effects(
                progress_callback,
                {
                    "event": "epoch_completed",
                    "variant": str(variant),
                    "fold": int(fold),
                    "seed": int(seed),
                    "epoch": epoch + 1,
                    "epochs": epochs,
                    "optimizer_updates": global_step,
                    "total_optimizer_updates": total_updates,
                    "mean_total_loss": epoch_summary["mean_total_loss"],
                    "epoch_seconds": float(time.monotonic() - epoch_started),
                    "job_elapsed_seconds": float(time.time() - started),
                    "device": str(device),
                },
                device,
            )

    lineage = _expected_lineage(
        context,
        cohort,
        splits,
        fold,
        seed,
        variant,
        training_implementation_digest=training_implementation_digest,
    )
    model_state = {
        key: value.detach().cpu() for key, value in model.state_dict().items()
    }
    checkpoint: dict[str, Any] = {
        **lineage,
        "lineage_digest": canonical_json_digest(lineage),
        "model_state": model_state,
        "model_state_digest": _state_digest(model_state),
        "projector_state": {
            key: value.detach().cpu() for key, value in projector.state_dict().items()
        },
        "initial_target_state": initial_target_state,
        "initial_state_digest": initial_state_digest,
        "objective_inputs": ["coordinates", "validity"],
        "sampling": "source_uniform_then_sequence_uniform",
        "rng_streams": ["sampling", "mask", "reflection"],
        "source_draw_counts": dict(sorted(draw_counts.items())),
        "reflection_probability": reflection_probability,
        "epochs": epochs,
        "updates_per_epoch": updates_per_epoch,
        "optimizer_updates": global_step,
        "history": history,
        "wall_seconds": float(time.time() - started),
        "device": str(device),
    }
    validate_checkpoint_lineage(checkpoint, lineage)
    if progress_callback is not None:
        progress_callback(
            {
                "event": "checkpoint_saving",
                "variant": str(variant),
                "fold": int(fold),
                "seed": int(seed),
                "epoch": epochs,
                "epochs": epochs,
                "job_elapsed_seconds": checkpoint["wall_seconds"],
                "device": str(device),
            }
        )
    _atomic_torch_save(output_path, checkpoint)
    return checkpoint


def train_selected(
    context: ExperimentContext,
    cohort: PreparedCohort,
    splits: dict[str, Any],
    *,
    progress_callback: ProgressCallback | None = None,
) -> list[dict[str, Any]]:
    """Train or reuse all selected jobs while emitting behavior-neutral progress events.

    The callback receives dictionaries whose ``event`` field is one of
    ``run_started``, ``job_started``, ``epoch_completed``, ``checkpoint_saving``,
    ``job_completed``, ``job_failed``, or ``run_completed``. A failing display
    callback is disabled after one warning so it cannot discard an expensive run.
    """

    jobs = [
        (str(variant), int(fold), int(seed))
        for variant in context.variants
        for fold in context.folds
        for seed in context.seeds
    ]
    callback_enabled = progress_callback is not None

    def emit(event: dict[str, Any]) -> None:
        nonlocal callback_enabled
        if not callback_enabled or progress_callback is None:
            return
        try:
            progress_callback(dict(event))
        except Exception as error:  # Progress must never waste a valid training run.
            callback_enabled = False
            warnings.warn(
                f"Training progress display failed and was disabled: {error}",
                RuntimeWarning,
                stacklevel=2,
            )

    profile = context.profile_config
    max_train_sources = max(len(item["train_sources"]) for item in splits["folds"])
    updates_per_epoch = int(math.ceil(max_train_sources / int(profile["batch_size"])))
    epochs = int(profile["epochs"])
    cached_candidates = sum(
        checkpoint_path(context.artifact_root, variant, fold, seed).exists()
        for variant, fold, seed in jobs
    )
    run_started = time.monotonic()
    emit(
        {
            "event": "run_started",
            "profile": context.profile,
            "total_jobs": len(jobs),
            "cached_candidate_jobs": int(cached_candidates),
            "new_candidate_jobs": int(len(jobs) - cached_candidates),
            "epochs_per_job": epochs,
            "updates_per_epoch": updates_per_epoch,
            "total_optimizer_updates_per_job": epochs * updates_per_epoch,
            "device": str(resolve_device()),
        }
    )

    summaries: list[dict[str, Any]] = []
    reused_jobs = 0
    trained_jobs = 0
    for job_index, (variant, fold, seed) in enumerate(jobs, start=1):
        output_path = checkpoint_path(context.artifact_root, variant, fold, seed)
        checkpoint_existed = output_path.exists()
        job_started = time.monotonic()

        def job_progress(event: dict[str, Any]) -> None:
            emit(
                {
                    **event,
                    "job_index": job_index,
                    "total_jobs": len(jobs),
                }
            )

        try:
            checkpoint = train_fold(
                context,
                cohort,
                splits,
                fold,
                seed,
                variant,
                progress_callback=job_progress,
            )
        except BaseException as error:
            emit(
                {
                    "event": "job_failed",
                    "job_index": job_index,
                    "total_jobs": len(jobs),
                    "variant": variant,
                    "fold": fold,
                    "seed": seed,
                    "error_type": type(error).__name__,
                    "error": str(error),
                    "run_elapsed_seconds": float(time.monotonic() - run_started),
                }
            )
            raise

        if checkpoint_existed:
            reused_jobs += 1
        else:
            trained_jobs += 1
        summaries.append(
            {
                "variant": variant,
                "fold": fold,
                "seed": seed,
                "checkpoint_reused": checkpoint_existed,
                "optimizer_updates": checkpoint["optimizer_updates"],
                "final_loss": checkpoint["history"][-1]["mean_total_loss"],
                "history": checkpoint["history"],
                "minimum_source_draws": min(
                    checkpoint["source_draw_counts"].values()
                ),
                "maximum_source_draws": max(
                    checkpoint["source_draw_counts"].values()
                ),
                "checkpoint": str(output_path),
            }
        )
        emit(
            {
                "event": "job_completed",
                "job_index": job_index,
                "total_jobs": len(jobs),
                "completed_jobs": len(summaries),
                "variant": variant,
                "fold": fold,
                "seed": seed,
                "checkpoint_reused": checkpoint_existed,
                "epochs": int(checkpoint["epochs"]),
                "optimizer_updates": int(checkpoint["optimizer_updates"]),
                "final_loss": float(checkpoint["history"][-1]["mean_total_loss"]),
                "historical_training_seconds": float(checkpoint["wall_seconds"]),
                "job_elapsed_seconds": float(time.monotonic() - job_started),
                "run_elapsed_seconds": float(time.monotonic() - run_started),
                "device": str(checkpoint["device"]),
                "reused_jobs": reused_jobs,
                "trained_jobs": trained_jobs,
            }
        )
    emit(
        {
            "event": "run_completed",
            "total_jobs": len(jobs),
            "completed_jobs": len(summaries),
            "reused_jobs": reused_jobs,
            "trained_jobs": trained_jobs,
            "run_elapsed_seconds": float(time.monotonic() - run_started),
        }
    )
    return summaries
