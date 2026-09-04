from __future__ import annotations

import hashlib
import math
import os
import platform
import random
import tempfile
import time
from collections import Counter
from pathlib import Path
from typing import Any, Iterator

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


def implementation_digest() -> str:
    digest = hashlib.sha256()
    module_root = Path(__file__).resolve().parent
    for path in sorted(module_root.glob("*.py")):
        digest.update(path.name.encode("utf-8"))
        digest.update(path.read_bytes())
    return digest.hexdigest()


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
) -> dict[str, Any]:
    fold_payload = get_fold(splits, fold)
    return {
        "schema": "neurips_laterality_checkpoint/v2",
        "protocol_digest": context.protocol_digest,
        "context_digest": context.context_digest,
        "cohort_digest": cohort.cohort_digest,
        "split_digest": splits["split_digest"],
        "implementation_digest": implementation_digest(),
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
    validate_checkpoint_lineage(
        checkpoint, _expected_lineage(context, cohort, splits, fold, seed, variant)
    )
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
) -> dict[str, Any]:
    output_path = checkpoint_path(context.artifact_root, variant, fold, seed)
    if output_path.exists() and reuse_valid:
        return load_checkpoint(output_path, context, cohort, splits, fold, seed, variant)

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

    set_reproducible_seed(seed)
    device = resolve_device()
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
        history.append(
            {
                "epoch": epoch + 1,
                "mean_total_loss": float(np.mean(epoch_losses)),
                "optimizer_updates": len(epoch_losses),
            }
        )

    lineage = _expected_lineage(context, cohort, splits, fold, seed, variant)
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
    _atomic_torch_save(output_path, checkpoint)
    return checkpoint


def train_selected(context: ExperimentContext, cohort: PreparedCohort, splits: dict[str, Any]) -> list[dict[str, Any]]:
    summaries: list[dict[str, Any]] = []
    for variant in context.variants:
        for fold in context.folds:
            for seed in context.seeds:
                checkpoint = train_fold(context, cohort, splits, fold, seed, variant)
                summaries.append(
                    {
                        "variant": variant,
                        "fold": fold,
                        "seed": seed,
                        "optimizer_updates": checkpoint["optimizer_updates"],
                        "final_loss": checkpoint["history"][-1]["mean_total_loss"],
                        "history": checkpoint["history"],
                        "minimum_source_draws": min(
                            checkpoint["source_draw_counts"].values()
                        ),
                        "maximum_source_draws": max(
                            checkpoint["source_draw_counts"].values()
                        ),
                        "checkpoint": str(
                            checkpoint_path(context.artifact_root, variant, fold, seed)
                        ),
                    }
                )
    return summaries
