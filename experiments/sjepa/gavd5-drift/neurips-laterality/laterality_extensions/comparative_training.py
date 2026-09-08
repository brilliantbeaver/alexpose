"""Paired masking experiments with explicit budgets and a dense target path.

The original model and completed experiments are imported without modification.
Default settings are synthetic. A real run requires explicit confirmation.
"""
from __future__ import annotations

import copy
from dataclasses import asdict, dataclass, replace
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import shutil
import tempfile
import time
from typing import Any, Mapping
import warnings

import numpy as np
import pandas as pd
import torch
from torch.nn import functional as F

from laterality.config import SUITE_ROOT, canonical_json_digest
from laterality.geometry import FULL_MIRROR_PAIRS
from laterality.model import (
    SJEPAGait,
    VICRegProjector,
    authorized_pool,
    geometric_view,
    sjepa_cross_entropy,
    vicreg_loss,
)
from laterality.training import source_balanced_epoch_batches
from laterality_extensions.masked_learning import (
    GAIT_JOINTS, LearningDataset, LearningSettings, configure_learning_runtime,
    load_learning_dataset, resolve_learning_device,
)
from laterality_extensions.comparative_masks import MaskBudget, MaskPolicy, coverage_summary, eligible_landmarks, sample_mask

SCHEMA = "comparative_masking/v1"
STRUCTURED = {"whole_trajectory", "connected_region", "temporal_gap"}


def dense_prediction(model, view, target, valid_patch, target_mask):
    """Keep [clip, token, feature] identities until the loss selects targets."""
    if target_mask.dtype != torch.bool or target_mask.shape != valid_patch.shape:
        raise ValueError("Target mask must match boolean token validity")
    flat_mask = target_mask.flatten(1)
    flat_valid = valid_patch.flatten(1)
    if (target_mask & ~valid_patch).any() or (flat_mask.sum(1) < 1).any():
        raise ValueError("Each clip needs observed prediction targets")
    if ((flat_valid & ~flat_mask).sum(1) < 1).any():
        raise ValueError("Each clip needs visible context")
    context = model.view_encoder(view, valid_patch, hide_mask=target_mask)
    p = model.predictor
    tokens = p.encoder_to_predictor(context)
    tokens = torch.where(flat_mask[..., None], p.mask_token.expand_as(tokens), tokens)
    position = (p.time_pos[:, None] + p.joint_pos[None]).reshape(1, -1, tokens.shape[-1])
    predicted = p.output(p.norm(p.blocks(tokens + position, src_key_padding_mask=~flat_valid)))
    with torch.no_grad():
        targets = model.target_encoder(target, valid_patch)
    return predicted, targets


def per_clip_prediction_loss(predicted, targets, mask, center):
    """Average hidden targets within each clip, giving clips equal loss weight."""
    hidden = mask.flatten(1)
    if predicted.shape != targets.shape or predicted.shape[:2] != hidden.shape:
        raise ValueError("Dense features and mask dimensions disagree")
    probability = F.softmax((targets - center) / 0.06, dim=-1).detach()
    token_loss = -(probability * F.log_softmax(predicted / 0.10, dim=-1)).sum(-1)
    counts = hidden.sum(1)
    if (counts < 1).any():
        raise ValueError("Cannot average an empty target set")
    return ((token_loss * hidden).sum(1) / counts).mean()


def _encode_prevalidated(encoder, coordinates, valid_patch, *, hide_mask=None):
    """Run a previously validated encoder batch without device-to-host branches."""
    tokens = encoder.patch_embed(encoder.patchify(coordinates))
    if hide_mask is not None:
        tokens = tokens.masked_fill(hide_mask[..., None], 0.0)
    tokens = (
        tokens
        + encoder.time_pos[None, :, None, :]
        + encoder.joint_pos[None, None, :, :]
    )
    batch = len(tokens)
    padding = ~valid_patch.reshape(batch, -1)
    encoded = encoder.norm(
        encoder.blocks(
            tokens.reshape(batch, encoder.segments * encoder.joints, encoder.embed_dim),
            src_key_padding_mask=padding,
        )
    )
    return encoded.masked_fill(padding[..., None], 0.0)


def _prediction_prevalidated(
    model,
    view,
    target,
    valid_patch,
    target_mask,
    *,
    equal_target_counts,
):
    """Use the packed target path whenever every clip has the same target count."""
    context = _encode_prevalidated(
        model.view_encoder, view, valid_patch, hide_mask=target_mask
    )
    flat_mask = target_mask.reshape(len(target), -1)
    counts = flat_mask.sum(1)
    if equal_target_counts:
        predicted = model.predictor(context, target_mask, valid_patch)
        with torch.no_grad():
            target_tokens = _encode_prevalidated(
                model.target_encoder, target, valid_patch
            )
            selected = target_tokens[flat_mask].reshape(
                len(target), -1, target_tokens.shape[-1]
            )
        loss = sjepa_cross_entropy(
            predicted,
            selected,
            model.target_center,
            predictor_temperature=0.10,
            target_temperature=0.06,
        )
        return loss, selected, counts

    predictor = model.predictor
    tokens = predictor.encoder_to_predictor(context)
    tokens = torch.where(
        flat_mask[..., None], predictor.mask_token.expand_as(tokens), tokens
    )
    position = (
        predictor.time_pos[:, None] + predictor.joint_pos[None]
    ).reshape(1, -1, tokens.shape[-1])
    predicted = predictor.output(
        predictor.norm(
            predictor.blocks(
                tokens + position,
                src_key_padding_mask=~valid_patch.reshape(len(target), -1),
            )
        )
    )
    with torch.no_grad():
        selected = _encode_prevalidated(
            model.target_encoder, target, valid_patch
        )
    return (
        per_clip_prediction_loss(
            predicted, selected, target_mask, model.target_center
        ),
        selected,
        counts,
    )


def _synchronize_device(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elif device.type == "mps":
        torch.mps.synchronize()


def _cpu_tree(value):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu()
    if isinstance(value, dict):
        return {key: _cpu_tree(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_cpu_tree(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_cpu_tree(item) for item in value)
    return value


def _diagnostic_values(arm, *, label):
    """Copy the small loss trace once and reject a poisoned training chunk."""
    if not arm["diagnostics"]:
        return np.empty((0, 3), dtype=np.float64)
    values = torch.stack(arm["diagnostics"]).detach().cpu().numpy()
    if not np.isfinite(values).all():
        raise FloatingPointError(f"Non-finite training diagnostics for {label}")
    return values


def _resume_path(destination):
    destination = Path(destination)
    return destination.parent / f".{destination.name}.resume.pt"


def _resume_checksum_path(path):
    path = Path(path)
    return path.with_name(f"{path.name}.sha256")


def _file_digest(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _validate_tensor_state(saved, reference, *, label):
    if not isinstance(saved, dict) or set(saved) != set(reference):
        raise RuntimeError(f"Training resume {label} has different state entries")
    for name, expected in reference.items():
        value = saved[name]
        if (
            not isinstance(value, torch.Tensor)
            or value.shape != expected.shape
            or value.dtype != expected.dtype
        ):
            raise RuntimeError(
                f"Training resume {label} tensor {name!r} has an invalid shape or dtype"
            )
        if (value.is_floating_point() or value.is_complex()) and not torch.isfinite(value).all():
            raise RuntimeError(
                f"Training resume {label} tensor {name!r} is non-finite"
            )


def _write_training_resume(path, identity, completed_step, arms, elapsed):
    """Atomically retain both independent arms at one shared update boundary."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema": "comparative_masking_resume/v1",
        "identity_digest": canonical_json_digest(identity),
        "completed_step": int(completed_step),
        "conditions": list(arms),
        "elapsed_training_seconds": float(elapsed),
        "arms": {},
    }
    for name, arm in arms.items():
        new_values = _diagnostic_values(arm, label=name)
        history = np.concatenate(
            (arm["history_prefix"], new_values), axis=0
        )
        payload["arms"][name] = {
            "model": _cpu_tree(arm["model"].state_dict()),
            "projector": _cpu_tree(arm["projector"].state_dict()),
            "optimizer": _cpu_tree(arm["optimizer"].state_dict()),
            # Keep the resume payload inside PyTorch's ``weights_only`` safe
            # type set.  A NumPy array would require broadening deserialization.
            "history": torch.as_tensor(history),
            "snapshots": _cpu_tree(arm["snapshots"]),
            "initial_state_digest": arm["initial_digest"],
            "initial_projector_digest": arm["initial_projector_digest"],
        }
    checksum_path = _resume_checksum_path(path)
    temporary = path.with_name(f"{path.name}.{os.getpid()}.tmp")
    temporary_checksum = checksum_path.with_name(
        f"{checksum_path.name}.{os.getpid()}.tmp"
    )
    try:
        torch.save(payload, temporary)
        temporary_checksum.write_text(f"{_file_digest(temporary)}\n")
        os.replace(temporary, path)
        os.replace(temporary_checksum, checksum_path)
    finally:
        temporary.unlink(missing_ok=True)
        temporary_checksum.unlink(missing_ok=True)


def _load_training_resume(path, identity, arms, device):
    """Restore an exact atomic checkpoint or fail closed on any mismatch."""
    path = Path(path)
    checksum_path = _resume_checksum_path(path)
    if not path.exists():
        if checksum_path.exists():
            raise RuntimeError(
                f"Training resume checkpoint is incomplete: {path}"
            )
        return 0, 0.0
    if not checksum_path.is_file():
        raise RuntimeError(f"Training resume checkpoint has no checksum: {path}")
    expected_file_digest = checksum_path.read_text().strip()
    if (
        len(expected_file_digest) != 64
        or any(character not in "0123456789abcdef" for character in expected_file_digest)
        or _file_digest(path) != expected_file_digest
    ):
        raise RuntimeError(f"Training resume checkpoint failed its checksum: {path}")
    try:
        payload = torch.load(path, map_location="cpu", weights_only=True)
    except Exception as error:
        raise RuntimeError(f"Training resume checkpoint is unreadable: {path}") from error
    expected = canonical_json_digest(identity)
    if (
        payload.get("schema") != "comparative_masking_resume/v1"
        or payload.get("identity_digest") != expected
        or payload.get("conditions") != list(arms)
        or set(payload.get("arms", {})) != set(arms)
    ):
        raise RuntimeError(f"Training resume checkpoint lineage mismatch: {path}")
    completed = payload.get("completed_step")
    if (
        not isinstance(completed, int)
        or completed < 1
        or completed > identity["settings"]["steps"]
    ):
        raise RuntimeError(f"Training resume checkpoint has an invalid step: {path}")
    for name, arm in arms.items():
        saved = payload["arms"][name]
        if (
            saved.get("initial_state_digest") != arm["initial_digest"]
            or saved.get("initial_projector_digest")
            != arm["initial_projector_digest"]
        ):
            raise RuntimeError(
                f"Training resume initialization mismatch for {name}: {path}"
            )
        history = np.asarray(saved.get("history"))
        if history.shape != (completed, 3) or not np.isfinite(history).all():
            raise RuntimeError(
                f"Training resume history is invalid for {name}: {path}"
            )
        _validate_tensor_state(
            saved.get("model"), arm["model"].state_dict(), label=f"model for {name}"
        )
        _validate_tensor_state(
            saved.get("projector"),
            arm["projector"].state_dict(),
            label=f"projector for {name}",
        )
        snapshots = saved.get("snapshots")
        expected_snapshots = {
            step for step in identity["checkpoint_steps"] if step <= completed
        }
        if not isinstance(snapshots, dict) or set(snapshots) != expected_snapshots:
            raise RuntimeError(
                f"Training resume snapshots are incomplete for {name}: {path}"
            )
        for step, snapshot in snapshots.items():
            _validate_tensor_state(
                snapshot,
                arm["model"].state_dict(),
                label=f"update-{step} model for {name}",
            )

        saved_optimizer = saved.get("optimizer")
        reference_optimizer = arm["optimizer"].state_dict()
        if (
            not isinstance(saved_optimizer, dict)
            or set(saved_optimizer) != {"state", "param_groups"}
            or len(saved_optimizer["param_groups"])
            != len(reference_optimizer["param_groups"])
        ):
            raise RuntimeError(
                f"Training resume optimizer structure is invalid for {name}: {path}"
            )
        for saved_group, reference_group in zip(
            saved_optimizer["param_groups"], reference_optimizer["param_groups"]
        ):
            if (
                len(saved_group.get("params", ()))
                != len(reference_group.get("params", ()))
                or {key: value for key, value in saved_group.items() if key != "params"}
                != {key: value for key, value in reference_group.items() if key != "params"}
            ):
                raise RuntimeError(
                    f"Training resume optimizer settings differ for {name}: {path}"
                )

        parameter_ids = [
            parameter_id
            for group in saved_optimizer["param_groups"]
            for parameter_id in group["params"]
        ]
        if (
            len(parameter_ids) != len(arm["trainable"])
            or len(set(parameter_ids)) != len(parameter_ids)
            or set(saved_optimizer["state"]) != set(parameter_ids)
        ):
            raise RuntimeError(
                f"Training resume optimizer state is incomplete for {name}: {path}"
            )
        for parameter_id, parameter in zip(parameter_ids, arm["trainable"]):
            state = saved_optimizer["state"][parameter_id]
            if not isinstance(state, dict) or set(state) != {
                "step", "exp_avg", "exp_avg_sq"
            }:
                raise RuntimeError(
                    f"Training resume Adam state is invalid for {name}: {path}"
                )
            step_value = state["step"]
            if (
                not isinstance(step_value, torch.Tensor)
                or step_value.numel() != 1
                or not torch.isfinite(step_value).all()
                or float(step_value.item()) != completed
            ):
                raise RuntimeError(
                    f"Training resume optimizer step differs for {name}: {path}"
                )
            for moment_name in ("exp_avg", "exp_avg_sq"):
                moment = state[moment_name]
                if (
                    not isinstance(moment, torch.Tensor)
                    or moment.shape != parameter.shape
                    or moment.dtype != parameter.dtype
                    or not torch.isfinite(moment).all()
                ):
                    raise RuntimeError(
                        f"Training resume {moment_name} is invalid for {name}: {path}"
                    )

        arm["model"].load_state_dict(saved["model"], strict=True)
        arm["projector"].load_state_dict(saved["projector"], strict=True)
        try:
            arm["optimizer"].load_state_dict(saved_optimizer)
        except (KeyError, TypeError, ValueError) as error:
            raise RuntimeError(
                f"Training resume optimizer state is invalid for {name}: {path}"
            ) from error
        for state in arm["optimizer"].state.values():
            for key, value in state.items():
                if isinstance(value, torch.Tensor):
                    if (
                        (value.is_floating_point() or value.is_complex())
                        and not torch.isfinite(value).all()
                    ):
                        raise RuntimeError(
                            f"Training resume optimizer state is non-finite for {name}: {path}"
                        )
                    state[key] = value.to(device)
        arm["history_prefix"] = history
        arm["snapshots"] = saved["snapshots"]
    elapsed = float(payload.get("elapsed_training_seconds", 0.0))
    if not math.isfinite(elapsed) or elapsed < 0:
        raise RuntimeError(f"Training resume timing is invalid: {path}")
    return completed, elapsed


def _digest_array(values):
    x = np.ascontiguousarray(values)
    return hashlib.sha256(str(x.dtype).encode() + str(x.shape).encode() + x.tobytes()).hexdigest()


def state_digest(model):
    h = hashlib.sha256()
    for name, value in sorted(model.state_dict().items()):
        h.update(name.encode()); h.update(value.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def training_implementation_digest():
    """Hash code that can change fitted weights, without evaluation-only code."""
    extension_root = Path(__file__).resolve().parent
    laterality_root = extension_root.parent / "laterality"
    paths = [
        extension_root / "comparative_masks.py",
        extension_root / "masked_learning.py",
        *[
            laterality_root / name
            for name in (
                "config.py", "data.py", "geometry.py", "model.py",
                "splitting.py", "training.py",
            )
        ],
    ]
    digest = hashlib.sha256()
    for path in paths:
        if not path.is_file():
            raise FileNotFoundError(path)
        digest.update(path.name.encode())
        digest.update(path.read_bytes())
    source = Path(__file__).read_bytes()
    boundary = b"\ndef saved_reference_recipe("
    if boundary not in source:
        raise RuntimeError("Comparative training digest boundary is missing")
    digest.update(b"comparative_training_fit_section")
    digest.update(source.split(boundary, 1)[0])
    return digest.hexdigest()


def implementation_digest():
    """Compatibility name for the fitted-model implementation identity."""
    return training_implementation_digest()


def _new_model(settings, frames, device):
    # Model construction happens on the CPU. Seed only the CPU generator so a
    # notebook's MPS/CUDA random stream is neither initialized nor changed.
    cpu_rng_state = torch.random.get_rng_state()
    try:
        torch.random.default_generator.manual_seed(settings.seed)
        # PyTorch announces that its optional nested-tensor fast path is
        # unavailable with this architecture's pre-norm transformer. The
        # ordinary transformer path is intentional and numerically unchanged.
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message=(
                    "enable_nested_tensor is True, but self.use_nested_tensor "
                    "is False because encoder_layer.norm_first was True"
                ),
                category=UserWarning,
            )
            model = SJEPAGait(frames=frames, joints=33, coordinate_dim=3,
                segment_length=settings.segment_length, embed_dim=settings.embed_dim,
                encoder_depth=settings.encoder_depth, predictor_depth=settings.predictor_depth,
                heads=settings.heads)
        projector = VICRegProjector(settings.embed_dim)
    finally:
        torch.random.set_rng_state(cpu_rng_state)
    return model.to(device), projector.to(device)


def source_schedule(dataset, settings):
    return declared_source_schedule(dataset.source_ids, dataset.train_sources, settings)


def declared_source_schedule(source_ids, train_sources, settings):
    """Recreate the deterministic training-row schedule from its declaration."""
    table = pd.DataFrame({"video_id": np.asarray(source_ids).astype(str)})
    rng = np.random.default_rng(np.random.SeedSequence([settings.seed, settings.fold, 101]))
    schedule = []
    while len(schedule) < settings.steps:
        for rows, _ in source_balanced_epoch_batches(table, list(train_sources),
                batch_size=settings.batch_size,
                updates_per_epoch=math.ceil(len(train_sources) / settings.batch_size), rng=rng):
            schedule.append(rows)
            if len(schedule) == settings.steps:
                break
    return np.stack(schedule)


def _policy_rng(settings, name, step, row):
    code = int.from_bytes(hashlib.sha256(name.encode()).digest()[:4], "little")
    return np.random.default_rng(np.random.SeedSequence([settings.seed, settings.fold, 102, code, step, row]))


def masks_for_batch(dataset, rows, settings, conditions, *, budgets=None, matched_to=None, step=0):
    """Construct paired masks before training; infeasibility never drops a clip."""
    budgets = dict(budgets or {})
    for name, budget in budgets.items():
        if name not in conditions:
            raise ValueError("A budget names an unknown condition")
        budget.validate(conditions[name])
    valid = dataset.valid[rows].reshape(len(rows), -1, settings.segment_length, 33).all(2)
    masks = {name: [] for name in conditions}
    coverage = {name: [] for name in conditions}
    if matched_to is None:
        if any(p.name in STRUCTURED for p in conditions.values()):
            raise ValueError("A structured comparison must name its matched_to condition")
        available = [int(valid[..., list(eligible_landmarks(p))].sum((1, 2)).min())
                     for p in conditions.values()]
        count = min(max(1, int(valid[..., GAIT_JOINTS].sum((1, 2)).min() * settings.mask_fraction)),
                    *available, int(valid.sum((1, 2)).min()) - 1)
        explicit = {b.hidden_count for b in budgets.values() if b.hidden_count is not None}
        if len(explicit) > 1:
            raise ValueError("Scattered comparisons must request one shared hidden count")
        if explicit:
            count = next(iter(explicit))
        if count < 1:
            raise ValueError("No shared positive target count is feasible")
    elif matched_to not in conditions or matched_to not in budgets:
        raise ValueError("matched_to requires a named condition and its declared budget")
    for batch_row, dataset_row in enumerate(rows):
        counts = count if matched_to is None else None
        order = list(conditions) if matched_to is None else [matched_to, *[n for n in conditions if n != matched_to]]
        for name in order:
            if name == matched_to:
                budget = budgets[name]
            else:
                if conditions[name].name in STRUCTURED:
                    raise ValueError("Compare one intact structure with scattered references per experiment")
                if name in budgets and budgets[name].hidden_count != counts:
                    raise ValueError("An explicit reference budget differs from the paired realized count")
                budget = MaskBudget(hidden_count=int(counts))
            result = sample_mask(dataset.xyz[dataset_row], valid[batch_row], conditions[name], budget,
                _policy_rng(settings, name, step, batch_row), segment_length=settings.segment_length,
                observation_valid=dataset.valid[dataset_row])
            if name == matched_to:
                counts = int(result.mask.sum())
            masks[name].append(result.mask)
            coverage[name].append(result.coverage)
    stacked = {name: np.stack(values) for name, values in masks.items()}
    counts = [m.sum((1, 2)) for m in stacked.values()]
    if any(not np.array_equal(c, counts[0]) for c in counts[1:]):
        raise AssertionError("Paired policies hid different counts in a clip")
    return stacked, coverage


@dataclass(frozen=True)
class _PreparedComparisonInputs:
    """Immutable fold/seed inputs shared by every independently fitted arm."""

    schedule: np.ndarray
    reflections: np.ndarray
    xyz: torch.Tensor
    valid: torch.Tensor
    valid_patch: torch.Tensor
    effective_rows: torch.Tensor
    target_masks: dict[str, torch.Tensor]
    target_counts: dict[str, list[list[int]]]
    equal_counts: dict[str, list[bool]]
    coverage: dict[str, list[list[dict[str, Any]]]]
    view_digest: str
    resident_bytes: int


def _prepare_comparison_inputs(
    dataset,
    settings,
    conditions,
    *,
    budgets,
    matched_to,
    device,
):
    """Precompute random choices on CPU and keep reusable tensors on the device."""
    schedule = source_schedule(dataset, settings)
    if set(dataset.source_ids[schedule.ravel()].astype(str)) & set(
        dataset.test_sources
    ):
        raise AssertionError("Test videos reached training")

    masks_by_name = {name: [] for name in conditions}
    original_coverage = {name: [] for name in conditions}
    for step, rows in enumerate(schedule):
        masks, coverage = masks_for_batch(
            dataset,
            rows,
            settings,
            conditions,
            budgets=budgets,
            matched_to=matched_to,
            step=step,
        )
        for name in conditions:
            masks_by_name[name].append(masks[name])
            original_coverage[name].append(coverage[name])

    reflection_rng = np.random.default_rng(
        np.random.SeedSequence([settings.seed, settings.fold, 103])
    )
    reflections = (
        reflection_rng.random(schedule.shape) < settings.reflection_probability
    )
    permutation = np.arange(33)
    for left, right in FULL_MIRROR_PAIRS:
        permutation[[left, right]] = permutation[[right, left]]

    clean_xyz = np.where(dataset.valid[..., None], dataset.xyz, 0)
    mirrored_xyz = clean_xyz[..., permutation, :].copy()
    mirrored_xyz[..., 0] *= -1
    mirrored_valid = dataset.valid[..., permutation].copy()
    combined_xyz = np.concatenate((clean_xyz, mirrored_xyz), axis=0)
    combined_valid = np.concatenate((dataset.valid, mirrored_valid), axis=0)
    combined_patch = combined_valid.reshape(
        len(combined_valid), -1, settings.segment_length, 33
    ).all(axis=2)
    effective_rows = schedule + reflections.astype(np.int64) * len(dataset.xyz)

    transformed_masks = {}
    target_counts = {}
    equal_counts = {}
    recorded_coverage = {name: [] for name in conditions}
    for name in conditions:
        stacked = np.stack(masks_by_name[name]).copy()
        for step in range(settings.steps):
            reflected_rows = reflections[step]
            if reflected_rows.any():
                stacked[step, reflected_rows] = stacked[
                    step, reflected_rows
                ][..., permutation]
        transformed_masks[name] = stacked
        counts = stacked.reshape(
            settings.steps, settings.batch_size, -1
        ).sum(axis=2)
        target_counts[name] = counts.astype(int).tolist()
        equal_counts[name] = [
            bool(np.all(step_counts == step_counts[0]))
            for step_counts in counts
        ]

        for step in range(settings.steps):
            observed = combined_patch[effective_rows[step]]
            step_coverage = []
            for batch_row, (hidden, valid_patch) in enumerate(
                zip(stacked[step], observed)
            ):
                details = {
                    **original_coverage[name][step][batch_row],
                    **coverage_summary(
                        hidden,
                        valid_patch,
                        segment_length=settings.segment_length,
                    ),
                    "reflected": bool(reflections[step, batch_row]),
                    "coordinate_frame": (
                        "training view after anatomical reflection"
                    ),
                }
                if reflections[step, batch_row]:
                    for key in ("selected_landmarks", "eligible_landmarks"):
                        if key in details:
                            details[key] = sorted(
                                map(int, permutation[details[key]])
                            )
                step_coverage.append(details)
            recorded_coverage[name].append(step_coverage)

    device_xyz = torch.as_tensor(
        combined_xyz, dtype=torch.float32, device=device
    ).contiguous()
    device_valid = torch.as_tensor(
        combined_valid, dtype=torch.bool, device=device
    ).contiguous()
    device_patch = torch.as_tensor(
        combined_patch, dtype=torch.bool, device=device
    ).contiguous()
    device_rows = torch.as_tensor(
        effective_rows, dtype=torch.long, device=device
    )
    device_masks = {
        name: torch.as_tensor(
            values, dtype=torch.bool, device=device
        ).contiguous()
        for name, values in transformed_masks.items()
    }
    resident = [
        device_xyz,
        device_valid,
        device_patch,
        device_rows,
        *device_masks.values(),
    ]
    view_digest = canonical_json_digest({
        "algorithm": "shared_geometric_view/v1",
        "source_schedule": _digest_array(schedule),
        "reflection_schedule": _digest_array(reflections),
        "seeds": [
            settings.seed + 100_003 * (settings.fold + 1) + step
            for step in range(settings.steps)
        ],
        "max_rotation_degrees": 8.0,
        "max_translation": 0.03,
    })
    return _PreparedComparisonInputs(
        schedule=schedule,
        reflections=reflections,
        xyz=device_xyz,
        valid=device_valid,
        valid_patch=device_patch,
        effective_rows=device_rows,
        target_masks=device_masks,
        target_counts=target_counts,
        equal_counts=equal_counts,
        coverage=recorded_coverage,
        view_digest=view_digest,
        resident_bytes=sum(
            tensor.numel() * tensor.element_size() for tensor in resident
        ),
    )


def default_conditions():
    return {"gait_targets": MaskPolicy("gait"), "all_landmark_targets": MaskPolicy("uniform")}


def anatomical_conditions():
    return {**default_conditions(), **{f"random_set_{seed}": MaskPolicy("random_subset", subset_seed=seed)
        for seed in (31, 47, 59)}, "soft_gait_preference": MaskPolicy("soft_gait")}


def comparison_identity(dataset, settings, conditions, budgets, matched_to, checkpoint_steps):
    device = resolve_learning_device(settings.device)
    runtime = {
        "python": platform.python_version(),
        "torch": str(torch.__version__),
        "numpy": np.__version__,
        "device": str(device),
        "device_type": device.type,
        "platform": platform.platform(),
        "machine": platform.machine(),
    }
    if device.type == "cpu":
        runtime["torch_threads"] = torch.get_num_threads()
    elif device.type == "cuda":
        index = device.index if device.index is not None else torch.cuda.current_device()
        runtime.update({
            "device_index": int(index),
            "device_name": torch.cuda.get_device_name(index),
            "compute_capability": list(torch.cuda.get_device_capability(index)),
            "cuda": torch.version.cuda,
            "cudnn": torch.backends.cudnn.version(),
        })
    elif device.type == "mps":
        runtime["macos"] = platform.mac_ver()[0]
    return {"schema": SCHEMA, "settings": asdict(settings), "frames": int(dataset.xyz.shape[1]),
        "conditions": {n: asdict(p) for n, p in conditions.items()},
        "budgets": {n: asdict(b) for n, b in (budgets or {}).items()}, "matched_to": matched_to,
        "checkpoint_steps": list(checkpoint_steps), "synthetic": dataset.synthetic,
        "train_sources": list(dataset.train_sources), "test_sources": list(dataset.test_sources),
        "sequence_ids": dataset.sequence_ids.astype(str).tolist(),
        "source_ids": dataset.source_ids.astype(str).tolist(),
        "data": {n: _digest_array(getattr(dataset, n)) for n in ("xyz", "valid", "targets")},
        "cohort_digest": dataset.cohort_digest, "split_digest": dataset.split_digest,
        "implementation": implementation_digest(),
        "runtime": runtime}


def train_comparison(dataset: LearningDataset, settings: LearningSettings,
        conditions: Mapping[str, MaskPolicy] | None = None, *, budgets=None, matched_to=None,
        output_dir=None, checkpoint_steps=(), reuse=True, progress_callback=None,
        progress_interval=1, resume_interval=0):
    """Fit one paired comparison, optionally retaining completed compatible runs.

    Inputs, random choices, masks, and coverage are prevalidated once. The
    independently optimized arms then share resident batches and the exact same
    augmented views. The progress interval changes reporting cadence only.
    """
    settings.validate(); dataset.validate(settings.segment_length)
    if settings.fold != dataset.fold:
        raise ValueError("Dataset fold and settings fold differ")
    if settings.symmetry_weight:
        raise ValueError("Symmetry loss is a separate comparison")
    if not dataset.synthetic and not settings.confirm_real_run:
        raise PermissionError("Real training requires confirm_real_run=True")
    conditions = dict(conditions or default_conditions())
    if len(conditions) < 2 or any(not name or not name.replace("_", "").isalnum() for name in conditions):
        raise ValueError("Supply at least two conditions with simple distinct names")
    if any(n not in conditions for n in (budgets or {})):
        raise ValueError("A budget names an unknown condition")
    if not isinstance(progress_interval, int) or progress_interval < 1:
        raise ValueError("progress_interval must be a positive integer")
    if not isinstance(resume_interval, int) or resume_interval < 0:
        raise ValueError("resume_interval must be a nonnegative integer")
    checkpoint_steps = tuple(sorted(set([*checkpoint_steps, settings.steps])))
    if any(not isinstance(s, int) or s < 1 or s > settings.steps for s in checkpoint_steps):
        raise ValueError("Checkpoint steps must be within the declared training exposure")
    device = resolve_learning_device(settings.device)
    settings = replace(settings, device=str(device))
    identity = comparison_identity(dataset, settings, conditions, budgets, matched_to, checkpoint_steps)
    key = canonical_json_digest(identity)
    destination = Path(output_dir) / key if output_dir is not None else None
    if destination is not None and destination.exists():
        if not reuse:
            raise FileExistsError(destination)
        reused = load_comparison(destination, identity)
        stale_resume = _resume_path(destination)
        for path in (stale_resume, _resume_checksum_path(stale_resume)):
            path.unlink(missing_ok=True)
        return reused
    prepared = _prepare_comparison_inputs(
        dataset,
        settings,
        conditions,
        budgets=budgets,
        matched_to=matched_to,
        device=device,
    )

    template_model, template_projector = _new_model(
        settings, dataset.xyz.shape[1], device
    )
    arms = {}
    for arm_index, (name, policy) in enumerate(conditions.items()):
        model = template_model if arm_index == 0 else copy.deepcopy(template_model)
        projector = (
            template_projector
            if arm_index == 0
            else copy.deepcopy(template_projector)
        )
        initial_model = copy.deepcopy(model).eval().to("cpu")
        initial_digest = state_digest(model)
        initial_projector_digest = state_digest(projector)
        trainable = [
            *model.view_encoder.parameters(),
            *model.predictor.parameters(),
            *projector.parameters(),
        ]
        optimizer_options = {"fused": True} if device.type == "cuda" else {}
        optimizer = torch.optim.AdamW(
            trainable,
            lr=settings.learning_rate,
            weight_decay=settings.weight_decay,
            betas=(0.9, 0.95),
            **optimizer_options,
        )
        model.train()
        projector.train()
        arms[name] = {
            "index": arm_index,
            "policy": policy,
            "model": model,
            "projector": projector,
            "initial_model": initial_model,
            "initial_digest": initial_digest,
            "initial_projector_digest": initial_projector_digest,
            "trainable": trainable,
            "optimizer": optimizer,
            "diagnostics": [],
            "history_prefix": np.empty((0, 3), dtype=np.float64),
            "snapshots": {},
        }

    resume_path = (
        _resume_path(destination)
        if destination is not None and resume_interval
        else None
    )
    resumed_from_step, earlier_elapsed = (
        _load_training_resume(resume_path, identity, arms, device)
        if resume_path is not None
        else (0, 0.0)
    )
    if device.type == "cuda":
        random_device_index = (
            device.index if device.index is not None else torch.cuda.current_device()
        )
        random_devices = [random_device_index]
        random_device_type = "cuda"
    elif device.type == "mps":
        random_device_index = 0
        random_devices = [random_device_index]
        random_device_type = "mps"
    else:
        random_device_index = None
        random_devices = []
        random_device_type = None
    _synchronize_device(device)
    started = time.monotonic()
    names = tuple(conditions)
    for step in range(resumed_from_step, settings.steps):
        row_index = prepared.effective_rows[step]
        xyz = prepared.xyz.index_select(0, row_index)
        valid = prepared.valid.index_select(0, row_index)
        patches = prepared.valid_patch.index_select(0, row_index)
        view_seed = settings.seed + 100_003 * (settings.fold + 1) + step
        with torch.random.fork_rng(
            devices=random_devices,
            device_type=random_device_type,
        ):
            if device.type == "cuda":
                with torch.cuda.device(random_device_index):
                    torch.cuda.manual_seed(view_seed)
            elif device.type == "mps":
                torch.mps.manual_seed(view_seed)
            else:
                torch.random.default_generator.manual_seed(view_seed)
            both = geometric_view(
                torch.cat((xyz, xyz), dim=0),
                torch.cat((valid, valid), dim=0),
                max_degrees=8,
                translate=0.03,
            )
        view_a, _ = both.chunk(2)
        both_valid = torch.cat((patches, patches), dim=0)

        for name in names:
            arm = arms[name]
            model = arm["model"]
            projector = arm["projector"]
            mask = prepared.target_masks[name][step]
            equal_counts = prepared.equal_counts[name][step]
            prediction_loss, target, counts = _prediction_prevalidated(
                model,
                view_a,
                xyz,
                patches,
                mask,
                equal_target_counts=equal_counts,
            )
            tokens = _encode_prevalidated(
                model.view_encoder, both, both_valid
            ).reshape(
                2 * settings.batch_size,
                -1,
                33,
                settings.embed_dim,
            )
            projected = projector(
                authorized_pool(tokens, both_valid, GAIT_JOINTS)
            )
            regularizer = vicreg_loss(*projected.chunk(2))
            loss = prediction_loss + settings.vicreg_weight * regularizer
            # A CPU scalar check is cheap and fails before gradients are used.
            # Accelerators validate the buffered trace at each resume boundary,
            # avoiding a device synchronization on every optimizer update.
            if device.type == "cpu" and not bool(torch.isfinite(loss).all()):
                raise FloatingPointError(f"Non-finite training loss for {name}")
            arm["optimizer"].zero_grad(set_to_none=True)
            loss.backward()
            if any(
                parameter.grad is not None
                for parameter in model.target_encoder.parameters()
            ):
                raise AssertionError("Teacher received gradients")
            torch.nn.utils.clip_grad_norm_(arm["trainable"], 1.0)
            arm["optimizer"].step()
            model.update_target(settings.ema_momentum)
            with torch.no_grad():
                if equal_counts:
                    mean = target.mean(dim=(0, 1))
                else:
                    hidden = mask.flatten(1)
                    mean = (
                        (target * hidden[..., None]).sum(1)
                        / counts[:, None]
                    ).mean(0)
                model.target_center.mul_(0.9).add_(mean, alpha=0.1)
            diagnostics = torch.stack((
                loss.detach(),
                prediction_loss.detach(),
                regularizer.detach(),
            ))
            arm["diagnostics"].append(diagnostics)
            if step + 1 in checkpoint_steps:
                arm["snapshots"][step + 1] = {
                    state_name: value.detach().cpu().clone()
                    for state_name, value in model.state_dict().items()
                }
            should_report = (
                progress_callback is not None
                and (
                    (step + 1) % progress_interval == 0
                    or step + 1 == settings.steps
                )
            )
            if should_report:
                loss_value = float(diagnostics[0].cpu())
                if not math.isfinite(loss_value):
                    raise FloatingPointError("Non-finite training loss")
                progress_callback({
                    "condition": name,
                    "condition_index": int(arm["index"]),
                    "step": step + 1,
                    "total_steps": settings.steps,
                    "condition_count": len(conditions),
                    "comparison_completed_steps": (
                        step * len(conditions) + int(arm["index"]) + 1
                    ),
                    "comparison_total_steps": settings.steps * len(conditions),
                    "loss": loss_value,
                })
        if (
            resume_path is not None
            and (
                (step + 1) % resume_interval == 0
                or step + 1 == settings.steps
            )
        ):
            _synchronize_device(device)
            _write_training_resume(
                resume_path,
                identity,
                step + 1,
                arms,
                earlier_elapsed + time.monotonic() - started,
            )

    _synchronize_device(device)
    elapsed = earlier_elapsed + time.monotonic() - started
    runs = {}
    for name in names:
        arm = arms[name]
        new_values = _diagnostic_values(arm, label=name)
        values = np.concatenate(
            (arm["history_prefix"], new_values), axis=0
        )
        if values.shape != (settings.steps, 3):
            raise RuntimeError("Training history does not cover every update")
        if not np.isfinite(values).all():
            raise FloatingPointError("Non-finite training diagnostics")
        counts = prepared.target_counts[name]
        history = pd.DataFrame([
            {
                "step": step + 1,
                "loss": float(values[step, 0]),
                "masked_prediction_loss": float(values[step, 1]),
                "variance_regularizer": float(values[step, 2]),
                "hidden_count_min": int(min(counts[step])),
                "hidden_count_max": int(max(counts[step])),
            }
            for step in range(settings.steps)
        ])
        runs[name] = {
            "model": arm["model"].eval(),
            "initial_model": arm["initial_model"],
            "projector": arm["projector"].eval(),
            "settings": asdict(settings),
            "policy": asdict(arm["policy"]),
            "history": history,
            "initial_state_digest": arm["initial_digest"],
            "initial_projector_digest": arm["initial_projector_digest"],
            "source_draw_digest": _digest_array(prepared.schedule),
            "view_digest": prepared.view_digest,
            "hidden_token_counts": counts,
            "coverage": prepared.coverage[name],
            "checkpoints": arm["snapshots"],
            "elapsed_training_seconds": elapsed,
            "resident_input_bytes": prepared.resident_bytes,
            "execution_layout": (
                "resident fold tensors; shared augmented views; "
                "independent arm parameters and optimizer states"
            ),
        }
    pairing = {"same_initialization": len({r["initial_state_digest"] for r in runs.values()}) == 1,
        "same_projector_initialization": len({r["initial_projector_digest"] for r in runs.values()}) == 1,
        "same_source_draws": len({r["source_draw_digest"] for r in runs.values()}) == 1,
        "same_geometric_views": len({r["view_digest"] for r in runs.values()}) == 1,
        "same_hidden_counts": len({json.dumps(r["hidden_token_counts"]) for r in runs.values()}) == 1}
    if not all(pairing.values()):
        raise AssertionError(f"Comparison controls failed: {pairing}")
    result = {"runs": runs, "pairing": pairing, "identity": identity,
        "source_schedule": prepared.schedule, "reflection_schedule": prepared.reflections,
        "synthetic": dataset.synthetic, "reused": False,
        "resumed_from_step": resumed_from_step}
    if destination is not None:
        save_comparison(result, destination)
        if resume_path is not None:
            for path in (resume_path, _resume_checksum_path(resume_path)):
                path.unlink(missing_ok=True)
    return result


def save_comparison(result, destination):
    """Publish a complete result directory atomically; never overwrite a run."""
    destination = Path(destination)
    if destination.exists():
        raise FileExistsError(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=".incomplete_", dir=destination.parent))
    try:
        manifest = {"schema": SCHEMA, "complete": True, "identity": result["identity"],
            "pairing": result["pairing"],
            "resumed_from_step": int(result.get("resumed_from_step", 0)),
            "runs": {}, "files": {}}
        for name, run in result["runs"].items():
            torch.save({"model": run["model"].state_dict(), "initial_model": run["initial_model"].state_dict(),
                "projector": run["projector"].state_dict(), "checkpoints": run["checkpoints"]}, staging / f"{name}.pt")
            run["history"].to_csv(staging / f"{name}_training.csv", index=False)
            manifest["runs"][name] = {k: run[k] for k in ("settings", "policy", "initial_state_digest", "initial_projector_digest",
                "source_draw_digest", "view_digest", "hidden_token_counts", "coverage", "elapsed_training_seconds",
                "resident_input_bytes", "execution_layout")}
        np.save(staging / "source_schedule.npy", result["source_schedule"], allow_pickle=False)
        np.save(staging / "reflection_schedule.npy", result["reflection_schedule"], allow_pickle=False)
        for path in staging.iterdir():
            manifest["files"][path.name] = hashlib.sha256(path.read_bytes()).hexdigest()
        (staging / "manifest.json").write_text(json.dumps(manifest, indent=2, allow_nan=False) + "\n")
        # A competing completed directory is nonempty, so rename cannot replace it.
        if destination.exists():
            raise FileExistsError(destination)
        staging.rename(destination)
    finally:
        shutil.rmtree(staging, ignore_errors=True)


def load_comparison(destination, expected_identity):
    destination = Path(destination)
    manifest = json.loads((destination / "manifest.json").read_text())
    if manifest.get("schema") != SCHEMA or not manifest.get("complete"):
        raise ValueError("Saved comparison is incomplete or uses another format")
    if canonical_json_digest(manifest.get("identity")) != canonical_json_digest(expected_identity):
        raise ValueError("Saved comparison is incompatible with the requested experiment")
    names = set(expected_identity["conditions"])
    expected_files = {"source_schedule.npy", "reflection_schedule.npy", *[f"{n}.pt" for n in names], *[f"{n}_training.csv" for n in names]}
    if set(manifest.get("runs", {})) != names or set(manifest.get("files", {})) != expected_files:
        raise ValueError("Saved comparison is missing or duplicating declared artifacts")
    for name, digest in manifest["files"].items():
        path = destination / name
        if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != digest:
            raise ValueError(f"Saved comparison artifact failed verification: {name}")
    runs = {}
    settings = LearningSettings(**expected_identity["settings"])
    schedule = np.load(destination / "source_schedule.npy", allow_pickle=False)
    reflections = np.load(destination / "reflection_schedule.npy", allow_pickle=False)
    shape = (settings.steps, settings.batch_size)
    if (schedule.shape != shape or not np.issubdtype(schedule.dtype, np.integer)
            or schedule.min() < 0 or schedule.max() >= len(expected_identity["source_ids"])
            or reflections.shape != shape or reflections.dtype != bool):
        raise ValueError("Saved source or reflection schedule has an invalid shape or value")
    schedule_sources = np.asarray(expected_identity["source_ids"])[schedule.ravel()]
    if not set(schedule_sources) <= set(expected_identity["train_sources"]):
        raise ValueError("Saved source schedule contains a test video")
    if not np.array_equal(schedule, declared_source_schedule(
            expected_identity["source_ids"], expected_identity["train_sources"], settings)):
        raise ValueError("Saved source schedule disagrees with its declared sampling seed")
    reflection_rng = np.random.default_rng(np.random.SeedSequence([settings.seed, settings.fold, 103]))
    expected_reflections = reflection_rng.random(shape) < settings.reflection_probability
    if not np.array_equal(reflections, expected_reflections):
        raise ValueError("Saved reflection schedule disagrees with its declared sampling seed")
    for name, metadata in manifest["runs"].items():
        model, projector = _new_model(settings, expected_identity["frames"], settings.device)
        initial_model = copy.deepcopy(model).to("cpu")
        if (metadata["settings"] != expected_identity["settings"]
                or metadata["policy"] != expected_identity["conditions"][name]
                or metadata["source_draw_digest"] != _digest_array(schedule)
                or metadata["initial_state_digest"] != state_digest(initial_model)
                or metadata["initial_projector_digest"] != state_digest(projector)):
            raise ValueError("Saved run metadata disagree with the experiment or its controls")
        # Validate cached tensors on CPU. Repeated scalar checks on MPS/CUDA
        # would otherwise serialize the accelerator before any useful work.
        saved = torch.load(
            destination / f"{name}.pt", map_location="cpu", weights_only=True
        )
        if set(saved["checkpoints"]) != set(expected_identity["checkpoint_steps"]):
            raise ValueError("Saved checkpoints do not match the declared update positions")
        for state in (saved["model"], saved["initial_model"], *saved["checkpoints"].values()):
            if set(state) != set(model.state_dict()) or any(
                    v.shape != model.state_dict()[k].shape or not torch.isfinite(v).all()
                    for k, v in state.items()):
                raise ValueError("Saved model or checkpoint is incomplete or nonfinite")
        if (set(saved["projector"]) != set(projector.state_dict()) or any(
                value.shape != projector.state_dict()[key].shape or not torch.isfinite(value).all()
                for key, value in saved["projector"].items())):
            raise ValueError("Saved projector is incomplete or nonfinite")
        if any(not torch.equal(v, saved["checkpoints"][settings.steps][k]) for k, v in saved["model"].items()):
            raise ValueError("Final model and final declared checkpoint disagree")
        model.load_state_dict(saved["model"]); initial_model.load_state_dict(saved["initial_model"])
        projector.load_state_dict(saved["projector"])
        history = pd.read_csv(destination / f"{name}_training.csv")
        if (history.step.tolist() != list(range(1, settings.steps + 1))
                or not np.isfinite(history.select_dtypes(include=np.number)).all().all()):
            raise ValueError("Training history does not cover the complete run")
        counts = np.asarray(metadata["hidden_token_counts"])
        if counts.shape != shape or not np.issubdtype(counts.dtype, np.integer) or counts.min() < 1:
            raise ValueError("Saved hidden counts have invalid coverage")
        if len(metadata["coverage"]) != settings.steps:
            raise ValueError("Saved mask coverage omits training updates")
        if (
            not isinstance(metadata.get("resident_input_bytes"), int)
            or metadata["resident_input_bytes"] < 1
            or not isinstance(metadata.get("execution_layout"), str)
            or not metadata["execution_layout"]
        ):
            raise ValueError("Saved execution metadata are invalid")
        for step, coverage in enumerate(metadata["coverage"]):
            if len(coverage) != settings.batch_size:
                raise ValueError("Saved mask coverage omits clips")
            for row, c in enumerate(coverage):
                if (c["hidden_tokens"] != counts[step, row] or c["context_tokens"] < 1
                        or c["valid_tokens"] != c["hidden_tokens"] + c["context_tokens"]
                        or c["reflected"] != bool(reflections[step, row])
                        or sum(c[k] for k in ("left_hidden_tokens", "right_hidden_tokens", "midline_hidden_tokens")) != c["hidden_tokens"]):
                    raise ValueError("Saved mask coverage and target counts disagree")
        if state_digest(initial_model) != metadata["initial_state_digest"]:
            raise ValueError("Initial checkpoint does not match the retained controls")
        runs[name] = {**metadata, "model": model.eval(), "initial_model": initial_model.eval(),
            "projector": projector.eval(), "history": history, "checkpoints": saved["checkpoints"]}
    recomputed = {"same_initialization": len({r["initial_state_digest"] for r in runs.values()}) == 1,
        "same_projector_initialization": len({r["initial_projector_digest"] for r in runs.values()}) == 1,
        "same_source_draws": len({r["source_draw_digest"] for r in runs.values()}) == 1,
        "same_geometric_views": len({r["view_digest"] for r in runs.values()}) == 1,
        "same_hidden_counts": len({json.dumps(r["hidden_token_counts"]) for r in runs.values()}) == 1}
    if manifest["pairing"] != recomputed or not all(recomputed.values()):
        raise ValueError("Saved comparison controls did not pass")
    resumed_from_step = manifest.get("resumed_from_step", 0)
    if (
        not isinstance(resumed_from_step, int)
        or resumed_from_step < 0
        or resumed_from_step > settings.steps
    ):
        raise ValueError("Saved resume provenance is invalid")
    return {"runs": runs, "pairing": manifest["pairing"], "identity": expected_identity,
        "source_schedule": schedule, "reflection_schedule": reflections,
        "synthetic": expected_identity["synthetic"], "reused": True,
        "resumed_from_step": resumed_from_step}


def saved_reference_recipe(root=SUITE_ROOT):
    """Read all 25 retained pairs and require one consistent reference recipe."""
    manifests = sorted((Path(root) / "artifacts/research_extensions/masking").glob("exploratory_real_*/manifest.json"))
    recipes, pairs = [], []
    for path in manifests:
        manifest = json.loads(path.read_text())
        if not manifest.get("complete") or manifest.get("synthetic"):
            continue
        if not all(manifest.get("pairing", {}).values()):
            raise ValueError("A retained reference comparison failed its controls")
        arm_recipes = []
        for run in manifest["runs"].values():
            s = dict(run["settings"])
            arm_recipes.append({k: v for k, v in s.items() if k not in {"seed", "fold", "mask_policy", "device", "confirm_real_run"}})
        if any(r != arm_recipes[0] for r in arm_recipes):
            raise ValueError("Reference arms differ in an unexpected setting")
        recipes.append(arm_recipes[0]); pairs.append((s["fold"], s["seed"]))
    if len(recipes) != 25 or set(pairs) != {(f, s) for f in range(5) for s in range(42, 47)}:
        raise ValueError("The expected complete reference grid was not found")
    if any(recipe != recipes[0] for recipe in recipes):
        raise ValueError("Saved reference recipes disagree")
    return {**recipes[0], "optimizer": "AdamW", "optimizer_betas": (0.9, .95),
        "learning_rate_schedule": "constant", "teacher_momentum_schedule": "constant",
        "target_center_beta": .9, "predictor_temperature": .10, "target_temperature": .06,
        "gradient_clip_norm": 1., "view_rotation_degrees": 8., "view_translation": .03,
        "input_landmarks": 33, "regularizer_landmarks": list(GAIT_JOINTS),
        "reference_pairs": len(pairs), "reference_trained_encoders": 2 * len(pairs)}


def real_settings(*, fold=0, seed=42, device="cpu", confirm_real_run=False):
    recipe = saved_reference_recipe()
    fields = set(LearningSettings.__dataclass_fields__)
    return LearningSettings(**{k: v for k, v in recipe.items() if k in fields},
        fold=fold, seed=seed, device=device, confirm_real_run=confirm_real_run)


def plan_real_comparison(*, conditions=None, folds=tuple(range(5)), seeds=tuple(range(42, 47)),
        device="cpu", output_dir=None, budgets=None, matched_to=None, validate_inputs=False):
    """Describe every requested fit. Optional input validation performs no training."""
    conditions = dict(conditions or default_conditions())
    folds, seeds = tuple(folds), tuple(seeds)
    if not folds or not seeds or len(set(folds)) != len(folds) or len(set(seeds)) != len(seeds):
        raise ValueError("Declare nonempty unique folds and seeds")
    if not set(folds) <= set(range(5)) or not set(seeds) <= set(range(42, 47)):
        raise ValueError("The declared real comparison uses folds 0–4 and seeds 42–46")
    settings = real_settings(device=device)
    rows = [{"fold": f, "seed": s, "condition": n, "updates": settings.steps,
             "batch_size": settings.batch_size} for f in folds for s in seeds for n in conditions]
    checks = []
    if validate_inputs:
        for fold in folds:
            data = load_learning_dataset(real=True, fold=fold)
            setting = replace(settings, fold=fold)
            # One pass over all training clips checks policy feasibility; full
            # scheduled masks are checked again before actual model training.
            for start in range(0, len(data.train_rows), setting.batch_size):
                masks_for_batch(data, data.train_rows[start:start + setting.batch_size], setting,
                    conditions, budgets=budgets, matched_to=matched_to, step=start)
            checks.append({"fold": fold, "train_clips": len(data.train_rows), "test_clips": len(data.test_rows),
                "train_sources": len(data.train_sources), "test_sources": len(data.test_sources)})
    return {"workload": pd.DataFrame(rows), "training_runs": len(rows),
        "optimizer_updates": len(rows) * settings.steps,
        "sample_presentations": len(rows) * settings.steps * settings.batch_size,
        "scope": "full declared comparison" if folds == tuple(range(5)) and seeds == tuple(range(42, 47)) else "pilot",
        "settings": asdict(settings), "recipe": saved_reference_recipe(),
        "conditions": {n: asdict(p) for n, p in conditions.items()},
        "budgets": {n: asdict(b) for n, b in (budgets or {}).items()}, "matched_to": matched_to,
        "folds": folds, "seeds": seeds, "input_checks": pd.DataFrame(checks),
        "output_dir": str(output_dir or SUITE_ROOT / "artifacts/comparative_masking"),
        "runtime_estimate": "Measure a training-source pilot; equal updates do not imply equal runtime."}


def _brief_duration(seconds):
    seconds = max(0, int(round(seconds)))
    hours, remainder = divmod(seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    if hours:
        return f"{hours}h {minutes:02d}m"
    if minutes:
        return f"{minutes}m {seconds:02d}s"
    return f"{seconds}s"


def run_real_comparison(
    plan,
    *,
    enabled=False,
    progress_callback=None,
    progress_interval=100,
    resume_interval=300,
    log=print,
):
    """Execute a real grid with compact, reader-oriented status messages."""
    emit = log if log is not None else (lambda _message: None)
    conditions = {n: MaskPolicy(**p) for n, p in plan["conditions"].items()}
    budgets = {n: MaskBudget(**b) for n, b in plan["budgets"].items()}
    comparison_jobs = len(plan["folds"]) * len(plan["seeds"])
    base_settings = LearningSettings(**plan["settings"])
    runtime = configure_learning_runtime(base_settings.device)
    base_settings = replace(base_settings, device=runtime["device"])
    backend = {
        "mps": "MPS (Apple Metal GPU)",
        "cuda": "CUDA (NVIDIA GPU)",
        "cpu": f"CPU ({runtime['cpu_threads']} PyTorch threads)",
    }[torch.device(runtime["device"]).type]

    emit("Controlled masking experiment")
    emit(
        f"What will run: {comparison_jobs} fold/seed jobs. Each job compares "
        f"{len(conditions)} independently optimized masking encoders at "
        f"{base_settings.steps:,} updates per encoder."
    )
    emit(
        f"Total work: {plan['training_runs']} encoders and "
        f"{plan['optimizer_updates']:,} optimizer updates. Hardware: {backend}."
    )
    emit(
        "Memory strategy: run one job at a time; keep its data and masks on the "
        f"selected device; share each augmented batch across all {len(conditions)} "
        "encoder arms."
    )
    emit(
        f"Reuse and recovery: completed jobs are reused. During a new job, "
        f"all {len(conditions)} encoder arms are saved together every "
        f"{resume_interval} updates per encoder "
        f"and after their final update."
        if resume_interval
        else "Reuse and recovery: completed jobs are reused; interrupted-job checkpoints are disabled."
    )
    emit(f"Results will be saved under: {plan['output_dir']}")
    if not enabled:
        emit("Training is disabled. Set the notebook's real-training switch to run this plan.")
        return {"status": "Training disabled", "plan": plan, "runtime": runtime}

    completed, predictions, expected_rows = [], [], []
    counts = {
        "training_reused": 0,
        "training_resumed": 0,
        "training_new": 0,
        "evaluation_reused": 0,
        "evaluation_new": 0,
    }
    experiment_started = time.monotonic()
    job_index = 0
    for fold in plan["folds"]:
        data = load_learning_dataset(real=True, fold=fold)
        expected_rows.append(pd.DataFrame({"sequence_id": data.sequence_ids[data.test_rows],
            "source_id": data.source_ids[data.test_rows], "fold": fold}))
        for seed in plan["seeds"]:
            job_index += 1
            settings = replace(
                base_settings,
                fold=fold,
                seed=seed,
                confirm_real_run=True,
            )
            expected_identity = comparison_identity(
                data,
                settings,
                conditions,
                budgets,
                plan["matched_to"],
                (settings.steps,),
            )
            destination = Path(plan["output_dir"]) / canonical_json_digest(
                expected_identity
            )
            resume_candidate = _resume_path(destination)
            if destination.exists():
                action = "checking and reusing the completed training cache"
            elif resume_interval and resume_candidate.exists():
                action = (
                    "checking the interrupted-job checkpoint, then resuming "
                    f"all {len(conditions)} encoder arms"
                )
            else:
                action = (
                    f"training all {len(conditions)} encoder arms from their "
                    "shared initialization"
                )
            emit(
                f"[{job_index:02d}/{comparison_jobs:02d}] Starting outer fold {fold}, "
                f"seed {seed}: {action}."
            )

            def report(update):
                if progress_callback is not None:
                    progress_callback({
                        **update,
                        "fold": fold,
                        "seed": seed,
                        "job": job_index,
                        "total_jobs": comparison_jobs,
                    })

            job_started = time.monotonic()
            result = train_comparison(
                data,
                settings,
                conditions,
                budgets=budgets,
                matched_to=plan["matched_to"],
                output_dir=plan["output_dir"],
                progress_callback=report if progress_callback is not None else None,
                progress_interval=progress_interval,
                resume_interval=resume_interval,
            )
            resumed_from = int(result.get("resumed_from_step", 0))
            if result["reused"]:
                training_status = "completed cache reused"
                counts["training_reused"] += 1
            elif resumed_from:
                training_status = f"resumed after update {resumed_from} and completed"
                counts["training_resumed"] += 1
            else:
                training_status = f"{len(conditions)} new encoders trained"
                counts["training_new"] += 1
            evaluation = evaluate_comparison(result, data, settings,
                output_dir=Path(plan["output_dir"]) / "evaluations")
            evaluation_reused = bool(evaluation.get("evaluation_reused", False))
            counts["evaluation_reused" if evaluation_reused else "evaluation_new"] += 1
            predictions.append(evaluation["predictions"])
            completed.append({
                "fold": fold,
                "seed": seed,
                "identity": result["identity"],
                "evaluation_identity": evaluation["evaluation_identity"],
                "reused": result["reused"],
                "resumed_from_step": resumed_from,
                "evaluation_reused": evaluation_reused,
            })
            emit(
                f"[{job_index:02d}/{comparison_jobs:02d}] Finished outer fold {fold}, "
                f"seed {seed}. Training: {training_status}. Frozen-feature evaluation: "
                f"{'reused' if evaluation_reused else 'computed and saved'}. Elapsed: "
                f"{_brief_duration(time.monotonic() - job_started)}."
            )
            del result
    emit("All fold/seed jobs are complete. Building the source-balanced held-out summary.")
    from laterality_extensions.comparative_evaluation import aggregate_predictions, paired_source_bootstrap
    table = pd.concat(predictions, ignore_index=True)
    expected = pd.concat(expected_rows, ignore_index=True)
    summaries = aggregate_predictions(table, expected, seeds=plan["seeds"], conditions=tuple(conditions),
        representations=tuple(table.representation.unique()), observations=tuple(table.observation.unique()))
    intervals = []
    first = next(iter(conditions))
    for other in list(conditions)[1:]:
        intervals.append(paired_source_bootstrap(table, first=other, reference=first,
            representation="pretrained_teacher"))
    tables = {"predictions": table, **summaries, "paired_intervals": pd.DataFrame(intervals)}
    summary_identity = {"runs": [canonical_json_digest(c["identity"]) for c in completed],
        "evaluations": [c["evaluation_identity"] for c in completed],
        "scope": plan["scope"], "aggregation": "pooled outer predictions per seed; source balanced",
        "training_implementation": training_implementation_digest(),
        "evaluation_implementation": evaluation_implementation_digest(),
        "aggregation_implementation": summary_implementation_digest()}
    _save_tables(tables, summary_identity, Path(plan["output_dir"]) / "summaries")
    emit(
        f"Finished {comparison_jobs}/{comparison_jobs} jobs in "
        f"{_brief_duration(time.monotonic() - experiment_started)}. Training jobs: "
        f"{counts['training_reused']} reused · {counts['training_resumed']} resumed · "
        f"{counts['training_new']} new. Evaluation: "
        f"{counts['evaluation_reused']} reused · {counts['evaluation_new']} new."
    )
    return {
        "status": "Complete",
        "completed": completed,
        "plan": plan,
        "runtime": runtime,
        "run_counts": counts,
        **tables,
    }


def summary_implementation_digest():
    """Hash grid aggregation separately from fitted-model implementation."""
    source = Path(__file__).read_bytes()
    start = b"\ndef run_real_comparison("
    end = b"\ndef evaluation_implementation_digest("
    if start not in source or end not in source:
        raise RuntimeError("Comparative summary digest boundary is missing")
    section = start + source.split(start, 1)[1].split(end, 1)[0]
    return hashlib.sha256(
        b"comparative_masking_summary/v1\0" + section
    ).hexdigest()


def evaluation_implementation_digest():
    """Hash evaluation code separately so prose/log edits do not retrain models."""
    extension_root = Path(__file__).resolve().parent
    laterality_root = extension_root.parent / "laterality"
    paths = [
        extension_root / "comparative_evaluation.py",
        *[
            laterality_root / name
            for name in ("evaluation.py", "geometry.py", "metrics.py", "model.py")
        ],
    ]
    digest = hashlib.sha256()
    for path in paths:
        if not path.is_file():
            raise FileNotFoundError(path)
        digest.update(path.name.encode())
        digest.update(path.read_bytes())
    source = Path(__file__).read_bytes()
    boundary = b"\ndef _save_tables("
    if boundary not in source:
        raise RuntimeError("Comparative evaluation digest boundary is missing")
    digest.update(b"comparative_training_evaluation_section")
    digest.update(boundary + source.split(boundary, 1)[1])
    return digest.hexdigest()


def _load_tables(identity, parent, names):
    """Load a complete immutable table cache before repeating evaluation work."""
    destination = Path(parent) / canonical_json_digest(identity)
    if not destination.exists():
        return None
    manifest_path = destination / "manifest.json"
    if not manifest_path.is_file():
        raise ValueError("Saved evaluation is missing its manifest")
    manifest = json.loads(manifest_path.read_text())
    expected = {f"{name}.csv" for name in names}
    if (
        not manifest.get("complete")
        or canonical_json_digest(manifest.get("identity"))
        != canonical_json_digest(identity)
        or set(manifest.get("files", {})) != expected
    ):
        raise ValueError("Saved evaluation is incomplete or incompatible")
    for name, digest in manifest["files"].items():
        path = destination / name
        if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != digest:
            raise ValueError("Saved evaluation failed its content check")
    tables = {
        name: pd.read_csv(destination / f"{name}.csv")
        for name in names
    }
    return tables, destination


def _save_tables(tables, identity, parent):
    """Save immutable tables, or verify an identical concurrent/cache result."""
    parent = Path(parent)
    destination = parent / canonical_json_digest(identity)
    parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=".incomplete_evaluation_", dir=parent))
    names = tuple(tables)
    try:
        for name, table in tables.items():
            table.to_csv(temporary / f"{name}.csv", index=False)
        new_files = {
            path.name: hashlib.sha256(path.read_bytes()).hexdigest()
            for path in temporary.iterdir()
        }
        manifest = {
            "complete": True,
            "identity": identity,
            "files": new_files,
        }
        (temporary / "manifest.json").write_text(
            json.dumps(manifest, indent=2) + "\n"
        )

        def verify_existing():
            loaded = _load_tables(identity, parent, names)
            if loaded is None:
                raise FileExistsError(destination)
            existing_manifest = json.loads(
                (destination / "manifest.json").read_text()
            )
            if existing_manifest["files"] != new_files:
                raise ValueError(
                    "Saved evaluation disagrees with newly computed tables"
                )

        if destination.exists():
            verify_existing()
            return destination
        try:
            temporary.rename(destination)
        except OSError:
            if not destination.exists():
                raise
            verify_existing()
        return destination
    finally:
        shutil.rmtree(temporary, ignore_errors=True)


def evaluate_comparison(result, dataset, settings, *, output_dir=None):
    """Apply the same fixed evaluation bank and training-only readout to all arms.

    The default missing-input scores concern already prepared coordinates.
    Raw missing-data preparation is a separate demonstrated helper in tutorial 13.
    """
    from laterality_extensions.comparative_evaluation import (
        evaluate_frozen_representations, make_evaluation_mask_bank,
        prepared_observation_sensitivity, predictor_diagnostics,
    )
    import sklearn
    table_names = (
        "predictions", "predictor_diagnostics", "readout_validation",
        "training_feature_diagnostics",
    )
    identity = {
        "training": canonical_json_digest(result["identity"]),
        "model_states": {
            name: {
                "final": state_digest(run["model"]),
                "initial": state_digest(run["initial_model"]),
            }
            for name, run in result["runs"].items()
        },
        "evaluation": {
            "alphas": [.01, .1, 1., 10., 100.],
            "inner_folds": 3,
            "mask_bank_seed": 813,
            "mismatch_seed": 991,
            "missingness": "prepared-coordinate sensitivity",
        },
        "implementation": evaluation_implementation_digest(),
        "runtime": {
            "python": platform.python_version(),
            "torch": str(torch.__version__),
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "sklearn": sklearn.__version__,
        },
    }
    if output_dir is not None:
        cached = _load_tables(identity, output_dir, table_names)
        if cached is not None:
            tables, destination = cached
            tables["output_path"] = destination
            tables["evaluation_reused"] = True
            tables["evaluation_identity"] = canonical_json_digest(identity)
            return tables

    patches = dataset.valid.reshape(len(dataset.xyz), -1, settings.segment_length, 33).all(2)
    bank = make_evaluation_mask_bank(patches)
    observations = {f"prepared_{name}": prepared_observation_sensitivity(dataset, mask,
        segment_length=settings.segment_length) for name, mask in bank.items()}
    # The comparison definition is shared across folds and seeds; each training
    # cache identity separately identifies its particular fitted models.
    design = {k: result["identity"][k] for k in ("conditions", "budgets", "matched_to", "implementation")}
    design["training_settings"] = {k: v for k, v in result["identity"]["settings"].items()
                                   if k not in {"seed", "fold", "confirm_real_run"}}
    comparison_id = canonical_json_digest(design)
    prediction_tables, diagnostics, validation, feature_checks = [], [], [], []
    feature_cache = {}
    diagnostic_cache = {}
    run_values = list(result["runs"].values())
    evaluation_devices = {
        str(next(run["model"].parameters()).device) for run in run_values
    }
    if len(evaluation_devices) != 1:
        raise ValueError("Compared encoders must use one evaluation device")
    recorded_initial_digests = {
        run["initial_state_digest"] for run in run_values
    }
    actual_initial_digests = {
        state_digest(run["initial_model"]) for run in run_values
    }
    if (
        len(recorded_initial_digests) != 1
        or actual_initial_digests != recorded_initial_digests
    ):
        raise ValueError("Compared encoders do not share the declared initial state")
    evaluation_device = next(iter(evaluation_devices))
    # Evaluate one independent copy of the verified common initialization.
    # The caller's retained CPU baselines therefore stay unchanged on failure.
    shared_initial_model = copy.deepcopy(
        run_values[0]["initial_model"]
    ).to(evaluation_device)
    for name, run in result["runs"].items():
        state_names = identity["model_states"][name]
        evaluation = evaluate_frozen_representations(
            run["model"],
            shared_initial_model,
            dataset,
            settings,
            condition=name,
            observation_datasets=observations,
            comparison_id=comparison_id,
            feature_cache=feature_cache,
            feature_cache_names={
                "pretrained_online": f"{state_names['final']}:online",
                "pretrained_teacher": f"{state_names['final']}:teacher",
                "initial_online": f"{state_names['initial']}:online",
                "direct_pose": "shared_direct_pose",
            },
        )
        prediction_tables.append(evaluation["predictions"])
        for representation, readout in evaluation["readouts"].items():
            validation.append(readout.validation.assign(condition=name, representation=representation,
                selected_alpha=readout.selected_alpha, fold=dataset.fold, seed=settings.seed))
            feature_checks.append({"condition": name, "representation": representation,
                **readout.training_feature_diagnostics})
        for state, model in (("pretrained", run["model"]), ("initial", shared_initial_model)):
            model_digest = state_names["final" if state == "pretrained" else "initial"]
            if model_digest not in diagnostic_cache:
                diagnostic_cache[model_digest] = predictor_diagnostics(
                    model, dataset, bank, condition="shared_state"
                )
            diagnostics.append(
                diagnostic_cache[model_digest].assign(
                    condition=name,
                    encoder_state=state,
                    fold=dataset.fold,
                    seed=settings.seed,
                )
            )
    tables = {"predictions": pd.concat(prediction_tables, ignore_index=True),
        "predictor_diagnostics": pd.concat(diagnostics, ignore_index=True),
        "readout_validation": pd.concat(validation, ignore_index=True),
        "training_feature_diagnostics": pd.DataFrame(feature_checks)}
    if output_dir is not None:
        tables["output_path"] = _save_tables(tables, identity, output_dir)
    tables["evaluation_reused"] = False
    tables["evaluation_identity"] = canonical_json_digest(identity)
    return tables
