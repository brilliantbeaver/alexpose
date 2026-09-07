"""Reflection-aware learning experiments, separate from the registered study.

These helpers use the registered anatomical reflection but do not modify its
training code, checkpoints, or decision thresholds. A small reflection error
is a consistency measurement; useful predictive information needs its own test.
"""

from __future__ import annotations

from dataclasses import replace
from typing import Any

import numpy as np
import torch

from laterality.geometry import FULL_MIRROR_PAIRS


def reflection_permutation(joints: int = 33) -> list[int]:
    """Return the supplied BlazePose joint action, including unpaired points."""
    if joints != 33:
        raise ValueError("This experiment requires the registered 33-joint schema")
    permutation = list(range(joints))
    for left, right in FULL_MIRROR_PAIRS:
        permutation[left], permutation[right] = right, left
    return permutation


def reflect_training_batch(
    coordinates: torch.Tensor,
    valid: torch.Tensor,
    target_mask: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
    """Reflect coordinates and apply the same joint swap to both masks.

    Coordinates are [batch, frames, 33, 3]. ``valid`` can describe frames or
    temporal patches, and ``target_mask`` describes the hidden temporal patches.
    Reflection leaves the temporal axis unchanged. Nothing is modified in place.
    """
    if coordinates.ndim != 4 or coordinates.shape[-2:] != (33, 3):
        raise ValueError("coordinates must have shape [batch, frames, 33, 3]")
    if valid.ndim != 3 or valid.shape[0] != len(coordinates) or valid.shape[-1] != 33:
        raise ValueError("valid must have shape [batch, frames-or-patches, 33]")
    if valid.dtype != torch.bool:
        raise ValueError("valid must contain Boolean observation flags")
    index = torch.tensor(reflection_permutation(), device=coordinates.device)
    reflected = coordinates.index_select(-2, index).clone()
    reflected[..., 0] *= -1
    reflected_valid = valid.index_select(-1, index.to(valid.device))
    reflected_mask = None
    if target_mask is not None:
        if target_mask.shape != valid.shape or target_mask.dtype != torch.bool:
            raise ValueError("target_mask must be Boolean and match patch validity")
        if (target_mask & ~valid).any():
            raise ValueError("The target mask contains an invalid prediction target")
        reflected_mask = target_mask.index_select(-1, index.to(target_mask.device))
    return reflected, reflected_valid, reflected_mask


def _aligned_tokens(
    original: torch.Tensor,
    reflected: torch.Tensor,
    valid_original: torch.Tensor,
    valid_reflected: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if original.shape != reflected.shape or original.ndim != 4:
        raise ValueError("Both token arrays must have shape [batch, patches, 33, channels]")
    if original.shape[-2] != 33 or valid_original.shape != original.shape[:-1]:
        raise ValueError("Token shape and original validity disagree")
    if valid_reflected.shape != valid_original.shape:
        raise ValueError("The two validity masks must have the same shape")
    if valid_original.dtype != torch.bool or valid_reflected.dtype != torch.bool:
        raise ValueError("Validity masks must be Boolean")
    index = torch.tensor(reflection_permutation(), device=original.device)
    expected = original.index_select(-2, index)
    common = valid_reflected & valid_original.index_select(-1, index)
    if not common.flatten(1).any(dim=1).all():
        raise ValueError("Every sequence needs at least one aligned valid token")
    # Values at invalid positions are excluded before arithmetic. This also
    # prevents NaN placeholders in a diagnostic export from affecting the result.
    expected = torch.where(common[..., None], expected, 0.0)
    observed = torch.where(common[..., None], reflected, 0.0)
    if not torch.isfinite(expected).all() or not torch.isfinite(observed).all():
        raise ValueError("Nonfinite features appear at valid token positions")
    return expected, observed, common


def token_reflection_error(
    original: torch.Tensor,
    reflected: torch.Tensor,
    valid_original: torch.Tensor,
    valid_reflected: torch.Tensor,
    *,
    energy_floor: float = 1e-12,
) -> torch.Tensor:
    """Per-sequence squared reflection mismatch normalized by token energy.

    Zero-energy sequences are rejected, not assigned a perfect score. A constant
    nonzero representation can still score zero; test feature variation too.
    """
    expected, observed, _ = _aligned_tokens(
        original, reflected, valid_original, valid_reflected
    )
    energy = (expected.square() + observed.square()).flatten(1).sum(dim=1)
    if (energy <= energy_floor).any():
        raise ValueError("A zero-energy representation cannot pass a symmetry test")
    difference = (observed - expected).square().flatten(1).sum(dim=1)
    return difference / energy


def token_equivariance_loss(
    original: torch.Tensor,
    reflected: torch.Tensor,
    valid_original: torch.Tensor,
    valid_reflected: torch.Tensor,
) -> torch.Tensor:
    """Average the normalized mismatch over sequences without fitting a map.

    The detached denominator prevents this additional term from rewarding a
    larger feature norm. This is the identity-channel action used by the current
    strict test, with supplied anatomical joint swapping. The training loss does
    not itself rule out constant nonzero features.
    """
    expected, observed, _ = _aligned_tokens(
        original, reflected, valid_original, valid_reflected
    )
    energy = (expected.square() + observed.square()).flatten(1).sum(dim=1)
    difference = (observed - expected).square().flatten(1).sum(dim=1)
    return (difference / energy.detach().clamp_min(1e-12)).mean()


def token_symmetry_penalty(
    encoder: torch.nn.Module,
    coordinates: torch.Tensor,
    valid_patch: torch.Tensor,
) -> torch.Tensor:
    """Shared-runner hook: two unmasked encoder forwards and a paired loss."""
    reflected, reflected_valid, _ = reflect_training_batch(coordinates, valid_patch)
    original_tokens = encoder(coordinates, valid_patch)
    reflected_tokens = encoder(reflected, reflected_valid)
    batch, patches, joints = valid_patch.shape
    original_tokens = original_tokens.reshape(batch, patches, joints, -1)
    reflected_tokens = reflected_tokens.reshape(batch, patches, joints, -1)
    return token_equivariance_loss(
        original_tokens, reflected_tokens, valid_patch, reflected_valid
    )


def feature_variation(features: np.ndarray, source_ids: np.ndarray) -> dict[str, Any]:
    """Source-weighted descriptive checks; none is a clinical validity test."""
    values = np.asarray(features, dtype=np.float64)
    groups = np.asarray(source_ids).astype(str)
    if values.ndim != 2 or values.shape[0] != len(groups) or len(groups) < 2:
        raise ValueError("At least two finite feature rows and matching source IDs are required")
    if not np.isfinite(values).all() or values.shape[1] < 1:
        raise ValueError("Features must be finite and have at least one channel")
    unique, counts = np.unique(groups, return_counts=True)
    count_by_source = dict(zip(unique, counts))
    weights = np.asarray([1.0 / count_by_source[group] for group in groups])
    weights /= weights.sum()
    centered = values - np.sum(weights[:, None] * values, axis=0)
    variances = np.sum(weights[:, None] * centered**2, axis=0)
    singular_values = np.linalg.svd(centered * np.sqrt(weights[:, None]), compute_uv=False)
    spectrum = singular_values**2
    total = float(spectrum.sum())
    nonzero = spectrum[spectrum > 0] / total if total > 1e-24 else np.asarray([])
    effective_rank = float(np.exp(-np.sum(nonzero * np.log(nonzero)))) if len(nonzero) else 0.0
    return {
        "source_count": int(len(unique)),
        "mean_channel_standard_deviation": float(np.sqrt(variances).mean()),
        "effective_rank": effective_rank,
        "constant_features": bool(total <= 1e-24),
        "interpretation": "Variation checks cannot establish useful prediction or clinical meaning.",
    }


@torch.no_grad()
def held_out_encoder_diagnostics(
    encoder: torch.nn.Module, dataset: Any, *, device: str = "cpu", batch_size: int = 8
) -> dict[str, float]:
    """Measure strict token mismatch on held-out sources without fitting a map."""
    from laterality.metrics import source_weights
    from laterality.model import valid_patches

    was_training = encoder.training
    encoder.eval()
    errors = []
    try:
        for start in range(0, len(dataset.test_rows), batch_size):
            rows = dataset.test_rows[start : start + batch_size]
            coordinates = torch.as_tensor(dataset.xyz[rows], dtype=torch.float32, device=device)
            valid = torch.as_tensor(dataset.valid[rows], dtype=torch.bool, device=device)
            patches = valid_patches(valid, encoder.segment_length)
            reflected, reflected_valid, _ = reflect_training_batch(coordinates, patches)
            shape = (*patches.shape, encoder.embed_dim)
            original_tokens = encoder(coordinates, patches).reshape(shape)
            reflected_tokens = encoder(reflected, reflected_valid).reshape(shape)
            errors.extend(token_reflection_error(
                original_tokens, reflected_tokens, patches, reflected_valid
            ).cpu().tolist())
    finally:
        encoder.train(was_training)
    weights = source_weights(dataset.source_ids[dataset.test_rows])
    return {"reflection_error": float(np.average(errors, weights=weights))}


def run_symmetry_comparison(dataset: Any, settings: Any, *, symmetry_weight: float = 1.0) -> Any:
    """Run matched update counts; the explicit penalty uses extra encoder work.

    Local-data authorization is handled by the shared data loader. This function
    does not load new data, mutate registered settings, or tune the penalty.
    """
    from .masked_learning import run_matched_comparison
    import pandas as pd

    if not np.isfinite(symmetry_weight) or symmetry_weight <= 0:
        raise ValueError("symmetry_weight must be a finite, positive planned setting")
    fixed = replace(settings, reflection_probability=0.0, symmetry_weight=0.0, mask_policy="gait")
    variants = {
        "base_objective": {"reflection_probability": 0.0, "symmetry_weight": 0.0},
        "mirrored_training": {"reflection_probability": 0.5, "symmetry_weight": 0.0},
        "explicit_reflection_loss": {"reflection_probability": 0.0, "symmetry_weight": float(symmetry_weight)},
    }
    comparison = run_matched_comparison(
        dataset, fixed, variants=variants, symmetry_penalty=token_symmetry_penalty
    )
    labels = {
        "base_objective": "Base objective",
        "mirrored_training": "Mirrored training clips",
        "explicit_reflection_loss": "Explicit reflection loss",
    }
    rows, diagnostics = [], []
    test = dataset.test_rows
    groups = dataset.source_ids[test]
    for variant, run in comparison["runs"].items():
        metric = run["metrics"].set_index("representation").loc["learned_encoder"]
        consistency = held_out_encoder_diagnostics(
            run["evaluated_encoder"], dataset, device=settings.device
        )
        variation = feature_variation(run["features"]["learned_encoder"][test], groups)
        rows.append({"Condition": labels[variant], "Held-out R²": metric["r2"],
                     "Held-out MAE": metric["mae"],
                     "Reflection error": consistency["reflection_error"],
                     "Feature std": variation["mean_channel_standard_deviation"],
                     "Effective rank": variation["effective_rank"]})
        diagnostics.append({"variant": variant, "representation": "learned_encoder",
                            **consistency, **variation})
    reference = comparison["runs"]["base_objective"]
    initial_consistency = held_out_encoder_diagnostics(
        reference["initial_encoder"], dataset, device=settings.device
    )
    initial_variation = feature_variation(reference["initial_features"][test], groups)
    diagnostics.append({"variant": "base_objective", "representation": "initial_encoder",
                        **initial_consistency, **initial_variation})
    for lane, label in (("initial_encoder", "Paired initial encoder"),
                        ("raw_pose", "Direct pose summaries"),
                        ("training_mean", "Training-source mean")):
        metric = reference["metrics"].set_index("representation").loc[lane]
        row = {"Condition": label, "Held-out R²": metric["r2"], "Held-out MAE": metric["mae"],
               "Reflection error": np.nan, "Feature std": np.nan, "Effective rank": np.nan}
        if lane == "initial_encoder":
            row.update({"Reflection error": initial_consistency["reflection_error"],
                        "Feature std": initial_variation["mean_channel_standard_deviation"],
                        "Effective rank": initial_variation["effective_rank"]})
        rows.append(row)
    comparison["summary"] = pd.DataFrame(rows)
    comparison["diagnostics"] = pd.DataFrame(diagnostics)
    comparison["compute_scope"] = "Matched optimizer updates; the reflection penalty adds two encoder forwards and gradients per update."
    return comparison
