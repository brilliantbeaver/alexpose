"""Explicit JEPA objective spaces, ragged losses, and branch diagnostics."""
from __future__ import annotations

import torch
from torch.nn import functional as F


def per_example_mean(values: torch.Tensor, valid: torch.Tensor):
    """Equal-example reduction, not a pooled-token loss favoring dense examples.

    Returns scalar loss, [B] per-example means (NaN if unsupported), and support.
    Entirely unsupported batches have a differentiable zero but must be skipped
    by the trainer; they are never logged as successful zero-error predictions.
    """
    if values.shape != valid.shape or values.ndim < 2:
        raise ValueError("values and validity must share [batch,...] shape")
    finite = torch.isfinite(values)
    if (valid & ~finite).any():
        raise FloatingPointError("Nonfinite supported loss")
    flat_valid = valid.reshape(len(values), -1).bool()
    counts = flat_valid.sum(1)
    safe = torch.where(valid, values, torch.zeros_like(values)).reshape(len(values), -1)
    means = safe.sum(1) / counts.clamp_min(1)
    supported = counts > 0
    scalar = means[supported].mean() if supported.any() else safe.sum() * 0
    return scalar, torch.where(supported, means, torch.full_like(means, float("nan"))), supported


def predictive_loss(predicted: torch.Tensor, target: torch.Tensor,
                    valid: torch.Tensor, *, objective: str,
                    center: torch.Tensor, student_temperature: float = .1,
                    teacher_temperature: float = .06):
    """CE acts on feature logits; regression acts on per-token LayerNorm space.

    `centered_ce_v1` matches the historical local centered/sharpened CE formula,
    not its entire architecture/data/training implementation. Regression is a
    separately named adaptation and is never called an exact S-JEPA replay.
    """
    if predicted.shape != target.shape or predicted.shape[:-1] != valid.shape:
        raise ValueError("Prediction, teacher and validity shapes disagree")
    target = target.detach().float()
    predicted = predicted.float()
    if objective == "centered_ce_v1":
        if min(student_temperature, teacher_temperature) <= 0:
            raise ValueError("Temperatures must be positive")
        probabilities = ((target - center.float()) / teacher_temperature).softmax(-1)
        errors = -(probabilities * (predicted / student_temperature).log_softmax(-1)).sum(-1)
    elif objective == "feature_regression_v1":
        dimension = (predicted.shape[-1],)
        errors = F.smooth_l1_loss(F.layer_norm(predicted, dimension),
                                  F.layer_norm(target, dimension), reduction="none").mean(-1)
    else:
        raise ValueError(f"Unknown objective {objective!r}")
    return per_example_mean(errors, valid.bool())


def vicreg_loss(first: torch.Tensor, second: torch.Tensor) -> torch.Tensor:
    """Historical local 25 invariance +25 variance +1 covariance convention.

    Applied to two small translated prepared-feature views. This augmentation
    is declared as part of the new 2D recipe, not a 3D geometric reproduction.
    A singleton batch cannot estimate useful between-example covariance.
    """
    if len(first) < 2:
        return (first.sum() + second.sum()) * 0
    first, second = first.float(), second.float()
    invariance = F.mse_loss(first, second)
    variance = .5 * (F.relu(1 - (first.var(0, unbiased=False) + 1e-4).sqrt()).mean()
                     + F.relu(1 - (second.var(0, unbiased=False) + 1e-4).sqrt()).mean())
    covariance = first.new_zeros(())
    for value in (first, second):
        centered = value - value.mean(0)
        cov = centered.T @ centered / (len(value) - 1)
        offdiag = cov - torch.diag_embed(torch.diagonal(cov))
        covariance = covariance + offdiag.square().sum() / (2 * value.shape[1])
    return 25 * invariance + 25 * variance + covariance


@torch.no_grad()
def feature_diagnostics(features: torch.Tensor, valid: torch.Tensor | None = None) -> dict:
    """Rank is a failure diagnostic, not a success criterion or uncertainty CI."""
    values = features.detach().float().reshape(-1, features.shape[-1])
    if valid is not None:
        values = values[valid.reshape(-1)]
    values = values[torch.isfinite(values).all(-1)]
    if len(values) == 0:
        return {"count": 0, "mean_std": None, "effective_rank": None}
    centered = values - values.mean(0)
    singular = torch.linalg.svdvals(centered)
    total = singular.sum()
    if total <= 1e-12:
        rank = 0.
    else:
        probabilities = singular / total
        rank = float((-(probabilities * probabilities.clamp_min(1e-12).log()).sum()).exp())
    return {"count": len(values), "mean_std": float(values.std(0, unbiased=False).mean()),
            "effective_rank": rank}
