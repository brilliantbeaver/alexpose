"""Uniform observed-token masking; True always means a prediction target."""
from __future__ import annotations

import torch

JOINT_IDS = (11, 12, 13, 14, 15, 16, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32)


def _validate(valid: torch.Tensor) -> None:
    if valid.ndim != 3 or valid.shape[-1] != len(JOINT_IDS):
        raise ValueError("valid must be [batch,time,16] in the declared joint order")
    if valid.dtype != torch.bool:
        raise ValueError("valid must be boolean")


def uniform_mask(valid: torch.Tensor, ratio: float = 0.5,
                 generator: torch.Generator | None = None) -> torch.Tensor:
    """Sample without replacement among valid tokens, never by motion magnitude.

    At least one context and one target token are retained per sequence. Missing
    keypoints are neither visible context nor prediction targets.
    """
    _validate(valid)
    if not 0 < ratio < 1:
        raise ValueError("ratio must be strictly between zero and one")
    target = torch.zeros_like(valid)
    for i, row in enumerate(valid.flatten(1)):
        indices = row.nonzero(as_tuple=True)[0]
        if len(indices) < 2:
            raise ValueError("Each sequence needs at least two observed tokens")
        count = max(1, min(len(indices) - 1, round(len(indices) * ratio)))
        random_device = generator.device if generator is not None else valid.device
        order = torch.randperm(len(indices), generator=generator, device=random_device).to(valid.device)
        target[i].view(-1)[indices[order[:count]]] = True
    return target


def forecast_mask(valid: torch.Tensor, context_frames: int) -> torch.Tensor:
    """Hide every observed token after a prefix. No future context is retained."""
    _validate(valid)
    if not 0 < context_frames < valid.shape[1]:
        raise ValueError("context_frames must leave both context and future frames")
    target = valid.clone()
    target[:, :context_frames] = False
    if not target.flatten(1).any(1).all() or not valid[:, :context_frames].flatten(1).any(1).all():
        raise ValueError("Every forecast needs observed prefix and future tokens")
    return target
