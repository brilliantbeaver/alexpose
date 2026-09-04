"""Strict, identity-channel representation-equivariance diagnostics.

For one sequence, let ``z`` be ``E(x)`` and ``z_mirror`` be ``E(Mx)``.  The
expected representation under the registered group action is ``S z``, where
``S`` swaps only the anatomical joint axis and leaves every feature channel
unchanged.  On tokens valid in both views, this module reports

    ||z_mirror - S z||^2 / (||z_mirror||^2 + ||S z||^2).

The value is zero for exact equivariance and has expectation near one for
uncorrelated, equal-energy representations.  No channel fitting, sign search,
Procrustes alignment, centering, or read-out is permitted.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

import numpy as np

from .geometry import FULL_MIRROR_PAIRS


@dataclass(frozen=True)
class EquivarianceError:
    """Energy decomposition for one sequence's common-valid tokens."""

    value: float
    residual_energy: float
    representation_energy: float
    common_token_count: int
    channel_count: int


def _validated_pairs(
    pairs: Iterable[Sequence[int]], joint_count: int
) -> tuple[tuple[int, int], ...]:
    normalized: list[tuple[int, int]] = []
    used: set[int] = set()
    for raw_pair in pairs:
        if len(raw_pair) != 2:
            raise ValueError("Every joint-swap entry must contain exactly two indices")
        left, right = (int(raw_pair[0]), int(raw_pair[1]))
        if left == right:
            raise ValueError("A joint cannot be swapped with itself")
        if left < 0 or right < 0 or left >= joint_count or right >= joint_count:
            raise ValueError(
                f"Joint-swap pair {(left, right)} is outside [0, {joint_count - 1}]"
            )
        if left in used or right in used:
            raise ValueError("Joint-swap pairs must be disjoint")
        used.update((left, right))
        normalized.append((left, right))
    if not normalized:
        raise ValueError("At least one joint-swap pair is required")
    return tuple(normalized)


def swap_token_joints(
    tokens: np.ndarray,
    pairs: Iterable[Sequence[int]] = FULL_MIRROR_PAIRS,
) -> np.ndarray:
    """Apply ``S`` to ``[..., joint, channel]`` tokens; channels are untouched."""
    values = np.asarray(tokens)
    if values.ndim < 2:
        raise ValueError("Tokens must have at least joint and channel axes")
    if values.shape[-1] < 1 or values.shape[-2] < 1:
        raise ValueError("Joint and channel axes must be non-empty")
    normalized_pairs = _validated_pairs(pairs, values.shape[-2])
    swapped = values.copy()
    for left, right in normalized_pairs:
        swapped[..., [left, right], :] = values[..., [right, left], :]
    return swapped


def swap_token_validity(
    valid: np.ndarray,
    pairs: Iterable[Sequence[int]] = FULL_MIRROR_PAIRS,
) -> np.ndarray:
    """Apply the same joint permutation to a ``[..., joint]`` validity mask."""
    values = np.asarray(valid, dtype=bool)
    if values.ndim < 1 or values.shape[-1] < 1:
        raise ValueError("Validity must have a non-empty joint axis")
    normalized_pairs = _validated_pairs(pairs, values.shape[-1])
    swapped = values.copy()
    for left, right in normalized_pairs:
        swapped[..., [left, right]] = values[..., [right, left]]
    return swapped


def strict_equivariance_error(
    original_tokens: np.ndarray,
    mirrored_tokens: np.ndarray,
    original_valid: np.ndarray,
    mirrored_valid: np.ndarray,
    *,
    pairs: Iterable[Sequence[int]] = FULL_MIRROR_PAIRS,
    minimum_common_tokens: int = 1,
    minimum_representation_energy: float = 0.0,
) -> EquivarianceError:
    """Measure strict joint-swap equivariance for one sequence.

    Tokens must have shape ``[time_patch, joint, channel]`` and validity masks
    shape ``[time_patch, joint]``.  A token is compared only when the mirrored
    token and its joint-swapped original counterpart are both valid.  Values at
    every other token may be arbitrary or non-finite and cannot affect the
    result.

    A collapsed all-zero common representation is rejected rather than being
    credited with perfect equivariance.
    """
    original = np.asarray(original_tokens, dtype=np.float64)
    mirrored = np.asarray(mirrored_tokens, dtype=np.float64)
    if original.ndim != 3:
        raise ValueError("original_tokens must have shape [time_patch, joint, channel]")
    if mirrored.shape != original.shape:
        raise ValueError("original_tokens and mirrored_tokens must have identical shapes")
    if original.shape[0] < 1 or original.shape[1] < 1 or original.shape[2] < 1:
        raise ValueError("Token axes must be non-empty")

    valid_original = np.asarray(original_valid, dtype=bool)
    valid_mirrored = np.asarray(mirrored_valid, dtype=bool)
    expected_valid_shape = original.shape[:-1]
    if valid_original.shape != expected_valid_shape:
        raise ValueError(
            f"original_valid shape {valid_original.shape} does not match "
            f"{expected_valid_shape}"
        )
    if valid_mirrored.shape != expected_valid_shape:
        raise ValueError(
            f"mirrored_valid shape {valid_mirrored.shape} does not match "
            f"{expected_valid_shape}"
        )
    if isinstance(minimum_common_tokens, bool) or int(minimum_common_tokens) != minimum_common_tokens:
        raise ValueError("minimum_common_tokens must be a positive integer")
    minimum_count = int(minimum_common_tokens)
    if minimum_count < 1:
        raise ValueError("minimum_common_tokens must be a positive integer")
    energy_floor = float(minimum_representation_energy)
    if not np.isfinite(energy_floor) or energy_floor < 0.0:
        raise ValueError("minimum_representation_energy must be finite and non-negative")

    normalized_pairs = _validated_pairs(pairs, original.shape[-2])
    expected = swap_token_joints(original, normalized_pairs)
    expected_valid = swap_token_validity(valid_original, normalized_pairs)
    common = valid_mirrored & expected_valid
    common_count = int(common.sum())
    if common_count < minimum_count:
        raise ValueError(
            f"Only {common_count} common-valid tokens; need at least {minimum_count}"
        )

    observed_values = mirrored[common]
    expected_values = expected[common]
    if not np.isfinite(observed_values).all() or not np.isfinite(expected_values).all():
        raise ValueError("A common-valid representation token is non-finite")

    residual_energy = float(np.square(observed_values - expected_values).sum())
    representation_energy = float(
        np.square(observed_values).sum() + np.square(expected_values).sum()
    )
    if not np.isfinite(residual_energy) or not np.isfinite(representation_energy):
        raise FloatingPointError("Representation energy overflowed")
    if representation_energy <= energy_floor:
        raise ValueError(
            "Common-valid representation energy is too small for an equivariance ratio"
        )
    return EquivarianceError(
        value=residual_energy / representation_energy,
        residual_energy=residual_energy,
        representation_energy=representation_energy,
        common_token_count=common_count,
        channel_count=int(original.shape[-1]),
    )


def strict_equivariance_errors(
    original_tokens: np.ndarray,
    mirrored_tokens: np.ndarray,
    original_valid: np.ndarray,
    mirrored_valid: np.ndarray,
    **kwargs,
) -> tuple[EquivarianceError, ...]:
    """Apply :func:`strict_equivariance_error` to a leading sequence axis."""
    original = np.asarray(original_tokens)
    mirrored = np.asarray(mirrored_tokens)
    valid_original = np.asarray(original_valid)
    valid_mirrored = np.asarray(mirrored_valid)
    if original.ndim != 4:
        raise ValueError(
            "Batched original_tokens must have shape [sequence, time_patch, joint, channel]"
        )
    if mirrored.shape != original.shape:
        raise ValueError("Batched original and mirrored token shapes differ")
    if valid_original.shape != original.shape[:-1] or valid_mirrored.shape != original.shape[:-1]:
        raise ValueError("Batched validity shapes must match token shapes without channels")
    return tuple(
        strict_equivariance_error(
            original[index],
            mirrored[index],
            valid_original[index],
            valid_mirrored[index],
            **kwargs,
        )
        for index in range(len(original))
    )


def source_balanced_mean(errors: Sequence[float], source_ids: Sequence[object]) -> float:
    """Average sequence errors within source, then give every source equal weight."""
    values = np.asarray(errors, dtype=np.float64)
    sources = np.asarray(source_ids).astype(str)
    if values.ndim != 1 or sources.ndim != 1 or len(values) != len(sources):
        raise ValueError("errors and source_ids must be aligned one-dimensional arrays")
    if not len(values):
        raise ValueError("At least one sequence error is required")
    if not np.isfinite(values).all():
        raise ValueError("Sequence equivariance errors must be finite")
    if np.any(np.char.str_len(sources) == 0):
        raise ValueError("Source identifiers must be non-empty")
    per_source = [values[sources == source].mean() for source in np.unique(sources)]
    return float(np.mean(per_source))


def per_seed_source_balanced_mean(
    errors: Sequence[float],
    source_ids: Sequence[object],
    seed_ids: Sequence[object],
    *,
    require_same_sources: bool = True,
) -> dict[str, float]:
    """Return source-balanced sequence-error means separately for every seed."""
    values = np.asarray(errors, dtype=np.float64)
    sources = np.asarray(source_ids).astype(str)
    seeds = np.asarray(seed_ids).astype(str)
    if (
        values.ndim != 1
        or sources.ndim != 1
        or seeds.ndim != 1
        or len(values) != len(sources)
        or len(values) != len(seeds)
    ):
        raise ValueError("errors, source_ids, and seed_ids must be aligned 1-D arrays")
    if not len(values):
        raise ValueError("At least one sequence error is required")
    unique_seeds = np.unique(seeds)
    if require_same_sources:
        reference = set(sources[seeds == unique_seeds[0]])
        for seed in unique_seeds[1:]:
            if set(sources[seeds == seed]) != reference:
                raise ValueError("Every seed must cover the same source identifiers")
    return {
        str(seed): source_balanced_mean(values[seeds == seed], sources[seeds == seed])
        for seed in unique_seeds
    }


__all__ = [
    "EquivarianceError",
    "per_seed_source_balanced_mean",
    "source_balanced_mean",
    "strict_equivariance_error",
    "strict_equivariance_errors",
    "swap_token_joints",
    "swap_token_validity",
]
