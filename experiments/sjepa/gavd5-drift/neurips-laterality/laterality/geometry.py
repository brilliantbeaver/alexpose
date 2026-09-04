from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

import numpy as np


FULL_MIRROR_PAIRS: tuple[tuple[int, int], ...] = (
    (1, 4),
    (2, 5),
    (3, 6),
    (7, 8),
    (9, 10),
    (11, 12),
    (13, 14),
    (15, 16),
    (17, 18),
    (19, 20),
    (21, 22),
    (23, 24),
    (25, 26),
    (27, 28),
    (29, 30),
    (31, 32),
)


@dataclass(frozen=True)
class TargetResult:
    value: float
    pair_contrasts: tuple[float, ...]
    common_transition_counts: tuple[int, ...]
    usable_pair_count: int


@dataclass(frozen=True)
class PreparedPose:
    """Separate model tensors from the untouched, validity-aware target lane."""

    model_xyz: np.ndarray
    model_valid: np.ndarray
    target_xyz: np.ndarray
    target_valid: np.ndarray
    frame_times: np.ndarray
    observed: np.ndarray
    body_scale: float


def anatomical_mirror(
    xyz: np.ndarray,
    valid: np.ndarray | None = None,
) -> np.ndarray | tuple[np.ndarray, np.ndarray]:
    coordinates = np.asarray(xyz)
    if coordinates.ndim < 3 or coordinates.shape[-2:] != (33, 3):
        raise ValueError(f"Expected [..., 33, 3], received {coordinates.shape}")
    mirrored = coordinates.copy()
    mirrored[..., 0] *= -1
    for left, right in FULL_MIRROR_PAIRS:
        mirrored[..., [left, right], :] = mirrored[..., [right, left], :]
    if valid is None:
        return mirrored
    validity = np.asarray(valid, dtype=bool)
    if validity.shape != coordinates.shape[:-1]:
        raise ValueError(
            f"Validity shape {validity.shape} does not match {coordinates.shape[:-1]}"
        )
    mirrored_valid = validity.copy()
    for left, right in FULL_MIRROR_PAIRS:
        mirrored_valid[..., [left, right]] = mirrored_valid[..., [right, left]]
    return mirrored, mirrored_valid


def observed_mask(raw_sequence: np.ndarray, visibility_threshold: float) -> np.ndarray:
    raw = np.asarray(raw_sequence)
    if raw.ndim != 3 or raw.shape[1:] != (33, 4):
        raise ValueError(f"Expected [T, 33, 4], received {raw.shape}")
    visibility = np.nan_to_num(raw[..., 3], nan=0.0, posinf=0.0, neginf=0.0)
    finite = np.isfinite(raw[..., :3]).all(axis=-1)
    return finite & (visibility >= visibility_threshold)


def interpolate_short_gaps(
    xyz: np.ndarray,
    observed: np.ndarray,
    max_gap: int,
) -> tuple[np.ndarray, np.ndarray]:
    coordinates = np.asarray(xyz, dtype=np.float64).copy()
    valid = np.asarray(observed, dtype=bool)
    if coordinates.shape[:-1] != valid.shape:
        raise ValueError("Coordinate and validity shapes disagree")
    filled = valid.copy()
    coordinates[~valid] = np.nan
    for joint in range(coordinates.shape[1]):
        endpoints = np.flatnonzero(valid[:, joint])
        for left, right in zip(endpoints[:-1], endpoints[1:]):
            gap = int(right - left - 1)
            if not 0 < gap <= max_gap:
                continue
            fraction = np.arange(1, gap + 1, dtype=np.float64)[:, None] / (gap + 1)
            coordinates[left + 1 : right, joint] = (
                coordinates[left, joint][None, :] * (1.0 - fraction)
                + coordinates[right, joint][None, :] * fraction
            )
            filled[left + 1 : right, joint] = True
    coordinates[~filled] = np.nan
    return coordinates, filled


def _bilateral_scale(xyz: np.ndarray, valid: np.ndarray, epsilon: float = 1e-8) -> float:
    candidates: list[np.ndarray] = []
    for left, right in ((11, 12), (23, 24)):
        ok = valid[:, left] & valid[:, right]
        if ok.any():
            candidates.append(np.linalg.norm(xyz[ok, left] - xyz[ok, right], axis=-1))
    if not candidates:
        return 1.0
    scale = float(np.nanmedian(np.concatenate(candidates)))
    return scale if np.isfinite(scale) and scale > epsilon else 1.0


def pelvis_normalize(
    xyz: np.ndarray,
    valid: np.ndarray,
    *,
    allow_fallback: bool,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Normalize without ever using a sentinel as a coordinate observation."""
    coordinates = np.asarray(xyz, dtype=np.float64).copy()
    validity = np.asarray(valid, dtype=bool).copy()
    if coordinates.shape[:-1] != validity.shape:
        raise ValueError("Coordinate and validity shapes disagree")

    pelvis_ok = validity[:, 23] & validity[:, 24]
    pelvis = np.full((len(coordinates), 3), np.nan, dtype=np.float64)
    pelvis[pelvis_ok] = 0.5 * (
        coordinates[pelvis_ok, 23] + coordinates[pelvis_ok, 24]
    )
    if allow_fallback and pelvis_ok.any():
        fallback = np.nanmedian(pelvis[pelvis_ok], axis=0)
        pelvis[~pelvis_ok] = fallback
    elif allow_fallback:
        pelvis[:] = 0.0

    normalized = coordinates - pelvis[:, None, :]
    normalized_valid = validity & np.isfinite(pelvis).all(axis=-1)[:, None]
    scale = _bilateral_scale(coordinates, validity)
    normalized /= scale
    normalized[~normalized_valid] = np.nan
    return normalized, normalized_valid, scale


def temporal_resize(array: np.ndarray, frames: int) -> np.ndarray:
    values = np.asarray(array)
    if len(values) == frames:
        return values.copy()
    if len(values) < 1:
        raise ValueError("Cannot resize an empty sequence")
    if len(values) == 1:
        return np.repeat(values, frames, axis=0)
    old_time = np.linspace(0.0, 1.0, len(values))
    new_time = np.linspace(0.0, 1.0, frames)
    flat = values.reshape(len(values), -1)
    resized = np.stack(
        [np.interp(new_time, old_time, flat[:, column]) for column in range(flat.shape[1])],
        axis=1,
    )
    return resized.reshape(frames, *values.shape[1:])


def prepare_pose(
    raw_sequence: np.ndarray,
    frame_numbers: np.ndarray,
    fps: float,
    *,
    frames: int,
    visibility_threshold: float,
    max_interpolation_gap: int,
) -> PreparedPose:
    raw = np.asarray(raw_sequence, dtype=np.float64)
    frame_numbers = np.asarray(frame_numbers, dtype=np.float64)
    if frame_numbers.shape != (len(raw),):
        raise ValueError("frame_numbers must have one entry per pose frame")
    if not np.isfinite(fps) or fps <= 0:
        raise ValueError("fps must be positive")

    observed = observed_mask(raw, visibility_threshold)
    raw_xyz = raw[..., :3].copy()
    raw_xyz[~observed] = np.nan

    # This lane is never zero-filled or naively resized. It defines the target.
    target_xyz, target_valid, target_scale = pelvis_normalize(
        raw_xyz, observed, allow_fallback=False
    )

    # Interpolation exists only for model input. Validity travels alongside it.
    filled_xyz, filled_valid = interpolate_short_gaps(
        raw_xyz, observed, max_interpolation_gap
    )
    model_normalized, model_valid, _ = pelvis_normalize(
        filled_xyz, filled_valid, allow_fallback=True
    )
    model_xyz = temporal_resize(np.nan_to_num(model_normalized, nan=0.0), frames)
    model_valid_resized = temporal_resize(model_valid.astype(np.float64), frames) >= 0.999
    model_xyz[~model_valid_resized] = 0.0
    return PreparedPose(
        model_xyz=model_xyz.astype(np.float32),
        model_valid=model_valid_resized.astype(bool),
        target_xyz=target_xyz,
        target_valid=target_valid,
        frame_times=(frame_numbers - frame_numbers[0]) / float(fps),
        observed=observed,
        body_scale=float(target_scale),
    )


def paired_valid_target(
    xyz: np.ndarray,
    valid: np.ndarray,
    pairs: Sequence[Sequence[int]],
    minimum_common_transitions_per_pair: int,
    minimum_usable_pairs: int,
    epsilon: float,
    frame_times: np.ndarray | None = None,
) -> TargetResult:
    coordinates = np.asarray(xyz, dtype=np.float64)
    validity = np.asarray(valid, dtype=bool)
    if coordinates.ndim != 3 or coordinates.shape[-1] != 3:
        raise ValueError("xyz must have shape [T, J, 3]")
    if validity.shape != coordinates.shape[:-1]:
        raise ValueError("valid must have shape [T, J]")
    if frame_times is None:
        times = np.arange(len(coordinates), dtype=np.float64)
    else:
        times = np.asarray(frame_times, dtype=np.float64)
        if times.shape != (len(coordinates),):
            raise ValueError("frame_times must have shape [T]")
    dt = np.diff(times)
    finite_dt = np.isfinite(dt) & (dt > 0)

    contrasts: list[float] = []
    counts: list[int] = []
    for left, right in pairs:
        common = (
            validity[:-1, left]
            & validity[1:, left]
            & validity[:-1, right]
            & validity[1:, right]
            & finite_dt
        )
        counts.append(int(common.sum()))
        if common.sum() < minimum_common_transitions_per_pair:
            contrasts.append(float("nan"))
            continue
        left_speed = np.linalg.norm(
            coordinates[1:, left] - coordinates[:-1, left], axis=-1
        ) / dt
        right_speed = np.linalg.norm(
            coordinates[1:, right] - coordinates[:-1, right], axis=-1
        ) / dt
        left_motion = float(np.median(left_speed[common]))
        right_motion = float(np.median(right_speed[common]))
        contrast = (left_motion - right_motion) / (left_motion + right_motion + epsilon)
        contrasts.append(float(contrast))

    usable = np.isfinite(contrasts)
    usable_count = int(usable.sum())
    value = float(np.mean(np.asarray(contrasts)[usable])) if usable_count >= minimum_usable_pairs else float("nan")
    return TargetResult(value, tuple(contrasts), tuple(counts), usable_count)


def missingness_feature(valid: np.ndarray, pairs: Iterable[Sequence[int]]) -> np.ndarray:
    fraction = np.asarray(valid, dtype=np.float64).mean(axis=0)
    features: list[float] = []
    for left, right in pairs:
        features.extend((fraction[left] - fraction[right], fraction[left] + fraction[right]))
    return np.asarray(features, dtype=np.float64)
