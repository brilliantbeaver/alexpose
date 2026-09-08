"""Label-blind masks for the comparative tutorials, without changing old runs.

One sample is [time, 33 landmarks, xyz], and one target token contains four
successive observations of one landmark. A deliberate mask is always a subset
of valid tokens. Structured masks retain their declared shape even when that
makes a requested count infeasible. Counts may differ between samples; callers
must preserve their identities when selecting targets for a loss.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from typing import Any, Sequence

import numpy as np

from laterality.geometry import FULL_MIRROR_PAIRS


GAIT_JOINTS = (11, 12, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32)
POLICY_NAMES = (
    "gait", "uniform", "random_subset", "soft_gait", "motion",
    "whole_trajectory", "connected_region", "temporal_gap",
)
SCATTERED_POLICIES = frozenset(POLICY_NAMES[:5])
LEFT_JOINTS = frozenset(left for left, _ in FULL_MIRROR_PAIRS)
RIGHT_JOINTS = frozenset(right for _, right in FULL_MIRROR_PAIRS)
LANDMARK_NAMES = (
    "nose", "left eye inner", "left eye", "left eye outer", "right eye inner",
    "right eye", "right eye outer", "left ear", "right ear", "left mouth",
    "right mouth", "left shoulder", "right shoulder", "left elbow", "right elbow",
    "left wrist", "right wrist", "left little finger", "right little finger",
    "left index finger", "right index finger", "left thumb", "right thumb",
    "left hip", "right hip", "left knee", "right knee", "left ankle", "right ankle",
    "left heel", "right heel", "left foot tip", "right foot tip",
)

# Anatomical adjacency, not consecutive storage indices. The face-to-mouth and
# ear-to-shoulder links explicitly connect the head with the body. They are
# declared masking connections, not measured skeletal bones or new landmarks.
ANATOMICAL_EDGES = (
    (0, 1), (1, 2), (2, 3), (3, 7), (0, 4), (4, 5), (5, 6), (6, 8),
    (0, 9), (0, 10), (9, 10), (7, 11), (8, 12), (11, 12),
    (11, 13), (13, 15), (15, 17), (15, 19), (15, 21), (17, 19),
    (12, 14), (14, 16), (16, 18), (16, 20), (16, 22), (18, 20),
    (11, 23), (12, 24), (23, 24), (23, 25), (24, 26), (25, 27), (26, 28),
    (27, 29), (28, 30), (29, 31), (30, 32), (27, 31), (28, 32),
)


class InfeasibleMaskBudget(ValueError):
    """The requested intact mask cannot preserve any valid context."""


@dataclass(frozen=True)
class MaskPolicy:
    name: str
    eligibility: str = "all"
    subset_seed: int = 31
    subset_kind: str = "individual"
    gait_weight: float = 0.5
    motion_weight: float = 0.75
    motion_clip_quantile: float = 0.95

    def validate(self) -> None:
        if self.name not in POLICY_NAMES:
            raise ValueError(f"Unknown mask policy {self.name!r}; expected {POLICY_NAMES}")
        if self.eligibility not in {"all", "gait", "random_subset"}:
            raise ValueError("Eligibility must be all, gait, or random_subset")
        if self.subset_kind not in {"individual", "bilateral_pairs"}:
            raise ValueError("A fixed subset must use individual landmarks or bilateral_pairs")
        if not isinstance(self.subset_seed, (int, np.integer)) or self.subset_seed < 0:
            raise ValueError("Subset selection seed must be a nonnegative integer")
        if not 0 <= self.gait_weight < 1 or not 0 <= self.motion_weight < 1:
            raise ValueError("Mixture weights must lie in [0, 1), retaining uniform coverage")
        if not 0.5 <= self.motion_clip_quantile <= 1:
            raise ValueError("Motion clipping quantile must lie in [0.5, 1]")
        if self.name in {"uniform", "soft_gait", "temporal_gap"} and self.eligibility != "all":
            raise ValueError(f"{self.name} is defined over all landmarks; use eligibility='all'")
        if self.name == "gait" and self.eligibility not in {"all", "gait"}:
            raise ValueError("The gait policy always uses the fixed gait pool")
        if self.name == "random_subset" and self.eligibility not in {"all", "random_subset"}:
            raise ValueError("The random_subset policy always uses its recorded fixed pool")


@dataclass(frozen=True)
class MaskBudget:
    hidden_count: int | None = None
    trajectories: int | None = None
    region_size: int | None = None
    interval_blocks: int | None = None

    def validate(self, policy: MaskPolicy) -> None:
        for name in ("hidden_count", "trajectories", "region_size", "interval_blocks"):
            value = getattr(self, name)
            if value is not None and (
                not isinstance(value, (int, np.integer)) or isinstance(value, bool) or value < 1
            ):
                raise ValueError(f"{name} must be a positive integer when supplied")
        if policy.name in SCATTERED_POLICIES:
            if self.hidden_count is None:
                raise ValueError("Scattered masking requires an explicit hidden_count")
            if any(value is not None for value in (self.trajectories, self.region_size, self.interval_blocks)):
                raise ValueError("Scattered masking has no trajectory, region, or interval size")
        elif policy.name == "whole_trajectory":
            if self.hidden_count is None and self.trajectories is None:
                raise ValueError("Specify a whole-trajectory count or an exact hidden-token count")
            if self.region_size is not None or self.interval_blocks is not None:
                raise ValueError("Whole trajectories always span the complete input")
        elif policy.name == "connected_region":
            if self.region_size is None or self.interval_blocks is None:
                raise ValueError("Connected regions require region_size and interval_blocks")
            if self.trajectories is not None:
                raise ValueError("Connected-region budgets do not use trajectories")
        elif policy.name == "temporal_gap":
            if self.interval_blocks is None:
                raise ValueError("A temporal gap requires an explicit interval_blocks")
            if self.trajectories is not None or self.region_size is not None:
                raise ValueError("A temporal gap hides all valid landmarks in its interval")


@dataclass(frozen=True)
class MaskResult:
    mask: np.ndarray
    target_indices: np.ndarray
    coverage: dict[str, Any]

    @property
    def context_count(self) -> int:
        return int(self.coverage["context_tokens"])


def fixed_subset(policy: MaskPolicy) -> tuple[int, ...]:
    """Select once with its own seed, independent of every mask-draw generator."""
    policy.validate()
    rng = np.random.default_rng(policy.subset_seed)
    if policy.subset_kind == "individual":
        chosen = rng.choice(33, size=12, replace=False)
    else:
        pairs = rng.choice(len(FULL_MIRROR_PAIRS), size=6, replace=False)
        chosen = [joint for index in pairs for joint in FULL_MIRROR_PAIRS[int(index)]]
    return tuple(sorted(map(int, chosen)))


def eligible_landmarks(policy: MaskPolicy) -> tuple[int, ...]:
    policy.validate()
    if policy.name == "gait" or policy.eligibility == "gait":
        return GAIT_JOINTS
    if policy.name == "random_subset" or policy.eligibility == "random_subset":
        return fixed_subset(policy)
    return tuple(range(33))


def _eligible_tokens(valid: np.ndarray, policy: MaskPolicy) -> np.ndarray:
    eligible = np.zeros_like(valid)
    eligible[:, eligible_landmarks(policy)] = valid[:, eligible_landmarks(policy)]
    return eligible


def shared_scattered_count(
    token_valid: np.ndarray, policies: Sequence[MaskPolicy], requested_count: int,
) -> int:
    """Declare the largest common count up to a requested count before training.

The returned count can be smaller than requested; callers must display and
record that new budget and rerun references at it. This helper never changes a
structured mask. It accepts one clip or a batch when checking feasibility.
"""
    valid = np.asarray(token_valid)
    if valid.dtype != np.dtype(bool) or valid.ndim not in {2, 3} or valid.shape[-1] != 33:
        raise ValueError("Token validity must be boolean [blocks, 33] or [batch, blocks, 33]")
    if not policies or not isinstance(requested_count, (int, np.integer)) or requested_count < 1:
        raise ValueError("Provide policies and a positive requested count")
    samples = valid[None] if valid.ndim == 2 else valid
    feasible = int(requested_count)
    for policy in policies:
        policy.validate()
        if policy.name not in SCATTERED_POLICIES:
            raise ValueError("This helper matches scattered policies only")
        for sample in samples:
            feasible = min(feasible, int(sample.sum()) - 1, int(_eligible_tokens(sample, policy).sum()))
    if feasible < 1:
        raise InfeasibleMaskBudget("No shared hidden target can retain valid context")
    return feasible


def _validate_input(
    xyz: np.ndarray, token_valid: np.ndarray, segment_length: int,
    observation_valid: np.ndarray | None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    coordinates, valid = np.asarray(xyz, dtype=float), np.asarray(token_valid)
    if not isinstance(segment_length, (int, np.integer)) or segment_length < 1:
        raise ValueError("Segment length must be a positive integer")
    if coordinates.ndim != 3 or coordinates.shape[1:] != (33, 3):
        raise ValueError("Coordinates must have shape [time, 33, 3]")
    if valid.dtype != np.dtype(bool) or valid.shape != (len(coordinates) // segment_length, 33):
        raise ValueError("Token validity must be boolean [time / segment_length, 33]")
    if not len(coordinates) or len(coordinates) % segment_length:
        raise ValueError("Input length must be nonzero and divisible by segment length")
    if observation_valid is None:
        observed = np.repeat(valid, segment_length, axis=0)
    else:
        observed = np.asarray(observation_valid)
        if observed.dtype != np.dtype(bool) or observed.shape != coordinates.shape[:-1]:
            raise ValueError("Observation validity must be boolean [time, 33]")
        expected = observed.reshape(-1, segment_length, 33).all(axis=1)
        if not np.array_equal(expected, valid):
            raise ValueError("Token validity must require every constituent observation")
    if not np.isfinite(coordinates[observed]).all():
        raise ValueError("Observed coordinates must be finite")
    if valid.sum() < 2:
        raise InfeasibleMaskBudget("At least two valid tokens are needed for a target and context")
    return coordinates, valid, observed


def motion_scores(
    xyz: np.ndarray, token_valid: np.ndarray, *, segment_length: int = 4,
    timestamps: np.ndarray | None = None, observation_valid: np.ndarray | None = None,
    clip_quantile: float = 0.95,
) -> tuple[np.ndarray, dict[str, Any]]:
    """A declared motion-sampling adaptation, not an exact MAMP reproduction.

Displacement spans one token length, as in MAMP's masking motion. We divide by
elapsed time, take the median valid transition speed within a token, and cap
positive token scores at a declared quantile. The median controls isolated
one-frame jumps. At least two valid transitions are needed per token. The first
block reuses the second block's score where both blocks are valid. With no
timestamps, speed is in prepared coordinate units per prepared time step.
"""
    coordinates, valid, observed = _validate_input(xyz, token_valid, segment_length, observation_valid)
    if not 0.5 <= clip_quantile <= 1:
        raise ValueError("Motion clipping quantile must lie in [0.5, 1]")
    times = np.arange(len(coordinates), dtype=float) if timestamps is None else np.asarray(timestamps, float)
    if times.shape != (len(coordinates),) or not np.isfinite(times).all() or np.any(np.diff(times) <= 0):
        raise ValueError("Timestamps must be finite, increasing, and match the permitted input")
    scores = np.zeros_like(valid, dtype=float)
    supported = np.zeros_like(valid, dtype=int)
    stride = segment_length
    if len(valid) > 1:
        available = observed[stride:] & observed[:-stride]
        safe_coordinates = np.where(observed[..., None], coordinates, 0.0)
        displacement = safe_coordinates[stride:] - safe_coordinates[:-stride]
        speed = np.linalg.norm(displacement, axis=-1) / (times[stride:] - times[:-stride])[:, None]
        available &= np.isfinite(speed)
        speed = np.where(available, speed, np.nan).reshape(-1, stride, 33)
        supported[1:] = available.reshape(-1, stride, 33).sum(axis=1)
        # Explicit per-column reduction avoids warnings for missing columns and
        # has at most 33 iterations, independent of the number of time blocks.
        for joint in range(33):
            usable = (supported[1:, joint] >= 2) & valid[1:, joint]
            if usable.any():
                scores[1:, joint][usable] = np.nanmedian(speed[usable, :, joint], axis=1)
    if len(valid) > 1:
        first_supported = valid[0] & valid[1]
        scores[0, first_supported] = scores[1, first_supported]
        supported[0, first_supported] = supported[1, first_supported]
    positive = scores[(scores > 0) & np.isfinite(scores)]
    cap = float(np.quantile(positive, clip_quantile)) if len(positive) else 0.0
    scores = np.minimum(np.where(np.isfinite(scores), scores, 0.0), cap)
    return scores, {
        "motion_units": "coordinate units / second" if timestamps is not None else "coordinate units / prepared step",
        "motion_stride_steps": int(stride), "motion_clip_quantile": float(clip_quantile),
        "motion_score_cap": cap, "motion_supported_tokens": int(((supported >= 2) & valid).sum()),
        "motion_fallback_uniform": not bool(np.any(scores > 0)),
    }


def _mirror_ids(indices: Sequence[int]) -> tuple[int, ...]:
    permutation = np.arange(33)
    for left, right in FULL_MIRROR_PAIRS:
        permutation[left], permutation[right] = right, left
    return tuple(sorted(map(int, permutation[list(indices)])))


@lru_cache(maxsize=128)
def connected_region_bank(landmarks: tuple[int, ...], region_size: int) -> tuple[tuple[int, ...], ...]:
    """A fixed, fully declared bank of connected regions covering the pool.

Grow breadth-first from every eligible landmark using anatomical edges and
sorted neighboring identities, then include valid reflected counterparts.
This is a restricted bank, not uniform sampling of all connected subgraphs.
"""
    allowed = set(landmarks)
    if not allowed or any(j < 0 or j >= 33 for j in allowed) or len(allowed) != len(landmarks):
        raise ValueError("Region landmarks must be distinct valid identities")
    if not isinstance(region_size, (int, np.integer)) or not 1 <= region_size <= len(allowed):
        raise InfeasibleMaskBudget("Region size exceeds the declared candidate pool")
    graph = {joint: set() for joint in allowed}
    for left, right in ANATOMICAL_EDGES:
        if left in allowed and right in allowed:
            graph[left].add(right)
            graph[right].add(left)
    regions = set()
    for root in sorted(allowed):
        selected, queue = [], [root]
        while queue and len(selected) < region_size:
            joint = queue.pop(0)
            if joint in selected:
                continue
            selected.append(joint)
            queue.extend(sorted(graph[joint] - set(selected)))
        if len(selected) == region_size:
            region = tuple(sorted(selected))
            regions.add(region)
            reflected = _mirror_ids(region)
            if set(reflected) <= allowed:
                regions.add(reflected)
    covered = set().union(*map(set, regions)) if regions else set()
    if covered != allowed:
        raise InfeasibleMaskBudget("No connected-region bank of this size covers every eligible landmark")
    return tuple(sorted(regions))


def _whole_trajectory_mask(valid, eligible, budget, rng):
    joints = np.flatnonzero(eligible.any(axis=0))
    counts = eligible.sum(axis=0)
    if budget.trajectories is not None and budget.trajectories > len(joints):
        raise InfeasibleMaskBudget("Fewer observed eligible trajectories than requested")
    if budget.hidden_count is None:
        selected = rng.choice(joints, size=budget.trajectories, replace=False)
    else:
        # Dynamic programming retains intact trajectories and finds an exact
        # feasible subset. Randomized search order avoids a fixed side priority.
        states = {(0, 0): ()}
        for joint in rng.permutation(joints):
            for (size, total), subset in list(states.items()):
                new_size, new_total = size + 1, total + int(counts[joint])
                if new_total > budget.hidden_count:
                    continue
                if budget.trajectories is not None and new_size > budget.trajectories:
                    continue
                states.setdefault((new_size, new_total), subset + (int(joint),))
        options = [subset for (size, total), subset in states.items()
                   if total == budget.hidden_count and (budget.trajectories is None or size == budget.trajectories)]
        if not options:
            raise InfeasibleMaskBudget("Exact count cannot be formed from intact eligible trajectories")
        selected = options[int(rng.integers(len(options)))]
    mask = np.zeros_like(valid)
    mask[:, list(selected)] = eligible[:, list(selected)]
    return mask, {"selected_landmarks": list(map(int, selected)), "interval_start": 0, "interval_blocks": len(valid)}


def _interval_mask(valid, eligible, policy, budget, rng):
    length = budget.interval_blocks
    if policy.name == "temporal_gap":
        regions = (tuple(range(33)),)
        starts = range(1, len(valid) - length)
    else:
        regions = connected_region_bank(eligible_landmarks(policy), budget.region_size)
        starts = range(0, len(valid) - length + 1)
    candidates = []
    feasible_counts = set()
    for start in starts:
        if policy.name == "temporal_gap" and (not valid[:start].any() or not valid[start + length:].any()):
            continue
        for region in regions:
            mask = np.zeros_like(valid)
            mask[start:start + length, list(region)] = eligible[start:start + length, list(region)]
            count = int(mask.sum())
            if not 0 < count < int(valid.sum()):
                continue
            # Every nominal region landmark must have a supervised observation;
            # missing cells within its interval remain naturally missing.
            if policy.name == "connected_region" and not mask[:, list(region)].any(axis=0).all():
                continue
            feasible_counts.add(count)
            if budget.hidden_count is None or count == budget.hidden_count:
                candidates.append((mask, start, region))
    if not candidates:
        raise InfeasibleMaskBudget(
            f"No intact {policy.name} satisfies the budget; feasible observed counts: {sorted(feasible_counts)}"
        )
    mask, start, region = candidates[int(rng.integers(len(candidates)))]
    return mask, {"selected_landmarks": list(region), "interval_start": int(start), "interval_blocks": int(length)}


def coverage_summary(mask: np.ndarray, token_valid: np.ndarray, *, segment_length: int = 4) -> dict[str, Any]:
    hidden, valid = np.asarray(mask), np.asarray(token_valid)
    if (hidden.dtype != np.dtype(bool) or valid.dtype != np.dtype(bool)
            or hidden.shape != valid.shape or valid.ndim != 2 or valid.shape[-1] != 33):
        raise ValueError("Mask and validity must be boolean arrays with equal shapes")
    if np.any(hidden & ~valid):
        raise ValueError("Naturally missing measurements cannot become supervised targets")
    count, available = int(hidden.sum()), int(valid.sum())
    if not 0 < count < available:
        raise InfeasibleMaskBudget("A mask needs at least one hidden target and one valid context token")
    positions = np.argwhere(hidden)
    joints = sorted(map(int, np.flatnonzero(hidden.any(axis=0))))
    longest = 0
    for joint in joints:
        padded = np.r_[False, hidden[:, joint], False].astype(int)
        starts, ends = np.flatnonzero(np.diff(padded) == 1), np.flatnonzero(np.diff(padded) == -1)
        longest = max(longest, int(np.max(ends - starts)))
    return {
        "valid_tokens": available, "hidden_tokens": count, "context_tokens": available - count,
        "naturally_missing_tokens": int(valid.size - available), "hidden_fraction": count / available,
        "hidden_landmarks": joints, "landmark_count": len(joints),
        "left_hidden_tokens": int(hidden[:, sorted(LEFT_JOINTS)].sum()),
        "right_hidden_tokens": int(hidden[:, sorted(RIGHT_JOINTS)].sum()),
        "midline_hidden_tokens": int(hidden[:, 0].sum()),
        "first_hidden_block": int(positions[:, 0].min()), "last_hidden_block": int(positions[:, 0].max()),
        "hidden_span_blocks": int(positions[:, 0].max() - positions[:, 0].min() + 1),
        "longest_observed_hidden_run_blocks": longest, "segment_length": int(segment_length),
    }


def sample_mask(
    xyz: np.ndarray, token_valid: np.ndarray, policy: MaskPolicy, budget: MaskBudget,
    rng: np.random.Generator, *, segment_length: int = 4,
    timestamps: np.ndarray | None = None, observation_valid: np.ndarray | None = None,
) -> MaskResult:
    """Construct one label-blind mask with unique [time block, landmark] targets.

The caller supplies only permitted observations. Ordinary masked pretraining
can use its whole input to score motion; a forecasting caller must first limit
the input and all adaptive calculations to the observed prefix.
"""
    policy.validate()
    budget.validate(policy)
    if not isinstance(rng, np.random.Generator):
        raise TypeError("Pass a dedicated numpy.random.Generator")
    coordinates, valid, observed = _validate_input(xyz, token_valid, segment_length, observation_valid)
    eligible = _eligible_tokens(valid, policy)
    metadata: dict[str, Any] = {}
    if policy.name in SCATTERED_POLICIES:
        candidates = np.flatnonzero(eligible.ravel())
        count = budget.hidden_count
        if count > len(candidates) or count >= valid.sum():
            raise InfeasibleMaskBudget(
                f"Requested {count} targets, with {len(candidates)} eligible and {int(valid.sum())} total valid tokens"
            )
        weights = np.ones(len(candidates), dtype=float) / len(candidates)
        if policy.name == "soft_gait":
            gait = np.isin(candidates % 33, GAIT_JOINTS)
            if gait.any():
                weights = (1 - policy.gait_weight) * weights + policy.gait_weight * gait / gait.sum()
            metadata["gait_fallback_uniform"] = not bool(gait.any())
        elif policy.name == "motion":
            scores, metadata = motion_scores(
                coordinates, valid, segment_length=segment_length, timestamps=timestamps,
                observation_valid=observed, clip_quantile=policy.motion_clip_quantile,
            )
            selected_scores = scores.ravel()[candidates]
            if selected_scores.sum() > 0:
                weights = (1 - policy.motion_weight) * weights + policy.motion_weight * selected_scores / selected_scores.sum()
            else:
                metadata["motion_fallback_uniform"] = True
        selected = rng.choice(candidates, size=count, replace=False, p=weights)
        mask = np.zeros(valid.size, dtype=bool)
        mask[selected] = True
        mask = mask.reshape(valid.shape)
    elif policy.name == "whole_trajectory":
        mask, metadata = _whole_trajectory_mask(valid, eligible, budget, rng)
    else:
        mask, metadata = _interval_mask(valid, eligible, policy, budget, rng)
    coverage = coverage_summary(mask, valid, segment_length=segment_length)
    if budget.hidden_count is not None and coverage["hidden_tokens"] != budget.hidden_count:
        raise AssertionError("Mask construction violated the explicit exact-count contract")
    coverage.update(metadata)
    coverage.update({"policy": policy.name, "eligible_landmarks": list(eligible_landmarks(policy)),
                     "eligible_tokens": int(eligible.sum()), "requested_hidden_count": budget.hidden_count})
    return MaskResult(mask=mask, target_indices=np.argwhere(mask), coverage=coverage)


def reflect_sample(
    xyz: np.ndarray, token_valid: np.ndarray, mask: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Reflect x and exchange anatomical identities in values, validity and mask."""
    coordinates = np.asarray(xyz).copy()
    valid, hidden = np.asarray(token_valid).copy(), np.asarray(mask).copy()
    if coordinates.ndim != 3 or coordinates.shape[1:] != (33, 3) or valid.shape != hidden.shape or valid.shape[-1] != 33:
        raise ValueError("Expected one 33-landmark clip and compatible token masks")
    coordinates[..., 0] *= -1
    for left, right in FULL_MIRROR_PAIRS:
        coordinates[:, [left, right]] = coordinates[:, [right, left]]
        valid[:, [left, right]] = valid[:, [right, left]]
        hidden[:, [left, right]] = hidden[:, [right, left]]
    return coordinates, valid, hidden
