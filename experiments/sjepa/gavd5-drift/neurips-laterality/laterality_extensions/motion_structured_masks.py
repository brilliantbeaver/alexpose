"""Motion and structure experiments for notebooks 15–18.

MAMP's published equations and its official implementation differ in intensity
normalization. ``mamp_motion`` follows the latter on fully observed inputs:
mean absolute patch displacement, per-clip maximum normalization, and Gumbel
top-k. Missing-transition handling is our explicit pose-data adaptation.
The established robust motion and anatomical samplers are reused unchanged.
See docs/MOTION_STRUCTURED_MASKING.md for sources and experimental boundaries.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib

import numpy as np

from .comparative_masks import (
    ANATOMICAL_EDGES, GAIT_JOINTS, InfeasibleMaskBudget, MaskBudget, MaskPolicy,
    MaskResult, _validate_input, coverage_summary, sample_mask,
    motion_scores,
)


@dataclass(frozen=True)
class StudyArm:
    name: str
    temperature: float = 0.8
    motion_weight: float = 0.75
    clip_quantile: float = 0.95
    trajectories: int = 3
    region_size: int = 6
    interval_blocks: int = 2

    def validate(self):
        if self.name not in {"uniform", "mamp_motion", "robust_motion",
                             "whole_trajectory", "connected_region", "temporal_gap"}:
            raise ValueError("Unknown motion/structure arm")
        if not np.isfinite(self.temperature) or self.temperature <= 0:
            raise ValueError("Motion temperature must be finite and positive")
        MaskPolicy("motion", motion_weight=self.motion_weight,
                   motion_clip_quantile=self.clip_quantile).validate()
        for value in (self.trajectories, self.region_size, self.interval_blocks):
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError("Structure sizes must be positive integers")


def mamp_logits(xyz, token_valid, *, segment_length=4, observation_valid=None,
                temperature=0.8):
    """Log weights in prepared-coordinate units; no physical-speed claim.

Each transition spans one token, at the same within-token offset. Both endpoints
must be observed. Replication into block zero requires corresponding observed
endpoints too. Stationary/single-block inputs have uniform log weights.
"""
    x, valid, observed = _validate_input(xyz, token_valid, segment_length, observation_valid)
    if not np.isfinite(temperature) or temperature <= 0:
        raise ValueError("Motion temperature must be finite and positive")
    patches = x.reshape(-1, segment_length, 33, 3)
    obs = observed.reshape(-1, segment_length, 33)
    intensity = np.zeros(valid.shape, dtype=float)
    support = np.zeros(valid.shape, dtype=int)
    if len(valid) > 1:
        transition = obs[1:] & obs[:-1]
        safe = np.where(obs[..., None], patches, 0.0)
        delta = np.abs(safe[1:] - safe[:-1]).mean(axis=-1)
        support[1:] = transition.sum(axis=1)
        intensity[1:] = np.divide(np.where(transition, delta, 0).sum(axis=1),
                                  support[1:], out=np.zeros_like(intensity[1:]),
                                  where=support[1:] > 0)
        intensity[0] = np.where(valid[0] & valid[1], intensity[1], 0)
        support[0] = np.where(valid[0] & valid[1], support[1], 0)
    intensity[~valid] = 0
    maximum = float(intensity.max())
    # Algebraically matches official code's max * tau + 1e-10 denominator.
    logits = intensity / (maximum * temperature + 1e-10)
    logits[~valid] = -np.inf
    return logits, {"motion_units": "mean absolute displacement per prepared token",
                    "motion_stride_steps": segment_length, "temperature": temperature,
                    "motion_maximum": maximum, "supported_tokens": int(((support > 0) & valid).sum()),
                    "motion_fallback_uniform": maximum == 0,
                    "normalization": "per-clip maximum, as in official MAMP code"}


def _cached_scores(xyz, valid, observed, arm, segment_length, cache):
    """Cache deterministic clip scores only; never cache a random mask draw.

Content keys include invalid observations too, so an in-place data change cannot
reuse stale weights. Training passes only its permitted rows to this function.
"""
    digest = hashlib.sha256(repr((arm, segment_length)).encode())
    for value in (xyz, valid, observed):
        array = np.ascontiguousarray(value)
        digest.update(str((array.shape, array.dtype)).encode())
        digest.update(array.tobytes())
    key = digest.hexdigest()
    if key not in cache:
        cache[key] = (mamp_logits(xyz, valid, segment_length=segment_length,
            observation_valid=observed, temperature=arm.temperature) if arm.name == "mamp_motion"
            else motion_scores(xyz, valid, segment_length=segment_length,
                observation_valid=observed, clip_quantile=arm.clip_quantile))
    scores, metadata = cache[key]
    return scores, dict(metadata)


def sample_study_mask(xyz, valid, arm, hidden_count, rng, *, segment_length=4,
                      observation_valid=None, score_cache=None):
    """Hide unique valid targets; every arm retains observed context."""
    arm.validate()
    if arm.name == "mamp_motion":
        MaskBudget(hidden_count=hidden_count).validate(MaskPolicy("uniform"))
        logits, metadata = (mamp_logits(xyz, valid, segment_length=segment_length,
            observation_valid=observation_valid, temperature=arm.temperature) if score_cache is None
            else _cached_scores(xyz, valid, observation_valid, arm, segment_length, score_cache))
        candidates = np.flatnonzero(valid)
        if hidden_count >= len(candidates):
            raise InfeasibleMaskBudget("MAMP targets must retain valid context")
        u = np.clip(rng.random(len(candidates)), np.finfo(float).tiny, 1 - np.finfo(float).eps)
        priorities = logits.ravel()[candidates] - np.log(-np.log(u))
        selected = candidates[np.argsort(priorities, kind="stable")[-hidden_count:]]
        mask = np.zeros_like(valid)
        mask.ravel()[selected] = True
        coverage = {**coverage_summary(mask, valid, segment_length=segment_length), **metadata,
                    "policy": arm.name, "eligible_landmarks": list(range(33))}
        return MaskResult(mask, np.argwhere(mask), coverage)
    if arm.name == "robust_motion" and score_cache is not None:
        # Same candidate order, arithmetic, RNG call and metadata as sample_mask.
        # Only deterministic motion_scores (33 nanmedians/clip) are memoized.
        MaskBudget(hidden_count=hidden_count).validate(MaskPolicy("motion"))
        xyz, valid, observed = _validate_input(xyz, valid, segment_length, observation_valid)
        candidates = np.flatnonzero(valid.ravel())
        if hidden_count >= len(candidates):
            raise InfeasibleMaskBudget("Motion targets must retain valid context")
        scores, metadata = _cached_scores(xyz, valid, observed, arm, segment_length, score_cache)
        weights = np.ones(len(candidates), dtype=float) / len(candidates)
        selected_scores = scores.ravel()[candidates]
        if selected_scores.sum() > 0:
            weights = ((1 - arm.motion_weight) * weights
                + arm.motion_weight * selected_scores / selected_scores.sum())
        else:
            metadata["motion_fallback_uniform"] = True
        selected = rng.choice(candidates, size=hidden_count, replace=False, p=weights)
        mask = np.zeros_like(valid)
        mask.ravel()[selected] = True
        coverage = {**coverage_summary(mask, valid, segment_length=segment_length), **metadata,
            "policy": "motion", "eligible_landmarks": list(range(33)),
            "eligible_tokens": int(valid.sum()), "requested_hidden_count": hidden_count}
        return MaskResult(mask, np.argwhere(mask), coverage)
    name = "motion" if arm.name == "robust_motion" else arm.name
    policy = MaskPolicy(name, motion_weight=arm.motion_weight,
                        motion_clip_quantile=arm.clip_quantile)
    if name == "whole_trajectory":
        budget = MaskBudget(trajectories=arm.trajectories)
    elif name == "connected_region":
        budget = MaskBudget(region_size=arm.region_size, interval_blocks=arm.interval_blocks)
    elif name == "temporal_gap":
        budget = MaskBudget(interval_blocks=arm.interval_blocks)
    else:
        budget = MaskBudget(hidden_count=hidden_count)
    return sample_mask(xyz, valid, policy, budget, rng, segment_length=segment_length,
                       observation_valid=observation_valid)


def study_arms(experiment, blocks):
    """A small declared comparison; structure has its own scattered reference."""
    if blocks < 3:
        raise ValueError("Study examples require at least three time blocks")
    if experiment == "motion":
        return {name: StudyArm(name) for name in ("uniform", "mamp_motion", "robust_motion")}
    specifications = {
        "regions": StudyArm("connected_region", interval_blocks=max(1, blocks // 2)),
        "trajectories": StudyArm("whole_trajectory"),
        "completion": StudyArm("temporal_gap", interval_blocks=max(1, blocks // 4)),
    }
    if experiment not in specifications:
        raise ValueError("Choose motion, regions, trajectories, or completion")
    return {"uniform": StudyArm("uniform"), specifications[experiment].name: specifications[experiment]}


def paired_study_masks(dataset, rows, settings, arms, *, step=0, score_cache=None):
    """Mask RNGs cannot alter source draws or geometric augmentation streams.

Counts match between arms for each clip. A structure's realized count determines
its scattered reference, including with missing measurements; no trimming occurs.
"""
    if len(arms) < 2:
        raise ValueError("A study needs at least two arms")
    for key, arm in arms.items():
        arm.validate()
        if key != arm.name:
            raise ValueError("Arm keys must equal their declared names")
    structured = [n for n in arms if n in {"whole_trajectory", "connected_region", "temporal_gap"}]
    if len(structured) > 1:
        raise ValueError("Each intact structure requires a separate comparison")
    valid = dataset.valid[rows].reshape(len(rows), -1, settings.segment_length, 33).all(2)
    count = max(1, int(valid[:, :, GAIT_JOINTS].sum((1, 2)).min() * settings.mask_fraction))
    results = {n: [] for n in arms}
    order = [*structured, *[n for n in arms if n not in structured]]
    for offset, row in enumerate(rows):
        clip_count = count
        for name in order:
            code = int.from_bytes(hashlib.sha256(name.encode()).digest()[:4], "little")
            rng = np.random.default_rng(np.random.SeedSequence([settings.seed, settings.fold, 151, code, step, offset]))
            result = sample_study_mask(dataset.xyz[row], valid[offset], arms[name], clip_count, rng,
                segment_length=settings.segment_length, observation_valid=dataset.valid[row],
                score_cache=score_cache)
            if name in structured:
                clip_count = int(result.mask.sum())
            result.coverage["reflected"] = False
            results[name].append(result)
    masks = {n: np.stack([r.mask for r in values]) for n, values in results.items()}
    counts = [m.sum((1, 2)) for m in masks.values()]
    if any(not np.array_equal(counts[0], value) for value in counts[1:]):
        raise AssertionError("Hidden counts differ across paired arms")
    return masks, {n: [r.coverage for r in values] for n, values in results.items()}


def context_cue_audit(mask, valid):
    """Describe available local cues, without claiming a learned shortcut.

Temporal bracketing requires visible immediate previous AND next tokens of the
same landmark. Spatial cues count at least one visible graph neighbor at the
same time. Graph links are the declared masking anatomy from Notebook 11.
"""
    coverage_summary(mask, valid)
    visible = valid & ~mask
    bracket = np.zeros_like(mask)
    bracket[1:-1] = visible[:-2] & visible[2:]
    neighbor = np.zeros_like(mask)
    for a, b in ANATOMICAL_EDGES:
        neighbor[:, a] |= visible[:, b]
        neighbor[:, b] |= visible[:, a]
    return {"hidden_tokens": int(mask.sum()),
            "temporal_bracket_fraction": float(bracket[mask].mean()),
            "visible_neighbor_fraction": float(neighbor[mask].mean())}
