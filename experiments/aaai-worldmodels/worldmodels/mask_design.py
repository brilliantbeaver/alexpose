"""Executable mask-design examples, not fitted or validated research methods.

Only an observed prefix enters this API. Masks describe artificial corruption,
not natural missingness or future targets. Structured policies use an exact
token budget and can truncate their final block. No mask performance is claimed.
"""
from __future__ import annotations

import numpy as np

from .data import JOINT_IDS

POLICIES = ('uniform', 'time_blocks', 'anatomical_tubes', 'motion_balanced')
REFERENCE_JOINTS = (11, 12, 23, 24)
GROUPS = ((25, 27, 29, 31), (26, 28, 30, 32), (13, 15), (14, 16))


def design_prefix_mask(prefix, observed, times, policy, *, ratio=.4, seed=27):
    """Return hidden/protected masks and counts for one [time,16,2] prefix.

    All observed reference joints and each joint's latest observation remain
    visible. Motion strata use only adjacent observed pairs and actual times.
    Missing motion is a separate pool. Seeds never depend on labels or errors.
    The policy may use values it subsequently hides, so motion-aware selection
    needs a mask-pattern-only control in a future fitted experiment.
    """
    x = np.asarray(prefix, dtype=float)
    valid = np.asarray(observed)
    t = np.asarray(times, dtype=float)
    if x.ndim != 3 or x.shape[1:] != (len(JOINT_IDS), 2):
        raise ValueError('Expected prefix [time,16,2] in the declared joint order')
    if valid.shape != x.shape[:-1] or valid.dtype != np.bool_:
        raise ValueError('Observed support must be a Boolean [time,16] array')
    if t.ndim != 1 or len(t) != len(x) or len(t) < 2 or not np.isfinite(t).all() or np.any(np.diff(t) <= 0):
        raise ValueError('Need increasing finite prefix timestamps')
    if not np.isfinite(x[valid]).all():
        raise ValueError('Observed coordinates must be finite')
    if policy not in POLICIES or not 0 < ratio < 1:
        raise ValueError('Unknown policy or invalid corruption fraction')
    rng = np.random.default_rng(seed)
    lookup = {j: i for i, j in enumerate(JOINT_IDS)}
    protected = np.zeros_like(valid)
    protected[:, [lookup[j] for j in REFERENCE_JOINTS]] = valid[:, [lookup[j] for j in REFERENCE_JOINTS]]
    for j in range(len(JOINT_IDS)):
        indexes = np.flatnonzero(valid[:, j])
        if len(indexes):
            protected[indexes[-1], j] = True
    eligible = valid & ~protected
    ids = np.flatnonzero(eligible)
    if len(ids) < 2:
        raise ValueError('Need at least two eligible non-protected observations')
    budget = max(1, min(len(ids)-1, int(np.floor(ratio * len(ids) + .5))))
    chunks = []
    if policy == 'uniform':
        chunks = [rng.permutation(ids)]
    elif policy == 'time_blocks':
        for start in rng.permutation(np.arange(0, len(x), 4)):
            block = np.zeros_like(valid)
            block[start:start+4] = eligible[start:start+4]
            chunks.append(rng.permutation(np.flatnonzero(block)))
    elif policy == 'anatomical_tubes':
        candidates = [(start, group) for start in range(0, len(x), 4) for group in GROUPS]
        for index in rng.permutation(len(candidates)):
            start, group = candidates[index]
            block = np.zeros_like(valid)
            js = [lookup[j] for j in group]
            block[start:start+4, js] = eligible[start:start+4, js]
            chunks.append(rng.permutation(np.flatnonzero(block)))
    else:
        # NaN means no measured motion pair, not zero measured velocity.
        motion = np.full(valid.shape, np.nan)
        paired = valid[1:] & valid[:-1]
        safe = np.where(valid[..., None], x, 0.)
        speed = np.linalg.norm(np.diff(safe, axis=0), axis=-1) / np.diff(t)[:, None]
        motion[1:] = np.where(paired, speed, np.nan)
        # Reserve 20% of the budget for uniform draws, including tokens whose
        # motion cannot be measured. All eligible tokens can be selected.
        uniform_count = max(1, int(np.ceil(.2 * budget)))
        uniform_ids = rng.permutation(ids)[:uniform_count]
        remaining = ids[~np.isin(ids, uniform_ids)]
        measured = remaining[np.isfinite(motion.ravel()[remaining])]
        ranked = measured[np.argsort(motion.ravel()[measured], kind='stable')]
        low, high = np.array_split(ranked, 2)
        low, high = rng.permutation(low), rng.permutation(high)
        alternating = []
        for k in range(max(len(low), len(high))):
            if k < len(low):
                alternating.append(low[k])
            if k < len(high):
                alternating.append(high[k])
        unknown = rng.permutation(remaining[~np.isfinite(motion.ravel()[remaining])])
        chunks = [uniform_ids, np.asarray(alternating, dtype=int), unknown]
    order = np.concatenate(chunks)
    if len(order) != len(ids) or len(np.unique(order)) != len(ids):
        raise AssertionError('Policy failed to cover eligible tokens exactly once')
    hidden = np.zeros_like(valid)
    hidden.ravel()[order[:budget]] = True
    cumulative = np.cumsum([len(c) for c in chunks])
    partial = bool(policy in ('time_blocks', 'anatomical_tubes') and budget not in cumulative)
    return hidden, protected, {
        'policy': policy, 'eligible_tokens': int(len(ids)), 'hidden_tokens': int(budget),
        'observed_tokens': int(valid.sum()), 'visible_tokens': int((valid & ~hidden).sum()),
        'requested_ratio': ratio, 'realized_eligible_ratio': float(budget / len(ids)),
        'partial_final_block': partial,
        'status': 'design example; no model trained',
    }
