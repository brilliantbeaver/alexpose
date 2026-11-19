"""Canonical prefix-corruption policies for medwm-reference-masking-v2.

These policies never receive future coordinates or future visibility. True in
``hidden`` denotes a deliberately removed observed prefix token. A token is one
joint's xy pair. Natural observation validity is never modified in place.
"""
from __future__ import annotations

from dataclasses import dataclass
import numpy as np

from .data import JOINT_IDS

CONTEXT = 16
REFERENCES = (11, 12, 23, 24)
GROUPS = ((25, 27, 29, 31), (26, 28, 30, 32), (13, 15), (14, 16))
POLICIES = ('none', 'uniform', 'time_blocks', 'anatomical_tubes',
            'motion_balanced', 'random_groups', 'motion_focused')
POLICY_LABELS = dict(zip(POLICIES, ('M0', 'M1', 'M2', 'M3', 'M4', 'M3-random', 'M4-focused')))


@dataclass(frozen=True)
class MaskConfig:
    policy: str = 'none'
    ratio: float = .4
    block_length: int = 4
    uniform_fraction: float = .2

    def __post_init__(self):
        if self.policy not in POLICIES:
            raise ValueError('Unknown corruption policy')
        if not 0 < self.ratio < 1 or not 0 < self.uniform_fraction <= 1:
            raise ValueError('Mask ratio and uniform fraction must be in (0,1) and (0,1]')
        if not isinstance(self.block_length, int) or self.block_length < 1:
            raise ValueError('Need a positive integer block length')


def _inputs(prefix, observed, times):
    x, v, t = np.asarray(prefix, float), np.asarray(observed), np.asarray(times, float)
    if x.shape != (CONTEXT, len(JOINT_IDS), 2):
        raise ValueError('Mask input must be the exact observed [16,16,2] prefix')
    if v.shape != x.shape[:-1] or v.dtype != np.bool_:
        raise ValueError('Natural support must be Boolean [16,16]')
    if t.shape != (CONTEXT,) or not np.isfinite(t).all() or np.any(np.diff(t) <= 0):
        raise ValueError('Need sixteen finite increasing actual prefix times')
    if not np.isfinite(x[v]).all():
        raise ValueError('Naturally observed coordinates must be finite')
    return x, v, t


def protected_tokens(observed):
    """Protect every observed reference and each joint's latest observation."""
    v = np.asarray(observed)
    if v.shape != (CONTEXT, len(JOINT_IDS)) or v.dtype != np.bool_:
        raise ValueError('Natural support must be Boolean [16,16]')
    p = np.zeros_like(v)
    refs = [JOINT_IDS.index(j) for j in REFERENCES]
    p[:, refs] = v[:, refs]
    for j in range(len(JOINT_IDS)):
        ii = np.flatnonzero(v[:, j])
        if len(ii):
            p[ii[-1], j] = True
    return p


def prefix_motion(prefix, observed, times):
    """Observed adjacent-pair 2D speed proxy. NaN denotes no measured pair."""
    x, v, t = _inputs(prefix, observed, times)
    safe = np.where(v[..., None], x, 0.)
    movement = np.full(v.shape, np.nan)
    paired = v[1:] & v[:-1]
    speed = np.linalg.norm(np.diff(safe, axis=0), axis=-1) / np.diff(t)[:, None]
    movement[1:] = np.where(paired, speed, np.nan)
    return movement


def _structured_chunks(eligible, groups, starts, rng, block_length):
    candidates = [(int(start), group) for start in starts for group in groups]
    chunks = []
    for k in rng.permutation(len(candidates)):
        start, group = candidates[k]
        block = np.zeros_like(eligible)
        block[start:start+block_length, group] = eligible[start:start+block_length, group]
        chunks.append(rng.permutation(np.flatnonzero(block)))
    return chunks


def make_mask(prefix, observed, times, config: MaskConfig | None = None, *, seed=27):
    """Return ``hidden, protected, audit`` for one observed prefix.

    Structured policies prioritize whole disjoint blocks but truncate their last
    block to meet the exact rounded budget. Random groups share anatomical group
    sizes and time extent, but are a shuffled partition of eligible joint IDs.
    All motion policies reserve >=20% of their budget for uniform draws, giving
    observations without measured movement positive selection probability.
    """
    c = config or MaskConfig()
    x, v, t = _inputs(prefix, observed, times)
    p = protected_tokens(v)
    eligible = v & ~p
    ids = np.flatnonzero(eligible)
    budget = 0 if c.policy == 'none' else int(np.floor(c.ratio * len(ids) + .5))
    if c.policy != 'none' and len(ids) < 2:
        raise ValueError('Need at least two eligible non-protected tokens')
    budget = min(max(budget, int(c.policy != 'none')), max(0, len(ids)-1))
    rng = np.random.default_rng(seed)
    chunks = []
    if c.policy == 'none':
        order = ids
    elif c.policy == 'uniform':
        chunks = [rng.permutation(ids)]
        order = chunks[0]
    elif c.policy == 'time_blocks':
        chunks = _structured_chunks(eligible, [list(range(16))],
            range(0, CONTEXT, c.block_length), rng, c.block_length)
        order = np.concatenate(chunks)
    elif c.policy in ('anatomical_tubes', 'random_groups'):
        groups = [[JOINT_IDS.index(j) for j in group] for group in GROUPS]
        # One joint partition is held fixed across time blocks. Sizes4,4,2,2
        # and block extent match the anatomical arm, while membership changes.
        if c.policy == 'random_groups':
            perm = rng.permutation([j for group in groups for j in group])
            lengths = np.cumsum([0] + [len(g) for g in groups])
            groups = [perm[lengths[k]:lengths[k+1]].tolist() for k in range(len(groups))]
        chunks = _structured_chunks(eligible, groups,
            range(0, CONTEXT, c.block_length), rng, c.block_length)
        order = np.concatenate(chunks)
    else:
        motion = prefix_motion(x, v, t).ravel()
        uniform_count = min(budget, max(1, int(np.ceil(c.uniform_fraction * budget))))
        uniform = rng.permutation(ids)[:uniform_count]
        remaining = ids[~np.isin(ids, uniform)]
        measured = remaining[np.isfinite(motion[remaining])]
        unknown = rng.permutation(remaining[~np.isfinite(motion[remaining])])
        if c.policy == 'motion_balanced':
            # Stable sorting resolves ties independently of outcomes/labels.
            ranked = measured[np.argsort(motion[measured], kind='stable')]
            low, high = np.array_split(ranked, 2)
            low, high = rng.permutation(low), rng.permutation(high)
            balance = []
            # Alternate high/low priority according to the seed, so an odd
            # remaining budget does not always privilege the low-motion bin.
            bins = (low, high) if rng.integers(2) == 0 else (high, low)
            for k in range(max(len(low), len(high))):
                for b in bins:
                    if k < len(b):
                        balance.append(int(b[k]))
            priority = np.asarray(balance, int)
        else:
            # Gumbel-top-k is weighted sampling without replacement. Robust
            # within-window rank weights1..2 bound the motion preference.
            rank = np.empty(len(measured), float)
            rank[np.argsort(motion[measured], kind='stable')] = np.arange(len(measured))
            weights = 1 + rank / max(1, len(measured)-1)
            keys = np.log(weights) + rng.gumbel(size=len(measured))
            priority = measured[np.argsort(-keys)]
        chunks = [uniform, priority, unknown]
        order = np.concatenate(chunks)
    if len(order) != len(ids) or len(np.unique(order)) != len(ids) or set(order) != set(ids):
        raise AssertionError('Mask policy must cover every eligible token exactly once')
    hidden = np.zeros_like(v)
    hidden.ravel()[order[:budget]] = True
    structured = c.policy in ('time_blocks', 'anatomical_tubes', 'random_groups')
    partial = bool(structured and budget not in np.cumsum([len(b) for b in chunks]))
    if np.any(hidden & (~v | p)):
        raise AssertionError('A hidden token was missing or protected')
    return hidden, p, dict(policy=c.policy, arm=POLICY_LABELS[c.policy],
        eligible_tokens=int(len(ids)), hidden_tokens=int(hidden.sum()),
        visible_tokens=int((v & ~hidden).sum()), observed_tokens=int(v.sum()),
        protected_tokens=int(p.sum()), partial_final_block=partial,
        realized_fraction=float(hidden.sum()/max(1, len(ids))),
        uniform_draws=int(min(budget, max(1, np.ceil(c.uniform_fraction*budget))))
            if c.policy.startswith('motion_') else 0)


def batch_masks(prefix, observed, times, config=None, *, seed=27):
    """Deterministic batch convenience API; rows get independent RNG seeds."""
    x, v, t = np.asarray(prefix), np.asarray(observed), np.asarray(times)
    if x.ndim != 4 or x.shape[1:] != (16,16,2) or v.shape != x.shape[:-1] or t.shape != (len(x),16):
        raise ValueError('Expected prefix[N,16,16,2], valid[N,16,16], times[N,16]')
    seeds = np.random.SeedSequence(seed).spawn(len(x))
    entries = [make_mask(a,b,c,config,seed=np.random.default_rng(s).integers(2**32))
               for a,b,c,s in zip(x,v,t,seeds)]
    return np.stack([e[0] for e in entries]), [e[2] for e in entries]
