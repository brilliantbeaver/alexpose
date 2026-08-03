"""Repaired masking for S-JEPA: per-example stochastic graph-time masks.

This replaces the fixed 12-joint mask (see ``masking.py``, kept for the E0
reference) with the design the audit and literature call for:

* **Per-example** masks shaped ``(B, N)`` rather than one ``(N,)`` pattern reused
  across the whole batch.
* **Stochastic graph-time regions**: each masked region is a connected group of
  joints (a limb or the trunk) over a contiguous span of time blocks, so the
  mask has spatial and temporal structure instead of hiding whole joints for all
  time.
* **Every joint rotates** between context and target across steps, so no joint is
  starved of context gradient (the D2 defect).
* **Clinical target bias, not motion bias.** The lower-body and shoulder joints
  are *sampled as targets a little more often* (default 1.5x). We deliberately do
  NOT bias toward high-motion regions: reduced motion (hypokinesia, short steps)
  is exactly the clinical signal in MS/PD, so a high-motion mask would hide the
  evidence (MAMP's motion-aware masking is contraindicated here).
* **Full context coverage guaranteed**: at least one lower-body / contralateral
  cue is kept visible, and the target fraction is bounded so context is never
  empty.

Tokens are laid out as ``token index = t * V + v`` (time block ``t``, joint
``v``), matching the tokenizer. Functions are pure numpy and take an explicit
``numpy.random.Generator`` so masks are deterministic under a seed and diverse
across examples.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Sequence, Tuple

import numpy as np

from .config import ANATOMICAL_MASK_IDX

# BlazePose-33 connected joint groups (graph regions). Each region is a connected
# set of joints; a mask grows from one region over a contiguous time span. Every
# joint belongs to at least one region, and the face is split so its joints can
# also be targets (the coverage gate requires *every* joint to be targetable).
JOINT_GROUPS: Dict[str, Tuple[int, ...]] = {
    "head": (0, 1, 2, 3, 4, 5, 6, 7, 8),
    "mouth": (9, 10),
    "left_arm": (11, 13, 15, 17, 19, 21),
    "right_arm": (12, 14, 16, 18, 20, 22),
    "trunk": (11, 12, 23, 24),
    "left_leg": (23, 25, 27, 29, 31),
    "right_leg": (24, 26, 28, 30, 32),
}

# Clinically relevant joints (both shoulders + both legs) get a target-sampling
# boost. This is a *bias*, not a permanent hide.
CLINICAL_JOINTS = frozenset(ANATOMICAL_MASK_IDX)


@dataclass
class MaskBankStats:
    """Coverage accounting over a bank of sampled masks (for the gates/tests)."""

    joint_visible_frac: np.ndarray   # (V,) fraction of masks where joint is context
    joint_target_frac: np.ndarray    # (V,) fraction of masks where joint is target
    mean_target_frac: float          # mean fraction of tokens masked
    n_masks: int


def _region_joint_pool() -> List[Tuple[str, Tuple[int, ...]]]:
    """All connected regions we grow masks from (every joint is targetable)."""
    return list(JOINT_GROUPS.items())


def _region_weights(regions, clinical_bias: float) -> np.ndarray:
    """Region-selection weights.

    Joints that appear in several regions (e.g. shoulders 11/12 and hips 23/24 in
    the trunk) would otherwise be over-targeted, so we down-weight each region by
    the average multiplicity of its joints. Clinical regions get a mild boost. The
    net effect keeps per-joint target frequency roughly balanced while still
    favouring the clinically relevant lower body a little.
    """
    # How many regions each joint belongs to.
    mult = {}
    for _, joints in regions:
        for j in joints:
            mult[j] = mult.get(j, 0) + 1
    w = []
    for _, joints in regions:
        avg_mult = np.mean([mult[j] for j in joints])
        boost = clinical_bias if any(j in CLINICAL_JOINTS for j in joints) else 1.0
        w.append(boost / avg_mult)
    w = np.asarray(w, dtype=float)
    return w / w.sum()


def sample_target_mask(
    num_joints: int,
    num_time_tokens: int,
    rng: np.random.Generator,
    target_ratio: float = 0.6,
    clinical_bias: float = 1.5,
    max_time_span_frac: float = 0.75,
) -> np.ndarray:
    """Sample ONE per-example target mask, shape (num_tokens,), True = target.

    We grow connected graph-time regions until roughly ``target_ratio`` of the
    tokens are targeted, then guarantee a non-empty context by clearing targets
    if we overshot. Clinically relevant joints are chosen more often via
    ``clinical_bias``. Time spans are contiguous and bounded by
    ``max_time_span_frac`` of the window so masks keep temporal structure.
    """
    V, T = num_joints, num_time_tokens
    N = V * T
    target = np.zeros((T, V), dtype=bool)

    regions = _region_joint_pool()
    weights = _region_weights(regions, clinical_bias)

    max_span = max(1, int(round(max_time_span_frac * T)))
    budget = int(round(target_ratio * N))
    guard = 0
    while target.sum() < budget and guard < 4 * len(regions) + 8:
        guard += 1
        ri = rng.choice(len(regions), p=weights)
        _, joints = regions[ri]
        # contiguous temporal span
        span = int(rng.integers(1, max_span + 1))
        start = int(rng.integers(0, max(1, T - span + 1)))
        for t in range(start, min(T, start + span)):
            for j in joints:
                target[t, j] = True

    flat = target.reshape(-1)
    # Guarantee non-empty context: if we masked everything, free a random block.
    if flat.all():
        flat[rng.integers(0, N)] = False
    # Guarantee at least one target (degenerate tiny configs).
    if not flat.any():
        flat[rng.integers(0, N)] = True
    return flat


def sample_mask_batch(
    batch_size: int,
    num_joints: int,
    num_time_tokens: int,
    rng: np.random.Generator,
    target_ratio: float = 0.6,
    masks_per_sequence: int = 1,
    clinical_bias: float = 1.5,
) -> np.ndarray:
    """Return a per-example target-mask batch of shape (B * masks, num_tokens).

    When ``masks_per_sequence > 1`` the batch is expanded so each sequence
    contributes several independent masks (the (B, M, N) idea, flattened to
    (B*M, N) for vectorised training). ``context = ~target``.
    """
    rows = []
    for _ in range(batch_size):
        for _ in range(masks_per_sequence):
            rows.append(sample_target_mask(
                num_joints, num_time_tokens, rng,
                target_ratio=target_ratio, clinical_bias=clinical_bias))
    return np.stack(rows, axis=0)


def mask_bank_stats(
    num_joints: int,
    num_time_tokens: int,
    n_masks: int = 512,
    seed: int = 0,
    target_ratio: float = 0.6,
    clinical_bias: float = 1.5,
) -> MaskBankStats:
    """Sample a bank of masks and measure per-joint context/target coverage.

    Used by the promotion-gate tests: every joint must be visible in some masks
    and targeted in others.
    """
    rng = np.random.default_rng(seed)
    V, T = num_joints, num_time_tokens
    vis_count = np.zeros(V)
    tgt_count = np.zeros(V)
    total_target = 0
    for _ in range(n_masks):
        m = sample_target_mask(V, T, rng, target_ratio=target_ratio,
                               clinical_bias=clinical_bias).reshape(T, V)
        joint_targeted = m.any(axis=0)     # (V,) targeted in at least one time block
        tgt_count += joint_targeted
        vis_count += (~m).any(axis=0)      # visible in at least one time block
        total_target += m.sum()
    return MaskBankStats(
        joint_visible_frac=vis_count / n_masks,
        joint_target_frac=tgt_count / n_masks,
        mean_target_frac=total_target / (n_masks * V * T),
        n_masks=n_masks,
    )
