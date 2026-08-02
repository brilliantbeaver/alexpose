"""Anatomical masking for S-JEPA.

The original S-JEPA paper borrows MAMP's *motion-aware* masking: it hides the
joint-time tokens that move the most, chosen fresh for every clip. For a clinical
gait study that rule is a poor fit, because in many conditions the informative
signal is *reduced* motion (short steps, stiff knees, reduced arm swing). So this
project does something different and deliberately simple.

We mask a fixed, hand chosen set of neurologically relevant joints and nothing
else. The set comes from ``mapping-data/ms-pd-mapping.md`` and, after removing
duplicates and sorting, is exactly these twelve BlazePose-33 landmarks:

    11 LEFT_SHOULDER    12 RIGHT_SHOULDER
    23 LEFT_HIP         24 RIGHT_HIP
    25 LEFT_KNEE        26 RIGHT_KNEE
    27 LEFT_ANKLE       28 RIGHT_ANKLE
    29 LEFT_HEEL        30 RIGHT_HEEL
    31 LEFT_FOOT_INDEX  32 RIGHT_FOOT_INDEX

That is both shoulders plus both complete legs. The remaining joints (the face
and the arms) form the visible context. The mask never changes and never looks
at the motion signal, which is the whole point.

The functions below are pure and do not import torch, so they are cheap to call
and easy to test. They return boolean numpy arrays that the training code turns
into torch tensors.
"""

from __future__ import annotations

from typing import List, Tuple

import numpy as np

from .config import ANATOMICAL_MASK_IDX


# A frozen copy so callers cannot mutate the shared list by accident.
MASKED_JOINTS: Tuple[int, ...] = tuple(ANATOMICAL_MASK_IDX)


def joint_token_mask(num_joints: int, num_time_tokens: int) -> np.ndarray:
    """Return a boolean array of shape (num_tokens,), True where a token is masked.

    Tokens are laid out as ``(time_token, joint)`` in row major order, matching
    the tokenizer: token index ``t * num_joints + v`` belongs to time block ``t``
    and joint ``v``. Every temporal token of a masked joint is a target, so the
    mask is the same pattern repeated for each time block.
    """
    masked = set(MASKED_JOINTS)
    per_joint = np.array([v in masked for v in range(num_joints)], dtype=bool)
    # Repeat the joint pattern for each time block.
    full = np.tile(per_joint, num_time_tokens)
    assert full.shape[0] == num_joints * num_time_tokens
    return full


class AnatomicalMaskSampler:
    """Produces the fixed target/context split for a batch.

    Unlike a motion-aware sampler this takes no input signal. It computes the
    mask once from the joint layout and hands back the same partition every call.
    """

    def __init__(self, num_joints: int, num_time_tokens: int):
        self.num_joints = num_joints
        self.num_time_tokens = num_time_tokens
        self._target = joint_token_mask(num_joints, num_time_tokens)
        self._context = ~self._target
        if not self._target.any():
            raise ValueError("Anatomical mask selected zero target tokens.")
        if not self._context.any():
            raise ValueError("Anatomical mask left zero visible context tokens.")

    @property
    def target_mask(self) -> np.ndarray:
        """Boolean (num_tokens,) marking joints we hide and predict."""
        return self._target.copy()

    @property
    def context_mask(self) -> np.ndarray:
        """Boolean (num_tokens,) marking the visible joints (face and arms)."""
        return self._context.copy()

    @property
    def target_indices(self) -> np.ndarray:
        return np.nonzero(self._target)[0]

    @property
    def context_indices(self) -> np.ndarray:
        return np.nonzero(self._context)[0]

    def summary(self) -> str:
        nt = int(self._target.sum())
        nc = int(self._context.sum())
        return (
            f"AnatomicalMaskSampler: {nt} target tokens, {nc} context tokens "
            f"(masked joints={list(MASKED_JOINTS)})"
        )


def masked_joint_names(joint_names: List[str]) -> List[str]:
    """Convenience: map the masked indices to their names for display."""
    return [joint_names[i] for i in MASKED_JOINTS]
