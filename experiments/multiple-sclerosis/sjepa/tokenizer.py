"""Skeleton tokenizer for S-JEPA.

A window is a small movie of stick figures with shape ``(B, T, V, C)`` where
``T`` is frames, ``V`` is joints (33), and ``C`` is channels (x, y, visibility).
Following the paper we group ``l`` adjacent frames of one joint into a single
token, so one token summarises how that joint moved over a short slice of time.

Concretely we reshape ``(B, T, V, C)`` into ``(B, T//l, V, l*C)`` and project each
``l*C`` vector to the model width with a linear layer. We then add two separate
learnable position embeddings, one for which joint the token is (spatial) and one
for which time block it is (temporal). The flattened output has shape
``(B, N, dim)`` with ``N = (T//l) * V`` and token index ``t * V + v``, which is the
exact layout the mask in ``masking.py`` assumes.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class SkeletonTokenizer(nn.Module):
    def __init__(self, num_joints: int, in_channels: int, frame_group: int,
                 num_time_tokens: int, dim: int):
        super().__init__()
        self.num_joints = num_joints
        self.in_channels = in_channels
        self.frame_group = frame_group
        self.num_time_tokens = num_time_tokens
        self.dim = dim

        self.proj = nn.Linear(frame_group * in_channels, dim)
        # Separate spatial (per joint) and temporal (per time block) embeddings.
        self.spatial_emb = nn.Parameter(torch.zeros(1, 1, num_joints, dim))
        self.temporal_emb = nn.Parameter(torch.zeros(1, num_time_tokens, 1, dim))
        nn.init.trunc_normal_(self.spatial_emb, std=0.02)
        nn.init.trunc_normal_(self.temporal_emb, std=0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """(B, T, V, C) -> (B, N, dim) with N = (T//l) * V."""
        B, T, V, C = x.shape
        l = self.frame_group
        if T % l != 0:
            raise ValueError(f"T={T} not divisible by frame_group={l}")
        Tt = T // l
        # Group frames: (B, T, V, C) -> (B, Tt, l, V, C) -> (B, Tt, V, l*C)
        x = x.view(B, Tt, l, V, C).permute(0, 1, 3, 2, 4).reshape(B, Tt, V, l * C)
        tokens = self.proj(x)                          # (B, Tt, V, dim)
        tokens = tokens + self.spatial_emb + self.temporal_emb
        return tokens.reshape(B, Tt * V, self.dim)     # (B, N, dim)
