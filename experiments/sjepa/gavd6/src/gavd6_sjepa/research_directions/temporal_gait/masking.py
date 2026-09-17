"""Prepared-feature masks and auditable source→bout→window sampling.

These masks do not claim raw-observation withholding: normalization and the
causal observation resampler have already prepared the features upstream.
"""
from __future__ import annotations

from collections import defaultdict
import math

import numpy as np
import torch
from torch.nn import functional as F


def patch_validity(valid: torch.Tensor, patch_size: int) -> torch.Tensor:
    """A token is usable if any original prepared sample in its patch is valid.

    Missing samples remain explicitly flagged inside the patch; structural
    right-padding is invalid and never creates an additional observation.
    """
    if valid.ndim != 3 or patch_size < 1:
        raise ValueError("Expected [batch,time,joint] validity and positive patch")
    padded = F.pad(valid.bool(), (0, 0, 0, (-valid.shape[1]) % patch_size))
    return padded.reshape(len(valid), -1, patch_size, valid.shape[2]).any(2)


def sample_feature_mask(valid: torch.Tensor, fraction: float,
                        rng: np.random.Generator) -> torch.Tensor:
    """Ragged per-example uniform masks, leaving at least one context token.

    No valid token means no loss, not a zero-error example. Singleton examples
    also receive no target because they cannot retain informative context.
    """
    if not 0 < fraction < 1:
        raise ValueError("mask_fraction must lie strictly between zero and one")
    eligible = valid.detach().cpu().numpy().astype(bool)
    result = np.zeros_like(eligible)
    for row, flags in enumerate(eligible.reshape(len(eligible), -1)):
        candidates = np.flatnonzero(flags)
        if len(candidates) < 2:
            continue
        count = min(len(candidates) - 1, max(1, math.floor(len(candidates) * fraction)))
        selected = rng.choice(candidates, size=count, replace=False)
        result[row].reshape(-1)[selected] = True
    return torch.as_tensor(result, dtype=torch.bool, device=valid.device)


class SourceBoutSampler:
    """Independent uniform source, then bout, then window draws with replacement."""

    def __init__(self, records: list[dict], rng: np.random.Generator,
                 allowed: list[int] | None = None):
        self.rng = rng
        grouped: dict[str, dict[str, list[int]]] = defaultdict(lambda: defaultdict(list))
        for i in range(len(records)) if allowed is None else allowed:
            record = records[i]
            grouped[str(record["video_id"])][str(record["sequence_id"])].append(i)
        self.groups = {source: dict(bouts) for source, bouts in grouped.items()}
        self.sources = sorted(self.groups)
        if not self.sources:
            raise ValueError("Training requires at least one eligible source window")

    def draw(self, count: int) -> list[int]:
        result = []
        for _ in range(count):
            source = self.sources[int(self.rng.integers(len(self.sources)))]
            bouts = sorted(self.groups[source])
            bout = bouts[int(self.rng.integers(len(bouts)))]
            windows = self.groups[source][bout]
            result.append(windows[int(self.rng.integers(len(windows)))])
        return result


def control_target_indices(records: list[dict], indices: list[int], arm: str,
                           rng: np.random.Generator, minimum_separation: float) -> list[int]:
    """Wrong-source excludes the connected group; wrong-time retains each window.

    The trainer implements `future_wrong_time` as a fixed circular permutation
    of the SAME window's encoded future horizons (wrong_horizon_order). This is
    deliberately not a disjoint-future-window claim. minimum_separation remains
    in this compatibility signature but is unused by this order control.
    """
    if arm in {"future", "future_wrong_time"}:
        return list(indices)
    if arm != "future_wrong_source":
        raise ValueError(f"Unknown future control {arm!r}")
    output = []
    for index in indices:
        current = records[index]
        allowed = [i for i, row in enumerate(records)
                   if row["group_id"] != current["group_id"]
                   and row["video_id"] != current["video_id"]]
        if not allowed:
            raise ValueError("Wrong-source requires a different training connected group")
        output.extend(SourceBoutSampler(records, rng, allowed).draw(1))
    return output
