"""Matched controls generated strictly inside the current fitting partition."""

import numpy as np
from scipy.optimize import linear_sum_assignment

from .fi_contracts import stable_key


def block_shuffle(history, window_id, block=4):
    if len(history) % block:
        raise ValueError("Shuffle block must divide context length")
    blocks = history.reshape(len(history) // block, block, *history.shape[1:])
    rng = np.random.default_rng(int(stable_key("shuffle", window_id)[:16], 16))
    order = rng.permutation(len(blocks))
    if np.array_equal(order, np.arange(len(blocks))):
        order = np.roll(order, 1)
    return blocks[order].reshape(history.shape)


def mismatch_donors(video_ids, metadata):
    video_ids = np.asarray(video_ids, dtype=str)
    metadata = np.asarray(metadata, dtype=float)
    if (
        len(set(video_ids)) < 2
        or metadata.shape[0] != len(video_ids)
        or not np.isfinite(metadata).all()
    ):
        raise ValueError(
            "Mismatch requires finite context metadata from at least two sources"
        )
    # Donor matching is a control construction, not learned model preprocessing.
    # Its statistics are confined to this partition, including validation/test.
    scaled = (metadata - metadata.mean(axis=0)) / np.maximum(metadata.std(axis=0), 1e-8)
    cost = ((scaled[:, None] - scaled[None]) ** 2).sum(axis=-1)
    cost[video_ids[:, None] == video_ids[None]] = np.inf
    try:
        recipients, donors = linear_sum_assignment(cost)
        result = np.empty(len(video_ids), dtype=int)
        result[recipients] = donors
        replacement = False
    except ValueError:
        # Only necessary when a source occupies more than half the partition.
        result = np.argmin(cost, axis=1)
        replacement = True
    if (
        np.any(video_ids == video_ids[result])
        or not np.isfinite(cost[np.arange(len(result)), result]).all()
    ):
        raise ValueError("Invalid same-source mismatch donor")
    return result, replacement


def controlled_history(arm, skeleton, window_ids, video_ids, metadata):
    if arm in ("real-skeleton", "background-target"):
        return skeleton.copy(), None
    if arm == "time-shuffle":
        return np.stack(
            [block_shuffle(s, w) for s, w in zip(skeleton, window_ids)]
        ), None
    if arm == "no-skeleton":
        value = skeleton.copy()
        value[..., :3] = 0
        return value, None
    if arm == "clip-mismatch":
        donors, replacement = mismatch_donors(video_ids, metadata)
        return skeleton[donors].copy(), {
            "donor_window_ids": np.asarray(window_ids)[donors].tolist(),
            "replacement": replacement,
        }
    raise ValueError(f"Unknown arm: {arm}")
