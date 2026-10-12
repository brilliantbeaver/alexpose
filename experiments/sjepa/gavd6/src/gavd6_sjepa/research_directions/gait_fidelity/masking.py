"""Matched body12 pretraining masks, independent of reference support.

Intervals use cyclic starts to balance edge exposure. They are artificial query
masks, not simulations of physically contiguous image occlusion. The final
region may be truncated to match the exact observed-token hiding budget.
"""
from __future__ import annotations

import hashlib
import numpy as np

POLICIES = ("time_blocks", "uniform_tokens", "graph_time", "shuffled_topology", "random_joint_intervals")
# Named left/right regions plus connected axial regions; no clinical weighting.
REGIONS = ((0, 2, 4), (1, 3, 5), (6, 8, 10), (7, 9, 11),
           (0, 1, 6, 7), (0, 6, 8), (1, 7, 9), (0, 2), (1, 3), (8, 10), (9, 11))
EDGES = ((0, 1), (0, 2), (2, 4), (1, 3), (3, 5), (0, 6), (1, 7),
         (6, 7), (6, 8), (8, 10), (7, 9), (9, 11))
ALIASES = {"topology_shuffled": "shuffled_topology", "random_intervals": "random_joint_intervals"}


def patch_support(observed, patch_size=4):
    observed = np.asarray(observed)
    if observed.dtype != np.bool_ or observed.ndim != 3 or observed.shape[-1] != 12:
        raise ValueError("Observed must be boolean [batch,time,12]")
    if patch_size < 1 or observed.shape[1] % patch_size:
        raise ValueError("Patch size must divide the time dimension")
    return observed.reshape(len(observed), -1, patch_size, 12).any(2)


def realized_run_lengths(mask):
    """Linear crop run lengths after overlaps, sparse support and truncation.

    A cyclic interval crossing the crop edge produces two runs here, matching
    what the model sees. Counts pool example/joint runs, not independent people.
    """
    mask = np.asarray(mask, bool)
    lengths = []
    for sample in mask:
        for joint in range(sample.shape[1]):
            padded = np.r_[False, sample[:, joint], False].astype(np.int8)
            starts = np.flatnonzero(np.diff(padded) == 1)
            ends = np.flatnonzero(np.diff(padded) == -1)
            lengths.extend((ends - starts).tolist())
    return lengths


def sample_mask(observed, policy, *, rng, fraction=.5, patch_size=4,
                topology_seed=271828, max_interval_patches=None, return_receipt=False):
    """Hide an exactly matched number of available tokens, retaining context.

    The same number is hidden for every policy on each example. With one or
    fewer available tokens no artificial masking is added. A fresh advancing
    NumPy Generator, separate from the batch RNG, is required.
    """
    policy = ALIASES.get(policy, policy)
    if policy not in POLICIES or not 0 < fraction < 1:
        raise ValueError("Unknown policy or masking fraction outside (0,1)")
    eligible = patch_support(observed, patch_size)
    result = np.zeros_like(eligible)
    blocks = eligible.shape[1]
    max_len = max(1, blocks // 2) if max_interval_patches is None else int(max_interval_patches)
    if not 1 <= max_len <= blocks:
        raise ValueError("Interval length must be between one and the patch count")
    topology = np.random.default_rng(topology_seed).permutation(12)
    # A fixed bijection preserves region sizes and durations; it changes only
    # which anatomical names occupy the graph. Store it in every receipt.
    sparse_fill_tokens = 0
    requested_intervals = []
    for row, available in zip(result, eligible):
        n = int(available.sum())
        budget = min(max(1, round(n * fraction)), n - 1) if n > 1 else 0
        if budget == 0:
            continue
        if policy == "uniform_tokens":
            slots = np.flatnonzero(available)
            row.flat[rng.choice(slots, budget, replace=False)] = True
        elif policy == "time_blocks":
            joint_order = rng.permutation(12)
            start = int(rng.integers(blocks))
            order = [(int((start + k) % blocks), int(j)) for k in range(blocks) for j in joint_order
                     if available[(start + k) % blocks, j]]
            for t, j in order[:budget]:
                row[t, j] = True
        else:
            attempts = 0
            while row.sum() < budget and attempts < 100 * blocks:
                attempts += 1
                region = np.asarray(REGIONS[int(rng.integers(len(REGIONS)))])
                # Draw duration/start before policy-specific joint draws, and
                # report achieved run lengths; equal distributions are not an
                # assertion of identical unions after overlap/truncation.
                duration, start = int(rng.integers(1, max_len + 1)), int(rng.integers(blocks))
                requested_intervals.append(duration)
                if policy == "shuffled_topology":
                    region = topology[region]
                elif policy == "random_joint_intervals":
                    region = rng.choice(12, len(region), replace=False)
                for offset in range(duration):
                    for joint in rng.permutation(region):
                        t = (start + offset) % blocks
                        if available[t, joint] and not row[t, joint]:
                            row[t, joint] = True
                            if row.sum() == budget:
                                break
                    if row.sum() == budget:
                        break
            # Sparse support cannot hang a sampler. This declared uniform fill
            # preserves the budget and is measured in the receipt.
            if row.sum() < budget:
                remaining = np.flatnonzero(available & ~row)
                sparse_fill_tokens += budget - int(row.sum())
                row.flat[rng.choice(remaining, budget - int(row.sum()), replace=False)] = True
    if not return_receipt:
        return result
    counts = eligible.sum((1, 2))
    receipt = {"policy": policy, "sha256": hashlib.sha256(result.tobytes()).hexdigest(),
        "observed_tokens": counts.tolist(), "hidden_tokens": result.sum((1, 2)).tolist(),
        "context_tokens": (eligible & ~result).sum((1, 2)).tolist(),
        "realized_fraction": float(result.sum() / max(1, eligible.sum())),
        "cyclic_intervals": True, "topology_permutation": topology.tolist() if policy == "shuffled_topology" else None,
        "sparse_no_added_mask": int((counts <= 1).sum()),
        "sparse_uniform_fill_tokens": sparse_fill_tokens,
        "requested_interval_lengths": requested_intervals,
        "realized_linear_crop_run_lengths": realized_run_lengths(result),
        "final_region_may_be_truncated": True}
    return result, receipt


def coverage_audit(observed, *, draws=512, seed=17, **kwargs):
    """Expose joint-time probabilities and complete-joint context coverage."""
    rng = np.random.default_rng(seed)
    bank = np.stack([sample_mask(observed, rng=rng, **kwargs) for _ in range(draws)])
    available = patch_support(observed, kwargs.get("patch_size", 4))
    denominator = available.sum(0) * draws
    hidden = bank.sum((0, 1))
    probability = np.divide(hidden, denominator, out=np.full_like(hidden, np.nan, dtype=float), where=denominator > 0)
    return {"draws": draws, "masked_probability_by_patch_joint": probability.tolist(),
            "always_hidden_eligible_slots": int(((hidden == denominator) & (denominator > 0)).sum()),
            "never_hidden_eligible_slots": int(((hidden == 0) & (denominator > 0)).sum()),
            "context_somewhere_by_joint": (available[None] & ~bank).any(2).mean((0, 1)).tolist()}


def audit_matching(observed, *, draws=128, seed=17, fraction=.5, patch_size=4,
                   probability_tolerance=.10, run_length_tolerance=.10):
    """Audit residual confounds in nominal topology/duration controls.

    Exact hiding budgets are enforced by construction. Node marginals and union
    run lengths need not be equal after overlapping regions. Their empirical
    deviations are retained under frozen tolerances and restrict a claim that
    anatomy alone explains any observed difference.
    """
    from scipy.stats import wasserstein_distance
    available = patch_support(observed, patch_size)
    denominator = available.sum(0) * draws
    banks, probabilities, runs = {}, {}, {}
    for policy in ("graph_time", "shuffled_topology", "random_joint_intervals"):
        rng = np.random.default_rng(seed)
        banks[policy] = np.stack([sample_mask(observed, policy, rng=rng, fraction=fraction,
                                             patch_size=patch_size) for _ in range(draws)])
        hidden = banks[policy].sum((0, 1))
        probabilities[policy] = np.divide(hidden, denominator, out=np.zeros_like(hidden, float), where=denominator > 0)
        runs[policy] = realized_run_lengths(banks[policy].reshape(-1, *available.shape[1:]))
    comparisons = {}
    for policy in ("shuffled_topology", "random_joint_intervals"):
        delta = np.abs(probabilities[policy] - probabilities["graph_time"])
        slots = denominator > 0
        maximum = float(delta[slots].max()) if slots.any() else None
        distance = float(wasserstein_distance(runs[policy], runs["graph_time"]) / available.shape[1]) if runs[policy] and runs["graph_time"] else None
        budget_equal = bool(np.array_equal(banks[policy].sum((-1, -2)), banks["graph_time"].sum((-1, -2))))
        passed = budget_equal and maximum is not None and maximum <= probability_tolerance and distance is not None and distance <= run_length_tolerance
        comparisons[policy] = dict(exact_hidden_counts_matched=budget_equal,
            maximum_joint_time_probability_difference=maximum,
            mean_joint_time_probability_difference=float(delta[slots].mean()) if slots.any() else None,
            normalized_run_length_Wasserstein=distance, tolerance_passed=bool(passed),
            pure_connectivity_attribution_supported=bool(passed))
    return dict(seed=seed, draws=draws, probability_tolerance=probability_tolerance,
                run_length_tolerance=run_length_tolerance,
                probability_by_policy={k: v.tolist() for k, v in probabilities.items()},
                run_length_histogram_by_policy={k: np.bincount(v, minlength=available.shape[1]+1).tolist() for k,v in runs.items()},
                comparisons=comparisons,
                claim_limit="A failed matching tolerance leaves joint exposure or temporal persistence as an alternative explanation; retain every control.")
