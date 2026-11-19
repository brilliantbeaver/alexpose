"""Verify proposed U/J/A mask laws without loading data or training a model.

Run with Python 3.10 or newer. The script only prints a JSON report. Its local
random.Random sampler exercises the proposed laws; it is not a replacement for
the historical Torch evaluation bank or a production protocol artifact.
"""

from collections import Counter
from fractions import Fraction
from itertools import combinations
import json
from math import comb
import random


JOINTS = (11, 12, 13, 14, 15, 16, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32)
TIMES = tuple(range(8))
GRID = tuple((t, j) for t in TIMES for j in JOINTS)
GROUPS = (
    ("torso", (11, 12, 23, 24)),
    ("left_arm", (13, 15)),
    ("right_arm", (14, 16)),
    ("left_distal_leg", (25, 27, 29, 31)),
    ("right_distal_leg", (26, 28, 30, 32)),
)
MIRROR = dict(zip(JOINTS, (12, 11, 14, 13, 16, 15, 24, 23, 26, 25, 28, 27, 30, 29, 32, 31)))
EDGES = (
    (11, 12), (11, 13), (13, 15), (12, 14), (14, 16),
    (11, 23), (12, 24), (23, 24), (23, 25), (24, 26),
    (25, 27), (26, 28), (27, 29), (29, 31), (27, 31),
    (28, 30), (30, 32), (28, 32),
)
LOWER_JOINTS = frozenset(range(25, 33))
MASK_BUDGET = 64
CORE_SIZE = 8
FILL_SIZE = MASK_BUDGET - CORE_SIZE


def is_connected(joints):
    remaining = set(joints)
    reached = {remaining.pop()}
    while remaining:
        new = {v for u, v in EDGES if u in reached and v in remaining}
        new |= {u for u, v in EDGES if v in reached and u in remaining}
        if not new:
            return False
        reached |= new
        remaining -= new
    return True


def make_cores(policy):
    if policy == "J":
        return tuple(frozenset((t, j) for t in TIMES) for j in JOINTS)
    if policy == "A":
        result = []
        for _, joints in GROUPS:
            h = CORE_SIZE // len(joints)
            for start in range(0, len(TIMES), h):
                result.append(frozenset((t, j) for t in range(start, start + h) for j in joints))
        return tuple(result)
    raise ValueError(policy)


def fill_core(core, rng):
    remaining = tuple(cell for cell in GRID if cell not in core)
    return frozenset(core) | frozenset(rng.sample(remaining, FILL_SIZE))


def sample_mask(policy, rng):
    if policy == "U":
        return frozenset(rng.sample(GRID, MASK_BUDGET))
    return fill_core(rng.choice(make_cores(policy)), rng)


def probability_hidden(cell, cores):
    return sum((Fraction(1) if cell in core else Fraction(FILL_SIZE, len(GRID) - CORE_SIZE))
               for core in cores) / len(cores)


def probability_pair_hidden(pair, cores):
    total = Fraction(0)
    for core in cores:
        n_inside = len(set(pair) & core)
        if n_inside == 2:
            p = Fraction(1)
        elif n_inside == 1:
            p = Fraction(FILL_SIZE, len(GRID) - CORE_SIZE)
        else:
            p = Fraction(comb(FILL_SIZE, 2), comb(len(GRID) - CORE_SIZE, 2))
        total += p
    return total / len(cores)


def verify_core_law(policy):
    cores = make_cores(policy)
    assert len(cores) == 16 and len(set(cores)) == 16
    incidence = Counter(cell for core in cores for cell in core)
    assert set(incidence) == set(GRID)
    assert set(incidence.values()) == {1}
    for core in cores:
        assert len(core) == CORE_SIZE and core <= set(GRID)
        times = sorted({t for t, _ in core})
        assert times == list(range(times[0], times[-1] + 1))
        assert min(times) >= 0 and max(times) < 8
        assert is_connected({j for _, j in core})
        assert frozenset((t, MIRROR[j]) for t, j in core) in cores
        assert frozenset((7 - t, j) for t, j in core) in cores
    assert all(Fraction(incidence[cell], len(cores)) == Fraction(1, 16) for cell in GRID)
    assert all(probability_hidden(cell, cores) == Fraction(1, 2) for cell in GRID)
    pair_probs = [probability_pair_hidden(pair, cores) for pair in combinations(GRID, 2)]
    assert sum(pair_probs) == comb(MASK_BUDGET, 2)
    return {
        "core_count": len(cores),
        "core_size": CORE_SIZE,
        "core_partition_covers_each_token": 1,
        "core_probability_for_every_token": "1/16",
        "hidden_probability_for_every_token": "1/2",
        "pair_hidden_probability_range": [str(min(pair_probs)), str(max(pair_probs))],
        "pair_probability_sum": str(sum(pair_probs)),
        "left_right_reflection_closed": True,
        "time_reversal_closed": True,
        "all_spatial_cores_connected": True,
        "all_temporal_cores_contiguous_without_wrap": True,
    }


def verify_leg_bank_structure():
    # The seed only makes this verification exercise repeatable. Production
    # must record its own exact RNG implementation, bank bytes and checksum.
    rng = random.Random(17)
    bank = []
    for joints in ((25, 27, 29, 31), (26, 28, 30, 32)):
        for start in (0, 2, 4, 6):
            for repeat in (0, 1):
                core = frozenset((t, j) for t in (start, start + 1) for j in joints)
                mask = fill_core(core, rng)
                assert len(mask) == 64 and core <= mask
                # Only the core is a scored target in this diagnostic.
                bank.append({"core": core, "mask": mask, "repeat": repeat})
    targets = Counter(cell for item in bank for cell in item["core"])
    assert len(bank) == 16
    assert set(targets) == {(t, j) for t in TIMES for j in LOWER_JOINTS}
    assert set(targets.values()) == {2}
    assert len(targets) == 64 and sum(targets.values()) == 128
    lower_p = Fraction(1, 8) + Fraction(7, 8) * Fraction(56, 120)
    other_p = Fraction(56, 120)
    assert lower_p == Fraction(8, 15) and other_p == Fraction(7, 15)
    return {
        "mask_count": 16,
        "construction": "2 sides x 4 aligned starts x 2 repeats",
        "hidden_tokens_per_mask": 64,
        "scored_core_tokens_per_mask": 8,
        "distinct_scored_target_tokens": 64,
        "scored_target_instances_before_validity": 128,
        "repeats_per_distinct_target": 2,
        "expected_lower_limb_hidden_probability": str(lower_p),
        "expected_other_joint_hidden_probability": str(other_p),
        "same_distribution_as_primary_uniform_evaluation": False,
    }


def main():
    assert len(GRID) == 128 and len(set(GRID)) == 128
    assert tuple(sorted(j for _, joints in GROUPS for j in joints)) == JOINTS
    assert all(MIRROR[MIRROR[j]] == j for j in JOINTS)
    assert Fraction(comb(127, 63), comb(128, 64)) == Fraction(1, 2)
    for policy in ("U", "J", "A"):
        rng = random.Random(17)
        for _ in range(2048):
            mask = sample_mask(policy, rng)
            assert len(mask) == MASK_BUDGET and mask <= set(GRID)
    result = {
        "status": "PASS",
        "scope": "Exact proposed mask laws and standalone sampler invariants only; no GAVD data or models loaded.",
        "joints": list(JOINTS),
        "temporal_tokens": 8,
        "frames_per_token": 4,
        "grid_tokens": 128,
        "hidden_tokens_per_primary_mask": 64,
        "sampled_budget_allowlist_checks": 6144,
        "uniform_hidden_probability_for_every_token": "1/2",
        "uniform_pair_hidden_probability": str(Fraction(64 * 63, 128 * 127)),
        "J": verify_core_law("J"),
        "A": verify_core_law("A"),
        "E_leg_structure": verify_leg_bank_structure(),
        "production_bank_warning": "This script uses Python random.Random for checks. Preserve the historical Torch E_U bank and record production bank hashes separately.",
    }
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
