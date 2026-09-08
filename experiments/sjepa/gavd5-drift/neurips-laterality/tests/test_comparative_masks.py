"""Scientific contracts for label-blind scattered and intact structured masks."""

from dataclasses import replace
import inspect
from pathlib import Path
import sys
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from laterality.geometry import FULL_MIRROR_PAIRS
from laterality_extensions.comparative_masks import (
    ANATOMICAL_EDGES, GAIT_JOINTS, InfeasibleMaskBudget, MaskBudget, MaskPolicy,
    connected_region_bank, coverage_summary, eligible_landmarks, fixed_subset,
    motion_scores, reflect_sample, sample_mask, shared_scattered_count,
)


def fixture(blocks=6):
    time = np.arange(blocks * 4, dtype=float)
    xyz = np.zeros((len(time), 33, 3))
    xyz[:, :, 1] = np.arange(33) / 33
    xyz[:, 27, 0] = time * 0.1
    xyz[:, 28, 0] = time * 0.02
    return xyz, np.ones((blocks, 33), dtype=bool)


def draw(policy, budget, *, seed=3, blocks=6):
    xyz, valid = fixture(blocks)
    return sample_mask(xyz, valid, policy, budget, np.random.default_rng(seed))


def check_scattered_reproduces_and_preserves_valid_context(name):
    xyz, valid = fixture()
    valid[1, 27] = False
    xyz[4:8, 27] = np.nan
    args = xyz, valid, MaskPolicy(name), MaskBudget(hidden_count=17)
    first = sample_mask(*args, np.random.default_rng(1))
    second = sample_mask(*args, np.random.default_rng(1))
    assert np.array_equal(first.mask, second.mask)
    assert first.mask.sum() == 17
    assert not (first.mask & ~valid).any()
    assert len(np.unique(first.target_indices, axis=0)) == 17
    assert np.array_equal(first.target_indices, np.argwhere(first.mask))
    assert first.context_count == int(valid.sum()) - 17
    assert first.coverage["naturally_missing_tokens"] == 1


def test_fixed_subsets_use_separate_selection_seed_and_different_mask_draws():
    policy = MaskPolicy("random_subset", subset_seed=21)
    first = draw(policy, MaskBudget(hidden_count=17), seed=1)
    second = draw(policy, MaskBudget(hidden_count=17), seed=2)
    assert len(fixed_subset(policy)) == 12
    assert first.coverage["eligible_landmarks"] == second.coverage["eligible_landmarks"]
    assert not np.array_equal(first.mask, second.mask)
    assert set(first.coverage["hidden_landmarks"]) <= set(fixed_subset(policy))
    pairs = fixed_subset(replace(policy, subset_kind="bilateral_pairs"))
    assert all((left in pairs) == (right in pairs) for left, right in FULL_MIRROR_PAIRS)
    assert len(pairs) == 12 and 0 not in pairs


def test_soft_preference_does_not_exclude_non_gait_and_falls_back():
    xyz, valid = fixture()
    rng = np.random.default_rng(42)
    seen = np.zeros_like(valid)
    for _ in range(40):
        seen |= sample_mask(xyz, valid, MaskPolicy("soft_gait"), MaskBudget(hidden_count=20), rng).mask
    assert seen[:, 0].any() and seen[:, 15].any()
    valid[:, GAIT_JOINTS] = False
    result = sample_mask(xyz, valid, MaskPolicy("soft_gait"), MaskBudget(hidden_count=12), rng)
    assert result.coverage["gait_fallback_uniform"]
    with unittest.TestCase().assertRaisesRegex(ValueError, "uniform coverage"):
        draw(MaskPolicy("soft_gait", gait_weight=1), MaskBudget(hidden_count=12))


def test_motion_increases_hiding_probability_for_reliable_movement():
    xyz, valid = fixture()
    rng = np.random.default_rng(74)
    counts = np.zeros_like(valid, dtype=int)
    for _ in range(400):
        counts += sample_mask(xyz, valid, MaskPolicy("motion"), MaskBudget(hidden_count=1), rng).mask
    # A repeated sampling property of the configured probabilities; this says
    # nothing about whether any trained representation will improve.
    assert counts[:, 27].sum() > 2 * counts[:, 28].sum()
    assert counts[:, 27].sum() > 10 * counts[:, 15].sum()
    assert counts[:, 15].sum() > 0


def test_motion_units_stationary_missing_and_isolated_jump():
    xyz, valid = fixture()
    scores, note = motion_scores(xyz, valid)
    seconds, seconds_note = motion_scores(xyz, valid, timestamps=np.arange(len(xyz)) / 10)
    np.testing.assert_allclose(seconds, scores * 10)
    assert note["motion_units"] == "coordinate units / prepared step"
    assert seconds_note["motion_units"] == "coordinate units / second"
    xyz[:] = 0
    xyz[9, 15, 0] = 1e6
    jump_scores, _ = motion_scores(xyz, valid)
    assert not jump_scores.any()  # One erroneous observation cannot dominate the sampler.
    stationary = sample_mask(xyz * 0, valid, MaskPolicy("motion"), MaskBudget(hidden_count=12), np.random.default_rng(1))
    assert stationary.coverage["motion_fallback_uniform"]
    observed = np.repeat(valid, 4, axis=0)
    observed[4, 27] = False
    with unittest.TestCase().assertRaisesRegex(ValueError, "every constituent observation"):
        sample_mask(xyz, valid, MaskPolicy("motion"), MaskBudget(hidden_count=12), np.random.default_rng(1), observation_valid=observed)
    valid[1, 27] = False
    xyz[4, 27] = np.nan
    result = sample_mask(xyz, valid, MaskPolicy("motion"), MaskBudget(hidden_count=12), np.random.default_rng(1), observation_valid=observed)
    assert not result.mask[1, 27]


def test_whole_trajectories_keep_their_shape_and_exact_feasible_counts():
    xyz, valid = fixture(blocks=16)
    result = sample_mask(xyz, valid, MaskPolicy("whole_trajectory"), MaskBudget(hidden_count=96), np.random.default_rng(6))
    assert result.mask.sum() == 96
    assert np.all(result.mask == result.mask[0])
    assert result.coverage["landmark_count"] == 6
    with unittest.TestCase().assertRaisesRegex(InfeasibleMaskBudget, "intact"):
        sample_mask(xyz, valid, MaskPolicy("whole_trajectory"), MaskBudget(hidden_count=95), np.random.default_rng(6))


def test_structured_missingness_can_create_ragged_counts_without_trimming():
    xyz, valid = fixture()
    policy, budget = MaskPolicy("whole_trajectory", eligibility="gait"), MaskBudget(trajectories=1)
    # All candidate trajectories in the second clip have fewer valid blocks.
    first = sample_mask(xyz, valid, policy, budget, np.random.default_rng(7))
    valid[2, GAIT_JOINTS] = False
    second = sample_mask(xyz, valid, policy, budget, np.random.default_rng(7))
    assert len(first.target_indices) == 6 and len(second.target_indices) == 5
    assert second.coverage["interval_blocks"] == 6
    joint = second.coverage["selected_landmarks"][0]
    assert np.array_equal(second.mask[:, joint], valid[:, joint])


def is_connected(region):
    seen, remaining = {region[0]}, set(region[1:])
    while remaining:
        neighbors = {b for a, b in ANATOMICAL_EDGES if a in seen} | {a for a, b in ANATOMICAL_EDGES if b in seen}
        new = neighbors & remaining
        if not new:
            return False
        seen.update(new)
        remaining.difference_update(new)
    return True


def check_connected_bank_covers_declared_pool_and_is_anatomical(pool):
    regions = connected_region_bank(pool, 3)
    assert set().union(*map(set, regions)) == set(pool)
    assert all(is_connected(region) for region in regions)
    assert any(27 in region for region in regions) and any(28 in region for region in regions)
    result = draw(MaskPolicy("connected_region", eligibility="gait" if pool == GAIT_JOINTS else "all"), MaskBudget(region_size=3, interval_blocks=2, hidden_count=6))
    selected = result.coverage["selected_landmarks"]
    assert is_connected(selected)
    assert result.mask.sum() == 6
    start = result.coverage["interval_start"]
    assert not result.mask[:start].any() and not result.mask[start + 2:].any()
    assert result.mask[start:start + 2, selected].all()


def test_connected_bank_is_closed_under_reflection():
    permutation = np.arange(33)
    for left, right in FULL_MIRROR_PAIRS:
        permutation[left], permutation[right] = right, left
    bank = connected_region_bank(tuple(range(33)), 3)
    assert all(tuple(sorted(permutation[list(region)])) in bank for region in bank)


def test_interior_gap_preserves_both_sides_of_time_and_rejects_impossible_budget():
    result = draw(MaskPolicy("temporal_gap"), MaskBudget(interval_blocks=2, hidden_count=66), blocks=16)
    start = result.coverage["interval_start"]
    assert start > 0 and start + 2 < 16
    assert result.mask[start:start + 2].all()
    assert not result.mask[:start].any() and not result.mask[start + 2:].any()
    with unittest.TestCase().assertRaisesRegex(InfeasibleMaskBudget, "feasible observed counts"):
        draw(MaskPolicy("temporal_gap"), MaskBudget(interval_blocks=3, hidden_count=96), blocks=16)
    trajectories = {16 * n for n in range(1, 33)}
    complete_times = {33 * n for n in range(1, 16)}
    assert not trajectories & complete_times


def test_count_matching_records_lower_common_count_and_equal_reference():
    xyz, valid = fixture()
    policies = [MaskPolicy("gait"), MaskPolicy("uniform"), MaskPolicy("random_subset")]
    count = shared_scattered_count(valid, policies, requested_count=100)
    assert count == 72
    results = [sample_mask(xyz, valid, policy, MaskBudget(hidden_count=count), np.random.default_rng(3)) for policy in policies]
    assert {result.mask.sum() for result in results} == {72}
    structured = draw(MaskPolicy("whole_trajectory"), MaskBudget(trajectories=2))
    reference = draw(MaskPolicy("uniform"), MaskBudget(hidden_count=structured.coverage["hidden_tokens"]))
    assert structured.mask.sum() == reference.mask.sum()


def test_reflection_exchanges_coordinates_validity_and_targets_together():
    xyz, valid = fixture()
    valid[0, 27] = False
    mask = sample_mask(xyz, valid, MaskPolicy("gait"), MaskBudget(hidden_count=10), np.random.default_rng(1)).mask
    mirrored, mirrored_valid, mirrored_mask = reflect_sample(xyz, valid, mask)
    np.testing.assert_allclose(mirrored[:, 28, 0], -xyz[:, 27, 0])
    np.testing.assert_array_equal(mirrored_valid[:, 28], valid[:, 27])
    np.testing.assert_array_equal(mirrored_mask[:, 28], mask[:, 27])
    restored = reflect_sample(mirrored, mirrored_valid, mirrored_mask)
    for original, returned in zip((xyz, valid, mask), restored):
        np.testing.assert_array_equal(original, returned)


def test_label_blind_interface_and_dedicated_generator_do_not_use_global_state():
    assert not {"labels", "targets", "conditions", "clinical_condition"} & set(inspect.signature(sample_mask).parameters)
    np.random.seed(7)
    before = np.random.get_state()
    draw(MaskPolicy("random_subset"), MaskBudget(hidden_count=12))
    after = np.random.get_state()
    assert all(np.array_equal(a, b) for a, b in zip(before, after))


def test_invalid_empty_and_all_hidden_requests_fail():
    xyz, valid = fixture()
    with unittest.TestCase().assertRaises(InfeasibleMaskBudget):
        sample_mask(xyz, valid, MaskPolicy("uniform"), MaskBudget(hidden_count=valid.size), np.random.default_rng(1))
    with unittest.TestCase().assertRaisesRegex(ValueError, "Naturally missing"):
        invalid = valid.copy()
        invalid[0, 0] = False
        coverage_summary(valid, invalid)
    with unittest.TestCase().assertRaisesRegex(ValueError, "all landmarks"):
        draw(MaskPolicy("temporal_gap", eligibility="gait"), MaskBudget(interval_blocks=1))
    with unittest.TestCase().assertRaises(InfeasibleMaskBudget):
        draw(MaskPolicy("temporal_gap"), MaskBudget(interval_blocks=6))


def test_both_sides_of_body_can_be_selected():
    xyz, valid = fixture()
    left_seen, right_seen = False, False
    for seed in range(40):
        result = sample_mask(xyz, valid, MaskPolicy("whole_trajectory"), MaskBudget(trajectories=1), np.random.default_rng(seed))
        left_seen |= result.coverage["left_hidden_tokens"] > 0
        right_seen |= result.coverage["right_hidden_tokens"] > 0
    assert left_seen and right_seen


class ComparativeMaskParameterizedTests(unittest.TestCase):
    def test_scattered_families(self):
        for name in ("gait", "uniform", "random_subset", "soft_gait", "motion"):
            with self.subTest(policy=name):
                check_scattered_reproduces_and_preserves_valid_context(name)

    def test_anatomical_region_pools(self):
        for pool in (tuple(range(33)), GAIT_JOINTS):
            with self.subTest(pool=pool):
                check_connected_bank_covers_declared_pool_and_is_anatomical(pool)


def load_tests(loader, standard_tests, pattern):
    """Use the project's unittest runner without adding a test dependency."""
    suite = unittest.TestSuite(standard_tests)
    suite.addTests(unittest.FunctionTestCase(function)
                   for name, function in sorted(globals().items())
                   if name.startswith("test_") and callable(function))
    return suite


if __name__ == "__main__":
    unittest.main()
