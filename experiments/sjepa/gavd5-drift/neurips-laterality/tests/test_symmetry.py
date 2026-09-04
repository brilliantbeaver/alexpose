"""Adversarial tests for strict representation-equivariance diagnostics."""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np


SUITE_ROOT = Path(__file__).resolve().parents[1]
if str(SUITE_ROOT) not in sys.path:
    sys.path.insert(0, str(SUITE_ROOT))

from laterality.symmetry import (  # noqa: E402
    per_seed_source_balanced_mean,
    source_balanced_mean,
    strict_equivariance_error,
    strict_equivariance_errors,
    swap_token_joints,
    swap_token_validity,
)


class StrictEquivarianceTests(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(20260904)
        self.original = rng.normal(size=(7, 33, 5))
        self.valid = rng.random((7, 33)) > 0.18

    def test_exact_joint_swap_has_zero_error(self):
        mirrored = swap_token_joints(self.original)
        mirrored_valid = swap_token_validity(self.valid)
        result = strict_equivariance_error(
            self.original, mirrored, self.valid, mirrored_valid
        )

        self.assertEqual(result.value, 0.0)
        self.assertEqual(result.residual_energy, 0.0)
        self.assertGreater(result.representation_energy, 0.0)
        self.assertEqual(result.common_token_count, int(self.valid.sum()))
        self.assertEqual(result.channel_count, self.original.shape[-1])

    def test_energy_ratio_is_scale_invariant(self):
        rng = np.random.default_rng(91)
        mirrored_valid = swap_token_validity(self.valid)
        mirrored = swap_token_joints(self.original)
        mirrored[mirrored_valid] += rng.normal(
            scale=0.35, size=mirrored[mirrored_valid].shape
        )
        baseline = strict_equivariance_error(
            self.original, mirrored, self.valid, mirrored_valid
        ).value
        self.assertGreater(baseline, 0.0)

        for scale in (1e-6, 0.01, 0.5, 17.0, 1_000.0, 1e6):
            with self.subTest(scale=scale):
                scaled = strict_equivariance_error(
                    scale * self.original,
                    scale * mirrored,
                    self.valid,
                    mirrored_valid,
                ).value
                self.assertAlmostEqual(scaled, baseline, delta=2e-14)

    def test_values_outside_common_valid_support_cannot_affect_error(self):
        original_valid = self.valid.copy()
        mirrored_valid = swap_token_validity(original_valid)
        # Remove different tokens from the two views so common support is the
        # intersection after applying the joint permutation.
        original_valid[1, 11] = False
        original_valid[3, 27] = False
        mirrored_valid[2, 12] = False
        mirrored_valid[5, 28] = False

        mirrored = swap_token_joints(self.original)
        mirrored += 0.1
        expected = strict_equivariance_error(
            self.original, mirrored, original_valid, mirrored_valid
        )

        common_mirrored = mirrored_valid & swap_token_validity(original_valid)
        common_original = swap_token_validity(common_mirrored)
        poisoned_original = self.original.copy()
        poisoned_mirrored = mirrored.copy()
        poisoned_original[~common_original] = np.nan
        poisoned_mirrored[~common_mirrored] = np.array([1e100] * self.original.shape[-1])
        actual = strict_equivariance_error(
            poisoned_original,
            poisoned_mirrored,
            original_valid,
            mirrored_valid,
        )

        self.assertEqual(actual.common_token_count, expected.common_token_count)
        self.assertAlmostEqual(actual.value, expected.value, delta=1e-15)
        self.assertAlmostEqual(
            actual.residual_energy, expected.residual_energy, delta=1e-12
        )
        self.assertAlmostEqual(
            actual.representation_energy,
            expected.representation_energy,
            delta=1e-10,
        )

    def test_wrong_identity_joint_action_is_detectably_nonzero(self):
        # A one-hot identity channel makes every swapped joint orthogonal to
        # its partner, so omitting S cannot be hidden by shared channel energy.
        original = np.repeat(np.eye(33, dtype=np.float64)[None, :, :], 4, axis=0)
        valid = np.ones((4, 33), dtype=bool)

        wrong_mirrored = original.copy()  # E(Mx)=E(x), not S E(x)
        result = strict_equivariance_error(
            original, wrong_mirrored, valid, valid
        )
        self.assertGreater(result.value, 0.01)

    def test_unrelated_equal_energy_representations_are_near_one(self):
        rng = np.random.default_rng(7)
        original = rng.normal(size=(100, 33, 24))
        mirrored = rng.normal(size=original.shape)
        valid = np.ones(original.shape[:-1], dtype=bool)
        result = strict_equivariance_error(original, mirrored, valid, valid)
        self.assertAlmostEqual(result.value, 1.0, delta=0.04)

    def test_common_valid_nonfinite_and_collapsed_representations_are_rejected(self):
        valid = np.ones((2, 33), dtype=bool)
        original = np.ones((2, 33, 2), dtype=np.float64)
        mirrored = swap_token_joints(original)
        mirrored[0, 0, 0] = np.nan
        with self.assertRaisesRegex(ValueError, "common-valid.*non-finite"):
            strict_equivariance_error(original, mirrored, valid, valid)

        zeros = np.zeros_like(original)
        with self.assertRaisesRegex(ValueError, "energy is too small"):
            strict_equivariance_error(zeros, zeros, valid, valid)

    def test_batched_api_preserves_sequence_results(self):
        mirrored = swap_token_joints(self.original)
        originals = np.stack((self.original, 2.0 * self.original))
        mirrors = np.stack((mirrored, 2.0 * mirrored))
        validity = np.stack((self.valid, self.valid))
        results = strict_equivariance_errors(
            originals, mirrors, validity, swap_token_validity(validity)
        )
        self.assertEqual(len(results), 2)
        self.assertEqual([result.value for result in results], [0.0, 0.0])


class EquivarianceAggregationTests(unittest.TestCase):
    def test_source_balancing_prevents_long_sources_from_dominating(self):
        errors = [0.0, 1.0, 2.0, 9.0]
        sources = ["long", "long", "long", "short"]
        # Mean within long=1, mean within short=9, then equal-source mean=5.
        self.assertEqual(source_balanced_mean(errors, sources), 5.0)
        self.assertNotEqual(source_balanced_mean(errors, sources), np.mean(errors))

    def test_per_seed_aggregation_is_separate_and_requires_source_coverage(self):
        errors = [0.0, 2.0, 4.0, 6.0, 1.0, 3.0, 5.0, 7.0]
        sources = ["a", "a", "b", "b"] * 2
        seeds = ["41"] * 4 + ["42"] * 4
        self.assertEqual(
            per_seed_source_balanced_mean(errors, sources, seeds),
            {"41": 3.0, "42": 4.0},
        )

        with self.assertRaisesRegex(ValueError, "same source"):
            per_seed_source_balanced_mean(
                [0.0, 1.0, 2.0], ["a", "b", "a"], ["41", "41", "42"]
            )


if __name__ == "__main__":
    unittest.main()
