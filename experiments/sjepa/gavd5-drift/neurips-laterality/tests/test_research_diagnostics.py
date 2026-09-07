from pathlib import Path
import sys
import unittest

import numpy as np

SUITE_ROOT = Path(__file__).resolve().parents[1]
if str(SUITE_ROOT) not in sys.path:
    sys.path.insert(0, str(SUITE_ROOT))

from laterality_extensions.diagnostics import (
    reconstruct_processed_target,
    summarize_reconstruction,
    temporal_pooling_example,
)
from laterality.geometry import anatomical_mirror


class ResearchDiagnosticsTests(unittest.TestCase):
    def test_pooling_example_has_equal_means_and_opposite_motion(self):
        result = temporal_pooling_example()
        np.testing.assert_array_equal(result["mean_position"], np.full((2, 2), 0.5))
        np.testing.assert_array_equal(result["motion_contrast"], [1.0, -1.0])

    def test_time_reversal_preserves_speed_in_the_example(self):
        trajectories = temporal_pooling_example()["trajectories"]
        original = np.median(np.abs(np.diff(trajectories, axis=1)), axis=1)
        reversed_speed = np.median(np.abs(np.diff(trajectories[:, ::-1], axis=1)), axis=1)
        np.testing.assert_array_equal(original, reversed_speed)

    def test_source_weighting_does_not_treat_repeated_clips_as_new_sources(self):
        result = summarize_reconstruction(
            np.array([0.0, 0.0, 1.0]),
            np.array([0.0, 0.0, 0.0]),
            np.array(["source_a", "source_a", "source_b"]),
        )
        self.assertAlmostEqual(result["weighted_mae"], 0.5)
        self.assertAlmostEqual(result["direct_agreement_r2"], -1.0)

    def test_missing_recomputations_are_counted(self):
        result = summarize_reconstruction(np.array([0.0, 1.0, 2.0]), np.array([0.0, 1.0, np.nan]), np.array(["a", "b", "c"]))
        self.assertEqual(result["finite_sequences"], 2)
        self.assertEqual(result["finite_sources"], 2)
        self.assertEqual(result["excluded_sequences"], 1)
        self.assertEqual(result["direct_agreement_r2"], 1.0)

    def test_constant_target_has_undefined_r2(self):
        result = summarize_reconstruction(np.ones(3), np.ones(3), np.array(["a", "b", "c"]))
        self.assertTrue(np.isnan(result["direct_agreement_r2"]))
        self.assertTrue(np.isnan(result["weighted_correlation"]))

    def test_invalid_inputs_are_rejected(self):
        with self.assertRaises(ValueError):
            summarize_reconstruction(np.array([1.0]), np.array([1.0, 2.0]), np.array(["a"]))
        with self.assertRaises(ValueError):
            summarize_reconstruction(np.array([1.0]), np.array([np.nan]), np.array(["a"]))
        with self.assertRaises(ValueError):
            summarize_reconstruction(np.array([1.0]), np.array([1.0]), np.array([""]))

    def test_reconstruction_retains_mirror_rule_without_mutating_inputs(self):
        xyz = np.zeros((1, 16, 33, 3), dtype=np.float64)
        xyz[0, :, 11, 0] = np.arange(16) * 2.0
        xyz[0, :, 12, 0] = np.arange(16)
        valid = np.ones(xyz.shape[:-1], dtype=bool)
        config = {"pairs": [[11, 12]], "minimum_common_transitions_per_pair": 8, "minimum_usable_pairs": 1, "epsilon": 1e-8}
        saved = xyz.copy()
        original = reconstruct_processed_target(xyz, valid, config)
        mirrored_xyz, mirrored_valid = anatomical_mirror(xyz, valid)
        mirrored = reconstruct_processed_target(mirrored_xyz, mirrored_valid, config)
        np.testing.assert_allclose(original, -mirrored)
        np.testing.assert_array_equal(xyz, saved)
        self.assertAlmostEqual(original[0], 1.0 / 3.0)

    def test_reconstruction_retains_missing_result(self):
        xyz = np.zeros((1, 16, 33, 3))
        valid = np.zeros(xyz.shape[:-1], dtype=bool)
        config = {"pairs": [[11, 12]], "minimum_common_transitions_per_pair": 8, "minimum_usable_pairs": 1, "epsilon": 1e-8}
        self.assertTrue(np.isnan(reconstruct_processed_target(xyz, valid, config)[0]))


if __name__ == "__main__":
    unittest.main()
