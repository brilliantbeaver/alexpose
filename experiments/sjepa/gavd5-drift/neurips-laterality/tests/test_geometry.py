"""Contract tests for the paired-valid laterality geometry."""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np


SUITE_ROOT = Path(__file__).resolve().parents[1]
if str(SUITE_ROOT) not in sys.path:
    sys.path.insert(0, str(SUITE_ROOT))

from laterality.geometry import (  # noqa: E402
    FULL_MIRROR_PAIRS,
    anatomical_mirror,
    paired_valid_target,
)


TARGET_PAIRS = (
    (11, 12),
    (23, 24),
    (25, 26),
    (27, 28),
    (29, 30),
    (31, 32),
)


def asymmetric_motion(frames: int = 18) -> tuple[np.ndarray, np.ndarray]:
    """Make a non-degenerate gait-like fixture with stronger left motion."""
    time = np.linspace(0.0, 1.0, frames, dtype=np.float64)
    xyz = np.zeros((frames, 33, 3), dtype=np.float64)
    for pair_index, (left, right) in enumerate(TARGET_PAIRS):
        phase = 0.17 * pair_index
        left_amplitude = 1.2 + 0.12 * pair_index
        right_amplitude = 0.35 + 0.04 * pair_index
        xyz[:, left, 0] = left_amplitude * time
        xyz[:, right, 0] = right_amplitude * time
        xyz[:, left, 1] = 0.20 * np.sin(2.0 * np.pi * time + phase)
        xyz[:, right, 1] = 0.07 * np.sin(2.0 * np.pi * time + phase)
        xyz[:, left, 2] = 0.05 * np.cos(2.0 * np.pi * time + phase)
        xyz[:, right, 2] = 0.02 * np.cos(2.0 * np.pi * time + phase)
    return xyz, np.ones((frames, 33), dtype=bool)


def target_value(xyz: np.ndarray, valid: np.ndarray) -> float:
    result = paired_valid_target(
        xyz,
        valid,
        pairs=TARGET_PAIRS,
        minimum_common_transitions_per_pair=4,
        minimum_usable_pairs=3,
        epsilon=1e-8,
    )
    if result.value is None:
        raise AssertionError("the fully usable synthetic fixture produced no target")
    return float(result.value)


class MirrorGeometryTests(unittest.TestCase):
    def test_full_mirror_mapping_covers_every_bilateral_landmark(self):
        expected = {
            (1, 4),
            (2, 5),
            (3, 6),
            (7, 8),
            (9, 10),
            (11, 12),
            (13, 14),
            (15, 16),
            (17, 18),
            (19, 20),
            (21, 22),
            (23, 24),
            (25, 26),
            (27, 28),
            (29, 30),
            (31, 32),
        }
        self.assertEqual({tuple(pair) for pair in FULL_MIRROR_PAIRS}, expected)

    def test_mirror_is_an_involution_for_coordinates_and_validity(self):
        rng = np.random.default_rng(20260904)
        xyz = rng.normal(size=(13, 33, 3))
        valid = rng.random((13, 33)) > 0.23
        xyz_before = xyz.copy()
        valid_before = valid.copy()

        mirrored_xyz, mirrored_valid = anatomical_mirror(xyz, valid=valid)
        restored_xyz, restored_valid = anatomical_mirror(
            mirrored_xyz, valid=mirrored_valid
        )

        np.testing.assert_allclose(restored_xyz, xyz, rtol=0.0, atol=1e-12)
        np.testing.assert_array_equal(restored_valid, valid)
        np.testing.assert_array_equal(xyz, xyz_before)
        np.testing.assert_array_equal(valid, valid_before)

    def test_mirror_negates_the_paired_valid_target(self):
        xyz, valid = asymmetric_motion()
        original = target_value(xyz, valid)
        mirrored_xyz, mirrored_valid = anatomical_mirror(xyz, valid=valid)
        mirrored = target_value(mirrored_xyz, mirrored_valid)

        self.assertGreater(abs(original), 0.1, "antisymmetry test must not be vacuous")
        self.assertAlmostEqual(mirrored, -original, delta=1e-10)


class PairedValidTargetTests(unittest.TestCase):
    def test_invalid_coordinate_values_cannot_change_target(self):
        xyz, valid = asymmetric_motion()
        # Each invalid time removes adjacent transitions, but enough common paired
        # transitions remain for all five registered motion pairs.
        for pair_index, (left, right) in enumerate(TARGET_PAIRS):
            valid[3 + pair_index % 2, left] = False
            valid[10 + pair_index % 3, right] = False

        expected = target_value(xyz, valid)
        sentinel = xyz.copy()
        sentinel[~valid] = np.array([1e12, -7e11, 3e11])
        with_nan = xyz.copy()
        with_nan[~valid] = np.nan

        self.assertAlmostEqual(target_value(sentinel, valid), expected, delta=1e-12)
        self.assertAlmostEqual(target_value(with_nan, valid), expected, delta=1e-12)

    def test_normalized_target_is_scale_invariant(self):
        xyz, valid = asymmetric_motion()
        baseline = target_value(xyz, valid)

        for scale in (0.125, 7.5, 100.0):
            with self.subTest(scale=scale):
                self.assertAlmostEqual(
                    target_value(scale * xyz, valid), baseline, delta=1e-6
                )


if __name__ == "__main__":
    unittest.main()
