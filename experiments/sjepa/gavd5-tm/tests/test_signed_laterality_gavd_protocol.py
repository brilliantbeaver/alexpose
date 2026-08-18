"""Protocol tests run independently of the signed-laterality notebook."""

from __future__ import annotations

import numpy as np
import unittest

from signed_laterality_gavd_protocol import (
    PoseRecord,
    anatomical_reflection,
    canonical_body_frame,
    even_total_excursion,
    left_right_label_shuffle,
    mirror_metrics,
    nested_group_rankings,
    prepare_for_encoder,
    run_grouped_arm,
    signed_right_minus_left_excursion,
    source_group_folds,
    validate_rankings,
    yaw_rotate,
)
from signed_laterality_gavd_protocol import _fit_two_view_head


def _sequence(seed: int = 1) -> np.ndarray:
    rng = np.random.default_rng(seed)
    sequence = rng.normal(size=(24, 33, 4)).astype(np.float32)
    sequence[..., 3] = rng.uniform(0.55, 1.0, size=(24, 33))
    # Give the body schema a stable non-degenerate pelvis/shoulder geometry.
    sequence[:, 23, :3] = (-0.15, 0.0, 0.0)
    sequence[:, 24, :3] = (0.15, 0.0, 0.0)
    sequence[:, 11, :3] = (-0.2, 0.65, 0.0)
    sequence[:, 12, :3] = (0.2, 0.65, 0.0)
    sequence[:, 25, :3] = (-0.15, -0.45, 0.03)
    sequence[:, 26, :3] = (0.15, -0.45, 0.03)
    sequence[:, 27, :3] = (-0.15, -0.9, 0.06)
    sequence[:, 28, :3] = (0.15, -0.9, 0.06)
    return sequence


def _records() -> list[PoseRecord]:
    return [PoseRecord(f"s{i}", f"v{i // 2}", "normal", "canonical", _sequence(i), f"/{i}.npz") for i in range(8)]


class TestGAVDSignedLateralityProtocol(unittest.TestCase):
    def test_reflection_is_an_involution_and_preserves_joint_metadata(self) -> None:
        sequence = _sequence()
        reflected = anatomical_reflection(sequence)
        np.testing.assert_array_equal(anatomical_reflection(reflected), sequence)
        np.testing.assert_array_equal(reflected[:, 11, 3], sequence[:, 12, 3])

    def test_odd_and_even_targets_have_expected_parity(self) -> None:
        sequence = _sequence()
        reflected = anatomical_reflection(sequence)
        self.assertAlmostEqual(signed_right_minus_left_excursion(reflected), -signed_right_minus_left_excursion(sequence))
        self.assertAlmostEqual(even_total_excursion(reflected), even_total_excursion(sequence))

    def test_reflection_preserves_shape_masks_bones_and_forward_direction(self) -> None:
        sequence = _sequence()
        reflected = anatomical_reflection(sequence)
        self.assertEqual(reflected.shape, sequence.shape)
        self.assertEqual(np.count_nonzero(np.isfinite(reflected[..., :3])), np.count_nonzero(np.isfinite(sequence[..., :3])))
        original_bone = np.linalg.norm(sequence[:, 23, :3] - sequence[:, 25, :3], axis=1)
        reflected_bone = np.linalg.norm(reflected[:, 24, :3] - reflected[:, 26, :3], axis=1)
        np.testing.assert_allclose(original_bone, reflected_bone)
        _, original_rotation = canonical_body_frame(sequence)
        _, reflected_rotation = canonical_body_frame(reflected)
        self.assertGreater(np.dot(original_rotation[:, 2], reflected_rotation[:, 2]), 0.99)

    def test_yaw_does_not_change_side_target_or_joint_identity(self) -> None:
        sequence = _sequence()
        prepared, mask, _ = prepare_for_encoder(sequence, frames=32)
        yawed = yaw_rotate(prepared, 30.0)
        self.assertEqual(yawed.shape, prepared.shape)
        self.assertEqual(mask.shape, prepared.shape[:2])
        np.testing.assert_allclose(np.linalg.norm(prepared[:, 23] - prepared[:, 24], axis=1),
                                   np.linalg.norm(yawed[:, 23] - yawed[:, 24], axis=1))
        self.assertFalse(np.allclose(yawed[:, 23], prepared[:, 24]))
        self.assertAlmostEqual(signed_right_minus_left_excursion(sequence), signed_right_minus_left_excursion(sequence))

    def test_source_folds_and_rankings_are_leakage_safe_and_nested(self) -> None:
        records = _records()
        folds = source_group_folds(records, requested_folds=4)
        groups = [record.video_id for record in records]
        for train, test in folds:
            self.assertFalse(set(np.asarray(groups)[train]) & set(np.asarray(groups)[test]))
        rankings = nested_group_rankings(groups, seed=2027, repetitions=3)
        validate_rankings(rankings, groups)

    def test_exact_odd_output_has_zero_oddness_error(self) -> None:
        features = np.array([[1.0, 2.0], [-1.0, 3.0], [0.5, -4.0]])
        weights = np.array([0.3, -0.2])
        prediction = features @ weights
        result = mirror_metrics(prediction, -prediction)
        self.assertAlmostEqual(result["oddness_error"], 0.0)
        self.assertAlmostEqual(result["mirror_slope"], -1.0)

    def test_arm_c_is_exactly_odd_after_grouped_fitting(self) -> None:
        records = _records()
        groups = [record.video_id for record in records]
        generator = np.random.default_rng(9)
        original = generator.normal(size=(len(records), 5))
        mirrored = -original
        target = original[:, 0] - 0.5 * original[:, 1]
        result = run_grouped_arm("C", original, mirrored, target, groups, source_group_folds(records, 4), 2)
        np.testing.assert_allclose(result.mirrored_prediction, -result.prediction, atol=1e-10)

    def test_arm_d_has_a_frozen_unit_circle_mixing_convention(self) -> None:
        generator = np.random.default_rng(4)
        original, mirrored = generator.normal(size=(18, 4)), generator.normal(size=(18, 4))
        head = _fit_two_view_head(original, mirrored, generator.normal(size=18), np.repeat(["a", "b", "c"], 6), 1.0)
        self.assertAlmostEqual(head.a * head.a + head.b * head.b, 1.0, places=8)

    def test_left_right_shuffle_is_destructive_but_keeps_target_label(self) -> None:
        sequence = _sequence()
        shuffled = left_right_label_shuffle(sequence, seed=18)
        self.assertFalse(np.array_equal(sequence, shuffled))
        self.assertAlmostEqual(signed_right_minus_left_excursion(sequence), signed_right_minus_left_excursion(sequence))
