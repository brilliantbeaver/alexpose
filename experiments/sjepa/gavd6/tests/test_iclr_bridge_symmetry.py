"""Synthetic reflection/time algebra and source-held observability checks."""

import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from gavd6_sjepa.research_directions.iclr_bridge.symmetry_calibration import (
    ALGEBRA_ATOL, FULL_MIRROR_PAIRS, OBSERVABLE_PAIRS, PoseHistory,
    _algebra_fixture, motion_observables, paired_teacher_components,
    parity_project, reflect, reverse_time, run_calibration,
)


class SymmetryCalibrationTests(unittest.TestCase):
    def test_reflection_carries_confidence_and_validity_with_coordinates(self):
        history = _algebra_fixture()
        valid = history.valid.copy()
        valid[2, 11] = False
        xyz = history.xyz.copy()
        xyz[2, 11] = np.nan
        original = PoseHistory(xyz, history.confidence, valid, history.times)
        mirrored = reflect(original)
        for left, right in FULL_MIRROR_PAIRS:
            np.testing.assert_array_equal(mirrored.confidence[:, right], original.confidence[:, left])
            np.testing.assert_array_equal(mirrored.valid[:, right], original.valid[:, left])
            np.testing.assert_array_equal(mirrored.xyz[:, right, 1:], original.xyz[:, left, 1:])
            np.testing.assert_array_equal(mirrored.xyz[:, right, 0], -original.xyz[:, left, 0])
        restored = reflect(mirrored)
        for name in ("xyz", "confidence", "valid", "times"):
            np.testing.assert_array_equal(getattr(restored, name), getattr(original, name))
        self.assertFalse(original.xyz.flags.writeable)

    def test_reversal_changes_intervals_and_commutes_with_reflection(self):
        original = _algebra_fixture()
        reversed_history = reverse_time(original)
        np.testing.assert_array_equal(np.diff(reversed_history.times), np.diff(original.times)[::-1])
        for name in ("xyz", "confidence", "valid", "times"):
            np.testing.assert_array_equal(getattr(reverse_time(reversed_history), name), getattr(original, name))
            np.testing.assert_array_equal(getattr(reflect(reversed_history), name),
                                          getattr(reverse_time(reflect(original)), name))

    def test_wrong_reversal_intervals_change_speed_contrast(self):
        # Regression fixture: reversing positions but keeping irregular intervals
        # changes the measurement. The complete reversal retains the speed target.
        template = _algebra_fixture()
        rng = np.random.default_rng(913)
        xyz = np.cumsum(rng.normal(size=template.xyz.shape), axis=0)
        history = PoseHistory(xyz, template.confidence, template.valid, template.times)
        wrong = PoseHistory(history.xyz[::-1], history.confidence[::-1],
                            history.valid[::-1], history.times)
        target = motion_observables(history)["bilateral_speed_contrast"]
        self.assertAlmostEqual(motion_observables(reverse_time(history))["bilateral_speed_contrast"], target, places=12)
        self.assertGreater(abs(motion_observables(wrong)["bilateral_speed_contrast"] - target), 0.001)

    def test_invalid_coordinates_have_no_observable_influence(self):
        template = _algebra_fixture()
        valid = template.valid.copy()
        for left, _ in OBSERVABLE_PAIRS:
            valid[[2, 5], left] = False
        first, second = template.xyz.copy(), template.xyz.copy()
        first[~valid] = np.nan
        second[~valid] = 1e100
        a = PoseHistory(first, template.confidence, valid, template.times)
        b = PoseHistory(second, template.confidence, valid, template.times)
        self.assertEqual(motion_observables(a), motion_observables(b))
        self.assertTrue(all(n == 12 for n in motion_observables(a)["common_transition_counts"]))
        target = motion_observables(a)["bilateral_speed_contrast"]
        self.assertAlmostEqual(motion_observables(reflect(a))["bilateral_speed_contrast"], -target, places=12)
        self.assertAlmostEqual(motion_observables(reverse_time(a))["bilateral_speed_contrast"], target, places=12)

    def test_missing_support_and_invalid_schema_fail_explicitly(self):
        h = _algebra_fixture()
        with self.assertRaises(ValueError):
            motion_observables(PoseHistory(h.xyz, h.confidence, np.zeros_like(h.valid), h.times))
        with self.assertRaises(ValueError):
            PoseHistory(h.xyz, h.confidence, h.valid.astype(float), h.times)
        with self.assertRaises(ValueError):
            PoseHistory(h.xyz, h.confidence, h.valid, np.zeros_like(h.times))
        wrong = h.xyz.copy()
        wrong[0, 0] = np.nan
        with self.assertRaises(ValueError):
            PoseHistory(wrong, h.confidence, h.valid, h.times)

    def test_four_projectors_reconstruct_and_separate_function_components(self):
        history = _algebra_fixture()
        feature = lambda h: np.r_[h.xyz[0, [11, 12], :].ravel(),
                                  h.xyz[-1, [11, 12], :].ravel(), 1.0]
        projected = {(a, b): parity_project(feature, history, a, b)
                     for a in (-1, 1) for b in (-1, 1)}
        np.testing.assert_allclose(sum(projected.values()), feature(history), atol=ALGEBRA_ATOL, rtol=0)
        for (a, b), value in projected.items():
            for c in (-1, 1):
                for d in (-1, 1):
                    twice = parity_project(lambda h: parity_project(feature, h, a, b), history, c, d)
                    expected = value if (a, b) == (c, d) else np.zeros_like(value)
                    np.testing.assert_allclose(twice, expected, atol=ALGEBRA_ATOL, rtol=0)

    def test_pooled_teacher_components_require_two_arrays_and_explicit_identity(self):
        original = np.arange(12, dtype=float).reshape(3, 4)
        reflected = original[:, ::-1] + 0.5
        options = dict(window_ids=("a", "b", "c"), original_evidence_id="original-encoding",
                       mirrored_evidence_id="reflected-encoding", shared_basis_id="fixed-projection")
        parts = paired_teacher_components(original, reflected, **options)
        np.testing.assert_array_equal(parts["even"] + parts["odd"], original)
        np.testing.assert_array_equal(parts["even"] - parts["odd"], reflected)
        self.assertFalse(parts["provenance"]["external_lineage_verified_by_this_function"])
        with self.assertRaises(ValueError):
            paired_teacher_components(original[None], reflected[None], **options)
        with self.assertRaises(ValueError):
            paired_teacher_components(original, np.full_like(original, np.nan), **options)
        with self.assertRaises(ValueError):
            paired_teacher_components(original, reflected, **{**options, "window_ids": ("a", "a", "c")})
        with self.assertRaises(ValueError):
            paired_teacher_components(original, reflected, **{**options, "mirrored_evidence_id": "original-encoding"})

    def test_full_calibration_has_source_separation_and_recomputable_scores(self):
        result = run_calibration()
        self.assertEqual(result["status"], "passed")
        self.assertEqual(result["zero_feature_control"]["feature_variance"], 0)
        rows = result["observable_parities"]
        self.assertEqual({(r["reflection_sign"], r["time_sign"]) for r in rows},
                         {(1, 1), (1, -1), (-1, 1), (-1, -1)})
        fixture = result["temporal_observability"]
        self.assertFalse(set(fixture["train_sources"]) & set(fixture["test_sources"]))
        y = np.array(fixture["test_targets"])
        for row in fixture["scores"]:
            self.assertEqual(set(row["fit_sources"]), set(fixture["train_sources"]))
            self.assertFalse(set(row["fit_clip_ids"]) & set(fixture["test_clip_ids"]))
            expected = 1 - np.mean((y - row["test_predictions"])**2) / np.mean((y - fixture["training_target_mean"])**2)
            self.assertAlmostEqual(expected, row["r2_training_mean_reference"], places=12)

    def test_saved_calibration_roundtrip_and_no_overwrite(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "calibration.json"
            result = run_calibration(path)
            self.assertEqual(json.loads(path.read_text()), result)
            before = path.read_bytes()
            with self.assertRaises(FileExistsError):
                run_calibration(path)
            self.assertEqual(path.read_bytes(), before)


if __name__ == "__main__":
    unittest.main()
