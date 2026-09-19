"""CPU calibration controls must separate fit labels from inference geometry."""
import copy
import importlib.util
import json
from pathlib import Path
import unittest

import numpy as np

from gavd6_sjepa.research_directions.synthetic_training_v2.contracts import TrackBundle
from gavd6_sjepa.research_directions.synthetic_training_v2.data import fixture_bundle, normalize_inputs


SCRIPT = Path(__file__).resolve().parents[2] / "scripts/research_directions/synthetic_training_v2/diagnostics/calibration.py"
SPEC = importlib.util.spec_from_file_location("stv2_postrun_calibration", SCRIPT)
CAL = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(CAL)


class CalibrationTests(unittest.TestCase):
    def setUp(self):
        self.train = fixture_bundle().subset("train")

    def with_residual(self, bundle, residual):
        normalized, norm = normalize_inputs(bundle.inputs)
        clean = np.where(bundle.inputs["observed"][..., None], normalized["xy"], 0.)
        bundle.targets["xy"] = norm.invert(clean + residual)
        bundle.targets["valid"] = bundle.inputs["observed"].copy()
        bundle.targets["visible"] = bundle.targets["valid"].copy()
        return normalized, norm

    def test_constant_offset_recovery_and_json_roundtrip(self):
        offset = np.arange(24).reshape(12, 2) * .001 - .012
        self.with_residual(self.train, offset[None, None])
        models = CAL.fit_calibrations(self.train, "fixture-held")
        for model in models.values():
            recovered = CAL.apply_calibration(json.loads(json.dumps(model, allow_nan=False)), self.train.inputs)
            mask = self.train.inputs["observed"]
            np.testing.assert_allclose(recovered[mask], self.train.targets["xy"][mask], atol=3e-5, rtol=0)
            self.assertTrue(np.isnan(recovered[~mask]).all())
        np.testing.assert_allclose(models["joint_offset"]["offset"], offset, atol=1e-7)

    def test_jointwise_affine_recovers_known_cross_axis_correction(self):
        normalized, _ = normalize_inputs(self.train.inputs)
        xy = np.where(self.train.inputs["observed"][..., None], normalized["xy"], 0.)
        slopes = np.array([[.08, -.04], [.03, .06]])
        residual = xy @ slopes + np.array([.004, -.007])
        self.with_residual(self.train, residual)
        models = CAL.fit_calibrations(self.train, "fixture-held", ridge=1e-9)
        got = CAL.apply_calibration(models["joint_affine"], self.train.inputs)
        mask = self.train.inputs["observed"]
        np.testing.assert_allclose(got[mask], self.train.targets["xy"][mask], atol=6e-5, rtol=0)
        offset = CAL.apply_calibration(models["joint_offset"], self.train.inputs)
        self.assertGreater(float(np.abs(offset[mask] - self.train.targets["xy"][mask]).mean()), .05)

    def test_rejects_development_before_touching_targets(self):
        class Trap:
            records = fixture_bundle().records
            evidence_status = "fixture-tested"
            @property
            def targets(self):
                raise AssertionError("Evaluation targets must never be accessed")
        with self.assertRaisesRegex(ValueError, "only explicit historical training"):
            CAL.fit_calibrations(Trap(), "fixture-held")
        mislabeled = copy.deepcopy(self.train)
        mislabeled.records[0]["original_split"] = "validation"
        with self.assertRaisesRegex(ValueError, "historical training"):
            CAL.fit_calibrations(mislabeled, "fixture-held")

    def test_rejects_held_id_family_and_forged_family(self):
        for excluded in ("vitpose", "vitpose_base"):
            held = copy.deepcopy(self.train)
            held.records[0].update(extractor="alternate_vitpose", extractor_family="vitpose")
            with self.assertRaisesRegex(ValueError, "Held extractor"):
                CAL.fit_calibrations(held, excluded)
        forged = copy.deepcopy(self.train)
        forged.records[0].update(extractor="vitpose_base", extractor_family="hrnet")
        with self.assertRaisesRegex(ValueError, "known family"):
            CAL.fit_calibrations(forged, "fixture-held")

    def test_input_only_normalization_and_target_free_inference(self):
        self.with_residual(self.train, np.array([.01, -.02]))
        first = CAL.fit_calibrations(self.train, "fixture-held")
        changed = copy.deepcopy(self.train)
        changed.targets["eval_scale"] *= 500
        changed.targets["visible"][:] = False
        second = CAL.fit_calibrations(changed, "fixture-held")
        for key in ("offset",):
            np.testing.assert_array_equal(first["joint_offset"][key], second["joint_offset"][key])
        np.testing.assert_array_equal(first["joint_affine"]["coefficients"], second["joint_affine"]["coefficients"])
        dev = fixture_bundle().subset("development")
        old = CAL.apply_calibration(first["joint_offset"], dev.inputs)
        dev.targets["xy"][:] = 1e20
        np.testing.assert_equal(CAL.apply_calibration(first["joint_offset"], dev.inputs), old)
        with self.assertRaisesRegex(ValueError, "allow-list"):
            CAL.apply_calibration(first["joint_offset"], dict(dev.inputs, targets=dev.targets))

    def test_unknown_source_held_id_cannot_masquerade_as_family(self):
        source = copy.deepcopy(self.train)
        source.evidence_status = "automated-source-screen"
        for row in source.records:
            row.update(extractor="rtmpose_m", extractor_family="rtmpose",
                       locomotion_status="algorithm_screened_locomotion",
                       review_mode="automated_development", reserved="unknown")
        self.assertEqual(set(CAL.fit_calibrations(source, "vitpose")), {"joint_offset", "joint_affine"})
        for unknown in ("vitpose_custom", "vitpose-large", "unregistered_family"):
            with self.subTest(held=unknown), self.assertRaisesRegex(ValueError, "Unresolvable held extractor"):
                CAL.fit_calibrations(source, unknown)

    def test_offset_is_constant_within_window_and_preserves_pixel_displacements(self):
        offsets = np.arange(24).reshape(12, 2) * .001 - .012
        self.with_residual(self.train, offsets[None, None])
        model = CAL.fit_calibrations(self.train, "fixture-held")["joint_offset"]
        dev = fixture_bundle().subset("development")
        factors = np.linspace(.5, 3., len(dev.records))[:, None, None, None]
        dev.inputs["xy"] = dev.inputs["xy"] * factors + 40.
        _, normalization = normalize_inputs(dev.inputs)
        self.assertGreater(float(np.ptp(normalization.scale)), 50.)
        prediction = CAL.apply_calibration(model, dev.inputs)
        observed = dev.inputs["observed"]
        expected = normalization.scale[:, None, None, None] * np.asarray(model["offset"])[None, None]
        expected = np.broadcast_to(expected, prediction.shape)
        # The existing normalizer stores normalized coordinates in float32;
        # round-trip pixel error is bounded here at these enlarged image scales.
        np.testing.assert_allclose((prediction - dev.inputs["xy"])[observed],
                                   expected[observed], atol=3e-4, rtol=0)
        endpoints = observed[:, 5:] & observed[:, :-5]
        output_displacement = prediction[:, 5:] - prediction[:, :-5]
        input_displacement = dev.inputs["xy"][:, 5:] - dev.inputs["xy"][:, :-5]
        np.testing.assert_allclose(output_displacement[endpoints], input_displacement[endpoints],
                                   atol=3e-4, rtol=0)

    def test_people_dominate_unequal_windows_and_missing_frame_counts(self):
        # One first-person row has 64 frames; four second-person rows have only
        # 32 supported frames for joint zero. Each person's total mass is 1/2.
        selected = [0, 6, 7, 8, 9]
        bundle = TrackBundle({k: v[selected].copy() for k, v in self.train.inputs.items()},
                             {k: v[selected].copy() for k, v in self.train.targets.items()},
                             [copy.deepcopy(self.train.records[i]) for i in selected],
                             self.train.evidence_status, self.train.provenance)
        bundle.inputs["observed"][1:, :32, 0] = False
        bundle.inputs["xy"][1:, :32, 0] = np.nan
        bundle.inputs["confidence"][1:, :32, 0] = np.nan
        residual = np.zeros(bundle.inputs["xy"].shape)
        residual[1:, :, :, 0] = .2
        self.with_residual(bundle, residual)
        model = CAL.fit_calibrations(bundle, "fixture-held")["joint_offset"]
        self.assertAlmostEqual(model["offset"][0][0], .1, places=6)
        self.assertEqual(model["provenance"]["joint_support"][0]["people"], 2)
        self.assertEqual(model["provenance"]["joint_support"][0]["frames"], 64 + 4 * 32)

    def test_missing_targets_excluded_and_unsupported_joint_fails(self):
        self.with_residual(self.train, np.array([.01, -.02]))
        self.train.targets["valid"][0, :10, 2] = False
        self.train.targets["visible"][0, :10, 2] = False
        self.train.targets["xy"][0, :10, 2] = np.nan
        model = CAL.fit_calibrations(self.train, "fixture-held")["joint_offset"]
        np.testing.assert_allclose(model["offset"][2], [.01, -.02], atol=1e-7)
        self.train.targets["valid"][:, :, 2] = False
        self.train.targets["visible"][:, :, 2] = False
        with self.assertRaisesRegex(ValueError, "Unsupported training joint"):
            CAL.fit_calibrations(self.train, "fixture-held")

    def test_invalid_ridge_is_rejected(self):
        for value in (0, -1, float("inf"), float("nan"), True):
            with self.subTest(ridge=value), self.assertRaises(ValueError):
                CAL.fit_calibrations(self.train, "fixture-held", ridge=value)


if __name__ == "__main__":
    unittest.main()
