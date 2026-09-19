"""Post-run timing diagnostics expose censoring without changing old metrics."""
import importlib.util
from pathlib import Path
import unittest

import numpy as np

from gavd6_sjepa.research_directions.synthetic_training_v2.evaluation import evaluate_predictions


ROOT = Path(__file__).resolve().parents[2]
SPEC = importlib.util.spec_from_file_location(
    "stv2_postrun_timing", ROOT / "scripts/research_directions/synthetic_training_v2/diagnostics/timing.py"
)
TIMING = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(TIMING)


def fixture(reference_peaks=(10, 35, 60), predicted_peaks=None, length=80):
    times = np.arange(length, dtype=float)[None] / 25
    truth = np.zeros((1, length, 12, 2))
    truth[0, list(reference_peaks), 10, 0] = 20
    prediction = truth.copy()
    if predicted_peaks is not None:
        prediction[:] = 0
        prediction[0, list(predicted_peaks), 10, 0] = 20
    mask = np.ones(truth.shape[:-1], dtype=bool)
    targets = dict(xy=truth, valid=mask.copy(), visible=mask.copy(),
                   eval_scale=np.full(times.shape, 100.))
    records = [dict(person_id="p0", motion_id="m0", window_id="w0", variant="clean",
                    extractor="e0", split="development", seed=17, target_kind="synthetic_proxy")]
    return prediction, targets, times, records


def score(*args, tolerance_s=.12):
    return TIMING.timing_diagnostics(*args, method="candidate",
                                    evidence_status="automated-source-screen",
                                    tolerance_s=tolerance_s).iloc[0]


class PostrunTimingTests(unittest.TestCase):
    def test_oracle_exposes_reference_ceiling_and_zero_timing_error(self):
        pred, targets, times, records = fixture()
        out = score(targets["xy"], targets, times, records)
        self.assertTrue(out.reference_eligible)
        self.assertTrue(out.original_support)
        self.assertEqual(out.reference_event_count, 3)
        self.assertEqual(out.matched_event_count, 3)
        self.assertEqual(out.missed_event_count, 0)
        self.assertEqual(out.extra_event_count, 0)
        self.assertEqual(out.event_recall, 1)
        self.assertEqual(out.event_precision, 1)
        self.assertEqual(out.matched_timing_mae_s, 0)
        self.assertEqual(out.eligible_window_denominator, 1)
        self.assertEqual(out.reference_event_denominator, 3)
        self.assertEqual(out.matched_timing_denominator, 3)

    def test_count_mismatch_is_attributed_to_prediction_not_reference(self):
        out = score(*fixture(predicted_peaks=(10, 60)))
        self.assertTrue(out.reference_eligible)
        self.assertFalse(out.original_support)
        self.assertEqual(out.prediction_status, "event_count_mismatch")
        self.assertEqual(out.missed_event_count, 1)
        self.assertEqual(out.matched_event_count, 2)
        self.assertEqual(out.extra_event_count, 0)
        self.assertAlmostEqual(out.event_recall, 2 / 3)
        self.assertEqual(out.matched_timing_mae_s, 0)
        self.assertEqual(out.reference_event_denominator, 3)

    def test_extra_prediction_event_is_not_discarded(self):
        out = score(*fixture(predicted_peaks=(10, 23, 35, 60)))
        self.assertFalse(out.original_support)
        self.assertEqual(out.extra_event_count, 1)
        self.assertEqual(out.missed_event_count, 0)
        self.assertAlmostEqual(out.event_precision, .75)

    def test_missing_prediction_counts_reference_misses_without_false_precision(self):
        args = fixture()
        args[0][0, 3, 11, 1] = np.nan
        out = score(*args)
        self.assertTrue(out.reference_eligible)
        self.assertFalse(out.original_support)
        self.assertEqual(out.matching_status, "prediction_incomplete")
        self.assertEqual(out.matched_event_count, 0)
        self.assertEqual(out.missed_event_count, 3)
        self.assertEqual(out.event_recall, 0)
        self.assertEqual(out.complete_prediction_window_denominator, 0)
        for key in ("predicted_event_count", "extra_event_count", "event_precision", "matched_timing_mae_s"):
            self.assertTrue(np.isnan(out[key]), key)

    def test_visibility_gaps_censor_reference_even_for_oracle(self):
        pred, targets, times, records = fixture()
        targets["visible"][0, 4, 10] = False
        out = score(targets["xy"], targets, times, records)
        self.assertFalse(out.reference_eligible)
        self.assertFalse(out.original_support)
        self.assertEqual(out.reference_status, "reference_visibility_gap")
        self.assertEqual(out.reference_event_denominator, 0)
        self.assertEqual(out.eligible_window_denominator, 0)
        for key in ("matched_event_count", "missed_event_count", "event_recall", "event_precision", "matched_timing_mae_s"):
            self.assertTrue(np.isnan(out[key]), key)

    def test_invalid_nonfinite_reference_is_explicit(self):
        args = fixture()
        args[1]["xy"][0, 4, 10, 1] = np.nan
        out = score(*args)
        self.assertFalse(out.reference_eligible)
        self.assertFalse(out.original_metric_input_valid)
        self.assertEqual(out.original_event_status, "invalid_original_metric_input")
        self.assertEqual(out.reference_status, "reference_nonfinite_ankles")
        self.assertTrue(np.isnan(out.reference_event_count))

    def test_one_peak_and_zero_amplitude_never_become_successful_zeros(self):
        for args in (fixture(reference_peaks=(35,)), fixture(reference_peaks=())):
            out = score(*args)
            self.assertFalse(out.reference_eligible)
            self.assertFalse(out.original_support)
            self.assertIn("fewer_than_two_reference_peaks", out.reference_reasons)
            self.assertTrue(np.isnan(out.event_recall))
            self.assertTrue(np.isnan(out.matched_timing_mae_s))

    def test_finite_flat_prediction_fails_recall(self):
        out = score(*fixture(predicted_peaks=()))
        self.assertTrue(out.reference_eligible)
        self.assertEqual(out.predicted_event_count, 0)
        self.assertEqual(out.missed_event_count, 3)
        self.assertEqual(out.extra_event_count, 0)
        self.assertEqual(out.event_recall, 0)
        self.assertEqual(out.event_f1, 0)
        self.assertTrue(np.isnan(out.event_precision))
        self.assertTrue(np.isnan(out.matched_timing_mae_s))

    def test_matching_tolerance_does_not_rewrite_original_metric(self):
        args = fixture(predicted_peaks=(13, 38, 63))
        inside = score(*args, tolerance_s=.12)
        outside = score(*args, tolerance_s=.11)
        self.assertTrue(inside.original_support)
        self.assertTrue(outside.original_support)
        self.assertAlmostEqual(inside.original_event_timing_mae_s, .12)
        self.assertAlmostEqual(outside.original_event_timing_mae_s, .12)
        self.assertEqual(inside.matched_event_count, 3)
        self.assertAlmostEqual(inside.matched_timing_mae_s, .12)
        self.assertEqual(outside.matched_event_count, 0)
        self.assertEqual(outside.missed_event_count, 3)
        self.assertEqual(outside.extra_event_count, 3)
        self.assertTrue(np.isnan(outside.matched_timing_mae_s))

    def test_original_status_and_timing_match_existing_evaluator(self):
        examples = [fixture(), fixture(predicted_peaks=(10, 60)),
                    fixture(reference_peaks=(35,)), fixture(reference_peaks=()),
                    fixture(predicted_peaks=(15, 40, 65))]
        for args in examples:
            original = evaluate_predictions(*args).iloc[0]
            out = score(*args)
            self.assertEqual(out.original_event_status, original.event_status)
            self.assertEqual(out.original_event_count, original.event_count)
            np.testing.assert_allclose(out.original_event_timing_mae_s, original.event_timing_mae_s,
                                       equal_nan=True)

    def test_minimum_duration_and_frame_support(self):
        out = score(*fixture(reference_peaks=(2, 5), length=7))
        self.assertFalse(out.reference_eligible)
        self.assertIn("fewer_than_eight_frames", out.reference_reasons)
        self.assertIn("duration_below_one_second", out.reference_reasons)
        args = fixture()
        args[2][:] /= 10
        out = score(*args)
        self.assertFalse(out.reference_eligible)
        self.assertEqual(out.reference_status, "duration_below_one_second")

    def test_scale_variation_and_raw_amplitude_are_visible(self):
        args = fixture()
        baseline = score(*args)
        self.assertEqual(baseline.bbox_scale_cv, 0)
        self.assertAlmostEqual(baseline.reference_amplitude_normalized,
                               baseline.reference_amplitude_raw_px / 100)
        args[1]["eval_scale"][0, 10] = 200
        out = score(*args)
        self.assertGreater(out.bbox_scale_cv, 0)
        self.assertEqual(out.bbox_scale_min_px, 100)
        self.assertEqual(out.bbox_scale_max_px, 200)
        self.assertEqual(out.reference_amplitude_raw_px, baseline.reference_amplitude_raw_px)
        self.assertNotEqual(out.reference_amplitude_normalized, baseline.reference_amplitude_normalized)

    def test_monotone_timestamps_shapes_masks_scales_and_tolerance(self):
        for value in (np.nan, -1.):
            with self.assertRaises(ValueError):
                score(*fixture(), tolerance_s=value)
        for kind in ("duplicate", "reverse", "nonfinite"):
            args = fixture()
            if kind == "duplicate":
                args[2][0, 2] = args[2][0, 1]
            elif kind == "reverse":
                args[2][:] = args[2][:, ::-1]
            else:
                args[2][0, 0] = np.nan
            with self.assertRaisesRegex(ValueError, "timestamps"):
                score(*args)
        args = fixture()
        args[1]["valid"] = args[1]["valid"].astype(int)
        with self.assertRaisesRegex(ValueError, "Boolean"):
            score(*args)
        args = fixture()
        args[1]["eval_scale"][0, 0] = 0
        with self.assertRaisesRegex(ValueError, "scale"):
            score(*args)

    def test_dynamic_program_avoids_nearest_greedy_losing_a_match(self):
        # Nearest-first chooses .18 for reference .10, leaving .20 unmatched;
        # ordered cardinality-first matching chooses .00 then .18.
        pairs = TIMING._ordered_matches([.10, .20], [.00, .18], .11)
        np.testing.assert_array_equal(pairs, [[0, 0], [1, 1]])

    def test_dynamic_program_minimizes_cost_after_cardinality(self):
        pairs = TIMING._ordered_matches([.10], [.00, .09], .11)
        np.testing.assert_array_equal(pairs, [[0, 1]])
        self.assertEqual(TIMING._ordered_matches([], [.1], .12).shape, (0, 2))
        self.assertEqual(TIMING._ordered_matches([.1], [], .12).shape, (0, 2))


if __name__ == "__main__":
    unittest.main()
