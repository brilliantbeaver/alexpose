"""Expanded diagnostics preserve source evidence and separate statistical units."""
from dataclasses import replace
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np
import pandas as pd

from gavd6_sjepa.research_directions.synthetic_training_v2.config import RunConfig, STAGES
from gavd6_sjepa.research_directions.synthetic_training_v2.contracts import atomic_json, sha256_file
from gavd6_sjepa.research_directions.synthetic_training_v2.data import fixture_bundle
from gavd6_sjepa.research_directions.synthetic_training_v2.workflow import run_stage

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts/research_directions/synthetic_training_v2"))
import expansion_results as expansion


def tree_hashes(root):
    return {str(p.relative_to(root)): sha256_file(p) for p in root.rglob("*") if p.is_file()}


class LeftRightDiagnosticTests(unittest.TestCase):
    def setUp(self):
        self.bundle = fixture_bundle().subset("development")
        self.records = [{**r, "seed": 17} for r in self.bundle.records]

    def score(self, prediction, **kwargs):
        return expansion.left_right_metrics(prediction, self.bundle.targets, self.records,
            method="test", evidence_status="fixture-tested", **kwargs)

    def test_common_translation_cancels_but_one_sided_error_is_retained(self):
        prediction = self.bundle.targets["xy"].copy()
        prediction += np.array([4., -7.], np.float32)
        self.assertTrue(np.allclose(self.score(prediction).left_right_nle, 0, atol=1e-7))
        prediction[:, :, 10, 0] += 36.
        frame = self.score(prediction)
        np.testing.assert_allclose(frame.loc[frame.pair.eq("ankle"), "left_right_nle"], .1, atol=1e-7)
        np.testing.assert_allclose(frame.loc[frame.pair.eq("hip"), "left_right_nle"], 0, atol=1e-7)

    def test_missing_prediction_penalty_and_hidden_reference_support(self):
        prediction = self.bundle.targets["xy"].copy()
        prediction[:, :, 10] = np.nan
        visible = self.score(prediction)
        all_valid = self.score(prediction, endpoint="all_valid_synthetic")
        ankles = visible.loc[visible.pair.eq("ankle")]
        self.assertTrue((ankles.left_right_nle == 1.).all())
        self.assertTrue((ankles.missing_prediction_frames == ankles.reference_frames).all())
        self.assertTrue((all_valid.reference_frames == 64).all())
        self.assertTrue((visible.reference_frames < 64).any())
        self.assertFalse(np.isnan(prediction[:, :, 6]).any())

    def test_real_hidden_joints_and_invalid_scale_are_rejected(self):
        self.records[0]["target_kind"] = "real_annotation"
        with self.assertRaisesRegex(ValueError, "synthetic-proxy"):
            self.score(self.bundle.targets["xy"], endpoint="all_valid_synthetic")
        self.bundle.targets["eval_scale"][0, 0] = 0
        with self.assertRaisesRegex(ValueError, "scale"):
            self.score(self.bundle.targets["xy"])

    def test_balancing_keeps_people_and_unsupported_reference_rows(self):
        frame = self.score(self.bundle.targets["xy"])
        frame["scale_policy"] = "frame_reference"
        # An entirely unsupported motion must remain missing in its stratum.
        choose = frame.pair.eq("ankle") & frame.extractor.eq("fixture-held")
        frame.loc[frame.index[choose][0], "left_right_nle"] = np.nan
        balanced = expansion._balanced(frame, ("left_right_nle",), ("pair",))
        failed = balanced.loc[balanced.pair.eq("ankle") & balanced.extractor.eq("fixture-held")
                              & balanced.condition.eq("all_conditions")]
        self.assertTrue(failed.value.isna().all())
        self.assertEqual(set(balanced.people), {3})

    def test_constant_scale_preserves_offset_dynamics_and_reference_arrays(self):
        from gavd6_sjepa.research_directions.synthetic_training_v2.evaluation import evaluate_predictions
        from diagnostics.timing import timing_diagnostics

        targets = {key: value[:1].copy() for key, value in self.bundle.targets.items()}
        targets["visible"][:] = True
        phase = np.arange(64) * (2 * np.pi / 25)
        targets["xy"][0, :, 10, 0] = 200 + 15 * np.sin(phase)
        targets["xy"][0, :, 11, 0] = 200 - 15 * np.sin(phase)
        targets["eval_scale"][0] = 300 + 40 * np.sin(phase * 3 + .4)
        original = {key: value.copy() for key, value in targets.items()}
        prediction = targets["xy"].copy()
        prediction[:, :, 10, 0] += 35
        record = [self.records[0]]
        constant = expansion.evaluation_targets(targets, record, "visible", "window_median_reference")
        scores = evaluate_predictions(prediction, constant, self.bundle.inputs["timestamps"][:1], record)
        events = timing_diagnostics(prediction, constant, self.bundle.inputs["timestamps"][:1], record,
                                    method="offset", evidence_status="fixture-tested")
        self.assertAlmostEqual(float(scores.displacement_nle.iloc[0]), 0., places=7)
        self.assertAlmostEqual(float(scores.amplitude_error.iloc[0]), 0., places=7)
        self.assertAlmostEqual(float(events.original_event_timing_mae_s.iloc[0]), 0., places=7)
        varying = evaluate_predictions(prediction, targets, self.bundle.inputs["timestamps"][:1], record)
        self.assertGreater(float(varying.amplitude_error.iloc[0]), 1e-4)
        for key in original:
            np.testing.assert_array_equal(original[key], targets[key])


class ExpansionArtifactTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temporary = tempfile.TemporaryDirectory(prefix="expansion results ")
        cls.base = Path(cls.temporary.name)
        cls.cfg = replace(RunConfig.fixture("verified-fixture", str(cls.base)),
                          updates=1, readout_updates=1, bootstrap_draws=10)
        cls.config = cls.base / "scope.json"
        atomic_json(cls.config, cls.cfg.as_dict())
        for stage in STAGES:
            run_stage(cls.cfg, stage, ROOT)

    @classmethod
    def tearDownClass(cls):
        cls.temporary.cleanup()

    def test_complete_scope_verifies_without_writing_and_fixture_requires_explicit_flag(self):
        before = tree_hashes(self.cfg.root)
        with self.assertRaisesRegex(ValueError, "source mode"):
            expansion.verify_scope(self.config)
        result = expansion.verify_scope(self.config, allow_fixture_tests=True)
        self.assertEqual(result["status"], "SCOPE_ARTIFACTS_COMPLETE")
        self.assertEqual(result["evidence_status"], "fixture-tested")
        self.assertEqual(result["development_people"], 3)
        self.assertIn("Not checked", result["scheduler"])
        self.assertEqual(before, tree_hashes(self.cfg.root))

    def test_changed_configuration_and_metric_bytes_fail_closed(self):
        path = self.base / "changed-config.json"
        atomic_json(path, replace(self.cfg, updates=2).as_dict())
        with self.assertRaisesRegex(ValueError, "identity changed"):
            expansion.verify_scope(path, allow_fixture_tests=True)
        metric = self.cfg.root / "evaluation/per-window.csv"
        original = metric.read_bytes()
        try:
            metric.write_bytes(original + b"\n")
            with self.assertRaisesRegex(ValueError, "Stale/corrupt"):
                expansion.verify_scope(self.config, allow_fixture_tests=True)
        finally:
            metric.write_bytes(original)

    def test_analysis_reports_support_scales_and_suite_rejects_duplicate_seeds_or_tampering(self):
        before = tree_hashes(self.cfg.root)
        output = self.base / "diagnostics"
        result = expansion.run_analysis(self.config, output, allow_fixture_tests=True, max_windows=1)
        self.assertEqual(result["status"], "EXPANSION_ANALYSIS_COMPLETE")
        self.assertEqual(result["standard_checks"]["status"], "POSTRUN_CHECKS_COMPLETE")
        self.assertFalse(result["scientific_gate_authorized"])
        self.assertEqual(before, tree_hashes(self.cfg.root))
        coordinate = pd.read_csv(output / "exploratory-per-window.csv")
        self.assertEqual(set(coordinate.endpoint), set(expansion.ENDPOINTS))
        self.assertEqual(set(coordinate.scale_policy), set(expansion.SCALE_POLICIES))
        self.assertTrue(set(self.cfg.arms) <= set(coordinate.method))
        timing = pd.read_csv(output / "exploratory-timing.csv")
        oracle = timing.loc[timing.method.eq("reference_oracle")]
        self.assertTrue((oracle.original_support == oracle.reference_eligible).all())
        visible = oracle.loc[oracle.endpoint.eq("visible")]
        all_valid = oracle.loc[oracle.endpoint.eq("all_valid_synthetic")]
        self.assertGreater(all_valid.reference_eligible.sum(), visible.reference_eligible.sum())
        with self.assertRaises(FileExistsError):
            expansion.run_analysis(self.config, output, allow_fixture_tests=True)
        suite = self.base / "suite"
        expansion.report_suite([self.config], [output], suite, allow_fixture_tests=True)
        summary = pd.read_csv(suite / "training-seed-summary.csv")
        self.assertEqual(set(summary.seed_count), {1})
        self.assertEqual(set(summary.independent_people), {3})
        self.assertTrue(summary.standard_deviation.isna().all())
        self.assertEqual(set(pd.read_csv(suite / "per-seed-balanced.csv").updates), {1})
        with self.assertRaisesRegex(ValueError, "Duplicate training seed"):
            expansion.report_suite([self.config, self.config], [output, output], self.base / "duplicate", allow_fixture_tests=True)
        changed = output / "exploratory-balanced.csv"
        changed.write_text(changed.read_text() + "\n")
        with self.assertRaisesRegex(ValueError, "Changed diagnostic artifact"):
            expansion.report_suite([self.config], [output], self.base / "changed", allow_fixture_tests=True)


if __name__ == "__main__":
    unittest.main()
