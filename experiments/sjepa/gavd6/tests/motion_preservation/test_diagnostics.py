"""Mechanism diagnostics must separate damage, retain identity, and preserve fits."""
from dataclasses import replace
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

from gavd6_sjepa.research_directions.motion_preservation import diagnostics, workflow
from gavd6_sjepa.research_directions.motion_preservation import body_geometry as geometry
from gavd6_sjepa.research_directions.motion_preservation.config import RunConfig


class DiagnosticTests(unittest.TestCase):
    def fixture(self):
        clean, event = geometry.demo_motion(frames=16, family="foot_clearance")
        truth = event.joints.copy()
        raw = truth.copy(); raw[:, 7, 0] += .03
        case = dict(raw=raw, truth=truth, clean=clean.joints, event_reference=truth.copy(),
                    observed=np.ones(raw.shape[:2], bool), prior=truth+.015,
                    bridge=raw.copy(), bridge_error=0.,
                    bone_lengths=geometry.estimate_bone_lengths(raw))
        meta = dict(case_id="one", person_id="person1", role="development", event_family="foot_clearance",
                    fixture="factorial", event_present=True, noise_present=True)
        return case, meta

    def test_repair_and_new_damage_are_separate(self):
        case, meta = self.fixture()
        prediction = case["truth"].copy()
        prediction[:, 21, 0] += .01  # A previously accurate wrist is damaged.
        score = diagnostics.score_prediction(prediction, case, meta, "example")
        self.assertEqual(score["affected_count"], 16)
        self.assertEqual(score["unaffected_count"], 16*21)
        self.assertEqual(score["affected_mse_m2"], 0.)
        self.assertGreater(score["unaffected_mse_m2"], 0.)
        total = (score["affected_mse_m2"]*score["affected_count"] +
                 score["unaffected_mse_m2"]*score["unaffected_count"])/(16*22)
        self.assertAlmostEqual(total, score["observed_mse_m2"], places=10)
        matched = diagnostics.score_prediction(prediction, case, {**meta, "fixture": "matched"}, "example")
        self.assertTrue(np.isnan(matched["affected_mse_m2"]))

    def test_oracle_is_bounded_before_projection_and_bridge_placeholder_is_not_scored(self):
        case, _ = self.fixture()
        values = diagnostics.diagnostic_candidates(case)
        original_error = np.sum((case["raw"]-case["truth"])**2, axis=-1)
        oracle_error = np.sum((values["oracle_mixture_unprojected"]-case["truth"])**2, axis=-1)
        self.assertTrue(np.all(oracle_error <= original_error+1e-10))
        case["bridge_error"] = np.nan
        values = diagnostics.diagnostic_candidates(case)
        self.assertNotIn("bridge_unprojected", values)

    def test_reference_length_variation_is_visible_even_when_observation_is_exact(self):
        case, meta = self.fixture()
        parent = geometry.PARENTS[20]
        truth = case["truth"].copy()
        vector = truth[:, 20]-truth[:, parent]
        truth[:, 20] = truth[:, parent]+np.linspace(.9, 1.1, len(truth))[:, None]*vector
        case.update(truth=truth, raw=truth.copy(), bone_lengths=geometry.estimate_bone_lengths(truth))
        rows = pd.DataFrame(diagnostics.bone_length_rows(case, meta)).set_index("joint")
        self.assertGreater(rows.loc[20, "reference_length_std_m"], .001)
        self.assertAlmostEqual(rows.loc[20, "bias_m"], 0., places=7)
        candidates = diagnostics.diagnostic_candidates(case)
        self.assertGreater(np.mean((candidates["truth_projected_reference_lengths"]-truth)**2), 1e-7)

    def test_person_summary_does_not_weight_repeated_clips_as_new_people(self):
        case, meta = self.fixture()
        row = diagnostics.score_prediction(case["raw"], case, meta, "raw")
        rows = [{**row, "case_id": f"repeat{i}", "observed_mse_m2": 1.} for i in range(3)]
        rows.append({**row, "case_id": "second", "person_id": "person2", "observed_mse_m2": 3.})
        summary = diagnostics.summarize_cases(pd.DataFrame(rows)).iloc[0]
        self.assertEqual(summary.observed_mse_m2, 2.)
        self.assertEqual(summary.n_people, 2)
        self.assertEqual(summary.n_cases, 4)

    def test_final_is_rejected_before_cache_access(self):
        with tempfile.TemporaryDirectory() as folder:
            with patch.object(workflow, "_load_cases", side_effect=AssertionError("must not open caches")):
                with self.assertRaisesRegex(ValueError, "final"):
                    diagnostics.run_diagnostics(RunConfig(run_root=folder), roles=("development", "final"))
            self.assertEqual(list(Path(folder).iterdir()), [])

    def test_cache_only_run_preserves_fit_and_has_an_exact_zero_endpoint(self):
        with tempfile.TemporaryDirectory() as folder:
            cfg = RunConfig(run_root=folder, mode="demo", device="cpu", prior_id="demo_smoothing",
                            prior_backend="demo_smoothing", flow_backend="demo_farneback",
                            max_motions_per_role=1, duration_s=.8, image_size=32)
            workflow.build_pairs(cfg, roles=("calibration", "development"))
            workflow.cache_predictions(cfg, roles=("calibration", "development"))
            protected = [cfg.root/"config.json", cfg.root/"models/training.json", cfg.root/"calibration/locked.json"]
            for path in protected[1:]:
                path.parent.mkdir(exist_ok=True); path.write_text('"Existing fitted artifact"\n')
            before = {path: path.read_bytes() for path in protected}
            # Device in a saved training configuration does not request GPU work.
            report = diagnostics.run_diagnostics(replace(cfg, device="cuda"), max_trace_cases=1)
            self.assertTrue((report["output_dir"]/"case_scores.csv").exists())
            self.assertTrue((report["output_dir"]/"flow_pairs.csv").exists())
            self.assertEqual(len(report["traces"]), 1)
            self.assertEqual(report["traces"][0]["role"], "development")
            for path, original in before.items(): self.assertEqual(path.read_bytes(), original)
            self.assertFalse((cfg.root/"final_opened.json").exists())
            curves = report["strength_case_scores"]
            zero = curves.loc[curves.strength.eq(0) & curves.projection.eq("none")]
            np.testing.assert_allclose(zero.mean_displacement_m, 0., atol=1e-10)
            np.testing.assert_allclose(zero.noise_removal.dropna(), 0., atol=1e-8)
            self.assertFalse(report["notes"]["fits_or_calibration_changed"])


if __name__ == "__main__":
    unittest.main()
