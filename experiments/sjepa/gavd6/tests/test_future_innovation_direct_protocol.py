"""The direct protocol measures skeleton increment without selectivity prerequisites."""

from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

from gavd6_sjepa.research_directions.future_innovation.fi_cohort import select_eligible, validate_cohort
from gavd6_sjepa.research_directions.future_innovation.fi_contracts import (
    initialize_run, DIRECT_PROTOCOL, DIRECT_ARMS, ARMS, audit_summary_path, experiment_arms, read_json, write_json, sha256_file,
)
from gavd6_sjepa.research_directions.future_innovation.fi_feature_cache import load_cache
from gavd6_sjepa.research_directions.future_innovation.fi_gate_decision import decide_gate
from gavd6_sjepa.research_directions.future_innovation.fi_metrics import (
    source_error_sums, source_bootstrap_counts, score_source_sums, source_bootstrap_indices, score_arrays,
)
from gavd6_sjepa.research_directions.future_innovation.fi_smoke import synthetic_cache
from gavd6_sjepa.research_directions.future_innovation.fi_token_regions import pool_context, pool_target, region_masks
from gavd6_sjepa.research_directions.future_innovation.fi_validity_audits import require_audits, ValidityAuditRejected


class DirectProtocolTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = synthetic_cache(Path(self.temporary.name) / "direct", protocol=DIRECT_PROTOCOL)

    def test_non50_gate_is_rejected_before_initialization(self):
        with self.assertRaisesRegex(ValueError, "exactly 50"):
            initialize_run(Path(self.temporary.name) / "larger", protocol=DIRECT_PROTOCOL, cohort_size=70,
                           sequence_manifest="unused", video_manifest="unused", annotations=[],
                           pose_model="unused", vjepa_root="unused", checkpoint="unused", change_reason="test")

    def test_non_object_readiness_is_an_execution_error(self):
        for value in (None, [], {"artifacts": None}):
            write_json(self.root / "qc/readiness-summary.json", value)
            with self.assertRaises(ValueError) as error:
                require_audits(self.root)
            self.assertNotIsInstance(error.exception, ValidityAuditRejected)

    def test_corrupt_config_preserves_notebook_diagnostic_routing(self):
        from gavd6_sjepa.research_directions.future_innovation import fi_notebook_workflow as workflow
        import subprocess
        path = self.root / "config/protocol-contract.json"
        path.write_text(path.read_text() + "\n")
        with patch.object(workflow, "run_stage", side_effect=subprocess.CalledProcessError(1, ["audit"])) as stage:
            self.assertIsInstance(workflow.attempt_stage("audit-teacher", self.root), str)
            stage.assert_called_once()
        with patch.object(workflow, "run_stage", side_effect=[subprocess.CalledProcessError(1, ["score"]), None]) as stage:
            self.assertIs(workflow.build_notebook_report(self.root), False)
            self.assertEqual([call.args[0] for call in stage.call_args_list], ["score-gate", "build-report"])

    def test_blocked_direct_report_requires_matching_protocol_and_no_jepa_permission(self):
        from gavd6_sjepa.research_directions.future_innovation import fi_notebook_workflow as workflow
        from gavd6_sjepa.research_directions.future_innovation.fi_reporting import build_report
        table = self.root / "qc/causal-leakage.csv"
        rows = pd.read_csv(table)
        rows.loc[0, ["max_abs_difference", "passed"]] = [0.1, False]
        rows.to_csv(table, index=False)
        path = self.root / "qc/readiness-summary.json"
        summary = read_json(path)
        summary["artifacts"]["qc/causal-leakage.csv"] = sha256_file(table)
        summary["checks"]["causal_leakage_absent"] = summary["passed"] = False
        write_json(path, summary)
        original = build_report(self.root)
        self.assertFalse(original["allow_jepa_training_comparison"])
        self.assertFalse(original["measurement_complete"])
        self.assertIsNone(original["metrics"])
        for changes in ({"allow_jepa_training_comparison": True}, {"protocol": "legacy-v1"}):
            write_json(self.root / "reports/gate-decision.json", {**original, **changes})
            with patch.object(workflow, "run_stage"), self.assertRaisesRegex(ValueError, "does not match"):
                workflow.build_notebook_report(self.root)
        self.assertEqual(original["readiness_sha256"], sha256_file(path))
        self.assertEqual(original["manifest_sha256"], sha256_file(self.root / "manifests/gate-windows.csv"))

    def metrics(self):
        return {"delta_r2_real": 0.06, "delta_r2_time_shuffle": 0.02, "delta_r2_clip_mismatch": 0.005,
                "delta_r2_no_skeleton": 0.01, "seed_real_gains": [0.06] * 3,
                "seed_skeleton_increments": [0.05] * 3, "bootstrap_positive_fraction": 0.95,
                "skeleton_increment_positive_fraction": 0.95,
                **{k: True for k in ("data_contract_valid", "evaluation_contract_valid", "controls_complete",
                                    "input_audit_complete", "teacher_stable", "causal_leakage_absent", "target_variance_valid")}}

    def test_direct_readiness_accepts_50_clip_gate_without_pixel_artifacts(self):
        self.assertEqual(experiment_arms(self.root), DIRECT_ARMS)
        self.assertTrue(require_audits(self.root)["passed"])
        self.assertEqual(audit_summary_path(self.root).name, "readiness-summary.json")
        self.assertFalse((self.root / "qc/target-sensitivity.csv").exists())
        self.assertFalse((self.root / "qc/pixel-edit-contact-sheets").exists())
        cohort, arrays = load_cache(self.root)
        self.assertEqual(arrays["skeleton"].shape, (50, 32, 33, 4))
        self.assertEqual(cohort.video_id.nunique(), 25)

    def test_gate_selects_50_clips_deterministically_from_larger_pool(self):
        rows = [{"window_id": f"w{i}", "sequence_id": f"q{i}", "video_id": f"v{i // 4}"} for i in range(280)]
        selected = select_eligible(rows, count=50)
        self.assertEqual(len(selected), 50)
        self.assertGreaterEqual(selected.video_id.nunique(), 25)
        self.assertLessEqual(selected.groupby("video_id").size().max(), 2)
        validate_cohort(selected, read_json(self.root / "config/run-contract.json"))
        self.assertTrue(selected.equals(select_eligible(list(reversed(rows)), count=50)))

    def test_new_contract_has_no_quality_or_background_decision_cutoffs(self):
        run = read_json(self.root / "config/run-contract.json")
        self.assertEqual(run["cohort_size"], 50)
        self.assertEqual(run["minimum_crop_retention"], 0)
        self.assertEqual(run["minimum_pose_coverage"], 0)
        thresholds = read_json(self.root / "config/thresholds.json")
        self.assertNotIn("motion_to_background_change_min", thresholds)
        self.assertNotIn("person_ablation_reduction_min", thresholds)

    def test_absent_background_region_is_allowed_for_direct_pooling(self):
        box = np.array([0, 0, 1, 1], dtype=float)
        with self.assertRaisesRegex(ValueError, "Empty background"):
            region_masks(box)
        person, background = region_masks(box, allow_empty_background=True)
        self.assertTrue(person.all())
        self.assertFalse(background.any())
        boxes = np.tile(box, (64, 1))
        context = pool_context(np.ones((16 * 576, 2)), boxes, allow_empty_background=True)
        np.testing.assert_array_equal(context, [1, 1, 1, 1, 0, 0])
        target, unused_background = pool_target(np.ones((32 * 576, 2)), boxes, allow_empty_background=True)
        np.testing.assert_array_equal(target, [1, 1])
        np.testing.assert_array_equal(unused_background, [0, 0])

    def test_background_target_variance_cannot_block_the_primary_comparison(self):
        cohort, arrays = load_cache(self.root)
        arrays["background"][:] = 0
        with patch("gavd6_sjepa.research_directions.future_innovation.fi_readiness.load_cache", return_value=(cohort, arrays)):
            self.assertTrue(require_audits(self.root)["passed"])

    def test_readiness_rejects_corrupt_records_and_causal_leakage(self):
        summary_path = audit_summary_path(self.root)
        summary = read_json(summary_path)
        table_path = self.root / "qc/causal-leakage.csv"
        table = pd.read_csv(table_path)
        table.loc[0, "max_abs_difference"] = 0.1
        table.loc[0, "passed"] = False
        table.to_csv(table_path, index=False)
        summary["artifacts"]["qc/causal-leakage.csv"] = sha256_file(table_path)
        write_json(summary_path, summary)
        with self.assertRaisesRegex(ValueError, "flags disagree"):
            require_audits(self.root)
        summary["checks"]["causal_leakage_absent"] = False
        summary["passed"] = False
        write_json(summary_path, summary)
        with self.assertRaises(ValidityAuditRejected):
            require_audits(self.root)

    def test_rgb_only_capacity_gain_cannot_count_as_skeleton_evidence(self):
        metrics = {**self.metrics(), "delta_r2_no_skeleton": 0.06, "seed_skeleton_increments": [0.0] * 3}
        result = decide_gate(metrics, protocol=DIRECT_PROTOCOL)
        self.assertEqual(result["decision"], "STOP")
        self.assertFalse(result["allow_jepa_training_comparison"])
        self.assertFalse(result["checks"]["skeleton_increment"])

    def test_supported_increment_recommends_jepa_comparison_only(self):
        result = decide_gate(self.metrics(), protocol=DIRECT_PROTOCOL)
        self.assertEqual(result["decision"], "ADVANCE")
        self.assertEqual(result["next_action"], "design_full_gavd_jepa_comparison")
        self.assertFalse(result["allow_adapter_training"])
        self.assertTrue(result["allow_full_experiment"])
        self.assertTrue(result["allow_jepa_training_comparison"])
        metrics = {**self.metrics(), "skeleton_increment_positive_fraction": 0.6}
        self.assertEqual(decide_gate(metrics, protocol=DIRECT_PROTOCOL)["decision"], "INCONCLUSIVE")

    def test_legacy_protocol_does_not_silently_drop_its_requirements(self):
        root = synthetic_cache(Path(self.temporary.name) / "legacy")
        self.assertEqual(experiment_arms(root), ARMS)
        self.assertEqual(audit_summary_path(root).name, "validity-summary.json")
        self.assertTrue(require_audits(root)["passed"])

    def test_source_sums_match_clip_resampling_with_unequal_source_sizes(self):
        rng = np.random.default_rng(42)
        y, base, full = [rng.normal(size=(7, 5)) for _ in range(3)]
        sources = np.array(["a", "a", "a", "b", "c", "c", "d"])
        weights = np.array([1/3, 1/3, 1/3, 1, .5, .5, 1])
        valid = np.array([True, False, True, True, True])
        sums = source_error_sums(y, base, full, weights, sources)
        for indices, multiplicities in zip(source_bootstrap_indices(sources, 20), source_bootstrap_counts(sources, 20)):
            expected = score_arrays(y[indices], base[indices], full[indices], weights[indices], valid)[0]
            actual = score_source_sums(sums, multiplicities, valid)
            for key in actual:
                self.assertAlmostEqual(actual[key], expected[key], places=12)


if __name__ == "__main__":
    unittest.main()
