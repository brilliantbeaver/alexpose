"""future feature prediction / gate / test decision."""


import contextlib
import io
import json
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

from gavd6_sjepa.research_directions.future_prediction import notebook_workflow as workflow
from gavd6_sjepa.research_directions.future_prediction.cohort import (
    select_eligible,
    validate_cohort,
)
from gavd6_sjepa.research_directions.future_prediction.contracts import (
    ARMS,
    DIRECT_ARMS,
    DIRECT_PROTOCOL,
    audit_summary_path,
    experiment_arms,
    initialize_run,
    read_json,
    sha256_file,
    write_json,
)
from gavd6_sjepa.research_directions.future_prediction.decisions import decide_gate
from gavd6_sjepa.research_directions.future_prediction.feature_cache import load_cache
from gavd6_sjepa.research_directions.future_prediction.metrics import (
    score_arrays,
    score_source_sums,
    source_bootstrap_counts,
    source_bootstrap_indices,
    source_error_sums,
)
from gavd6_sjepa.research_directions.future_prediction.reporting import build_report
from gavd6_sjepa.research_directions.future_prediction.smoke import synthetic_audit, synthetic_cache
from gavd6_sjepa.research_directions.future_prediction.token_regions import (
    pool_context,
    pool_target,
    region_masks,
)
from gavd6_sjepa.research_directions.future_prediction.validity_audits import (
    ValidityAuditRejected,
    require_audits,
    verify_audits,
)

# A rejected measurement blocks training without hiding execution defects.


class AuditOutcomeTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = synthetic_cache(Path(self.temporary.name) / "run")
        synthetic_audit(self.root, ratio=1.36, direction_count=8)

    def change_summary(self, mutate):
        path = self.root / "qc/validity-summary.json"
        value = read_json(path)
        mutate(value)
        write_json(path, value)

    def change_table(self, name, mutate):
        path = self.root / name
        value = pd.read_csv(path)
        mutate(value)
        value.to_csv(path, index=False)
        self.change_summary(lambda s: s["artifacts"].update({name: sha256_file(path)}))

    def assert_corrupt(self):
        with self.assertRaises((ValueError, OSError, KeyError, TypeError)) as caught:
            require_audits(self.root)
        self.assertNotIsInstance(caught.exception, ValidityAuditRejected)

    def test_verified_rejection_finishes_notebook_without_training(self):
        with patch.object(workflow, "run_stage") as run, contextlib.redirect_stdout(io.StringIO()):
            for command in ("audit-teacher", "run-gate"):
                outcome = workflow.attempt_stage(command, self.root)
                workflow.require_stage_success(outcome)
                self.assertIsInstance(outcome, ValidityAuditRejected)
            run.assert_not_called()
        self.assertFalse(list(self.root.glob("models/fold-*/*")))

    def test_pass_and_rejection_use_complete_numeric_verification(self):
        self.assertFalse(verify_audits(self.root)["passed"])
        with self.assertRaisesRegex(ValidityAuditRejected, "1.360 < 2"):
            require_audits(self.root)
        synthetic_audit(self.root)
        self.assertTrue(require_audits(self.root)["passed"])

    def test_missing_and_inconsistent_summary_fields_are_errors(self):
        original = read_json(self.root / "qc/validity-summary.json")
        mutations = [
            lambda s: s.update(passed=True),
            lambda s: s.update(passed=0),
            lambda s: s["checks"].update(target_sensitivity=0),
            lambda s: s["checks"].update(target_sensitivity=True),
            lambda s: s["checks"].update(target_variance_valid=False),
            lambda s: s.update(motion_to_background_change_ratio=3.0),
            lambda s: s.update(motion_to_background_change_ratio=True),
            lambda s: s.update(person_edit_direction_fraction="0.8"),
            lambda s: s["artifacts"].pop("qc/target-sensitivity.csv"),
            lambda s: s["artifacts"].pop(next(p for p in s["artifacts"] if p.endswith(".jpg"))),
        ]
        for mutate in mutations:
            with self.subTest(mutate=mutate):
                write_json(self.root / "qc/validity-summary.json", original)
                self.change_summary(mutate)
                self.assert_corrupt()

    def test_overflow_and_nonstandard_json_infinity_cannot_pass_the_gate(self):
        synthetic_audit(self.root)
        self.change_table("qc/target-sensitivity.csv", lambda t: t.__setitem__("person_change", 1e308))
        summary = read_json(self.root / "qc/validity-summary.json")
        summary["motion_to_background_change_ratio"] = float("inf")
        # Deliberately bypass the strict writer to model a malformed input file.
        (self.root / "qc/validity-summary.json").write_text(json.dumps(summary))
        self.assert_corrupt()

    def test_rehashed_corrupt_rows_still_fail(self):
        cases = [
            ("target-sensitivity", "person_change", -1.0),
            ("target-sensitivity", "background_change", float("nan")),
            ("target-sensitivity", "person_larger", False),
            ("target-sensitivity", "donor_window_id", "synthetic-window-00"),
            ("target-sensitivity", "window_id", "synthetic-window-01"),
            ("teacher-stability", "passed", False),
            ("teacher-stability", "context_max_abs", 0.01),
            ("causal-leakage", "tolerance", 0.1),
            ("causal-leakage", "mean_abs_difference", 0.1),
            ("causal-leakage", "stable", False),
        ]
        for name, column, value in cases:
            with self.subTest(name=name, column=column):
                synthetic_audit(self.root, ratio=1.36, direction_count=8)
                self.change_table(f"qc/{name}.csv", lambda t: t.loc.__setitem__((0, column), value))
                self.assert_corrupt()

    def test_missing_artifact_and_modified_plan_are_errors(self):
        sheet = next(self.root.glob("qc/pixel-edit-contact-sheets/*.jpg"))
        original = sheet.read_bytes()
        sheet.unlink()
        self.assert_corrupt()
        sheet.write_bytes(original)
        plan = self.root / "manifests/audit-windows.csv"
        plan.write_text(plan.read_text() + "\n")
        self.assert_corrupt()

    def test_consistent_stability_and_leakage_rejections_remain_blocked(self):
        self.change_table("qc/teacher-stability.csv", lambda t: t.loc.__setitem__((0, "target_max_abs"), 0.1))
        self.change_table("qc/teacher-stability.csv", lambda t: t.loc.__setitem__((0, "passed"), False))
        self.change_table("qc/causal-leakage.csv", lambda t: t.loc.__setitem__((0, "max_abs_difference"), 0.1))
        self.change_table("qc/causal-leakage.csv", lambda t: t.loc.__setitem__((0, "passed"), False))
        self.change_summary(lambda s: s["checks"].update(teacher_stable=False, causal_leakage_absent=False))
        with self.assertRaises(ValidityAuditRejected) as caught:
            require_audits(self.root)
        self.assertEqual(set(caught.exception.failed_checks),
                         {"target_sensitivity", "teacher_stable", "causal_leakage_absent"})

    def test_report_is_current_unsealed_stop_without_predictive_metrics(self):
        with patch.object(workflow, "run_stage", side_effect=lambda command, root: build_report(root)) as stage:
            attempt = workflow.build_notebook_report(self.root)
            result = workflow.finish_notebook_report(self.root, scoring_succeeded=attempt)
        self.assertEqual(stage.call_args.args[0], "build-report")
        self.assertFalse(result["measurement_complete"])
        self.assertIsNone(result["metrics"])
        self.assertFalse(result["allow_full_experiment"])
        self.assertFalse(result["allow_adapter_training"])
        self.assertFalse((self.root / "reports/final-report-contract.json").exists())
        (self.root / "reports/gate-report.md").write_text("stale narrative")
        with self.assertRaisesRegex(ValueError, "report changed"):
            workflow.finish_notebook_report(self.root, scoring_succeeded=attempt)

    def test_legacy_stop_or_complete_report_cannot_certify_current_rejection(self):
        original = build_report(self.root)
        for changes in ({"audit_summary_sha256": None}, {"measurement_complete": True},
                        {"allow_full_experiment": True}, {"allow_adapter_training": True},
                        {"metrics": {}}, {"synthetic": False}, {"run_id": "wrong-run"}):
            with self.subTest(changes=changes):
                write_json(self.root / "reports/gate-decision.json", {**original, **changes})
                with patch.object(workflow, "run_stage"), self.assertRaisesRegex(ValueError, "does not match"):
                    workflow.build_notebook_report(self.root)

    def test_scoring_runtime_error_is_not_hidden_by_stale_stop(self):
        build_report(self.root)
        synthetic_audit(self.root)  # A passing fixture still has no fitted predictions.
        with patch.object(workflow, "run_stage", side_effect=[subprocess.CalledProcessError(1, ["score"]), None]):
            attempt = workflow.build_notebook_report(self.root)
        self.assertIs(attempt, False)
        with self.assertRaises(RuntimeError):
            workflow.finish_notebook_report(self.root, scoring_succeeded=attempt)

    def test_corrupt_audit_still_builds_diagnostics_and_fails(self):
        self.change_summary(lambda s: s.update(passed=True))
        def stage(command, root):
            if command == "score-gate":
                raise subprocess.CalledProcessError(1, [command])
            return build_report(root)
        with patch.object(workflow, "run_stage", side_effect=stage) as run:
            attempt = workflow.build_notebook_report(self.root)
        self.assertIs(attempt, False)
        self.assertEqual([call.args[0] for call in run.call_args_list], ["score-gate", "build-report"])
        decision = read_json(self.root / "reports/gate-decision.json")
        self.assertIn("passed flag disagrees", decision["reason"])
        self.assertNotIn("workflow_status", decision)
        with self.assertRaises(RuntimeError):
            workflow.finish_notebook_report(self.root, scoring_succeeded=attempt)

    def test_reused_audit_cli_returns_rejection_code_without_teacher(self):
        with patch("gavd6_sjepa.research_directions.future_prediction.cli.teacher_arguments") as args:
            from gavd6_sjepa.research_directions.future_prediction.cli import audits_main
            args.return_value.run_root = self.root
            with patch("gavd6_sjepa.research_directions.future_prediction.vjepa.FrozenVJEPAAdapter.from_run") as teacher:
                self.assertEqual(audits_main(), 2)
                teacher.assert_not_called()

    def test_new_audit_cli_failure_requires_verified_evidence(self):
        (self.root / "qc/validity-summary.json").unlink()
        def rejected_stage(*args):
            synthetic_audit(self.root, ratio=1.36, direction_count=8)
            raise subprocess.CalledProcessError(2, ["audit"])
        with patch.object(workflow, "run_stage", side_effect=rejected_stage):
            self.assertIsInstance(workflow.attempt_stage("audit-teacher", self.root), ValidityAuditRejected)


# The direct protocol measures skeleton increment without selectivity prerequisites.


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
        from gavd6_sjepa.research_directions.future_prediction import notebook_workflow as workflow
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
        from gavd6_sjepa.research_directions.future_prediction import notebook_workflow as workflow
        from gavd6_sjepa.research_directions.future_prediction.reporting import build_report
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
        with patch("gavd6_sjepa.research_directions.future_prediction.readiness.load_cache", return_value=(cohort, arrays)):
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


class FutureInnovationGateTests(unittest.TestCase):
    def valid(self):
        return {
            "delta_r2_real": 0.06,
            "delta_r2_time_shuffle": 0.02,
            "delta_r2_clip_mismatch": 0.005,
            "delta_r2_background_target": 0.02,
            "delta_r2_no_skeleton": 0.005,
            "motion_to_background_change_ratio": 3.0,
            "person_edit_direction_fraction": 0.9,
            "bootstrap_positive_fraction": 0.95,
            "seed_real_gains": [0.06, 0.06, 0.06],
            **{
                key: True
                for key in (
                    "data_contract_valid",
                    "evaluation_contract_valid",
                    "controls_complete",
                    "target_audit_complete",
                    "target_variance_valid",
                    "teacher_stable",
                    "causal_leakage_absent",
                )
            },
        }

    def test_only_complete_stable_point_pass_advances_without_adapter_authority(self):
        result = decide_gate(self.valid())
        self.assertEqual(result["decision"], "ADVANCE")
        self.assertTrue(result["allow_full_experiment"])
        self.assertFalse(result["allow_adapter_training"])

    def test_every_validity_failure_stops(self):
        for key in (
            "data_contract_valid",
            "evaluation_contract_valid",
            "controls_complete",
            "target_audit_complete",
            "target_variance_valid",
            "teacher_stable",
            "causal_leakage_absent",
        ):
            for value in (False, None, 1):
                with self.subTest(key=key, value=value):
                    self.assertEqual(
                        decide_gate({**self.valid(), key: value})["decision"], "STOP"
                    )

    def test_point_failures_and_direction_consistency(self):
        for change in (
            {"delta_r2_time_shuffle": 0.04},
            {"delta_r2_clip_mismatch": 0.02},
            {"delta_r2_background_target": 0.04},
            {"motion_to_background_change_ratio": 1.0},
            {"person_edit_direction_fraction": 0.7},
            {"delta_r2_real": 0.04, "seed_real_gains": [0.04, 0.04, 0.04]},
        ):
            self.assertEqual(
                decide_gate({**self.valid(), **change})["decision"], "STOP"
            )

    def test_instability_is_inconclusive(self):
        for change in (
            {"bootstrap_positive_fraction": 0.7},
            {"seed_real_gains": [0.16, 0.01, 0.01]},
        ):
            self.assertEqual(
                decide_gate({**self.valid(), **change})["decision"], "INCONCLUSIVE"
            )

    def test_retired_manual_capacity_field_does_not_gate_advance(self):
        result = decide_gate({**self.valid(), "capacity_control_clear": False})
        self.assertEqual(result["decision"], "ADVANCE")
        self.assertNotIn("capacity_control_clear", result["checks"])

    def test_missing_seed_nonfinite_and_inconsistent_aggregation_stop(self):
        for change in (
            {"seed_real_gains": [0.06, 0.06]},
            {"seed_real_gains": [0.07, 0.07, 0.07]},
            {"delta_r2_no_skeleton": float("nan")},
            {"delta_r2_real": float("inf")},
            {"bootstrap_positive_fraction": 1.1},
        ):
            self.assertEqual(
                decide_gate({**self.valid(), **change})["decision"], "STOP"
            )


if __name__ == "__main__":
    unittest.main()
