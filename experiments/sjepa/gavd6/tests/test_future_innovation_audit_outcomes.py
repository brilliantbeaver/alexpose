"""A rejected measurement blocks training without hiding execution defects."""

import contextlib
import io
import json
import tempfile
import unittest
import subprocess
from pathlib import Path
from unittest.mock import patch

import pandas as pd

from gavd6_sjepa.research_directions.future_innovation import fi_notebook_workflow as workflow
from gavd6_sjepa.research_directions.future_innovation.fi_contracts import read_json, write_json, sha256_file
from gavd6_sjepa.research_directions.future_innovation.fi_smoke import synthetic_cache, synthetic_audit
from gavd6_sjepa.research_directions.future_innovation.fi_validity_audits import (
    ValidityAuditRejected, require_audits, verify_audits,
)
from gavd6_sjepa.research_directions.future_innovation.fi_reporting import build_report


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
        with patch("gavd6_sjepa.research_directions.future_innovation.fi_entrypoint.teacher_arguments") as args:
            from gavd6_sjepa.research_directions.future_innovation.fi_entrypoint import audits_main
            args.return_value.run_root = self.root
            with patch("gavd6_sjepa.research_directions.future_innovation.fi_vjepa_adapter.FrozenVJEPAAdapter.from_run") as teacher:
                self.assertEqual(audits_main(), 2)
                teacher.assert_not_called()

    def test_new_audit_cli_failure_requires_verified_evidence(self):
        (self.root / "qc/validity-summary.json").unlink()
        def rejected_stage(*args):
            synthetic_audit(self.root, ratio=1.36, direction_count=8)
            raise subprocess.CalledProcessError(2, ["audit"])
        with patch.object(workflow, "run_stage", side_effect=rejected_stage):
            self.assertIsInstance(workflow.attempt_stage("audit-teacher", self.root), ValidityAuditRejected)


if __name__ == "__main__":
    unittest.main()
