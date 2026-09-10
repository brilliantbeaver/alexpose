"""Operational changes must not invalidate good data or poison interrupted runs."""

import contextlib
import io
import os
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch

from gavd6_sjepa.research_directions.future_innovation import fi_contracts as contracts
from gavd6_sjepa.research_directions.future_innovation import fi_entrypoint as cli
from gavd6_sjepa.research_directions.future_innovation.fi_nested_training import fit_outer_fold
from gavd6_sjepa.research_directions.future_innovation.fi_reporting import build_report, score_gate
from gavd6_sjepa.research_directions.future_innovation.fi_smoke import synthetic_cache


class FutureInnovationResumptionTests(unittest.TestCase):
    def test_initialization_does_not_require_exact_torch_build_or_change_reason(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            fixture = root / "input"
            fixture.write_text("initialization fixture")
            (root / "youtube/all").mkdir(parents=True)
            args = ["fi", "--run-root", str(root / "run"), "--youtube-dir", str(root / "youtube")]
            for flag in ("--sequence-manifest", "--video-manifest", "--annotations", "--pose-model", "--checkpoint"):
                args.extend([flag, str(fixture)])
            args.extend(["--vjepa-root", str(root)])
            with patch("sys.argv", args), patch.object(
                cli.subprocess, "check_output", return_value=contracts.VJEPA_COMMIT
            ), patch.object(
                cli.subprocess, "run", return_value=SimpleNamespace(stdout="test environment")
            ), patch.object(torch, "__version__", "compatible-build"), contextlib.redirect_stdout(io.StringIO()):
                cli.init_main()
            self.assertEqual(contracts.check_run(root / "run")["change_reason"], "Experiment 0 initialization")

    def test_requeued_job_gets_a_new_runtime_record(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with patch.dict(os.environ, {"SLURM_JOB_ID": "123"}), patch.object(torch.cuda, "is_available", return_value=False):
                cli.record_teacher_runtime(root, "cache")
                with patch.object(torch, "__version__", "compatible-update"):
                    cli.record_teacher_runtime(root, "cache")
            self.assertEqual(len(list(root.glob("logs/cache-runtime-123-*.json"))), 2)

    def test_invalid_configuration_still_leaves_report_diagnostic(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            synthetic_cache(root)
            (root / "config/thresholds.json").write_text("{}")
            with patch.dict(os.environ, {"FI_RUN_ROOT": str(root)}), patch("sys.argv", ["fi"]), contextlib.redirect_stderr(io.StringIO()):
                self.assertEqual(cli.report_main(), 1)
            diagnostic = contracts.read_json(root / "reports/pipeline-error.json")
            self.assertIn("Frozen configuration changed", diagnostic["error"])
            self.assertFalse(diagnostic["allow_full_experiment"])
            self.assertFalse((root / "reports/final-report-contract.json").exists())

    def test_code_and_runtime_drift_preserve_contracts_and_are_recorded(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            synthetic_cache(root)
            original = (root / "config/run-contract.json").read_bytes()
            with patch.object(contracts, "code_fingerprint", return_value="updated-code"), patch.object(
                contracts, "runtime_versions", return_value={"python": "compatible-update"}
            ):
                contracts.check_run(root)
                contracts.check_run(root)
            self.assertEqual((root / "config/run-contract.json").read_bytes(), original)
            records = [contracts.read_json(p) for p in root.glob("logs/provenance/*.json")]
            changed = [r for r in records if r["code_sha256"] == "updated-code"]
            self.assertEqual(len(changed), 1)
            self.assertTrue(changed[0]["code_changed"])
            self.assertTrue(changed[0]["runtime_changed"])
            self.assertEqual(changed[0]["runtime"], {"python": "compatible-update"})
            # Actual edits to the frozen protocol still invalidate the run.
            (root / "config/thresholds.json").write_text("{}")
            with self.assertRaisesRegex(ValueError, "Frozen configuration changed"):
                contracts.check_run(root)

    def test_completed_cache_and_audits_do_not_load_teacher(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            synthetic_cache(root)
            with patch.dict(os.environ, {"FI_RUN_ROOT": str(root)}), patch(
                "sys.argv", ["fi"]
            ), patch(
                "gavd6_sjepa.research_directions.future_innovation.fi_vjepa_adapter.FrozenVJEPAAdapter.from_run",
                side_effect=AssertionError("completed stages must not load a model"),
            ), contextlib.redirect_stdout(io.StringIO()):
                cli.cache_main()
                self.assertEqual(cli.audits_main(), 0)

    def test_early_stop_can_resume_but_complete_results_stay_sealed(self):
        torch.set_num_threads(1)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            synthetic_cache(root)
            early = build_report(root)
            self.assertEqual(early["decision"], "STOP")
            self.assertFalse(early["measurement_complete"])
            self.assertFalse((root / "reports/final-report-contract.json").exists())
            # Simulate an old implementation that sealed an incomplete STOP.
            legacy = dict(early)
            legacy.pop("measurement_complete")
            contracts.write_json(root / "reports/gate-decision.json", legacy)
            contracts.write_json(root / "reports/final-report-contract.json", {
                "artifacts": {
                    name: contracts.sha256_file(root / name)
                    for name in ("reports/gate-decision.json", "reports/gate-report.md")
                }
            })
            for fold in range(5):
                fit_outer_fold(root, fold)
            score_gate(root)
            final = build_report(root)
            self.assertTrue(final["measurement_complete"])
            self.assertTrue(final["synthetic"])
            self.assertFalse(final["allow_full_experiment"])
            self.assertEqual(len(list(root.glob("reports/attempts/*/gate-decision.json"))), 1)
            sealed = (root / "reports/final-report-contract.json").read_bytes()
            fit_outer_fold(root, 0)  # A completed fold is safe to reuse after reporting.
            self.assertEqual(build_report(root), final)
            self.assertEqual((root / "reports/final-report-contract.json").read_bytes(), sealed)
            (root / "reports/gate-report.md").write_text("tampered")
            with self.assertRaisesRegex(ValueError, "Sealed final report changed"):
                build_report(root)
