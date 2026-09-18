"""Submission retries must not allocate duplicate jobs or erase accepted IDs."""
import contextlib
import importlib.util
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

from gavd6_sjepa.research_directions.synthetic_training_v2.config import RunConfig


REPO = Path(__file__).resolve().parents[2]
SPEC = importlib.util.spec_from_file_location(
    "stv2_submission_safety", REPO / "scripts/research_directions/synthetic_training_v2/submit.py"
)
SUBMIT = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(SUBMIT)


class SubmissionSafetyTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        bundle = self.root / "bundle"
        bundle.mkdir()
        (bundle / "manifest.json").write_text("{}\n")
        ledger = self.root / "prior.json"
        ledger.write_text(json.dumps({"scope_authorized": True, "entries": []}))
        self.cfg = RunConfig(
            run_id="submission-fixture", output_root=str(self.root), mode="source",
            bundle=str(bundle), device="cuda", authorized_gpu_hours=3,
            projected_gpu_hours=.9, cost_ledger=str(ledger),
        )
        self.config = self.root / "source.json"
        self.config.write_text(json.dumps(self.cfg.as_dict()))
        self.record = self.cfg.root / "submission.json"
        self.environment = patch.dict(os.environ, {"STV2_PYTHON": "/test/bin/python"})
        self.environment.start()
        self.addCleanup(self.environment.stop)

    def launch(self, *extra):
        with patch.object(sys, "argv", ["submit.py", "--config", str(self.config), *extra]), \
                contextlib.redirect_stdout(io.StringIO()):
            SUBMIT.main()

    def test_repeated_submission_refuses_and_preserves_job_ids(self):
        with patch.object(SUBMIT.subprocess, "check_output", side_effect=[f"{1000 + i}\n" for i in range(10)]) as sbatch:
            self.launch()
            original = self.record.read_bytes()
            self.assertFalse((self.cfg.root / "logs/submission-pending.json").exists())
            with self.assertRaisesRegex(FileExistsError, "[Ss]ubmission.*recorded"):
                self.launch()
            self.assertEqual(sbatch.call_count, 10)
        self.assertEqual(self.record.read_bytes(), original)
        self.assertEqual(len(json.loads(original)), 10)

    def test_partial_failure_keeps_each_accepted_job_and_blocks_full_retry(self):
        calls = []

        def accept_then_fail(command, text, env=None):
            calls.append(command)
            if len(calls) == 1:
                return "2001;cluster\n"
            previous = json.loads(self.record.read_text())
            self.assertEqual(len(previous), len(calls) - 1)
            if len(calls) == 3:
                raise subprocess.CalledProcessError(1, command, "injected scheduler failure")
            self.assertIn("--dependency=afterok:2001", command)
            return "2002\n"

        with patch.object(SUBMIT.subprocess, "check_output", side_effect=accept_then_fail) as sbatch:
            with self.assertRaises(subprocess.CalledProcessError):
                self.launch()
            original = self.record.read_bytes()
            with self.assertRaisesRegex(FileExistsError, "squeue|sacct"):
                self.launch()
            self.assertEqual(sbatch.call_count, 3)
        self.assertEqual(self.record.read_bytes(), original)
        self.assertEqual(json.loads(original), [
            {"stage": "audit", "job_id": "2001"}, {"stage": "data", "job_id": "2002"},
        ])

    def test_concurrent_submission_is_locked_before_first_job_is_recorded(self):
        calls = []

        def scheduler(command, text, env=None):
            calls.append(command)
            if len(calls) == 1:
                self.assertFalse(self.record.exists())
                with self.assertRaisesRegex(ValueError, "already running: submit"):
                    self.launch()
            return f"{3000 + len(calls)}\n"

        with patch.object(SUBMIT.subprocess, "check_output", side_effect=scheduler):
            self.launch()
        self.assertEqual(len(calls), 10)
        self.assertEqual(len(json.loads(self.record.read_text())), 10)

    def test_dry_run_does_not_create_run_directory_or_submit(self):
        before = {str(p.relative_to(self.root)): p.read_bytes() for p in self.root.rglob("*") if p.is_file()}
        with patch.object(SUBMIT.subprocess, "check_output") as sbatch:
            self.launch("--dry-run")
            sbatch.assert_not_called()
        self.assertFalse(self.cfg.root.exists())
        after = {str(p.relative_to(self.root)): p.read_bytes() for p in self.root.rglob("*") if p.is_file()}
        self.assertEqual(before, after)

    def test_exported_account_and_partition_override_batch_defaults(self):
        with patch.dict(os.environ, {"ST_ACCOUNT": "research-account", "ST_PARTITION": "gpu-partition"}):
            commands = SUBMIT.commands(self.config, dependency="999")
        self.assertEqual(len(commands), 10)
        for command in commands:
            self.assertIn("--account=research-account", command)
            self.assertIn("--partition=gpu-partition", command)
        self.assertIn("--dependency=afterok:999", commands[0])
        self.assertEqual(sum("--gres=gpu:h100:1" in command for command in commands), 2)

    def test_automatic_requeue_is_disabled_in_launcher_and_batch_defaults(self):
        for command in SUBMIT.commands(self.config):
            self.assertIn("--no-requeue", command)
        for name in ("preflight.sbatch", "prepare.sbatch", "stage.sbatch"):
            contents = (REPO / "slurm/synthetic-training-v2" / name).read_text()
            self.assertIn("#SBATCH --no-requeue", contents, name)

    def test_submission_removes_inherited_slurm_overrides_only(self):
        with patch.dict(os.environ, {
            "SBATCH_GPUS": "8", "SBATCH_GRES": "gpu:8", "SBATCH_NODES": "4",
            "SBATCH_REQUEUE": "1", "ST_MODEL_ROOT": "/keep/models",
        }):
            with patch.object(SUBMIT.subprocess, "check_output",
                              side_effect=[f"{4000 + i}\n" for i in range(10)]) as sbatch:
                self.launch()
            self.assertEqual(os.environ["SBATCH_GPUS"], "8")
        for call in sbatch.call_args_list:
            env = call.kwargs.get("env")
            self.assertIsNotNone(env, "A sanitized environment must be passed to sbatch")
            self.assertFalse(any(key.startswith("SBATCH_") for key in env))
            self.assertEqual(env["ST_MODEL_ROOT"], "/keep/models")
            self.assertEqual(env["STV2_PYTHON"], "/test/bin/python")

    def test_lost_first_scheduler_response_blocks_retry_without_erasing_uncertainty(self):
        marker = self.cfg.root / "logs/submission-pending.json"

        def uncertain(command, text, env):
            self.assertTrue(marker.is_file(), "Record uncertainty before contacting Slurm")
            pending = json.loads(marker.read_text())
            self.assertEqual(pending["stage"], "audit")
            self.assertEqual(pending["command"], command)
            raise subprocess.CalledProcessError(1, command, "scheduler response was lost")

        with patch.object(SUBMIT.subprocess, "check_output", side_effect=uncertain) as sbatch:
            with self.assertRaises(subprocess.CalledProcessError):
                self.launch()
            self.assertFalse(self.record.exists())
            original = marker.read_bytes()
            with self.assertRaisesRegex(FileExistsError, "[Uu]ncertain.*squeue|sacct"):
                self.launch()
            self.assertEqual(sbatch.call_count, 1)
        self.assertEqual(marker.read_bytes(), original)


if __name__ == "__main__":
    unittest.main()
