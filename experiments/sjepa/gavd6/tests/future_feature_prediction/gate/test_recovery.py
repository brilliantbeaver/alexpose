"""future feature prediction / gate / test recovery."""


import contextlib
import importlib.util
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch

from gavd6_sjepa.research_directions.future_prediction import cli, contracts
from gavd6_sjepa.research_directions.future_prediction.nested_training import fit_outer_fold
from gavd6_sjepa.research_directions.future_prediction.reporting import build_report, score_gate
from gavd6_sjepa.research_directions.future_prediction.smoke import synthetic_cache
from tests.support import REPO_ROOT

# Operational changes must not invalidate good data or poison interrupted runs.


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
                "gavd6_sjepa.research_directions.future_prediction.vjepa.FrozenVJEPAAdapter.from_run",
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


ROOT = REPO_ROOT

spec = importlib.util.spec_from_file_location('source_curve_stage_runner', ROOT/'scripts/research_directions/source_scaling/run_stage.py')

runner = importlib.util.module_from_spec(spec)

spec.loader.exec_module(runner)

class StageExecutionTests(unittest.TestCase):
    def test_failure_and_retry_keep_distinct_logs_and_exit_codes(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for code in (7, 0):
                result = runner.run_stage(root, 'run', [sys.executable, '-c',
                    f'print("synthetic subprocess fixture"); raise SystemExit({code})'], fold=2)
                self.assertEqual(result, code)
            records = [json.loads(p.read_text()) for p in (root/'logs/stages').glob('*.json')]
            self.assertEqual(len(records), 2)
            self.assertEqual({r['status'] for r in records}, {'failed', 'passed'})
            for record in records:
                self.assertEqual(record['fold'], 2)
                self.assertIn('synthetic subprocess', (root/record['log']).read_text())
            self.assertFalse((root/'reports').exists())

    def test_startup_failure_retains_error(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self.assertEqual(runner.run_stage(root, 'prepare', ['/nonexistent/source-curve-test-executable']), 2)
            record = json.loads(next((root/'logs/stages').glob('*.json')).read_text())
            self.assertEqual(record['status'], 'failed')
            self.assertIn('FileNotFoundError', record['error'])

    def test_verification_record_requires_existing_unchanged_bindings(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            command = [sys.executable, '-c', 'print("synthetic verification command fixture")']
            self.assertEqual(runner.run_stage(root, 'verify', command), 2)
            for relative in runner.BINDINGS:
                p = root/relative; p.parent.mkdir(parents=True, exist_ok=True)
                p.write_text('{"protocol":"source-learning-curve-v1"}')
            before = runner.binding(root)
            self.assertEqual(runner.run_stage(root, 'verify', command), 0)
            command = [sys.executable, '-c', 'import os; from pathlib import Path; '
                       '(Path(os.environ["FI_RUN_ROOT"])/"reports/learning-curve.json").write_text("changed")']
            self.assertEqual(runner.run_stage(root, 'verify', command), 2)
            self.assertNotEqual(runner.binding(root), before)

    def test_actual_cli_failure_reaches_structured_log(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)/'missing study'
            result = subprocess.run([sys.executable, runner.__file__, 'run', '--run-root', str(root), '--fold', '3'],
                                    capture_output=True, text=True)
            self.assertNotEqual(result.returncode, 0)
            record = json.loads(next((root/'logs/stages').glob('*.json')).read_text())
            self.assertEqual(record['returncode'], result.returncode)
            self.assertEqual(record['status'], 'failed')
            self.assertIn('study.json', record['output_tail'])
            self.assertIn('--fold', record['command'])

    def test_historical_gate_and_escaped_logs_are_rejected_without_writes(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)/'gate'; (root/'config').mkdir(parents=True)
            (root/'config/run-contract.json').write_text('{"protocol":"direct-v2"}')
            with self.assertRaisesRegex(ValueError, 'historical gate'):
                runner.run_stage(root, 'run', [sys.executable, '-c', 'pass'])
            self.assertFalse((root/'logs').exists())
            root = Path(tmp)/'child'; root.mkdir()
            other = Path(tmp)/'parent'; other.mkdir()
            (root/'logs').symlink_to(other, target_is_directory=True)
            with self.assertRaisesRegex(ValueError, 'stay inside'):
                runner.run_stage(root, 'run', [sys.executable, '-c', 'pass'])
            self.assertEqual(list(other.iterdir()), [])

    def test_all_jobs_route_to_logged_stages_and_cache_failure_stops_audits(self):
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            interpreter = directory/'python-standin'
            interpreter.write_text(f'#!{sys.executable}\n' + '''import json,os,sys
from pathlib import Path
p=Path(os.environ['TEST_STAGE_COMMANDS'])
rows=json.loads(p.read_text()) if p.exists() else []
rows.append(sys.argv[1:]); p.write_text(json.dumps(rows))
raise SystemExit(int(os.environ.get('TEST_STAGE_EXIT','0')))
''')
            interpreter.chmod(0o755)
            output = directory/'commands.json'
            env = {**os.environ, 'GAVD6_ROOT':str(ROOT),'FI_RUN_ROOT':str(directory/'new run'),
                   'FI_SCALING_ROOT':str(directory/'new run'),'FI_PARENT_ROOT':str(directory/'gate-v2'),
                   'FI_PYTHON':str(interpreter),'FI_ANNOTATION_ROOT':str(directory/'annotations'),
                   'FI_VIDEO_ROOT':str(directory/'videos'),'FI_POSE_MODEL':str(directory/'pose.task'),
                   'VJEPA2_ROOT':str(directory/'teacher'),'FI_TEACHER_CHECKPOINT':str(directory/'teacher.pt'),
                   'SLURM_ARRAY_TASK_ID':'3','TEST_STAGE_COMMANDS':str(output), 'TEST_STAGE_EXIT':'0'}
            jobs = [('initialize.sbatch',['initialize']), ('prepare-data.sbatch',['prepare']),
                    ('cache-features.sbatch',['cache','audit-teacher','plan']),
                    ('fit-models.sbatch',['run']), ('build-report.sbatch',['report','verify'])]
            for script, stages in jobs:
                result = subprocess.run(['bash',str(ROOT/'slurm/source-scaling'/script)],
                                        env=env,capture_output=True,text=True)
                self.assertEqual(result.returncode,0,result.stderr)
                commands = json.loads(output.read_text())
                self.assertEqual([c[1] for c in commands],stages)
                self.assertTrue(all(c[0].endswith('scripts/research_directions/source_scaling/run_stage.py') for c in commands))
                if stages==['run']: self.assertEqual(commands[0][-2:],['--fold','3'])
                output.unlink()
            result = subprocess.run(['bash',str(ROOT/'slurm/source-scaling/cache-features.sbatch')],
                                    env={**env,'TEST_STAGE_EXIT':'7'},capture_output=True,text=True)
            self.assertEqual(result.returncode,7)
            self.assertEqual(len(json.loads(output.read_text())),1)


if __name__ == "__main__":
    unittest.main()
