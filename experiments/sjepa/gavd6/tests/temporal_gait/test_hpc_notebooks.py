"""Temporal-gait notebook execution and source-data-free Slurm orchestration."""
from __future__ import annotations

import importlib.util
from contextlib import redirect_stdout
from dataclasses import replace
import io
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import MagicMock, patch

import nbformat

ROOT = Path(__file__).resolve().parents[2]
SCRIPTS = ROOT / "scripts/research_directions/temporal_gait"
sys.path.insert(0, str(SCRIPTS))


def load(name, filename):
    spec = importlib.util.spec_from_file_location(name, SCRIPTS / filename)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


runner = load("temporal_gait_notebook_executor_test", "execute_notebook.py")
builder = load("temporal_gait_notebook_builder_test", "build_notebooks.py")
submitter = load("temporal_gait_submitter_test", "submit.py")
verifier = load("temporal_gait_software_verifier_test", "verify_software.py")
stage_cli = sys.modules["run_stage"]
WORKFLOW = "gavd6_sjepa.research_directions.temporal_gait.workflow"


class NotebookTests(unittest.TestCase):
    def test_generated_sources_reproduce_and_are_output_free(self):
        generated = builder.render_all()
        self.assertEqual(set(generated), set(runner.NOTEBOOKS.values()))
        for name, expected in generated.items():
            actual = nbformat.read(runner.SOURCE / name, as_version=4)
            self.assertEqual(actual, expected, name)
            codes = [cell for cell in actual.cells if cell.cell_type == "code"]
            self.assertGreaterEqual(len(codes), 4)
            for cell in codes:
                self.assertEqual(cell.outputs, [])
                self.assertIsNone(cell.execution_count)
                compile(cell.source, name, "exec")

    def test_inspection_keeps_execution_disabled_and_clears_stale_task(self):
        captured = []
        def client(notebook, **kwargs):
            return MagicMock(execute=lambda **options: captured.append(options["env"]))
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            clean = {key: value for key, value in os.environ.items() if not key.startswith("TG_")}
            clean["TG_TASK_ID"] = "999"
            with patch.dict(os.environ, clean, clear=True), \
                 patch.object(runner, "KernelManager", return_value=MagicMock(has_kernel=False)), \
                 patch.object(runner, "NotebookClient", side_effect=client):
                destination = runner.execute_notebook("00", run_root=root / "run")
                runner.execute_notebook("03", run_root=root / "run", phase="develop", task_id=7,
                                        execute=True)
            self.assertEqual(captured[0]["TG_EXECUTE"], "0")
            self.assertNotIn("TG_TASK_ID", captured[0])
            self.assertEqual(captured[1]["TG_EXECUTE"], "1")
            self.assertEqual(captured[1]["TG_TASK_ID"], "7")
            self.assertEqual(captured[1]["TG_PHASE"], "develop")
            self.assertEqual(nbformat.read(destination, as_version=4).metadata.temporal_gait_execution.artifact_status,
                             "plan_only")

    def test_failure_keeps_partial_output_and_source_is_unchanged(self):
        def client(notebook, **kwargs):
            def execute(**options):
                cell = next(c for c in notebook.cells if c.cell_type == "code")
                cell.outputs = [nbformat.v4.new_output("stream", name="stdout", text="retained diagnostic")]
                cell.execution_count = 1
                raise RuntimeError("fixture failure")
            return MagicMock(execute=execute)
        original = (runner.SOURCE / runner.NOTEBOOKS["00"]).read_bytes()
        with tempfile.TemporaryDirectory() as temporary:
            folder = Path(temporary) / "executed"
            with patch.object(runner, "KernelManager", return_value=MagicMock(has_kernel=False)), \
                 patch.object(runner, "NotebookClient", side_effect=client):
                with self.assertRaisesRegex(RuntimeError, "fixture failure"):
                    runner.execute_notebook("00", run_root=Path(temporary), output_dir=folder)
            saved = nbformat.read(folder / runner.NOTEBOOKS["00"], as_version=4)
            self.assertEqual(saved.metadata.temporal_gait_execution.status, "failed")
            self.assertTrue(any("retained diagnostic" in output.get("text", "")
                                for c in saved.cells for output in c.get("outputs", [])))
            with self.assertRaises(FileExistsError):
                runner.execute_notebook("00", run_root=Path(temporary), output_dir=folder)
        self.assertEqual(original, (runner.SOURCE / runner.NOTEBOOKS["00"]).read_bytes())

    def test_wrong_notebook_cannot_open_test(self):
        with tempfile.TemporaryDirectory() as temporary:
            with self.assertRaisesRegex(ValueError, "cannot execute"):
                runner.execute_notebook("03", run_root=Path(temporary), stage="test")

    def test_executed_fixture_metadata_cannot_look_like_real_execution(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            config = root / "synthetic.json"
            config.write_text(json.dumps({"mode": "synthetic"}))
            clean = {key: value for key, value in os.environ.items() if not key.startswith("TG_")}
            with patch.dict(os.environ, clean, clear=True), \
                 patch.object(runner, "KernelManager", return_value=MagicMock(has_kernel=False)), \
                 patch.object(runner, "NotebookClient", return_value=MagicMock()):
                destination = runner.execute_notebook("00", run_root=root / "run", config=config, execute=True)
            record = nbformat.read(destination, as_version=4).metadata.temporal_gait_execution
            self.assertEqual(record.mode, "synthetic")
            self.assertEqual(record.artifact_status, "synthetic_software")
            self.assertEqual(record.status, "completed")

    def test_real_plan_cells_never_load_manifests_or_run_stage(self):
        from gavd6_sjepa.research_directions.temporal_gait import manifests, workflow
        from gavd6_sjepa.research_directions.temporal_gait.config import MANIFEST_FIELDS
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            config = root / "real-plan.json"
            config.write_text(json.dumps({"mode": "real", "run_root": str(root / "uncreated-run"),
                **{name: str(root / "nonexistent" / f"{name}.json") for name in MANIFEST_FIELDS}}))
            environment = {key: value for key, value in os.environ.items() if not key.startswith("TG_")}
            environment.update(GAVD6_ROOT=str(ROOT), TG_CONFIG=str(config), TG_EXECUTE="0")
            namespace = {}
            with patch.dict(os.environ, environment, clear=True), \
                 patch.object(manifests, "read_manifest", side_effect=AssertionError("No manifest reads")) as read, \
                 patch.object(workflow, "run_stage", side_effect=AssertionError("No workflow execution")) as execute, \
                 redirect_stdout(io.StringIO()):
                for cell in builder.render_all()[runner.NOTEBOOKS["00"]].cells:
                    if cell.cell_type == "code":
                        exec(compile(cell.source, "real-plan-cell", "exec"), namespace)
            read.assert_not_called()
            execute.assert_not_called()
            self.assertEqual(namespace["result"]["status"], "plan_only")
            self.assertFalse(namespace["result"]["media_opened"])
            self.assertFalse((root / "uncreated-run").exists())


class CliTests(unittest.TestCase):
    def test_explicit_phase_reaches_workflow_and_incomplete_fails(self):
        function = MagicMock(return_value={"status": "complete"})
        cfg = SimpleNamespace(mode="synthetic")
        with patch.object(stage_cli, "configuration", return_value=cfg), \
             patch.dict(sys.modules, {WORKFLOW: SimpleNamespace(run_stage=function)}), \
             redirect_stdout(io.StringIO()):
            stage_cli.main(["--stage", "evaluate", "--phase", "develop"])
            function.assert_called_once_with(cfg, "evaluate", task_id=None, role="development", phase="develop")
            function.return_value = {"status": "incomplete", "missing_tasks": [5]}
            with self.assertRaisesRegex(RuntimeError, "incomplete"):
                stage_cli.main(["--stage", "evaluate", "--phase", "pilot"])

    def test_software_verifier_help_is_source_data_free(self):
        result = subprocess.run([sys.executable, str(SCRIPTS / "verify_software.py"), "--help"],
                                text=True, capture_output=True, check=True)
        self.assertIn("--run-root", result.stdout)
        self.assertNotIn("--mode", result.stdout)
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary)
            (output / "existing-result").write_text("preserve")
            with self.assertRaises(FileExistsError):
                verifier.verify_software(output)
            self.assertEqual((output / "existing-result").read_text(), "preserve")

    def test_software_verifier_plan_is_complete_and_bounded(self):
        from gavd6_sjepa.research_directions.temporal_gait.config import RunConfig
        payload = json.loads((ROOT / "slurm/temporal-gait/synthetic.example.json").read_text())
        payload["run_root"] = "/tmp/explicit-unused-software-plan"
        cfg = RunConfig(**payload).validate()
        plan = verifier.verification_plan(cfg)
        self.assertEqual([stage for _, stage, _ in plan[:3]], ["inventory", "prepare", "audit"])
        self.assertEqual([task for _, stage, task in plan if stage in ("masked", "future")], [0, 1, 2, 3, 4])
        self.assertEqual([stage for _, stage, _ in plan[-6:]],
                         ["extensions", "cache-video", "evaluate", "calibrate", "test", "aggregate"])
        self.assertEqual(cfg.mode, "synthetic")
        self.assertLessEqual(cfg.updates, 2)

    def test_frozen_task_grid_ignores_resume_path_but_not_science(self):
        from gavd6_sjepa.research_directions.temporal_gait.config import RunConfig
        with tempfile.TemporaryDirectory() as temporary:
            cfg = RunConfig(mode="synthetic", device="cpu", run_root=temporary)
            path = stage_cli.freeze_task_grid(cfg)
            original = path.read_bytes()
            resumed = replace(cfg, resume_from=str(Path(temporary) / "checkpoint-000300.pt"))
            self.assertEqual(stage_cli.freeze_task_grid(resumed), path)
            self.assertEqual(path.read_bytes(), original)
            self.assertIsNone(json.loads(original)["resolved_config"]["resume_from"])
            with self.assertRaisesRegex(ValueError, "Incompatible frozen artifact"):
                stage_cli.freeze_task_grid(replace(resumed, learning_rate=0.002))
            self.assertEqual(path.read_bytes(), original)


class SlurmTests(unittest.TestCase):
    def setUp(self):
        environment = {key: value for key, value in os.environ.items() if not key.startswith("TG_")}
        environment_patch = patch.dict(os.environ, environment, clear=True)
        environment_patch.start()
        self.addCleanup(environment_patch.stop)

    @staticmethod
    def tasks(cfg, phase):
        all_tasks = [dict(task_id=i, arm=arm, seed=seed, fold=0)
                     for i, (arm, seed) in enumerate((arm, seed)
                         for arm in ("masked_index", "masked", "future", "future_wrong_source", "future_wrong_time")
                         for seed in (42, 43, 44, 45, 46))]
        allowed = (42,) if phase == "pilot" else (42, 43, 44) if phase == "develop" else (42, 43, 44, 45, 46)
        return [row for row in all_tasks if row["seed"] in allowed]

    def test_dry_run_never_freezes_grid_submits_or_reads_decision(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "not-created"
            cfg = SimpleNamespace(root=root, mode="real")
            with patch.object(submitter, "configuration", return_value=cfg), \
                 patch.dict(sys.modules, {WORKFLOW: SimpleNamespace(plan_tasks=self.tasks)}), \
                 patch.object(submitter, "freeze_task_grid") as freeze, \
                 patch.object(submitter, "require_expansion") as decision, \
                 patch.object(submitter.subprocess, "run") as sbatch:
                records = submitter.submit("confirm", dry_run=True)
            freeze.assert_not_called()
            decision.assert_not_called()
            sbatch.assert_not_called()
            self.assertFalse(root.exists())
            self.assertEqual(records[-1]["stage"], "evaluate")
            self.assertEqual(set(records[-1]["dependencies"]), {"DRY_masked", "DRY_future"})

    def test_emitted_real_pilot_cuda_stages_request_accelerators(self):
        from gavd6_sjepa.research_directions.temporal_gait.config import MANIFEST_FIELDS
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "not-created"
            environment = {"TG_RUN_ROOT": str(root)}
            environment.update({"TG_" + name.upper(): str(Path(temporary) / "nonexistent" / (name + ".json"))
                                for name in MANIFEST_FIELDS})
            output = io.StringIO()
            with patch.dict(os.environ, environment), \
                 patch.object(submitter.subprocess, "run", side_effect=AssertionError("Dry run must not submit")) as sbatch, \
                 redirect_stdout(output):
                records = submitter.submit("pilot", config=ROOT / "slurm/temporal-gait/pilot.example.json", dry_run=True)
            commands = [shlex.split(line) for line in output.getvalue().splitlines() if line.startswith("sbatch ")]
            self.assertEqual(len(commands), len(records))
            emitted = {record["stage"]: Path(command[-1]) for record, command in zip(records, commands)}
            for stage in ("audit", "masked", "future", "evaluate"):
                launcher = emitted[stage]
                self.assertEqual(launcher, ROOT / "slurm/temporal-gait" / submitter.SCRIPTS[stage])
                self.assertIn("#SBATCH --gres=gpu:h100:1", launcher.read_text(),
                              f"Real CUDA-consuming stage {stage} has no accelerator allocation")
            self.assertFalse(root.exists())
            sbatch.assert_not_called()

    def test_mock_sbatch_receives_complete_arrays_and_phase(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            cfg = SimpleNamespace(root=root / "run", mode="synthetic")
            binary = root / "bin"
            binary.mkdir()
            fake = binary / "sbatch"
            fake.write_text("#!/usr/bin/env python3\nimport json,os,pathlib,sys\n"
                "p=pathlib.Path(os.environ['TG_FAKE_LOG'])\n"
                "n=100+(len(p.read_text().splitlines()) if p.exists() else 0)\n"
                "with p.open('a') as f: f.write(json.dumps({'args':sys.argv[1:],'phase':os.environ['TG_PHASE']})+'\\n')\n"
                "print(str(n)+';fixture-cluster')\n")
            fake.chmod(0o755)
            log = root / "calls.jsonl"
            env = {key: value for key, value in os.environ.items() if not key.startswith("TG_")}
            env.update(PATH=str(binary) + os.pathsep + os.environ["PATH"], TG_FAKE_LOG=str(log))
            def freeze(value):
                path = value.root / "manifests/task-grid.json"
                path.parent.mkdir(parents=True)
                path.write_text(json.dumps(self.tasks(value, "confirm")))
            with patch.dict(os.environ, env, clear=True), \
                 patch.object(submitter, "configuration", return_value=cfg), \
                 patch.dict(sys.modules, {WORKFLOW: SimpleNamespace(plan_tasks=self.tasks)}), \
                 patch.object(submitter, "freeze_task_grid", side_effect=freeze):
                records = submitter.submit("pilot")
            calls = [json.loads(line) for line in log.read_text().splitlines()]
            self.assertEqual([r["stage"] for r in records], ["inventory", "prepare", "audit", "masked", "future", "evaluate"])
            self.assertTrue(all(c["phase"] == "pilot" for c in calls))
            self.assertIn("--array=0,5%2", calls[3]["args"])
            self.assertIn("--array=10,15,20%2", calls[4]["args"])
            self.assertIn("--dependency=afterok:103:104", calls[5]["args"])
            self.assertTrue((cfg.root / "manifests/task-grid.json").is_file())
            self.assertEqual(len((cfg.root / "logs/submissions.tsv").read_text().splitlines()), 6)
            self.assertFalse(any("evaluate-test" in a for call in calls for a in call["args"]))

    def test_stop_decision_prevents_expansion_before_grid_write(self):
        with tempfile.TemporaryDirectory() as temporary:
            cfg = SimpleNamespace(root=Path(temporary), mode="real")
            (cfg.root / "decisions").mkdir()
            (cfg.root / "decisions/development.json").write_text(json.dumps({"ready_for_expansion": False}))
            with patch.object(submitter, "configuration", return_value=cfg), \
                 patch.dict(sys.modules, {WORKFLOW: SimpleNamespace(plan_tasks=self.tasks)}), \
                 patch.object(submitter, "freeze_task_grid") as freeze, \
                 patch.object(submitter.subprocess, "run") as sbatch:
                with self.assertRaisesRegex(RuntimeError, "did not authorize"):
                    submitter.submit("develop")
            freeze.assert_not_called()
            sbatch.assert_not_called()

    def test_scheduler_failure_stops_dependent_submission(self):
        with tempfile.TemporaryDirectory() as temporary:
            cfg = SimpleNamespace(root=Path(temporary), mode="synthetic")
            with patch.object(submitter, "configuration", return_value=cfg), \
                 patch.dict(sys.modules, {WORKFLOW: SimpleNamespace(plan_tasks=self.tasks)}), \
                 patch.object(submitter, "freeze_task_grid"), \
                 patch.object(submitter.subprocess, "run", side_effect=subprocess.CalledProcessError(1, "sbatch")) as sbatch:
                with self.assertRaises(subprocess.CalledProcessError):
                    submitter.submit("pilot")
            self.assertEqual(sbatch.call_count, 1)

    def test_unimplemented_extension_never_allocates_job(self):
        cfg = SimpleNamespace(mode="synthetic")
        with patch.object(submitter, "configuration", return_value=cfg), \
             patch.object(submitter, "freeze_task_grid") as freeze, \
             patch.object(submitter.subprocess, "run") as sbatch, \
             redirect_stdout(io.StringIO()):
            self.assertEqual(submitter.submit("extensions"), [])
            self.assertEqual(submitter.submit("cache-video"), [])
        freeze.assert_not_called()
        sbatch.assert_not_called()

    def test_array_cannot_share_one_notebook_output_folder(self):
        cfg = SimpleNamespace(mode="synthetic")
        with patch.object(submitter, "configuration", return_value=cfg), \
             patch.dict(os.environ, {"TG_NOTEBOOK_OUTPUT_DIR": "/tmp/unused-shared-output"}), \
             patch.dict(sys.modules, {WORKFLOW: SimpleNamespace(plan_tasks=self.tasks)}), \
             patch.object(submitter, "freeze_task_grid") as freeze:
            with self.assertRaisesRegex(ValueError, "cannot be shared"):
                submitter.submit("pilot")
        freeze.assert_not_called()

    def test_single_task_resume_keeps_frozen_grid_and_exports_exact_checkpoint(self):
        from gavd6_sjepa.research_directions.temporal_gait.config import RunConfig
        with tempfile.TemporaryDirectory() as temporary:
            cfg = RunConfig(mode="synthetic", device="cpu", run_root=temporary)
            grid = stage_cli.freeze_task_grid(cfg)
            original = grid.read_bytes()
            checkpoint = str(Path(temporary) / "training/task-0010/checkpoint-000300.pt")
            resumed = replace(cfg, resume_from=checkpoint)
            completed = subprocess.CompletedProcess(["sbatch"], 0, stdout="700;mock-cluster\n", stderr="")
            with patch.object(submitter, "configuration", return_value=resumed), \
                 patch.object(submitter.subprocess, "run", return_value=completed) as sbatch, \
                 redirect_stdout(io.StringIO()):
                records = submitter.submit("future", phase="pilot", task_id=10)
            self.assertEqual(len(records), 1)
            self.assertEqual(records[0]["task_ids"], [10])
            self.assertEqual(records[0]["resume_from"], checkpoint)
            self.assertIn("--array=10%2", sbatch.call_args.args[0])
            self.assertEqual(sbatch.call_args.kwargs["env"]["TG_RESUME_FROM"], checkpoint)
            self.assertEqual(grid.read_bytes(), original)
            self.assertTrue((Path(temporary) / "logs/submissions.tsv").read_text().rstrip().endswith(checkpoint))
            with patch.object(submitter, "configuration", return_value=replace(resumed, learning_rate=0.002)), \
                 patch.object(submitter.subprocess, "run") as sbatch:
                with self.assertRaisesRegex(ValueError, "Incompatible frozen artifact"):
                    submitter.submit("future", phase="pilot", task_id=10)
            sbatch.assert_not_called()

    def test_resume_rejects_multi_task_and_wrong_family_before_submission(self):
        from gavd6_sjepa.research_directions.temporal_gait.config import RunConfig
        cfg = RunConfig(mode="synthetic", device="cpu", run_root="/tmp/unused-resume-plan",
                        resume_from="/tmp/unused-resume-plan/training/task-0010/checkpoint-000300.pt")
        with patch.object(submitter, "configuration", return_value=cfg), \
             patch.object(submitter, "freeze_task_grid") as freeze, \
             patch.object(submitter.subprocess, "run") as sbatch:
            for command in ("pilot", "future", "evaluate"):
                with self.assertRaisesRegex(ValueError, "Checkpoint resume requires"):
                    submitter.submit(command)
            with self.assertRaisesRegex(ValueError, "No masked tasks"):
                submitter.submit("masked", task_id=10)
            with self.assertRaisesRegex(ValueError, "not in the pilot"):
                submitter.submit("future", task_id=11)
        freeze.assert_not_called()
        sbatch.assert_not_called()

    def test_all_shell_entrypoints_parse(self):
        for suffix in ("*.sh", "*.sbatch"):
            for path in (ROOT / "slurm/temporal-gait").glob(suffix):
                subprocess.run(["bash", "-n", str(path)], check=True, capture_output=True)


if __name__ == "__main__":
    unittest.main()
