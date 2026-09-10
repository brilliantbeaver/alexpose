"""Notebook adapters must preserve CLI routing, failure visibility and protocol scope."""

import contextlib
import hashlib
import importlib.util
import io
import json
import os
import re
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import MagicMock, patch

import nbformat

from gavd6_sjepa.research_directions.future_innovation import fi_notebook_workflow as workflow


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts/research_directions/future_innovation"
with patch.object(sys, "path", [str(SCRIPTS), *sys.path]):
    spec = importlib.util.spec_from_file_location("fi_notebook_executor", SCRIPTS / "execute_future_innovation_notebook.py")
    executor = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(executor)
    import build_future_innovation_notebooks as builder


class FutureInnovationNotebookExecutionTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name).resolve()

    def test_cli_uses_current_python_and_keeps_failure_output(self):
        with patch.dict(os.environ, {"FI_NOTEBOOK_LOG_DIR": str(self.root / "logs"),
                                    "FI_TORCH_THREADS": "3", "OMP_NUM_THREADS": "9"}), contextlib.redirect_stdout(io.StringIO()):
            workflow.run_stage("run-gate", self.root, "--help")
            with self.assertRaises(subprocess.CalledProcessError):
                workflow.run_stage("run-gate", self.root, "--outer-fold", "9")
        receipts = [json.loads(p.read_text().strip().splitlines()[-1]) for p in self.root.glob("logs/*.log")]
        self.assertEqual({r["status"] for r in receipts}, {"passed", "failed"})
        for receipt in receipts:
            self.assertEqual(receipt["argv"][:6], [sys.executable, "-u", "-m",
                             "gavd6_sjepa.command_line_interface", "future-innovation", "run-gate"])
            self.assertEqual(receipt["argv"][-2:], ["--run-root", str(self.root.resolve())])
            self.assertEqual(receipt["cwd"], str(ROOT))
            self.assertEqual(receipt["execution_controls"]["OMP_NUM_THREADS"], "3")
            self.assertEqual(receipt["execution_controls"]["PYTHONHASHSEED"], "260905")
        self.assertTrue(any("invalid choice" in p.read_text() for p in self.root.glob("logs/*.log")))
        self.assertEqual(len(list(self.root.glob("logs/*"))), 2)

    def test_initialization_resumes_without_input_environment(self):
        (self.root / "config").mkdir()
        (self.root / "config/run-contract.json").write_text("{}")
        with patch.dict(os.environ, {}, clear=True), patch.object(workflow, "check_run") as check, patch.object(workflow, "run_stage") as run:
            workflow.initialize_from_environment(self.root)
        check.assert_called_once_with(self.root.resolve())
        run.assert_not_called()

    def test_new_initialization_requires_and_uses_exact_official_partitions(self):
        data = self.root / "data with spaces"
        annotations = data / "annotations/GAVD/data"
        annotations.mkdir(parents=True)
        env = {"GAVD_FULL_ROOT": str(data), "VJEPA2_ROOT": str(self.root / "teacher"),
               "FI_TEACHER_CHECKPOINT": str(self.root / "checkpoint"), "FI_POSE_MODEL": str(self.root / "pose")}
        with patch.dict(os.environ, {}, clear=True), self.assertRaisesRegex(ValueError, "GAVD_FULL_ROOT"):
            workflow.initialize_from_environment(self.root)
        with patch.dict(os.environ, env, clear=True), patch.object(workflow, "run_stage") as run:
            with self.assertRaisesRegex(FileNotFoundError, "Annotations_1"):
                workflow.initialize_from_environment(self.root)
            run.assert_not_called()
            for part in ("1", "2", "3", "4", "5", "1.1", "1.2"):
                (annotations / f"GAVD_Clinical_Annotations_{part}.csv").touch()
            workflow.initialize_from_environment(self.root)
        args = run.call_args.args
        selected = args[args.index("--annotations") + 1:args.index("--youtube-dir")]
        self.assertEqual([p.name for p in selected], [f"GAVD_Clinical_Annotations_{p}.csv" for p in range(1, 6)])
        self.assertEqual(args[:2], ("init-run", self.root.resolve()))

    def execute_stage_cell(self, number, *, fold=None, stage=None):
        stage = stage or MagicMock(return_value=None)
        env = {} if fold is None else {"FI_NOTEBOOK_FOLD": fold}
        namespace = {"MODE": "execute", "RUN_ROOT": self.root, "run_stage": stage, "attempt_stage": stage,
                     "initialize_from_environment": stage, "build_notebook_report": stage,
                     "display": lambda value: None, "os": os}
        (self.root / "config").mkdir(exist_ok=True)
        (self.root / "config/model-contract.json").write_text("{}")
        with patch.dict(os.environ, env, clear=True):
            exec(builder.execution_cells(number)[1].source, namespace)
        return stage

    def test_generated_execute_cells_cover_every_stage_and_default_full_grid(self):
        self.assertEqual(self.execute_stage_cell("00").call_args.args, (self.root,))
        for number, commands in (("01", ["build-cohort", "extract-poses"]),
                                 ("02", ["cache-teacher", "audit-teacher"])):
            self.assertEqual([c.args[0] for c in self.execute_stage_cell(number).call_args_list], commands)
        self.assertEqual(self.execute_stage_cell("03").call_args.args,
                         ("run-gate", self.root, "--device", "cpu"))
        self.assertEqual(self.execute_stage_cell("03", fold="4").call_args.args[-2:], ("--outer-fold", "4"))
        with self.assertRaisesRegex(ValueError, "0–4"):
            self.execute_stage_cell("03", fold="5")
        self.assertEqual(self.execute_stage_cell("04").call_args.args, (self.root,))

    def test_failed_preparation_or_cache_does_not_start_dependent_stage(self):
        for number in ("01", "02"):
            stage = MagicMock(return_value="stage failed")
            self.execute_stage_cell(number, stage=stage)
            self.assertEqual(stage.call_count, 1)
        with patch.object(workflow, "run_stage", side_effect=subprocess.CalledProcessError(2, ["fi"])):
            error = workflow.attempt_stage("audit-teacher", self.root)
        with self.assertRaisesRegex(RuntimeError, "audit-teacher failed"):
            workflow.require_stage_success(error)

    def test_report_fallback_does_not_hide_scoring_failure(self):
        with patch.object(workflow, "run_stage", side_effect=[subprocess.CalledProcessError(2, ["fi"]), None]) as run:
            self.assertFalse(workflow.build_notebook_report(self.root))
        self.assertEqual([c.args[0] for c in run.call_args_list], ["score-gate", "build-report"])
        evidence = {"decision": {"measurement_complete": True, "decision": "STOP"},
                    "seal_verified": True, "state": "COMPLETE / STOP"}
        with patch.object(workflow, "inspect_report", return_value=evidence):
            self.assertEqual(workflow.require_complete_report(self.root, scoring_succeeded=True)["decision"], "STOP")
            with self.assertRaises(RuntimeError):
                workflow.require_complete_report(self.root, scoring_succeeded=False)
            for changed in ({"decision": None}, {"seal_verified": False}, {"state": "UNVERIFIED"},
                            {"decision": {"measurement_complete": False}}):
                with patch.object(workflow, "inspect_report", return_value={**evidence, **changed}), self.assertRaises(RuntimeError):
                    workflow.require_complete_report(self.root, scoring_succeeded=True)

    def test_executor_validates_scope_before_creating_outputs(self):
        cases = ({"number": "00", "mode": "execute"},
                 {"number": "01", "outer_fold": 0},
                 {"number": "03", "mode": "execute", "run_root": self.root, "outer_fold": 5},
                 {"number": "00", "timeout": 0})
        for kwargs in cases:
            with self.assertRaises(ValueError):
                executor.execute_notebook(**kwargs, output_parent=self.root)
        self.assertEqual(list(self.root.iterdir()), [])

    def test_executor_saves_failed_cell_without_changing_source_and_clears_inherited_fold(self):
        source = builder.DESTINATION / builder.NAMES["03"]
        before = hashlib.sha256(source.read_bytes()).hexdigest()
        observed = []

        def fake_client(notebook, **kwargs):
            def execute(**options):
                observed.append(options["env"])
                cell = notebook.cells[2]
                kwargs["on_cell_start"](cell=cell, cell_index=2)
                cell.execution_count = 1
                cell.outputs = [nbformat.v4.new_output("error", ename="RuntimeError", evalue="fixture", traceback=["fixture"])]
                kwargs["on_cell_executed"](cell=cell, cell_index=2)
                # A completed/error cell has already been checkpointed before exit.
                saved = nbformat.read(next(self.root.glob("notebooks/03*.ipynb")), as_version=4)
                self.assertEqual(saved.cells[2].outputs[0].evalue, "fixture")
                raise RuntimeError("fixture")
            return MagicMock(execute=execute)

        with patch.object(executor, "KernelManager", return_value=MagicMock(has_kernel=False)), patch.object(
            executor, "NotebookClient", side_effect=fake_client
        ), patch.dict(os.environ, {"FI_TUTORIAL_MODE": "teach", "FI_NOTEBOOK_FOLD": "1"}):
            with self.assertRaisesRegex(RuntimeError, "fixture"):
                executor.execute_notebook("03", mode="execute", run_root=self.root,
                                          output_parent=self.root / "notebooks")
        self.assertEqual(observed[0]["FI_TUTORIAL_MODE"], "execute")
        self.assertNotIn("FI_NOTEBOOK_FOLD", observed[0])
        receipts = list(self.root.glob("notebooks/*.ipynb"))
        self.assertEqual(len(receipts), 1)
        record = nbformat.read(receipts[0], as_version=4).metadata.fi_execution
        self.assertEqual(record["status"], "failed")
        self.assertEqual(record["last_finished_cell"], 2)
        self.assertEqual(before, hashlib.sha256(source.read_bytes()).hexdigest())
        self.assertTrue(all(p.suffix == ".ipynb" for p in (self.root / "notebooks").iterdir()))

    def test_shared_folder_keeps_fold_outputs_separate_and_rejects_duplicate_writers(self):
        folder = self.root / "notebooks"
        def client(notebook, **kwargs):
            def execute(**options):
                fold = int(options["env"]["FI_NOTEBOOK_FOLD"])
                with self.assertRaisesRegex(ValueError, "Another job"):
                    executor.execute_notebook("03", mode="execute", run_root=self.root / "alternate-run",
                                              outer_fold=fold, output_parent=folder)
            return MagicMock(execute=execute)
        with patch.object(executor, "KernelManager", return_value=MagicMock(has_kernel=False)), patch.object(
            executor, "NotebookClient", side_effect=client
        ):
            paths = [executor.execute_notebook("03", mode="execute", run_root=self.root,
                                              outer_fold=fold, output_parent=folder) for fold in (0, 1)]
        self.assertEqual(len(list(folder.iterdir())), 2)
        self.assertEqual({nbformat.read(p, as_version=4).metadata.fi_execution.outer_fold for p in paths}, {0, 1})
        self.assertTrue(all(p.parent == folder and p.suffix == ".ipynb" for p in paths))

    def test_executed_document_links_follow_new_folder_and_array_filenames(self):
        notebook = builder.render("02")
        original_code = [c.source for c in notebook.cells if c.cell_type == "code"]
        with patch.dict(os.environ, {"FI_NOTEBOOK_ARRAY": "1"}):
            executor.relocate_links(notebook, self.root)
        for cell in notebook.cells:
            if cell.cell_type == "markdown":
                for target in re.findall(r"\]\(([^)]+)\)", cell.source):
                    if target.endswith(".ipynb"):
                        self.assertEqual(target, "03_matched_predictors_and_controls_fold-0.ipynb")
                    elif "://" not in target and not target.startswith("#"):
                        self.assertTrue((self.root / target.split("#")[0]).exists())
        self.assertEqual(original_code, [c.source for c in notebook.cells if c.cell_type == "code"])


if __name__ == "__main__":
    unittest.main()
