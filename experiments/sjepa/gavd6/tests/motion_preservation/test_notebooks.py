"""motion preservation / test notebooks."""


import ast
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import MagicMock, patch

import nbformat

from tests.support import REPO_ROOT

ROOT = REPO_ROOT

SCRIPTS = ROOT / "scripts/research_directions/motion_preservation"

spec = importlib.util.spec_from_file_location("motion_preservation_notebook_runner", SCRIPTS / "execute_notebook.py")

runner = importlib.util.module_from_spec(spec)

spec.loader.exec_module(runner)

class MotionPreservationNotebookTests(unittest.TestCase):
    def test_six_notebooks_have_readable_steps_and_compile(self):
        for number, name in runner.NOTEBOOKS.items():
            notebook = nbformat.read(runner.SOURCE / name, as_version=4)
            code_cells = [c for c in notebook.cells if c.cell_type == "code"]
            self.assertGreaterEqual(len(code_cells), 4, name)
            self.assertGreater(len([c for c in notebook.cells if c.cell_type == "markdown"]), 5, name)
            for cell in code_cells:
                self.assertEqual(cell.outputs, [])
                self.assertIsNone(cell.execution_count)
                compile(cell.source, name, "exec")
                for statement in ast.walk(ast.parse(cell.source)):
                    if isinstance(statement, ast.ImportFrom) and statement.module == \
                            "gavd6_sjepa.research_directions.motion_preservation":
                        for alias in statement.names:
                            importlib.import_module(f"{statement.module}.{alias.name}")
            if number == "04":
                source = "\n".join(c.source for c in code_cells)
                self.assertNotIn("workflow.train_gate(", source)
                self.assertNotIn("workflow.calibrate(", source)

    def test_pilot_uses_afterok_and_leaves_final_unopened(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            fake_bin = root / "bin"
            fake_bin.mkdir()
            log = root / "jobs.jsonl"
            fake_sbatch = fake_bin / "sbatch"
            fake_sbatch.write_text(
                "#!/usr/bin/env python3\n"
                "import json, os, pathlib, sys\n"
                "p=pathlib.Path(os.environ['FAKE_JOB_LOG'])\n"
                "number=100 + (len(p.read_text().splitlines()) if p.exists() else 0)\n"
                "with p.open('a') as stream: stream.write(json.dumps({'args':sys.argv[1:],"
                "'split':os.environ['MP_EVALUATION_SPLIT'],"
                "'mode':os.environ.get('MP_MODE'),'device':os.environ.get('MP_DEVICE')})+'\\n')\n"
                "print(str(number)+';fake-cluster')\n"
            )
            fake_sbatch.chmod(0o755)
            environment = {**os.environ, "GAVD6_ROOT": str(ROOT), "MP_RUN_ROOT": str(root / "run"),
                           "PATH": str(fake_bin) + os.pathsep + os.environ["PATH"], "FAKE_JOB_LOG": str(log)}
            environment.pop("MP_NOTEBOOK_OUTPUT_DIR", None)
            environment.pop("MP_DEPENDENCY", None)
            environment.pop("MP_MODE", None)
            environment.pop("MP_DEVICE", None)
            environment.pop("MP_CONFIG", None)
            subprocess.run(["bash", str(ROOT / "slurm/motion-preservation/submit.sh"), "pilot"],
                           env=environment, capture_output=True, text=True, check=True)
            jobs = [json.loads(line) for line in log.read_text().splitlines()]
            self.assertEqual(len(jobs), 5)
            for index, job in enumerate(jobs):
                self.assertEqual(job["split"], "development")
                self.assertIsNone(job["mode"])
                self.assertIsNone(job["device"])
                self.assertTrue(job["args"][-1].endswith(("inventory.sbatch", "controlled-pairs.sbatch",
                                                        "cache-evidence.sbatch", "train-calibrate.sbatch",
                                                        "evaluate.sbatch")[index]))
                dependencies = [a for a in job["args"] if a.startswith("--dependency=")]
                self.assertEqual(dependencies, [] if index == 0 else [f"--dependency=afterok:{99 + index}"])
            environment.update(MP_DEPENDENCY="afterok:71:72", MP_ACCOUNT="test-account", MP_PARTITION="test-partition")
            subprocess.run(["bash", str(ROOT / "slurm/motion-preservation/submit.sh"), "final"],
                           env=environment, capture_output=True, text=True, check=True)
            final = json.loads(log.read_text().splitlines()[-1])
            self.assertEqual(final["split"], "final")
            self.assertTrue(final["args"][-1].endswith("evaluate.sbatch"))
            for option in ("--dependency=afterok:71:72", "--account=test-account", "--partition=test-partition"):
                self.assertIn(option, final["args"])

    def test_runner_only_overrides_mode_and_device_when_requested(self):
        captured = []

        def make_client(notebook, **kwargs):
            return MagicMock(execute=lambda **options: captured.append(options["env"]))

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            config = root / "settings.json"
            config.write_text(json.dumps({"mode": "demo", "device": "cpu"}))
            clean_environment = {key: value for key, value in os.environ.items()
                                 if key not in {"MP_MODE", "MP_DEVICE", "MP_CONFIG"}}
            with patch.dict(os.environ, clean_environment, clear=True), \
                 patch.object(runner, "KernelManager", return_value=MagicMock(has_kernel=False)), \
                 patch.object(runner, "NotebookClient", side_effect=make_client):
                runner.execute_notebook("00", run_root=root / "run", config=config,
                                        output_dir=root / "configured")
                runner.execute_notebook("00", run_root=root / "run", config=config,
                                        output_dir=root / "overridden", mode="real", device="cuda")
            self.assertNotIn("MP_MODE", captured[0])
            self.assertNotIn("MP_DEVICE", captured[0])
            self.assertEqual(captured[0]["MP_CONFIG"], str(config.resolve()))
            self.assertEqual(captured[1]["MP_MODE"], "real")
            self.assertEqual(captured[1]["MP_DEVICE"], "cuda")

    def test_notebook_failure_keeps_partial_outputs(self):
        def make_client(notebook, **kwargs):
            def execute(**options):
                cell = next(c for c in notebook.cells if c.cell_type == "code")
                cell.execution_count = 1
                cell.outputs = [nbformat.v4.new_output("stream", name="stdout", text="partial diagnostic\n")]
                raise RuntimeError("synthetic model loading failure")
            return MagicMock(execute=execute)

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = runner.SOURCE / runner.NOTEBOOKS["00"]
            original = source.read_bytes()
            with patch.object(runner, "KernelManager", return_value=MagicMock(has_kernel=False)), \
                 patch.object(runner, "NotebookClient", side_effect=make_client):
                with self.assertRaisesRegex(RuntimeError, "synthetic model loading failure"):
                    runner.execute_notebook("00", run_root=root / "run", output_dir=root / "executed", device="cpu")
            saved = nbformat.read(root / "executed" / source.name, as_version=4)
            self.assertEqual(saved.metadata.motion_preservation_execution.status, "failed")
            self.assertTrue(any("partial diagnostic" in output.get("text", "")
                                for cell in saved.cells for output in cell.get("outputs", [])))
            self.assertEqual(original, source.read_bytes())


if __name__ == "__main__":
    unittest.main()
