"""motion preservation / test notebooks."""


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
                "'split':os.environ['MP_EVALUATION_SPLIT']})+'\\n')\n"
                "print(str(number)+';fake-cluster')\n"
            )
            fake_sbatch.chmod(0o755)
            environment = {**os.environ, "GAVD6_ROOT": str(ROOT), "MP_RUN_ROOT": str(root / "run"),
                           "PATH": str(fake_bin) + os.pathsep + os.environ["PATH"], "FAKE_JOB_LOG": str(log)}
            environment.pop("MP_NOTEBOOK_OUTPUT_DIR", None)
            environment.pop("MP_DEPENDENCY", None)
            subprocess.run(["bash", str(ROOT / "slurm/motion-preservation/submit.sh"), "pilot"],
                           env=environment, capture_output=True, text=True, check=True)
            jobs = [json.loads(line) for line in log.read_text().splitlines()]
            self.assertEqual(len(jobs), 5)
            for index, job in enumerate(jobs):
                self.assertEqual(job["split"], "development")
                self.assertTrue(job["args"][-1].endswith(("inventory.sbatch", "controlled-pairs.sbatch",
                                                        "cache-evidence.sbatch", "train-calibrate.sbatch",
                                                        "evaluate.sbatch")[index]))
                dependencies = [a for a in job["args"] if a.startswith("--dependency=")]
                self.assertEqual(dependencies, [] if index == 0 else [f"--dependency=afterok:{99 + index}"])
            subprocess.run(["bash", str(ROOT / "slurm/motion-preservation/submit.sh"), "final"],
                           env=environment, capture_output=True, text=True, check=True)
            final = json.loads(log.read_text().splitlines()[-1])
            self.assertEqual(final["split"], "final")
            self.assertTrue(final["args"][-1].endswith("evaluate.sbatch"))

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
