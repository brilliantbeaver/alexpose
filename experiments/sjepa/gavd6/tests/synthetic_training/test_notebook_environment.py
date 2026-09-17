"""Exercise a real study notebook kernel without study assets or model weights."""
import importlib.util
import json
import os
from pathlib import Path
import socket
import sys
import tempfile
import unittest
from unittest.mock import patch

import nbformat


SCRIPT = Path(__file__).resolve().parents[2] / "scripts/research_directions/synthetic_training/execute_notebook.py"
SPEC = importlib.util.spec_from_file_location("synthetic_training_notebook_executor", SCRIPT)
executor = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(executor)


class NotebookEnvironmentTests(unittest.TestCase):
    def test_real_kernel_uses_selected_python_and_discards_parent_python_overrides(self):
        # A sandbox can prohibit the same loopback bind Jupyter needs. Detect
        # that host limitation before Jupyter starts its background threads.
        # With permission, the rest of this test always uses a real kernel.
        with socket.socket() as probe:
            try:
                probe.bind(("127.0.0.1", 0))
            except PermissionError as error:
                self.skipTest(f"Host blocks Jupyter loopback ports: {error}; run outside the sandbox")
        with tempfile.TemporaryDirectory(prefix="st-notebook-environment-") as temporary:
            root = Path(temporary)
            source, run = root / "source", root / "run"
            source.mkdir()
            # This synthetic fixture only inspects the kernel. It cannot read
            # experiment assets or be mistaken for an executed research stage.
            notebook = nbformat.v4.new_notebook(cells=[
                nbformat.v4.new_markdown_cell("Synthetic kernel-isolation test; no experimental evidence."),
                nbformat.v4.new_code_cell(
                    "import json, os, sys\n"
                    f"assert sys.executable == {sys.executable!r}\n"
                    "assert 'PYTHONHOME' not in os.environ\n"
                    "assert 'PYTHONPATH' not in os.environ\n"
                    "assert os.environ['PYTHONNOUSERSITE'] == '1'\n"
                    "print(json.dumps({'python': sys.executable, 'isolated': True}))"
                ),
            ])
            nbformat.write(notebook, source / executor.NOTEBOOKS["00"])
            poisoned = {
                "PYTHONHOME": str(root / "invalid-python-home"),
                "PYTHONPATH": str(root / "unrelated-project-packages"),
                "PYTHONNOUSERSITE": "0",
                "ST_NOTEBOOK_OUTPUT_DIR": str(root / "ignored-output"),
            }
            with patch.object(executor, "SOURCE", source), patch.dict(os.environ, poisoned):
                destination = executor.execute_notebook(
                    "00", run_root=run, output_dir=root / "executed", timeout=30,
                )
                # Sanitizing the child must not mutate the caller's environment.
                self.assertEqual(os.environ["PYTHONHOME"], poisoned["PYTHONHOME"])
                self.assertEqual(os.environ["PYTHONPATH"], poisoned["PYTHONPATH"])
            executed = nbformat.read(destination, as_version=4)
            record = executed.metadata["synthetic_training_execution"]
            self.assertEqual(record["status"], "completed")
            self.assertEqual(record["python"], sys.executable)
            self.assertEqual(executed.cells[1].execution_count, 1)
            stdout = "".join(output.text for output in executed.cells[1].outputs
                             if output.output_type == "stream" and output.name == "stdout")
            self.assertEqual(json.loads(stdout), {"python": sys.executable, "isolated": True})
            self.assertEqual(nbformat.read(source / executor.NOTEBOOKS["00"], as_version=4).cells[1].outputs, [])


if __name__ == "__main__":
    unittest.main()
