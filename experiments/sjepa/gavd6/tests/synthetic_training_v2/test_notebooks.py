"""Contracts for the canonical study notebooks; no fitting during these tests."""
from __future__ import annotations

import ast
import importlib.util
import os
from pathlib import Path
import re
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import nbformat

ROOT = Path(__file__).resolve().parents[2]
BUILDER = ROOT / "scripts/research_directions/synthetic_training_v2/build_notebooks.py"
NOTEBOOKS = ROOT / "notebooks/synthetic_training_v2"
SPEC = importlib.util.spec_from_file_location("stv2_notebook_builder", BUILDER)
builder = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(builder)
EXPECTED = {
    "00_artifact_audit.ipynb": ["audit"],
    "01_paired_data.ipynb": ["data"],
    "02_image_adaptation.ipynb": ["adaptation"],
    "03_information_ladder.ipynb": ["information"],
    "04_coordinate_vs_jepa.ipynb": ["direct", "jepa"],
    "05_evaluation.ipynb": ["evaluate"],
    "06_optional_gates.ipynb": ["optional"],
    "07_development_snapshot.ipynb": ["freeze"],
    "08_evidence_report.ipynb": ["report"],
}


class NotebookContracts(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.notebooks = {p.name: nbformat.read(p, as_version=4) for p in NOTEBOOKS.glob("*.ipynb")}

    def test_schema_output_free_and_executable_syntax(self):
        self.assertEqual(set(self.notebooks), set(EXPECTED))
        for name, notebook in self.notebooks.items():
            with self.subTest(notebook=name):
                nbformat.validate(notebook)
                self.assertEqual(notebook.metadata.kernelspec.name, "python3")
                self.assertEqual(len({cell.id for cell in notebook.cells}), len(notebook.cells))
                narrative = "\n".join(cell.source for cell in notebook.cells if cell.cell_type == "markdown")
                for section in ("Question", "Inputs", "Computation", "Outputs and checks", "Interpretation", "Next gate"):
                    self.assertIn(f"## {section}", narrative)
                self.assertIn("fixture", narrative)
                for cell in notebook.cells:
                    if cell.cell_type == "code":
                        self.assertIsNone(cell.execution_count)
                        self.assertEqual(cell.outputs, [])
                        compile(cell.source, name, "exec")

    def test_stage_calls_match_metadata_and_prerequisites(self):
        # Keep the dependency contract independent of the generated cell text.
        dependencies = {"data": {"audit"}, "adaptation": {"data"}, "information": {"data"},
                        "direct": {"information"}, "jepa": {"direct"}, "evaluate": {"jepa"},
                        "optional": {"audit"}, "freeze": {"evaluate"},
                        "report": {"evaluate", "adaptation", "optional", "freeze"}}
        completed = set()
        for name, expected in EXPECTED.items():
            notebook = self.notebooks[name]
            self.assertEqual(notebook.metadata.synthetic_training_v2.stages, expected)
            calls = []
            for cell in notebook.cells:
                if cell.cell_type != "code":
                    continue
                for node in ast.walk(ast.parse(cell.source)):
                    if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == "run_stage":
                        self.assertIsInstance(node.args[0], ast.Name)
                        self.assertEqual(node.args[0].id, "cfg")
                        self.assertIsInstance(node.args[1], ast.Constant)
                        self.assertEqual([keyword.arg for keyword in node.keywords], ["repo_root"])
                        calls.append(node.args[1].value)
            self.assertEqual(calls, expected)
            for stage in calls:
                self.assertTrue(dependencies.get(stage, set()) <= completed, stage)
                completed.add(stage)

    def test_builder_is_deterministic_and_matches_checked_in_files(self):
        with tempfile.TemporaryDirectory() as temporary:
            target = Path(temporary)
            command = [sys.executable, str(BUILDER), "--output-dir", str(target)]
            subprocess.run(command, check=True, capture_output=True, text=True)
            first = {p.name: p.read_bytes() for p in target.glob("*.ipynb")}
            subprocess.run(command, check=True, capture_output=True, text=True)
            second = {p.name: p.read_bytes() for p in target.glob("*.ipynb")}
            canonical = {p.name: p.read_bytes() for p in NOTEBOOKS.glob("*.ipynb")}
            self.assertEqual(first, second)
            self.assertEqual(first, canonical)

    def test_all_local_markdown_links_resolve(self):
        documents = [(NOTEBOOKS / name, "\n".join(cell.source for cell in nb.cells
                                                  if cell.cell_type == "markdown"))
                     for name, nb in self.notebooks.items()]
        documents.append((NOTEBOOKS / "README.md", (NOTEBOOKS / "README.md").read_text()))
        for path, source in documents:
            for target in re.findall(r"\[[^\]]*\]\(([^)]+)\)", source):
                if "://" in target or target.startswith("#"):
                    continue
                relative = target.split("#", 1)[0]
                self.assertTrue((path.parent / relative).exists(), f"{path.name}: {target}")

    def test_setup_requires_explicit_existing_configuration_before_study_import(self):
        for name, notebook in self.notebooks.items():
            setup = next(cell.source for cell in notebook.cells if cell.cell_type == "code")
            with self.subTest(notebook=name), patch.dict(os.environ, {}, clear=True):
                with self.assertRaisesRegex(RuntimeError, "Set STV2_CONFIG"):
                    exec(compile(setup, name, "exec"), {})
            with self.subTest(notebook=name, missing_file=True), tempfile.TemporaryDirectory() as temporary:
                with patch.dict(os.environ, {"STV2_CONFIG": str(Path(temporary) / "absent.json")}):
                    with self.assertRaisesRegex(FileNotFoundError, "configuration does not exist"):
                        exec(compile(setup, name, "exec"), {})

    def test_confirmation_boundary_and_no_embedded_training(self):
        for name, notebook in self.notebooks.items():
            self.assertFalse(notebook.metadata.synthetic_training_v2.opens_confirmation)
            sources = "\n".join(cell.source for cell in notebook.cells if cell.cell_type == "code")
            tree = ast.parse(sources)
            # Training loops, shell commands and subprocess submissions belong outside notebooks.
            self.assertFalse(any(isinstance(node, (ast.For, ast.While, ast.FunctionDef, ast.AsyncFunctionDef))
                                 for node in ast.walk(tree)), name)
            self.assertNotIn("subprocess", sources)
            self.assertNotIn("sbatch", sources)
        freeze = self.notebooks["07_development_snapshot.ipynb"]
        text = "\n".join(cell.source for cell in freeze.cells)
        self.assertIn("development snapshot", text)
        self.assertIn("confirmation_opened", text)
        self.assertIn("unavailable", text)


if __name__ == "__main__":
    unittest.main()
