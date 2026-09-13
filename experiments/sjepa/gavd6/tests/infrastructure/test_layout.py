"""infrastructure / test layout."""


from __future__ import annotations

import ast
from collections import defaultdict
import json
from pathlib import Path
import re
import unittest

from tests.support import REPO_ROOT

PROJECT_ROOT = REPO_ROOT

SOURCE_PACKAGE = PROJECT_ROOT / "src" / "gavd6_sjepa"

SCRIPTS_ROOT = PROJECT_ROOT / "scripts"

class ResearchDirectoryOwnershipTests(unittest.TestCase):
    def test_source_root_contains_only_package_and_command_router(self) -> None:
        names = {path.name for path in SOURCE_PACKAGE.glob("*.py")}
        self.assertEqual(names, {"__init__.py", "cli.py"})

    def test_scripts_root_contains_no_python_entrypoints(self) -> None:
        self.assertEqual(list(SCRIPTS_ROOT.glob("*.py")), [])

    def test_active_source_never_imports_archive_modules(self) -> None:
        active_roots = (
            SOURCE_PACKAGE / "data_foundations",
            SOURCE_PACKAGE / "shared_infrastructure",
            SOURCE_PACKAGE / "research_directions",
            SOURCE_PACKAGE / "workspace_validation",
        )
        violations: list[str] = []
        for root in active_roots:
            for path in root.rglob("*.py"):
                tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
                for node in ast.walk(tree):
                    module = ""
                    if isinstance(node, ast.ImportFrom):
                        module = node.module or ""
                    elif isinstance(node, ast.Import):
                        module = " ".join(alias.name for alias in node.names)
                    if "gavd6_sjepa.archive" in module:
                        violations.append(str(path.relative_to(PROJECT_ROOT)))
        self.assertEqual(violations, [])


ROOT = REPO_ROOT

class StudyLayoutTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.registry = json.loads((ROOT / "docs/repository/studies.json").read_text())

    def test_study_entry_points_resolve(self):
        studies = self.registry["studies"]
        self.assertEqual(len({study["id"] for study in studies}), len(studies))
        for study in studies:
            paths = [study["overview"], study["documentation"]]
            for role in ("code", "notebooks", "scripts", "slurm", "tests"):
                paths.extend(study[role])
            for path in paths:
                with self.subTest(study=study["id"], path=path):
                    self.assertTrue(any(ROOT.glob(path)), f"Missing study entry: {path}")

    def test_canonical_notebooks_have_one_owner(self):
        owners = defaultdict(list)
        for study in self.registry["studies"]:
            for directory in study["notebooks"]:
                for path in (ROOT / directory).rglob("*.ipynb"):
                    if not path.is_symlink() and ".ipynb_checkpoints" not in path.parts:
                        owners[path].append(study["id"])
        for directory in self.registry["shared"]["data"]:
            if directory.startswith("notebooks/"):
                for path in (ROOT / directory).rglob("*.ipynb"):
                    if not path.is_symlink() and ".ipynb_checkpoints" not in path.parts:
                        owners[path].append("shared-data")
        canonical = {
            path for path in (ROOT / "notebooks").rglob("*.ipynb")
            if not path.is_symlink() and ".ipynb_checkpoints" not in path.parts
        }
        self.assertEqual(set(owners), canonical)
        self.assertEqual({str(p): v for p, v in owners.items() if len(v) != 1}, {})


# Validate package-local Python names and unittest discovery boundaries.

class ScopedPythonNamingTests(unittest.TestCase):
    def test_python_names_and_test_discovery_packages(self):
        for directory in (REPO_ROOT / "src", REPO_ROOT / "scripts", REPO_ROOT / "tests"):
            for path in directory.rglob("*.py"):
                with self.subTest(path=str(path.relative_to(REPO_ROOT))):
                    self.assertRegex(path.stem, r"^[a-z_][a-z0-9_]*$")
        test_root = REPO_ROOT / "tests"
        for path in test_root.rglob("test_*.py"):
            self.assertFalse(path.is_symlink(), f"Test aliases cause duplicate discovery: {path}")
            for directory in (path.parent, *path.parent.parents):
                if not directory.is_relative_to(test_root):
                    break
                self.assertTrue((directory / "__init__.py").is_file(),
                                f"Missing unittest discovery package: {directory}")


if __name__ == "__main__":
    unittest.main()
