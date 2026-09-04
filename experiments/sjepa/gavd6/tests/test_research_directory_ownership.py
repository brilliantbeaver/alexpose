"""Guard the active-research, shared-foundation, and archive boundaries."""

from __future__ import annotations

import ast
from pathlib import Path
import unittest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SOURCE_PACKAGE = PROJECT_ROOT / "src" / "gavd6_sjepa"
SCRIPTS_ROOT = PROJECT_ROOT / "scripts"


class ResearchDirectoryOwnershipTests(unittest.TestCase):
    def test_source_root_contains_only_package_and_command_router(self) -> None:
        names = {path.name for path in SOURCE_PACKAGE.glob("*.py")}
        self.assertEqual(names, {"__init__.py", "command_line_interface.py"})

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


if __name__ == "__main__":
    unittest.main()
