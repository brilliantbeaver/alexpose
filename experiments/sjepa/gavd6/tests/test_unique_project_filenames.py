"""Prevent ambiguous filenames from returning to scripts or source code."""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path
import unittest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CHECKED_DIRECTORIES = (PROJECT_ROOT / "scripts", PROJECT_ROOT / "src")
CHECKED_SUFFIXES = {".py", ".md", ".js"}


class UniqueProjectFilenameTests(unittest.TestCase):
    def test_scripts_and_source_have_globally_unique_basenames(self) -> None:
        paths_by_name: dict[str, list[Path]] = defaultdict(list)
        for directory in CHECKED_DIRECTORIES:
            for path in directory.rglob("*"):
                if path.is_file() and path.suffix in CHECKED_SUFFIXES:
                    paths_by_name[path.name].append(path.relative_to(PROJECT_ROOT))
        duplicates = {
            name: paths
            for name, paths in paths_by_name.items()
            if len(paths) > 1
        }
        self.assertEqual(duplicates, {})


if __name__ == "__main__":
    unittest.main()
