"""Tests for the discoverable, lazy ``gavd6`` command router."""

from __future__ import annotations

import subprocess
import sys
import unittest
from unittest.mock import patch

from gavd6_sjepa import command_line_interface


class CommandLineInterfaceTests(unittest.TestCase):
    def test_global_help_lists_each_workflow_group(self) -> None:
        help_text = command_line_interface._help()
        for group in ("amass", "gavd", "laterality", "swap-probe", "notebooks"):
            self.assertIn(group, help_text)

    def test_command_arguments_are_forwarded_without_importing_eagerly(self) -> None:
        observed: list[str] = []

        def handler() -> int:
            observed.extend(sys.argv)
            return 7

        with patch.object(command_line_interface, "_resolve_handler", return_value=handler):
            status = command_line_interface.main(
                ["laterality", "benchmark", "--synthetic-smoke"]
            )
        self.assertEqual(status, 7)
        self.assertEqual(
            observed,
            ["gavd6 laterality benchmark", "--synthetic-smoke"],
        )

    def test_unknown_command_fails_without_resolving_a_handler(self) -> None:
        with patch.object(command_line_interface, "_resolve_handler") as resolver:
            status = command_line_interface.main(["amass", "unknown"])
        self.assertEqual(status, 2)
        resolver.assert_not_called()

    def test_help_for_environment_configured_command_never_executes_it(self) -> None:
        with patch.object(command_line_interface, "_resolve_handler") as resolver:
            status = command_line_interface.main(["amass", "inventory", "--help"])
        self.assertEqual(status, 0)
        resolver.assert_not_called()

    def test_help_does_not_import_heavy_scientific_dependencies(self) -> None:
        code = (
            "import sys; "
            "from gavd6_sjepa.command_line_interface import main; "
            "status=main(['--help']); "
            "assert status == 0; "
            "assert 'torch' not in sys.modules; "
            "assert 'cv2' not in sys.modules"
        )
        completed = subprocess.run(
            [sys.executable, "-c", code],
            check=False,
            capture_output=True,
            text=True,
        )
        self.assertEqual(completed.returncode, 0, completed.stderr)


if __name__ == "__main__":
    unittest.main()
