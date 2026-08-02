"""Regression tests for process-level warning suppression behavior."""

import os
import subprocess
import sys

import pytest


@pytest.mark.fast
def test_importing_ambient_preserves_stderr_file_descriptor() -> None:
    """Importing the package must not replace the host process's stderr."""
    script = """
import os

before = os.fstat(2)
import ambient
after = os.fstat(2)

assert (after.st_dev, after.st_ino, after.st_mode) == (
    before.st_dev,
    before.st_ino,
    before.st_mode,
)
"""

    result = subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True,
        check=False,
        text=True,
        timeout=30,
    )

    assert result.returncode == 0, result.stderr
