"""Regression checks for the swap-probe Slurm launcher defaults."""

from __future__ import annotations

from pathlib import Path
import unittest


class RunSwapProbeSbatchTests(unittest.TestCase):
    def test_artifact_paths_are_explicit_and_sha_is_not_pinned(self) -> None:
        script = (
            Path(__file__).resolve().parents[1] / "slurm" / "run-swap-probe.sbatch"
        ).read_text(encoding="utf-8")

        self.assertIn('CHECKPOINT="$SWAP_PROBE_CHECKPOINT"', script)
        self.assertIn('OUTPUT_DIR="$SWAP_PROBE_OUTPUT_DIR"', script)
        self.assertIn(
            "python -m gavd6_sjepa.command_line_interface swap-probe run",
            script,
        )
        self.assertNotIn('CHECKPOINT_SHA256=', script)
        self.assertNotIn('--expected-checkpoint-sha256', script)


if __name__ == "__main__":
    unittest.main()
