"""Regression checks for the swap-probe Slurm launcher defaults."""

from __future__ import annotations

from pathlib import Path
import unittest


class RunSwapProbeSbatchTests(unittest.TestCase):
    def test_default_artifact_paths_use_amass_run_root(self) -> None:
        script = (
            Path(__file__).resolve().parents[1] / "slurm" / "run-swap-probe.sbatch"
        ).read_text(encoding="utf-8")

        self.assertIn(
            'CHECKPOINT="${SWAP_PROBE_CHECKPOINT:-$AMASS_RUN_ROOT/outputs/'
            'repaired-jepa-seed7-v2/seed-7_standard_sjepa_best.pt}"',
            script,
        )
        self.assertIn(
            'OUTPUT_DIR="${SWAP_PROBE_OUTPUT_DIR:-$AMASS_RUN_ROOT/outputs/'
            'swap-probe-seed7}"',
            script,
        )
        self.assertNotIn('$GAVD6_ROOT/outputs/repaired-jepa-seed7-v2', script)


if __name__ == "__main__":
    unittest.main()
