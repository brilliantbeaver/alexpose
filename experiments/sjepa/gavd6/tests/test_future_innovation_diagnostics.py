"""Verify mechanism interventions and immutable, relocated artifact inspection."""

import importlib.util
from pathlib import Path
import shutil
import sys
import tempfile
import unittest

import numpy as np
import torch

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts/research_directions/future_innovation"
spec = importlib.util.spec_from_file_location(
    "diagnose_future_innovation_fits", SCRIPTS / "diagnose_future_innovation_fits.py"
)
diagnostic = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = diagnostic
spec.loader.exec_module(diagnostic)
probe_spec = importlib.util.spec_from_file_location(
    "probe_future_innovation_residual_failure", SCRIPTS / "probe_future_innovation_residual_failure.py"
)
probe = importlib.util.module_from_spec(probe_spec)
probe_spec.loader.exec_module(probe)

from gavd6_sjepa.research_directions.future_innovation.fi_nested_training import fit_outer_fold
from gavd6_sjepa.research_directions.future_innovation.fi_smoke import synthetic_cache


def snapshot(root):
    return {str(path.relative_to(root)): (path.stat().st_mtime_ns, diagnostic.sha256_file(path))
            for path in root.rglob("*") if path.is_file()}


class FutureInnovationDiagnosticTests(unittest.TestCase):
    def test_null_removal_preserves_fit_but_changes_unseen_directions(self):
        rng = np.random.default_rng(19)
        x = rng.normal(size=(8, 80))
        x -= x.mean(axis=0)
        basis, _, _ = diagnostic.row_basis(x)
        self.assertEqual(len(basis), 7)
        weight = rng.normal(size=(12, 80))
        null = diagnostic.null_weights(weight, basis)
        np.testing.assert_allclose(x @ null.T, 0, atol=1e-12)
        self.assertGreater(np.mean((rng.normal(size=(40, 80)) @ null.T) ** 2), 1)

    def test_zero_residual_exposes_harm_and_zero_initialization_removes_it(self):
        result = probe.run_probe(seeds=(7,), n_train=12, n_test=32,
                                 baseline_dim=160, width=8, updates=(25, 100))
        ordinary, zero = result["zero_residual_probes"]
        self.assertLess(ordinary["training_history"][-1]["training_mse"], 1e-4)
        self.assertGreater(ordinary["heldout_residual_mse"], 0.1)
        self.assertLess(ordinary["null_removal_max_training_change"], 1e-10)
        self.assertLess(ordinary["heldout_mse_after_null_removal"], 1e-4)
        self.assertEqual(zero["heldout_residual_mse"], 0.0)

    def test_diagnosis_reconstructs_copied_fits_without_writes_and_rejects_tampering(self):
        torch.set_num_threads(1)
        with tempfile.TemporaryDirectory() as temporary:
            original = synthetic_cache(Path(temporary) / "original", protocol="direct-v2")
            fit_outer_fold(original, 0)
            copied = Path(temporary) / "relocated"
            shutil.copytree(original, copied)
            before = snapshot(copied)
            result = diagnostic.diagnose(copied, [0])
            self.assertTrue(result["synthetic"])
            self.assertEqual(snapshot(copied), before)
            fold = result["folds"][0]
            self.assertEqual(len(fold["models"]), 12)
            for model in fold["models"]:
                self.assertLess(model["saved_prediction_max_abs_difference"], 1e-6)
                self.assertLess(model["null_removal_max_training_prediction_change"], 1e-10)
                self.assertAlmostEqual(
                    model["heldout_added_mse"],
                    model["heldout_correction_mse"] - model["heldout_twice_residual_alignment"],
                    places=8,
                )
            path = next(copied.glob("models/fold-0/*/residual-head.pt"))
            with path.open("ab") as handle:
                handle.write(b"tampered")
            with self.assertRaisesRegex(ValueError, "checksum"):
                diagnostic.diagnose(copied, [0])


if __name__ == "__main__":
    unittest.main()
