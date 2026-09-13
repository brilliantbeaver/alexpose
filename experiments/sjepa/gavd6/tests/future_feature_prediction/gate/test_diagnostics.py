"""future feature prediction / gate / test diagnostics."""


import importlib.util
import json
from pathlib import Path
import shutil
import sys
import tempfile
import unittest

import numpy as np
import pandas as pd
import torch

from gavd6_sjepa.research_directions.future_prediction.nested_training import fit_outer_fold
from gavd6_sjepa.research_directions.future_prediction.residual_models import TrainingScaler
from gavd6_sjepa.research_directions.future_prediction.smoke import run_smoke, synthetic_cache
from tests.support import REPO_ROOT, artifact_snapshot as snapshot

SCRIPTS = REPO_ROOT / "scripts/research_directions/future_prediction"

spec = importlib.util.spec_from_file_location(
    "inspect_fits", SCRIPTS / "inspect_fits.py"
)

diagnostic = importlib.util.module_from_spec(spec)

sys.modules[spec.name] = diagnostic

spec.loader.exec_module(diagnostic)

probe_spec = importlib.util.spec_from_file_location(
    "probe_residual_head", SCRIPTS / "probe_residual_head.py"
)

probe = importlib.util.module_from_spec(probe_spec)

probe_spec.loader.exec_module(probe)


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


sys.path.insert(0, str(SCRIPTS))

import evaluate_failures as forensic

sys.path.remove(str(SCRIPTS))


class FailureEvaluationTests(unittest.TestCase):
    def test_unseen_missingness_is_distinguished_from_supported_variation(self):
        scaler = TrainingScaler.fit(np.array([[0., 0.], [0., 1.], [0., 2.]]),
                                    np.ones(3), ["a", "b", "c"])
        heldout = np.array([[0.21875, 100.], [1., 1.]])
        np.testing.assert_array_equal(forensic.unstable_columns(scaler, heldout), [0])
        self.assertEqual(scaler.transform(heldout)[1, 0], 1e8)
        self.assertLess(scaler.transform(heldout)[0, 1], 200.)

    def test_complete_recomputation_is_read_only_and_detects_wrong_sealed_scores(self):
        torch.set_num_threads(1)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "fixture"
            run_smoke(root, protocol="direct-v2")
            before = snapshot(root)
            result = forensic.evaluate(root, zero_init_outer=True)
            self.assertEqual(before, snapshot(root))
            self.assertTrue(result["synthetic"])
            self.assertTrue(result["readiness_receipt_checked"])
            self.assertEqual(len(result["fits"]), 60)
            self.assertEqual(result["bootstrap_rows_recomputed"], 32 * 4)
            scores = pd.DataFrame(result["scores"])
            self.assertEqual(len(scores[scores.variant == "zero_initialized_refit"]), 3)
            # The small fixture has full feature rank: null removal must not invent gains.
            base = scores[scores.variant == "saved_full"].set_index(["arm", "seed"])
            removed = scores[scores.variant == "remove_x_null"].set_index(["arm", "seed"])
            np.testing.assert_allclose(base.r2_full, removed.r2_full, atol=1e-10)
            path = root / "reports/aggregate-metrics.csv"
            table = pd.read_csv(path)
            table.loc[0, "r2_full"] += 0.01
            table.to_csv(path, index=False)
            seal_path = root / "reports/scores-contract.json"
            seal = json.loads(seal_path.read_text())
            seal["artifacts"]["reports/aggregate-metrics.csv"] = forensic.sha256_file(path)
            seal_path.write_text(json.dumps(seal))
            # Even with a matching file digest, score arithmetic must be checked afresh.
            with self.assertRaises(AssertionError):
                forensic.evaluate(root)


if __name__ == "__main__":
    unittest.main()
