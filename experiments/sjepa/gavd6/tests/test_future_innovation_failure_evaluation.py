"""Forensic calculations must reproduce held-out evidence without trusting scores."""

import json
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np
import pandas as pd
import torch

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts/research_directions/future_innovation"
sys.path.insert(0, str(SCRIPTS))
import evaluate_future_innovation_failures as forensic
sys.path.remove(str(SCRIPTS))

from gavd6_sjepa.research_directions.future_innovation.fi_residual_models import TrainingScaler
from gavd6_sjepa.research_directions.future_innovation.fi_smoke import run_smoke


def snapshot(root):
    return {str(p.relative_to(root)): (p.stat().st_mtime_ns, forensic.sha256_file(p))
            for p in root.rglob("*") if p.is_file()}


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
