import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from gavd6_sjepa.research_directions.future_innovation.fi_contracts import (
    ARMS,
    ModelContract,
    equal_source_weights,
)
from gavd6_sjepa.research_directions.future_innovation.fi_feature_cache import (
    load_cache,
)
from gavd6_sjepa.research_directions.future_innovation.fi_reporting import assemble_oof
from gavd6_sjepa.research_directions.future_innovation.fi_residual_models import (
    TrainingScaler,
    fit_baseline,
    train_head,
)
from gavd6_sjepa.research_directions.future_innovation.fi_smoke import run_smoke


class FutureInnovationTrainingTests(unittest.TestCase):
    def test_weighted_training_only_scaling_and_forward_loss(self):
        torch.set_num_threads(1)
        values = np.array([[0.0, np.nan], [2.0, 1.0], [10.0, 3.0]])
        weights = equal_source_weights(["a", "a", "b"])
        scaler = TrainingScaler.fit(values, weights, ["w0", "w1", "w2"])
        self.assertAlmostEqual(scaler.mean[0], 5.5)
        self.assertAlmostEqual(scaler.mean[1], 7 / 3)
        np.testing.assert_array_equal(
            scaler.mean, TrainingScaler.fit(values, weights, ["w0", "w1", "w2"]).mean
        )
        self.assertTrue(np.isfinite(scaler.transform([[1e9, np.nan]])).all())
        target = np.tile(np.array([0.0, 2.0, 10.0])[:, None], (1, 256))
        base = fit_baseline(
            values, target, ["w0", "w1", "w2"], ["a", "a", "b"], 1.0, 1e-10
        )
        residual = base.y_scaler.transform(target) - base.predict(values)
        head, history = train_head(
            np.ones((3, 32, 33, 4)),
            base.x_scaler.transform(values),
            residual,
            ["a", "a", "b"],
            base.valid_features,
            seed=7,
            weight_decay=0.1,
            updates=[2],
            model_contract=ModelContract(width=8),
        )
        self.assertEqual(head.output.out_features, 256)
        self.assertTrue(np.isfinite(history[0]["training_mse"]))

    def test_all_fold_seed_arm_checkpoint_prediction_report_flow(self):
        torch.set_num_threads(1)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "smoke"
            result = run_smoke(root)
            self.assertTrue(result["synthetic"])
            self.assertEqual(result["stage"], "all five outer folds scored")
            self.assertFalse(result["allow_full_experiment"])
            self.assertFalse(result["allow_adapter_training"])
            checkpoints = list(root.glob("models/fold-*/*/residual-head.pt"))
            self.assertEqual(len(checkpoints), 5 * 5 * 3)
            scores = pd.read_csv(root / "reports/aggregate-metrics.csv")
            self.assertEqual(len(scores), 5 * 3)
            self.assertEqual(set(scores.arm), set(ARMS))
            cohort, cache = load_cache(root)
            predictions = pd.concat(
                [
                    pd.read_parquet(root / f"predictions/fold-{fold}.parquet")
                    for fold in range(5)
                ]
            )
            config = ModelContract(
                width=8,
                updates=(1, 2),
                ridge_alphas=(1.0, 10.0),
                weight_decays=(0.1,),
                bootstrap_repetitions=32,
            )
            assemble_oof(cohort, cache, predictions, config)
            broken = predictions.copy()
            broken.loc[broken.index[0], "target_mean"] = 1e6
            with self.assertRaisesRegex(ValueError, "outer-training"):
                assemble_oof(cohort, cache, broken, config)
            with self.assertRaises(ValueError):
                assemble_oof(cohort, cache, predictions.iloc[:-1], config)
            cache_path = root / "teacher-cache" / f"{cohort.window_id.iloc[0]}.npz"
            with cache_path.open("ab") as handle:
                handle.write(b"tampered")
            with self.assertRaisesRegex(ValueError, "checksum"):
                load_cache(root)
