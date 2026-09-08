import unittest

import numpy as np

from gavd6_sjepa.research_directions.future_innovation.fi_contracts import (
    equal_source_weights,
)
from gavd6_sjepa.research_directions.future_innovation.fi_metrics import (
    featurewise_predictive_r2,
    score_arrays,
    source_bootstrap_indices,
)


class FutureInnovationMetricTests(unittest.TestCase):
    def test_analytical_predictive_r2_and_gain(self):
        target = np.array([[1.0, 2.0], [3.0, 4.0]])
        baseline = target * 0.5
        full = target * 0.75
        score, _, _, _ = score_arrays(
            target, baseline, full, np.ones(2), np.ones(2, dtype=bool)
        )
        self.assertAlmostEqual(score["r2_baseline"], 0.75)
        self.assertAlmostEqual(score["r2_full"], 0.9375)
        self.assertAlmostEqual(score["delta_r2"], 0.1875)
        self.assertAlmostEqual(score["f8"], 0.75)
        self.assertAlmostEqual(score["delta_r2"], score["delta_feature_mean"])
        self.assertIsNone(
            score_arrays(target, target, target, np.ones(2), np.ones(2, dtype=bool))[0][
                "f8"
            ]
        )

    def test_source_weights_and_bootstrap_preserve_draw_multiplicity(self):
        ids = np.array(["a", "a", "b", "c"])
        weights = equal_source_weights(ids)
        self.assertAlmostEqual(weights[:2].sum(), weights[2])
        self.assertAlmostEqual(weights[2], weights[3])
        draws = list(source_bootstrap_indices(ids, repetitions=30, seed=1))
        self.assertTrue(any(len(set(ids[index])) < 3 for index in draws))
        for index in draws:
            self.assertAlmostEqual(weights[index].sum(), 4)
            for source in set(ids[index]):
                count = np.sum(ids[index] == source)
                self.assertEqual(count % np.sum(ids == source), 0)

    def test_constant_features_and_nonfinite_values_fail(self):
        target = np.ones((3, 2))
        score, mask = featurewise_predictive_r2(
            target, target, target * 0, np.ones(3), np.array([True, False])
        )
        self.assertEqual(score[0], 1)
        self.assertTrue(np.isnan(score[1]))
        self.assertFalse(mask[1])
        with self.assertRaises(ValueError):
            featurewise_predictive_r2(
                target, target, target * 0, np.ones(3), np.zeros(2, dtype=bool)
            )
        for bad in (np.nan, np.inf):
            prediction = target.copy()
            prediction[0, 0] = bad
            with self.assertRaises(ValueError):
                featurewise_predictive_r2(
                    target, prediction, target * 0, np.ones(3), np.ones(2, dtype=bool)
                )
        with self.assertRaises(ValueError):
            featurewise_predictive_r2(
                target, target, target * 0, np.array([1, 0, 1]), np.ones(2, dtype=bool)
            )
