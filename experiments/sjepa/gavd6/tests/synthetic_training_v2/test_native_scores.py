"""Native MMPose heatmap/SimCC maxima are scores, not probabilities."""
import json
from types import SimpleNamespace
import unittest

import numpy as np
import torch

from gavd6_sjepa.research_directions.synthetic_training_v2.contracts import validate_inputs
from gavd6_sjepa.research_directions.synthetic_training_v2.data import fixture_bundle, normalize_inputs
from gavd6_sjepa.research_directions.synthetic_training_v2.extraction import extract_tracks
from gavd6_sjepa.research_directions.synthetic_training_v2.models import (
    ModelConfig, RestorationModel, validate_inputs as validate_model_inputs,
)


def extract_score_fixture(body_scores, *, threshold=0.):
    scores = np.concatenate([np.ones(5, np.float32), np.asarray(body_scores, np.float32)])[None]
    class NativeScoreModel:
        def eval(self):
            return self

        def test_step(self, batch):
            return [SimpleNamespace(pred_instances=SimpleNamespace(
                keypoints=np.full((1, 17, 2), 5., np.float32), keypoint_scores=scores))]
    estimator = SimpleNamespace(model=NativeScoreModel(), _batch=lambda images, boxes: None)
    return extract_tracks(estimator, [np.zeros((8, 8, 3), np.uint8)], [[0, 0, 8, 8]], [0.],
                          box_source="fixture_supplied", threshold=threshold)


class NativeScoreTests(unittest.TestCase):
    def test_unbounded_scores_are_preserved_and_codec_invalid_scores_are_missing(self):
        scores = np.array([1.25, 3.5, .8, 0., -.2, np.nan, np.inf, .4, .3, .2, .1, .9], np.float32)
        track = extract_score_fixture(scores)
        np.testing.assert_equal(track.confidence[0], scores)
        expected = np.isfinite(scores) & (scores > 0)
        np.testing.assert_array_equal(track.observed[0], expected)
        self.assertTrue(np.isnan(track.xy[0, ~expected]).all())
        self.assertEqual(track.status_counts["score_semantics"], "native_mmpose_keypoint_scores_not_probabilities")
        self.assertAlmostEqual(track.status_counts["native_score_min"], -.2, places=6)
        self.assertEqual(track.status_counts["native_score_max"], 3.5)
        self.assertEqual(track.status_counts["nonfinite_score_joints"], 2)
        self.assertEqual(track.status_counts["nonpositive_score_joints"], 2)
        json.dumps(track.status_counts, allow_nan=False)

    def test_all_missing_scores_have_json_safe_absent_range(self):
        track = extract_score_fixture(np.full(12, np.nan))
        self.assertFalse(track.observed.any())
        self.assertIsNone(track.status_counts["native_score_min"])
        self.assertIsNone(track.status_counts["native_score_max"])
        json.dumps(track.status_counts, allow_nan=False)

    def test_native_score_threshold_is_not_a_probability(self):
        track = extract_score_fixture([1.1, 1.2, 1.3] * 4, threshold=1.2)
        np.testing.assert_array_equal(track.observed[0], np.array([False, False, True] * 4))
        for threshold in (-.1, np.inf, np.nan):
            with self.subTest(threshold=threshold), self.assertRaises(ValueError):
                extract_score_fixture(np.ones(12), threshold=threshold)

    def test_observed_joint_requires_a_finite_positive_native_score(self):
        for score in (-.2, 0., np.nan, np.inf):
            inputs = {key: value[:1].copy() for key, value in fixture_bundle().inputs.items()}
            inputs['observed'][0, 0, 0] = True
            inputs['confidence'][0, 0, 0] = score
            with self.subTest(contract='raw', score=score), self.assertRaisesRegex(ValueError, '[Ss]core'):
                validate_inputs(inputs)
            # Test the model boundary independently, without having the raw
            # validator prevent malformed imported arrays from reaching it.
            tensors = {key: torch.as_tensor(value, dtype=torch.bool if key == 'observed' else torch.float32)
                       for key, value in inputs.items()}
            with self.subTest(contract='model', score=score), self.assertRaisesRegex(ValueError, '[Ss]core'):
                validate_model_inputs(tensors)

    def test_nonpositive_scores_are_retained_when_the_joint_is_unobserved(self):
        inputs = {key: value[:1].copy() for key, value in fixture_bundle().inputs.items()}
        inputs['observed'][0, 0, :2] = False
        inputs['xy'][0, 0, :2] = np.nan
        inputs['confidence'][0, 0, :2] = [-.2, 0.]
        validate_inputs(inputs)
        normalized, _ = normalize_inputs(inputs)
        np.testing.assert_array_equal(normalized['confidence'][0, 0, :2], inputs['confidence'][0, 0, :2])
        tensors = {key: torch.as_tensor(value, dtype=torch.bool if key == 'observed' else torch.float32)
                   for key, value in normalized.items()}
        validate_model_inputs(tensors)

    def test_raw_contract_normalization_and_every_model_accept_native_scores(self):
        inputs = {key: value[:2].copy() for key, value in fixture_bundle().inputs.items()}
        inputs["confidence"][inputs["observed"]] = 3.5
        inputs["observed"][:, 0, 0] = False
        inputs["xy"][:, 0, 0] = np.nan
        inputs["confidence"][:, 0, 0] = -.2
        inputs["observed"][:, 0, 1] = False
        inputs["xy"][:, 0, 1] = np.nan
        inputs["confidence"][:, 0, 1] = np.nan
        validate_inputs(inputs)
        normalized, _ = normalize_inputs(inputs)
        self.assertEqual(normalized["confidence"][0, 1, 0], 3.5)
        self.assertAlmostEqual(normalized["confidence"][0, 0, 0], -.2)
        self.assertEqual(normalized["confidence"][0, 0, 1], 0.)
        self.assertTrue(np.isnan(inputs["confidence"][0, 0, 1]))
        tensors = {key: torch.as_tensor(value, dtype=torch.bool if key == "observed" else torch.float32)
                   for key, value in normalized.items()}
        cfg = ModelConfig(width=8, encoder_layers=1, predictor_layers=1, heads=2)
        for arm in ("direct", "smoothnet", "static"):
            with self.subTest(arm=arm), torch.inference_mode():
                self.assertTrue(torch.isfinite(RestorationModel(arm, cfg)(tensors)).all())


if __name__ == "__main__":
    unittest.main()
