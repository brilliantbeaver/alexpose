"""Supervision geometry and independent head updates, without model downloads."""

from copy import deepcopy
from types import SimpleNamespace
import tempfile
from pathlib import Path
import unittest

import numpy as np
import torch

from gavd6_sjepa.research_directions.synthetic_training.estimators import (
    MMPoseEstimator, StudentSpec, _adaptation_pipelines, _images_and_boxes,
    coco_training_labels,
)


class _GradientFixture(torch.nn.Module):
    """Small torch fixture that exercises the adapter, never used by experiments."""

    def __init__(self):
        super().__init__()
        self.backbone = torch.nn.Sequential(
            torch.nn.Linear(3, 4), torch.nn.BatchNorm1d(4), torch.nn.Dropout(0.9)
        )
        self.head = torch.nn.Sequential(torch.nn.BatchNorm1d(4), torch.nn.Linear(4, 34))
        self.dataset_meta = {}
        self.data_preprocessor = lambda batch, training: batch

    def forward(self, inputs, data_samples, mode):
        output = self.head(self.backbone(inputs)).reshape(-1, 17, 2)
        targets = torch.from_numpy(np.concatenate([x["keypoints"] for x in data_samples]))
        weights = torch.from_numpy(np.concatenate([x["keypoints_visible"] for x in data_samples]))
        return {"loss_kpt": ((output - targets).square() * weights[..., None]).mean(),
                "accuracy": torch.tensor(7.0)}


def _collate(records):
    return {"inputs": torch.tensor(np.stack([r["img"].mean(axis=(0, 1)) for r in records]),
                                   dtype=torch.float32) / 255,
            "data_samples": records}


class EstimatorTests(unittest.TestCase):
    def test_missing_labels_cannot_become_face_supervision_or_nan_targets(self):
        xy = np.arange(48, dtype=np.float32).reshape(2, 12, 2)
        visible = np.ones((2, 12), dtype=bool)
        visible[0, 3] = False
        xy[0, 3] = np.nan
        targets, weights = coco_training_labels(xy, visible)
        self.assertTrue(np.isfinite(targets).all())
        np.testing.assert_array_equal(weights[:, :5], 0)
        np.testing.assert_array_equal(weights[:, 5:], visible)
        np.testing.assert_array_equal(targets[1, 5:], xy[1])
        np.testing.assert_array_equal(targets[0, 8], 0)

    def test_visible_nan_label_rejected(self):
        xy = np.zeros((1, 12, 2), dtype=np.float32)
        xy[0, 0] = np.nan
        with self.assertRaisesRegex(ValueError, "Visible"):
            coco_training_labels(xy, np.ones((1, 12), dtype=bool))

    def test_boxes_are_not_recomputed_from_model_or_truth(self):
        images = [np.zeros((20, 40, 3), dtype=np.uint8), np.zeros((10, 30, 3), dtype=np.uint8)]
        _, defaults = _images_and_boxes(images)
        np.testing.assert_array_equal(defaults, [[0, 0, 40, 20], [0, 0, 30, 10]])
        _, supplied = _images_and_boxes(images, [[5, 2, 15, 18], [-2, 1, 25, 12]])
        np.testing.assert_array_equal(supplied, [[5, 2, 15, 18], [-2, 1, 25, 12]])
        with self.assertRaises(ValueError):
            _images_and_boxes(images, [[5, 2, 4, 18], [1, 1, 2, 2]])

    def test_training_keeps_author_crop_and_udp_codec(self):
        pipeline = [{"type": "LoadImage"}, {"type": "GetBBoxCenterScale", "padding": 1.3},
                    {"type": "TopdownAffine", "input_size": (192, 256), "use_udp": True},
                    {"type": "PackPoseInputs"}]
        codec = {"type": "UDPHeatmap", "input_size": (192, 256), "heatmap_size": (48, 64)}
        config = SimpleNamespace(test_dataloader=SimpleNamespace(dataset=SimpleNamespace(pipeline=pipeline)),
                                 model=SimpleNamespace(head=SimpleNamespace(decoder=codec)))
        prediction, training = _adaptation_pipelines(config)
        self.assertEqual(prediction, pipeline)
        self.assertEqual(training[:3], pipeline[:3])
        self.assertEqual(training[3]["encoder"], codec)
        training[1]["padding"] = 99
        self.assertEqual(pipeline[1]["padding"], 1.3)

    def test_backbone_buffers_and_dropout_stay_frozen_and_heads_restore(self):
        torch.manual_seed(3)
        model = _GradientFixture()
        spec = StudentSpec("fixture", "fixture", "unused", "unused")
        estimator = MMPoseEstimator(model, spec, lambda x: x, lambda x: x, _collate)
        images = np.random.default_rng(7).integers(0, 256, (4, 8, 8, 3), dtype=np.uint8)
        xy = np.ones((4, 12, 2), dtype=np.float32)
        visible = np.ones((4, 12), dtype=bool)
        backbone_before = deepcopy(model.backbone.state_dict())
        initial = estimator.head_state()
        self.assertEqual(initial._metadata, model.head.state_dict()._metadata)
        optimizer = estimator.make_optimizer(1e-2)
        loss = estimator.train_batch(images, xy, visible, optimizer)
        self.assertTrue(np.isfinite(loss))
        self.assertFalse(model.backbone.training)
        self.assertTrue(model.head.training)
        self.assertTrue(all(p.grad is None for p in model.backbone.parameters()))
        for key, value in backbone_before.items():
            torch.testing.assert_close(model.backbone.state_dict()[key], value, rtol=0, atol=0)
        first_branch = estimator.head_state()
        self.assertTrue(any(not torch.equal(first_branch[k], initial[k]) for k in initial))
        estimator.load_head_state(initial)
        repeated_optimizer = estimator.make_optimizer(1e-2)
        self.assertEqual(len(repeated_optimizer.state), 0)
        estimator.train_batch(images, xy, visible, repeated_optimizer)
        for key, value in first_branch.items():
            torch.testing.assert_close(estimator.head_state()[key], value, rtol=0, atol=0)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "head.pt"
            estimator.save_head(path)
            estimator.load_head_state(initial)
            estimator.load_head(path)
            for key, value in first_branch.items():
                torch.testing.assert_close(estimator.head_state()[key], value, rtol=0, atol=0)

    def test_rgb_is_converted_once_for_author_bgr_input(self):
        model = _GradientFixture()
        estimator = MMPoseEstimator(model, StudentSpec("fixture", "fixture", "", ""),
                                    lambda x: x, lambda x: x, _collate)
        image = np.full((2, 2, 3), [10, 20, 30], dtype=np.uint8)
        batch = estimator._batch([image], None)
        np.testing.assert_array_equal(batch["data_samples"][0]["img"][0, 0], [30, 20, 10])

    def test_failed_predictions_are_retained_for_reference_penalty(self):
        model = _GradientFixture()
        coordinates = np.arange(34, dtype=np.float32).reshape(1, 17, 2)
        coordinates[0, 8, 0] = np.inf
        model.test_step = lambda batch: [SimpleNamespace(pred_instances=SimpleNamespace(keypoints=coordinates))
                                         for _ in batch["data_samples"]]
        estimator = MMPoseEstimator(model, StudentSpec("fixture", "fixture", "", ""),
                                    lambda x: x, lambda x: x, _collate)
        image = np.zeros((2, 2, 3), dtype=np.uint8)
        predictions = estimator.predict([image])
        self.assertEqual(predictions.shape, (1, 12, 2))
        self.assertTrue(np.isnan(predictions[0, 3]).all())
        np.testing.assert_array_equal(predictions[0, 0], coordinates[0, 5])


if __name__ == "__main__":
    unittest.main()
