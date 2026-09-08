from dataclasses import replace
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from laterality.model import sjepa_cross_entropy
from laterality_extensions.masked_learning import LearningSettings, load_learning_dataset
from laterality_extensions.comparative_masks import MaskPolicy, MaskBudget
from laterality_extensions.comparative_training import (
    _new_model, dense_prediction, per_clip_prediction_loss, train_comparison,
    state_digest, masks_for_batch, plan_real_comparison, real_settings, run_real_comparison,
)


class ComparativeTrainingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        cls.data = load_learning_dataset()
        cls.settings = LearningSettings(steps=2)

    def test_dense_equal_count_agrees_with_original(self):
        model, _ = _new_model(self.settings, 16, "cpu")
        model.eval()
        xyz = torch.randn(2, 16, 33, 3)
        valid = torch.ones(2, 4, 33, dtype=torch.bool)
        mask = torch.zeros_like(valid); mask[:, 0, :7] = True
        dense, target = dense_prediction(model, xyz, xyz, valid, mask)
        old, old_target = model(xyz, xyz, valid, mask)
        torch.testing.assert_close(dense[mask.flatten(1)].reshape(2, 7, -1), old)
        torch.testing.assert_close(target[mask.flatten(1)].reshape(2, 7, -1), old_target)
        torch.testing.assert_close(per_clip_prediction_loss(dense, target, mask, model.target_center),
            sjepa_cross_entropy(old, old_target, model.target_center))

    def test_ragged_targets_equal_independent_clip_losses(self):
        model, _ = _new_model(self.settings, 16, "cpu")
        xyz = torch.randn(2, 16, 33, 3)
        valid = torch.ones(2, 4, 33, dtype=torch.bool)
        mask = torch.zeros_like(valid); mask[0, 0, :3] = True; mask[1, :2, :5] = True
        dense, target = dense_prediction(model, xyz, xyz, valid, mask)
        expected = []
        for i in range(2):
            p, t = model(xyz[i:i+1], xyz[i:i+1], valid[i:i+1], mask[i:i+1])
            expected.append(sjepa_cross_entropy(p, t, model.target_center))
        torch.testing.assert_close(per_clip_prediction_loss(dense, target, mask, model.target_center),
            torch.stack(expected).mean(), atol=2e-5, rtol=2e-5)

    def test_hidden_content_isolation_with_mask_held_fixed(self):
        model, _ = _new_model(self.settings, 16, "cpu")
        xyz = torch.randn(2, 16, 33, 3)
        valid = torch.ones(2, 4, 33, dtype=torch.bool)
        mask = torch.zeros_like(valid); mask[:, :2, 11] = True
        changed = xyz.clone(); changed[:, :8, 11] += 10000
        p, _ = dense_prediction(model, xyz, xyz, valid, mask)
        q, _ = dense_prediction(model, changed, xyz, valid, mask)
        torch.testing.assert_close(p, q, atol=0, rtol=0)

    def test_paired_training_and_labels_do_not_change_pretraining(self):
        first = train_comparison(self.data, self.settings)
        changed = replace(self.data, targets=self.data.targets[::-1].copy())
        second = train_comparison(changed, self.settings)
        self.assertTrue(all(first["pairing"].values()))
        for name, run in first["runs"].items():
            self.assertEqual(state_digest(run["model"]), state_digest(second["runs"][name]["model"]))
            self.assertTrue(all(p.grad is None for p in run["model"].target_encoder.parameters()))
            self.assertEqual(set(run["checkpoints"]), {2})

    def test_outer_test_values_do_not_change_trained_weights(self):
        first = train_comparison(self.data, self.settings)
        xyz = self.data.xyz.copy(); xyz[self.data.test_rows] += 3
        changed = replace(self.data, xyz=xyz)
        second = train_comparison(changed, self.settings)
        for name in first["runs"]:
            self.assertEqual(state_digest(first["runs"][name]["model"]), state_digest(second["runs"][name]["model"]))

    def test_structure_and_reference_counts_stay_paired(self):
        conditions = {"trajectories": MaskPolicy("whole_trajectory"), "scattered": MaskPolicy("uniform")}
        m, c = masks_for_batch(self.data, self.data.train_rows[:3], self.settings, conditions,
            budgets={"trajectories": MaskBudget(trajectories=2)}, matched_to="trajectories")
        np.testing.assert_array_equal(m["trajectories"].sum((1,2)), m["scattered"].sum((1,2)))
        self.assertEqual(len(c["trajectories"]), 3)
        result = train_comparison(self.data, self.settings, conditions,
            budgets={"trajectories": MaskBudget(trajectories=2)}, matched_to="trajectories")
        self.assertTrue(all(result["pairing"].values()))

    def test_completed_cache_reuse_and_corruption_rejection(self):
        with tempfile.TemporaryDirectory() as folder:
            first = train_comparison(self.data, self.settings, output_dir=folder)
            second = train_comparison(self.data, self.settings, output_dir=folder)
            self.assertTrue(second["reused"])
            for name in first["runs"]:
                self.assertEqual(state_digest(first["runs"][name]["model"]), state_digest(second["runs"][name]["model"]))
            with self.assertRaises(FileExistsError):
                train_comparison(self.data, self.settings, output_dir=folder, reuse=False)
            path = next(Path(folder).glob("*/source_schedule.npy"))
            path.write_bytes(b"incomplete")
            with self.assertRaises(ValueError):
                train_comparison(self.data, self.settings, output_dir=folder)

    def test_real_training_requires_enablement(self):
        with self.assertRaises(PermissionError):
            train_comparison(replace(self.data, synthetic=False), self.settings)
        settings = real_settings()
        self.assertEqual((settings.steps, settings.embed_dim, settings.batch_size), (1200, 96, 20))
        plan = plan_real_comparison()
        self.assertEqual(plan["training_runs"], 50)
        self.assertEqual(plan["optimizer_updates"], 60000)
        self.assertEqual(run_real_comparison(plan)["status"], "Training disabled")


if __name__ == "__main__":
    unittest.main()
