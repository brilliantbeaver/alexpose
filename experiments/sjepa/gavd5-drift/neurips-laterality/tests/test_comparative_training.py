from dataclasses import replace
from pathlib import Path
import contextlib
import hashlib
import io
import sys
import tempfile
import unittest
from unittest import mock

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from laterality.model import sjepa_cross_entropy
from laterality_extensions.masked_learning import LearningSettings, load_learning_dataset
from laterality_extensions.comparative_masks import MaskPolicy, MaskBudget
from laterality_extensions.comparative_training import (
    _new_model, _save_tables, dense_prediction, per_clip_prediction_loss, train_comparison,
    state_digest, masks_for_batch, plan_real_comparison, real_settings, run_real_comparison,
    evaluate_comparison,
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
                self.assertEqual(
                    first["runs"][name]["resident_input_bytes"],
                    second["runs"][name]["resident_input_bytes"],
                )
                self.assertEqual(
                    first["runs"][name]["execution_layout"],
                    second["runs"][name]["execution_layout"],
                )
            with self.assertRaises(FileExistsError):
                train_comparison(self.data, self.settings, output_dir=folder, reuse=False)
            path = next(Path(folder).glob("*/source_schedule.npy"))
            path.write_bytes(b"incomplete")
            with self.assertRaises(ValueError):
                train_comparison(self.data, self.settings, output_dir=folder)

        import pandas as pd
        with tempfile.TemporaryDirectory() as folder:
            identity = {"test": "immutable summary cache"}
            original = {"scores": pd.DataFrame({"value": [1.0]})}
            _save_tables(original, identity, folder)
            _save_tables(original, identity, folder)
            with self.assertRaisesRegex(ValueError, "disagrees"):
                _save_tables(
                    {"scores": pd.DataFrame({"value": [2.0]})},
                    identity,
                    folder,
                )

    def test_interrupted_pair_resumes_exactly_from_shared_checkpoint(self):
        settings = replace(self.settings, steps=3)
        uninterrupted = train_comparison(self.data, settings)

        def interrupt_during_second_step(update):
            if update["condition_index"] == 1 and update["step"] == 2:
                raise KeyboardInterrupt("simulated notebook interruption")

        with tempfile.TemporaryDirectory() as folder:
            with self.assertRaises(KeyboardInterrupt):
                train_comparison(
                    self.data,
                    settings,
                    output_dir=folder,
                    resume_interval=1,
                    progress_callback=interrupt_during_second_step,
                )
            resumed = train_comparison(
                self.data,
                settings,
                output_dir=folder,
                resume_interval=1,
            )
            self.assertEqual(resumed["resumed_from_step"], 1)
            self.assertFalse(resumed["reused"])
            self.assertFalse(list(Path(folder).glob(".*.resume.pt")))
            self.assertFalse(list(Path(folder).glob(".*.resume.pt.sha256")))
            for name in uninterrupted["runs"]:
                self.assertEqual(
                    state_digest(uninterrupted["runs"][name]["model"]),
                    state_digest(resumed["runs"][name]["model"]),
                )
                np.testing.assert_array_equal(
                    uninterrupted["runs"][name]["history"].drop(
                        columns=["hidden_count_min", "hidden_count_max"]
                    ).to_numpy(),
                    resumed["runs"][name]["history"].drop(
                        columns=["hidden_count_min", "hidden_count_max"]
                    ).to_numpy(),
                )

        with tempfile.TemporaryDirectory() as folder:
            with self.assertRaises(KeyboardInterrupt):
                train_comparison(
                    self.data,
                    settings,
                    output_dir=folder,
                    resume_interval=1,
                    progress_callback=interrupt_during_second_step,
                )
            resume_path = next(Path(folder).glob(".*.resume.pt"))
            payload = torch.load(resume_path, map_location="cpu", weights_only=True)
            first_arm = next(iter(payload["arms"].values()))
            first_arm["optimizer"]["state"] = {}
            torch.save(payload, resume_path)
            checksum = hashlib.sha256(resume_path.read_bytes()).hexdigest()
            resume_path.with_name(f"{resume_path.name}.sha256").write_text(
                f"{checksum}\n"
            )
            with self.assertRaisesRegex(RuntimeError, "optimizer state is incomplete"):
                train_comparison(
                    self.data,
                    settings,
                    output_dir=folder,
                    resume_interval=1,
                )

    def test_real_training_requires_enablement(self):
        with self.assertRaises(PermissionError):
            train_comparison(replace(self.data, synthetic=False), self.settings)
        settings = real_settings()
        self.assertEqual((settings.steps, settings.embed_dim, settings.batch_size), (1200, 96, 20))
        plan = plan_real_comparison()
        self.assertEqual(plan["training_runs"], 50)
        self.assertEqual(plan["optimizer_updates"], 60000)
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            self.assertEqual(run_real_comparison(plan)["status"], "Training disabled")
        log = output.getvalue()
        self.assertIn("What will run: 25 fold/seed jobs", log)
        self.assertIn("60,000 optimizer updates", log)
        self.assertIn("every 300 updates per encoder", log)
        self.assertIn("Hardware:", log)
        self.assertNotIn("condition  updates  batch_size", log)

        import pandas as pd
        with tempfile.TemporaryDirectory() as folder:
            one_job = plan_real_comparison(
                folds=(0,), seeds=(42,), output_dir=folder
            )
            fake_training = {
                "identity": {"test": "one concise log job"},
                "reused": False,
                "resumed_from_step": 0,
            }
            fake_evaluation = {
                "predictions": pd.DataFrame([{
                    "representation": "pretrained_teacher",
                    "observation": "unaltered",
                }]),
                "evaluation_identity": "evaluation-test-digest",
                "evaluation_reused": False,
            }
            output = io.StringIO()
            with (
                mock.patch(
                    "laterality_extensions.comparative_training.load_learning_dataset",
                    return_value=self.data,
                ),
                mock.patch(
                    "laterality_extensions.comparative_training.train_comparison",
                    return_value=fake_training,
                ),
                mock.patch(
                    "laterality_extensions.comparative_training.evaluate_comparison",
                    return_value=fake_evaluation,
                ),
                mock.patch(
                    "laterality_extensions.comparative_training._save_tables"
                ),
                mock.patch(
                    "laterality_extensions.comparative_evaluation.aggregate_predictions",
                    return_value={"per_seed": pd.DataFrame()},
                ),
                mock.patch(
                    "laterality_extensions.comparative_evaluation.paired_source_bootstrap",
                    return_value={"difference": 0.0},
                ),
                contextlib.redirect_stdout(output),
            ):
                run_real_comparison(one_job, enabled=True)
            job_lines = [
                line for line in output.getvalue().splitlines()
                if line.startswith("[01/01]")
            ]
            self.assertEqual(len(job_lines), 2)
            self.assertIn("Starting outer fold 0, seed 42", job_lines[0])
            self.assertIn("Training: 2 new encoders trained", job_lines[1])
            self.assertIn(
                "Frozen-feature evaluation: computed and saved", job_lines[1]
            )

    def test_training_to_retained_evaluation_is_complete(self):
        from laterality_extensions.comparative_evaluation import aggregate_predictions
        import pandas as pd
        with tempfile.TemporaryDirectory() as folder:
            result = train_comparison(self.data, self.settings)
            evaluated = evaluate_comparison(result, self.data, self.settings, output_dir=folder)
            table = evaluated["predictions"]
            self.assertEqual(set(table.condition), set(result["runs"]))
            self.assertEqual(set(table.representation), {"pretrained_online", "pretrained_teacher",
                "initial_online", "direct_pose", "training_mean"})
            expected = pd.DataFrame({"sequence_id": self.data.sequence_ids[self.data.test_rows],
                "source_id": self.data.source_ids[self.data.test_rows], "fold": self.data.fold})
            scored = aggregate_predictions(table, expected, seeds=(self.settings.seed,),
                conditions=tuple(result["runs"]), representations=tuple(table.representation.unique()),
                observations=tuple(table.observation.unique()))
            self.assertTrue((scored["per_seed"].retained_rows == len(self.data.test_rows)).all())
            self.assertTrue((evaluated["output_path"] / "predictions.csv").is_file())
            with mock.patch(
                "laterality_extensions.comparative_evaluation.evaluate_frozen_representations",
                side_effect=AssertionError("evaluation should have been loaded before feature extraction"),
            ):
                repeated = evaluate_comparison(result, self.data, self.settings, output_dir=folder)
            self.assertEqual(repeated["output_path"], evaluated["output_path"])
            self.assertTrue(repeated["evaluation_reused"])
            self.assertEqual(
                repeated["evaluation_identity"], evaluated["evaluation_identity"]
            )
            altered = result["runs"]["all_landmark_targets"]["initial_model"]
            with torch.no_grad():
                next(altered.parameters()).add_(1)
            with self.assertRaisesRegex(ValueError, "initial state"):
                evaluate_comparison(result, self.data, self.settings)


if __name__ == "__main__":
    unittest.main()
