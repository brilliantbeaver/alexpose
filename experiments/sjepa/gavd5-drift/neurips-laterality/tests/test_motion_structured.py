"""Behavioral checks for the new masking study and its information boundaries."""
from dataclasses import asdict, replace
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from laterality_extensions.comparative_masks import InfeasibleMaskBudget, reflect_sample
from laterality_extensions.comparative_training import _new_model, dense_prediction, state_digest
from laterality_extensions.comparative_evaluation import encode_laterality_features
from laterality_extensions.masked_learning import LearningSettings, load_learning_dataset
from laterality_extensions.motion_structured_masks import (
    StudyArm, context_cue_audit, mamp_logits, paired_study_masks, sample_study_mask, study_arms,
)
from laterality_extensions.motion_structured_training import plan_mask_study, run_mask_study, train_mask_study
from laterality_extensions.motion_readout import (
    aggregate_motion_study, bilateral_summaries, encode_motion_summaries,
    evaluate_motion_readouts, evaluate_retained_comparison, pooling_positive_control,
)


class MotionMaskTests(unittest.TestCase):
    def setUp(self):
        self.xyz = np.zeros((16, 33, 3))
        self.xyz[:, 27, 0] = np.arange(16) * 0.2
        self.valid = np.ones((4, 33), bool)

    def test_official_code_log_weights_on_complete_inputs(self):
        rng = np.random.default_rng(20)
        xyz = rng.normal(size=self.xyz.shape)
        x = torch.tensor(xyz.reshape(4, 4, 33, 3).transpose(0, 2, 1, 3).reshape(1, 4, 33, 12))
        motion = torch.zeros_like(x)
        motion[:, 1:] = (x[:, 1:] - x[:, :-1]).abs()
        motion[:, 0] = motion[:, 1]
        intensity = motion.mean(-1).flatten(1)
        official = intensity / (intensity.max(1, keepdim=True).values * 0.8 + 1e-10)
        actual, _ = mamp_logits(xyz, self.valid)
        np.testing.assert_allclose(actual.ravel(), official.numpy()[0], atol=1e-12)

    def test_motion_bias_reproducibility_and_missing_values(self):
        hits = []
        for name in ("uniform", "mamp_motion", "robust_motion"):
            arm = StudyArm(name)
            rng = np.random.default_rng(15)
            hits.append(sum(sample_study_mask(self.xyz, self.valid, arm, 16, rng).mask[:, 27].sum()
                            for _ in range(400)))
            first = sample_study_mask(self.xyz, self.valid, arm, 16, np.random.default_rng(1))
            second = sample_study_mask(self.xyz, self.valid, arm, 16, np.random.default_rng(1))
            np.testing.assert_array_equal(first.mask, second.mask)
            self.assertEqual(len(first.target_indices), 16)
        self.assertGreater(hits[1], 1.5 * hits[0])
        self.assertGreater(hits[2], hits[1])
        valid = self.valid.copy(); valid[1, 27] = False
        observed = np.repeat(valid, 4, axis=0)
        changed = self.xyz.copy(); changed[~observed] = np.nan
        clean, _ = mamp_logits(changed, valid, observation_valid=observed)
        changed[~observed] = 1e12
        other, _ = mamp_logits(changed, valid, observation_valid=observed)
        np.testing.assert_array_equal(clean, other)
        result = sample_study_mask(changed, valid, StudyArm("mamp_motion"), 12,
                                   np.random.default_rng(1), observation_valid=observed)
        self.assertFalse((result.mask & ~valid).any())

    def test_stationary_fallback_and_reflection(self):
        logits, metadata = mamp_logits(self.xyz * 0, self.valid)
        self.assertTrue(metadata["motion_fallback_uniform"])
        np.testing.assert_array_equal(logits, np.zeros_like(logits))
        xyz, valid, _ = reflect_sample(self.xyz, self.valid, np.zeros_like(self.valid))
        reflected, _ = mamp_logits(xyz, valid)
        original, _ = mamp_logits(self.xyz, self.valid)
        np.testing.assert_array_equal(reflected[:, 28], original[:, 27])
        with self.assertRaises(InfeasibleMaskBudget):
            sample_study_mask(self.xyz, self.valid, StudyArm("mamp_motion"), 132, np.random.default_rng(1))
        with self.assertRaises(ValueError):
            mamp_logits(self.xyz, self.valid, temperature=0)

    def test_structures_remove_local_cues(self):
        for experiment in ("regions", "trajectories", "completion"):
            arms = study_arms(experiment, 4)
            arm = list(arms.values())[1]
            result = sample_study_mask(self.xyz, self.valid, arm, 24, np.random.default_rng(3))
            meta = result.coverage
            expected = np.zeros_like(self.valid)
            start = meta["interval_start"]
            expected[start:start + meta["interval_blocks"], meta["selected_landmarks"]] = True
            np.testing.assert_array_equal(result.mask, expected)
            self.assertGreater(result.context_count, 0)
            if experiment == "trajectories":
                self.assertEqual(context_cue_audit(result.mask, self.valid)["temporal_bracket_fraction"], 0)


class StudyTrainingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.environment = patch.dict(os.environ, {"LATERALITY_RESEARCH_CPU_THREADS": "1"})
        cls.environment.start()
        torch.set_num_threads(1)
        cls.data = load_learning_dataset()
        cls.settings = LearningSettings(steps=2)

    @classmethod
    def tearDownClass(cls):
        cls.environment.stop()

    def test_paired_training_cache_and_teacher_isolation(self):
        with tempfile.TemporaryDirectory() as folder:
            result = train_mask_study(self.data, self.settings, output_dir=folder, checkpoint_steps=(1,))
            self.assertTrue(all(result["pairing"].values()))
            reused = train_mask_study(self.data, self.settings, output_dir=folder, checkpoint_steps=(1,))
            self.assertTrue(reused["reused"])
            for name, run in result["runs"].items():
                self.assertEqual(state_digest(run["model"]), state_digest(reused["runs"][name]["model"]))
                self.assertTrue(all(p.grad is None for p in run["model"].target_encoder.parameters()))
            path = next(Path(folder).glob("*/*_training.csv"))
            path.write_text("incomplete")
            with self.assertRaises(ValueError):
                train_mask_study(self.data, self.settings, output_dir=folder, checkpoint_steps=(1,))

    def test_labels_and_outer_test_content_cannot_change_training(self):
        first = train_mask_study(self.data, self.settings)
        xyz = self.data.xyz.copy(); xyz[self.data.test_rows] += 10
        altered = replace(self.data, xyz=xyz, targets=self.data.targets[::-1].copy())
        second = train_mask_study(altered, self.settings)
        for name in first["runs"]:
            self.assertEqual(state_digest(first["runs"][name]["model"]), state_digest(second["runs"][name]["model"]))

    def test_dense_predictions_exclude_fixed_hidden_content(self):
        model, _ = _new_model(self.settings, 16, "cpu")
        xyz = torch.tensor(self.data.xyz[:2], dtype=torch.float32)
        valid = torch.ones(2, 4, 33, dtype=torch.bool)
        mask = torch.zeros_like(valid); mask[0, :2, 27] = True; mask[1, :, 25:29] = True
        changed = xyz.clone(); changed[mask.repeat_interleave(4, dim=1)] += 999
        first, _ = dense_prediction(model, xyz, xyz, valid, mask)
        second, _ = dense_prediction(model, changed, xyz, valid, mask)
        torch.testing.assert_close(first, second, atol=0, rtol=0)

    def test_structured_training_and_budget_pairing(self):
        for experiment in ("regions", "trajectories", "completion"):
            result = train_mask_study(self.data, replace(self.settings, steps=1), experiment=experiment)
            self.assertTrue(all(result["pairing"].values()))

    def test_plan_without_local_artifacts_and_disabled_run(self):
        plan = plan_mask_study()
        self.assertEqual(plan["training_runs"], 125)
        self.assertEqual(plan["optimizer_updates"], 150000)
        self.assertEqual(plan["settings"]["embed_dim"], 96)
        with patch("laterality_extensions.motion_structured_training.train_mask_study", side_effect=AssertionError("must not train")):
            self.assertEqual(run_mask_study(plan, enabled=False, log=None)["status"], "Training disabled")

    def test_enabled_runner_saves_reuses_and_rejects_misreported_workload(self):
        with tempfile.TemporaryDirectory() as folder:
            plan = plan_mask_study(experiments=("motion",), folds=(0,), seeds=(42,), output_dir=folder)
            plan["settings"] = asdict(replace(self.settings, seed=42, steps=1))
            plan["workload"]["updates"] = 1
            plan["optimizer_updates"] = 3
            with patch("laterality_extensions.motion_structured_training.load_learning_dataset", return_value=self.data):
                result = run_mask_study(plan, enabled=True, log=None)
                self.assertEqual(result["status"], "Complete")
                self.assertTrue(result["predictions"].synthetic.all())
                self.assertTrue(list(Path(folder).glob("evaluations/*/predictor_diagnostics.csv")))
                reused = run_mask_study(plan, enabled=True, log=None)
                pd.testing.assert_frame_equal(result["predictions"], reused["predictions"])
                plan["workload"].loc[0, "seed"] = 43
                with self.assertRaisesRegex(ValueError, "Workload"):
                    run_mask_study(plan, enabled=True, log=None)

    def test_readout_positive_control_and_mean_compatibility(self):
        score, _ = pooling_positive_control()
        self.assertGreater(score.set_index("summary").loc["mean_motion", "r2"], .99)
        self.assertTrue(score.set_index("summary").loc["mean", "near_constant"])
        model, _ = _new_model(self.settings, 16, "cpu")
        old, _ = encode_laterality_features(model.view_encoder, self.data)
        new = encode_motion_summaries(model.view_encoder, self.data)["mean"]
        np.testing.assert_allclose(old, new, atol=1e-6)

    def test_readout_selection_ignores_outer_test_values(self):
        result = train_mask_study(self.data, replace(self.settings, steps=1))
        first = evaluate_motion_readouts(result, self.data, self.settings, alphas=(1., 1000.))
        xyz = self.data.xyz.copy(); xyz[self.data.test_rows] *= 3
        targets = self.data.targets.copy(); targets[self.data.test_rows] += 100
        second = evaluate_motion_readouts(result, replace(self.data, xyz=xyz, targets=targets), self.settings, alphas=(1., 1000.))
        pd.testing.assert_frame_equal(first["selection"], second["selection"])
        expected = pd.DataFrame({"sequence_id": self.data.sequence_ids[self.data.test_rows],
            "source_id": self.data.source_ids[self.data.test_rows], "fold": 0})
        predictions = first["predictions"].assign(experiment="motion")
        plan = {"experiments": ("motion",), "seeds": (7,), "arms": {"motion": result["runs"]}}
        summary = aggregate_motion_study(predictions, expected, plan)
        self.assertEqual(summary["per_seed"].evaluated_clips.unique().tolist(), [len(expected)])
        with self.assertRaises(ValueError):
            aggregate_motion_study(pd.concat((predictions, predictions.iloc[:1])), expected, plan)
        with self.assertRaises(ValueError):
            aggregate_motion_study(predictions.iloc[1:], expected, plan)

    def test_retained_comparison_can_be_reanalysed_without_encoder_training(self):
        from laterality_extensions.comparative_training import train_comparison
        with tempfile.TemporaryDirectory() as folder:
            train_comparison(self.data, replace(self.settings, steps=1), output_dir=folder)
            saved = next(Path(folder).glob("*/manifest.json")).parent
            original = {p.name: p.read_bytes() for p in saved.iterdir()}
            with patch("laterality_extensions.comparative_training.train_comparison", side_effect=AssertionError("must not train")):
                result = evaluate_retained_comparison(saved, output_dir=Path(folder) / "readouts")
            self.assertFalse(result["predictor_diagnostics"].empty)
            self.assertEqual(original, {p.name: p.read_bytes() for p in saved.iterdir()})


if __name__ == "__main__":
    unittest.main()
