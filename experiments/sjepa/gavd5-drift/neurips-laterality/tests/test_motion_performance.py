"""Check numerical equivalence, cache boundaries and optimizer recovery."""
from dataclasses import replace
from pathlib import Path
import os
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from laterality_extensions.masked_learning import LearningSettings, load_learning_dataset
from laterality_extensions.comparative_training import _new_model, _prediction_prevalidated, state_digest
from laterality_extensions.motion_runtime import prediction_resident, motion_numerical_context
from laterality_extensions.motion_structured_masks import paired_study_masks, study_arms
from laterality_extensions.motion_structured_training import train_mask_study, mask_study_identity
from laterality_extensions import motion_structured_masks as masks_module
from laterality_extensions.motion_readout import prepare_motion_evaluation_inputs, encode_motion_summaries


class MotionPerformanceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.environment = patch.dict(os.environ, {"LATERALITY_RESEARCH_CPU_THREADS": "1"})
        cls.environment.start()
        torch.set_num_threads(1)
        cls.data = load_learning_dataset()
        cls.settings = LearningSettings(steps=4)

    @classmethod
    def tearDownClass(cls):
        cls.environment.stop()

    def test_score_cache_preserves_draws_metadata_and_detects_mutated_coordinates(self):
        data = self.data
        rows = data.train_rows[:4]
        arms = study_arms("motion", data.xyz.shape[1] // 4)
        cache = {}
        for step in (0, 1, 17):
            expected, metadata = paired_study_masks(data, rows, self.settings, arms, step=step)
            actual, cached_metadata = paired_study_masks(data, rows, self.settings, arms,
                step=step, score_cache=cache)
            self.assertEqual(metadata, cached_metadata)
            for name in arms:
                np.testing.assert_array_equal(expected[name], actual[name])
        with patch.object(masks_module, "motion_scores", side_effect=AssertionError("must reuse")), \
             patch.object(masks_module, "mamp_logits", side_effect=AssertionError("must reuse")):
            paired_study_masks(data, rows, self.settings, arms, step=23, score_cache=cache)
        altered = replace(data, xyz=data.xyz.copy())
        altered.xyz[rows[0], 5:9, 27, 0] += 0.1
        previous = len(cache)
        paired_study_masks(altered, rows, self.settings, arms, score_cache=cache)
        self.assertGreater(len(cache), previous)

    def test_gather_and_ragged_losses_and_gradients_match_reference(self):
        model, _ = _new_model(self.settings, 16, "cpu")
        xyz = torch.tensor(self.data.xyz[:2], dtype=torch.float32)
        valid = torch.ones(2, 4, 33, dtype=torch.bool)
        for equal in (True, False):
            mask = torch.zeros_like(valid)
            mask[:, :2, 27] = True
            if not equal:
                mask[1, :, 25:28] = True
            indices = torch.tensor(np.nonzero(mask.flatten(1).numpy())[1].reshape(2, -1)) if equal else None
            model.zero_grad(set_to_none=True)
            reference = _prediction_prevalidated(model, xyz, xyz, valid, mask, equal_target_counts=equal)[0]
            reference.backward()
            gradients = {name: p.grad.clone() for name, p in model.named_parameters() if p.grad is not None}
            model.zero_grad(set_to_none=True)
            actual = prediction_resident(model, xyz, xyz, valid, mask, indices, precision="fp32")[0]
            actual.backward()
            torch.testing.assert_close(actual, reference, rtol=0, atol=0)
            for name, parameter in model.named_parameters():
                if name in gradients:
                    torch.testing.assert_close(parameter.grad, gradients[name], rtol=0, atol=0)

    def test_resume_matches_uninterrupted_and_fold_tensor_cache_is_reused(self):
        cache = {}
        reference = train_mask_study(self.data, self.settings, input_cache=cache)
        with tempfile.TemporaryDirectory() as folder:
            def interrupt(update):
                if update["step"] == 3:
                    raise KeyboardInterrupt("simulated interruption after saved step 2")
            with self.assertRaises(KeyboardInterrupt):
                train_mask_study(self.data, self.settings, output_dir=folder,
                    resume_interval=2, input_cache=cache, progress=interrupt)
            resumed = train_mask_study(self.data, self.settings, output_dir=folder,
                resume_interval=2, input_cache=cache)
            self.assertEqual(resumed["resumed_from_step"], 2)
            self.assertTrue(resumed["tensor_cache_reused"])
            for name in reference["runs"]:
                self.assertEqual(state_digest(reference["runs"][name]["model"]),
                                 state_digest(resumed["runs"][name]["model"]))
                pd.testing.assert_frame_equal(reference["runs"][name]["history"], resumed["runs"][name]["history"])

    def test_runtime_restored_on_failure_and_cpu_bf16_is_rejected(self):
        previous = torch.get_float32_matmul_precision()
        try:
            torch.set_float32_matmul_precision("high")
            with self.assertRaises(RuntimeError), motion_numerical_context():
                self.assertEqual(torch.get_float32_matmul_precision(), "highest")
                raise RuntimeError("stop")
            self.assertEqual(torch.get_float32_matmul_precision(), "high")
        finally:
            torch.set_float32_matmul_precision(previous)
        with self.assertRaisesRegex(ValueError, "CUDA"):
            train_mask_study(self.data, self.settings, precision="bf16")

    def test_evaluation_cache_rejects_in_place_changes(self):
        data = replace(self.data, xyz=self.data.xyz.copy())
        model, _ = _new_model(self.settings, 16, "cpu")
        prepared = prepare_motion_evaluation_inputs(data, segment_length=4, device="cpu")
        encode_motion_summaries(model.view_encoder, data, prepared_inputs=prepared)
        data.xyz[0, 0, 27, 0] += 1
        with self.assertRaisesRegex(ValueError, "stale"):
            encode_motion_summaries(model.view_encoder, data, prepared_inputs=prepared)

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA validation requires a CUDA-enabled kernel")
    def test_cuda_bf16_uses_distinct_cache_and_retains_fp32_teacher_without_gradients(self):
        settings = replace(self.settings, device="cuda", steps=4)
        arms = study_arms("motion", 4)
        self.assertNotEqual(mask_study_identity(self.data, settings, "motion", arms),
            mask_study_identity(self.data, settings, "motion", arms, precision="bf16"))
        with tempfile.TemporaryDirectory() as folder:
            first = train_mask_study(self.data, settings, precision="bf16", output_dir=folder)
            second = train_mask_study(self.data, settings, precision="bf16", output_dir=folder)
            self.assertTrue(second["reused"])
            self.assertTrue(all(first["pairing"].values()))
            for name, run in first["runs"].items():
                self.assertTrue(np.isfinite(run["history"].loss).all())
                self.assertTrue(all(p.dtype == torch.float32 and p.grad is None
                    for p in run["model"].target_encoder.parameters()))
                self.assertEqual(state_digest(run["model"]), state_digest(second["runs"][name]["model"]))
            resumed_folder = Path(folder) / "resume_check"
            def interrupt(update):
                if update["step"] == 3:
                    raise KeyboardInterrupt()
            with self.assertRaises(KeyboardInterrupt):
                train_mask_study(self.data, settings, precision="bf16", output_dir=resumed_folder,
                    resume_interval=2, progress=interrupt)
            resumed = train_mask_study(self.data, settings, precision="bf16", output_dir=resumed_folder,
                resume_interval=2)
            self.assertEqual(resumed["resumed_from_step"], 2)
            for name, run in first["runs"].items():
                self.assertEqual(state_digest(run["model"]), state_digest(resumed["runs"][name]["model"]))
            prepared = prepare_motion_evaluation_inputs(self.data, segment_length=4, device="cuda")
            self.assertEqual(prepared.device, torch.device("cuda", torch.cuda.current_device()))
            features = encode_motion_summaries(next(iter(first["runs"].values()))["model"].view_encoder,
                self.data, prepared_inputs=prepared)
            self.assertTrue(all(value.dtype == np.float32 and np.isfinite(value).all()
                for value in features.values()))


if __name__ == "__main__":
    unittest.main()
