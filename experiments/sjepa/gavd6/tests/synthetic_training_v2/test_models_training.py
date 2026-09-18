"""Contract tests: these tiny CPU fits are software evidence, never source results."""
from dataclasses import replace
from pathlib import Path
import tempfile
import unittest

import numpy as np
import torch

from gavd6_sjepa.research_directions.synthetic_training_v2.models import (
    ARMS, Body12Encoder, ModelConfig, RestorationModel,
)
from gavd6_sjepa.research_directions.synthetic_training_v2.training import (
    TrainConfig, coordinate_loss, smoothnet_loss, fit_arm, load_fitted, predict,
)


def example():
    rng = np.random.default_rng(8)
    xy = rng.normal(size=(4, 8, 12, 2)).astype(np.float32) * .2
    observed = np.ones(xy.shape[:-1], bool)
    observed[:, :, 10] = False
    targets = {"xy": xy.copy() + .025, "valid": np.ones_like(observed)}
    xy[~observed] = np.nan
    inputs = {"xy": xy, "observed": observed, "confidence": observed.astype(np.float32) * .8,
              "timestamps": np.broadcast_to(np.arange(8, dtype=np.float32) / 25, (4, 8)).copy()}
    return inputs, targets


def tensors(inputs):
    return {k: torch.from_numpy(v) for k, v in inputs.items()}


class ModelTrainingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def setUp(self):
        self.model_cfg = ModelConfig(width=16, encoder_layers=1, predictor_layers=1,
                                     heads=2, patch_size=2, window_size=8)
        self.train_cfg = TrainConfig(updates=2, readout_updates=2, batch_size=2,
                                     learning_rate=1e-3, log_every=2, checkpoint_every=1)
        self.inputs, self.targets = example()

    def test_missing_input_valid_target_and_target_isolation(self):
        model = RestorationModel("direct", self.model_cfg).eval()
        inputs = tensors(self.inputs)
        result = model(inputs)
        self.assertTrue(torch.isfinite(result).all())
        valid = torch.from_numpy(self.targets["valid"])
        target = torch.from_numpy(self.targets["xy"])
        missing_support = valid & ~inputs["observed"]
        loss, _, supported = coordinate_loss(result, target, missing_support)
        self.assertTrue(supported.all())
        loss.backward()
        self.assertGreater(float(model.readout.network[-1].bias.grad.abs().sum()), 0)
        target[:] = 1000
        valid[:] = False
        torch.testing.assert_close(result, model(inputs), rtol=0, atol=0)
        with self.assertRaisesRegex(ValueError, "Inference accepts only"):
            model(dict(inputs, target_valid=valid))

    def test_reference_acceleration_penalizes_attenuation_and_masks_missing_triples(self):
        times = torch.arange(8)[None].float() / 25
        target = (times.square()[:, :, None, None]).expand(1, 8, 12, 2).clone()
        predicted = (target * .5).requires_grad_()
        valid = torch.ones(1, 8, 12, dtype=torch.bool)
        positional = smoothnet_loss(predicted, target, valid, times, 0)[0]
        full, _, _, diagnostics = smoothnet_loss(predicted, target, valid, times, .1)
        self.assertGreater(float(full), float(positional))
        self.assertAlmostEqual(diagnostics["reference_acceleration_l1"], 1., places=4)
        valid[:, 2:4, 0] = False
        target[~valid] = float("nan")
        masked, _, supported, diagnostics = smoothnet_loss(predicted, target, valid, times, .1)
        self.assertTrue(torch.isfinite(masked))
        self.assertTrue(supported.all())
        self.assertEqual(diagnostics["acceleration_triple_count"], 68)
        masked.backward()
        self.assertTrue(torch.isfinite(predicted.grad).all())

    def test_missing_garbage_and_artificial_hidden_coordinates_do_not_leak(self):
        encoder = Body12Encoder(self.model_cfg).eval()
        inputs = tensors(self.inputs)
        altered = dict(inputs, xy=inputs["xy"].clone())
        altered["xy"][~inputs["observed"]] = 1e8
        torch.testing.assert_close(encoder(inputs), encoder(altered), rtol=0, atol=0)
        hidden = torch.zeros(4, 4, 12, dtype=torch.bool)
        hidden[:, 1, :3] = True
        altered["xy"][:, 2:4, :3] = 1e6
        torch.testing.assert_close(encoder(inputs, hidden), encoder(altered, hidden), rtol=0, atol=0)
        self.assertGreater(float((encoder(inputs) - encoder(altered)).abs().max()), 1e-5)

    def test_static_changes_only_with_current_coordinates_or_shared_auxiliary(self):
        model = RestorationModel("static", self.model_cfg).eval()
        torch.nn.init.normal_(model.practical.network[-1].weight)
        inputs = tensors(self.inputs)
        expected = model(inputs)
        altered = dict(inputs, xy=inputs["xy"].clone())
        altered["xy"][:, 4, :10] += 30
        actual = model(altered)
        keep = [0, 1, 2, 3, 5, 6, 7]
        torch.testing.assert_close(actual[:, keep], expected[:, keep], rtol=0, atol=0)
        self.assertGreater(float((actual[:, 4] - expected[:, 4]).abs().max()), .1)
        changed_aux = dict(inputs, confidence=inputs["confidence"].clone())
        changed_aux["confidence"][:, 4] *= .5
        self.assertGreater(float((model(changed_aux)[:, 0] - expected[:, 0]).abs().max()), 1e-6)

    def test_teacher_has_no_gradient_and_ema_initial_snapshot_is_immutable(self):
        model = RestorationModel("paired_jepa", self.model_cfg)
        initial = {k: v.clone() for k, v in model.initialized_encoder.state_dict().items()}
        before = {k: v.clone() for k, v in model.teacher.state_dict().items()}
        with torch.no_grad():
            for p in model.encoder.parameters():
                p.add_(.3)
        model.update_teacher(.75)
        for key, value in model.teacher.state_dict().items():
            torch.testing.assert_close(value, before[key] * .75 + model.encoder.state_dict()[key] * .25)
        for key, value in model.initialized_encoder.state_dict().items():
            torch.testing.assert_close(value, initial[key], rtol=0, atol=0)
        model(tensors(self.inputs)).square().mean().backward()
        self.assertTrue(all(p.grad is None for p in model.teacher.parameters()))
        self.assertTrue(all(p.grad is None for p in model.initialized_encoder.parameters()))

    def test_all_arms_fit_and_emit_finite_missing_predictions(self):
        with tempfile.TemporaryDirectory() as directory:
            for arm in ARMS:
                with self.subTest(arm=arm):
                    cfg = replace(self.train_cfg, updates=1, readout_updates=1)
                    donors = np.array([1, 0, 3, 2]) if arm == "shuffled_jepa" else None
                    model = fit_arm(arm, self.inputs, self.targets, self.model_cfg, cfg,
                                    Path(directory) / arm, {"kind": "fixture"}, donor_indices=donors,
                                    roles=["train"] * 4)
                    result = predict(model, self.inputs, batch_size=3)
                    self.assertEqual(result.shape, self.inputs["xy"].shape)
                    self.assertTrue(np.isfinite(result).all())
                    self.assertEqual(model.training_report["status"], "complete")
                    self.assertTrue((Path(directory) / arm / "checkpoint-000000.pt").exists())
                    restored = load_fitted(Path(directory) / arm / "model.pt", identity={"kind": "fixture"})
                    np.testing.assert_allclose(predict(restored, self.inputs, batch_size=3), result, rtol=0, atol=0)

    def test_exact_resume_across_pretraining_and_readout(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            cfg = replace(self.train_cfg, updates=3, readout_updates=2)
            expected = fit_arm("paired_jepa", self.inputs, self.targets, self.model_cfg,
                               cfg, root / "full", {"fixture": 1})
            initial = fit_arm("paired_jepa", self.inputs, self.targets, self.model_cfg,
                              replace(cfg, stop_after_updates=2), root / "resume", {"fixture": 1})
            self.assertEqual(initial.training_report["status"], "interrupted")
            resumed = fit_arm("paired_jepa", self.inputs, self.targets, self.model_cfg,
                replace(cfg, resume_from=str(root / "resume/checkpoint-000002.pt")), root / "resume", {"fixture": 1})
            for key, value in expected.state_dict().items():
                torch.testing.assert_close(value, resumed.state_dict()[key], rtol=0, atol=0)
            self.assertEqual(expected.training_report["history"], resumed.training_report["history"])
            self.assertEqual(expected.training_report["draws"], resumed.training_report["draws"])

    def test_coordinate_and_paired_queries_are_matched_including_missing_targets(self):
        with tempfile.TemporaryDirectory() as directory:
            fits = {}
            for arm in ("coordinate", "paired_jepa", "ordinary_jepa"):
                fits[arm] = fit_arm(arm, self.inputs, self.targets, self.model_cfg, self.train_cfg,
                                    Path(directory) / arm, {"fixture": "query_matching"})
            a, b, ordinary = [fits[k].training_report["draws"][0]
                              for k in ("coordinate", "paired_jepa", "ordinary_jepa")]
            for key in ("indices", "mask_sha256", "query_sha256", "target_support"):
                self.assertEqual(a[key], b[key])
            self.assertGreater(a["missing_input_query_tokens"], 0)
            self.assertAlmostEqual(a["hidden_observed_token_fraction"], .5)
            self.assertGreater(sum(a["target_support"]), sum(ordinary["target_support"]))
            self.assertEqual(a["query_sha256"], ordinary["query_sha256"])

    def test_practical_seconds_budget_uses_actual_timer(self):
        with tempfile.TemporaryDirectory() as directory:
            cfg = replace(self.train_cfg, end_to_end_updates=10000, total_seconds_budget=.05)
            model = fit_arm("smoothnet", self.inputs, self.targets, self.model_cfg, cfg,
                            directory, {"fixture": "timer"})
            report = model.training_report
            self.assertEqual(report["termination"], "time_budget_reached")
            self.assertGreater(report["optimizer_updates"], 0)
            self.assertLess(report["optimizer_updates"], 10000)
            self.assertGreaterEqual(report["elapsed_seconds"], .05)
            self.assertIsNotNone(report["time_budget_overrun_seconds"])
            self.assertTrue((Path(directory) / "model.pt").exists())

    def test_resume_rejects_stale_data_and_protocol(self):
        with tempfile.TemporaryDirectory() as directory:
            cfg = replace(self.train_cfg, stop_after_updates=1)
            fit_arm("coordinate", self.inputs, self.targets, self.model_cfg, cfg, directory, {"v": 1})
            resume = replace(self.train_cfg, resume_from=str(Path(directory) / "checkpoint-000001.pt"))
            altered = dict(self.inputs, confidence=self.inputs["confidence"].copy())
            altered["confidence"][0, 0, 0] = .1
            with self.assertRaisesRegex(ValueError, "Incompatible resume"):
                fit_arm("coordinate", altered, self.targets, self.model_cfg, resume, directory, {"v": 1})
            with self.assertRaisesRegex(ValueError, "Incompatible resume"):
                fit_arm("coordinate", self.inputs, self.targets, self.model_cfg, resume, directory, {"v": 2})

    def test_no_test_role_or_self_pair_or_unsupported_success(self):
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(ValueError, "train-role"):
                fit_arm("direct", self.inputs, self.targets, self.model_cfg, self.train_cfg,
                        directory, {"fixture": 1}, roles=["train", "train", "confirmation", "train"])
            with self.assertRaisesRegex(ValueError, "same-window"):
                fit_arm("shuffled_jepa", self.inputs, self.targets, self.model_cfg, self.train_cfg,
                        directory, {"fixture": 1}, donor_indices=np.arange(4))
            target = dict(self.targets, valid=np.zeros_like(self.targets["valid"]))
            with self.assertRaisesRegex(ValueError, "unsupported batches"):
                fit_arm("direct", self.inputs, target, self.model_cfg, self.train_cfg,
                        directory, {"fixture": 1})
            self.assertFalse((Path(directory) / "model.pt").exists())

    def test_schema_and_clock_errors_are_rejected(self):
        model = RestorationModel("direct", self.model_cfg)
        inputs = tensors(self.inputs)
        with self.assertRaisesRegex(ValueError, "12,2"):
            model(dict(inputs, xy=inputs["xy"][:, :, :11]))
        with self.assertRaisesRegex(ValueError, "strictly increasing"):
            model(dict(inputs, timestamps=inputs["timestamps"].flip(1)))
        changed = dict(inputs, timestamps=inputs["timestamps"] * 2)
        encoder = model.encoder
        self.assertGreater(float((encoder(inputs) - encoder(changed)).abs().max()), 1e-6)


if __name__ == "__main__":
    unittest.main()
