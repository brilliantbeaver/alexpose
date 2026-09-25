from copy import deepcopy
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch

from gavd6_sjepa.research_directions.gait_fidelity import training
from gavd6_sjepa.research_directions.gait_fidelity.common import read_json
from gavd6_sjepa.research_directions.gait_fidelity.data import fixture_bundle
from gavd6_sjepa.research_directions.gait_fidelity.response_calibration import (
    calibrate_response, prepare_response_batch, response_forward_losses,
    verified_response_identity, response_data_identity)
from gavd6_sjepa.research_directions.gait_fidelity.response_objectives import (
    centered_logit_errors, latent_response_loss, coordinate_response_loss,
    predictive_diagnostics, response_support)
from gavd6_sjepa.research_directions.synthetic_training_v2.models import ModelConfig, RestorationModel
from gavd6_sjepa.research_directions.temporal_gait.objectives import predictive_loss


class ResponseObjectiveTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        self.cfg = ModelConfig(width=8, encoder_layers=1, predictor_layers=1, heads=2, patch_size=4, window_size=8)
        self.queries = torch.ones(4, 2, 12, dtype=torch.bool)
        self.valid = torch.ones(4, 8, 12, dtype=torch.bool)

    def test_algebra_channel_shift_and_detached_teacher(self):
        torch.manual_seed(31)
        p = torch.randn(4, 24, 8, requires_grad=True)
        t = torch.randn_like(p, requires_grad=True)
        center = torch.randn(8, requires_grad=True)
        self.valid[0, 1, 0] = False
        self.queries[2, 1, 4] = False
        support = response_support(self.queries, self.valid, self.cfg)
        self.assertFalse(support[0, 0]); self.assertFalse(support[1, 16])
        self.assertEqual(int(support.sum()), 46)
        delta, diagnostic = latent_response_loss(p, t, self.queries, self.valid, self.cfg,
            variant="jepa_delta_v1", center=center)
        endpoint, _ = latent_response_loss(p, t, self.queries, self.valid, self.cfg,
            variant="jepa_endpoint_v1", center=center)
        a, b = centered_logit_errors(p, t, center).reshape(2, 2, 24, 8).unbind(1)
        cross = (a * b).mean(-1)
        expected_cross = ((cross * support).sum(1) / support.sum(1)).mean()
        torch.testing.assert_close(delta, endpoint - expected_cross)
        shifted, _ = latent_response_loss(p + torch.randn(4, 24, 1), t, self.queries, self.valid, self.cfg,
            variant="jepa_delta_v1", center=center)
        torch.testing.assert_close(delta, shifted)
        delta.backward()
        self.assertIsNone(t.grad); self.assertIsNone(center.grad)
        self.assertGreater(float(p.grad.abs().sum()), 0)
        self.assertEqual(diagnostic["supported_pairs"], 2)

    def test_coordinate_common_scale_and_nan_exclusion(self):
        scales = torch.tensor([2., 6., 3., 9.])
        pixel_error = torch.ones(4, 8, 12, 2)
        pixel_error[1] *= 3; pixel_error[3] *= 5
        target = torch.zeros_like(pixel_error)
        predicted = (pixel_error / scales[:, None, None, None]).requires_grad_()
        loss, _ = coordinate_response_loss(predicted, target, self.queries, self.valid, scales, self.cfg)
        expected = .5 * (((3 - 1) / 4) ** 2 / 2 + ((5 - 1) / 6) ** 2 / 2)
        self.assertAlmostEqual(float(loss.detach()), expected, places=6)
        self.valid[0, 0, 0] = False
        target[0, 0, 0] = float("nan")
        loss, report = coordinate_response_loss(predicted, target, self.queries, self.valid, scales, self.cfg)
        loss.backward()
        self.assertTrue(torch.isfinite(predicted.grad).all())
        self.assertEqual(report["supported_tokens"], 47)

    def test_unsupported_auxiliary_is_differentiable_and_explicit(self):
        p = torch.randn(4, 24, 8, requires_grad=True)
        loss, diagnostic = latent_response_loss(p, torch.randn_like(p), self.queries & False,
            self.valid, self.cfg, variant="jepa_endpoint_v1", center=torch.zeros(8))
        self.assertEqual(float(loss), 0)
        self.assertIsNone(diagnostic["loss"])
        self.assertEqual(diagnostic["unsupported_pairs"], 2)
        loss.backward()
        self.assertEqual(float(p.grad.abs().sum()), 0)

    def test_entropy_kl_use_exact_ce_probabilities_and_equal_example_reduction(self):
        torch.manual_seed(11)
        p, t = torch.randn(2, 5, 8), torch.randn(2, 5, 8)
        valid = torch.ones(2, 5, dtype=torch.bool); valid[0, 1:] = False
        center = torch.randn(8)
        report = predictive_diagnostics(p, t, valid, center=center)
        loss = predictive_loss(p, t, valid, objective="centered_ce_v1", center=center)[0]
        self.assertAlmostEqual(float(loss), report["predictive_ce"], places=5)
        self.assertAlmostEqual(report["predictive_ce"], report["teacher_entropy"] + report["predictive_kl"], places=5)


class ResponseTrainingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        cls.temp = tempfile.TemporaryDirectory(prefix="response-training-")
        cls.root = Path(cls.temp.name)
        cls.bundle = fixture_bundle(samples=16, people=4)
        cls.config = dict(fixture=True, device="cpu",
            model=dict(width=8, encoder_layers=1, predictor_layers=1, heads=2, patch_size=4, window_size=16),
            training=dict(pretraining_updates=3, readout_updates=2, end_to_end_updates=4,
                batch_size=4, learning_rate=.0003, sampling="person_motion", log_every=1, checkpoint_every=1),
            measurement=dict(min_frames=8, min_coverage=.75, min_segment_px=1.))
        before = torch.get_rng_state().clone()
        cls.receipt = calibrate_response(cls.bundle, cls.config, cls.root / "calibration")
        torch.testing.assert_close(before, torch.get_rng_state(), rtol=0, atol=0)
        cls.config["response"] = dict(calibration_receipt=cls.receipt["calibration_receipt"])

    @classmethod
    def tearDownClass(cls):
        cls.temp.cleanup()

    def recipe(self, variant):
        return dict(encoder="coordinate" if variant == "coordinate_delta_v1" else "paired_jepa",
                    pretraining_mask="graph_time", representation_variant=variant,
                    readout_or_training_objective="paired_change", recipe_id=variant)

    def test_calibration_shared_coefficient_rng_and_timing_identity(self):
        r = self.receipt
        self.assertEqual(len(r["batches"]), 32)
        self.assertEqual(r["coefficients"]["jepa_delta_v1"], r["coefficients"]["jepa_endpoint_v1"])
        maximum = max(r["gradients"][k]["rms"] for k in ("jepa_delta_v1", "jepa_endpoint_v1"))
        self.assertAlmostEqual(r["coefficients"]["jepa_delta_v1"] * maximum / r["gradients"]["jepa_base"]["rms"], .1)
        shortened = deepcopy(self.config); shortened["training"]["pretraining_updates"] = 1
        with self.assertRaisesRegex(ValueError, "schedule"):
            verified_response_identity(shortened, "jepa_delta_v1", r["data_sha256"])
        shortened["response"]["timing_probe"] = True
        self.assertTrue(verified_response_identity(shortened, "jepa_delta_v1", r["data_sha256"])["timing_probe"])
        shortened["training"]["mask_fraction"] = .4
        with self.assertRaisesRegex(ValueError, "protocol"):
            verified_response_identity(shortened, "jepa_delta_v1", r["data_sha256"])
        with self.assertRaisesRegex(ValueError, "data_sha256"):
            verified_response_identity(self.config, "jepa_delta_v1", "wrong")

    def test_three_variants_train_and_fresh_readouts_freeze(self):
        for variant in ("jepa_delta_v1", "jepa_endpoint_v1", "coordinate_delta_v1"):
            with self.subTest(variant=variant):
                recipe = self.recipe(variant)
                pre = training.train_phase(self.bundle, recipe, "pretrain", 17, self.config, self.root / (variant + "-pre"))
                history = read_json(self.root / (variant + "-pre") / "history.json")
                self.assertGreater(history[-1]["response_auxiliary"]["supported_tokens"], 0)
                self.assertIn("response_gradients", history[-1])
                self.assertIn("gradient_was_clipped", history[-1])
                if variant != "coordinate_delta_v1":
                    self.assertAlmostEqual(history[-1]["predictive_ce"], history[-1]["teacher_entropy"] + history[-1]["predictive_kl"], places=4)
                final = training.train_phase(self.bundle, recipe, "readout", 17, self.config,
                    self.root / (variant + "-readout"), Path(pre["checkpoint"]))
                self.assertTrue(final["frozen_encoder_unchanged"])
                payload = torch.load(final["checkpoint"], weights_only=True)
                self.assertEqual(payload["signature"]["representation_variant"], variant)
                base_recipe = dict(recipe, readout_or_training_objective="base", recipe_id=variant+"-base")
                base = training.train_phase(self.bundle, base_recipe, "readout", 17, self.config,
                    self.root / (variant + "-base-readout"), Path(pre["checkpoint"]))
                self.assertTrue(base["frozen_encoder_unchanged"])
                base_payload = torch.load(base["checkpoint"], weights_only=True)
                self.assertEqual(final["frozen_encoder_sha256"], base["frozen_encoder_sha256"])
                self.assertEqual(payload["signature"]["upstream_sha256"], base_payload["signature"]["upstream_sha256"])
        with self.assertRaisesRegex(ValueError, "representation_variant"):
            training.train_phase(self.bundle, self.recipe("jepa_endpoint_v1"), "readout", 17,
                self.config, self.root / "wrong-parent", self.root / "jepa_delta_v1-pre/checkpoint.pt")

    def test_interrupted_resume_matches_uninterrupted_state_and_history(self):
        recipe = self.recipe("jepa_delta_v1")
        expected = training.train_phase(self.bundle, recipe, "pretrain", 29, self.config, self.root / "full")
        original = training._save
        def interrupted(path, payload):
            original(path, payload)
            if payload["updates"] == 1:
                raise RuntimeError("simulated worker interruption")
        with patch.object(training, "_save", side_effect=interrupted):
            with self.assertRaisesRegex(RuntimeError, "simulated worker"):
                training.train_phase(self.bundle, recipe, "pretrain", 29, self.config, self.root / "resumed")
        partial = torch.load(self.root / "resumed/checkpoint.pt", weights_only=True)
        self.assertEqual(partial["status"], "interrupted")
        actual = training.train_phase(self.bundle, recipe, "pretrain", 29, self.config, self.root / "resumed")
        a, b = (torch.load(row["checkpoint"], weights_only=True) for row in (expected, actual))
        self.assertEqual(a["history"], b["history"])
        for key in a["model"]:
            torch.testing.assert_close(a["model"][key], b["model"][key], rtol=0, atol=0)

    def test_missing_aux_support_retains_supported_base_loss(self):
        rows = training.paired_indices(self.bundle.records)[0]
        cfg = ModelConfig(**self.config["model"])
        hidden = np.zeros((2, 4, 12), bool)
        hidden[0, 0] = True; hidden[1, 1] = True
        batch = prepare_response_batch(self.bundle, rows, hidden, cfg)
        # All fixture joints in this clear cell are observed; disjoint queries give no aux support.
        batch["queries"] = torch.as_tensor(hidden)
        model = RestorationModel("paired_jepa", cfg)
        result = response_forward_losses(model, batch, self.config["training"], "jepa_delta_v1")
        self.assertTrue(result["supported"].all())
        self.assertGreater(float(result["base_loss"]), 0)
        self.assertEqual(result["extra"]["response_auxiliary"]["supported_pairs"], 0)

    def test_first_update_base_forward_and_exposure_match_legacy_exactly(self):
        recipe = self.recipe("jepa_delta_v1")
        legacy = {k: v for k, v in recipe.items() if k != "representation_variant"}
        training.train_phase(self.bundle, legacy, "pretrain", 43, self.config, self.root / "legacy-parity")
        training.train_phase(self.bundle, recipe, "pretrain", 43, self.config, self.root / "response-parity")
        a, b = (read_json(self.root / name / "history.json")[0] for name in ("legacy-parity", "response-parity"))
        for key in ("predictive_loss", "vicreg_loss", "endpoint_indices", "target_indices", "mask", "normalization_fallbacks"):
            self.assertEqual(a[key], b[key], key)
        payload = torch.load(self.root / "legacy-parity/checkpoint.pt", weights_only=True)
        self.assertNotIn("representation_variant", payload["signature"])
        self.assertNotIn("response", payload["signature"])

    def test_calibration_rejects_zero_gradient_energy_and_restores_rng(self):
        before = torch.get_rng_state().clone()
        with patch("gavd6_sjepa.research_directions.gait_fidelity.response_calibration.gradient_statistics",
                   return_value=dict(squared_l2=0., l2=0., rms=0., parameters=1)):
            with self.assertRaisesRegex(FloatingPointError, "Zero or nonfinite"):
                calibrate_response(self.bundle, self.config, self.root / "invalid-calibration")
        torch.testing.assert_close(before, torch.get_rng_state(), rtol=0, atol=0)
        self.assertFalse((self.root / "invalid-calibration/calibration.json").exists())

    def test_source_deadline_saves_interrupted_state_before_any_update(self):
        cfg = deepcopy(self.config)
        cfg["fixture"] = False
        cfg["response"]["deadline_utc"] = "2000-01-01T00:00:00Z"
        with self.assertRaises(training.ResponseDeadlineReached):
            training.train_phase(self.bundle, self.recipe("jepa_delta_v1"), "pretrain", 17,
                                 cfg, self.root / "deadline")
        payload = torch.load(self.root / "deadline/checkpoint.pt", weights_only=True)
        self.assertEqual(payload["status"], "interrupted")
        self.assertEqual(payload["updates"], 0)


if __name__ == "__main__":
    unittest.main()
