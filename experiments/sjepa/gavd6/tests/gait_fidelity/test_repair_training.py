from copy import deepcopy
from contextlib import redirect_stdout
import io
import json
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch

from gavd6_sjepa.research_directions.gait_fidelity import training, repair_training as repair
from gavd6_sjepa.research_directions.gait_fidelity.common import atomic_json, digest, read_json, sha256
from gavd6_sjepa.research_directions.gait_fidelity.data import fixture_bundle
from gavd6_sjepa.research_directions.gait_fidelity.measurements import measurement_loss
from gavd6_sjepa.research_directions.gait_fidelity.repair_objectives import repair_measurement_terms, angle_gradient_support
from gavd6_sjepa.research_directions.gait_fidelity.response_calibration import calibrate_response


class TrainingTargetsOnly:
    def __init__(self, source, records):
        self.source, self.records = source, records
        self.shape, self.dtype, self.ndim = source.shape, source.dtype, source.ndim

    def __len__(self):
        return len(self.source)

    def __getitem__(self, key):
        row_key = key[0] if isinstance(key, tuple) else key
        indices = np.arange(len(self.source))[row_key]
        if any(self.records[int(i)]["split"] != "train" for i in np.atleast_1d(indices)):
            raise AssertionError("Held-out reference tensors were opened")
        return self.source[key]


class RepairObjectiveTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        bundle = fixture_bundle(samples=128, people=3)
        rows = training.paired_indices(bundle.records)[:2].reshape(-1)
        self.truth = torch.tensor(bundle.targets["xy"][rows]).reshape(2, 2, 128, 12, 2)
        self.valid = torch.tensor(bundle.targets["valid"][rows]).reshape(2, 2, 128, 12)
        rng = torch.Generator().manual_seed(741)
        self.predicted = (self.truth+torch.randn(self.truth.shape, generator=rng)*3).requires_grad_()

    def test_scalar_and_geometry_exactly_reproduce_existing_measurement_loss(self):
        terms = repair_measurement_terms(self.predicted, self.truth, self.valid)
        old, report = measurement_loss(self.predicted, self.truth, self.valid, objective="paired_change")
        torch.testing.assert_close(terms["scalar"]+terms["geometry"], old)
        self.assertEqual(terms["diagnostic"]["supported_pairs"], report["supported_pairs"])
        self.assertEqual(terms["diagnostic"]["short_segment_penalty"], report["short_segment_penalty"])
        sparse = angle_gradient_support(terms)
        self.assertLessEqual(sparse["scalar"]["fraction"], 4/128)
        self.assertGreater(sparse["dense"]["fraction"], .95)

    def test_scalar_permutation_invariance_does_not_hold_for_dense_change(self):
        original = repair_measurement_terms(self.predicted, self.truth, self.valid)
        permuted = repair_measurement_terms(self.predicted.flip(2), self.truth, self.valid)
        torch.testing.assert_close(original["scalar"], permuted["scalar"], rtol=1e-5, atol=1e-7)
        self.assertGreater(abs(float((original["dense"]-permuted["dense"]).detach())), 1e-7)

    def test_invalid_nan_targets_and_coordinate_operands_are_masked_before_arithmetic(self):
        truth = self.truth.clone(); valid = self.valid.clone()
        valid[:, :, 5, 8] = False; truth[:, :, 5, 8] = float("nan")
        terms = repair_measurement_terms(self.predicted, truth, valid)
        (terms["scalar"]+terms["dense"]+terms["geometry"]).backward()
        self.assertTrue(torch.isfinite(self.predicted.grad).all())
        predicted = torch.tensor([[[[1., 2.], [float("nan"), float("nan")]]]], requires_grad=True)
        target = torch.tensor([[[[0., 0.], [float("nan"), float("nan")]]]])
        mask = torch.tensor([[[True, False]]])
        value, _, _ = repair.safe_coordinate_loss(predicted, target, mask)
        value.backward()
        self.assertTrue(torch.isfinite(predicted.grad).all())
        self.assertEqual(float(predicted.grad[0, 0, 1].abs().sum()), 0.)

    def test_short_geometry_support_cannot_be_removed_by_prediction(self):
        bad = self.predicted.detach().clone()
        bad[:, :, :, 8] = bad[:, :, :, 6]
        terms = repair_measurement_terms(bad.requires_grad_(), self.truth, self.valid)
        self.assertEqual(terms["diagnostic"]["supported_pairs"], 2)
        self.assertGreater(float(terms["geometry"].detach()), 0.)
        self.assertGreater(terms["diagnostic"]["predicted_short_segments"], 0)


class RepairTrainingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        cls.temp = tempfile.TemporaryDirectory(prefix="repair-training-")
        cls.root = Path(cls.temp.name)
        cls.bundle = fixture_bundle(samples=16, people=4)
        cls.config = dict(fixture=True, device="cpu",
            model=dict(width=8, encoder_layers=1, predictor_layers=1, heads=2, patch_size=4, window_size=16),
            training=dict(pretraining_updates=2, readout_updates=3, end_to_end_updates=4, batch_size=4,
                learning_rate=.0003, sampling="person_motion", log_every=1, checkpoint_every=1, change_weight=2.),
            measurement=dict(min_frames=8, min_coverage=.75, min_segment_px=1.))
        response = calibrate_response(cls.bundle, cls.config, cls.root/"response-calibration")
        cls.config["response"] = dict(calibration_receipt=response["calibration_receipt"])
        cls.config["repair"] = dict(calibration_batches=3, upstream_bindings={})
        cls.upstreams = {}
        for variant in repair.VARIANTS:
            recipe = dict(encoder="paired_jepa", pretraining_mask="graph_time", representation_variant=variant,
                          readout_or_training_objective="paired_change", recipe_id=variant)
            pre = training.train_phase(cls.bundle, recipe, "pretrain", 17, cls.config, cls.root/(variant+"-pre"))
            checkpoint = Path(pre["checkpoint"])
            payload = torch.load(checkpoint, weights_only=True)
            cls.config["repair"]["upstream_bindings"][variant+"/seed-17"] = dict(
                checkpoint=str(checkpoint), checkpoint_sha256=sha256(checkpoint), signature_sha256=digest(payload["signature"]))
            cls.upstreams[variant] = checkpoint
        cls.upstream = cls.upstreams["jepa_delta_v1"]
        cls.cal = repair.calibrate_repair(cls.bundle, cls.upstream, cls.config, cls.root/"calibration")
        cls.calibration = Path(cls.cal["calibration_receipt"])

    @classmethod
    def tearDownClass(cls):
        cls.temp.cleanup()

    def test_calibration_matches_low_scalar_gradient_and_preserves_rng(self):
        before = torch.get_rng_state().clone()
        receipt = repair.calibrate_repair(self.bundle, self.upstreams["jepa_endpoint_v1"], self.config, self.root/"endpoint-calibration")
        torch.testing.assert_close(before, torch.get_rng_state(), rtol=0, atol=0)
        coeff, grad = receipt["coefficients"], receipt["gradients"]
        self.assertAlmostEqual(coeff["scalar_low"], .2)
        self.assertEqual(coeff["geometry"], 2.)
        self.assertAlmostEqual(coeff["dense_change"]*grad["dense"]["rms"], coeff["scalar_low"]*grad["scalar"]["rms"])
        for batch in receipt["batches"]:
            self.assertTrue(all(self.bundle.records[i]["split"] == "train" for i in batch["endpoint_indices"]))
            self.assertIn("cosine_with_coordinate", batch["gradients"]["dense"])

    def test_frozen_encoder_same_initialization_sampling_and_standard_inference(self):
        results = [repair.train_repair(self.bundle, self.upstream, self.config, self.root/objective,
            objective=objective, seed=17, calibration=self.calibration) for objective in repair.OBJECTIVES]
        a, b = (torch.load(row["checkpoint"], weights_only=True) for row in results)
        self.assertEqual(results[0]["initial_readout_sha256"], results[1]["initial_readout_sha256"])
        self.assertEqual(results[0]["frozen_encoder_sha256"], results[1]["frozen_encoder_sha256"])
        for left, right in zip(a["history"], b["history"]):
            self.assertEqual(left["endpoint_indices"], right["endpoint_indices"])
            self.assertEqual(left["geometry_coefficient"], right["geometry_coefficient"])
            self.assertEqual(left["geometry_coefficient"], 2.)
        self.assertEqual(a["history"][0]["coordinate_loss"], b["history"][0]["coordinate_loss"])
        first = training.load_model(results[0]["checkpoint"])[1]
        self.assertEqual(first["signature"]["representation_variant"], "jepa_delta_v1")
        for row in results:
            prediction = np.load(row["predictions"])
            self.assertTrue(np.isfinite(prediction["xy"]).all())
            self.assertTrue(all(self.bundle.records[i]["split"] == "development" for i in prediction["indices"]))
        recipe = dict(encoder="paired_jepa", pretraining_mask="graph_time", representation_variant="jepa_delta_v1",
                      readout_or_training_objective="paired_change", recipe_id="legacy")
        legacy = training.train_phase(self.bundle, recipe, "readout", 17, self.config, self.root/"legacy-readout", self.upstream)
        old = torch.load(legacy["checkpoint"], weights_only=True)
        self.assertEqual([r["endpoint_indices"] for r in a["history"]], [r["endpoint_indices"] for r in old["history"]])
        self.assertEqual(a["history"][0]["coordinate_loss"], old["history"][0]["coordinate_or_practical_loss"])

    def test_calibration_and_training_do_not_open_development_targets(self):
        protected = SimpleNamespace(inputs=self.bundle.inputs, records=self.bundle.records,
            evidence_status=self.bundle.evidence_status, provenance=self.bundle.provenance,
            targets={k: TrainingTargetsOnly(v, self.bundle.records) for k, v in self.bundle.targets.items()})
        cal = repair.calibrate_repair(protected, self.upstream, self.config, self.root/"protected-calibration")
        result = repair.train_repair(protected, self.upstream, self.config, self.root/"protected-fit",
            objective="dense_change", seed=17, calibration=Path(cal["calibration_receipt"]))
        self.assertEqual(result["status"], "complete")
        altered = SimpleNamespace(**vars(protected)); altered.records = deepcopy(protected.records)
        altered.records[-1]["split"] = "confirmation"
        with self.assertRaisesRegex(PermissionError, "Confirmation"):
            repair.calibrate_repair(altered, self.upstream, self.config, self.root/"forbidden")

    def test_upstream_calibration_and_hyperparameter_tampering_rejected(self):
        changed = deepcopy(self.config); changed["training"]["learning_rate"] *= 2
        with self.assertRaisesRegex(ValueError, "upstream mismatch: training"):
            repair.calibrate_repair(self.bundle, self.upstream, changed, self.root/"wrong-learning")
        with self.assertRaisesRegex(ValueError, "seed mismatch"):
            repair.train_repair(self.bundle, self.upstream, self.config, self.root/"wrong-seed", objective="dense_change", seed=29, calibration=self.calibration)
        changed = deepcopy(self.config)
        changed["repair"]["upstream_bindings"]["jepa_delta_v1/seed-17"]["checkpoint_sha256"] = "0"*64
        with self.assertRaisesRegex(ValueError, "binding mismatch: checkpoint_sha256"):
            repair.calibrate_repair(self.bundle, self.upstream, changed, self.root/"wrong-hash")
        broken = deepcopy(self.cal); broken["coefficients"]["dense_change"] *= 2
        path = self.root/"tampered.json"; atomic_json(path, broken)
        with self.assertRaisesRegex(ValueError, "calibration identity mismatch"):
            repair.train_repair(self.bundle, self.upstream, self.config, self.root/"tampered-fit", objective="dense_change", seed=17, calibration=path)
        broken["identity_sha256"] = digest({k: v for k, v in broken.items() if k != "identity_sha256"})
        atomic_json(path, broken)
        with self.assertRaisesRegex(ValueError, "coefficients differ"):
            repair.train_repair(self.bundle, self.upstream, self.config, self.root/"tampered-fit", objective="dense_change", seed=17, calibration=path)

    def test_resume_is_exact_and_deadline_saves_zero_update_checkpoint(self):
        expected = repair.train_repair(self.bundle, self.upstream, self.config, self.root/"full",
            objective="dense_change", seed=17, calibration=self.calibration)
        original = training._save
        def interrupted(path, payload):
            original(path, payload)
            if payload["updates"] == 1:
                raise RuntimeError("simulated interruption")
        with patch.object(training, "_save", side_effect=interrupted):
            with self.assertRaisesRegex(RuntimeError, "simulated"):
                repair.train_repair(self.bundle, self.upstream, self.config, self.root/"resume",
                    objective="dense_change", seed=17, calibration=self.calibration)
        actual = repair.train_repair(self.bundle, self.upstream, self.config, self.root/"resume",
            objective="dense_change", seed=17, calibration=self.calibration)
        a, b = (torch.load(row["checkpoint"], weights_only=True) for row in (expected, actual))
        self.assertEqual(a["history"], b["history"])
        for key in a["model"]:
            torch.testing.assert_close(a["model"][key], b["model"][key], rtol=0, atol=0)
        with patch.object(repair, "_deadline_reached", return_value=True):
            with self.assertRaises(training.ResponseDeadlineReached):
                repair.train_repair(self.bundle, self.upstream, self.config, self.root/"deadline",
                    objective="dense_change", seed=17, calibration=self.calibration)
        partial = torch.load(self.root/"deadline/checkpoint.pt", weights_only=True)
        self.assertEqual(partial["updates"], 0); self.assertEqual(partial["status"], "interrupted")

    def test_source_requires_cuda_and_timezone_aware_deadline(self):
        changed = deepcopy(self.config); changed["fixture"] = False
        with self.assertRaisesRegex(RuntimeError, "allocated CUDA"):
            repair.calibrate_repair(self.bundle, self.upstream, changed, self.root/"source-cpu")
        with self.assertRaisesRegex(ValueError, "timezone-aware"):
            repair._deadline(dict(fixture=False, repair=dict(deadline_utc="2026-09-25T15:00:00")))
        with self.assertRaisesRegex(ValueError, "explicit deadline"):
            repair._deadline(dict(fixture=False, repair={}))

    def test_prediction_export_is_exact_and_partial_deadline_file_is_not_published(self):
        result = repair.train_repair(self.bundle, self.upstream, self.config, self.root/"export-fit",
            objective="dense_change", seed=17, calibration=self.calibration)
        observed = SimpleNamespace(inputs={key: value[:5] for key, value in self.bundle.inputs.items()})
        expected = training.predict(observed, result["checkpoint"], self.config)
        path = self.root/"guarded-predictions.npy"
        repair.predict_with_deadline(observed, result["checkpoint"], self.config, output=path)
        np.testing.assert_array_equal(np.load(path), expected)
        stopped = self.root/"stopped-predictions.npy"
        with patch.object(repair, "_deadline_reached", return_value=True):
            with self.assertRaises(training.ResponseDeadlineReached):
                repair.predict_with_deadline(observed, result["checkpoint"], self.config, output=stopped)
        self.assertFalse(stopped.exists())
        self.assertTrue(stopped.with_name(stopped.name+".partial.npy").exists())

    def test_live_progress_logs_real_updates_and_does_not_duplicate_completed_resume(self):
        output = self.root/"logged-fit"
        captured = io.StringIO()
        with redirect_stdout(captured):
            repair.train_repair(self.bundle, self.upstream, self.config, output,
                objective="dense_change", seed=17, calibration=self.calibration)
        events = [json.loads(line) for line in (output/"training.jsonl").read_text().splitlines()]
        self.assertEqual([row["update"] for row in events], [1, 2, 3])
        self.assertEqual([json.loads(line) for line in captured.getvalue().splitlines()], events)
        self.assertTrue(all(row["total_updates"] == 3 and row["elapsed_seconds"] >= 0 for row in events))
        self.assertTrue(all("coordinate_loss" in row and "dense_loss" in row and "gradient_was_clipped" in row for row in events))
        retained = (output/"training.jsonl").read_bytes()
        with redirect_stdout(io.StringIO()):
            repair.train_repair(self.bundle, self.upstream, self.config, output,
                objective="dense_change", seed=17, calibration=self.calibration)
        self.assertEqual((output/"training.jsonl").read_bytes(), retained)


if __name__ == "__main__":
    unittest.main()
