"""Small synthetic arithmetic, leakage, gradient and restart tests; no GAVD IO."""
from __future__ import annotations

import copy
from dataclasses import dataclass
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest

import numpy as np
import torch

from gavd6_sjepa.research_directions.temporal_gait.masking import (
    SourceBoutSampler, control_target_indices, patch_validity, sample_feature_mask,
)
from gavd6_sjepa.research_directions.temporal_gait.models import JEPA
from gavd6_sjepa.research_directions.temporal_gait.objectives import (
    feature_diagnostics, per_example_mean, predictive_loss,
)
from gavd6_sjepa.research_directions.temporal_gait.training import (
    context_batch, load_model, train_condition,
)


@dataclass
class FixtureDataset:
    arrays: dict
    records: list

    def __len__(self):
        return len(self.records)


def config(**overrides):
    values = dict(mode="synthetic", run_root="unused", device="cpu", seeds=[42], pilot_seeds=[42],
                  updates=3, checkpoint_updates=[0, 1, 2, 3], batch_size=2, hidden_dim=16,
                  encoder_depth=1, predictor_depth=1, heads=4, patch_size=4, clock_channels=False,
                  learning_rate=.001, weight_decay=0., ema_start=.999, ema_end=.999,
                  mask_fraction=.5, prefix_seconds=2.56, grid_hz=25., horizons=[.25, .5, 1.],
                  target_interval_seconds=.08, precision="fp32", objective="centered_ce_v1",
                  student_temperature=.1, teacher_temperature=.06, center_momentum=.9,
                  vicreg_weight=.05, view_translation=.02, lr_schedule="constant",
                  resume_from=None, stop_after_updates=None)
    values.update(overrides)
    return SimpleNamespace(**values)


def fixture():
    rng = np.random.default_rng(7)
    n, t, j, h = 8, 64, 33, 3
    query = np.arange(-64, 0, dtype=np.float32) / 25
    coordinates = rng.normal(0, .1, (n, t, j, 2)).astype(np.float32)
    coordinates[..., 0] += np.sin(np.arange(t)[None, :, None] / 7).astype(np.float32)
    valid = np.ones((n, t, j), bool)
    valid[0, :20, 2:8] = False
    valid[3, :, 10] = False
    coordinates[~valid] = 0
    arrays = {"context": coordinates, "context_valid": valid,
              "context_times": np.broadcast_to(query[None, :, None], (n, t, j)).copy(),
              "context_age": np.zeros((n, t, j), np.float32),
              "query_times": np.broadcast_to(query[None], (n, t)).copy(),
              "future": rng.normal(0, .2, (n, h, 2, j, 2)).astype(np.float32),
              "future_valid": np.ones((n, h, 2, j), bool)}
    arrays["future_valid"][2, 0] = False
    arrays["future"][~arrays["future_valid"]] = 0
    for key in ("context", "context_valid", "context_times", "context_age", "query_times"):
        arrays["index_" + key] = arrays[key].copy()
    # Different supports must not change the matched target-mask positions.
    arrays["index_context_valid"][0, 20:24, 2:8] = False
    arrays["index_context"][~arrays["index_context_valid"]] = 0
    records = [dict(window_id=f"w{i}", video_id=f"v{i // 4}", group_id=f"g{i // 4}",
                    sequence_id=f"s{i // 4}", issue_time=10. * (i % 4 + 1), role="train")
               for i in range(n)]
    return FixtureDataset(arrays, records)


def task(arm="masked"):
    return dict(task_id="fixture-42", arm=arm, seed=42, fold=0)


class TrainingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.threads)

    def test_ragged_loss_is_equal_example_and_has_correct_gradient(self):
        value = torch.tensor([[1., 900., 100.], [4., 8., 99.]], requires_grad=True)
        flags = torch.tensor([[True, False, False], [True, True, False]])
        loss, per_example, support = per_example_mean(value, flags)
        self.assertEqual(float(loss.detach()), 3.5)
        torch.testing.assert_close(per_example, torch.tensor([1., 6.]))
        loss.backward()
        torch.testing.assert_close(value.grad, torch.tensor([[.5, 0., 0.], [.25, .25, 0.]]))
        self.assertTrue(support.all())
        unsupported, per_example, support = per_example_mean(value, torch.zeros_like(flags))
        self.assertEqual(float(unsupported.detach()), 0.)
        self.assertTrue(torch.isnan(per_example).all())
        self.assertFalse(support.any())

    def test_partial_patch_padding_and_ragged_mask(self):
        valid = torch.ones(2, 2, 33, dtype=torch.bool)
        valid[0, :, 1:] = False
        patched = patch_validity(valid, 4)
        self.assertEqual(tuple(patched.shape), (2, 1, 33))
        mask = sample_feature_mask(patched, .5, np.random.default_rng(4))
        self.assertEqual(int(mask[0].sum()), 0)
        self.assertEqual(int(mask[1].sum()), 16)
        self.assertFalse((mask & ~patched).any())

    def test_hidden_features_have_zero_effect_and_zero_gradient(self):
        cfg = config(clock_channels=True)
        model = JEPA(cfg).eval()
        context = context_batch(fixture(), [0, 1])
        context["context"].requires_grad_()
        hidden = sample_feature_mask(patch_validity(context["context_valid"], 4), .5,
                                     np.random.default_rng(1))
        prediction, _, valid = model.masked_prediction(context, hidden)
        changed = {key: value.clone() for key, value in context.items()}
        frames = hidden.repeat_interleave(4, 1)
        changed["context"] = torch.where(frames[..., None], 1000., changed["context"])
        changed["context_times"] = torch.where(frames, -1000., changed["context_times"])
        changed["context_age"] = torch.where(frames, 1000., changed["context_age"])
        altered, _, _ = model.masked_prediction(changed, hidden)
        torch.testing.assert_close(prediction, altered, rtol=0, atol=0)
        prediction[valid].square().mean().backward()
        self.assertEqual(float(context["context"].grad[frames].abs().sum()), 0.)
        self.assertGreater(float(context["context"].grad[~frames].abs().sum()), 0.)
        self.assertTrue(all(p.grad is None for p in model.teacher.parameters()))

    def test_future_api_boundary_and_independent_teacher_intervals(self):
        cfg, data = config(), fixture()
        model = JEPA(cfg).eval()
        context = context_batch(data, [0, 1])
        prediction = model.predict_future(context)
        self.assertEqual(tuple(prediction.shape), (2, 3, 16))
        future = torch.as_tensor(data.arrays["future"][:2]).clone()
        valid = torch.as_tensor(data.arrays["future_valid"][:2])
        encoded = model.encode_future_targets(future, valid)
        future[:, 1] += 50
        changed = model.encode_future_targets(future, valid)
        torch.testing.assert_close(encoded[:, 0], changed[:, 0], rtol=0, atol=0)
        torch.testing.assert_close(encoded[:, 2], changed[:, 2], rtol=0, atol=0)
        self.assertFalse(torch.allclose(encoded[:, 1], changed[:, 1]))
        torch.testing.assert_close(prediction, model.predict_future(context), rtol=0, atol=0)
        with self.assertRaisesRegex(ValueError, "Context-only"):
            model.predict_future(dict(context, future=future))
        context["context_times"][0, 0, 0] = .001
        with self.assertRaisesRegex(ValueError, "boundary"):
            model.encode(context)

    def test_all_invalid_is_explicit_and_objective_stops_teacher_gradient(self):
        data, cfg = fixture(), config()
        context = context_batch(data, [0, 1])
        context["context_valid"][:] = False
        model = JEPA(cfg)
        torch.testing.assert_close(model.encode(context), torch.zeros(2, 16))
        self.assertTrue(torch.isfinite(model.predict_future(context)).all())
        predicted = torch.randn(2, 3, 16, requires_grad=True)
        teacher = torch.randn(2, 3, 16, requires_grad=True)
        for objective in ("centered_ce_v1", "feature_regression_v1"):
            loss, _, _ = predictive_loss(predicted, teacher, torch.ones(2, 3, dtype=torch.bool),
                                         objective=objective, center=torch.zeros(16))
            loss.backward()
            self.assertIsNone(teacher.grad)
        self.assertEqual(feature_diagnostics(torch.ones(8, 16))["effective_rank"], 0.)

    def test_sampler_balances_sources_not_window_counts_and_controls(self):
        records = fixture().records
        records = records[:1] + records[4:] * 20
        sampler = SourceBoutSampler(records, np.random.default_rng(1))
        samples = sampler.draw(4000)
        proportion = np.mean([records[i]["video_id"] == "v0" for i in samples])
        self.assertAlmostEqual(proportion, .5, delta=.03)
        records = fixture().records
        indices = list(range(8))
        wrong_source = control_target_indices(records, indices, "future_wrong_source",
                                               np.random.default_rng(2), 3.64)
        wrong_time = control_target_indices(records, indices, "future_wrong_time",
                                             np.random.default_rng(2), 3.64)
        for i, other in enumerate(wrong_source):
            self.assertNotEqual(records[i]["group_id"], records[other]["group_id"])
        for i, other in enumerate(wrong_time):
            self.assertEqual(i, other)
        with self.assertRaises(ValueError):
            control_target_indices(records[:1], [0], "future_wrong_source", np.random.default_rng(1), 3.64)

    def test_train_load_all_arms_and_matched_mask_positions(self):
        with tempfile.TemporaryDirectory() as temporary:
            records = {}
            for arm in ("masked", "masked_index", "future", "future_wrong_source", "future_wrong_time"):
                cfg = config(updates=1, checkpoint_updates=[0, 1])
                row = train_condition(cfg, task(arm), fixture(), Path(temporary) / arm)
                self.assertEqual(row["status"], "trained")
                self.assertGreater(row["usage"]["wall_seconds_this_invocation"], 0)
                self.assertEqual(row["usage"]["optimizer_updates_this_invocation"], 1)
                self.assertIsNone(row["usage"]["gpu_hours"])
                self.assertIsNone(row["usage"]["cuda_peak_allocated_bytes"])
                model = load_model(cfg, row["checkpoint"])
                context = context_batch(fixture(), [0, 1], arm=arm)
                self.assertEqual(tuple(model.encode(context).shape), (2, 16))
                self.assertTrue(any(not torch.equal(a, b) for a, b in
                                    zip(model.online.parameters(), model.initialized.parameters())))
                self.assertTrue(all(not p.requires_grad for p in model.teacher.parameters()))
                records[arm] = json.loads(Path(row["draws"]).read_text())
                self.assertEqual(row["unique_training_draws"]["windows"],
                                 len(set(records[arm][0]["window_ids"])))
                self.assertEqual(sum(records[arm][0]["valid_target_counts_per_example"]),
                                 records[arm][0]["target_tokens"])
                if arm.startswith("masked"):
                    draw = records[arm][0]
                    self.assertTrue(all(0 <= v <= 1 for v in draw["realized_whole_body_mask_fraction"]))
                    self.assertTrue(all(v is None or 0 <= v <= 2
                                        for v in draw["visible_temporal_neighbors_per_target"]))
            for key in ("mask_sha256", "window_ids", "matched_target_candidates"):
                self.assertEqual(records["masked"][0][key], records["masked_index"][0][key])
            wrong_order = records["future_wrong_time"][0]
            self.assertEqual(wrong_order["target_window_ids"], records["future"][0]["target_window_ids"])
            self.assertEqual(wrong_order["target_horizon_indices"], [2, 0, 1])
            self.assertEqual(wrong_order["target_tokens"], records["future"][0]["target_tokens"])

    def test_exact_resume_and_incompatible_data_or_schedule_rejected(self):
        data = fixture()
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            cfg = config(updates=4, checkpoint_updates=[0, 2, 4], lr_schedule="warmup_cosine")
            full = train_condition(cfg, task("future"), data, root / "full")
            interrupted_cfg = config(**vars(cfg))
            interrupted_cfg.stop_after_updates = 2
            first = train_condition(interrupted_cfg, task("future"), data, root / "resume")
            self.assertEqual(first["status"], "interrupted")
            resumed_cfg = config(**vars(cfg))
            resumed_cfg.resume_from = first["checkpoint"]
            final = train_condition(resumed_cfg, task("future"), data, root / "resume")
            self.assertEqual(final["usage"]["optimizer_updates_this_invocation"], 2)
            self.assertEqual(final["usage"]["optimizer_updates_total"], 4)
            a, b = load_model(cfg, full["checkpoint"]), load_model(cfg, final["checkpoint"])
            for name, value in a.state_dict().items():
                torch.testing.assert_close(value, b.state_dict()[name], rtol=0, atol=0, msg=name)
            self.assertEqual(Path(full["draws"]).read_text(), Path(final["draws"]).read_text())
            self.assertEqual(Path(full["history"]).read_text(), Path(final["history"]).read_text())
            bad_data = copy.deepcopy(data)
            bad_data.arrays["context"][0, 0, 0, 0] += .001
            with self.assertRaisesRegex(ValueError, "Incompatible resume"):
                train_condition(resumed_cfg, task("future"), bad_data, root / "bad-data")
            resumed_cfg.updates = 5
            with self.assertRaisesRegex(ValueError, "Incompatible resume"):
                train_condition(resumed_cfg, task("future"), data, root / "bad-schedule")


if __name__ == "__main__":
    unittest.main()
