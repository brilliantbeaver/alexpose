"""Hand-checkable observability, source estimand, and frozen-readout tests."""
from __future__ import annotations

import copy
from dataclasses import dataclass
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch

from gavd6_sjepa.research_directions.temporal_gait.config import RunConfig
from gavd6_sjepa.research_directions.temporal_gait import evaluation as ev
from gavd6_sjepa.research_directions.temporal_gait.statistics import (
    aggregate_window_errors, paired_group_bootstrap, primary_window_errors,
    secondary_motion_errors,
)
from gavd6_sjepa.research_directions.temporal_gait.training import train_condition


@dataclass
class Fixture:
    arrays: dict
    records: list

    def __len__(self):
        return len(self.records)


def fixture(role="train", prefix="fit", videos=4):
    n, t, j, h = videos * 2, 64, 33, 3
    query = np.arange(-64, 0, dtype=np.float32) / 25
    times = np.broadcast_to(query[None, :, None], (n, t, j)).copy()
    slope = .1 + np.arange(n, dtype=np.float32) * .01
    offset = np.arange(n, dtype=np.float32) * .1
    context = np.zeros((n, t, j, 2), np.float32)
    context[..., 0] = offset[:, None, None] + slope[:, None, None] * times + np.arange(j)[None, None] * .01
    context[..., 1] = -.3 + .5 * slope[:, None, None] * times
    endpoint = np.zeros((n, h, j, 2), np.float32)
    future = np.zeros((n, h, 2, j, 2), np.float32)
    for hi, horizon in enumerate((.25, .5, 1.)):
        endpoint[:, hi, :, 0] = offset[:, None] + slope[:, None] * horizon + np.arange(j)[None] * .01
        endpoint[:, hi, :, 1] = -.3 + .5 * slope[:, None] * horizon
        for ti, target in enumerate((horizon - .04, horizon)):
            future[:, hi, ti, :, 0] = offset[:, None] + slope[:, None] * target + np.arange(j)[None] * .01
            future[:, hi, ti, :, 1] = -.3 + .5 * slope[:, None] * target
    arrays = {"context": context, "context_valid": np.ones((n, t, j), bool),
              "context_times": times, "context_age": np.zeros((n, t, j), np.float32),
              "query_times": np.broadcast_to(query[None], (n, t)).copy(),
              "future": future, "future_valid": np.ones((n, h, 2, j), bool),
              "endpoint": endpoint, "endpoint_valid": np.ones((n, h, j), bool),
              "endpoint_times": np.broadcast_to(np.array([.25, .5, 1.])[None, :, None], (n, h, j)).copy() + 10,
              "scale_valid": np.ones(n, bool), "scale": np.ones(n), "origin": np.zeros((n, 2))}
    for key in ev.CONTEXT_KEYS:
        arrays["index_" + key] = arrays[key].copy()
    records = [dict(window_id=f"{prefix}-w{i}", video_id=f"{prefix}-v{i // 2}",
                    sequence_id=f"{prefix}-b{i // 2}", group_id=f"{prefix}-g{i // 2}",
                    role=role, issue_time=10. + i % 2, bout_start=0., bout_end=15.,
                    mode="synthetic", scale_reason="valid") for i in range(n)]
    return Fixture(arrays, records)


def config(root, **updates):
    values = dict(mode="synthetic", run_root=str(root), device="cpu", seeds=[42], pilot_seeds=[42],
                  development_seeds=[42], updates=1, checkpoint_updates=[0, 1],
                  batch_size=2, hidden_dim=16, encoder_depth=1, predictor_depth=1,
                  direct_updates=2, bootstrap_samples=100, ridge_alphas=[.1, 10.],
                  arms=["future"])
    values.update(updates)
    return RunConfig(**values)


class EvaluationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.threads)

    def test_primary_pairs_scale_and_missing_prediction_are_not_zero_error(self):
        y = np.zeros((3, 1, 33, 2))
        p = np.broadcast_to([3., 4.], y.shape).copy()
        valid = np.ones((3, 1, 33), bool)
        valid[1, 0, [26, 28]] = False
        measured = primary_window_errors(p, y, valid, np.array([True, True, False]))
        self.assertEqual(measured["errors"][0, 0], 5.)
        self.assertTrue(np.isnan(measured["errors"][1:]).all())
        self.assertEqual(int(measured["joint_support"].sum()), 8)
        p[0, 0, 25] = np.nan
        missing = primary_window_errors(p, y, valid, np.array([True, True, False]))
        self.assertTrue(missing["eligible"][0, 0])
        self.assertFalse(missing["prediction_complete"][0, 0])
        rows = [dict(window_id=str(i), video_id="v", sequence_id="b", group_id="g") for i in range(3)]
        summarized = aggregate_window_errors(rows, missing, [.5])
        self.assertIsNone(summarized["summary"][0]["score"])
        self.assertEqual(summarized["summary"][0]["status"], "missing_predictions")

    def test_joint_window_bout_video_order_not_pooled_windows(self):
        records = [dict(window_id=str(i), video_id=v, sequence_id=b, group_id=g) for i, (v, b, g) in
                   enumerate([("a", "a1", "g1"), ("a", "a1", "g1"), ("a", "a2", "g1"),
                              ("b", "b1", "g1"), ("c", "c1", "g2")])]
        measured = dict(errors=np.array([[0.], [2.], [10.], [20.], [30.]]),
                        eligible=np.ones((5, 1), bool), prediction_complete=np.ones((5, 1), bool))
        result = aggregate_window_errors(records, measured, [.5])
        self.assertEqual(result["sources"][0]["score"], 5.5)
        self.assertEqual(result["summary"][0]["score"], 18.5)
        self.assertNotEqual(result["summary"][0]["score"], measured["errors"].mean())

    def test_root_translation_and_articulation_are_separate_secondaries(self):
        target = np.zeros((2, 1, 33, 2))
        target[..., 0] = np.arange(33)
        target[..., 1] = np.arange(33) * .5
        prediction = target + [3., 4.]
        valid = np.ones((2, 1, 33), bool)
        scale = np.ones(2, bool)
        primary = primary_window_errors(prediction, target, valid, scale)
        secondary = secondary_motion_errors(prediction, target, valid, scale)
        np.testing.assert_array_equal(primary["errors"], 5.)
        np.testing.assert_array_equal(secondary["root_displacement_2d"]["errors"], 5.)
        np.testing.assert_array_equal(secondary["root_relative_lower_limb_2d"]["errors"], 0.)
        valid[1, 0, 23] = False
        missing_hip = secondary_motion_errors(prediction, target, valid, scale)
        for measured in missing_hip.values():
            self.assertFalse(measured["eligible"][1, 0])
            self.assertTrue(np.isnan(measured["errors"][1, 0]))
            self.assertFalse(measured["joint_support"][1, 0].any())
        after = primary_window_errors(prediction, target, valid, scale)
        np.testing.assert_array_equal(after["joint_support"], primary["joint_support"])
        np.testing.assert_array_equal(after["errors"], primary["errors"])
        # An observed hip with a missing prediction is model failure, not an
        # excuse to remove that otherwise eligible secondary observation.
        prediction[0, 0, 24] = np.nan
        incomplete = secondary_motion_errors(prediction, target, valid, scale)
        for measured in incomplete.values():
            self.assertTrue(measured["eligible"][0, 0])
            self.assertFalse(measured["prediction_complete"][0, 0])
            self.assertTrue(np.isnan(measured["errors"][0, 0]))

    def test_group_bootstrap_keeps_videos_and_seeds_together(self):
        first, reference = [], []
        for seed in (1, 2):
            for video, group, score in (("a", "g1", 0.), ("b", "g1", 10.), ("c", "g2", 100.)):
                first.append(dict(seed=seed, video_id=video, group_id=group, score=score))
                reference.append(dict(seed=seed, video_id=video, group_id=group, score=101.))
        result = paired_group_bootstrap(first, reference, repetitions=400, seed=9)
        self.assertAlmostEqual(result["first_mean"], 110 / 3)
        self.assertAlmostEqual(result["difference"], 110 / 3 - 101)
        self.assertEqual(result["lower_95"], -96.)
        self.assertEqual(result["upper_95"], -1.)
        self.assertEqual(result["seed_sd_difference"], 0.)
        self.assertEqual(len(result["bootstrap_differences"]), 400)
        with self.assertRaisesRegex(ValueError, "identical seeds and videos"):
            paired_group_bootstrap(first[:-1], reference, repetitions=10)

    def test_velocity_uses_actual_time_and_forecasts_ignore_future(self):
        data = fixture(videos=1)
        query = data.arrays["query_times"]
        # Last observation is one bin older than its held display query.
        actual = np.floor((query - .04) / .08) * .08
        data.arrays["context_times"][:] = actual[:, :, None]
        data.arrays["context_age"][:] = (query - actual)[:, :, None]
        data.arrays["context"][..., 0] = 1 + 2 * actual[:, :, None]
        data.arrays["context"][..., 1] = 3 - actual[:, :, None]
        first = ev.kinematic_predictions(data, [.25, .5, 1.])
        np.testing.assert_allclose(first["robust_velocity"][:, 1, :, 0], 2., atol=2e-6)
        np.testing.assert_allclose(first["robust_velocity"][:, 1, :, 1], 2.5, atol=2e-6)
        data.arrays["endpoint"] += 100
        data.arrays["future"] -= 200
        data.arrays["endpoint_valid"][:] = False
        second = ev.kinematic_predictions(data, [.25, .5, 1.])
        for method in first:
            np.testing.assert_array_equal(first[method], second[method])

    def test_causal_and_role_contamination_fails_before_fit(self):
        train, dev = fixture(), fixture("development", "dev", videos=2)
        dev.records[0]["group_id"] = train.records[0]["group_id"]
        with tempfile.TemporaryDirectory() as folder, patch.object(ev, "_fit_baselines", side_effect=AssertionError("fit called")):
            cfg = config(folder)
            with self.assertRaisesRegex(ValueError, "contamination"):
                ev.evaluate_baselines(cfg, train, dev, Path(folder) / "out")
            test = fixture("test", "test", videos=2)
            with self.assertRaisesRegex(ValueError, "frozen fitted_path"):
                ev.evaluate_baselines(cfg, train, test, Path(folder) / "out")
        bad = fixture()
        bad.arrays["context_times"][0, -1, 0] = .01
        with self.assertRaisesRegex(ValueError, "Future observation"):
            ev.kinematic_predictions(bad, [.5])

    def test_baseline_freeze_replay_and_prefix_only_mean_fallback(self):
        train, dev = fixture(), fixture("development", "dev", videos=2)
        # A primary joint has an observed future but no prefix; the frozen
        # train mean supplies a forecast without consulting its future support.
        dev.arrays["context_valid"][0, :, 25] = False
        with tempfile.TemporaryDirectory() as folder:
            cfg = config(folder)
            record = ev.evaluate_baselines(cfg, train, dev, Path(folder) / "dev")
            fitted = ev._unpack_fitted(record["fitted_path"])
            self.assertEqual(fitted["train_role"], "train")
            self.assertEqual(fitted["baselines"]["direct_mlp"]["kind"], "mlp")
            self.assertEqual(len(fitted["baselines"]["direct_mlp"]["training"]["loss_trace"]), 2)
            self.assertIn("inner held groups remain SSL-train-visible", fitted["baselines"]["raw_pose_ridge"]["selection"]["ssl_exposure"])
            with np.load(record["predictions_path"], allow_pickle=False) as stored:
                self.assertTrue(stored["persistence_prefix_mean_fallback"][0, :, 25].all())
                np.testing.assert_allclose(stored["persistence"][0, :, 25], fitted["baselines"]["static_train_mean"][:, 25])
                self.assertTrue(np.isfinite(stored["persistence"]).all())
                for metric in ("root_displacement_2d", "root_relative_lower_limb_2d"):
                    self.assertEqual(stored[f"secondary__{metric}__persistence__errors"].shape, (len(dev), 3))
                    self.assertEqual(stored[f"secondary__{metric}__persistence__joint_support"].shape, (len(dev), 3, 33))
            scores = json.loads(Path(record["scores_path"]).read_text())
            for metric, values in scores["secondary"].items():
                self.assertNotIn(metric, record["methods"])
                self.assertFalse(values["used_for_selection_or_gate"])
                self.assertEqual(len(values["sources"]), len(ev.BASELINES) * 3 * 2)
            test = fixture("test", "test", videos=2)
            with patch.object(ev, "_fit_baselines", side_effect=AssertionError("forbidden test fit")):
                final = ev.evaluate_baselines(cfg, None, test, Path(folder) / "test", fitted_path=record["fitted_path"])
            self.assertTrue(final["fitted_reused"])
            self.assertEqual(final["fitted_sha256"], record["fitted_sha256"])
            with self.assertRaisesRegex(FileExistsError, "new/empty"):
                ev.evaluate_baselines(cfg, None, test, Path(folder) / "test", fitted_path=record["fitted_path"])

    def test_checkpoint_readouts_replay_without_future_leakage(self):
        train, dev = fixture(), fixture("development", "dev", videos=2)
        task = dict(task_id="future-42", arm="future", seed=42, fold=0)
        with tempfile.TemporaryDirectory() as folder:
            cfg = config(folder)
            checkpoint = train_condition(cfg, task, train, Path(folder) / "model")
            record = ev.evaluate_condition(cfg, task, train, dev, checkpoint["checkpoint"], Path(folder) / "dev")
            test = fixture("test", "test", videos=2)
            changed = copy.deepcopy(test)
            changed.arrays["endpoint"] += 5
            changed.arrays["future"] += 10
            with patch.object(ev, "_fit_baselines", side_effect=AssertionError("forbidden test fit")), \
                 patch.object(ev, "_fit_feature_decoder", side_effect=AssertionError("forbidden decoder fit")):
                a = ev.evaluate_condition(cfg, task, None, test, checkpoint["checkpoint"], Path(folder) / "test-a", fitted_path=record["fitted_path"])
                b = ev.evaluate_condition(cfg, task, None, changed, checkpoint["checkpoint"], Path(folder) / "test-b", fitted_path=record["fitted_path"])
            with np.load(a["predictions_path"], allow_pickle=False) as first, np.load(b["predictions_path"], allow_pickle=False) as second:
                for method in record["methods"]:
                    if method != "observed_future_teacher_ridge":
                        np.testing.assert_array_equal(first[method], second[method])
                self.assertFalse(np.array_equal(first["observed_future_teacher_ridge"], second["observed_future_teacher_ridge"]))
            self.assertEqual(record["task_id"], task["task_id"])
            self.assertEqual(record["fold"], 0)

    def test_unobserved_secondary_decoder_is_not_estimable_and_roundtrips(self):
        train = fixture()
        features = np.random.default_rng(20).normal(size=(len(train), 3, 4)).astype(np.float32)
        with tempfile.TemporaryDirectory() as folder:
            cfg = config(folder)
            complete = ev._fit_feature_decoder(cfg, features, train, 42)
            legacy = copy.deepcopy(complete)
            legacy.pop("horizon_support")
            for readout in legacy["readouts"]:
                readout["selection"].pop("support")
            np.testing.assert_array_equal(ev._predict_feature_decoder(features, legacy),
                                          ev._predict_feature_decoder(features, complete))
            train.arrays["endpoint_valid"][:, 2] = False
            train.arrays["future_valid"][:, 2] = False
            shared = ev._fit_feature_decoder(cfg, features[:, 1], train, 42)
            self.assertTrue(np.isnan(ev._predict_feature_decoder(features[:, 1], shared)[:, 2]).all())
            fitted = ev._fit_feature_decoder(cfg, features, train, 42)
            unsupported = fitted["readouts"][2]
            self.assertEqual(unsupported["kind"], "not_estimable")
            self.assertEqual(unsupported["selection"]["selection_horizon_seconds"], 1.)
            self.assertIsNone(unsupported["selection"]["selected_alpha"])
            self.assertEqual(unsupported["selection"]["support"]["observed_joint_endpoints"], 0)
            self.assertEqual(fitted["horizon_support"][2]["status"], "not_estimable")
            prediction = ev._predict_feature_decoder(features, fitted)
            self.assertTrue(np.isnan(prediction[:, 2]).all())
            np.testing.assert_array_equal(prediction[:, :2], ev._predict_feature_decoder(features, complete)[:, :2])
            changed = features.copy()
            changed[:, 2] = np.nan  # An unsupported slot cannot feed another horizon.
            np.testing.assert_array_equal(ev._predict_feature_decoder(changed, fitted), prediction)
            path = Path(folder) / "decoder.json"
            ev._pack_fitted(path, fitted)
            with patch.object(ev, "_fit_ridge", side_effect=AssertionError("forbidden replay fit")):
                restored = ev._unpack_fitted(path)
                np.testing.assert_array_equal(ev._predict_feature_decoder(features, restored), prediction)
            self.assertEqual(restored["horizon_support"], fitted["horizon_support"])

    def test_partial_secondary_support_fits_only_observed_joints(self):
        train = fixture()
        train.arrays["endpoint_valid"][:, 2, [25, 26]] = False
        features = np.random.default_rng(21).normal(size=(len(train), 3, 4)).astype(np.float32)
        with tempfile.TemporaryDirectory() as folder:
            cfg = config(folder)
            fitted = ev._fit_feature_decoder(cfg, features, train, 42)
            readout = fitted["readouts"][2]
            self.assertEqual(readout["kind"], "ridge")
            self.assertEqual(readout["selection"]["selection_horizon_seconds"], 1.)
            self.assertEqual(readout["selection"]["support"]["eligible_training_windows"], len(train))
            prediction = ev._predict_feature_decoder(features, fitted)
            self.assertTrue(np.isnan(prediction[:, 2, [25, 26]]).all())
            self.assertTrue(np.isfinite(prediction[:, 2, 27:33]).all())
            self.assertEqual(fitted["horizon_support"][2]["fitted_joint_count"], 31)
            self.assertEqual(fitted["horizon_support"][2]["status"], "partial_joint_support")

    def test_secondary_inner_holdout_absence_is_not_borrowed_but_primary_still_fails(self):
        train = fixture()
        groups = sorted({r["group_id"] for r in train.records})
        held_group = groups[np.random.default_rng(42).permutation(len(groups))[0]]
        held = np.array([r["group_id"] == held_group for r in train.records])
        features = np.random.default_rng(22).normal(size=(len(train), 3, 4)).astype(np.float32)
        with tempfile.TemporaryDirectory() as folder:
            cfg = config(folder)
            train.arrays["endpoint_valid"][held, 2] = False
            fitted = ev._fit_feature_decoder(cfg, features, train, 42)
            readout = fitted["readouts"][2]
            self.assertEqual(readout["kind"], "not_estimable")
            self.assertGreater(readout["selection"]["support"]["observed_joint_endpoints"], 0)
            self.assertEqual(readout["selection"]["support"]["eligible_inner_validation_windows"], 0)
            self.assertTrue(np.isnan(ev._predict_feature_decoder(features, fitted)[:, 2]).all())
            for selected in (held, np.ones(len(train), bool)):
                no_primary = copy.deepcopy(train)
                no_primary.arrays["endpoint_valid"][selected, 1] = False
                with self.assertRaisesRegex(ValueError, "no complete observed primary support"):
                    ev._fit_feature_decoder(cfg, features, no_primary, 42)
                with self.assertRaisesRegex(ValueError, "no complete observed primary support"):
                    ev._fit_feature_decoder(cfg, features[:, 1], no_primary, 42)

    def test_checkpoint_readouts_with_unobserved_secondary_reuse_nan_predictions(self):
        train, dev = fixture(), fixture("development", "dev", videos=2)
        for dataset in (train, dev):
            dataset.arrays["endpoint_valid"][:, 2] = False
            dataset.arrays["future_valid"][:, 2] = False
            dataset.arrays["future"][:, 2] = 0
        task = dict(task_id="future-42", arm="future", seed=42, fold=0)
        with tempfile.TemporaryDirectory() as folder:
            cfg = config(folder)
            checkpoint = train_condition(cfg, task, train, Path(folder) / "model")
            record = ev.evaluate_condition(cfg, task, train, dev, checkpoint["checkpoint"], Path(folder) / "dev")
            metadata = json.loads(Path(record["records_path"]).read_text())
            self.assertEqual(metadata["decoder_horizon_support"], record["decoder_horizon_support"])
            with np.load(record["predictions_path"], allow_pickle=False) as predictions:
                for method, support in record["decoder_horizon_support"].items():
                    self.assertEqual(support[2]["status"], "not_estimable")
                    self.assertTrue(np.isnan(predictions[method][:, 2]).all())
                    self.assertTrue(np.isfinite(predictions[method][:, 1]).all())
            # Test observations cannot retrospectively make the missing training
            # horizon estimable: the frozen readout still emits NaNs there.
            test = fixture("test", "test", videos=2)
            with patch.object(ev, "_fit_feature_decoder", side_effect=AssertionError("forbidden test refit")):
                final = ev.evaluate_condition(cfg, task, None, test, checkpoint["checkpoint"],
                    Path(folder) / "test", fitted_path=record["fitted_path"])
            self.assertEqual(final["fitted_sha256"], record["fitted_sha256"])
            with np.load(final["predictions_path"], allow_pickle=False) as predictions:
                for method in record["decoder_horizon_support"]:
                    self.assertTrue(np.isnan(predictions[method][:, 2]).all())
            summary = next(row for row in final["summary"] if row["method"] == "online_future_ridge"
                           and row["horizon_seconds"] == 1.)
            self.assertEqual(summary["status"], "missing_predictions")
            self.assertIsNone(summary["score"])

    def _fake_grid(self, folder, *, role="development"):
        methods = list(ev.BASELINES) + [state + "_" + space + "_ridge" for state in ev.STATES for space in ("context", "future")] + ["observed_future_teacher_ridge"]
        result = []
        for seed in (42, 43):
            path = Path(folder) / f"{role}-{seed}.json"
            sources = []
            for method in methods:
                value = .8 if method == "online_context_ridge" else .9 if method.startswith(("online_", "teacher_")) else 1.
                if method == "observed_future_teacher_ridge":
                    value = 0.
                for index in range(2):
                    sources.append(dict(method=method, seed=seed, horizon_seconds=.5,
                        video_id=f"v{index}", group_id=f"g{index}", score=value,
                        eligible_windows=2, scored_windows=2, missing_predictions=0, scored_bouts=1))
            ev._json(path, {"sources": sources, "summary": []})
            result.append(dict(role=role, arm="masked", seed=seed, mode="synthetic", scores_path=str(path),
                support_sha256="identical-target-support", checkpoint_path=f"checkpoint-{seed}",
                fitted_path=f"fitted-{seed}", fitted_reused=role == "test", artifact_path=str(path)))
        return result

    def test_selection_requires_complete_grid_and_never_selects_on_test(self):
        with tempfile.TemporaryDirectory() as folder:
            cfg = config(folder, seeds=[42, 43], development_seeds=[42, 43])
            records = self._fake_grid(folder)
            tasks = [dict(arm="masked", seed=s) for s in (42, 43)]
            with self.assertRaisesRegex(ValueError, "Explicit expected_tasks"):
                ev.compare_conditions(cfg, records, Path(folder) / "absent")
            with self.assertRaisesRegex(ValueError, "Incomplete/unexpected"):
                ev.compare_conditions(cfg, records[:1], Path(folder) / "missing", expected_tasks=tasks)
            selected = ev.compare_conditions(cfg, records, Path(folder) / "select", expected_tasks=tasks)
            self.assertEqual(selected["selected_method"], "online_context_ridge")
            self.assertEqual(selected["statistical_status"], "success")
            self.assertFalse(selected["ready_for_expansion"])
            self.assertEqual(selected["status"], "synthetic_only")
            test = self._fake_grid(folder, role="test")
            with self.assertRaisesRegex(ValueError, "development-only"):
                ev.compare_conditions(cfg, test, Path(folder) / "forbidden", expected_tasks=tasks)
            final = ev.summarize_locked_conditions(cfg, test, selected, Path(folder) / "final")
            self.assertEqual(final["selected_method"], selected["selected_method"])
            self.assertEqual(final["selection_scope"], "frozen development selection; no test refit/reselection")

    def test_gate_effect_classification(self):
        with tempfile.TemporaryDirectory() as folder:
            cfg = config(folder)
            base = [dict(seed=42, video_id=str(i), group_id=str(i), score=1.) for i in range(3)]
            for score, expected in ((.9, "success"), (.99, "subthreshold"), (1.1, "failure")):
                first = [{**r, "score": score} for r in base]
                self.assertEqual(ev._contrast(cfg, first, base)["status"], expected)
            first = [{**r, "score": v} for r, v in zip(base, [.3, .3, 2.])]
            self.assertEqual(ev._contrast(cfg, first, base)["status"], "inconclusive")


if __name__ == "__main__":
    unittest.main()
