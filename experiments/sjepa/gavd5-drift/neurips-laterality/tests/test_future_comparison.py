"""Scientific controls for the separate future-feature decoder experiment."""
from dataclasses import replace
from pathlib import Path
import copy
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from laterality_extensions.forecasting import JOINTS, TimedPose, synthetic_records
from laterality_extensions.future_comparison import (
    ALL_JOINTS, ComparisonForecastSpec, ForecastComparisonModel,
    aggregate_future_error_rows, forecast_error_rows, paired_future_source_interval,
    compatibility_reference, feature_diagnostics, fit_coordinate_decoder,
    fit_forecast_model, future_motion_baselines, load_future_result,
    plan_future_comparison, prepare_future_examples, prepare_prefix,
    run_future_comparison, run_real_future_comparison, save_future_result,
    score_future_predictions, source_roles,
)


class FutureComparisonTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        cls.spec = ComparisonForecastSpec(updates=2)
        cls.records = synthetic_records(sources=6, clips_per_source=1)
        cls.data = prepare_future_examples(cls.records, cls.spec, status="SYNTHETIC TEST")
        sources = sorted(set(cls.data.sources))
        cls.train_sources, cls.test_sources = sources[:4], sources[4:]
        cls.roles = {"train_sources": cls.train_sources, "test_sources": cls.test_sources}
        cls.result = run_future_comparison(cls.data, cls.spec, **cls.roles)

    def test_twelve_and_all_inputs_have_identical_endpoints(self):
        other = prepare_future_examples(self.records, replace(self.spec, input_joints=ALL_JOINTS), status="SYNTHETIC TEST")
        self.assertEqual(other.past.shape[2], 33)
        for name in ("endpoint", "endpoint_valid", "endpoint_times", "horizon", "sequence_ids", "sources"):
            np.testing.assert_array_equal(getattr(self.data, name), getattr(other, name))
        np.testing.assert_array_equal(self.data.past, other.past[:, :, list(JOINTS)])

    def test_future_values_visibility_and_later_timestamps_cannot_change_prefix(self):
        original = self.records[0]
        later = original.times - original.times[0] > self.spec.context_seconds + 1e-10
        xyz, valid, times = original.xyz.copy(), original.valid.copy(), original.times.copy()
        xyz[later], valid[later], times[later] = 1e8, False, times[later] + 0.4
        changed = TimedPose(original.source, original.sequence_id, times, xyz, valid)
        for inputs in (JOINTS, ALL_JOINTS):
            spec = replace(self.spec, input_joints=inputs)
            before, after = prepare_prefix(original, spec), prepare_prefix(changed, spec)
            for field in before:
                np.testing.assert_array_equal(before[field], after[field])
            with torch.random.fork_rng():
                torch.manual_seed(spec.seed)
                model = ForecastComparisonModel(spec).eval()
            predictions = []
            with torch.no_grad():
                for context in (before, after):
                    predictions.append(model.predict(torch.tensor(context["xyz"])[None],
                        torch.tensor(context["valid"])[None], torch.tensor(context["times"])[None], torch.tensor([0.25])))
            torch.testing.assert_close(predictions[0], predictions[1], rtol=0, atol=0)

    def test_outer_future_values_cannot_change_pretraining(self):
        changed = copy.deepcopy(self.data)
        _, held_out = source_roles(changed, **self.roles)
        changed.future[held_out] = 1e6
        changed.future_valid[held_out] = False
        changed.future_times[held_out] = 1e4
        trained = fit_forecast_model(changed, self.spec, **self.roles)
        for name, value in trained["model"].state_dict().items():
            torch.testing.assert_close(value, self.result["matched"]["model"].state_dict()[name], rtol=0, atol=0)

    def test_teacher_has_no_gradient_and_paired_source_schedules_hold(self):
        matched, wrong = self.result["matched"], self.result["mismatched"]
        self.assertEqual(matched["source_schedule"], wrong["source_schedule"])
        for parameter in matched["model"].teacher.parameters():
            self.assertFalse(parameter.requires_grad)
            self.assertIsNone(parameter.grad)
        for original, targets in zip(wrong["source_schedule"], wrong["target_schedule"]):
            self.assertTrue(np.all(self.data.sources[original] != self.data.sources[targets]))
            np.testing.assert_array_equal(self.data.horizon[original], self.data.horizon[targets])
            self.assertTrue(set(self.data.sources[targets]) <= set(self.train_sources))

    def test_initial_weights_match_between_input_choices(self):
        specs = [self.spec, replace(self.spec, input_joints=ALL_JOINTS)]
        models = []
        for spec in specs:
            with torch.random.fork_rng():
                torch.manual_seed(spec.seed)
                models.append(ForecastComparisonModel(spec))
        for name, value in models[0].state_dict().items():
            torch.testing.assert_close(value, models[1].state_dict()[name], rtol=0, atol=0)

    def test_decoder_selection_uses_training_sources_only(self):
        feature = self.data.past.mean((1, 2))
        first = fit_coordinate_decoder(feature, self.data, self.spec, **self.roles)
        changed = copy.deepcopy(self.data)
        _, held_out = source_roles(changed, **self.roles)
        changed.endpoint[held_out] = 1e6
        altered_feature = feature.copy()
        altered_feature[held_out] = -1e6
        second = fit_coordinate_decoder(altered_feature, changed, self.spec, **self.roles)
        for field in ("mean", "scale", "coefficients", "intercepts", "selected_alphas"):
            np.testing.assert_array_equal(getattr(first, field), getattr(second, field))
        self.assertEqual(set(first.fitted_sources), set(self.train_sources))
        for joint in first.validation_sources:
            for split in joint.get("splits", []):
                self.assertFalse(set(split["fit_sources"]) & set(split["validation_sources"]))
                self.assertFalse(set(split["validation_sources"]) & set(self.test_sources))

    def test_simple_readout_recovers_a_known_synthetic_signal(self):
        data = copy.deepcopy(self.data)
        rng = np.random.default_rng(911)
        feature = rng.normal(size=(len(data.past), 2))
        endpoint = np.column_stack((2 * feature[:, 0], -feature[:, 1], feature.sum(1)))
        data.endpoint = np.repeat(endpoint[:, None], len(JOINTS), axis=1)
        data.endpoint_valid[:] = True
        spec = replace(self.spec, ridge_alphas=(1e-6,))
        decoder = fit_coordinate_decoder(feature, data, spec, **self.roles)
        _, test = source_roles(data, **self.roles)
        error = np.square(decoder.predict(feature, data.horizon)[test] - data.endpoint[test]).mean()
        self.assertLess(error, 1e-8)

    def test_observed_and_predicted_features_use_the_same_decoder(self):
        self.assertIn("Matched future-feature decoder", self.result["decoders"])
        self.assertIn("Matched: observed future features decoded (diagnostic)", self.result["predictions"])
        self.assertIn("Matched: predicted future features decoded", self.result["predictions"])
        self.assertEqual(len(self.result["scores"]), 33)
        self.assertTrue(np.isfinite(self.result["scores"].source_balanced_rmse).all())

    def test_constant_feature_diagnostic_is_explicit(self):
        diagnosis = feature_diagnostics(np.ones((8, 6)))
        self.assertEqual(diagnosis["effective_rank"], 0)
        self.assertIn("Nearly constant", diagnosis["status"])

    def test_motion_baselines_do_not_read_future_values_or_measurement_times(self):
        changed = copy.deepcopy(self.data)
        changed.endpoint_times[:] = 1e9
        changed.future[:] = 1e6
        changed.future_times[:] = 1e6
        for first, second in zip(future_motion_baselines(self.data), future_motion_baselines(changed)):
            np.testing.assert_array_equal(first, second)

    def test_coverage_reports_unavailable_predictions(self):
        first, second = future_motion_baselines(self.data)
        _, test = source_roles(self.data, **self.roles)
        second[test[0], 0] = np.nan
        scores, coverage = score_future_predictions({"one": first, "two": second}, self.data, test)
        self.assertEqual(int(coverage.query("method == 'two'").unavailable_predictions.sum()), 1)
        for _, group in scores.groupby("horizon_seconds"):
            self.assertEqual(group.common_endpoints.nunique(), 1)
        with self.assertRaises(ValueError):
            score_future_predictions({"one": first}, self.data, np.r_[test, test[:1]])

    def test_source_overlap_and_duplicate_examples_fail(self):
        with self.assertRaises(ValueError):
            source_roles(self.data, self.train_sources, self.train_sources)
        with self.assertRaises(ValueError):
            prepare_future_examples(self.records + self.records[:1], self.spec, status="SYNTHETIC TEST")

    def test_all_joint_tiny_training_executes(self):
        spec = replace(self.spec, input_joints=ALL_JOINTS, updates=1)
        data = prepare_future_examples(self.records, spec, status="SYNTHETIC TEST")
        result = run_future_comparison(data, spec, **self.roles)
        self.assertTrue(np.isfinite(result["scores"].source_balanced_rmse).all())

    def test_save_reuse_corruption_and_overwrite_controls(self):
        reference = compatibility_reference(self.data, self.spec, **self.roles)
        with tempfile.TemporaryDirectory() as temporary:
            destination = Path(temporary) / "new_result"
            save_future_result(destination, self.result, self.data, reference)
            loaded = load_future_result(destination, reference)
            self.assertEqual(len(loaded["scores"]), 33)
            np.testing.assert_array_equal(loaded["arrays"]["target"], self.data.endpoint[self.result["test_indices"]])
            states = torch.load(destination / "checkpoints.pt", weights_only=True)
            self.assertTrue(any(key.startswith("matched.teacher.") for key in states))
            self.assertTrue(any(key.startswith("matched.encoder.") for key in states))
            with self.assertRaises(FileExistsError):
                save_future_result(destination, self.result, self.data, reference)
            with self.assertRaises(ValueError):
                load_future_result(destination, {**reference, "data": "incompatible"})
            with (destination / "predictions.npz").open("ab") as stream:
                stream.write(b"corrupted")
            with self.assertRaises(ValueError):
                load_future_result(destination, reference)

    def test_disabled_real_path_only_displays_the_plan(self):
        plan = plan_future_comparison()
        self.assertEqual(plan["model_fits"], 50)
        self.assertEqual(plan["optimizer_updates"], 60000)
        self.assertEqual(plan["recipe"]["updates"], 1200)
        with patch("laterality_extensions.future_comparison.load_future_records") as loader, patch("builtins.print"):
            result = run_real_future_comparison(enabled=False)
        loader.assert_not_called()
        self.assertIn("Training disabled", result["status"])
        with self.assertRaises(ValueError):
            plan_future_comparison(folds=(0, 0))

    def test_real_input_validation_does_not_train(self):
        splits = {0: self.roles}
        with patch("laterality_extensions.future_comparison.load_future_records", return_value=(self.records, splits)), \
             patch("laterality_extensions.future_comparison.fit_forecast_model") as trainer, patch("builtins.print"):
            result = run_real_future_comparison(enabled=False, validate_inputs=True,
                                                spec=self.spec, folds=(0,), seeds=(42,))
        trainer.assert_not_called()
        self.assertEqual(result["status"], "Inputs validated; training disabled")

    def test_forecasting_aggregation_validates_coverage_and_pools_predictions(self):
        test = self.result["test_indices"]
        rows = forecast_error_rows(
            {name: value[test] for name, value in self.result["predictions"].items()},
            sources=self.data.sources[test], sequence_ids=self.data.sequence_ids[test],
            horizon=self.data.horizon[test], target=self.data.endpoint[test], valid=self.data.endpoint_valid[test],
            seed=self.spec.seed, fold=0, input_landmarks=12,
        )
        expected = rows[["sequence_id", "source_id", "fold", "horizon_seconds"]].drop_duplicates()
        declaration = {"seeds": (self.spec.seed,), "input_sizes": (12,), "methods": tuple(self.result["predictions"])}
        summary = aggregate_future_error_rows(rows, expected, **declaration)
        actual = summary["per_seed"].sort_values(["method", "horizon_seconds"]).source_balanced_rmse.to_numpy()
        original = self.result["scores"].sort_values(["method", "horizon_seconds"]).source_balanced_rmse.to_numpy()
        np.testing.assert_allclose(actual, original)
        with self.assertRaises(ValueError):
            aggregate_future_error_rows(rows.iloc[1:], expected, **declaration)
        with self.assertRaises(ValueError):
            aggregate_future_error_rows(__import__("pandas").concat([rows, rows.iloc[:1]]), expected, **declaration)
        inconsistent = rows.copy()
        inconsistent.loc[0, "fold"] = 1
        with self.assertRaises(ValueError):
            aggregate_future_error_rows(inconsistent, expected, **declaration)
        inconsistent = rows.copy()
        inconsistent.loc[0, "coverage_reference"] = "different endpoint set"
        with self.assertRaises(ValueError):
            aggregate_future_error_rows(inconsistent, expected, **declaration)
        interval = paired_future_source_interval(
            rows, expected, first="Matched: predicted future features decoded",
            reference="Last observed position", input_landmarks=12,
            horizon_seconds=0.25, seeds=(self.spec.seed,), repetitions=20,
        )
        self.assertEqual(interval["source_videos"], 2)
        self.assertTrue(np.isfinite(interval["rmse_difference"]))
        self.assertIn("conditional on fitted models", interval["scope"])


if __name__ == "__main__":
    unittest.main()
