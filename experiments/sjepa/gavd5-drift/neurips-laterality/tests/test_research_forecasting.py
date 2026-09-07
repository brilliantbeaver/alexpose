from pathlib import Path
import sys
import unittest

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from laterality_extensions.forecasting import (
    ForecastSpec, FutureJEPA, JOINTS, TimedPose, as_tensors,
    load_local_forecasting, motion_baselines, perturb_future, prefix_input,
    prepare_examples, run_forecasting_study, score_forecasts, synthetic_records,
    shuffled_target_indices,
    fit_future_model,
)


class ForecastingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        cls.spec = ForecastSpec(updates=2, embed_dim=8, batch_size=4)
        cls.records = synthetic_records(sources=6, clips_per_source=1)
        cls.data = prepare_examples(cls.records, cls.spec, status="SYNTHETIC TEST")

    def test_future_coordinates_and_visibility_cannot_change_prefix(self):
        before = prefix_input(self.records[0], self.spec)
        after = prefix_input(perturb_future(self.records[0], self.spec), self.spec)
        for field in before:
            np.testing.assert_array_equal(before[field], after[field])

    def test_nonfinite_future_does_not_change_context(self):
        record = self.records[0]
        xyz = record.xyz.copy()
        xyz[record.times > self.spec.context_seconds] = np.nan
        changed = TimedPose(record.source, record.sequence_id, record.times, xyz, record.valid)
        np.testing.assert_array_equal(prefix_input(record, self.spec)["xyz"], prefix_input(changed, self.spec)["xyz"])

    def test_future_timestamp_changes_do_not_change_prefix(self):
        r = self.records[0]
        times = r.times.copy()
        times[times > self.spec.context_seconds] += 0.37
        changed = TimedPose(r.source, r.sequence_id, times, r.xyz, r.valid)
        before, after = prefix_input(r, self.spec), prefix_input(changed, self.spec)
        for field in before:
            np.testing.assert_array_equal(before[field], after[field])

    def test_held_out_future_cannot_change_fitted_model(self):
        import copy
        train = np.flatnonzero(np.isin(self.data.sources, sorted(set(self.data.sources))[:4]))
        test = np.setdiff1d(np.arange(len(self.data.sources)), train)
        poisoned = copy.deepcopy(self.data)
        poisoned.future[test] = 1e5
        poisoned.future_valid[test] = False
        poisoned.future_times[test] = 20
        original, _, _ = fit_future_model(self.data, train, self.spec)
        changed, _, _ = fit_future_model(poisoned, train, self.spec)
        for name, tensor in original.state_dict().items():
            torch.testing.assert_close(tensor, changed.state_dict()[name], rtol=0, atol=0)

    def test_prediction_has_no_future_input(self):
        model = FutureJEPA(8).eval()
        record = self.records[0]
        contexts = [prefix_input(r, self.spec) for r in (record, perturb_future(record, self.spec))]
        outputs = []
        with torch.no_grad():
            for context in contexts:
                outputs.append(model.predict(torch.tensor(context["xyz"])[None], torch.tensor(context["valid"])[None], torch.tensor(context["times"])[None], torch.tensor([0.25])))
        torch.testing.assert_close(outputs[0], outputs[1], rtol=0, atol=0)

    def test_future_windows_strictly_follow_boundary(self):
        self.assertTrue(np.all(self.data.future_times[self.data.future_valid] > 0))
        self.assertTrue(np.all(self.data.past_times[self.data.past_valid] <= 1e-8))
        self.assertEqual(len(self.data.past), len(self.records) * len(self.spec.horizons))

    def test_invalid_settings_fail(self):
        with self.assertRaises(ValueError):
            ForecastSpec(horizons=(0.1,))
        with self.assertRaises(ValueError):
            ForecastSpec(horizons=(0.75, 0.25))

    def test_invalid_timestamps_fail(self):
        r = self.records[0]
        bad_times = r.times.copy()
        bad_times[2] = bad_times[1]
        with self.assertRaises(ValueError):
            prefix_input(TimedPose(r.source, r.sequence_id, bad_times, r.xyz, r.valid), self.spec)

    def test_baselines_finite_and_common_scoring(self):
        persistence, velocity = motion_baselines(self.data)
        self.assertTrue(np.isfinite(persistence).all())
        self.assertTrue(np.isfinite(velocity).all())
        scores = score_forecasts({"persistence": persistence, "velocity": velocity}, self.data, np.arange(len(persistence)))
        self.assertEqual(len(scores), 6)
        for _, rows in scores.groupby("horizon_seconds"):
            self.assertEqual(rows.observed_joint_endpoints.nunique(), 1)

    def test_static_input_persistence_is_exact(self):
        r = self.records[0]
        xyz = np.repeat(r.xyz[:1], len(r.times), axis=0)
        static = prepare_examples([TimedPose(r.source, r.sequence_id, r.times, xyz, r.valid)], self.spec, status="SYNTHETIC STATIC")
        persistence, velocity = motion_baselines(static)
        np.testing.assert_allclose(persistence, static.future[:, -1], atol=1e-7)
        np.testing.assert_allclose(velocity, static.future[:, -1], atol=1e-7)

    def test_source_overlap_rejected(self):
        with self.assertRaises(ValueError):
            run_forecasting_study(self.data, self.spec, train_sources=["a", "b"], test_sources=["b", "c"])

    def test_shuffled_future_draws_sources_before_clips(self):
        sources = np.array(["context", "small"] + ["large"] * 100)
        horizons = np.full(len(sources), 0.25)
        chosen = shuffled_target_indices(np.zeros(2000, dtype=int), np.arange(len(sources)), sources, horizons, np.random.default_rng(4))
        small_fraction = np.mean(sources[chosen] == "small")
        self.assertGreater(small_fraction, 0.45)
        self.assertLess(small_fraction, 0.55)
        self.assertNotIn("context", sources[chosen])

    def test_tiny_training_comparison_runs(self):
        groups = sorted(set(self.data.sources))
        result = run_forecasting_study(self.data, self.spec, train_sources=groups[:4], test_sources=groups[4:])
        self.assertEqual(len(result["scores"]), 18)
        self.assertTrue(np.isfinite(result["scores"].source_balanced_rmse).all())
        self.assertEqual(len(result["history"]), 2)
        self.assertGreaterEqual(result["variation"]["effective_rank"], 0)
        train_groups = set(self.data.sources[result["train_indices"]])
        test_groups = set(self.data.sources[result["test_indices"]])
        self.assertFalse(train_groups & test_groups)


if __name__ == "__main__":
    unittest.main()
