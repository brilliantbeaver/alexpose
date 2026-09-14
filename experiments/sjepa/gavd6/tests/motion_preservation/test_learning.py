"""motion preservation / test learning."""


import unittest

import numpy as np
import pandas as pd
import torch

from gavd6_sjepa.research_directions.motion_preservation.body_geometry import (
    PARENTS,
    project_bone_lengths,
)
from gavd6_sjepa.research_directions.motion_preservation.preservation_metrics import (
    paired_person_interval,
    score_case,
    summarize,
)
from gavd6_sjepa.research_directions.motion_preservation.repair_models import (
    FLOW_FIELDS,
    FLOW_START,
    feature_view,
    fit_gate,
    infer_gate,
    make_features,
    project_torch,
)

# Small CPU checks of the actual research gate and person-level comparisons.


class GateTests(unittest.TestCase):
    def test_training_and_scoring_projection_agree_including_collapsed_bones(self):
        rng = np.random.default_rng(31)
        x = rng.normal(size=(2, 5, 22, 3)).astype(np.float32)
        x[0, 2, 4] = x[0, 2, PARENTS[4]]
        lengths = rng.uniform(.1, .5, (2, 22)).astype(np.float32)
        tensor = torch.tensor(x, requires_grad=True)
        actual = project_torch(tensor, torch.from_numpy(lengths))
        expected = np.stack([project_bone_lengths(x[i], lengths[i]) for i in range(2)])
        np.testing.assert_allclose(actual.detach(), expected, atol=1e-6)
        actual.square().mean().backward()
        self.assertTrue(torch.isfinite(tensor.grad).all())

    def test_missing_uncertainty_is_distinct_from_measured_zero(self):
        raw = np.zeros((5, 22, 3), np.float32)
        confidence = np.ones((5, 22), np.float32)
        diagnostic = {key: np.zeros((4, 22), np.float32) for key in FLOW_FIELDS}
        diagnostic["evidence_valid"] = np.ones((4, 22), bool)
        missing = {key: value.copy() for key, value in diagnostic.items()}
        missing["flow_uncertainty_px"][:] = np.nan
        known = make_features(raw, raw, confidence, confidence, diagnostic, diagnostic, 128)
        unknown = make_features(raw, raw, confidence, confidence, missing, diagnostic, 128)
        self.assertFalse(np.array_equal(known, unknown))
        self.assertTrue(np.isfinite(unknown).all())
        coordinates = feature_view(unknown[None], "coordinates")
        np.testing.assert_allclose(coordinates[..., FLOW_START:], 0)
        np.testing.assert_array_equal(coordinates[..., :FLOW_START], unknown[None, ..., :FLOW_START])

    def test_small_cpu_fit_improves_and_inference_uses_no_reference(self):
        old_threads = torch.get_num_threads()
        torch.set_num_threads(1)
        try:
            count, frames = 6, 10
            rng = np.random.default_rng(8)
            lengths = np.full((count, 22), .15, np.float32)
            truth = np.zeros((count, frames, 22, 3), np.float32)
            for joint in range(1, 22):
                truth[:, :, joint] = truth[:, :, PARENTS[joint]] + np.array([.03, .14, .04])
            truth = np.stack([project_bone_lengths(x, lengths[i]) for i, x in enumerate(truth)])
            raw = truth + rng.normal(0, .03, truth.shape).astype(np.float32)
            prior = truth.copy()
            features = rng.normal(size=(count, frames, 22, 47)).astype(np.float32)
            arrays = dict(features=features, raw=raw, prior=prior, truth=truth, bone_lengths=lengths,
                          observed=np.ones((count, frames, 22), np.float32),
                          event_support=np.zeros((count, frames, 22), np.float32),
                          event_present=np.array([0, 1, 0, 1, 0, 1], np.float32))
            payload, history = fit_gate(arrays, epochs=12, hidden_dim=8, batch_size=count,
                                        learning_rate=.01, device="cpu", seed=3)
            self.assertLess(history[-1]["restoration_loss"], history[0]["restoration_loss"])
            # Prediction takes measurement features only. Identical available
            # observations get identical predictions even with hidden labels.
            duplicated = np.repeat(features[:1], 2, axis=0)
            retention, logits = infer_gate(payload, duplicated)
            self.assertEqual(retention.shape, (2, frames, 22))
            self.assertTrue(((retention >= 0) & (retention <= 1)).all())
            np.testing.assert_array_equal(retention[0], retention[1])
            np.testing.assert_array_equal(logits[0], logits[1])
        finally:
            torch.set_num_threads(old_threads)

class PersonAggregationTests(unittest.TestCase):
    def _scores(self):
        rows = []
        for person, retention, brier in (("a", .2, .1), ("b", .6, .3), ("c", .8, .8)):
            for method, offset in (("baseline", 0), ("adapter", .1)):
                rows.append(dict(person_id=person, method=method, fixture="factorial", noise_present=True, event_present=True,
                                 retention=retention+offset, noise_removal=.3, descriptor_abs_error=.1,
                                 mse_m2=.01, decided=1., correct=1., brier=brier))
        return pd.DataFrame(rows)

    def test_repeated_trials_do_not_create_more_independent_people(self):
        original = self._scores()
        repeated = pd.concat([original, *([original.loc[original.person_id.eq("a")]] * 20)])
        before = paired_person_interval(original, "adapter", "baseline", samples=100)
        after = paired_person_interval(repeated, "adapter", "baseline", samples=100)
        self.assertEqual(before["n_people"], 3)
        self.assertEqual(after["n_people"], 3)
        self.assertAlmostEqual(before["mean"], after["mean"])
        first = summarize(original).set_index("method")
        second = summarize(repeated).set_index("method")
        np.testing.assert_allclose(first.brier, second.brier)
        np.testing.assert_allclose(first.retention, second.retention)


class RepairMetricTests(unittest.TestCase):
    def test_gap_filling_does_not_count_as_repairing_observed_joints(self):
        from gavd6_sjepa.research_directions.motion_preservation.body_geometry import demo_motion
        clean, event = demo_motion(frames=16)
        truth = event.joints.copy()
        raw = truth.copy()
        raw[:, 0, 0] += .02
        observed = np.ones(raw.shape[:2], bool)
        observed[4:9, 10] = False
        # Cached raw gaps are interpolated. A poor interpolation can dominate
        # total error even when the remaining measured errors are unchanged.
        raw[4:9, 10, 1] -= .25
        repaired = raw.copy()
        repaired[~observed] = truth[~observed]
        case = dict(raw=raw, truth=truth, clean=clean.joints,
                    event_reference=truth, observed=observed)
        metadata = dict(case_id="gap", person_id="p", role="development",
                        event_family="foot_clearance", fixture="factorial",
                        event_present=True, noise_present=True)
        score = score_case(repaired, case, metadata, "gap_fill_only")
        self.assertAlmostEqual(score["noise_removal"], 0.)
        self.assertAlmostEqual(score["completion_mse_m2"], 0.)
        self.assertLess(score["mse_m2"], score["raw_mse_m2"])
        repaired[observed] = truth[observed] + .5 * (raw[observed] - truth[observed])
        score = score_case(repaired, case, metadata, "half_observed_error")
        self.assertAlmostEqual(score["noise_removal"], .75, places=5)


if __name__ == "__main__":
    unittest.main()
