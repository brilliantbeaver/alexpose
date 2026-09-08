"""Scientific-control tests for the new masking evaluation helpers."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
import sys
import unittest

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from laterality.model import SJEPAGait, valid_patches
from laterality_extensions.masked_learning import LearningSettings, load_learning_dataset
from laterality_extensions.comparative_evaluation import (
    aggregate_predictions, encode_laterality_features, evaluate_frozen_representations,
    feature_diagnostics, fit_source_readout, make_evaluation_mask_bank,
    paired_source_bootstrap, predictor_diagnostics, prepare_raw_missing_observations,
    prepared_observation_sensitivity, validate_prediction_coverage,
)


def prediction_fixture():
    expected = pd.DataFrame([
        {"sequence_id": f"{source}_{clip}", "source_id": source, "fold": fold}
        for source, fold, clips in (("a", 0, 2), ("b", 0, 1), ("c", 1, 3), ("d", 1, 2))
        for clip in range(clips)
    ])
    rows = []
    for condition in ("candidate", "reference"):
        for seed in (4, 9):
            for index, identity in expected.iterrows():
                target = (-1.4 + index * 0.4)
                error = 0.2 if condition == "candidate" else 0.4
                rows.append({**identity.to_dict(), "seed": seed, "condition": condition,
                             "representation": "pretrained_online", "observation": "unaltered",
                             "target": target, "prediction": target + error * (1 if seed == 4 else -1),
                             "available": True, "checkpoint": "final",
                             "comparison_id": "same_data_and_training_controls"})
    return pd.DataFrame(rows), expected


DECLARATION = dict(seeds=(4, 9), conditions=("candidate", "reference"),
                   representations=("pretrained_online",))


class SourceReadoutTests(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(81)
        self.sources = np.repeat(np.array(list("abcdefgh")), 5)
        self.x = rng.normal(size=(40, 4))
        self.y = 2 * self.x[:, 0] - 0.5 * self.x[:, 1]
        self.x[::3, 3] = np.nan
        self.train, self.test = tuple("abcdef"), tuple("gh")

    def fitted(self, x=None, y=None):
        return fit_source_readout(self.x if x is None else x, self.y if y is None else y,
                                  self.sources, train_sources=self.train, test_sources=self.test,
                                  alphas=(0.001, 0.1, 10), inner_folds=3)

    def test_known_signal_recovered_without_training_on_test_sources(self):
        fitted = self.fitted()
        test = np.isin(self.sources, self.test)
        error = np.mean((fitted.predict(self.x[test]) - self.y[test]) ** 2)
        self.assertLess(error, 0.01)
        self.assertEqual(fitted.train_sources, self.train)
        self.assertFalse(fitted.training_feature_diagnostics["near_constant"])

    def test_outer_test_values_cannot_change_fitting_or_penalty(self):
        first = self.fitted()
        x, y = self.x.copy(), self.y.copy()
        test = np.isin(self.sources, self.test)
        x[test] = 1e9
        y[test] = np.nan  # These labels are not required for fitting.
        second = self.fitted(x, y)
        np.testing.assert_array_equal(first.regression.coef_, second.regression.coef_)
        np.testing.assert_array_equal(first.scaler.impute_, second.scaler.impute_)
        np.testing.assert_array_equal(first.scaler.mean_, second.scaler.mean_)
        self.assertEqual(first.selected_alpha, second.selected_alpha)
        pd.testing.assert_frame_equal(first.validation, second.validation)

    def test_source_roles_and_penalties_are_checked(self):
        with self.assertRaisesRegex(ValueError, "disjoint"):
            fit_source_readout(self.x, self.y, self.sources, train_sources=self.train, test_sources=("a", "g", "h"))
        with self.assertRaisesRegex(ValueError, "exactly"):
            fit_source_readout(self.x, self.y, self.sources, train_sources=self.train, test_sources=("g",))
        with self.assertRaisesRegex(ValueError, "positive"):
            fit_source_readout(self.x, self.y, self.sources, train_sources=self.train, test_sources=self.test, alphas=(0,))

    def test_constant_and_nearly_constant_features_are_flagged(self):
        constant = feature_diagnostics(np.ones((8, 3)), np.array(list("abcdefgh")))
        self.assertTrue(constant["near_constant"])
        self.assertEqual(constant["effective_rank"], 0)
        close = np.ones((8, 3)) + np.arange(8)[:, None] * 1e-10
        self.assertTrue(feature_diagnostics(close, np.array(list("abcdefgh")))["near_constant"])
        repeated = np.repeat([[0., 1.], [2., 3.]], [1, 5], axis=0)
        unique = feature_diagnostics(np.array([[0., 1.], [2., 3.]]), np.array(["a", "b"]))
        duplicated = feature_diagnostics(repeated, np.array(["a", "b", "b", "b", "b", "b"]))
        self.assertAlmostEqual(unique["mean_channel_sd"], duplicated["mean_channel_sd"])


class CoverageTests(unittest.TestCase):
    def test_pool_per_seed_before_averaging_scores(self):
        table, expected = prediction_fixture()
        result = aggregate_predictions(table, expected, **DECLARATION)
        self.assertEqual(len(result["per_seed"]), 4)
        self.assertTrue((result["per_seed"].evaluated_clips == len(expected)).all())
        self.assertTrue((result["summary"].mean_r2 < 1).all())
        # The seed-averaged predictions are perfect, but that ensemble is not the declared endpoint.
        averaged = table.groupby(["condition", "sequence_id"])[["target", "prediction"]].mean()
        np.testing.assert_allclose(averaged.target, averaged.prediction)

    def test_duplicate_source_overlap_and_missing_rows_fail(self):
        table, expected = prediction_fixture()
        with self.assertRaisesRegex(ValueError, "duplicate"):
            validate_prediction_coverage(pd.concat([table, table.iloc[:1]]), expected, **DECLARATION)
        overlapping = expected.copy()
        overlapping.loc[1, "fold"] = 1
        with self.assertRaisesRegex(ValueError, "overlap"):
            validate_prediction_coverage(table, overlapping, **DECLARATION)
        with self.assertRaisesRegex(ValueError, "Incomplete"):
            validate_prediction_coverage(table.iloc[1:], expected, **DECLARATION)
        summary = aggregate_predictions(table.iloc[1:], expected, allow_partial=True, **DECLARATION)
        self.assertTrue(summary["per_seed"].scope.str.contains("partial").all())

    def test_changed_targets_or_incompatible_comparisons_fail(self):
        table, expected = prediction_fixture()
        changed = table.copy()
        changed.loc[0, "target"] += 1
        with self.assertRaisesRegex(ValueError, "changed the target"):
            validate_prediction_coverage(changed, expected, **DECLARATION)
        changed = table.copy()
        changed.loc[0, "comparison_id"] = "different_exposure"
        with self.assertRaisesRegex(ValueError, "Incompatible"):
            validate_prediction_coverage(changed, expected, **DECLARATION)
        changed = table.copy()
        changed.loc[0, "checkpoint"] = "update_1"
        with self.assertRaisesRegex(ValueError, "Mixed checkpoints"):
            validate_prediction_coverage(changed, expected, **DECLARATION)

    def test_unavailable_predictions_are_reported_and_cannot_enter_bootstrap(self):
        table, expected = prediction_fixture()
        table.loc[0, "available"] = False
        table.loc[0, "prediction"] = np.nan
        summary = aggregate_predictions(table, expected, **DECLARATION)
        self.assertEqual(summary["per_seed"].unavailable_clips.sum(), 1)
        self.assertTrue(summary["per_seed"].scope.str.contains("available cases").any())
        with self.assertRaisesRegex(ValueError, "finite shared coverage"):
            paired_source_bootstrap(table, first="candidate", reference="reference",
                                    representation="pretrained_online", repetitions=10)

    def test_source_bootstrap_keeps_seeds_and_conditions_paired(self):
        table, _ = prediction_fixture()
        kwargs = dict(first="candidate", reference="reference", representation="pretrained_online", repetitions=80)
        first = paired_source_bootstrap(table, **kwargs)
        second = paired_source_bootstrap(table, **kwargs)
        self.assertEqual(first, second)
        self.assertEqual(first["sources"], 4)
        self.assertEqual(first["seeds"], 2)
        self.assertGreater(first["difference"], 0)  # Deliberately constructed prediction errors.
        self.assertIn("excludes retraining", first["uncertainty_scope"])
        with self.assertRaisesRegex(ValueError, "Incomplete"):
            paired_source_bootstrap(table.iloc[1:], **kwargs)


class MissingObservationTests(unittest.TestCase):
    def test_raw_hidden_values_cannot_affect_prepared_input(self):
        rng = np.random.default_rng(9)
        raw = rng.normal(size=(16, 33, 4))
        raw[..., 3] = 1
        raw[:, 23, :3] = [-0.5, 0, 0]
        raw[:, 24, :3] = [0.5, 0, 0]
        hidden = np.zeros((16, 33), bool)
        hidden[4:12, [23, 24, 27, 28]] = True
        first = prepare_raw_missing_observations(raw, np.arange(16), 30, hidden,
                                                 original_target=0.23, frames=16, max_interpolation_gap=0)
        changed = raw.copy()
        changed[hidden, :3] = 1e9
        changed[hidden, 3] = 0.2
        second = prepare_raw_missing_observations(changed, np.arange(16), 30, hidden,
                                                  original_target=0.23, frames=16, max_interpolation_gap=0)
        np.testing.assert_array_equal(first["xyz"], second["xyz"])
        np.testing.assert_array_equal(first["valid"], second["valid"])
        self.assertEqual(first["target"], 0.23)
        self.assertEqual(first["deliberately_removed_observations"], int(hidden.sum()))

    def test_bank_is_shared_readonly_and_preserves_original_targets(self):
        dataset = load_learning_dataset()
        valid = dataset.valid.reshape(len(dataset.xyz), -1, 4, 33).all(axis=2)
        bank = make_evaluation_mask_bank(valid)
        again = make_evaluation_mask_bank(valid)
        for name, mask in bank.items():
            np.testing.assert_array_equal(mask, again[name])
            self.assertFalse(mask.flags.writeable)
            corrupted = prepared_observation_sensitivity(dataset, mask)
            np.testing.assert_array_equal(corrupted.targets, dataset.targets)
            self.assertTrue(np.all(corrupted.valid <= dataset.valid))
            np.testing.assert_array_equal(corrupted.removed_valid_tokens, (mask & valid).sum((1, 2)))
        self.assertTrue(bank["left_leg_gap"][:, :, [25, 27, 29, 31]].any())
        self.assertTrue(bank["right_leg_gap"][:, :, [26, 28, 30, 32]].any())


class ModelEvaluationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.previous_threads = torch.get_num_threads()
        torch.set_num_threads(1)
        cls.dataset = load_learning_dataset()
        cls.settings = LearningSettings(steps=1, embed_dim=8, heads=2)
        torch.manual_seed(4)
        cls.model = SJEPAGait(frames=cls.dataset.xyz.shape[1], embed_dim=8,
                              encoder_depth=1, predictor_depth=1, heads=2)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.previous_threads)

    def test_normal_predictor_path_and_fixed_hidden_value_isolation(self):
        dataset = self.dataset
        xyz = torch.as_tensor(dataset.xyz[:2], dtype=torch.float32)
        patches = valid_patches(torch.as_tensor(dataset.valid[:2]), 4)
        masks = make_evaluation_mask_bank(patches.numpy())["scattered_gap"]
        first_mask = torch.as_tensor(masks[:1].copy())
        changed = xyz[:1].clone()
        changed[torch.repeat_interleave(first_mask, 4, dim=1)] = 10000
        self.model.eval()
        with torch.no_grad():
            first, _ = self.model(xyz[:1], xyz[:1], patches[:1], first_mask)
            second, _ = self.model(changed, xyz[:1], patches[:1], first_mask)
        torch.testing.assert_close(first, second)
        self.assertTrue(all(parameter.grad is None for parameter in self.model.target_encoder.parameters()))

    def test_diagnostics_do_not_mix_clips_with_unequal_target_counts(self):
        valid = self.dataset.valid.reshape(len(self.dataset.xyz), -1, 4, 33).all(axis=2)
        mask = np.zeros_like(valid)
        for row in range(len(valid)):
            mask[row].flat[np.flatnonzero(valid[row])[:row % 3 + 1]] = True
        result = predictor_diagnostics(self.model, self.dataset, {"unequal_counts": mask}, condition="initial")
        self.assertEqual(result.iloc[0].evaluated_clips, len(self.dataset.test_rows))
        self.assertTrue(np.isfinite(result.iloc[0].normalized_error))
        self.assertIn("target_clip_effective_rank", result)
        self.assertTrue(all(parameter.grad is None for parameter in self.model.target_encoder.parameters()))

    def test_frozen_readouts_include_controls_and_keep_teacher_distinct(self):
        valid = self.dataset.valid.reshape(len(self.dataset.xyz), -1, 4, 33).all(axis=2)
        bank = make_evaluation_mask_bank(valid)
        altered = prepared_observation_sensitivity(self.dataset, bank["left_leg_gap"])
        result = evaluate_frozen_representations(self.model, self.model, self.dataset, self.settings,
                                                 condition="initial_demo", alphas=(0.1, 1),
                                                 observation_datasets={"left_gap_sensitivity": altered})
        self.assertEqual(set(result["predictions"].representation),
                         {"pretrained_online", "pretrained_teacher", "initial_online", "direct_pose", "training_mean"})
        self.assertEqual(set(result["predictions"].observation), {"unaltered", "left_gap_sensitivity"})
        for readout in result["readouts"].values():
            self.assertEqual(set(readout.train_sources), set(self.dataset.train_sources))
        changed = replace(altered, targets=altered.targets + 1)
        with self.assertRaisesRegex(ValueError, "reference targets"):
            evaluate_frozen_representations(self.model, self.model, self.dataset, self.settings,
                                            condition="bad", observation_datasets={"changed": changed})

    def test_unavailable_encoder_row_remains_missing(self):
        valid = self.dataset.valid.copy()
        valid[0] = False
        altered = replace(self.dataset, valid=valid)
        features, available = encode_laterality_features(self.model.view_encoder, altered)
        self.assertFalse(available[0])
        self.assertTrue(np.isnan(features[0]).all())

    def test_infeasible_missing_region_is_never_scored_as_unchanged_input(self):
        valid = self.dataset.valid.copy()
        row = self.dataset.test_rows[0]
        valid[row, :, [25, 27, 29, 31]] = False
        dataset = replace(self.dataset, valid=valid)
        patches = valid.reshape(len(valid), -1, 4, 33).all(axis=2)
        bank = make_evaluation_mask_bank(patches)
        altered = prepared_observation_sensitivity(dataset, bank["left_leg_gap"])
        self.assertGreater(altered.requested_hidden_tokens[row], 0)
        self.assertEqual(altered.removed_valid_tokens[row], 0)
        self.assertFalse(altered.corruption_feasible[row])
        result = evaluate_frozen_representations(self.model, self.model, dataset, self.settings,
                                                 condition="synthetic", alphas=(1,),
                                                 observation_datasets={"left_gap": altered})
        selected = result["predictions"].query("observation == 'left_gap'")
        selected = selected[selected.sequence_id == dataset.sequence_ids[row]]
        self.assertTrue((selected.status == "no_observed_token_removed").all())
        self.assertTrue(selected.prediction.isna().all())
        self.assertFalse(selected.available.any())


if __name__ == "__main__":
    unittest.main()
