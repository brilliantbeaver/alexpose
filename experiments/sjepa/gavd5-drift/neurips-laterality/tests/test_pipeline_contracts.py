"""Small tests for leakage-sensitive contracts outside the geometry/split suites."""
from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd


SUITE_ROOT = Path(__file__).resolve().parents[1]
if str(SUITE_ROOT) not in sys.path:
    sys.path.insert(0, str(SUITE_ROOT))

from laterality.evaluation import (  # noqa: E402
    EXPECTED_LANES,
    REPRESENTATION_AUDIT_COLUMNS,
    WeightedScaler,
    _fit_readout,
    select_alpha,
    validate_evaluation_frame,
)
from laterality.config import canonical_json_digest  # noqa: E402
from laterality.config import load_context  # noqa: E402
from laterality.data import (  # noqa: E402
    PreparedCohort,
    load_cohort,
    prepare_cohort,
    save_cohort,
)
from laterality.geometry import anatomical_mirror, prepare_pose  # noqa: E402
from laterality.metrics import source_weights, weighted_mae, weighted_r2  # noqa: E402
from laterality.model import anatomical_reflect_tensor  # noqa: E402
from laterality.reporting import (  # noqa: E402
    _bootstrap_delta,
    _checkpoint_r2_bootstrap_row,
    _native_symmetry_bootstrap_row,
    seed_average_predictions,
    symmetry_gate_decisions,
)
from laterality.training import (  # noqa: E402
    _state_digest,
    make_rng_streams,
    source_balanced_epoch_batches,
    validate_checkpoint_lineage,
)


class SourceWeightTests(unittest.TestCase):
    def test_duplicate_clips_from_one_source_do_not_change_primary_metrics(self):
        truth = np.array([-1.0, -0.4, 0.5, 0.9])
        prediction = np.array([-0.8, -0.1, 0.2, 1.1])
        groups = np.array(["a", "a", "b", "b"])
        baseline_r2 = weighted_r2(truth, prediction, source_weights(groups))
        baseline_mae = weighted_mae(truth, prediction, source_weights(groups))

        # Replicate every clip from source a equally; its total weight remains one.
        take = np.array([0, 1, 0, 1, 0, 1, 2, 3])
        expanded_groups = groups[take]
        self.assertAlmostEqual(
            weighted_r2(truth[take], prediction[take], source_weights(expanded_groups)),
            baseline_r2,
        )
        self.assertAlmostEqual(
            weighted_mae(truth[take], prediction[take], source_weights(expanded_groups)),
            baseline_mae,
        )

    def test_constrained_readout_is_odd_about_the_origin(self):
        features = np.array([[-2.0, 1.0], [-1.0, 0.4], [1.0, -0.4], [2.0, -1.0]])
        target = np.array([-0.8, -0.3, 0.3, 0.8])
        groups = np.array(["a", "b", "c", "d"])
        scaler, model = _fit_readout(
            features,
            target,
            groups,
            alpha=0.1,
            constrained_odd=True,
        )
        prediction = model.predict(scaler.transform(features))
        mirror_prediction = model.predict(scaler.transform(-features))
        np.testing.assert_allclose(prediction + mirror_prediction, 0.0, atol=1e-12)
        self.assertFalse(scaler.center)
        self.assertFalse(model.fit_intercept)

    def test_constrained_readout_rejects_nonfinite_features(self):
        with self.assertRaises(ValueError):
            _fit_readout(
                np.array([[1.0], [np.nan]]),
                np.array([0.1, -0.1]),
                np.array(["a", "b"]),
                alpha=1.0,
                constrained_odd=True,
            )

    def test_duplicate_source_clips_do_not_change_scaler_ridge_or_alpha(self):
        features = np.array(
            [
                [-1.2, 0.1],
                [-0.8, 0.3],
                [-0.3, 0.5],
                [0.0, 0.4],
                [0.4, -0.2],
                [0.7, -0.4],
                [1.0, -0.7],
                [1.3, -0.8],
            ]
        )
        target = np.array([-1.0, -0.7, -0.3, -0.1, 0.3, 0.6, 0.8, 1.1])
        groups = np.repeat(np.array(["a", "b", "c", "d"]), 2)
        # Replicate every sequence from source a three times. Source weighting must
        # leave all fitted quantities and inner-fold scores unchanged.
        take = np.array([0, 1, 0, 1, 0, 1, 2, 3, 4, 5, 6, 7])
        base_scaler, base_model = _fit_readout(
            features, target, groups, alpha=0.1, constrained_odd=False
        )
        expanded_scaler, expanded_model = _fit_readout(
            features[take],
            target[take],
            groups[take],
            alpha=0.1,
            constrained_odd=False,
        )
        np.testing.assert_allclose(base_scaler.impute_, expanded_scaler.impute_)
        np.testing.assert_allclose(base_scaler.mean_, expanded_scaler.mean_)
        np.testing.assert_allclose(base_scaler.scale_, expanded_scaler.scale_)
        np.testing.assert_allclose(base_model.coef_, expanded_model.coef_)
        np.testing.assert_allclose(base_model.intercept_, expanded_model.intercept_)
        np.testing.assert_allclose(
            base_model.predict(base_scaler.transform(features)),
            expanded_model.predict(expanded_scaler.transform(features)),
        )

        inner_folds = [
            {"train_sources": ["a", "b"], "validation_sources": ["c", "d"]},
            {"train_sources": ["c", "d"], "validation_sources": ["a", "b"]},
        ]
        alphas = [0.01, 0.1, 1.0, 10.0]
        base_alpha, base_scores = select_alpha(
            features,
            target,
            groups,
            inner_folds,
            alphas,
            constrained_odd=False,
        )
        expanded_alpha, expanded_scores = select_alpha(
            features[take],
            target[take],
            groups[take],
            inner_folds,
            alphas,
            constrained_odd=False,
        )
        self.assertEqual(base_alpha, expanded_alpha)
        np.testing.assert_allclose(
            [base_scores[str(alpha)] for alpha in alphas],
            [expanded_scores[str(alpha)] for alpha in alphas],
        )


class SamplingTests(unittest.TestCase):
    def test_reflection_draws_cannot_perturb_sampling_or_mask_streams(self):
        vanilla = make_rng_streams(42, 3)
        augmented = make_rng_streams(42, 3)
        _ = augmented["reflection"].random(1000)
        np.testing.assert_array_equal(
            vanilla["sampling"].integers(0, 100, size=40),
            augmented["sampling"].integers(0, 100, size=40),
        )
        np.testing.assert_array_equal(
            vanilla["mask"].integers(0, 100, size=40),
            augmented["mask"].integers(0, 100, size=40),
        )

    def test_epoch_covers_sources_and_never_reads_annotations(self):
        table = pd.DataFrame(
            {
                "video_id": ["a", "a", "b", "c", "c", "c"],
                "condition": ["secret"] * 6,
            }
        )
        batches = list(
            source_balanced_epoch_batches(
                table,
                ["a", "b", "c"],
                batch_size=2,
                updates_per_epoch=2,
                rng=np.random.default_rng(9),
            )
        )
        drawn = [source for _, sources in batches for source in sources]
        self.assertTrue({"a", "b", "c"} <= set(drawn))
        for indices, sources in batches:
            self.assertEqual(len(indices), len(sources))
            for row, source in zip(indices, sources):
                self.assertEqual(table.loc[row, "video_id"], source)


class ArtifactLineageTests(unittest.TestCase):
    def test_smoke_cohort_cannot_be_loaded_as_paper_data(self):
        with tempfile.TemporaryDirectory() as directory:
            with patch.dict(
                "os.environ",
                {
                    "LATERALITY_PROFILE": "smoke",
                    "LATERALITY_ARTIFACT_ROOT": directory,
                },
                clear=False,
            ):
                smoke_context = load_context()
                save_cohort(smoke_context, prepare_cohort(smoke_context))
            with patch.dict(
                "os.environ",
                {
                    "LATERALITY_PROFILE": "paper",
                    "LATERALITY_ARTIFACT_ROOT": directory,
                },
                clear=False,
            ):
                paper_context = load_context()
                with self.assertRaisesRegex(RuntimeError, "context"):
                    load_cohort(paper_context)

    def test_checkpoint_cache_rejects_mutated_lineage_or_state(self):
        import torch

        expected = {
            "schema": "test",
            "fold": 2,
            "train_sources": ["a", "b"],
            "forbidden_test_sources": ["c"],
            "labels_enter_objective": False,
        }
        checkpoint = {
            **expected,
            "lineage_digest": canonical_json_digest(expected),
            "objective_inputs": ["coordinates", "validity"],
            "initial_target_state": {},
            "initial_state_digest": _state_digest({}),
            "model_state": {"weight": torch.tensor([1.0])},
            "model_state_digest": _state_digest({"weight": torch.tensor([1.0])}),
            "sampling": "source_uniform_then_sequence_uniform",
            "rng_streams": ["sampling", "mask", "reflection"],
        }
        validate_checkpoint_lineage(checkpoint, expected)
        checkpoint["fold"] = 3
        with self.assertRaises(RuntimeError):
            validate_checkpoint_lineage(checkpoint, expected)
        checkpoint["fold"] = 2
        checkpoint["model_state"]["weight"][0] = 2.0
        with self.assertRaisesRegex(RuntimeError, "model state"):
            validate_checkpoint_lineage(checkpoint, expected)

    def test_evaluation_frame_requires_exact_locked_rows_and_targets(self):
        context = load_context(profile="smoke")
        table = pd.DataFrame(
            {
                "sequence_id": ["s1", "s2"],
                "video_id": ["v1", "v2"],
                "condition": ["normal", "stroke"],
                "target": [0.1, -0.2],
            }
        )
        cohort = PreparedCohort(
            table=table,
            model_xyz=np.empty((2, 0, 33, 3)),
            model_valid=np.empty((2, 0, 33), dtype=bool),
            pair_contrasts=np.empty((2, 0)),
            missingness=np.empty((2, 0)),
            cohort_digest="fixture",
            attrition={},
        )
        rows = []
        for lane in EXPECTED_LANES:
            for row in table.itertuples(index=False):
                rows.append(
                    {
                        "sequence_id": row.sequence_id,
                        "video_id": row.video_id,
                        "condition": row.condition,
                        "target": row.target,
                        "fold": 0,
                        "seed": 7,
                        "variant": "vanilla",
                        "lane": lane,
                        "prediction": 0.0,
                        "mirror_prediction": 0.0,
                        "source_weight": 1.0,
                        "learned_strict_equivariance_error": 0.1,
                        "learned_strict_equivariance_residual_energy": 1.0,
                        "learned_strict_equivariance_total_energy": 10.0,
                        "learned_strict_equivariance_common_tokens": 10,
                        "initial_strict_equivariance_error": 0.2,
                        "initial_strict_equivariance_residual_energy": 2.0,
                        "initial_strict_equivariance_total_energy": 10.0,
                        "initial_strict_equivariance_common_tokens": 10,
                    }
                )
        frame = pd.DataFrame(rows)
        fold = {
            "test_sequence_ids": ["s1", "s2"],
            "test_sources": ["v1", "v2"],
        }
        self.assertTrue(set(REPRESENTATION_AUDIT_COLUMNS) <= set(frame.columns))
        validate_evaluation_frame(frame, context, cohort, fold, 0, 7, "vanilla")
        with self.assertRaises(RuntimeError):
            validate_evaluation_frame(
                frame.iloc[:-1], context, cohort, fold, 0, 7, "vanilla"
            )
        corrupted = frame.copy()
        corrupted.loc[0, "target"] += 0.01
        with self.assertRaisesRegex(RuntimeError, "target"):
            validate_evaluation_frame(
                corrupted, context, cohort, fold, 0, 7, "vanilla"
            )


class BootstrapTests(unittest.TestCase):
    def test_vectorized_source_bootstrap_matches_explicit_block_resampling(self):
        source_by_sequence = ["a", "a", "b", "c", "c", "c"]
        target = np.array([-1.0, -0.6, 0.2, 0.5, 0.8, 1.1])
        prediction_a = np.array([-0.8, -0.5, 0.1, 0.4, 0.7, 1.0])
        prediction_b = np.array([-0.1, -0.2, 0.0, 0.1, 0.2, 0.3])
        rows = []
        for lane, prediction in (("a", prediction_a), ("b", prediction_b)):
            for index, (source, truth, estimate) in enumerate(
                zip(source_by_sequence, target, prediction)
            ):
                rows.append(
                    {
                        "variant": "v",
                        "lane": lane,
                        "sequence_id": f"s{index}",
                        "video_id": source,
                        "target": truth,
                        "prediction": estimate,
                    }
                )
        frame = pd.DataFrame(rows)
        repetitions = 100
        seed = 123
        result = _bootstrap_delta(frame, "v", "a", "b", repetitions, seed)

        sources = np.asarray(["a", "b", "c"])
        draws = np.random.default_rng(seed).integers(
            0, len(sources), size=(repetitions, len(sources))
        )
        explicit = []
        for draw in draws:
            truth_parts = []
            a_parts = []
            b_parts = []
            replica_groups = []
            for replica, source_index in enumerate(draw):
                mask = np.asarray(source_by_sequence) == sources[source_index]
                truth_parts.extend(target[mask])
                a_parts.extend(prediction_a[mask])
                b_parts.extend(prediction_b[mask])
                replica_groups.extend([str(replica)] * int(mask.sum()))
            truth_array = np.asarray(truth_parts)
            weights = source_weights(np.asarray(replica_groups))
            explicit.append(
                weighted_r2(truth_array, np.asarray(a_parts), weights)
                - weighted_r2(truth_array, np.asarray(b_parts), weights)
            )
        self.assertAlmostEqual(result["ci95_low"], np.nanquantile(explicit, 0.025))
        self.assertAlmostEqual(result["ci95_high"], np.nanquantile(explicit, 0.975))

    def test_native_symmetry_squares_before_seed_aggregation(self):
        rows = []
        for seed, residual in ((1, 1.0), (2, -1.0)):
            for index, source in enumerate(("a", "b", "c")):
                rows.append(
                    {
                        "variant": "vanilla",
                        "lane": "learned_single_free",
                        "sequence_id": f"s{index}",
                        "video_id": source,
                        "seed": seed,
                        "prediction": residual,
                        "mirror_prediction": residual,
                        "training_target_scale": 1.0,
                    }
                )
        result = _native_symmetry_bootstrap_row(
            pd.DataFrame(rows),
            variant_a="vanilla",
            lane_a="learned_single_free",
            variant_b=None,
            lane_b=None,
            repetitions=100,
            seed=9,
            comparison_type="test",
        )
        # Averaging signed predictions first would incorrectly produce zero.
        self.assertAlmostEqual(result["estimate"], 1.0)
        self.assertAlmostEqual(result["ci95_low"], 1.0)
        self.assertAlmostEqual(result["ci95_high"], 1.0)

    def test_relative_random_advantage_cannot_replace_absolute_utility(self):
        targets = np.array([-1.5, -1.0, -0.5, 0.5, 1.0, 1.5])
        rows = []
        for lane, prediction in (
            ("learned_single_free", np.zeros_like(targets)),
            ("random_single_free", -10.0 * targets),
        ):
            for index, (target, estimate) in enumerate(zip(targets, prediction)):
                rows.append(
                    {
                        "variant": "vanilla",
                        "lane": lane,
                        "sequence_id": f"s{index}",
                        "video_id": f"v{index}",
                        "target": target,
                        "seed": 1,
                        "prediction": estimate,
                    }
                )
        frame = pd.DataFrame(rows)
        absolute = _checkpoint_r2_bootstrap_row(
            frame,
            variant_a="vanilla",
            lane_a="learned_single_free",
            variant_b=None,
            lane_b=None,
            repetitions=500,
            seed=11,
            comparison_type="absolute",
        )
        relative = _checkpoint_r2_bootstrap_row(
            frame,
            variant_a="vanilla",
            lane_a="learned_single_free",
            variant_b="vanilla",
            lane_b="random_single_free",
            repetitions=500,
            seed=11,
            comparison_type="relative",
        )
        self.assertGreater(relative["ci95_low"], 0.0)
        self.assertLessEqual(absolute["ci95_low"], 0.0)
        decisions = symmetry_gate_decisions(
            primary_r2_ci_low=absolute["ci95_low"],
            minimum_primary_r2=0.0,
            native_output_error_ci_high=0.01,
            native_output_error_margin=0.1,
            primary_minus_initial_r2_ci_low=relative["ci95_low"],
            representation_error_ci_high=0.01,
            representation_error_margin=0.1,
            learned_minus_initial_representation_error_ci_high=-0.01,
        )
        self.assertFalse(decisions["absolute_native_predictive_utility_ci"])
        self.assertFalse(
            decisions["training_induced_symmetry_supported_conditional_on_pose_schema"]
        )


class SeedAggregationTests(unittest.TestCase):
    @staticmethod
    def fixture() -> pd.DataFrame:
        rows = []
        for seed in (1, 2):
            rows.append(
                {
                    "sequence_id": "s",
                    "video_id": "v",
                    "condition": "annotation",
                    "fold": 0,
                    "variant": "vanilla",
                    "lane": "learned_two_pass_odd_zero",
                    "role": "primary",
                    "seed": seed,
                    "target": 0.2,
                    "prediction": 0.1 + 0.01 * seed,
                    "mirror_prediction": -0.1 - 0.01 * seed,
                    "neutral_threshold": 0.03,
                    "training_target_scale": 0.2,
                }
            )
        return pd.DataFrame(rows)

    def test_seed_averaging_requires_all_seeds(self):
        frame = self.fixture().iloc[:1]
        with self.assertRaises(RuntimeError):
            seed_average_predictions(frame, (1, 2))

    def test_seed_averaging_rejects_changed_target_or_train_statistic(self):
        for column in ("target", "neutral_threshold", "training_target_scale"):
            frame = self.fixture()
            frame.loc[1, column] += 0.01
            with self.subTest(column=column), self.assertRaises(RuntimeError):
                seed_average_predictions(frame, (1, 2))


class PreprocessingSymmetryTests(unittest.TestCase):
    def test_tensor_reflection_can_select_only_some_batch_rows(self):
        import torch

        rng = np.random.default_rng(13)
        coordinates = rng.normal(size=(4, 8, 33, 3)).astype(np.float32)
        validity = rng.random((4, 8, 33)) > 0.2
        selected = np.array([True, False, True, False])
        reflected, reflected_valid = anatomical_reflect_tensor(
            torch.from_numpy(coordinates),
            torch.from_numpy(validity),
            torch.from_numpy(selected),
        )
        expected_xyz, expected_valid = anatomical_mirror(
            coordinates[selected], validity[selected]
        )
        np.testing.assert_allclose(reflected.numpy()[selected], expected_xyz)
        np.testing.assert_array_equal(reflected_valid.numpy()[selected], expected_valid)
        np.testing.assert_array_equal(reflected.numpy()[~selected], coordinates[~selected])
        np.testing.assert_array_equal(reflected_valid.numpy()[~selected], validity[~selected])

    def test_preprocessing_commutes_with_anatomical_reflection(self):
        rng = np.random.default_rng(41)
        raw = np.zeros((20, 33, 4), dtype=np.float32)
        raw[..., :3] = rng.normal(size=(20, 33, 3))
        raw[..., 3] = 0.99
        mirrored_xyz, mirrored_valid = anatomical_mirror(
            raw[..., :3], np.ones((20, 33), dtype=bool)
        )
        mirrored_raw = np.concatenate(
            (mirrored_xyz, mirrored_valid[..., None].astype(np.float32)), axis=-1
        )
        kwargs = dict(
            frames=16,
            visibility_threshold=0.45,
            max_interpolation_gap=4,
        )
        prepared = prepare_pose(raw, np.arange(20), 30.0, **kwargs)
        prepared_mirror = prepare_pose(
            mirrored_raw, np.arange(20), 30.0, **kwargs
        )
        expected_xyz, expected_valid = anatomical_mirror(
            prepared.model_xyz, prepared.model_valid
        )
        np.testing.assert_allclose(prepared_mirror.model_xyz, expected_xyz, atol=2e-6)
        np.testing.assert_array_equal(prepared_mirror.model_valid, expected_valid)


if __name__ == "__main__":
    unittest.main()
