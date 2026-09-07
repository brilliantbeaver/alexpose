from __future__ import annotations

from dataclasses import replace
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock
import warnings

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import laterality_extensions.masked_learning as masked_learning
from laterality_extensions.masked_learning import (
    GAIT_JOINTS, LearningSettings, candidate_mask, configure_learning_runtime,
    evaluate_features, load_cached_comparison, load_learning_dataset,
    matched_target_masks, random_target_subset, raw_pose_features,
    resolve_learning_device, run_masked_learning, run_matched_comparison,
    save_comparison, summarize_cross_fitted_predictions,
)
from tutorials.research_08 import build_notebook


class MatchedMaskTests(unittest.TestCase):
    def test_same_actual_count_and_validity(self):
        valid = np.ones((3, 4, 33), bool)
        valid[1, :2, 11] = False
        valid[2, 0, 3:10] = False
        masks = matched_target_masks(valid, mask_fraction=0.5, rng=np.random.default_rng(8))
        gait, uniform = masks["gait"], masks["uniform"]
        np.testing.assert_array_equal(gait.sum((1, 2)), uniform.sum((1, 2)))
        self.assertEqual(len(set(gait.sum((1, 2)))), 1)
        self.assertFalse((gait & ~valid).any())
        self.assertFalse((uniform & ~valid).any())
        forbidden = np.ones(33, bool)
        forbidden[list(GAIT_JOINTS)] = False
        self.assertFalse(gait[:, :, forbidden].any())
        self.assertTrue(np.all(gait.sum((1, 2)) < valid.sum((1, 2))))

    def test_reproducible_draws(self):
        valid = np.ones((2, 4, 33), bool)
        first = matched_target_masks(valid, mask_fraction=0.5, rng=np.random.default_rng(12))
        second = matched_target_masks(valid, mask_fraction=0.5, rng=np.random.default_rng(12))
        for key in first:
            np.testing.assert_array_equal(first[key], second[key])

    def test_random_subset_can_join_shared_budget(self):
        valid = np.ones((2, 4, 33), bool)
        masks = matched_target_masks(valid, mask_fraction=0.5, rng=np.random.default_rng(12),
                                     policies=("gait", "uniform", "random_subset"), subset_seed=31)
        self.assertEqual(len(random_target_subset(31)), 12)
        self.assertNotEqual(random_target_subset(31), random_target_subset(32))
        counts = [m.sum((1, 2)).tolist() for m in masks.values()]
        self.assertTrue(all(c == counts[0] for c in counts))
        self.assertFalse((masks["random_subset"] & ~candidate_mask(valid, "random_subset", 31)).any())

    def test_empty_candidate_policy_fails(self):
        with self.assertRaises(ValueError):
            matched_target_masks(np.zeros((2, 4, 33), bool), mask_fraction=0.5,
                                 rng=np.random.default_rng(1))

    def test_context_remains_under_high_mask_fraction(self):
        valid = np.zeros((2, 1, 33), bool)
        valid[:, :, [11, 12]] = True
        masks = matched_target_masks(valid, mask_fraction=0.99, rng=np.random.default_rng(1))
        self.assertEqual(int(masks["gait"].sum()), 2)


class NotebookBootstrapTests(unittest.TestCase):
    def test_reload_precedes_binding_new_masking_api_names(self):
        notebook = build_notebook()
        bootstrap = next(cell.source for cell in notebook.cells if cell.cell_type == "code")
        self.assertNotIn("from laterality_extensions.masked_learning import", bootstrap)
        self.assertIn("Path(loaded_module).resolve() != expected_module", bootstrap)
        reload_at = bootstrap.index("masked_learning = importlib.reload(masked_learning)")
        bind_at = bootstrap.index("comparison_cache_path = masked_learning.comparison_cache_path")
        self.assertLess(reload_at, bind_at)
        self.assertIn("MASKING_RUNNER_API_VERSION", bootstrap)


class TutorialLearningTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.prior_threads = torch.get_num_threads()
        torch.set_num_threads(1)
        cls.dataset = load_learning_dataset()
        cls.settings = LearningSettings(steps=2)
        cls.progress_events = []
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            cls.result = run_matched_comparison(
                cls.dataset,
                cls.settings,
                progress_callback=cls.progress_events.append,
            )

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.prior_threads)

    def test_synthetic_loader_disjoint_and_no_new_artifacts(self):
        data = self.dataset
        self.assertTrue(data.synthetic)
        self.assertFalse(set(data.train_sources) & set(data.test_sources))
        self.assertEqual(set(data.train_sources) | set(data.test_sources), set(data.source_ids))

    def test_pairing_checks_and_baselines(self):
        self.assertTrue(all(self.result["pairing"].values()))
        rows = self.result["metrics"]
        self.assertEqual(set(rows.representation), {"learned_encoder", "initial_encoder", "raw_pose", "training_mean"})
        self.assertTrue(np.isfinite(rows[["r2", "mae"]]).all().all())
        for representation in ("initial_encoder", "raw_pose", "training_mean"):
            self.assertEqual(rows[rows.representation == representation].r2.nunique(), 1)

    def test_device_validation_resolution_and_cpu_runtime_configuration(self):
        LearningSettings(device="auto").validate()
        LearningSettings(device="cpu").validate()
        self.assertEqual(resolve_learning_device("cpu"), torch.device("cpu"))
        self.assertIn(resolve_learning_device("auto").type, {"cpu", "cuda", "mps"})
        with self.assertRaises(ValueError):
            LearningSettings(device="not-a-torch-device").validate()

        accelerator_cases = (
            ("cuda", torch.cuda.is_available()),
            ("mps", torch.backends.mps.is_available()),
        )
        for requested, available in accelerator_cases:
            with self.subTest(device=requested, available=available):
                if available:
                    self.assertEqual(resolve_learning_device(requested).type, requested)
                else:
                    with self.assertRaises(RuntimeError):
                        resolve_learning_device(requested)

        prior_threads = torch.get_num_threads()
        try:
            runtime = configure_learning_runtime("cpu", cpu_threads=1)
            self.assertEqual(runtime, {
                "device": "cpu", "cpu_threads": 1, "accelerated": False,
            })
        finally:
            torch.set_num_threads(prior_threads)

    def test_vectorized_accounting_runtime_metadata_and_shared_controls(self):
        gait = self.result["runs"]["gait_targets"]
        uniform = self.result["runs"]["uniform_targets"]
        self.assertTrue(all(self.result["pairing"].values()))
        self.assertIs(gait["features"]["initial_encoder"], uniform["features"]["initial_encoder"])
        self.assertIs(gait["features"]["raw_pose"], uniform["features"]["raw_pose"])
        for run in (gait, uniform):
            self.assertEqual(run["encoder_forward_calls"], 4 * self.settings.steps)
            self.assertEqual(run["encoder_invocations"], 3 * self.settings.steps)
            self.assertGreater(run["resident_input_bytes"], 0)
            self.assertEqual(run["runtime_device"], "cpu")
            self.assertEqual(run["cpu_threads"], 1)
            self.assertEqual(
                set(run["runtime_versions"]),
                {"python", "numpy", "torch", "sklearn"},
            )
        self.assertEqual(gait["resident_input_bytes"], uniform["resident_input_bytes"])
        self.assertEqual(gait["source_draw_digest"], uniform["source_draw_digest"])
        self.assertEqual(gait["hidden_token_counts"], uniform["hidden_token_counts"])

    def test_content_addressed_cache_save_load_round_trip(self):
        with tempfile.TemporaryDirectory() as directory, mock.patch.object(
            masked_learning, "SUITE_ROOT", Path(directory)
        ):
            destination = save_comparison(self.result, study="masking_test")
            self.assertEqual(
                destination,
                masked_learning.comparison_cache_path(
                    self.dataset, self.settings, study="masking_test",
                ),
            )
            manifest = json.loads((destination / "manifest.json").read_text())
            self.assertTrue(manifest["complete"])
            self.assertEqual(manifest["cache_key"], self.result["cache_key"])
            self.assertEqual(set(manifest["runs"]), set(self.result["runs"]))
            cached = load_cached_comparison(
                self.dataset, self.settings, study="masking_test",
            )
            self.assertIsNotNone(cached)
            self.assertTrue(cached["cache_reused"])
            self.assertEqual(cached["artifact_path"], destination)
            with self.assertRaises(FileExistsError):
                save_comparison(self.result, study="masking_test")

    def test_cached_predictions_equal_computed_predictions(self):
        with tempfile.TemporaryDirectory() as directory, mock.patch.object(
            masked_learning, "SUITE_ROOT", Path(directory)
        ):
            save_comparison(self.result, study="masking_test")
            cached = load_cached_comparison(
                self.dataset, self.settings, study="masking_test",
            )
            self.assertIsNotNone(cached)
            for label, computed in self.result["runs"].items():
                loaded = cached["runs"][label]
                pd.testing.assert_frame_equal(
                    loaded["predictions"],
                    computed["predictions"],
                    check_dtype=False,
                    check_exact=False,
                    rtol=1e-12,
                    atol=1e-12,
                )
                pd.testing.assert_frame_equal(
                    loaded["metrics"],
                    computed["metrics"],
                    check_dtype=False,
                    check_exact=False,
                    rtol=1e-12,
                    atol=1e-12,
                )

    def test_corrupted_cache_fails_closed(self):
        with tempfile.TemporaryDirectory() as directory, mock.patch.object(
            masked_learning, "SUITE_ROOT", Path(directory)
        ):
            destination = save_comparison(self.result, study="masking_test")
            metrics_path = destination / "metrics.csv"
            with metrics_path.open("ab") as stream:
                stream.write(b"corruption")
            with self.assertRaisesRegex(RuntimeError, "content digest mismatch"):
                load_cached_comparison(
                    self.dataset, self.settings, study="masking_test",
                )

    def test_cache_key_tracks_cpu_thread_configuration(self):
        with tempfile.TemporaryDirectory() as directory, mock.patch.object(
            masked_learning, "SUITE_ROOT", Path(directory)
        ):
            first_path = save_comparison(self.result, study="masking_test")
            prior_threads = torch.get_num_threads()
            try:
                torch.set_num_threads(2 if prior_threads != 2 else 3)
                second_path = masked_learning.comparison_cache_path(
                    self.dataset, self.settings, study="masking_test",
                )
                self.assertNotEqual(first_path, second_path)
                self.assertIsNone(load_cached_comparison(
                    self.dataset, self.settings, study="masking_test",
                ))
            finally:
                torch.set_num_threads(prior_threads)

    def test_custom_symmetry_penalty_has_no_unsafe_cache_identity(self):
        variants = {
            "base": {"symmetry_weight": 0.0},
            "custom": {"symmetry_weight": 1.0},
        }
        with self.assertRaisesRegex(ValueError, "custom penalty"):
            masked_learning.comparison_cache_key(
                self.dataset, self.settings, variants,
            )

    def test_progress_callback_reports_every_optimizer_update(self):
        events = self.progress_events
        self.assertEqual(len(events), 2 * self.settings.steps)
        self.assertEqual(
            [event["comparison_completed_steps"] for event in events],
            list(range(1, 2 * self.settings.steps + 1)),
        )
        self.assertEqual({event["comparison_total_steps"] for event in events}, {4})
        self.assertEqual(
            [event["variant"] for event in events],
            ["gait_targets", "gait_targets", "uniform_targets", "uniform_targets"],
        )
        self.assertTrue(all(np.isfinite(event["loss"]) for event in events))

    def test_teacher_has_no_gradient(self):
        for run in self.result["runs"].values():
            self.assertTrue(all(p.grad is None for p in run["model"].target_encoder.parameters()))

    def test_real_training_requires_explicit_confirmation(self):
        with self.assertRaises(PermissionError):
            run_masked_learning(replace(self.dataset, synthetic=False), self.settings)

    def test_source_overlap_rejected(self):
        wrong = replace(self.dataset, test_sources=self.dataset.test_sources + self.dataset.train_sources[:1])
        with self.assertRaises(ValueError):
            wrong.validate()

    def test_bad_hyperparameters_rejected(self):
        for change in ({"steps": 0}, {"batch_size": 1}, {"mask_fraction": 1},
                       {"ridge_alpha": 0}, {"reflection_probability": 1.1}):
            with self.assertRaises(ValueError):
                replace(self.settings, **change).validate()

    def test_test_features_do_not_fit_preprocessing_or_readout(self):
        features = raw_pose_features(self.dataset)
        reference = evaluate_features(features, self.dataset)
        changed = features.copy()
        changed[self.dataset.test_rows] = 1e8
        altered = evaluate_features(changed, self.dataset)
        np.testing.assert_array_equal(reference["scaler"].mean_, altered["scaler"].mean_)
        np.testing.assert_array_equal(reference["scaler"].scale_, altered["scaler"].scale_)
        np.testing.assert_array_equal(reference["readout"].coef_, altered["readout"].coef_)

    def test_probe_labels_do_not_enter_representation_training(self):
        changed = replace(self.dataset, targets=self.dataset.targets[::-1].copy())
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            rerun = run_masked_learning(changed, self.settings)
        reference = self.result["runs"]["gait_targets"]
        for name, tensor in reference["model"].state_dict().items():
            torch.testing.assert_close(tensor, rerun["model"].state_dict()[name], rtol=0, atol=0)

    def test_variant_update_budget_cannot_change(self):
        with self.assertRaises(ValueError):
            run_matched_comparison(self.dataset, self.settings, variants={"wrong": {"steps": 100}})

    def test_fold_mismatch_rejected(self):
        with self.assertRaises(ValueError):
            run_masked_learning(self.dataset, replace(self.settings, fold=1))

    def test_positive_symmetry_weight_requires_callback(self):
        with self.assertRaises(ValueError):
            run_masked_learning(self.dataset, replace(self.settings, symmetry_weight=1))

    def test_source_metadata_covers_the_fitted_partition(self):
        partition = self.result["runs"]["gait_targets"]["source_partition"]
        self.assertEqual(set(partition["train_sources"]), set(self.dataset.train_sources))
        self.assertEqual(set(partition["test_sources"]), set(self.dataset.test_sources))

    def test_cross_fitted_scoring_retains_source_weighting(self):
        run = self.result["runs"]["gait_targets"]
        frame = run["predictions"].assign(fold=0, seed=7, variant="gait_targets")
        scores = summarize_cross_fitted_predictions(frame).set_index("representation")
        expected = run["metrics"].set_index("representation")
        np.testing.assert_allclose(scores.loc[expected.index, "r2"], expected.r2)
        np.testing.assert_allclose(scores.loc[expected.index, "mae"], expected.mae)

    def test_repeated_test_sequences_rejected(self):
        frame = self.result["runs"]["gait_targets"]["predictions"].assign(fold=0, seed=7, variant="gait_targets")
        with self.assertRaises(ValueError):
            summarize_cross_fitted_predictions(pd.concat([frame, frame], ignore_index=True))

    def test_different_seed_coverage_rejected(self):
        frame = self.result["runs"]["gait_targets"]["predictions"].assign(fold=0, seed=7, variant="gait_targets")
        partial = frame[frame.sequence_id != frame.sequence_id.iloc[0]].assign(seed=8)
        with self.assertRaises(ValueError):
            summarize_cross_fitted_predictions(pd.concat([frame, partial], ignore_index=True))


if __name__ == "__main__":
    unittest.main()
