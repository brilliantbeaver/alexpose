from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import torch

from gavd6_sjepa.research_directions.reflection_equivariance.amass_core11_training_pipeline import (
    MIRROR_CHANNEL,
    MIRROR_PAIRS,
    SyntheticCore11Dataset,
    core11_train_config,
)
from gavd6_sjepa.research_directions.reflection_equivariance.jepa_model_architecture import build_model
from gavd6_sjepa.research_directions.reflection_equivariance.swap_probe_evaluation_pipeline import (
    CorruptionConfig,
    _fit_path_temperature,
    _structured_path_nll_numpy,
    _temperature_scale,
    anchored_path_from_edges,
    apply_corruption,
    boundary_features,
    build_draws,
    candidate_prior,
    candidate_paths,
    edge_metric_row,
    edge_labels,
    kinematic_targets,
    make_window_draw,
    map_candidate_path,
    posterior_edge_marginals,
    run_probe,
    select_nonoverlapping_windows,
    semantic_permute_by_block,
    semantic_permute_patch_validity,
    sha256_file,
    sensor_reflect,
    split_calibration_identities,
    structured_path_posterior,
    structured_path_nll,
    tolerant_event_f1,
    validate_encoder_contract,
)


class BilateralSemanticCorrectionProbeTests(unittest.TestCase):
    @staticmethod
    def _rows(split: str, size: int) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "window_id": [f"{split}:{index}" for index in range(size)],
                "identity": [f"identity::{split}::{index}" for index in range(size)],
                "tensor_relative_path": [f"synthetic/{split}/{index}" for index in range(size)],
                "start_frame": [0] * size,
                "split": [split] * size,
            }
        )

    def test_semantic_permutation_and_sensor_reflection_are_distinct_involutions(self):
        coordinates = torch.arange(2 * 64 * 11 * 3, dtype=torch.float32).reshape(
            2, 64, 11, 3
        )
        path = torch.ones(2, 16, dtype=torch.bool)
        permuted = semantic_permute_by_block(coordinates, path)
        restored = semantic_permute_by_block(permuted, path)
        torch.testing.assert_close(restored, coordinates)

        bits = torch.ones(2, dtype=torch.bool)
        reflected = sensor_reflect(coordinates, bits)
        torch.testing.assert_close(sensor_reflect(reflected, bits), coordinates)
        self.assertFalse(torch.equal(permuted, reflected))
        torch.testing.assert_close(
            sensor_reflect(permuted, bits),
            semantic_permute_by_block(reflected, path),
        )

    def test_oracle_inverse_recovers_matched_nuisance_reference_exactly(self):
        generator = torch.Generator().manual_seed(11)
        clean = torch.randn(3, 64, 11, 3, generator=generator)
        valid = torch.ones(3, 16, 11, dtype=torch.bool)
        draws = [make_window_draw(f"window:{index}", 7, CorruptionConfig()) for index in range(3)]
        path = torch.tensor([draw.semantic_path for draw in draws], dtype=torch.bool)
        sensor = torch.tensor([draw.sensor_reflection_bit for draw in draws], dtype=torch.bool)
        occlusion = torch.tensor([draw.occlusion_keep for draw in draws], dtype=torch.bool)
        observed, observed_valid, reference, reference_valid = apply_corruption(
            clean,
            valid,
            path,
            sensor,
            occlusion,
            [draw.noise_seed for draw in draws],
            CorruptionConfig(),
        )
        restored = semantic_permute_by_block(observed, path)
        restored_valid = semantic_permute_by_block(
            observed_valid.repeat_interleave(4, dim=1), path
        ).reshape(3, 16, 4, 11).all(dim=2)
        self.assertTrue(torch.equal(restored, reference))
        self.assertTrue(torch.equal(restored_valid, reference_valid))

    def test_nuisance_randomness_is_domain_separated_from_event_length(self):
        short = make_window_draw(
            "same-window", 7, CorruptionConfig(swapped_blocks=2, event_probability=0.999)
        )
        long = make_window_draw(
            "same-window", 7, CorruptionConfig(swapped_blocks=6, event_probability=0.999)
        )
        self.assertEqual(short.sensor_reflection_bit, long.sensor_reflection_bit)
        self.assertEqual(short.noise_seed, long.noise_seed)
        self.assertEqual(short.occlusion_keep, long.occlusion_keep)
        self.assertNotEqual(short.semantic_path, long.semantic_path)

    def test_frozen_contract_rejects_nonfour_frame_blocks(self):
        with self.assertRaisesRegex(ValueError, "exactly four"):
            CorruptionConfig(block_frames=8).validate()

    def test_draws_include_matched_clean_no_switch_examples(self):
        config = CorruptionConfig(event_probability=0.50)
        paths = [
            make_window_draw(f"window:{index}", 17, config).semantic_path
            for index in range(100)
        ]
        self.assertTrue(any(not any(path) for path in paths))
        self.assertTrue(any(any(path) for path in paths))

    def test_kinematic_targets_have_declared_parity(self):
        generator = torch.Generator().manual_seed(21)
        coordinates = torch.randn(5, 64, 11, 3, generator=generator)
        valid = torch.rand(5, 16, 11, generator=generator) > 0.10
        valid[..., 0:5] = True
        path = torch.ones(5, 16, dtype=torch.bool)
        odd, even, usable = kinematic_targets(coordinates, valid)
        swapped_coordinates = semantic_permute_by_block(coordinates, path)
        swapped_valid = semantic_permute_by_block(
            valid.repeat_interleave(4, dim=1), path
        ).reshape(5, 16, 4, 11).all(dim=2)
        swapped_odd, swapped_even, swapped_usable = kinematic_targets(
            swapped_coordinates, swapped_valid
        )
        self.assertTrue(torch.equal(usable, swapped_usable))
        torch.testing.assert_close(swapped_odd[usable], -odd[usable])
        torch.testing.assert_close(swapped_even[usable], even[usable])

    def test_local_swap_preserves_even_target_under_asymmetric_validity(self):
        generator = torch.Generator().manual_seed(23)
        coordinates = torch.randn(2, 64, 11, 3, generator=generator)
        valid = torch.ones(2, 16, 11, dtype=torch.bool)
        valid[0, 4, 5] = False
        valid[0, 9, 8] = False
        valid[1, 3, 10] = False
        path = torch.zeros(2, 16, dtype=torch.bool)
        path[:, 5:9] = True
        swapped_coordinates = semantic_permute_by_block(coordinates, path)
        swapped_valid = semantic_permute_patch_validity(valid, path)
        odd, even, usable = kinematic_targets(coordinates, valid, block_mask=path)
        swapped_odd, swapped_even, swapped_usable = kinematic_targets(
            swapped_coordinates, swapped_valid, block_mask=path
        )
        self.assertTrue(torch.equal(usable, swapped_usable))
        torch.testing.assert_close(swapped_odd[usable], -odd[usable])
        torch.testing.assert_close(swapped_even[usable], even[usable])

    def test_relative_features_are_invariant_to_global_semantic_flip(self):
        generator = torch.Generator().manual_seed(31)
        coordinates = torch.randn(4, 64, 11, 3, generator=generator)
        valid = torch.rand(4, 16, 11, generator=generator) > 0.15
        global_path = torch.ones(4, 16, dtype=torch.bool)
        flipped = semantic_permute_by_block(coordinates, global_path)
        flipped_valid = semantic_permute_by_block(
            valid.repeat_interleave(4, dim=1), global_path
        ).reshape(4, 16, 4, 11).all(dim=2)
        original_features = boundary_features(coordinates, valid)
        flipped_features = boundary_features(flipped, flipped_valid)
        for original, transformed in zip(original_features, flipped_features):
            torch.testing.assert_close(original, transformed)

    def test_continuity_margin_identifies_a_synthetic_swap(self):
        coordinates = torch.zeros(1, 64, 11, 3)
        for frame in range(64):
            coordinates[0, frame, :, 0] = frame * 0.01
            for left, right in ((1, 2), (3, 4), (5, 6), (7, 8), (9, 10)):
                coordinates[0, frame, left, 2] = -1.0 - 0.1 * left
                coordinates[0, frame, right, 2] = 1.0 + 0.1 * left
        path = torch.zeros(1, 16, dtype=torch.bool)
        path[:, 6:10] = True
        observed = semantic_permute_by_block(coordinates, path)
        features, _, _, _ = boundary_features(
            observed, torch.ones(1, 16, 11, dtype=torch.bool)
        )
        margin = features[0, :, -3]
        labels = edge_labels(path)[0]
        self.assertTrue((margin[labels] > 0).all())
        self.assertTrue((margin[~labels] < 0).all())

    def test_aggregate_continuity_margin_uses_all_five_pairs(self):
        coordinates = torch.zeros(1, 64, 11, 3)
        for frame in range(64):
            coordinates[0, frame, :, 0] = frame * 0.01
            for pair_index, (left, right) in enumerate(((1, 2), (3, 4), (5, 6), (7, 8), (9, 10))):
                lateral = 0.0 if pair_index == 0 else 1.0 + pair_index
                coordinates[0, frame, left, 2] = -lateral
                coordinates[0, frame, right, 2] = lateral
        path = torch.zeros(1, 16, dtype=torch.bool)
        path[:, 4:8] = True
        features, _, _, _ = boundary_features(
            semantic_permute_by_block(coordinates, path),
            torch.ones(1, 16, 11, dtype=torch.bool),
        )
        labels = edge_labels(path)[0]
        self.assertTrue(torch.allclose(features[0, labels, 2], torch.zeros(2)))
        self.assertTrue((features[0, labels, -3] > 0).all())

    def test_all_invalid_boundary_features_are_finite_and_zero_coverage(self):
        coordinates = torch.zeros(2, 64, 11, 3)
        valid = torch.zeros(2, 16, 11, dtype=torch.bool)
        features, same, swapped, coverage = boundary_features(coordinates, valid)
        self.assertTrue(torch.isfinite(features).all())
        self.assertTrue(torch.equal(same, torch.zeros_like(same)))
        self.assertTrue(torch.equal(swapped, torch.zeros_like(swapped)))
        self.assertTrue(torch.equal(coverage, torch.zeros_like(coverage)))

    def test_edge_integration_and_structured_posterior_support(self):
        edges = torch.tensor([[0, 1, 0], [1, 1, 0]], dtype=torch.bool)
        root = torch.tensor([0, 1], dtype=torch.bool)
        path = anchored_path_from_edges(edges, root)
        self.assertTrue(torch.equal(edge_labels(path), edges))

        config = CorruptionConfig()
        candidates = candidate_paths(config)
        self.assertEqual(tuple(candidates.shape), (12, 16))
        target = candidates[6:7]
        probabilities = torch.where(edge_labels(target), 0.999, 0.001)
        posterior = structured_path_posterior(
            probabilities,
            candidates,
            config.event_probability,
            edge_prior=0.80 * 2 / 15,
        )
        recovered = map_candidate_path(posterior, candidates)
        self.assertTrue(torch.equal(recovered, target))
        self.assertTrue(torch.equal(edge_labels(recovered), edge_labels(target)))
        marginals = posterior_edge_marginals(posterior, candidates)
        self.assertTrue(torch.isfinite(marginals).all())

    def test_structured_posterior_maps_zero_edge_evidence_to_clean_path(self):
        config = CorruptionConfig()
        candidates = candidate_paths(config)
        posterior = structured_path_posterior(
            torch.full((2, 15), 1e-7),
            candidates,
            config.event_probability,
            edge_prior=0.80 * 2 / 15,
        )
        recovered = map_candidate_path(posterior, candidates)
        self.assertTrue(torch.equal(recovered, candidates[0:1].expand(2, -1)))

    def test_structured_posterior_applies_the_path_prior_once(self):
        config = CorruptionConfig()
        candidates = candidate_paths(config)
        edge_prior = config.event_probability * 2 / 15
        posterior = structured_path_posterior(
            torch.full((3, 15), edge_prior),
            candidates,
            config.event_probability,
            edge_prior=edge_prior,
        )
        expected = candidate_prior(candidates, config.event_probability).expand(3, -1)
        torch.testing.assert_close(posterior, expected)

    def test_path_temperature_is_fit_against_held_out_complete_paths(self):
        config = CorruptionConfig()
        candidates = candidate_paths(config)
        edge_prior = config.event_probability * 2 / 15
        targets = torch.cat((candidates[0:1], candidates[1:4]), dim=0)
        raw = np.full((len(targets), 15), 0.08)
        for row, target in enumerate(targets):
            raw[row, edge_labels(target[None]).numpy().ravel()] = 0.35
        coverage = np.ones_like(raw)
        temperature = _fit_path_temperature(
            raw,
            targets,
            coverage,
            candidates,
            config.event_probability,
            edge_prior=edge_prior,
        )
        baseline = _structured_path_nll_numpy(
            raw,
            targets,
            candidates,
            config.event_probability,
            edge_prior=edge_prior,
        )
        calibrated = _structured_path_nll_numpy(
            _temperature_scale(raw, temperature, edge_prior),
            targets,
            candidates,
            config.event_probability,
            edge_prior=edge_prior,
        )
        self.assertGreater(temperature, 0.0)
        self.assertLessEqual(calibrated, baseline + 1e-8)
        with self.assertRaisesRegex(ValueError, "clean and event"):
            _fit_path_temperature(
                raw[0:1],
                targets[0:1],
                coverage[0:1],
                candidates,
                config.event_probability,
                edge_prior=edge_prior,
            )

    def test_temperature_scaling_preserves_the_edge_prior(self):
        config = CorruptionConfig()
        candidates = candidate_paths(config)
        edge_prior = config.event_probability * 2 / 15
        for temperature in (0.05, 0.50, 1.0, 20.0):
            probabilities = _temperature_scale(
                np.full((2, 15), edge_prior), temperature, edge_prior
            )
            posterior = structured_path_posterior(
                torch.from_numpy(probabilities.astype(np.float32)),
                candidates,
                config.event_probability,
                edge_prior=edge_prior,
            )
            expected = candidate_prior(candidates, config.event_probability).expand(2, -1)
            torch.testing.assert_close(posterior, expected)

    def test_structured_path_nll_does_not_cap_confident_wrong_paths(self):
        config = CorruptionConfig()
        candidates = candidate_paths(config)
        nll = structured_path_nll(
            torch.full((1, 15), 1.0 - 1e-7),
            candidates[0:1],
            candidates,
            config.event_probability,
            edge_prior=config.event_probability * 2 / 15,
        )
        self.assertGreater(float(nll), 16.2)

    def test_anchored_metrics_do_not_hide_a_global_flip(self):
        config = CorruptionConfig()
        candidates = candidate_paths(config)
        target = candidates[1:2]
        wrong = ~target
        log_posterior = torch.full((1, len(candidates)), -torch.inf)
        log_posterior[:, 0] = 0.0
        row = edge_metric_row(
            edge_labels(target).numpy().astype(np.int64).ravel(),
            np.full(15, 0.1),
            wrong,
            target,
            "test",
            structured_log_posterior=log_posterior,
            candidates=candidates,
        )
        self.assertEqual(row["path_hamming_anchored"], 1.0)
        self.assertEqual(row["path_hamming_up_to_flip"], 0.0)

    def test_tolerant_event_matching_is_maximal_not_nearest_greedy(self):
        predicted = torch.tensor([[0, 1, 1]], dtype=torch.bool)
        target = torch.tensor([[1, 1, 0]], dtype=torch.bool)
        self.assertEqual(float(tolerant_event_f1(predicted, target, tolerance=1)[0]), 1.0)

    def test_calibration_identities_are_disjoint_and_exhaustive(self):
        rows = self._rows("train", 10)
        fit, calibration = split_calibration_identities(rows, seed=7)
        self.assertFalse(fit & calibration)
        self.assertEqual(fit | calibration, set(rows.identity))
        self.assertTrue(calibration)

    def test_calibration_identity_split_preserves_required_path_support(self):
        rows = self._rows("train", 16)
        draws = build_draws(
            rows,
            seed=7,
            config=CorruptionConfig(event_probability=0.50),
        )
        fit, calibration = split_calibration_identities(
            rows,
            seed=7,
            draws=draws,
        )

        support = {
            identity: draws[window_id].event_present
            for identity, window_id in zip(rows.identity, rows.window_id)
        }
        self.assertEqual({support[identity] for identity in calibration}, {False, True})
        self.assertIn(True, {support[identity] for identity in fit})
        self.assertEqual(len(calibration), 3)
        self.assertFalse(fit & calibration)
        self.assertEqual(fit | calibration, set(rows.identity))

    def test_frozen_contract_rejects_incorrect_bilateral_metadata(self):
        encoder = SimpleNamespace(
            config=SimpleNamespace(
                frames=64,
                segment_length=4,
                joints=11,
                mirror_pairs=MIRROR_PAIRS,
                mirror_channel=MIRROR_CHANNEL,
            )
        )
        validate_encoder_contract(encoder)
        encoder.config.mirror_pairs = ((1, 2),)
        with self.assertRaisesRegex(ValueError, "bilateral-pair"):
            validate_encoder_contract(encoder)
        encoder.config.mirror_pairs = MIRROR_PAIRS
        encoder.config.mirror_channel = (MIRROR_CHANNEL + 1) % 3
        with self.assertRaisesRegex(ValueError, "mediolateral"):
            validate_encoder_contract(encoder)

    def test_nonoverlapping_selection_is_stable_and_has_no_shared_frames(self):
        rows = []
        for sequence in range(2):
            for start in (0, 32, 64, 96, 128):
                rows.append(
                    {
                        "window_id": f"{sequence}:{start}",
                        "sequence_index": sequence,
                        "tensor_relative_path": f"sequence-{sequence}.npz",
                        "identity": f"identity-{sequence}",
                        "split": "train",
                        "start_frame": start,
                    }
                )
        index = pd.DataFrame(rows)
        first = select_nonoverlapping_windows(index, "train", None, 7)
        second = select_nonoverlapping_windows(index, "train", None, 7)
        pd.testing.assert_frame_equal(first, second)
        self.assertTrue(first.sequence_index.is_monotonic_increasing)
        for _, group in first.groupby("tensor_relative_path"):
            starts = sorted(group["start_frame"])
            self.assertTrue(all(right - left >= 64 for left, right in zip(starts, starts[1:])))

    def test_end_to_end_synthetic_smoke_writes_complete_artifacts(self):
        train_size, validation_size = 16, 16
        train = SyntheticCore11Dataset("train", train_size, 101)
        validation = SyntheticCore11Dataset("validation", validation_size, 202)
        model = build_model(core11_train_config("smoke"), "standard_sjepa", seed=7)
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "probe"
            result = run_probe(
                train_dataset=train,
                train_rows=self._rows("train", train_size),
                validation_dataset=validation,
                validation_rows=self._rows("validation", validation_size),
                encoder=model.target_encoder,
                device=torch.device("cpu"),
                output_dir=output,
                corruption=CorruptionConfig(event_probability=0.50),
                seed=7,
                batch_size=4,
                encoder_batch_size=8,
                num_workers=0,
                provenance={"synthetic_smoke": True},
            )
            self.assertEqual(set(result["summary"]["arm"]), {
                "clean_reference",
                "corrupted_uncorrected",
                "continuity_map",
                "learned_map",
                "learned_structured_posterior",
                "oracle",
            })
            expected = {
                "COMPLETE.json",
                "corruption_manifest.csv",
                "effective_config.json",
                "lightweight_models.joblib",
                "summary.csv",
                "validation_edge_metrics.csv",
                "validation_prior_sensitivity.csv",
                "validation_reliability.csv",
                "validation_condition_summary.csv",
                "validation_uncertainty.csv",
                "validation_window_metrics.csv",
            }
            self.assertEqual({path.name for path in output.iterdir()}, expected)
            metrics = pd.read_csv(output / "validation_window_metrics.csv")
            self.assertEqual(len(metrics), validation_size * 6)
            self.assertEqual(set(metrics["condition"]), {"clean_no_switch", "swapped_event"})
            self.assertTrue((metrics.loc[metrics.arm == "oracle", "embedding_cosine_distance_to_nuisance_reference"].abs() < 1e-6).all())
            self.assertEqual(
                set(pd.read_csv(output / "corruption_manifest.csv")["event_present"]),
                {False, True},
            )
            complete = json.loads((output / "COMPLETE.json").read_text())
            self.assertEqual(complete["status"], "complete")
            self.assertEqual(set(complete["artifact_sha256"]), expected - {"COMPLETE.json"})
            for name, digest in complete["artifact_sha256"].items():
                self.assertEqual(digest, sha256_file(output / name))
            effective_config = json.loads((output / "effective_config.json").read_text())
            self.assertEqual(
                effective_config["path_temperature_calibration"]["objective"],
                "held-out complete-path negative log likelihood",
            )
            sensitivity = pd.read_csv(output / "validation_prior_sensitivity.csv")
            self.assertEqual(
                set(sensitivity["evaluation_event_probability"]), {0.2, 0.5}
            )


if __name__ == "__main__":
    unittest.main()
