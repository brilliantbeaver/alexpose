from __future__ import annotations

import hashlib
import json
import unittest
from dataclasses import replace
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace

import numpy as np
import pandas as pd
import torch

from gavd6_sjepa.research_directions.reflection_equivariance.amass_core11_training_pipeline import (
    BalancedGroupBatchPlan,
    MIRROR_CHANNEL,
    MIRROR_PAIRS,
    checkpoint_payload,
    core11_train_config,
)
from gavd6_sjepa.research_directions.latent_laterality.laterality_gauge_training_pipeline import semantic_gauge_objective
from gavd6_sjepa.research_directions.latent_laterality.laterality_gauge_evaluation_pipeline import GaugeWindows, _encoder_features, evaluate
from gavd6_sjepa.research_directions.reflection_equivariance.jepa_model_architecture import (
    VICRegProjector,
    build_model,
    lift_orbit,
    orbit_closed_target_masks,
    permute_bilateral_tokens,
)
from gavd6_sjepa.research_directions.reflection_equivariance.gavd_core11_probe_evaluation import AdapterConfig, _body_frame
from gavd6_sjepa.research_directions.latent_laterality.laterality_corruption_inference import (
    SequenceGaugeConfig,
    TwoStateDurationModel,
    apply_sequence_draw,
    generate_sequence_draw,
    path_hamming_up_to_global_flip,
    run_length_decode,
    semantic_permute,
    sequence_gauge_config_json,
    sensor_reflect,
    slice_corrupted_windows,
    structured_parity_prediction_loss,
)
from gavd6_sjepa.research_directions.latent_laterality.laterality_sequence_benchmarking import (
    load_manifest_sequence_config,
    make_manifest_examples,
    one_example_per_chart_pair,
    run_sequence_benchmark,
)
from gavd6_sjepa.research_directions.latent_laterality.laterality_study_protocol import ARM_SPECS, require_benchmark_gate, source_screen_jobs
from gavd6_sjepa.research_directions.latent_laterality.laterality_manifest_construction import build as build_amass_gauge_manifest
from gavd6_sjepa.data_foundations.amass_core11_conversion_pipeline import (
    ConversionConfig,
    convert_sequence_arrays,
    estimate_forward_frame,
)


def synthetic_sequence(frames: int = 128) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    time = np.arange(frames, dtype=np.float32)
    coordinates = np.zeros((frames, 11, 3), dtype=np.float32)
    pelvis_world = np.stack(
        [0.02 * time, 0.15 * np.sin(time / 23.0), np.zeros_like(time)], axis=1
    )
    coordinates[:, 0] = pelvis_world
    for pair_index, (left, right) in enumerate(MIRROR_PAIRS):
        phase = time / (5.0 + pair_index) + 0.2 * pair_index
        coordinates[:, left, 0] = pelvis_world[:, 0] + 0.04 * np.sin(phase)
        coordinates[:, right, 0] = pelvis_world[:, 0] + 0.04 * np.sin(phase + np.pi)
        coordinates[:, left, 1] = 0.8 - 0.12 * pair_index
        coordinates[:, right, 1] = 0.8 - 0.12 * pair_index
        coordinates[:, left, 2] = 0.12 + 0.01 * pair_index
        coordinates[:, right, 2] = -0.12 - 0.01 * pair_index
    valid = np.ones((frames, 11), dtype=bool)
    return coordinates, valid, pelvis_world


class GaugeNeutralFrameTests(unittest.TestCase):
    def test_amass_orientation_never_needs_named_side_joints(self):
        coordinates, valid, _ = synthetic_sequence()
        # Make the pelvis the only valid token. A side-reading orientation rule
        # cannot pass this fixture.
        pelvis_only = np.zeros_like(valid)
        pelvis_only[:, 0] = True
        frame = estimate_forward_frame(
            coordinates,
            pelvis_only,
            min_travel_m=0.1,
            min_travel_straightness=0.1,
            min_abs_lateral_hip_alignment=1.0,
            forward_policy="gauge-neutral-travel",
        )
        swapped = coordinates.copy()
        swapped[:, 1:] = swapped[:, 1:][:, ::-1]
        other = estimate_forward_frame(
            swapped,
            pelvis_only,
            min_travel_m=0.1,
            min_travel_straightness=0.1,
            min_abs_lateral_hip_alignment=1.0,
            forward_policy="gauge-neutral-travel",
        )
        np.testing.assert_allclose(frame.world_to_body_transform, other.world_to_body_transform)
        self.assertEqual(
            frame.method,
            "gauge_neutral_pelvis_travel_pca_signed_by_net_displacement",
        )
        self.assertIsNone(frame.lateral_hip_alignment)

    def test_global_token_relabeling_does_not_change_amass_physical_frame(self):
        coordinates, valid, _ = synthetic_sequence()
        config = ConversionConfig(
            min_travel_m=0.1,
            min_travel_straightness=0.1,
            forward_policy="gauge-neutral-travel",
        )
        first = convert_sequence_arrays(
            coordinates, valid, source_fps=30.0, config=config
        )
        second = convert_sequence_arrays(
            semantic_permute(coordinates),
            semantic_permute(valid),
            source_fps=30.0,
            config=config,
        )
        np.testing.assert_allclose(
            first["frame"].world_to_body_transform,
            second["frame"].world_to_body_transform,
        )
        np.testing.assert_allclose(
            semantic_permute(first["coordinates"]), second["coordinates"], atol=1e-6
        )

    def test_gavd_travel_frame_is_side_permutation_neutral(self):
        coordinates, valid, _ = synthetic_sequence()
        config = AdapterConfig(
            minimum_travel_leg_lengths=0.01,
            frame_policy="gauge-neutral-travel-or-image",
        )
        first, method = _body_frame(coordinates, valid, 1.0, config)
        second, other_method = _body_frame(
            semantic_permute(coordinates), semantic_permute(valid), 1.0, config
        )
        np.testing.assert_allclose(first, second)
        self.assertEqual(method, other_method)
        self.assertTrue(method.startswith("gauge_neutral"))

    def test_gavd_static_fallback_declares_image_chart(self):
        coordinates, valid, _ = synthetic_sequence(16)
        coordinates[:, 0] = 0.0
        config = AdapterConfig(frame_policy="gauge-neutral-travel-or-image")
        first, method = _body_frame(coordinates, valid, 1.0, config)
        second, other_method = _body_frame(
            semantic_permute(coordinates), semantic_permute(valid), 1.0, config
        )
        np.testing.assert_array_equal(first, second)
        self.assertEqual(method, "declared_image_space_unanchored")
        self.assertEqual(other_method, method)


class TypedInputTests(unittest.TestCase):
    def test_sensor_semantic_and_global_chart_actions_are_distinct(self):
        coordinates, _, _ = synthetic_sequence(8)
        semantic = semantic_permute(coordinates)
        sensor = sensor_reflect(coordinates, 1)
        np.testing.assert_allclose(semantic[..., MIRROR_CHANNEL], semantic_permute(coordinates)[..., MIRROR_CHANNEL])
        np.testing.assert_allclose(sensor[..., :MIRROR_CHANNEL], coordinates[..., :MIRROR_CHANNEL])
        np.testing.assert_allclose(sensor[..., MIRROR_CHANNEL], -coordinates[..., MIRROR_CHANNEL])
        self.assertFalse(np.allclose(semantic, sensor))

    def test_validity_is_an_explicit_encoder_input_and_invalid_targets_fail(self):
        model = build_model(core11_train_config("smoke"), "standard_sjepa", seed=7)
        tokenizer = model.encoder.tokenizer
        with torch.no_grad():
            tokenizer.validity_embedding.weight[0].fill_(-1.0)
            tokenizer.validity_embedding.weight[1].fill_(1.0)
        coordinates = torch.zeros(1, model.config.frames, model.config.joints, 3)
        valid = torch.ones(1, tokenizer.segments, model.config.joints, dtype=torch.bool)
        invalid = valid.clone()
        invalid[:, 0, 0] = False
        valid_tokens = tokenizer(coordinates, valid)
        invalid_tokens = tokenizer(coordinates, invalid)
        self.assertFalse(torch.equal(valid_tokens[:, 0, 0], invalid_tokens[:, 0, 0]))
        target = torch.zeros_like(valid)
        target[:, 0, 0] = True
        with self.assertRaisesRegex(ValueError, "Invalid tokens"):
            model(coordinates, target, valid_patch=invalid)

    def test_paired_validity_and_masks_follow_the_same_permutation(self):
        model = build_model(
            core11_train_config("smoke"), "reflection_equivariant", seed=7
        )
        segments = model.config.frames // model.config.segment_length
        coordinates = torch.randn(1, model.config.frames, 11, 3)
        valid = torch.ones(1, segments, 11, dtype=torch.bool)
        valid[:, 2, 1] = False
        paired_valid = torch.stack(
            [valid, permute_bilateral_tokens(valid, MIRROR_PAIRS)], dim=1
        )
        target = torch.zeros_like(valid)
        target[:, 2, 2] = True
        paired_target = orbit_closed_target_masks(target, MIRROR_PAIRS)
        orbit = lift_orbit(coordinates, MIRROR_PAIRS, MIRROR_CHANNEL)
        predicted, targets = model(orbit, paired_target, valid_patch=paired_valid)
        self.assertEqual(predicted.shape, targets.shape)
        self.assertTrue(torch.equal(paired_valid[:, 1], permute_bilateral_tokens(paired_valid[:, 0], MIRROR_PAIRS)))
        self.assertTrue(torch.equal(paired_target[:, 1], permute_bilateral_tokens(paired_target[:, 0], MIRROR_PAIRS)))


class SequenceBenchmarkTests(unittest.TestCase):
    def test_paired_manifest_has_identical_observations_and_one_effective_draw(self):
        coordinates, valid, pelvis = synthetic_sequence()
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            tensor_root = root / "tensors"
            tensor_root.mkdir()
            np.savez(
                tensor_root / "sequence.npz",
                coordinates=coordinates,
                valid=valid,
                pelvis_world_m=pelvis,
            )
            source_manifest = root / "source.csv"
            pd.DataFrame(
                [
                    {
                        "tensor_relative_path": "sequence.npz",
                        "identity": "participant-1",
                        "split": "train",
                        "canonical_frames": len(coordinates),
                        "coordinate_frame": "gauge-neutral-travel-v1",
                        "status": "converted",
                        "motion_id": "reused-readable-name",
                    }
                ]
            ).to_csv(source_manifest, index=False)
            output_manifest = root / "gauge-v2.csv"
            config = SequenceGaugeConfig(
                boundary_mode="gap", boundary_radius_frames=1
            )
            written = build_amass_gauge_manifest(
                source_manifest,
                tensor_root,
                output_manifest,
                draws=1,
                seed=7,
                config=config,
                chart_pairs=True,
            )
            self.assertEqual(len(written), 2)
            self.assertEqual(written.source_chart_bit.tolist(), [0, 1])
            self.assertEqual(written.latent_chart_bit.tolist(), [0, 1])
            self.assertEqual(
                written.sequence_gauge_config.iloc[0], sequence_gauge_config_json(config)
            )
            self.assertEqual(load_manifest_sequence_config(output_manifest), config)
            examples = make_manifest_examples(output_manifest, tensor_root, config=config)
            self.assertEqual(len(examples), 2)
            np.testing.assert_array_equal(
                examples[0].corrupted["coordinates"], examples[1].corrupted["coordinates"]
            )
            np.testing.assert_array_equal(
                examples[0].corrupted["valid"], examples[1].corrupted["valid"]
            )
            self.assertEqual(len(one_example_per_chart_pair(examples)), 1)
            self.assertEqual(
                len(
                    make_manifest_examples(
                        output_manifest,
                        tensor_root,
                        config=config,
                        chart_members=(0,),
                    )
                ),
                1,
            )

    def test_training_gate_rejects_a_decision_for_a_different_manifest(self):
        with TemporaryDirectory() as temporary:
            path = Path(temporary) / "gate.json"
            path.write_text(
                '{"ready_for_sg_jepa": true, "gauge_manifest_sha256": "a"}'
            )
            with self.assertRaisesRegex(RuntimeError, "different gauge manifest"):
                require_benchmark_gate(path, gauge_manifest_sha256="b")

    def test_common_readout_extracts_standard_and_semantic_orbit_features(self):
        coordinates, valid, _ = synthetic_sequence(64)
        windows = GaugeWindows(
            coordinates=np.stack([coordinates, coordinates]),
            valid=np.stack([valid[::4], valid[::4]]),
            targets=np.zeros((2, 2), dtype=np.float64),
            metadata=pd.DataFrame({"identity": ["a", "b"]}),
        )
        for variant in ("standard_sjepa", "reflection_equivariant"):
            model = build_model(core11_train_config("smoke"), variant, seed=7)
            features = _encoder_features(
                model, windows, batch_size=2, device=torch.device("cpu")
            )
            self.assertEqual(features.shape, (2, 2 * model.config.embed_dim))
            self.assertTrue(np.isfinite(features).all())

    def test_common_readout_is_test_only_and_binds_runs_to_the_gated_manifest(self):
        coordinates, valid, pelvis = synthetic_sequence(64)
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            tensor_root = root / "tensors"
            tensor_root.mkdir()
            source_rows = []
            for index, split in enumerate(
                ("train", "train", "train", "train", "train", "validation", "validation", "test", "test")
            ):
                relative = f"sequence-{index}.npz"
                np.savez(
                    tensor_root / relative,
                    coordinates=coordinates + index * 1e-4,
                    valid=valid,
                    pelvis_world_m=pelvis,
                )
                source_rows.append(
                    {
                        "tensor_relative_path": relative,
                        "identity": f"participant-{index}",
                        "split": split,
                        "canonical_frames": len(coordinates),
                        "coordinate_frame": "gauge-neutral-travel-v1",
                        "status": "converted",
                    }
                )
            source_manifest = root / "source.csv"
            pd.DataFrame(source_rows).to_csv(source_manifest, index=False)
            gauge_manifest = root / "gauge-v2.csv"
            config = SequenceGaugeConfig(
                clean_probability=0.0,
                global_probability=0.0,
                local_probability=1.0,
                repeated_probability=0.0,
                local_durations=(2,),
                boundary_mode="gap",
                boundary_radius_frames=1,
                nuisance_selection="independent",
            )
            build_amass_gauge_manifest(
                source_manifest,
                tensor_root,
                gauge_manifest,
                draws=1,
                seed=7,
                config=config,
                chart_pairs=True,
            )
            digest = hashlib.sha256(gauge_manifest.read_bytes()).hexdigest()
            gate_dir = root / "gate"
            gate = run_sequence_benchmark(
                gate_dir,
                seed=7,
                config=config,
                examples=make_manifest_examples(gauge_manifest, tensor_root, config=config),
                gauge_manifest_sha256=digest,
            )
            self.assertTrue(gate["ready_for_sg_jepa"], gate)
            run_dirs = []
            for arm, variant in (
                ("correction_first_sjepa", "standard_sjepa"),
                ("sg_jepa", "reflection_equivariant"),
                ("uniform_posterior", "reflection_equivariant"),
            ):
                run_dir = root / arm
                run_dir.mkdir()
                model = build_model(core11_train_config("smoke"), variant, seed=7)
                projector = VICRegProjector(model.config.embed_dim)
                optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
                checkpoint = run_dir / "best.pt"
                torch.save(
                    checkpoint_payload(
                        model,
                        projector,
                        optimizer,
                        variant=model.variant,
                        seed=7,
                        config=model.config,
                        updates=0,
                    ),
                    checkpoint,
                )
                (run_dir / "run_result.json").write_text(
                    json.dumps(
                        {
                            "arm": arm,
                            "gauge_manifest_sha256": digest,
                            "result": {"best_checkpoint": str(checkpoint)},
                        }
                    )
                )
                run_dirs.append(f"{arm}={run_dir}")
            output = root / "test-readout"
            summary = evaluate(
                SimpleNamespace(
                    gate_decision=gate_dir / "gate_decision.json",
                    gauge_manifest=gauge_manifest,
                    tensor_root=tensor_root,
                    run_dir=run_dirs,
                    output_dir=output,
                    mode="decisive",
                    split="test",
                    device="cpu",
                    batch_size=2,
                )
            )
            self.assertEqual(len(summary), 6)
            contract = json.loads((output / "evaluation_contract.json").read_text())
            self.assertTrue(contract["test_split_evaluated"])
            self.assertEqual(contract["evaluation_split"], "test")

    def test_every_path_family_and_overlap_contract(self):
        coordinates, valid, pelvis = synthetic_sequence()
        families = {
            "clean": (1.0, 0.0, 0.0, 0.0),
            "global_swap": (0.0, 1.0, 0.0, 0.0),
            "local_segment": (0.0, 0.0, 1.0, 0.0),
            "repeated_switches": (0.0, 0.0, 0.0, 1.0),
        }
        for expected, probabilities in families.items():
            config = SequenceGaugeConfig(
                clean_probability=probabilities[0],
                global_probability=probabilities[1],
                local_probability=probabilities[2],
                repeated_probability=probabilities[3],
                window_stride=32,
            )
            draw, path = generate_sequence_draw(
                coordinates,
                valid,
                sequence_id="sequence",
                identity="identity",
                split="train",
                corruption_draw=0,
                seed=7,
                config=config,
                pelvis_world=pelvis,
            )
            self.assertEqual(draw.path_family, expected)
            np.testing.assert_array_equal(run_length_decode(draw.gauge_path_rle), path)
            corrupted = apply_sequence_draw(coordinates, valid, draw, config=config)
            windows = slice_corrupted_windows(corrupted, config=config)
            self.assertGreaterEqual(len(windows), 2)
            np.testing.assert_array_equal(
                windows[0]["coordinates"][32:], windows[1]["coordinates"][:32]
            )
            np.testing.assert_array_equal(
                windows[0]["observed_frame_path"][32:],
                windows[1]["observed_frame_path"][:32],
            )

    def test_matched_interpolation_does_not_reveal_switches_through_validity(self):
        coordinates, valid, pelvis = synthetic_sequence()
        local = SequenceGaugeConfig(
            clean_probability=0.0,
            global_probability=0.0,
            local_probability=1.0,
            repeated_probability=0.0,
        )
        clean = replace(
            local,
            clean_probability=1.0,
            local_probability=0.0,
        )
        local_draw, _ = generate_sequence_draw(
            coordinates,
            valid,
            sequence_id="same",
            identity="id",
            split="validation",
            corruption_draw=0,
            seed=19,
            config=local,
            pelvis_world=pelvis,
        )
        clean_draw, _ = generate_sequence_draw(
            coordinates,
            valid,
            sequence_id="same",
            identity="id",
            split="validation",
            corruption_draw=0,
            seed=19,
            config=clean,
            pelvis_world=pelvis,
        )
        local_result = apply_sequence_draw(coordinates, valid, local_draw, config=local)
        clean_result = apply_sequence_draw(coordinates, valid, clean_draw, config=clean)
        np.testing.assert_array_equal(local_result["valid"], semantic_permute_by_chart(valid, local_result["observed_frame_path"]))
        np.testing.assert_array_equal(clean_result["valid"], semantic_permute_by_chart(valid, clean_result["observed_frame_path"]))
        self.assertEqual(len(local_draw.nuisance_boundary_frames), len(clean_draw.nuisance_boundary_frames))

    def test_duration_model_returns_map_and_calibrated_marginals_up_to_flip(self):
        model = TwoStateDurationModel(max_duration=8)
        target = np.asarray([0, 0, 0, 1, 1, 1, 0, 0], dtype=np.int8)
        edges = target[1:] != target[:-1]
        logits = np.where(edges, 8.0, -8.0)
        posterior = model.infer(logits)
        self.assertEqual(path_hamming_up_to_global_flip(posterior.map_path, target), 0.0)
        self.assertTrue(((posterior.block_swap_probability >= 0) & (posterior.block_swap_probability <= 1)).all())
        self.assertTrue(((posterior.edge_switch_probability >= 0) & (posterior.edge_switch_probability <= 1)).all())
        first = model.equivalence_path_nll(logits, target)
        second = model.equivalence_path_nll(logits, 1 - target)
        self.assertAlmostEqual(first, second, places=10)

    def test_structured_jepa_loss_detaches_correspondence_posterior(self):
        shape = (2, 4, 3)
        predicted_even = torch.randn(shape, requires_grad=True)
        predicted_odd = torch.randn(shape, requires_grad=True)
        target_even = torch.randn(shape)
        target_odd = torch.randn(shape)
        posterior = torch.full((2, 4), 0.3, requires_grad=True)
        loss = structured_parity_prediction_loss(
            predicted_even,
            predicted_odd,
            target_even,
            target_odd,
            posterior,
        )
        loss.backward()
        self.assertIsNone(posterior.grad)
        self.assertIsNotNone(predicted_even.grad)
        self.assertIsNotNone(predicted_odd.grad)

    def test_semantic_gauge_objective_runs_with_orbit_closed_masks(self):
        model = build_model(
            core11_train_config("smoke"), "reflection_equivariant", seed=7
        )
        coordinates = torch.randn(2, 64, 11, 3)
        teacher = coordinates.clone()
        valid = torch.ones(2, 16, 11, dtype=torch.bool)
        target = torch.zeros_like(valid)
        target[:, 3, 1] = True
        probability = torch.full((2, 16), 0.25, requires_grad=True)
        loss, terms = semantic_gauge_objective(
            model,
            coordinates,
            valid,
            teacher,
            valid,
            probability,
            target,
        )
        loss.backward()
        self.assertTrue(torch.isfinite(loss))
        self.assertIsNone(probability.grad)
        self.assertEqual(terms["even_features"].shape, (2, model.config.embed_dim))

    def test_common_arm_and_source_screen_contract_is_complete(self):
        required = {
            "raw_temporal",
            "standard_sjepa",
            "standard_mirror_aug",
            "paired_unconstrained",
            "reflection_equivariant",
            "correction_first_sjepa",
            "sg_jepa",
            "uniform_posterior",
            "oracle_correction",
            "raw_downstream",
            "random_encoder",
        }
        self.assertEqual(set(ARM_SPECS), required)
        jobs = source_screen_jobs(seed=7)
        self.assertEqual(len(jobs), 6)
        self.assertTrue(all(seed == 7 for _, _, seed in jobs))

    def test_source_video_balanced_batches_do_not_weight_prolific_groups(self):
        groups = ["prolific"] * 20 + ["small"]
        plan = BalancedGroupBatchPlan(groups, batch_size=200, updates=1, seed=7)
        batch = next(iter(plan))
        sampled = [groups[index] for index in batch]
        fraction = sampled.count("small") / len(sampled)
        self.assertGreater(fraction, 0.35)
        self.assertLess(fraction, 0.65)


def semantic_permute_by_chart(values: np.ndarray, frame_path: np.ndarray) -> np.ndarray:
    result = values.copy()
    selected = np.asarray(frame_path, dtype=bool)
    result[selected] = semantic_permute(result[selected])
    return result


if __name__ == "__main__":
    unittest.main()
