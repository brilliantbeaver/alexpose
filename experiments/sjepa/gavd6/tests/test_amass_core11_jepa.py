from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Subset

from gavd6_sjepa.amass_core11_jepa import (
    CHANNEL_NAMES,
    JOINT_NAMES,
    MIRROR_CHANNEL,
    MIRROR_PAIRS,
    Core11WindowDataset,
    build_window_index,
    configure_worker_tensor_sharing,
    core11_train_config,
    evaluate_variant,
    fit_variant,
    load_conversion_manifest,
    make_train_loader,
    make_synthetic_core11_datasets,
    mirror_validity,
    validate_archives,
    window_starts,
)
from gavd6_sjepa.gait_parity_jepa import (
    VARIANTS,
    VICRegProjector,
    anatomical_mirror,
    build_model,
    complete_commutation_audit,
    lift_orbit,
    sample_mask,
    sjepa_distribution_metrics,
    trainable_parameter_count,
)
from gavd6_sjepa.train_amass import main as train_amass_main


class AmassCore11JepaTests(unittest.TestCase):
    def _fixture(self, root: Path) -> Path:
        rows = []
        for index, split in enumerate(("train", "validation", "test")):
            frames = 68
            relative = f"dataset/sequence-{index}_core11.npz"
            path = root / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            coordinates = np.random.default_rng(index).normal(
                0, 0.1, (frames, 11, 3)
            ).astype(np.float32)
            valid = np.ones((frames, 11), dtype=bool)
            np.savez_compressed(
                path,
                coordinates=coordinates,
                valid=valid,
                joint_names=np.asarray(JOINT_NAMES),
                channel_names=np.asarray(CHANNEL_NAMES),
            )
            rows.append(
                {
                    "tensor_relative_path": relative,
                    "identity": f"dataset::subject-{index}",
                    "split": split,
                    "canonical_fps": 30.0,
                    "canonical_frames": frames,
                }
            )
        manifest_path = root / "manifest.csv"
        pd.DataFrame(rows).to_csv(manifest_path, index=False)
        return manifest_path

    @staticmethod
    def _model_inputs(config):
        generator = torch.Generator().manual_seed(91)
        coordinates = torch.randn(
            2, config.frames, config.joints, 3, generator=generator
        )
        valid = torch.ones(
            2,
            config.frames // config.segment_length,
            config.joints,
            dtype=torch.bool,
        )
        target_mask = sample_mask(
            valid,
            config.mask_fraction,
            torch.Generator().manual_seed(92),
            config.mask_joints,
        )
        return coordinates, target_mask

    def test_frozen_window_policy(self):
        self.assertEqual(window_starts(63), [])
        self.assertEqual(window_starts(64), [0])
        self.assertEqual(window_starts(96), [0, 32])
        self.assertEqual(window_starts(97), [0, 32, 33])

    def test_core11_mirror_is_an_involution(self):
        coordinates = torch.randn(2, 64, 11, 3)
        mirrored = anatomical_mirror(coordinates, MIRROR_PAIRS, MIRROR_CHANNEL)
        restored = anatomical_mirror(mirrored, MIRROR_PAIRS, MIRROR_CHANNEL)
        torch.testing.assert_close(restored, coordinates)
        swap = torch.tensor([0, 2, 1, 4, 3, 6, 5, 8, 7, 10, 9])
        torch.testing.assert_close(
            mirrored[..., 0], coordinates[..., 0].index_select(2, swap)
        )
        torch.testing.assert_close(
            mirrored[..., 1], coordinates[..., 1].index_select(2, swap)
        )
        torch.testing.assert_close(
            mirrored[..., 2], -coordinates[..., 2].index_select(2, swap)
        )

        valid = torch.zeros(1, 16, 11, dtype=torch.bool)
        valid[..., 1] = True
        mirrored_valid = mirror_validity(valid)
        self.assertTrue(mirrored_valid[..., 2].all())
        self.assertFalse(mirrored_valid[..., 1].any())

    def test_loader_contract_and_identity_disjoint_splits(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            manifest = load_conversion_manifest(self._fixture(root))
            windows = build_window_index(manifest)
            self.assertEqual(validate_archives(manifest, root), 3)
            identities = {
                split: set(manifest.loc[manifest.split == split, "identity"])
                for split in ("train", "validation", "test")
            }
            self.assertFalse(identities["train"] & identities["validation"])
            self.assertFalse(identities["train"] & identities["test"])
            dataset = Core11WindowDataset(manifest, windows, root, split="train")
            sample = dataset[0]
            self.assertEqual(tuple(sample["coordinates"].shape), (64, 11, 3))
            self.assertEqual(tuple(sample["valid"].shape), (16, 11))

    def test_multi_worker_loader_bounds_descriptor_pressure(self):
        datasets = make_synthetic_core11_datasets()
        config = core11_train_config("smoke")
        prior_strategy = torch.multiprocessing.get_sharing_strategy()
        try:
            loader = make_train_loader(
                datasets["train"], config, updates=1, seed=7, num_workers=1
            )
            self.assertEqual(
                torch.multiprocessing.get_sharing_strategy(), "file_system"
            )
            self.assertFalse(loader.persistent_workers)
            self.assertEqual(loader.prefetch_factor, 1)
            self.assertEqual(configure_worker_tensor_sharing(), "file_system")
        finally:
            torch.multiprocessing.set_sharing_strategy(prior_strategy)

    def test_variant_names_capacity_and_paired_predictor_contracts(self):
        self.assertEqual(
            VARIANTS,
            [
                "paired_shared_no_cross",
                "reflection_equivariant",
                "paired_unconstrained",
            ],
        )
        config = core11_train_config("smoke")
        counts = []
        embedding_dims = []
        for variant in VARIANTS:
            model = build_model(config, variant, seed=7)
            embedding_dims.append(model.config.embed_dim)
            projector = VICRegProjector(model.config.embed_dim)
            counts.append(trainable_parameter_count(model, projector))
            coordinates, target_mask = self._model_inputs(model.config)
            orbit = lift_orbit(
                coordinates, model.config.mirror_pairs, model.config.mirror_channel
            )
            state = model.encoder(orbit, keep_mask=~target_mask)
            predicted = model.predictor(state, target_mask)
            self.assertEqual(predicted.shape[:2], (2, 2))
            self.assertEqual(predicted.shape[-1], model.config.embed_dim)
            self.assertEqual(predicted.shape[2], int(target_mask[0].sum()))

            restored = model.predictor.restore_branch(state[0], target_mask)
            positions = (
                model.predictor.time_pos[:, None] + model.predictor.joint_pos[None]
            ).reshape(1, -1, model.config.embed_dim)
            mask = target_mask.flatten(1)
            torch.testing.assert_close(
                (restored - positions)[~mask], state[0].flatten(0, 1)
            )
            expected_mask = model.predictor.mask_token.expand_as(restored)
            torch.testing.assert_close(
                (restored - positions)[mask], expected_mask[mask]
            )

            paired_layers = list(model.encoder.layers) + list(model.predictor.layers)
            gates = [
                float(layer.cross_scale.detach())
                for layer in paired_layers
                if hasattr(layer, "cross_scale")
            ]
            if variant == "paired_shared_no_cross":
                self.assertEqual(gates, [])
            else:
                self.assertTrue(gates)
                self.assertTrue(all(abs(value - 0.1) < 1e-7 for value in gates))
        self.assertLess(max(counts) / min(counts) - 1, 0.05)
        self.assertEqual(len(set(embedding_dims)), 1)

    def test_complete_swap_commutation_contract(self):
        config = core11_train_config("smoke")
        audits = {}
        for variant in VARIANTS:
            model = build_model(config, variant, seed=11)
            coordinates, target_mask = self._model_inputs(model.config)
            summary, layers = complete_commutation_audit(
                model, coordinates, target_mask, device=torch.device("cpu")
            )
            audits[variant] = summary
            self.assertEqual(set(layers.module), {"online_encoder", "target_encoder"})
        for variant in ("paired_shared_no_cross", "reflection_equivariant"):
            self.assertLess(max(audits[variant].values()), 1e-5)
        self.assertGreater(max(audits["paired_unconstrained"].values()), 1e-4)

    def test_cross_entropy_entropy_kl_decomposition(self):
        predicted = torch.randn(3, 2, 5, 17)
        targets = torch.randn(3, 2, 5, 17)
        center = torch.randn(17)
        cross_entropy, entropy, kl = sjepa_distribution_metrics(
            predicted, targets, center
        )
        torch.testing.assert_close(cross_entropy, entropy + kl, rtol=1e-5, atol=1e-6)
        self.assertGreaterEqual(float(kl), -1e-6)

    def test_evaluation_is_deterministic_and_does_not_mutate_state(self):
        datasets = make_synthetic_core11_datasets()
        model = build_model(core11_train_config("smoke"), "reflection_equivariant", 7)
        projector = VICRegProjector(model.config.embed_dim)
        before_model = {name: value.clone() for name, value in model.state_dict().items()}
        before_projector = {
            name: value.clone() for name, value in projector.state_dict().items()
        }
        first, _ = evaluate_variant(
            model,
            projector,
            datasets["validation"],
            torch.device("cpu"),
            seed=7,
            split="validation",
        )
        second, _ = evaluate_variant(
            model,
            projector,
            datasets["validation"],
            torch.device("cpu"),
            seed=7,
            split="validation",
        )
        self.assertEqual(first, second)
        for name, value in model.state_dict().items():
            torch.testing.assert_close(value, before_model[name])
        for name, value in projector.state_dict().items():
            torch.testing.assert_close(value, before_projector[name])

    def test_validation_selected_checkpoint_survives_reload(self):
        datasets = make_synthetic_core11_datasets()
        train = Subset(datasets["train"], range(8))
        validation = datasets["validation"]
        model = build_model(
            core11_train_config("smoke"), "reflection_equivariant", seed=7
        )
        with tempfile.TemporaryDirectory() as temporary:
            best_model, best_projector, history, result, _ = fit_variant(
                model,
                train,
                validation,
                torch.device("cpu"),
                seed=7,
                output_dir=Path(temporary),
                num_workers=0,
                max_epochs=2,
                patience=2,
            )
            required = {
                "train_jepa_cross_entropy",
                "train_teacher_entropy",
                "train_kl_divergence",
                "train_even_invariance",
                "train_even_variance",
                "train_even_covariance",
                "train_odd_invariance",
                "train_odd_variance",
                "train_odd_covariance",
                "validation_jepa_cross_entropy",
                "validation_teacher_entropy",
                "validation_kl_divergence",
                "validation_even_effective_rank",
                "validation_odd_mean_pairwise_cosine",
            }
            self.assertTrue(required.issubset(history.columns))
            eligible = history.loc[history.eligible]
            selected = int(
                eligible.loc[eligible.validation_kl_divergence.idxmin(), "epoch"]
            )
            self.assertEqual(result["selected_epoch"], selected)
            self.assertTrue(Path(result["best_checkpoint"]).is_file())
            self.assertEqual(
                {path.name for path in Path(temporary).iterdir()},
                {
                    "seed-7_reflection_equivariant_best.pt",
                    "seed-7_reflection_equivariant_history.csv",
                },
            )
            persisted_history = pd.read_csv(
                Path(temporary) / "seed-7_reflection_equivariant_history.csv"
            )
            pd.testing.assert_frame_equal(
                persisted_history,
                history,
                check_dtype=False,
            )
            reloaded_metrics, _ = evaluate_variant(
                best_model,
                best_projector,
                validation,
                torch.device("cpu"),
                seed=7,
                split="validation",
            )
            self.assertAlmostEqual(
                reloaded_metrics["kl_divergence"],
                result["validation"]["kl_divergence"],
                places=7,
            )

    def test_completed_epoch_history_survives_later_interruption(self):
        datasets = make_synthetic_core11_datasets()
        train = Subset(datasets["train"], range(8))
        model = build_model(
            core11_train_config("smoke"), "paired_unconstrained", seed=7
        )
        evaluations = 0

        def interrupt_second_evaluation(*args, **kwargs):
            nonlocal evaluations
            evaluations += 1
            if evaluations == 2:
                raise RuntimeError("simulated scheduler interruption")
            return evaluate_variant(*args, **kwargs)

        with tempfile.TemporaryDirectory() as temporary:
            history_path = (
                Path(temporary) / "seed-7_paired_unconstrained_history.csv"
            )
            with patch(
                "gavd6_sjepa.amass_core11_jepa.evaluate_variant",
                side_effect=interrupt_second_evaluation,
            ):
                with self.assertRaisesRegex(RuntimeError, "scheduler interruption"):
                    fit_variant(
                        model,
                        train,
                        datasets["validation"],
                        torch.device("cpu"),
                        seed=7,
                        output_dir=Path(temporary),
                        num_workers=0,
                        max_epochs=2,
                        patience=2,
                    )

            persisted = pd.read_csv(history_path)
            self.assertEqual(persisted.epoch.tolist(), [1])
            self.assertFalse(history_path.with_suffix(".csv.tmp").exists())

    def test_training_entrypoint_writes_only_compact_artifact_contract(self):
        with tempfile.TemporaryDirectory() as temporary:
            output_dir = Path(temporary)
            environment = {
                "AMASS_RUN_TRAINING": "1",
                "AMASS_PROFILE": "smoke",
                "AMASS_SYNTHETIC_SMOKE": "1",
                "AMASS_DEVICE": "cpu",
                "AMASS_NUM_WORKERS": "0",
                "AMASS_SEEDS": "7,19",
                "AMASS_EVALUATE_TEST": "1",
                "AMASS_OUTPUT_DIR": str(output_dir),
            }
            with patch.dict(os.environ, environment, clear=True):
                train_amass_main()

            expected = {"summary.csv", "run_config.json", "capacity.csv"}
            for seed in (7, 19):
                for variant in VARIANTS:
                    expected.add(f"seed-{seed}_{variant}_history.csv")
                    expected.add(f"seed-{seed}_{variant}_best.pt")
            self.assertEqual({path.name for path in output_dir.iterdir()}, expected)


if __name__ == "__main__":
    unittest.main()
