from __future__ import annotations

import unittest
from dataclasses import asdict, replace
from pathlib import Path
from types import SimpleNamespace
import tempfile

import numpy as np
import torch
from torch import nn

from gavd6_sjepa.research_directions.reflection_equivariance.gavd_core11_probe_evaluation import (
    AdapterConfig,
    adapt_mediapipe_sequence,
    build_probe_windows,
    frozen_target_encoder,
    nested_ridge_probes,
    parity_sequence_features,
    raw_coordinate_features,
    validity_only_features,
)
from gavd6_sjepa.research_directions.reflection_equivariance.amass_core11_training_pipeline import MIRROR_PAIRS
from gavd6_sjepa.research_directions.reflection_equivariance.jepa_model_architecture import (
    OrbitJEPA,
    PROFILES,
    VICRegProjector,
    save_checkpoint,
)


ASPECT_RATIO = 16.0 / 9.0


def synthetic_mediapipe_walk(frames: int = 70, fps: float = 30.0) -> tuple[np.ndarray, float]:
    """Create an upright pseudo-3D MediaPipe walk moving toward image right."""

    sequence = np.zeros((frames, 33, 4), dtype=np.float32)
    sequence[..., 3] = 1.0
    travel = np.linspace(0.25, 0.55, frames, dtype=np.float32)
    pelvis = np.stack(
        [travel, np.zeros(frames, dtype=np.float32), np.full(frames, 0.75, dtype=np.float32)],
        axis=1,
    )
    offsets = {
        23: [0.00, 0.05, 0.00],
        24: [0.00, -0.05, 0.00],
        25: [0.01, 0.05, -0.13],
        26: [-0.01, -0.05, -0.13],
        27: [0.02, 0.05, -0.27],
        28: [-0.02, -0.05, -0.27],
        29: [-0.01, 0.05, -0.29],
        30: [-0.04, -0.05, -0.29],
        31: [0.07, 0.05, -0.29],
        32: [0.04, -0.05, -0.29],
    }
    for landmark, offset in offsets.items():
        world = pelvis + np.asarray(offset, dtype=np.float32)
        sequence[:, landmark, 0] = world[:, 0]
        sequence[:, landmark, 1] = -world[:, 2] * ASPECT_RATIO
        sequence[:, landmark, 2] = world[:, 1]
    return sequence, fps


class AdapterTests(unittest.TestCase):
    def test_body_frame_matches_core11_channel_and_laterality_contract(self):
        sequence, fps = synthetic_mediapipe_walk()
        adapted = adapt_mediapipe_sequence(
            sequence, source_fps=fps, aspect_ratio=ASPECT_RATIO
        )
        coordinates = adapted["coordinates"]
        self.assertEqual(coordinates.shape, (70, 11, 3))
        np.testing.assert_allclose(coordinates[:, 0], 0.0, atol=1e-6)
        self.assertGreater(float(coordinates[:, 1, 2].mean()), 0.0)
        self.assertLess(float(coordinates[:, 2, 2].mean()), 0.0)
        self.assertLess(float(coordinates[:, 5, 1].mean()), 0.0)
        self.assertTrue(adapted["frame_method"].startswith("pelvis_travel"))

    def test_24_fps_is_resampled_before_short_clip_padding(self):
        sequence, _ = synthetic_mediapipe_walk(frames=49, fps=24.0)
        adapted = adapt_mediapipe_sequence(
            sequence, source_fps=24.0, aspect_ratio=ASPECT_RATIO
        )
        self.assertEqual(len(adapted["coordinates"]), 61)
        record = {
            **adapted,
            "sequence_id": "short",
            "video_id": "video",
            "condition": "normal",
        }
        windows, valid, sequence_indices, table = build_probe_windows([record])
        self.assertEqual(windows.shape, (1, 64, 11, 3))
        self.assertEqual(int(table.iloc[0]["observed_frames"]), 61)
        self.assertFalse(valid[0, -11:].any())
        self.assertEqual(sequence_indices.tolist(), [0])

    def test_orbit_token_validity_requires_both_bilateral_partners(self):
        sequence, fps = synthetic_mediapipe_walk(frames=64, fps=30.0)
        sequence[:4, 23, 3] = 0.0
        config = AdapterConfig(maximum_gap_frames=0)
        adapted = adapt_mediapipe_sequence(
            sequence,
            source_fps=fps,
            aspect_ratio=ASPECT_RATIO,
            config=config,
        )
        record = {
            **adapted,
            "sequence_id": "paired",
            "video_id": "video",
            "condition": "normal",
        }
        windows, token_valid, _, _ = build_probe_windows([record], config=config)
        first_patch = token_valid.reshape(1, 16, 11)[0, 0]
        self.assertFalse(first_patch[1])
        self.assertFalse(first_patch[2])
        features = raw_coordinate_features(
            windows, token_valid, np.asarray([0]), sequence_count=1
        )
        self.assertEqual(features.shape, (1, 66))
        self.assertTrue(np.isfinite(features).all())
        validity_features = validity_only_features(
            token_valid, np.asarray([0]), sequence_count=1
        )
        self.assertEqual(validity_features.shape, (1, 176))
        np.testing.assert_array_equal(validity_features[0], token_valid[0])


class ProbeTests(unittest.TestCase):
    def test_parity_pooling_order_is_even_mean_std_then_odd_mean_std(self):
        class FakeEncoder(nn.Module):
            def __init__(self):
                super().__init__()
                self.config = SimpleNamespace(
                    embed_dim=2,
                    mirror_pairs=MIRROR_PAIRS,
                    mirror_channel=2,
                )

            def forward(self, orbit):
                marker = orbit[:, 0, 0, 0, 0]
                first = torch.stack([marker, 2 * marker], dim=1)[:, None].expand(-1, 176, -1)
                second = torch.stack([-marker, torch.zeros_like(marker)], dim=1)[:, None].expand(-1, 176, -1)
                return first, second

        windows = np.zeros((2, 64, 11, 3), dtype=np.float32)
        windows[:, 0, 0, 0] = [1.0, 3.0]
        token_valid = np.zeros((2, 176), dtype=bool)
        token_valid[:, 0] = True
        features = parity_sequence_features(
            FakeEncoder(),
            windows,
            token_valid,
            np.asarray([0, 0]),
            sequence_count=1,
            batch_size=2,
        )
        np.testing.assert_allclose(
            features[0],
            [0.0, 2.0, 0.0, 1.0, 2.0, 2.0, 1.0, 1.0],
            atol=1e-6,
        )

    def test_checkpoint_loader_selects_and_does_not_mutate_ema_target(self):
        variant = "paired_shared_no_cross"
        config = replace(
            PROFILES["smoke"],
            frames=64,
            stride=32,
            segment_length=4,
            embed_dim=8,
            heads=2,
            feedforward_dim=16,
            joints=11,
            mask_joints=tuple(range(11)),
            mirror_pairs=MIRROR_PAIRS,
            mirror_channel=2,
            capacity_variant=variant,
        )
        model = OrbitJEPA(config, variant)
        with torch.no_grad():
            next(model.encoder.parameters()).add_(10.0)
        projector = VICRegProjector(config.embed_dim)
        metadata = {"variant": variant, "seed": 7, "train_config": asdict(config)}
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "checkpoint.pt"
            save_checkpoint(path, model, projector, metadata)
            encoder, loaded_metadata = frozen_target_encoder(path)
        self.assertEqual(loaded_metadata["variant"], variant)
        self.assertFalse(encoder.training)
        self.assertFalse(any(parameter.requires_grad for parameter in encoder.parameters()))
        for name, value in encoder.state_dict().items():
            np.testing.assert_array_equal(value.numpy(), model.target_encoder.state_dict()[name].numpy())
        before = {name: value.clone() for name, value in encoder.state_dict().items()}
        features = parity_sequence_features(
            encoder,
            np.zeros((1, 64, 11, 3), dtype=np.float32),
            np.ones((1, 176), dtype=bool),
            np.asarray([0]),
            sequence_count=1,
        )
        self.assertEqual(features.shape, (1, 32))
        for name, value in encoder.state_dict().items():
            torch.testing.assert_close(value, before[name], rtol=0.0, atol=0.0)

    def test_nested_probe_runs_identical_outer_folds_and_reports_dimensions(self):
        rng = np.random.default_rng(4)
        labels = np.repeat(["a", "b", "c"], 6)
        videos = np.asarray([f"video_{index // 2}" for index in range(len(labels))])
        feature_sets = {
            "small": rng.normal(size=(18, 3)),
            "large": rng.normal(size=(18, 7)),
        }
        folds, summary, predictions = nested_ridge_probes(
            feature_sets,
            labels,
            videos,
            seed=3,
            outer_splits=3,
            inner_splits=2,
            alphas=(0.1, 1.0),
        )
        self.assertEqual(len(folds), 6)
        self.assertEqual(len(predictions), 36)
        dimensions = summary.set_index("representation")["feature_dimension"].to_dict()
        self.assertEqual(dimensions, {"small": 3, "large": 7})
        first = predictions[predictions.representation == "small"].sort_values("row_index")
        second = predictions[predictions.representation == "large"].sort_values("row_index")
        self.assertEqual(first.row_index.tolist(), second.row_index.tolist())


if __name__ == "__main__":
    unittest.main()
