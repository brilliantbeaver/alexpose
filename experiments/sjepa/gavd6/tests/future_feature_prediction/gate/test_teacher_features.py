"""future feature prediction / gate / test teacher features."""


import os
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np
import pandas as pd
from torch import nn
import torch

from gavd6_sjepa.research_directions.future_prediction.contracts import FrameContract
from gavd6_sjepa.research_directions.future_prediction.controls import (
    block_shuffle,
    controlled_history,
    mismatch_donors,
)
from gavd6_sjepa.research_directions.future_prediction.nested_training import (
    isolated_split,
    partition_control,
)
from gavd6_sjepa.research_directions.future_prediction.residual_models import SkeletonResidualHead
from gavd6_sjepa.research_directions.future_prediction.token_regions import (
    context_indices,
    fixed_projection,
    pool_context,
    pool_target,
    region_masks,
    token_index,
)
from gavd6_sjepa.research_directions.future_prediction.validity_audits import (
    future_pixel_leakage_test,
)
from gavd6_sjepa.research_directions.future_prediction.video_pose import CropGeometry
from gavd6_sjepa.research_directions.future_prediction.vjepa import FrozenVJEPAAdapter
from gavd6_sjepa.shared_infrastructure.artifact_io import sha256_file

class MixingBlock(nn.Module):
    def forward(self, x):
        return x + x.mean(dim=1, keepdim=True)

class TinyEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.patch_embed = nn.Module()
        self.patch_embed.proj = nn.Conv3d(3, 8, (2, 16, 16), stride=(2, 16, 16))
        self.blocks = nn.ModuleList([MixingBlock()])

    def forward(self, x, masks=None):
        x = self.patch_embed.proj(x).flatten(2).transpose(1, 2)
        if masks is not None:
            x = torch.gather(x, 1, masks.unsqueeze(-1).expand(-1, -1, x.shape[-1]))
        return self.blocks[0](x)

def test_processor(frames):
    pixels = CropGeometry(*frames[0].shape[:2], resolution=64).pixels(frames)
    tensor = torch.from_numpy(pixels).permute(3, 0, 1, 2).float() / 255
    return [
        (tensor - torch.tensor([0.485, 0.456, 0.406])[:, None, None, None])
        / torch.tensor([0.229, 0.224, 0.225])[:, None, None, None]
    ]

class FutureInnovationCausalityTests(unittest.TestCase):
    def test_mask_before_attention_future_invariance_and_geometry(self):
        torch.set_num_threads(1)
        torch.manual_seed(1)
        adapter = FrozenVJEPAAdapter(
            TinyEncoder(), test_processor, frame=FrameContract(resolution=64)
        )
        video = np.random.default_rng(1).integers(
            0, 256, (64, 72, 96, 3), dtype=np.uint8
        )
        self.assertLessEqual(adapter.verify_geometry(video), 1e-6)
        result = future_pixel_leakage_test(adapter, video, np.random.default_rng(4))
        self.assertTrue(result["passed"])
        self.assertEqual(result["max_abs_difference"], 0)
        edited = video.copy()
        edited[32:] = 0
        self.assertFalse(
            np.array_equal(
                adapter.encode_full_target(video), adapter.encode_full_target(edited)
            )
        )
        with self.assertRaises(ValueError):
            adapter.encode_past_context(video, np.arange(512))

    def test_intentionally_leaky_adapter_is_detected(self):
        class Leaky:
            def encode_past_context(self, video):
                return np.array([video.mean()], dtype=float)

        video = np.zeros((64, 4, 4, 3), dtype=np.uint8)
        self.assertFalse(
            future_pixel_leakage_test(Leaky(), video, np.random.default_rng(1))[
                "passed"
            ]
        )


class FutureInnovationControlTests(unittest.TestCase):
    def test_shuffle_preserves_joint_channels_and_four_frame_blocks(self):
        history = np.arange(32 * 33 * 4).reshape(32, 33, 4)
        result = block_shuffle(history, "w")
        self.assertEqual(result.shape, history.shape)
        self.assertFalse(np.array_equal(result, history))
        np.testing.assert_array_equal(result, block_shuffle(history, "w"))
        self.assertEqual(
            {tuple(row.flatten()) for row in result.reshape(8, 4, 33, 4)},
            {tuple(row.flatten()) for row in history.reshape(8, 4, 33, 4)},
        )

    def test_mismatch_is_partition_local_different_source_and_without_replacement(self):
        sources = np.array(["a", "a", "b", "b", "c", "c"])
        metadata = np.arange(12).reshape(6, 2)
        donors, replacement = mismatch_donors(sources, metadata)
        self.assertFalse(replacement)
        self.assertEqual(len(set(donors)), 6)
        self.assertFalse(np.any(sources == sources[donors]))
        with self.assertRaises(ValueError):
            mismatch_donors(["a", "a"], metadata[:2])
        cohort = pd.DataFrame(
            {"window_id": [f"w{i}" for i in range(6)], "video_id": sources}
        )
        arrays = {
            "skeleton": np.arange(6 * 32 * 33 * 4).reshape(6, 32, 33, 4),
            "matching": metadata,
        }
        audit = []
        partition_control(
            "clip-mismatch", np.array([0, 1, 2, 3]), cohort, arrays, "train", audit
        )
        self.assertLessEqual(
            set(audit[0]["donor_window_ids"]), {"w0", "w1", "w2", "w3"}
        )
        with self.assertRaises(ValueError):
            isolated_split(cohort, np.array([0, 2, 4]), np.array([1, 3, 5]))

    def test_validity_only_keeps_shape_and_parameter_count(self):
        history = np.ones((4, 32, 33, 4))
        result, _ = controlled_history(
            "no-skeleton", history, list("abcd"), list("abcd"), np.ones((4, 2))
        )
        np.testing.assert_array_equal(result[..., :3], 0)
        np.testing.assert_array_equal(result[..., 3], history[..., 3])
        counts = [
            sum(p.numel() for p in SkeletonResidualHead(132, 32).parameters())
            for _ in range(5)
        ]
        self.assertEqual(len(set(counts)), 1)


# Optional source integration, enabled by FI_TEST_VJEPA_ROOT on HAIC/local QA.


@unittest.skipUnless(
    os.environ.get("FI_TEST_VJEPA_ROOT"),
    "Set FI_TEST_VJEPA_ROOT to the reviewed V-JEPA source for integration testing",
)
class FutureInnovationOfficialAdapterTests(unittest.TestCase):
    def test_reviewed_hub_builds_base_encoder_and_predictor_without_downloads(self):
        torch.set_num_threads(1)
        encoder, predictor = torch.hub.load(
            os.environ["FI_TEST_VJEPA_ROOT"],
            "vjepa2_1_vit_base_384",
            source="local",
            pretrained=False,
        )
        self.assertEqual(encoder.embed_dim, 768)
        self.assertEqual(encoder.num_frames, 64)
        self.assertEqual(len(encoder.blocks), 12)
        self.assertIsNone(encoder.out_layers)
        self.assertFalse(encoder.return_hierarchical)
        self.assertIsNotNone(predictor)

    def test_official_mask_order_preprocessor_and_final_layer(self):
        root = Path(os.environ["FI_TEST_VJEPA_ROOT"])
        sys.path.insert(0, str(root))
        try:
            from app.vjepa_2_1.models.vision_transformer import VisionTransformer
            from evals.hub.preprocessor import vjepa2_preprocessor

            torch.set_num_threads(1)
            torch.manual_seed(1)
            encoder = VisionTransformer(
                img_size=(32, 32),
                patch_size=16,
                num_frames=64,
                tubelet_size=2,
                embed_dim=48,
                depth=12,
                num_heads=4,
                use_rope=True,
                use_sdpa=True,
                img_temporal_dim_size=1,
                interpolate_rope=True,
            )
            frame = FrameContract(resolution=32)
            adapter = FrozenVJEPAAdapter(
                encoder, vjepa2_preprocessor(crop_size=32), frame=frame
            )
            rng = np.random.default_rng(1)
            video = rng.integers(0, 256, (64, 48, 64, 3), dtype=np.uint8)
            self.assertLessEqual(adapter.verify_geometry(video), 1e-6)
            past = adapter.encode_past_context(video)
            edited = video.copy()
            edited[32:] = rng.integers(0, 256, edited[32:].shape, dtype=np.uint8)
            np.testing.assert_array_equal(past, adapter.encode_past_context(edited))
            self.assertEqual(past.shape, (16 * 4, 48))
            self.assertEqual(adapter.encode_full_target(video).shape, (32 * 4, 48))
            # Actual official PatchEmbed3D flatten order, not just a mock adapter.
            with torch.no_grad():
                conv = encoder.patch_embed.proj
                conv.weight.zero_()
                conv.bias.zero_()
                conv.weight[0, 0] = 1 / (2 * 16 * 16)
                pixels = torch.zeros(1, 3, 64, 32, 32)
                for t in range(32):
                    for row in range(2):
                        for col in range(2):
                            pixels[
                                0,
                                0,
                                2 * t : 2 * t + 2,
                                row * 16 : (row + 1) * 16,
                                col * 16 : (col + 1) * 16,
                            ] = t * 4 + row * 2 + col
                np.testing.assert_allclose(
                    encoder.patch_embed(pixels)[0, :, 0].numpy(), np.arange(128)
                )
        finally:
            sys.path.remove(str(root))


class FutureInnovationProjectionTests(unittest.TestCase):
    def test_scaled_orthogonal_determinism_and_checksum(self):
        projection = fixed_projection(768)
        # Accumulate in float64 so this checks the saved matrix, not float32 BLAS rounding.
        np.testing.assert_allclose(
            projection.astype(float).T @ projection.astype(float),
            np.eye(256) * 3,
            atol=2e-7,
        )
        np.testing.assert_array_equal(projection, fixed_projection(768))
        with tempfile.TemporaryDirectory() as directory:
            a, b = Path(directory) / "a.npy", Path(directory) / "b.npy"
            np.save(a, projection)
            np.save(b, fixed_projection(768))
            self.assertEqual(sha256_file(a), sha256_file(b))
        with self.assertRaises(ValueError):
            fixed_projection(128)


class FutureInnovationTokenTests(unittest.TestCase):
    def test_temporal_major_pooling_known_indices(self):
        tokens = np.arange(32 * 24 * 24, dtype=float)[:, None]
        boxes = np.tile([0.4, 0.3, 0.6, 0.7], (64, 1))
        person, background = region_masks(boxes[0])
        pooled_person, pooled_background = pool_target(tokens, boxes)
        target = tokens.reshape(32, 576, 1)[19]
        np.testing.assert_array_equal(pooled_person, target[person].mean(axis=0))
        np.testing.assert_array_equal(
            pooled_background, target[background].mean(axis=0)
        )
        self.assertEqual(token_index(19, 4, 5), 19 * 576 + 4 * 24 + 5)
        pools = pool_context(tokens[context_indices()], boxes)
        self.assertAlmostEqual(pools[0], np.arange(16 * 576).mean())
        self.assertAlmostEqual(
            pools[1], tokens.reshape(32, 576, 1)[15, person].mean(), places=3
        )
        self.assertFalse((person & background).any())
        self.assertGreater((~person & ~background).sum(), 0)

    def test_empty_person_and_background_fail(self):
        with self.assertRaises(ValueError):
            region_masks([0.001, 0.001, 0.002, 0.002])
        with self.assertRaises(ValueError):
            region_masks([0, 0, 1, 1])


if __name__ == "__main__":
    unittest.main()
