"""Optional source integration, enabled by FI_TEST_VJEPA_ROOT on HAIC/local QA."""

import os
import sys
import unittest
from pathlib import Path

import numpy as np
import torch

from gavd6_sjepa.research_directions.future_innovation.fi_contracts import FrameContract
from gavd6_sjepa.research_directions.future_innovation.fi_vjepa_adapter import (
    FrozenVJEPAAdapter,
)


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
