import unittest

import numpy as np
import torch
from torch import nn

from gavd6_sjepa.research_directions.future_innovation.fi_contracts import FrameContract
from gavd6_sjepa.research_directions.future_innovation.fi_validity_audits import (
    future_pixel_leakage_test,
)
from gavd6_sjepa.research_directions.future_innovation.fi_video_pose import CropGeometry
from gavd6_sjepa.research_directions.future_innovation.fi_vjepa_adapter import (
    FrozenVJEPAAdapter,
)


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
