"""Context timing, feature semantics and frozen encoder pooling."""

from contextlib import nullcontext
from types import SimpleNamespace
import unittest

import numpy as np
import torch

from gavd6_sjepa.research_directions.synthetic_training.context_features import (
    ContextEncoder, ContextSpec, _clean_encoder_state, load_context_encoder,
    sample_clip,
)


class ContextTests(unittest.TestCase):
    def setUp(self):
        self.clip = np.stack([np.full((8, 12, 3), value, dtype=np.uint8)
                              for value in [0, 20, 40, 60, 80, 100]])

    def test_temporal_shuffle_changes_order_but_not_frame_exposure(self):
        original = sample_clip(self.clip, 4)
        shuffled = sample_clip(self.clip, 4, shuffle_seed=13)
        np.testing.assert_array_equal(np.sort(original[:, 0, 0, 0]), np.sort(shuffled[:, 0, 0, 0]))
        self.assertFalse(np.array_equal(original, shuffled))
        np.testing.assert_array_equal(shuffled, sample_clip(self.clip, 4, shuffle_seed=13))

    def test_simple_features_keep_dimension_and_measure_image_change(self):
        encoder = load_context_encoder(ContextSpec(kind="simple", frames=6), "cpu")
        dynamic = encoder.encode(self.clip)
        static = encoder.encode(np.repeat(self.clip[:1], 6, axis=0))
        self.assertEqual(dynamic.shape, (16,))
        self.assertEqual(dynamic.dtype, np.float32)
        self.assertTrue(np.isfinite(dynamic).all())
        np.testing.assert_allclose(dynamic[6:8], 0)
        np.testing.assert_allclose(dynamic[8:11], 20 / 255, atol=1e-5)
        np.testing.assert_allclose(static[8:14], 0)
        np.testing.assert_allclose(dynamic[-2:], np.log1p([12, 8]))

    def test_context_none_has_no_information(self):
        encoder = load_context_encoder(ContextSpec(kind="none", frames=4), "cpu")
        self.assertEqual(encoder.encode(self.clip).shape, (0,))

    def test_checkpoint_prefix_cleaning_is_not_arbitrary_inner_replacement(self):
        value = torch.tensor(2)
        state = _clean_encoder_state({"module.backbone.block.module.weight": value})
        self.assertEqual(list(state), ["block.module.weight"])
        with self.assertRaisesRegex(ValueError, "colliding"):
            _clean_encoder_state({"module.weight": value, "weight": value})

    def test_vjepa_uses_only_frozen_final_tokens(self):
        class EncoderFixture(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.scale = torch.nn.Parameter(torch.tensor(1.0))

            def forward(self, tensor):
                return torch.tensor([[[1., 2., 4.], [2., 1., 4.]]]) * self.scale

        model = EncoderFixture()
        spec = ContextSpec(kind="vjepa", frames=4, image_size=8)
        encoder = ContextEncoder(spec, "cpu", model=model,
                                 processor=lambda clip: [torch.zeros(3, 4, 8, 8)],
                                 author_repository=SimpleNamespace(activate=nullcontext))
        feature = encoder.encode(self.clip)
        self.assertEqual(feature.shape, (3,))
        self.assertAlmostEqual(float(np.linalg.norm(feature)), 1.0, places=6)
        self.assertFalse(model.training)
        self.assertFalse(model.scale.requires_grad)
        self.assertIsNone(model.scale.grad)

    def test_image_temporal_pooling_is_order_invariant(self):
        class ImageFixture(torch.nn.Module):
            def forward(self, tensor):
                mean = tensor.float().mean(dim=(1, 2, 3))
                return torch.stack([mean, mean.square()], dim=1)

        encoder = ContextEncoder(ContextSpec(kind="image", frames=4), "cpu",
                                 model=ImageFixture(), processor=lambda x: x)
        np.testing.assert_allclose(encoder.encode(self.clip), encoder.encode(self.clip, shuffle_seed=8), atol=1e-7)


if __name__ == "__main__":
    unittest.main()
