"""Small mathematical and gradient checks for the proposed symmetry experiment."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np
import torch

SUITE_ROOT = Path(__file__).resolve().parents[1]
if str(SUITE_ROOT) not in sys.path:
    sys.path.insert(0, str(SUITE_ROOT))

from laterality.model import SkeletonPatchEncoder
from laterality_extensions.symmetry_learning import (
    feature_variation,
    reflect_training_batch,
    reflection_permutation,
    token_equivariance_loss,
    token_reflection_error,
    token_symmetry_penalty,
    run_symmetry_comparison,
)


class ResearchSymmetryTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(7)
        self.tokens = torch.randn(3, 4, 33, 8)
        self.valid = torch.ones(3, 4, 33, dtype=torch.bool)
        self.permutation = reflection_permutation()

    def test_reflection_is_an_involution_for_coordinates_and_masks(self):
        xyz = torch.randn(3, 16, 33, 3)
        valid = self.valid.clone()
        valid[:, :, 11] = False
        hidden = torch.zeros_like(valid)
        hidden[:, 1, 12] = True
        mirrored, mirrored_valid, mirrored_hidden = reflect_training_batch(xyz, valid, hidden)
        restored, restored_valid, restored_hidden = reflect_training_batch(
            mirrored, mirrored_valid, mirrored_hidden
        )
        torch.testing.assert_close(restored, xyz)
        self.assertTrue(torch.equal(restored_valid, valid))
        self.assertTrue(torch.equal(restored_hidden, hidden))
        self.assertFalse((mirrored_hidden & ~mirrored_valid).any())
        self.assertTrue(mirrored_hidden[:, 1, 11].all())

    def test_invalid_targets_are_rejected(self):
        valid = self.valid.clone()
        valid[0, 0, 11] = False
        hidden = ~valid
        with self.assertRaises(ValueError):
            reflect_training_batch(torch.randn(3, 16, 33, 3), valid, hidden)

    def test_aligned_joint_swap_has_zero_error(self):
        expected = self.tokens[:, :, self.permutation]
        error = token_reflection_error(self.tokens, expected, self.valid, self.valid)
        torch.testing.assert_close(error, torch.zeros(3))

    def test_zero_features_cannot_pass_diagnostic(self):
        with self.assertRaises(ValueError):
            token_reflection_error(torch.zeros_like(self.tokens), torch.zeros_like(self.tokens), self.valid, self.valid)

    def test_invalid_placeholders_do_not_change_error(self):
        valid = self.valid.clone()
        valid[:, :, 11] = False
        mirror_valid = valid[:, :, self.permutation]
        original = self.tokens.clone()
        reflected = self.tokens[:, :, self.permutation] + 0.2
        before = token_reflection_error(original, reflected, valid, mirror_valid)
        original[~valid] = float("nan")
        reflected[~mirror_valid] = float("nan")
        after = token_reflection_error(original, reflected, valid, mirror_valid)
        torch.testing.assert_close(before, after)

    def test_loss_has_finite_gradients(self):
        original = self.tokens.clone().requires_grad_()
        reflected = torch.randn_like(original, requires_grad=True)
        loss = token_equivariance_loss(original, reflected, self.valid, self.valid)
        loss.backward()
        self.assertTrue(torch.isfinite(original.grad).all())
        self.assertTrue(torch.isfinite(reflected.grad).all())
        self.assertGreater(float(original.grad.abs().sum()), 0.0)

    def test_constant_features_show_why_symmetry_alone_is_inadequate(self):
        constant = torch.ones_like(self.tokens)
        error = token_reflection_error(constant, constant, self.valid, self.valid)
        self.assertEqual(float(error.max()), 0.0)
        summary = feature_variation(np.ones((3, 8)), np.array(["a", "b", "c"]))
        self.assertTrue(summary["constant_features"])
        self.assertEqual(summary["effective_rank"], 0.0)

    def test_variation_is_unchanged_by_duplicate_rows_within_source(self):
        features = np.array([[0.0, 1.0], [1.0, -1.0], [2.0, 4.0]])
        groups = np.array(["a", "b", "c"])
        expected = feature_variation(features, groups)
        duplicate = feature_variation(np.vstack([features, features[:1]]), np.append(groups, "a"))
        self.assertAlmostEqual(expected["mean_channel_standard_deviation"], duplicate["mean_channel_standard_deviation"])
        self.assertAlmostEqual(expected["effective_rank"], duplicate["effective_rank"])

    def test_hook_trains_the_existing_encoder_without_modifying_architecture(self):
        encoder = SkeletonPatchEncoder(frames=16, segment_length=4, embed_dim=8, depth=1, heads=2)
        loss = token_symmetry_penalty(encoder, torch.randn(3, 16, 33, 3), self.valid)
        loss.backward()
        gradients = [parameter.grad for parameter in encoder.parameters() if parameter.grad is not None]
        self.assertTrue(gradients)
        self.assertTrue(all(torch.isfinite(gradient).all() for gradient in gradients))

    def test_small_comparison_keeps_controls_and_reports_extra_compute(self):
        from laterality_extensions.masked_learning import LearningSettings, load_learning_dataset

        dataset = load_learning_dataset(real=False, fold=0)
        comparison = run_symmetry_comparison(
            dataset, LearningSettings(steps=1, embed_dim=8, heads=2)
        )
        self.assertTrue(comparison["synthetic"])
        self.assertTrue(all(comparison["pairing"].values()))
        self.assertEqual(len(comparison["summary"]), 6)
        self.assertEqual(comparison["runs"]["base_objective"]["encoder_forward_calls"], 4)
        self.assertEqual(comparison["runs"]["explicit_reflection_loss"]["encoder_forward_calls"], 6)
        self.assertTrue(np.isfinite(comparison["summary"].iloc[:4]["Reflection error"]).all())
        self.assertEqual(len(comparison["diagnostics"]), 4)


if __name__ == "__main__":
    unittest.main()
