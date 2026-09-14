"""motion preservation / test adapters."""


import importlib
import json
import os
from pathlib import Path
import sys
import tempfile
from types import ModuleType, SimpleNamespace
import unittest

import numpy as np
import torch

from gavd6_sjepa.research_directions.motion_preservation.pretrained_models import (
    FlowResult,
    HumanML3DBridge,
    OpticalFlowEstimator,
    _AuthorRepository,
    _world_to_y_up,
    bilinear_sample,
    joint_transport_diagnostics,
    load_external_prior,
    restore_flow_endpoints,
)

# Scientific coordinate/interface tests; no pretrained downloads or GPU runs.


class TransportTests(unittest.TestCase):
    def test_propagation_samples_updated_position_and_ignores_missing_reanchor(self):
        from gavd6_sjepa.research_directions.motion_preservation.rendering import Camera
        from gavd6_sjepa.research_directions.motion_preservation.workflow import _propagate_flow

        camera = Camera(np.zeros(3), np.eye(3), width=40, height=40, focal_px=20)
        xy = np.array([[[10., 20.]], [[16., 20.]], [[16., 20.]]])
        raw = np.zeros((3, 1, 3), dtype=np.float32)
        raw[..., 0] = (xy[..., 0]-19.5)/20*2
        raw[..., 1] = -(xy[..., 1]-19.5)/20*2
        raw[..., 2] = 2
        field = np.zeros((2, 40, 40, 2), np.float32)
        field[..., 0] = np.arange(40)[None, None, :] * .1
        flow = FlowResult(field)
        result = _propagate_flow(raw, camera, flow, neighborhood_radius=1)
        projected, depth = camera.project(result)
        # First propagated location is 0.8*(10+1)+0.2*16 = 12.
        # Next field sample must be at 12, not the original raw location16.
        np.testing.assert_allclose(projected[:, 0, 0], [10, 12, 13.76], atol=1e-5)
        np.testing.assert_allclose(depth, 2)
        observed = np.array([[True], [False], [False]])
        result = _propagate_flow(raw, camera, flow, observed, neighborhood_radius=1)
        projected, _ = camera.project(result)
        np.testing.assert_allclose(projected[:, 0, 0], [10, 11, 12.1], atol=1e-5)

    def test_bilinear_affine_field_and_missing_support(self):
        yy, xx = np.meshgrid(np.arange(5), np.arange(7), indexing="ij")
        field = np.stack((2 * xx + yy, -xx + 3 * yy), axis=-1)
        sampled, valid = bilinear_sample(field, np.array([[1.5, 2.25], [-1, 2], [6, 4]]))
        np.testing.assert_allclose(sampled[[0, 2]], [[5.25, 5.25], [16, 6]])
        self.assertEqual(valid.tolist(), [True, False, True])
        self.assertTrue(np.isnan(sampled[1]).all())

    def test_supported_motion_and_unsupported_tracker_jump_separate(self):
        forward = np.zeros((1, 20, 24, 2), np.float32)
        forward[..., 0] = 2
        flow = FlowResult(forward, -forward)
        joints = np.array([[[8., 10.], [8., 10.]], [[10., 10.], [13., 10.]]])
        diagnostics = joint_transport_diagnostics(flow, joints, neighborhood_radius=2)
        np.testing.assert_allclose(diagnostics["transport_error_px"], [[0, 3]])
        np.testing.assert_allclose(diagnostics["forward_backward_error_px"], 0)
        self.assertTrue(diagnostics["evidence_valid"].all())
        # Missing model uncertainty must not become a perfect-confidence input.
        self.assertTrue(np.isnan(diagnostics["flow_uncertainty_px"]).all())

    def test_neighborhood_median_resists_one_flow_outlier(self):
        forward = np.zeros((1, 15, 15, 2), np.float32)
        forward[..., 0] = 1
        forward[0, 7, 7] = [5, 0]
        points = np.array([[[7., 7.]], [[8., 7.]]])
        checks = joint_transport_diagnostics(FlowResult(forward), points, neighborhood_radius=1)
        np.testing.assert_allclose(checks["transport_error_px"], 0)

    def test_occluded_and_out_of_image_tracks_are_unknown(self):
        forward = np.zeros((1, 10, 10, 2), np.float32)
        points = np.array([[[5., 5.], [5., 5.]], [[5., 5.], [20., 5.]]])
        available = np.array([[False, True], [False, True]])
        checks = joint_transport_diagnostics(FlowResult(forward), points, valid=available)
        self.assertFalse(checks["evidence_valid"].any())
        self.assertTrue(np.isnan(checks["transport_error_px"]).all())

    def test_moving_crop_uses_both_endpoint_transforms(self):
        initial = np.array([[2., 0., 10.], [0., 2., 20.], [0., 0., 1.]])
        following = np.array([[3., 0., 40.], [0., 3., 50.], [0., 0., 1.]])
        start, end = restore_flow_endpoints(np.array([[5., 6.]]), np.array([[1., 2.]]), initial, following)
        np.testing.assert_allclose(start, [[20, 32]])
        np.testing.assert_allclose(end, [[58, 74]])
        # A stationary material point can move in crop coordinates.
        first = np.eye(3)
        second = np.eye(3)
        second[0, 2] = 3
        forward = np.zeros((1, 30, 30, 2), np.float32)
        forward[..., 0] = -3
        points = np.array([[[15., 15.]], [[12., 15.]]])
        checks = joint_transport_diagnostics(FlowResult(forward, -forward), points,
                                              crop_to_full=np.stack([first, second]))
        np.testing.assert_allclose(checks["local_flow_xy"], 0)
        np.testing.assert_allclose(checks["transport_error_px"], 0)
        np.testing.assert_allclose(checks["forward_backward_error_px"], 0)

    def test_brightness_diagnostic_compares_material_endpoints(self):
        yy, xx = np.meshgrid(np.arange(20), np.arange(24), indexing="ij")
        first = np.repeat((xx * 5 + yy)[..., None], 3, axis=-1).astype(np.uint8)
        following = np.roll(first, 2, axis=1)
        forward = np.zeros((1, 20, 24, 2), np.float32)
        forward[..., 0] = 2
        points = np.array([[[10., 10.]], [[12., 10.]]])
        checks = joint_transport_diagnostics(FlowResult(forward), points, rgb=np.stack([first, following]))
        np.testing.assert_allclose(checks["photometric_error"], 0, atol=1e-8)
        self.assertGreater(float(checks["texture_std"][0, 0]), 0)

class FlowInterfaceTests(unittest.TestCase):
    def test_sea_raft_pixel_scaling_and_uncertainty_with_mock_model(self):
        class FakeSEA(torch.nn.Module):
            def forward(self, first, second, *, iters, test_mode):
                assert first.shape[-2:] == (128, 128)
                assert first.max() == 255  # SEA-RAFT does its own normalization.
                flow = torch.ones((len(first), 2, 128, 128))
                info = torch.zeros((len(first), 4, 128, 128))
                return {"flow": [flow], "info": [info]}

        estimator = OpticalFlowEstimator.__new__(OpticalFlowEstimator)
        estimator.device = torch.device("cpu")
        estimator.backend, estimator.scale = "sea_raft", 0
        estimator.batch_size, estimator.iterations = 2, 4
        estimator.repository = None
        estimator.model, estimator.checkpoint = FakeSEA(), "mock_not_pretrained"
        estimator.args = SimpleNamespace(use_var=True, var_min=0, var_max=10)
        result = estimator.estimate(np.full((3, 32, 64, 3), 255, np.uint8))
        self.assertEqual(result.forward.shape, (2, 32, 64, 2))
        self.assertEqual(result.backward.shape, result.forward.shape)
        np.testing.assert_allclose(result.forward[..., 0], 0.5)
        np.testing.assert_allclose(result.forward[..., 1], 0.25)
        np.testing.assert_allclose(result.uncertainty[..., 0], 0.5)
        np.testing.assert_allclose(result.uncertainty[..., 1], 0.25)

    def test_author_import_does_not_overwrite_notebook_utils(self):
        old = sys.modules.get("utils")
        existing = ModuleType("utils")
        existing.owner = "notebook"
        sys.modules["utils"] = existing
        try:
            with tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary)
                (root / "utils.py").write_text("owner = 'author'\n")
                repo = _AuthorRepository(root)
                with repo.activate():
                    self.assertEqual(importlib.import_module("utils").owner, "author")
                self.assertIs(sys.modules["utils"], existing)
                with repo.activate():
                    self.assertEqual(importlib.import_module("utils").owner, "author")
                self.assertIs(sys.modules["utils"], existing)
        finally:
            if old is None:
                sys.modules.pop("utils", None)
            else:
                sys.modules["utils"] = old

    def test_axis_conversion_is_proper_rotation(self):
        values = np.array([[[1., 2., 3.]]])
        converted, rotation = _world_to_y_up(values, "amass_z_up")
        np.testing.assert_allclose(converted, [[[1., 3., -2.]]])
        self.assertAlmostEqual(float(np.linalg.det(rotation)), 1)
        np.testing.assert_allclose(converted @ rotation, values)

    def test_external_prior_has_explicit_temporal_and_coordinate_contract(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "prior.npz"
            metadata = {"model": "external_mdm", "checkpoint": "supplied_by_user",
                        "coordinate_system": "world_y_up", "context": "whole_supplied_clip"}
            np.savez(path, joints=np.zeros((4, 22, 3)), frame_indices=np.arange(4),
                     metadata_json=np.array(json.dumps(metadata)))
            result = load_external_prior(path)
            self.assertEqual(result.metadata["integration"], "external_cached_predictions")
            np.testing.assert_array_equal(result.frame_indices, [0, 1, 2, 3])
            # Truncating fractional indices would silently align the wrong poses.
            np.savez(path, joints=np.zeros((4, 22, 3)), frame_indices=[0., 1.9, 2.9, 3.9],
                     metadata_json=np.array(json.dumps(metadata)))
            with self.assertRaisesRegex(ValueError, "array of integers"):
                load_external_prior(path)

@unittest.skipUnless(os.environ.get("MOTION_PRESERVATION_MOMASK_REPO"),
                     "Set MOTION_PRESERVATION_MOMASK_REPO for the actual author conversion check")
class AuthorHumanMLTests(unittest.TestCase):
    """Uses only the author's small example and code; weights are not required."""

    def test_real_author_roundtrip_restores_axes_heading_scale_and_translation(self):
        bridge = HumanML3DBridge(os.environ["MOTION_PRESERVATION_MOMASK_REPO"])
        features = np.load(bridge.target_path, allow_pickle=False)
        original = bridge.motion.recover_from_ric(torch.as_tensor(features).float(), 22).numpy()[:65]
        angle = 0.73
        yaw = np.array([[np.cos(angle), 0, np.sin(angle)], [0, 1, 0], [-np.sin(angle), 0, np.cos(angle)]])
        transformed = (original @ yaw.T) * 1.37 + np.array([7., 2., -3.])
        result = bridge.roundtrip(transformed)
        self.assertEqual(result.joints.shape, (64, 22, 3))
        self.assertEqual(result.input_features.shape, (64, 263))
        self.assertLess(result.bridge_roundtrip_error_m, 2e-5)
        _, axis = _world_to_y_up(transformed, "amass_z_up")
        z_up = transformed @ axis
        z_result = bridge.roundtrip(z_up, "amass_z_up")
        np.testing.assert_allclose(z_result.joints, z_up[:-1], atol=3e-5)


if __name__ == "__main__":
    unittest.main()
