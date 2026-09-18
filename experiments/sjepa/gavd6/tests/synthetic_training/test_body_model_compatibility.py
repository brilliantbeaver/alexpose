"""Opt-in test of the real pinned body backend with invented mesh coefficients.

Run with ST_TEST_BODY_MODEL=1 in the Python 3.11 / Torch 2.6 study environment.
No licensed body assets, pretrained models, renderer, CUDA or HAIC are needed.
The tiny mesh validates arithmetic and API compatibility, not human anatomy.
"""
from dataclasses import replace
import importlib.metadata
import inspect
import os
from pathlib import Path
import tempfile
import unittest

import numpy as np


@unittest.skipUnless(os.environ.get("ST_TEST_BODY_MODEL") == "1",
                     "Set ST_TEST_BODY_MODEL=1 in the pinned body-model test environment")
class BodyModelCompatibilityTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import torch
        from human_body_prior.body_model.body_model import BodyModel
        if (torch.__version__.split("+")[0] != "2.6.0"
                or np.__version__ != "1.26.4"
                or importlib.metadata.version("human-body-prior") != "2.2.2.0"):
            raise RuntimeError("This integration check requires Torch 2.6.0, NumPy 1.26.4 and human-body-prior 2.2.2.0")
        cls.torch, cls.backend = torch, BodyModel

    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        body, dmpl = self.root / "smplh/male", self.root / "dmpls/male"
        body.mkdir(parents=True)
        dmpl.mkdir(parents=True)
        # SMPL-H has 52 articulated joints. These coefficients are entirely
        # synthetic; one vertex per joint keeps expected results analytical.
        self.template = np.arange(52 * 3, dtype=np.float32).reshape(52, 3) / 100
        shape = np.zeros((52, 3, 16), np.float32)
        shape[:, 0, 0] = 0.1
        dynamic = np.zeros((52, 3, 8), np.float32)
        dynamic[:, 1, 0] = 0.2
        np.savez(body / "model.npz", v_template=self.template,
                 f=np.array([[0, 1, 2]], np.int32), shapedirs=shape,
                 posedirs=np.zeros((52, 3, 51 * 9), np.float32),
                 J_regressor=np.eye(52, dtype=np.float32),
                 kintree_table=np.stack([np.arange(52) - 1, np.arange(52)]),
                 weights=np.eye(52, dtype=np.float32))
        np.savez(dmpl / "model.npz", eigvec=dynamic)
        self.model = self.backend(bm_fname=str(body / "model.npz"), num_betas=16,
                                  dmpl_fname=str(dmpl / "model.npz"), num_dmpls=8)

    def test_required_api_and_real_forward_preserve_dynamic_shape(self):
        self.assertTrue({"bm_fname", "dmpl_fname"} <= set(inspect.signature(self.backend).parameters))
        self.assertEqual(self.model.model_type, "smplh")
        self.assertTrue(self.model.use_dmpl)
        self.assertEqual(self.model.dmpldirs.shape[-1], 8)
        zeros = self.torch.zeros
        dmpls = zeros((2, 8)); dmpls[1, 0] = 1
        betas = zeros((2, 16)); betas[:, 0] = 2
        with self.torch.inference_mode():
            output = self.model(betas=betas, dmpls=dmpls)
        expected = np.stack([self.template, self.template]).copy()
        expected[:, :, 0] += 0.2
        expected[1, :, 1] += 0.2
        np.testing.assert_allclose(output.v.numpy(), expected, atol=2e-6, rtol=0)
        np.testing.assert_allclose(output.Jtr.numpy(), expected, atol=2e-6, rtol=0)

    def test_pose_rotation_and_translation_are_not_ignored(self):
        root_orient = self.torch.tensor([[0, 0, np.pi / 2]], dtype=self.torch.float32)
        trans = self.torch.tensor([[2., 3., 4.]])
        with self.torch.inference_mode():
            base = self.model()
            moved = self.model(root_orient=root_orient, trans=trans)
        pelvis = base.Jtr[:, :1]
        rotation = self.torch.tensor([[0., -1., 0.], [1., 0., 0.], [0., 0., 1.]])
        expected = (base.v - pelvis) @ rotation.T + pelvis + trans[:, None]
        np.testing.assert_allclose(moved.v.numpy(), expected.numpy(), atol=2e-6, rtol=0)

    def test_study_adapter_batches_and_rotates_the_same_body_outputs(self):
        from gavd6_sjepa.research_directions.motion_preservation.body_geometry import AMASS_TO_Y_UP, SMPLHBody
        from gavd6_sjepa.research_directions.motion_preservation.motion_data import MotionParameters
        frames = 5
        dynamic = np.zeros((frames, 8), np.float32)
        dynamic[:, 0] = np.arange(frames)
        motion = MotionParameters(poses=np.zeros((frames, 156), np.float32),
                                  trans=np.zeros((frames, 3), np.float32),
                                  betas=np.zeros(16, np.float32), dmpls=dynamic,
                                  gender="male", fps=20., timestamps=np.arange(frames) / 20)
        adapter = SMPLHBody(self.root, device="cpu", batch_size=2)
        actual = adapter.forward(motion)
        expected = np.repeat(self.template[None], frames, axis=0)
        expected[:, :, 1] += dynamic[:, 0, None] * 0.2
        expected = expected @ AMASS_TO_Y_UP.T
        np.testing.assert_allclose(actual.vertices, expected, atol=2e-6, rtol=0)
        np.testing.assert_allclose(actual.joints, expected[:, :22], atol=2e-6, rtol=0)
        static = adapter.forward(replace(motion, dmpls=np.zeros_like(dynamic)))
        self.assertGreater(float(np.abs(actual.vertices - static.vertices).max()), 0.7)


if __name__ == "__main__":
    unittest.main()
