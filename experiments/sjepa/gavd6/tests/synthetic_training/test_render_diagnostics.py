"""Empty renders must identify the input and geometry, without guessing a cause."""
import importlib.util
import json
from pathlib import Path
import re
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
from PIL import Image

from gavd6_sjepa.research_directions.synthetic_training.rendering import (
    RenderRecipe, SMPL_INDICES, TexturedBodyRenderer,
)
from gavd6_sjepa.research_directions.synthetic_training.data import _write_rendered_clip


@unittest.skipUnless(importlib.util.find_spec("pyrender") and importlib.util.find_spec("trimesh"),
                     "Rendering diagnostics need pyrender and trimesh, but no OpenGL context")
class RenderDiagnosticTests(unittest.TestCase):
    def setUp(self):
        import trimesh

        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        root = Path(temporary.name)
        for name in ("textures", "backgrounds"):
            (root / name).mkdir()
            Image.new("RGB", (64, 64), (100, 150, 200)).save(root / name / "image.png")
        mesh = trimesh.creation.box(extents=(0.6, 1.8, 0.3))
        uv = root / "uv.npz"
        np.savez(uv, uv=np.array([[0., 0.], [1., 0.], [0., 1.]]),
                 face_uv=np.tile([0, 1, 2], (len(mesh.faces), 1)), faces=mesh.faces)
        joints = np.zeros((3, 22, 3))
        joints[:, 1, 0], joints[:, 2, 0] = .1, -.1
        self.body = SimpleNamespace(
            coordinate_system="y_up", faces=mesh.faces,
            vertices=np.repeat(mesh.vertices[None], 3, axis=0), joints=joints,
            timestamps=np.array([20., 20.25, 20.5]),
            metadata={"relative_path": "fixture/motion.npz"},
        )
        self.renderer = TexturedBodyRenderer(uv, root / "textures", root / "backgrounds", 256, 320)
        self.addCleanup(self.renderer.close)
        backend = patch("pyrender.OffscreenRenderer")
        self.backend = backend.start().return_value
        self.addCleanup(backend.stop)
        self.rgba = np.zeros((320, 256, 4), np.uint8)
        self.empty_depth = np.zeros((320, 256), np.float32)

    def test_travelling_body_stays_in_view_and_labels_match_the_render_camera(self):
        # Exercise the actual renderer's camera/label wiring without an OpenGL
        # context. The real pyrender projection matrix is an independent check.
        frames = 64
        self.body.vertices = np.repeat(self.body.vertices[:1], frames, axis=0)
        self.body.joints = np.repeat(self.body.joints[:1], frames, axis=0)
        self.body.timestamps = np.arange(frames) / 15
        shift = np.zeros((frames, 1, 3))
        shift[:, 0, 0] = np.linspace(-3, 3, frames)
        shift[:, 0, 2] = np.linspace(-4, 4, frames)
        self.body.vertices += shift
        self.body.joints += shift
        original_vertices = self.body.vertices.copy()
        for angle in (0, 45, 90):
            with self.subTest(angle=angle):
                poses, expected_joints = [], []

                def render_frame(scene, flags):
                    camera_node = scene.main_camera_node
                    pose = scene.get_pose(camera_node)
                    projection = camera_node.camera.get_projection_matrix(256, 320)
                    points = np.concatenate((self.body.vertices[len(poses)],
                                             self.body.joints[len(poses), SMPL_INDICES]))
                    homogeneous = np.column_stack((points, np.ones(len(points))))
                    clip = homogeneous @ np.linalg.inv(pose).T @ projection.T
                    self.assertTrue((clip[:, 3] > .05).all(), "Geometry crossed the near plane")
                    ndc = clip[:, :2] / clip[:, 3:4]
                    pixels = (ndc * [1, -1] + 1) * [128, 160] - .5
                    mesh_pixels = pixels[:-12]
                    self.assertTrue((mesh_pixels >= 1.99).all(), "Geometry left the left/top edge")
                    self.assertTrue((mesh_pixels <= [253.01, 317.01]).all(),
                                    "Geometry left the right/bottom edge")
                    poses.append(pose)
                    expected_joints.append(pixels[-12:])
                    return self.rgba, np.ones((320, 256), np.float32)

                self.backend.render.side_effect = render_frame
                rendered = self.renderer.render(self.body, RenderRecipe("moving", angle))
                self.assertEqual(len(rendered["images"]), frames)
                np.testing.assert_allclose(poses, np.broadcast_to(poses[0], (frames, 4, 4)))
                np.testing.assert_allclose(rendered["keypoints"], expected_joints, atol=2e-5)
                np.testing.assert_array_equal(self.body.vertices, original_vertices)
                if angle == 45:
                    with tempfile.TemporaryDirectory() as folder:
                        _write_rendered_clip(Path(folder), rendered, 'moving', {})
                        scene = json.loads((Path(folder) / 'synthetic/moving/scene.json').read_text())
                    self.assertEqual(scene['camera_framing'], rendered['camera_framing'])
                    self.assertEqual(scene['camera_framing']['policy'], 'fixed_camera_full_clip_v1')
                    self.assertGreater(scene['camera_framing']['distance_scale'], 1.)
                    self.assertGreater(scene['camera_framing']['projected_person_height_fraction_min'], 0.)
                    self.assertLess(scene['camera_framing']['projected_person_height_fraction_max'], 1.)

    def test_moving_clip_error_identifies_input_after_framing(self):
        shift = np.zeros((3, 1, 3))
        shift[:, 0, 0] = [-3., 0., 3.]
        self.body.vertices += shift
        self.body.joints += shift
        self.backend.render.return_value = (self.rgba, self.empty_depth)
        with self.assertRaises(ValueError) as error:
            self.renderer.render(self.body, RenderRecipe("front", 0.), seed=20)
        message = str(error.exception)
        self.assertIn("motion='fixture/motion.npz'", message)
        self.assertIn("recipe=front", message)
        self.assertIn("frame_index=0/3", message)
        self.assertIn("source_time_s=20.000000", message)
        self.assertIn("seed=20", message)
        self.assertIn("viewport=256x320", message)
        bounds = re.search(r"projected_x_px=\[([^,]+), ([^\]]+)\]", message)
        self.assertIsNotNone(bounds)
        self.assertGreaterEqual(float(bounds[1]), 1.99)
        self.assertLessEqual(float(bounds[2]), 253.01)
        self.assertIn("framing_scale=", message)

    def test_in_view_empty_render_is_not_mislabeled_as_offscreen(self):
        self.backend.render.side_effect = [
            (self.rgba, np.ones((320, 256), np.float32)),
            (self.rgba, self.empty_depth),
        ]
        with self.assertRaises(ValueError) as error:
            self.renderer.render(self.body, RenderRecipe("front", 0.), seed=17)
        message = str(error.exception)
        self.assertIn("frame_index=1/3", message)
        self.assertIn("source_time_s=20.250000", message)
        self.assertIn("depth_m=", message)
        bounds = re.search(r"projected_x_px=\[([^,]+), ([^\]]+)\]", message)
        self.assertIsNotNone(bounds)
        self.assertGreater(float(bounds[1]), 0)
        self.assertLess(float(bounds[2]), 256)
        self.assertNotIn("body is outside", message.lower())


if __name__ == "__main__":
    unittest.main()
