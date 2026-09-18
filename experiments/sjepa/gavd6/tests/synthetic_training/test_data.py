"""Scientific data boundaries, camera labels and human-reference import."""
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd
from PIL import Image

from gavd6_sjepa.research_directions.synthetic_training.annotations import (
    import_gavd_annotations, select_gavd_panel,
)
from gavd6_sjepa.research_directions.synthetic_training.data import (
    BOX_COLUMNS, PoseFrameDataset, assign_amass_roles, context_clip_indices,
)
from gavd6_sjepa.research_directions.synthetic_training.rendering import (
    KEYPOINT_NAMES, camera_pose, fitted_camera_pose, load_uv_topology, project_points,
)


class TeachingDataTests(unittest.TestCase):
    def test_unlabeled_context_does_not_open_landmark_file(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "frame.png"
            Image.fromarray(np.zeros((20, 30, 3), np.uint8)).save(path)
            index = pd.DataFrame([dict(
                frame_id="f0", clip_id="c0", frame_index=0, image_path=str(path),
                label_path="this_label_file_must_not_be_opened.npz", role="train_context",
                bbox_x1=1, bbox_y1=2, bbox_x2=20, bbox_y2=18,
            )])
            with patch("numpy.load", side_effect=AssertionError("Reference leakage")):
                batch = PoseFrameDataset(index, labeled=False).get_batch([0])
            self.assertNotIn("keypoints", batch)
            self.assertNotIn("visible", batch)
            self.assertEqual(batch["images"][0].shape, (20, 30, 3))
            np.testing.assert_array_equal(batch["boxes"], [[1, 2, 20, 18]])

    def test_person_roles_preserve_existing_validation_and_test(self):
        rows = [dict(person_id=f"{split}_{person}", original_split=split, raw_path=f"{split}_{person}_{clip}")
                for split, count in (("train", 12), ("validation", 6), ("test", 4))
                for person in range(count) for clip in range(2)]
        table = assign_amass_roles(pd.DataFrame(rows))
        self.assertTrue((table.groupby("person_id").teaching_pool.nunique() == 1).all())
        self.assertEqual(set(table.loc[table.original_split.eq("test"), "teaching_pool"]), {"unused_test"})
        self.assertEqual(set(table.loc[table.original_split.eq("validation"), "teaching_pool"]),
                         {"validation_context", "validation_reference"})
        self.assertEqual(set(table.loc[table.original_split.eq("train"), "teaching_pool"]),
                         {"support", "train_context", "train_reference"})

    def test_context_clips_sorted_using_positional_indices(self):
        index = pd.DataFrame([dict(frame_id=str(i), clip_id="b" if i == 0 else "a", frame_index=t,
                                   image_path="unused.png", role="context", bbox_x1=0, bbox_y1=0,
                                   bbox_x2=10, bbox_y2=10) for i, t in enumerate([0, 5, 2])],
                             index=[100, 200, 300])
        groups = list(context_clip_indices(index))
        self.assertEqual(groups[0][0], "a")
        np.testing.assert_array_equal(groups[0][1], [2, 1])

    def test_projection_matches_camera_center_and_orientation(self):
        joints = np.zeros((2, 22, 3))
        # SMPL neutral subject-left is +X, and toe/forward direction is +Z.
        joints[:, 1, 0], joints[:, 2, 0] = 0.1, -0.1
        joints[:, [10, 11], 2] = 0.2
        pose = camera_pose(joints, 0, 4)
        self.assertGreater(pose[2, 3], 0)
        points = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]])
        xy, depth = project_points(points, pose, 200, 100, np.pi / 2)
        np.testing.assert_allclose(xy, [[99.5, 49.5], [112, 49.5], [99.5, 37]])
        np.testing.assert_allclose(depth, 4)
        self.assertAlmostEqual(np.linalg.det(pose[:3, :3]), 1)

    def test_uv_seams_and_topology_are_retained(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "template.obj"
            path.write_text("v 0 0 0\nv 1 0 0\nv 0 1 0\nvt 0 0\nvt 1 0\nvt 0 1\nf 1/3 2/1 3/2\n")
            faces = np.array([[0, 1, 2]])
            uv, face_uv = load_uv_topology(path, faces)
            np.testing.assert_array_equal(face_uv, [[2, 0, 1]])
            self.assertEqual(uv.shape, (3, 2))
            with self.assertRaisesRegex(ValueError, "exactly match"):
                load_uv_topology(path, faces[:, ::-1])


class CameraFramingTests(unittest.TestCase):
    def setUp(self):
        self.vertices = np.array([[x, y, z] for x in (-.3, .3) for y in (-.9, .9)
                                  for z in (-.15, .15)])[None].repeat(64, axis=0)
        self.joints = np.zeros((64, 22, 3))
        self.joints[:, 1, 0], self.joints[:, 2, 0] = .1, -.1
        self.yfov = np.deg2rad(50.)

    def fit(self, fraction, width=256, height=320, angle=45):
        pose, details = fitted_camera_pose(self.vertices, self.joints, angle, fraction,
                                           width, height, self.yfov)
        pixels, depth = project_points(self.vertices, pose, width, height, self.yfov)
        self.assertTrue(np.isfinite(pixels).all())
        self.assertTrue((depth > .05).all())
        self.assertTrue((pixels >= 1.99).all())
        self.assertTrue((pixels <= [width - 2.99, height - 2.99]).all())
        return pose, details, pixels

    def test_stationary_clip_keeps_the_original_camera(self):
        for fraction in (.25, .45, .65):
            with self.subTest(fraction=fraction):
                pose, details, _ = self.fit(fraction)
                nominal = max(2., 1.8 / (2 * np.tan(self.yfov / 2) * fraction))
                np.testing.assert_allclose(pose, camera_pose(self.joints, 45, nominal))
                self.assertEqual(details['distance_scale'], 1.)

    def test_large_travel_preserves_the_resolution_distance_ratios(self):
        shift = np.zeros((64, 1, 3))
        shift[:, 0, 0] = np.linspace(-5, 5, 64)
        self.vertices += shift
        self.joints += shift
        for width, height in ((256, 320), (64, 512), (512, 64)):
            with self.subTest(viewport=(width, height)):
                results = [self.fit(fraction, width, height) for fraction in (.65, .45, .25)]
                scales = [result[1]['distance_scale'] for result in results]
                self.assertGreater(scales[0], 1.)
                np.testing.assert_allclose(scales, scales[0])
                distances = [result[1]['distance_m'] for result in results]
                np.testing.assert_allclose(np.array(distances) / distances[0], [.65/.65, .65/.45, .65/.25])
                heights = [np.ptp(result[2][..., 1], axis=1).mean() for result in results]
                self.assertGreater(heights[0], heights[1])
                self.assertGreater(heights[1], heights[2])

    def test_motion_toward_camera_and_global_world_offset(self):
        shift = np.zeros((64, 1, 3))
        shift[:, 0, 2] = np.linspace(-8, 8, 64)
        self.vertices += shift
        self.joints += shift
        pose, _, pixels = self.fit(.65, angle=0)
        offset = np.array([150., -25., 30.])
        self.vertices += offset
        self.joints += offset
        moved_pose, _, moved_pixels = self.fit(.65, angle=0)
        np.testing.assert_allclose(pose[:3, 3] + offset, moved_pose[:3, 3])
        np.testing.assert_allclose(pixels, moved_pixels, atol=2e-5)

    def test_invalid_person_fraction_fails_clearly(self):
        for fraction in (0, -1, 1.1, np.nan, np.inf):
            with self.subTest(fraction=fraction), self.assertRaisesRegex(ValueError, 'person_height_fraction'):
                self.fit(fraction)


class GAVDPanelTests(unittest.TestCase):
    def _table(self):
        table, assignments = [], []
        for view in ("side", "oblique", "frontal_rear"):
            for resolution, height in (("low", 80), ("high", 180)):
                for recording in range(3):
                    uid = f"{view}_{resolution}_{recording}"
                    table.append(dict(sequence_id=uid, video_id=uid, available=True))
                    assignments.append(dict(sequence_id=uid, coarse_view=view, view_checked=True,
                                            person_height_px=height, bbox_x1=0, bbox_y1=0,
                                            bbox_x2=100, bbox_y2=200))
        return pd.DataFrame(table), pd.DataFrame(assignments)

    def test_panel_is_recording_disjoint_and_uses_person_resolution(self):
        table, assignments = self._table()
        panel = select_gavd_panel(table, assignments, counts=(1, 1, 1))
        self.assertEqual(len(panel), 18)
        self.assertFalse(panel.video_id.duplicated().any())
        self.assertEqual(panel.domain_id.nunique(), 6)
        self.assertEqual(set(panel.role), {"context", "early", "confirmation"})
        self.assertTrue(panel.resolution_boundary_px.eq(130).all())

    def test_panel_rejects_missing_cells_and_related_recording_reuse(self):
        table, assignments = self._table()
        assignments["related_recording_id"] = "one_original_source"
        with self.assertRaisesRegex(ValueError, "eligible recordings"):
            select_gavd_panel(table, assignments, counts=(1, 1, 1))
        table, assignments = self._table()
        assignments.loc[0, "person_height_px"] = np.nan
        with self.assertRaisesRegex(ValueError, "person heights"):
            select_gavd_panel(table, assignments, counts=(1, 1, 1))

    def test_protected_original_also_excludes_its_reupload(self):
        table, assignments = self._table()
        table.attrs["protected_video_ids"] = ["protected_original"]
        assignments["related_recording_id"] = assignments.sequence_id
        assignments.loc[0, "related_recording_id"] = "protected_original"
        with self.assertRaisesRegex(ValueError, "eligible recordings"):
            select_gavd_panel(table, assignments, counts=(1, 1, 1))

    def test_human_reference_scale_does_not_change_estimator_crop(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "data").mkdir()
            frame = dict(frame_id="f0", clip_id="c0", frame_index=10, image_path="unused.png",
                         role="early", label_path="", width=100, height=100,
                         bbox_x1=1, bbox_y1=2, bbox_x2=99, bbox_y2=98)
            pd.DataFrame([frame]).to_csv(root / "data" / "gavd_evaluation.csv", index=False)
            annotations = pd.DataFrame([dict(frame_id="f0", landmark=name, x=10, y=20, visible=True,
                                             box_x1=0, box_y1=0, box_x2=30, box_y2=40,
                                             annotator="annotator A", reviewer="") for name in KEYPOINT_NAMES])
            path = root / "gavd-early-annotations.csv"
            annotations.to_csv(path, index=False)
            cfg = SimpleNamespace(root=root, gavd_annotations_csv=str(root / "gavd-{split}-annotations.csv"))
            labeled = import_gavd_annotations(cfg, split="early")
            np.testing.assert_array_equal(labeled[BOX_COLUMNS], [[1, 2, 99, 98]])
            self.assertEqual(labeled.reference_scale.iloc[0], 50)
            self.assertTrue(Path(labeled.label_path.iloc[0]).is_file())
            # The confirmation file does not exist, proving early import did not need it.
            self.assertFalse((root / "gavd-confirmation-annotations.csv").exists())
            annotations["visible"] = annotations.visible.astype(object)
            annotations.loc[0, "visible"] = ""
            annotations.to_csv(path, index=False)
            with self.assertRaisesRegex(ValueError, "visible must"):
                import_gavd_annotations(cfg, split="early")


if __name__ == "__main__":
    unittest.main()
