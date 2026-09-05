from __future__ import annotations

import importlib.util
from pathlib import Path
import sys
import unittest

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = ROOT / "neurips-brain-body" / "pose_geometry.py"
SPEC = importlib.util.spec_from_file_location("brainbody_pose_geometry", MODULE_PATH)
pose_geometry = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = pose_geometry
SPEC.loader.exec_module(pose_geometry)


class PoseGeometryTests(unittest.TestCase):
    def annotation(self) -> dict[str, object]:
        return {
            "bbox": {"top": 125.0, "left": 156.0, "height": 497.0, "width": 228.0},
            "vid_info": {"height": 720, "width": 1280},
        }

    def test_projection_is_resolution_aware_for_reported_sequence(self) -> None:
        normalized = pose_geometry.normalized_padded_bounds(self.annotation())
        self.assertEqual(
            pose_geometry.project_normalized_bounds(normalized, (360, 640, 3)),
            (60, 25, 210, 349),
        )
        self.assertEqual(
            pose_geometry.project_normalized_bounds(normalized, (720, 1280, 3)),
            (121, 50, 419, 697),
        )

    def test_cached_normalized_bounds_reproject_without_half_scale_bug(self) -> None:
        old_pixels = (60, 25, 210, 349)
        normalized = pose_geometry.normalize_pixel_bounds(old_pixels, (640, 360))
        reprojected = pose_geometry.project_normalized_bounds(
            normalized, (720, 1280, 3)
        )
        self.assertEqual(reprojected, (120, 50, 420, 698))
        expected = pose_geometry.scaled_crop_bounds(
            self.annotation(), (720, 1280, 3)
        )
        self.assertGreater(pose_geometry.bounds_iou(reprojected, expected), 0.98)

    def test_invalid_annotation_does_not_silently_become_full_frame(self) -> None:
        annotation = self.annotation()
        annotation["bbox"] = {"top": 10, "left": 10, "height": 0, "width": 20}
        with self.assertRaises(pose_geometry.PoseGeometryError):
            pose_geometry.scaled_crop_bounds(annotation, (720, 1280, 3))


if __name__ == "__main__":
    unittest.main()
