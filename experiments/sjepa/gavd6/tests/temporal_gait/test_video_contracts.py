"""Actual reader acceptance with an explicitly SYNTHETIC temporary video.

No GAVD/HAIC path is searched, opened, or downloaded. No workflow, experiment,
training, or run-mode transition is executed. Metadata-only inventory tests use
the explicit-input manifest schema (whose validator requires ``mode=real``),
but every artifact is labelled synthetic and only lives in TemporaryDirectory.
Passing these unit tests is not empirical validation of pose extraction or gait.
"""
import copy
from pathlib import Path
import shutil
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from gavd6_sjepa.research_directions.temporal_gait.config import RunConfig
from gavd6_sjepa.research_directions.temporal_gait.contracts import atomic_json, sha256_file
from gavd6_sjepa.research_directions.temporal_gait.manifests import SCHEMA, inventory, read_manifest
from gavd6_sjepa.research_directions.temporal_gait.video import load_pose, probe_pts


PACKAGE = "gavd6_sjepa.research_directions.temporal_gait"


class SyntheticVideoReaderTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if shutil.which("ffprobe") is None:
            raise unittest.SkipTest("Actual video-reader fixture requires ffprobe")
        try:
            import cv2
        except ImportError as exc:
            raise unittest.SkipTest("Synthetic video encoder requires optional OpenCV") from exc
        cls._temporary = tempfile.TemporaryDirectory(prefix="SYNTHETIC-video-reader-")
        cls.addClassCleanup(cls._temporary.cleanup)
        cls.fixture_root = Path(cls._temporary.name)
        path = cls.fixture_root / "SYNTHETIC-full-source-16frames.avi"
        width, height, count, fps = 64, 48, 16, 25.
        writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"), fps,
                                 (width, height))
        if not writer.isOpened():
            writer.release()
            raise unittest.SkipTest("OpenCV MJPG encoder unavailable for synthetic fixture")
        try:
            for i in range(count):
                pixels = np.zeros((height, width, 3), dtype=np.uint8)
                pixels[..., 0] = i * 11
                pixels[10:20, i:i + 8, 1] = 255
                writer.write(pixels)
        finally:
            writer.release()
        if not path.is_file() or not path.stat().st_size:
            raise unittest.SkipTest("OpenCV encoder did not create the synthetic fixture")
        cls.video = dict(video_id="SYNTHETIC-video", video_path=str(path),
                         sha256=sha256_file(path), is_full_video=True)
        # Actual ffprobe result, not a mocked clock or derived FPS substitute.
        cls.timing = probe_pts(cls.video)
        cls.expected_shape = (count, width, height)

    def setUp(self):
        temporary = tempfile.TemporaryDirectory(prefix="SYNTHETIC-pose-contract-")
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        pts, width, height = self.timing
        # A COMPLETE bout is a strict subset of the full video. The first and
        # final source frames must still be probed; only the declared bout is
        # represented in its pose cache, with its original nonzero frame IDs.
        self.bout = dict(sequence_id="SYNTHETIC-complete-bout",
                         video_id=self.video["video_id"], start_pts=float(pts[2]),
                         end_pts_exclusive=float(pts[12]),
                         boundary_convention="pts_half_open",
                         annotation_provenance="SYNTHETIC-unit-fixture-not-gait",
                         walking_status="complete_walking_bout")
        indices = np.arange(2, 12, dtype=np.int64)
        coords = np.empty((len(indices), 33, 2), dtype=np.float32)
        coords[..., 0] = np.arange(33)[None] + indices[:, None] / 10
        coords[..., 1] = 20.
        observed = np.ones((len(indices), 33), dtype=bool)
        confidence = np.ones((len(indices), 33), dtype=np.float32)
        # Entire failed detection: it must stay in the cache as a frame row.
        coords[3] = np.nan
        observed[3] = False
        confidence[3] = 0.
        confidence[4, 30] = .2
        self.arrays = dict(times=pts[indices].copy(), coords=coords,
                           observed=observed, confidence=confidence,
                           frame_indices=indices, width=np.array(width),
                           height=np.array(height))

    def write_pose(self, arrays=None, name="SYNTHETIC-complete-pose.npz"):
        path = self.root / name
        np.savez(path, **(self.arrays if arrays is None else arrays))
        return dict(sequence_id=self.bout["sequence_id"], pose_path=str(path),
                    sha256=sha256_file(path), extractor="SYNTHETIC-fixture",
                    extractor_version="unit-only-v1", checkpoint_sha256="f" * 64,
                    tracking_provenance="SYNTHETIC-no-actual-tracking",
                    causal_preprocessing=True, coordinate_frame="full_frame_pixels",
                    joint_order="mediapipe33", time_kind="video_pts", complete_bout=True,
                    source_video_sha256=self.video["sha256"])

    def load(self, entry):
        return load_pose(entry, self.bout, video_timing=self.timing,
                         visibility_threshold=.5)

    def write_inventory(self, *, pose=None, role="train", exposure="certified_unexposed",
                        duplicate_role=None):
        videos = [dict(self.video)]
        if duplicate_role is not None:
            # Same exact content is a duplicate even without a supplied link.
            videos.append({**self.video, "video_id": "SYNTHETIC-same-content"})
        roles = [role] if duplicate_role is None else [role, duplicate_role]
        docs = {
            "video": dict(rows=videos),
            "sequence": dict(rows=[dict(self.bout)]),
            "pose": dict(rows=[pose if pose is not None else self.write_pose()]),
            "identity": dict(verified_links_complete=True, rows=[]),
            "split": dict(rows=[dict(video_id=v["video_id"], role=r,
                                      reason="SYNTHETIC-metadata-only")
                                for v, r in zip(videos, roles)]),
            "exposure": dict(historical_laterality_93_checked=True,
                             audit_provenance="SYNTHETIC-no-human-sources",
                             rows=[dict(video_id=v["video_id"], status=exposure,
                                        historical_laterality_member=False,
                                        provenance="SYNTHETIC-unit-fixture") for v in videos]),
            "reservation": dict(rows=[dict(video_id=v["video_id"], protected=False,
                                            reason="SYNTHETIC-unit-fixture") for v in videos]),
        }
        paths = {}
        for kind, body in docs.items():
            path = self.root / f"SYNTHETIC-{kind}.json"
            atomic_json(path, dict(schema_version=SCHEMA, kind=kind, **body))
            paths[kind + "_manifest"] = str(path)
        # Only inventory() is called. Never freeze, prepare, or train a real run.
        return RunConfig(run_root=str(self.root / "SYNTHETIC-NEVER-RUN"), **paths)

    def metadata_inventory(self, cfg):
        # Provenance validation is metadata-only and cannot open media here.
        with patch(PACKAGE + ".video.probe_pts", side_effect=AssertionError("media access")), \
             patch(PACKAGE + ".video.load_pose", side_effect=AssertionError("pose access")), \
             patch("numpy.load", side_effect=AssertionError("NPZ access")), \
             patch("subprocess.run", side_effect=AssertionError("decoder access")):
            return inventory(cfg)

    def test_actual_probe_retains_full_source_pts_and_dimensions(self):
        pts, width, height = probe_pts(self.video)
        self.assertEqual((len(pts), width, height), self.expected_shape)
        np.testing.assert_allclose(pts, np.arange(16) / 25., rtol=0, atol=1e-6)
        self.assertTrue(np.all(np.diff(pts) > 0))
        self.assertLess(pts[0], self.bout["start_pts"])
        self.assertGreater(pts[-1], self.bout["end_pts_exclusive"])

    def test_complete_bout_preserves_failed_rows_and_visibility_filter(self):
        pose = self.load(self.write_pose())
        np.testing.assert_array_equal(pose.frame_indices, np.arange(2, 12))
        np.testing.assert_array_equal(pose.times, self.timing[0][2:12])
        self.assertEqual(len(pose.times), 10)
        self.assertFalse(pose.observed[3].any())
        self.assertTrue(np.isnan(pose.coords[3]).all())
        self.assertFalse(pose.observed[4, 30])
        self.assertTrue(pose.observed[4, 29])
        self.assertEqual(int(pose.frame_indices[3]), 5)

    def test_video_and_pose_sha_mismatches_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "Full-video SHA256 changed"):
            probe_pts({**self.video, "sha256": "0" * 64})
        entry = self.write_pose()
        with self.assertRaisesRegex(ValueError, "Pose cache SHA256 changed"):
            self.load({**entry, "sha256": "0" * 64})

    def test_missing_failed_detection_row_rejected_not_compressed(self):
        arrays = {k: (np.delete(v, 3, axis=0) if v.ndim else v)
                  for k, v in self.arrays.items()}
        with self.assertRaisesRegex(ValueError, "retain every source frame"):
            self.load(self.write_pose(arrays))

    def test_including_exclusive_bout_endpoint_is_rejected(self):
        arrays = copy.deepcopy(self.arrays)
        # Original PTS remains correct; only COMPLETE [start,end) support is wrong.
        arrays["frame_indices"][-1] = 12
        arrays["times"][-1] = self.timing[0][12]
        with self.assertRaisesRegex(ValueError, "COMPLETE bout"):
            self.load(self.write_pose(arrays))

    def test_wrong_full_frame_aspect_is_rejected(self):
        for field in ("width", "height"):
            with self.subTest(field=field):
                arrays = copy.deepcopy(self.arrays)
                arrays[field] = arrays[field] + 1
                with self.assertRaisesRegex(ValueError, "aspect ratio"):
                    self.load(self.write_pose(arrays))

    def test_wrong_pts_is_rejected_even_when_monotone(self):
        arrays = copy.deepcopy(self.arrays)
        arrays["times"][0] += .001
        with self.assertRaisesRegex(ValueError, "original video presentation timestamp"):
            self.load(self.write_pose(arrays))

    def test_missing_or_noninteger_original_frame_ids_are_rejected(self):
        arrays = copy.deepcopy(self.arrays)
        arrays.pop("frame_indices")
        with self.assertRaisesRegex(ValueError, "Pose cache requires"):
            self.load(self.write_pose(arrays))
        arrays = copy.deepcopy(self.arrays)
        arrays["frame_indices"] = arrays["frame_indices"].astype(float)
        with self.assertRaisesRegex(ValueError, "original frame IDs"):
            self.load(self.write_pose(arrays))

    def test_manifest_rejects_noncausal_and_unaligned_pose_provenance(self):
        pose = self.write_pose()
        self.assertEqual(self.metadata_inventory(self.write_inventory(pose=pose))["declared_bouts"], 1)
        invalid = dict(causal_preprocessing=False, coordinate_frame="cropped_pixels",
                       joint_order="unknown", time_kind="frame_index", complete_bout=False)
        for field, value in invalid.items():
            with self.subTest(field=field):
                cfg = self.write_inventory(pose={**pose, field: value})
                with self.assertRaisesRegex(ValueError, "Noncausal/unaligned/partial"):
                    self.metadata_inventory(cfg)
        cfg = self.write_inventory(pose={**pose, "source_video_sha256": "0" * 64})
        with self.assertRaisesRegex(ValueError, "content identity mismatch"):
            self.metadata_inventory(cfg)

    def test_manifest_unknown_exposure_cannot_be_untouched_test(self):
        cfg = self.write_inventory(role="test", exposure="unknown")
        with self.assertRaisesRegex(ValueError, "Exposed/unknown"):
            self.metadata_inventory(cfg)

    def test_exact_content_duplicates_merge_and_cannot_cross_partitions(self):
        same = self.metadata_inventory(self.write_inventory(duplicate_role="train"))
        self.assertEqual(same["declared_videos"], 2)
        self.assertEqual(same["groups"], 1)
        self.assertEqual(len({v["group_id"] for v in same["videos"]}), 1)
        cfg = self.write_inventory(duplicate_role="test")
        with self.assertRaisesRegex(ValueError, "crosses source partitions"):
            self.metadata_inventory(cfg)


class ManifestSchemaTests(unittest.TestCase):
    def test_unknown_schema_wrong_kind_and_missing_rows_rejected(self):
        with tempfile.TemporaryDirectory(prefix="SYNTHETIC-manifest-schema-") as directory:
            path = Path(directory) / "SYNTHETIC-pose-manifest.json"
            valid = dict(schema_version=SCHEMA, kind="pose", rows=[])
            atomic_json(path, valid)
            self.assertEqual(read_manifest(path, "pose"), valid)
            for field, value in (("schema_version", "unknown"), ("kind", "video"),
                                 ("rows", None)):
                with self.subTest(field=field):
                    atomic_json(path, {**valid, field: value})
                    with self.assertRaisesRegex(ValueError, "Expected versioned pose"):
                        read_manifest(path, "pose")


if __name__ == "__main__":
    unittest.main()
