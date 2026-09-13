"""future feature prediction / gate / test input data."""


import os
from pathlib import Path
import tempfile
import unittest

import cv2
import numpy as np
import pandas as pd

from gavd6_sjepa.research_directions.future_prediction.cohort import (
    build_candidates,
    deterministic_window_start,
    select_eligible,
    validate_cohort,
)
from gavd6_sjepa.research_directions.future_prediction.contracts import (
    FRAME,
    FrameContract,
    initialize_run,
    stage_lock,
    write_once_json,
)
from gavd6_sjepa.research_directions.future_prediction.source_inventory import discover_sources
from gavd6_sjepa.research_directions.future_prediction.validity_audits import pixel_edits
from gavd6_sjepa.research_directions.future_prediction.video_pose import (
    CropGeometry,
    annotation_boxes,
    decode_exact_window,
    normalize_skeleton,
)
from tests.support import source_discovery_fixture

class FutureInnovationBoxTests(unittest.TestCase):
    def test_boxes_and_pixels_share_resize_crop(self):
        geometry = CropGeometry(100, 200)
        self.assertEqual(geometry.resized_hw, (438, 876))
        np.testing.assert_array_equal(geometry.offset_xy, [246, 27])
        boxes, retention = geometry.boxes(np.array([[0.4, 0.2, 0.6, 0.8]]))
        expected = (
            np.array([[0.4 * 876, 0.2 * 438, 0.6 * 876, 0.8 * 438]])
            - [246, 27, 246, 27]
        ) / 384
        np.testing.assert_allclose(boxes, expected)
        np.testing.assert_allclose(retention, [1])
        video = np.zeros((1, 100, 200, 3), dtype=np.uint8)
        video[:, 20:80, 80:120] = 255
        resized = geometry.pixels(video)
        y, x = np.where(resized[0, :, :, 0] > 127)
        np.testing.assert_allclose(
            [x.min(), y.min(), x.max() + 1, y.max() + 1], boxes[0] * 384, atol=1
        )

    def test_severe_cropping_and_invalid_annotation(self):
        _, retention = CropGeometry(100, 200).boxes(np.array([[0, 0, 0.1, 1]]))
        self.assertLess(retention[0], 0.9)
        with self.assertRaises(ValueError):
            annotation_boxes([{"bbox": "{}", "vid_info": "{}"}])

    def test_pixel_edits_keep_past_and_opposite_regions_fixed(self):
        rng = np.random.default_rng(4)
        video = rng.integers(0, 256, (64, 32, 48, 3), dtype=np.uint8)
        donor = rng.integers(0, 256, video.shape, dtype=np.uint8)
        boxes = np.tile([0.25, 0.25, 0.75, 0.75], (64, 1))
        person, background = pixel_edits(video, boxes, donor, boxes, feather=3)
        np.testing.assert_array_equal(person[:32], video[:32])
        np.testing.assert_array_equal(background[:32], video[:32])
        np.testing.assert_array_equal(person[32:, :, :12], video[32:, :, :12])
        np.testing.assert_array_equal(
            background[32:, 8:24, 12:36], video[32:, 8:24, 12:36]
        )
        self.assertGreater(np.abs(person.astype(float) - video).sum(), 0)
        self.assertGreater(np.abs(background.astype(float) - video).sum(), 0)


class FutureInnovationCohortTests(unittest.TestCase):
    def test_duplicate_stage_writers_are_rejected_and_lock_is_released(self):
        with tempfile.TemporaryDirectory() as directory:
            with stage_lock(directory, "fold-0"):
                with (
                    self.assertRaisesRegex(ValueError, "Another job"),
                    stage_lock(directory, "fold-0"),
                ):
                    self.fail("duplicate lock acquired")
                with stage_lock(directory, "fold-1"):
                    pass
            with stage_lock(directory, "fold-0"):
                pass

    def candidates(self):
        return [
            {
                "window_id": f"w{i}",
                "sequence_id": f"seq{i}",
                "video_id": f"source{i // 4}",
            }
            for i in range(200)
        ]

    def test_deterministic_selection_source_caps_and_folds(self):
        candidates = self.candidates()
        selected = select_eligible(candidates)
        self.assertTrue(selected.equals(select_eligible(list(reversed(candidates)))))
        validate_cohort(selected)
        self.assertEqual(len(selected), 50)
        self.assertLessEqual(selected.groupby("video_id").size().max(), 2)
        self.assertEqual(selected.groupby("video_id").outer_fold.nunique().max(), 1)

    def test_small_cohort_and_source_overlap_fail(self):
        with self.assertRaisesRegex(ValueError, "exactly 50"):
            select_eligible(self.candidates()[:40])
        cohort = select_eligible(self.candidates())
        pair = cohort[cohort.video_id.duplicated(keep=False)].index[:1]
        self.assertTrue(len(pair))
        cohort.loc[pair, "outer_fold"] = (cohort.loc[pair, "outer_fold"] + 1) % 5
        with self.assertRaisesRegex(ValueError, "multiple folds"):
            validate_cohort(cohort)

    def test_window_start_and_no_short_sequence(self):
        self.assertEqual(deterministic_window_start(10, 73, "a"), 10)
        self.assertIn(deterministic_window_start(10, 100, "a"), range(10, 38))
        with self.assertRaises(ValueError):
            deterministic_window_start(10, 72, "a")

    def test_frozen_contract_cannot_change(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "contract.json"
            write_once_json(path, {"seed": 7})
            write_once_json(path, {"seed": 7})
            with self.assertRaises(ValueError):
                write_once_json(path, {"seed": 8})
            with self.assertRaises(ValueError):
                write_once_json(Path(directory) / "nan.json", {"score": float("nan")})


class FutureInnovationFrameTests(unittest.TestCase):
    def test_exact_decode_absolute_zero_based_index(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "video.avi"
            writer = cv2.VideoWriter(
                str(path), cv2.VideoWriter_fourcc(*"MJPG"), 25, (48, 32)
            )
            self.assertTrue(writer.isOpened())
            for i in range(75):
                writer.write(np.full((32, 48, 3), i * 3, dtype=np.uint8))
            writer.release()
            frames, fps = decode_exact_window(path, 4)
            self.assertEqual(frames.shape, (64, 32, 48, 3))
            self.assertEqual(fps, 25)
            # MJPEG's YUV round trip shifts this uniform-color fixture by up
            # to 4/3 intensity units on HAIC's OpenCV/FFmpeg build. A 1.5
            # tolerance still rejects a one-frame shift: tags are 3 apart.
            np.testing.assert_allclose(
                frames.mean(axis=(1, 2, 3)), np.arange(4, 68) * 3, atol=1.5
            )
            with self.assertRaises(ValueError):
                decode_exact_window(path, 20)

    def test_context_and_target_tubelets(self):
        self.assertEqual(FRAME.target_tubelet_start // FRAME.tubelet_size, 19)
        self.assertEqual(FRAME.context_stop_exclusive // FRAME.tubelet_size, 16)
        with self.assertRaises(ValueError):
            FrameContract(context_stop_exclusive=31)
        with self.assertRaises(ValueError):
            FrameContract(
                target_tubelet_start=30,
                target_tubelet_stop_exclusive=32,
                target_horizon_frames=0,
            )

    def test_past_normalization_preserves_missingness(self):
        raw = np.ones((32, 33, 4), dtype=np.float32) * 0.5
        raw[..., 3] = 0.9
        raw[:, 11:13, 1] = 0.25
        raw[0, 5, :2] = np.nan
        boxes = np.tile([0, 0, 1, 1], (32, 1))
        history, scale = normalize_skeleton(raw, boxes)
        self.assertAlmostEqual(scale, 0.25)
        np.testing.assert_array_equal(history[0, 5, :2], 0)
        self.assertEqual(history[0, 5, 3], 0)
        self.assertAlmostEqual(history[0, 5, 2], 0.9, places=6)
        with self.assertRaises(ValueError):
            normalize_skeleton(np.tile(raw, (2, 1, 1)), np.tile(boxes, (2, 1)))


class FutureInnovationSourceDiscoveryTests(unittest.TestCase):
    def test_nested_cache_and_manifest_path_and_missing_source(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            nested = root / "cache/condition"
            nested.mkdir(parents=True)
            (nested / "source-a.AVI").write_bytes(b"video")
            explicit = root / "original title.mp4"
            explicit.write_bytes(b"video")
            table = pd.DataFrame({"video_id": ["source-a", "source-b", "source-c"],
                                  "video_path": [None, "original title.mp4", None]})
            found = discover_sources(table, root / "videos.csv", root / "cache")
            self.assertEqual(found["source-a"]["path"], (nested / "source-a.AVI").resolve())
            self.assertEqual(found["source-b"]["path"], explicit.resolve())
            self.assertIn("not found", found["source-c"]["error"])

    def test_ambiguous_ids_are_not_silently_assigned_and_symlinks_deduplicate(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            a, b = root / "a", root / "b"
            a.mkdir(); b.mkdir()
            (a / "source.mp4").write_bytes(b"original")
            (b / "source.mp4").symlink_to(a / "source.mp4")
            os.link(a / "source.mp4", b / "source.mkv")
            table = pd.DataFrame({"video_id": ["source"]})
            self.assertIsNotNone(discover_sources(table, root / "videos.csv", root)["source"]["path"])
            (b / "source.webm").write_bytes(b"different export")
            found = discover_sources(table, root / "videos.csv", root)
            self.assertIn("Ambiguous", found["source"]["error"])
            table["video_path"] = str(a / "source.mp4")
            self.assertEqual(discover_sources(table, root / "videos.csv", root)["source"]["path"], a / "source.mp4")

    def test_additional_roots_use_exact_ids_and_ignore_partial_files(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            cache, additional = root / "cache", root / "external"
            cache.mkdir(); additional.mkdir()
            (additional / "abc.mp4").write_bytes(b"video")
            (additional / "abc-longer.mp4").write_bytes(b"wrong source")
            (cache / "abc.mp4.part").write_bytes(b"partial")
            table = pd.DataFrame({"video_id": ["abc", "abc-long"]})
            found = discover_sources(table, root / "manifest.csv", cache, [additional])
            self.assertEqual(found["abc"]["path"], additional / "abc.mp4")
            self.assertIsNone(found["abc-long"]["path"])

    fixture = staticmethod(source_discovery_fixture)

    def test_gap_does_not_discard_another_complete_annotation_window(self):
        with tempfile.TemporaryDirectory() as temporary:
            args, missing = self.fixture(Path(temporary).resolve(), gap=True)
            build_candidates(*args)
            candidates = pd.read_csv(args[0] / "manifests/candidates.csv")
            row = candidates.set_index("sequence_id").loc["sequence-0"]
            self.assertEqual(len(candidates), 50)
            self.assertFalse(row.source_first_frame + 1 <= missing <= row.source_last_frame + 1)
            self.assertEqual(row.source_last_frame - row.source_first_frame, 63)

    def test_incomplete_discovery_can_retry_after_cached_source_arrives(self):
        with tempfile.TemporaryDirectory() as temporary:
            args, _ = self.fixture(Path(temporary).resolve())
            path = args[-1] / "all/video-0.mp4"
            path.rename(path.with_suffix(".part"))
            with self.assertRaisesRegex(ValueError, "Only 48 candidates"):
                build_candidates(*args)
            self.assertFalse((args[0] / "config/candidates-contract.json").exists())
            path.with_suffix(".part").rename(path)
            build_candidates(*args)
            self.assertEqual(len(pd.read_csv(args[0] / "manifests/candidates.csv")), 50)

    def test_short_annotated_sequence_is_not_padded_to_force_a_pass(self):
        with tempfile.TemporaryDirectory() as temporary:
            args, _ = self.fixture(Path(temporary).resolve(), short=True)
            with self.assertRaisesRegex(ValueError, "Only 49 candidates"):
                build_candidates(*args)
            exclusions = pd.read_csv(args[0] / "manifests/exclusions.csv")
            self.assertEqual(exclusions.reason.tolist(), ["Sequence shorter than 64 frames"])
            self.assertTrue(pd.read_csv(args[0] / "manifests/source-availability.csv").available.all())


if __name__ == "__main__":
    unittest.main()
