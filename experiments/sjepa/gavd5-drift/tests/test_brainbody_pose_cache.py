from __future__ import annotations

import importlib.util
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = ROOT / "neurips-brain-body" / "pose_cache.py"
SPEC = importlib.util.spec_from_file_location("brainbody_pose_cache", MODULE_PATH)
pose_cache = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = pose_cache
SPEC.loader.exec_module(pose_cache)


class PoseCacheTests(unittest.TestCase):
    def legacy_payload(self) -> dict[str, np.ndarray]:
        return {
            "sequence": np.zeros((3, 33, 4), dtype=np.float32),
            "frame_numbers": np.asarray([4, 6, 9], dtype=np.int32),
            "crop_bounds": np.zeros((3, 4), dtype=np.int32),
            "fps": np.asarray(30.0, dtype=np.float32),
            "sequence_id": np.asarray("sequence-a"),
            "video_id": np.asarray("video-a"),
            "condition": np.asarray("normal"),
            "source_csv": np.asarray("old.csv"),
            "source_video": np.asarray("video.mp4"),
            "extraction_version": np.asarray("gavd5_pose_v2_video_mode"),
            "pose_model": np.asarray("pose.task"),
            "pose_model_sha256": np.asarray("model-hash"),
            "visibility_threshold": np.asarray(0.45, dtype=np.float32),
        }

    def expected(self) -> dict[str, object]:
        return {
            "sequence_id": "sequence-a",
            "video_id": "video-a",
            "condition": "normal",
            "frame_numbers": np.asarray([4, 6, 9]),
            "pose_model_sha256": "model-hash",
            "visibility_threshold": 0.45,
            "source_csv": "current.csv",
        }

    def provenance(self) -> dict[str, object]:
        return {
            "eligibility_stage": "decoded_frame_eligible",
            "outer_fold": 0,
            "split_role": "train",
            "split_version": "split-v1",
            "split_seed": 42,
            "manifest_sha256": "manifest-hash",
            "split_sha256": "split-hash",
            "split_roles_json": '{"0": "train"}',
        }

    def test_refresh_preserves_arrays_and_adds_current_provenance(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "pose.npz"
            np.savez_compressed(path, **self.legacy_payload())
            before = pose_cache.read_pose_payload(path)

            old_version = pose_cache.refresh_pose_cache_provenance(
                path,
                expected=self.expected(),
                provenance=self.provenance(),
            )
            after = pose_cache.validate_current_pose_cache(
                path,
                expected=self.expected(),
                provenance=self.provenance(),
            )

            self.assertEqual(old_version, "gavd5_pose_v2_video_mode")
            np.testing.assert_array_equal(after["sequence"], before["sequence"])
            np.testing.assert_array_equal(
                after["frame_numbers"], before["frame_numbers"]
            )
            self.assertEqual(
                after["extraction_version"].item(),
                pose_cache.CURRENT_EXTRACTION_VERSION,
            )
            self.assertEqual(after["source_csv"].item(), "current.csv")
            self.assertFalse(list(path.parent.glob("*.tmp.npz")))

    def test_refresh_rejects_frame_mismatch_without_replacing_original(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "pose.npz"
            np.savez_compressed(path, **self.legacy_payload())
            expected = self.expected()
            expected["frame_numbers"] = np.asarray([4, 6, 10])

            with self.assertRaises(pose_cache.PoseCacheIntegrityError):
                pose_cache.refresh_pose_cache_provenance(
                    path,
                    expected=expected,
                    provenance=self.provenance(),
                )

            with np.load(path, allow_pickle=False) as unchanged:
                self.assertEqual(
                    unchanged["extraction_version"].item(),
                    "gavd5_pose_v2_video_mode",
                )

    def test_retries_only_transient_failures(self) -> None:
        calls = []

        def transient_then_success():
            calls.append(1)
            if len(calls) == 1:
                raise pose_cache.TransientPoseExtractionError("decoder busy")
            return "ok"

        result, attempts = pose_cache.run_with_retries(
            transient_then_success,
            max_retries=2,
            backoff_seconds=0,
        )
        self.assertEqual(result, "ok")
        self.assertEqual(len(calls), 2)
        self.assertEqual([item["status"] for item in attempts], ["failed", "succeeded"])

        calls.clear()

        def permanent_failure():
            calls.append(1)
            raise pose_cache.PermanentPoseExtractionError("video too short")

        with self.assertRaises(pose_cache.PoseOperationFailed) as caught:
            pose_cache.run_with_retries(
                permanent_failure,
                max_retries=3,
                backoff_seconds=0,
            )
        self.assertEqual(len(calls), 1)
        self.assertFalse(caught.exception.disposition.retryable)

    def test_rejects_partial_resolution_safe_geometry(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "pose.npz"
            payload = self.legacy_payload()
            payload["frame_sizes"] = np.tile(
                np.asarray([[640, 360]], dtype=np.int32), (3, 1)
            )
            np.savez_compressed(path, **payload)
            with self.assertRaises(pose_cache.PoseCacheIntegrityError):
                pose_cache.refresh_pose_cache_provenance(
                    path,
                    expected=self.expected(),
                    provenance=self.provenance(),
                )


if __name__ == "__main__":
    unittest.main()
