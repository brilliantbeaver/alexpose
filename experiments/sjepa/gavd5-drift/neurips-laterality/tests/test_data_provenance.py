from __future__ import annotations

import hashlib
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

SUITE_ROOT = Path(__file__).resolve().parents[1]
if str(SUITE_ROOT) not in sys.path:
    sys.path.insert(0, str(SUITE_ROOT))

from laterality.data import (
    PreparedCohort,
    SPLIT_PROVENANCE_SCHEMA_VERSION,
    load_real_pose_records,
)


class RealPoseProvenanceTests(unittest.TestCase):
    sequence_id = "sequence-a"
    video_id = "video-a"
    condition = "normal"

    def prepared_cohort(self, *, archive_sha256: str = "archive-a") -> PreparedCohort:
        table = pd.DataFrame(
            {
                "sequence_id": [self.sequence_id],
                "video_id": [self.video_id],
                "condition": [self.condition],
                "target": [0.25],
                "usable_pair_count": [12],
                "authorized_coverage": [0.9],
                "authorized_patch_count": [10],
                "raw_frame_count": [80],
                "archive_sha256": [archive_sha256],
                "extraction_version": ["gavd5_pose_v2_video_mode"],
            }
        )
        return PreparedCohort(
            table=table,
            model_xyz=np.zeros((1, 2, 33, 3), dtype=np.float32),
            model_valid=np.ones((1, 2, 33), dtype=bool),
            pair_contrasts=np.asarray([[0.1, 0.2]], dtype=np.float64),
            missingness=np.asarray([[0.0, 0.1]], dtype=np.float64),
            cohort_digest="container-sensitive-digest",
            attrition={},
        )

    @staticmethod
    def inventory_digest(values: list[str]) -> str:
        return hashlib.sha256(
            ("\n".join(sorted(values)) + "\n").encode("utf-8")
        ).hexdigest()

    def make_fixture(
        self,
        root: Path,
        *,
        source_csv: str,
        schema_version: str = SPLIT_PROVENANCE_SCHEMA_VERSION,
        origin_version: str | None = "gavd5_pose_v2_video_mode",
    ) -> tuple[Path, Path, dict[str, object]]:
        annotation_root = root / "annotations"
        pose_root = root / "poses"
        annotation_dir = annotation_root / self.condition
        pose_dir = pose_root / self.condition
        annotation_dir.mkdir(parents=True)
        pose_dir.mkdir(parents=True)

        annotation_path = annotation_dir / f"{self.sequence_id}.csv"
        pd.DataFrame(
            {
                "seq": [self.sequence_id],
                "id": [self.video_id],
                "gait_pat": [self.condition],
            }
        ).to_csv(annotation_path, index=False)

        payload: dict[str, np.ndarray] = {
            "sequence": np.zeros((3, 33, 4), dtype=np.float32),
            "frame_numbers": np.asarray([1, 2, 3]),
            "sequence_id": np.asarray(self.sequence_id),
            "video_id": np.asarray(self.video_id),
            "condition": np.asarray(self.condition),
            "fps": np.asarray(30.0),
            "source_csv": np.asarray(source_csv),
            "extraction_version": np.asarray(schema_version),
            "pose_model": np.asarray("pose.task"),
            "pose_model_sha256": np.asarray("model-hash"),
            "visibility_threshold": np.asarray(0.45),
        }
        if origin_version is not None:
            payload["cache_origin_version"] = np.asarray(origin_version)
        np.savez_compressed(pose_dir / f"{self.sequence_id}.npz", **payload)

        annotation_inventory = [f"{self.condition}/{annotation_path.name}"]
        pose_inventory = [f"{self.condition}/{self.sequence_id}"]
        mapping = f"{self.condition}/{self.sequence_id}\t{self.video_id}\t{self.condition}"
        data_config: dict[str, object] = {
            "inventory_contract": {
                "annotation_count": 1,
                "annotation_inventory_sha256": self.inventory_digest(
                    annotation_inventory
                ),
                "pose_archive_count": 1,
                "pose_inventory_sha256": self.inventory_digest(pose_inventory),
                "annotation_source_mapping_sha256": hashlib.sha256(
                    f"{mapping}\n".encode("utf-8")
                ).hexdigest(),
            },
            "extraction_provenance": {
                "extraction_version_counts": {
                    origin_version or schema_version: 1,
                },
                "pose_model": "pose.task",
                "pose_model_sha256": "model-hash",
                "visibility_threshold": 0.45,
            },
        }
        return pose_root, annotation_root, data_config

    def test_windows_path_and_metadata_wrapper_preserve_origin_generation(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            source = rf"C:\data-gavd\{self.condition}\{self.sequence_id}.csv"
            pose_root, annotation_root, config = self.make_fixture(
                Path(directory), source_csv=source
            )

            records, audit = load_real_pose_records(
                pose_root, annotation_root, [self.condition], config
            )

            self.assertEqual(records[0].extraction_version, "gavd5_pose_v2_video_mode")
            self.assertEqual(
                audit["extraction_version_counts"],
                {"gavd5_pose_v2_video_mode": 1},
            )
            self.assertEqual(
                audit["archive_schema_version_counts"],
                {SPLIT_PROVENANCE_SCHEMA_VERSION: 1},
            )

    def test_posix_legacy_path_remains_supported(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            source = f"/data-gavd/{self.condition}/{self.sequence_id}.csv"
            pose_root, annotation_root, config = self.make_fixture(
                Path(directory),
                source_csv=source,
                schema_version="gavd5_pose_v2_video_mode",
                origin_version=None,
            )

            records, audit = load_real_pose_records(
                pose_root, annotation_root, [self.condition], config
            )

            self.assertEqual(records[0].extraction_version, "gavd5_pose_v2_video_mode")
            self.assertEqual(
                audit["archive_schema_version_counts"],
                {"gavd5_pose_v2_video_mode": 1},
            )

    def test_source_csv_filename_mismatch_is_still_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            source = rf"C:\data-gavd\{self.condition}\wrong-sequence.csv"
            pose_root, annotation_root, config = self.make_fixture(
                Path(directory), source_csv=source
            )

            with self.assertRaisesRegex(
                ValueError, "source_csv/annotation filename mismatch"
            ):
                load_real_pose_records(
                    pose_root, annotation_root, [self.condition], config
                )

    def test_metadata_wrapper_requires_an_origin_generation(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            source = f"/data-gavd/{self.condition}/{self.sequence_id}.csv"
            pose_root, annotation_root, config = self.make_fixture(
                Path(directory), source_csv=source, origin_version=None
            )

            with self.assertRaisesRegex(ValueError, "missing 'cache_origin_version'"):
                load_real_pose_records(
                    pose_root, annotation_root, [self.condition], config
                )

    def test_scientific_digest_ignores_container_hash_but_covers_model_inputs(self) -> None:
        original = self.prepared_cohort()
        repacked = self.prepared_cohort(archive_sha256="archive-b")
        self.assertEqual(
            original.scientific_content_digest,
            repacked.scientific_content_digest,
        )

        repacked.missingness[0, 0] = 1.0
        self.assertNotEqual(
            original.scientific_content_digest,
            repacked.scientific_content_digest,
        )


if __name__ == "__main__":
    unittest.main()
