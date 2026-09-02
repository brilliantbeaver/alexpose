"""Regression tests for augmented-pose path resolution and legacy migration."""
from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
EXTRACTOR = ROOT / "notes" / "extract_augmented_poses.py"
MIGRATOR = ROOT / "notes" / "migrate_augmented_pose_artifacts.py"
CONTRACT = json.loads((ROOT / "pose_cache_contract.json").read_text())
CURRENT_EXTRACTION_VERSION = CONTRACT["current_extraction_version"]
LEGACY_EXTRACTION_VERSION = "gavd5_pose_v2_video_mode"


def load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def archive_digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_archive(
    path: Path,
    sequence_id: str,
    extraction_version: str = CURRENT_EXTRACTION_VERSION,
) -> None:
    frames = 2
    sequence = np.zeros((frames, 33, 4), dtype=np.float32)
    sequence[..., 3] = 0.9
    np.savez_compressed(
        path,
        sequence=sequence,
        sequence_id=np.asarray(sequence_id),
        video_id=np.asarray(f"video-{sequence_id}"),
        condition=np.asarray("normal"),
        frame_numbers=np.arange(1, frames + 1, dtype=np.int32),
        crop_bounds=np.zeros((frames, 4), dtype=np.int32),
        fps=np.asarray(30.0, dtype=np.float32),
        source_csv=np.asarray(f"{sequence_id}.csv"),
        source_video=np.asarray(f"{sequence_id}.mp4"),
        pose_model=np.asarray("pose_landmarker_lite.task"),
        pose_model_sha256=np.asarray("fixture"),
        extraction_version=np.asarray(extraction_version),
        cohort=np.asarray("augmented_normal"),
        bbox_source=np.asarray("mediapipe_pose_auto"),
        clip_id=np.asarray(sequence_id),
    )


def write_legacy_cohort(root: Path, count: int = 63) -> list[str]:
    pose_dir = root / "poses_augmented" / "normal"
    pose_dir.mkdir(parents=True)
    ids = [f"aug-fixture-{index:02d}" for index in range(count)]
    for sequence_id in ids:
        write_archive(
            pose_dir / f"{sequence_id}.npz",
            sequence_id,
            extraction_version=LEGACY_EXTRACTION_VERSION,
        )
    pd.DataFrame(
        {
            "sequence_id": ids,
            "status": ["accepted"] * count,
            "neuro_observed": [0.9] * count,
        }
    ).to_csv(root / "augmented_pose_extraction_report.csv", index=False)
    return ids


class ExtractorEnvironmentTests(unittest.TestCase):
    def test_honours_real_mode_cache_and_artifact_roots(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            temp = Path(temp_dir)
            env = {
                "GAVD_MODE": "real",
                "GAVD_CACHE_DIR": str(temp / "chosen-cache"),
                "GAVD_ARTIFACT_DIR": str(temp / "chosen-artifacts"),
            }
            with patch.dict(os.environ, env, clear=False):
                module = load_module(EXTRACTOR, "extractor_real_fixture")
            self.assertEqual(module.MODE, "real")
            self.assertEqual(module.CACHE_DIR, temp / "chosen-cache")
            self.assertEqual(module.ARTIFACT_DIR, temp / "chosen-artifacts" / "real")
            self.assertEqual(
                module.OUT_POSE_DIR,
                temp / "chosen-artifacts" / "real" / "poses_augmented" / "normal",
            )
            with self.assertRaises(FileNotFoundError):
                module.preflight()
            self.assertFalse((temp / "chosen-artifacts").exists())

    def test_rejects_smoke_mode_before_writing(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            temp = Path(temp_dir)
            env = {
                "GAVD_MODE": "smoke",
                "GAVD_CACHE_DIR": str(temp / "cache"),
                "GAVD_ARTIFACT_DIR": str(temp / "artifacts"),
            }
            with patch.dict(os.environ, env, clear=False):
                module = load_module(EXTRACTOR, "extractor_smoke_fixture")
            with self.assertRaisesRegex(RuntimeError, "real-data only"):
                module.preflight()
            self.assertFalse((temp / "artifacts").exists())


class PoseCacheContractTests(unittest.TestCase):
    def test_current_version_is_accepted_by_every_contract_consumer(self):
        self.assertIn(
            CURRENT_EXTRACTION_VERSION,
            CONTRACT["compatible_extraction_versions"],
        )
        with patch.dict(os.environ, {"GAVD_MODE": "real"}, clear=False):
            extractor = load_module(EXTRACTOR, "extractor_contract_fixture")
            migrator = load_module(MIGRATOR, "migrator_contract_fixture")
        self.assertEqual(
            extractor.CURRENT_EXTRACTION_VERSION,
            CURRENT_EXTRACTION_VERSION,
        )
        self.assertIn(
            CURRENT_EXTRACTION_VERSION,
            migrator.COMPATIBLE_EXTRACTION_VERSIONS,
        )

        for notebook_name in (
            "02_extract_and_watch_skeletons.ipynb",
            "04_pretrain_sjepa_on_normal.ipynb",
        ):
            notebook = json.loads((ROOT / notebook_name).read_text())
            code = "\n".join(
                "".join(cell["source"])
                for cell in notebook["cells"]
                if cell["cell_type"] == "code"
            )
            self.assertIn("pose_cache_contract.json", code)
            self.assertIn("COMPATIBLE_EXTRACTION_VERSIONS", code)
            if notebook_name.startswith("02_"):
                self.assertIn(
                    "extraction_version=np.asarray(CURRENT_EXTRACTION_VERSION)",
                    code,
                )
            else:
                self.assertIn(
                    "extraction_version not in COMPATIBLE_EXTRACTION_VERSIONS",
                    code,
                )

    def test_legacy_source_discovery_skips_incomplete_locations(self):
        with patch.dict(os.environ, {"GAVD_MODE": "real"}, clear=False):
            migrator = load_module(MIGRATOR, "migrator_discovery_fixture")
        with tempfile.TemporaryDirectory() as temp_dir:
            temp = Path(temp_dir)
            incomplete = temp / "incomplete"
            complete = temp / "complete"
            (complete / "poses_augmented" / "normal").mkdir(parents=True)
            (complete / "augmented_pose_extraction_report.csv").write_text(
                "sequence_id,status,neuro_observed\n"
            )
            self.assertEqual(
                migrator.choose_legacy_source_root([incomplete, complete]),
                complete,
            )


class LegacyMigrationTests(unittest.TestCase):
    def run_migrator(self, source: Path, destination: Path, *extra: str):
        env = {
            **os.environ,
            "GAVD_MODE": "real",
            "GAVD_ARTIFACT_DIR": str(destination.parent),
            "MPLCONFIGDIR": tempfile.gettempdir(),
        }
        return subprocess.run(
            [
                sys.executable,
                str(MIGRATOR),
                "--source-root",
                str(source),
                "--destination-root",
                str(destination),
                *extra,
            ],
            text=True,
            capture_output=True,
            env=env,
            check=False,
        )

    def test_dry_run_apply_and_idempotence_for_63_archives(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            temp = Path(temp_dir)
            source = temp / "legacy" / "real"
            destination = temp / "active" / "real"
            ids = write_legacy_cohort(source)

            dry_run = self.run_migrator(source, destination)
            self.assertEqual(dry_run.returncode, 0, dry_run.stderr)
            self.assertIn("Validated 63 eligible", dry_run.stdout)
            self.assertIn("Dry run only", dry_run.stdout)
            self.assertFalse(destination.exists())

            applied = self.run_migrator(source, destination, "--apply")
            self.assertEqual(applied.returncode, 0, applied.stderr)
            self.assertIn("Migrated and validated 63", applied.stdout)
            source_dir = source / "poses_augmented" / "normal"
            destination_dir = destination / "poses_augmented" / "normal"
            self.assertEqual(len(list(destination_dir.glob("*.npz"))), 63)
            self.assertEqual(
                archive_digest(source_dir / f"{ids[0]}.npz"),
                archive_digest(destination_dir / f"{ids[0]}.npz"),
            )

            repeated = self.run_migrator(source, destination, "--apply")
            self.assertEqual(repeated.returncode, 0, repeated.stderr)
            self.assertIn("Archives to copy: 0", repeated.stdout)

    def test_refuses_to_overwrite_a_different_destination_archive(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            temp = Path(temp_dir)
            source = temp / "legacy" / "real"
            destination = temp / "active" / "real"
            ids = write_legacy_cohort(source)
            destination_dir = destination / "poses_augmented" / "normal"
            destination_dir.mkdir(parents=True)
            write_archive(destination_dir / f"{ids[0]}.npz", ids[0])
            with np.load(destination_dir / f"{ids[0]}.npz", allow_pickle=False) as original:
                altered = original["sequence"].copy()
            altered[0, 0, 0] = 1.0
            np.savez_compressed(
                destination_dir / f"{ids[0]}.npz",
                sequence=altered,
                sequence_id=np.asarray(ids[0]),
                video_id=np.asarray("different"),
                condition=np.asarray("normal"),
                frame_numbers=np.asarray([1, 2], dtype=np.int32),
                crop_bounds=np.zeros((2, 4), dtype=np.int32),
                fps=np.asarray(30.0, dtype=np.float32),
                source_csv=np.asarray("different.csv"),
                source_video=np.asarray("different.mp4"),
                pose_model=np.asarray("pose_landmarker_lite.task"),
                pose_model_sha256=np.asarray("different"),
                extraction_version=np.asarray(CURRENT_EXTRACTION_VERSION),
                cohort=np.asarray("augmented_normal"),
                bbox_source=np.asarray("mediapipe_pose_auto"),
                clip_id=np.asarray(ids[0]),
            )
            result = self.run_migrator(source, destination, "--apply")
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("Refusing to overwrite different destination artifact", result.stderr)
            self.assertFalse((destination / "augmented_pose_extraction_report.csv").exists())


if __name__ == "__main__":
    unittest.main()
