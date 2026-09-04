from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from gavd6_sjepa.data_foundations.amass_core11_conversion_pipeline import (
    CHANNEL_NAMES,
    COORDINATE_FRAME,
    CORE11_NAMES,
    SCHEMA,
    ConversionConfig,
    ConversionError,
    _attach_subject_splits,
    _atomic_save_npz,
    _read_inventory,
    _safe_relative_path,
    canonical_times,
    convert_inventory_row,
    convert_sequence_arrays,
    estimate_forward_frame,
    load_amass_sequence,
    reconstruct_world,
    resample_at_times,
    sha256_file,
)


def synthetic_walk(num_frames: int = 61) -> tuple[np.ndarray, np.ndarray]:
    """Asymmetric named skeleton travelling along +x in a z-up world."""

    coordinates = np.zeros((num_frames, 11, 3), dtype=np.float32)
    x = np.linspace(0.0, 1.0, num_frames, dtype=np.float32)
    pelvis = np.stack([x, np.zeros_like(x), np.ones_like(x)], axis=1)
    offsets = np.array(
        [
            [0.00, 0.00, 0.00],  # pelvis
            [0.00, 0.16, 0.00],  # left hip
            [0.00, -0.14, 0.00],  # right hip
            [0.03, 0.17, -0.48],  # left knee
            [-0.02, -0.14, -0.46],  # right knee
            [0.06, 0.18, -0.94],  # left ankle
            [-0.04, -0.15, -0.91],  # right ankle
            [0.00, 0.18, -1.00],  # left heel
            [-0.10, -0.15, -0.97],  # right heel
            [0.25, 0.19, -0.98],  # left forefoot
            [0.16, -0.16, -0.95],  # right forefoot
        ],
        dtype=np.float32,
    )
    coordinates[:] = pelvis[:, None, :] + offsets[None, :, :]
    return coordinates, np.ones((num_frames, 11), dtype=bool)


class FakeBackend:
    def __init__(self, coordinates: np.ndarray, valid: np.ndarray) -> None:
        self.coordinates = coordinates
        self.valid = valid

    def model_info(self, gender: str):
        return {
            "backend": "fake-test-backend",
            "backend_version": "1",
            "body_model_family": "Extended SMPL+H",
            "body_model_sha256": f"body-{gender}",
            "dmpl_model_sha256": f"dmpl-{gender}",
        }

    def forward(self, sequence):
        return self.coordinates.copy(), self.valid.copy()


class BodyFrameTests(unittest.TestCase):
    def test_conversion_config_enforces_frozen_contract(self):
        with self.assertRaisesRegex(ValueError, "exactly 30"):
            ConversionConfig(canonical_fps=25.0)
        with self.assertRaisesRegex(ValueError, "nonnegative"):
            ConversionConfig(min_travel_m=-0.1)
        with self.assertRaisesRegex(ValueError, r"\[0, 1\]"):
            ConversionConfig(min_travel_straightness=float("nan"))

    def test_travel_frame_is_right_handed_and_preserves_anatomical_names(self):
        world, valid = synthetic_walk()
        converted = convert_sequence_arrays(
            world,
            valid,
            source_fps=60.0,
            config=ConversionConfig(canonical_fps=30.0),
        )
        frame = converted["frame"]

        self.assertTrue(frame.method.startswith("pelvis_travel"))
        np.testing.assert_allclose(frame.forward_world, [1.0, 0.0, 0.0], atol=1e-6)
        np.testing.assert_allclose(frame.up_world, [0.0, 0.0, 1.0], atol=1e-6)
        np.testing.assert_allclose(frame.lateral_world, [0.0, 1.0, 0.0], atol=1e-6)
        physical_basis = np.stack(
            [frame.forward_world, frame.lateral_world, frame.up_world], axis=1
        )
        self.assertAlmostEqual(float(np.linalg.det(physical_basis)), 1.0, places=6)
        self.assertAlmostEqual(
            float(np.linalg.det(frame.world_to_body_transform)), -1.0, places=6
        )
        self.assertGreater(converted["coordinates"][0, 1, 2], 0.0)
        self.assertLess(converted["coordinates"][0, 2, 2], 0.0)
        self.assertEqual(converted["coordinates"].shape, (31, 11, 3))
        self.assertEqual(converted["valid"].dtype, np.bool_)

    def test_yaw_and_translation_do_not_change_body_coordinates(self):
        world, valid = synthetic_walk()
        config = ConversionConfig(canonical_fps=30.0)
        reference = convert_sequence_arrays(world, valid, source_fps=60.0, config=config)

        angle = np.deg2rad(73.0)
        rotation = np.array(
            [
                [np.cos(angle), -np.sin(angle), 0.0],
                [np.sin(angle), np.cos(angle), 0.0],
                [0.0, 0.0, 1.0],
            ]
        )
        moved = world @ rotation.T + np.array([7.0, -4.0, 2.5])
        transformed = convert_sequence_arrays(moved, valid, source_fps=60.0, config=config)

        np.testing.assert_allclose(
            transformed["coordinates"], reference["coordinates"], atol=2e-6
        )

    def test_static_sequence_uses_declared_hip_facing_fallback(self):
        world, valid = synthetic_walk(num_frames=5)
        world[:, :, 0] = world[0, :, 0]
        frame = estimate_forward_frame(
            world,
            valid,
            min_travel_m=0.1,
            min_travel_straightness=0.2,
            min_abs_lateral_hip_alignment=0.5,
            forward_policy="travel-or-hips",
        )
        self.assertEqual(frame.method, "hip_facing_fallback")
        np.testing.assert_allclose(frame.forward_world, [1.0, 0.0, 0.0], atol=1e-6)
        np.testing.assert_allclose(frame.lateral_world, [0.0, 1.0, 0.0], atol=1e-6)

    def test_require_travel_rejects_static_sequence(self):
        world, valid = synthetic_walk(num_frames=5)
        world[:, :, 0] = world[0, :, 0]
        with self.assertRaisesRegex(ConversionError, "require-travel"):
            estimate_forward_frame(
                world,
                valid,
                min_travel_m=0.1,
                min_travel_straightness=0.2,
                min_abs_lateral_hip_alignment=0.5,
                forward_policy="require-travel",
            )

    def test_side_step_uses_hip_facing_to_keep_anatomical_mirror_axis(self):
        world, valid = synthetic_walk()
        travel = np.linspace(0.0, 1.0, len(world), dtype=np.float32)
        world[:, :, 0] -= np.linspace(0.0, 1.0, len(world), dtype=np.float32)[:, None]
        world[:, :, 1] += travel[:, None]
        converted = convert_sequence_arrays(
            world,
            valid,
            source_fps=60.0,
            config=ConversionConfig(canonical_fps=30.0),
        )
        self.assertEqual(
            converted["frame"].method,
            "hip_facing_fallback_due_to_travel_anatomy_misalignment",
        )
        self.assertGreater(converted["coordinates"][0, 1, 2], 0.0)
        self.assertLess(converted["coordinates"][0, 2, 2], 0.0)

    def test_invalid_pelvis_invalidates_the_entire_frame(self):
        world, valid = synthetic_walk()
        valid[30, 0] = False
        converted = convert_sequence_arrays(
            world,
            valid,
            source_fps=60.0,
            config=ConversionConfig(canonical_fps=30.0),
        )
        self.assertFalse(converted["valid"][15].any())
        np.testing.assert_array_equal(converted["coordinates"][15], 0.0)

    def test_saved_transform_reconstructs_resampled_world_coordinates(self):
        world, valid = synthetic_walk()
        converted = convert_sequence_arrays(
            world,
            valid,
            source_fps=60.0,
            config=ConversionConfig(canonical_fps=30.0),
        )
        reconstructed = reconstruct_world(
            converted["coordinates_m"],
            converted["pelvis_world_m"],
            converted["frame"].world_to_body_transform,
        )
        expected = world[::2]
        np.testing.assert_allclose(reconstructed, expected, atol=2e-6)


class ResamplingTests(unittest.TestCase):
    def test_grid_includes_exact_one_second_endpoint(self):
        times = canonical_times(61, source_fps=60.0, target_fps=30.0)
        self.assertEqual(len(times), 31)
        self.assertEqual(times[0], 0.0)
        self.assertEqual(times[-1], 1.0)

    def test_linear_interpolation_and_conservative_validity(self):
        values = np.array([[[0.0]], [[1.0]], [[2.0]]], dtype=np.float32)
        valid = np.array([[True], [False], [True]])
        times = np.arange(5, dtype=np.float64) / 4.0
        result, result_valid = resample_at_times(
            values, valid, source_fps=2.0, target_times=times
        )
        np.testing.assert_array_equal(result_valid[:, 0], [True, False, False, False, True])
        np.testing.assert_allclose(result[:, 0, 0], [0.0, 0.0, 0.0, 0.0, 2.0])

    def test_resampling_refuses_extrapolation(self):
        values = np.zeros((2, 1, 3), dtype=np.float32)
        valid = np.ones((2, 1), dtype=bool)
        with self.assertRaisesRegex(ConversionError, "inside the source"):
            resample_at_times(
                values,
                valid,
                source_fps=30.0,
                target_times=np.array([0.0, 1.0]),
            )


class ArchiveAndIntegrationTests(unittest.TestCase):
    def _write_amass(self, path: Path, frames: int = 61, gender: object = "female") -> None:
        np.savez_compressed(
            path,
            poses=np.zeros((frames, 156), dtype=np.float32),
            trans=np.zeros((frames, 3), dtype=np.float32),
            betas=np.zeros(16, dtype=np.float32),
            dmpls=np.zeros((frames, 8), dtype=np.float32),
            gender=np.asarray(gender),
            mocap_framerate=np.asarray(60.0),
        )

    def test_load_decodes_byte_gender_and_slices_required_parameters(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "walk_poses.npz"
            self._write_amass(path, gender=b"male")
            sequence = load_amass_sequence(path)
        self.assertEqual(sequence.gender, "male")
        self.assertEqual(sequence.poses.shape, (61, 156))
        self.assertEqual(sequence.betas.shape, (16,))
        self.assertEqual(sequence.dmpls.shape, (61, 8))

    def test_load_rejects_wrong_pose_width(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "bad_poses.npz"
            np.savez_compressed(
                path,
                poses=np.zeros((2, 72), dtype=np.float32),
                trans=np.zeros((2, 3), dtype=np.float32),
                betas=np.zeros(16, dtype=np.float32),
                dmpls=np.zeros((2, 8), dtype=np.float32),
                gender=np.asarray("female"),
                mocap_framerate=np.asarray(60.0),
            )
            with self.assertRaisesRegex(ConversionError, r"\[T, 156\]"):
                load_amass_sequence(path)

    def test_path_traversal_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            with self.assertRaisesRegex(ConversionError, "unsafe"):
                _safe_relative_path(root, "../escape_poses.npz")
            with self.assertRaisesRegex(ConversionError, "unsafe"):
                _safe_relative_path(root, "/absolute_poses.npz")

    def test_subject_splits_are_attached_and_cross_split_identity_is_rejected(self):
        rows = [{"relative_path": "A/s/m_poses.npz", "subject_id_candidate": "A::s"}]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "splits.csv"
            path.write_text(
                "subject_id_candidate,identity,split\nA::s,A::person-1,train\n",
                encoding="utf-8",
            )
            attached = _attach_subject_splits(rows, path)
            self.assertEqual(attached[0]["identity"], "A::person-1")
            self.assertEqual(attached[0]["split"], "train")

            path.write_text(
                "subject_id_candidate,identity,split\n"
                "A::s,A::person-1,train\n"
                "A::other,A::person-1,test\n",
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ConversionError, "cross subject splits"):
                _attach_subject_splits(rows, path)

    def test_inventory_detects_duplicate_normalized_paths(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "inventory.csv"
            path.write_text(
                "relative_path,status\n"
                "A/subject/walk_poses.npz,ok\n"
                "A//subject/./walk_poses.npz,ok\n",
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ConversionError, "duplicate relative paths"):
                _read_inventory(path)

    def test_output_npz_reopens_without_pickle(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "out.npz"
            _atomic_save_npz(
                path,
                {
                    "joint_names": np.asarray(CORE11_NAMES),
                    "channel_names": np.asarray(CHANNEL_NAMES),
                    "provenance_json": np.asarray(json.dumps({"schema": SCHEMA})),
                },
            )
            with np.load(path, allow_pickle=False) as archive:
                self.assertEqual(archive["joint_names"].tolist(), list(CORE11_NAMES))
                self.assertEqual(json.loads(str(archive["provenance_json"].item()))["schema"], SCHEMA)

    def test_inventory_row_end_to_end_and_idempotent_skip(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            amass_root = root / "amass"
            output_root = root / "converted"
            source = amass_root / "ACCAD" / "subject" / "walk_poses.npz"
            source.parent.mkdir(parents=True)
            self._write_amass(source)
            world, valid = synthetic_walk()
            backend = FakeBackend(world, valid)
            row = {
                "relative_path": "ACCAD/subject/walk_poses.npz",
                "source_dataset": "ACCAD",
                "subject_id_candidate": "ACCAD::subject",
                "motion_id": "walk",
                "sha256": sha256_file(source),
                "status": "ok",
            }
            config = ConversionConfig(canonical_fps=30.0)

            first = convert_inventory_row(
                row,
                amass_root=amass_root.resolve(),
                output_root=output_root.resolve(),
                backend=backend,
                config=config,
                overwrite=False,
                verify_source_sha256=False,
            )
            second = convert_inventory_row(
                row,
                amass_root=amass_root.resolve(),
                output_root=output_root.resolve(),
                backend=backend,
                config=config,
                overwrite=False,
                verify_source_sha256=False,
            )

            self.assertEqual(first["status"], "converted")
            self.assertEqual(second["status"], "skipped_valid_existing")
            with self.assertRaisesRegex(ConversionError, "stale or unreadable provenance"):
                convert_inventory_row(
                    row,
                    amass_root=amass_root.resolve(),
                    output_root=output_root.resolve(),
                    backend=backend,
                    config=config,
                    overwrite=False,
                    verify_source_sha256=True,
                )
            changed_row = {**row, "identity": "ACCAD::subject", "split": "train"}
            with self.assertRaisesRegex(ConversionError, "stale or unreadable provenance"):
                convert_inventory_row(
                    changed_row,
                    amass_root=amass_root.resolve(),
                    output_root=output_root.resolve(),
                    backend=backend,
                    config=config,
                    overwrite=False,
                    verify_source_sha256=False,
                )
            output = output_root / "ACCAD" / "subject" / "walk_core11.npz"
            with np.load(output, allow_pickle=False) as archive:
                self.assertEqual(archive["coordinates"].shape, (31, 11, 3))
                self.assertEqual(archive["valid"].shape, (31, 11))
                replay = reconstruct_world(
                    archive["coordinates_m"],
                    archive["pelvis_world_m"],
                    archive["world_to_body_transform"],
                )
                provenance = json.loads(str(archive["provenance_json"].item()))
            np.testing.assert_allclose(replay, world[::2], atol=2e-6)
            self.assertEqual(provenance["schema"]["name"], SCHEMA)
            self.assertEqual(provenance["coordinate_frame"]["name"], COORDINATE_FRAME)
            self.assertTrue(provenance["body_model"]["dmpls_used"])


if __name__ == "__main__":
    unittest.main()
