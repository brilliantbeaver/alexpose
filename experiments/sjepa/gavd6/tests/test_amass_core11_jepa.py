from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from amass_core11_jepa import (
    CHANNEL_NAMES,
    COORDINATE_FRAME,
    JOINT_NAMES,
    MIRROR_CHANNEL,
    MIRROR_PAIRS,
    SCHEMA,
    VALID_FINGERPRINTS,
    VARIANTS,
    Core11WindowDataset,
    build_window_index,
    core11_train_config,
    load_conversion_manifest,
    mirror_validity,
    train_streaming_variant,
    validate_archives,
    window_starts,
)
from gait_parity_jepa import anatomical_mirror, build_model


class AmassCore11JepaTests(unittest.TestCase):
    def _fixture(self, root: Path) -> Path:
        rows = []
        for index, (split, gender) in enumerate(
            (("train", "female"), ("validation", "male"), ("test", "male"))
        ):
            frames = 68
            relative = f"dataset/sequence-{index}_core11.npz"
            path = root / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            coordinates = np.random.default_rng(index).normal(
                0, 0.1, (frames, 11, 3)
            ).astype(np.float32)
            valid = np.ones((frames, 11), dtype=bool)
            source_sha256 = "a" + f"{index + 1:063x}"
            body_sha256 = "b" + f"{index:063x}"
            dmpl_sha256 = "d" + f"{index:063x}"
            source = {
                "relative_path": f"dataset/sequence-{index}_poses.npz",
                "source_dataset": "dataset",
                "subject_id_candidate": f"dataset::subject-{index}",
                "identity": f"dataset::subject-{index}",
                "split": split,
                "motion_id": f"sequence-{index}",
                "gender": gender,
                "sha256": source_sha256,
                "sha256_verified_against_current_file": True,
            }
            provenance = {
                "source": source,
                "body_model": {
                    "body_model_sha256": body_sha256,
                    "dmpl_model_sha256": dmpl_sha256,
                },
                "schema": {
                    "name": SCHEMA,
                    "joint_names": JOINT_NAMES,
                    "channel_names": CHANNEL_NAMES,
                },
                "resampling": {"canonical_fps": 30, "canonical_frames": frames},
                "conversion": {"fingerprint": VALID_FINGERPRINTS[gender]},
            }
            np.savez_compressed(
                path,
                coordinates=coordinates,
                coordinates_m=coordinates,
                valid=valid,
                canonical_times_s=np.arange(frames, dtype=np.float64) / 30,
                pelvis_world_m=np.zeros((frames, 3), dtype=np.float32),
                world_to_body_transform=np.eye(3, dtype=np.float64),
                body_to_world_transform=np.eye(3, dtype=np.float64),
                physical_basis_world=np.eye(3, dtype=np.float64),
                leg_length_m=np.asarray(1.0, dtype=np.float64),
                joint_names=np.asarray(JOINT_NAMES),
                channel_names=np.asarray(CHANNEL_NAMES),
                provenance_json=np.asarray(json.dumps(provenance)),
            )
            rows.append(
                {
                    "relative_path": source["relative_path"],
                    "tensor_relative_path": relative,
                    "source_dataset": source["source_dataset"],
                    "subject_id_candidate": source["subject_id_candidate"],
                    "identity": source["identity"],
                    "split": split,
                    "motion_id": source["motion_id"],
                    "gender": gender,
                    "canonical_fps": 30.0,
                    "canonical_frames": frames,
                    "source_sha256": source_sha256,
                    "source_sha256_verified": True,
                    "body_model_sha256": body_sha256,
                    "dmpl_model_sha256": dmpl_sha256,
                    "schema": SCHEMA,
                    "coordinate_frame": COORDINATE_FRAME,
                    "conversion_fingerprint": VALID_FINGERPRINTS[gender],
                    "status": "converted",
                }
            )
        manifest_path = root / "manifest.csv"
        pd.DataFrame(rows).to_csv(manifest_path, index=False)
        return manifest_path

    def test_frozen_window_policy(self):
        self.assertEqual(window_starts(63), [])
        self.assertEqual(window_starts(64), [0])
        self.assertEqual(window_starts(96), [0, 32])
        self.assertEqual(window_starts(97), [0, 32, 33])

    def test_core11_mirror_is_an_involution(self):
        coordinates = torch.randn(2, 64, 11, 3)
        mirrored = anatomical_mirror(coordinates, MIRROR_PAIRS, MIRROR_CHANNEL)
        restored = anatomical_mirror(mirrored, MIRROR_PAIRS, MIRROR_CHANNEL)
        torch.testing.assert_close(restored, coordinates)
        swap = torch.tensor([0, 2, 1, 4, 3, 6, 5, 8, 7, 10, 9])
        torch.testing.assert_close(
            mirrored[..., 0], coordinates[..., 0].index_select(2, swap)
        )
        torch.testing.assert_close(
            mirrored[..., 1], coordinates[..., 1].index_select(2, swap)
        )
        torch.testing.assert_close(
            mirrored[..., 2], -coordinates[..., 2].index_select(2, swap)
        )

        valid = torch.zeros(1, 16, 11, dtype=torch.bool)
        valid[..., 1] = True
        mirrored_valid = mirror_validity(valid)
        self.assertTrue(mirrored_valid[..., 2].all())
        self.assertFalse(mirrored_valid[..., 1].any())

    def test_loader_contract_and_one_update_per_variant(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            manifest = load_conversion_manifest(self._fixture(root))
            windows = build_window_index(manifest)
            self.assertEqual(validate_archives(manifest, root), 3)
            dataset = Core11WindowDataset(manifest, windows, root, split="train")
            sample = dataset[0]
            self.assertEqual(tuple(sample["coordinates"].shape), (64, 11, 3))
            self.assertEqual(tuple(sample["valid"].shape), (16, 11))

            config = core11_train_config("smoke")
            for variant in VARIANTS:
                model = build_model(config, variant, seed=7)
                _, _, _, history, _ = train_streaming_variant(
                    model,
                    dataset,
                    config,
                    torch.device("cpu"),
                    updates=1,
                    seed=7,
                    num_workers=0,
                )
                self.assertEqual(len(history), 1)
                self.assertTrue(np.isfinite(history["total_loss"]).all())


if __name__ == "__main__":
    unittest.main()
