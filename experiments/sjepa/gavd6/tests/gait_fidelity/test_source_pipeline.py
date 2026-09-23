"""Source-branch integration with licensed geometry and GPU backends substituted.

The identity registry, reservation join, screening, edits, camera fitting,
StudentSpec construction, extraction adapter, side interventions, persistence
and shard contracts are the actual implementation. These tests provide software
evidence only; they cannot certify CUDA, OpenGL or pretrained checkpoints.
"""
from contextlib import ExitStack
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

from gavd6_sjepa.research_directions.gait_fidelity.data import load_dataset
from gavd6_sjepa.research_directions.gait_fidelity.preparation import prepare, merge_datasets
from gavd6_sjepa.research_directions.motion_preservation.body_geometry import _demo_body
from gavd6_sjepa.research_directions.motion_preservation.motion_data import MotionParameters
from gavd6_sjepa.research_directions.synthetic_training.estimators import StudentSpec
from gavd6_sjepa.research_directions.synthetic_training.rendering import SMPL_INDICES, project_points
from gavd6_sjepa.research_directions.synthetic_training_v2.contracts import atomic_json, sha256_file


def _motion(row, start_s=0., duration_s=5.12, fps=25.):
    n = round(duration_s * fps); t = start_s + np.arange(n) / fps
    phase = 2 * np.pi * 1.2 * (t - start_s)
    poses = np.zeros((n, 52, 3), np.float32)
    poses[:, 1, 0], poses[:, 2, 0] = .35 * np.sin(phase), -.35 * np.sin(phase)
    poses[:, 4, 0], poses[:, 5, 0] = .35 + .3 * np.sin(phase), .35 - .3 * np.sin(phase)
    trans = np.column_stack([np.zeros(n), np.full(n, .92), .15 * (t - start_s)]).astype(np.float32)
    return MotionParameters(poses.reshape(n, 156), trans, np.zeros(16, np.float32),
                            np.zeros((n, 8), np.float32), "neutral", fps, t,
                            dict(relative_path=row["relative_path"]))


class _Body:
    def __init__(self, body_model_root, dmpl_root, device="cuda"):
        assert device == "cuda"
        self.body_root = Path(body_model_root) / "smplh"
        self.dmpl_root = Path(dmpl_root)

    def forward(self, motion):
        assert isinstance(motion, MotionParameters)
        return _demo_body(motion)


class _EstimatorModel:
    def eval(self): return self

    def test_step(self, batch):
        points = np.column_stack([np.linspace(230, 340, 17), np.linspace(100, 390, 17)]).astype(np.float32)
        return [SimpleNamespace(pred_instances=SimpleNamespace(keypoints=points[None],
                 keypoint_scores=np.full((1, 17), .75, np.float32))) for _ in batch]


class _Estimator:
    model = _EstimatorModel()

    def _batch(self, images, boxes):
        assert boxes.shape == (len(images), 4)
        assert all(im.dtype == np.uint8 and im.shape == (480, 640, 3) for im in images)
        return images


def _load_estimator(spec, device="cuda"):
    assert isinstance(spec, StudentSpec)
    assert device == "cuda"
    return _Estimator()


def _render(renderer, body, recipe, seed, camera, original_faces):
    keypoints, depth = project_points(body.joints[:, SMPL_INDICES], camera, 640, 480, np.deg2rad(50.))
    assert np.min(depth) > 0
    xy, _ = project_points(body.vertices, camera, 640, 480, np.deg2rad(50.))
    boxes = np.concatenate([xy.min(1), xy.max(1)], -1)
    visible = np.ones(keypoints.shape[:-1], bool)
    if recipe.occlusion_fraction: visible[:, 8:] = False
    return dict(images=np.broadcast_to(np.zeros((1, 480, 640, 3), np.uint8), (len(body.timestamps), 480, 640, 3)),
                keypoints=keypoints, visible=visible, boxes=boxes,
                texture_path=str(renderer.textures[0]), background_path=str(renderer.backgrounds[0]))


class SourcePipelineTest(unittest.TestCase):
    def setUp(self):
        self.temporary = TemporaryDirectory(); self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.config = self.make_source_configuration()

    def make_source_configuration(self):
        from PIL import Image
        r = self.root
        for name in ("manifests", "amass", "body/smplh/neutral", "body/dmpls/neutral", "textures", "backgrounds", "old/bundle", "old/evidence"):
            (r / name).mkdir(parents=True, exist_ok=True)
        for name in ("body/smplh/neutral/model.npz", "body/dmpls/neutral/model.npz", "uv.obj", "checkpoint.pth"):
            (r / name).write_bytes(b"explicit-test-backend-placeholder")
        (r / "estimator.py").write_text("model = dict(type='TestBackend')\n")
        Image.new("RGB", (16, 16), "white").save(r / "textures/a.png")
        Image.new("RGB", (16, 16), "gray").save(r / "backgrounds/a.png")
        inventory, registry, splits, audit, reservation, old_records = [], [], [], [], [], []
        for person, split, starts in (("person-train", "train", (5., 8.)), ("person-dev", "validation", (5.,))):
            relative = f"{person}/walk.npz"; raw = r / "amass" / relative
            raw.parent.mkdir(); raw.write_bytes(person.encode()); source_hash = sha256_file(raw)
            inventory.append(dict(relative_path=relative, subject_id_candidate=person, status="ok", num_frames=1000, mocap_framerate=25))
            registry.append(dict(subject_id_candidate=person, identity=person, identity_audit_status="approved", excluded=False))
            splits.append(dict(identity=person, split=split))
            reservation.append(dict(person_id=person, canonical_person_id=person, original_split=split, reserved=False, exposure="known_development"))
            for start in starts:
                evidence = r / "old/evidence" / f"{person}-{start:g}.json"
                atomic_json(evidence, dict(screen_version="stv2-kinematic-screen-v1", reviewed_by="algorithm", status="pass",
                    review_mode="automated_development", relative_path=relative, canonical_person_id=person,
                    start_s=start, metrics={"test_backend": True}, thresholds={"test_backend": True}, source_sha256=source_hash))
                audit.append(dict(relative_path=relative, start_s=start, locomotion_status="algorithm_screened_locomotion",
                    audit_reviewer="stv2-kinematic-screen-v1", audit_evidence=str(evidence), audit_date="2026-09-22",
                    exposure="known_development", canonical_person_id=person))
                old_records.append(dict(relative_path=relative, start_s=start, canonical_person_id=person,
                    split="train" if split == "train" else "development", motion_hash=source_hash))
        for name, rows in (("amass_raw_inventory_eligible", inventory), ("amass_subject_registry", registry), ("amass_subject_splits", splits)):
            pd.DataFrame(rows).to_csv(r / "manifests" / f"{name}.csv", index=False)
        pd.DataFrame(audit).to_csv(r / "old/audit.csv", index=False)
        pd.DataFrame(reservation).to_csv(r / "old/reservations.csv", index=False)
        atomic_json(r / "old/bundle/manifest.json", dict(records=old_records))
        prep = dict(manifest_dir=str(r / "manifests"), amass_root=str(r / "amass"),
                    locomotion_audit=str(r / "old/audit.csv"), reservation_csv=str(r / "old/reservations.csv"),
                    body_model_root=str(r / "body"), dmpl_root=str(r / "body/dmpls"), uv_path=str(r / "uv.obj"),
                    texture_dir=str(r / "textures"), background_dir=str(r / "backgrounds"), review_mode="automated_development",
                    estimators=[dict(student_id=name, family=family, config=str(r / "estimator.py"), checkpoint=str(r / "checkpoint.pth"))
                                for name, family in (("rtmpose_m", "rtmpose"), ("vitpose_base", "vitpose"))])
        return dict(mode="source", source_bundle=str(r / "old/bundle"), preparation=prep,
                    data=dict(samples=128, hz=25., movement_levels_deg=[0, 5, 10, 15], held_level_deg=15, keep_videos=False))

    def test_source_branch_preserves_real_adapter_contracts_and_held_family(self):
        with ExitStack() as stack:
            for target, replacement in (
                ("gavd6_sjepa.research_directions.synthetic_training_v2.runtime.require_haic_runtime", lambda: {"test_backend": True}),
                ("gavd6_sjepa.research_directions.motion_preservation.motion_data.load_motion", _motion),
                ("gavd6_sjepa.research_directions.motion_preservation.body_geometry.SMPLHBody", _Body),
                ("gavd6_sjepa.research_directions.synthetic_training.estimators.load_estimator", _load_estimator),
                ("gavd6_sjepa.research_directions.gait_fidelity.preparation._render_fixed", _render),
            ): stack.enter_context(patch(target, replacement))
            path = prepare(self.config, self.root / "prepared")
        b = load_dataset(path)
        self.assertEqual(b.evidence_status, "automated-source-screen")
        self.assertTrue(all(r["extractor_family"] != "vitpose" for r in b.records if r["split"] == "train"))
        self.assertTrue(any(r["extractor_family"] == "vitpose" for r in b.records if r["split"] == "development"))
        self.assertTrue(all(r["movement_magnitude"] != 15 for r in b.records if r["split"] == "train"))
        self.assertTrue(any(r["movement_magnitude"] == 15 for r in b.records if r["split"] == "development"))
        starts = {r["start_s"] for r in b.records if r["split"] == "train"}
        np.testing.assert_allclose(sorted(starts), [5., 10.12], rtol=0, atol=1e-12)
        self.assertTrue(all("screening" in r["audit_evidence"] for r in b.records))
        # Naming conditions retain the exact same camera across all physical edits.
        for family in {r["source_family_id"] for r in b.records}:
            for camera in ("oblique", "side"):
                self.assertEqual(len({r["camera_hash"] for r in b.records if r["source_family_id"] == family and r["camera_id"] == camera}), 1)
        merged = merge_datasets([path], self.root / "merged")
        self.assertEqual(len(load_dataset(merged).records), len(b.records))
        self.assertGreater(len(list((self.root / "prepared/images").glob("*.png"))), 0)


if __name__ == "__main__": unittest.main()
