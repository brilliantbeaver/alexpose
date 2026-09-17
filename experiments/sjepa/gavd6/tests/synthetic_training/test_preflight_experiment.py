"""Preflight failures must not alter scientific run state or use protected video."""
import importlib.util
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd
from PIL import Image


SCRIPT = Path(__file__).resolve().parents[2] / "scripts/research_directions/synthetic_training/preflight_experiment.py"
SPEC = importlib.util.spec_from_file_location("preflight_experiment", SCRIPT)
PREFLIGHT = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(PREFLIGHT)


class PreflightBoundaryTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        root = Path(self.temporary.name)
        self.root = root
        assets = root / "assets"
        assets.mkdir()
        self.images = assets / "images"
        self.images.mkdir()
        Image.fromarray(np.zeros((32, 32, 3), np.uint8)).save(self.images / "person.jpg")
        weights = assets / "model.pth"
        weights.write_bytes(b"CPU path fixture; never loaded as a model")
        pose_config = assets / "pose.py"
        pose_config.write_text("# fixture\n")
        (assets / "hubconf.py").write_text("# fixture\n")
        body = assets / "body"
        for group in ("smplh", "dmpls"):
            (body / group / "male").mkdir(parents=True)
        self.faces = np.array([[0, 1, 2]])
        np.savez(body / "smplh/male/model.npz", f=self.faces)
        np.savez(body / "dmpls/male/model.npz", eigvec=np.zeros((3, 3, 8)))
        self.uv = assets / "uv.npz"
        self.write_uv(self.faces)
        self.reservation = assets / "source-reservation.csv"
        self.reservation.write_text("video_id,role\n" + "".join(
            f"v{i},development\n" for i in range(18)) + "protected,confirmation\n")
        coco = assets / "person_keypoints_train2017.json"
        coco.write_text(json.dumps({
            "images": [{"id": 1, "file_name": "person.jpg", "width": 32, "height": 32}],
            "annotations": [{"id": 1, "image_id": 1, "keypoints": [8, 8, 2] * 17,
                             "bbox": [0, 0, 32, 32], "category_id": 1}],
        }))
        run = root / "run"
        (run / "source").mkdir(parents=True)
        (run / "source/outcomes.csv").write_text("existing outcomes must remain unchanged\n")
        (run / "config.json").write_text('{"existing": "preserve"}\n')
        self.output = run / "preflight/check"
        self.output.mkdir(parents=True)
        self.cfg = PREFLIGHT.RunConfig(
            run_root=str(run), amass_root=str(assets), amass_manifest_dir=str(assets),
            body_model_root=str(body), render_texture_dir=str(self.images),
            render_background_dir=str(self.images), render_uv_path=str(self.uv),
            coco_image_root=str(self.images), coco_annotations_json=str(coco),
            gavd_manifest_dir=str(assets), gavd_video_root=str(assets),
            gavd_reservation_csv=str(self.reservation), context_repo=str(assets),
            context_checkpoint=str(weights), image_checkpoint=str(weights), coco_replay_images=1,
            gavd_context_recordings=1, gavd_early_recordings=1, gavd_confirmation_recordings=1,
            students=[dict(student_id=role, family=role, role=role,
                           config=str(pose_config), checkpoint=str(weights))
                      for role in ("train", "validation", "held")],
        )
        self.rows = pd.DataFrame([dict(relative_path="motion.npz", raw_path=str(assets / "motion.npz"), duration_s=8)])
        self.gavd = pd.DataFrame(dict(video_id=[*[f"v{i}" for i in range(18)], "protected"], available=True))
        for name, value in (
            ("selected_motions", (self.rows, {"support": {}})),
            ("load_motion", SimpleNamespace(poses=np.zeros((64, 156)), gender="male")),
            ("load_gavd_manifest", self.gavd),
        ):
            patcher = patch.object(PREFLIGHT, name, return_value=value)
            patcher.start()
            self.addCleanup(patcher.stop)

    def write_uv(self, faces):
        np.savez(self.uv, uv=np.array([[0., 0.], [1., 0.], [0., 1.]]), face_uv=self.faces, faces=faces)

    def test_new_artifacts_stay_under_preflight_and_existing_bytes_survive(self):
        before = {p.relative_to(self.root): p.read_bytes() for p in self.root.rglob("*") if p.is_file()}
        report = {}
        _, replay = PREFLIGHT.check_assets(self.cfg, self.output, report)
        self.assertEqual(len(replay), 1)
        self.assertEqual(report["decoded_coco_replay_images"], 1)
        self.assertEqual(report["gavd_readiness"]["available_permitted_unique_video_ids"], 18)
        for relative, content in before.items():
            self.assertEqual((self.root / relative).read_bytes(), content)
        for path in self.root.rglob("*"):
            if path.is_file() and path.relative_to(self.root) not in before:
                self.assertTrue(path.is_relative_to(self.output), path)

    def test_reservation_directory_is_rejected(self):
        cfg = PREFLIGHT.replace(self.cfg, gavd_reservation_csv=str(self.reservation.parent))
        with self.assertRaisesRegex(FileNotFoundError, "gavd_reservation_csv"):
            PREFLIGHT.check_assets(cfg, self.output, {})

    def test_protected_recording_cannot_fill_the_minimum(self):
        # 17 permitted + one protected = 18 existing files, but only 17 eligible.
        self.gavd.loc[self.gavd.video_id.eq("v0"), "available"] = False
        with self.assertRaisesRegex(ValueError, "17 available permitted.*at least 18"):
            PREFLIGHT.check_assets(self.cfg, self.output, {})

    def test_any_protected_assignment_wins_over_development(self):
        with self.reservation.open("a") as stream:
            stream.write("v0,confirmation\n")
        with self.assertRaisesRegex(ValueError, "17 available permitted"):
            PREFLIGHT.check_assets(self.cfg, self.output, {})

    def test_incompatible_uv_face_order_is_rejected(self):
        self.write_uv(self.faces[:, ::-1])
        with self.assertRaisesRegex(ValueError, "exactly match"):
            PREFLIGHT.check_assets(self.cfg, self.output, {})

    def test_corrupt_appearance_image_is_rejected(self):
        (self.images / "corrupt.jpg").write_bytes(b"invalid image")
        with self.assertRaisesRegex(OSError, "cannot identify image"):
            PREFLIGHT.check_assets(self.cfg, self.output, {})
