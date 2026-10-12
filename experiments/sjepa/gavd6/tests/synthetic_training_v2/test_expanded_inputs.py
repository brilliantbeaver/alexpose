"""Expanded source selection never manufactures people or review evidence."""
import contextlib
import csv
import importlib.util
import io
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

from gavd6_sjepa.research_directions.motion_preservation.body_geometry import demo_motion
from gavd6_sjepa.research_directions.synthetic_training_v2.preparation import audited_motion_windows


REPO = Path(__file__).resolve().parents[2]
SCRIPT = REPO / "scripts/research_directions/synthetic_training_v2/expanded_inputs.py"
SPEC = importlib.util.spec_from_file_location("stv2_expanded_inputs", SCRIPT)
EXPANDED = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(EXPANDED)
BODY = "gavd6_sjepa.research_directions.motion_preservation.body_geometry.SMPLHBody"


class FixtureBody:
    def __init__(self, *args, **kwargs):
        pass

    def forward(self, motion):
        # Procedural geometry is a software fixture, not a licensed source run.
        body, _ = demo_motion(frames=64, fps=25)
        return body


class ExpandedInputTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.amass = self.root / "amass"
        self.manifests = self.root / "manifests"
        self.work = self.root / "work"
        self.output = self.work / "inputs/expanded-01"
        self.manifests.mkdir()
        (self.work / "config").mkdir(parents=True)
        inventory, registry, splits = [], [], []
        for split, person in [("train", "t1"), ("train", "t2"), ("train", "t3"),
                              ("validation", "v1"), ("validation", "v2"),
                              ("validation", "v3"), ("test", "z1")]:
            relative = f"Dataset/{person}/0000_treadmill_norm_poses.npz"
            source = self.amass / relative
            source.parent.mkdir(parents=True)
            np.savez(source, poses=np.zeros((451, 156)), trans=np.zeros((451, 3)),
                     dmpls=np.zeros((451, 8)), betas=np.zeros(16), gender="male", mocap_framerate=25)
            inventory.append(dict(relative_path=relative, subject_id_candidate=person, status="ok",
                                  num_frames=451, mocap_framerate=25, gender="male"))
            registry.append(dict(subject_id_candidate=person, identity=person,
                                 identity_audit_status="approved", excluded=False))
            splits.append(dict(identity=person, split=split))
        for filename, values in (("amass_raw_inventory_eligible.csv", inventory),
                                 ("amass_subject_registry.csv", registry),
                                 ("amass_subject_splits.csv", splits)):
            pd.DataFrame(values).to_csv(self.manifests / filename, index=False)
        self.config = self.work / "config/preparation.json"
        self.config.write_text(json.dumps(dict(manifest_dir=str(self.manifests), amass_root=str(self.amass),
                                             body_model_root=str(self.root / "body"),
                                             dmpl_root=str(self.root / "dmpls"))))
        self.argv = ["--config", str(self.config), "--output", str(self.output),
                     "--train-people", "2", "--development-people", "2"]

    def run_fixture(self, extra=None, body=FixtureBody):
        with patch(BODY, body), contextlib.redirect_stdout(io.StringIO()):
            return EXPANDED.run(EXPANDED.parser().parse_args(self.argv + (extra or [])))

    def reservations(self, rows):
        folder = self.work / "inputs/review-drafts"
        folder.mkdir(parents=True, exist_ok=True)
        path = folder / "person-reservations.csv"
        with path.open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=["person_id", "canonical_person_id",
                                                        "original_split", "reserved", "exposure"])
            writer.writeheader()
            writer.writerows(rows)
        return path

    def old_bundle(self, *people):
        path = self.root / "old-bundle"
        path.mkdir()
        (path / "manifest.json").write_text(json.dumps(dict(
            schema="coco-body12-xy-v1",
            records=[dict(person_id=person, canonical_person_id=person) for person in people],
        )))
        return path

    def test_defaults_are_larger_people_panel_with_nonoverlapping_physical_windows(self):
        args = EXPANDED.parser().parse_args(["--config", "input.json", "--output", "out"])
        self.assertEqual((args.train_people, args.development_people, args.windows_per_person), (24, 8, 4))
        self.assertEqual(args.max_candidates, 512)
        self.assertEqual(EXPANDED.window_starts(4), [5., 8., 11., 14.])
        self.assertGreaterEqual(min(np.diff(EXPANDED.window_starts(4))), 64 / 25)
        with self.assertRaises(ValueError):
            EXPANDED.window_starts(True)

    def test_output_is_accepted_by_existing_preparation_contract(self):
        original = self.config.read_bytes()
        report = self.run_fixture()
        self.assertEqual(report["counts"], {"train": 2, "validation": 2})
        self.assertEqual(report["selected_recordings"], 4)
        self.assertEqual(report["selected_windows"], 16)
        self.assertEqual(report["minimum_duration_s"], 16.52)
        allowed, rejected = audited_motion_windows(
            self.manifests, self.amass, self.output / "locomotion-audit.csv",
            self.output / "person-reservations.csv", review_mode="automated_development")
        self.assertEqual(len(allowed), 16)
        self.assertTrue(rejected.empty)
        self.assertEqual(allowed.groupby("canonical_person_id").size().tolist(), [4, 4, 4, 4])
        self.assertTrue(allowed.audit_reviewer.eq(EXPANDED.legacy.VERSION).all())
        self.assertTrue(allowed.exposure.eq("unknown").all())
        self.assertEqual(self.config.read_bytes(), original)
        evidence = json.loads(Path(allowed.audit_evidence.iloc[0]).read_text())
        self.assertEqual(evidence["helper_sha256"], report["helper_sha256"])
        self.assertEqual(evidence["screening_helper_sha256"], report["screening_helper_sha256"])
        self.assertNotEqual(evidence["helper_sha256"], evidence["screening_helper_sha256"])
        with np.load(self.output / "playback/joints.npz", allow_pickle=False) as data:
            self.assertEqual(data["joints"].shape, (16, 64, 22, 3))
            self.assertEqual(data["timestamps"].shape, (16, 64))

    def test_previous_training_and_development_people_are_both_excluded(self):
        old = self.old_bundle("t1", "v1")
        report = self.run_fixture(["--exclude-bundle", str(old)])
        self.assertEqual(report["selected_canonical_people"], ["t2", "t3", "v2", "v3"])
        self.assertEqual(report["excluded_previous_people"], ["t1", "v1"])
        self.assertEqual(len(report["exclusion_sources"]), 1)
        self.assertFalse((old / "inputs.npz").exists())  # identity-only exclusion needs no labels

    def test_previous_identity_exclusion_follows_current_aliases(self):
        old = self.old_bundle("t1")
        people = {"t1": {"canonical_person_id": "renamed"}, "t2": {"canonical_person_id": "renamed"}}
        excluded, _ = EXPANDED.excluded_people([old], people)
        self.assertEqual(excluded, {"t1", "renamed"})

    def test_candidates_round_robin_people_before_second_recordings(self):
        people = {person: dict(person_id=person, canonical_person_id=person) for person in ("a", "b")}
        rows = [dict(person_id=person, original_split="train", available=True, duration_s=20.,
                     relative_path=f"Dataset/{person}/{name}.npz")
                for person, name in (("a", "0000_treadmill_norm"), ("a", "0001_walk"), ("b", "0001_walk"))]
        candidates = EXPANDED.candidate_rows(pd.DataFrame(rows), people, set(), set(), [5., 8., 11., 14.])
        self.assertEqual([row["person_id"] for row in candidates], ["a", "b", "a"])

    def test_reserved_and_test_aliases_never_enter_screening(self):
        path = self.reservations([
            dict(person_id="t1", canonical_person_id="same", original_split="train", reserved="", exposure=""),
            dict(person_id="z1", canonical_person_id="same", original_split="test", reserved="", exposure=""),
            dict(person_id="v1", canonical_person_id="v1", original_split="validation", reserved="true", exposure="known"),
        ])
        before = path.read_bytes()
        report = self.run_fixture()
        self.assertEqual(report["selected_canonical_people"], ["t2", "t3", "v2", "v3"])
        attempted = {row["person_id"] for row in report["attempts"]}
        self.assertTrue(attempted.isdisjoint({"t1", "z1", "v1"}))
        self.assertEqual(path.read_bytes(), before)

    def test_short_recordings_are_excluded_before_geometry(self):
        path = self.manifests / "amass_raw_inventory_eligible.csv"
        table = pd.read_csv(path)
        table.loc[table.subject_id_candidate.isin(["t1", "v1"]), "num_frames"] = 401
        table.to_csv(path, index=False)
        report = self.run_fixture()
        self.assertEqual(report["selected_canonical_people"], ["t2", "t3", "v2", "v3"])
        self.assertFalse(any(row["person_id"] in {"t1", "v1"} for row in report["attempts"]))

    def test_shortage_fails_with_counts_without_manufacturing_smaller_panel(self):
        old = self.old_bundle("t1", "t2")
        with patch(BODY) as body:
            with self.assertRaisesRegex(ValueError, "no smaller panel was accepted"):
                EXPANDED.run(EXPANDED.parser().parse_args(self.argv + ["--exclude-bundle", str(old)]))
            body.assert_not_called()
        report = json.loads((self.output / "screen.json").read_text())
        self.assertEqual(report["status"], "insufficient_metadata_candidates")
        self.assertEqual(report["counts"], {"train": 0, "validation": 0})
        self.assertEqual(report["available_people"], {"train": 1, "validation": 3})
        self.assertEqual(report["attempts"], [])
        self.assertEqual(report["requested_people"], {"train": 2, "validation": 2})
        self.assertFalse((self.output / "locomotion-audit.csv").exists())
        self.assertFalse((self.output / "person-reservations.csv").exists())

    def test_candidate_limit_counts_failed_attempts_and_does_not_relax_rule(self):
        class StaticBody(FixtureBody):
            def forward(self, motion):
                body = super().forward(motion)
                body.joints[:] = body.joints[0]
                return body
        with self.assertRaisesRegex(ValueError, "no smaller panel was accepted"):
            self.run_fixture(["--max-candidates", "4"], body=StaticBody)
        report = json.loads((self.output / "screen.json").read_text())
        self.assertEqual(len(report["attempts"]), 4)
        self.assertEqual(report["counts"], {"train": 0, "validation": 0})
        self.assertTrue(all(not row["passed"] for row in report["attempts"]))

    def test_existing_output_and_missing_explicit_inputs_fail_before_writes(self):
        self.output.mkdir(parents=True)
        marker = self.output / "keep.txt"
        marker.write_text("preserved")
        with self.assertRaises(FileExistsError):
            self.run_fixture()
        self.assertEqual(marker.read_text(), "preserved")

    def test_missing_exclusion_and_reservation_paths_fail_closed(self):
        for flag in ("--exclude-bundle", "--reservation-csv"):
            with self.subTest(flag=flag), self.assertRaises(FileNotFoundError):
                self.run_fixture([flag, str(self.root / "absent")])
            self.assertFalse(self.output.exists())

    def test_help_runs_without_site_packages(self):
        result = subprocess.run([sys.executable, "-S", str(SCRIPT), "--help"],
                                capture_output=True, text=True, check=False)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("--exclude-bundle", result.stdout)


if __name__ == "__main__":
    unittest.main()
