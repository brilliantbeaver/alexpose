"""Bounded automated development input policy, using explicit software geometry."""
import contextlib
import csv
import importlib.util
import io
import json
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

from gavd6_sjepa.research_directions.motion_preservation.body_geometry import demo_motion


REPO = Path(__file__).resolve().parents[2]
SCRIPT = REPO / "scripts/research_directions/synthetic_training_v2/automated_inputs.py"
sys.path.insert(0, str(SCRIPT.parent))
SPEC = importlib.util.spec_from_file_location("stv2_automated_inputs", SCRIPT)
AUTO = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(AUTO)
BODY = "gavd6_sjepa.research_directions.motion_preservation.body_geometry.SMPLHBody"


class FixtureBody:
    def __init__(self, *args, **kwargs):
        pass

    def forward(self, motion):
        # Procedural software fixture: intentionally not actual licensed SMPL-H.
        clean, _ = demo_motion(frames=64, fps=25)
        return clean


class AutomatedInputTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.amass = self.root / "amass"
        self.manifests = self.root / "manifests"
        self.work = self.root / "work"
        self.output = self.work / "inputs/automated-01"
        self.manifests.mkdir()
        (self.work / "config").mkdir(parents=True)
        self.records = []
        registry, splits = [], []
        for split, person in [("train", "t1"), ("train", "t2"), ("train", "t3"),
                              ("validation", "v1"), ("validation", "v2"), ("test", "z1")]:
            relative = f"Dataset/{person}/0000_treadmill_norm_poses.npz"
            source = self.amass / relative
            source.parent.mkdir(parents=True)
            np.savez(source, poses=np.zeros((301, 156)), trans=np.zeros((301, 3)),
                     dmpls=np.zeros((301, 8)), betas=np.zeros(16), gender="male", mocap_framerate=25)
            self.records.append(dict(relative_path=relative, subject_id_candidate=person, status="ok",
                                     num_frames=301, mocap_framerate=25, gender="male"))
            registry.append(dict(subject_id_candidate=person, identity=person,
                                 identity_audit_status="approved", excluded=False))
            splits.append(dict(identity=person, split=split))
        for filename, values in (("amass_raw_inventory_eligible.csv", self.records),
                                 ("amass_subject_registry.csv", registry),
                                 ("amass_subject_splits.csv", splits)):
            pd.DataFrame(values).to_csv(self.manifests / filename, index=False)
        self.config = self.work / "config/preparation.json"
        self.config.write_text(json.dumps(dict(manifest_dir=str(self.manifests), amass_root=str(self.amass),
                                             body_model_root=str(self.root / "bodies"),
                                             dmpl_root=str(self.root / "dmpls"))))
        self.argv = ["--config", str(self.config), "--output-dir", str(self.output)]

    def run_fixture(self, argv=None, body=FixtureBody):
        with patch(BODY, body), contextlib.redirect_stdout(io.StringIO()):
            return AUTO.run(AUTO.parser().parse_args(self.argv if argv is None else argv))

    def write_reservations(self, rows):
        folder = self.work / "inputs/review-drafts"
        folder.mkdir(parents=True, exist_ok=True)
        path = folder / "person-reservations.csv"
        with path.open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=["person_id", "canonical_person_id",
                                                        "original_split", "reserved", "exposure"])
            writer.writeheader()
            writer.writerows(rows)
        return path

    def test_heuristics_accept_fixture_alternation_but_reject_static_and_corrupt_geometry(self):
        clean, _ = demo_motion(64, 25)
        self.assertTrue(AUTO.screen_joints(clean.joints)["passed"])
        static = np.repeat(clean.joints[:1], 64, axis=0)
        rejected = AUTO.screen_joints(static)
        self.assertFalse(rejected["passed"])
        self.assertIn("ankle_excursion", rejected["failures"])
        corrupt = clean.joints.copy()
        corrupt[0, 0, 0] = np.nan
        self.assertEqual(AUTO.screen_joints(corrupt)["failures"], ["nonfinite_or_wrong_shape"])

    def test_end_to_end_writes_eight_machine_records_without_altering_inputs(self):
        before = self.config.read_bytes()
        report = self.run_fixture()
        self.assertEqual(report["status"], "inputs_created")
        self.assertEqual(report["counts"], {"train": 2, "validation": 2})
        audits = pd.read_csv(self.output / "locomotion-audit.csv", keep_default_na=False)
        reservations = pd.read_csv(self.output / "person-reservations.csv", keep_default_na=False)
        self.assertEqual(len(audits), 8)
        self.assertTrue(audits.locomotion_status.eq("algorithm_screened_locomotion").all())
        self.assertTrue(audits.audit_reviewer.eq(AUTO.VERSION).all())
        self.assertTrue(reservations.reserved.eq("unknown").all())
        self.assertTrue(reservations.exposure.eq("unknown").all())
        self.assertEqual(set(reservations.person_id), {"t1", "t2", "v1", "v2"})
        for row in audits.to_dict("records"):
            evidence = json.loads(Path(row["audit_evidence"]).read_text())
            self.assertEqual(evidence["status"], "pass")
            self.assertEqual(evidence["reviewed_by"], "algorithm")
            self.assertEqual(evidence["screen_version"], AUTO.VERSION)
            self.assertEqual(evidence["relative_path"], row["relative_path"])
            self.assertEqual(evidence["start_s"], row["start_s"])
        self.assertEqual(self.config.read_bytes(), before)
        self.assertEqual(len(list((self.output / "evidence").glob("*.json"))), 8)
        with np.load(self.output / "playback/joints.npz", allow_pickle=False) as data:
            self.assertEqual(data["joints"].shape, (8, 64, 22, 3))

    def test_existing_reservation_true_is_excluded_and_other_facts_are_preserved(self):
        path = self.write_reservations([
            dict(person_id="t1", canonical_person_id="t1", original_split="train", reserved="true", exposure="prior"),
            dict(person_id="v1", canonical_person_id="v1", original_split="validation", reserved="false", exposure="known"),
        ])
        before = path.read_bytes()
        self.run_fixture()
        reservations = pd.read_csv(self.output / "person-reservations.csv", keep_default_na=False)
        self.assertEqual(set(reservations.person_id), {"t2", "t3", "v1", "v2"})
        record = reservations.set_index("person_id").loc["v1"]
        self.assertEqual(record["reserved"], "false")
        self.assertEqual(record["exposure"], "known")
        self.assertEqual(path.read_bytes(), before)

    def test_known_test_alias_protects_training_identity(self):
        self.write_reservations([
            dict(person_id="t1", canonical_person_id="same-person", original_split="train", reserved="", exposure=""),
            dict(person_id="z1", canonical_person_id="same-person", original_split="test", reserved="", exposure=""),
        ])
        self.run_fixture()
        people = pd.read_csv(self.output / "person-reservations.csv")
        self.assertNotIn("t1", set(people.person_id))
        self.assertNotIn("z1", set(people.person_id))

    def test_insufficient_motion_does_not_manufacture_audit_rows(self):
        class StaticBody(FixtureBody):
            def forward(self, motion):
                result = super().forward(motion)
                result.joints[:] = result.joints[0]
                return result
        with self.assertRaisesRegex(ValueError, "Could not screen two people"):
            self.run_fixture(self.argv + ["--max-candidates", "4"], StaticBody)
        report = json.loads((self.output / "screen.json").read_text())
        self.assertEqual(report["status"], "insufficient_candidates")
        self.assertEqual(len(report["attempts"]), 4)
        self.assertFalse((self.output / "locomotion-audit.csv").exists())
        self.assertFalse((self.output / "person-reservations.csv").exists())

    def test_existing_output_is_not_replaced(self):
        self.output.mkdir(parents=True)
        marker = self.output / "keep.txt"
        marker.write_text("existing")
        with self.assertRaises(FileExistsError):
            self.run_fixture()
        self.assertEqual(marker.read_text(), "existing")

    def test_missing_explicit_reservations_fail_instead_of_becoming_unknown(self):
        with self.assertRaisesRegex(FileNotFoundError, "Explicit reservation"):
            self.run_fixture(self.argv + ["--reservation-csv", str(self.root / "missing.csv")])
        self.assertFalse(self.output.exists())

    def test_help_needs_no_site_packages(self):
        result = subprocess.run([sys.executable, "-S", str(SCRIPT), "--help"], text=True,
                                capture_output=True, check=False)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("--max-candidates", result.stdout)


if __name__ == "__main__":
    unittest.main()
