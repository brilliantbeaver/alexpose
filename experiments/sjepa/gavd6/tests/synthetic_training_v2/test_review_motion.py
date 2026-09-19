"""Offline playback tests; licensed body-model execution is explicitly mocked."""
import contextlib
import csv
import hashlib
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


REPO = Path(__file__).resolve().parents[2]
SCRIPT = REPO / "scripts/research_directions/synthetic_training_v2/review_motion.py"
SPEC = importlib.util.spec_from_file_location("stv2_review_motion", SCRIPT)
REVIEW = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(REVIEW)
BODY = "gavd6_sjepa.research_directions.motion_preservation.body_geometry.SMPLHBody"


class FixtureBody:
    """Software fixture geometry, never a stand-in for scientific review evidence."""
    def __init__(self, root, dmpl_root, device, batch_size):
        self.dmpl_root = Path(dmpl_root)
        self.device, self.batch_size = device, batch_size

    def forward(self, motion):
        joints = np.zeros((64, 22, 3), np.float32)
        joints[:, :, 1] = np.linspace(0.9, 1.7, 22)
        joints[:, 1, 0], joints[:, 2, 0] = -0.12, 0.12
        joints[:, :, 2] = motion.trans[:, 0, None]
        joints[:, 7, 1] = 0.1 + 0.05 * np.sin(motion.timestamps * 7)
        return SimpleNamespace(joints=joints, coordinate_system="y_up")


class MotionReviewTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.amass = self.root / "amass"
        self.manifests = self.root / "manifests"
        self.amass.mkdir()
        self.manifests.mkdir()
        self.relative = "Dataset/person/walk_poses.npz"
        source = self.amass / self.relative
        source.parent.mkdir(parents=True)
        times = np.arange(301) / 25
        np.savez(source, poses=np.zeros((301, 156)), trans=np.stack(
            [times * 0.2, np.zeros(301), np.zeros(301)], axis=1),
            betas=np.zeros(16), dmpls=np.zeros((301, 8)), gender="male", mocap_framerate=25)
        self.source = source
        pd.DataFrame([dict(relative_path=self.relative, subject_id_candidate="person",
                           status="ok", num_frames=301, mocap_framerate=25, gender="male")]).to_csv(
            self.manifests / "amass_raw_inventory_eligible.csv", index=False)
        pd.DataFrame([dict(subject_id_candidate="person", identity="person",
                           identity_audit_status="approved", excluded=False)]).to_csv(
            self.manifests / "amass_subject_registry.csv", index=False)
        pd.DataFrame([dict(identity="person", split="train")]).to_csv(
            self.manifests / "amass_subject_splits.csv", index=False)
        self.config = self.root / "preparation.json"
        self.config.write_text(json.dumps(dict(
            manifest_dir=str(self.manifests), amass_root=str(self.amass),
            body_model_root=str(self.root / "bodies"), dmpl_root=str(self.root / "dmpls"))))
        self.output = self.root / "review"
        self.argv = ["--config", str(self.config), "--relative-path", self.relative,
                     "--start-s", "5", "--start-s", "8", "--output-dir", str(self.output)]

    def arguments(self, argv=None):
        return REVIEW.parser().parse_args(self.argv if argv is None else argv)

    def run_fixture(self, argv=None):
        with patch(BODY, FixtureBody), contextlib.redirect_stdout(io.StringIO()):
            return REVIEW.create_review(self.arguments(argv))

    def csv(self, rows):
        path = self.root / "audit.csv"
        with path.open("w", newline="") as stream:
            writer = csv.DictWriter(stream, ["relative_path", "start_s", "audit_reviewer"])
            writer.writeheader()
            writer.writerows(rows)
        return path

    def test_real_loader_and_html_renderer_preserve_samples_and_inputs(self):
        before = {path: path.read_bytes() for path in (self.source, self.config)}
        report = self.run_fixture()
        self.assertEqual(report["status"], "playback_created_human_review_required")
        self.assertEqual([w["start_s"] for w in report["windows"]], [5, 8])
        self.assertEqual([w["end_s"] for w in report["windows"]], [7.52, 10.52])
        for path, content in before.items():
            self.assertEqual(path.read_bytes(), content)
        with np.load(self.output / "joints.npz", allow_pickle=False) as trace:
            self.assertEqual(trace["joints"].shape, (2, 64, 22, 3))
            np.testing.assert_allclose(trace["timestamps"][1], 8 + np.arange(64) / 25)
            self.assertGreater(float(np.ptp(trace["joints"][0, :, 0, 2])), 0.4)
        html = (self.output / "review.html").read_text()
        self.assertIn("Front view (initial orientation)", html)
        self.assertIn("Side view (initial orientation)", html)
        self.assertIn("#window=2", html)
        self.assertNotIn("audited_locomotion", html)
        self.assertEqual(report["windows"][0]["source_sha256"],
                         hashlib.sha256(self.source.read_bytes()).hexdigest())
        self.assertEqual(json.loads((self.output / "metadata.json").read_text()), report)

    def test_blank_human_fields_in_candidate_csv_are_not_modified(self):
        path = self.csv([dict(relative_path=self.relative, start_s=5, audit_reviewer="")])
        before = path.read_bytes()
        argv = ["--config", str(self.config), "--audit-csv", str(path), "--output-dir", str(self.output)]
        self.run_fixture(argv)
        self.assertEqual(path.read_bytes(), before)

    def test_invalid_candidate_requests_fail_before_output(self):
        for rows in ([], [dict(relative_path=self.relative, start_s="nan")],
                     [dict(relative_path="../escape.npz", start_s=0)],
                     [dict(relative_path=self.relative, start_s=0)] * 2,
                     [dict(relative_path=self.relative, start_s=index) for index in range(33)]):
            with self.subTest(rows=rows):
                path = self.csv(rows)
                argv = ["--config", str(self.config), "--audit-csv", str(path),
                        "--output-dir", str(self.output)]
                with self.assertRaises(ValueError):
                    self.run_fixture(argv)
                self.assertFalse(self.output.exists())

    def test_manifest_duration_and_real_source_duration_are_both_checked(self):
        args = self.arguments()
        args.start_s = [10]
        with patch(BODY, FixtureBody), self.assertRaisesRegex(ValueError, "manifest duration"):
            REVIEW.create_review(args)
        inventory = self.manifests / "amass_raw_inventory_eligible.csv"
        table = pd.read_csv(inventory)
        table["num_frames"] = 10000
        table.to_csv(inventory, index=False)
        with patch(BODY, FixtureBody), self.assertRaisesRegex(ValueError, "extends past"):
            REVIEW.create_review(args)
        self.assertFalse(self.output.exists())

    def test_existing_output_and_source_directory_are_rejected(self):
        self.output.mkdir()
        marker = self.output / "keep.txt"
        marker.write_text("existing review")
        with self.assertRaises(FileExistsError):
            self.run_fixture()
        self.assertEqual(marker.read_text(), "existing review")
        args = self.arguments()
        args.output_dir = self.amass / "do-not-write"
        with self.assertRaisesRegex(ValueError, "outside the source"):
            REVIEW.create_review(args)
        self.assertFalse(args.output_dir.exists())

    def test_original_test_people_are_not_previewed(self):
        pd.DataFrame([dict(identity="person", split="test")]).to_csv(
            self.manifests / "amass_subject_splits.csv", index=False)
        with self.assertRaisesRegex(ValueError, "original test"):
            self.run_fixture()
        self.assertFalse(self.output.exists())

    def test_nonfinite_body_geometry_is_rejected(self):
        class BrokenBody(FixtureBody):
            def forward(self, motion):
                result = super().forward(motion)
                result.joints[0, 0, 0] = np.nan
                return result
        with patch(BODY, BrokenBody), self.assertRaisesRegex(ValueError, "finite"):
            REVIEW.create_review(self.arguments())
        self.assertFalse(self.output.exists())

    def test_help_works_without_site_packages(self):
        result = subprocess.run([sys.executable, "-S", str(SCRIPT), "--help"],
                                capture_output=True, text=True, check=False)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("--audit-csv", result.stdout)
        self.assertIn("--device {cpu,cuda}", result.stdout)

    def test_cli_runs_real_renderer_and_reports_actionable_errors(self):
        with patch(BODY, FixtureBody), contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(REVIEW.main(self.argv), 0)
        with contextlib.redirect_stderr(io.StringIO()) as err:
            self.assertEqual(REVIEW.main(self.argv), 1)
        self.assertIn("choose a new directory", err.getvalue())

    def test_html_escapes_payload_script_terminators(self):
        path = self.root / "escaped.html"
        REVIEW._write_html(path, {"windows": [{"relative_path": "</script><script>bad()"}]})
        self.assertNotIn("</script><script>bad()", path.read_text())
        self.assertIn("\\u003c/script>", path.read_text())


if __name__ == "__main__":
    unittest.main()
