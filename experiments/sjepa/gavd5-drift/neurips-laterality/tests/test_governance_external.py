from __future__ import annotations

import csv
import json
import sys
import tempfile
import unittest
from pathlib import Path


SUITE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SUITE_ROOT))

from laterality.external import (  # noqa: E402
    ExternalEvaluationBlocked,
    ExternalManifestError,
    validate_external_manifest,
)
from laterality.governance import load_governance, submission_readiness  # noqa: E402


REPOSITORY_GOVERNANCE = SUITE_ROOT / "governance" / "status.json"


def write_governance(path: Path, *, resolved: bool) -> None:
    status = "resolved" if resolved else "unresolved"
    reference = "IRB-EXTERNAL-001" if resolved else ""
    date = "2026-09-04" if resolved else ""
    payload = {
        "schema": "neurips_laterality_external_governance/v1",
        "dataset_reference": "CUSTODIAN-DATASET-001",
        "authorization_scope": "subject_disjoint_pose_evaluation",
        "ethics_determination": {
            "status": status,
            "reference": reference,
            "date": date,
        },
        "data_use_review": {
            "status": status,
            "reference": reference,
            "date": date,
        },
        "derived_pose_release_review": {
            "status": status,
            "reference": reference,
            "date": date,
        },
    }
    path.write_text(json.dumps(payload), encoding="utf-8")


def write_manifest(path: Path, rows: list[dict[str, str]], fieldnames=None) -> None:
    columns = fieldnames or [
        "dataset_reference",
        "sequence_id",
        "subject_id",
        "pose_path",
        "split",
        "joint_schema",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


class GovernanceGateTests(unittest.TestCase):
    def test_checked_in_governance_status_is_fail_closed(self):
        readiness = submission_readiness(load_governance(REPOSITORY_GOVERNANCE))
        self.assertFalse(readiness["ready"])
        self.assertEqual(
            set(readiness["unresolved"]),
            {
                "ethics_determination",
                "data_use_review",
                "derived_pose_release_review",
            },
        )

    def test_unresolved_governance_blocks_before_manifest_authorization(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            governance = root / "governance.json"
            write_governance(governance, resolved=False)

            # The manifest need not even exist: unresolved governance is the first
            # and decisive gate, and no placeholder cohort is returned.
            with self.assertRaises(ExternalEvaluationBlocked):
                validate_external_manifest(root / "absent.csv", governance)

    def test_gavd_governance_cannot_authorize_an_external_dataset(self):
        with self.assertRaisesRegex(ExternalEvaluationBlocked, "external-dataset-scoped"):
            validate_external_manifest(Path("absent.csv"), REPOSITORY_GOVERNANCE)


class ExternalManifestTests(unittest.TestCase):
    def setUp(self):
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary_directory.name)
        self.governance = self.root / "governance.json"
        write_governance(self.governance, resolved=True)
        (self.root / "train_a.pose").write_bytes(b"custodian supplied pose")
        (self.root / "train_b.pose").write_bytes(b"custodian supplied pose")
        (self.root / "test.pose").write_bytes(b"custodian supplied pose")

    def tearDown(self):
        self.temporary_directory.cleanup()

    def valid_rows(self) -> list[dict[str, str]]:
        return [
            {
                "sequence_id": "train-sequence-a",
                "dataset_reference": "CUSTODIAN-DATASET-001",
                "subject_id": "subject-train",
                "pose_path": "train_a.pose",
                "split": "train",
                "joint_schema": "BlazePose33",
            },
            {
                "sequence_id": "train-sequence-b",
                "dataset_reference": "CUSTODIAN-DATASET-001",
                "subject_id": "subject-train",
                "pose_path": "train_b.pose",
                "split": "train",
                "joint_schema": "BlazePose33",
            },
            {
                "sequence_id": "test-sequence",
                "dataset_reference": "CUSTODIAN-DATASET-001",
                "subject_id": "subject-test",
                "pose_path": "test.pose",
                "split": "test",
                "joint_schema": "BlazePose33",
            },
        ]

    def test_valid_manifest_authorizes_only_existing_subject_disjoint_poses(self):
        manifest = self.root / "external.csv"
        write_manifest(manifest, self.valid_rows())

        cohort = validate_external_manifest(manifest, self.governance)

        self.assertTrue(cohort.manifest_authorized)
        self.assertEqual(cohort.dataset_reference, "CUSTODIAN-DATASET-001")
        self.assertEqual(cohort.n_sequences, 3)
        self.assertEqual(cohort.n_subjects, 2)
        self.assertEqual(cohort.train_subject_ids, ("subject-train",))
        self.assertEqual(cohort.test_subject_ids, ("subject-test",))
        self.assertEqual(cohort.validation_subject_ids, ())
        self.assertTrue(all(row.pose_path.is_file() for row in cohort.rows))

    def test_manifest_requires_every_contract_column(self):
        manifest = self.root / "missing-subject.csv"
        write_manifest(
            manifest,
            self.valid_rows(),
            fieldnames=["sequence_id", "pose_path", "split", "joint_schema"],
        )
        with self.assertRaisesRegex(ExternalManifestError, "subject_id"):
            validate_external_manifest(manifest, self.governance)

    def test_manifest_dataset_must_match_external_governance(self):
        rows = self.valid_rows()
        for row in rows:
            row["dataset_reference"] = "DIFFERENT-DATASET"
        manifest = self.root / "wrong-dataset.csv"
        write_manifest(manifest, rows)
        with self.assertRaisesRegex(ExternalEvaluationBlocked, "does not match"):
            validate_external_manifest(manifest, self.governance)

    def test_manifest_rejects_subject_leakage(self):
        rows = self.valid_rows()
        rows[-1]["subject_id"] = "subject-train"
        manifest = self.root / "leaking.csv"
        write_manifest(manifest, rows)
        with self.assertRaisesRegex(ExternalManifestError, "exactly one partition"):
            validate_external_manifest(manifest, self.governance)

    def test_manifest_requires_blazepose33(self):
        rows = self.valid_rows()
        rows[-1]["joint_schema"] = "COCO17"
        manifest = self.root / "wrong-schema.csv"
        write_manifest(manifest, rows)
        with self.assertRaisesRegex(ExternalManifestError, "BlazePose33"):
            validate_external_manifest(manifest, self.governance)

    def test_manifest_requires_existing_pose_files(self):
        rows = self.valid_rows()
        rows[-1]["pose_path"] = "not-present.pose"
        manifest = self.root / "missing-pose.csv"
        write_manifest(manifest, rows)
        with self.assertRaisesRegex(ExternalManifestError, "existing file"):
            validate_external_manifest(manifest, self.governance)

    def test_manifest_requires_both_train_and_test_subjects(self):
        rows = self.valid_rows()[:-1]
        manifest = self.root / "train-only.csv"
        write_manifest(manifest, rows)
        with self.assertRaisesRegex(ExternalManifestError, "train and test"):
            validate_external_manifest(manifest, self.governance)


if __name__ == "__main__":
    unittest.main()
