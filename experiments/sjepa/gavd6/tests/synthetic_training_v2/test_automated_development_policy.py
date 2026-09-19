"""Machine screens remain explicit development evidence; never fabricated audits."""
import copy
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import pandas as pd

from gavd6_sjepa.research_directions.synthetic_training_v2.contracts import sha256_file
from gavd6_sjepa.research_directions.synthetic_training_v2.data import fixture_bundle
from gavd6_sjepa.research_directions.synthetic_training_v2.decisions import adjudicate_gate_b
from gavd6_sjepa.research_directions.synthetic_training_v2.evaluation import scientific_gate
from gavd6_sjepa.research_directions.synthetic_training_v2.preparation import audited_motion_windows, preparation_provenance
from tests.synthetic_training_v2.test_preparation_provenance import assets, metadata


class AutomatedDevelopmentPolicyTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.base, self.audit, self.reservation = metadata()
        self.audit.pop("reserved")
        self.audit.update(locomotion_status="algorithm_screened_locomotion",
                          audit_reviewer="stv2-kinematic-screen-v1", exposure="unknown",
                          audit_evidence=str(self.root / "evidence.json"))
        self.reservation.update(reserved="unknown", exposure="unknown")
        self.evidence = dict(status="pass", screen_version="stv2-kinematic-screen-v1",
                             reviewed_by="algorithm", review_mode="automated_development",
                             relative_path=self.audit["relative_path"], start_s=self.audit["start_s"],
                             canonical_person_id=self.audit["canonical_person_id"], source_sha256="a" * 64,
                             metrics={"software_fixture": 1}, thresholds={"software_fixture": 1})

    def run_inputs(self, *, mode="automated_development", aliases=()):
        (self.root / "evidence.json").write_text(json.dumps(self.evidence))
        pd.DataFrame([self.audit]).to_csv(self.root / "audit.csv", index=False)
        pd.DataFrame([self.reservation, *aliases]).to_csv(self.root / "reservation.csv", index=False)
        with patch("gavd6_sjepa.research_directions.motion_preservation.motion_data.load_amass_manifest", return_value=self.base):
            return audited_motion_windows("registry", "source", self.root / "audit.csv", self.root / "reservation.csv",
                                          review_mode=mode)

    def test_unknown_is_retained_only_under_explicit_machine_mode(self):
        admitted, rejected = self.run_inputs()
        self.assertEqual(len(rejected), 0)
        self.assertEqual(admitted.iloc[0]["reserved"], "unknown")
        self.assertEqual(admitted.iloc[0]["exposure"], "unknown")
        self.assertEqual(admitted.iloc[0]["review_mode"], "automated_development")
        with self.assertRaisesRegex(ValueError, "explicit true/false"):
            self.run_inputs(mode="human_audited")

    def test_known_reserved_alias_and_original_test_stay_excluded(self):
        alias = dict(self.reservation, person_id="known-alias", reserved=True)
        admitted, rejected = self.run_inputs(aliases=[alias])
        self.assertEqual(len(admitted), 0)
        self.assertEqual(rejected.iloc[0]["exclusion_reason"], "reserved_identity")
        self.base["original_split"] = "test"
        self.reservation["original_split"] = "test"
        admitted, rejected = self.run_inputs()
        self.assertEqual(len(admitted), 0)
        self.assertEqual(rejected.iloc[0]["exclusion_reason"], "existing_test_identity")

    def test_unknown_alias_cannot_be_silently_treated_as_false(self):
        self.reservation["reserved"] = False
        alias = dict(self.reservation, person_id="unknown-alias", reserved="unknown")
        admitted, _ = self.run_inputs(aliases=[alias])
        self.assertEqual(admitted.iloc[0]["reserved"], "unknown")

    def test_machine_evidence_must_match_exact_window_and_agent(self):
        for key, wrong in (("start_s", 3), ("relative_path", "other.npz"), ("status", "fail"),
                           ("canonical_person_id", "other-person"), ("reviewed_by", "human")):
            with self.subTest(field=key):
                original = self.evidence[key]
                self.evidence[key] = wrong
                with self.assertRaisesRegex(ValueError, "exact window"):
                    self.run_inputs()
                self.evidence[key] = original

    def test_automated_status_cannot_masquerade_as_a_human_audit(self):
        self.reservation["reserved"] = False
        with self.assertRaisesRegex(ValueError, "audited_locomotion"):
            self.run_inputs(mode="human_audited")
        self.audit["locomotion_status"] = "audited_locomotion"
        with self.assertRaisesRegex(ValueError, "algorithm_screened_locomotion"):
            self.run_inputs()

    def test_evidence_hash_and_screened_source_are_bound_to_provenance(self):
        config, table = assets(self.root / "producer")
        evidence = self.root / "producer/evidence.json"
        evidence.write_text(json.dumps({"source_sha256": sha256_file(table.iloc[0].raw_path)}))
        table["audit_evidence"] = str(evidence)
        config["review_mode"] = "automated_development"
        initial = preparation_provenance(config, table, self.root / "producer")
        evidence.write_text(json.dumps({"source_sha256": sha256_file(table.iloc[0].raw_path), "extra": "changed"}))
        changed = preparation_provenance(config, table, self.root / "producer")
        self.assertNotEqual(initial["identity"], changed["identity"])
        Path(table.iloc[0].raw_path).write_text("source changed after screening")
        with self.assertRaisesRegex(ValueError, "since algorithm screening"):
            preparation_provenance(config, table, self.root / "producer")

    def test_machine_bundle_has_no_confirmation_or_scientific_gate_success(self):
        bundle = fixture_bundle()
        bundle.evidence_status = "automated-source-screen"
        for row in bundle.records:
            row.update(locomotion_status="algorithm_screened_locomotion", review_mode="automated_development", reserved="unknown")
        bundle.validate()
        invalid = copy.deepcopy(bundle)
        invalid.evidence_status = "source-run"
        with self.assertRaisesRegex(ValueError, "Unaudited"):
            invalid.validate()
        invalid = copy.deepcopy(bundle)
        invalid.records[-1].update(split="confirmation", exposure="unexposed_verified")
        with self.assertRaisesRegex(ValueError, "confirmation"):
            invalid.validate()
        for reserved in (True, None, "", 0, "false"):
            with self.subTest(reserved=reserved):
                invalid = copy.deepcopy(bundle)
                invalid.records[0]["reserved"] = reserved
                with self.assertRaisesRegex(ValueError, "reserved must be"):
                    invalid.validate()
        verdict = scientific_gate(evidence_status=bundle.evidence_status, relative_improvement=.9,
                                  ci95=[.8, 1], preservation_ok=True, clean_retention_ok=True)
        self.assertEqual(verdict["status"], "insufficient_evidence")

    def test_automatic_gate_cannot_be_reopened_by_a_decision_spec(self):
        metrics = pd.DataFrame({"evidence_status": ["automated-source-screen"]})
        cfg = SimpleNamespace(decision_spec="must-not-open-this-calibration.json")
        with patch("gavd6_sjepa.research_directions.synthetic_training_v2.decisions.load_decision_spec",
                   side_effect=AssertionError("Automated screen must not read a scientific approval specification")) as read_spec:
            result = adjudicate_gate_b(metrics, cfg)
        read_spec.assert_not_called()
        self.assertEqual(result["status"], "insufficient_evidence")
        self.assertIn("Machine-screened development", result["reason"])


if __name__ == "__main__":
    unittest.main()
