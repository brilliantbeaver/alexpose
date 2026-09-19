"""Automatic development acceptance never certifies a human overlay review."""
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from tests.synthetic_training_v2 import test_haic_manager as fixtures

MANAGER = fixtures.MANAGER
CONTRACTS = fixtures.CONTRACTS


class AutomatedSourceTests(unittest.TestCase):
    def setUp(self):
        self.fixture = fixtures.ManagedHaicTests()
        self.fixture.setUp()
        self.addCleanup(self.fixture.doCleanups)
        self.state = self.fixture.state
        _, self.accounting = self.fixture.prepare_job()
        self.args = SimpleNamespace(retry=False, dry_run=False, overlays_reviewed=False, automated_screen=True)
        self.metadata = dict(split="development", extractor_family="vitpose", review_mode="automated_development",
                             locomotion_status="algorithm_screened_locomotion", reserved="unknown")
        self.bundle = SimpleNamespace(evidence_status="automated-source-screen",
            provenance={"configuration": {"review_mode": "automated_development"}},
            validate=lambda held: None, records=[self.metadata])

    def execute(self, *, bundle=None, args=None):
        with patch.object(MANAGER, "scheduler_snapshot", return_value={"100": self.accounting}), \
                patch(CONTRACTS + ".verify_preservation"), \
                patch(CONTRACTS + ".TrackBundle.load", return_value=bundle or self.bundle), \
                patch.object(MANAGER, "submit_one", return_value="123") as submit:
            MANAGER.source(self.state, args or self.args)
        return submit

    def test_automatic_source_records_no_human_review_and_submits_full_chain(self):
        submit = self.execute()
        self.assertEqual(submit.call_count, 10)
        self.assertNotIn("overlays_reviewed", self.state)
        acceptance = self.state["automated_screen"]
        self.assertFalse(acceptance["human_reviewed"])
        self.assertEqual(acceptance["review_mode"], "automated_development")
        self.assertEqual(acceptance["evidence_status"], "automated-source-screen")
        self.assertEqual(acceptance["preparation_job"], "100")
        self.assertEqual(MANAGER.read(self.fixture.work / "control.json")["automated_screen"], acceptance)

    def test_automatic_dry_run_does_not_persist_acceptance(self):
        before = self.fixture.snapshot_files()
        args = SimpleNamespace(**{**vars(self.args), "dry_run": True})
        submit = self.execute(args=args)
        submit.assert_not_called()
        self.assertNotIn("automated_screen", self.state)
        self.assertEqual(before, self.fixture.snapshot_files())

    def test_automatic_retry_uses_saved_automatic_acceptance(self):
        self.execute()
        before = dict(self.state["automated_screen"])
        args = SimpleNamespace(**{**vars(self.args), "retry": True, "automated_screen": False})
        submit = self.execute(args=args)
        self.assertEqual(submit.call_count, 10)
        self.assertEqual(self.state["automated_screen"], before)
        self.assertNotIn("overlays_reviewed", self.state)

    def test_mode_status_and_flag_mismatches_are_rejected_without_source_config(self):
        variants = [
            SimpleNamespace(**{**vars(self.bundle), "evidence_status": "source-run"}),
            SimpleNamespace(**{**vars(self.bundle), "provenance": {"configuration": {}}}),
            self.fixture.bundle(),
        ]
        for bundle in variants:
            with self.subTest(bundle=bundle), self.assertRaises(ValueError):
                self.execute(bundle=bundle)
        self.assertIsNone(self.state["source_config"])
        self.assertNotIn("automated_screen", self.state)

    def test_human_flag_cannot_claim_automatic_records_were_reviewed(self):
        for automatic in (False, True):
            args = SimpleNamespace(**{**vars(self.args), "overlays_reviewed": True, "automated_screen": automatic})
            with self.subTest(automatic=automatic), self.assertRaises(ValueError):
                self.execute(args=args)
        self.assertNotIn("overlays_reviewed", self.state)

    def test_records_must_remain_algorithmic_development_with_no_known_reservation(self):
        for field, value in (("review_mode", "human_audited"), ("locomotion_status", "audited_locomotion"),
                             ("reserved", True), ("reserved", "false"), ("split", "confirmation")):
            record = {**self.metadata, field: value}
            bundle = SimpleNamespace(**{**vars(self.bundle), "records": [record]})
            with self.subTest(field=field, value=value), self.assertRaises(ValueError):
                self.execute(bundle=bundle)
        self.assertIsNone(self.state["source_config"])

    def test_automatic_flag_does_not_change_existing_human_source_behavior(self):
        args = SimpleNamespace(**{**vars(self.args), "overlays_reviewed": True, "automated_screen": False})
        submit = self.execute(bundle=self.fixture.bundle(), args=args)
        self.assertEqual(submit.call_count, 10)
        self.assertIn("overlays_reviewed", self.state)
        self.assertNotIn("automated_screen", self.state)


if __name__ == "__main__":
    unittest.main()
