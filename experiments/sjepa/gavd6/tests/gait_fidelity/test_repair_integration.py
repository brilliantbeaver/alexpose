"""Real tiny model execution across dependencies, repair and locked confirmation."""
from pathlib import Path
import os
import tempfile
import unittest

from gavd6_sjepa.research_directions.gait_fidelity.common import read_json, sha256
from gavd6_sjepa.research_directions.gait_fidelity.repair_fixture import run_fixture


class RepairIntegrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        retained = os.environ.get("GF_REPAIR_INTEGRATION_WORK")
        cls.temporary = None if retained else tempfile.TemporaryDirectory(prefix="repair-integration-")
        cls.work = Path(retained if retained else cls.temporary.name)
        cls.result = run_fixture(cls.work)

    @classmethod
    def tearDownClass(cls):
        if cls.temporary is not None:
            cls.temporary.cleanup()

    def test_complete_real_three_seed_matrix_retains_fixture_boundary(self):
        result = self.result
        self.assertEqual(result["status"], "REPAIR_SOFTWARE_FIXTURE_COMPLETE")
        self.assertEqual(result["new_readouts"], 12)
        self.assertEqual(result["calibrations"], 6)
        self.assertEqual(result["evaluation_fits"], 27)
        self.assertEqual(result["evaluation_methods"], 9)
        self.assertEqual(result["response_fits"], 18)
        self.assertEqual(result["seeds"], [17, 29, 43])
        self.assertFalse(result["scientific_results"])
        self.assertFalse(result["independent_confirmation"])
        self.assertFalse(result["clinical_validation"])
        for path in result["reports"].values():
            self.assertTrue(Path(path).is_file())

    def test_confirmation_generated_independently_and_uses_locked_fits(self):
        parent_cfg = read_json(self.work/"repair/config.json")
        parent = read_json(Path(parent_cfg["repair"]["parent_bundle"])/"manifest.json")
        confirmation = read_json(self.work/"repair/confirmation/bundle/manifest.json")
        first = {row["canonical_person_id"] for row in parent["records"]}
        second = {row["canonical_person_id"] for row in confirmation["records"]}
        self.assertFalse(first & second)
        self.assertEqual(len(second), self.result["generated_confirmation_people"])
        lock = read_json(self.work/"repair/confirmation/lock.json")
        self.assertEqual(len(lock["fit_manifest"]), 27)
        self.assertTrue(lock["fixture"])
        self.assertFalse(lock["independent_confirmation"])
        for item in lock["fit_manifest"]:
            self.assertEqual(sha256(item["checkpoint"]), item["sha256"])

    def test_completed_fixture_is_reverified_without_retraining(self):
        ledger = self.work/"repair/ledger.json"
        before = sha256(ledger)
        self.assertEqual(run_fixture(self.work), self.result)
        self.assertEqual(sha256(ledger), before)


if __name__ == "__main__":
    unittest.main()
