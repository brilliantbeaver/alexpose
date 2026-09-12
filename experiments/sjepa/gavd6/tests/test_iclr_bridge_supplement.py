"""Tamper the copied pilot coherently, including all affected integrity hashes."""
from pathlib import Path
import shutil
import tempfile
import unittest

from gavd6_sjepa.research_directions.future_innovation.fi_contracts import read_json, write_json, sha256_file
from gavd6_sjepa.research_directions.iclr_bridge.cached_panel import verify_cached_panel
from gavd6_sjepa.research_directions.iclr_bridge.verification_supplement import verify_panel_supplement

PILOT = Path(__file__).resolve().parents[1] / "outputs/iclr-bridge-cached-20260911"


@unittest.skipUnless((PILOT / "reports/completion.json").exists(), "Saved cached accessibility pilot is not installed")
class SupplementalVerificationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp = tempfile.TemporaryDirectory()
        cls.root = Path(cls.temp.name) / "copied-pilot"
        shutil.copytree(PILOT, cls.root)

    @classmethod
    def tearDownClass(cls):
        cls.temp.cleanup()

    def restore(self):
        for relative in (
            "reports/panel-report.json", "reports/completion.json",
            "models/posture/fold-0/selection.json", "models/posture/fold-0/diagnostics.json",
            "models/posture/fold-0/complete.json",
        ):
            shutil.copy2(PILOT / relative, self.root / relative)

    def setUp(self):
        self.restore()

    def tearDown(self):
        self.restore()

    def tamper(self, relative, change):
        path = self.root / relative
        value = read_json(path); change(value); write_json(path, value)
        seal_path = self.root / "reports/completion.json"; seal = read_json(seal_path)
        if relative != "reports/completion.json":
            seal["artifacts"][relative] = sha256_file(path)
            if relative.startswith("models/"):
                receipt_path = path.parent / "complete.json"; receipt = read_json(receipt_path)
                receipt["artifacts"][path.name] = sha256_file(path); write_json(receipt_path, receipt)
                seal["artifacts"][str(receipt_path.relative_to(self.root))] = sha256_file(receipt_path)
            write_json(seal_path, seal)

    def test_saved_pilot_passes(self):
        result = verify_panel_supplement(self.root)
        self.assertEqual(result["checked_candidate_display_records"], 1480)
        self.assertEqual(result["checked_fitted_state_diagnostics"], 40)

    def test_changed_report_metadata_rejected(self):
        self.tamper("reports/panel-report.json", lambda r: r.update(clips=5000))
        # This reproduces the old verifier's omission before the supplement.
        self.assertEqual(verify_cached_panel(self.root)["status"], "passed")
        with self.assertRaisesRegex(ValueError, "metadata"):
            verify_panel_supplement(self.root)

    def test_omitted_seal_artifact_rejected(self):
        self.tamper("reports/completion.json", lambda r: r["artifacts"].pop("reports/numerical-verification.json"))
        with self.assertRaisesRegex(ValueError, "omits or adds"):
            verify_panel_supplement(self.root)

    def test_changed_seal_version_and_completeness_rejected(self):
        for field, value in (("version", "invented"), ("measurement_complete", False)):
            with self.subTest(field=field):
                self.restore()
                self.tamper("reports/completion.json", lambda r: r.update({field: value}))
                with self.assertRaises(ValueError):
                    verify_panel_supplement(self.root)

    def test_misleading_candidate_display_fields_rejected(self):
        for field, value in (("improvement_over_baseline", 1000.), ("selected", True),
                             ("selection_reason", "selected by outer result")):
            with self.subTest(field=field):
                self.restore()
                def change(r):
                    candidate = next(c for c in r["real-skeleton"]["candidates"] if not c["selected"])
                    candidate[field] = value
                self.tamper("models/posture/fold-0/selection.json", change)
                with self.assertRaises((ValueError, AssertionError)):
                    verify_panel_supplement(self.root)

    def test_changed_training_diagnostics_rejected(self):
        self.tamper("models/posture/fold-0/diagnostics.json", lambda rows: rows[0].update(training_mse=123.))
        with self.assertRaises((ValueError, AssertionError)):
            verify_panel_supplement(self.root)


if __name__ == '__main__':
    unittest.main()
