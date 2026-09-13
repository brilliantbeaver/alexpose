"""future feature prediction / accessibility / test evidence."""


import json
from pathlib import Path
import shutil
import tempfile
import unittest
from unittest.mock import patch

from gavd6_sjepa.research_directions.future_prediction.contracts import (
    read_json,
    sha256_file,
    write_json,
)
from gavd6_sjepa.research_directions.target_accessibility.cached_panel import verify_cached_panel
from gavd6_sjepa.research_directions.target_accessibility.inspection import inspect_cached_panel
from gavd6_sjepa.research_directions.target_accessibility.verification import (
    expected_sealed_artifacts,
    verify_panel_supplement,
)
from tests.support import REPO_ROOT

# File inspection must never become reconstruction or scientific verification.


class InspectionTests(unittest.TestCase):
    def fixture(self, root):
        names = expected_sealed_artifacts({})
        for name in names:
            path = root / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text('fixture bytes')
        (root / 'reports/panel-report.json').write_text(json.dumps({'status': 'synthetic_saved_status'}))
        (root / 'reports/completion.json').write_text(json.dumps({'artifacts': {
            name: sha256_file(root / name) for name in names}}))

    def test_no_fit_or_model_loading_and_no_claim_of_reconstruction(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder); self.fixture(root)
            before = {str(p): (sha256_file(p), p.stat().st_mtime_ns) for p in root.rglob('*') if p.is_file()}
            with patch('gavd6_sjepa.research_directions.target_accessibility.inspection._load_frozen', return_value=({}, None, None)), \
                 patch('joblib.load', side_effect=AssertionError('Inspection must not load models')), \
                 patch('gavd6_sjepa.research_directions.future_prediction.joint_models.JointRidge.fit', side_effect=AssertionError('No refit')):
                result = inspect_cached_panel(root)
            self.assertEqual(result['status'], 'integrity_checked')
            self.assertFalse(result['model_refits'])
            self.assertFalse(result['numerical_reconstruction_performed'])
            self.assertEqual(before, {str(p): (sha256_file(p), p.stat().st_mtime_ns) for p in root.rglob('*') if p.is_file()})

    def test_changed_digest_and_incomplete_inventory_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder); self.fixture(root)
            with patch('gavd6_sjepa.research_directions.target_accessibility.inspection._load_frozen', return_value=({}, None, None)):
                (root / 'models/posture/fold-0/baseline.joblib').write_text('altered')
                with self.assertRaisesRegex(ValueError, 'digest mismatch'): inspect_cached_panel(root)
                self.fixture(root)
                seal = json.loads((root / 'reports/completion.json').read_text())
                seal['artifacts'].pop('predictions/oof.parquet')
                (root / 'reports/completion.json').write_text(json.dumps(seal))
                with self.assertRaisesRegex(ValueError, 'inventory'): inspect_cached_panel(root)


PILOT = REPO_ROOT / "outputs/iclr-bridge-cached-20260911"

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


if __name__ == "__main__":
    unittest.main()
