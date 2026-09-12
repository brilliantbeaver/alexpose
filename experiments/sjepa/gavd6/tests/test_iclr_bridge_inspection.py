"""File inspection must never become reconstruction or scientific verification."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from gavd6_sjepa.research_directions.iclr_bridge.inspection import inspect_cached_panel
from gavd6_sjepa.research_directions.iclr_bridge.verification_supplement import expected_sealed_artifacts
from gavd6_sjepa.research_directions.future_innovation.fi_contracts import sha256_file


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
            with patch('gavd6_sjepa.research_directions.iclr_bridge.inspection._load_frozen', return_value=({}, None, None)), \
                 patch('joblib.load', side_effect=AssertionError('Inspection must not load models')), \
                 patch('gavd6_sjepa.research_directions.future_innovation.fi_joint_models.JointRidge.fit', side_effect=AssertionError('No refit')):
                result = inspect_cached_panel(root)
            self.assertEqual(result['status'], 'integrity_checked')
            self.assertFalse(result['model_refits'])
            self.assertFalse(result['numerical_reconstruction_performed'])
            self.assertEqual(before, {str(p): (sha256_file(p), p.stat().st_mtime_ns) for p in root.rglob('*') if p.is_file()})

    def test_changed_digest_and_incomplete_inventory_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder); self.fixture(root)
            with patch('gavd6_sjepa.research_directions.iclr_bridge.inspection._load_frozen', return_value=({}, None, None)):
                (root / 'models/posture/fold-0/baseline.joblib').write_text('altered')
                with self.assertRaisesRegex(ValueError, 'digest mismatch'): inspect_cached_panel(root)
                self.fixture(root)
                seal = json.loads((root / 'reports/completion.json').read_text())
                seal['artifacts'].pop('predictions/oof.parquet')
                (root / 'reports/completion.json').write_text(json.dumps(seal))
                with self.assertRaisesRegex(ValueError, 'inventory'): inspect_cached_panel(root)


if __name__ == '__main__': unittest.main()
