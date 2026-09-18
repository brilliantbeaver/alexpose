"""Missing transfers and altered historical evidence require different repairs."""
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

from gavd6_sjepa.research_directions.synthetic_training_v2.contracts import (
    sha256_file, verify_preservation,
)

ROOT = Path(__file__).resolve().parents[2]


class PreservationDiagnosticsTests(unittest.TestCase):
    def make_repo(self, folder):
        root = Path(folder)
        paths = ['notebook_runs/synthetic-training/results.csv',
                 'notebook_runs/synthetic-training/.DS_Store',
                 'docs/retained.md']
        hashes = {}
        for name in paths:
            path = root / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text('retained ' + name)
            hashes[name] = sha256_file(path)
        manifest = root / 'docs/studies/synthetic-training-v2/preservation-manifest.json'
        manifest.parent.mkdir(parents=True)
        manifest.write_text(json.dumps({'files': hashes}))
        return root, paths

    def test_missing_transfer_is_distinguished_from_changed_contents(self):
        with tempfile.TemporaryDirectory() as folder:
            root, names = self.make_repo(folder)
            (root / names[0]).unlink()
            (root / names[1]).write_text('different Finder metadata')
            with self.assertRaises(RuntimeError) as error:
                verify_preservation(root)
            message = str(error.exception)
            self.assertIn('Missing or not regular files (1)', message)
            self.assertIn(names[0], message)
            self.assertIn('SHA256 mismatches (1)', message)
            self.assertIn(names[1], message)
            self.assertNotIn(names[2], message)
            self.assertIn('notebook_runs is not transferred by Git', message)

    def test_success_retains_existing_result_contract(self):
        with tempfile.TemporaryDirectory() as folder:
            root, _ = self.make_repo(folder)
            self.assertEqual(verify_preservation(root), {'status': 'pass', 'files_checked': 3})

    def test_unreadable_evidence_remains_a_failure(self):
        with tempfile.TemporaryDirectory() as folder:
            root, names = self.make_repo(folder)
            real_hash = sha256_file
            def read_hash(path):
                if Path(path) == root / names[0]:
                    raise PermissionError('test read denied')
                return real_hash(path)
            with patch('gavd6_sjepa.research_directions.synthetic_training_v2.contracts.sha256_file', side_effect=read_hash):
                with self.assertRaisesRegex(RuntimeError, 'Unreadable files \\(1\\)'):
                    verify_preservation(root)

    def test_cli_reports_failure_as_json_and_does_not_write_evidence(self):
        with tempfile.TemporaryDirectory() as folder:
            root, names = self.make_repo(folder)
            (root / names[0]).unlink()
            (root / names[1]).write_text('changed metadata')
            before = {str(p.relative_to(root)): p.read_bytes() for p in root.rglob('*') if p.is_file()}
            result = subprocess.run([sys.executable, str(ROOT / 'scripts/research_directions/synthetic_training_v2/check_history.py'),
                                     '--repo', str(root), '--json'], text=True, capture_output=True)
            self.assertEqual(result.returncode, 1)
            report = json.loads(result.stdout)
            self.assertEqual(report['missing'], [names[0]])
            self.assertEqual([row['path'] for row in report['mismatched']], [names[1]])
            self.assertEqual(report['matched'], 1)
            self.assertEqual(before, {str(p.relative_to(root)): p.read_bytes() for p in root.rglob('*') if p.is_file()})


if __name__ == '__main__':
    unittest.main()
