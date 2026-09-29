"""Offline transfer regressions; execute the real remote program via local Python."""
import contextlib
import importlib.util
import io
import json
from pathlib import Path
import shlex
import subprocess
import sys
import tarfile
import tempfile
import unittest
from unittest.mock import patch


SPEC = importlib.util.spec_from_file_location(
    'download_paper_evidence', Path(__file__).with_name('download_paper_evidence.py'))
download = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(download)
RUN = subprocess.run


class TransferTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.base = Path(self.temporary.name)
        self.assets = self.base/'assets'
        self.output = self.base/'downloaded'
        self.work = self.assets/'outputs/gait-fidelity/walking-core-01'
        for name, (source, key) in download.RUNS.items():
            allowlist = download.ROOT/f'docs/studies/gait-fidelity/manuscript/paper-evidence-{key}-files.txt'
            for relative in allowlist.read_text().splitlines():
                path = self.assets/'outputs/gait-fidelity'/source/relative
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text('{}\n')

    def run_cli(self, *args):
        def local_remote(command, *, input, stdout, check):
            self.assertFalse(check)
            remote_args = shlex.split(command[-1])
            return RUN([sys.executable, '-', remote_args[-1]],
                       input=input, stdout=stdout, stderr=subprocess.PIPE)

        argv = ['download', '--asset-root', str(self.assets), '--output', str(self.output), *args]
        captured = io.StringIO()
        with patch.object(sys, 'argv', argv), patch.object(download.subprocess, 'run', local_remote):
            with contextlib.redirect_stdout(captured):
                code = download.main()
        return code, captured.getvalue()

    def preview(self):
        records = list(self.output.glob('transfer-preview-*.json'))
        self.assertEqual(len(records), 1)
        return json.loads(records[0].read_text())

    def resize(self, path, size):
        with path.open('wb') as stream:
            stream.truncate(size)

    def test_oversized_apply_reports_file_and_preserves_inventory_only(self):
        self.resize(self.work/'report.md', 1024**2+1)
        code, text = self.run_cli('--apply', '--max-file-mib', '1')
        self.assertEqual(code, 2)
        self.assertIn('walking-core/report.md: 1,048,577 bytes', text)
        self.assertIn('per-file limit of 1,048,576 bytes', text)
        self.assertNotIn("Command '['ssh'", text)
        self.assertTrue(self.preview()['transfer_blocked'])
        self.assertFalse((self.output/'walking-core').exists())

    def test_total_limit_reports_actual_total(self):
        self.resize(self.work/'report.md', 600*1024)
        self.resize(self.work/'evaluation/per-person.csv', 600*1024)
        code, text = self.run_cli('--apply', '--max-file-mib', '1', '--max-total-mib', '1')
        self.assertEqual(code, 2)
        self.assertIn('exceeds the total limit of 1,048,576 bytes', text)
        self.assertGreater(self.preview()['selected_bytes'], 1024**2)
        self.assertFalse((self.output/'walking-core').exists())

    def test_rejected_symlink_reason_is_visible(self):
        target = self.work/'report.md'
        target.unlink()
        target.symlink_to(self.work/'config.json')
        code, text = self.run_cli('--apply')
        self.assertEqual(code, 2)
        self.assertIn('walking-core/report.md: symbolic links are not included', text)
        self.assertTrue(self.preview()['transfer_blocked'])

    def test_empty_source_has_distinct_diagnostic(self):
        self.assets = self.base/'absent'
        code, text = self.run_cli()
        self.assertEqual(code, 2)
        self.assertIn('No eligible files were found', text)
        self.assertNotIn('exceeds the per-file limit', text)
        self.assertTrue(self.preview()['transfer_blocked'])

    def test_successful_preview_saves_inventory_without_evidence(self):
        code, text = self.run_cli()
        self.assertEqual(code, 0)
        self.assertFalse(self.preview()['transfer_blocked'])
        self.assertIn('excludes oversized, rejected and missing files', text)
        self.assertFalse((self.output/'walking-core').exists())

    def test_measured_large_tables_fit_explicit_50_80_limits(self):
        # Rounded HAIC sizes: 17,940.2 KiB core and 50,499.5 KiB response.
        self.resize(self.work/'evaluation/by-condition-person.csv', round(17940.2*1024))
        response = self.assets/'outputs/gait-fidelity/jepa-response-02/evaluation'
        self.resize(response/'response-by-condition-person.csv', round(50499.5*1024))
        # Represent the remaining 9.34 MiB without needing real study data.
        self.resize(self.work/'report.md', round(9.34*1024**2))
        code, _ = self.run_cli('--apply', '--max-file-mib', '50', '--max-total-mib', '80')
        self.assertEqual(code, 0)
        record = next(self.output.glob('transfer-inventory-*.json'))
        self.assertAlmostEqual(json.loads(record.read_text())['selected_bytes']/1024**2, 76.18, places=2)
        for run, name, source in [
            ('walking-core', 'by-condition-person.csv', self.work/'evaluation'),
            ('jepa-response', 'response-by-condition-person.csv', response),
        ]:
            self.assertEqual(download.digest(self.output/run/'evaluation'/name),
                             download.digest(source/name))

    def test_successful_download_is_verified_and_repeatable(self):
        for _ in range(2):
            code, text = self.run_cli('--apply')
            self.assertEqual(code, 0)
            self.assertIn('SHA-256 verified', text)
        self.assertEqual((self.output/'walking-core/report.md').read_text(), '{}\n')
        self.assertEqual(len(list(self.output.glob('transfer-inventory-*.json'))), 2)

    def test_missing_required_is_explicit_partial_download(self):
        (self.work/'report.md').unlink()
        code, text = self.run_cli('--apply')
        self.assertEqual(code, 3)
        self.assertIn('INCOMPLETE EVIDENCE', text)
        self.assertIn('walking-core/report.md', text)

    def test_existing_different_evidence_is_preserved(self):
        target = self.output/'walking-core/report.md'
        target.parent.mkdir(parents=True)
        target.write_text('preserve this existing result')
        with self.assertRaisesRegex(FileExistsError, 'Existing file differs'):
            self.run_cli('--apply')
        self.assertEqual(target.read_text(), 'preserve this existing result')
        self.assertFalse((self.output/'walking-core/config.json').exists())

    def test_transport_error_does_not_echo_full_command(self):
        args = ['download', '--asset-root', str(self.assets), '--output', str(self.output), '--apply']
        failure = subprocess.CompletedProcess(['ssh', 'private payload'], 255)
        with patch.object(sys, 'argv', args), patch.object(download.subprocess, 'run', return_value=failure):
            with self.assertRaisesRegex(RuntimeError, 'status 255') as error:
                download.main()
        self.assertNotIn('private payload', str(error.exception))
        self.assertFalse(self.output.exists())

    def test_noninventory_remote_exit_two_is_not_a_policy_refusal(self):
        args = ['download', '--asset-root', str(self.assets), '--output', str(self.output), '--apply']
        failure = subprocess.CompletedProcess(['ssh'], 2)
        with patch.object(sys, 'argv', args), patch.object(download.subprocess, 'run', return_value=failure):
            with self.assertRaisesRegex(RuntimeError, 'no valid inventory'):
                download.main()
        self.assertFalse(self.output.exists())

    def test_unsafe_archive_and_checksum_mismatch_are_rejected(self):
        for name, expected in [('walking-core/../../escape', 'Unsafe'),
                               ('walking-core/report.md', 'checksum mismatch')]:
            with self.subTest(name=name):
                archive = self.base/'packet.tar.gz'
                inventory = dict(files=[dict(destination=name, status='selected', bytes=3, sha256='wrong')])
                with tarfile.open(archive, 'w:gz') as stream:
                    for member_name, data in [(name, b'bad'), ('_inventory.json', json.dumps(inventory).encode())]:
                        member = tarfile.TarInfo(member_name); member.size = len(data)
                        stream.addfile(member, io.BytesIO(data))
                with tempfile.TemporaryDirectory(dir=self.base) as staging:
                    with self.assertRaisesRegex(ValueError, expected):
                        download.unpack_verified(archive, Path(staging), 1024**2, 2*1024**2)


if __name__ == '__main__':
    unittest.main()
