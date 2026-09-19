"""Exercise the actual shell entry points without SSH, Slurm, or a GPU."""
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[2]
SHELL = ROOT / 'slurm/synthetic-training-v2'


class HaicShellTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix='stv2 shell ')
        self.addCleanup(self.temporary.cleanup)
        self.directory = Path(self.temporary.name)
        self.repo = self.directory / 'checkout'
        self.env = {key: value for key, value in os.environ.items()
                    if not key.startswith(('ST_', 'STV2_'))
                    and key not in {'GAVD6_ROOT', 'PYTHONPATH'}}
        self.env['USER'] = 'testuser'

    def profile(self, extra):
        env = {**self.env, **extra}
        result = subprocess.run(
            ['bash', '-c', 'source "$1"; "$2" -c "import json, os; print(json.dumps(dict(os.environ)))"',
             'test', str(SHELL / 'study.env'), sys.executable],
            env=env, text=True, capture_output=True, check=True)
        return json.loads(result.stdout)

    def test_profile_reuses_original_paths_without_changing_them(self):
        original = dict(GAVD6_ROOT='/custom/gavd6', ST_PYTHON='/custom/env/bin/python',
                        ST_RUN_ROOT='/old/pilot-01', ST_CONFIG='/old/pilot-01/config.json',
                        ST_MODEL_ROOT='/custom/models', ST_GAVD_RESERVATION='/old/video-reservations.csv')
        actual = self.profile(original)
        self.assertEqual(actual['STV2_ROOT'], original['GAVD6_ROOT'])
        self.assertEqual(actual['STV2_PYTHON'], original['ST_PYTHON'])
        self.assertEqual(actual['STV2_WORK'], '/custom/gavd6/outputs/synthetic-training-v2/source-smoke-01')
        self.assertEqual(actual['PYTHONPATH'], '/custom/gavd6/src')
        self.assertNotIn('STV2_CONFIG', actual)
        for key, value in original.items():
            self.assertEqual(actual[key], value)

    def test_profile_explicit_run_settings_and_checkout_fallback(self):
        actual = self.profile(dict(GAVD6_ROOT='/old/checkout', ST_PYTHON='/old/python',
                                   STV2_ROOT='/chosen/checkout', STV2_PYTHON='/chosen/python',
                                   STV2_WORK='/chosen/work', PYTHONPATH='/wrong/src'))
        self.assertEqual(actual['STV2_ROOT'], '/chosen/checkout')
        self.assertEqual(actual['STV2_PYTHON'], '/chosen/python')
        self.assertEqual(actual['STV2_WORK'], '/chosen/work')
        self.assertEqual(actual['PYTHONPATH'], '/chosen/checkout/src')
        fallback = self.profile({})
        self.assertEqual(fallback['STV2_ROOT'], str(ROOT))
        self.assertEqual(fallback['STV2_PYTHON'], '/hai/scratch/testuser/envs/synthetic-training-cu124/bin/python')

    def test_submit_preserves_argument_boundaries_and_exit_status(self):
        shell = self.repo / 'slurm/synthetic-training-v2'
        shell.mkdir(parents=True)
        for name in ('study.env', 'submit.sh'):
            shutil.copyfile(SHELL / name, shell / name)
        cli = self.repo / 'scripts/research_directions/synthetic_training_v2/haic.py'
        cli.parent.mkdir(parents=True)
        cli.write_text('import json, os, sys\n'
                       'print(json.dumps(dict(args=sys.argv[1:], root=os.environ["STV2_ROOT"], '
                       'old=os.environ["ST_CONFIG"])))\n'
                       'raise SystemExit(23)\n')
        result = subprocess.run(
            ['bash', str(shell / 'submit.sh'), 'init', '--work', 'a path with spaces', '--gpu-hours', '4'],
            cwd=self.directory,
            env={**self.env, 'STV2_PYTHON': sys.executable, 'ST_CONFIG': '/old/config.json'},
            text=True, capture_output=True)
        self.assertEqual(result.returncode, 23, result.stderr)
        actual = json.loads(result.stdout)
        self.assertEqual(actual['args'], ['init', '--work', 'a path with spaces', '--gpu-hours', '4'])
        self.assertEqual(actual['root'], str(self.repo.resolve()))
        self.assertEqual(actual['old'], '/old/config.json')

    def sync_fixture(self):
        trees = ('src/gavd6_sjepa', 'scripts/research_directions/synthetic_training',
                 'scripts/research_directions/synthetic_training_v2', 'slurm/synthetic-training',
                 'slurm/synthetic-training-v2', 'docs/studies/synthetic-training-v2',
                 'notebooks/synthetic_training_v2')
        for tree in trees:
            folder = self.repo / tree
            folder.mkdir(parents=True, exist_ok=True)
            (folder / 'README.md').write_text('study file\n')
        script = self.repo / 'slurm/synthetic-training-v2/sync-to-haic.sh'
        shutil.copyfile(SHELL / 'sync-to-haic.sh', script)
        for name in ('slurm/synthetic-training/pilot-01.env',
                     'slurm/synthetic-training-v2/session.env',
                     'slurm/synthetic-training/wheels/cache.whl',
                     'src/gavd6_sjepa/__pycache__/cached.pyc',
                     'src/gavd6_sjepa/.settings/private.json',
                     'outputs/private-run/config.json'):
            path = self.repo / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text('must not transfer\n')
        (self.repo / 'slurm/synthetic-training-v2/study.env').write_text('distribution profile\n')
        support = self.repo / 'src/gavd6_sjepa/data_foundations/amass_conversion.py'
        support.parent.mkdir(parents=True)
        support.write_text('required shared code\n')
        frozen = {}
        for name in ('notebook_runs/synthetic-training/.DS_Store',
                     'notebook_runs/synthetic-training/source/trial.json',
                     'docs/studies/temporal-gait/protocol/protocol.md'):
            path = self.repo / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text('verified historical evidence\n')
            frozen[name] = hashlib.sha256(path.read_bytes()).hexdigest()
        manifest = self.repo / 'docs/studies/synthetic-training-v2/preservation-manifest.json'
        manifest.write_text(json.dumps(dict(files=frozen)))
        command_log = self.directory / 'rsync-call.json'
        mock_bin = self.directory / 'bin'
        mock_bin.mkdir()
        mock = mock_bin / 'rsync'
        mock.write_text(f'#!{sys.executable}\n'
                        'import json, os, pathlib, sys\n'
                        'args = sys.argv[1:]\n'
                        'listing = next(a.split("=", 1)[1] for a in args if a.startswith("--files-from="))\n'
                        'files = pathlib.Path(listing).read_bytes().decode().split("\\0")[:-1]\n'
                        'pathlib.Path(os.environ["SYNC_LOG"]).write_text(json.dumps(dict(args=args, files=files)))\n')
        mock.chmod(0o755)
        env = {**self.env, 'PATH': str(mock_bin) + os.pathsep + self.env['PATH'], 'SYNC_LOG': str(command_log)}
        return script, env, command_log, frozen

    def test_sync_preview_preserves_profiles_and_selects_exact_history(self):
        script, env, log, frozen = self.sync_fixture()
        before = {str(path): path.read_bytes() for path in self.repo.rglob('*') if path.is_file()}
        result = subprocess.run(['bash', str(script), 'tester@haic.example:/hai/scratch/tester/gavd6'],
                                env=env, text=True, capture_output=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        call = json.loads(log.read_text())
        self.assertIn('--dry-run', call['args'])
        self.assertIn('--checksum', call['args'])
        self.assertIn('--backup', call['args'])
        self.assertFalse(any(argument.startswith('--delete') for argument in call['args']))
        self.assertTrue(any(argument.startswith('--backup-dir=/hai/scratch/tester/gavd6/outputs/synthetic-training-v2/sync-backup-')
                            for argument in call['args']))
        historical = {name for name in call['files'] if name.startswith('notebook_runs/')}
        self.assertEqual(historical, {name for name in frozen if name.startswith('notebook_runs/')})
        self.assertTrue(set(frozen).issubset(call['files']))
        self.assertIn('src/gavd6_sjepa/data_foundations/amass_conversion.py', call['files'])
        self.assertIn('slurm/synthetic-training-v2/study.env', call['files'])
        self.assertFalse(any(name.startswith('outputs/') or '/__pycache__/' in name
                             or '/wheels/' in name or '/.settings/' in name for name in call['files']))
        self.assertEqual([name for name in call['files'] if name.endswith('.env')],
                         ['slurm/synthetic-training-v2/study.env'])
        self.assertEqual(before, {str(path): path.read_bytes() for path in self.repo.rglob('*') if path.is_file()})

    def test_sync_apply_retains_backup_and_uses_unique_destination(self):
        script, env, log, _ = self.sync_fixture()
        backups = []
        for _ in range(2):
            result = subprocess.run(['bash', str(script), 'tester@haic.example:/hai/scratch/tester/gavd6', '--apply'],
                                    env=env, text=True, capture_output=True)
            self.assertEqual(result.returncode, 0, result.stderr)
            args = json.loads(log.read_text())['args']
            self.assertNotIn('--dry-run', args)
            self.assertIn('--backup', args)
            backups.append(next(argument for argument in args if argument.startswith('--backup-dir=')))
        self.assertNotEqual(*backups)

    def test_sync_missing_or_changed_history_stops_before_remote_command(self):
        script, env, log, frozen = self.sync_fixture()
        missing = self.repo / next(iter(frozen))
        missing.unlink()
        result = subprocess.run(['bash', str(script), 'tester@haic.example:/hai/scratch/tester/gavd6', '--apply'],
                                env=env, text=True, capture_output=True)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn('Missing or unsafe historical file', result.stderr)
        self.assertFalse(log.exists())
        missing.write_text('modified evidence\n')
        result = subprocess.run(['bash', str(script), 'tester@haic.example:/hai/scratch/tester/gavd6', '--apply'],
                                env=env, text=True, capture_output=True)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn('SHA256 mismatch', result.stderr)
        self.assertFalse(log.exists())

    def test_history_only_recovery_does_not_transfer_unpublished_scientific_code(self):
        script, env, log, frozen = self.sync_fixture()
        # A local implementation change must never replace the pulled revision
        # during recovery of historical evidence outside Git.
        changed = self.repo / 'src/gavd6_sjepa/local-unpublished.py'
        changed.write_text('unpublished scientific change\n')
        for apply in (False, True):
            with self.subTest(apply=apply):
                command = ['bash', str(script), 'tester@haic.example:/hai/scratch/tester/gavd6', '--history-only']
                if apply:
                    command.append('--apply')
                result = subprocess.run(command, env=env, text=True, capture_output=True)
                self.assertEqual(result.returncode, 0, result.stderr)
                call = json.loads(log.read_text())
                self.assertEqual(set(call['files']), set(frozen))
                self.assertEqual('--dry-run' in call['args'], not apply)
                self.assertIn('--backup', call['args'])
                self.assertIn('Historical files selected:', result.stdout)
        # Integrity checks still run before any remote call in this mode.
        Path(self.repo / next(iter(frozen))).write_text('corrupted historical source\n')
        log.unlink()
        result = subprocess.run(['bash', str(script), 'tester@haic.example:/hai/scratch/tester/gavd6',
                                 '--history-only', '--apply'], env=env, text=True, capture_output=True)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn('SHA256 mismatch', result.stderr)
        self.assertFalse(log.exists())

    def test_sync_rejects_remote_shell_expressions_and_parent_traversal(self):
        script, env, log, _ = self.sync_fixture()
        for destination in ('tester@haic.example:/hai/$(touch injected)',
                            'tester@haic.example:/hai/../etc', 'tester@haic.example:/'):
            result = subprocess.run(['bash', str(script), destination, '--apply'],
                                    env=env, text=True, capture_output=True)
            self.assertEqual(result.returncode, 2, result.stderr)
            self.assertFalse(log.exists())


if __name__ == '__main__':
    unittest.main()
