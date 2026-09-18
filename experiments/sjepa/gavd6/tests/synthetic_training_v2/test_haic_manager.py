"""Exercise the managed workflow with a simulated scheduler, never real GPU jobs."""
import contextlib
import importlib.util
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

REPO = Path(__file__).resolve().parents[2]
SCRIPTS = REPO / 'scripts/research_directions/synthetic_training_v2'
sys.path.insert(0, str(SCRIPTS))
SPEC = importlib.util.spec_from_file_location('stv2_haic_manager', SCRIPTS / 'haic.py')
MANAGER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MANAGER)
CONTRACTS = 'gavd6_sjepa.research_directions.synthetic_training_v2.contracts'


class ManagedHaicTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.work = Path(temporary.name) / 'run'
        self.work.mkdir()
        (self.work / 'config').mkdir()
        self.state = dict(name='test-run', work=str(self.work), python=sys.executable,
            authorized_gpu_hours=4., prior_entries=[dict(stage='setup', gpu_seconds=120.)],
            updates=200, readout_updates=200, seeds=[17], held_extractor='vitpose', batch_size=64,
            account='my-account', partition='my-partition', jobs=[], source_config=None, pending_submission=None)
        MANAGER.write_new(self.work / 'config/preparation.json', dict(scope_config='not-yet-created', estimators=[]))
        MANAGER.save(self.state)
        output = contextlib.redirect_stdout(io.StringIO())
        output.__enter__()
        self.addCleanup(output.__exit__, None, None, None)

    def snapshot_files(self):
        return {str(path.relative_to(self.work)): path.read_bytes()
                for path in self.work.rglob('*') if path.is_file()}

    def prepare_job(self, identifier='100', state='COMPLETED', elapsed=100):
        output = self.work / 'paired-01'
        output.mkdir(exist_ok=True)
        MANAGER.write_new(output / 'preparation-status.json', dict(status='source_prepared'))
        job = dict(phase='prepare', stage='prepare', job_id=identifier, gpu=True,
                   scope=str(self.work / 'config/prepare-01.json'), output=str(output), token='old')
        self.state['jobs'].append(job)
        MANAGER.save(self.state)
        row = dict(state=state, exit_code='0:0' if state == 'COMPLETED' else '1:0',
                   gpu_seconds=elapsed, allocation='cpu=8,gres/gpu=1,gres/gpu:h100=1')
        return job, row

    def bundle(self):
        return SimpleNamespace(evidence_status='source-run', validate=lambda held: None,
            records=[dict(split='development', extractor_family='vitpose')])

    def test_initialization_inherits_assets_and_freezes_explicit_scope_without_jobs(self):
        args = SimpleNamespace(name='source-smoke-01', gpu_hours=4., prior_gpu_hours=.25,
                               prior_ledger=None, updates=200, seeds=[17])
        env = dict(USER='another-user', ST_MODEL_ROOT='/existing/models', ST_AMASS_ROOT='/existing/amass',
                   ST_BODY_MODEL_ROOT='/existing/body', ST_ACCOUNT='custom', ST_PARTITION='batch')
        with patch.object(MANAGER, 'ROOT', self.work), patch.object(MANAGER, 'environment_ok', return_value=True), \
                patch.object(MANAGER, 'create_drafts', return_value='drafts'), \
                patch.object(MANAGER.subprocess, 'check_output') as scheduler:
            created = MANAGER.initialize(args, env)
            scheduler.assert_not_called()
            original = (created / 'control.json').read_bytes()
            with self.assertRaises(FileExistsError):
                MANAGER.initialize(args, env)
        settings = MANAGER.read(created / 'control.json')
        preparation = MANAGER.read(created / 'config/preparation.json')
        self.assertEqual(settings['prior_entries'][0]['gpu_seconds'], 900)
        self.assertEqual(settings['account'], 'custom')
        self.assertEqual(settings['jobs'], [])
        self.assertEqual(preparation['amass_root'], '/existing/amass')
        self.assertEqual([row['student_id'] for row in preparation['estimators']], ['rtmpose_m', 'hrnet_w32', 'vitpose_base'])
        self.assertEqual(original, (created / 'control.json').read_bytes())
        self.assertIn('export STV2_PYTHON=', (created / 'session.env').read_text())
        self.assertFalse((created / 'inputs/locomotion-audit.csv').exists())

    def test_incompatible_interpreter_refuses_before_creating_run(self):
        args = SimpleNamespace(name='new-run', gpu_hours=4., prior_gpu_hours=0., prior_ledger=None, updates=200, seeds=[17])
        with patch.object(MANAGER, 'ROOT', self.work), patch.object(MANAGER, 'environment_ok', return_value=False):
            with self.assertRaisesRegex(ValueError, 'No run files created'):
                MANAGER.initialize(args)
        self.assertFalse((self.work / 'outputs').exists())

    def test_help_and_environment_entry_point_do_not_require_scientific_imports(self):
        # -S removes site packages, including NumPy/Torch, from this interpreter.
        result = subprocess.run([sys.executable, '-S', str(SCRIPTS / 'haic.py'), '--help'],
                                capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn('Manage one HAIC development run', result.stdout)

    def test_preparation_preview_does_not_write_or_submit(self):
        before = self.snapshot_files()
        with patch.object(MANAGER, 'check'), patch.object(MANAGER.subprocess, 'check_output') as scheduler:
            MANAGER.prepare(self.state, SimpleNamespace(retry=False, dry_run=True))
            scheduler.assert_not_called()
        self.assertEqual(before, self.snapshot_files())

    def test_accounting_uses_only_allocation_rows_and_counts_failed_attempts(self):
        job, _ = self.prepare_job(state='FAILED')
        raw = '100|FAILED|1:0|100|cpu=8,gres/gpu=1,gres/gpu:h100=1|\n100.batch|FAILED|1:0|100|gres/gpu=1|\n100.extern|COMPLETED|0:0|100|gres/gpu=1|\n'
        with patch.object(MANAGER.subprocess, 'check_output', return_value=raw):
            snapshot = MANAGER.scheduler_snapshot([job])
        self.assertEqual(set(snapshot), {'100'})
        entries = MANAGER.ledger_entries(self.state, snapshot)
        self.assertEqual(sum(row['gpu_seconds'] for row in entries), 221)
        self.assertEqual(entries[-1]['status'], 'FAILED')
        snapshot['100']['allocation'] = 'gres/gpu=2,gres/gpu:h100=1'
        with self.assertRaises(ValueError):
            MANAGER.ledger_entries(self.state, snapshot)
        snapshot['100'].update(allocation='gres/gpu=1', gpu_seconds=0)
        self.assertEqual(MANAGER.ledger_entries(self.state, snapshot)[-1]['gpu_seconds'], 1)
        snapshot['100'].update(allocation='', state='CANCELLED')
        self.assertEqual(MANAGER.ledger_entries(self.state, snapshot)[-1]['gpu_seconds'], 0)

    def test_requeued_allocations_are_not_silently_overwritten(self):
        job, _ = self.prepare_job()
        raw = '100|NODE_FAIL|1:0|100|gres/gpu=1|\n100|COMPLETED|0:0|20|gres/gpu=1|\n'
        with patch.object(MANAGER.subprocess, 'check_output', return_value=raw) as accounting:
            with self.assertRaisesRegex(RuntimeError, 'Multiple allocation records'):
                MANAGER.scheduler_snapshot([job])
            self.assertIn('-D', accounting.call_args.args[0])

    def test_unknown_or_active_scheduler_state_blocks_next_submission(self):
        self.prepare_job()
        for snapshot in ({}, {'100': dict(state='PENDING')}):
            with self.subTest(snapshot=snapshot), patch.object(MANAGER, 'scheduler_snapshot', return_value=snapshot):
                with self.assertRaisesRegex(RuntimeError, 'still active or accounting'):
                    MANAGER.settled(self.state)

    def test_failed_sbatch_preserves_ambiguous_attempt_until_explicit_recovery(self):
        error = subprocess.CalledProcessError(1, 'sbatch', output='12345\n')
        with patch.object(MANAGER.subprocess, 'check_output', side_effect=error):
            with self.assertRaises(subprocess.CalledProcessError):
                MANAGER.submit_one(self.state, 'prepare', True, 'prepare.sbatch', [], {}, phase='prepare', scope='scope.json')
        pending = MANAGER.read(self.work / 'control.json')['pending_submission']
        self.assertIsNotNone(pending)
        with self.assertRaisesRegex(RuntimeError, 'uncertain'):
            MANAGER.settled(self.state)
        token = pending['record']['token']
        with patch.dict(os.environ, USER='fixture-user'), patch.object(MANAGER.subprocess, 'check_output',
                return_value=f'12345|stv2-prepare-{token}|fixture-user|\n'):
            MANAGER.recover(self.state, SimpleNamespace(job_id='12345', not_submitted=False))
        self.assertIsNone(self.state['pending_submission'])
        self.assertEqual(self.state['jobs'][0]['job_id'], '12345')

    def test_stale_scheduler_environment_cannot_add_gpus_or_arrays(self):
        with patch.object(MANAGER.subprocess, 'check_output', return_value='123\n') as sbatch:
            MANAGER.submit_one(self.state, 'audit', False, 'stage.sbatch', [],
                dict(SBATCH_GPUS='2', SBATCH_ARRAY_INX='1-100', STV2_PYTHON=sys.executable),
                phase='source', scope='scope.json')
        self.assertEqual(sbatch.call_args.kwargs['env'], dict(STV2_PYTHON=sys.executable))
        # If Slurm nevertheless allocated a GPU to a nominal CPU job, retain its cost.
        accounting = {'123': dict(state='COMPLETED', exit_code='0:0', gpu_seconds=50, allocation='gres/gpu=1')}
        self.assertEqual(MANAGER.ledger_entries(self.state, accounting)[-1]['gpu_seconds'], 51)

    def test_prepare_retry_carries_allocation_costs_into_fresh_scope(self):
        _, row = self.prepare_job(state='FAILED')
        with patch.object(MANAGER, 'check'), patch.object(MANAGER, 'scheduler_snapshot', return_value={'100': row}), \
                patch.object(MANAGER.subprocess, 'check_output', return_value='101\n') as sbatch:
            MANAGER.prepare(self.state, SimpleNamespace(retry=True, dry_run=False))
        scope = MANAGER.read(self.state['jobs'][-1]['scope'])
        ledger = MANAGER.read(scope['cost_ledger'])
        self.assertAlmostEqual(scope['measured_gpu_hours'], 221 / 3600)
        self.assertEqual(len(ledger['entries']), 2)
        command = sbatch.call_args.args[0]
        self.assertIn('--account=my-account', command)
        self.assertIn('--gres=gpu:h100:1', command)
        self.assertIn('--no-requeue', command)
        self.assertTrue(Path(sbatch.call_args.kwargs['env']['STV2_PREPARATION_CONFIG']).is_file())

    def test_training_preview_and_submission_preserve_prior_scope_and_block_duplicates(self):
        _, row = self.prepare_job()
        before = self.snapshot_files()
        with patch.object(MANAGER, 'scheduler_snapshot', return_value={'100': row}), \
                patch(CONTRACTS + '.verify_preservation'), patch(CONTRACTS + '.TrackBundle.load', return_value=self.bundle()), \
                patch.object(MANAGER.subprocess, 'check_output') as sbatch:
            MANAGER.source(self.state, SimpleNamespace(retry=False, dry_run=True, overlays_reviewed=True))
            sbatch.assert_not_called()
        self.assertEqual(before, self.snapshot_files())
        commands = []
        def scheduler(command, **kwargs):
            # All accepted IDs must be on disk before the next scheduler call.
            self.assertEqual(len(MANAGER.read(self.work / 'control.json')['jobs']), 1 + len(commands))
            commands.append(command)
            return f'{200 + len(commands)};cluster\n'
        with patch.object(MANAGER, 'scheduler_snapshot', return_value={'100': row}), \
                patch(CONTRACTS + '.verify_preservation'), patch(CONTRACTS + '.TrackBundle.load', return_value=self.bundle()), \
                patch.object(MANAGER.subprocess, 'check_output', side_effect=scheduler):
            MANAGER.source(self.state, SimpleNamespace(retry=False, dry_run=False, overlays_reviewed=True))
        self.assertEqual(len(commands), 10)
        self.assertEqual(sum('--gres=gpu:h100:1' in cmd for cmd in commands), 2)
        self.assertNotIn('--dependency=afterok:201', commands[0])
        self.assertIn('--dependency=afterok:201', commands[1])
        self.assertEqual(len(self.state['jobs']), 11)
        cfg = MANAGER.RunConfig.load(self.state['source_config'])
        self.assertAlmostEqual(cfg.measured_gpu_hours, 221 / 3600)
        terminal = {job['job_id']: row for job in self.state['jobs']}
        with patch.object(MANAGER, 'scheduler_snapshot', return_value=terminal), patch.object(MANAGER, 'submit_one') as repeat:
            with self.assertRaisesRegex(ValueError, 'already submitted'):
                MANAGER.source(self.state, SimpleNamespace(retry=False, dry_run=False, overlays_reviewed=True))
            repeat.assert_not_called()

    def test_training_requires_human_overlay_review_and_full_allocation_allowance(self):
        _, row = self.prepare_job()
        with patch.object(MANAGER, 'scheduler_snapshot', return_value={'100': row}):
            with self.assertRaisesRegex(ValueError, 'Inspect'):
                MANAGER.source(self.state, SimpleNamespace(retry=False, dry_run=False, overlays_reviewed=False))
        self.state['authorized_gpu_hours'] = 2
        with patch.object(MANAGER, 'scheduler_snapshot', return_value={'100': row}), \
                patch(CONTRACTS + '.verify_preservation'), patch(CONTRACTS + '.TrackBundle.load', return_value=self.bundle()):
            with self.assertRaisesRegex(PermissionError, 'two|2 one-hour'):
                MANAGER.source(self.state, SimpleNamespace(retry=False, dry_run=False, overlays_reviewed=True))

    def test_source_retry_reconciles_all_allocations_once_without_rewriting_identity_inputs(self):
        _, prepared = self.prepare_job()
        initial = MANAGER.ledger_entries(self.state, {'100': prepared})
        path, config = MANAGER.make_scope(self.state, 'source-01', self.work / 'paired-01/bundle', initial)
        self.state['source_config'] = str(path)
        self.state['jobs'].extend([
            dict(phase='source', stage='audit', job_id='201', gpu=False, scope=str(path), output=''),
            dict(phase='source', stage='direct', job_id='202', gpu=True, scope=str(path), output='')])
        snapshot = {'100': prepared,
                    '201': dict(state='COMPLETED', exit_code='0:0', gpu_seconds=50, allocation='gres/gpu=1'),
                    '202': dict(state='FAILED', exit_code='1:0', gpu_seconds=100, allocation='gres/gpu=1')}
        MANAGER.write_new(config.root / 'costs/direct-attempt.json',
                          dict(stage='direct', slurm_job='202', gpu_seconds=75, status='failed'))
        frozen_config, frozen_ledger = path.read_bytes(), Path(config.cost_ledger).read_bytes()
        with patch.object(MANAGER, 'scheduler_snapshot', return_value=snapshot), \
                patch(CONTRACTS + '.verify_preservation'), patch(CONTRACTS + '.TrackBundle.load', return_value=self.bundle()), \
                patch.object(MANAGER, 'submit_one'):
            for _ in range(2):
                MANAGER.source(self.state, SimpleNamespace(retry=True, dry_run=False, overlays_reviewed=False))
        records = [MANAGER.read(path) for path in (config.root / 'costs').glob('*.json')]
        self.assertEqual(sum(row['gpu_seconds'] for row in records if row['slurm_job'] == '201'), 51)
        self.assertEqual(sum(row['gpu_seconds'] for row in records if row['slurm_job'] == '202'), 101)
        self.assertEqual(len(records), 3)
        self.assertEqual(path.read_bytes(), frozen_config)
        self.assertEqual(Path(config.cost_ledger).read_bytes(), frozen_ledger)


if __name__ == '__main__':
    unittest.main()
