"""Adversarial controller tests use a simulated scheduler and no allocated jobs."""
import contextlib
import errno
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
SPEC = importlib.util.spec_from_file_location('stv2_automated_run', SCRIPTS / 'automated_run.py')
AUTO = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(AUTO)


class AutomatedRunTests(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.work = Path(directory.name).resolve() / 'run'
        (self.work / 'config').mkdir(parents=True)
        self.state = dict(work=str(self.work), python=sys.executable, jobs=[], source_config=None,
                          pending_submission=None, held_extractor='vitpose', account='fixture-account',
                          partition='fixture-partition', authorized_gpu_hours=4., prior_entries=[])
        self.save()
        self.config_path = self.work / 'config/preparation.json'
        self.config_path.write_text(json.dumps(dict(locomotion_audit='old.csv', reservation_csv='old-people.csv')))
        output = contextlib.redirect_stdout(io.StringIO())
        output.__enter__()
        self.addCleanup(output.__exit__, None, None, None)

    def save(self):
        (self.work / 'control.json').write_text(json.dumps(self.state))

    def job(self, stage, identifier, phase='source', **extra):
        row = dict(stage=stage, job_id=identifier, phase=phase, gpu=stage in {'prepare', 'direct', 'jepa'}, **extra)
        self.state['jobs'].append(row)
        self.save()
        return row

    @staticmethod
    def accounting(state='COMPLETED', exit_code='0:0'):
        return dict(state=state, exit_code=exit_code, gpu_seconds=0., allocation='')

    def test_wait_uses_latest_attempt_and_does_not_count_old_failure_twice(self):
        self.job('prepare', '100', phase='prepare')
        newest = self.job('prepare', '101', phase='prepare')
        snapshot = {'100': self.accounting('FAILED', '1:0'), '101': self.accounting()}
        with patch.object(AUTO.haic, 'scheduler_snapshot', return_value=snapshot), patch.object(AUTO.time, 'sleep') as sleep:
            actual = AUTO.wait_jobs(self.work, 'prepare', poll_seconds=0)
        self.assertEqual(actual, {'prepare': newest})
        sleep.assert_not_called()

    def test_status_prints_authoritative_automated_input_directory(self):
        output = str(self.work / 'inputs/automated-02')
        self.job('automated_inputs', '100', phase='automated_inputs', output=output)
        buffer = io.StringIO()
        with patch.object(AUTO.haic, 'scheduler_snapshot', return_value={'100': self.accounting()}), \
                contextlib.redirect_stdout(buffer):
            AUTO.haic.status(self.state)
        self.assertIn(output, buffer.getvalue())

    def test_failed_source_cancels_only_its_pending_dependents_then_stops(self):
        self.job('direct', '100')
        self.job('jepa', '101')
        self.job('evaluate', '102')
        active = {'100': self.accounting(), '101': self.accounting('FAILED', '1:0'),
                  '102': self.accounting('PENDING', '0:0')}
        finished = dict(active, **{'102': self.accounting('CANCELLED', '0:0')})
        with patch.object(AUTO.haic, 'scheduler_snapshot', side_effect=[active, finished]), \
                patch.object(AUTO.subprocess, 'check_output', return_value='100|RUNNING\n102|PENDING\n999|PENDING\n'), \
                patch.object(AUTO.subprocess, 'run') as cancel, patch.object(AUTO.time, 'sleep'):
            with self.assertRaises(AUTO.PhaseFailed):
                AUTO.wait_jobs(self.work, 'source', poll_seconds=0)
        # One cancellation per queue poll is harmless; unrelated job 999 and active job 100 are untouched.
        for action in cancel.call_args_list:
            self.assertEqual(action.args[0], ['scancel', '102'])

    def test_unknown_scheduler_state_is_not_an_infinite_wait(self):
        self.job('prepare', '100', phase='prepare')
        with patch.object(AUTO.haic, 'scheduler_snapshot', return_value={'100': self.accounting('UNRECOGNIZED')}), \
                patch.object(AUTO.time, 'sleep') as sleep:
            with self.assertRaisesRegex(ValueError, 'Unexpected scheduler states'):
                AUTO.wait_jobs(self.work, 'prepare', poll_seconds=0)
        sleep.assert_not_called()

    def test_scheduler_errors_are_not_eligible_for_automatic_retry(self):
        self.job('prepare', '100', phase='prepare')
        for error in (RuntimeError('Multiple allocation records'), ValueError('Invalid elapsed time')):
            with self.subTest(error=error), patch.object(AUTO, 'wait_jobs', side_effect=error), \
                    patch.object(AUTO.subprocess, 'run') as submit:
                with self.assertRaises(type(error)):
                    AUTO.execute_phase(self.work, 'prepare', retry=True, poll_seconds=0)
                submit.assert_not_called()

    def test_accounting_command_failure_retries_at_most_three_reads(self):
        self.job('prepare', '100', phase='prepare')
        failure = subprocess.CalledProcessError(1, 'sacct')
        with patch.object(AUTO.haic, 'scheduler_snapshot', side_effect=failure) as read, \
                patch.object(AUTO.time, 'sleep') as sleep:
            with self.assertRaises(subprocess.CalledProcessError):
                AUTO.wait_jobs(self.work, 'prepare', poll_seconds=0)
        self.assertEqual(read.call_count, 3)
        self.assertEqual(sleep.call_count, 2)

    def test_source_submission_uses_automatic_flag_without_human_claim(self):
        with patch.object(AUTO, 'wait_jobs') as wait, patch.object(AUTO.subprocess, 'run') as submit:
            AUTO.execute_phase(self.work, 'source', retry=False, poll_seconds=30)
        command = submit.call_args.args[0]
        self.assertIn('--automated-screen', command)
        self.assertNotIn('--overlays-reviewed', command)
        self.assertNotIn('--retry', command)
        self.assertEqual(submit.call_args.kwargs['env']['STV2_WORK'], str(self.work))
        wait.assert_called_once_with(self.work, 'source', poll_seconds=30)

    def test_partial_successful_source_chain_requires_explicit_retry(self):
        self.job('audit', '100')
        with patch.object(AUTO, 'wait_jobs'), patch.object(AUTO.subprocess, 'run') as submit:
            with self.assertRaisesRegex(ValueError, 'partly submitted'):
                AUTO.execute_phase(self.work, 'source', retry=False, poll_seconds=0)
            submit.assert_not_called()
        with patch.object(AUTO, 'wait_jobs') as wait, patch.object(AUTO.subprocess, 'run') as submit:
            AUTO.execute_phase(self.work, 'source', retry=True, poll_seconds=0)
        self.assertIn('--retry', submit.call_args.args[0])
        self.assertIn('--automated-screen', submit.call_args.args[0])
        self.assertEqual(wait.call_count, 2)

    def test_terminal_job_failure_requires_retry_and_never_silently_resubmits(self):
        self.job('prepare', '100', phase='prepare')
        for retry in (False, True):
            with self.subTest(retry=retry), patch.object(AUTO, 'wait_jobs', side_effect=[AUTO.PhaseFailed('failed'), {}]), \
                    patch.object(AUTO.subprocess, 'run') as submit:
                if retry:
                    AUTO.execute_phase(self.work, 'prepare', retry=True, poll_seconds=0)
                    self.assertIn('--retry', submit.call_args.args[0])
                else:
                    with self.assertRaises(AUTO.PhaseFailed):
                        AUTO.execute_phase(self.work, 'prepare', retry=False, poll_seconds=0)
                    submit.assert_not_called()

    def test_automated_inputs_are_cpu_only_and_bound_after_validator_succeeds(self):
        previous = self.config_path.read_bytes()
        output = self.work / 'inputs/automated-01'
        def submit(state, stage, gpu, script, arguments, env, **details):
            self.assertFalse(gpu)
            self.assertEqual(script, 'automated-inputs.sbatch')
            self.assertEqual(AUTO.read(arguments[0])['review_mode'], 'automated_development')
            self.assertEqual(arguments[1], str(output))
            state['jobs'].append(dict(job_id='100', stage=stage, gpu=gpu,
                                     **{key: str(value) if isinstance(value, Path) else value
                                        for key, value in details.items()}))
            AUTO.haic.save(state)
        with patch.object(AUTO.haic, 'submit_one', side_effect=submit), \
                patch.object(AUTO, 'wait_jobs', return_value={'automated_inputs': {'output': str(output)}}), \
                patch('haic_inputs.check_inputs', return_value={'status': 'pass'}) as validate:
            AUTO.prepare_automated_inputs(self.work, retry=False, poll_seconds=0)
        bound = AUTO.read(self.config_path)
        self.assertEqual(bound['review_mode'], 'automated_development')
        self.assertEqual(bound['locomotion_audit'], str(output / 'locomotion-audit.csv'))
        self.assertEqual(bound['reservation_csv'], str(output / 'person-reservations.csv'))
        self.assertEqual(validate.call_args.args, (bound, 'vitpose'))
        self.assertEqual(AUTO.read(self.config_path.with_name('preparation.before-automation.json')), json.loads(previous))

    def test_bad_automated_inputs_leave_authoritative_configuration_unchanged(self):
        self.job('automated_inputs', '100', phase='automated_inputs', output=str(self.work / 'inputs/automated-01'))
        before = self.config_path.read_bytes()
        with patch.object(AUTO.haic, 'scheduler_snapshot', return_value={'100': self.accounting()}), \
                patch.object(AUTO, 'wait_jobs', return_value={'automated_inputs': self.state['jobs'][0]}), \
                patch('haic_inputs.check_inputs', side_effect=ValueError('bad metadata')):
            with self.assertRaisesRegex(ValueError, 'bad metadata'):
                AUTO.prepare_automated_inputs(self.work, retry=False, poll_seconds=0)
        self.assertEqual(self.config_path.read_bytes(), before)
        self.assertFalse(self.config_path.with_name('preparation.before-automation.json').exists())

    def test_uncertain_submission_blocks_before_new_jobs(self):
        self.state['pending_submission'] = {'token': 'uncertain'}
        self.save()
        with patch.object(AUTO.haic, 'submit_one') as submit:
            with self.assertRaisesRegex(RuntimeError, 'Uncertain submission'):
                AUTO.prepare_automated_inputs(self.work, retry=True, poll_seconds=0)
            submit.assert_not_called()

    def test_existing_manual_experiment_is_not_adopted(self):
        self.job('prepare', '100', phase='prepare')
        with patch.object(AUTO.haic, 'submit_one') as submit:
            with self.assertRaisesRegex(ValueError, 'different experiment'):
                AUTO.prepare_automated_inputs(self.work, retry=True, poll_seconds=0)
            submit.assert_not_called()

    def test_failed_input_retry_cannot_mutate_inputs_after_gpu_submission(self):
        self.job('automated_inputs', '100', phase='automated_inputs')
        self.job('prepare', '101', phase='prepare')
        snapshot = {'100': self.accounting('FAILED', '1:0'), '101': self.accounting()}
        with patch.object(AUTO.haic, 'scheduler_snapshot', return_value=snapshot), patch.object(AUTO.haic, 'submit_one') as submit:
            with self.assertRaisesRegex(ValueError, 'Do not change source inputs'):
                AUTO.prepare_automated_inputs(self.work, retry=True, poll_seconds=0)
            submit.assert_not_called()

    def test_running_controller_does_not_launch_a_duplicate(self):
        with AUTO.haic.stage_lock(self.work, 'automation'), patch.object(AUTO.subprocess, 'Popen') as child:
            AUTO.launch(self.work)
        child.assert_not_called()

    def test_reused_live_pid_does_not_block_a_free_controller_lock(self):
        (self.work / 'automation-process.json').write_text(json.dumps({'pid': os.getpid()}))
        with patch.object(AUTO.subprocess, 'Popen', return_value=SimpleNamespace(pid=12345)) as child:
            AUTO.launch(self.work)
        child.assert_called_once()
        self.assertEqual(AUTO.read(self.work / 'automation-process.json')['pid'], 12345)

    def test_running_probe_uses_shared_lock_without_creating_or_mutating_it(self):
        path = self.work / 'locks/process/automation.lock'
        self.assertFalse(AUTO.controller_running(self.work))
        self.assertFalse(path.exists())
        with AUTO.haic.stage_lock(self.work, 'automation'):
            self.assertTrue(AUTO.controller_running(self.work))
        before = path.read_bytes()
        self.assertFalse(AUTO.controller_running(self.work))
        self.assertEqual(path.read_bytes(), before)

    def test_running_probe_supports_nfs_exclusive_locks(self):
        # Linux NFS emulates flock with byte-range locks: LOCK_EX needs a
        # writable descriptor even though native local flock may accept 'r'.
        flock = AUTO.fcntl.flock

        def nfs_flock(stream, operation):
            mode = AUTO.fcntl.fcntl(stream.fileno(), AUTO.fcntl.F_GETFL) & os.O_ACCMODE
            if operation & AUTO.fcntl.LOCK_EX and mode == os.O_RDONLY:
                raise OSError(errno.EBADF, os.strerror(errno.EBADF))
            return flock(stream, operation)

        path = self.work / 'locks/process/automation.lock'
        path.parent.mkdir(parents=True)
        path.write_bytes(b'preserve existing lock contents\n')
        before = path.read_bytes()
        with patch.object(AUTO.fcntl, 'flock', side_effect=nfs_flock):
            self.assertFalse(AUTO.controller_running(self.work))
            with AUTO.haic.stage_lock(self.work, 'automation'), \
                    patch.object(AUTO.subprocess, 'Popen') as child:
                self.assertTrue(AUTO.controller_running(self.work))
                AUTO.launch(self.work)
                child.assert_not_called()
            self.assertFalse(AUTO.controller_running(self.work))
            with patch.object(AUTO.subprocess, 'Popen', return_value=SimpleNamespace(pid=12345)) as child:
                AUTO.launch(self.work)
                child.assert_called_once()
            with patch.object(AUTO.haic, 'status'):
                AUTO.status(self.work)
        self.assertEqual(path.read_bytes(), before)

    def test_launch_is_detached_noninteractive_and_preserves_retry_flag(self):
        with patch.object(AUTO.subprocess, 'Popen', return_value=SimpleNamespace(pid=12345)) as child:
            AUTO.launch(self.work, retry=True)
        command = child.call_args.args[0]
        self.assertIn('--retry', command)
        self.assertIn('run', command)
        self.assertTrue(child.call_args.kwargs['start_new_session'])
        self.assertEqual(child.call_args.kwargs['cwd'], REPO)
        self.assertEqual(AUTO.read(self.work / 'automation-process.json')['pid'], 12345)

    def test_full_controller_phase_order_and_verified_completion(self):
        events = []
        def stage(work, phase, **kwargs):
            events.append(phase)
        checker = SimpleNamespace(check_results=lambda work: events.append('verify') or {'report': 'fixture-report'})
        with patch.object(AUTO.haic, 'environment_ok', return_value=True), \
                patch('gavd6_sjepa.research_directions.synthetic_training_v2.contracts.verify_preservation'), \
                patch('gavd6_sjepa.research_directions.synthetic_training_v2.audit.reconstruct_pilot'), \
                patch.object(AUTO, 'prepare_automated_inputs', side_effect=lambda *a, **k: events.append('inputs')), \
                patch.object(AUTO, 'execute_phase', side_effect=stage), patch.dict(sys.modules, {'check_results': checker}):
            AUTO.run(self.work, poll_seconds=10)
        self.assertEqual(events, ['inputs', 'prepare', 'source', 'verify'])
        self.assertEqual(AUTO.read(self.work / 'automation.json')['status'], 'complete')

    def test_failed_precheck_records_failure_without_submission(self):
        with patch.object(AUTO.haic, 'environment_ok', return_value=False), \
                patch.object(AUTO, 'prepare_automated_inputs') as inputs:
            with self.assertRaisesRegex(ValueError, 'environment check'):
                AUTO.run(self.work, poll_seconds=10)
        inputs.assert_not_called()
        self.assertEqual(AUTO.read(self.work / 'automation.json')['status'], 'failed')


if __name__ == '__main__':
    unittest.main()
