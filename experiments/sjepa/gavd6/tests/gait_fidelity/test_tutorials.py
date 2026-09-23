"""Workflow checks for restart safety, batch relocation and tutorial structure."""
from __future__ import annotations

import importlib.util
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[2]
SLURM = ROOT / 'slurm/gait-fidelity'
NOTEBOOKS = ROOT / 'notebooks/gait_fidelity'


def import_file(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


submit = import_file('gait_fidelity_slurm_submit', SLURM / 'submit.py')


class TutorialStructureTests(unittest.TestCase):
    def test_notebooks_are_ordered_and_code_compiles(self):
        notebooks = sorted(NOTEBOOKS.glob('*.ipynb'))
        self.assertEqual(len(notebooks), 7)
        for index, path in enumerate(notebooks):
            self.assertTrue(path.name.startswith(f'{index:02d}_'))
        experiments = sorted((NOTEBOOKS / 'experiments').glob('*.ipynb'))
        self.assertEqual(len(experiments), 5)
        for index, path in enumerate(experiments):
            self.assertTrue(path.name.startswith(chr(ord('A') + index) + '_'))
        for path in notebooks + experiments:
            notebook = json.loads(path.read_text())
            self.assertEqual(notebook['nbformat'], 4)
            self.assertTrue(notebook['cells'][0]['source'].startswith('# '))
            for cell_index, cell in enumerate(notebook['cells']):
                if cell['cell_type'] == 'code':
                    compile(cell['source'], f'{path.name}:cell{cell_index}', 'exec')
                    self.assertEqual(cell['outputs'], [])

    def test_shell_syntax(self):
        for path in sorted(SLURM.glob('*')):
            if path.suffix in {'.sh', '.sbatch'}:
                subprocess.run(['bash', '-n', str(path)], check=True, capture_output=True, text=True)

    def test_batch_script_works_when_slurm_moves_it(self):
        # Slurm executes a copy from its spool; BASH_SOURCE is not the release path.
        with tempfile.TemporaryDirectory() as temporary:
            folder = Path(temporary)
            interpreter, captured = folder / 'python', folder / 'arguments.txt'
            interpreter.write_text('#!/usr/bin/env bash\nprintf "%s\\n" "$@" > ' + shlex.quote(str(captured)) + '\n')
            interpreter.chmod(0o755)
            work = folder / 'work'
            work.mkdir()
            (work / 'session.env').write_text('\n'.join([
                f'export GF_ROOT={shlex.quote(str(ROOT))}',
                f'export GF_WORK={shlex.quote(str(work))}',
                f'export GF_PYTHON={shlex.quote(str(interpreter))}',
            ]) + '\n')
            env = dict(os.environ, GF_ROOT=str(ROOT), SLURM_JOB_ID='123')
            spool = folder / 'slurm_script'
            spool.write_text((SLURM / 'worker.sbatch').read_text())
            subprocess.run(['bash', str(spool), str(work), 'phase-1', str(folder / 'attempt-1')],
                           env=env, check=True, capture_output=True, text=True)
            self.assertIn('--phase-id\nphase-1\n--attempt\n', captured.read_text())
            spool.write_text((SLURM / 'coordinator.sbatch').read_text())
            subprocess.run(['bash', str(spool), str(work), '8', 'prepare-only'],
                           env=env, check=True, capture_output=True, text=True)
            self.assertIn('--prepare-only', captured.read_text())
            for stage in ('gavd-only','gavd-confirmation'):
                subprocess.run(['bash',str(spool),str(work),'8',stage],env=env,
                               check=True,capture_output=True,text=True)
                self.assertIn('--'+stage,captured.read_text())

    def test_unsafe_remote_destination_rejected(self):
        for name, args in [('sync-to-haic.sh', ['haic:/tmp/path;touch-injection', '--apply']),
                           ('retrieve.sh', ['haic:/tmp/../private', '/tmp/gf-not-created'])]:
            result = subprocess.run(['bash', str(SLURM / name), *args], capture_output=True, text=True)
            self.assertNotEqual(result.returncode, 0)


class CoordinatorSubmissionTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.work = Path(self.temporary.name)
        (self.work / 'config.json').write_text(json.dumps({'fixture': False}))
        self.env = patch.dict(os.environ, {'GF_ROOT': str(ROOT)})
        self.env.start()

    def tearDown(self):
        self.env.stop()
        self.temporary.cleanup()

    def invoke(self, extra=()):
        with patch.object(sys, 'argv', ['submit.py', '--work', str(self.work), *extra]):
            return submit.main()

    def test_active_submission_not_duplicated(self):
        calls = []

        def run(command, **kwargs):
            calls.append(command)
            return subprocess.CompletedProcess(command, 0, '12345\n', '')

        with patch.object(submit.subprocess, 'run', side_effect=run):
            self.invoke(['--prepare-only'])
            self.invoke()
        self.assertEqual(sum(command[0] == 'sbatch' for command in calls), 1)
        self.assertEqual(calls[0][-1], 'prepare-only')
        receipt = json.loads((self.work / 'control/coordinator.json').read_text())
        self.assertEqual(receipt['job_id'], '12345')

    def test_ambiguous_submission_blocks_duplicate(self):
        result = subprocess.CompletedProcess(['sbatch'], 0, 'unexpected response\n', '')
        with patch.object(submit.subprocess, 'run', return_value=result) as runner:
            with self.assertRaises(SystemExit):
                self.invoke()
            with self.assertRaises(SystemExit):
                self.invoke()
        self.assertEqual(runner.call_count, 1)
        self.assertEqual(json.loads((self.work / 'control/coordinator.json').read_text())['state'], 'submitting')

    def test_finished_coordinator_reconciled_with_sacct(self):
        responses = [subprocess.CompletedProcess([], 1, '', 'Invalid job id specified'),
                     subprocess.CompletedProcess([], 0, '12345|COMPLETED\n', '')]
        with patch.object(submit.subprocess, 'run', side_effect=responses):
            self.assertFalse(submit.active_job('12345'))

    def test_transport_error_does_not_authorize_duplicate(self):
        result = subprocess.CompletedProcess(['sbatch'], 1, '', 'Socket timed out on send/recv operation')
        with patch.object(submit.subprocess, 'run', return_value=result) as runner:
            with self.assertRaises(SystemExit):
                self.invoke()
            with self.assertRaises(SystemExit):
                self.invoke()
        self.assertEqual(runner.call_count, 1)
        self.assertEqual(json.loads((self.work / 'control/coordinator.json').read_text())['state'], 'submitting')

    def test_fixture_cannot_consume_cluster_allocation(self):
        (self.work / 'config.json').write_text(json.dumps({'fixture': True}))
        with patch.object(submit.subprocess, 'run') as runner:
            with self.assertRaises(SystemExit):
                self.invoke()
            runner.assert_not_called()

    def test_saved_resources_and_gavd_stage_are_used(self):
        (self.work/'config.json').write_text(json.dumps({'fixture':False,'resources':{
            'account':'research','partition':'compute','controller_memory':'96G'}}))
        with patch.object(submit.subprocess,'run',return_value=subprocess.CompletedProcess([],0,'12345\n','')) as runner:
            self.invoke(['--gavd-only'])
        command=runner.call_args.args[0]
        self.assertIn('--account=research',command)
        self.assertIn('--partition=compute',command)
        self.assertIn('--mem=96G',command)
        self.assertEqual(command[-1],'gavd-only')


if __name__ == '__main__':
    unittest.main()
