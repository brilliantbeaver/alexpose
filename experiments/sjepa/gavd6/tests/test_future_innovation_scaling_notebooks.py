"""Real-kernel inspection tests plus scheduler and interruption regressions."""
import hashlib
from contextlib import redirect_stdout
import io
import importlib.util
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import MagicMock, patch

import nbformat

ROOT = Path(__file__).resolve().parents[1]
LAUNCH = ROOT / 'slurm/future-innovation-scaling/launch'
spec = importlib.util.spec_from_file_location('source_curve_notebook_execution', LAUNCH / 'notebooks.py')
runner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runner)


def fixture(root, complete=False):
    (root / 'config').mkdir(parents=True)
    (root / 'reports').mkdir()
    (root / 'config/study.json').write_text('{"protocol":"source-learning-curve-v1","synthetic":true}')
    if complete:
        # Illustrative report schema only; explicitly synthetic and not a fit.
        report = dict(status='synthetic_calibration_only', measurement_complete=True, synthetic=True,
                      arm_means=[dict(arm='real-skeleton',r2_full=.2,r2_baseline=.1)],
                      contrasts=[dict(contrast='matched_increment',estimate=.05)],
                      per_subset_contrasts=[dict(size='40',subset_seed=261201,matched_increment=.05)])
        (root / 'reports/learning-curve.json').write_text(json.dumps(report))
        (root / 'reports/learning-curve.svg').write_text(
            '<svg xmlns="http://www.w3.org/2000/svg" width="120" height="30"><text x="2" y="20">Synthetic fixture</text></svg>')


def fake_success(notebook, **kwargs):
    def execute(**options):
        count=0
        for i,cell in enumerate(notebook.cells):
            kwargs['on_cell_start'](cell=cell,cell_index=i)
            if cell.cell_type=='code':
                count+=1; cell.execution_count=count
            kwargs['on_cell_executed'](cell=cell,cell_index=i)
    return MagicMock(execute=execute)


class SourceNotebookTests(unittest.TestCase):
    def test_source_generator_and_output_free_cells(self):
        path,notebook=runner.load_source('23')
        self.assertEqual(len([c for c in notebook.cells if c.cell_type=='code']),5)
        for cell in notebook.cells:
            if cell.cell_type=='code':
                self.assertEqual(cell.outputs,[])
                self.assertIsNone(cell.execution_count)
                compile(cell.source,str(path),'exec')

    def test_wrong_study_and_unsafe_output_rejected_before_outputs(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)/'run'; fixture(root)
            with self.assertRaises(ValueError): runner.execute_notebooks(root,timeout=0)
            with self.assertRaises(ValueError): runner.execute_notebooks(root,output_dir=ROOT)
            (root/'config/study.json').unlink()
            (root/'config/run-contract.json').write_text('{"protocol":"direct-v2"}')
            with self.assertRaisesRegex(ValueError,'historical gate'): runner.check_inputs(root)
            self.assertFalse((root/'notebook_runs').exists())

    def test_kernel_startup_failure_is_saved_without_starting_notebook_client(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)/'run'; fixture(root)
            manager=MagicMock(has_kernel=False)
            manager.start_kernel.side_effect=PermissionError('kernel port denied')
            with patch.object(runner,'KernelManager',return_value=manager), patch.object(runner,'NotebookClient') as client:
                with self.assertRaisesRegex(PermissionError,'kernel port denied'):
                    runner.execute_notebooks(root)
                client.assert_not_called()
            manager.cleanup_resources.assert_called_once()
            saved=nbformat.read(next((root/'notebook_runs').glob('*/*.ipynb')),as_version=4)
            self.assertEqual(saved.metadata.fi_execution.status,'failed')
            self.assertFalse(saved.metadata.fi_execution.execution_completed)

    def test_real_kernel_incomplete_run_uses_explicit_root_and_preserves_sources(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)/'selected run'; fixture(root)
            source,_=runner.load_source('23'); before=runner.digest(source)
            inputs=runner.input_snapshot(root)
            with patch.dict(os.environ,{'FI_RUN_ROOT':'/wrong/inherited/run','FI_TUTORIAL_MODE':'execute'}):
                output=runner.execute_notebooks(root)
            notebook=nbformat.read(output/source.name,as_version=4)
            record=notebook.metadata.fi_execution
            self.assertEqual(record.status,'passed')
            self.assertTrue(record.execution_completed)
            self.assertEqual(record.executed_code_cells,5)
            self.assertEqual(record.run_root,str(root.resolve()))
            self.assertFalse(record.numerical_verification_performed)
            self.assertFalse(record.artifact_integrity_verified)
            self.assertTrue(record.input_snapshot_unchanged)
            text=''.join(o.get('text','') for c in notebook.cells for o in c.get('outputs',[]))
            self.assertIn(str(root.resolve()),text)
            self.assertNotIn('/wrong/inherited/run',text)
            self.assertIn('execution is incomplete',text)
            self.assertEqual(runner.input_snapshot(root),inputs)
            self.assertEqual(runner.digest(source),before)
            self.assertEqual([p.suffix for p in output.iterdir()],['.ipynb'])
            self.assertTrue((root/'logs/notebooks'/output.name/'23-execution.json').exists())
            for cell in notebook.cells:
                if cell.cell_type=='markdown':
                    for target in re.findall(r'\]\(([^)]+)\)',cell.source):
                        if not target.startswith('#'):
                            self.assertTrue((output/target.split('#')[0]).exists(),target)

    def test_real_kernel_displays_saved_synthetic_report_and_embeds_svg(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)/'run'; fixture(root,complete=True)
            inputs=runner.input_snapshot(root)
            output=runner.execute_notebooks(root)
            notebook=nbformat.read(output/runner.NOTEBOOKS['23'],as_version=4)
            self.assertEqual(notebook.metadata.fi_execution.status,'passed')
            text=''.join(o.get('text','') for c in notebook.cells for o in c.get('outputs',[]))
            self.assertIn('synthetic_calibration_only',text)
            self.assertIn('not independently verified',text)
            self.assertIn('Synthetic fixture: True',text)
            self.assertTrue(any('image/svg+xml' in o.get('data',{}) for c in notebook.cells for o in c.get('outputs',[])))
            self.assertEqual(runner.input_snapshot(root),inputs)

    def test_real_kernel_failure_retains_traceback_and_partial_output(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)/'run'; fixture(root)
            (root/'reports/cohort-audit.json').write_text('{malformed JSON')
            with self.assertRaises(Exception): runner.execute_notebooks(root)
            notebook=nbformat.read(next((root/'notebook_runs').glob('*/*.ipynb')),as_version=4)
            self.assertEqual(notebook.metadata.fi_execution.status,'failed')
            self.assertFalse(notebook.metadata.fi_execution.execution_completed)
            self.assertEqual(notebook.cells[1].execution_count,1)
            self.assertTrue(any(o.get('output_type')=='error' for c in notebook.cells for o in c.get('outputs',[])))

    def test_expanded_counts_failures_and_stale_verification_are_distinguished(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)/'run'; fixture(root, complete=True)
            (root/'data/manifests').mkdir(parents=True)
            (root/'manifests').mkdir()
            (root/'logs/stages').mkdir(parents=True)
            (root/'config/availability-contract.json').write_text(json.dumps(dict(version='available-development-v1',
                summary=dict(included_development_recordings=2,unavailable_development_recordings=1))))
            (root/'config/media-availability.csv').write_text(
                'video_id,role,included,annotated_sequences,exclusion_reason\nv0,development,True,1,\nv1,development,True,2,\nv2,development,False,2,synthetic missing-file fixture\n')
            (root/'data/cohort-complete.json').write_text(json.dumps(dict(eligible_windows=3, eligible_sources=2,
                                                                       failed_pose_windows=0, confirmation_processed=False)))
            (root/'data/manifests/development-windows.csv').write_text(
                'window_id,video_id,outer_fold,evidence_origin\nw0,v0,0,parent_cache\nw1,v1,1,new_processing\nw2,v1,1,new_processing\n')
            (root/'data/cache-complete.json').write_text('{"reused_windows":1,"new_windows":2}')
            (root/'manifests/learning-plan.json').write_text('{"entries":[],"fits":{}}')
            (root/'reports/complete.json').write_text('{}')
            bindings = {p: runner.digest(root/p) for p in ['config/study.json','manifests/learning-plan.json',
                                                         'reports/complete.json','reports/learning-curve.json']}
            failure = dict(stage='cache',status='failed',returncode=7,log='logs/stages/synthetic-failure.log',
                           output_tail='synthetic GPU-error fixture')
            verified = dict(stage='verify',status='passed',returncode=0,log='logs/stages/synthetic-verify.log',
                            binding_before=bindings,binding_after=bindings,finished_utc='synthetic timestamp')
            (root/'logs/stages/failure.json').write_text(json.dumps(failure))
            (root/'logs/stages/verify.json').write_text(json.dumps(verified))
            def inspect():
                namespace = {}; output = io.StringIO()
                with patch.dict(os.environ, {'FI_RUN_ROOT':str(root),'GAVD6_ROOT':str(ROOT)}), \
                     patch('IPython.display.display'), redirect_stdout(output):
                    for cell in runner.load_source('23')[1].cells:
                        if cell.cell_type == 'code': exec(cell.source, namespace)
                return output.getvalue()
            text = inspect()
            self.assertIn('Saved completed preparation', text)
            self.assertIn('Unavailable development recordings excluded before processing: 1', text)
            self.assertIn('synthetic GPU-error fixture', text)
            self.assertIn('Saved numerical-verification job succeeded', text)
            before = runner.input_snapshot(root)
            with (root/'reports/learning-curve.json').open('a') as f: f.write(' ')
            self.assertIn('No successful numerical-verification log matches', inspect())
            self.assertNotEqual(runner.input_snapshot(root), before)
            before = runner.input_snapshot(root)
            (root/'logs/stages/failure.json').write_text(json.dumps({**failure,'returncode':9}))
            self.assertNotEqual(runner.input_snapshot(root), before)

    def test_duplicate_writer_rejected_and_completed_batches_not_overwritten(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)/'run'; fixture(root)
            output=root/'notebook_runs/batch'
            def client(notebook,**kwargs):
                result=fake_success(notebook,**kwargs)
                # MagicMock(execute=...) holds an ordinary callable.
                execute=result.execute
                def with_collision(**options):
                    with self.assertRaisesRegex(ValueError,'Another job'):
                        runner.execute_notebooks(root,output_dir=output)
                    return execute(**options)
                return MagicMock(execute=with_collision)
            with patch.object(runner,'KernelManager',return_value=MagicMock(has_kernel=False)), patch.object(runner,'NotebookClient',side_effect=client):
                runner.execute_notebooks(root,output_dir=output)
            before=runner.digest(output/runner.NOTEBOOKS['23'])
            with self.assertRaises(FileExistsError): runner.execute_notebooks(root,output_dir=output)
            self.assertEqual(before,runner.digest(output/runner.NOTEBOOKS['23']))

    def test_changed_inputs_and_unexecuted_cells_cannot_get_passed_status(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)/'run'; fixture(root)
            def client(notebook,**kwargs):
                result=fake_success(notebook,**kwargs)
                execute=result.execute
                def change(**options):
                    execute(**options)
                    (root/'reports/learning-curve.svg').write_text('<svg/>')
                return MagicMock(execute=change)
            with patch.object(runner,'KernelManager',return_value=MagicMock(has_kernel=False)), patch.object(runner,'NotebookClient',side_effect=client):
                with self.assertRaisesRegex(ValueError,'changed during'):
                    runner.execute_notebooks(root)
            saved=nbformat.read(next((root/'notebook_runs').glob('*/*.ipynb')),as_version=4)
            self.assertEqual(saved.metadata.fi_execution.status,'failed')
            self.assertFalse(saved.metadata.fi_execution.input_snapshot_unchanged)
            with patch.object(runner,'KernelManager',return_value=MagicMock(has_kernel=False)), patch.object(runner,'NotebookClient',return_value=MagicMock()):
                with self.assertRaisesRegex(RuntimeError,'every code cell'):
                    runner.execute_notebooks(root)

    def test_notebook_only_submission_needs_no_parent_models_or_media(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)/'run'; fixture(root)
            environment={k:v for k,v in os.environ.items() if not k.startswith(('FI_','GAVD','VJEPA'))}
            environment.update(GAVD6_ROOT=str(ROOT),FI_RUN_ROOT=str(root),FI_PYTHON=sys.executable)
            result=subprocess.run(['bash',str(LAUNCH/'submit.sh'),'notebooks','--dry-run'],
                                  env=environment,capture_output=True,text=True)
            self.assertEqual(result.returncode,0,result.stderr)
            commands=[line for line in result.stdout.splitlines() if line.startswith('sbatch ')]
            self.assertEqual(len(commands),1)
            self.assertIn('24-notebooks.sbatch',commands[0])
            self.assertNotIn('--dependency=',commands[0])
            self.assertFalse((root/'logs').exists())

    def test_sbatch_uses_selected_environment_and_propagates_executor_failure(self):
        with tempfile.TemporaryDirectory() as tmp:
            directory=Path(tmp); root=directory/'run'; fixture(root)
            interpreter=directory/'environment/bin/python'
            interpreter.parent.mkdir(parents=True)
            interpreter.write_text(f'#!{sys.executable}\n' + '''import json,os,sys
from pathlib import Path
Path(os.environ['TEST_NOTEBOOK_ARGS']).write_text(json.dumps(sys.argv[1:]))
raise SystemExit(int(os.environ.get('TEST_NOTEBOOK_EXIT','0')))
''')
            interpreter.chmod(0o755)
            environment={k:v for k,v in os.environ.items() if not k.startswith(('FI_','GAVD','VJEPA'))}
            environment.update(GAVD6_ROOT=str(ROOT),FI_RUN_ROOT=str(root),
                               FI_ENVIRONMENT=str(interpreter.parent.parent),
                               TEST_NOTEBOOK_ARGS=str(directory/'args.json'))
            for exit_code in (0,7):
                result=subprocess.run(['bash',str(LAUNCH/'24-notebooks.sbatch')],
                                      env={**environment,'TEST_NOTEBOOK_EXIT':str(exit_code)},capture_output=True,text=True)
                self.assertEqual(result.returncode,exit_code,result.stderr)
                self.assertEqual(json.loads((directory/'args.json').read_text()),
                                 ['slurm/future-innovation-scaling/launch/notebooks.py','--run-root',str(root)])


if __name__=='__main__':
    unittest.main()
