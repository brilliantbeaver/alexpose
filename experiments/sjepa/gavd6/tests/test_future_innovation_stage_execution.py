"""Stage records are operational evidence, separate from scientific artifacts."""
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('source_curve_stage_runner', ROOT/'slurm/future-innovation-scaling/launch/run_stage.py')
runner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runner)


class StageExecutionTests(unittest.TestCase):
    def test_failure_and_retry_keep_distinct_logs_and_exit_codes(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for code in (7, 0):
                result = runner.run_stage(root, 'run', [sys.executable, '-c',
                    f'print("synthetic subprocess fixture"); raise SystemExit({code})'], fold=2)
                self.assertEqual(result, code)
            records = [json.loads(p.read_text()) for p in (root/'logs/stages').glob('*.json')]
            self.assertEqual(len(records), 2)
            self.assertEqual({r['status'] for r in records}, {'failed', 'passed'})
            for record in records:
                self.assertEqual(record['fold'], 2)
                self.assertIn('synthetic subprocess', (root/record['log']).read_text())
            self.assertFalse((root/'reports').exists())

    def test_startup_failure_retains_error(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self.assertEqual(runner.run_stage(root, 'prepare', ['/nonexistent/source-curve-test-executable']), 2)
            record = json.loads(next((root/'logs/stages').glob('*.json')).read_text())
            self.assertEqual(record['status'], 'failed')
            self.assertIn('FileNotFoundError', record['error'])

    def test_verification_record_requires_existing_unchanged_bindings(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            command = [sys.executable, '-c', 'print("synthetic verification command fixture")']
            self.assertEqual(runner.run_stage(root, 'verify', command), 2)
            for relative in runner.BINDINGS:
                p = root/relative; p.parent.mkdir(parents=True, exist_ok=True)
                p.write_text('{"protocol":"source-learning-curve-v1"}')
            before = runner.binding(root)
            self.assertEqual(runner.run_stage(root, 'verify', command), 0)
            command = [sys.executable, '-c', 'import os; from pathlib import Path; '
                       '(Path(os.environ["FI_RUN_ROOT"])/"reports/learning-curve.json").write_text("changed")']
            self.assertEqual(runner.run_stage(root, 'verify', command), 2)
            self.assertNotEqual(runner.binding(root), before)

    def test_actual_cli_failure_reaches_structured_log(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)/'missing study'
            result = subprocess.run([sys.executable, runner.__file__, 'run', '--run-root', str(root), '--fold', '3'],
                                    capture_output=True, text=True)
            self.assertNotEqual(result.returncode, 0)
            record = json.loads(next((root/'logs/stages').glob('*.json')).read_text())
            self.assertEqual(record['returncode'], result.returncode)
            self.assertEqual(record['status'], 'failed')
            self.assertIn('study.json', record['output_tail'])
            self.assertIn('--fold', record['command'])

    def test_historical_gate_and_escaped_logs_are_rejected_without_writes(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)/'gate'; (root/'config').mkdir(parents=True)
            (root/'config/run-contract.json').write_text('{"protocol":"direct-v2"}')
            with self.assertRaisesRegex(ValueError, 'historical gate'):
                runner.run_stage(root, 'run', [sys.executable, '-c', 'pass'])
            self.assertFalse((root/'logs').exists())
            root = Path(tmp)/'child'; root.mkdir()
            other = Path(tmp)/'parent'; other.mkdir()
            (root/'logs').symlink_to(other, target_is_directory=True)
            with self.assertRaisesRegex(ValueError, 'stay inside'):
                runner.run_stage(root, 'run', [sys.executable, '-c', 'pass'])
            self.assertEqual(list(other.iterdir()), [])

    def test_all_jobs_route_to_logged_stages_and_cache_failure_stops_audits(self):
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            interpreter = directory/'python-standin'
            interpreter.write_text(f'#!{sys.executable}\n' + '''import json,os,sys
from pathlib import Path
p=Path(os.environ['TEST_STAGE_COMMANDS'])
rows=json.loads(p.read_text()) if p.exists() else []
rows.append(sys.argv[1:]); p.write_text(json.dumps(rows))
raise SystemExit(int(os.environ.get('TEST_STAGE_EXIT','0')))
''')
            interpreter.chmod(0o755)
            output = directory/'commands.json'
            env = {**os.environ, 'GAVD6_ROOT':str(ROOT),'FI_RUN_ROOT':str(directory/'new run'),
                   'FI_SCALING_ROOT':str(directory/'new run'),'FI_PARENT_ROOT':str(directory/'gate-v2'),
                   'FI_PYTHON':str(interpreter),'FI_ANNOTATION_ROOT':str(directory/'annotations'),
                   'FI_VIDEO_ROOT':str(directory/'videos'),'FI_POSE_MODEL':str(directory/'pose.task'),
                   'VJEPA2_ROOT':str(directory/'teacher'),'FI_TEACHER_CHECKPOINT':str(directory/'teacher.pt'),
                   'SLURM_ARRAY_TASK_ID':'3','TEST_STAGE_COMMANDS':str(output), 'TEST_STAGE_EXIT':'0'}
            jobs = [('launch/19-initialize.sbatch',['initialize']), ('20-prepare-scaling.sbatch',['prepare']),
                    ('21-cache-scaling.sbatch',['cache','audit-teacher','plan']),
                    ('22-fit-scaling.sbatch',['run']), ('23-report-scaling.sbatch',['report','verify'])]
            for script, stages in jobs:
                result = subprocess.run(['bash',str(ROOT/'slurm/future-innovation-scaling'/script)],
                                        env=env,capture_output=True,text=True)
                self.assertEqual(result.returncode,0,result.stderr)
                commands = json.loads(output.read_text())
                self.assertEqual([c[1] for c in commands],stages)
                self.assertTrue(all(c[0].endswith('launch/run_stage.py') for c in commands))
                if stages==['run']: self.assertEqual(commands[0][-2:],['--fold','3'])
                output.unlink()
            result = subprocess.run(['bash',str(ROOT/'slurm/future-innovation-scaling/21-cache-scaling.sbatch')],
                                    env={**env,'TEST_STAGE_EXIT':'7'},capture_output=True,text=True)
            self.assertEqual(result.returncode,7)
            self.assertEqual(len(json.loads(output.read_text())),1)


if __name__ == '__main__':
    unittest.main()
