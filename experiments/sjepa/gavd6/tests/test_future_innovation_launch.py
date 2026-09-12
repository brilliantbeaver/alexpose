"""README-style launcher: real shell submission routing and frozen-init recovery."""
import importlib.util
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
LAUNCH = ROOT / 'slurm/future-innovation-scaling/launch'
spec = importlib.util.spec_from_file_location('source_curve_launch', LAUNCH / 'initialize.py')
launch = importlib.util.module_from_spec(spec)
spec.loader.exec_module(launch)


def fixture(directory):
    parent = directory / 'gates/gate-v2'
    (parent / 'config').mkdir(parents=True)
    (parent / 'manifests').mkdir()
    (parent / 'config/run-contract.json').write_text('{"protocol":"direct-v2"}')
    (parent / 'manifests/gate-windows.csv').write_text('video_id\nparent-source\n')
    full = directory / 'full'
    (full / 'manifests').mkdir(parents=True)
    (full / 'manifests/gavd_full_sequences.csv').write_text('sequence_id,video_id\nq,parent-source\n')
    (full / 'manifests/gavd_full_videos.csv').write_text('video_id\nparent-source\n')
    annotations = full / 'annotations/GAVD/data'
    annotations.mkdir(parents=True)
    for i in range(1, 6):
        (annotations / f'GAVD_Clinical_Annotations_{i}.csv').touch()
    (full / 'youtube/all').mkdir(parents=True)
    teacher = directory / 'teacher'
    teacher.mkdir()
    pose, checkpoint = directory / 'pose.task', directory / 'teacher.pt'
    pose.touch(); checkpoint.touch()
    return dict(GAVD6_ROOT=str(ROOT), FI_RUN_ROOT=str(directory / 'new study'),
                FI_PARENT_ROOT=str(parent), GAVD_FULL_ROOT=str(full),
                FI_POSE_MODEL=str(pose), FI_TEACHER_CHECKPOINT=str(checkpoint),
                VJEPA2_ROOT=str(teacher), FI_PYTHON=sys.executable)


class LaunchTests(unittest.TestCase):
    def test_relative_roots_and_parent_protection(self):
        cfg = launch.settings({'FI_RUN_ROOT':'outputs/new-run','FI_PARENT_ROOT':'outputs/gate-v2'})
        self.assertEqual(cfg['root'], ROOT / 'outputs/new-run')
        for name in ('outputs/gate-v2', 'outputs/gate-v2/nested', 'outputs'):
            with self.assertRaisesRegex(ValueError, 'separate'):
                launch.settings({'FI_RUN_ROOT':name,'FI_PARENT_ROOT':'outputs/gate-v2'})

    def test_known_exposures_and_both_gates_are_collected(self):
        with tempfile.TemporaryDirectory() as tmp:
            env = fixture(Path(tmp)); cfg = launch.settings(env)
            sibling = cfg['parent'].parent / 'gate-v1/manifests'
            sibling.mkdir(parents=True)
            (sibling / 'gate-windows.csv').write_text('video_id\nolder-source\n')
            extra = Path(tmp) / 'extra.csv'
            extra.write_text('video_id\nextra-source\n')
            env['FI_INSPECTED_MANIFESTS'] = str(extra)
            plan = launch.input_plan(cfg, env)
            known = launch.source_ids(LAUNCH / 'known-exposure.csv')
            self.assertEqual(len(known), 128)
            self.assertEqual(set(plan['source_ids']), known | {'parent-source','older-source','extra-source'})
            self.assertFalse(cfg['root'].exists())

    def test_missing_or_changed_exposure_evidence_is_not_silently_omitted(self):
        with tempfile.TemporaryDirectory() as tmp:
            env = fixture(Path(tmp)); cfg = launch.settings(env)
            sibling = cfg['parent'].parent / 'gate-v1/config'
            sibling.mkdir(parents=True)
            (sibling / 'cohort-contract.json').write_text('{}')
            with self.assertRaisesRegex(FileNotFoundError, 'Historical gate'):
                launch.input_plan(cfg, env)
            (sibling / 'cohort-contract.json').unlink()
            copied = Path(tmp) / 'launcher'
            shutil.copytree(LAUNCH, copied)
            with (copied / 'known-exposure.csv').open('a') as f:
                f.write('invented,none\n')
            with self.assertRaisesRegex(ValueError, 'checksum'):
                launch.input_plan({**cfg,'launcher':copied}, env)

    def test_missing_input_reports_exact_path_before_creating_run(self):
        with tempfile.TemporaryDirectory() as tmp:
            env = fixture(Path(tmp)); cfg = launch.settings(env)
            path = cfg['full'] / 'annotations/GAVD/data/GAVD_Clinical_Annotations_3.csv'
            path.unlink()
            with self.assertRaisesRegex(FileNotFoundError, 'Annotations_3.csv'):
                launch.preflight(cfg, env, 'all')
            self.assertFalse(cfg['root'].exists())

    def test_refuses_gate_v1_as_parent_and_existing_gate_as_output(self):
        with tempfile.TemporaryDirectory() as tmp:
            env = fixture(Path(tmp)); cfg = launch.settings(env)
            (cfg['parent'] / 'config/run-contract.json').write_text('{"protocol":"legacy-v1"}')
            with self.assertRaisesRegex(ValueError, 'direct-v2'):
                launch.preflight(cfg, env, 'all')
            (cfg['root'] / 'config').mkdir(parents=True)
            with self.assertRaisesRegex(ValueError, 'another experiment'):
                launch.initialize(cfg, env)

    def test_dry_run_accepts_new_root_and_exports_old_jobs_alias(self):
        with tempfile.TemporaryDirectory() as tmp:
            env = fixture(Path(tmp))
            # Simulate stale variables from the earlier instructions.
            env['FI_SCALING_ROOT'] = '/unused/old-root'
            for mode, count in [('all',6),('prepare',2)]:
                result = subprocess.run(['bash',str(LAUNCH/'submit.sh'),mode,'--dry-run'],
                                        env={**os.environ,**env},capture_output=True,text=True)
                self.assertEqual(result.returncode,0,result.stderr)
                commands = [line for line in result.stdout.splitlines() if line.startswith('sbatch ')]
                self.assertEqual(len(commands),count)
                self.assertIn('19-initialize.sbatch',commands[0])
                self.assertEqual(sum('--dependency=afterok:' in c for c in commands),4 if mode=='all' else 1)
                if mode=='all':
                    self.assertIn('--dependency=afterany:DRY_initialize:DRY_pose:DRY_cache:DRY_fit:DRY_report',commands[-1])
                self.assertNotIn('/unused/old-root',result.stdout)
                self.assertFalse(Path(env['FI_RUN_ROOT']).exists())

    def test_actual_submission_chain_with_scheduler_standin(self):
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp); env = fixture(directory)
            binary = directory/'bin'; binary.mkdir()
            scheduler = binary/'sbatch'
            scheduler.write_text(f'#!{sys.executable}\n' + '''import json,os,sys
from pathlib import Path
p=Path(os.environ['TEST_SUBMISSIONS'])
rows=json.loads(p.read_text()) if p.exists() else []
rows.append({'args':sys.argv[1:],'run':os.environ['FI_RUN_ROOT'],
             'alias':os.environ['FI_SCALING_ROOT'],'video':os.environ['FI_VIDEO_ROOT']})
p.write_text(json.dumps(rows))
print(800+len(rows))
''')
            scheduler.chmod(0o755)
            env.update(PATH=str(binary)+os.pathsep+os.environ['PATH'],
                       TEST_SUBMISSIONS=str(directory/'submissions.json'))
            result = subprocess.run(['bash',str(LAUNCH/'submit.sh'),'all'],
                                    env={**os.environ,**env},capture_output=True,text=True)
            self.assertEqual(result.returncode,0,result.stderr)
            rows = json.loads((directory/'submissions.json').read_text())
            self.assertEqual(len(rows),6)
            for i,row in enumerate(rows):
                self.assertEqual(row['run'],row['alias'])
                self.assertEqual(row['video'],str(Path(env['GAVD_FULL_ROOT'])/'youtube/all'))
                deps = [arg for arg in row['args'] if arg.startswith('--dependency=')]
                expected=[] if i==0 else ([f'--dependency=afterany:801:802:803:804:805'] if i==5 else [f'--dependency=afterok:{800+i}'])
                self.assertEqual(deps,expected)
            self.assertEqual(len((Path(env['FI_RUN_ROOT'])/'logs/submissions.tsv').read_text().splitlines()),6)

    def test_partial_calibration_starts_a_new_attempt_and_complete_reuses(self):
        with tempfile.TemporaryDirectory() as tmp:
            env = fixture(Path(tmp)); cfg = launch.settings(env)
            (cfg['root']/'launch/calibration/attempt-001').mkdir(parents=True)
            calls=[]
            def runner(command, **kwargs):
                calls.append(command)
                dest=Path(command[-1]); dest.mkdir(parents=True)
                launch.write_json(dest/'calibration.json',dict(status='synthetic_calibration_only',
                                  passed=True,software=launch.fingerprint()))
            result = launch.calibrate(cfg,runner)
            self.assertEqual(result.parent.name,'attempt-002')
            self.assertEqual(launch.calibrate(cfg,runner),result)
            self.assertEqual(len(calls),1)
            launch.write_json(result,dict(passed=False,software=launch.fingerprint()))
            with self.assertRaisesRegex(ValueError,'Calibration failed'):
                launch.calibrate(cfg,runner)

    def test_initialization_lock_and_changed_captured_inputs(self):
        with tempfile.TemporaryDirectory() as tmp:
            env = fixture(Path(tmp)); cfg = launch.settings(env)
            record = launch.capture_inputs(cfg,env)
            with launch.stage_lock(cfg['root'],'scaling-initialize'):
                with self.assertRaises(ValueError):
                    launch.initialize(cfg,env)
            path=cfg['root']/'launch/inputs/inspected-recordings.csv'
            path.write_text('video_id\nchanged\n')
            with self.assertRaisesRegex(ValueError,'input changed'):
                launch.capture_inputs(cfg,env)


class RetainedStudyIntegrationTests(unittest.TestCase):
    def test_launcher_does_not_change_any_preexisting_fingerprinted_file(self):
        for relative,digest in launch.fingerprint().items():
            if relative == 'historical_future_innovation':
                continue
            saved=subprocess.run(['git','show',f'HEAD:./{relative}'],cwd=ROOT,capture_output=True)
            self.assertEqual(saved.returncode,0,saved.stderr)
            self.assertEqual(digest,hashlib.sha256(saved.stdout).hexdigest(),relative)

    def test_real_parent_freeze_publish_recovery_and_exact_reservation(self):
        parent=ROOT/'outputs/future-innovation'
        historical=ROOT/'outputs/future-innovation-source-curve-dev-20260911-v2'
        if not (historical/'config/calibration.json').exists():
            self.skipTest('Retained local calibration is unavailable')
        calibration=Path(os.environ.get('FI_LAUNCH_TEST_CALIBRATION',historical/'config/calibration.json'))
        if launch.read_json(calibration)['software'] != launch.fingerprint():
            self.skipTest('Supply FI_LAUNCH_TEST_CALIBRATION with a passing calibration for the current checkout')
        from gavd6_sjepa.research_directions.future_innovation.fi_cache_reuse import snapshot,verify_snapshot
        before=snapshot(parent)
        with tempfile.TemporaryDirectory() as tmp:
            env=dict(GAVD6_ROOT=str(ROOT), FI_RUN_ROOT=str(Path(tmp)/'source-curve'),
                     FI_PARENT_ROOT=str(parent),GAVD_FULL_ROOT=str(Path(tmp)/'full'))
            cfg=launch.settings(env)
            attempt=cfg['root']/'launch/calibration/attempt-001'
            attempt.mkdir(parents=True)
            shutil.copyfile(calibration,attempt/'calibration.json')
            def interrupted_publish(config,staging):
                shutil.copytree(staging/'config',config['root']/'config')
                raise RuntimeError('injected crash after config publish')
            def no_calibration(*args,**kwargs):
                self.fail('Completed matching calibration should be reused')
            with patch.object(launch,'publish',side_effect=interrupted_publish):
                with self.assertRaisesRegex(RuntimeError,'injected crash'):
                    launch.initialize(cfg,env,no_calibration)
            launch.initialize(cfg,env,no_calibration)
            launch.check_study(cfg)
            expected=pd.read_csv(historical/'config/source-reservation.csv')
            actual=pd.read_csv(cfg['root']/'config/source-reservation.csv')
            pd.testing.assert_frame_equal(actual,expected)
            self.assertEqual(launch.read_json(cfg['root']/'config/study.json')['run_id'],'source-curve')
            files_before=snapshot(cfg['root']/'config')
            launch.initialize(cfg,env,no_calibration)
            verify_snapshot(cfg['root']/'config',files_before)
            verify_snapshot(parent,before)


if __name__ == '__main__':
    unittest.main()
