import concurrent.futures
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import json

from gavd6_sjepa.research_directions.gait_fidelity.scheduler import reserve,parse_accounting,update_attempt,preparation_phase_ids
from gavd6_sjepa.research_directions.gait_fidelity.spec import build_plan


class SchedulerTests(unittest.TestCase):
    def test_full_matrix_dependency_counts(self):
        p=build_plan()
        self.assertEqual(p['counts'],dict(recipes=34,final_fits=102,pretraining_phases=33,optimization_phases=135))
        seen={'prepare'}
        for phase in p['phases']:
            self.assertTrue(set(phase['depends_on'])<=seen)
            self.assertNotIn(phase['phase_id'],seen)
            seen.add(phase['phase_id'])

    def test_core_protocol_retains_primary_comparison_and_attribution_controls(self):
        p=build_plan(experiment_set='core')
        self.assertEqual(p['counts'],dict(recipes=10,final_fits=30,pretraining_phases=9,optimization_phases=39))
        self.assertEqual({r['encoder'] for r in p['recipes']},
                         {'coordinate','paired_jepa','direct','initialized','shuffled_jepa'})
        self.assertEqual({r['readout_or_training_objective'] for r in p['recipes']}, {'base','paired_change'})
        with self.assertRaises(ValueError):
            build_plan(experiment_set='best_after_evaluation')

    def test_shards_are_not_limited_to_gpu_count(self):
        self.assertEqual(preparation_phase_ids({'fixture':True,'data':{}}),['prepare'])
        keys=preparation_phase_ids({'fixture':False,'data':{'num_shards':128}})
        self.assertEqual(len(keys),128)
        self.assertEqual(len(set(keys)),128)
        self.assertEqual(keys[127],'prepare-shard-00127')
        with self.assertRaises(ValueError):
            preparation_phase_ids({'fixture':False,'data':{'num_shards':0}})

    def test_evaluation_receipt_detects_baseline_or_calibration_tampering(self):
        from gavd6_sjepa.research_directions.gait_fidelity.evaluation import retain_evaluation_receipt
        from gavd6_sjepa.research_directions.gait_fidelity.cli import verify_study
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);(root/'evaluation').mkdir()
            (root/'config.json').write_text('{}')
            (root/'ledger.json').write_text(json.dumps({'completed':{}}))
            (root/'report.md').write_text('Development evidence')
            (root/'evaluation/summary.json').write_text('{}')
            baseline=root/'evaluation/joint_affine.npy';baseline.write_bytes(b'original')
            calibration=root/'evaluation/calibration.json';calibration.write_text('{}')
            retain_evaluation_receipt({'work':folder})
            baseline.write_bytes(b'changed')
            with patch('gavd6_sjepa.research_directions.gait_fidelity.scheduler._verify_frozen'):
                with self.assertRaisesRegex(RuntimeError,'Artifact changed:.*joint_affine'):
                    verify_study({'work':folder})
            baseline.write_bytes(b'original');calibration.write_text('{"ridge":99}')
            with patch('gavd6_sjepa.research_directions.gait_fidelity.scheduler._verify_frozen'):
                with self.assertRaisesRegex(RuntimeError,'Artifact changed:.*calibration'):
                    verify_study({'work':folder})

    def test_atomic_budget_with_concurrent_workers(self):
        with tempfile.TemporaryDirectory() as folder:
            def attempt(index):
                try:
                    reserve(folder,str(index),1.,limit=3.,max_jobs=8)
                    return True
                except RuntimeError:
                    return False
            with concurrent.futures.ThreadPoolExecutor(8) as pool:
                accepted=list(pool.map(attempt,range(8)))
            self.assertEqual(sum(accepted),3)

    def test_scheduler_parent_count_only(self):
        text='10|COMPLETED|0:0|3600|cpu=4,gres/gpu=1,gres/gpu:h100=1\n10.batch|COMPLETED|0:0|3600|gres/gpu=1\n'
        self.assertEqual(parse_accounting(text,'10')['allocated_gpu_hours'],1.)
        with self.assertRaises(ValueError):
            parse_accounting('10|COMPLETED|0:0|60|cpu=4','10')

    def test_unresolved_submission_keeps_budget_and_prevents_duplicate(self):
        with tempfile.TemporaryDirectory() as folder:
            row=reserve(folder,'fit',2.,limit=3.,max_jobs=8)
            with self.assertRaises(RuntimeError):
                reserve(folder,'fit',1.,limit=3.,max_jobs=8)
            with self.assertRaises(RuntimeError):
                reserve(folder,'other',2.,limit=3.,max_jobs=8)
            update_attempt(folder,row['id'],dict(status='failed',allocated_gpu_hours=.5))
            reserve(folder,'fit',2.,limit=3.,max_jobs=8)


if __name__=='__main__':
    unittest.main()
