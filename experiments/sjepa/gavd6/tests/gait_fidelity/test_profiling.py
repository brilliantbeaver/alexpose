import unittest
from pathlib import Path
import tempfile
from unittest.mock import patch

from gavd6_sjepa.research_directions.gait_fidelity.common import atomic_json
from gavd6_sjepa.research_directions.gait_fidelity.profiling import profile_training,select_budget
from gavd6_sjepa.research_directions.gait_fidelity.spec import build_plan


class ProfilingTests(unittest.TestCase):
    def test_worker_and_artifact_overhead_recur_without_update_scaling(self):
        with tempfile.TemporaryDirectory() as folder:
            work=Path(folder)
            atomic_json(work/'plan.json',build_plan([17],experiment_set='core'))
            cfg=dict(work=str(work),experiment_set='core',seeds=[17],
                     training=dict(pretraining_updates=2000,readout_updates=2000,end_to_end_updates=4000),
                     resources=dict(gpu_hours=360,phase_wall_minutes=120))
            clock=[0.]
            def train(bundle,recipe,phase,seed,local,output,parent):
                output.mkdir()
                checkpoint=output/'checkpoint.pt';checkpoint.write_bytes(b'profile checkpoint')
                clock[0]+=100.
                return dict(checkpoint=str(checkpoint),elapsed_seconds=20.,updates=40 if phase=='end_to_end' else 20)
            def hash_file(path):
                self.assertTrue(path.is_file())
                clock[0]+=2.
                return 'test-artifact-hash'
            with patch('gavd6_sjepa.research_directions.gait_fidelity.training.train_phase',side_effect=train), \
                 patch('gavd6_sjepa.research_directions.gait_fidelity.profiling.time.monotonic',side_effect=lambda:clock[0]), \
                 patch('gavd6_sjepa.research_directions.gait_fidelity.profiling.sha256',side_effect=hash_file):
                result=profile_training(None,cfg,work/'profile',3.,worker_setup_seconds=30.)['budget']
            # Four 100-second phases, their own 2-second hashes, then the
            # complete 8-second profile tree verification. Setup is charged
            # once here; its doubled allowance applies only to future fits.
            self.assertEqual(result['measured_profile_seconds'],416.)
            self.assertEqual(result['measured_profile_gpu_hours'],446./3600)
            self.assertAlmostEqual(result['charged_before_profile'],3.+446./3600)
            for timing in result['timings_seconds'].values():
                self.assertEqual(timing['fixed_seconds'],80.+2.+60.+8.)
            self.assertEqual(result['selected']['projected_phase_seconds']['readout'],2150.)
            self.assertEqual(result['alternatives'][1]['projected_phase_seconds']['readout'],1150.)

    def test_cohort_hashing_and_export_overhead_is_charged_once(self):
        cfg=dict(training=dict(pretraining_updates=2000,readout_updates=2000,end_to_end_updates=4000),
                 experiment_set='core',seeds=[17],resources=dict(gpu_hours=360,phase_wall_minutes=120))
        times={k:dict(optimization_seconds=20.,fixed_seconds=600.,wall_seconds=620.,profile_updates=20 if k!='direct' else 40)
               for k in ('coordinate_pretrain','jepa_pretrain','readout','direct')}
        selected=select_budget(cfg,times,0.)['selected']
        self.assertIsNotNone(selected)
        self.assertEqual(selected['projected_phase_seconds']['readout'],2600.)

    def test_selects_one_common_schedule_from_time_only(self):
        cfg=dict(training=dict(pretraining_updates=2000,readout_updates=2000,end_to_end_updates=4000),
                 resources=dict(gpu_hours=360,phase_wall_minutes=120))
        times={k:1. for k in ('coordinate_pretrain','jepa_pretrain','readout','direct','refiner')}
        result=select_budget(cfg,times,10.)
        self.assertEqual(result['selected']['divisor'],1)
        times={k:90. for k in times}
        result=select_budget(cfg,times,10.)
        self.assertEqual(result['selected']['divisor'],2)
        self.assertEqual(result['selected']['updates']['readout_updates'],1000)
        times={k:1000. for k in times}
        self.assertIsNone(select_budget(cfg,times,10.)['selected'])


if __name__=='__main__':unittest.main()
