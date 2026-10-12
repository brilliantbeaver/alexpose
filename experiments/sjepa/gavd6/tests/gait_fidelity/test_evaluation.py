import unittest
import numpy as np
import pandas as pd

from gavd6_sjepa.research_directions.gait_fidelity.evaluation import angular_waveform,crossed_interval,response_contrasts,measurement_rows
from gavd6_sjepa.research_directions.gait_fidelity.measurements import knee_angles
from gavd6_sjepa.research_directions.gait_fidelity.data import fixture_bundle


class EvaluationTests(unittest.TestCase):
    def test_independent_angle_implementations_agree(self):
        x=np.random.default_rng(42).normal(size=(4,64,12,2))*100
        a,valid=angular_waveform(x)
        b,other=knee_angles(x,min_segment_px=2.)
        np.testing.assert_allclose(a,b,atol=1e-8,equal_nan=True)
        np.testing.assert_array_equal(valid,other)

    def test_missing_prediction_cannot_improve_support(self):
        bundle=fixture_bundle(samples=32)
        cfg={'measurement':dict(min_segment_px=2,min_frames=16,min_coverage=.8,sign_tolerance_deg=1)}
        pred=bundle.targets['xy'].copy()
        table=measurement_rows(bundle,pred,cfg,'oracle',17)
        self.assertEqual(table.A_error.max(),0.)
        pred[0,0,8]=np.nan
        failed=measurement_rows(bundle,pred,cfg,'missing',17)
        self.assertEqual(failed.iloc[0].support_frames,32)
        self.assertEqual(failed.iloc[0].A_error,360.)
        self.assertFalse(failed.iloc[0].prediction_success)
        responses=response_contrasts(table)
        self.assertEqual(responses.response_error.max(),0.)
        self.assertIn('no_change',responses.movement_state.values)

    def test_crossed_factor_bootstrap_retains_seed_effect(self):
        rows=[]
        for p in range(8):
            for seed in range(3):
                for method,error in [('candidate',1.),('control',2.+seed)]:
                    rows.append(dict(canonical_person_id=str(p),seed=seed,method=method,error=error))
        result=crossed_interval(pd.DataFrame(rows),'candidate','control','error',draws=300)
        self.assertEqual(result['improvement'],2.)
        self.assertEqual(result['person_conditional_ci95'],[2.,2.])
        self.assertLess(result['crossed_person_seed_ci95'][0],2.)
        self.assertGreater(result['crossed_person_seed_ci95'][1],2.)


if __name__=='__main__':
    unittest.main()
