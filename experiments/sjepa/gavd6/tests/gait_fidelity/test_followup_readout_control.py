"""Protect the primary contrast while adding the core-motivated readout control."""
from datetime import datetime, timezone
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

from tests.gait_fidelity.test_followup import retained_parent
from tests.gait_fidelity.test_response_evaluation import configuration
from gavd6_sjepa.research_directions.gait_fidelity.common import read_json, sha256
from gavd6_sjepa.research_directions.gait_fidelity.config import load_config
from gavd6_sjepa.research_directions.gait_fidelity.data import fixture_bundle
from gavd6_sjepa.research_directions.gait_fidelity.followup import initialize_followup, admit_matrix, response_timing_key
from gavd6_sjepa.research_directions.gait_fidelity.spec import build_response_plan
from gavd6_sjepa.research_directions.gait_fidelity.response_evaluation import (
    declared_response_recipes, summarize_response_predictions, readout_control_comparisons)


class ReadoutControlTests(unittest.TestCase):
    def test_extended_plan_shares_pretraining_and_retains_original_phases(self):
        original = build_response_plan()
        extended = build_response_plan(include_base_readouts=True)
        self.assertEqual(extended['phases'][:18], original['phases'])
        self.assertEqual(extended['counts'], dict(recipes=6, final_fits=18, pretraining_phases=9, optimization_phases=27))
        pre = {p['phase_id'] for p in extended['phases'] if p['phase']=='pretrain'}
        for p in extended['phases'][18:]:
            self.assertEqual(p['recipe']['readout_or_training_objective'],'base')
            self.assertIn(p['depends_on'][0],pre)
        self.assertNotEqual(extended['identity'], original['identity'])

    def test_custom_deadline_and_both_parent_readouts_survive_config_validation(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);parent,_=retained_parent(root)
            before={str(p):sha256(p) for p in parent.rglob('*') if p.is_file()}
            child=root/'readout-child'
            cfg=initialize_followup(child,parent_work=parent,include_base_readouts=True,
                                    deadline_utc='2026-09-25T08:00:00-07:00')
            self.assertEqual(cfg,load_config(child))
            self.assertEqual(cfg['followup']['deadline_utc'],'2026-09-25T15:00:00Z')
            self.assertEqual(cfg['resources']['gpu_hours'],0.)
            self.assertEqual(cfg['followup']['total_gpu_hours'],48.)
            self.assertEqual(len(cfg['followup']['parent_binding']['baseline_phases']),30)
            self.assertTrue(cfg['evaluation']['primary_candidate'].endswith('paired_change'))
            self.assertEqual(before,{str(p):sha256(p) for p in parent.rglob('*') if p.is_file()})
            self.assertEqual(initialize_followup(child,parent_work=parent),cfg)
            with self.assertRaisesRegex(ValueError,'deadline cannot be changed'):
                initialize_followup(child,parent_work=parent,deadline_utc='2026-09-26T15:00:00Z')
            with self.assertRaisesRegex(ValueError,'matrix cannot be changed'):
                initialize_followup(child,parent_work=parent,include_base_readouts=False)

    def test_admission_requires_all_base_timings_and_reserves_evaluation(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);parent,_=retained_parent(root)
            cfg=initialize_followup(root/'child',parent_work=parent,include_base_readouts=True,
                                    deadline_utc='2026-09-25T15:00:00Z')
            cfg['fixture']=False
            phases=build_response_plan(include_base_readouts=True)['phases']
            timings={response_timing_key(p['recipe'],p['phase']):dict(fixed_seconds=1,optimization_seconds=1,profile_updates=1)
                     for p in phases}
            admission=admit_matrix(cfg,timings,now=datetime(2026,9,24,18,tzinfo=timezone.utc))
            self.assertTrue(admission['admitted'])
            self.assertEqual(len(admission['predicted_phase_seconds']),27)
            self.assertGreater(admission['required_wall_hours'],14.)
            self.assertFalse(admit_matrix(cfg,timings,now=datetime(2026,9,25,2,tzinfo=timezone.utc))['admitted'])
            timings.pop('jepa_delta_v1:readout:base')
            with self.assertRaises(KeyError):admit_matrix(cfg,timings)

    def test_missing_or_relabelled_readout_cannot_complete(self):
        plan=build_response_plan(include_base_readouts=True)
        final=[p for p in plan['phases'] if p['phase']=='readout']
        cfg=dict(followup=dict(include_base_readouts=True))
        names=declared_response_recipes(final,cfg)
        self.assertTrue(names['jepa_delta_v1','paired_change'].endswith('paired_change'))
        self.assertTrue(names['jepa_delta_v1','base'].endswith('-base'))
        with self.assertRaises(ValueError):declared_response_recipes(final[:-1],cfg)
        with self.assertRaises(ValueError):declared_response_recipes(final,dict(followup={}))

    def test_failure_decomposition_keeps_the_all_attempted_denominator(self):
        bundle=fixture_bundle(samples=32,people=5,windows=2).subset('development')
        truth=np.asarray(bundle.targets['xy']);failed=truth.copy()
        changed=next(i for i,r in enumerate(bundle.records) if r['movement_level_deg']==10)
        failed[changed,0,8]=np.nan
        cfg=configuration();cfg.update(fixture=True,followup=dict(include_base_readouts=True))
        exports=[(m,s,failed if m=='candidate' else truth) for m in ('candidate','control') for s in (17,29,43)]
        with tempfile.TemporaryDirectory() as temp, patch(
                'gavd6_sjepa.research_directions.gait_fidelity.response_evaluation.readout_control_comparisons',return_value={}):
            people,_,_=summarize_response_predictions(bundle,iter(exports),cfg,temp)
            np.testing.assert_allclose(people.response_error,
                people.response_failure_contribution+people.response_success_contribution,rtol=0,atol=1e-12)
            np.testing.assert_allclose(people.response_failure_contribution,720*people.response_failure_rate)
            self.assertGreater(people.response_failure_contribution.max(),0.)
            self.assertAlmostEqual(people.response_success_contribution.max(),0.)
            ref=people.pivot(index=['canonical_person_id','seed'],columns='method',values='zero_response_error')
            np.testing.assert_allclose(ref.candidate,ref.control)

    def test_secondary_interaction_preserves_sign_and_primary_objective(self):
        metrics=['response_error','nuisance_error','A_error','waveform_error','visible_nle','synthetic_all_nle',
                 'response_failure_contribution','response_success_contribution','zero_response_error']
        rows=[]
        values={('jepa_delta_v1','base'):8.,('jepa_delta_v1','paired_change'):9.,
                ('jepa_endpoint_v1','base'):10.,('jepa_endpoint_v1','paired_change'):14.,
                ('coordinate_delta_v1','base'):10.,('coordinate_delta_v1','paired_change'):11.}
        for (variant,objective),value in values.items():
            for person in ('a','b'):
                for seed in (17,29,43):
                    rows.append(dict(method=f'F-response-{variant}-graph_time-{objective}',canonical_person_id=person,
                                     seed=seed,**{m:value for m in metrics}))
        result=readout_control_comparisons(pd.DataFrame(rows),configuration())['comparisons']
        self.assertEqual(result['delta_vs_endpoint_base']['response_error']['improvement'],2.)
        self.assertEqual(result['coupling_by_readout_interaction']['response_error']['improvement'],3.)

    def test_deadline_stops_new_evaluation_but_allows_later_reconstruction(self):
        bundle=fixture_bundle(samples=32,people=5,windows=2).subset('development')
        cfg=configuration();cfg.update(fixture=False,followup=dict(deadline_utc='2000-01-01T00:00:00Z'))
        truth=np.asarray(bundle.targets['xy'])
        exports=[(m,s,truth) for m in ('candidate','control') for s in (17,29,43)]
        with tempfile.TemporaryDirectory() as temp:
            with self.assertRaisesRegex(RuntimeError,'deadline'):
                summarize_response_predictions(bundle,iter(exports),cfg,temp,enforce_deadline=True)
            people,_,_=summarize_response_predictions(bundle,iter(exports),cfg,temp)
            self.assertEqual(people.response_error.max(),0.)


if __name__=='__main__':unittest.main()
