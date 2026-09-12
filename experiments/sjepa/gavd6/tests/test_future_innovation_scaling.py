"""Prospective invariants for source scaling, normalization and paired scoring."""
import copy
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

from gavd6_sjepa.research_directions.future_innovation_scaling.fi_scaling_cohort import (
    POLICY, reserve_sources, source_groups, make_plan)
from gavd6_sjepa.research_directions.future_innovation_scaling.fi_scaling_training import (
    compute_curve, prediction_payload, run, report, verify_fit)
from gavd6_sjepa.research_directions.future_innovation.fi_joint_calibration import fixture
from gavd6_sjepa.research_directions.future_innovation.fi_joint_models import JointRidge, JointModelContract, baseline_schema
from gavd6_sjepa.research_directions.future_innovation_scaling.fi_scaling_nested import nested_partition
from gavd6_sjepa.research_directions.future_innovation.fi_contracts import (
    DIRECT_ARMS, write_json, read_json, sha256_file, equal_source_weights, stage_lock)
from gavd6_sjepa.research_directions.future_innovation_scaling.fi_scaling_data import seal, verify_seal, verify_prepared_inputs, verify_data_identity


def inventory(n=100):
    return pd.DataFrame([dict(sequence_id=f'q{i}-{j}',video_id=f'v{i}',first_frame=1,last_frame=100,n_annotated_frames=100)
                         for i in range(n) for j in range(1+i%3)])


class ScalingPlanTests(unittest.TestCase):
    def test_reservation_excludes_prior_sources_and_all_their_clips(self):
        sequences=inventory(); exposed={'v0','v1','v2'}
        roster=reserve_sources(sequences,exposed)
        self.assertFalse(roster.loc[roster.role=='confirmation','previously_exposed'].any())
        self.assertEqual(sum(roster.role=='confirmation'),20)
        changed=sequences.sample(frac=1,random_state=19).assign(diagnosis='irrelevant')
        pd.testing.assert_frame_equal(roster,reserve_sources(changed,exposed))
        self.assertEqual(sum(roster.annotated_sequences),len(sequences))

    def test_participant_components_cross_no_boundaries(self):
        people=pd.DataFrame({'video_id':['v0','v1','v1','v2','v3'], 'participant_id':['a','a','b','b','']})
        groups=source_groups(inventory().video_id,people)
        self.assertEqual(groups['v0'],groups['v2'])
        self.assertNotEqual(groups['v3'],groups['v4'])
        roster=reserve_sources(inventory(),{'v0'},people).set_index('video_id')
        self.assertEqual(roster.loc[['v0','v1','v2'],'outer_fold'].nunique(),1)
        self.assertTrue((roster.loc[['v0','v1','v2'],'role']=='development').all())
        with self.assertRaises(ValueError): source_groups(['v0'],people)

    def test_nested_subsets_fixed_test_windows_and_deduplicated_all(self):
        sequences=inventory()
        roster=reserve_sources(sequences,set(sequences.video_id))
        cohort=sequences.rename(columns={'sequence_id':'window_id'})
        policy={**POLICY,'nominal_source_sizes':[12,24,1000]}
        plan=make_plan(cohort,roster,policy)
        for fold in range(5):
            previous=None; tests=[]
            for seed in policy['subset_seeds']:
                rows=[e for e in plan['entries'] if e['outer_fold']==fold and e['subset_seed']==seed and e['status']=='planned']
                trainsets=[]
                for row in rows:
                    fit=plan['fits'][row['fit_id']]
                    trainsets.append(set(fit['train_sources'])); tests.append(fit['test_windows'])
                    self.assertFalse(set(fit['train_groups'])&set(fit['test_groups']))
                    for source in fit['train_sources']:
                        self.assertEqual(set(cohort.loc[cohort.video_id==source,'window_id']),
                                         set(fit['train_windows'])&set(cohort.loc[cohort.video_id==source,'window_id']))
                self.assertTrue(trainsets[0]<trainsets[1]<=trainsets[2])
            self.assertTrue(all(t==tests[0] for t in tests))
            allids={e['fit_id'] for e in plan['entries'] if e['outer_fold']==fold and e['size']=='all'}
            self.assertEqual(len(allids),1)
        self.assertTrue(all(e['status']=='unavailable' for e in plan['entries'] if e['size']=='1000'))

    def test_confirmation_cannot_enter_learning_plan(self):
        sequences=inventory(); roster=reserve_sources(sequences,set())
        with self.assertRaises(ValueError): make_plan(sequences.rename(columns={'sequence_id':'window_id'}),roster)

    def test_historical_gate_still_rejects_expansion(self):
        from gavd6_sjepa.research_directions.future_innovation.fi_cohort import select_eligible
        with self.assertRaises(ValueError): select_eligible([],count=1800)

    def test_duplicate_writer_and_changed_receipt_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp); p=root/'value.txt'; p.write_text('original')
            seal(root,'receipt.json',[p])
            verify_seal(root,'receipt.json')
            p.write_text('changed')
            with self.assertRaises(ValueError): verify_seal(root,'receipt.json')
            with stage_lock(root,'same'):
                with self.assertRaises(ValueError):
                    with stage_lock(root,'same'): pass

    def test_nested_pose_receipt_does_not_hide_changed_decoded_frames(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp); (root/'data/frames').mkdir(parents=True); (root/'data/poses').mkdir()
            frame=root/'data/frames/window.npz'; frame.write_bytes(b'original')
            receipt=root/'data/poses/window.json'
            seal(root,str(receipt.relative_to(root)),[frame])
            seal(root,'data/cohort-complete.json',[receipt])
            verify_prepared_inputs(root)
            frame.write_bytes(b'altered')
            # The outer receipt's hash still matches; recursive input checking must fail.
            verify_seal(root,'data/cohort-complete.json')
            with self.assertRaises(ValueError): verify_prepared_inputs(root)

    def test_teacher_relocation_preserves_scientific_identity(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp); (root/'config').mkdir(); (root/'data/config').mkdir(parents=True)
            original={'repository_path':'old','checkpoint_path':'old.pt','checkpoint_sha256':'same','builder':'teacher'}
            current={**original,'repository_path':'new','checkpoint_path':'new.pt'}
            write_json(root/'config/teacher-contract.json',original)
            path=root/'data/config/teacher-contract.json'; write_json(path,current)
            run=root/'data/config/run-contract.json'
            write_json(run,{'config_sha256':{'teacher-contract.json':sha256_file(path)}})
            verify_data_identity(root)
            current['checkpoint_sha256']='different'; write_json(path,current)
            write_json(run,{'config_sha256':{'teacher-contract.json':sha256_file(path)}})
            with self.assertRaises(ValueError): verify_data_identity(root)


class ScalingNumericsTests(unittest.TestCase):
    def test_unscaled_orchestration_matches_historical_implementation(self):
        from gavd6_sjepa.research_directions.future_innovation.fi_joint_training import nested_partition as historical
        cohort,arrays,schema=fixture(261306,'temporal')
        train,test=np.arange(100),np.arange(100,160)
        with threadpool_limits(1):
            original=historical(cohort,arrays,train,test,schema,JointModelContract())
            isolated=nested_partition(cohort,arrays,train,test,schema,JointModelContract())
        self.assertEqual(original['selection'],isolated['selection'])
        self.assertEqual(original['split'],isolated['split'])
        for arm in DIRECT_ARMS:
            np.testing.assert_array_equal(original['predictions'][arm],isolated['predictions'][arm])

    def test_actual_fit_reload_reuse_and_rehashed_prediction_rejection(self):
        cohort,arrays,_=fixture(261305,'temporal')
        roster=reserve_sources(cohort.assign(sequence_id=cohort.window_id),set(cohort.video_id))
        cohort=cohort.drop(columns='outer_fold').merge(roster[['video_id','source_group','outer_fold']],on='video_id')
        full_plan=make_plan(cohort,roster)
        identity=next(iter(full_plan['fits']))
        plan={**full_plan,'fits':{identity:full_plan['fits'][identity]}}
        with tempfile.TemporaryDirectory() as tmp,threadpool_limits(1):
            root=Path(tmp); (root/'config').mkdir(); (root/'manifests').mkdir()
            write_json(root/'config/nuisance-schema.json',{'context_embedding_columns':16,'columns':[]})
            write_json(root/'manifests/learning-plan.json',plan)
            module='gavd6_sjepa.research_directions.future_innovation_scaling.fi_scaling_training.'
            with patch(module+'read_study',return_value=({'synthetic':True},None)), \
                 patch(module+'load_expanded',return_value=(cohort,arrays)),patch(module+'checked_plan',return_value=plan):
                run(root)
                before={str(p):p.stat().st_mtime_ns for p in root.rglob('*') if p.is_file()}
                run(root)
                self.assertEqual(before,{str(p):p.stat().st_mtime_ns for p in root.rglob('*') if p.is_file()})
            path=root/f'predictions/{identity}.npz'
            with np.load(path,allow_pickle=False) as data: values={k:data[k] for k in data.files}
            values['y_true']=values['y_true']*100
            np.savez_compressed(path,**values)
            receipt=root/f'models/{identity}/complete.json'
            saved=read_json(receipt); saved['artifacts'][str(path.relative_to(root))]=sha256_file(path); write_json(receipt,saved)
            with self.assertRaises(AssertionError): verify_fit(root,identity,cohort,arrays,plan)

    def test_average_loss_penalty_is_invariant_to_duplicate_rows(self):
        rng=np.random.default_rng(261301)
        x=rng.normal(size=(20,4)); s=rng.normal(size=(20,3)); y=rng.normal(size=(20,2))
        base=JointRidge.fit(x,s,y,np.ones(20),20*.25,20*2.5)
        twice=JointRidge.fit(np.repeat(x,2,axis=0),np.repeat(s,2,axis=0),np.repeat(y,2,axis=0),np.ones(40),40*.25,40*2.5)
        old=JointRidge.fit(np.repeat(x,2,axis=0),np.repeat(s,2,axis=0),np.repeat(y,2,axis=0),np.ones(40),20*.25,20*2.5)
        np.testing.assert_allclose(base.predict(x,s),twice.predict(x,s),rtol=1e-10,atol=1e-10)
        self.assertGreater(np.max(np.abs(old.predict(x,s)-base.predict(x,s))),.001)

    def test_penalties_use_each_inner_fit_count_and_holdout_never_refits(self):
        cohort,arrays,schema=fixture(261302,'temporal')
        train=np.arange(100); test=np.arange(100,160)
        with threadpool_limits(1):
            fit=nested_partition(cohort,arrays,train,test,schema,JointModelContract(),penalty_reference_windows=40)
        for records in [fit['baseline_selection'],*[v['candidates'] for v in fit['selection'].values()]]:
            for candidate in records:
                for row in candidate['inner_folds']:
                    self.assertAlmostEqual(row['effective_lambda_x'],candidate['lambda_x']*row['fit_windows']/40)
                    self.assertAlmostEqual(row['fit_weight_sum'],row['fit_windows'])
        before=copy.deepcopy(fit['baseline'].x_scaler.record())
        fit['baseline'].predict(arrays['baseline'][test]*1e8)
        self.assertEqual(before,fit['baseline'].x_scaler.record())
        for arm in DIRECT_ARMS: self.assertTrue(fit['selection'][arm]['complete'])

    def test_scoring_preserves_source_multiplicity_and_masks(self):
        policy={**POLICY,'nominal_source_sizes':[1],'subset_seeds':[11,22,33], 'outer_folds':2,'bootstrap_repetitions':2000}
        payloads={}
        for fold in range(2):
            y=np.array([[1.,3.],[2.,5.]]) if fold==0 else np.array([[3.,4.]])
            for size,scale in [('1',.5),('all',.75)]:
                identity=f'{fold}-{size}'
                payload=dict(window_id=np.array([f'w{fold}-{j}' for j in range(len(y))]),video_id=np.repeat(f'v{fold}',len(y)),
                             y_true=y,y_raw=y,baseline=np.zeros_like(y),target_mean=np.zeros(2),target_scale=np.ones(2),valid_features=np.array([True,False]))
                payload.update({arm:(y*scale if arm=='real-skeleton' else np.zeros_like(y)) for arm in DIRECT_ARMS})
                payloads[identity]=payload
        entries=[dict(size=size,subset_seed=seed,outer_fold=f,status='planned',fit_id=f'{f}-{size}') for size in ['1','all'] for seed in policy['subset_seeds'] for f in range(2)]
        plan=dict(entries=entries,common_evaluation_windows=['w0-0','w0-1','w1-0'],common_evaluation_sources=['v0','v1'])
        result,metrics,boot,raw=compute_curve(plan,payloads,policy,synthetic=True)
        self.assertEqual(result['status'],'synthetic_calibration_only')
        real=metrics[metrics.arm=='real-skeleton']
        np.testing.assert_allclose(real.loc[real['size']=='1','r2_full'],.75)
        np.testing.assert_allclose(real.loc[real['size']=='all','r2_full'],.9375)
        self.assertAlmostEqual(result['growth']['estimate'],.1875)
        # Equal source weights: source means 2.5 and 9, not the unweighted clip mean.
        self.assertAlmostEqual(raw.iloc[0].baseline_mse,(2.5+9)/2)
        broken=copy.deepcopy(payloads); broken['0-all']['window_id'][0]='bad'
        with self.assertRaises(ValueError): compute_curve(plan,broken,policy)


if __name__=='__main__': unittest.main()
