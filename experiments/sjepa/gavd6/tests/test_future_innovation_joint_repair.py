"""Regression, source isolation, typed fallback and read-only numerical verification."""
import copy
from dataclasses import asdict
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import joblib
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

from gavd6_sjepa.research_directions.future_innovation.fi_joint_models import (
    JointModelContract, SupportedInput, TargetStandardizer, JointRidge, temporal_features,
    baseline_schema, skeleton_schema, fit_joint_baseline)
from gavd6_sjepa.research_directions.future_innovation.fi_joint_calibration import exact_calibration
from gavd6_sjepa.research_directions.future_innovation.fi_joint_training import (
    choose_candidate, nested_partition, predict_selected, verify_joint_fold)
from gavd6_sjepa.research_directions.future_innovation.fi_cache_reuse import (
    initialize_cached_run, read_parent, load_reused_cache, snapshot, verify_snapshot)
from gavd6_sjepa.research_directions.future_innovation.fi_joint_reporting import verify_numerics, decide_joint_gate
from gavd6_sjepa.research_directions.future_innovation.fi_contracts import read_json, write_json, sha256_file, equal_source_weights, stage_lock
from gavd6_sjepa.research_directions.future_innovation.fi_smoke import synthetic_cache
from gavd6_sjepa.research_directions.future_innovation.fi_nested_training import fit_outer_fold
from gavd6_sjepa.research_directions.future_innovation.fi_reporting import score_gate, build_report, assemble_oof
from gavd6_sjepa.research_directions.future_innovation.fi_residual_models import TrainingScaler

ROOT=Path(__file__).resolve().parents[1]


class InputRepairTests(unittest.TestCase):
    def test_exact_calibration_regressions(self):
        self.assertTrue(exact_calibration()['heldout_statistics_unchanged'])

    def test_scaler_validation_roundtrip_and_weighted_missingness(self):
        vids=['a','a','b','c']; w=equal_source_weights(vids)
        values=np.array([[0.,np.nan,0.],[0.,np.nan,1e-9],[0.,1.,2e-9],[0.,3.,3e-9]])
        ids=list('wxyz'); names=('constant','some_missing','near')
        scaler=SupportedInput.fit(values,w,ids,vids,names,('fraction',)*3)
        self.assertAlmostEqual(scaler.mean[1],2.)
        self.assertEqual(scaler.support_counts.tolist(),[4,2,4])
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp)/'scaler.joblib'; joblib.dump(scaler,p); loaded=joblib.load(p)
            np.testing.assert_array_equal(loaded.transform([[1.,1.,1.]]),scaler.transform([[1.,1.,1.]]))
        before=copy.deepcopy(scaler.record()); scaler.transform([[1e9,np.nan,1e9]])
        self.assertEqual(before,scaler.record())
        with self.assertRaises(ValueError): scaler.transform([[1.,1.,1.]],names=names[::-1])
        broken=copy.deepcopy(scaler); broken.scale[0]=1e-8
        with self.assertRaises(ValueError): broken.validate()
        for invalid in ([1.,-1.,1.,1.],[1.,np.nan,1.,1.],[1.,1.]):
            with self.assertRaises(ValueError): SupportedInput.fit(values,invalid,ids,vids,names,('fraction',)*3)

    def test_target_mask_does_not_erase_heldout_variation(self):
        y=TargetStandardizer.fit(np.array([[0.,0.],[0.,1.]]),np.ones(2),['a','b'])
        np.testing.assert_array_equal(y.inverse(y.transform(np.array([[1.,2.]]))),[[1.,2.]])
        self.assertEqual(y.transform(np.array([[1.,2.]]))[0,0],1e8)

    def test_valid_velocity_and_temporal_positions(self):
        sk=np.zeros((1,32,33,4)); sk[0,0,0]=[1.,0.,1.,1.]; sk[0,2,0]=[100.,0.,1.,1.]
        f=temporal_features(sk); self.assertTrue(np.isnan(f[0,2:4]).all())
        sk[0,3,0]=[102.,0.,1.,1.]; self.assertEqual(temporal_features(sk)[0,2],2.)
        self.assertEqual(len(skeleton_schema()[0]),924)

    def test_real_recorded_missing_partitions(self):
        parent=ROOT/'outputs/future-innovation'
        if not (parent/'models/fold-1/split-audit.json').exists(): self.skipTest('Parent real cache not installed')
        _,cohort,arrays,_=read_parent(parent)
        schema=baseline_schema(read_json(parent/'config/nuisance-schema.json'))
        lookup={w:i for i,w in enumerate(cohort.window_id)}
        for outer,inner,count in ((1,1,19),(3,0,20)):
            split=read_json(parent/f'models/fold-{outer}/split-audit.json')['inner'][inner]
            tr=np.array([lookup[w] for w in split['train']]); va=np.array([lookup[w] for w in split['validation']])
            w=equal_source_weights(cohort.iloc[tr].video_id)
            old=TrainingScaler.fit(arrays['baseline'][tr],w,cohort.iloc[tr].window_id)
            new=SupportedInput.fit(arrays['baseline'][tr],w,cohort.iloc[tr].window_id,cohort.iloc[tr].video_id,*schema)
            bad=(old.variance<=1e-16)&(np.max(np.abs(old.transform(arrays['baseline'][va])),axis=0)>1e6)
            self.assertEqual(bad.sum(),count)
            self.assertEqual(np.max(np.abs(old.transform(arrays['baseline'][va]))),1e8)
            np.testing.assert_array_equal(new.transform(arrays['baseline'][va])[:,bad],0.)
            self.assertLess(new.diagnostics(arrays['baseline'][va])['transformed_max_abs'],1000.)

    def test_explicit_fallback_is_exact_after_reload(self):
        rng=np.random.default_rng(7); x=rng.normal(size=(12,3)); y=x[:,:2]
        ids=[str(i) for i in range(12)]
        base=fit_joint_baseline(x,y,ids,ids,1.,(('a','b','c'),('rgb',)*3))
        saved={'model_version':'joint-ridge-v1','checkpoint_type':'baseline_only','model':None,'lambda_s':None}
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp)/'fallback.joblib'; joblib.dump((saved,base),p); saved,base=joblib.load(p)
            np.testing.assert_array_equal(predict_selected(saved,base,x,None)-base.predict(x),0.)
        records=[{'candidate_id':'baseline_only','candidate_type':'baseline_only','pooled_loss':0.,'valid':True,'lambda_x':1.,'lambda_s':None},
                 {'candidate_id':'inferior','candidate_type':'joint_ridge','pooled_loss':1.,'valid':True,'lambda_x':1.,'lambda_s':1.}]
        self.assertEqual(choose_candidate(records,JointModelContract())[0]['candidate_id'],'baseline_only')
        records[1]['pooled_loss']=1e-12
        self.assertEqual(choose_candidate(records,JointModelContract())[0]['candidate_id'],'baseline_only')


class CachedRepairIntegrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp=tempfile.TemporaryDirectory(); cls.addClassCleanup(cls.temp.cleanup)
        cls.base=Path(cls.temp.name)
        with threadpool_limits(limits=1): cls.parent=synthetic_cache(cls.base/'parent',protocol='direct-v2')
        cls.before=snapshot(cls.parent)
        cls.model=JointModelContract(ridge_alphas=(1.,100.),skeleton_alphas=(1.,100.))
        cls.cal=cls.base/'synthetic-calibration.json'
        write_json(cls.cal,{'passed':True,'model_contract':asdict(cls.model),'synthetic':True,'fixture':'injected integration readiness only'})
        cls.doc=ROOT/'docs/studies/future-innovation/direct-v3-repair-protocol.md'
        cls.root=cls.base/'child'
        with threadpool_limits(limits=1):
            initialize_cached_run(cls.root,cls.parent,cls.doc,cls.cal,cls.model)
            for f in range(5): fit_outer_fold(cls.root,f)
            score_gate(cls.root)
            cls.result=build_report(cls.root)

    def test_completed_cached_comparison_and_readonly_reconstruction(self):
        self.assertTrue(self.result['measurement_complete'],self.result)
        self.assertEqual(self.result['decision'],'STOP')
        self.assertFalse(self.result['allow_full_experiment'])
        before=snapshot(self.root)
        result=verify_numerics(self.root)
        self.assertEqual(result['prediction_rows'],51200)
        self.assertEqual(result['bootstrap_rows'],8000)
        self.assertEqual(before,snapshot(self.root))
        verify_snapshot(self.parent,self.before)

    def test_completed_reuse_and_duplicate_stage_lock(self):
        files=snapshot(self.root/'models')
        fit_outer_fold(self.root,0)
        self.assertEqual(files,snapshot(self.root/'models'))
        with stage_lock(self.root,'fold-0'):
            with self.assertRaisesRegex(ValueError,'Another job'):
                with stage_lock(self.root,'fold-0'): pass
        with stage_lock(self.root,'fold-0'): pass

    def test_incorrect_aggregate_rejected_even_with_updated_hash(self):
        p=self.root/'reports/aggregate-metrics.csv'; seal=self.root/'reports/scores-contract.json'
        before,oldseal=p.read_bytes(),seal.read_bytes()
        try:
            df=pd.read_csv(p); df.loc[0,'r2_full']+=.01; df.to_csv(p,index=False)
            record=read_json(seal); record['artifacts']['reports/aggregate-metrics.csv']=sha256_file(p); write_json(seal,record)
            with self.assertRaisesRegex(ValueError,'Numerical verification'): verify_numerics(self.root)
        finally: p.write_bytes(before); seal.write_bytes(oldseal)

    def test_wrong_units_and_masks_rejected(self):
        c,a=load_reused_cache(self.root)
        p=pd.concat([pd.read_parquet(self.root/f'predictions/fold-{f}.parquet') for f in range(5)],ignore_index=True)
        for column,value in (('target_mean',1e6),('valid_feature',False),('y_reference',1.)):
            broken=p.copy(); broken.loc[0,column]=value
            with self.assertRaises(ValueError): assemble_oof(c,a,broken,self.model,arms=('real-skeleton','time-shuffle','clip-mismatch','no-skeleton'))
        with self.assertRaises(ValueError): assemble_oof(c,a,p.iloc[:-1],self.model,arms=('real-skeleton','time-shuffle','clip-mismatch','no-skeleton'))

    def test_changed_parent_rejected_without_rewriting_it(self):
        p=self.parent/'config/projection-256.npy'; content=p.read_bytes(); stat=p.stat()
        try:
            p.write_bytes(content+b'altered')
            with self.assertRaisesRegex(ValueError,'Parent artifacts changed'): load_reused_cache(self.root)
        finally:
            import os
            p.write_bytes(content); os.utime(p,ns=(stat.st_atime_ns,stat.st_mtime_ns))
        verify_snapshot(self.parent,self.before)

    def test_missing_inherited_cache_or_audit_never_loads_teacher(self):
        import sys
        from gavd6_sjepa.research_directions.future_innovation import fi_entrypoint
        root=self.base/'missing-evidence'
        initialize_cached_run(root,self.parent,self.doc,self.cal,self.model)
        for filename,handler in (('config/cache-contract.json',fi_entrypoint.cache_main),
                                 ('qc/readiness-summary.json',fi_entrypoint.audits_main)):
            path=root/filename; content=path.read_bytes(); path.unlink()
            try:
                with patch.object(sys,'argv',['fi','--run-root',str(root),'--device','cpu']), patch(
                    'gavd6_sjepa.research_directions.future_innovation.fi_vjepa_adapter.FrozenVJEPAAdapter.from_run') as teacher:
                    with self.assertRaises(FileNotFoundError): handler()
                    teacher.assert_not_called()
            finally: path.write_bytes(content)

    def test_initialization_recovers_without_changing_frozen_parent(self):
        root=self.base/'init-recovery'
        from gavd6_sjepa.research_directions.future_innovation import fi_cache_reuse as reuse
        real_write=reuse.write_once_json
        def crash(path,value):
            if str(path).endswith('cache-contract.json'): raise RuntimeError('injected init crash')
            real_write(path,value)
        with patch.object(reuse,'write_once_json',side_effect=crash), self.assertRaisesRegex(RuntimeError,'init crash'):
            initialize_cached_run(root,self.parent,self.doc,self.cal,self.model)
        initialize_cached_run(root,self.parent,self.doc,self.cal,self.model)
        load_reused_cache(root)
        verify_snapshot(self.parent,self.before)

    def test_invalid_checkpoint_type_rejected_with_updated_receipt(self):
        model=self.root/'models/fold-0/real-skeleton/selected-model.joblib'
        receipt=self.root/'models/fold-0/fold-complete.json'
        oldmodel,oldreceipt=model.read_bytes(),receipt.read_bytes()
        try:
            saved=joblib.load(model); saved['checkpoint_type']='random_zero_update_head'; joblib.dump(saved,model)
            record=read_json(receipt); record['artifacts'][str(model.relative_to(self.root))]=sha256_file(model); write_json(receipt,record)
            with self.assertRaises(ValueError): verify_joint_fold(self.root,0)
        finally: model.write_bytes(oldmodel); receipt.write_bytes(oldreceipt)

    def test_interrupted_fold_recovery_and_failed_candidate_incomplete(self):
        root=self.base/'recovered'
        with threadpool_limits(limits=1):
            initialize_cached_run(root,self.parent,self.doc,self.cal,self.model)
            model=JointRidge.fit
            count=[0]
            def fail_one(*args,**kwargs):
                count[0]+=1
                if count[0]==8: raise ValueError('injected numerical failure')
                return model(*args,**kwargs)
            with patch.object(JointRidge,'fit',side_effect=fail_one): fit_outer_fold(root,0)
            record=read_json(root/'models/fold-0/fold-complete.json')
            self.assertFalse(record['all_candidates_valid'])
            self.assertFalse(build_report(root)['measurement_complete'])
            # Simulate a crash before a receipt was written in a separate fold.
            directory=root/'models/fold-1'; directory.mkdir(); (directory/'partial.tmp').write_text('interrupted')
            fit_outer_fold(root,1)
            verify_joint_fold(root,1)


if __name__=='__main__': unittest.main()
