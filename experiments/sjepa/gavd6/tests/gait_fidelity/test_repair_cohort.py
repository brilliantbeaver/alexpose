"""Independent identities, immutable protocols and real source QC for repair tests."""
from contextlib import ExitStack
from copy import deepcopy
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import pandas as pd

from gavd6_sjepa.research_directions.gait_fidelity.common import atomic_json, read_json, sha256
from gavd6_sjepa.research_directions.gait_fidelity.data import load_dataset, fixture_bundle
from gavd6_sjepa.research_directions.gait_fidelity.repair_cohort import (
    plan_slim_cohort, lock_slim_confirmation, verify_slim_lock, prepare_slim,
    load_slim_cohort, scientific_config,
)


def _source_fits(root):
    from gavd6_sjepa.research_directions.gait_fidelity.repair_evaluation import EXPECTED_FITS
    fits,payloads=[],{}
    for index,(method,seed) in enumerate(sorted(EXPECTED_FITS)):
        path=root/f'source-fit-{index}.pt';path.write_bytes(f'{method}-{seed}'.encode())
        signature=dict(seed=seed,evidence_status='technical-source-screen')
        if method.startswith('P-'):
            signature.update(phase='end_to_end',encoder='direct',policy=None,objective='base')
        else:
            variant='jepa_endpoint_v1' if 'jepa_endpoint_v1' in method else 'jepa_delta_v1'
            objective=method.rsplit('-',1)[1]
            signature.update(phase='readout',encoder='paired_jepa',policy='graph_time',objective=objective,representation_variant=variant)
            if method.startswith('R-'):signature['repair']=dict(format='gait-fidelity-readout-repair-v1')
        fits.append(dict(method=method,seed=seed,checkpoint=str(path),sha256=sha256(path)))
        payloads[str(path.resolve())]=dict(signature=signature)
    return fits,payloads


class RepairCohortTest(unittest.TestCase):
    def setUp(self):
        tmp = TemporaryDirectory(); self.addCleanup(tmp.cleanup)
        self.root = Path(tmp.name)
        self.config = dict(mode='fixture', seed=11, seeds=[11],
            data=dict(samples=16,hz=25.), model={}, training={}, measurement={}, evaluation={},
            repair=dict(deadline_utc='2100-01-01T00:00:00Z',fixture_confirmation_people=3,
                cohort_plan=str(self.root/'slim.json'),confirmation_lock=str(self.root/'lock.json'),
                statistical_protocol=dict(primary='dense-minus-scalar',unit='person',seeds=[11])))
        self.checkpoint = self.root/'checkpoint.pt'; self.checkpoint.write_bytes(b'fixed-completed-test-model')
        self.fits = [dict(method='dense',seed=11,checkpoint=str(self.checkpoint),checkpoint_sha256=sha256(self.checkpoint))]
        self.payloads={str(self.checkpoint.resolve()):dict(signature=dict(evidence_status='fixture-tested',seed=11))}
        loader=patch('gavd6_sjepa.research_directions.gait_fidelity.training.load_model',
            side_effect=lambda path:(None,self.payloads[str(path)]))
        loader.start();self.addCleanup(loader.stop)

    def plan(self):
        return plan_slim_cohort(self.config,self.config['repair']['cohort_plan'])

    def lock(self, exposure=None):
        return lock_slim_confirmation(self.config,self.fits,exposure_ledger=exposure,
            reviewed_by='test reviewer',evidence='explicit software test',output=self.config['repair']['confirmation_lock'])

    def source_config(self):
        manifests=self.root/'manifests'; manifests.mkdir()
        raw_root=self.root/'amass'; raw_root.mkdir()
        inventory,registry,splits=[],[],[]
        for p,split,source in [('train','train','BioMotionLab_NTroje'),('dev','validation','KIT'),
                               ('held-a','test','BioMotionLab_NTroje'),('held-b','test','KIT')]:
            registry.append(dict(subject_id_candidate=p,identity=p,identity_audit_status='approved',excluded=False))
            splits.append(dict(identity=p,split=split))
            for m in range(3):
                relative=f'{p}/walking-{m}.npz'; raw=raw_root/relative
                raw.parent.mkdir(exist_ok=True);raw.write_bytes(relative.encode())
                inventory.append(dict(relative_path=relative,source_dataset=source,subject_id_candidate=p,
                    motion_id=f'walking-{m}',sha256=sha256(raw),num_frames=400,mocap_framerate=25,status='ok'))
        for name,rows in [('amass_raw_inventory_eligible',inventory),('amass_subject_registry',registry),('amass_subject_splits',splits)]:
            pd.DataFrame(rows).to_csv(manifests/f'{name}.csv',index=False)
        self.config['mode']='source';self.config['data']['samples']=128
        self.fits,self.payloads=_source_fits(self.root)
        self.config['cohort']=dict(preset='named_walking')
        self.config['preparation']=dict(manifest_dir=str(manifests),amass_root=str(raw_root),
            estimators=[dict(family=x,student_id=x) for x in ('vitpose','rtmpose','hrnet')])
        ledger=self.root/'exposure.csv'
        pd.DataFrame([dict(person_id=p,canonical_person_id=p,original_split='test',reserved=False,exposure='unexposed_verified')
            for p in ('held-a','held-b')]).to_csv(ledger,index=False)
        return ledger

    def test_metadata_selection_keeps_all_people_distinct_motions_and_original_test(self):
        self.source_config()
        # Planning may stat availability, but must not read any NPZ contents.
        with patch('numpy.load',side_effect=AssertionError('Raw motion opened during planning')):
            first=self.plan();second=plan_slim_cohort(self.config)
        self.assertEqual(first,second)
        self.assertEqual(first['person_ids'],['held-a','held-b'])
        self.assertEqual(first['summary']['planned_windows'],4)
        self.assertEqual(first['summary']['rendered_frames'],4*3*2*2*2*128)
        for person in first['person_ids']:
            rows=[r for r in first['records'] if r['canonical_person_id']==person]
            self.assertEqual(len({r['motion_hash'] for r in rows}),2)
            self.assertEqual({r['original_split'] for r in rows},{'test'})
            self.assertEqual({r['split'] for r in rows},{'confirmation'})
            self.assertEqual({r['exposure'] for r in rows},{'unknown'})
        self.assertEqual(first['panel']['extractor_families'],['vitpose','rtmpose'])
        self.assertEqual(first['summary']['source_people'],{'BioMotionLab_NTroje':1,'KIT':1})
        self.assertEqual(first['summary']['source_windows'],{'BioMotionLab_NTroje':2,'KIT':2})

    def test_inadequate_distinct_motions_cannot_silently_drop_a_person(self):
        self.source_config()
        path=self.root/'manifests/amass_raw_inventory_eligible.csv'
        frame=pd.read_csv(path)
        frame=frame[~(frame.subject_id_candidate.eq('held-a') & ~frame.motion_id.eq('walking-0'))]
        frame.to_csv(path,index=False)
        with self.assertRaisesRegex(ValueError,'no person was dropped'): self.plan()

    def test_content_identity_split_conflict_rejected(self):
        self.source_config()
        path=self.root/'manifests/amass_raw_inventory_eligible.csv';frame=pd.read_csv(path)
        frame.loc[frame.subject_id_candidate.eq('held-b'),'sha256']=frame.loc[frame.subject_id_candidate.eq('train'),'sha256'].iloc[0]
        frame.to_csv(path,index=False)
        with self.assertRaisesRegex(ValueError,'crosses an identity'): self.plan()

    def test_missing_source_file_keeps_original_shard_assignment_and_reports_loss(self):
        ledger=self.source_config();plan=self.plan();self.lock(ledger)
        Path(plan['records'][0]['raw_path']).unlink()
        receipt=prepare_slim(self.config,self.root/'missing-shard',shard_index=0,num_shards=4)
        self.assertEqual(receipt['status'],'no_work')
        self.assertEqual(receipt['planned_windows'],1)
        self.assertEqual(receipt['retained_windows'],0)
        self.assertEqual(receipt['unavailable_source_families'],[plan['records'][0]['source_family_id']])
        other=deepcopy(self.config);other['_repair_lock_config']=self.config
        other['_repair_shard_index']=1;other['_repair_num_shards']=4
        rows,_,_=load_slim_cohort(other)
        self.assertEqual(list(rows.source_family_id),[plan['records'][1]['source_family_id']])

    def test_source_exposure_is_required_exact_explicit_and_immutable(self):
        ledger=self.source_config();self.plan()
        with self.assertRaisesRegex(PermissionError,'real separately reviewed'): self.lock()
        frame=pd.read_csv(ledger);frame.loc[0,'exposure']='unknown';frame.to_csv(ledger,index=False)
        with self.assertRaisesRegex(PermissionError,'unexposed_verified'): self.lock(ledger)
        frame.loc[0,'exposure']='unexposed_verified';frame.to_csv(ledger,index=False)
        self.lock(ledger); rows,_,_=load_slim_cohort(self.config)
        self.assertEqual(set(rows.exposure),{'unexposed_verified'})
        self.assertEqual(set(rows.locomotion_status),{'metadata_candidate'})
        frame.loc[0,'reserved']=True;frame.to_csv(ledger,index=False)
        with self.assertRaisesRegex(PermissionError,'ledger changed'): verify_slim_lock(self.config)

    def test_prior_known_exposure_cannot_be_erased_by_new_ledger(self):
        ledger=self.source_config()
        history=self.root/'history.csv';frame=pd.read_csv(ledger);frame['exposure']='known_development';frame.to_csv(history,index=False)
        self.config['cohort']['reservation_csv']=str(history)
        self.plan()
        with self.assertRaisesRegex(PermissionError,'cannot be erased'): self.lock(ledger)

    def test_lock_rejects_checkpoint_and_population_tampering(self):
        self.plan();self.lock()
        self.checkpoint.write_bytes(b'changed weights')
        with self.assertRaisesRegex(PermissionError,'checkpoint changed'): verify_slim_lock(self.config)
        self.checkpoint.write_bytes(b'fixed-completed-test-model')
        plan=read_json(self.config['repair']['cohort_plan']);plan['records'][0]['split']='development'
        atomic_json(self.config['repair']['cohort_plan'],plan)
        with self.assertRaisesRegex(ValueError,'identity changed'): verify_slim_lock(self.config)

    def test_statistics_and_primary_settings_are_frozen_operational_receipts_are_not(self):
        self.plan();self.lock()
        self.config['repair']['prediction_receipts']=[dict(path='later-output')]
        verify_slim_lock(self.config)
        self.config['repair']['statistical_protocol']['primary']='selected-after-test'
        with self.assertRaisesRegex(PermissionError,'Scientific configuration'): verify_slim_lock(self.config)

    def test_prepare_without_lock_or_after_deadline_never_calls_source_backend(self):
        self.plan()
        with patch('gavd6_sjepa.research_directions.gait_fidelity.preparation.prepare') as backend:
            with self.assertRaises(FileNotFoundError): prepare_slim(self.config,self.root/'prepared')
            backend.assert_not_called()
        self.config['repair']['deadline_utc']='2000-01-01T00:00:00Z'
        self.config['mode']='source'
        with self.assertRaises(TimeoutError): self.lock()

    def test_fixture_has_genuinely_separate_generated_ids_and_no_training_access(self):
        plan=self.plan();self.lock()
        receipt=prepare_slim(self.config,self.root/'prepared')
        self.assertTrue(receipt['fixture']);self.assertFalse(receipt['independent_confirmation'])
        self.assertTrue(receipt['complete_planned_population'])
        bundle=load_dataset(receipt['bundle'],allow_confirmation=True)
        self.assertEqual({r['canonical_person_id'] for r in bundle.records},set(plan['person_ids']))
        train_ids={r['canonical_person_id'] for r in fixture_bundle(samples=16).records}
        self.assertFalse(train_ids & set(plan['person_ids']))
        self.assertEqual({r['split'] for r in bundle.records},{'confirmation'})
        self.assertEqual({r['movement_magnitude'] for r in bundle.records},{0.,5.,15.})
        self.assertEqual({r['extractor_family'] for r in bundle.records},{'vitpose','rtmpose'})
        self.assertEqual(bundle.provenance['repair_confirmation_lock_sha256'],sha256(self.config['repair']['confirmation_lock']))
        with self.assertRaises(PermissionError): load_dataset(receipt['bundle'])
        with self.assertRaises(FileExistsError): prepare_slim(self.config,self.root/'prepared')

    def test_fixture_shards_partition_exact_locked_family_population(self):
        plan=self.plan();self.lock();sets=[]
        for index in range(4):
            receipt=prepare_slim(self.config,self.root/f'shard-{index}',shard_index=index,num_shards=4)
            data=load_dataset(receipt['bundle'],allow_confirmation=True)
            sets.append({r['source_family_id'] for r in data.records})
        self.assertEqual(set.union(*sets),set(plan['source_family_ids']))
        self.assertEqual(sum(map(len,sets)),len(set.union(*sets)))


class RepairActualCheckpointTest(unittest.TestCase):
    def test_registered_direct_checkpoint_uses_null_policy_and_passes_attribution(self):
        import torch
        from gavd6_sjepa.research_directions.gait_fidelity.training import train_phase,load_model
        from gavd6_sjepa.research_directions.gait_fidelity.repair_cohort import _fits
        torch.set_num_threads(1)
        bundle=fixture_bundle(samples=16,people=3)
        recipes=read_json(Path(__file__).resolve().parents[2]/'src/gavd6_sjepa/research_directions/gait_fidelity/recipes.json')
        recipe=next(r for r in recipes['recipes'] if r['recipe_id']=='P-direct-none-base')
        config=dict(device='cpu',model=dict(width=8,encoder_layers=1,predictor_layers=1,heads=2,patch_size=4,window_size=16),
            training=dict(pretraining_updates=1,readout_updates=1,end_to_end_updates=1,batch_size=4,
                learning_rate=.0003,repaired_trials=2,log_every=100),
            measurement=dict(min_frames=8,min_coverage=.8,min_segment_px=2.))
        with TemporaryDirectory() as temporary:
            result=train_phase(bundle,recipe,'end_to_end',17,config,Path(temporary)/'fit')
            _,payload=load_model(result['checkpoint'])
            self.assertIsNone(payload['signature']['policy'])
            fits=_fits([dict(method='P-direct-none-base',seed=17,checkpoint=result['checkpoint'],
                sha256=result['checkpoint_sha256'])],fixture=True)
            self.assertEqual(fits[0]['method'],'P-direct-none-base')


class RepairSourcePreparationTest(unittest.TestCase):
    def test_source_pipeline_keeps_true_test_labels_and_reports_reference_qc_loss(self):
        from tests.gait_fidelity.test_source_pipeline import SourcePipelineTest,_Body,_load_estimator,_motion,_render
        with TemporaryDirectory() as temporary:
            helper=SourcePipelineTest();helper.root=Path(temporary)
            config=helper.make_source_configuration();root=helper.root
            inv_path=root/'manifests/amass_raw_inventory_eligible.csv'
            inv=pd.read_csv(inv_path);inv['source_dataset']='test';inv['motion_id']='normal_walk'
            inv['sha256']=[sha256(root/'amass'/r) for r in inv.relative_path]
            extra=[]
            for m in range(2):
                relative=f'held/walk-{m}.npz';raw=root/'amass'/relative;raw.parent.mkdir(exist_ok=True);raw.write_bytes(relative.encode())
                extra.append(dict(relative_path=relative,subject_id_candidate='held',status='ok',num_frames=128,
                                  mocap_framerate=25,source_dataset='test',motion_id='normal_walk',sha256=sha256(raw)))
            pd.concat([inv,pd.DataFrame(extra)]).to_csv(inv_path,index=False)
            for filename,row in [('amass_subject_registry.csv',dict(subject_id_candidate='held',identity='held',identity_audit_status='approved',excluded=False)),
                                 ('amass_subject_splits.csv',dict(identity='held',split='test'))]:
                path=root/'manifests'/filename;frame=pd.read_csv(path);pd.concat([frame,pd.DataFrame([row])]).to_csv(path,index=False)
            config['cohort']=dict(preset='named_walking')
            config['repair']=dict(deadline_utc='2100-01-01T00:00:00Z',cohort_plan=str(root/'slim.json'),
                confirmation_lock=str(root/'lock.json'),statistical_protocol=dict(primary='fixed',unit='person'))
            plan=plan_slim_cohort(config,config['repair']['cohort_plan'])
            fits,payloads=_source_fits(root)
            ledger=root/'held-exposure.csv';pd.DataFrame([dict(person_id='held',canonical_person_id='held',
                original_split='test',reserved=False,exposure='unexposed_verified')]).to_csv(ledger,index=False)
            with patch('gavd6_sjepa.research_directions.gait_fidelity.training.load_model',
                side_effect=lambda path:(None,payloads[str(path)])):
                lock_slim_confirmation(config,fits,
                    exposure_ledger=ledger,reviewed_by='test',evidence='source adapter software test',output=config['repair']['confirmation_lock'])
            from gavd6_sjepa.research_directions.gait_fidelity.preparation import technical_geometry_screen
            calls=[]
            def screen(body):
                calls.append(True)
                result=technical_geometry_screen(body)
                if len(calls)==1: result['passed']=False
                return result
            with ExitStack() as stack:
                stack.enter_context(patch('gavd6_sjepa.research_directions.gait_fidelity.training.load_model',
                    side_effect=lambda path:(None,payloads[str(path)])))
                for target,replacement in (
                    ('gavd6_sjepa.research_directions.synthetic_training_v2.runtime.require_haic_runtime',lambda:{'test_backend':True}),
                    ('gavd6_sjepa.research_directions.motion_preservation.motion_data.load_motion',_motion),
                    ('gavd6_sjepa.research_directions.motion_preservation.body_geometry.SMPLHBody',_Body),
                    ('gavd6_sjepa.research_directions.synthetic_training.estimators.load_estimator',_load_estimator),
                    ('gavd6_sjepa.research_directions.gait_fidelity.preparation._render_fixed',_render),
                    ('gavd6_sjepa.research_directions.gait_fidelity.preparation.technical_geometry_screen',screen)):
                    stack.enter_context(patch(target,replacement))
                receipt=prepare_slim(config,root/'prepared')
            self.assertEqual(receipt['planned_windows'],2);self.assertEqual(receipt['retained_windows'],1)
            self.assertFalse(receipt['complete_planned_population'])
            self.assertEqual(len(receipt['missing_source_families']),1)
            bundle=load_dataset(receipt['bundle'],allow_confirmation=True)
            self.assertEqual({r['split'] for r in bundle.records},{'confirmation'})
            self.assertEqual({r['locomotion_status'] for r in bundle.records},{'metadata_candidate'})
            self.assertEqual(bundle.provenance['source_selection'],'repair_confirmation')
            self.assertEqual(bundle.provenance['repair_cohort_identity'],plan['identity'])
            self.assertEqual({r['movement_magnitude'] for r in bundle.records},{0.,5.,15.})


if __name__=='__main__': unittest.main()
