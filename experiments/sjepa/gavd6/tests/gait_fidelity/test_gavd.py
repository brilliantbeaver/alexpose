"""Real-video boundary tests; synthetic media certify software, not real transfer."""
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import json
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

from gavd6_sjepa.research_directions.gait_fidelity import gavd


class GAVDTests(unittest.TestCase):
    def test_linked_people_and_duplicate_media_never_cross_splits(self):
        videos=[f'v{i}' for i in range(20)]
        ids=pd.DataFrame({'video_id':['v0','v1','v1','v2'],'person_id':['p','p','q','q']})
        groups=gavd.recording_groups(videos,ids,{'v2':'same','v3':'same'})
        self.assertEqual(len({groups[f'v{i}'] for i in range(4)}),1)
        rows=pd.DataFrame({'video_id':videos,'gait_pattern_annotation':['normal']*10+['stroke']*10})
        reserved=pd.DataFrame([{'video_id':'v0','role':'confirmation'}])
        splits=gavd.assign_splits(rows,groups,reservations=reserved)
        self.assertEqual(splits[groups['v3']],'protected')
        self.assertEqual(splits,gavd.assign_splits(rows.sample(frac=1,random_state=2),groups,reservations=reserved))

    def test_nearest_frames_are_distinct_nonoverlapping_and_within_annotations(self):
        windows=gavd.sample_windows(19,700,30.,128,25.)
        self.assertGreater(len(windows),1)
        all_frames=np.concatenate(windows)
        self.assertEqual(len(all_frames),len(np.unique(all_frames)))
        self.assertTrue(np.all(np.diff(all_frames)>0));self.assertLessEqual(all_frames[-1],700)
        self.assertEqual(gavd.sample_windows(0,1000,20.,128,25.),[])
        self.assertEqual(gavd.sample_windows(0,80,30.,128,25.),[])

    def test_annotation_geometry_fails_closed(self):
        b=gavd.annotation_box("{'left': 10, 'top': 20, 'width': 30, 'height': 40}",
                              "{'width': 100, 'height': 100}",(200,200,3))
        np.testing.assert_array_equal(b,[20,40,80,120])
        with self.assertRaises(KeyError):
            gavd.annotation_box('{}',"{'width': 100, 'height': 100}",(200,200,3))

    def test_probe_imputation_and_scaling_use_training_only(self):
        x=np.array([[0.,np.nan],[1.,2.],[3.,4.],[4.,np.nan]])
        model=gavd.fit_probe(x,['normal','normal','stroke','stroke'],['a','a','b','c'])
        before={k:v.copy() for k,v in model.items()}
        out=gavd.probe_predict(model,[[10000.,np.nan]])
        self.assertEqual(len(out),1)
        for key in model: np.testing.assert_array_equal(model[key],before[key])
        self.assertEqual(model['median'][1],3.)

    def test_features_retain_missingness_and_input_scale(self):
        rng=np.random.default_rng(9);x=rng.normal(size=(128,12,2));t=np.arange(128)/25
        o=np.ones((128,12),bool);o[:,0]=False;x[:,0]=np.nan
        f=gavd.pose_features(x,o,t,input_origin=np.zeros(2),input_scale=2.)
        self.assertEqual(f.shape,(112,));self.assertTrue(np.isnan(f).any())
        larger=gavd.pose_features(x*2,o,t,input_origin=np.zeros(2),input_scale=2.)
        self.assertFalse(np.allclose(f,larger,equal_nan=True))

    def test_group_score_not_inflated_by_many_clips(self):
        rows=pd.DataFrame({'sequence_id':['a','b','c','d'],'source_group':['v1','v2','v3','v4'],
                           'label':['normal','normal','stroke','stroke'],
                           'predicted':['normal','stroke','stroke','normal']})
        one=gavd.grouped_scores(rows,draws=0)
        many=gavd.grouped_scores(pd.concat([rows,*[rows.iloc[[0]]]*100]),draws=0)
        self.assertEqual(one['macro_recall'],many['macro_recall'])
        self.assertEqual(many['groups'],4)
        pair=pd.concat([rows.assign(method='unchanged'),rows.assign(method='model')])
        self.assertEqual(gavd.paired_group_interval(pair,'model',draws=0)['improvement'],0.)

    def test_seeded_comparison_does_not_count_seeds_as_people(self):
        rows=[]
        for seed in (17,29,43):
            for i,label in enumerate(['normal','normal','stroke','stroke']):
                for method in ('candidate','comparator'):
                    rows.append(dict(sequence_id=f's{i}',source_group=f'v{i}',label=label,predicted=label,
                                     method=f'{method}-seed-{seed}'))
        result=gavd.seeded_primary_comparison(pd.DataFrame(rows),'candidate','comparator',draws=0)
        self.assertEqual(result['groups'],4);self.assertEqual(result['seeds'],3);self.assertEqual(result['improvement'],0.)

    def test_fixture_or_unknown_checkpoints_cannot_become_real_evidence(self):
        for evidence in ('fixture-tested',None):
            payload={'signature':{'model':{'window_size':128},'sampling_hz':25.,'evidence_status':evidence}}
            with self.assertRaises(ValueError):gavd._validate_restorer_signature(payload,{'samples':128,'hz':25.})
        with self.assertRaisesRegex(ValueError,'sampling rate'):
            gavd._validate_restorer_signature({'signature':{'model':{'window_size':128},'sampling_hz':30.,'evidence_status':'technical-source-screen'}},{'samples':128,'hz':25.})

    def test_exposed_reservations_cannot_enter_confirmation(self):
        rows=pd.DataFrame({'video_id':[f'v{i}' for i in range(20)],'gait_pattern_annotation':['normal']*20})
        groups=gavd.recording_groups(rows.video_id)
        exposed=pd.DataFrame({'video_id':rows.video_id,'previously_exposed':[True]*20})
        roles=gavd.assign_splits(rows,groups,reservations=exposed)
        self.assertNotIn('confirmation',roles.values())
        for column,value in [('reserved','true'),('protected','true')]:
            ledger=pd.DataFrame({'video_id':rows.video_id,column:value})
            self.assertEqual(set(gavd.assign_splits(rows,groups,reservations=ledger).values()),{'protected'})
        for column,value in [('role','development'),('role','train'),('exposure','known_development'),('exposure_status','previously_exposed')]:
            ledger=pd.DataFrame({'video_id':rows.video_id,column:value})
            self.assertNotIn('confirmation',gavd.assign_splits(rows,groups,reservations=ledger).values())
        with self.assertRaises(ValueError):gavd.assign_splits(rows,groups,reservations=rows[['video_id']])

    def test_historical_role_and_split_preserve_strictest_reservation(self):
        rows=pd.DataFrame({'video_id':[f'v{i}' for i in range(10)],'gait_pattern_annotation':['normal']*10})
        groups=gavd.recording_groups(rows.video_id)
        for role,split in [('', 'confirmation'),('development','confirmation'),('test','train')]:
            ledger=pd.DataFrame({'video_id':rows.video_id,'role':role,'split':split})
            self.assertEqual(set(gavd.assign_splits(rows,groups,reservations=ledger).values()),{'protected'})
        ledger=pd.DataFrame({'video_id':rows.video_id,'role':'','split':'development'})
        self.assertNotIn('confirmation',gavd.assign_splits(rows,groups,reservations=ledger).values())
        ledger['split']='unsupported_history'
        with self.assertRaises(ValueError):gavd.assign_splits(rows,groups,reservations=ledger)

    def test_real_plan_extract_and_probe_paths_with_fake_pose_backend(self):
        import cv2
        with TemporaryDirectory() as temp:
            root=Path(temp);manifest=root/'manifests';manifest.mkdir();media=root/'videos';media.mkdir();ann=root/'annotations';ann.mkdir()
            config=root/'pose.py';config.write_text('# dummy fixture configuration\n');checkpoint=root/'pose.pth';checkpoint.write_bytes(b'fixture')
            rows=[];frames=[]
            for i in range(12):
                name=f'vid{i}';path=media/f'{name}.avi'
                # Use a supported discovery suffix with actual MPEG-4 encoding.
                path=media/f'{name}.mp4';writer=cv2.VideoWriter(str(path),cv2.VideoWriter_fourcc(*'mp4v'),25.,(64,64))
                for f in range(40):
                    writer.write(np.full((64,64,3),(i*17+f)%255,np.uint8))
                writer.release()
                seq=f'seq{i}';label='normal' if i<6 else 'stroke'
                rows.append(dict(sequence_id=seq,video_id=name,url='',first_frame=1,last_frame=40,n_annotated_frames=40,source_height=64,
                                 dataset_annotation='Normal Gait' if label=='normal' else 'Abnormal Gait',gait_pattern_annotation=label,cam_view='left side'))
                for f in range(1,41):
                    frames.append(dict(seq=seq,id=name,frame_num=f,bbox="{'left': 2, 'top': 2, 'width': 60, 'height': 60}",vid_info="{'width': 64, 'height': 64}"))
            pd.DataFrame(rows).to_csv(manifest/'gavd_full_sequences.csv',index=False)
            pd.DataFrame({'video_id':[r['video_id'] for r in rows]}).to_csv(manifest/'gavd_full_videos.csv',index=False)
            pd.DataFrame(frames).to_csv(ann/'GAVD_Clinical_Annotations_1.csv',index=False)
            cfg=dict(work=str(root/'run'),asset_root=str(root),code_root=str(root),data=dict(samples=32,hz=25.),
                     preparation=dict(estimators=[dict(student_id='mock',family='mock',config=str(config),checkpoint=str(checkpoint))]))
            reservation=root/'historical-reservations.csv'
            pd.DataFrame({'video_id':['vid0','vid1'],'source_group':['known-shared-person']*2,
                          'role':['development']*2,'previously_exposed':[False]*2}).to_csv(reservation,index=False)
            with patch('gavd6_sjepa.research_directions.synthetic_training_v2.preparation._config_files',return_value={str(config):gavd.sha256(config)}):
                result=gavd.plan_gavd(cfg,video_root=media,annotation_root=ann,manifest_dir=manifest,reservation_csv=reservation,shards=2)
            self.assertEqual(result['sequences'],12)
            planned=pd.read_csv(root/'run/gavd/sequences.csv').set_index('video_id')
            self.assertEqual(planned.loc['vid0','source_group'],planned.loc['vid1','source_group'])
            self.assertNotEqual(planned.loc['vid0','split'],'confirmation')
            class Model:
                def eval(self): return self
                def test_step(self,batch):
                    points=np.column_stack([np.linspace(10,50,17),np.linspace(10,55,17)]).astype(np.float32)
                    return [SimpleNamespace(pred_instances=SimpleNamespace(keypoints=points[None],keypoint_scores=np.ones((1,17),np.float32))) for _ in batch]
            estimator=SimpleNamespace(model=Model(),_batch=lambda images,boxes:images)
            with patch('gavd6_sjepa.research_directions.synthetic_training_v2.runtime.require_haic_runtime',return_value={}),patch('gavd6_sjepa.research_directions.synthetic_training.estimators.load_estimator',return_value=estimator):
                for i in range(2): gavd.extract_gavd(cfg,i)
            result=gavd.evaluate_gavd(cfg,[],device='cpu')
            self.assertIn('unchanged/mock',result['scores']);self.assertIn('view_only/mock',result['scores'])
            self.assertFalse(result['clinical_validation'])
            with self.assertRaises(FileNotFoundError):gavd.extract_gavd(cfg,0,split='confirmation')
            summary=json.loads((Path(result['output'])/'scores.json').read_text())
            self.assertEqual(summary['unchanged/mock']['evaluation_groups'],2)
            saved=pd.read_csv(root/'run/gavd/sequences.csv',keep_default_na=False)
            ledger=root/'exposure.csv';selected=saved[saved.split.eq('confirmation')]
            pd.DataFrame({'video_id':selected.video_id,'exposure':'unknown','reviewed_by':'reviewer','evidence':'audit'}).to_csv(ledger,index=False)
            # A source-status checkpoint stub isolates lock policy from fitting.
            final=root/'final.pt';final.write_bytes(b'completed checkpoint test stand-in')
            with patch('gavd6_sjepa.research_directions.gait_fidelity.training.load_model',return_value=(None,{'signature':{'model':{'window_size':32},'sampling_hz':25.,'evidence_status':'technical-source-screen'}})):
                with self.assertRaises(ValueError):gavd.lock_gavd(cfg['work'],[f'model={final}'],ledger)
                pd.DataFrame({'video_id':selected.video_id,'exposure':'unexposed_verified','reviewed_by':'reviewer','evidence':'audit'}).to_csv(ledger,index=False)
                gavd.lock_gavd(cfg['work'],[f'model={final}'],ledger)
                final.write_bytes(b'changed')
                with self.assertRaises(ValueError):gavd._check_confirmation(root/'run/gavd',gavd.read_json(root/'run/gavd/plan.json'))


if __name__=='__main__':unittest.main()
