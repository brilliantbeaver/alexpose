"""Full manifest selection, leakage boundaries and immutable interval plans."""
from contextlib import ExitStack
import copy
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

from gavd6_sjepa.research_directions.gait_fidelity.cohort import plan_cohort, load_cohort
from gavd6_sjepa.research_directions.gait_fidelity.common import atomic_json, sha256
from gavd6_sjepa.research_directions.gait_fidelity.preparation import prepare, merge_datasets, technical_geometry_screen
from gavd6_sjepa.research_directions.gait_fidelity.data import load_dataset
from tests.gait_fidelity.test_source_pipeline import SourcePipelineTest, _Body, _Estimator, _load_estimator, _motion, _render


class CohortTest(unittest.TestCase):
    def setUp(self):
        temporary = TemporaryDirectory(); self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        (self.root/'manifest').mkdir(); (self.root/'amass').mkdir()
        inventory, registry, splits = [], [], []
        for person, split, name, frames in [('p0','train','normal_walk',400),
                ('p1','validation','treadmill_slow',300), ('p2','test','treadmill_fast',300),
                ('p3','train','sitting',300), ('p4','train','short_walk',64)]:
            relative = f'{person}/{name}.npz'
            raw=self.root/'amass'/relative;raw.parent.mkdir();raw.write_bytes(person.encode())
            inventory.append(dict(relative_path=relative, source_dataset='test',subject_id_candidate=person,
                motion_id=name,sha256=sha256(raw),num_frames=frames,mocap_framerate=25.,status='ok'))
            registry.append(dict(subject_id_candidate=person,identity=person,identity_audit_status='approved',excluded=False))
            splits.append(dict(identity=person,split=split))
        for filename, rows in [('amass_raw_inventory_eligible.csv',inventory),('amass_subject_registry.csv',registry),('amass_subject_splits.csv',splits)]:
            pd.DataFrame(rows).to_csv(self.root/'manifest'/filename,index=False)
        self.config=dict(data=dict(samples=128,hz=25.), preparation=dict(manifest_dir=str(self.root/'manifest'),amass_root=str(self.root/'amass')),
            cohort=dict(preset='named_walking',plan_path=str(self.root/'cohort/manifest.json')))

    def save(self): return plan_cohort(self.config,self.root/'cohort')

    def test_declared_missing_historical_reservation_fails_closed(self):
        from gavd6_sjepa.research_directions.gait_fidelity.config import initialize
        prep=dict(self.config['preparation'],reservation_csv=str(self.root/'missing-reservations.csv'))
        with patch('gavd6_sjepa.research_directions.gait_fidelity.config._previous_preparation',return_value=(prep,self.root/'preparation.json')):
            with self.assertRaisesRegex(FileNotFoundError,'historical AMASS reservation'):
                initialize(self.root/'run',root=self.root)
            replacement=self.root/'replacement.csv'
            pd.DataFrame(columns=['person_id','canonical_person_id','original_split','reserved','exposure']).to_csv(replacement,index=False)
            cfg=initialize(self.root/'run',root=self.root,reservation_csv=replacement)
            self.assertEqual(cfg['cohort']['reservation_csv'],str(replacement.resolve()))
            for changed in ({'experiment_set':'full'},{'cohort_preset':'all_eligible'},{'num_shards':9}):
                with self.assertRaisesRegex(ValueError,'new run directory'):
                    initialize(self.root/'run',root=self.root,**changed)
            self.assertEqual(initialize(self.root/'run',root=self.root)['cohort'],cfg['cohort'])

    def test_complete_inventory_nonoverlap_and_original_test_quarantine(self):
        plan=self.save()
        self.assertEqual(len(plan['inventory']),5)
        self.assertEqual(plan['summary']['planned_windows'],7)
        for split in ('train','development','confirmation'): self.assertEqual(plan['summary']['roles'][split]['people'],1)
        rows,excluded,_=load_cohort(self.config)
        self.assertEqual(set(rows.original_split),{'train','validation'})
        self.assertTrue(set(rows.reserved)=={'unknown'})
        for _,group in rows.groupby('relative_path'):
            self.assertTrue(np.all(np.diff(group.start_s)>=128/25.-1e-8))
        with self.assertRaises(PermissionError): load_cohort(self.config,'confirmation')

    def test_person_alias_split_conflicts_and_duplicate_content_rejected(self):
        path=self.root/'manifest/amass_subject_splits.csv'
        frame=pd.read_csv(path);frame.loc[len(frame)]=dict(identity='p0',split='test');frame.to_csv(path,index=False)
        with self.assertRaisesRegex(ValueError,'exactly one'): plan_cohort(self.config)
        frame.iloc[:-1].to_csv(path,index=False)
        path=self.root/'manifest/amass_raw_inventory_eligible.csv'
        frame=pd.read_csv(path);frame.loc[1,'sha256']=frame.loc[0,'sha256'];frame.to_csv(path,index=False)
        with self.assertRaisesRegex(ValueError,'content crosses'): plan_cohort(self.config)

    def test_reviews_override_filename_and_can_include_asymmetric_actions(self):
        reviews=self.root/'reviews.csv'
        pd.DataFrame([dict(relative_path='p3/sitting.npz',start_s=0.,end_s=11.96,decision='include',motion_label='reviewed_asymmetric_walk',reviewer='human-a',evidence='review-01')]).to_csv(reviews,index=False)
        self.config['cohort']['motion_review_csv']=str(reviews)
        plan=plan_cohort(self.config)
        included=[r for r in plan['records'] if r['canonical_person_id']=='p3']
        self.assertEqual(len(included),2)
        self.assertTrue(all(r['label_source']=='reviewed_interval' for r in included))

    def test_authority_or_saved_population_cannot_change(self):
        self.save()
        path=Path(self.config['cohort']['plan_path']);plan=json.loads(path.read_text());plan['records'][0]['start_s']+=1
        atomic_json(path,plan)
        with self.assertRaisesRegex(ValueError,'records were changed'): load_cohort(self.config)

    def test_adjacent_reviews_never_share_boundary_sample_or_exceed_review(self):
        reviews=self.root/'adjacent-reviews.csv'
        # Each interval individually fits 128 samples, but their common frame
        # at 5.08 seconds cannot belong to two independent source windows.
        pd.DataFrame([dict(relative_path='p1/treadmill_slow.npz',start_s=start,end_s=end,
            decision='include',motion_label='treadmill_slow',reviewer='human-a',evidence='review-01')
            for start,end in [(0.,5.08),(5.08,10.16)]]).to_csv(reviews,index=False)
        self.config['cohort']['motion_review_csv']=str(reviews)
        plan=plan_cohort(self.config)
        rows=[r for r in plan['records'] if r['canonical_person_id']=='p1']
        self.assertEqual([(r['start_s'],r['end_s']) for r in rows],[(0.,5.08)])
        # When sufficient reviewed support exists, the adjusted start is frozen
        # in the plan itself and still ends within the second reviewed interval.
        table=pd.read_csv(reviews);table.loc[1,'end_s']=10.20;table.to_csv(reviews,index=False)
        rows=[r for r in plan_cohort(self.config)['records'] if r['canonical_person_id']=='p1']
        np.testing.assert_allclose([r['start_s'] for r in rows],[0.,5.12])
        self.assertLessEqual(rows[1]['end_s'],10.20+1e-9)

    def test_reserved_excluded_unknown_exploratory_and_postfit_test_review(self):
        plan=self.save()
        checkpoint=self.root/'checkpoint.pt';checkpoint.write_bytes(b'frozen weights')
        ledger=self.root/'confirmation-exposure.csv'
        pd.DataFrame([dict(person_id='p2',canonical_person_id='p2',original_split='test',reserved=False,exposure='unexposed_verified')]).to_csv(ledger,index=False)
        lock=self.root/'lock.json'
        atomic_json(lock,dict(schema='gf-confirmation-lock-v1',cohort_identity=plan['identity'],reviewed_by='reviewer',evidence='exposure-search-record',
            person_ids=['p2'],checkpoint_hashes={str(checkpoint):sha256(checkpoint)},exposure_ledger=str(ledger),exposure_ledger_sha256=sha256(ledger)))
        self.config['cohort']['confirmation_lock']=str(lock)
        rows,_,_=load_cohort(self.config,'confirmation')
        self.assertEqual(set(rows.exposure),{'unexposed_verified'})
        self.assertTrue(all(v is False for v in rows.reserved.tolist()))
        checkpoint.write_bytes(b'changed weights')
        with self.assertRaisesRegex(PermissionError,'checkpoint changed'): load_cohort(self.config,'confirmation')

    def test_technical_qc_does_not_require_alternation(self):
        from gavd6_sjepa.research_directions.motion_preservation.body_geometry import _demo_body
        body=_demo_body(_motion({'relative_path':'software-fixture.npz'},duration_s=5.12))
        body.joints[:]=body.joints[0]
        result=technical_geometry_screen(body)
        self.assertTrue(result['passed'])
        self.assertFalse(result['locomotion_diagnostics']['passed'])


class FullSourcePipelineTest(SourcePipelineTest):
    """Exercise the full source branch with the same substituted GPU backends."""
    def test_full_manifest_chunks_resume_and_screen_evidence(self):
        cfg=copy.deepcopy(self.config)
        inventory_path=self.root/'manifests/amass_raw_inventory_eligible.csv'
        table=pd.read_csv(inventory_path)
        table['source_dataset']='test';table['motion_id']='normal_walk'
        table['sha256']=[sha256(self.root/'amass'/r) for r in table.relative_path]
        # Two complete training windows and two development windows.
        table['num_frames']=256;table.to_csv(inventory_path,index=False)
        cfg['data'].update(source_selection='full_manifest',partition='development',review_videos_per_person=1)
        cfg['cohort']=dict(preset='named_walking',plan_path=str(self.root/'cohort/manifest.json'))
        plan_cohort(cfg,self.root/'cohort')
        with ExitStack() as stack:
            for target,replacement in (
                ('gavd6_sjepa.research_directions.synthetic_training_v2.runtime.require_haic_runtime',lambda:{'test_backend':True}),
                ('gavd6_sjepa.research_directions.motion_preservation.motion_data.load_motion',_motion),
                ('gavd6_sjepa.research_directions.motion_preservation.body_geometry.SMPLHBody',_Body),
                ('gavd6_sjepa.research_directions.synthetic_training.estimators.load_estimator',_load_estimator),
                ('gavd6_sjepa.research_directions.gait_fidelity.preparation._render_fixed',_render)):
                stack.enter_context(patch(target,replacement))
            stack.enter_context(patch('gavd6_sjepa.research_directions.gait_fidelity.preparation.select_intervals',
                side_effect=AssertionError('Full-manifest preparation must preserve frozen cohort intervals')))
            first=prepare(cfg,self.root/'full-prepared')
            cfg['data']['resume_preparation_paths']=[str(self.root/'full-prepared')]
            with patch('gavd6_sjepa.research_directions.gait_fidelity.preparation._render_fixed',side_effect=AssertionError('Complete families should resume without rerendering')):
                second=prepare(cfg,self.root/'resumed')
        a,b=load_dataset(first),load_dataset(second)
        self.assertEqual(a.evidence_status,'technical-source-screen')
        self.assertTrue(all(r['locomotion_status']=='metadata_candidate' for r in a.records))
        self.assertEqual(len(a.records),len(b.records))
        np.testing.assert_array_equal(a.inputs['xy'],b.inputs['xy'])
        merged=merge_datasets([second],self.root/'full-merged')
        self.assertIsInstance(load_dataset(merged).inputs['xy'],np.memmap)


if __name__=='__main__': unittest.main()
