import copy
import tempfile
from pathlib import Path
import unittest
from types import SimpleNamespace
import numpy as np
from gavd6_sjepa.research_directions.synthetic_training_v2.contracts import (JOINTS,schema_permutation,cache_identity,TrackBundle)
from gavd6_sjepa.research_directions.synthetic_training_v2.data import (fixed_time_grid,sample_observations,audit_pair,PAIR_FACTORS,shuffled_donors,fixture_bundle,normalize_inputs)
from gavd6_sjepa.research_directions.synthetic_training_v2.extraction import extract_tracks,perturb_boxes

class DataTests(unittest.TestCase):
    def test_named_schema_permutation_no_invented_joints(self):
        self.assertEqual(schema_permutation(JOINTS[::-1]).tolist(),list(range(11,-1,-1)))
        for names in (JOINTS[:-1],JOINTS[:-1]+('nose',),JOINTS[:-1]+(JOINTS[0],)):
            with self.assertRaises(ValueError):schema_permutation(names)
    def test_physical_duration_missing_frames_and_nearest_clock(self):
        t=np.arange(100)/30
        grid=fixed_time_grid(t,.2)
        self.assertAlmostEqual(grid[-1]-grid[0],2.52)
        with self.assertRaises(ValueError):fixed_time_grid(t,2.)
        xy=np.ones((100,12,2));scores=np.ones((100,12));observed=np.ones((100,12),bool);observed[1,0]=False
        result,actual=sample_observations(xy,scores,observed,t,0.)
        self.assertFalse(result['observed'][1,0]);self.assertTrue(np.isnan(result['xy'][1,0]).all())
        self.assertNotEqual(actual[1],result['timestamps'][1])
    def test_pair_mismatch_camera_and_motion(self):
        clean={k:0 for k in PAIR_FACTORS};clean.update(blur_px=0,target_hash='original')
        changed={**clean,'blur_px':2}
        self.assertEqual(audit_pair(clean,changed,['blur_px'])['status'],'pass')
        with self.assertRaises(ValueError):audit_pair(clean,{**changed,'motion_hash':3},['blur_px'])
        with self.assertRaises(ValueError):audit_pair(clean,{**clean,'camera_hash':1},['camera_hash'])
    def test_bundle_content_identity_and_target_isolation(self):
        bundle=fixture_bundle();before,norm=normalize_inputs(bundle.inputs)
        different=copy.deepcopy(bundle);different.targets['xy']*=100;different.targets['valid'][:]=False
        after,other=normalize_inputs(different.inputs)
        for key in before:np.testing.assert_equal(before[key],after[key])
        with tempfile.TemporaryDirectory() as d:
            folder=Path(d)/'bundle';bundle.save(folder)
            with self.assertRaises(ValueError):TrackBundle.load(folder,expected_identity='stale')
            with (folder/'inputs.npz').open('ab') as f:f.write(b'tampered')
            with self.assertRaises(ValueError):TrackBundle.load(folder)
        args=dict(source_hash='source',checkpoint_hash='checkpoint',config={'fps':25},code_hash='code')
        self.assertNotEqual(cache_identity(**args),cache_identity(**{**args,'code_hash':'changed'}))
    def test_shuffled_donors_distinct_window_same_strata(self):
        rows=fixture_bundle().subset('train').records
        for i,j in enumerate(shuffled_donors(rows,17)):
            self.assertNotEqual(rows[i]['window_id'],rows[j]['window_id'])
            for key in ('canonical_person_id','variant','extractor'):self.assertEqual(rows[i][key],rows[j][key])
    def test_extraction_actual_scores_missing_detection_and_no_label_argument(self):
        class Model:
            def eval(self):pass
            def test_step(self,batch):
                return [SimpleNamespace(pred_instances=SimpleNamespace(keypoints=np.ones((1,17,2)),keypoint_scores=np.linspace(0,1,17)[None])),
                        SimpleNamespace(pred_instances=SimpleNamespace(keypoints=np.empty((0,17,2))))]
        estimator=SimpleNamespace(model=Model(),_batch=lambda images,boxes:None)
        result=extract_tracks(estimator,[np.zeros((8,8,3),np.uint8)]*2,np.array([[0,0,8,8]]*2),[0,.04],box_source='supplied_fixture',threshold=.5)
        np.testing.assert_equal(result.confidence[0],np.linspace(0,1,17)[5:])
        self.assertEqual(result.status_counts['missing_detections'],1);self.assertFalse(result.observed[1].any())
        self.assertTrue(np.isnan(result.confidence[1]).all())
        boxes=np.array([[1,2,11,22.]])
        np.testing.assert_equal(perturb_boxes(boxes),boxes)
        self.assertFalse(np.array_equal(perturb_boxes(boxes,shift_fraction=.1),boxes))

if __name__=='__main__':unittest.main()
