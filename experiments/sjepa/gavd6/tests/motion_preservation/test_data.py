"""motion preservation / test data."""


from pathlib import Path
import tempfile
import unittest

import numpy as np
import pandas as pd

from gavd6_sjepa.research_directions.motion_preservation.body_geometry import (
    AMASS_TO_Y_UP,
    BodySequence,
    PARENTS,
    demo_motion,
    descriptor,
    estimate_bone_lengths,
    project_bone_lengths,
)
from gavd6_sjepa.research_directions.motion_preservation.motion_data import (
    load_amass_manifest,
    load_motion,
    make_observation_cases,
    tracking_noise,
)
from gavd6_sjepa.research_directions.motion_preservation.rendering import Camera, render_sequence

# Scientific-construction checks; no licensed data or checkpoints required.


class ObservationConstructionTests(unittest.TestCase):
    def test_exact_pair_and_full_factorial(self):
        clean,event=demo_motion(frames=16)
        noise=tracking_noise(clean.joints.shape,joints=(18,20))
        observed=np.ones((16,22),bool);observed[4:7,20]=False
        cases=make_observation_cases(clean.joints,event.joints,noise,observed=observed)
        self.assertEqual({(c['event_present'],c['noise_present']) for c in cases[:4]},
                         {(False,False),(False,True),(True,False),(True,True)})
        for key in ('raw','observed','confidence'):
            np.testing.assert_array_equal(cases[4][key],cases[5][key])
        self.assertFalse(np.array_equal(cases[4]['truth'],cases[5]['truth']))
        np.testing.assert_array_equal(cases[3]['raw'][~observed],0)
        # Noise is identical with and without a real event, including overlap.
        np.testing.assert_allclose((cases[3]['raw']-cases[2]['raw'])[observed],
                                   (cases[1]['raw']-cases[0]['raw'])[observed],atol=1e-7)

    def test_angle_events_keep_lengths_and_have_nonzero_descriptors(self):
        for family in ('arm_leg_timing','foot_clearance','trunk_pelvis_timing'):
            clean,event=demo_motion(frames=32,family=family)
            clean_lengths=np.linalg.norm(clean.joints[:,1:]-clean.joints[:,PARENTS[1:]],axis=-1)
            event_lengths=np.linalg.norm(event.joints[:,1:]-event.joints[:,PARENTS[1:]],axis=-1)
            np.testing.assert_allclose(clean_lengths,event_lengths,atol=1e-6)
            self.assertGreater(abs(descriptor(clean.joints,family)-descriptor(event.joints,family)),.001)

    def test_projection_uses_passed_lengths_and_preserves_root(self):
        clean,_=demo_motion(frames=8)
        lengths=estimate_bone_lengths(clean.joints)
        noisy=clean.joints+tracking_noise(clean.joints.shape)
        projected=project_bone_lengths(noisy,lengths)
        np.testing.assert_array_equal(projected[:,0],noisy[:,0])
        actual=np.linalg.norm(projected[:,1:]-projected[:,PARENTS[1:]],axis=-1)
        np.testing.assert_allclose(actual,np.broadcast_to(lengths[1:],actual.shape),atol=1e-6)
        np.testing.assert_allclose(AMASS_TO_Y_UP@AMASS_TO_Y_UP.T,np.eye(3))
        self.assertAlmostEqual(np.linalg.det(AMASS_TO_Y_UP),1)

class ManifestTests(unittest.TestCase):
    def test_aliases_share_roles_and_existing_test_is_final(self):
        with tempfile.TemporaryDirectory() as temporary:
            folder=Path(temporary)
            candidates=['alias_a','alias_b','cal','dev','held']
            pd.DataFrame(dict(subject_id_candidate=candidates,source_dataset=['D']*5,
                              relative_path=[f'{c}/motion.npz' for c in candidates],
                              num_frames=[200]*5,mocap_framerate=[20]*5,status=['ok']*5)).to_csv(
                                  folder/'amass_raw_inventory_eligible.csv',index=False)
            pd.DataFrame(dict(subject_id_candidate=candidates,identity=['same','same','c','d','h'],
                              identity_audit_status=['approved']*5,excluded=[False]*5)).to_csv(
                                  folder/'amass_subject_registry.csv',index=False)
            pd.DataFrame(dict(identity=['same','c','d','h'],split=['train','validation','validation','test'])).to_csv(
                folder/'amass_subject_splits.csv',index=False)
            table=load_amass_manifest(folder,folder/'raw')
            self.assertEqual(table.loc[table.person_id.eq('same'),'role'].unique().tolist(),['train'])
            self.assertEqual(table.loc[table.person_id.eq('h'),'role'].tolist(),['final'])
            self.assertEqual(set(table.loc[table.person_id.isin(['c','d']),'role']),{'calibration','development'})
            self.assertFalse(table.available.any())

    def test_so3_resampling_crosses_pi_without_folding_to_zero(self):
        with tempfile.TemporaryDirectory() as temporary:
            path=Path(temporary)/'motion.npz'
            poses=np.zeros((3,156),np.float32)
            poses[:,2]=np.deg2rad([179,-179,-177])
            np.savez(path,poses=poses,trans=np.array([[0,0,0],[1,0,0],[2,0,0]],np.float32),
                     betas=np.zeros(16,np.float32),dmpls=np.zeros((3,8),np.float32),
                     gender='male',mocap_framerate=2.)
            motion=load_motion({'raw_path':str(path),'person_id':'p'},duration_s=.75,fps=4)
            self.assertAlmostEqual(abs(motion.poses[1,2]),np.pi,places=5)
            np.testing.assert_allclose(motion.timestamps,[0,.25,.5])
            np.testing.assert_allclose(motion.trans[:,0],[0,.5,1])

class MeshTransportTests(unittest.TestCase):
    def test_reference_flow_matches_translated_surface(self):
        vertices=np.array([[[-.5,-.5,0],[.5,-.5,0],[0,.5,0]],
                           [[-.4,-.5,0],[.6,-.5,0],[.1,.5,0]]],np.float32)
        body=BodySequence(np.zeros((2,22,3),np.float32),vertices,np.array([[0,1,2]]),20,
                          np.array([0,.05]),{'evidence_origin':'unit_geometry'})
        camera=Camera.look_at(target=(0,0,0),eye=(0,0,3),width=48,height=48,focal_px=48)
        rendered=render_sequence(body,camera)
        self.assertGreater(rendered.flow_valid.sum(),30)
        expected=np.array([48*.1/3,0])
        np.testing.assert_allclose(rendered.flow[rendered.flow_valid],
                                   np.broadcast_to(expected,rendered.flow[rendered.flow_valid].shape),atol=1e-5)
        blocked=render_sequence(body,camera,occlusion=(0,0,1,1))
        self.assertFalse(blocked.flow_valid.any())
        self.assertFalse(blocked.joint_visible.any())
        self.assertFalse(blocked.foreground.any())


if __name__ == "__main__":
    unittest.main()
