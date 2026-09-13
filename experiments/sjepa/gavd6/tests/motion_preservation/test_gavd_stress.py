"""motion preservation / test gavd stress."""


from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import cv2
import numpy as np
import pandas as pd

from gavd6_sjepa.research_directions.motion_preservation.gavd_stress import (
    decode_gavd_window,
    discover_reservations,
    gavd_stress,
    load_pose_overlay,
)

# GAVD sampling and observability checks, using no actual clinical videos.


class GavdStressTests(unittest.TestCase):
    def test_source_reservation_keeps_confirmation_out(self):
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary)
            path=root/'reservation.csv'
            pd.DataFrame(dict(video_id=['held','open'],role=['confirmation','development'])).to_csv(path,index=False)
            _,roster=discover_reservations(SimpleNamespace(root=root,gavd_reservation_csv=str(path)))
            self.assertEqual(roster.set_index('video_id').reserved.to_dict(),{'held':True,'open':False})

    def test_missing_reservation_does_not_decode_or_load_models(self):
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary)
            manifest=root/'manifests';manifest.mkdir()
            pd.DataFrame(dict(sequence_id=['s'],video_id=['v'],first_frame=[1],last_frame=[50])).to_csv(
                manifest/'gavd_full_sequences.csv',index=False)
            cfg=SimpleNamespace(root=root/'run',mode='real',gavd_manifest_dir=str(manifest),
                                gavd_video_root=str(root/'videos'),gavd_reservation_csv=None)
            with patch('gavd6_sjepa.research_directions.motion_preservation.gavd_stress.discover_reservations',return_value=([],None)), \
                 patch('gavd6_sjepa.research_directions.motion_preservation.gavd_stress.OpticalFlowEstimator') as model:
                report=gavd_stress(cfg)
            self.assertEqual(report.iloc[0].status,'not_run_missing_source_reservation')
            model.assert_not_called()

    def test_manifest_one_based_frame_and_actual_time_alignment(self):
        with tempfile.TemporaryDirectory() as temporary:
            path=Path(temporary)/'index.avi'
            writer=cv2.VideoWriter(str(path),cv2.VideoWriter_fourcc(*'MJPG'),30,(32,24))
            self.assertTrue(writer.isOpened())
            for index in range(30):
                writer.write(np.full((24,32,3),index*5,np.uint8))
            writer.release()
            rgb,frames,timestamps,info=decode_gavd_window(
                {'video_path':str(path),'first_frame':4,'last_frame':25},duration_s=.2,target_fps=20,max_side=32)
            np.testing.assert_array_equal(frames,[3,5,6,7])  # round-to-even nearest source frame
            np.testing.assert_allclose(timestamps,frames/30)
            self.assertAlmostEqual(float(rgb[0].mean()),15,delta=2)
            self.assertEqual(info['manifest_frame_base'],1)
            self.assertEqual(info['cache_frame_base'],0)

    def test_pose_overlay_requires_original_frame_geometry(self):
        with tempfile.TemporaryDirectory() as temporary:
            path=Path(temporary)/'pose.npz'
            points=np.full((2,22,2),[20,30],np.float32)
            np.savez(path,source_frames=[10,12],joints2d=points,coordinate_system='full_frame_pixels')
            result=load_pose_overlay(path,[10,11,12],[.5,.25])['joints2d']
            np.testing.assert_allclose(result[0],np.broadcast_to([10,7.5],(22,2)))
            self.assertTrue(np.isnan(result[1]).all())
            np.savez(path,source_frames=[10,12],joints3d=np.zeros((2,22,3)))
            with self.assertRaisesRegex(ValueError,'Pose overlay needs'):
                load_pose_overlay(path,[10,12],[1,1])


if __name__ == "__main__":
    unittest.main()
