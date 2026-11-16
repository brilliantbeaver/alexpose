"""Regression for annotation gaps changing the actual window sampling interval."""
import numpy as np
from worldmodels.pose import make_windows
from worldmodels.demo import make_demo


def test_window_fps_uses_actual_frame_spacing():
    example=make_demo(n=1,frames=16)['x'][0]
    pose=dict(xy=example,confidence=np.ones(example.shape[:-1]),frame_numbers=1+4*np.arange(16),
        fps=np.array(30.),sample_fps=np.array(15.),sequence_id='s',source_id='v',label='normal',fingerprint='fixture')
    x,valid,meta=make_windows(pose)
    assert len(x)==1
    assert meta.iloc[0].sample_fps==7.5
