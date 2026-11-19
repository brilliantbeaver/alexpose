"""External-transfer ordering and population invariants on synthetic fixtures."""
import numpy as np
import pandas as pd
import pytest

from worldmodels.v2_transfer import participant_errors,average_components
from worldmodels.v2_models import TARGET_JOINTS


def test_participant_mean_gives_equal_recording_mass_and_equal_leg_mass():
    # Participant A has two windows in recording1 and one in recording2.
    # Recording means (1,9) must average5, rather than window mean11/3.
    x=np.zeros((4,24,16,2))
    v=np.ones((4,24,16),bool)
    pred=np.zeros((4,8,8,2));pred[0:2]=1;pred[2]=3;pred[3]=2
    meta=pd.DataFrame({'source_id':['A','A','A','B'],
        'participant_id':['A','A','A','B'],'component_id':['A','A','A','B'],
        'sequence_id':['one','one','two','three'],'label':['normal','normal','normal','parkinsons']})
    scores=participant_errors(meta,x,v,pred).set_index('source_id')
    assert scores.loc['A','mse']==5
    assert scores.loc['B','mse']==4
    # Sparse left support still receives half the loss mass.
    v[3,16:,TARGET_JOINTS[:4]]=False
    v[3,16,TARGET_JOINTS[0]]=True
    pred[3,:,0:4]=1
    scores=participant_errors(meta,x,v,pred).set_index('source_id')
    assert scores.loc['B','mse']==2.5


def test_tail_average_requires_all_fifteen_identical_participant_sets():
    frame=pd.DataFrame({'source_id':['A','B'],'label':['normal','parkinsons'],
                        'component_id':['A','B'],'two_sided':[.2,.8]})
    frames=[frame.copy() for _ in range(15)]
    frames[0]['two_sided']=[.5,.5]
    result=average_components(frames,['two_sided'])
    np.testing.assert_allclose(result.two_sided,[(14*.2+.5)/15,(14*.8+.5)/15])
    with pytest.raises(ValueError,match='fifteen'):
        average_components(frames[:14],['two_sided'])
    frames[1]=frames[1].iloc[:1]
    with pytest.raises(ValueError,match='identical'):
        average_components(frames,['two_sided'])


def test_recording_identifier_cannot_be_substituted_for_participant():
    x=np.zeros((1,24,16,2));v=np.ones((1,24,16),bool);p=np.zeros((1,8,8,2))
    meta=pd.DataFrame({'source_id':['file1'],'participant_id':['person1'],
        'component_id':['person1'],'sequence_id':['file1'],'label':['normal']})
    with pytest.raises(ValueError,match='participants'):
        participant_errors(meta,x,v,p)
