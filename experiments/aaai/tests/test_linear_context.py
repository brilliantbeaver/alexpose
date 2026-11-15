import numpy as np
import pytest
from aaai_gait.linear_context import patches,context_features,fit_ridge_context

def test_hidden_and_excluded_values_cannot_enter_linear_context():
    x=np.random.default_rng(1).normal(size=(2,32,16,2)).astype('float32')
    p=patches(x);mask=np.zeros((2,128),bool);mask[:,0]=True
    before=context_features(p,mask,0,0,'own')
    edited=p.copy();edited[:,np.arange(128)%16!=0]+=1000;edited[:,0]+=1000
    np.testing.assert_array_equal(before,context_features(edited,mask,0,0,'own'))

def test_ridge_rejects_abnormal_fit():
    with pytest.raises(ValueError,match='normal'):
        fit_ridge_context(np.zeros((2,32,16,2)),source_ids=['a','b'],labels=['normal','stroke'])
