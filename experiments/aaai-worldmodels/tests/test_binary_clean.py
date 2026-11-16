"""The declared primary must survive an adversarial validation ranking."""
import json
import joblib
import numpy as np
import pandas as pd
import pytest
from sklearn.dummy import DummyClassifier
from worldmodels import binary_clean


def test_primary_mixed_is_fixed_even_when_every_alternative_ranks_higher():
    choices={name:{'validation_binary_ba':1.0} for name in ['rf',*binary_clean.ENCODERS]}
    choices['jepa_mixed']['validation_binary_ba']=0.0
    assert binary_clean.fixed_primary_jepa(choices)=='jepa_mixed'
    del choices['jepa_mixed']
    with pytest.raises(ValueError,match='mixed'): binary_clean.fixed_primary_jepa(choices)


def test_develop_scores_validation_only_and_seals_fixed_primary(tmp_path,monkeypatch):
    prepared=tmp_path/'prepared';baseline=tmp_path/'baseline';prepared.mkdir();baseline.mkdir()
    x=np.zeros((12,4,16,2),np.float32);v=np.ones(x.shape[:-1],bool)
    meta=pd.DataFrame(dict(source_id=[f's{i}' for i in range(12)],sequence_id=[f'q{i}' for i in range(12)],
        label=['normal','stroke']*6,split=['train']*4+['validation']*4+['test']*4,sample_fps=15.))
    np.savez_compressed(prepared/'windows.npz',x=x,valid=v)
    meta.to_csv(prepared/'windows.csv',index=False);(prepared/'preparation.json').write_text('{}')
    monkeypatch.setattr(binary_clean,'load_prepared',lambda *_:(x,v,meta))
    monkeypatch.setattr(binary_clean,'baseline_lineage',lambda *_:{'synthetic_fixture':True})
    def matrix(name,x,valid,metadata,baseline):
        assert set(metadata.split)=={'train','validation'}
        assert len(x)==8
        return np.arange(len(x)*2).reshape(len(x),2)
    monkeypatch.setattr(binary_clean,'matrix',matrix)
    seen=[]
    def factory(features,labels,sources,seed,**config):
        assert set(labels)=={'normal','abnormal'}
        assert len(features)==4 and seed==27
        seen.append(config)
        return DummyClassifier(strategy='prior').fit(features,labels)
    monkeypatch.setattr(binary_clean,'fit_rf',factory);monkeypatch.setattr(binary_clean,'fit_probe',factory)
    for name in ['rf',*binary_clean.ENCODERS]:
        old=DummyClassifier(strategy='uniform').fit(np.zeros((5,2)),['normal','parkinsons','stroke','cerebral palsy','myopathic'])
        joblib.dump(old,baseline/f'{name}_classifier.joblib')
    result=binary_clean.develop(prepared,baseline,tmp_path/'development')
    assert result['primary_jepa']=='jepa_mixed'
    assert result['identity']=='direct-binary-person-reviewed-clean-v1'
    assert result['test_metrics_inspected'] is False
    assert seen[:2]==[{'max_depth':4},{'max_depth':8}]
    assert seen[2:6]==[{'C':.01},{'C':.1},{'C':1.},{'C':10.}]
    assert (tmp_path/'development/seal.json').is_file()


def test_excluded_source_cannot_reenter_development(monkeypatch,tmp_path):
    m=pd.DataFrame(dict(source_id=['CMls0LwK5OU']))
    monkeypatch.setattr(binary_clean,'load_prepared',lambda *_:(np.zeros((1,4,16,2)),np.ones((1,4,16),bool),m))
    with pytest.raises(ValueError,match='Excluded'):
        binary_clean.develop(tmp_path,tmp_path,tmp_path/'output')
