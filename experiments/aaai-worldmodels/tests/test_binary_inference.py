"""Binary column alignment, exact threshold, quality and artifact guards."""
import json
from pathlib import Path
import numpy as np
import pytest
from worldmodels import binary_inference as module
from worldmodels.demo import make_demo

@pytest.fixture
def fixture(tmp_path,monkeypatch):
    development=tmp_path/'binary';development.mkdir();(development/'selection.json').write_text('{}')
    spec=dict(identity='direct-binary-person-reviewed-clean-v1',baseline='fixture',
              choices={n:{'classifier_sha256':'fixture'} for n in ('rf','jepa_mixed')})
    monkeypatch.setattr(module,'verify_development',lambda *_:(spec,{'pose_model_sha256':'fixed'}))
    d=make_demo(n=1,frames=24)
    pose=dict(xy=d['x'][0]*100+[300,200],confidence=np.ones((24,16)),frame_numbers=np.arange(1,49,2),
        fps=np.array(30.),sequence_id='unlabeled',source_id='fixture',label='',fingerprint='fixture',
        provenance={'pose_model_sha256':'fixed'})
    def matrix(name,x,valid,meta,baseline):
        assert x.shape==(2,16,16,2)
        assert meta.label.tolist()==['','']
        return np.zeros((len(x),2))
    monkeypatch.setattr(module,'matrix',matrix)
    class Estimator:
        classes_=np.array(['abnormal','normal'])  # Reverse the declared report order.
        def __init__(self,name): self.name=name
        def predict_proba(self,x):
            return np.array([[.8,.2],[.6,.4]]) if self.name=='rf' else np.array([[.2,.8],[.4,.6]])
    monkeypatch.setattr(module.joblib,'load',lambda p:Estimator(Path(p).stem))
    return development,pose

def test_true_binary_probabilities_are_aligned_and_averaged(fixture):
    path,pose=fixture;report=module.classify_pose(pose,path)
    rf=report['classifiers']['rf'];mixed=report['classifiers']['jepa_mixed']
    assert rf['probabilities']==pytest.approx({'normal':.3,'abnormal':.7})
    assert mixed['probabilities']==pytest.approx({'normal':.7,'abnormal':.3})
    assert rf['predicted_annotation']=='abnormal' and mixed['predicted_annotation']=='normal'
    assert report['decision_threshold']==.5 and report['primary_jepa']=='jepa_mixed'
    assert report['person_review_status']=='required'
    json.dumps(report,allow_nan=False)

def test_quality_failure_produces_no_classifiers(fixture):
    path,pose=fixture;pose['confidence'][:]=0
    report=module.classify_pose(pose,path)
    assert report['status']=='quality_failure' and report['classifiers']=={}

def test_pose_weight_mismatch_is_rejected(fixture):
    path,pose=fixture;pose['provenance']['pose_model_sha256']='changed'
    with pytest.raises(ValueError,match='pose model'): module.classify_pose(pose,path)

def test_changed_selection_is_rejected_before_loading(tmp_path,monkeypatch):
    (tmp_path/'selection.json').write_text('{}')
    (tmp_path/'seal.json').write_text(json.dumps({'selection_sha256':'0'*64}))
    monkeypatch.setattr(module.joblib,'load',lambda *_:pytest.fail('Changed artifacts must not be deserialized'))
    with pytest.raises(ValueError,match='selection'): module.verify_development(tmp_path)

def test_probability_tie_uses_declared_abnormal_boundary(fixture,monkeypatch):
    path,pose=fixture
    class EqualEstimator:
        classes_=np.array(['normal','abnormal'])
        def predict_proba(self,x): return np.full((len(x),2),.5)
    monkeypatch.setattr(module.joblib,'load',lambda *_:EqualEstimator())
    report=module.classify_pose(pose,path)
    assert all(v['predicted_annotation']=='abnormal' for v in report['classifiers'].values())
