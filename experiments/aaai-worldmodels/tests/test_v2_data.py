"""Cohort eligibility, linked-source partitioning, and reuse identities."""
import json

import numpy as np
import pandas as pd
import pytest

from worldmodels import v2_data
from worldmodels.data import sha256_file


def test_common_target_gate_preserves_missingness_and_both_legs():
    v=np.ones((4,24,16),bool)
    v[1,16:20,v2_data.LEGS[0]]=False
    v[2,:16,v2_data.LEGS[1]]=False
    v[3,16:,v2_data.LEGS[0]]=False
    v[3,16:,v2_data.LEGS[0][0]]=True
    original=v.copy()
    np.testing.assert_array_equal(v2_data.common_support(v),[True,False,False,True])
    np.testing.assert_array_equal(v,original)
    with pytest.raises(ValueError):
        v2_data.common_support(v.astype(int))


def test_components_only_join_verified_identical_footage():
    media=pd.DataFrame([('a','hash1'),('b','hash1'),('b','hash2'),('c','hash2')],columns=['source_id','sha256'])
    components,edges=v2_data.relation_components(['a','b','c','d'],media)
    assert components=={'a':'a','b':'a','c':'a','d':'d'}
    assert len(edges)==2
    assert all(e['relation']=='byte_identical_original_footage' for e in edges)


def test_linked_sources_stay_in_one_fold_and_mixed_labels_fail():
    rows=[]
    for label in ['normal','stroke']:
        for i in range(6):
            rows.append(dict(source_id=f'{label}{i}',component_id=f'{label}{i}',label=label))
    rows.append(dict(source_id='normal_alias',component_id='normal0',label='normal'))
    meta=pd.DataFrame(rows)
    folds=v2_data.component_folds(meta,n_splits=3)
    assert folds[0]==folds[-1]
    np.testing.assert_array_equal(folds,v2_data.component_folds(meta,n_splits=3))
    meta.loc[len(meta)-1,'label']='stroke'
    with pytest.raises(ValueError,match='Mixed-label'):
        v2_data.component_folds(meta,n_splits=3)


def test_loader_reads_only_declared_rows_and_rejects_changed_inputs(tmp_path,monkeypatch):
    prepared=tmp_path/'prepared';prepared.mkdir()
    x=np.zeros((3,24,16,2),np.float32);x[1]=1;x[2]=2
    valid=np.ones((3,24,16),bool)
    np.savez(prepared/'windows.npz',x=x,valid=valid)
    root=tmp_path/'new';root.mkdir()
    times=json.dumps((np.arange(24)*.1).tolist())
    m=pd.DataFrame([dict(source_id='s',sequence_id='a',label='normal',window_id='one',prepared_index=2,component_id='s',outer_fold=0,timestamp_seconds=times),
                    dict(source_id='t',sequence_id='b',label='stroke',window_id='two',prepared_index=0,component_id='t',outer_fold=1,timestamp_seconds=times)])
    m.to_csv(root/'cohort.csv',index=False)
    identity=dict(prepared_directory='prepared',prepared_inputs={'windows.npz':sha256_file(prepared/'windows.npz')},cohort_sha256=sha256_file(root/'cohort.csv'))
    (root/'identity.json').write_text(json.dumps(identity))
    monkeypatch.setattr(v2_data,'REPOSITORY',tmp_path)
    values,mask,meta,t=v2_data.load_cohort(root)
    np.testing.assert_array_equal(values,x[[2,0]])
    assert t.shape==(2,24) and mask.all()
    with (prepared/'windows.npz').open('ab') as stream:
        stream.write(b'changed')
    with pytest.raises(ValueError,match='hash mismatch'):
        v2_data.load_cohort(root)
