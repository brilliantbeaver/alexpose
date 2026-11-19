"""Expanded-data invariants with synthetic annotations and source identities."""
import numpy as np
import pandas as pd
import pytest
from aaai_gait.study_data import (STUDY_LABELS,build_study_manifest,source_table,
                                  make_outer_folds,fold_metadata,load_study_windows)
from aaai_gait.data import CSV_COLUMNS,POSE_VERSION
import json


def inventory(tmp_path):
    for i,label in enumerate(STUDY_LABELS):
        path=tmp_path/'data/gavd'/label/f's{i}.csv';path.parent.mkdir(parents=True)
        source=f'{i:011d}'
        rows=[dict(seq=f's{i}',frame_num=str(t),cam_view='front',gait_event='',dataset='Abnormal Gait',gait_pat=label,
            bbox="{'left': 0, 'top': 0, 'width': 100, 'height': 100}",vid_info="{'width': 100, 'height': 100}",
            id=source,url=f'https://www.youtube.com/watch?v={source}') for t in range(1,65)]
        pd.DataFrame(rows,columns=CSV_COLUMNS).to_csv(path,index=False)
    source=tmp_path/'data/gavd/source';source.mkdir()
    (source/'duplicate.csv').write_text('not,an,annotation\n')
    return build_study_manifest(tmp_path)


def metadata():
    rows=[]
    for i in range(28):
        rows.append(dict(source_id=f'n{i:02}',sequence_id=f'normal{i}',label='normal',binary_label=0))
    for j,label in enumerate(c for c in STUDY_LABELS if c!='normal'):
        for i in range(5):
            rows.append(dict(source_id=f'a{j:02}{i}',sequence_id=f'{j}_{i}',label=label,binary_label=1))
    # Same recording has two abnormal categories, but is one binary unit.
    rows.append(dict(source_id='a000',sequence_id='mixed',label='style',binary_label=1))
    return pd.DataFrame(rows)


def test_fresh_all_folders_binary_rule_and_source_cache_excluded(tmp_path):
    m=inventory(tmp_path)
    assert len(m)==12 and m.n_frames.sum()==768
    assert m.loc[m.label.eq('normal'),'binary_label'].eq(0).all()
    assert m.loc[m.label.ne('normal'),'binary_label'].eq(1).all()
    assert m.loc[m.label.eq('normal'),'dataset_groups'].iloc[0]=='Abnormal Gait'
    before=m.attrs['manifest_sha256'];path=tmp_path/'data/gavd/style/s11.csv'
    table=pd.read_csv(path,dtype=str,keep_default_na=False);table.loc[0,'gait_event']='changed';table.to_csv(path,index=False)
    assert build_study_manifest(tmp_path).attrs['manifest_sha256']!=before


def test_unique_sources_balanced_folds_and_frozen_inner_roles():
    meta=metadata();r=make_outer_folds(meta)
    assert len(r)==83 and not r.source_id.duplicated().any()
    mixed=r[r.source_id.eq('a000')].iloc[0]
    assert mixed.categories=='abnormal|style'
    normal=r[r.binary_label.eq(0)].groupby('outer_fold').size()
    assert normal.tolist()==[7]*4
    abnormal=r[r.binary_label.eq(1)].groupby('outer_fold').size()
    assert abnormal.max()-abnormal.min()<=1
    pd.testing.assert_frame_equal(r,make_outer_folds(meta.sample(frac=1,random_state=9)))
    for fold in range(4):
        assigned=fold_metadata(meta,r,fold)
        assert assigned.groupby('source_id').split.nunique().max()==1
        assert assigned[assigned.split.eq('val')].source_id.nunique()==5
        assert assigned[assigned.split.eq('train')].source_id.nunique()==16
        assert assigned[assigned.split.isin(['train','val'])].binary_label.eq(0).all()
        assert assigned[assigned.split.eq('auxiliary')].fit_eligible.eq(False).all()
        assert set(assigned[assigned.split.eq('test')].source_id)==set(r[r.outer_fold.eq(fold)].source_id)


def test_fold_guard_rejects_ineligible_and_normal_abnormal_mixing():
    meta=metadata()
    with pytest.raises(ValueError,match='Filter'):make_outer_folds(meta.assign(eligible=False))
    changed=pd.concat([meta,pd.DataFrame([dict(source_id='n00',sequence_id='bad',label='stroke',binary_label=1)])])
    with pytest.raises(ValueError,match='mixes normal'):source_table(changed)
    with pytest.raises(ValueError,match='at least four'):make_outer_folds(meta,inner_normal_count=3)
    r=make_outer_folds(meta)
    with pytest.raises(ValueError,match='already frozen'):fold_metadata(meta,r,0,seed=73)
    altered=r.copy();i=altered.index[altered.outer_fold.eq(0)][0]
    altered.loc[i,'role_fold_0']='train'
    with pytest.raises(ValueError,match='Outer-test role'):fold_metadata(meta,altered,0)
    altered=r.copy();i=altered.index[altered.outer_fold.ne(0)&altered.binary_label.eq(1)][0]
    altered.loc[i,'role_fold_0']='train'
    with pytest.raises(ValueError,match='remain auxiliary'):fold_metadata(meta,altered,0)


def test_study_loader_has_no_old_five_class_coverage_requirement(tmp_path):
    manifest=inventory(tmp_path);cache=tmp_path/'poses';cache.mkdir()
    # One eligible normal sequence is enough to create a pre-fold eligibility
    # ledger. It is not enough to fit the eventual four-fold experiment.
    row=manifest[manifest.label.eq('normal')].iloc[0]
    raw=np.random.default_rng(2).normal(size=(64,33,2)).astype(np.float32)
    provenance=dict(version=POSE_VERSION,csv_sha256=row.csv_sha256,source_id=row.source_id,
                    frame_number_base=1,min_visibility=.5,bbox_margin=.05,model_sha256='synthetic')
    path=cache/f'{row.sequence_id}.npz'
    np.savez(path,raw_xy=raw,xy=raw,confidence=np.ones((64,33)),frame_numbers=np.arange(1,65),fps=15,provenance=json.dumps(provenance))
    x,meta,ledger=load_study_windows(manifest,cache)
    assert len(x)==3 and 'split' not in meta and ledger.eligible.sum()==1
    before=meta.attrs['corpus_sha256'];raw[2,25,0]+=.1
    np.savez(path,raw_xy=raw,xy=raw,confidence=np.ones((64,33)),frame_numbers=np.arange(1,65),fps=15,provenance=json.dumps(provenance))
    assert load_study_windows(manifest,cache)[1].attrs['corpus_sha256']!=before
