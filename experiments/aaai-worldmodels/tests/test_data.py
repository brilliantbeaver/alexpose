from pathlib import Path
import numpy as np
import pandas as pd
import pytest
from worldmodels.data import LABELS, JOINT_IDS, assign_splits, build_inventory, save_inventory
from worldmodels.pose import normalize_window, make_windows


def fixture_table():
    return pd.DataFrame([dict(source_id=f'{label}-{i}',sequence_id=f'{label}-{i}-{j}',label=label)
                         for label in LABELS for i in range(6) for j in range(3)])


def skeleton():
    xy = np.zeros((16,16,2))
    xy[:,:,0] = np.arange(16)
    for joint in (11,12):
        xy[:,JOINT_IDS.index(joint),1] = -5
    return xy, np.ones((16,16))


def test_partition_is_source_disjoint_and_reproducible():
    table=fixture_table()
    a,b=assign_splits(table),assign_splits(table.sample(frac=1,random_state=3))
    assert a.groupby('source_id').split.nunique().max()==1
    assert a.groupby('source_id').split.first().equals(b.groupby('source_id').split.first())
    assert set(a.split)=={'train','validation','test'}
    assert a.groupby(['label','split']).size().size==15


def test_mixed_label_source_fails():
    table=fixture_table(); table.loc[0,'label']='stroke'
    with pytest.raises(ValueError,match='Mixed-label'):
        assign_splits(table)


def test_normalization_uses_no_future_scale():
    xy,conf=skeleton(); original,valid=normalize_window(xy,conf)
    xy[8:]*=100
    changed,_=normalize_window(xy,conf)
    np.testing.assert_array_equal(original[:8],changed[:8])
    assert valid.all()


def test_missing_pelvis_is_missing_frame():
    xy,conf=skeleton(); conf[3,JOINT_IDS.index(23)]=0
    normalized,valid=normalize_window(xy,conf)
    assert not valid[3].any()
    assert (normalized[3]==0).all()
    conf[:8,JOINT_IDS.index(11)]=0
    with pytest.raises(ValueError,match='context'):
        normalize_window(xy,conf)


def test_fresh_real_inventory_and_frozen_publication(tmp_path):
    repo=Path(__file__).resolve().parents[3]
    table,audit=build_inventory(repo)
    assert audit['all_sequences']==1874
    assert audit['all_rows']==458116
    assert audit['selected_sequences']==666
    assert audit['selected_sources']==103
    assert table.sequence_id.is_unique
    assert not table.loc[table.source_id=='2LbERGI34lA','eligible'].any()
    split=assign_splits(table)
    save_inventory(split,audit,tmp_path)
    changed=split.copy(); changed.loc[0,'label']='changed'
    with pytest.raises(FileExistsError):
        save_inventory(changed,audit,tmp_path)
