"""Guards that real tutorial mode cannot silently use synthetic or test inputs."""
import numpy as np
import pandas as pd
import pytest
from worldmodels.data import LABELS
from worldmodels import notebook_support as support


def test_real_lesson_selection_stays_in_training(monkeypatch,tmp_path):
    rows=[]
    for split in ('train','validation','test'):
        for label in LABELS:
            for source in range(3):
                rows.append(dict(split=split,label=label,source_id=f'{split}-{label}-{source}',sequence_id=f'{split}-{label}-{source}-clip',first_frame=1,sample_fps=15))
    metadata=pd.DataFrame(rows)
    x=np.zeros((len(rows),16,16,2),np.float32);valid=np.ones(x.shape[:-1],bool)
    monkeypatch.setenv('WM_DATA_MODE','real')
    monkeypatch.setattr(support,'real_inputs',lambda:(x,valid,metadata))
    monkeypatch.setattr(support,'prepared_path',lambda:tmp_path)
    (tmp_path/'windows.npz').write_bytes(b'unit-test-identity')
    (tmp_path/'windows.csv').write_text('unit-test-identity')
    result=support.lesson_data(n=10,conditions=True)
    assert set(result['metadata'].split)=={'train'}
    assert set(result['labels'])==set(LABELS)
    assert len(result['source_ids'])==len(set(result['source_ids']))==10
    assert result['evidence_type']=='real GAVD prepared poses'


def test_real_mode_failure_never_falls_back_to_demo(monkeypatch):
    monkeypatch.setenv('WM_DATA_MODE','real')
    def missing(): raise FileNotFoundError('Real cohort missing')
    monkeypatch.setattr(support,'real_inputs',missing)
    with pytest.raises(FileNotFoundError,match='Real cohort'):
        support.lesson_data()


def test_unknown_data_mode_is_rejected(monkeypatch):
    monkeypatch.setenv('WM_DATA_MODE','guess')
    with pytest.raises(ValueError,match='WM_DATA_MODE'):
        support.data_mode()
