"""Controlled objective change and synthetic full workflow; no clinical claim."""
from dataclasses import asdict,replace
import json
import numpy as np
import pandas as pd
import pytest
import torch
from worldmodels.data import sha256_file
from worldmodels.demo import make_demo
from worldmodels.revision import RevisionProtocol,develop as primary_develop,freeze as primary_freeze,evaluate_once as primary_evaluate
from worldmodels.objective_audit import declared_change,develop,freeze,evaluate_once

def test_exactly_one_declared_field_changes():
    p=RevisionProtocol();q=declared_change(p)
    assert q.coordinate_weight==10
    assert {k for k in asdict(p) if asdict(p)[k]!=asdict(q)[k]}=={'coordinate_weight'}
    with pytest.raises(ValueError): declared_change(replace(p,coordinate_weight=2))

def test_matched_secondary_pipeline_and_one_final_access(tmp_path):
    old=torch.get_num_threads();torch.set_num_threads(1)
    try:
        data=make_demo(n=9,frames=4);input_dir=tmp_path/'prepared';input_dir.mkdir()
        np.savez_compressed(input_dir/'windows.npz',x=data['x'],valid=data['valid'])
        meta=pd.DataFrame(dict(source_id=data['source_ids'],sequence_id=[f'q{i}' for i in range(9)],
            label='normal',split=['train']*3+['validation']*3+['test']*3,sample_fps=15.))
        meta.to_csv(input_dir/'windows.csv',index=False)
        (input_dir/'preparation.json').write_text(json.dumps(dict(context_frames=2,
            data_sha256=sha256_file(input_dir/'windows.npz'),metadata_sha256=sha256_file(input_dir/'windows.csv'))))
        p=RevisionProtocol(identity='synthetic-objective-audit-fixture',frames=4,context_frames=2,horizons=(1,2),
            seeds=(7,),dim=16,encoder_layers=1,pretrain_steps=1,forecast_steps=1,batch_size=2,prior_samples=2)
        reference=tmp_path/'primary';primary_develop(input_dir,reference,p);primary_freeze(reference)
        audit=tmp_path/'audit';result=develop(reference,audit)
        assert result['declaration']['changed_field']=={'coordinate_weight':[1.,10.]}
        assert len(result['records'])==1
        saved=torch.load(audit/'seed_7/residual_det_weight10.pt',weights_only=True)
        assert saved['metadata']['source_schedule_sha256']==sha256_file(reference/'seed_7/source_schedule.json')
        assert saved['metadata']['parent_pretrain_sha256']==sha256_file(reference/'seed_7/uniform_pretrain.pt')
        assert not (audit/'test_access.json').exists()
        freeze(audit)
        with pytest.raises(FileNotFoundError): evaluate_once(audit,tmp_path/'not-released',tmp_path/'forbidden')
        assert not (tmp_path/'forbidden').exists()
        primary_evaluate(reference,tmp_path/'primary-test')
        final=evaluate_once(audit,tmp_path/'primary-test',tmp_path/'audit-test')
        assert final['status']=='secondary_final_complete'
        assert final['primary_unchanged'] is True
        with pytest.raises(FileExistsError): evaluate_once(audit,tmp_path/'primary-test',tmp_path/'repeat')
    finally:
        torch.set_num_threads(old)
