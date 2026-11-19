"""Exercise scientific-role separation and paired scoring through the runner."""
from pathlib import Path
import json

import numpy as np
import pandas as pd
import pytest
import torch

from aaai_gait import study_runner as runner
from aaai_gait.context_prediction import ContextBundle, ContextConfig, load_context_bundle
from aaai_gait.linear_context import fit_ridge_context
from aaai_gait.model import SJEPA
from aaai_gait.training import synthetic_windows


def small_config():
    torch.set_num_threads(2)
    return ContextConfig(pretrain_updates=1,predictor_updates=1,batch_size=4,
                         validation_interval=1,validation_batches=1,dim=16,depth=2,heads=2)


def test_scoring_chunk_size_preserves_global_rows_masks_and_ridge_support():
    cfg=small_config();x=synthetic_windows(5);meta=pd.DataFrame({'row':range(len(x))})
    bundle=ContextBundle(SJEPA(cfg.model_config()),cfg).eval()
    ridge=fit_ridge_context(x,source_ids=['a','a','b','b','c'],labels=['normal']*5,
                            config=cfg,draws=64)
    first=runner._score_batches(bundle,x,meta,masks=3,ridge=ridge,chunk_size=2)
    second=runner._score_batches(bundle,x,meta,masks=3,ridge=ridge,chunk_size=5)
    for a,b in zip(first,second):
        pd.testing.assert_frame_equal(a,b,atol=2e-5,rtol=2e-5)
        assert set(a.window_index)==set(range(5))
    coordinate=first[0][first[0].target_type.eq('coordinate')].reset_index(drop=True)
    keys=['window_index','joint','patch_index','target_count','requested_count']
    pd.testing.assert_frame_equal(coordinate[keys],first[1][keys])
    ridge.counts[(2,0)]=7
    with pytest.raises(ValueError,match='prespecified ridge_draws'):
        runner._score_batches(bundle,x,meta,masks=3,ridge=ridge)


def test_score_cache_rejects_changed_bytes_and_provenance(tmp_path):
    path=tmp_path/'losses.csv.gz';table=pd.DataFrame({'window_index':[0,1]})
    identity={'run_id':'abc','mask_seed':17}
    runner._write_score_cache(path,table,identity)
    pd.testing.assert_frame_equal(runner._read_score_cache(path,identity),table)
    with pytest.raises(ValueError,match='provenance'):
        runner._read_score_cache(path,dict(identity,mask_seed=18))
    with path.open('ab') as f:f.write(b'changed')
    with pytest.raises(ValueError,match='contents changed'):
        runner._read_score_cache(path,identity)


def test_single_fold_fit_reference_test_roles_and_checkpoint_resume(tmp_path,monkeypatch):
    cfg=small_config();seen=[]
    # Full model fitting, ridge fitting, scoring, aggregation and references run.
    # These two independent reporting paths have their own smoke coverage.
    def no_summary(*args,**kwargs):raise AssertionError('Parallel-fold mode must defer summary')
    monkeypatch.setattr(runner,'summarize_study',no_summary)
    def capstone(bundle,x,metadata,folder,seed):
        seen.append(metadata.copy())
        assert metadata.loc[metadata.split.eq('train'),'label'].eq('normal').all()
        assert metadata.loc[metadata.split.eq('val'),'label'].eq('normal').all()
    monkeypatch.setattr(runner,'condition_capstone',capstone)
    out=runner.run_study(demo=True,config=cfg,mask_repeats=16,ridge_draws=64,
                         include_untrained=True,folds=(0,),output_root=tmp_path,finalize=False)
    folder=out/'fold_0';roles=json.loads((folder/'source_roles.json').read_text())
    for a in ('train','val','test','auxiliary'):
        for b in ('train','val','test','auxiliary'):
            if a!=b:assert not set(roles[a])&set(roles[b])
    assert len(roles['val'])==5 and len(seen)==1
    bundle=load_context_bundle(folder/'trained.pt')
    assert bundle.provenance['train_source_ids']==roles['train']
    assert bundle.provenance['validation_source_ids']==roles['val']
    assert bundle.history['pretrain_updates_completed']==1
    assert bundle.history['predictor_updates_completed_per_head']==1
    references=json.loads((folder/'trained_references.json').read_text())
    for value in references.values():
        for reference in value.values():
            assert set(reference['source_ids'])==set(roles['val'])
    scores=pd.read_csv(folder/'trained_scores.csv')
    assert set(scores.source_id)==set(roles['test'])
    assert scores.profile_score.notna().all()
    assert (folder/'ridge.joblib').exists()
    assert (out/'window_index_before_profile_qc.csv').exists()
    assert pd.read_csv(out/'profile_eligibility.csv').profile_eligible.all()
    identity=json.loads((out/'protocol.json').read_text())
    assert identity['profile_eligibility_sha256'] and identity['retained_metadata_sha256']
    for method in ('ridge','untrained'):
        runner._assert_budget_selection(pd.read_csv(folder/f'{method}_budget4_selection.csv'),
                                         pd.read_csv(folder/'trained_budget4_selection.csv'))
    untrained=load_context_bundle(folder/'untrained.pt')
    assert untrained.history['pretrain_updates_completed']==0
    def no_fit(*args,**kwargs):
        raise AssertionError('Resuming an intact scientific run must reuse fitted checkpoints and losses')
    monkeypatch.setattr(runner,'fit_context_models',no_fit)
    monkeypatch.setattr(runner,'fit_ridge_context',no_fit)
    resumed=runner.run_study(demo=True,config=cfg,mask_repeats=16,ridge_draws=64,
                             include_untrained=True,folds=(0,),output_root=tmp_path,finalize=False)
    assert resumed==out
    assert json.loads((out/'protocol.json').read_text())['started_utc']==identity['started_utc']


def test_profile_qc_excludes_entire_sources_before_identity_and_preserves_attrs():
    x,meta,_,_=runner.prepare_study(demo=True)
    original=meta.copy();meta.attrs['extra_preprocessing_rule']='unchanged'
    source=meta.source_id.iloc[0];bad=meta.source_id.eq(source).to_numpy()
    x[bad,4:,4]=np.nan
    values,filtered,ledger=runner._freeze_profile_cohort(x,meta,small_config(),16)
    assert source not in set(filtered.source_id)
    assert len(values)==len(x)-bad.sum()
    assert filtered.window_index.tolist()==list(range(len(filtered)))
    assert filtered.pre_profile_window_index.tolist()==meta.loc[~bad,'window_index'].tolist()
    assert filtered.attrs['extra_preprocessing_rule']=='unchanged'
    assert filtered.attrs['pre_profile_corpus_sha256']==meta.attrs['corpus_sha256']
    assert filtered.attrs['corpus_sha256']!=meta.attrs['corpus_sha256']
    assert not ledger.set_index('source_id').loc[source,'profile_eligible']
    pd.testing.assert_frame_equal(meta,original,check_flags=False)


def test_support_and_budget_mismatch_raise_before_method_comparison():
    a=pd.DataFrame(dict(window_index=[0],joint=[25],patch_index=[0],target_type=['coordinate'],
                        target_count=[2],requested_count=[3],loss_own=[2.],loss_patch=[1.],loss_full=[.5]))
    runner._assert_compact_support(a,a.assign(loss_own=500.))
    with pytest.raises(ValueError,match='support weights'):
        runner._assert_compact_support(a,a.assign(target_count=1))
    selected=pd.DataFrame(dict(source_id=['source'],joint=[25],sequence_id=['seq'],
                               patch_start_ns=[0],patch_end_ns=[266666667]))
    with pytest.raises(ValueError,match='selections differ'):
        runner._assert_budget_selection(selected,selected.assign(patch_start_ns=266666667))


def test_parallel_protocol_publication_preserves_first_creation(tmp_path):
    from concurrent.futures import ThreadPoolExecutor
    path=tmp_path/'protocol.json'
    base=dict(run_id='id',corpus='corpus',profile_eligibility_sha256='qc',fold_registry_sha256='folds')
    def initialize(i):runner._initialize_protocol(path,dict(base,started_utc=f'time{i}'))
    with ThreadPoolExecutor(max_workers=4) as pool:list(pool.map(initialize,range(12)))
    created=json.loads(path.read_text())['started_utc']
    initialize(99)
    assert json.loads(path.read_text())['started_utc']==created
    assert sorted(p.name for p in tmp_path.iterdir())==['protocol.json']
    with pytest.raises(ValueError,match='identity differs'):
        runner._initialize_protocol(path,dict(base,corpus='changed'))
