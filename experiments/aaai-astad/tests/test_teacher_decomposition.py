"""Falsify the frozen teacher/context factorization and its support contract."""
import numpy as np
import pandas as pd
import pytest
import torch

from aaai_gait.context_prediction import ContextBundle,ContextConfig,make_mask_bank,score_context_models
from aaai_gait.model import JOINT_INDICES,SJEPA
from aaai_gait.teacher_decomposition import (
    FACTORS,KEYS,LOSSES,decompose_teacher_losses,score_teacher_decomposition,
)
from aaai_gait.training import synthetic_windows


@pytest.fixture
def bundle():
    torch.set_num_threads(2)
    torch.manual_seed(5)
    cfg=ContextConfig(pretrain_updates=0,predictor_updates=1,dim=16,depth=2,heads=2)
    return ContextBundle(SJEPA(cfg.model_config()),cfg).eval()


def test_identical_inputs_duplicate_all_four_factors_and_match_main_scorer(bundle):
    x=synthetic_windows(2)
    factors=score_teacher_decomposition(bundle,x,x,mask_repeats=2,joints=(13,27))
    main=score_context_models(bundle,x,mask_repeats=2,joints=(13,27))
    main=main[main.target_type.eq('latent')].sort_values(list(KEYS)).reset_index(drop=True)
    for table in factors.values():
        np.testing.assert_allclose(table[list(LOSSES)],main[list(LOSSES)],rtol=0,atol=0,equal_nan=True)
    deltas=decompose_teacher_losses(factors)
    columns=[c for c in deltas if c not in KEYS and c!='valid']
    assert (deltas.loc[deltas.valid,columns]==0).all().all()


def test_contextual_teacher_changes_while_own_joint_predictor_stays_fixed(bundle):
    original=synthetic_windows(2);edited=original.copy()
    own_slot=JOINT_INDICES.index(27)
    edited[:,:,[j for j in range(16) if j!=own_slot]]+=100.
    bundle.train()
    before={name:value.clone() for name,value in bundle.state_dict().items()}
    tables=score_teacher_decomposition(bundle,original,edited,mask_repeats=2,joints=(27,))
    assert bundle.training  # A diagnostic must restore the caller's mode.
    for name,value in before.items():torch.testing.assert_close(value,bundle.state_dict()[name],atol=0,rtol=0)
    deltas=decompose_teacher_losses(tables);valid=deltas.valid
    np.testing.assert_allclose(deltas.loc[valid,'predictor_own'],0.,atol=0,rtol=0)
    np.testing.assert_allclose(deltas.loc[valid,'interaction_own'],0.,atol=1e-12,rtol=0)
    assert np.abs(deltas.loc[valid,'teacher_own']).max()>1e-6
    for branch in ('own','patch','full'):
        total=deltas.loc[valid,[f'{c}_{branch}' for c in ('predictor','teacher','interaction')]].sum(axis=1)
        np.testing.assert_allclose(total,deltas.loc[valid,'joint_'+branch],rtol=1e-10,atol=1e-10)


def test_hidden_measurements_move_teacher_without_entering_any_predictor(bundle):
    original=synthetic_windows(2);edited=original.copy()
    mask=make_mask_bank(bundle.config,17,1).numpy().reshape(1,8,16).repeat(2,axis=0)
    edited[np.repeat(mask,4,axis=1)]+=80.
    tables=score_teacher_decomposition(bundle,original,edited,mask_repeats=1,joints=(13,27))
    first,changed=tables['p0y0'],tables['p1y0']
    np.testing.assert_allclose(first[list(LOSSES)],changed[list(LOSSES)],atol=0,rtol=0,equal_nan=True)
    deltas=decompose_teacher_losses(tables)
    assert np.abs(deltas.loc[deltas.valid,'teacher_full']).max()>1e-6


def test_all_factors_retain_common_missingness_and_own_history_eligibility(bundle):
    original=synthetic_windows(2);edited=original.copy()
    edited[:,4:,JOINT_INDICES.index(13)]=np.nan
    factors=score_teacher_decomposition(bundle,original,edited,mask_repeats=2,joints=(13,27))
    expected=[]
    for x in (original,edited):
        scored=score_context_models(bundle,x,mask_repeats=2,joints=(13,27))
        expected.append(scored[scored.target_type.eq('latent')].sort_values(list(KEYS)).reset_index(drop=True))
    common=expected[0].valid & expected[1].valid
    assert not common[expected[0].joint.eq(13)].any()
    assert common[expected[0].joint.eq(27)].any()
    for table in factors.values():
        np.testing.assert_array_equal(table.valid,common)
        assert table.loc[~table.valid,list(LOSSES)].isna().all().all()
        assert np.isfinite(table.loc[table.valid,list(LOSSES)]).all().all()
        assert table.attrs['valid_targets']==int(common.sum())


def test_decomposition_batch_size_does_not_change_masks_or_values(bundle):
    x=synthetic_windows(3);edited=x.copy();edited[:,:,2]+=.1
    a=score_teacher_decomposition(bundle,x,edited,mask_repeats=2,batch_size=1,joints=(13,27))
    b=score_teacher_decomposition(bundle,x,edited,mask_repeats=2,batch_size=3,joints=(13,27))
    for factor in FACTORS:
        pd.testing.assert_frame_equal(a[factor],b[factor],atol=2e-5,rtol=2e-5)


def test_factorization_rejects_mismatched_support_and_numerical_failure(bundle):
    x=synthetic_windows(2)
    factors=score_teacher_decomposition(bundle,x,x,mask_repeats=1,joints=(27,))
    altered={k:v.copy() for k,v in factors.items()}
    altered['p1y1'].loc[0,'valid']=not bool(altered['p1y1'].loc[0,'valid'])
    with pytest.raises(ValueError,match='identical target identities'):
        decompose_teacher_losses(altered)
    with pytest.raises(ValueError,match='identical shapes'):
        score_teacher_decomposition(bundle,x,x[:1],mask_repeats=1)
    with torch.no_grad():bundle.heads['latent_full'].output[-1].weight.fill_(torch.nan)
    with pytest.raises(FloatingPointError,match='Nonfinite'):
        score_teacher_decomposition(bundle,x,x,mask_repeats=1,joints=(27,))


def test_signed_interaction_formula_counts_neither_main_effect_twice():
    factors={}
    for name,value in zip(FACTORS,(1.,3.,4.,8.),strict=True):
        factors[name]=pd.DataFrame(dict(window_index=[0],joint=[27],patch_index=[0],
            mask_id=[0],target_type=['latent'],valid=[1],**{column:[value] for column in LOSSES}))
    result=decompose_teacher_losses(factors).iloc[0]
    for branch in ('own','patch','full'):
        assert result['predictor_'+branch]==2.
        assert result['teacher_'+branch]==3.
        assert result['interaction_'+branch]==2.
        assert result['joint_'+branch]==7.
