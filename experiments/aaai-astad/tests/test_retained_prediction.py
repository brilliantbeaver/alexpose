import numpy as np
import pandas as pd
import pytest

from aaai_gait.masking import make_mask_bank
from aaai_gait.retained_prediction import (
    RidgeStats, construct_features, fit_local_baseline, fit_correction_grid,
    make_training_plan, iter_score_batches, inner_candidate_report,
    select_correction_recipe, known_dependency_fixtures,
)


def sample_data(n=12, seed=3, prefix="fit"):
    rng=np.random.default_rng(seed)
    x=rng.normal(size=(n,32,16,2)).astype(np.float32)
    ids=np.asarray([f"{prefix}-{i//3}" for i in range(n)])
    return x,ids,np.full(n,"normal")


def test_hidden_coordinates_and_opposite_leg_cannot_enter_own_features():
    x,_,_=sample_data(3)
    hidden=make_mask_bank().hidden[:3].copy()
    hidden[:,3*16+10]=True  # Left ankle, patch 3.
    before=construct_features(x,hidden,27,3)
    changed=x.reshape(3,8,4,16,2).transpose(0,1,3,2,4).copy()
    changed[hidden.reshape(3,8,16)]+=10000
    changed=changed.transpose(0,1,3,2,4).reshape(3,32,16,2)
    after=construct_features(changed,hidden,27,3)
    for name in ("own","own_extra","other"):
        np.testing.assert_array_equal(getattr(before,name),getattr(after,name))
    changed=x.copy();changed[:,:,7::2]+=100
    after=construct_features(changed,hidden,27,3)
    np.testing.assert_array_equal(before.own,after.own)
    np.testing.assert_array_equal(before.own_extra,after.own_extra)
    assert not np.array_equal(before.other,after.other)
    assert np.isfinite(before.own).all()


def test_unmasked_target_is_rejected_and_missing_values_never_enter_features():
    x,_,_=sample_data(3)
    with pytest.raises(ValueError,match="must be hidden"):
        construct_features(x,np.zeros(128,bool),27,3)
    hidden=np.zeros((3,128),bool);hidden[:,3*16+10]=True
    x[:,:,8,:]=np.nan
    feat=construct_features(x,hidden,27,3)
    assert all(np.isfinite(getattr(feat,k)).all() for k in ("own","own_extra","other"))


def test_weighted_sufficient_statistics_match_batch_and_rank():
    rng=np.random.default_rng(9)
    x=rng.normal(size=(80,4));x=np.column_stack((x,x[:,0]))
    y=rng.normal(size=(80,8));weights=rng.uniform(.2,2,size=80)
    batch=RidgeStats(5);batch.add(x,y,weights)
    streamed=RidgeStats(5)
    for start in range(0,80,13): streamed.add(x[start:start+13],y[start:start+13],weights[start:start+13])
    a,b=batch.fit(1),streamed.fit(1)
    np.testing.assert_allclose(a.predict(x),b.predict(x),atol=1e-12)
    expected=np.sum(weights[:,None]*(a.predict(x)-y)**2)/(weights.sum()*8)
    assert batch.risk(a)==pytest.approx(expected,abs=1e-12)
    assert a.feature_rank==4


def test_training_sources_are_balanced_and_mask_policy_cannot_change_window_order():
    x,ids,labels=sample_data()
    uniform=make_training_plan(x,source_ids=ids,labels=labels,draws=24)
    joint=make_training_plan(x,source_ids=ids,labels=labels,draws=24,policy="J")
    np.testing.assert_array_equal(uniform.presentation_windows,joint.presentation_windows)
    row_sources=ids[joint.presentation_windows[joint.row_presentations]]
    totals=[joint.weights[row_sources==s].sum() for s in np.unique(ids)]
    np.testing.assert_allclose(totals,np.repeat(totals[0],len(totals)))
    found_tube=False
    for feat,_ in joint.batches():
        found_tube|=bool((feat.support["own_joint_count"]==0).any())
        assert (feat.support["own_times"]>=2).all()
        assert (feat.support["other_times"]>=2).all()
    assert found_tube
    labels[0]="stroke"
    with pytest.raises(ValueError,match="normal"):
        fit_local_baseline(x,source_ids=ids,labels=labels,draws=24)


def test_zero_correction_is_bit_identical_and_predictions_reconstruct_loss():
    x,ids,labels=sample_data()
    base=fit_local_baseline(x,source_ids=ids,labels=labels,draws=24)
    grid=fit_correction_grid(base,x,source_ids=ids,labels=labels,policy="A",alphas=(1,),draws=24)
    bundle=grid[1.].with_multipliers(0,0)
    scored=pd.concat(iter_score_batches(bundle,x[:2],make_mask_bank(repeats=2),return_predictions=True))
    np.testing.assert_array_equal(scored.loss_base,scored.loss_own)
    np.testing.assert_array_equal(scored.loss_base,scored.loss_both)
    y=scored[[f"Y_{k}" for k in range(8)]].to_numpy()
    pred=scored[[f"base_{k}" for k in range(8)]].to_numpy()
    np.testing.assert_allclose(scored.loss_base,np.square(y-pred).mean(1))
    assert bundle.metadata["own_feature_dimension"]==bundle.metadata["both_feature_dimension"]==154
    with pytest.raises(ValueError,match="identical fitting-source"):
        fit_correction_grid(base,x[:3],source_ids=ids[:3],labels=labels[:3],draws=24)
    with pytest.raises(ValueError,match="overlap"):
        inner_candidate_report(base,{"A":grid},x,source_ids=ids,labels=labels)
    val,vid,vlabel=sample_data(3,seed=4,prefix="validation")
    report=inner_candidate_report(base,{"A":grid},val,source_ids=vid,labels=vlabel,
                                  bank=make_mask_bank(repeats=2),inner_fold=0)
    assert len(report)==7 and set(report.source_id)==set(vid)
    assert np.isfinite(report.loss).all()


def test_internal_recipe_retains_failed_folds_and_zero_fallback():
    rows=[]
    for fold in range(3):
        rows.append(dict(source_id=f"v{fold}",policy="shared",branch="base",regularization=0.,multiplier=1.,loss=1.,inner_fold=fold))
        for branch in ("own","both"):
            for gate in (0.,1.):
                rows.append(dict(source_id=f"v{fold}",policy="U",branch=branch,regularization=1.,multiplier=gate,
                                 loss=1. if gate==0 else (.9 if branch=="own" else 1.1),inner_fold=fold))
    recipe=select_correction_recipe(pd.DataFrame(rows),"U")
    assert recipe["own_multiplier"]==1.
    assert recipe["both_multiplier"]==0. and not recipe["relational_supported"]
    assert recipe["internal_fold_count"]==3


def test_known_dependency_and_null_fixtures_preserve_target_marginal_law():
    report=known_dependency_fixtures()
    assert report["passed"],report["summary"]
    assert report["scope"]=="feature_level_estimator_fixture_not_gait"
    for replicate in range(5):
        cases=[r for r in report["cases"] if r["replicate"]==replicate]
        assert cases[0]["losses"]["base"]==cases[1]["losses"]["base"]
        assert cases[0]["losses"]["own"]==cases[1]["losses"]["own"]
