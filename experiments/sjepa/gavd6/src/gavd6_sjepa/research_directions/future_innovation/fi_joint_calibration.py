"""Prospectively fixed synthetic calibration for direct-v3. Never scientific evidence."""
from dataclasses import asdict
from pathlib import Path
import time

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from threadpoolctl import threadpool_limits

from .fi_contracts import DIRECT_ARMS, equal_source_weights, read_json, write_json, code_fingerprint
from .fi_cohort import assign_source_folds
from .fi_joint_models import (JointModelContract, JointRidge, SupportedInput, TargetStandardizer,
                              fit_joint_baseline, temporal_features)
from .fi_joint_training import nested_partition, choose_candidate
from .fi_metrics import score_arrays
from .fi_residual_models import TrainingScaler


def fixture(seed, kind):
    """80 sources x 2 clips; no source labels or targets enter model features.

    Per clip: 16 independent standard-normal RGB inputs; eight independent
    four-frame pose blocks on one joint. y = RGB + optional ordered pose contrast
    + independent noise. Four target coordinates repeat the latent with independent
    noise. The second clip adds no source identity input and stays with its source.
    """
    rng = np.random.default_rng(seed)
    n = 160
    videos = np.repeat([f'source-{i:03d}' for i in range(80)], 2)
    windows = np.array([f'fixture-{seed}-{i:03d}' for i in range(n)])
    folds = assign_source_folds(videos, 5)
    cohort = pd.DataFrame({'window_id': windows, 'video_id': videos,
                           'outer_fold': [folds[v] for v in videos]})
    x = rng.normal(size=(n, 16))
    sk = np.zeros((n, 32, 33, 4)); sk[..., 2:] = 1
    blocks = rng.normal(size=(n, 8))
    sk[:, :, 0, 0] = np.repeat(blocks, 4, axis=1)
    # Independent nuisance skeleton route competes with informative RGB.
    sk[:, :, 1, 1] = np.repeat(rng.normal(size=(n, 8)), 4, axis=1)
    temporal = blocks[:, 6:8].mean(axis=1) - blocks[:, 0:2].mean(axis=1)
    latent = 1.5 * x[:, 0]
    if kind in ('skeleton', 'temporal'): latent += 2.0 * temporal
    y = latent[:, None] + rng.normal(scale=0.25, size=(n, 4))
    arrays = {'baseline': x, 'person': y, 'skeleton': sk, 'matching': x[:, :3]}
    return cohort, arrays, (tuple(f'rgb-{i}' for i in range(16)), ('rgb',) * 16)


def source_held_fixture(seed, kind, config):
    cohort, arrays, schema = fixture(seed, kind)
    true = np.zeros_like(arrays['person']); base = true.copy()
    predictions = {arm: true.copy() for arm in DIRECT_ARMS}
    masks, selections = [], []
    for f in range(5):
        train = np.flatnonzero(cohort.outer_fold.to_numpy() != f)
        test = np.flatnonzero(cohort.outer_fold.to_numpy() == f)
        result = nested_partition(cohort, arrays, train, test, schema, config)
        baseline = result['baseline']
        true[test] = baseline.y_scaler.transform(arrays['person'][test])
        base[test] = baseline.predict(arrays['baseline'][test])
        masks.append(baseline.valid_features)
        for arm in DIRECT_ARMS:
            if not result['selection'][arm]['complete']: raise ValueError('Calibration candidate failed')
            predictions[arm][test] = result['predictions'][arm]
            selections.append({'fold': f, 'arm': arm, 'winner': result['selection'][arm]['winner']})
    valid = np.all(masks, axis=0)
    scores = {arm: score_arrays(true, base, p, equal_source_weights(cohort.video_id), valid)[0] for arm,p in predictions.items()}
    return {'seed': seed, 'kind': kind, 'scores': scores, 'selections': selections,
            'real_gain': scores['real-skeleton']['delta_r2'],
            'matched_increment': scores['real-skeleton']['r2_full'] - scores['no-skeleton']['r2_full'],
            'real_minus_shuffle': scores['real-skeleton']['r2_full'] - scores['time-shuffle']['r2_full']}


def exact_calibration():
    v = np.array([[0., np.nan, 0., 0.], [0., np.nan, 1e-9, 1.], [0., np.nan, 2e-9, 2.]])
    w, ids = np.ones(3), ['a','b','c']
    old = TrainingScaler.fit(v,w,ids)
    new = SupportedInput.fit(v,w,ids,ids,('constant','missing','near','varying'),('fraction',)*4)
    held = np.array([[1., 3., 1., 1.]])
    assert old.transform(held)[0,0] == 1e8
    before = new.record()
    np.testing.assert_array_equal(new.transform(held)[:,:3], 0)
    new.transform(held * 1e4)
    assert before == new.record()
    y = TargetStandardizer.fit(np.array([[0.,0.],[0.,1.],[0.,2.]]), w, ids)
    assert y.transform(np.array([[1.,1.]]))[0,0] == 1e8  # Target variation survives input policy.
    rng = np.random.default_rng(41)
    x, target = rng.normal(size=(14,27)), rng.normal(size=(14,4))
    vids = [str(i//2) for i in range(14)]; weights = equal_source_weights(vids)
    joint = JointRidge.fit(x,None,target,weights,10.)
    reference = Ridge(alpha=10.,solver='cholesky').fit(x,target,sample_weight=weights)
    np.testing.assert_allclose(joint.predict(x),reference.predict(x),rtol=1e-10,atol=1e-10)
    records = [{'candidate_id':'baseline_only','candidate_type':'baseline_only','valid':True,'pooled_loss':0.,'lambda_x':1.,'lambda_s':None},
               {'candidate_id':'bad','candidate_type':'joint_ridge','valid':True,'pooled_loss':1.,'lambda_x':1.,'lambda_s':1.},
               {'candidate_id':'failed','candidate_type':'joint_ridge','valid':False,'pooled_loss':None,'lambda_x':10.,'lambda_s':10.}]
    assert choose_candidate(records,JointModelContract())[0]['candidate_type']=='baseline_only'
    history = np.zeros((1,32,33,4)); history[:,0,0,:] = [1.,1.,1.,1.]; history[:,2,0,:] = [100.,100.,1.,1.]
    feats = temporal_features(history)
    assert np.isnan(feats[0,2:4]).all()
    return {'old_unsupported_value':1e8, 'repaired_unsupported_value':0.,'target_variation_preserved':True,
            'rgb_only_ridge_equivalence':True,'fallback_with_inferior_and_failed_candidates':True,
            'missing_endpoints_do_not_create_velocity':True,'heldout_statistics_unchanged':True}


def calibrate(output):
    output = Path(output)
    if output.exists(): raise ValueError('Calibration is immutable; choose a new output path')
    config = JointModelContract(); started = time.monotonic()
    with threadpool_limits(limits=1):
        exact = exact_calibration()
        rows = []
        for seed, kind in [(s,'rgb_only') for s in range(1101,1109)] + [(2201,'skeleton'),(3301,'temporal')]:
            row = source_held_fixture(seed,kind,config); rows.append(row)
            print(f"Calibration {kind} {seed}: gain={row['real_gain']:.6f}, matched={row['matched_increment']:.6f}, real-shuffle={row['real_minus_shuffle']:.6f}",flush=True)
    null = rows[:8]
    checks = {'null_mean_real_gain': float(np.mean([r['real_gain'] for r in null])) <= 0.03,
              'null_mean_matched_increment': float(np.mean([r['matched_increment'] for r in null])) <= 0.03,
              'planted_gain': rows[8]['real_gain'] > 0.10, 'planted_matched': rows[8]['matched_increment'] > 0.10,
              'temporal_gain': rows[9]['real_gain'] > 0.10, 'temporal_matched': rows[9]['matched_increment'] > 0.10,
              'temporal_shuffle_reduces_signal': rows[9]['real_minus_shuffle'] > 0.10}
    result = {'passed': all(checks.values()), 'synthetic': True, 'model_contract': asdict(config), 'checks': checks,
              'exact': exact, 'fixtures': rows, 'elapsed_seconds': time.monotonic()-started,
              'code_sha256': code_fingerprint(), 'interpretation': 'Software calibration only; no scientific ADVANCE'}
    write_json(output,result)
    if not result['passed']: raise ValueError(f'Calibration failed: {checks}')
    return result
