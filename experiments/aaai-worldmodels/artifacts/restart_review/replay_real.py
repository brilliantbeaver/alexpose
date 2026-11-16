"""Independent replay of completed real predictions and aggregation."""
from pathlib import Path
import hashlib
import json
import sys

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from threadpoolctl import threadpool_limits

PROJECT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT))
import worldmodels.restart_experiment as module


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    prepared = PROJECT/'artifacts/runs/restart-full10hz-v1'
    output = PROJECT/'artifacts/runs/restart-dynamics-v1'
    identity = json.loads((output/'identity.json').read_text())
    assert digest(module.__file__) == identity['code_sha256']
    assert digest(PROJECT/'docs/restart/protocol.md') == identity['protocol_sha256']
    for name, expected in identity['input_files'].items():
        assert digest(prepared/name) == expected
    hashes = json.loads((output/'artifact_hashes.json').read_text())
    for name, expected in hashes.items():
        assert digest(output/name) == expected
    with np.load(prepared/'windows.npz', allow_pickle=False) as z:
        x, valid = z['x'], z['valid']
    meta = pd.read_csv(prepared/'windows.csv')
    cohort = pd.read_csv(output/'cohort.csv')
    rows = meta.reset_index().set_index('window_id').loc[cohort.window_id, 'index'].to_numpy()
    x, valid = x[rows], valid[rows]
    np.testing.assert_array_equal(meta.iloc[rows].source_id.to_numpy(), cohort.source_id.to_numpy())
    assert len(meta)-len(cohort) == len(pd.read_csv(output/'support_exclusions.csv')) == 3
    membership = json.loads((output/'fold_membership.json').read_text())
    for fold in membership:
        expected_train = set(cohort.loc[(cohort.outer_fold != fold['fold']) & (cohort.label == 'normal'), 'source_id'])
        expected_test = set(cohort.loc[cohort.outer_fold == fold['fold'], 'source_id'])
        assert set(fold['train_sources']) == expected_train
        assert set(fold['heldout_sources']) == expected_test
        assert not expected_train & expected_test
        for inner in fold['inner']:
            assert not set(inner['train']) & set(inner['validation'])
            assert set(inner['train'] + inner['validation']) == expected_train
    selected = json.loads((output/'selection.json').read_text())
    for choice in selected:
        grid = choice['all_grid_mse']
        candidates = grid if choice['arm'].endswith('_shrunk') else [r for r in grid if r['beta'] == 1]
        best = min(candidates, key=lambda r: (r['mse'], r['beta'], -r['alpha']))
        assert (best['alpha'], best['beta']) == (choice['alpha'], choice['beta'])
        assert best['mse'] == choice['inner_normal_mse']
    stored = np.load(output/'forecast_predictions.npz', allow_pickle=False)
    errors = pd.read_csv(output/'window_predictions.csv')
    replayed = 0
    max_saved_difference = 0.
    solver_differences = []
    with threadpool_limits(limits=1):
        for path in sorted((output/'models').glob('*.joblib')):
            checkpoint = joblib.load(path)
            fold = int(path.name.split('_')[0][4:])
            test = cohort.outer_fold.to_numpy() == fold
            train = (cohort.outer_fold.to_numpy() != fold) & (cohort.label.to_numpy() == 'normal')
            leg = checkpoint['leg']
            arm = checkpoint['arm'] + ('_shrunk' if '_shrunk_' in path.name else '')
            assert set(checkpoint['training_sources']) == set(cohort.loc[train, 'source_id'])
            assert set(checkpoint['heldout_sources']) == set(cohort.loc[test, 'source_id'])
            prediction = module.predict_checkpoint(checkpoint, x[test], valid[test], cohort.loc[test].window_id.tolist())
            np.testing.assert_array_equal(prediction.astype(np.float32), stored[arm][test, :, leg*4:(leg+1)*4])
            max_saved_difference = max(max_saved_difference, float(np.max(np.abs(prediction - stored[arm][test, :, leg*4:(leg+1)*4]))))
            changed = x[test].copy(); changed[:, 16:] = np.nan
            changed_valid = valid[test].copy(); changed_valid[:, 16:] = False
            np.testing.assert_array_equal(prediction, module.predict_checkpoint(checkpoint, changed, changed_valid,
                                                                               cohort.loc[test].window_id.tolist()))
            joints = module.LEGS[leg]
            actual_target = x[test, 16:][:, :, joints]
            observed = valid[test, 16:][:, :, joints] & valid[test, :16][:, :, joints].any(1)[:, None]
            for horizon in (4, 8):
                coordinate_mask = np.repeat(observed[:, :horizon, :, None], 2, axis=-1)
                squared = np.where(coordinate_mask, (prediction[:, :horizon]-actual_target[:, :horizon])**2, 0.)
                direct_error = squared.sum((1,2,3))/coordinate_mask.sum((1,2,3))
                saved_rows = errors[(errors.arm==arm) & (errors.horizon==horizon)].set_index('window_id').loc[cohort.loc[test].window_id]
                column = 'left_mse' if leg==0 else 'right_mse'
                np.testing.assert_allclose(direct_error, saved_rows[column], atol=1e-12, rtol=1e-12)
            f, _, _ = module.features(x[train], valid[train], leg, checkpoint['arm'], cohort.loc[train].window_id.tolist())
            tm = cohort.loc[train]
            counts = tm.groupby(['source_id', 'sequence_id']).source_id.transform('size').to_numpy()
            sequences_per_source = tm[['source_id','sequence_id']].drop_duplicates().groupby('source_id').size()
            weights = 1/(counts*tm.source_id.map(sequences_per_source).to_numpy())
            weights *= len(weights)/weights.sum()
            mu = np.average(f, axis=0, weights=weights)
            sd = np.sqrt(np.average((f-mu)**2, axis=0, weights=weights)); sd[sd<1e-6] = 1
            np.testing.assert_allclose(mu, checkpoint['model']['mean'], atol=1e-12, rtol=1e-12)
            np.testing.assert_allclose(sd, checkpoint['model']['scale'], atol=1e-12, rtol=1e-12)
            # Independent library refit of one prespecified output coordinate in
            # each unshrunk bilateral leg/fold. No parameter selection is changed.
            if arm == 'bilateral':
                target, mask = module.targets(x[train], valid[train], leg)
                k = 63
                observed = mask[:, k]
                target_weights = weights*64/mask.sum(1)
                z = (f-mu)/sd
                reference = Ridge(alpha=checkpoint['alpha']).fit(z[observed], target[observed,k], sample_weight=target_weights[observed])
                actual = module.predict(checkpoint['model'], f)[:, k]
                expected = reference.predict(z)
                np.testing.assert_allclose(actual, expected, atol=1e-8, rtol=1e-8)
                solver_differences.append(float(np.max(np.abs(actual-expected))))
            replayed += 1
    assert replayed == 80
    for arm in stored.files:
        for horizon in (4, 8):
            leg_errors = []
            for leg, joints in enumerate(module.LEGS):
                target = x[:, 16:16+horizon][:, :, joints]
                support = valid[:, 16:16+horizon][:, :, joints] & valid[:, :16][:, :, joints].any(1)[:, None]
                mask = np.repeat(support[..., None], 2, axis=-1)
                forecast = stored[arm][:, :horizon, leg*4:(leg+1)*4].astype(float)
                value = np.where(mask, (forecast-target)**2, 0).sum((1,2,3))/mask.sum((1,2,3))
                leg_errors.append(value)
            saved_rows = errors[(errors.arm==arm) & (errors.horizon==horizon)].set_index('window_id').loc[cohort.window_id]
            # Stored coordinates use float32, while saved MSE uses full precision.
            np.testing.assert_allclose(np.mean(leg_errors, axis=0), saved_rows.mse, atol=2e-7, rtol=2e-7)
    source = errors.groupby(['source_id','sequence_id','label','fold','arm','horizon'])[['mse','left_mse','right_mse']].mean().reset_index()
    source = source.groupby(['source_id','label','fold','arm','horizon'])[['mse','left_mse','right_mse']].mean().reset_index()
    pd.testing.assert_frame_equal(source, pd.read_csv(output/'source_scores.csv'), check_exact=False, atol=1e-12, rtol=1e-12)
    results = json.loads((output/'results.json').read_text())
    for contrast in results['paired_contrasts']:
        normal = source[(source.label=='normal') & (source.horizon==contrast['horizon'])].pivot(index='source_id', columns='arm', values='mse')
        gains = (normal[contrast['reference']] - normal[contrast['method']]).to_numpy()
        rng = np.random.default_rng(812)
        interval = np.quantile(rng.choice(gains, (5000, len(gains)), replace=True).mean(1), [.025,.975])
        np.testing.assert_allclose(gains.mean(), contrast['estimate'], atol=1e-12, rtol=1e-12)
        np.testing.assert_allclose(interval, contrast['ci95'], atol=1e-12, rtol=1e-12)
    anomaly = pd.read_csv(output/'source_anomaly_scores.csv')
    for score, numbers in results['anomaly'].items():
        positive = anomaly.loc[anomaly.label!='normal', score].to_numpy()
        negative = anomaly.loc[anomaly.label=='normal', score].to_numpy()
        manual_auc = np.mean((positive[:,None] > negative).astype(float) + .5*(positive[:,None] == negative))
        np.testing.assert_allclose(manual_auc, numbers['auroc'], atol=1e-12, rtol=1e-12)
    record = dict(status='passed', real_data=True, windows=len(cohort), sources=int(cohort.source_id.nunique()),
                  artifact_hashes_verified=len(hashes), checkpoints_replayed=replayed,
                  all_checkpoint_float32_predictions_exact=True,
                  maximum_float32_rounding_difference=max_saved_difference,
                  independently_refit_output_coordinates=len(solver_differences),
                  maximum_independent_solver_prediction_difference=max(solver_differences),
                  code_sha256=digest(module.__file__), protocol_sha256=identity['protocol_sha256'],
                  results_sha256=digest(output/'results.json'), audit_script_sha256=digest(__file__),
                  checks=['input and output identities', 'all inner/outer source membership',
                          'all selected hyperparameter argmins', 'all 80 checkpoint predictions',
                          'all learned checkpoint window errors from absolute coordinates',
                          'all baseline and inference-control window errors from stored coordinates',
                          'hidden future invariance', 'all training-only scalers',
                          'ten independent sklearn coordinate refits', 'source aggregation',
                          'all paired means and intervals', 'manual pairwise anomaly AUROCs'],
                  limitation='Replay verifies this completed internal experiment; it does not establish patient identity, clinical labels or external generalization.')
    (Path(__file__).parent/'real_replay.json').write_text(json.dumps(record, indent=2)+'\n')
    print(json.dumps(record, indent=2))


if __name__ == '__main__':
    main()
