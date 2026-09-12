"""Nested source selection for direct-v3, with a typed exact baseline fallback."""
from dataclasses import asdict
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from .fi_contracts import (DIRECT_ARMS, check_run, equal_source_weights, load_model_contract,
                           read_json, sha256_file, write_json, write_once_json)
from .fi_cohort import assign_source_folds
from .fi_joint_models import (MODEL_VERSION, JointRidge, SupportedInput, TargetStandardizer,
                              baseline_schema, fit_joint_baseline, skeleton_schema, temporal_features)
from .fi_nested_training import isolated_split, partition_control, raw_source_weights
from .fi_feature_cache import load_cache
from .fi_cache_reuse import checked_file, verify_reused_readiness
from gavd6_sjepa.shared_infrastructure.artifact_io_operations import atomic_save_joblib


def candidate_specs(config):
    return [{'candidate_id': f'joint/x={x:g}/s={s:g}', 'candidate_type': 'joint_ridge',
             'lambda_x': float(x), 'lambda_s': float(s)}
            for x in config.ridge_alphas for s in config.skeleton_alphas]


def choose_candidate(records, config):
    """Baseline wins ties; otherwise larger penalties win tied joint losses."""
    valid = [r for r in records if r['valid'] and np.isfinite(r['pooled_loss'])]
    baseline = next(r for r in records if r['candidate_type'] == 'baseline_only')
    if not baseline['valid']: raise ValueError('Required shared baseline failed')
    best_loss = min(r['pooled_loss'] for r in valid)
    tolerance = config.tie_atol + config.tie_rtol * max(abs(best_loss), abs(baseline['pooled_loss']))
    tied = [r for r in valid if r['pooled_loss'] <= best_loss + tolerance]
    if baseline in tied:
        winner = baseline
        reason = 'baseline_only: no addition improves pooled inner loss beyond tolerance'
    else:
        winner = max(tied, key=lambda r: (r['lambda_x'], r['lambda_s']))
        reason = 'joint_ridge: lowest pooled inner loss; ties prefer larger RGB then skeleton penalty'
    return winner, reason


def error_record(y, pred, videos, valid):
    w = raw_source_weights(np.asarray(videos))
    if not valid.any() or not np.isfinite(pred).all(): raise ValueError('Invalid candidate prediction/mask')
    error = float(np.sum(w * ((y[:, valid] - pred[:, valid])**2).mean(axis=1)))
    if not np.isfinite(error): raise ValueError('Non-finite candidate loss')
    return {'validation_squared_error': error, 'validation_weight': float(w.sum()),
            'validation_mse': error / w.sum()}


def summarize(spec, folds):
    if len(folds) != 3 or [r['inner_fold'] for r in folds] != [0,1,2] or any(type(r['valid']) is not bool for r in folds):
        raise ValueError('Candidate requires exactly three identified inner folds')
    for row in folds:
        if row['valid'] and (not np.isfinite([row['validation_squared_error'],row['validation_weight'],row['validation_mse'],row['training_mse']]).all()
                or row['validation_squared_error'] < 0 or row['validation_weight'] <= 0
                or not np.isclose(row['validation_mse'],row['validation_squared_error']/row['validation_weight'],rtol=1e-12,atol=1e-12)):
            raise ValueError('Invalid per-fold candidate loss')
    valid = all(r['valid'] for r in folds)
    return {**spec, 'valid': valid, 'pooled_loss': (
        sum(r['validation_squared_error'] for r in folds) / sum(r['validation_weight'] for r in folds) if valid else None),
        'inner_folds': folds}


def nested_partition(cohort, arrays, train, test, schema, config):
    """The complete fitting/selection path, also used by fixed synthetic fixtures."""
    isolated_split(cohort, train, test)
    mapping = assign_source_folds(cohort.iloc[train].video_id, config.inner_folds)
    labels = np.array([mapping[v] for v in cohort.iloc[train].video_id])
    inner = [(train[labels != f], train[labels == f]) for f in range(config.inner_folds)]
    split = {'outer_train': cohort.iloc[train].window_id.tolist(), 'outer_test': cohort.iloc[test].window_id.tolist(),
             'inner': [], 'donors': [], 'preprocessing': []}
    for f, (fit, val) in enumerate(inner):
        isolated_split(cohort, fit, val)
        split['inner'].append({'fold': f, 'train': cohort.iloc[fit].window_id.tolist(),
                               'validation': cohort.iloc[val].window_id.tolist()})
    x, y = arrays['baseline'], arrays['person']
    weights = equal_source_weights(cohort.iloc[train].video_id)
    # Scalers are shared across penalties and arms at a given training boundary.
    baseline_grid, baseline_fits, preprocessing = [], {}, {}
    for alpha in config.ridge_alphas:
        rows, fits = [], []
        for f, (fit, val) in enumerate(inner):
            baseline = fit_joint_baseline(x[fit], y[fit], cohort.iloc[fit].window_id,
                                           cohort.iloc[fit].video_id, alpha, schema)
            rows.append({'inner_fold': f, 'valid': True, **error_record(baseline.y_scaler.transform(y[val]), baseline.predict(x[val]),
                                       cohort.iloc[val].video_id, baseline.valid_features),
                         'training_mse': error_record(baseline.y_scaler.transform(y[fit]), baseline.predict(x[fit]),
                                       cohort.iloc[fit].video_id, baseline.valid_features)['validation_mse'],
                         'loss_state': 'closed_form_fitted_model'})
            fits.append(baseline)
        baseline_grid.append(summarize({'candidate_id': f'rgb/x={alpha:g}', 'candidate_type': 'rgb_reference',
                                       'lambda_x': float(alpha), 'lambda_s': None}, rows))
        baseline_fits[alpha] = fits
    best = min(r['pooled_loss'] for r in baseline_grid)
    alpha = max(r['lambda_x'] for r in baseline_grid if r['pooled_loss'] <= best + config.tie_atol + config.tie_rtol * abs(best))
    reference = next(r for r in baseline_grid if r['lambda_x'] == alpha)
    inner_baselines = baseline_fits[alpha]
    baseline = fit_joint_baseline(x[train], y[train], cohort.iloc[train].window_id,
                                 cohort.iloc[train].video_id, alpha, schema)
    def keep_transform(scope, base, heldout):
        preprocessing[scope] = {'x': base.x_scaler.record(), 'y': {k: v.tolist() if isinstance(v, np.ndarray) else v for k, v in asdict(base.y_scaler).items()},
                                'heldout_x': base.x_scaler.diagnostics(x[heldout])}
        split['preprocessing'].append({'scope': scope, 'fitted_window_ids': list(base.x_scaler.training_window_ids),
                                      'target_fitted_window_ids': list(base.y_scaler.training_window_ids),
                                      'evaluated_window_ids': cohort.iloc[heldout].window_id.tolist()})
    for f, ((fit, val), base) in enumerate(zip(inner, inner_baselines)): keep_transform(f'inner-{f}', base, val)
    keep_transform('outer', baseline, test)
    models, selection, predictions, diagnostics = {}, {}, {}, []
    snames, skinds = skeleton_schema()
    for arm in DIRECT_ARMS:
        controls = []
        for f, ((fit, val), base) in enumerate(zip(inner, inner_baselines)):
            sf, sv = [temporal_features(partition_control(arm, ids, cohort, arrays, f'{arm}/inner-{f}/{label}', split['donors']))
                      for ids, label in ((fit, 'train'), (val, 'validation'))]
            scaler = SupportedInput.fit(sf, equal_source_weights(cohort.iloc[fit].video_id),
                                        cohort.iloc[fit].window_id, cohort.iloc[fit].video_id, snames, skinds)
            controls.append((scaler, scaler.transform(sf), scaler.transform(sv)))
            preprocessing[f'{arm}/inner-{f}'] = {'s': scaler.record(), 'heldout_s': scaler.diagnostics(sv)}
        candidates = [{**reference, 'candidate_id': 'baseline_only', 'candidate_type': 'baseline_only'}]
        for spec in candidate_specs(config):
            rows = []
            for f, ((fit, val), base, (_, sf, sv)) in enumerate(zip(inner, inner_baselines, controls)):
                try:
                    xf, xv = base.x_scaler.transform(x[fit]), base.x_scaler.transform(x[val])
                    yf, yv = base.y_scaler.transform(y[fit]), base.y_scaler.transform(y[val])
                    model = JointRidge.fit(xf, sf, yf, equal_source_weights(cohort.iloc[fit].video_id), spec['lambda_x'], spec['lambda_s'])
                    row = {'inner_fold': f, 'valid': True, **error_record(yv, model.predict(xv, sv), cohort.iloc[val].video_id, base.valid_features),
                           'training_mse': error_record(yf, model.predict(xf, sf), cohort.iloc[fit].video_id, base.valid_features)['validation_mse'],
                           'loss_state': 'closed_form_fitted_model'}
                except (ValueError, np.linalg.LinAlgError, FloatingPointError) as error:
                    row = {'inner_fold': f, 'valid': False, 'error': f'{type(error).__name__}: {error}',
                           'validation_squared_error': None, 'validation_weight': None, 'validation_mse': None, 'training_mse': None,
                           'loss_state': 'failed'}
                rows.append(row)
            candidates.append(summarize(spec, rows))
        winner, reason = choose_candidate(candidates, config)
        for candidate in candidates:
            candidate['improvement_over_baseline'] = reference['pooled_loss'] - candidate['pooled_loss'] if candidate['valid'] else None
            candidate['selected'] = candidate['candidate_id'] == winner['candidate_id']
            candidate['selection_reason'] = reason if candidate['selected'] else ('rejected: invalid numerical fit' if not candidate['valid'] else 'rejected by pooled inner loss/tie rule')
        selection[arm] = {'candidates': candidates, 'winner': winner['candidate_id'], 'reason': reason,
                          'complete': all(c['valid'] for c in candidates)}
        sf, st = [temporal_features(partition_control(arm, ids, cohort, arrays, f'{arm}/outer/{label}', split['donors']))
                  for ids, label in ((train, 'train'), (test, 'test'))]
        scaler = SupportedInput.fit(sf, weights, cohort.iloc[train].window_id, cohort.iloc[train].video_id, snames, skinds)
        preprocessing[f'{arm}/outer'] = {'s': scaler.record(), 'heldout_s': scaler.diagnostics(st)}
        model = None
        if winner['candidate_type'] == 'joint_ridge':
            model = JointRidge.fit(baseline.x_scaler.transform(x[train]), scaler.transform(sf), baseline.y_scaler.transform(y[train]),
                                   weights, winner['lambda_x'], winner['lambda_s'])
        models[arm] = {'checkpoint_type': winner['candidate_type'], 'model_version': MODEL_VERSION,
                       'candidate_id': winner['candidate_id'], 'arm': arm, 'seed': 0, 'target': 'person',
                       'training_window_ids': cohort.iloc[train].window_id.tolist(), 's_scaler': scaler,
                       'model': model, 'lambda_x': winner['lambda_x'], 'lambda_s': winner['lambda_s']}
        predictions[arm] = predict_selected(models[arm], baseline, x[test], st, features=True)
        training_pred = predict_selected(models[arm], baseline, x[train], sf, features=True)
        diagnostics.append({'arm': arm, 'selected_type': winner['candidate_type'], 'selected_candidate': winner['candidate_id'],
                            'x_features': len(baseline.x_scaler.mask), 'x_supported': int(baseline.x_scaler.mask.sum()),
                            's_features': len(scaler.mask), 's_supported': int(scaler.mask.sum()),
                            'coefficient_count': int((x.shape[1] + (sf.shape[1] if model else 0) + 1) * y.shape[1]),
                            'training_mse': error_record(baseline.y_scaler.transform(y[train]), training_pred, cohort.iloc[train].video_id, baseline.valid_features)['validation_mse'],
                            'selected_inner_mse': winner['pooled_loss'], 'loss_state': 'closed_form; training outer fit, validation inner fits',
                            'complete': selection[arm]['complete']})
    return {'baseline': baseline, 'models': models, 'predictions': predictions, 'selection': selection,
            'baseline_selection': baseline_grid, 'split': split, 'preprocessing': preprocessing, 'diagnostics': diagnostics}


def predict_selected(saved, baseline, x, skeleton, *, features=False):
    if saved['model_version'] != MODEL_VERSION or saved['checkpoint_type'] not in ('baseline_only', 'joint_ridge'):
        raise ValueError('Invalid selected checkpoint type/version')
    if saved['checkpoint_type'] == 'baseline_only':
        if saved['model'] is not None or saved['lambda_s'] is not None:
            raise ValueError('Baseline-only artifact contains a correction')
        return baseline.predict(x)
    sf = skeleton if features else temporal_features(skeleton)
    if not isinstance(saved['model'], JointRidge): raise ValueError('Wrong joint checkpoint implementation')
    return saved['model'].predict(baseline.x_scaler.transform(x), saved['s_scaler'].transform(sf, skeleton_schema()[0]))


def fit_joint_outer_fold(root, fold, device='cpu'):
    root = Path(root)
    if device != 'cpu': raise ValueError('direct-v3 is a cached CPU experiment')
    run = check_run(root)
    verify_reused_readiness(root)
    directory = root / f'models/fold-{fold}'
    if (directory / 'fold-complete.json').exists():
        verify_joint_fold(root, fold)
        return
    if (root / 'reports/final-report-contract.json').exists(): raise ValueError('Sealed run; use a new run ID')
    cohort, arrays = load_cache(root)
    config = load_model_contract(root)
    train = np.flatnonzero(cohort.outer_fold.to_numpy() != fold)
    test = np.flatnonzero(cohort.outer_fold.to_numpy() == fold)
    result = nested_partition(cohort, arrays, train, test, baseline_schema(read_json(root / 'config/nuisance-schema.json')), config)
    directory.mkdir(parents=True, exist_ok=True)
    artifacts = []
    def save_json(name, value):
        path = directory / name
        write_json(path, value)
        artifacts.append(path)
    basepath = directory / 'person-baseline.joblib'
    atomic_save_joblib(basepath, result['baseline']); artifacts.append(basepath)
    baseline = joblib.load(basepath)
    tables = []
    for arm, saved in result['models'].items():
        modelpath = directory / arm / 'selected-model.joblib'
        saved.update(baseline_sha256=sha256_file(basepath), run_contract_sha256=sha256_file(root / 'config/run-contract.json'),
                     cache_sha256=sha256_file(root / 'config/cache-contract.json'), outer_fold=fold)
        atomic_save_joblib(modelpath, saved); artifacts.append(modelpath)
        restored = joblib.load(modelpath)
        sk = partition_control(arm, test, cohort, arrays, f'{arm}/reload-test', [])
        full = predict_selected(restored, baseline, arrays['baseline'][test], sk)
        np.testing.assert_array_equal(full, result['predictions'][arm])
        n, d = len(test), arrays['person'].shape[1]
        tables.append(pd.DataFrame({
            'window_id': np.repeat(cohort.iloc[test].window_id.to_numpy(), d), 'video_id': np.repeat(cohort.iloc[test].video_id.to_numpy(), d),
            'outer_fold': fold, 'arm': arm, 'seed': 0, 'fit_identity': 'deterministic', 'target': 'person',
            'target_feature': np.tile(np.arange(d), n), 'y_true': baseline.y_scaler.transform(arrays['person'][test]).ravel(),
            'y_pred_baseline': baseline.predict(arrays['baseline'][test]).ravel(), 'y_pred_full': full.ravel(), 'y_reference': 0.,
            'target_mean': np.tile(baseline.y_scaler.mean, n), 'target_scale': np.tile(baseline.y_scaler.scale, n),
            'target_training_variance': np.tile(baseline.y_scaler.variance, n), 'valid_feature': np.tile(baseline.valid_features, n),
            'model_artifact': str(modelpath.relative_to(root)), 'model_sha256': sha256_file(modelpath),
            'checkpoint_type': saved['checkpoint_type']}))
    for name, key in (('selection.json','selection'), ('baseline-selection.json','baseline_selection'), ('split-audit.json','split'),
                       ('preprocessing.json','preprocessing'), ('fit-diagnostics.json','diagnostics')): save_json(name, result[key])
    selection_rows = [{ 'arm': arm, **{k:v for k,v in c.items() if k != 'inner_folds'}, **row}
                      for arm, data in result['selection'].items() for c in data['candidates'] for row in c['inner_folds']]
    path = directory / 'inner-selection.parquet'
    pd.DataFrame(selection_rows).to_parquet(path, index=False); artifacts.append(path)
    path = root / f'predictions/fold-{fold}.parquet'
    temp = path.with_suffix('.tmp'); pd.concat(tables, ignore_index=True).to_parquet(temp, index=False); temp.replace(path); artifacts.append(path)
    write_once_json(directory / 'fold-complete.json', {'outer_fold': fold, 'protocol': 'direct-v3',
        'model_version': MODEL_VERSION, 'all_candidates_valid': all(d['complete'] for d in result['diagnostics']),
        'cache_contract_sha256': sha256_file(root / 'config/cache-contract.json'), 'validity_sha256': sha256_file(root / 'qc/readiness-summary.json'),
        'code_sha256': code_fingerprint_for_fit(), 'artifacts': {str(p.relative_to(root)): sha256_file(p) for p in artifacts}})
    verify_joint_fold(root, fold)
    print(f'Completed direct-v3 fold {fold}: '+', '.join("{}={}".format(r["arm"], r["selected_type"]) for r in result['diagnostics']), flush=True)


def code_fingerprint_for_fit():
    from .fi_contracts import code_fingerprint
    return code_fingerprint()


def verify_joint_fold(root, fold, *, loaded=None):
    """Read-only reload and numerical reconstruction; no refitting or provenance writes."""
    root = Path(root)
    cohort, arrays = loaded if loaded is not None else load_cache(root)
    directory = root / f'models/fold-{fold}'
    receipt = read_json(directory / 'fold-complete.json')
    required = {f'models/fold-{fold}/{name}' for name in ('person-baseline.joblib','selection.json','baseline-selection.json','split-audit.json','preprocessing.json','fit-diagnostics.json','inner-selection.parquet')}
    required |= {f'models/fold-{fold}/{arm}/selected-model.joblib' for arm in DIRECT_ARMS}
    required.add(f'predictions/fold-{fold}.parquet')
    if (set(receipt['artifacts']) != required or receipt['outer_fold'] != fold or receipt['protocol'] != 'direct-v3' or receipt['model_version'] != MODEL_VERSION):
        raise ValueError('Fold receipt omits required artifacts or has wrong identity')
    checked_file(root, 'config/cache-contract.json', receipt['cache_contract_sha256'])
    checked_file(root, 'qc/readiness-summary.json', receipt['validity_sha256'])
    for path, digest in receipt['artifacts'].items(): checked_file(root, path, digest)
    train = np.flatnonzero(cohort.outer_fold.to_numpy() != fold); test = np.flatnonzero(cohort.outer_fold.to_numpy() == fold)
    split = read_json(directory / 'split-audit.json')
    if split['outer_train'] != cohort.iloc[train].window_id.tolist() or split['outer_test'] != cohort.iloc[test].window_id.tolist():
        raise ValueError('Wrong outer source partition')
    mapping = assign_source_folds(cohort.iloc[train].video_id, 3)
    for f, row in enumerate(split['inner']):
        expected_train = [w for w,v in zip(cohort.iloc[train].window_id,cohort.iloc[train].video_id) if mapping[v] != f]
        expected_val = [w for w,v in zip(cohort.iloc[train].window_id,cohort.iloc[train].video_id) if mapping[v] == f]
        if row != {'fold': f, 'train': expected_train, 'validation': expected_val}: raise ValueError('Invalid inner source partition')
    if len(split['inner']) != 3: raise ValueError('Missing inner split')
    _verify_training_transforms(root, cohort, arrays, split, directory)
    sources = dict(zip(cohort.window_id, cohort.video_id))
    for row in split['preprocessing']:
        if row['fitted_window_ids'] != row['target_fitted_window_ids'] or {sources[w] for w in row['fitted_window_ids']} & {sources[w] for w in row['evaluated_window_ids']}:
            raise ValueError('Preprocessing source leakage')
    for row in split['donors']:
        ids = row['recipient_window_ids']
        if len(ids) != len(row['donor_window_ids']) or any(d not in ids or sources[d] == sources[r] for r,d in zip(ids,row['donor_window_ids'])):
            raise ValueError('Invalid partition-local donor')
    baseline = joblib.load(directory / 'person-baseline.joblib')
    schema = baseline_schema(read_json(root / 'config/nuisance-schema.json'))
    baseline.x_scaler.validate(schema[0])
    if baseline.x_scaler.training_window_ids != tuple(split['outer_train']) or baseline.y_scaler.training_window_ids != tuple(split['outer_train']):
        raise ValueError('Wrong fitted scaler identities')
    target = TargetStandardizer.fit(arrays['person'][train], equal_source_weights(cohort.iloc[train].video_id), cohort.iloc[train].window_id)
    for name in ('mean','scale','variance'): np.testing.assert_allclose(getattr(baseline.y_scaler,name), getattr(target,name), rtol=1e-12, atol=1e-12)
    if not np.array_equal(baseline.valid_features, target.variance > 1e-10): raise ValueError('Wrong target mask')
    selection = read_json(directory / 'selection.json'); config = load_model_contract(root)
    reference_grid = read_json(directory / 'baseline-selection.json')
    if len(reference_grid) != len(config.ridge_alphas) or {r['lambda_x'] for r in reference_grid} != set(config.ridge_alphas):
        raise ValueError('Shared baseline search differs')
    for candidate in reference_grid:
        expected = summarize({k:candidate[k] for k in ('candidate_id','candidate_type','lambda_x','lambda_s')}, candidate['inner_folds'])
        if expected != candidate: raise ValueError('Incorrect reference loss aggregation')
    best = min(r['pooled_loss'] for r in reference_grid)
    alpha = max(r['lambda_x'] for r in reference_grid if r['pooled_loss'] <= best + config.tie_atol + config.tie_rtol * abs(best))
    if baseline.ridge.lambda_x != alpha or baseline.ridge.lambda_s is not None:
        raise ValueError('Incorrect selected shared RGB baseline')
    fitted = fit_joint_baseline(arrays['baseline'][train], arrays['person'][train], cohort.iloc[train].window_id,
                               cohort.iloc[train].video_id, alpha, schema)
    _same_model(baseline.ridge, fitted.ridge)
    ledger = pd.read_parquet(directory / 'inner-selection.parquet')
    expected_ledger = pd.DataFrame([{'arm': arm, **{k:v for k,v in c.items() if k != 'inner_folds'}, **row}
                                   for arm, data in selection.items() for c in data['candidates'] for row in c['inner_folds']])
    try:
        columns = list(expected_ledger.columns)
        keys = ['arm','candidate_id','inner_fold']
        pd.testing.assert_frame_equal(ledger[columns].sort_values(keys).reset_index(drop=True),
                                      expected_ledger.sort_values(keys).reset_index(drop=True), check_dtype=False)
    except AssertionError as error: raise ValueError('Selection ledger differs from candidate records') from error
    table = pd.read_parquet(root / f'predictions/fold-{fold}.parquet')
    complete = True
    for arm in DIRECT_ARMS:
        candidates = selection[arm]['candidates']
        if {c['candidate_id'] for c in candidates} != {'baseline_only', *(s['candidate_id'] for s in candidate_specs(config))} or len(candidates) != 1 + len(candidate_specs(config)):
            raise ValueError('Unequal or missing candidate opportunities')
        for c in candidates:
            expected = summarize({k:c[k] for k in ('candidate_id','candidate_type','lambda_x','lambda_s')}, c['inner_folds'])
            if c['valid'] != expected['valid'] or c['pooled_loss'] != expected['pooled_loss']: raise ValueError('Incorrect pooled candidate loss')
        base_candidate = next(c for c in candidates if c['candidate_type'] == 'baseline_only')
        reference = next(c for c in reference_grid if c['lambda_x'] == alpha)
        if base_candidate['inner_folds'] != reference['inner_folds'] or base_candidate['lambda_x'] != alpha or base_candidate['lambda_s'] is not None:
            raise ValueError('Arms do not share selected baseline candidate')
        for c in candidates:
            if c['candidate_type'] != 'baseline_only' and {k:c[k] for k in ('candidate_id','candidate_type','lambda_x','lambda_s')} not in candidate_specs(config):
                raise ValueError('Incorrect candidate identity or penalties')
            expected_gain = reference['pooled_loss'] - c['pooled_loss'] if c['valid'] else None
            if c['improvement_over_baseline'] != expected_gain: raise ValueError('Incorrect candidate improvement')
        winner, reason = choose_candidate(candidates, config)
        if sum(c['selected'] for c in candidates) != 1 or not winner['selected']:
            raise ValueError('Inconsistent selected candidate flags')
        if selection[arm]['winner'] != winner['candidate_id'] or selection[arm]['reason'] != reason:
            raise ValueError('Invalid selection or tie reason')
        complete &= all(c['valid'] for c in candidates)
        modelpath = directory / arm / 'selected-model.joblib'
        saved = joblib.load(modelpath)
        if (saved['training_window_ids'] != split['outer_train'] or saved['arm'] != arm or saved['seed'] != 0 or saved['outer_fold'] != fold or saved['target'] != 'person'
                or saved['candidate_id'] != winner['candidate_id'] or saved['checkpoint_type'] != winner['candidate_type']
                or saved['baseline_sha256'] != sha256_file(directory / 'person-baseline.joblib')
                or saved['run_contract_sha256'] != sha256_file(root / 'config/run-contract.json') or saved['cache_sha256'] != sha256_file(root / 'config/cache-contract.json')):
            raise ValueError('Selected model identity/baseline mismatch')
        saved['s_scaler'].validate(skeleton_schema()[0])
        if saved['s_scaler'].training_window_ids != tuple(split['outer_train']): raise ValueError('Skeleton scaler training identities differ')
        if (saved['lambda_x'], saved['lambda_s']) != (winner['lambda_x'],winner['lambda_s']):
            raise ValueError('Selected penalties differ from inner choice')
        if saved['checkpoint_type'] == 'joint_ridge':
            sk_train = partition_control(arm,train,cohort,arrays,'verification/train',[])
            solved = JointRidge.fit(baseline.x_scaler.transform(arrays['baseline'][train]),
                saved['s_scaler'].transform(temporal_features(sk_train)), baseline.y_scaler.transform(arrays['person'][train]),
                equal_source_weights(cohort.iloc[train].video_id),saved['lambda_x'],saved['lambda_s'])
            _same_model(saved['model'],solved)
        sk = partition_control(arm, test, cohort, arrays, 'verification', [])
        full = predict_selected(saved, baseline, arrays['baseline'][test], sk)
        part = table[table.arm == arm]
        if part.duplicated(['window_id','target_feature']).any() or len(part) != len(test) * 256: raise ValueError('Missing or duplicated fold predictions')
        for name, expected in (('y_pred_full', full), ('y_pred_baseline', baseline.predict(arrays['baseline'][test]))):
            actual = part.pivot(index='window_id',columns='target_feature',values=name).loc[cohort.iloc[test].window_id,range(256)].to_numpy()
            np.testing.assert_allclose(actual, expected, rtol=1e-10, atol=1e-10, err_msg='Model reconstruction mismatch')
        if (set(part.model_artifact) != {str(modelpath.relative_to(root))} or set(part.model_sha256) != {sha256_file(modelpath)}
                or set(part.checkpoint_type) != {saved['checkpoint_type']} or set(part.fit_identity) != {'deterministic'}):
            raise ValueError('Prediction artifact or fit identity mismatch')
    if receipt['all_candidates_valid'] is not complete: raise ValueError('Candidate completeness disagrees')
    return table


def _same_model(saved, solved):
    if (not isinstance(saved, JointRidge) or saved.version != solved.version
            or (saved.lambda_x, saved.lambda_s) != (solved.lambda_x, solved.lambda_s)):
        raise ValueError("Invalid joint model implementation/version")
    for name in ('x_weight','s_weight','intercept'):
        np.testing.assert_allclose(getattr(saved,name),getattr(solved,name),rtol=1e-10,atol=1e-10,
                                   err_msg='Saved solution differs from declared regularized training fit')


def _verify_training_transforms(root, cohort, arrays, split, directory):
    """Recompute all fitted statistics from declared training sources, never test."""
    preprocessing = read_json(directory/'preprocessing.json')
    lookup = {w:i for i,w in enumerate(cohort.window_id)}
    partitions = [(f"inner-{r['fold']}", r['train'], r['validation']) for r in split['inner']]
    partitions += [('outer',split['outer_train'],split['outer_test'])]
    schema = baseline_schema(read_json(root/'config/nuisance-schema.json'))
    expected = {}; donors = []; expected_audit = []
    for scope, train_ids, heldout_ids in partitions:
        tr = np.array([lookup[w] for w in train_ids]); ho = np.array([lookup[w] for w in heldout_ids])
        weights = equal_source_weights(cohort.iloc[tr].video_id)
        xs = SupportedInput.fit(arrays['baseline'][tr],weights,train_ids,cohort.iloc[tr].video_id,*schema)
        ys = TargetStandardizer.fit(arrays['person'][tr],weights,train_ids)
        expected[scope] = {'x':xs.record(),'y':{k:v.tolist() if isinstance(v,np.ndarray) else v for k,v in asdict(ys).items()},
                           'heldout_x':xs.diagnostics(arrays['baseline'][ho])}
        expected_audit.append({'scope':scope,'fitted_window_ids':train_ids,'target_fitted_window_ids':train_ids,
                               'evaluated_window_ids':heldout_ids})
        for arm in DIRECT_ARMS:
            sf = temporal_features(partition_control(arm,tr,cohort,arrays,f'{arm}/{scope}/train',donors))
            label = 'test' if scope == 'outer' else 'validation'
            sh = temporal_features(partition_control(arm,ho,cohort,arrays,f'{arm}/{scope}/{label}',donors))
            scaler = SupportedInput.fit(sf,weights,train_ids,cohort.iloc[tr].video_id,*skeleton_schema())
            expected[f'{arm}/{scope}'] = {'s':scaler.record(),'heldout_s':scaler.diagnostics(sh)}
            if scope == 'outer':
                saved = joblib.load(directory/arm/'selected-model.joblib')['s_scaler']
                if saved.record() != scaler.record(): raise ValueError('Skeleton scaler differs from training-only statistics')
    import json
    if preprocessing != json.loads(json.dumps(expected)):
        raise ValueError('Preprocessing statistics or diagnostics differ from training-only recomputation')
    if split['preprocessing'] != expected_audit:
        raise ValueError('Missing or unexpected preprocessing boundary')
    if sorted(split['donors'],key=lambda r:r['scope']) != sorted(donors,key=lambda r:r['scope']):
        raise ValueError('Donors differ from partition-local context-only assignment')
    base = joblib.load(directory/'person-baseline.joblib')
    if base.x_scaler.record() != expected['outer']['x']:
        raise ValueError('RGB input scaler differs from training-only statistics')
