"""Scaling-specific nested orchestration, isolated from sealed historical FI code.

Reuse the original model, scalers, controls, loss pooling and tie rules. The
separate orchestrator changes source grouping and per-fit penalty normalization.
Historical code is byte-preserved because old sealed runs bind its file digest.
"""
from dataclasses import asdict
import numpy as np
from ..future_innovation.fi_contracts import DIRECT_ARMS, equal_source_weights
from ..future_innovation.fi_cohort import assign_source_folds
from ..future_innovation.fi_joint_models import MODEL_VERSION, JointRidge, SupportedInput, fit_joint_baseline, skeleton_schema, temporal_features
from ..future_innovation.fi_joint_training import candidate_specs, choose_candidate, error_record, summarize, predict_selected
from ..future_innovation.fi_nested_training import isolated_split, partition_control


def nested_partition(cohort, arrays, train, test, schema, config, *,
                     penalty_reference_windows=None, group_column='video_id'):
    """The complete fitting/selection path, also used by fixed synthetic fixtures."""
    isolated_split(cohort, train, test)
    if penalty_reference_windows is not None and (not np.isfinite(penalty_reference_windows) or penalty_reference_windows <= 0):
        raise ValueError('Penalty reference window count must be finite and positive')
    def penalty(value, count):
        return value if penalty_reference_windows is None else value * count / penalty_reference_windows
    if set(cohort.iloc[train][group_column]) & set(cohort.iloc[test][group_column]):
        raise ValueError('Training and held-out source groups overlap')
    mapping = assign_source_folds(cohort.iloc[train][group_column], config.inner_folds)
    labels = np.array([mapping[v] for v in cohort.iloc[train][group_column]])
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
                                           cohort.iloc[fit].video_id, penalty(alpha, len(fit)), schema)
            rows.append({'inner_fold': f, 'valid': True, **error_record(baseline.y_scaler.transform(y[val]), baseline.predict(x[val]),
                                       cohort.iloc[val].video_id, baseline.valid_features),
                         'training_mse': error_record(baseline.y_scaler.transform(y[fit]), baseline.predict(x[fit]),
                                       cohort.iloc[fit].video_id, baseline.valid_features)['validation_mse'],
                         'loss_state': 'closed_form_fitted_model'})
            fits.append(baseline)
            if penalty_reference_windows is not None:
                rows[-1].update(fit_windows=len(fit), fit_weight_sum=float(equal_source_weights(cohort.iloc[fit].video_id).sum()),
                                effective_lambda_x=penalty(alpha, len(fit)), effective_lambda_s=None)
        baseline_grid.append(summarize({'candidate_id': f'rgb/x={alpha:g}', 'candidate_type': 'rgb_reference',
                                       'lambda_x': float(alpha), 'lambda_s': None}, rows))
        baseline_fits[alpha] = fits
    best = min(r['pooled_loss'] for r in baseline_grid)
    alpha = max(r['lambda_x'] for r in baseline_grid if r['pooled_loss'] <= best + config.tie_atol + config.tie_rtol * abs(best))
    reference = next(r for r in baseline_grid if r['lambda_x'] == alpha)
    inner_baselines = baseline_fits[alpha]
    baseline = fit_joint_baseline(x[train], y[train], cohort.iloc[train].window_id,
                                 cohort.iloc[train].video_id, penalty(alpha, len(train)), schema)
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
                    model = JointRidge.fit(xf, sf, yf, equal_source_weights(cohort.iloc[fit].video_id),
                                           penalty(spec['lambda_x'], len(fit)), penalty(spec['lambda_s'], len(fit)))
                    row = {'inner_fold': f, 'valid': True, **error_record(yv, model.predict(xv, sv), cohort.iloc[val].video_id, base.valid_features),
                           'training_mse': error_record(yf, model.predict(xf, sf), cohort.iloc[fit].video_id, base.valid_features)['validation_mse'],
                           'loss_state': 'closed_form_fitted_model'}
                except (ValueError, np.linalg.LinAlgError, FloatingPointError) as error:
                    row = {'inner_fold': f, 'valid': False, 'error': f'{type(error).__name__}: {error}',
                           'validation_squared_error': None, 'validation_weight': None, 'validation_mse': None, 'training_mse': None,
                           'loss_state': 'failed'}
                if penalty_reference_windows is not None:
                    row.update(fit_windows=len(fit), fit_weight_sum=float(equal_source_weights(cohort.iloc[fit].video_id).sum()),
                               effective_lambda_x=penalty(spec['lambda_x'], len(fit)), effective_lambda_s=penalty(spec['lambda_s'], len(fit)))
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
                                   weights, penalty(winner['lambda_x'], len(train)), penalty(winner['lambda_s'], len(train)))
        models[arm] = {'checkpoint_type': winner['candidate_type'], 'model_version': MODEL_VERSION,
                       'candidate_id': winner['candidate_id'], 'arm': arm, 'seed': 0, 'target': 'person',
                       'training_window_ids': cohort.iloc[train].window_id.tolist(), 's_scaler': scaler,
                       'model': model, 'lambda_x': winner['lambda_x'], 'lambda_s': winner['lambda_s']}
        if penalty_reference_windows is not None:
            models[arm].update(penalty_reference_windows=penalty_reference_windows,
                               rho_x=winner['lambda_x']/penalty_reference_windows,
                               rho_s=None if winner['lambda_s'] is None else winner['lambda_s']/penalty_reference_windows,
                               effective_lambda_x=penalty(winner['lambda_x'], len(train)),
                               effective_lambda_s=None if winner['lambda_s'] is None else penalty(winner['lambda_s'], len(train)),
                               fit_windows=len(train), fit_weight_sum=float(weights.sum()))
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
