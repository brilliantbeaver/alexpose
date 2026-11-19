"""Source-isolated calibration and endpoints for reference-masking study v2.

This module contains no outcome-driven model selection. Calibration rotations
hold their reference sources out of every fitted predictor and nuisance model.
Intervals condition on saved predictions, rather than incorporating retraining.
"""
from __future__ import annotations

import json
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import KFold

from .data import JOINT_IDS

FLOOR = 1e-8
ALPHAS = (.1, 1., 10., 100.)
DISTAL = [JOINT_IDS.index(j) for j in (25, 27, 29, 31, 26, 28, 30, 32)]


def _source_table(meta):
    required = {'source_id', 'label'}
    if not required.issubset(meta.columns) or not len(meta):
        raise ValueError('Need nonempty source_id and label metadata')
    frame = meta.copy()
    if 'component_id' not in frame:
        frame['component_id'] = frame.source_id
    keys = ['source_id', 'label', 'component_id']
    if 'outer_fold' in frame:
        keys.append('outer_fold')
    frame = frame[keys].drop_duplicates().sort_values('source_id').reset_index(drop=True)
    if frame.source_id.duplicated().any():
        raise ValueError('Source metadata must have one label, component, and fold')
    for col in ('label', 'outer_fold'):
        if col in frame and (frame.groupby('component_id')[col].nunique() != 1).any():
            raise ValueError(f'Related source component has conflicting {col}')
    return frame


def aggregate_windows(meta, values, *, name='value'):
    """Average windows per sequence, sequences per source, retaining grouping.

    Independent related-source components are averaged subsequently at endpoint
    calculation. A finite value is required for every common-support window;
    this function never silently changes the model cohort by dropping NaNs.
    """
    values = np.asarray(values, float)
    if values.shape != (len(meta),) or not np.isfinite(values).all():
        raise ValueError('One finite value is required per common-support window')
    if 'sequence_id' not in meta:
        raise ValueError('Sequence identity is required for hierarchical aggregation')
    sources = _source_table(meta)
    frame = meta[['source_id', 'sequence_id']].copy()
    frame[name] = values
    sequence = frame.groupby(['source_id', 'sequence_id'], sort=True)[name].mean()
    result = sequence.groupby('source_id').mean().reset_index()
    return sources.merge(result, on='source_id', validate='one_to_one')


def window_covariates(x, valid, times, context=16):
    """Observed-prefix motion in torso units/second and all-joint missingness.

    Motion averages each leg's observed adjacent-joint displacement rates, then
    gives both legs equal weight. Missing motion is never set to zero. Actual
    decoder times are required; future coordinates, support, and times are unused.
    """
    x, valid, times = np.asarray(x), np.asarray(valid), np.asarray(times, float)
    if x.ndim != 4 or x.shape[1] < context or x.shape[2:] != (16, 2):
        raise ValueError('Expected [N,T>=16,16,2] observations')
    if valid.shape != x.shape[:-1] or valid.dtype != np.bool_:
        raise ValueError('Expected Boolean joint support')
    if times.ndim != 2 or times.shape[0] != len(x) or times.shape[1] < context:
        raise ValueError('Expected measured prefix timestamps [N,T>=16]')
    a, v, t = x[:, :context], valid[:, :context], times[:, :context]
    dt = np.diff(t, axis=1)
    if not np.isfinite(t).all() or np.any(dt <= 0) or not np.isfinite(a[v]).all():
        raise ValueError('Observed prefix values and increasing timestamps must be finite')
    safe = np.where(v[..., None], a, 0)
    rate = np.linalg.norm(np.diff(safe[:, :, DISTAL], axis=1), axis=-1) / dt[..., None]
    pairs = v[:, 1:, DISTAL] & v[:, :-1, DISTAL]
    leg_motion = []
    counts = []
    for start in (0, 4):
        count = pairs[:, :, start:start+4].sum((1, 2))
        total = np.where(pairs[:, :, start:start+4], rate[:, :, start:start+4], 0).sum((1, 2))
        value = np.divide(total, count, out=np.full(len(x), np.nan), where=count > 0)
        leg_motion.append(value)
        counts.append(count)
    if not np.isfinite(leg_motion).all():
        raise ValueError('No observed adjacent-motion pair for a target leg; cohort amendment required')
    return pd.DataFrame({'prefix_motion': np.mean(leg_motion, axis=0),
                         'prefix_missingness': 1-v.mean((1, 2)),
                         'left_motion_pairs': counts[0], 'right_motion_pairs': counts[1]})


def source_covariates(x, valid, times, meta, context=16):
    """Construct exactly two source-level nuisance covariates without outcomes."""
    windows = window_covariates(x, valid, times, context)
    motion = aggregate_windows(meta, windows.prefix_motion, name='prefix_motion')
    missing = aggregate_windows(meta, windows.prefix_missingness, name='prefix_missingness')
    result = motion.merge(missing[['source_id', 'prefix_missingness']], on='source_id', validate='one_to_one')
    result['log_prefix_motion'] = np.log(result.prefix_motion + FLOOR)
    return result


def calibration_schedule(meta, outer_fold, *, seed=4701, rotations=3, nuisance_folds=3):
    """Plan disjoint predictor-fit, nuisance-OOF, calibration, and test roles.

    The nuisance model uses out-of-fit forecast errors within R. The predictor
    refitted on all R scores BOTH C and the held-out outer sources. Each rotation
    yields one calibrated score; the frozen algorithm averages its three scores.
    Group components, not individual windows or linked uploads, are partitioned.
    """
    sources = _source_table(meta)
    if 'outer_fold' not in sources:
        raise ValueError('Outer folds must be frozen in the cohort')
    if outer_fold not in set(sources.outer_fold):
        raise ValueError('Unknown outer fold')
    train = sources[(sources.outer_fold != outer_fold) & sources.label.eq('normal')]
    heldout = sources[sources.outer_fold == outer_fold]
    groups = np.array(sorted(train.component_id.unique()))
    if len(groups) < rotations or min(len(a) for a, _ in KFold(rotations).split(groups)) < nuisance_folds+1:
        raise ValueError('Insufficient normal components for nested calibration')
    splitter = KFold(rotations, shuffle=True, random_state=seed+int(outer_fold))
    output = []
    def group_sources(frame, ids):
        return sorted(frame.loc[frame.component_id.isin(ids), 'source_id'].tolist())
    for rotation, (ri, ci) in enumerate(splitter.split(groups)):
        rg, cg = groups[ri], groups[ci]
        inner = KFold(nuisance_folds, shuffle=True, random_state=seed+100+int(outer_fold)*10+rotation)
        oof = []
        for index, (fi, vi) in enumerate(inner.split(rg)):
            oof.append({'nuisance_fold': index,
                        'train_sources': group_sources(train, rg[fi]),
                        'validation_sources': group_sources(train, rg[vi])})
        record = {'outer_fold': int(outer_fold), 'rotation': rotation,
                  'predictor_fit_sources': group_sources(train, rg),
                  'calibration_sources': group_sources(train, cg),
                  'heldout_sources': sorted(heldout.source_id.tolist()),
                  'predictor_fit_components': rg.tolist(), 'calibration_components': cg.tolist(),
                  'nuisance_oof': oof, 'seed': seed}
        validate_calibration_schedule(record)
        output.append(record)
    return output


def validate_calibration_schedule(schedule):
    r, c, test = (set(schedule[k]) for k in ('predictor_fit_sources', 'calibration_sources', 'heldout_sources'))
    if not r or not c or not test or r & c or r & test or c & test:
        raise ValueError('Predictor, calibration, and held-out sources must be disjoint')
    seen = []
    for fold in schedule['nuisance_oof']:
        fit, val = set(fold['train_sources']), set(fold['validation_sources'])
        if fit & val or fit | val != r or not fit or not val:
            raise ValueError('Nuisance reference errors must be out-of-fit within R')
        seen.extend(fold['validation_sources'])
    if len(seen) != len(set(seen)) or set(seen) != r:
        raise ValueError('Every R source must have exactly one nuisance OOF forecast')
    return True


def _z(frame):
    motion = frame.prefix_motion.to_numpy(float)
    missing = frame.prefix_missingness.to_numpy(float)
    if not np.isfinite(motion).all() or np.any(motion < 0) or not np.isfinite(missing).all() or np.any((missing < 0) | (missing > 1)):
        raise ValueError('Finite nonnegative motion and missingness fractions are required')
    return np.column_stack([np.log(motion + FLOOR), missing])


def _log_error(frame):
    mse = frame.mse.to_numpy(float)
    if not np.isfinite(mse).all() or np.any(mse < 0):
        raise ValueError('Source MSE must be finite and nonnegative')
    return np.log(mse + FLOOR)


def _ridge(z, y, groups, alpha):
    # Equal component mass even when a verified component has multiple sources.
    _, inverse, counts = np.unique(groups, return_inverse=True, return_counts=True)
    weight = 1/counts[inverse]
    weight *= len(weight)/weight.sum()
    mean = np.average(z, axis=0, weights=weight)
    scale = np.sqrt(np.average((z-mean)**2, axis=0, weights=weight))
    scale[scale < 1e-6] = 1.
    zz = (z-mean)/scale
    center_z, center_y = np.average(zz, axis=0, weights=weight), np.average(y, weights=weight)
    centered = zz-center_z
    coefficient = np.linalg.solve((centered*weight[:, None]).T@centered+float(alpha)*np.eye(z.shape[1]),
                                  (centered*weight[:, None]).T@(y-center_y))
    return {'mean': mean.tolist(), 'scale': scale.tolist(), 'coefficient': coefficient.tolist(),
            'intercept': float(center_y-center_z@coefficient), 'alpha': float(alpha)}


def _predict_ridge(model, z):
    return ((z-np.asarray(model['mean']))/np.asarray(model['scale'])) @ np.asarray(model['coefficient'])+model['intercept']


def fit_nuisance_reference(frame, *, alphas=ALPHAS):
    """Select a small ridge on normal out-of-fit source errors by leave-group-out."""
    sources = _source_table(frame)
    if len(sources) != len(frame) or not frame.label.eq('normal').all():
        raise ValueError('Nuisance reference needs one out-of-fit row per normal source')
    z, y = _z(frame), _log_error(frame)
    groups = frame.get('component_id', frame.source_id).to_numpy()
    unique = sorted(set(groups))
    if len(unique) < 5:
        raise ValueError('At least five normal components are required for the nuisance reference')
    if not alphas or min(alphas) <= 0:
        raise ValueError('Positive ridge grid is required')
    validation = {float(alpha): [] for alpha in alphas}
    for group in unique:
        fit, val = groups != group, groups == group
        for alpha in validation:
            model = _ridge(z[fit], y[fit], groups[fit], alpha)
            validation[alpha].append(float(np.mean((_predict_ridge(model, z[val])-y[val])**2)))
    losses = {alpha: float(np.mean(values)) for alpha, values in validation.items()}
    chosen = min(losses, key=lambda a: (losses[a], -a))
    model = _ridge(z, y, groups, chosen)
    model.update(training_sources=sorted(frame.source_id.tolist()),
                 leave_component_out_mse={str(k): v for k, v in losses.items()},
                 covariates=['log_prefix_motion', 'prefix_missingness'],
                 response='log(source_mse + 1e-8)', numerical_floor=FLOOR,
                 covariate_min=z.min(axis=0).tolist(), covariate_max=z.max(axis=0).tolist(),
                 response_min=float(y.min()), response_max=float(y.max()))
    return model


def empirical_mid_cdf(reference, values):
    """Endpoint-smoothed mid-CDF: (0.5 + less + 0.5*equal)/(n+1)."""
    reference, values = np.asarray(reference, float), np.asarray(values, float)
    if reference.ndim != 1 or not len(reference) or not np.isfinite(reference).all() or not np.isfinite(values).all():
        raise ValueError('Finite nonempty calibration residuals are required')
    r = np.sort(reference)
    less = np.searchsorted(r, values, side='left')
    right = np.searchsorted(r, values, side='right')
    return (.5+less+.5*(right-less))/(len(r)+1)


def fit_source_calibrator(oof_reference, calibration, *, schedule, alphas=ALPHAS):
    """Fit g on R out-of-fit rows, then store C residuals for that same R model.

    The caller must obtain calibration MSE from the predictor fitted on exactly
    predictor_fit_sources. The schedule records the OOF fits that produced the
    nuisance response rows. It is validated here and must be saved with outputs.
    """
    validate_calibration_schedule(schedule)
    if set(oof_reference.source_id) != set(schedule['predictor_fit_sources']):
        raise ValueError('Nuisance reference rows differ from scheduled predictor-fit sources')
    if set(calibration.source_id) != set(schedule['calibration_sources']):
        raise ValueError('Calibration rows differ from held-out calibration sources')
    if len(_source_table(calibration)) != len(calibration) or not calibration.label.eq('normal').all():
        raise ValueError('Calibration requires one row per normal source')
    if set(oof_reference.get('component_id', oof_reference.source_id)) & set(calibration.get('component_id', calibration.source_id)):
        raise ValueError('A related component crossed the calibration boundary')
    expected = {source: set(fold['train_sources']) for fold in schedule['nuisance_oof']
                for source in fold['validation_sources']}
    for frame, role in ((oof_reference, 'out-of-fit reference'), (calibration, 'calibration')):
        if 'forecast_training_sources' not in frame:
            raise ValueError(f'{role} rows require forecast_training_sources provenance')
        for row in frame.itertuples():
            recorded = row.forecast_training_sources
            recorded = json.loads(recorded) if isinstance(recorded, str) else recorded
            if not isinstance(recorded, (list, tuple)):
                raise ValueError('Forecast provenance must be a source list or its JSON encoding')
            allowed = expected[row.source_id] if role == 'out-of-fit reference' else set(schedule['predictor_fit_sources'])
            if set(recorded) != allowed or len(recorded) != len(set(recorded)):
                raise ValueError(f'Forecast provenance violates {role} membership')
    model = fit_nuisance_reference(oof_reference, alphas=alphas)
    residuals = _log_error(calibration)-_predict_ridge(model, _z(calibration))
    # Endpoints use component means if one component contains several sources.
    reference = pd.DataFrame({'component_id': calibration.get('component_id', calibration.source_id), 'residual': residuals}).groupby('component_id', sort=True).residual.mean()
    return {'identity': 'reference-v2-source-calibrator-v1', 'nuisance_model': model,
            'nuisance_reference_rows': oof_reference.to_dict('records'),
            'calibration_sources': sorted(calibration.source_id.tolist()),
            'calibration_components': reference.index.tolist(),
            'calibration_residuals': reference.tolist(),
            'calibration_source_rows': calibration.assign(residual=residuals).to_dict('records'),
            'schedule': schedule, 'cdf': '(0.5 + count_less + 0.5*count_equal)/(n+1)',
            'interpretation': 'Exploratory normal-reference tail score; no conformal coverage guarantee'}


def score_source_calibrator(calibrator, frame, *, allow_reference=False):
    """Return scalar source residuals, normal-reference CDF, and two-sided scores."""
    if frame.source_id.duplicated().any():
        raise ValueError('Expected one scored row per source')
    sources = set(frame.source_id)
    training = set(calibrator['nuisance_model']['training_sources'])
    reference = set(calibrator['calibration_sources'])
    if not allow_reference and sources & (training | reference):
        raise ValueError('Scored sources overlap model or calibration sources')
    components = set(frame.get('component_id', frame.source_id))
    fitted_components = set(calibrator['schedule'].get('predictor_fit_components', training))
    fitted_components |= set(calibrator['schedule'].get('calibration_components', reference))
    if not allow_reference and components & fitted_components:
        raise ValueError('Scored components overlap model or calibration components')
    if 'forecast_training_sources' in frame:
        expected = set(calibrator['schedule']['predictor_fit_sources'])
        for recorded in frame.forecast_training_sources:
            recorded = json.loads(recorded) if isinstance(recorded, str) else recorded
            if not isinstance(recorded, (list, tuple)) or set(recorded) != expected or len(recorded) != len(set(recorded)):
                raise ValueError('Scored forecasts do not use the scheduled predictor training pool')
    residual = _log_error(frame)-_predict_ridge(calibrator['nuisance_model'], _z(frame))
    cdf = empirical_mid_cdf(calibrator['calibration_residuals'], residual)
    result = frame.copy()
    result['residual'], result['cdf'], result['two_sided'] = residual, cdf, np.abs(2*cdf-1)
    return result


def ensemble_rotation_scores(rotations, *, score='two_sided'):
    """Equal mean of the three frozen rotation scores, with no outcome selection."""
    if len(rotations) != 3:
        raise ValueError('The frozen algorithm requires all three calibration rotations')
    tables = []
    ids = set(rotations[0].source_id)
    for i, frame in enumerate(rotations):
        if set(frame.source_id) != ids or frame.source_id.duplicated().any():
            raise ValueError('Rotation scores need identical unique source support')
        if not np.isfinite(frame[score]).all():
            raise ValueError('Every rotation score must be finite')
        tables.append(frame[['source_id', score]].rename(columns={score: f'rotation_{i}'}))
    result = _source_table(rotations[0])
    for frame in tables:
        result = result.merge(frame, on='source_id', validate='one_to_one')
    result[score] = result[[f'rotation_{i}' for i in range(3)]].mean(axis=1)
    return result


def independent_score_table(frame, scores):
    """One row per verified independent component, equal mean across its sources."""
    scores = list(dict.fromkeys(scores))
    sources = _source_table(frame)
    if len(sources) != len(frame):
        raise ValueError('Scores require one row per source before component aggregation')
    if not np.isfinite(frame[list(scores)]).all().all():
        raise ValueError('Scores must be finite and share common source support')
    copy = frame.copy()
    if 'component_id' not in copy:
        copy['component_id'] = copy.source_id
    grouping = ['component_id', 'label']+(['outer_fold'] if 'outer_fold' in copy else [])
    return copy.groupby(grouping, sort=True)[list(scores)].mean().reset_index()


def _auc_pairs(positive, negative):
    difference = np.asarray(positive)[:, None]-np.asarray(negative)[None, :]
    return (difference > 0).astype(float)+.5*(difference == 0)


def within_fold_summary(frame, score):
    groups = independent_score_table(frame, [score])
    if 'outer_fold' not in groups:
        raise ValueError('Frozen outer-fold membership is required')
    rows = []
    for fold, part in groups.groupby('outer_fold', sort=True):
        y = part.label.ne('normal').to_numpy()
        negative, positive = part.loc[~y, score], part.loc[y, score]
        if not len(negative) or not len(positive):
            raise ValueError('Each fold needs both normal and abnormal independent components')
        auc = float(_auc_pairs(positive, negative).mean())
        rows.append({'fold': int(fold), 'normal': len(negative), 'abnormal': len(positive),
                     'pairs': len(negative)*len(positive), 'auroc': auc,
                     'average_precision': float(average_precision_score(y, part[score]))})
    weighted = float(np.average([r['auroc'] for r in rows], weights=[r['pairs'] for r in rows]))
    return {'score': score, 'pair_weighted_auroc': weighted,
            'equal_fold_auroc': float(np.mean([r['auroc'] for r in rows])), 'folds': rows,
            'independent_components': len(groups), 'source_rows': len(frame)}


def paired_within_fold_bootstrap(frame, candidate, reference, *, reps=5000, seed=5701):
    """Paired fold/class component bootstrap, excluding every cross-fold pair."""
    if not isinstance(reps, (int, np.integer)) or reps < 2:
        raise ValueError('At least two bootstrap repetitions are required')
    groups = independent_score_table(frame, [candidate, reference])
    if 'outer_fold' not in groups:
        raise ValueError('Frozen outer-fold membership is required')
    rng = np.random.default_rng(seed)
    totals = np.zeros((reps, 2))
    point = np.zeros(2)
    pair_count, per_fold = 0, []
    for fold, part in groups.groupby('outer_fold', sort=True):
        normal, abnormal = part[part.label.eq('normal')], part[part.label.ne('normal')]
        n, p = len(normal), len(abnormal)
        if not n or not p:
            raise ValueError('Each fold needs both classes')
        nw = rng.multinomial(n, np.full(n, 1/n), size=reps)
        pw = rng.multinomial(p, np.full(p, 1/p), size=reps)
        fold_auc = []
        for j, score in enumerate((candidate, reference)):
            matrix = _auc_pairs(abnormal[score], normal[score])
            totals[:, j] += np.einsum('bi,ij,bj->b', pw, matrix, nw, optimize=True)
            point[j] += matrix.sum()
            fold_auc.append(float(matrix.mean()))
        pair_count += n*p
        per_fold.append({'fold': int(fold), 'normal': n, 'abnormal': p,
                         'candidate_auroc': fold_auc[0], 'reference_auroc': fold_auc[1],
                         'difference': fold_auc[0]-fold_auc[1], 'pairs': n*p})
    totals /= pair_count
    point /= pair_count
    difference = totals[:, 0]-totals[:, 1]
    return {'candidate': candidate, 'reference': reference, 'estimate': float(point[0]-point[1]),
            'candidate_auroc': float(point[0]), 'reference_auroc': float(point[1]),
            'ci95': np.quantile(difference, [.025, .975]).tolist(),
            'candidate_ci95': np.quantile(totals[:, 0], [.025, .975]).tolist(),
            'reference_ci95': np.quantile(totals[:, 1], [.025, .975]).tolist(),
            'folds': per_fold, 'resamples': int(reps), 'seed': seed,
            'unit': 'verified source-relation component',
            'resampling': 'paired independently within outer fold and binary label',
            'uncertainty': 'Conditional on saved predictions; excludes retraining and unknown participant overlap'}


def pooled_rank_summary(frame, score, *, reps=2000, seed=2706):
    """Secondary pooled ranking; scores from separate folds must be comparable."""
    groups = independent_score_table(frame, [score])
    y, values = groups.label.ne('normal').to_numpy(), groups[score].to_numpy()
    indexes = [np.flatnonzero(~y), np.flatnonzero(y)]
    if any(not len(i) for i in indexes):
        raise ValueError('Both classes are required')
    rng, draws = np.random.default_rng(seed), []
    for _ in range(reps):
        ii = np.concatenate([rng.choice(i, len(i), replace=True) for i in indexes])
        draws.append((roc_auc_score(y[ii], values[ii]), average_precision_score(y[ii], values[ii])))
    draws = np.asarray(draws)
    return {'score': score, 'auroc': float(roc_auc_score(y, values)),
            'average_precision': float(average_precision_score(y, values)),
            'auroc_ci95': np.quantile(draws[:, 0], [.025, .975]).tolist(),
            'ap_ci95': np.quantile(draws[:, 1], [.025, .975]).tolist(),
            'prevalence': float(y.mean()), 'independent_components': len(groups),
            'resamples': reps, 'seed': seed, 'uncertainty': 'Conditional class-stratified component bootstrap'}


def normal_forecast_contrast(frame, method, reference, *, reps=5000, seed=5702):
    """Normal component-average forecast error and paired reference-minus-method."""
    normal = frame[frame.label.eq('normal')]
    groups = independent_score_table(normal, [method, reference])
    if not len(groups):
        raise ValueError('Normal source components are required')
    rng = np.random.default_rng(seed)
    indices = rng.integers(len(groups), size=(reps, len(groups)))
    prediction, baseline = groups[method].to_numpy(), groups[reference].to_numpy()
    if np.any(prediction < 0) or np.any(baseline < 0):
        raise ValueError('Forecast errors must be nonnegative')
    pp, bb = prediction[indices].mean(1), baseline[indices].mean(1)
    relative = np.divide(bb-pp, bb, out=np.full(reps, np.nan), where=bb > 0)
    return {'method': method, 'reference': reference, 'method_mse': float(prediction.mean()),
            'reference_mse': float(baseline.mean()), 'estimate': float((baseline-prediction).mean()),
            'ci95': np.quantile(bb-pp, [.025, .975]).tolist(),
            'method_ci95': np.quantile(pp, [.025, .975]).tolist(),
            'relative_reduction': float((baseline.mean()-prediction.mean())/baseline.mean()) if baseline.mean() > 0 else None,
            'relative_ci95': np.nanquantile(relative, [.025, .975]).tolist() if np.isfinite(relative).any() else None,
            'normal_components': len(groups), 'resamples': reps, 'seed': seed,
            'positive_favors': method, 'uncertainty': 'Conditional on saved cross-validation predictions'}


def fit_normal_distance(features, source_ids, *, shrinkage=.2):
    """Normal-only shrinkage Mahalanobis reference for explicit nuisance controls.

    Caller supplies only permitted normal training sources and a fixed feature
    definition. Coordinate/motion/missingness features are separate controls.
    No labels or evaluation scores select the shrinkage coefficient.
    """
    x = np.asarray(features, float)
    if x.ndim != 2 or len(x) < 3 or len(x) != len(source_ids) or len(set(source_ids)) != len(source_ids) or not np.isfinite(x).all():
        raise ValueError('Need finite features from at least three unique normal sources')
    if not 0 < shrinkage <= 1:
        raise ValueError('Fixed positive covariance shrinkage is required')
    mean, scale = x.mean(0), x.std(0)
    scale[scale < 1e-6] = 1.
    z = (x-mean)/scale
    cov = z.T@z/max(1, len(z)-1)
    cov = (1-shrinkage)*cov+shrinkage*np.eye(x.shape[1])
    return {'mean': mean.tolist(), 'scale': scale.tolist(), 'precision': np.linalg.inv(cov).tolist(),
            'shrinkage': shrinkage, 'training_sources': sorted(source_ids)}


def score_normal_distance(model, features):
    x = np.asarray(features, float)
    if x.ndim != 2 or not np.isfinite(x).all():
        raise ValueError('Finite source features are required')
    z = (x-np.asarray(model['mean']))/np.asarray(model['scale'])
    return np.maximum(0., np.einsum('ni,ij,nj->n', z, np.asarray(model['precision']), z))
