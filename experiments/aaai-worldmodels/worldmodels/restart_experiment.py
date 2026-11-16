"""Normal-only, source-cross-fitted fixed-time gait dynamics audit.

This is exploratory redevelopment because previous GAVD outcomes are known.
"""
from __future__ import annotations
import argparse
import hashlib
import importlib.metadata
import json
import time
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.model_selection import StratifiedKFold, KFold
from threadpoolctl import threadpool_limits
from .data import JOINT_IDS, sha256_file

C, H = 16, 8
LEGS = [[JOINT_IDS.index(j) for j in ids] for ids in ([25, 27, 29, 31], [26, 28, 30, 32])]
ARMS = ('own', 'bilateral', 'shifted', 'static')
ALPHAS = (.1, 1., 10., 100.)
BETAS = (0., .25, .5, 1.)
POSITIONS = np.array([0, 2, 4, 6, 9, 11, 13, 15])


def write_json(path, data):
    Path(path).write_text(json.dumps(data, indent=2, allow_nan=False, default=str) + '\n')


def hierarchical_weights(meta):
    """Equal source mass, then sequence mass, then window mass; mean weight one."""
    counts = meta.groupby(['source_id', 'sequence_id']).source_id.transform('size').to_numpy()
    seqs = meta[['source_id', 'sequence_id']].drop_duplicates().groupby('source_id').size()
    w = 1 / (counts * meta.source_id.map(seqs).to_numpy())
    return w * len(w) / w.sum()


def fill_prefix(x, valid):
    """Causal fill within the prefix, with zero before first observation."""
    if len(x.shape) != 4 or x.shape[1] < C or valid.shape != x.shape[:-1]:
        raise ValueError('Expected [N,T>=16,16,2] with matching validity')
    a = np.asarray(x[:, :C], float).copy()
    v = np.asarray(valid[:, :C], bool).copy()
    last = np.zeros_like(a[:, 0])
    known = np.zeros_like(v[:, 0])
    for t in range(C):
        last = np.where(v[:, t, :, None], a[:, t], last)
        known |= v[:, t]
        a[:, t] = last
    return a, v, last, known


def shift_offsets(window_ids):
    """Stable window-specific nonidentity offsets for seven selected observations."""
    return np.array([1 + int(hashlib.sha256(str(w).encode()).hexdigest()[:16], 16) % 6
                     for w in window_ids], dtype=int)


def features(x, valid, leg, arm, window_ids=None):
    """Prefix-only public inference boundary, independent of all future values."""
    if arm not in ARMS or leg not in (0, 1):
        raise ValueError('Unknown model arm or leg')
    a, v, anchor, known = fill_prefix(x, valid)
    own, opp = LEGS[leg], LEGS[1-leg]
    if arm == 'static':
        a[:] = anchor[:, None]
        v[:] = known[:, None]
    joints = own if arm == 'own' else own + opp
    selected = a[:, POSITIONS][:, :, joints].copy()
    observed = v[:, POSITIONS][:, :, joints].copy()
    if arm == 'shifted':
        if window_ids is None or len(window_ids) != len(x):
            raise ValueError('Shifted control needs stable window IDs')
        for i, offset in enumerate(shift_offsets(window_ids)):
            selected[i, :-1, 4:] = np.roll(selected[i, :-1, 4:], int(offset), axis=0)
            observed[i, :-1, 4:] = np.roll(observed[i, :-1, 4:], int(offset), axis=0)
    rel = selected - anchor[:, None, joints]
    f = np.concatenate([
        rel.reshape(len(x), -1),
        observed.reshape(len(x), -1),
        anchor[:, joints].reshape(len(x), -1)], axis=1)
    return f, anchor[:, own], known[:, own]


def targets(x, valid, leg):
    _, _, anchor, known = fill_prefix(x, valid)
    own = LEGS[leg]
    y = (x[:, C:, own] - anchor[:, None, own]).reshape(len(x), -1)
    mask = np.repeat((valid[:, C:, own] & known[:, None, own])[..., None], 2, axis=-1).reshape(len(x), -1)
    return y, mask


def fit_grid(f, y, mask, weights, alphas=ALPHAS):
    """Target-specific masked ridge fits; share solves for identical masks."""
    if not np.isfinite(f).all() or not np.isfinite(weights).all() or np.any(weights <= 0):
        raise ValueError('Finite features and positive finite weights required')
    mu = np.average(f, axis=0, weights=weights)
    sd = np.sqrt(np.average((f-mu)**2, axis=0, weights=weights))
    sd[sd < 1e-6] = 1
    z = (f-mu)/sd
    # Equal total coordinate-loss mass per window despite differing target support.
    count = mask.sum(axis=1)
    target_weights = np.divide(weights*y.shape[1], count,
                               out=np.zeros_like(weights, dtype=float), where=count > 0)
    groups = {}
    for k in range(y.shape[1]):
        groups.setdefault(np.packbits(mask[:, k]).tobytes(), []).append(k)
    models = {float(a): dict(mean=mu, scale=sd, coef=np.zeros((z.shape[1], y.shape[1])),
                            intercept=np.zeros(y.shape[1]), observed=np.zeros(y.shape[1], int)) for a in alphas}
    for columns in groups.values():
        use = mask[:, columns[0]]
        if not use.any():
            raise ValueError('No observed training target for an output coordinate')
        w, xx, yy = target_weights[use], z[use], y[use][:, columns]
        if not np.isfinite(yy).all():
            raise ValueError('Observed targets must be finite')
        xm, ym = np.average(xx, axis=0, weights=w), np.average(yy, axis=0, weights=w)
        xx, yy = xx-xm, yy-ym
        eig, vec = np.linalg.eigh((xx*w[:, None]).T @ xx)
        rhs = vec.T @ ((xx*w[:, None]).T @ yy)
        for alpha, model in models.items():
            coef = vec @ (rhs/(np.maximum(eig, 0)+alpha)[:, None])
            model['coef'][:, columns] = coef
            model['intercept'][columns] = ym-xm@coef
            model['observed'][columns] = use.sum()
    return models


def predict(model, f):
    return ((f-model['mean'])/model['scale']) @ model['coef'] + model['intercept']


def predict_checkpoint(checkpoint, x, valid, window_ids=None):
    """Return relative coordinates [N,8,4,2] using prefix observations only."""
    f, anchor, _ = features(x, valid, checkpoint['leg'], checkpoint['arm'], window_ids)
    residual = checkpoint['beta'] * predict(checkpoint['model'], f)
    return residual.reshape(len(x), H, 4, 2) + anchor[:, None]


def error_by_window(y, prediction, mask, horizon=H):
    t = y.reshape(len(y), H, 4, 2)[:, :horizon]
    p = prediction.reshape(len(y), H, 4, 2)[:, :horizon]
    m = mask.reshape(len(y), H, 4, 2)[:, :horizon]
    count = m.sum((1, 2, 3))
    sq = np.where(m, (t-p)**2, 0)
    return np.divide(sq.sum((1, 2, 3)), count, out=np.full(len(t), np.nan), where=count > 0)


def source_mean(meta, values):
    f = meta[['source_id', 'sequence_id', 'label']].copy()
    f['value'] = np.asarray(values)
    return f.groupby(['source_id', 'sequence_id', 'label'], sort=True).value.mean().groupby(['source_id', 'label']).mean().reset_index()


def source_folds(meta, n_splits=5, seed=2701):
    sources = meta[['source_id', 'label']].drop_duplicates().sort_values('source_id').reset_index(drop=True)
    if sources.groupby('source_id').label.nunique().max() != 1:
        raise ValueError('Mixed-label source')
    if sources.label.value_counts().min() < n_splits:
        raise ValueError('Insufficient sources per label')
    folds = {}
    splitter = StratifiedKFold(n_splits, shuffle=True, random_state=seed)
    for fold, (_, test) in enumerate(splitter.split(sources, sources.label)):
        folds.update({s: fold for s in sources.iloc[test].source_id})
    return folds


def bootstrap_mean(values, seed=812, reps=5000):
    a = np.asarray(values, float)
    if not len(a) or not np.isfinite(a).all():
        raise ValueError('Need finite source estimates')
    rng = np.random.default_rng(seed)
    b = rng.choice(a, (reps, len(a)), replace=True).mean(1)
    return dict(estimate=float(a.mean()), ci95=np.quantile(b, [.025, .975]).tolist(), n_sources=len(a), resamples=reps)


def rank_summary(frame, score, seed=271, reps=2000):
    y = (frame.label != 'normal').to_numpy(int)
    s = frame[score].to_numpy(float)
    rng = np.random.default_rng(seed)
    idx = [np.flatnonzero(y == k) for k in (0, 1)]
    draws = []
    for _ in range(reps):
        ii = np.concatenate([rng.choice(i, len(i), replace=True) for i in idx])
        draws.append((roc_auc_score(y[ii], s[ii]), average_precision_score(y[ii], s[ii])))
    return dict(auroc=float(roc_auc_score(y, s)), average_precision=float(average_precision_score(y, s)),
                auroc_ci95=np.quantile(np.array(draws)[:, 0], [.025, .975]).tolist(),
                ap_ci95=np.quantile(np.array(draws)[:, 1], [.025, .975]).tolist(),
                n_sources=len(frame), abnormal_prevalence=float(y.mean()))


def run(input_dir, output_dir):
    input_dir, output = Path(input_dir), Path(output_dir)
    if output.exists():
        raise FileExistsError('New run identity required')
    output.mkdir(parents=True)
    (output/'models').mkdir()
    started = time.monotonic()
    with np.load(input_dir/'windows.npz', allow_pickle=False) as z:
        x, valid = z['x'], z['valid']
    meta = pd.read_csv(input_dir/'windows.csv', keep_default_na=False)
    prep = json.loads((input_dir/'preparation.json').read_text())
    expected = {'frames': C+H, 'context_frames': C, 'sample_fps': 10.0}
    if any(prep.get(key) != value for key, value in expected.items()):
        raise ValueError('Preparation dimensions or nominal sample rate differ')
    for name, key in [('windows.npz', 'data_sha256'), ('windows.csv', 'metadata_sha256')]:
        if prep.get(key) != sha256_file(input_dir/name):
            raise ValueError(f'Preparation hash mismatch: {name}')
    if x.shape != (len(meta), C+H, 16, 2) or valid.shape != x.shape[:-1]:
        raise ValueError('Incorrect cohort contract')
    if not np.isfinite(x).all() or meta.window_id.duplicated().any():
        raise ValueError('Nonfinite normalized observations or duplicate window IDs')
    support = valid[:, C:] & valid[:, :C].any(1)[:, None]
    keep = np.ones(len(x), bool)
    for leg in LEGS:
        keep &= (support[:, :4, leg].sum((1, 2)) >= 4) & (support[:, :, leg].sum((1, 2)) >= 8)
    meta.loc[~keep].to_csv(output/'support_exclusions.csv', index=False)
    x, valid, meta = x[keep], valid[keep], meta.loc[keep].reset_index(drop=True)
    folds = source_folds(meta)
    meta['outer_fold'] = meta.source_id.map(folds)
    meta['shift_offset'] = shift_offsets(meta.window_id)
    meta.to_csv(output/'cohort.csv', index=False)
    protocol = Path(__file__).parents[1]/'docs/restart/protocol.md'
    identity = dict(code_sha256=sha256_file(__file__), protocol_sha256=sha256_file(protocol),
                    input_files={n: sha256_file(input_dir/n) for n in ('windows.npz', 'windows.csv', 'preparation.json')},
                    folds=folds, arms=ARMS, alphas=ALPHAS, betas=BETAS, inner_folds=3, outer_folds=5, seed=2701,
                    scope='Exploratory source cross-validation after earlier holdout exposure',
                    shift='Selected first seven opposite observations rolled by 1 + first64bits(SHA256(window_id)) modulo 6; latest preserved',
                    versions={n: importlib.metadata.version(n) for n in ('numpy', 'pandas', 'scikit-learn', 'scipy')})
    write_json(output/'identity.json', identity)
    records, selection, fit_ledger, fold_ledger = [], [], [], []
    names = (*ARMS, *(a+'_shrunk' for a in ARMS), 'persistence', 'damped_velocity', 'inference_shift', 'inference_static')
    predictions = {a: np.zeros((len(x), H, 8, 2), np.float32) for a in names}
    for fold in range(5):
        train = np.flatnonzero((meta.outer_fold != fold) & (meta.label == 'normal'))
        test = np.flatnonzero(meta.outer_fold == fold)
        tm, sm = meta.iloc[train].reset_index(drop=True), meta.iloc[test].reset_index(drop=True)
        sources = np.array(sorted(tm.source_id.unique()))
        inner = list(KFold(3, shuffle=True, random_state=3701+fold).split(sources))
        fold_ledger.append(dict(fold=fold, train_sources=sources.tolist(), heldout_sources=sorted(sm.source_id.unique()),
                                inner=[dict(train=sources[a].tolist(), validation=sources[b].tolist()) for a, b in inner]))
        errors = {arm: {h: [] for h in (4, 8)} for arm in names}
        for leg in range(2):
            target, mask = targets(x[train], valid[train], leg)
            sy, sv = targets(x[test], valid[test], leg)
            for arm in ARMS:
                f, _, _ = features(x[train], valid[train], leg, arm, tm.window_id)
                sf, sa, _ = features(x[test], valid[test], leg, arm, sm.window_id)
                validation = {(a, b): [] for a in ALPHAS for b in BETAS}
                for itr, ival in inner:
                    it = np.flatnonzero(tm.source_id.isin(sources[itr]))
                    iv = np.flatnonzero(tm.source_id.isin(sources[ival]))
                    grid = fit_grid(f[it], target[it], mask[it], hierarchical_weights(tm.iloc[it]))
                    for a, model in grid.items():
                        pp = predict(model, f[iv])
                        for beta in BETAS:
                            ev = error_by_window(target[iv], beta*pp, mask[iv])
                            validation[a, beta].extend(source_mean(tm.iloc[iv], ev).value.tolist())
                scores = {ab: float(np.mean(values)) for ab, values in validation.items()}
                if not np.isfinite(list(scores.values())).all():
                    raise ValueError('Inner validation lacks finite support')
                alpha = min(ALPHAS, key=lambda a: (scores[a, 1.], -a))
                ab = min(scores, key=lambda pair: (scores[pair], pair[1], -pair[0]))
                grid = fit_grid(f, target, mask, hierarchical_weights(tm), sorted({alpha, ab[0]}))
                for suffix, aa, beta in [('', alpha, 1.), ('_shrunk', ab[0], ab[1])]:
                    model = grid[aa]
                    pp = beta*predict(model, sf)
                    predictions[arm+suffix][test, :, leg*4:(leg+1)*4] = pp.reshape(len(test), H, 4, 2)+sa[:, None]
                    for horizon in (4, 8):
                        errors[arm+suffix][horizon].append(error_by_window(sy, pp, sv, horizon))
                    selection.append(dict(fold=fold, leg=leg, arm=arm+suffix, alpha=aa, beta=beta,
                                          inner_normal_mse=scores[aa, beta],
                                          all_grid_mse=[dict(alpha=a, beta=b, mse=s) for (a, b), s in scores.items()]))
                    path = output/'models'/f'fold{fold}_{arm+suffix}_leg{leg}.joblib'
                    joblib.dump(dict(model=model, arm=arm, leg=leg, beta=beta, training_sources=sources.tolist(),
                                     heldout_sources=sorted(sm.source_id.unique()), alpha=aa), path)
                    fit_ledger.append(dict(fold=fold, arm=arm+suffix, leg=leg, model_sha256=sha256_file(path),
                                           train_sources=len(sources), test_sources=int(sm.source_id.nunique()), training_windows=len(train)))
                    if arm == 'bilateral' and not suffix:
                        for control, mode in [('inference_shift', 'shifted'), ('inference_static', 'static')]:
                            ff, _, _ = features(x[test], valid[test], leg, mode, sm.window_id)
                            cp = predict(model, ff)
                            predictions[control][test, :, leg*4:(leg+1)*4] = cp.reshape(len(test), H, 4, 2)+sa[:, None]
                            for horizon in (4, 8):
                                errors[control][horizon].append(error_by_window(sy, cp, sv, horizon))
            a, v, last, _ = fill_prefix(x[test], valid[test])
            joints = LEGS[leg]
            velocity = (a[:, -1, joints]-a[:, -4, joints])/3
            velocity = np.where((v[:, -1, joints] & v[:, -4, joints])[..., None], velocity, 0)
            vv = (.5*np.arange(1, H+1)[None, :, None, None]*velocity[:, None]).reshape(len(test), -1)
            for arm, pp in [('persistence', np.zeros_like(sy)), ('damped_velocity', vv)]:
                predictions[arm][test, :, leg*4:(leg+1)*4] = pp.reshape(len(test), H, 4, 2)+last[:, None, joints]
                for horizon in (4, 8):
                    errors[arm][horizon].append(error_by_window(sy, pp, sv, horizon))
        for arm, by_horizon in errors.items():
            for horizon, legs in by_horizon.items():
                f = sm[['source_id', 'sequence_id', 'label', 'window_id']].copy()
                f['fold'], f['arm'], f['horizon'] = fold, arm, horizon
                f['mse'], f['left_mse'], f['right_mse'] = np.mean(legs, axis=0), legs[0], legs[1]
                records.append(f)
        write_json(output/'progress.json', dict(completed_folds=fold+1, seconds=time.monotonic()-started))
        print(json.dumps(dict(fold=fold, status='complete', normal_train_sources=len(sources),
                              test_sources=int(sm.source_id.nunique()), seconds=time.monotonic()-started)), flush=True)
    np.savez_compressed(output/'forecast_predictions.npz', **predictions)
    windows = pd.concat(records, ignore_index=True)
    windows.to_csv(output/'window_predictions.csv', index=False)
    sequence = windows.groupby(['source_id', 'sequence_id', 'label', 'fold', 'arm', 'horizon'])[['mse', 'left_mse', 'right_mse']].mean().reset_index()
    source = sequence.groupby(['source_id', 'label', 'fold', 'arm', 'horizon'])[['mse', 'left_mse', 'right_mse']].mean().reset_index()
    sequence.to_csv(output/'sequence_scores.csv', index=False)
    source.to_csv(output/'source_scores.csv', index=False)
    write_json(output/'selection.json', selection)
    write_json(output/'model_fit_ledger.json', fit_ledger)
    write_json(output/'fold_membership.json', fold_ledger)
    summaries, contrasts = [], []
    comparisons = [('persistence', 'bilateral'), ('own', 'bilateral'), ('shifted', 'bilateral'), ('static', 'bilateral'),
                   ('inference_shift', 'bilateral'), ('inference_static', 'bilateral'), ('persistence', 'bilateral_shrunk'),
                   ('bilateral', 'bilateral_shrunk'), ('own_shrunk', 'bilateral_shrunk'),
                   ('shifted_shrunk', 'bilateral_shrunk'), ('static_shrunk', 'bilateral_shrunk')]
    for horizon in (4, 8):
        normal = source[(source.label == 'normal') & (source.horizon == horizon)].pivot(index='source_id', columns='arm', values='mse')
        for arm in normal:
            summaries.append(dict(arm=arm, horizon=horizon, **bootstrap_mean(normal[arm])))
        for a, b in comparisons:
            contrasts.append(dict(reference=a, method=b, horizon=horizon, positive_favors=b, **bootstrap_mean(normal[a]-normal[b])))
    wide = source[source.horizon == 8].pivot(index=['source_id', 'label'], columns='arm', values='mse').reset_index()
    wide['excess_error'], wide['coordination_deficit'] = wide.bilateral-wide.persistence, wide.bilateral-wide.own
    quality = source_mean(meta, 1-meta.prefix_valid_fraction)
    wide = wide.merge(quality.rename(columns={'value': 'prefix_missingness'}), on=['source_id', 'label'], validate='one_to_one')
    wide.to_csv(output/'source_anomaly_scores.csv', index=False)
    anomaly = {score: rank_summary(wide, score) for score in ('bilateral', 'persistence', 'excess_error', 'coordination_deficit', 'prefix_missingness')}
    source.groupby(['label', 'arm', 'horizon']).agg(mse=('mse', 'mean'), sources=('source_id', 'nunique')).reset_index().to_csv(output/'condition_results.csv', index=False)
    summary = dict(status='complete', identity='medwm-dynamics-audit-v1', input=str(input_dir), windows=len(meta),
                   sequences=int(meta.sequence_id.nunique()), sources=int(meta.source_id.nunique()),
                   normal_sources=int(meta.loc[meta.label == 'normal', 'source_id'].nunique()),
                   cohort_by_label=meta.groupby('label').agg(windows=('source_id', 'size'), sequences=('sequence_id', 'nunique'), sources=('source_id', 'nunique')).reset_index().to_dict('records'),
                   normal_forecast=summaries, paired_contrasts=contrasts, anomaly=anomaly, seconds=time.monotonic()-started,
                   fit_count=len(fit_ledger), fit_count_note='80 saved arm/leg/fold checkpoints; some shrinkage variants reuse the same ridge fit.',
                   limitations=['Exploratory redevelopment after prior GAVD outcome exposure.',
                                'Source bootstrap intervals condition on fixed overlapping training folds and are not patient uncertainty.',
                                'GAVD source IDs do not prove independent patients or neurological diagnoses.',
                                'No treatment or clinical counterfactual is measured.'])
    write_json(output/'results.json', summary)
    write_json(output/'artifact_hashes.json', {str(p.relative_to(output)): sha256_file(p) for p in output.rglob('*') if p.is_file() and p.name != 'artifact_hashes.json'})
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    with threadpool_limits(limits=2):
        result = run(args.input, args.output)
    print(json.dumps({k: result[k] for k in ('status', 'windows', 'sources', 'seconds')}))


if __name__ == '__main__':
    main()
