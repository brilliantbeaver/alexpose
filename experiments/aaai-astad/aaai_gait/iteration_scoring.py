"""Observable, source-level measurements for the separate iteration02 study.

Nothing in this module selects a recipe from abnormal test outcomes. The two
score channels retain their different meanings; their fixed mixture is not a
probability of illness. All fitting operates on one row per source recording.
"""
from __future__ import annotations

from dataclasses import dataclass
import warnings
import numpy as np
import pandas as pd
from sklearn.covariance import LedoitWolf
from sklearn.preprocessing import RobustScaler, StandardScaler
from sklearn.metrics import roc_auc_score, average_precision_score

from .data import JOINTS
from .features import _angle, build_features

READOUTS = tuple(range(25, 33))
FOCAL = ('normal', 'parkinsons', 'stroke', 'cerebral palsy', 'myopathic')
LOSS_COLUMNS = ('loss_base', 'loss_own', 'loss_both', 'benefit', 'sham_benefit')


def _summary(values, kind, axis=1):
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', RuntimeWarning)
        if kind == 'median':
            return np.nanmedian(values, axis=axis)
        if kind == 'iqr':
            return np.nanpercentile(values, 75, axis=axis) - np.nanpercentile(values, 25, axis=axis)
        if kind == 'mad':
            return np.nanmedian(np.abs(values - np.expand_dims(np.nanmedian(values, axis=axis), axis)), axis=axis)
    raise ValueError(kind)


def local_movement_features(x, fps=15.):
    """Sixteen projected descriptors. Angles are degrees; XY is torso units.

    Velocities use only adjacent measured resampled observations and raw time.
    These full-observation summaries are for the detector, never a predictor
    with hidden values. No per-joint excursion normalization is performed.
    """
    x = np.asarray(x, float)
    if x.ndim != 4 or x.shape[2:] != (16, 2) or fps <= 0:
        raise ValueError('Expected [window,time,16,2] and positive sampling rate')
    index = {j: k for k, j in enumerate(JOINTS)}
    columns, names, knees, excursions = [], [], [], []
    for side, hip, knee, ankle in [('left', 23, 25, 27), ('right', 24, 26, 28)]:
        angle = _angle(x[:, :, index[hip]], x[:, :, index[knee]], x[:, :, index[ankle]])
        a = x[:, :, index[ankle]]
        med = _summary(angle, 'median'); exc = _summary(a, 'iqr')
        columns.extend([med, _summary(angle, 'iqr'), exc[:, 0], exc[:, 1],
                        _summary(np.diff(a, axis=1)[:, :, 0] * fps, 'mad'),
                        _summary(np.diff(a, axis=1)[:, :, 1] * fps, 'mad')])
        names.extend([f'{side}_knee_median_deg', f'{side}_knee_iqr_deg',
                      f'{side}_ankle_x_iqr_torso', f'{side}_ankle_y_iqr_torso',
                      f'{side}_ankle_vx_mad_torso_per_s', f'{side}_ankle_vy_mad_torso_per_s'])
        knees.append(med); excursions.append(exc[:, 0])
    torso = (x[:, :, index[11]] + x[:, :, index[12]] - x[:, :, index[23]] - x[:, :, index[24]]) / 2
    lean = np.degrees(np.arctan2(torso[:, :, 0], -torso[:, :, 1]))
    columns.extend([_summary(lean, 'median'), _summary(lean, 'iqr'),
                    knees[0] - knees[1], excursions[0] - excursions[1]])
    names.extend(['trunk_inclination_median_deg', 'trunk_inclination_iqr_deg',
                  'left_minus_right_knee_median_deg', 'left_minus_right_ankle_x_iqr_torso'])
    return np.stack(columns, axis=1), names


def elementary_timing_features(x, fps=15., max_lag=8, minimum_pairs=8):
    """Fixed raw-clock bilateral waveform controls, not a validated PCI.

    For each ankle axis: zero-lag correlation, largest absolute correlation,
    signed lag at that maximum, and same-time RMS left-right difference.
    Lags are searched from -8 to +8 samples, with at least eight finite pairs.
    Ties choose the smallest absolute lag, then the negative lag. A static
    signal has undefined correlation, never an assigned phase of zero.
    """
    x = np.asarray(x, float); idx = {j: k for k, j in enumerate(JOINTS)}
    out = np.full((len(x), 8), np.nan)
    lags = sorted(range(-max_lag, max_lag + 1), key=lambda v: (abs(v), v))
    for n, window in enumerate(x):
        for axis in range(2):
            left, right = window[:, idx[27], axis], window[:, idx[28], axis]
            both = np.isfinite(left) & np.isfinite(right)
            if both.sum() >= minimum_pairs:
                out[n, axis * 4 + 3] = np.sqrt(np.mean((left[both] - right[both]) ** 2))
            candidates = []
            for lag in lags:
                a = left[max(0, lag):len(left) + min(0, lag)]
                b = right[max(0, -lag):len(right) - max(0, lag)]
                valid = np.isfinite(a) & np.isfinite(b)
                if valid.sum() < minimum_pairs or np.std(a[valid]) < 1e-8 or np.std(b[valid]) < 1e-8:
                    continue
                corr = float(np.corrcoef(a[valid], b[valid])[0, 1])
                if lag == 0:
                    out[n, axis * 4] = corr
                candidates.append((abs(corr), lag))
            if candidates:
                best = max(candidates, key=lambda item: item[0])
                out[n, axis * 4 + 1:axis * 4 + 3] = best[0], best[1] / fps
    names = [f'ankle_{axis}_{suffix}' for axis in ('x', 'y') for suffix in
             ('zero_lag_correlation', 'max_abs_correlation', 'lag_seconds', 'bilateral_rms_torso')]
    return out, names


def aggregate_descriptors(values, names, metadata):
    """Equal windows per sequence, then equal sequences per recording."""
    if len(values) != len(metadata):
        raise ValueError('Descriptor/metadata alignment failed')
    table = pd.DataFrame(values, columns=names)
    table[['source_id', 'sequence_id']] = metadata[['source_id', 'sequence_id']].reset_index(drop=True)
    return table.groupby(['source_id', 'sequence_id'])[names].mean().groupby('source_id').mean()


def source_descriptor_tables(x, metadata):
    result = {}
    for name, fn in [('M', local_movement_features), ('motion36', build_features),
                     ('timing8', elementary_timing_features)]:
        values, columns = fn(x)
        result[name] = aggregate_descriptors(values, columns, metadata)
    available = np.isfinite(x).all(axis=-1).mean(axis=1)
    result['quality16'] = aggregate_descriptors(available, [f'available_{j}' for j in JOINTS], metadata)
    return result


@dataclass
class NormalDistance:
    medians: np.ndarray
    scaler: object
    covariance: object
    names: tuple
    fit_sources: tuple
    all_missing: tuple

    @classmethod
    def fit(cls, table, sources, robust=True):
        sources = tuple(sorted(sources)); a = table.loc[list(sources)].to_numpy(float)
        if len(sources) < 2 or len(set(sources)) != len(sources):
            raise ValueError('At least two unique fitting-normal source rows required')
        med = np.array([np.median(col[np.isfinite(col)]) if np.isfinite(col).any() else 0 for col in a.T])
        filled = np.where(np.isfinite(a), a, med)
        scaler = (RobustScaler() if robust else StandardScaler()).fit(filled)
        cov = LedoitWolf().fit(scaler.transform(filled))
        return cls(med, scaler, cov, tuple(table.columns), sources,
                   tuple(table.columns[~np.isfinite(a).any(axis=0)]))

    def score(self, table):
        a = table.loc[:, list(self.names)].to_numpy(float)
        z = self.scaler.transform(np.where(np.isfinite(a), a, self.medians))
        return pd.Series(np.maximum(self.covariance.mahalanobis(z), 0) / len(self.names), index=table.index)

    def audit(self):
        return dict(fit_sources=list(self.fit_sources), features=list(self.names),
                    imputation_medians=self.medians.tolist(), all_missing=list(self.all_missing),
                    covariance=self.covariance.covariance_.tolist(),
                    scaler=type(self.scaler).__name__, dimension_normalized=True,
                    aggregation='windows within sequences, sequences within source, then distance')


def compress_target_rows(rows):
    """Average mask draws within an identical window/joint/patch target.

    Later aggregation weights each target equally. Repeated masks never change
    the number of physical target patches used by the source support gate.
    """
    rows = rows.copy()
    if 'valid' in rows:
        rows = rows.loc[rows.valid.astype(bool)].copy()
    if rows.empty:
        return rows
    rows['sham_benefit'] = rows.loss_base - rows.loss_own
    rows['benefit'] = rows.loss_own - rows.loss_both
    keys = ['window_index', 'joint', 'patch_index']
    means = rows.groupby(keys, sort=True)[list(LOSS_COLUMNS)].mean()
    means['mask_draws'] = rows.groupby(keys).size()
    return means.reset_index()


def aggregate_target_profiles(compact, metadata, minimum_patches=4):
    """Target -> window -> sequence -> source, preserving the joint axis."""
    if compact.empty:
        return pd.DataFrame(), pd.DataFrame()
    meta = metadata.reset_index(drop=True).copy()
    if 'window_index' not in meta:
        meta['window_index'] = np.arange(len(meta))
    keep = ['window_index', 'source_id', 'sequence_id', 'start_seconds']
    table = compact.merge(meta[keep], on='window_index', validate='many_to_one')
    if table.source_id.isna().any():
        raise ValueError('A target row has no original source/time key')
    table['patch_start_ns'] = np.rint((table.start_seconds + table.patch_index * 4 / 15) * 1e9).astype(np.int64)
    distinct = table[['source_id', 'sequence_id', 'joint', 'patch_start_ns']].drop_duplicates()
    counts = distinct.groupby(['source_id', 'joint']).size().rename('distinct_patches')
    windows = table.groupby(['source_id', 'sequence_id', 'window_index', 'joint'])[list(LOSS_COLUMNS)].mean()
    sequences = windows.groupby(['source_id', 'sequence_id', 'joint']).mean()
    sources = sequences.groupby(['source_id', 'joint']).mean().join(counts)
    coverage = sources[['distinct_patches']].reset_index()
    eligible = []
    for source, group in sources.reset_index().groupby('source_id'):
        by = group.set_index('joint')
        if set(READOUTS).issubset(by.index) and by.loc[list(READOUTS), 'distinct_patches'].ge(minimum_patches).all():
            eligible.append(source)
    sources = sources.loc[sources.index.get_level_values('source_id').isin(eligible)]
    return sources.reset_index(), coverage


def bilateral_profiles(profiles, field='benefit'):
    """Average four joint-specific source values equally for each side."""
    if profiles.empty:
        return pd.DataFrame(columns=['left', 'right'])
    p = profiles.pivot(index='source_id', columns='joint', values=field)
    return pd.DataFrame({'left': p[[25, 27, 29, 31]].mean(axis=1),
                         'right': p[[26, 28, 30, 32]].mean(axis=1)})


@dataclass
class BilateralReference:
    mean: np.ndarray
    variance: np.ndarray
    sources: tuple
    floor: float = 1e-8

    @classmethod
    def fit(cls, values, sources, floor=1e-8):
        sources = tuple(sorted(sources))
        if len(sources) < 2 or len(set(sources)) != len(sources) or values.index.has_duplicates or not set(sources).issubset(values.index):
            raise ValueError('Required reserved normal reference sources lack support')
        a = values.loc[list(sources), ['left', 'right']].to_numpy(float)
        if not np.isfinite(a).all():
            raise ValueError('Reference has nonfinite values')
        s2 = np.var(a, axis=0, ddof=1)
        return cls(a.mean(axis=0), .5 * s2 + .5 * s2.mean() + floor, sources, floor)

    def score(self, values):
        a = values[['left', 'right']].to_numpy(float)
        return pd.Series(np.mean((a - self.mean) ** 2 / self.variance, axis=1), index=values.index)

    def audit(self):
        return dict(sources=list(self.sources), mean=self.mean.tolist(), variance=self.variance.tolist(),
                    variance_floor=self.floor, variance_ddof=1, shrinkage=.5)


def endpoint_frame(scores, endpoint):
    if endpoint == 'focal':
        return scores.loc[scores.focal_eligible.astype(bool)].copy()
    if endpoint == 'broad':
        return scores.copy()
    raise ValueError(endpoint)


def collapse_recording_duplicates(table, score_columns):
    """Retain source records, but give confirmed duplicate footage one rank.

    The decision uses footage review before fitting, never score similarity.
    An arithmetic mean combines the duplicate source score rows. Unknown
    repeated people remain an explicit limitation of this recording unit.
    """
    if 'cluster_id' not in table:
        return table.copy()
    rows=[]
    for cluster, group in table.groupby('cluster_id',sort=True):
        for field in ('outer_fold','binary_label','focal_eligible'):
            if group[field].nunique()!=1:
                raise ValueError(f'Confirmed duplicate cluster disagrees on {field}')
        row=group.iloc[0].to_dict()
        for column in score_columns:
            row[column]=group[column].mean() if group[column].notna().all() else np.nan
        row['source_id']='|'.join(sorted(group.source_id.astype(str)))
        row['source_id_count']=len(group);row['cluster_id']=cluster;rows.append(row)
    return pd.DataFrame(rows)


def ranking_metrics(table, column):
    folds = []; missing = 0; unavailable_folds=[]
    for fold, group in table.groupby('outer_fold'):
        missing += int((~np.isfinite(group[column])).sum())
        part = group.loc[np.isfinite(group[column])]
        n0 = int(part.binary_label.eq(0).sum()); n1 = int(part.binary_label.eq(1).sum())
        if not n0 or not n1:
            unavailable_folds.append(int(fold))
            continue
        folds.append(dict(outer_fold=int(fold), normal=n0, abnormal=n1, pairs=n0*n1,
                          prevalence=n1/(n0+n1), auroc=float(roc_auc_score(part.binary_label, part[column])),
                          average_precision=float(average_precision_score(part.binary_label, part[column]))))
    if not folds:
        return dict(status='unavailable', folds=[], missing=missing)
    weights = np.array([f['pairs'] for f in folds], float); weights /= weights.sum()
    complete = table.loc[np.isfinite(table[column])]
    return dict(status='complete' if not missing and not unavailable_folds else 'partial_support',
                folds=folds, weights=weights.tolist(), missing=missing, unavailable_folds=unavailable_folds,
                weighted_auroc=float(sum(w*f['auroc'] for w,f in zip(weights, folds))),
                equal_fold_auroc=float(np.mean([f['auroc'] for f in folds])),
                equal_fold_average_precision=float(np.mean([f['average_precision'] for f in folds])),
                pooled_secondary_auroc=float(roc_auc_score(complete.binary_label, complete[column])),
                n_sources=len(complete), prevalence=float(complete.binary_label.mean()))


def paired_bootstrap(table, first, second, repeats=1000, seed=42):
    """Stratified source/confirmed-duplicate-cluster bootstrap, fixed fold weights.

    The interval conditions on models, source split, and this exploratory
    redesign. It does not capture unknown repeated-person or training effects.
    """
    keep = np.isfinite(table[first]) & np.isfinite(table[second]); table = table.loc[keep].copy()
    if 'cluster_id' not in table:
        table['cluster_id'] = table.source_id
    ma, mb = ranking_metrics(table, first), ranking_metrics(table, second)
    if ma['status'] != 'complete' or mb['status'] != 'complete':
        return dict(status='unavailable', first=first, second=second)
    ids = [f['outer_fold'] for f in ma['folds']]; weights = np.asarray(ma['weights'])
    groups = {}
    for fold in ids:
        g = table.loc[table.outer_fold.eq(fold)]
        for label in (0,1):
            s = g.loc[g.binary_label.eq(label)]
            groups[fold,label] = [part for _,part in s.groupby('cluster_id')]
    rng = np.random.default_rng(seed); effects = []
    for _ in range(repeats):
        delta = []
        for fold in ids:
            sampled = []
            for label in (0,1):
                clusters = groups[fold,label]
                sampled.extend(clusters[k] for k in rng.integers(len(clusters), size=len(clusters)))
            sample = pd.concat(sampled, ignore_index=True)
            delta.append(roc_auc_score(sample.binary_label,sample[first])-roc_auc_score(sample.binary_label,sample[second]))
        effects.append(float(np.dot(weights,delta)))
    return dict(status='complete', first=first, second=second, repeats=int(repeats), seed=int(seed),
                n_common_sources=len(table), observed=ma['weighted_auroc']-mb['weighted_auroc'],
                conditional_95_interval=np.quantile(effects,[.025,.975]).tolist(),
                conditioning='fixed fitted models, split, protocol and original fold weights',
                resampling='whole recording/confirmed-duplicate clusters within fold and binary label',
                fixed_weights=weights.tolist(), folds=ids)
