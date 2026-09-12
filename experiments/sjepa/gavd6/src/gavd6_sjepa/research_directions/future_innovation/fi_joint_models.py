"""direct-v3: supported inputs, fixed temporal summaries and two-penalty ridge.

Historical TrainingScaler and checkpoint classes deliberately remain unchanged.
All solves are float64, with source weights summing to the number of fit windows.
"""
from dataclasses import asdict, dataclass

import numpy as np
from scipy.linalg import cho_factor, cho_solve

from .fi_contracts import equal_source_weights

INPUT_VERSION = 'supported-input-v1'
TARGET_VERSION = 'training-target-v1'
FEATURE_VERSION = 'ordered-bins-v1'
MODEL_VERSION = 'joint-ridge-v1'
BINS = (0, 8, 16, 24, 32)
CHANNELS = ('x', 'y', 'vx', 'vy', 'confidence', 'support', 'transition_support')
# (minimum usable standard deviation, scale floor), in each column's units.
POLICIES = {'rgb': (1e-8, 0.01), 'continuous': (1e-8, 0.001),
            'coordinate': (1e-6, 0.01), 'velocity': (1e-6, 0.01),
            'fraction': (1e-6, 0.1)}


@dataclass(frozen=True)
class JointModelContract:
    model_version: str = MODEL_VERSION
    input_version: str = INPUT_VERSION
    target_version: str = TARGET_VERSION
    feature_version: str = FEATURE_VERSION
    ridge_alphas: tuple = (0.1, 1., 10., 100., 1000., 10000.)
    skeleton_alphas: tuple = (0.1, 1., 10., 100., 1000., 10000.)
    seeds: tuple = (0,)
    seed_policy: str = 'deterministic_once; seed=0 is an identity, not a replicate'
    seed_aggregation: str = 'single_deterministic_score'
    inner_folds: int = 3
    bootstrap_repetitions: int = 2000
    target_variance_tolerance: float = 1e-10
    tie_atol: float = 1e-10
    tie_rtol: float = 1e-8
    failure_policy: str = 'reject_failed_candidate; any_required_failure_makes_measurement_incomplete'
    loss: str = 'summed_source_weighted_squared_error; fit_weights_sum_to_n; unpenalized_intercept'

    def __post_init__(self):
        expected = {'model_version': MODEL_VERSION, 'input_version': INPUT_VERSION,
                    'target_version': TARGET_VERSION, 'feature_version': FEATURE_VERSION}
        if any(getattr(self, k) != v for k, v in expected.items()):
            raise ValueError('Unsupported joint model/preprocessing version')
        if tuple(self.seeds) != (0,) or self.inner_folds != 3 or self.bootstrap_repetitions < 1:
            raise ValueError('Invalid deterministic seed/fold/bootstrap contract')
        for grid in (self.ridge_alphas, self.skeleton_alphas):
            if not grid or len(set(grid)) != len(grid) or not np.isfinite(grid).all() or min(grid) <= 0:
                raise ValueError('Penalties must be distinct, finite and positive')
        if self.tie_atol != 1e-10 or self.tie_rtol != 1e-8 or self.target_variance_tolerance != 1e-10:
            raise ValueError('Unsupported selection or target-mask tolerance')
        if self.seed_aggregation != 'single_deterministic_score' or self.failure_policy != JointModelContract.__dataclass_fields__['failure_policy'].default:
            raise ValueError('Unsupported seed/failure policy')


def validate_fit(values, weights, ids):
    v = np.asarray(values, dtype=np.float64)
    w = np.asarray(weights, dtype=np.float64)
    ids = tuple(map(str, ids))
    if (v.ndim != 2 or not len(v) or not v.shape[1] or len(ids) != len(v)
            or len(set(ids)) != len(ids) or w.shape != (len(v),)
            or not np.isfinite(w).all() or np.any(w <= 0) or np.isinf(v).any()):
        raise ValueError('Invalid fit arrays, identities or source weights')
    return v, w, ids


@dataclass
class SupportedInput:
    feature_names: tuple
    feature_kinds: tuple
    training_window_ids: tuple
    training_video_ids: tuple
    weights: np.ndarray
    support_counts: np.ndarray
    support_sources: np.ndarray
    support_weights: np.ndarray
    mean: np.ndarray
    variance: np.ndarray
    scale: np.ndarray
    mask: np.ndarray
    version: str = INPUT_VERSION

    @classmethod
    def fit(cls, values, weights, window_ids, video_ids, names, kinds):
        v, w, ids = validate_fit(values, weights, window_ids)
        vids = tuple(map(str, video_ids))
        if len(vids) != len(v) or len(names) != v.shape[1] or len(set(names)) != len(names) or len(kinds) != len(names) or any(k not in POLICIES for k in kinds):
            raise ValueError('Invalid feature schema or source identities')
        observed = np.isfinite(v)
        weighted = w[:, None] * observed
        support = weighted.sum(axis=0)
        mean = np.divide((weighted * np.nan_to_num(v)).sum(axis=0), support,
                         out=np.zeros(v.shape[1]), where=support > 0)
        variance = (w[:, None] * (np.where(observed, v, mean) - mean)**2).sum(axis=0) / w.sum()
        source_count = np.sum([observed[np.array(vids) == source].any(axis=0) for source in sorted(set(vids))], axis=0)
        minimum, floors = np.array([POLICIES[k] for k in kinds]).T
        mask = (observed.sum(axis=0) >= 2) & (source_count >= 2) & (np.sqrt(variance) > minimum)
        result = cls(tuple(names), tuple(kinds), ids, vids, w.copy(), observed.sum(axis=0),
                     source_count, support, mean, variance, np.maximum(np.sqrt(variance), floors), mask)
        result.validate()
        return result

    def validate(self, names=None):
        p = len(self.feature_names)
        if self.version != INPUT_VERSION or len(set(self.feature_names)) != p or len(self.feature_kinds) != p or any(k not in POLICIES for k in self.feature_kinds):
            raise ValueError('Input preprocessing version/schema mismatch')
        if names is not None and tuple(names) != self.feature_names:
            raise ValueError('Feature order/schema mismatch')
        n = len(self.training_window_ids)
        if (len(set(self.training_window_ids)) != n or len(self.training_video_ids) != n or self.weights.shape != (n,)
                or not np.isfinite(self.weights).all() or np.any(self.weights <= 0)):
            raise ValueError('Invalid saved preprocessing fit identities/weights')
        np.testing.assert_allclose(self.weights, equal_source_weights(self.training_video_ids), rtol=1e-12, atol=1e-12)
        for a in (self.support_counts, self.support_sources, self.support_weights, self.mean, self.variance, self.scale, self.mask):
            if np.shape(a) != (p,) or not np.isfinite(a).all():
                raise ValueError('Invalid saved input statistic shape/value')
        minimum, floors = np.array([POLICIES[k] for k in self.feature_kinds]).T
        if (np.any(self.variance < 0) or self.mask.dtype != bool
                or np.any(self.support_counts < 0) or np.any(self.support_counts > n)
                or np.any(self.support_sources > self.support_counts) or np.any(self.support_sources < 0)
                or np.any(self.support_weights < 0) or np.any(self.support_weights > self.weights.sum() + 1e-10)
                or not np.array_equal(self.mask, (self.support_counts >= 2) & (self.support_sources >= 2) & (np.sqrt(self.variance) > minimum))
                or not np.array_equal(self.scale, np.maximum(np.sqrt(self.variance), floors))):
            raise ValueError('Invalid saved support, masks or scales')
        return self

    def transform(self, values, names=None):
        self.validate(names)
        v = np.asarray(values, dtype=np.float64)
        if v.ndim != 2 or v.shape[1] != len(self.feature_names) or np.isinf(v).any():
            raise ValueError('Invalid input shape or infinite value')
        # Unsupported columns have exactly zero model influence in every partition.
        result = (np.where(np.isfinite(v), v, self.mean) - self.mean) / self.scale
        result[:, ~self.mask] = 0
        if not np.isfinite(result).all():
            raise ValueError('Non-finite transformed input')
        return result

    def diagnostics(self, values):
        v = np.asarray(values, dtype=np.float64)
        z = self.transform(v)
        changed = np.isfinite(v) & (v != self.mean) & ~self.mask
        return {'nominal_features': len(self.mask), 'supported_features': int(self.mask.sum()),
                'unsupported_observations': int(changed.sum()),
                'unsupported_columns_observed': [self.feature_names[i] for i in np.flatnonzero(changed.any(axis=0))],
                'transformed_min': float(z.min()), 'transformed_max': float(z.max()),
                'transformed_max_abs': float(np.max(np.abs(z)))}

    def record(self):
        return {k: v.tolist() if isinstance(v, np.ndarray) else v for k, v in asdict(self).items()}


@dataclass
class TargetStandardizer:
    """Training-derived target units; never apply an input support mask to Y."""
    mean: np.ndarray
    scale: np.ndarray
    variance: np.ndarray
    training_window_ids: tuple
    version: str = TARGET_VERSION

    @classmethod
    def fit(cls, values, weights, window_ids):
        v, w, ids = validate_fit(values, weights, window_ids)
        if not np.isfinite(v).all():
            raise ValueError('Targets must be finite')
        mean = (w[:, None] * v).sum(axis=0) / w.sum()
        variance = (w[:, None] * (v - mean)**2).sum(axis=0) / w.sum()
        return cls(mean, np.maximum(np.sqrt(variance), 1e-8), variance, ids)

    def transform(self, values):
        v = np.asarray(values, dtype=np.float64)
        if (self.version != TARGET_VERSION or v.ndim != 2 or v.shape[1:] != self.mean.shape
                or not all(np.isfinite(a).all() for a in (v, self.mean, self.scale, self.variance))
                or self.scale.shape != self.mean.shape or self.variance.shape != self.mean.shape
                or np.any(self.variance < 0) or not np.array_equal(self.scale, np.maximum(np.sqrt(self.variance), 1e-8))):
            raise ValueError('Invalid target standardizer or values')
        return (v - self.mean) / self.scale

    def inverse(self, values):
        return np.asarray(values) * self.scale + self.mean


def skeleton_schema():
    names = tuple(f'bin{a:02d}_{b:02d}/joint{j:02d}/{c}' for a, b in zip(BINS[:-1], BINS[1:]) for j in range(33) for c in CHANNELS)
    kinds = tuple('coordinate' if c in ('x', 'y') else 'velocity' if c in ('vx', 'vy') else 'fraction'
                  for _ in range(4 * 33) for c in CHANNELS)
    return names, kinds


def temporal_features(history):
    h = np.asarray(history, dtype=np.float64)
    if (h.ndim != 4 or h.shape[1:] != (32, 33, 4) or not np.isfinite(h).all()
            or not np.isin(h[..., 3], (0, 1)).all() or np.any((h[..., 2] < 0) | (h[..., 2] > 1))):
        raise ValueError('Invalid raw skeleton history')
    bins = []
    for a, b in zip(BINS[:-1], BINS[1:]):
        part = h[:, a:b]
        valid = part[..., 3].astype(bool)
        adjacent = valid[:, 1:] & valid[:, :-1]
        def mean(values, support):
            numerator = np.where(support[..., None], values, 0).sum(axis=1)
            denom = support.sum(axis=1)[..., None]
            return np.divide(numerator, denom, out=np.full_like(numerator, np.nan), where=denom > 0)
        xy = mean(part[..., :2], valid)
        velocity = mean(np.diff(part[..., :2], axis=1), adjacent)
        conf = mean(part[..., 2:3], valid)
        bins.append(np.concatenate((xy, velocity, conf, valid.mean(axis=1)[..., None], adjacent.mean(axis=1)[..., None]), axis=-1).reshape(len(h), -1))
    return np.concatenate(bins, axis=1)


def baseline_schema(schema):
    names = tuple([f'context_embedding_{i}' for i in range(schema['context_embedding_columns'])] + schema['columns'])
    def kind(name):
        if name.startswith('context_embedding_'): return 'rgb'
        if any(word in name for word in ('confidence', 'missingness', 'fraction', 'view_')): return 'fraction'
        return 'continuous'
    return names, tuple(map(kind, names))


@dataclass
class JointRidge:
    x_weight: np.ndarray
    s_weight: np.ndarray
    intercept: np.ndarray
    lambda_x: float
    lambda_s: float | None
    version: str = MODEL_VERSION

    @classmethod
    def fit(cls, x, s, y, weights, lambda_x, lambda_s=None):
        x, y, w = np.asarray(x, dtype=np.float64), np.asarray(y, dtype=np.float64), np.asarray(weights, dtype=np.float64)
        s = np.empty((len(x), 0)) if lambda_s is None else np.asarray(s, dtype=np.float64)
        if (x.ndim != 2 or s.ndim != 2 or y.ndim != 2 or len(s) != len(x) or len(y) != len(x)
                or w.shape != (len(x),) or not all(np.isfinite(v).all() for v in (x, s, y, w))
                or np.any(w <= 0) or not np.isclose(w.sum(), len(w), atol=1e-10)
                or not np.isfinite(lambda_x) or lambda_x <= 0
                or (lambda_s is not None and (not np.isfinite(lambda_s) or lambda_s <= 0))):
            raise ValueError('Invalid joint ridge inputs, weight normalization or penalties')
        xm, sm, ym = (np.average(v, axis=0, weights=w) for v in (x, s, y))
        sqrtw = np.sqrt(w)[:, None]
        xc, sc, yc = (sqrtw * (v - m) for v, m in ((x, xm), (s, sm), (y, ym)))
        kernel = xc @ xc.T / lambda_x + np.eye(len(x))
        if lambda_s is not None: kernel += sc @ sc.T / lambda_s
        dual = cho_solve(cho_factor(kernel, lower=True, check_finite=True), yc)
        wx = xc.T @ dual / lambda_x
        ws = sc.T @ dual / lambda_s if lambda_s is not None else np.zeros((0, y.shape[1]))
        result = cls(wx, ws, ym - xm @ wx - sm @ ws, float(lambda_x), None if lambda_s is None else float(lambda_s))
        result.predict(x, s)
        return result

    def predict(self, x, s=None):
        if self.version != MODEL_VERSION or not all(np.isfinite(a).all() for a in (self.x_weight, self.s_weight, self.intercept)):
            raise ValueError('Invalid joint checkpoint')
        prediction = np.asarray(x) @ self.x_weight + self.intercept
        if self.lambda_s is not None: prediction += np.asarray(s) @ self.s_weight
        if not np.isfinite(prediction).all(): raise ValueError('Non-finite joint prediction')
        return prediction


@dataclass
class JointBaseline:
    x_scaler: SupportedInput
    y_scaler: TargetStandardizer
    ridge: JointRidge
    valid_features: np.ndarray
    training_video_ids: tuple

    def predict(self, x):
        return self.ridge.predict(self.x_scaler.transform(x))


def fit_joint_baseline(x, y, ids, videos, alpha, schema):
    weights = equal_source_weights(videos)
    xs = SupportedInput.fit(x, weights, ids, videos, *schema)
    ys = TargetStandardizer.fit(y, weights, ids)
    valid = ys.variance > 1e-10
    if not valid.any(): raise ValueError('No nonconstant target features')
    model = JointRidge.fit(xs.transform(x), None, ys.transform(y), weights, alpha)
    return JointBaseline(xs, ys, model, valid, tuple(sorted(set(map(str, videos)))))
