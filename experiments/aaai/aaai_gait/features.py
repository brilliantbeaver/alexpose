"""Interpretable Random Forest features on exactly the S-JEPA input windows.

The core 15 summaries follow exp5's hip/knee/ankle construction. Extra summaries
are measured from normalized 2D coordinates. They are not clinical stride
length, calibrated gait speed, or diagnoses. No confidence or source metadata
enters these features.
"""
from __future__ import annotations
import numpy as np
from .data import JOINTS


def _nanmean(values, axis):
    values = np.asarray(values, float)
    finite = np.isfinite(values)
    total = np.where(finite, values, 0).sum(axis=axis)
    count = finite.sum(axis=axis)
    return np.divide(total, count, out=np.full_like(total, np.nan, dtype=float), where=count > 0)


def _nanstd(values, axis):
    mean = np.expand_dims(_nanmean(values, axis), axis=axis)
    return np.sqrt(_nanmean(np.square(values - mean), axis))


def _nanrange(values, axis):
    finite = np.isfinite(values)
    minimum = np.where(finite, values, np.inf).min(axis=axis)
    maximum = np.where(finite, values, -np.inf).max(axis=axis)
    return np.where(finite.any(axis=axis), maximum - minimum, np.nan)


def _angle(a, vertex, b):
    u, v = a - vertex, b - vertex
    denominator = np.linalg.norm(u, axis=-1) * np.linalg.norm(v, axis=-1)
    cosine = np.divide(np.sum(u * v, axis=-1), denominator,
                       out=np.full_like(denominator, np.nan), where=denominator > 1e-8)
    return np.degrees(np.arccos(np.clip(cosine, -1, 1)))


def build_features(x: np.ndarray, fps: float = 15) -> tuple[np.ndarray, list[str]]:
    """Return 36 observed angle/motion summaries per [N,T,16,2] window.

    Features share the same frames, joints, normalization and QC as S-JEPA.
    Missing observations remain NaN. Summaries use observed values only;
    adjacent-frame speeds require both observations. Entirely missing features
    remain NaN for the training-only imputer. Audit missingness as a nuisance.
    """
    x = np.asarray(x, float)
    if x.ndim != 4 or x.shape[2:] != (len(JOINTS), 2) or x.shape[1] < 2:
        raise ValueError("Expected [N,T>=2,16,2] in the declared joint order")
    if not np.isfinite(fps) or fps <= 0:
        raise ValueError("fps must be positive")
    index = {joint: j for j, joint in enumerate(JOINTS)}
    triples = [(11,23,25), (23,25,27), (25,27,31), (12,24,26), (24,26,28), (26,28,32)]
    angles = np.stack([_angle(x[:,:,index[a]], x[:,:,index[b]], x[:,:,index[c]]) for a,b,c in triples], axis=-1)
    means = _nanmean(angles, axis=1)
    ranges = _nanrange(angles, axis=1)
    stdev = _nanstd(angles, axis=1)
    angle_names = ["left_hip", "left_knee", "left_ankle", "right_hip", "right_knee", "right_ankle"]
    values = [means, np.abs(means[:, :3] - means[:, 3:]), ranges, stdev]
    names = [n + "_mean_deg" for n in angle_names] + [n + "_mean_asymmetry_deg" for n in ("hip","knee","ankle")]
    names += [n + "_range_deg" for n in angle_names] + [n + "_std_deg" for n in angle_names]
    for joint, name in ((15,"left_wrist"),(16,"right_wrist"),(27,"left_ankle"),(28,"right_ankle")):
        coordinates = x[:,:,index[joint]]
        speed = np.linalg.norm(np.diff(coordinates, axis=1), axis=-1) * fps
        excursion = np.linalg.norm(_nanrange(coordinates, axis=1), axis=-1)
        values.append(np.stack([_nanmean(speed, axis=1), _nanstd(speed, axis=1), excursion], axis=-1))
        names += [name + suffix for suffix in ("_speed_mean_torso_per_s","_speed_std_torso_per_s","_excursion_torso")]
    separation = np.linalg.norm(x[:,:,index[27]] - x[:,:,index[28]], axis=-1)
    torso = (x[:,:,index[11]] + x[:,:,index[12]]) / 2 - (x[:,:,index[23]] + x[:,:,index[24]]) / 2
    # Image y points downward. This is projected torso orientation, not a 3D angle.
    lean = np.arctan2(torso[:,:,0], -torso[:,:,1])
    values.append(np.stack([_nanmean(separation, axis=1), _nanstd(separation, axis=1), _nanstd(lean, axis=1)], axis=-1))
    names += ["ankle_separation_mean_torso", "ankle_separation_std_torso", "torso_orientation_std_rad"]
    matrix = np.concatenate(values, axis=1)
    assert matrix.shape[1] == len(names)
    return matrix.astype(np.float32), names


def build_ambient_features(x: np.ndarray, fps: float = 15) -> tuple[np.ndarray, list[str]]:
    """Optional exact ambient exp5 core-angle adapter, with only 15 enabled fields.

    The broader 82-field vector requires additional analyses. This adapter
    deliberately selects core_angles so uncomputed default zeros never masquerade
    as 82 measured features. Missing excluded landmarks have zero confidence.
    """
    from ambient.pose.joint_angles import get_joint_angles
    from ambient.pose.keypoint_data import MEDIAPIPE_33_NAMES
    from ambient.classification.features import GaitFeatureVector
    x = np.asarray(x)
    if x.ndim != 4 or x.shape[2:] != (16,2):
        raise ValueError("Expected [N,T,16,2]")
    names = GaitFeatureVector.get_feature_names(feature_groups=["core_angles"])
    vectors = []
    for n, window in enumerate(x):
        raw = np.zeros((len(window),33,3), dtype=float)
        raw[:,JOINTS,:2] = window
        raw[:,JOINTS,2] = 1
        frames = [[dict(id=j, name=MEDIAPIPE_33_NAMES[j], x=float(frame[j,0]), y=float(frame[j,1]), confidence=float(frame[j,2])) for j in range(33)] for frame in raw]
        angles = get_joint_angles(frames, keypoint_format="BLAZEPOSE_33", fps=fps, sequence_id=str(n))
        fv = GaitFeatureVector.from_joint_angles(angles, sample_id=str(n))
        vectors.append(np.full(len(names),np.nan) if fv is None else fv.to_array(feature_groups=["core_angles"]))
    return np.asarray(vectors), names


def equal_source_weights(groups) -> np.ndarray:
    """Each source contributes equal total training weight, regardless of windows."""
    groups = np.asarray(groups)
    _, inverse, counts = np.unique(groups, return_inverse=True, return_counts=True)
    weights = 1.0 / counts[inverse]
    return weights / weights.mean()


def fit_rf(x_train, y_train, seed: int = 42, groups=None, **kwargs):
    """Fit imputation and RF on training rows only, without internal ungrouped CV."""
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.impute import SimpleImputer
    from sklearn.pipeline import Pipeline
    parameters = dict(n_estimators=300, max_depth=8, min_samples_leaf=2,
                      max_features="sqrt", class_weight="balanced", random_state=seed, n_jobs=2)
    parameters.update(kwargs)
    estimator = Pipeline([("impute", SimpleImputer(strategy="median", keep_empty_features=True)),
                          ("rf", RandomForestClassifier(**parameters))])
    options = {}
    if groups is not None:
        weights = equal_source_weights(groups)
        labels = np.asarray(y_train)
        if len(labels) != len(weights):
            raise ValueError("One source group is required per training row")
        if parameters["class_weight"] == "balanced":
            # Balance classes after source weighting, rather than using raw
            # window counts that would partly undo source equalization.
            unique = np.unique(labels)
            total = weights.sum()
            for label in unique:
                selected = labels == label
                weights[selected] *= total / (len(unique) * weights[selected].sum())
            estimator.set_params(rf__class_weight=None)
        options["rf__sample_weight"] = weights
    estimator.fit(np.asarray(x_train), np.asarray(y_train), **options)
    return estimator
