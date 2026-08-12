"""The classical Random Forest branch, mirroring experiments/exp5.

This is the baseline we compare S-JEPA against. It reuses the exact ambient
pipeline that exp5 uses: cached keypoints turn into joint angles, joint angles
turn into a GaitFeatureVector, and a RandomForest is trained on those features.
The only differences from exp5 are that we read from our own cached .npz files
(no GAVD CSVs) and that we classify just the three conditions normal, ms, and pd.

Keeping this in the package means notebook 06 and the tests call the same code, so
the comparison is honest: both branches see identical train and test splits.
"""

from __future__ import annotations

from typing import List, Sequence, Tuple

import numpy as np

from .data import SequenceRecord

# Imported lazily inside functions: ambient.* and sklearn.


def sequence_to_feature_vector(record: SequenceRecord, fps: int = 15):
    """Turn one cached video into a GaitFeatureVector via joint angles.

    Uses the RAW (pixel) keypoints because joint-angle geometry is scale free and
    the ambient calculator expects pixel-like coordinates with confidence.
    """
    from ambient.pose.keypoint_data import MEDIAPIPE_33_NAMES
    from ambient.pose.joint_angles import get_joint_angles
    from ambient.classification.features import GaitFeatureVector

    arr = record.load_raw()                              # (T, 33, 3)
    frames = [
        [
            {
                "name": MEDIAPIPE_33_NAMES[j],
                "x": float(arr[t, j, 0]),
                "y": float(arr[t, j, 1]),
                "confidence": float(arr[t, j, 2]),
            }
            for j in range(33)
        ]
        for t in range(arr.shape[0])
    ]
    ja = get_joint_angles(frames, keypoint_format="BLAZEPOSE_33", fps=float(fps),
                          sequence_id=record.clip_name)
    return GaitFeatureVector.from_joint_angles(
        ja, sample_id=record.clip_name, condition_label=record.label
    )


def build_feature_matrix(records: Sequence[SequenceRecord], fps: int = 15
                         ) -> Tuple[np.ndarray, List[str], List[str], List[str]]:
    """Return (X, y, source_ids, feature_names) for a set of records."""
    from ambient.classification.features import GaitFeatureVector

    fvs = [sequence_to_feature_vector(r, fps=fps) for r in records]
    X = np.stack([fv.to_array() for fv in fvs], axis=0)
    y = [r.label for r in records]
    source_ids = [r.source_id for r in records]
    feature_names = GaitFeatureVector.get_feature_names()
    return X, y, source_ids, feature_names


def train_rf_and_predict(
    X_train: np.ndarray, y_train: Sequence[str],
    X_test: np.ndarray,
    n_estimators: int = 100, max_depth: int = 5, seed: int = 42,
) -> np.ndarray:
    """Train a StandardScaler + RandomForest and predict test labels.

    Mirrors exp5's RFClassifierConfig defaults (balanced classes, sqrt features).
    We drop all-zero feature columns first so unpopulated fields do not add noise.
    """
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.preprocessing import StandardScaler

    # Keep columns that vary in the training set.
    keep = X_train.std(axis=0) > 0
    Xtr = X_train[:, keep]
    Xte = X_test[:, keep]

    scaler = StandardScaler().fit(Xtr)
    Xtr_s = scaler.transform(Xtr)
    Xte_s = scaler.transform(Xte)

    clf = RandomForestClassifier(
        n_estimators=n_estimators, max_depth=max_depth, max_features="sqrt",
        class_weight="balanced", random_state=seed, n_jobs=-1,
    )
    clf.fit(Xtr_s, list(y_train))
    return clf.predict(Xte_s)
