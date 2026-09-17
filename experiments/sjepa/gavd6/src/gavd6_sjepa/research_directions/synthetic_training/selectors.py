"""Small outcome-based lesson selectors with explicit information ablations."""

from __future__ import annotations

from dataclasses import dataclass
import numpy as np
from sklearn.linear_model import Ridge
from sklearn.neighbors import KNeighborsRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


# Each list is the complete permitted input. In particular source_progress does
# not contain target_pre, target_delta, or anything that reconstructs them.
FEATURE_VIEWS = {
    "before": ("context", "target_pre", "diagnostic_pre", "descriptors", "budget"),
    "after": ("context", "target_post", "diagnostic_post", "descriptors", "budget"),
    "response": ("context", "target_pre", "target_delta", "diagnostic_pre", "diagnostic_delta", "descriptors", "budget"),
    "source_progress": ("context", "target_post", "diagnostic_pre", "diagnostic_post", "loss_history", "descriptors", "budget"),
    "full": ("context", "target_post", "target_delta", "diagnostic_pre", "diagnostic_post", "loss_history", "descriptors", "budget"),
    "magnitude": ("context", "target_post", "magnitude", "diagnostic_pre", "diagnostic_post", "loss_history", "descriptors", "budget"),
    "weakness": ("simple_context", "diagnostic_post", "descriptors", "budget"),
    "domain": ("simple_context", "budget"),
    "no_context": ("target_post", "target_delta", "diagnostic_pre", "diagnostic_post", "loss_history", "descriptors", "budget"),
    "simple_context": ("simple_context", "target_post", "target_delta", "diagnostic_pre", "diagnostic_post", "loss_history", "descriptors", "budget"),
    "image_context": ("image_context", "target_post", "target_delta", "diagnostic_pre", "diagnostic_post", "loss_history", "descriptors", "budget"),
    "shuffled_context": ("shuffled_context", "target_post", "target_delta", "diagnostic_pre", "diagnostic_post", "loss_history", "descriptors", "budget"),
}


def feature_vector(features: dict, view: str) -> np.ndarray:
    """Concatenate only declared fields; target reference errors are not inputs."""
    if view not in FEATURE_VIEWS:
        raise ValueError(f"Unknown feature view: {view}")
    arrays = [np.asarray(features[key], np.float32).reshape(-1) for key in FEATURE_VIEWS[view]]
    result = np.concatenate(arrays)
    if result.size == 0 or not np.isfinite(result).all():
        raise ValueError(f"Nonfinite or empty features in view {view}.")
    return result


@dataclass
class LessonSelector:
    """Predict signed gains; replay is the known zero-gain outside option."""

    view: str
    kind: str
    parameter: float
    lessons: list[str]
    model: object = None

    def fit(self, features: list[dict], gains: np.ndarray) -> "LessonSelector":
        x = np.stack([feature_vector(f, self.view) for f in features])
        y = np.asarray(gains, float)
        if y.shape != (len(x), len(self.lessons)) or not np.isfinite(y).all():
            raise ValueError("A complete finite gain vector is required for each source episode.")
        if self.kind == "nearest":
            model = KNeighborsRegressor(n_neighbors=min(int(self.parameter), len(x)), weights="uniform")
        elif self.kind == "ridge":
            model = Ridge(alpha=float(self.parameter))
        else:
            raise ValueError("Selector kind must be nearest or ridge.")
        self.model = make_pipeline(StandardScaler(), model).fit(x, y)
        return self

    def predict_gains(self, features: list[dict]) -> np.ndarray:
        if self.model is None:
            raise RuntimeError("Fit the selector on source episodes first.")
        x = np.stack([feature_vector(f, self.view) for f in features])
        return np.asarray(self.model.predict(x)).reshape(len(x), len(self.lessons))

    def select(self, features: list[dict]) -> list[str]:
        gains = self.predict_gains(features)
        # Put replay first: ties at zero prefer not adding synthetic training.
        choices = np.argmax(np.column_stack([np.zeros(len(gains)), gains]), axis=1)
        return [("replay", *self.lessons)[i] for i in choices]


def chosen_errors(actions, errors, lessons):
    """Look up true selected errors for source validation, never deployment."""
    order = ["replay", *lessons]
    errors = np.asarray(errors, dtype=float)
    if errors.shape != (len(actions), len(order)):
        raise ValueError("Errors must have replay followed by every candidate lesson.")
    return np.asarray([errors[i, order.index(action)] for i, action in enumerate(actions)])


def selection_regret(actions, errors, lessons):
    """Regret includes replay, so all harmful lessons may correctly be rejected."""
    return chosen_errors(actions, errors, lessons) - np.asarray(errors).min(axis=1)
