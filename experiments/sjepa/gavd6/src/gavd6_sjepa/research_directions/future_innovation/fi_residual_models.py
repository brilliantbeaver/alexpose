"""Source-weighted ridge and matched temporal residual heads."""

from dataclasses import dataclass

import numpy as np
import torch
from sklearn.linear_model import Ridge
from torch import nn

from .fi_contracts import equal_source_weights
from .fi_vjepa_adapter import deterministic_torch


@dataclass
class TrainingScaler:
    mean: np.ndarray
    scale: np.ndarray
    variance: np.ndarray
    training_window_ids: tuple[str, ...]

    @classmethod
    def fit(cls, values, weights, window_ids):
        values = np.asarray(values, dtype=np.float64)
        if values.ndim != 2 or len(values) != len(window_ids) or np.isinf(values).any():
            raise ValueError("Invalid training scaler input")
        observed = np.isfinite(values)
        weighted = weights[:, None] * observed
        support = weighted.sum(axis=0)
        mean = (weighted * np.nan_to_num(values)).sum(axis=0) / np.maximum(
            support, 1e-12
        )
        imputed = np.where(observed, values, mean)
        variance = (weights[:, None] * (imputed - mean) ** 2).sum(
            axis=0
        ) / weights.sum()
        return cls(
            mean,
            np.maximum(np.sqrt(variance), 1e-8),
            variance,
            tuple(map(str, window_ids)),
        )

    def transform(self, values):
        values = np.asarray(values, dtype=np.float64)
        if np.isinf(values).any():
            raise ValueError("Infinite model input")
        return (
            np.where(np.isfinite(values), values, self.mean) - self.mean
        ) / self.scale

    def inverse(self, values):
        return values * self.scale + self.mean


@dataclass
class FittedBaseline:
    x_scaler: TrainingScaler
    y_scaler: TrainingScaler
    ridge: Ridge
    valid_features: np.ndarray
    training_video_ids: tuple[str, ...]

    def predict(self, baseline_features):
        """Return predictions in this fit's training-standardized target units."""
        return self.ridge.predict(self.x_scaler.transform(baseline_features))


def fit_baseline(x, y, window_ids, video_ids, alpha, variance_tolerance):
    if y.ndim != 2 or not np.isfinite(y).all():
        raise ValueError("Baseline targets must be finite feature vectors")
    weights = equal_source_weights(video_ids)
    x_scaler = TrainingScaler.fit(x, weights, window_ids)
    y_scaler = TrainingScaler.fit(y, weights, window_ids)
    valid = y_scaler.variance > variance_tolerance
    if not valid.any():
        raise ValueError("No nonconstant training target features")
    ridge = Ridge(alpha=alpha, solver="cholesky")
    ridge.fit(x_scaler.transform(x), y_scaler.transform(y), sample_weight=weights)
    return FittedBaseline(
        x_scaler, y_scaler, ridge, valid, tuple(sorted(set(map(str, video_ids))))
    )


class SkeletonResidualHead(nn.Module):
    def __init__(self, input_channels, baseline_dim, width=64, target_dim=256):
        super().__init__()
        self.temporal = nn.Sequential(
            nn.Conv1d(input_channels, width, 3, padding=1),
            nn.GELU(),
            nn.Conv1d(width, width, 3, padding=1),
            nn.GELU(),
        )
        self.output = nn.Linear(width + baseline_dim, target_dim)

    def forward(self, skeleton, baseline_features):
        skeleton = skeleton.flatten(start_dim=2)
        encoded = self.temporal(skeleton.transpose(1, 2)).mean(dim=-1)
        return self.output(torch.cat([encoded, baseline_features], dim=-1))


@torch.inference_mode()
def predict_head(head, skeleton, scaled_x, device):
    head.eval()
    return (
        head(
            torch.as_tensor(skeleton, dtype=torch.float32, device=device),
            torch.as_tensor(scaled_x, dtype=torch.float32, device=device),
        )
        .cpu()
        .numpy()
    )


def train_head(
    skeleton,
    scaled_x,
    residual,
    video_ids,
    valid_features,
    *,
    seed,
    weight_decay,
    updates,
    model_contract,
    device="cpu",
    validation=None,
):
    """Full-batch AdamW; checkpointed validation only chooses from frozen budgets."""
    deterministic_torch(seed)
    head = SkeletonResidualHead(
        132, scaled_x.shape[1], model_contract.width, residual.shape[1]
    ).to(device)
    optimizer = torch.optim.AdamW(
        head.parameters(), lr=model_contract.learning_rate, weight_decay=weight_decay
    )
    sk = torch.as_tensor(skeleton, dtype=torch.float32, device=device)
    x = torch.as_tensor(scaled_x, dtype=torch.float32, device=device)
    target = torch.as_tensor(residual, dtype=torch.float32, device=device)
    weights = torch.as_tensor(
        equal_source_weights(video_ids), dtype=torch.float32, device=device
    )
    valid = torch.as_tensor(valid_features, dtype=torch.bool, device=device)
    if not valid.any():
        raise ValueError("Cannot train on constant targets")
    history = []
    for step in range(1, max(updates) + 1):
        head.train()
        optimizer.zero_grad(set_to_none=True)
        error = (head(sk, x)[:, valid] - target[:, valid]).square().mean(dim=-1)
        loss = (error * weights).sum() / weights.sum()
        if not torch.isfinite(loss):
            raise ValueError("Non-finite residual loss")
        loss.backward()
        nn.utils.clip_grad_norm_(
            head.parameters(), model_contract.gradient_clip, error_if_nonfinite=True
        )
        optimizer.step()
        if step in updates:
            record = {"updates": step, "training_mse": float(loss.detach().cpu())}
            if validation is not None:
                correction = predict_head(
                    head, validation["skeleton"], validation["x"], device
                )
                difference = (
                    correction[:, valid_features]
                    - validation["residual"][:, valid_features]
                )
                error = (difference**2).mean(axis=1)
                record["validation_squared_error"] = float(
                    np.sum(validation["weights"] * error)
                )
                record["validation_weight"] = float(validation["weights"].sum())
            history.append(record)
    head.eval()
    return head, history
