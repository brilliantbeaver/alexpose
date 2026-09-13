"""Small temporal retention gate and inexpensive restoration comparisons.

The inputs contain measurements only. Reference motion and event support are
used as training labels, never as gate features or test-time bone lengths.
"""
from __future__ import annotations

import numpy as np
from scipy.ndimage import gaussian_filter1d, median_filter
import torch
from torch import nn
from torch.nn import functional as F

from .body_geometry import PARENTS, project_bone_lengths


FLOW_START = 17
FLOW_FIELDS = ("transport_error_px", "forward_backward_error_px", "photometric_error",
               "flow_dispersion_px", "flow_uncertainty_px", "texture_std", "support_fraction")


def make_features(raw, prior, confidence, observed, raw_diagnostics, prior_diagnostics, image_size):
    """[T,J,C] features; append validity separately from zero-filled diagnostics."""
    raw, prior = np.asarray(raw), np.asarray(prior)
    base = [raw - raw[:, :1], prior - prior[:, :1], raw - prior,
            np.gradient(raw, axis=0), np.gradient(prior, axis=0),
            np.asarray(confidence)[..., None], np.asarray(observed)[..., None]]
    for diagnostic in (raw_diagnostics, prior_diagnostics):
        for key in FLOW_FIELDS:
            value = np.asarray(diagnostic[key], dtype=np.float32)
            if key.endswith("_px"):
                value = value / image_size
            padded = np.zeros(raw.shape[:2], dtype=np.float32)
            padded[:-1] = np.nan_to_num(value, nan=0, posinf=0, neginf=0)
            base.append(padded[..., None])
            available = np.zeros(raw.shape[:2], dtype=np.float32)
            available[:-1] = np.isfinite(value) & diagnostic["evidence_valid"]
            base.append(available[..., None])
        valid = np.zeros(raw.shape[:2], dtype=np.float32)
        valid[:-1] = diagnostic["evidence_valid"]
        base.append(valid[..., None])
    return np.concatenate(base, axis=-1).astype(np.float32)


def project_torch(joints, lengths):
    """Differentiable version of the final bone-length projection used at test."""
    result = [joints[..., 0, :]]
    for j in range(1, len(PARENTS)):
        direction = joints[..., j, :] - joints[..., PARENTS[j], :]
        length = torch.linalg.vector_norm(direction, dim=-1, keepdim=True)
        fallback = torch.tensor([0., -1., 0.], device=joints.device, dtype=joints.dtype)
        unit = torch.where(length < 1e-8, fallback, direction / length.clamp_min(1e-8))
        result.append(result[PARENTS[j]] + unit * lengths[:, None, j, None])
    return torch.stack(result, dim=-2)


class RetentionGate(nn.Module):
    """Two small temporal convolutions, joint embeddings and two output heads.

    This is an S-JEPA-compatible temporal adapter, not a claim that a pretrained
    S-JEPA checkpoint was used. Coordinate-only and random-feature arms expose
    whether its capacity or image-motion input explains the improvement.
    """
    def __init__(self, input_dim, hidden_dim=64, random_features=False):
        super().__init__()
        self.embed = nn.Linear(input_dim, hidden_dim)
        self.joint_embed = nn.Embedding(22, hidden_dim)
        self.temporal = nn.Sequential(nn.Conv1d(hidden_dim, hidden_dim, 5, padding=2), nn.GELU(),
                                      nn.Conv1d(hidden_dim, hidden_dim, 5, padding=4, dilation=2), nn.GELU())
        self.norm = nn.LayerNorm(hidden_dim)
        self.retain = nn.Linear(hidden_dim, 1)
        self.evidence = nn.Linear(hidden_dim, 1)
        if random_features:
            for module in (self.embed, self.joint_embed, self.temporal, self.norm):
                for parameter in module.parameters():
                    parameter.requires_grad_(False)

    def forward(self, x):
        b, t, j, _ = x.shape
        h = self.embed(x) + self.joint_embed(torch.arange(j, device=x.device))[None, None]
        temporal = self.temporal(h.permute(0, 2, 3, 1).reshape(b*j, -1, t))
        h = self.norm(h + temporal.reshape(b, j, -1, t).permute(0, 3, 1, 2))
        return torch.sigmoid(self.retain(h).squeeze(-1)), self.evidence(h.mean(dim=(1, 2))).squeeze(-1)


def feature_view(features, mode, permutation=None):
    x = np.array(features, copy=True)
    if mode == "coordinates":
        x[..., FLOW_START:] = 0
    elif mode == "shuffled_flow":
        if permutation is None:
            raise ValueError("Shuffled-flow control requires a fixed cross-case permutation")
        x[..., FLOW_START:] = x[permutation, ..., FLOW_START:]
    elif mode not in {"full", "random_features"}:
        raise ValueError(f"Unknown feature mode {mode}")
    return x


def fit_gate(arrays, *, seed=17, mode="full", epochs=25, batch_size=16,
             hidden_dim=64, learning_rate=1e-3, device="cpu"):
    """Fit only supplied training cases. No selection uses development truth."""
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    permutation = rng.permutation(len(arrays["features"]))
    x = feature_view(arrays["features"], mode, permutation)
    mean, std = x.mean(axis=(0, 1, 2)), x.std(axis=(0, 1, 2))
    std = np.maximum(std, 1e-3)
    x = (x - mean) / std
    model = RetentionGate(x.shape[-1], hidden_dim, mode == "random_features").to(device)
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=learning_rate)
    values = {k: torch.as_tensor(v, dtype=torch.float32, device=device) for k, v in arrays.items()
              if k in {"raw", "prior", "truth", "bone_lengths", "observed", "event_support", "event_present"}}
    xt = torch.as_tensor(x, device=device)
    history = []
    for epoch in range(epochs):
        model.train()
        totals = []
        for start in range(0, len(x), batch_size):
            if start == 0:
                order = rng.permutation(len(x))
            ids = torch.as_tensor(order[start:start+batch_size], device=device)
            retain, logit = model(xt[ids])
            retain = retain * values["observed"][ids]
            mixed = values["prior"][ids] + retain[..., None] * (values["raw"][ids] - values["prior"][ids])
            prediction = project_torch(mixed, values["bone_lengths"][ids])
            # Error measured in units of 3 cm. Extra weight on training event
            # support prevents a few genuine frames being overwhelmed by averages.
            errors = F.smooth_l1_loss(prediction / .03, values["truth"][ids] / .03, reduction="none").mean(-1)
            weight = 1 + 4 * values["event_support"][ids] * values["event_present"][ids, None, None]
            restoration = (errors * weight).sum() / weight.sum()
            evidence = F.binary_cross_entropy_with_logits(logit, values["event_present"][ids])
            loss = restoration + .1 * evidence
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            totals.append((float(loss.detach()), float(restoration.detach()), float(evidence.detach())))
        average = np.mean(totals, axis=0)
        history.append(dict(epoch=epoch+1, loss=average[0], restoration_loss=average[1], evidence_loss=average[2],
                            seed=seed, feature_mode=mode))
    payload = dict(state_dict={k: v.detach().cpu() for k, v in model.state_dict().items()},
                   input_dim=x.shape[-1], hidden_dim=hidden_dim, mode=mode, seed=seed,
                   feature_mean=mean, feature_std=std, training_cases=len(x))
    return payload, history


def infer_gate(payload, features, *, device="cpu"):
    rng = np.random.default_rng(payload["seed"] + 1000)
    x = feature_view(features, payload["mode"], rng.permutation(len(features)))
    x = (x - payload["feature_mean"]) / payload["feature_std"]
    model = RetentionGate(payload["input_dim"], payload["hidden_dim"], payload["mode"] == "random_features").to(device)
    model.load_state_dict(payload["state_dict"])
    model.eval()
    with torch.inference_mode():
        retention, logit = model(torch.as_tensor(x, dtype=torch.float32, device=device))
    return retention.cpu().numpy(), logit.cpu().numpy()


def robust_kalman(raw, confidence, process_variance=.0025):
    """Constant-velocity Kalman filter with clipped measurement innovations."""
    x = np.asarray(raw, dtype=float)
    state, velocity = x[0].copy(), np.zeros_like(x[0])
    variance = np.ones_like(state) * .01
    out = [state.copy()]
    for frame in range(1, len(x)):
        predicted = state + velocity
        variance += process_variance
        measurement = .0004 / np.maximum(confidence[frame, :, None], .05)
        gain = variance / (variance + measurement)
        innovation = np.clip(x[frame] - predicted, -.12, .12)
        updated = predicted + gain * innovation
        velocity = .7 * velocity + .3 * (updated - state)
        state, variance = updated, (1-gain) * variance
        out.append(state.copy())
    return np.asarray(out, dtype=np.float32)


def baseline_candidates(case):
    """Actual implemented baselines. Approximate objectives are labeled as such."""
    raw, prior = case["raw"], case["prior"]
    confidence, observed = case["confidence"], case["observed"]
    raw_error, prior_error = case["raw_transport"], case["prior_transport"]
    reliable = case["flow_valid"].astype(bool)
    ratio = np.where(reliable, np.nan_to_num(prior_error / (raw_error + prior_error + 1e-6), nan=.5), .5)
    clip_ratio = np.full_like(ratio, float(np.mean(ratio)))
    keep = ratio * observed
    result = {
        "raw": raw.copy(), "projected_raw": project_bone_lengths(raw, case["bone_lengths"]),
        "prior": prior.copy(), "conversion_only": case["bridge"].copy(),
        "prior_unprojected": prior.copy(), "conversion_only_unprojected": case["bridge"].copy(),
        "gaussian": gaussian_filter1d(raw, 1.2, axis=0),
        "median": median_filter(raw, size=(5, 1, 1)),
        "kalman": robust_kalman(raw, confidence),
        "confidence_gate": prior + (confidence * observed)[..., None] * (raw-prior),
        "flow_gate": prior + keep[..., None] * (raw-prior),
        "clip_flow_gate": prior + (clip_ratio * observed)[..., None] * (raw-prior),
    }
    # Pixel transport constrains two coordinates only. Preserve each raw point's
    # camera depth; backprojection never uses the reference trajectory.
    if "flow_propagated" in case:
        result["flow_propagation"] = case["flow_propagated"]
    # A bounded derivative-alignment comparator, not a reproduction of PVA-Net
    # or the full HTD-Refine paper. It uses measured raw derivatives explicitly.
    if len(raw) > 3:
        d1 = np.diff(np.eye(len(raw)), axis=0)
        d2 = np.diff(np.eye(len(raw)), n=2, axis=0)
        a = np.eye(len(raw)) + 2*d1.T@d1 + .5*d2.T@d2
        rhs = prior.reshape(len(raw), -1) + 2*d1.T@d1@raw.reshape(len(raw), -1)
        result["derivative_objective"] = np.linalg.solve(a, rhs).reshape(raw.shape)
    for key in result:
        if key not in {"raw", "prior_unprojected", "conversion_only_unprojected"}:
            result[key] = project_bone_lengths(result[key], case["bone_lengths"])
    return result


def mix_at_strength(raw, candidate, strength, lengths, observed=None):
    """Repair strength 0 retains raw; 1 accepts the candidate, then projects."""
    mixed = raw + float(strength) * (candidate-raw)
    if observed is not None:
        mixed = np.where(np.asarray(observed)[..., None], mixed, candidate)
    return project_bone_lengths(mixed, lengths)


def fit_linear_flow_gate(arrays):
    """Strong cheap comparator: ridge-fit flow evidence to training oracle weights.

    This has no temporal neural model. Its probability head sees mean observed
    diagnostics only and is fitted to training event labels.
    """
    from sklearn.linear_model import LogisticRegression, Ridge
    x = arrays["features"][..., FLOW_START:]
    mean, std = x.mean(axis=(0, 1, 2)), np.maximum(x.std(axis=(0, 1, 2)), 1e-3)
    normalized = (x - mean) / std
    delta = arrays["raw"] - arrays["prior"]
    target = np.clip(np.sum((arrays["truth"]-arrays["prior"])*delta, axis=-1) /
                     np.maximum(np.sum(delta**2, axis=-1), 1e-8), 0, 1)
    ridge = Ridge(alpha=100).fit(normalized.reshape(-1, x.shape[-1]), target.reshape(-1))
    classifier = LogisticRegression(C=1, max_iter=500).fit(normalized.mean(axis=(1, 2)), arrays["event_present"])
    return dict(mean=mean, std=std, ridge=ridge, classifier=classifier)


def infer_linear_flow_gate(payload, features):
    x = (features[..., FLOW_START:] - payload["mean"]) / payload["std"]
    weights = np.clip(payload["ridge"].predict(x.reshape(-1, x.shape[-1])), 0, 1).reshape(x.shape[:-1])
    return weights, payload["classifier"].decision_function(x.mean(axis=(1, 2)))
