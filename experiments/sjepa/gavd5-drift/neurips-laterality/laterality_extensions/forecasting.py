"""Past-only pose forecasting for tutorial 10, separate from registered results.

Coordinates are normalized image-space estimates. The prediction boundary is
an information boundary, not a claim about causal effects or physical units.
"""
from __future__ import annotations

import copy
from dataclasses import asdict, dataclass
from typing import Sequence

import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from torch import nn

from laterality.geometry import FULL_MIRROR_PAIRS, observed_mask
from laterality.metrics import source_weights


JOINTS = (11, 12, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32)
LOCAL_SWAP = np.array([JOINTS.index(dict(FULL_MIRROR_PAIRS + tuple((b, a) for a, b in FULL_MIRROR_PAIRS))[j]) for j in JOINTS])


@dataclass(frozen=True)
class ForecastSpec:
    context_seconds: float = 0.8
    horizons: tuple[float, ...] = (0.25, 0.50, 0.75)
    future_window_seconds: float = 0.20
    samples: int = 8
    max_past_age: float = 0.20
    target_time_tolerance: float = 0.055
    visibility_threshold: float = 0.45
    embed_dim: int = 24
    updates: int = 40
    batch_size: int = 8
    learning_rate: float = 0.001
    ema: float = 0.99
    variance_weight: float = 0.1
    ridge_alpha: float = 10.0
    seed: int = 17

    def __post_init__(self):
        continuous = (self.context_seconds, *self.horizons, self.future_window_seconds,
                      self.max_past_age, self.target_time_tolerance, self.visibility_threshold,
                      self.learning_rate, self.ema, self.variance_weight, self.ridge_alpha)
        if not np.isfinite(continuous).all():
            raise ValueError("Forecast settings must be finite")
        if not 0 <= self.ema < 1 or self.variance_weight < 0:
            raise ValueError("EMA must be in [0,1) and variance weight nonnegative")
        if self.learning_rate <= 0 or self.ridge_alpha <= 0 or not 0 <= self.visibility_threshold <= 1:
            raise ValueError("Invalid learning rate, ridge penalty, or visibility threshold")
        if self.context_seconds <= 0 or self.samples < 3:
            raise ValueError("Need positive context duration and at least three samples")
        if not self.horizons or any(h <= self.future_window_seconds for h in self.horizons):
            raise ValueError("Every target window must lie strictly after the boundary")
        if tuple(sorted(set(self.horizons))) != self.horizons:
            raise ValueError("Horizons must be unique and increasing")
        if self.future_window_seconds <= 0 or self.max_past_age <= 0 or self.target_time_tolerance <= 0:
            raise ValueError("Durations and tolerance must be positive")
        if self.embed_dim < 4 or self.embed_dim % 2 or self.batch_size < 2 or self.updates < 1:
            raise ValueError("Invalid small-model configuration")


@dataclass(frozen=True)
class TimedPose:
    source: str
    sequence_id: str
    times: np.ndarray
    xyz: np.ndarray
    valid: np.ndarray


@dataclass
class ForecastExamples:
    past: np.ndarray
    past_valid: np.ndarray
    past_times: np.ndarray
    future: np.ndarray
    future_valid: np.ndarray
    future_times: np.ndarray
    horizon: np.ndarray
    sources: np.ndarray
    sequence_ids: np.ndarray
    status: str
    coverage: dict


def validate_record(record: TimedPose) -> None:
    t = np.asarray(record.times)
    if t.ndim != 1 or len(t) < 3 or not np.isfinite(t).all() or np.any(np.diff(t) <= 0):
        raise ValueError("Timestamps must be finite and strictly increasing")
    if record.xyz.shape != (len(t), 33, 3) or record.valid.shape != (len(t), 33):
        raise ValueError("Expected timestamped [T,33,3] coordinates and [T,33] validity")


def prefix_input(record: TimedPose, spec: ForecastSpec) -> dict:
    """Prepare the context without reading coordinate/validity values after cut."""
    validate_record(record)
    times = record.times - record.times[0]
    observed_prefix = times <= spec.context_seconds + 1e-10
    prefix_t = times[observed_prefix]
    xyz = record.xyz[observed_prefix].copy()
    valid = record.valid[observed_prefix].copy()
    if not np.isfinite(xyz[valid]).all():
        raise ValueError("Observed prefix coordinates must be finite")
    if len(prefix_t) < 3 or prefix_t[-1] < spec.context_seconds - spec.max_past_age:
        raise ValueError("Insufficient observed prefix duration")
    pelvis_ok = valid[:, 23] & valid[:, 24]
    if not pelvis_ok.any():
        raise ValueError("No observed bilateral pelvis in the prefix")
    origin = ((xyz[pelvis_ok, 23] + xyz[pelvis_ok, 24]) / 2)[-1]
    widths = []
    for left, right in ((11, 12), (23, 24)):
        ok = valid[:, left] & valid[:, right]
        widths.extend(np.linalg.norm(xyz[ok, left, :2] - xyz[ok, right, :2], axis=-1))
    scale = float(np.median(widths)) if widths else float("nan")
    if not np.isfinite(scale) or scale <= 1e-6:
        raise ValueError("No usable observed image-plane body width in the prefix")
    grid = np.linspace(0, spec.context_seconds, spec.samples)
    out = np.zeros((spec.samples, len(JOINTS), 3), dtype=np.float32)
    mask = np.zeros(out.shape[:-1], dtype=bool)
    sampled_time = np.zeros_like(mask, dtype=np.float32)
    for sample, query in enumerate(grid):
        for col, joint in enumerate(JOINTS):
            available = np.flatnonzero(valid[:, joint] & (prefix_t <= query + 1e-10))
            if len(available) and query - prefix_t[available[-1]] <= spec.max_past_age + 1e-10:
                index = available[-1]
                out[sample, col] = (xyz[index, joint] - origin) / scale
                mask[sample, col] = True
                sampled_time[sample, col] = prefix_t[index] - spec.context_seconds
    if mask.mean() < 0.5:
        raise ValueError("Less than half of requested prefix landmarks are available")
    return {"xyz": out, "valid": mask, "times": sampled_time, "origin": origin, "scale": scale}


def prepare_examples(records: Sequence[TimedPose], spec: ForecastSpec, *, status: str) -> ForecastExamples:
    """One fixed boundary per clip; no future interpolation or target imputation."""
    values = {key: [] for key in ("past", "past_valid", "past_times", "future", "future_valid", "future_times", "horizon", "sources", "sequence_ids")}
    rejected: dict[str, int] = {}
    admitted_clips = set()
    for record in records:
        validate_record(record)
        try:
            prefix = prefix_input(record, spec)
        except ValueError as error:
            rejected[str(error)] = rejected.get(str(error), 0) + 1
            continue
        relative = record.times - record.times[0]
        for horizon in spec.horizons:
            endpoint = spec.context_seconds + horizon
            if relative[-1] < endpoint - spec.target_time_tolerance:
                rejected["future window unavailable"] = rejected.get("future window unavailable", 0) + 1
                continue
            queries = np.linspace(endpoint - spec.future_window_seconds, endpoint, spec.samples)
            future = np.zeros_like(prefix["xyz"])
            valid = np.zeros_like(prefix["valid"])
            times = np.zeros_like(prefix["times"])
            for row, query in enumerate(queries):
                index = int(np.argmin(np.abs(relative - query)))
                if relative[index] <= spec.context_seconds or abs(relative[index] - query) > spec.target_time_tolerance:
                    continue
                observed = record.valid[index, list(JOINTS)] & np.isfinite(record.xyz[index, list(JOINTS)]).all(-1)
                future[row, observed] = ((record.xyz[index, list(JOINTS)] - prefix["origin"]) / prefix["scale"])[observed]
                valid[row] = observed
                times[row, observed] = relative[index] - spec.context_seconds
            if valid.sum() < 4 or valid[-1].sum() < 2:
                rejected["future observation coverage"] = rejected.get("future observation coverage", 0) + 1
                continue
            for key, value in (("past", prefix["xyz"]), ("past_valid", prefix["valid"]), ("past_times", prefix["times"]), ("future", future), ("future_valid", valid), ("future_times", times), ("horizon", horizon), ("sources", record.source), ("sequence_ids", record.sequence_id)):
                values[key].append(value)
            admitted_clips.add(record.sequence_id)
    if not values["past"]:
        raise ValueError("No forecasting examples; inspect timing and observation coverage")
    arrays = {key: np.asarray(value) for key, value in values.items()}
    return ForecastExamples(**arrays, status=status, coverage={
        "input_clips": len(records), "accepted_clips": len(admitted_clips),
        "examples": len(arrays["past"]), "sources": len(set(arrays["sources"])),
        "reasons": rejected,
        "selection_note": "Some counts concern rejected horizons, not whole clips; future coverage affects eligibility but never predictor inputs.",
    })


def synthetic_records(seed: int = 17, sources: int = 12, clips_per_source: int = 2) -> list[TimedPose]:
    """Toy oscillators with a true clock; no medical or biomechanical simulation."""
    rng = np.random.default_rng(seed)
    records = []
    for source in range(sources):
        for clip in range(clips_per_source):
            fps = (15, 20, 25)[source % 3]
            t = np.arange(0, 2.2, 1 / fps)
            xyz = np.zeros((len(t), 33, 3))
            xyz[:, :, 0] = 0.5
            xyz[:, :, 1] = np.linspace(0.1, 0.9, 33)
            phase = rng.uniform(-np.pi, np.pi)
            frequency = rng.uniform(0.7, 1.4)
            asymmetry = rng.uniform(-0.35, 0.35)
            drift = rng.uniform(-0.025, 0.025) * t
            for left, right in FULL_MIRROR_PAIRS:
                xyz[:, left, 0] -= 0.10
                xyz[:, right, 0] += 0.10
            for pair_index, (left, right) in enumerate(((25, 26), (27, 28), (29, 30), (31, 32))):
                amplitude = 0.025 + pair_index * 0.007
                xyz[:, left, 0] += amplitude * (1 + asymmetry) * np.sin(2 * np.pi * frequency * t + phase)
                xyz[:, right, 0] += amplitude * (1 - asymmetry) * np.sin(2 * np.pi * frequency * t + phase + np.pi)
                xyz[:, [left, right], 2] = 0.02 * np.cos(2 * np.pi * frequency * t[:, None] + phase)
            xyz[:, :, 0] += drift[:, None]
            xyz += rng.normal(0, 0.0005, xyz.shape)
            valid = np.ones(xyz.shape[:-1], bool)
            records.append(TimedPose(f"synthetic_{source:02d}", f"toy_{source:02d}_{clip}", t, xyz.astype(np.float32), valid))
    return records


def perturb_future(record: TimedPose, spec: ForecastSpec) -> TimedPose:
    """Adversarially replace future coordinates AND visibility for leakage tests."""
    xyz, valid = record.xyz.copy(), record.valid.copy()
    later = record.times - record.times[0] > spec.context_seconds + 1e-10
    xyz[later] = 1e6
    valid[later] = False
    return TimedPose(record.source, record.sequence_id, record.times.copy(), xyz, valid)


class TimePoseEncoder(nn.Module):
    def __init__(self, dimension: int):
        super().__init__()
        self.input = nn.Linear(5, dimension)  # xyz, validity, actual observation time
        self.joint = nn.Parameter(torch.randn(len(JOINTS), dimension) * 0.02)
        block = nn.TransformerEncoderLayer(dimension, 2, dimension * 2, dropout=0.0, batch_first=True, norm_first=True)
        self.attention = nn.TransformerEncoder(block, 1, enable_nested_tensor=False)
        self.norm = nn.LayerNorm(dimension)

    def forward(self, xyz, valid, times):
        if not valid.flatten(1).any(1).all():
            raise ValueError("An encoder window must contain observed positions")
        xyz = torch.where(valid[..., None], xyz, torch.zeros_like(xyz))
        times = torch.where(valid, times, torch.zeros_like(times))
        tokens = self.input(torch.cat((xyz, valid[..., None].float(), times[..., None]), -1))
        tokens = tokens + self.joint[None, None]
        flat_valid = valid.flatten(1)
        features = self.norm(self.attention(tokens.flatten(1, 2), src_key_padding_mask=~flat_valid))
        return (features * flat_valid[..., None]).sum(1) / flat_valid.sum(1, keepdim=True)


class FutureJEPA(nn.Module):
    def __init__(self, dimension: int):
        super().__init__()
        self.encoder = TimePoseEncoder(dimension)
        self.teacher = copy.deepcopy(self.encoder).requires_grad_(False)
        self.predictor = nn.Sequential(nn.Linear(dimension + 1, dimension * 2), nn.GELU(), nn.Linear(dimension * 2, dimension))

    def predict(self, past, valid, times, horizon):
        encoded = self.encoder(past, valid, times)
        return self.predictor(torch.cat((encoded, horizon[:, None]), -1))


def as_tensors(data: ForecastExamples) -> dict[str, torch.Tensor]:
    return {name: torch.as_tensor(getattr(data, name), dtype=torch.bool if name.endswith("valid") else torch.float32)
             for name in ("past", "past_valid", "past_times", "future", "future_valid", "future_times", "horizon")}


def shuffled_target_indices(indices: np.ndarray, train: np.ndarray, sources: np.ndarray, horizons: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Change temporal pairing while retaining conditional source-uniform draws."""
    output = indices.copy()
    for row, original in enumerate(indices):
        candidates = train[(sources[train] != sources[original]) & (horizons[train] == horizons[original])]
        if not len(candidates):
            raise ValueError("No same-horizon, different-source training control")
        source = rng.choice(np.unique(sources[candidates]))
        output[row] = rng.choice(candidates[sources[candidates] == source])
    return output


def fit_future_model(data: ForecastExamples, train: np.ndarray, spec: ForecastSpec, *, shuffled_future: bool = False) -> tuple[FutureJEPA, TimePoseEncoder, list[dict]]:
    torch.manual_seed(spec.seed)
    model = FutureJEPA(spec.embed_dim)
    initial_encoder = copy.deepcopy(model.encoder).eval()
    optimizer = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad), lr=spec.learning_rate)
    tensor = as_tensors(data)
    rng = np.random.default_rng(spec.seed)
    pairing_rng = np.random.default_rng(spec.seed + 1)
    training_sources = np.unique(data.sources[train])
    if len(training_sources) < 2:
        raise ValueError("Need at least two training sources")
    by_source = {s: train[data.sources[train] == s] for s in training_sources}
    history = []
    for step in range(spec.updates):
        source_draw = rng.choice(training_sources, spec.batch_size)
        indices = np.array([rng.choice(by_source[s]) for s in source_draw])
        target_indices = indices.copy()
        if shuffled_future:
            target_indices = shuffled_target_indices(indices, train, data.sources, data.horizon, pairing_rng)
        prediction = model.predict(tensor["past"][indices], tensor["past_valid"][indices], tensor["past_times"][indices], tensor["horizon"][indices])
        context = model.encoder(tensor["past"][indices], tensor["past_valid"][indices], tensor["past_times"][indices])
        with torch.no_grad():
            target = model.teacher(tensor["future"][target_indices], tensor["future_valid"][target_indices], tensor["future_times"][target_indices])
        prediction_loss = (prediction - target).square().mean()
        variance_loss = torch.relu(1 - torch.sqrt(context.var(0, unbiased=False) + 1e-4)).mean()
        loss = prediction_loss + spec.variance_weight * variance_loss
        if not torch.isfinite(loss):
            raise FloatingPointError("Nonfinite forecasting loss")
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        with torch.no_grad():
            for teacher, online in zip(model.teacher.parameters(), model.encoder.parameters()):
                teacher.mul_(spec.ema).add_(online, alpha=1 - spec.ema)
        history.append({"update": step + 1, "prediction_loss": float(prediction_loss.detach()), "variance_penalty": float(variance_loss.detach())})
    return model.eval(), initial_encoder, history


def encode_features(encoder: TimePoseEncoder, data: ForecastExamples, *, reflected=False) -> np.ndarray:
    tensor = as_tensors(data)
    xyz, valid, times = tensor["past"], tensor["past_valid"], tensor["past_times"]
    if reflected:
        xyz = xyz[:, :, LOCAL_SWAP].clone()
        xyz[..., 0] *= -1
        valid, times = valid[:, :, LOCAL_SWAP], times[:, :, LOCAL_SWAP]
    chunks = []
    with torch.no_grad():
        for start in range(0, len(xyz), 64):
            chunks.append(encoder(xyz[start:start + 64], valid[start:start + 64], times[start:start + 64]).numpy())
    return np.concatenate(chunks)


def motion_baselines(data: ForecastExamples) -> tuple[np.ndarray, np.ndarray]:
    persistent = np.full((len(data.past), len(JOINTS), 3), np.nan)
    velocity = persistent.copy()
    for i in range(len(data.past)):
        for joint in range(len(JOINTS)):
            rows = np.flatnonzero(data.past_valid[i, :, joint])
            if not len(rows):
                continue
            last = rows[-1]
            persistent[i, joint] = data.past[i, last, joint]
            velocity[i, joint] = persistent[i, joint]
            earlier = rows[data.past_times[i, rows, joint] < data.past_times[i, last, joint] - 1e-6]
            if len(earlier):
                previous = earlier[-1]
                dt = data.past_times[i, last, joint] - data.past_times[i, previous, joint]
                speed = (data.past[i, last, joint] - data.past[i, previous, joint]) / dt
                velocity[i, joint] += speed * (data.horizon[i] - data.past_times[i, last, joint])
    return persistent, velocity


def ridge_forecasts(features: np.ndarray, data: ForecastExamples, train: np.ndarray, spec: ForecastSpec) -> np.ndarray:
    augmented = np.column_stack((features, data.horizon, features * data.horizon[:, None]))
    scaler = StandardScaler().fit(augmented[train], sample_weight=source_weights(data.sources[train]))
    scaled = scaler.transform(augmented)
    output = np.full((len(features), len(JOINTS), 3), np.nan)
    for joint in range(len(JOINTS)):
        observed = train[data.future_valid[train, -1, joint]]
        if len(np.unique(data.sources[observed])) < 2:
            continue
        fitted = Ridge(alpha=spec.ridge_alpha).fit(scaled[observed], data.future[observed, -1, joint], sample_weight=source_weights(data.sources[observed]))
        output[:, joint] = fitted.predict(scaled)
    return output


def score_forecasts(predictions: dict[str, np.ndarray], data: ForecastExamples, test: np.ndarray) -> pd.DataFrame:
    """Common observable endpoints, then equal total weight per test source."""
    eligible = data.future_valid[:, -1].copy()
    for pred in predictions.values():
        eligible &= np.isfinite(pred).all(-1)
    rows = []
    for horizon in np.unique(data.horizon[test]):
        selected = test[data.horizon[test] == horizon]
        selected = selected[eligible[selected].any(1)]
        if not len(selected):
            raise ValueError("No common evaluable endpoints at a requested horizon")
        for name, predicted in predictions.items():
            errors = np.square(predicted[selected] - data.future[selected, -1]).mean(-1)
            clip_mse = np.array([values[mask].mean() for values, mask in zip(errors, eligible[selected])])
            rmse = np.sqrt(np.average(clip_mse, weights=source_weights(data.sources[selected])))
            rows.append({"method": name, "horizon_seconds": float(horizon), "source_balanced_rmse": float(rmse),
                         "test_sources": len(set(data.sources[selected])), "test_examples": len(selected),
                         "observed_joint_endpoints": int(eligible[selected].sum()), "status": data.status})
    return pd.DataFrame(rows)


def run_forecasting_study(data: ForecastExamples, spec: ForecastSpec, *, train_sources: Sequence[str], test_sources: Sequence[str]) -> dict:
    allowed, held_out = set(train_sources), set(test_sources)
    if allowed & held_out:
        raise ValueError("Training and testing source sets overlap")
    if not set(data.sources) <= allowed | held_out:
        raise ValueError("Every example needs an explicit source role")
    train = np.flatnonzero(np.isin(data.sources, list(allowed)))
    test = np.flatnonzero(np.isin(data.sources, list(held_out)))
    if len(set(data.sources[train])) < 2 or len(set(data.sources[test])) < 2:
        raise ValueError("Need at least two available sources in each role")
    model, initial, history = fit_future_model(data, train, spec)
    shuffled, _, shuffled_history = fit_future_model(data, train, spec, shuffled_future=True)
    persistence, velocity = motion_baselines(data)
    raw = np.concatenate((data.past.reshape(len(data.past), -1), data.past_valid.reshape(len(data.past), -1), data.past_times.reshape(len(data.past), -1)), axis=1)
    learned = encode_features(model.encoder, data)
    initial_features = encode_features(initial, data)
    shuffled_features = encode_features(shuffled.encoder, data)
    predictions = {"Persistence": persistence, "Constant velocity": velocity,
        "Past coordinates + ridge": ridge_forecasts(raw, data, train, spec),
        "Initial encoder + ridge": ridge_forecasts(initial_features, data, train, spec),
        "Future JEPA + ridge": ridge_forecasts(learned, data, train, spec),
        "Shuffled-future JEPA + ridge": ridge_forecasts(shuffled_features, data, train, spec)}
    # Across-clip variation is checked once per sequence, not inflated by horizons.
    _, unique_indices = np.unique(data.sequence_ids[test], return_index=True)
    features = learned[test[unique_indices]]
    centered = features - features.mean(0)
    singular = np.linalg.svd(centered, compute_uv=False)
    energy = singular ** 2
    probabilities = energy / energy.sum() if energy.sum() > 1e-16 else np.zeros_like(energy)
    rank = float(np.exp(-np.sum(probabilities[probabilities > 0] * np.log(probabilities[probabilities > 0])))) if energy.sum() > 1e-16 else 0.0
    return {"configuration": asdict(spec), "scores": score_forecasts(predictions, data, test),
            "history": pd.DataFrame(history), "shuffled_history": pd.DataFrame(shuffled_history),
            "variation": {"mean_across_clip_std": float(features.std(0).mean()), "effective_rank": rank,
                          "scope": "Descriptive over distinct test clips; not a source-level confidence interval."},
            "models": {"future_jepa": model, "shuffled_future_jepa": shuffled, "initial_encoder": initial},
            "predictions": predictions, "train_indices": train, "test_indices": test,
            "status": data.status, "coverage": data.coverage,
            "claim": "Frozen context-feature usefulness after future-feature pretraining; not decoded rollout or a clinical forecast."}


def load_local_forecasting(spec: ForecastSpec, *, fold: int = 0) -> tuple[ForecastExamples, dict]:
    """Read validated raw archives; retain existing source roles, never rebuild them."""
    from laterality.config import load_context
    from laterality.data import load_cohort, load_real_pose_records
    from laterality.splitting import get_fold, load_splits

    context = load_context(profile="paper")
    cohort = load_cohort(context)
    split = get_fold(load_splits(context, cohort), fold)
    records, _ = load_real_pose_records(context.pose_root, context.annotation_root, context.protocol["data"]["conditions"], context.protocol["data"])
    accepted = set(cohort.table.sequence_id.astype(str))
    timed = [TimedPose(r.video_id, r.sequence_id, (r.frame_numbers - r.frame_numbers[0]) / r.fps,
                       r.sequence[..., :3], observed_mask(r.sequence, spec.visibility_threshold))
             for r in records if r.sequence_id in accepted]
    return prepare_examples(timed, spec, status="EXPLORATORY LOCAL DATA — new forecasting task"), split
