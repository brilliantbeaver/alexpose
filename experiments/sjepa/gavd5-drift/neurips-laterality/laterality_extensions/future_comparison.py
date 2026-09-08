"""Observable future prediction for tutorial 14, independent of completed runs.

The forecasting model sees a prepared past and a requested horizon. A decoder
fitted on training-video future features then tests what its predicted features
can express. No observed future value is supplied to deployable predictions.
"""
from __future__ import annotations

import copy
import hashlib
import json
import os
from dataclasses import asdict, dataclass, replace
from pathlib import Path
import time
from typing import Sequence

import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import Ridge
from sklearn.model_selection import GroupKFold
from torch import nn

from laterality.metrics import source_weights
from laterality_extensions.forecasting import (
    ForecastSpec, JOINTS, TimedPose, prefix_input, shuffled_target_indices,
    validate_record,
)

ENDPOINT_JOINTS = JOINTS
ALL_JOINTS = tuple(range(33))


@dataclass(frozen=True)
class ComparisonForecastSpec(ForecastSpec):
    """Toy defaults; ``real_forecast_spec`` supplies an explicit real recipe."""
    input_joints: tuple[int, ...] = JOINTS
    updates: int = 4
    embed_dim: int = 8
    batch_size: int = 4
    ridge_alphas: tuple[float, ...] = (0.1, 10.0)
    inner_folds: int = 3
    weight_decay: float = 0.01
    gradient_clip: float = 1.0

    def __post_init__(self):
        super().__post_init__()
        if self.input_joints not in (JOINTS, ALL_JOINTS):
            raise ValueError("Declare either the twelve gait inputs or all 33 inputs")
        if (not self.ridge_alphas or any(a <= 0 or not np.isfinite(a) for a in self.ridge_alphas)
                or len(set(self.ridge_alphas)) != len(self.ridge_alphas)):
            raise ValueError("Ridge penalties must be distinct finite positive values")
        if self.inner_folds < 2 or self.weight_decay < 0 or self.gradient_clip <= 0:
            raise ValueError("Invalid validation or optimizer settings")


@dataclass
class FutureExamples:
    past: np.ndarray
    past_valid: np.ndarray
    past_times: np.ndarray
    future: np.ndarray
    future_valid: np.ndarray
    future_times: np.ndarray
    endpoint: np.ndarray
    endpoint_valid: np.ndarray
    endpoint_times: np.ndarray
    horizon: np.ndarray
    sources: np.ndarray
    sequence_ids: np.ndarray
    input_joints: tuple[int, ...]
    status: str
    coverage: dict


def prepare_prefix(record: TimedPose, spec: ComparisonForecastSpec) -> dict:
    """Use the unchanged twelve-joint prefix to fix eligibility and geometry.

    Additional inputs use the same causal last-observation sampler. Neither
    normalization nor the choice between twelve and 33 changes the endpoints.
    """
    base = prefix_input(record, spec)
    if spec.input_joints == JOINTS:
        return base
    relative = record.times - record.times[0]
    before = relative <= spec.context_seconds + 1e-10
    t, xyz, valid = relative[before], record.xyz[before], record.valid[before]
    out = np.zeros((spec.samples, 33, 3), dtype=np.float32)
    mask = np.zeros((spec.samples, 33), dtype=bool)
    sampled = np.zeros(mask.shape, dtype=np.float32)
    for row, query in enumerate(np.linspace(0, spec.context_seconds, spec.samples)):
        for joint in ALL_JOINTS:
            available = np.flatnonzero(valid[:, joint] & (t <= query + 1e-10))
            if len(available) and query - t[available[-1]] <= spec.max_past_age + 1e-10:
                index = available[-1]
                out[row, joint] = (xyz[index, joint] - base["origin"]) / base["scale"]
                mask[row, joint] = True
                sampled[row, joint] = t[index] - spec.context_seconds
    return {"xyz": out, "valid": mask, "times": sampled,
            "origin": base["origin"], "scale": base["scale"]}


def prepare_future_examples(records: Sequence[TimedPose], spec: ComparisonForecastSpec,
                            *, status: str) -> FutureExamples:
    """Retain twelve common measured endpoints, with no future imputation."""
    fields = ("past", "past_valid", "past_times", "future", "future_valid",
              "future_times", "endpoint", "endpoint_valid", "endpoint_times",
              "horizon", "sources", "sequence_ids")
    values = {key: [] for key in fields}
    rejected, accepted, seen = {}, set(), set()
    endpoint_columns = [spec.input_joints.index(joint) for joint in ENDPOINT_JOINTS]
    for record in records:
        validate_record(record)
        if record.sequence_id in seen:
            raise ValueError("Duplicate sequence identity in forecasting records")
        seen.add(record.sequence_id)
        try:
            past = prepare_prefix(record, spec)
        except ValueError as error:
            rejected[str(error)] = rejected.get(str(error), 0) + 1
            continue
        relative = record.times - record.times[0]
        for horizon in spec.horizons:
            queries = np.linspace(spec.context_seconds + horizon - spec.future_window_seconds,
                                  spec.context_seconds + horizon, spec.samples)
            future = np.zeros_like(past["xyz"])
            valid = np.zeros_like(past["valid"])
            actual_time = np.zeros_like(past["times"])
            for row, query in enumerate(queries):
                index = int(np.argmin(np.abs(relative - query)))
                if (relative[index] <= spec.context_seconds
                        or abs(relative[index] - query) > spec.target_time_tolerance):
                    continue
                joint_xyz = record.xyz[index, list(spec.input_joints)]
                observed = record.valid[index, list(spec.input_joints)] & np.isfinite(joint_xyz).all(-1)
                future[row, observed] = ((joint_xyz - past["origin"]) / past["scale"])[observed]
                valid[row, observed] = True
                actual_time[row, observed] = relative[index] - spec.context_seconds
            common_valid = valid[:, endpoint_columns]
            if common_valid.sum() < 4 or common_valid[-1].sum() < 2:
                reason = "Insufficient common future endpoints or future-window observations"
                rejected[reason] = rejected.get(reason, 0) + 1
                continue
            row_values = (past["xyz"], past["valid"], past["times"], future, valid, actual_time,
                          future[-1, endpoint_columns], common_valid[-1], actual_time[-1, endpoint_columns],
                          horizon, record.source, record.sequence_id)
            for field, value in zip(fields, row_values):
                values[field].append(value)
            accepted.add(record.sequence_id)
    if not values["past"]:
        raise ValueError("No forecasting examples meet the declared timing and common endpoint coverage")
    arrays = {key: np.asarray(value) for key, value in values.items()}
    return FutureExamples(**arrays, input_joints=spec.input_joints, status=status,
                          coverage={"input_clips": len(records), "accepted_clips": len(accepted),
                                    "examples": len(arrays["past"]), "sources": len(set(arrays["sources"])),
                                    "rejected_clip_or_horizon_reasons": rejected,
                                    "endpoint_joint_ids": list(ENDPOINT_JOINTS)})


class ForecastEncoder(nn.Module):
    def __init__(self, dimension: int, input_joints: tuple[int, ...]):
        super().__init__()
        self.input_joints = input_joints
        self.input = nn.Linear(5, dimension)
        # A full shared identity table preserves comparable parameter shapes.
        self.joint = nn.Parameter(torch.randn(33, dimension) * 0.02)
        layer = nn.TransformerEncoderLayer(dimension, 2, 2 * dimension, dropout=0.0,
                                           batch_first=True, norm_first=True)
        self.attention = nn.TransformerEncoder(layer, 1, enable_nested_tensor=False)
        self.norm = nn.LayerNorm(dimension)

    def forward(self, xyz, valid, times):
        if xyz.shape[2] != len(self.input_joints) or not valid.flatten(1).any(1).all():
            raise ValueError("An encoder input needs the declared joints and some valid context")
        xyz = torch.where(valid[..., None], xyz, torch.zeros_like(xyz))
        times = torch.where(valid, times, torch.zeros_like(times))
        feature = self.input(torch.cat((xyz, valid[..., None].float(), times[..., None]), -1))
        feature = feature + self.joint[list(self.input_joints)][None, None]
        available = valid.flatten(1)
        encoded = self.norm(self.attention(feature.flatten(1, 2), src_key_padding_mask=~available))
        return (encoded * available[..., None]).sum(1) / available.sum(1, keepdim=True)


class ForecastComparisonModel(nn.Module):
    def __init__(self, spec: ComparisonForecastSpec):
        super().__init__()
        self.encoder = ForecastEncoder(spec.embed_dim, spec.input_joints)
        self.teacher = copy.deepcopy(self.encoder).requires_grad_(False)
        self.predictor = nn.Sequential(nn.Linear(spec.embed_dim + 1, spec.embed_dim * 2),
                                       nn.GELU(), nn.Linear(spec.embed_dim * 2, spec.embed_dim))

    def predict(self, past, valid, times, horizon):
        context = self.encoder(past, valid, times)
        return self.predictor(torch.cat((context, horizon[:, None]), -1))


def source_roles(data: FutureExamples, train_sources, test_sources) -> tuple[np.ndarray, np.ndarray]:
    training, testing = set(train_sources), set(test_sources)
    if training & testing or not set(data.sources) <= training | testing:
        raise ValueError("Every source needs one explicit, nonoverlapping outer role")
    train = np.flatnonzero(np.isin(data.sources, list(training)))
    test = np.flatnonzero(np.isin(data.sources, list(testing)))
    if len(set(data.sources[train])) < 3 or len(set(data.sources[test])) < 1:
        raise ValueError("Need at least three training sources and one test source")
    identity = list(zip(data.sequence_ids, data.horizon))
    if len(set(identity)) != len(identity):
        raise ValueError("Duplicate clip-horizon examples")
    return train, test


def _tensors(data: FutureExamples) -> dict:
    return {field: torch.as_tensor(getattr(data, field),
                                  dtype=torch.bool if field.endswith("valid") else torch.float32)
            for field in ("past", "past_valid", "past_times", "future", "future_valid", "future_times", "horizon")}


def fit_forecast_model(data: FutureExamples, spec: ComparisonForecastSpec, *,
                       train_sources, test_sources, mismatched=False) -> dict:
    train, _ = source_roles(data, train_sources, test_sources)
    if data.input_joints != spec.input_joints:
        raise ValueError("Prepared inputs and declared model inputs disagree")
    with torch.random.fork_rng():
        torch.manual_seed(spec.seed)
        model = ForecastComparisonModel(spec)
    initial = copy.deepcopy(model.encoder).eval()
    optimizer = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad),
                                  lr=spec.learning_rate, weight_decay=spec.weight_decay)
    tensor = _tensors(data)
    sample_rng = np.random.default_rng(np.random.SeedSequence([spec.seed, 101]))
    mismatch_rng = np.random.default_rng(np.random.SeedSequence([spec.seed, 202]))
    sources = np.unique(data.sources[train])
    members = {source: train[data.sources[train] == source] for source in sources}
    for horizon in np.unique(data.horizon[train]):
        if len(set(data.sources[train[data.horizon[train] == horizon]])) < 2:
            raise ValueError("Every training horizon needs at least two sources for its mismatched-future control")
    history, schedule, target_schedule = [], [], []
    started = time.perf_counter()
    for update in range(spec.updates):
        source_draws = sample_rng.choice(sources, size=spec.batch_size)
        indices = np.asarray([sample_rng.choice(members[source]) for source in source_draws])
        targets = shuffled_target_indices(indices, train, data.sources, data.horizon, mismatch_rng) if mismatched else indices
        context = model.encoder(tensor["past"][indices], tensor["past_valid"][indices], tensor["past_times"][indices])
        predicted = model.predictor(torch.cat((context, tensor["horizon"][indices, None]), -1))
        with torch.no_grad():
            target = model.teacher(tensor["future"][targets], tensor["future_valid"][targets], tensor["future_times"][targets])
        prediction_loss = (predicted - target).square().mean()
        variance_loss = torch.relu(1 - torch.sqrt(context.var(0, unbiased=False) + 1e-4)).mean()
        loss = prediction_loss + spec.variance_weight * variance_loss
        if not torch.isfinite(loss):
            raise FloatingPointError("Nonfinite forecasting loss")
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), spec.gradient_clip)
        optimizer.step()
        with torch.no_grad():
            for teacher, online in zip(model.teacher.parameters(), model.encoder.parameters()):
                teacher.mul_(spec.ema).add_(online, alpha=1 - spec.ema)
        schedule.append(indices.tolist())
        target_schedule.append(targets.tolist())
        history.append({"update": update + 1, "prediction_loss": float(prediction_loss.detach()),
                        "variance_penalty": float(variance_loss.detach())})
    return {"model": model.eval(), "initial_encoder": initial, "history": history,
            "source_schedule": schedule, "target_schedule": target_schedule,
            "runtime_seconds": time.perf_counter() - started,
            "train_sources": sorted(set(data.sources[train])), "mismatched": mismatched,
            "checkpoint": "final online encoder, final EMA teacher, final predictor"}


def extract_features(model: ForecastComparisonModel, data: FutureExamples,
                     initial: ForecastEncoder | None = None) -> dict[str, np.ndarray]:
    tensor = _tensors(data)
    parts = {name: [] for name in ("past_online", "past_teacher", "future_teacher", "predicted_future")}
    if initial is not None:
        parts["past_initial"] = []
    with torch.no_grad():
        for start in range(0, len(data.past), 64):
            take = slice(start, start + 64)
            past = [tensor[name][take] for name in ("past", "past_valid", "past_times")]
            future = [tensor[name][take] for name in ("future", "future_valid", "future_times")]
            parts["past_online"].append(model.encoder(*past).numpy())
            parts["past_teacher"].append(model.teacher(*past).numpy())
            parts["future_teacher"].append(model.teacher(*future).numpy())
            parts["predicted_future"].append(model.predict(*past, tensor["horizon"][take]).numpy())
            if initial is not None:
                parts["past_initial"].append(initial(*past).numpy())
    return {name: np.concatenate(values) for name, values in parts.items()}


def _design(features: np.ndarray, horizon: np.ndarray) -> np.ndarray:
    return np.column_stack((features, horizon, features * horizon[:, None]))


def _scaling(values: np.ndarray, sources: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    if not np.isfinite(values).all():
        raise ValueError("Prepared forecast features must be finite; unavailable endpoints remain unscored")
    weights = source_weights(sources)
    mean = np.average(values, axis=0, weights=weights)
    scale = np.sqrt(np.average((values - mean) ** 2, axis=0, weights=weights))
    return mean, np.where(scale > 1e-10, scale, 1.0)


@dataclass
class CoordinateDecoder:
    mean: np.ndarray
    scale: np.ndarray
    coefficients: np.ndarray
    intercepts: np.ndarray
    selected_alphas: np.ndarray
    fitted_sources: tuple[str, ...]
    validation_sources: list[dict]

    def predict(self, features: np.ndarray, horizon: np.ndarray) -> np.ndarray:
        design = (_design(features, horizon) - self.mean) / self.scale
        return np.einsum("nf,jcf->njc", design, self.coefficients) + self.intercepts


def fit_coordinate_decoder(features: np.ndarray, data: FutureExamples, spec: ComparisonForecastSpec,
                           *, train_sources, test_sources) -> CoordinateDecoder:
    """Tune only the readout using inner source groups; encoder is already fixed."""
    train, _ = source_roles(data, train_sources, test_sources)
    design = _design(features, data.horizon)
    mean, scale = _scaling(design[train], data.sources[train])
    coefficients = np.full((len(ENDPOINT_JOINTS), 3, design.shape[1]), np.nan)
    intercepts = np.full((len(ENDPOINT_JOINTS), 3), np.nan)
    alphas = np.full(len(ENDPOINT_JOINTS), np.nan)
    validation = []
    for joint in range(len(ENDPOINT_JOINTS)):
        observed = train[data.endpoint_valid[train, joint]]
        source_count = len(set(data.sources[observed]))
        if source_count < 3:
            validation.append({"joint_id": ENDPOINT_JOINTS[joint], "status": "Fewer than three training sources with endpoints"})
            continue
        splits = list(GroupKFold(min(spec.inner_folds, source_count)).split(observed, groups=data.sources[observed]))
        losses = []
        for alpha in spec.ridge_alphas:
            predictions = np.full((len(observed), 3), np.nan)
            for local_fit, local_valid in splits:
                fit_rows, valid_rows = observed[local_fit], observed[local_valid]
                fold_mean, fold_scale = _scaling(design[fit_rows], data.sources[fit_rows])
                fitted = Ridge(alpha=alpha).fit((design[fit_rows] - fold_mean) / fold_scale,
                                                data.endpoint[fit_rows, joint],
                                                sample_weight=source_weights(data.sources[fit_rows]))
                predictions[local_valid] = fitted.predict((design[valid_rows] - fold_mean) / fold_scale)
            error = np.square(predictions - data.endpoint[observed, joint]).mean(-1)
            losses.append(float(np.average(error, weights=source_weights(data.sources[observed]))))
        alpha = spec.ridge_alphas[int(np.argmin(losses))]
        fitted = Ridge(alpha=alpha).fit((design[observed] - mean) / scale,
                                        data.endpoint[observed, joint],
                                        sample_weight=source_weights(data.sources[observed]))
        coefficients[joint], intercepts[joint], alphas[joint] = fitted.coef_, fitted.intercept_, alpha
        validation.append({"joint_id": ENDPOINT_JOINTS[joint], "selected_alpha": alpha,
                           "candidate_source_balanced_mse": losses,
                           "splits": [{"fit_sources": sorted(set(data.sources[observed[a]])),
                                       "validation_sources": sorted(set(data.sources[observed[b]]))} for a, b in splits]})
    return CoordinateDecoder(mean, scale, coefficients, intercepts, alphas,
                             tuple(sorted(set(data.sources[train]))), validation)


def future_motion_baselines(data: FutureExamples) -> tuple[np.ndarray, np.ndarray]:
    persistent = np.full(data.endpoint.shape, np.nan)
    velocity = persistent.copy()
    for example in range(len(data.past)):
        for col, joint_id in enumerate(ENDPOINT_JOINTS):
            joint = data.input_joints.index(joint_id)
            observed = np.flatnonzero(data.past_valid[example, :, joint])
            if not len(observed):
                continue
            last = observed[-1]
            persistent[example, col] = data.past[example, last, joint]
            velocity[example, col] = persistent[example, col]
            earlier = observed[data.past_times[example, observed, joint] < data.past_times[example, last, joint] - 1e-6]
            if len(earlier):
                prior = earlier[-1]
                dt = data.past_times[example, last, joint] - data.past_times[example, prior, joint]
                rate = (data.past[example, last, joint] - data.past[example, prior, joint]) / dt
                # Actual future measurement time is reported only during scoring.
                velocity[example, col] += rate * (data.horizon[example] - data.past_times[example, last, joint])
    return persistent, velocity


def feature_diagnostics(features: np.ndarray) -> dict:
    centered = features - features.mean(0)
    energy = np.linalg.svd(centered, compute_uv=False) ** 2
    probabilities = energy / energy.sum() if energy.sum() > 1e-16 else np.zeros_like(energy)
    positive = probabilities[probabilities > 0]
    rank = float(np.exp(-np.sum(positive * np.log(positive)))) if len(positive) else 0.0
    variation = float(features.std(0).mean())
    return {"mean_feature_std": variation, "effective_rank": rank,
            "mean_feature_norm": float(np.linalg.norm(features, axis=1).mean()),
            "status": "Nearly constant features: usefulness is unresolved" if variation < 1e-6 else "Variation present; movement usefulness needs its own score"}


def score_future_predictions(predictions: dict[str, np.ndarray], data: FutureExamples,
                             test: np.ndarray) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Report the common intersection and every method's lost coverage explicitly."""
    if len(np.unique(test)) != len(test):
        raise ValueError("Duplicate evaluation rows")
    common = data.endpoint_valid.copy()
    for prediction in predictions.values():
        if prediction.shape != data.endpoint.shape:
            raise ValueError("Prediction endpoints differ from the declared endpoint array")
        common &= np.isfinite(prediction).all(-1)
    scores, coverage = [], []
    for horizon in np.unique(data.horizon[test]):
        selected = test[data.horizon[test] == horizon]
        scored = selected[common[selected].any(1)]
        for name, prediction in predictions.items():
            own = data.endpoint_valid[selected] & np.isfinite(prediction[selected]).all(-1)
            coverage.append({"method": name, "horizon_seconds": float(horizon),
                             "requested_examples": len(selected), "observed_endpoints": int(data.endpoint_valid[selected].sum()),
                             "unavailable_predictions": int((data.endpoint_valid[selected] & ~own).sum()),
                             "common_endpoints": int(common[selected].sum()),
                             "examples_without_common_endpoints": int((~common[selected].any(1)).sum())})
            rmse = float("nan")
            if len(scored):
                errors = np.square(prediction[scored] - data.endpoint[scored]).mean(-1)
                clip_error = np.asarray([row[valid].mean() for row, valid in zip(errors, common[scored])])
                rmse = float(np.sqrt(np.average(clip_error, weights=source_weights(data.sources[scored]))))
            actual = data.endpoint_times[selected][common[selected]]
            scores.append({"method": name, "horizon_seconds": float(horizon), "source_balanced_rmse": rmse,
                           "test_sources": len(set(data.sources[scored])), "test_clips": len(set(data.sequence_ids[scored])),
                           "test_examples": len(scored), "common_endpoints": int(common[selected].sum()),
                           "actual_time_min": float(actual.min()) if len(actual) else float("nan"),
                           "actual_time_max": float(actual.max()) if len(actual) else float("nan"),
                           "status": data.status})
    return pd.DataFrame(scores), pd.DataFrame(coverage)


def forecast_error_rows(predictions: dict[str, np.ndarray], *, sources, sequence_ids,
                        horizon, target, valid, seed: int, fold: int,
                        input_landmarks: int) -> pd.DataFrame:
    """One retained error per clip, horizon, and method, including unavailable rows.

    The common endpoint mask is retained as a reference so later aggregation can
    reject input/model comparisons measured on different landmark observations.
    """
    common = np.asarray(valid).copy()
    for predicted in predictions.values():
        if np.shape(predicted) != np.shape(target):
            raise ValueError("Forecast arrays use different coordinate endpoints")
        common &= np.isfinite(predicted).all(-1)
    rows = []
    for index in range(len(sources)):
        endpoint_reference = hashlib.sha256(np.asarray(target[index]).tobytes() + np.asarray(valid[index]).tobytes()).hexdigest()
        coverage_reference = "".join("1" if item else "0" for item in common[index])
        for method, predicted in predictions.items():
            available = bool(common[index].any())
            error = float(np.square(predicted[index, common[index]] - target[index, common[index]]).mean()) if available else float("nan")
            rows.append({"source_id": str(sources[index]), "sequence_id": str(sequence_ids[index]),
                         "horizon_seconds": float(horizon[index]), "seed": int(seed), "fold": int(fold),
                         "input_landmarks": int(input_landmarks), "method": method,
                         "clip_mse": error, "available": available,
                         "observed_endpoints": int(valid[index].sum()),
                         "common_endpoints": int(common[index].sum()),
                         "target_reference": endpoint_reference, "coverage_reference": coverage_reference})
    return pd.DataFrame(rows)


def saved_forecast_error_rows(saved: dict) -> pd.DataFrame:
    arrays = saved["arrays"]
    configuration = saved["reference"]["configuration"]
    return forecast_error_rows(
        {name: arrays[f"prediction_{index}"] for index, name in enumerate(saved["prediction_methods"])},
        sources=arrays["sources"], sequence_ids=arrays["sequence_ids"], horizon=arrays["horizon"],
        target=arrays["target"], valid=arrays["valid"], seed=configuration["seed"],
        fold=saved["reference"].get("outer_fold", -1), input_landmarks=len(configuration["input_joints"]),
    )


def validate_future_error_rows(rows: pd.DataFrame, expected: pd.DataFrame, *,
                               seeds, input_sizes, methods, allow_partial=False) -> pd.DataFrame:
    identity = ["sequence_id", "horizon_seconds"]
    keys = ["input_landmarks", "method", "seed", *identity]
    required = {*keys, "source_id", "fold", "clip_mse", "available", "common_endpoints",
                "observed_endpoints", "target_reference", "coverage_reference"}
    if rows.empty or not required <= set(rows) or rows.duplicated(keys).any():
        raise ValueError("Missing columns, empty data, or duplicate forecast predictions")
    if (not {*identity, "source_id", "fold"} <= set(expected)
            or expected.empty or expected.duplicated(identity).any()):
        raise ValueError("Declare one expected source/fold identity for every clip-horizon")
    if expected.groupby("source_id").fold.nunique().max() != 1 or rows.groupby("source_id").fold.nunique().max() != 1:
        raise ValueError("A source occurs in more than one outer test fold")
    joined = rows.merge(expected[[*identity, "source_id", "fold"]], on=identity,
                        how="left", suffixes=("", "_expected"), validate="many_to_one")
    if ((joined.source_id != joined.source_id_expected).any()
            or (joined.fold != joined.fold_expected).any()):
        raise ValueError("Forecast identities differ from declared evaluation coverage")
    for field, declaration in (("seed", seeds), ("input_landmarks", input_sizes), ("method", methods)):
        if not declaration or len(set(declaration)) != len(declaration) or not set(rows[field]) <= set(declaration):
            raise ValueError("Declare unique nonempty seeds, input choices, and methods before aggregation")
    if rows.groupby(identity).target_reference.nunique().max() != 1:
        raise ValueError("Compared predictions use different coordinate targets")
    if rows.groupby(identity).coverage_reference.nunique().max() != 1:
        raise ValueError("Compared predictions use different endpoint coverage; declare a shared comparison first")
    available = rows.available.to_numpy()
    if available.dtype != bool or not np.isfinite(rows.loc[available, "clip_mse"]).all():
        raise ValueError("Available forecasts need finite errors and boolean availability")
    if ((rows.loc[available, "clip_mse"] < 0).any()
            or rows.loc[~available, "clip_mse"].notna().any()):
        raise ValueError("Invalid forecast errors or unavailable predictions represented as successes")
    expected_keys = set(map(tuple, expected[identity].to_numpy()))
    complete = True
    for inputs in input_sizes:
        for method in methods:
            for seed in seeds:
                group = rows[(rows.input_landmarks == inputs) & (rows.method == method) & (rows.seed == seed)]
                if set(map(tuple, group[identity].to_numpy())) != expected_keys:
                    complete = False
                    if not allow_partial:
                        raise ValueError("Incomplete declared forecasting coverage; explicitly request a partial summary")
    output = rows.copy()
    output.attrs["scope"] = "Complete declared comparison" if complete else "Partial declared comparison"
    return output


def aggregate_future_error_rows(rows: pd.DataFrame, expected: pd.DataFrame, **declarations) -> dict:
    """Pool every outer fold within each seed, then summarize seed-specific RMSE."""
    checked = validate_future_error_rows(rows, expected, **declarations)
    summaries = []
    grouping = ["input_landmarks", "method", "horizon_seconds", "seed"]
    for names, group in checked.groupby(grouping):
        available = group[group.available]
        mse = float(np.average(available.clip_mse, weights=source_weights(available.source_id))) if len(available) else float("nan")
        summaries.append({**dict(zip(grouping, names)), "source_balanced_rmse": float(np.sqrt(mse)),
                          "evaluated_sources": available.source_id.nunique(),
                          "evaluated_clips": available.sequence_id.nunique(), "retained_examples": len(group),
                          "unavailable_examples": int((~group.available).sum()),
                          "common_endpoints": int(group.common_endpoints.sum()), "scope": checked.attrs["scope"]})
    per_seed = pd.DataFrame(summaries)
    summary = per_seed.groupby(grouping[:-1], as_index=False).agg(
        mean_rmse=("source_balanced_rmse", "mean"), seed_sd_rmse=("source_balanced_rmse", "std"),
        evaluated_seeds=("seed", "nunique"), evaluated_sources_min=("evaluated_sources", "min"),
        evaluated_clips_min=("evaluated_clips", "min"), unavailable_examples_max=("unavailable_examples", "max"),
        scope=("scope", "first"))
    return {"per_seed": per_seed, "summary": summary}


def paired_future_source_interval(rows: pd.DataFrame, expected: pd.DataFrame, *,
                                   first: str, reference: str, input_landmarks: int,
                                   horizon_seconds: float, seeds, repetitions=2000,
                                   random_seed=918) -> dict:
    """Resample entire videos jointly across each method and all training seeds."""
    if repetitions < 2 or first == reference:
        raise ValueError("Specify two methods and at least two source resamples")
    selected = rows[(rows.input_landmarks == input_landmarks) & (rows.horizon_seconds == horizon_seconds)
                    & rows.method.isin([first, reference])].copy()
    expected_horizon = expected[expected.horizon_seconds == horizon_seconds]
    checked = validate_future_error_rows(selected, expected_horizon, seeds=seeds,
                                         input_sizes=(input_landmarks,), methods=(first, reference))
    if not checked.available.all():
        raise ValueError("Paired source intervals require complete finite shared coverage")
    sources = np.array(sorted(checked.source_id.unique()))
    if len(sources) < 2:
        raise ValueError("Need at least two videos for paired source resampling")

    def difference(multiplicity):
        changes = []
        for seed in seeds:
            errors = []
            for method in (first, reference):
                group = checked[(checked.seed == seed) & (checked.method == method)]
                weight = source_weights(group.source_id) * group.source_id.map(multiplicity).fillna(0).to_numpy()
                errors.append(np.sqrt(np.average(group.clip_mse, weights=weight)))
            changes.append(errors[0] - errors[1])
        return float(np.mean(changes))

    rng = np.random.default_rng(random_seed)
    draws = []
    for _ in range(repetitions):
        sample, counts = np.unique(rng.choice(sources, len(sources), replace=True), return_counts=True)
        draws.append(difference(dict(zip(sample, counts))))
    low, high = np.quantile(draws, [0.025, 0.975])
    return {"rmse_difference": difference(dict.fromkeys(sources, 1)), "lower_95": float(low), "upper_95": float(high),
            "subtraction": f"{first} minus {reference}", "source_videos": len(sources), "training_seeds": len(seeds),
            "resamples": repetitions,
            "scope": "Source-resampling interval conditional on fitted models; retraining uncertainty excluded"}


def run_future_comparison(data: FutureExamples, spec: ComparisonForecastSpec, *,
                          train_sources, test_sources) -> dict:
    train, test = source_roles(data, train_sources, test_sources)
    fitted = fit_forecast_model(data, spec, train_sources=train_sources, test_sources=test_sources)
    mismatched = fit_forecast_model(data, spec, train_sources=train_sources, test_sources=test_sources, mismatched=True)
    if fitted["source_schedule"] != mismatched["source_schedule"]:
        raise AssertionError("Paired training arms received different source schedules")
    features = extract_features(fitted["model"], data, fitted["initial_encoder"])
    wrong_features = extract_features(mismatched["model"], data)
    persistent, velocity = future_motion_baselines(data)
    predictions = {"Last observed position": persistent, "Recent velocity continuation": velocity}
    raw = np.concatenate((data.past.reshape(len(data.past), -1),
                          data.past_valid.reshape(len(data.past), -1),
                          data.past_times.reshape(len(data.past), -1)), axis=1)
    readout_inputs = {"Past coordinates with regression": raw,
                      "Initial past encoder with regression": features["past_initial"],
                      "Trained past encoder with regression": features["past_online"],
                      "Past teacher with regression": features["past_teacher"],
                      "Mismatched-future past encoder with regression": wrong_features["past_online"]}
    decoders = {}
    for name, feature in readout_inputs.items():
        decoder = fit_coordinate_decoder(feature, data, spec, train_sources=train_sources, test_sources=test_sources)
        decoders[name] = decoder
        predictions[name] = decoder.predict(feature, data.horizon)
    for label, arm_features in (("Matched", features), ("Mismatched", wrong_features)):
        decoder = fit_coordinate_decoder(arm_features["future_teacher"], data, spec,
                                         train_sources=train_sources, test_sources=test_sources)
        decoders[f"{label} future-feature decoder"] = decoder
        predictions[f"{label}: predicted future features decoded"] = decoder.predict(arm_features["predicted_future"], data.horizon)
        predictions[f"{label}: observed future features decoded (diagnostic)"] = decoder.predict(arm_features["future_teacher"], data.horizon)
    scores, coverage = score_future_predictions(predictions, data, test)
    diagnostic_rows = []
    for arm, arm_features in (("Matched", features), ("Mismatched", wrong_features)):
        for field in ("past_online", "past_teacher", "future_teacher", "predicted_future"):
            diagnostic_rows.append({"arm": arm, "feature": field, **feature_diagnostics(arm_features[field][test])})
    latent = []
    for arm, arm_features in (("Matched", features), ("Mismatched", wrong_features)):
        target, predicted = arm_features["future_teacher"][test], arm_features["predicted_future"][test]
        denominator = float(np.square(target - target.mean(0)).mean())
        error = float(np.square(predicted - target).mean())
        latent.append({"arm": arm, "feature_mse": error, "target_variance_denominator": denominator,
                       "normalized_error": error / denominator if denominator > 1e-12 else None,
                       "scope": "Own-teacher diagnostic; teachers do not provide a common latent measurement scale"})
    return {"configuration": asdict(spec), "status": data.status, "scores": scores,
            "prediction_coverage": coverage, "feature_diagnostics": pd.DataFrame(diagnostic_rows),
            "latent_diagnostics": latent, "predictions": predictions, "decoders": decoders,
            "matched": fitted, "mismatched": mismatched, "train_indices": train, "test_indices": test,
            "coverage": data.coverage, "scope": "Independent horizons from one fixed prefix; no recursive rollout"}


def real_forecast_spec() -> ComparisonForecastSpec:
    """An explicit starting recipe for a separate forecasting experiment."""
    return ComparisonForecastSpec(updates=1200, embed_dim=24, batch_size=8,
                                  seed=42, ridge_alphas=(0.1, 1.0, 10.0, 100.0))


def plan_future_comparison(spec: ComparisonForecastSpec | None = None, *,
                           folds=(0, 1, 2, 3, 4), seeds=(42, 43, 44, 45, 46),
                           input_sets=(JOINTS,), output_root="outputs/masking_extensions/future") -> dict:
    spec = spec or real_forecast_spec()
    if (not folds or len(set(folds)) != len(folds) or not set(folds) <= set(range(5))
            or not seeds or len(set(seeds)) != len(seeds)
            or not input_sets or len(set(input_sets)) != len(input_sets)):
        raise ValueError("Declare unique existing folds, training seeds, and input sets")
    for joints in input_sets:
        replace(spec, input_joints=joints)
    rows = [{"fold": fold, "seed": seed, "input_landmarks": len(joints),
             "conditions": ["Matched future", "Mismatched future"], "updates_per_model": spec.updates,
             "output": str(Path(output_root) / f"joints_{len(joints)}" / f"fold_{fold}_seed_{seed}")}
            for joints in input_sets for fold in folds for seed in seeds]
    full = tuple(sorted(folds)) == tuple(range(5)) and tuple(sorted(seeds)) == (42, 43, 44, 45, 46)
    return {"scope": "Complete declared comparison" if full else "Pilot or partial declared comparison",
            "runs": rows, "model_fits": len(rows) * 2, "optimizer_updates": len(rows) * 2 * spec.updates,
            "source_example_draws": len(rows) * 2 * spec.updates * spec.batch_size,
            "recipe": {**asdict(spec), "optimizer": "AdamW", "learning_rate_schedule": "Constant",
                       "optimizer_betas": [0.9, 0.999], "optimizer_epsilon": 1e-8,
                       "encoder_layers": 1, "attention_heads": 2, "encoder_dropout": 0.0,
                       "encoder_feedforward_width": 2 * spec.embed_dim,
                       "encoder_activation": "ReLU in transformer feedforward",
                       "predictor_hidden_width": 2 * spec.embed_dim,
                       "predictor_activation": "GELU", "joint_identity_table_size": 33,
                       "variance_epsilon": 1e-4, "variance_target_std": 1.0,
                       "prediction_loss": "Mean squared error across batch and feature dimensions",
                       "source_sampling_stream": "SeedSequence([training_seed, 101])",
                       "mismatched_target_stream": "SeedSequence([training_seed, 202])",
                       "teacher_update": "EMA after each optimizer update", "augmentation": "None",
                       "forecasting_windows": "One fixed prefix per clip, separate strictly future windows",
                       "regularizer_pool": "Valid input landmarks in the observed prefix",
                       "normalization": "Prefix-only pelvis origin and body width",
                       "endpoint_joint_ids": list(ENDPOINT_JOINTS),
                       "checkpoint_selection": "Final update, fixed before outer evaluation",
                       "readout_selection": "Inner source-separated ridge validation only",
                       "resume": "Unsupported; incomplete runs are rejected"},
            "workload_note": "Runtime depends on input length and hardware; both wall time and update counts are saved.",
            "recipe_note": "Separate forecasting architecture and objective; the 1,200-update setting does not reproduce Notebook 08's training recipe."}


def _jsonable(value):
    if isinstance(value, np.ndarray):
        return _jsonable(value.tolist())
    if isinstance(value, (np.integer, np.floating)):
        return _jsonable(value.item())
    if isinstance(value, float) and not np.isfinite(value):
        return None
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    return value


def compatibility_reference(data: FutureExamples, spec: ComparisonForecastSpec, train_sources, test_sources) -> dict:
    source_roles(data, train_sources, test_sources)
    digest = hashlib.sha256()
    for field in ("past", "past_valid", "past_times", "future", "future_valid", "future_times",
                  "endpoint", "endpoint_valid", "endpoint_times", "horizon", "sources", "sequence_ids"):
        array = np.asarray(getattr(data, field))
        digest.update(field.encode()); digest.update(str(array.dtype).encode())
        digest.update(str(array.shape).encode()); digest.update(array.tobytes())
    from laterality_extensions import forecasting
    import sklearn
    implementation = hashlib.sha256(Path(__file__).read_bytes() + Path(forecasting.__file__).read_bytes()).hexdigest()
    return _jsonable({"configuration": asdict(spec), "data": digest.hexdigest(), "implementation": implementation,
                      "train_sources": sorted(set(train_sources)), "test_sources": sorted(set(test_sources)),
                      "libraries": {"torch": torch.__version__, "numpy": np.__version__, "sklearn": sklearn.__version__}})


def save_future_result(path: str | Path, result: dict, data: FutureExamples, reference: dict) -> None:
    """Create a new directory; completion becomes visible only after every file."""
    destination = Path(path)
    destination.mkdir(parents=True, exist_ok=False)
    payload = {"reference": reference, "status": result["status"], "coverage": result["coverage"],
               "scores": result["scores"].to_dict("records"),
               "prediction_coverage": result["prediction_coverage"].to_dict("records"),
               "feature_diagnostics": result["feature_diagnostics"].to_dict("records"),
               "latent_diagnostics": result["latent_diagnostics"], "scope": result["scope"],
               "decoders": {name: asdict(value) for name, value in result["decoders"].items()},
               "training": {arm: {key: result[arm][key] for key in ("history", "source_schedule", "target_schedule", "runtime_seconds", "train_sources", "checkpoint")}
                            for arm in ("matched", "mismatched")},
               "example_index": [{"source": str(source), "sequence_id": str(sequence), "horizon_seconds": float(horizon)}
                                 for source, sequence, horizon in zip(data.sources, data.sequence_ids, data.horizon)]}
    test = result["test_indices"]
    arrays = {"sources": data.sources[test], "sequence_ids": data.sequence_ids[test], "horizon": data.horizon[test],
              "target": data.endpoint[test], "valid": data.endpoint_valid[test], "actual_times": data.endpoint_times[test],
              "seed": np.full(len(test), result["configuration"]["seed"], dtype=int),
              "fold": np.full(len(test), reference.get("outer_fold", -1), dtype=int)}
    arrays.update({f"prediction_{index}": prediction[test] for index, prediction in enumerate(result["predictions"].values())})
    payload["prediction_methods"] = list(result["predictions"])
    payload["fold_seed_note"] = "Fold and seed are retained in the configuration/reference and run manifest."
    with (destination / "predictions.npz.partial").open("wb") as stream:
        np.savez_compressed(stream, **arrays)
    os.replace(destination / "predictions.npz.partial", destination / "predictions.npz")
    states = {f"{arm}.{name}": tensor.detach().cpu() for arm in ("matched", "mismatched")
              for name, tensor in result[arm]["model"].state_dict().items()}
    states.update({f"initial_encoder.{name}": tensor.detach().cpu() for name, tensor in result["matched"]["initial_encoder"].state_dict().items()})
    torch.save(states, destination / "checkpoints.pt.partial")
    os.replace(destination / "checkpoints.pt.partial", destination / "checkpoints.pt")
    (destination / "result.json.partial").write_text(json.dumps(_jsonable(payload), indent=2, allow_nan=False) + "\n")
    os.replace(destination / "result.json.partial", destination / "result.json")
    checksums = {name: hashlib.sha256((destination / name).read_bytes()).hexdigest()
                 for name in ("result.json", "checkpoints.pt", "predictions.npz")}
    (destination / "complete.json.partial").write_text(json.dumps({"files": checksums}) + "\n")
    os.replace(destination / "complete.json.partial", destination / "complete.json")


def load_future_result(path: str | Path, reference: dict) -> dict:
    destination = Path(path)
    if not (destination / "complete.json").is_file():
        raise ValueError("Incomplete forecasting run; automatic resume is unsupported")
    try:
        marker = json.loads((destination / "complete.json").read_text())
        if set(marker["files"]) != {"result.json", "checkpoints.pt", "predictions.npz"}:
            raise ValueError("Unexpected saved result file set")
        for name, checksum in marker["files"].items():
            if hashlib.sha256((destination / name).read_bytes()).hexdigest() != checksum:
                raise ValueError("Corrupted saved forecasting result")
        payload = json.loads((destination / "result.json").read_text())
        if payload["reference"] != _jsonable(reference):
            raise ValueError("Saved forecasting run is incompatible with this request")
        with np.load(destination / "predictions.npz", allow_pickle=False) as archive:
            arrays = {key: archive[key].copy() for key in archive.files}
        identities = list(zip(arrays["sequence_ids"], arrays["horizon"]))
        if len(set(identities)) != len(identities):
            raise ValueError("Duplicate saved clip-horizon predictions")
        expected = (len(identities), len(ENDPOINT_JOINTS), 3)
        if arrays["target"].shape != expected or any(arrays[f"prediction_{i}"].shape != expected for i in range(len(payload["prediction_methods"]))):
            raise ValueError("Saved prediction shape mismatch")
        payload["arrays"] = arrays
        return payload
    except (KeyError, OSError, json.JSONDecodeError) as error:
        raise ValueError("Malformed or incomplete forecasting result") from error


def load_future_records(*, visibility_threshold=0.45) -> tuple[list[TimedPose], dict[int, dict]]:
    """Read the accepted cohort and existing splits without modifying them."""
    from laterality.config import load_context
    from laterality.data import load_cohort, load_real_pose_records
    from laterality.geometry import observed_mask
    from laterality.splitting import get_fold, load_splits
    context = load_context(profile="paper")
    cohort = load_cohort(context)
    splits = load_splits(context, cohort)
    records, _ = load_real_pose_records(context.pose_root, context.annotation_root,
                                      context.protocol["data"]["conditions"], context.protocol["data"])
    accepted = set(cohort.table.sequence_id.astype(str))
    timed = [TimedPose(record.video_id, record.sequence_id,
                       (record.frame_numbers - record.frame_numbers[0]) / record.fps,
                       record.sequence[..., :3], observed_mask(record.sequence, visibility_threshold))
             for record in records if record.sequence_id in accepted]
    return timed, {fold: get_fold(splits, fold) for fold in range(5)}


def run_real_future_comparison(*, enabled=False, validate_inputs=False,
                               spec: ComparisonForecastSpec | None = None,
                               folds=(0, 1, 2, 3, 4), seeds=(42, 43, 44, 45, 46),
                               input_sets=(JOINTS,), output_root="outputs/masking_extensions/future") -> dict:
    spec = spec or real_forecast_spec()
    plan = plan_future_comparison(spec, folds=folds, seeds=seeds, input_sets=input_sets, output_root=output_root)
    print(json.dumps(_jsonable(plan), indent=2))
    if not enabled and not validate_inputs:
        return {"plan": plan, "status": "Training disabled; inputs have not been loaded"}
    records, splits = load_future_records(visibility_threshold=spec.visibility_threshold)
    prepared = {len(joints): prepare_future_examples(records, replace(spec, input_joints=joints),
                 status="REAL DATA — separate forecasting comparison") for joints in input_sets}
    validation = []
    for row in plan["runs"]:
        data, split = prepared[row["input_landmarks"]], splits[row["fold"]]
        train, test = source_roles(data, split["train_sources"], split["test_sources"])
        validation.append({"fold": row["fold"], "input_landmarks": row["input_landmarks"],
                           "train_examples": len(train), "test_examples": len(test), "coverage": data.coverage})
    if not enabled:
        return {"plan": plan, "input_validation": validation, "status": "Inputs validated; training disabled"}
    outputs = []
    for row in plan["runs"]:
        data, split = prepared[row["input_landmarks"]], splits[row["fold"]]
        run_spec = replace(spec, input_joints=data.input_joints, seed=row["seed"])
        reference = compatibility_reference(data, run_spec, split["train_sources"], split["test_sources"])
        reference["outer_fold"] = row["fold"]
        destination = Path(row["output"])
        if destination.exists():
            load_future_result(destination, reference)
            outputs.append({"output": str(destination), "status": "Compatible completed result reused"})
            continue
        result = run_future_comparison(data, run_spec, train_sources=split["train_sources"], test_sources=split["test_sources"])
        save_future_result(destination, result, data, reference)
        outputs.append({"output": str(destination), "status": "Completed new forecasting comparison"})
    return {"plan": plan, "input_validation": validation, "outputs": outputs, "status": "Requested runs complete"}
