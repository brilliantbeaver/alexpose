"""Source-separated evaluation for the comparative masking tutorials.

The fitted readout is a separate object from the features used to evaluate it.
This makes it possible to reuse the same readout under missing observations,
and to inspect which sources supplied every fitted preprocessing quantity.
All functions are independent of the completed research artifact writers.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import Ridge

from laterality.evaluation import WeightedScaler, laterality_features
from laterality.geometry import observed_mask, prepare_pose
from laterality.metrics import source_weights, weighted_mae, weighted_r2
from laterality.model import valid_patches
from laterality_extensions.masked_learning import LearningDataset, PROBE_PAIRS, raw_pose_features


@dataclass
class SourceReadout:
    """A frozen imputer/scaler and ridge model fitted on declared sources only."""

    scaler: WeightedScaler
    regression: Ridge
    train_sources: tuple[str, ...]
    selected_alpha: float
    validation: pd.DataFrame
    training_feature_diagnostics: dict[str, Any]

    def predict(self, features: np.ndarray) -> np.ndarray:
        return self.regression.predict(self.scaler.transform(features))


def feature_diagnostics(features: np.ndarray, source_ids: np.ndarray,
                        *, relative_floor: float = 1e-6) -> dict[str, Any]:
    """Measure source-balanced variation and covariance effective rank.

    Effective rank is exp(entropy of normalized covariance eigenvalues). A
    small absolute/relative variation flag is a numerical diagnostic, with no
    implication that high rank demonstrates useful movement information.
    """
    values = np.asarray(features, dtype=np.float64)
    sources = np.asarray(source_ids).astype(str)
    if values.ndim != 2 or sources.shape != (len(values),) or not len(values):
        raise ValueError("Features require [examples, channels] and one source per example")
    if not np.isfinite(values).all():
        raise ValueError("Feature diagnostics require finite values")
    weights = source_weights(sources)
    weights /= weights.sum()
    mean = np.sum(values * weights[:, None], axis=0)
    centered = values - mean
    variance = np.sum(centered ** 2 * weights[:, None], axis=0)
    singular = np.linalg.svd(centered * np.sqrt(weights[:, None]), compute_uv=False)
    eigenvalues = singular ** 2
    energy = float(eigenvalues.sum())
    probability = eigenvalues / energy if energy > 0 else np.zeros_like(eigenvalues)
    positive = probability[probability > 0]
    rank = float(np.exp(-np.sum(positive * np.log(positive)))) if len(positive) else 0.0
    rms = float(np.sqrt(np.sum(weights * np.mean(values ** 2, axis=1))))
    mean_sd = float(np.sqrt(variance).mean())
    floor = relative_floor * max(rms, 1.0)
    return {
        "mean_channel_sd": mean_sd,
        "mean_feature_norm": float(np.sum(weights * np.linalg.norm(values, axis=1))),
        "effective_rank": rank,
        "near_constant": bool(mean_sd <= floor),
        "variation_floor": float(floor),
        "examples": int(len(values)),
        "sources": int(len(set(sources))),
    }


def _fit_training_rows(features: np.ndarray, targets: np.ndarray, sources: np.ndarray,
                       alpha: float) -> tuple[WeightedScaler, Ridge]:
    """This low-level fit receives training rows only, including during inner CV."""
    weights = source_weights(sources)
    scaler = WeightedScaler(center=True).fit(features, weights)
    regression = Ridge(alpha=float(alpha), fit_intercept=True)
    regression.fit(scaler.transform(features), targets, sample_weight=weights)
    return scaler, regression


def fit_source_readout(
    features: np.ndarray,
    targets: np.ndarray,
    source_ids: np.ndarray,
    *,
    train_sources: Sequence[str],
    test_sources: Sequence[str],
    alphas: Sequence[float] = (0.01, 0.1, 1.0, 10.0, 100.0),
    inner_folds: int = 3,
) -> SourceReadout:
    """Tune ridge using pooled inner held-out source predictions, then refit.

    Only the readout is selected here. The features can have been pretrained
    on all outer-training sources. Assessing an entire pretraining recipe with
    inner validation requires separately excluding inner validation sources
    from that candidate encoder's training as well.
    """
    values = np.asarray(features, dtype=np.float64)
    labels = np.asarray(targets, dtype=np.float64)
    sources = np.asarray(source_ids).astype(str)
    train, test = tuple(map(str, train_sources)), tuple(map(str, test_sources))
    if values.ndim != 2 or labels.shape != (len(values),) or sources.shape != labels.shape:
        raise ValueError("Features, targets, and source identifiers have incompatible shapes")
    if len(set(train)) != len(train) or len(set(test)) != len(test):
        raise ValueError("Declared source roles contain duplicates")
    if not train or not test or set(train) & set(test):
        raise ValueError("Training and test sources must be nonempty and disjoint")
    if set(sources) != set(train) | set(test):
        raise ValueError("Declared source roles must cover the supplied examples exactly")
    penalties = tuple(sorted(set(map(float, alphas))))
    if not penalties or not np.isfinite(penalties).all() or min(penalties) <= 0:
        raise ValueError("Ridge penalties must be finite and positive")
    if inner_folds < 2 or len(train) < 2:
        raise ValueError("Readout selection needs at least two inner folds and training sources")

    # The fitting functions below receive no outer-test features or labels.
    train_rows = np.flatnonzero(np.isin(sources, train))
    x, y, groups = values[train_rows], labels[train_rows], sources[train_rows]
    if not np.isfinite(y).all():
        raise ValueError("Training labels must be finite")
    partitions = np.array_split(np.array(sorted(train)), min(inner_folds, len(train)))
    validation_rows = []
    for alpha in penalties:
        predicted = np.full(len(y), np.nan)
        for validation_sources in partitions:
            valid_rows = np.isin(groups, validation_sources)
            fit_rows = ~valid_rows
            scaler, regression = _fit_training_rows(x[fit_rows], y[fit_rows], groups[fit_rows], alpha)
            predicted[valid_rows] = regression.predict(scaler.transform(x[valid_rows]))
        weights = source_weights(groups)
        validation_rows.append({
            "alpha": alpha,
            "source_balanced_r2": weighted_r2(y, predicted, weights),
            "source_balanced_mae": weighted_mae(y, predicted, weights),
            "source_balanced_mse": float(np.average((y - predicted) ** 2, weights=weights)),
            "clips": len(y), "sources": len(set(groups)),
        })
    validation = pd.DataFrame(validation_rows)
    # Pooled MSE has the same ordering as pooled R² when its denominator exists.
    # A tied score selects the smaller penalty, a deterministic declared rule.
    selected = float(validation.sort_values(["source_balanced_mse", "alpha"]).iloc[0].alpha)
    scaler, regression = _fit_training_rows(x, y, groups, selected)
    diagnostics = feature_diagnostics(scaler.transform(x), groups)
    return SourceReadout(scaler, regression, tuple(sorted(train)), selected, validation, diagnostics)


def encode_laterality_features(encoder: torch.nn.Module, dataset: LearningDataset,
                               *, batch_size: int = 16) -> tuple[np.ndarray, np.ndarray]:
    """Keep the established five paired, common-time landmark summaries.

    Rows lacking any valid four-step token remain unavailable. A placeholder
    vector is never passed through the encoder as an observed recording.
    """
    if batch_size < 1:
        raise ValueError("Feature batch size must be positive")
    device = next(encoder.parameters()).device
    patches = dataset.valid.reshape(len(dataset.xyz), encoder.segments,
                                    encoder.segment_length, 33).all(axis=2)
    available = patches.any(axis=(1, 2))
    output = np.full((len(dataset.xyz), 2 * len(PROBE_PAIRS) * encoder.embed_dim), np.nan)
    rows = np.flatnonzero(available)
    encoder.eval()
    with torch.no_grad():
        for start in range(0, len(rows), batch_size):
            selected = rows[start:start + batch_size]
            xyz = torch.as_tensor(np.where(dataset.valid[selected, ..., None],
                                            dataset.xyz[selected], 0),
                                  dtype=torch.float32, device=device)
            valid = torch.as_tensor(patches[selected], dtype=torch.bool, device=device)
            tokens = encoder(xyz, valid).reshape(len(selected), encoder.segments, 33, encoder.embed_dim)
            output[selected] = laterality_features(tokens.cpu().numpy(), patches[selected], PROBE_PAIRS)
    return output, available


def evaluate_frozen_representations(
    model: torch.nn.Module,
    initial_model: torch.nn.Module,
    dataset: LearningDataset,
    settings: Any,
    *,
    condition: str,
    seed: int | None = None,
    checkpoint: str = "final",
    alphas: Sequence[float] = (0.01, 0.1, 1.0, 10.0, 100.0),
    inner_folds: int = 3,
    observation_datasets: Mapping[str, LearningDataset] | None = None,
    comparison_id: str = "declared_masking_comparison",
) -> dict[str, Any]:
    """Fit clean-training readouts once and evaluate a common observation bank.

    Corruption changes only held-out inputs at evaluation. Every row retains
    its unaltered-recording target, including rows whose prediction is unavailable.
    """
    observations = {"unaltered": dataset, **dict(observation_datasets or {})}
    for name, other in observations.items():
        if not (np.array_equal(other.sequence_ids, dataset.sequence_ids)
                and np.array_equal(other.source_ids, dataset.source_ids)
                and np.array_equal(other.targets, dataset.targets)
                and other.train_sources == dataset.train_sources
                and other.test_sources == dataset.test_sources):
            raise ValueError(f"Observation condition {name!r} changes source roles or reference targets")
    encoders = {
        "pretrained_online": model.view_encoder,
        "pretrained_teacher": model.target_encoder,
        "initial_online": initial_model.view_encoder,
    }
    features: dict[str, np.ndarray] = {}
    for name, encoder in encoders.items():
        features[name], available = encode_laterality_features(encoder, dataset)
        if not available[dataset.train_rows].all():
            raise ValueError("Unaltered training data contain inputs unavailable to the encoder")
    features["direct_pose"] = raw_pose_features(dataset)
    readouts = {
        name: fit_source_readout(
            values, dataset.targets, dataset.source_ids,
            train_sources=dataset.train_sources, test_sources=dataset.test_sources,
            alphas=alphas, inner_folds=inner_folds,
        )
        for name, values in features.items()
    }
    training_mean = float(np.average(dataset.targets[dataset.train_rows],
                                    weights=source_weights(dataset.source_ids[dataset.train_rows])))
    rows = []
    test = dataset.test_rows
    for observation, altered in observations.items():
        patch_valid = altered.valid.reshape(len(altered.xyz), -1, settings.segment_length, 33).all(axis=2)
        available = patch_valid.any(axis=(1, 2))
        for representation in (*encoders, "direct_pose", "training_mean"):
            if representation in encoders:
                x, _ = encode_laterality_features(encoders[representation], altered)
            elif representation == "direct_pose":
                x = raw_pose_features(altered)
            prediction = np.full(len(test), np.nan)
            valid_test = available[test]
            if representation == "training_mean":
                prediction[valid_test] = training_mean
                selected_alpha = np.nan
            else:
                prediction[valid_test] = readouts[representation].predict(x[test[valid_test]])
                selected_alpha = readouts[representation].selected_alpha
            for offset, index in enumerate(test):
                rows.append({
                    "sequence_id": str(dataset.sequence_ids[index]),
                    "source_id": str(dataset.source_ids[index]),
                    "fold": int(dataset.fold),
                    "seed": int(settings.seed if seed is None else seed),
                    "condition": condition, "representation": representation,
                    "checkpoint": "initial" if representation == "initial_online" else checkpoint,
                    "observation": observation, "target": float(dataset.targets[index]),
                    "prediction": float(prediction[offset]),
                    "available": bool(available[index]),
                    "status": "available" if available[index] else "no_valid_input_token",
                    "valid_input_tokens": int(patch_valid[index].sum()),
                    "selected_alpha": float(selected_alpha), "synthetic": dataset.synthetic,
                    "cohort_digest": dataset.cohort_digest, "split_digest": dataset.split_digest,
                    "comparison_id": comparison_id,
                })
    return {"predictions": pd.DataFrame(rows), "readouts": readouts, "training_mean": training_mean}


def make_evaluation_mask_bank(valid_patch: np.ndarray, *, seed: int = 813,
                              interval_length: int = 1) -> dict[str, np.ndarray]:
    """Prespecify scattered and connected leg gaps on both anatomical sides.

    Each region contains knee, ankle, heel, and foot tip. Its edges are
    knee--ankle, ankle--heel, ankle--foot tip, and heel--foot tip. A missing
    region occupies consecutive interior time blocks. Scattered masks match
    the left-region realized count per example; right coverage is reported
    separately because existing observation gaps can change its count.
    """
    valid = np.asarray(valid_patch, dtype=bool)
    if valid.ndim != 3 or valid.shape[-1] != 33 or valid.shape[1] < 3:
        raise ValueError("Evaluation bank requires at least three time blocks and 33 landmarks")
    if not 1 <= interval_length <= valid.shape[1] - 2:
        raise ValueError("The evaluation interval must leave time blocks before and after it")
    begin = (valid.shape[1] - interval_length) // 2
    bank = {}
    for name, joints in (("left_leg_gap", (25, 27, 29, 31)), ("right_leg_gap", (26, 28, 30, 32))):
        mask = np.zeros_like(valid)
        mask[:, begin:begin + interval_length, list(joints)] = True
        bank[name] = mask & valid
    rng = np.random.default_rng(seed)
    scattered = np.zeros_like(valid)
    for row in range(len(valid)):
        count = int(bank["left_leg_gap"][row].sum())
        if count:
            chosen = rng.choice(np.flatnonzero(valid[row]), size=count, replace=False)
            scattered[row].flat[chosen] = True
    bank["scattered_gap"] = scattered
    for name, mask in bank.items():
        invalid_rows = (mask.sum(axis=(1, 2)) == 0) | ((valid & ~mask).sum(axis=(1, 2)) == 0)
        # An all-false row is explicitly infeasible, handled by the evaluation.
        bank[name][invalid_rows] = False
        bank[name].setflags(write=False)
    return bank


def prepared_observation_sensitivity(dataset: LearningDataset, hidden: np.ndarray,
                                     *, segment_length: int = 4) -> LearningDataset:
    """Remove prepared tokens; this does not test raw missing-data preparation."""
    mask = np.asarray(hidden)
    expected = (len(dataset.xyz), dataset.xyz.shape[1] // segment_length, 33)
    if mask.dtype != bool or mask.shape != expected:
        raise ValueError("Hidden mask must be boolean with one value per prepared token")
    withheld = np.repeat(mask, segment_length, axis=1)
    valid = dataset.valid & ~withheld
    return replace(dataset, xyz=np.where(valid[..., None], dataset.xyz, 0), valid=valid,
                   dataset_note="Prepared-coordinate representation sensitivity; original target retained")


def prepare_raw_missing_observations(
    raw_sequence: np.ndarray,
    frame_numbers: np.ndarray,
    fps: float,
    hidden_observations: np.ndarray,
    *,
    original_target: float,
    frames: int = 64,
    visibility_threshold: float = 0.5,
    max_interpolation_gap: int = 2,
    segment_length: int = 4,
) -> dict[str, Any]:
    """Withhold raw measurements before unchanged interpolation/normalization.

    The original target is a supplied reference from the unaltered recording.
    Only permitted raw observations enter ``prepare_pose``. Interpolation may
    fill a short removed gap using remaining endpoints; its output is therefore
    reported separately from the number of measurements deliberately removed.
    """
    raw = np.asarray(raw_sequence, dtype=np.float64)
    hidden = np.asarray(hidden_observations)
    if raw.ndim != 3 or raw.shape[1:] != (33, 4) or hidden.shape != raw.shape[:-1] or hidden.dtype != bool:
        raise ValueError("Expected raw [frames, 33, 4] and a boolean observation mask")
    if frames % segment_length or not np.isfinite(original_target):
        raise ValueError("Output length must support complete tokens and the reference target must be finite")
    originally_observed = observed_mask(raw, visibility_threshold)
    permitted = raw.copy()
    permitted[hidden, :3] = np.nan
    permitted[hidden, 3] = 0.0
    prepared = prepare_pose(permitted, frame_numbers, fps, frames=frames,
                            visibility_threshold=visibility_threshold,
                            max_interpolation_gap=max_interpolation_gap)
    patches = prepared.model_valid.reshape(-1, segment_length, 33).all(axis=1)
    return {
        "xyz": prepared.model_xyz, "valid": prepared.model_valid,
        "target": float(original_target), "available": bool(patches.any()),
        "naturally_missing_observations": int((~originally_observed).sum()),
        "deliberately_removed_observations": int((hidden & originally_observed).sum()),
        "remaining_raw_observations": int((originally_observed & ~hidden).sum()),
        "prepared_valid_tokens": int(patches.sum()),
        "input_preparation": "raw removal before interpolation and normalization",
    }


def predictor_diagnostics(
    model: torch.nn.Module,
    dataset: LearningDataset,
    mask_bank: Mapping[str, np.ndarray],
    *,
    condition: str,
    rows: np.ndarray | None = None,
    mismatch_seed: int = 991,
) -> pd.DataFrame:
    """Inspect the normal online→predictor→own-teacher pathway at fixed masks.

    Clips are evaluated individually so unequal hidden counts cannot be mixed
    by the original predictor's batch reshape. Mismatched targets come from a
    different evaluation source with valid measurements at the same positions;
    this is a diagnostic permutation, never a fitted training condition.
    """
    selected_rows = dataset.test_rows if rows is None else np.asarray(rows, dtype=int)
    if len(np.unique(selected_rows)) != len(selected_rows) or not len(selected_rows):
        raise ValueError("Diagnostic rows must be nonempty and unique")
    if np.any(selected_rows < 0) or np.any(selected_rows >= len(dataset.xyz)):
        raise ValueError("Diagnostic row index is out of range")
    device = next(model.parameters()).device
    segment_length = model.view_encoder.segment_length
    patches = dataset.valid.reshape(len(dataset.xyz), -1, segment_length, 33).all(axis=2)
    xyz = torch.as_tensor(np.where(dataset.valid[..., None], dataset.xyz, 0),
                          dtype=torch.float32, device=device)
    validity = torch.as_tensor(patches, dtype=torch.bool, device=device)
    model.eval()
    outputs = []
    with torch.no_grad():
        for name, values in mask_bank.items():
            mask = np.asarray(values)
            if mask.dtype != bool or mask.shape != patches.shape or (mask & ~patches).any():
                raise ValueError("Every evaluation target must be a valid token in the declared bank")
            rng = np.random.default_rng(mismatch_seed)
            predictions, targets, clip_sources = [], [], []
            token_predictions, token_targets, token_sources = [], [], []
            mismatched_errors, errors, energies, row_sources = [], [], [], []
            unavailable, mismatch_unavailable = 0, 0
            for row in selected_rows:
                hidden = mask[row]
                if not hidden.any() or not (patches[row] & ~hidden).any():
                    unavailable += 1
                    continue
                hidden_tensor = torch.as_tensor(hidden[None].copy(), dtype=torch.bool, device=device)
                predicted, target = model(xyz[row:row + 1], xyz[row:row + 1],
                                          validity[row:row + 1], hidden_tensor)
                p, t = predicted[0].cpu().numpy(), target[0].cpu().numpy()
                source = str(dataset.source_ids[row])
                predictions.append(p.mean(axis=0)); targets.append(t.mean(axis=0)); clip_sources.append(source)
                token_predictions.extend(p); token_targets.extend(t); token_sources.extend([source] * len(t))
                errors.append(float(np.mean((p - t) ** 2)))
                energies.append(float(np.mean(t ** 2)))
                row_sources.append(source)
                candidates = [int(other) for other in selected_rows
                              if dataset.source_ids[other] != source
                              and patches[other][hidden].all()]
                if candidates:
                    other_sources = sorted(set(map(str, dataset.source_ids[candidates])))
                    other_source = str(rng.choice(other_sources))
                    same_source = [other for other in candidates if str(dataset.source_ids[other]) == other_source]
                    other = int(rng.choice(same_source))
                    other_tokens = model.target_encoder(xyz[other:other + 1], validity[other:other + 1])
                    different = other_tokens[0, hidden.reshape(-1)].cpu().numpy()
                    mismatched_errors.append((source, float(np.mean((p - different) ** 2))))
                else:
                    mismatch_unavailable += 1
            base = {"condition": condition, "evaluation_mask": name,
                    "requested_clips": len(selected_rows), "evaluated_clips": len(errors),
                    "evaluated_sources": len(set(row_sources)), "unavailable_clips": unavailable,
                    "mismatched_target_unavailable_clips": mismatch_unavailable,
                    "error_scope": "own teacher; errors do not rank usefulness across teachers"}
            if not errors:
                outputs.append({**base, "status": "no_feasible_target_with_context"})
                continue
            weights = source_weights(np.asarray(row_sources))
            mse = float(np.average(errors, weights=weights))
            energy = float(np.average(energies, weights=weights))
            # Energy normalizes scale only. Low variation can still make a task easy.
            normalized = mse / energy if energy > 1e-12 else float("nan")
            row = {**base, "status": "evaluated", "feature_mse": mse,
                   "target_mean_square": energy, "normalized_error": normalized,
                   "normalized_error_denominator": "source-balanced mean squared teacher channel value",
                   "mismatched_target_mse": float("nan")}
            if mismatched_errors:
                mismatch_sources, mismatch_values = zip(*mismatched_errors)
                row["mismatched_target_mse"] = float(np.average(
                    mismatch_values, weights=source_weights(np.asarray(mismatch_sources))))
            for prefix, features, sources in (
                ("target_clip", targets, clip_sources), ("prediction_clip", predictions, clip_sources),
                ("target_token", token_targets, token_sources), ("prediction_token", token_predictions, token_sources),
            ):
                for key, value in feature_diagnostics(np.asarray(features), np.asarray(sources)).items():
                    row[f"{prefix}_{key}"] = value
            outputs.append(row)
    return pd.DataFrame(outputs)


_PREDICTION_KEYS = ["condition", "representation", "observation", "seed", "sequence_id"]
_GROUP_KEYS = ["condition", "representation", "observation", "seed"]


def validate_prediction_coverage(
    predictions: pd.DataFrame,
    expected: pd.DataFrame,
    *,
    seeds: Sequence[int],
    conditions: Sequence[str],
    representations: Sequence[str],
    observations: Sequence[str] = ("unaltered",),
    allow_partial: bool = False,
) -> pd.DataFrame:
    """Verify row identity, source roles, expected coverage, and paired outcomes."""
    needed = set(_PREDICTION_KEYS + ["source_id", "fold", "target", "prediction", "available"])
    if missing := needed - set(predictions.columns):
        raise ValueError(f"Prediction columns are missing: {sorted(missing)}")
    if predictions.empty or predictions.duplicated(_PREDICTION_KEYS).any():
        raise ValueError("Predictions are empty or contain duplicate clip predictions")
    identity = ["sequence_id", "source_id", "fold"]
    if not set(identity) <= set(expected.columns) or expected.empty or expected.sequence_id.duplicated().any():
        raise ValueError("Expected coverage must contain one unique row per clip")
    if expected.groupby("source_id").fold.nunique().max() != 1:
        raise ValueError("Expected outer folds overlap source videos")
    if predictions.groupby("source_id").fold.nunique().max() != 1:
        raise ValueError("Prediction outer folds overlap source videos")
    expected_identity = expected[identity].copy()
    bound = predictions.merge(expected_identity, on="sequence_id", how="left", suffixes=("", "_expected"), validate="many_to_one")
    if (bound.source_id != bound.source_id_expected).any() or (bound.fold != bound.fold_expected).any():
        raise ValueError("Predictions disagree with declared clip/source/fold identities")
    for column in ("cohort_digest", "split_digest", "comparison_id", "synthetic"):
        if column in predictions and predictions[column].nunique(dropna=False) != 1:
            raise ValueError(f"Incompatible paired predictions: {column} differs")
    if predictions.groupby("sequence_id").target.nunique(dropna=False).max() != 1:
        raise ValueError("Paired comparisons changed the target for a recording")
    if not np.isfinite(predictions.target).all():
        raise ValueError("Reference targets must be finite")
    available = predictions.available.to_numpy()
    if available.dtype != bool:
        raise ValueError("Availability must be boolean")
    if not np.isfinite(predictions.loc[available, "prediction"]).all():
        raise ValueError("Available predictions must be finite")
    if predictions.loc[~available, "prediction"].notna().any():
        raise ValueError("Unavailable predictions must remain missing")
    declarations = {
        "condition": tuple(conditions), "representation": tuple(representations),
        "observation": tuple(observations), "seed": tuple(map(int, seeds)),
    }
    for name, values in declarations.items():
        if not values or len(set(values)) != len(values):
            raise ValueError(f"Declare nonempty unique {name} values")
        if not set(predictions[name]) <= set(values):
            raise ValueError(f"Predictions contain undeclared {name} values")
    expected_clips = set(expected.sequence_id)
    complete = True
    for condition in conditions:
        for representation in representations:
            for observation in observations:
                for seed in seeds:
                    group = predictions[(predictions.condition == condition)
                                        & (predictions.representation == representation)
                                        & (predictions.observation == observation)
                                        & (predictions.seed == seed)]
                    if set(group.sequence_id) != expected_clips:
                        complete = False
                        if not allow_partial:
                            raise ValueError("Incomplete declared prediction coverage; explicitly label a partial run")
    output = predictions.copy()
    output.attrs["complete_declared_coverage"] = complete
    output.attrs["declared_clips"] = len(expected)
    output.attrs["declared_sources"] = expected.source_id.nunique()
    return output


def aggregate_predictions(predictions: pd.DataFrame, expected: pd.DataFrame, **declarations: Any) -> dict[str, pd.DataFrame]:
    """Pool out-of-fold clips per seed, score, then summarize seed scores.

    Missing predictions remain counted. Scores with missing observations are
    labeled as available-case descriptions and cannot conceal their coverage.
    """
    checked = validate_prediction_coverage(predictions, expected, **declarations)
    complete = checked.attrs["complete_declared_coverage"]
    rows = []
    for names, group in checked.groupby(_GROUP_KEYS, sort=True):
        available = group[group.available]
        weights = source_weights(available.source_id.to_numpy())
        values = dict(zip(_GROUP_KEYS, names))
        values.update({
            "r2": weighted_r2(available.target, available.prediction, weights) if len(available) else np.nan,
            "mae": weighted_mae(available.target, available.prediction, weights) if len(available) else np.nan,
            "evaluated_clips": len(available), "evaluated_sources": available.source_id.nunique(),
            "retained_rows": len(group), "unavailable_clips": int((~group.available).sum()),
            "declared_clips": len(expected), "declared_sources": expected.source_id.nunique(),
            "scope": ("complete declared comparison" if complete else "partial declared comparison")
                     + ("; available cases only" if not group.available.all() else ""),
        })
        rows.append(values)
    per_seed = pd.DataFrame(rows)
    summary = per_seed.groupby(_GROUP_KEYS[:-1], as_index=False).agg(
        mean_r2=("r2", "mean"), seed_sd_r2=("r2", "std"), mean_mae=("mae", "mean"),
        seed_sd_mae=("mae", "std"), evaluated_seeds=("seed", "nunique"),
        evaluated_clips_min=("evaluated_clips", "min"), evaluated_clips_max=("evaluated_clips", "max"),
        evaluated_sources_min=("evaluated_sources", "min"), unavailable_clips_max=("unavailable_clips", "max"),
        scope=("scope", lambda values: "; ".join(sorted(set(values)))),
    )
    return {"per_seed": per_seed, "summary": summary}


def paired_source_bootstrap(
    predictions: pd.DataFrame,
    *,
    first: str,
    reference: str,
    representation: str,
    observation: str = "unaltered",
    repetitions: int = 2000,
    seed: int = 812,
    metric: str = "r2",
) -> dict[str, Any]:
    """Resample whole sources jointly across conditions, clips, and seeds.

    The estimate is the mean of paired seed-specific metric differences. The
    interval conditions on already fitted models; no model is retrained here.
    Supply already coverage-validated predictions; this function also checks
    completeness across the two selected conditions and observed seed set.
    """
    if repetitions < 2 or metric not in {"r2", "mae"} or first == reference:
        raise ValueError("Specify two conditions, at least two resamples, and R² or MAE")
    table = predictions[(predictions.condition.isin([first, reference]))
                        & (predictions.representation == representation)
                        & (predictions.observation == observation)].copy()
    if table.empty:
        raise ValueError("No predictions match the requested paired comparison")
    expected = table[["sequence_id", "source_id", "fold"]].drop_duplicates()
    checked = validate_prediction_coverage(table, expected, seeds=sorted(table.seed.unique()),
                                          conditions=(first, reference), representations=(representation,),
                                          observations=(observation,))
    if not checked.available.all():
        raise ValueError("Paired bootstrap requires complete finite shared coverage")
    if "checkpoint" in checked and checked.groupby("condition").checkpoint.nunique().max() != 1:
        raise ValueError("A condition contains mixed checkpoints")
    keys = ["seed", "sequence_id", "source_id", "fold", "target"]
    joined = checked[checked.condition == first][keys + ["prediction"]].merge(
        checked[checked.condition == reference][keys + ["prediction"]], on=keys,
        suffixes=("_first", "_reference"), validate="one_to_one")
    sources = np.array(sorted(joined.source_id.unique()))
    if len(sources) < 2:
        raise ValueError("Source resampling needs at least two source videos")
    score = weighted_r2 if metric == "r2" else weighted_mae

    def difference(source_multiplicity: Mapping[str, int]) -> float:
        values = []
        for _, group in joined.groupby("seed"):
            base = source_weights(group.source_id.to_numpy())
            multiplicity = group.source_id.map(source_multiplicity).fillna(0).to_numpy()
            weights = base * multiplicity
            kept = weights > 0
            values.append(score(group.target.to_numpy()[kept], group.prediction_first.to_numpy()[kept], weights[kept])
                          - score(group.target.to_numpy()[kept], group.prediction_reference.to_numpy()[kept], weights[kept]))
        return float(np.mean(values))

    observed = difference({str(source): 1 for source in sources})
    rng = np.random.default_rng(seed)
    draws = []
    for _ in range(repetitions):
        sampled, counts = np.unique(rng.choice(sources, len(sources), replace=True), return_counts=True)
        draws.append(difference(dict(zip(sampled, counts))))
    values = np.asarray(draws)
    finite = np.isfinite(values)
    interval = np.quantile(values[finite], [0.025, 0.975]) if finite.any() else [np.nan, np.nan]
    return {
        "difference": observed, "lower_95": float(interval[0]), "upper_95": float(interval[1]),
        "metric": metric, "subtraction": f"{first} minus {reference}",
        "sources": len(sources), "clips_per_seed": expected.sequence_id.nunique(),
        "seeds": joined.seed.nunique(), "requested_resamples": repetitions,
        "finite_resamples": int(finite.sum()),
        "uncertainty_scope": "source resampling conditional on fitted models; excludes retraining uncertainty",
    }
