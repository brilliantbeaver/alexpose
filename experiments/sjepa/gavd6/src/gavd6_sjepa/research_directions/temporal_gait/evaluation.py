"""Frozen observable readouts and development-only model comparison.

Endpoint targets are detector observations, not diagnoses. All fitted predictors
use train-role sources; ridge penalties use one connected-group train holdout.
Development chooses the reporting baseline/model, never fits their parameters.
Test/calibration can only replay serialized fitted predictors. The observed-
future decoder is a privileged diagnostic and cannot win a forecasting gate.
"""
from __future__ import annotations

from collections import defaultdict
import hashlib
import json
from pathlib import Path

import numpy as np

from .statistics import (aggregate_window_errors, paired_group_bootstrap,
                         primary_window_errors, secondary_motion_errors)
from .contracts import atomic_json, atomic_npz


CONTEXT_KEYS = ("context", "context_valid", "context_times", "context_age", "query_times")
BASELINES = ("persistence", "robust_velocity", "periodic", "static_train_mean",
             "raw_pose_ridge", "shuffled_past_ridge", "support_ridge", "direct_mlp")
STATES = ("initialized", "online", "teacher")
PRIMARY_HORIZON = .5


def _json(path, value):
    path = Path(path)
    if path.exists():
        raise FileExistsError(f"Refusing to overwrite immutable evaluation artifact: {path}")
    atomic_json(path, value)


def _sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _contract(cfg):
    return {key: getattr(cfg, key) for key in
            ("schema_version", "horizons", "prefix_seconds", "grid_hz",
             "max_observation_age_seconds", "endpoint_tolerance_seconds")}


def _role(dataset):
    roles = {r.get("role") for r in dataset.records}
    if len(roles) != 1 or next(iter(roles)) not in {"train", "development", "calibration", "test"}:
        raise ValueError("Dataset must be nonempty and have one explicit valid role")
    if len(dataset.records) != len(dataset.arrays["context"]):
        raise ValueError("Array/record length mismatch")
    return next(iter(roles))


def _partitions(train, evaluation, fitted=None):
    role = _role(evaluation)
    if role == "train":
        raise ValueError("Reported evaluation cannot use train-role windows")
    if fitted is None:
        if role != "development":
            raise ValueError("Test/calibration require frozen fitted_path; fitting is prohibited")
        if train is None or _role(train) != "train":
            raise ValueError("All fitting windows must have train role")
        train_groups = {str(r["group_id"]) for r in train.records}
        train_videos = {str(r["video_id"]) for r in train.records}
        train_windows = {str(r["window_id"]) for r in train.records}
        if len(train_groups) < 2:
            raise ValueError("Train-grouped ridge selection needs at least two connected groups")
    else:
        if fitted.get("train_role") != "train":
            raise ValueError("Frozen readout lacks train-only fitting provenance")
        train_groups, train_videos, train_windows = (set(fitted[key]) for key in
            ("train_groups", "train_videos", "train_windows"))
    for key, identities in (("group_id", train_groups), ("video_id", train_videos),
                            ("window_id", train_windows)):
        if identities & {str(r[key]) for r in evaluation.records}:
            raise ValueError(f"Train/evaluation contamination in {key}")
    modes = {r.get("mode") for r in evaluation.records}
    if len(modes) != 1:
        raise ValueError("Mixed synthetic/real records are prohibited")
    return role


def _causal_arrays(dataset):
    arrays = {key: np.asarray(dataset.arrays[key]) for key in CONTEXT_KEYS}
    x, valid, times = arrays["context"], arrays["context_valid"], arrays["context_times"]
    if x.ndim != 4 or x.shape[-2:] != (33, 2) or valid.shape != x.shape[:-1] or valid.dtype != bool:
        raise ValueError("Malformed causal context")
    if (times.shape != valid.shape or arrays["context_age"].shape != valid.shape
            or arrays["query_times"].shape != x.shape[:2]):
        raise ValueError("Malformed causal timing channels")
    if not np.isfinite(arrays["query_times"]).all() or (arrays["query_times"] >= 0).any():
        raise ValueError("Prediction inputs must precede the issue time")
    if not np.isfinite(x[valid]).all() or not np.isfinite(times[valid]).all():
        raise ValueError("Nonfinite observed context")
    if (times[valid] >= 0).any() or (times > arrays["query_times"][:, :, None] + 1e-6)[valid].any():
        raise ValueError("Future observation entered causal context")
    if not np.isfinite(arrays["context_age"][valid]).all() or (arrays["context_age"][valid] < 0).any():
        raise ValueError("Invalid observation ages")
    return arrays


def _history(arrays, row, joint):
    valid = arrays["context_valid"][row, :, joint]
    times = arrays["context_times"][row, valid, joint].astype(float)
    xy = arrays["context"][row, valid, joint].astype(float)
    if not len(times):
        return times, xy
    # Held observations are one observation, not many independent measurements.
    order = np.argsort(times, kind="stable")
    times, xy = times[order], xy[order]
    _, unique = np.unique(times, return_index=True)
    return times[unique], xy[unique]


def kinematic_predictions(dataset, horizons):
    """Past-only persistence, actual-time Theil–Sen velocity, harmonic forecast.

    Velocity uses the final 0.8 seconds and median pairwise slopes. Harmonic
    frequency selection uses prefix residuals only on a fixed 0.5–3Hz grid,
    with intercept, linear trend and two harmonics. Insufficient temporal span
    falls back to velocity; unavailable joint history stays explicitly NaN.
    """
    arrays = _causal_arrays(dataset)
    n, h = len(dataset.records), len(horizons)
    outputs = {key: np.full((n, h, 33, 2), np.nan) for key in BASELINES[:3]}
    query = np.asarray(horizons, float)
    for row in range(n):
        for joint in range(33):
            times, xy = _history(arrays, row, joint)
            if not len(times):
                continue
            outputs["persistence"][row, :, joint] = xy[-1]
            recent = times >= times[-1] - .8
            t, p = times[recent], xy[recent]
            velocity = np.zeros(2)
            if len(t) > 1:
                i, j = np.triu_indices(len(t), 1)
                velocity = np.median((p[j] - p[i]) / (t[j] - t[i])[:, None], axis=0)
            predicted = xy[-1] + (query - times[-1])[:, None] * velocity
            outputs["robust_velocity"][row, :, joint] = predicted
            outputs["periodic"][row, :, joint] = predicted
            if len(times) < 12 or times[-1] - times[0] < 1.2:
                continue
            best = None
            for frequency in np.linspace(.5, 3., 26):
                def design(z):
                    phase = 2 * np.pi * frequency * z
                    return np.column_stack([np.ones(len(z)), z, np.sin(phase), np.cos(phase),
                                            np.sin(2 * phase), np.cos(2 * phase)])
                matrix = design(times)
                penalty = np.diag([0., 1e-3, 1e-3, 1e-3, 1e-3, 1e-3])
                coef = np.linalg.solve(matrix.T @ matrix + penalty, matrix.T @ xy)
                residual = float(np.mean((matrix @ coef - xy) ** 2))
                if best is None or residual < best[0]:
                    best = residual, design(query) @ coef
            outputs["periodic"][row, :, joint] = best[1]
    return outputs


def _raw_features(dataset, *, support_only=False, shuffle_seed=None):
    a = _causal_arrays(dataset)
    valid = a["context_valid"]
    channels = [valid[..., None].astype(np.float32),
                np.where(valid, a["context_times"], 0)[..., None],
                np.where(valid, a["context_age"], 0)[..., None]]
    if not support_only:
        channels.insert(0, np.where(valid[..., None], a["context"], 0))
    features = np.concatenate(channels, -1)
    if shuffle_seed is not None:
        features = features.copy()
        for i, record in enumerate(dataset.records):
            token = f"{shuffle_seed}:{record['window_id']}".encode()
            seed = int.from_bytes(hashlib.sha256(token).digest()[:8], "little")
            # Shuffle pose positions, retaining the supplied query/validity clocks.
            # This is a deliberate order-destroyed diagnostic, not a new dataset.
            order = np.random.default_rng(seed).permutation(features.shape[1])
            features[i, :, :, :2] = features[i, order, :, :2]
    return features.reshape(len(features), -1).astype(np.float32)


def _weights(records):
    hierarchy = defaultdict(lambda: defaultdict(list))
    for i, row in enumerate(records):
        hierarchy[str(row["video_id"])][str(row["sequence_id"])].append(i)
    result = np.zeros(len(records), float)
    for bouts in hierarchy.values():
        for indices in bouts.values():
            result[indices] = 1 / (len(hierarchy) * len(bouts) * len(indices))
    return result


def _fit_projection(features, seed):
    """Train-only standardization and at most 128 train-fitted SVD directions."""
    from sklearn.utils.extmath import randomized_svd
    x = np.asarray(features, np.float32)
    mean = x.mean(0)
    sd = x.std(0)
    active = sd > 1e-6
    if not active.any():
        active[0] = True
    sd = np.where(sd > 1e-6, sd, 1)
    standardized = (x[:, active] - mean[active]) / sd[active]
    rank = min(128, max(len(x) - 1, 1), standardized.shape[1])
    if standardized.shape[1] <= rank:
        basis = np.eye(standardized.shape[1], dtype=np.float32)
    else:
        _, _, right = randomized_svd(standardized, n_components=rank, n_iter=3,
                                      random_state=int(seed), flip_sign=True)
        basis = right.T.astype(np.float32)
    return {"mean": mean, "sd": sd, "active": active, "basis": basis,
            "recipe": "all causal raw channels; train-only standardization and SVD capped at 128"}


def _project(features, projection):
    active = projection["active"]
    return (((features[:, active] - projection["mean"][active]) /
              projection["sd"][active]) @ projection["basis"]).astype(np.float64)


def _targets(dataset):
    y = np.asarray(dataset.arrays["endpoint"], float)
    valid = np.asarray(dataset.arrays["endpoint_valid"], bool)
    scale = np.asarray(dataset.arrays["scale_valid"], bool)
    if y.ndim != 4 or y.shape[-2:] != (33, 2) or valid.shape != y.shape[:-1]:
        raise ValueError("Malformed endpoint target arrays")
    valid = valid & scale[:, None, None]
    if not np.isfinite(y[valid]).all():
        raise ValueError("Nonfinite observed train endpoint")
    return np.where(valid[..., None], y, 0), valid


def _ridge_path(x, y, valid, records, alphas):
    n, h, _, _ = y.shape
    d = x.shape[-1]
    models = [{"coefficient": np.full((h, d, 66), np.nan),
               "intercept": np.full((h, 66), np.nan), "alpha": float(alpha)} for alpha in alphas]
    for horizon in range(h):
        masks = defaultdict(list)
        for joint in range(33):
            masks[valid[:, horizon, joint].tobytes()].append(joint)
        for joints in masks.values():
            selected = valid[:, horizon, joints[0]]
            if selected.sum() < 2:
                continue
            target_indices = np.array([[2 * j, 2 * j + 1] for j in joints]).ravel()
            values = y[:, horizon].reshape(n, 66)[selected][:, target_indices]
            selected_records = [r for r, keep in zip(records, selected) if keep]
            w = _weights(selected_records)
            w /= w.mean()
            xm, ym = np.average(x[selected], axis=0, weights=w), np.average(values, axis=0, weights=w)
            xc, yc = x[selected] - xm, values - ym
            eig, directions = np.linalg.eigh((xc * w[:, None]).T @ xc)
            projected = directions.T @ ((xc * w[:, None]).T @ yc)
            for model in models:
                coefficient = directions @ (projected / (np.maximum(eig, 0) + model["alpha"])[:, None])
                model["coefficient"][horizon][:, target_indices] = coefficient
                model["intercept"][horizon, target_indices] = ym - xm @ coefficient
    return models


def _ridge_predict(features, fitted):
    x = _project(features, fitted["projection"])
    model = fitted["model"]
    return np.stack([x @ coef + intercept for coef, intercept in
                    zip(model["coefficient"], model["intercept"])], 1).reshape(len(x), -1, 33, 2)


def _fit_ridge(cfg, features, dataset, seed, *, selection_horizon=PRIMARY_HORIZON):
    if _role(dataset) != "train":
        raise ValueError("Readout fit requires train role")
    groups = sorted({str(r["group_id"]) for r in dataset.records})
    if len(groups) < 2:
        raise ValueError("Ridge selection requires at least two training groups")
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(groups))
    held_groups = {groups[i] for i in order[:max(1, len(groups) // 5)]}
    held = np.array([str(r["group_id"]) in held_groups for r in dataset.records])
    fit = ~held
    y, valid = _targets(dataset)
    projection = _fit_projection(features[fit], seed)
    train_x, held_x = _project(features[fit], projection), _project(features[held], projection)
    fit_records = [r for r, keep in zip(dataset.records, fit) if keep]
    held_records = [r for r, keep in zip(dataset.records, held) if keep]
    candidates = _ridge_path(train_x, y[fit], valid[fit], fit_records, cfg.ridge_alphas)
    errors = []
    h_primary = list(cfg.horizons).index(selection_horizon)
    for model in candidates:
        pred = np.stack([held_x @ c + b for c, b in zip(model["coefficient"], model["intercept"])], 1)
        pred = pred.reshape(len(held_records), -1, 33, 2)
        measured = primary_window_errors(pred, y[held], valid[held], np.ones(held.sum(), bool))
        summary = aggregate_window_errors(held_records, measured, cfg.horizons)["summary"][h_primary]
        errors.append(summary["score"])
    available = [i for i, error in enumerate(errors) if error is not None]
    pair_counts = sum(valid[:, :, left] & valid[:, :, right]
                      for left, right in ((25, 26), (27, 28), (29, 30), (31, 32)))
    eligible = pair_counts[:, h_primary] >= 3
    selection = {"role": "train_group_holdout", "held_groups": sorted(held_groups),
                 "ssl_exposure": "inner held groups remain SSL-train-visible; readout-only selection, not fully nested SSL",
                 "alphas": list(cfg.ridge_alphas), "primary_errors": errors,
                 "selection_horizon_seconds": float(selection_horizon),
                 "support": {"observed_joint_endpoints": int(valid[:, h_primary].sum()),
                             "eligible_training_windows": int(eligible.sum()),
                             "eligible_inner_fit_windows": int(eligible[fit].sum()),
                             "eligible_inner_validation_windows": int(eligible[held].sum())}}
    if not available:
        if selection_horizon != PRIMARY_HORIZON:
            return {"kind": "not_estimable",
                    "reason": "No complete observed support on the frozen training-group validation at this secondary horizon",
                    "selection": {**selection, "selected_alpha": None, "boundary": None}}
        raise ValueError("Train-grouped ridge validation has no complete observed primary support")
    best = min(available, key=lambda i: (errors[i], -float(cfg.ridge_alphas[i])))
    final_projection = _fit_projection(features, seed)
    final_model = _ridge_path(_project(features, final_projection), y, valid,
                              dataset.records, [cfg.ridge_alphas[best]])[0]
    return {"kind": "ridge", "projection": final_projection, "model": final_model,
            "selection": {**selection, "selected_alpha": float(cfg.ridge_alphas[best]),
                          "boundary": best in {0, len(cfg.ridge_alphas) - 1}}}


def _fit_mlp(cfg, features, dataset, seed):
    import torch
    from torch import nn
    if _role(dataset) != "train":
        raise ValueError("Direct predictor fit requires train role")
    projection = _fit_projection(features, seed)
    x = _project(features, projection).astype(np.float32)
    y, valid = _targets(dataset)
    usable = valid.any((1, 2))
    if not usable.any():
        raise ValueError("No observed endpoints to fit direct predictor")
    weights = np.zeros(len(dataset.records), float)
    weights[usable] = _weights([r for r, keep in zip(dataset.records, usable) if keep])
    device = torch.device(cfg.device)
    torch.manual_seed(int(seed))
    width = 96
    model = nn.Sequential(nn.Linear(x.shape[1], width), nn.GELU(), nn.Linear(width, y.shape[1] * 66)).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    rng = np.random.default_rng(seed)
    losses = []
    for _ in range(int(cfg.direct_updates)):
        index = rng.choice(len(x), int(cfg.batch_size), replace=True, p=weights)
        tx = torch.from_numpy(x[index]).to(device)
        ty = torch.from_numpy(y[index].astype(np.float32)).to(device)
        mask = torch.from_numpy(valid[index]).to(device)
        prediction = model(tx).reshape_as(ty)
        squared = ((prediction - ty) ** 2).sum(-1)
        loss = ((squared * mask).sum((1, 2)) / mask.sum((1, 2)).clamp_min(1)).mean()
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        losses.append(float(loss.detach().cpu()))
    state = {name: value.detach().cpu().numpy().copy() for name, value in model.state_dict().items()}
    return {"kind": "mlp", "projection": projection, "weights": state, "hidden_dim": width,
            "horizons": len(cfg.horizons), "target_support": valid.any(0),
            "training": {"role": "train", "updates": int(cfg.direct_updates), "learning_rate": .001,
                         "weight_decay": .0001, "loss": "observed-coordinate MSE",
                         "sampling": "equal videos, then bouts, then windows with observed targets",
                         "loss_trace": losses, "manual_annotation_efficiency": False}}


def _mlp_predict(features, fitted):
    import torch
    from torch import nn
    x = _project(features, fitted["projection"]).astype(np.float32)
    model = nn.Sequential(nn.Linear(x.shape[1], fitted["hidden_dim"]), nn.GELU(),
                          nn.Linear(fitted["hidden_dim"], fitted["horizons"] * 66))
    model.load_state_dict({key: torch.from_numpy(value.copy()) for key, value in fitted["weights"].items()})
    model.eval()
    outputs = []
    with torch.no_grad():
        for start in range(0, len(x), 256):
            outputs.append(model(torch.from_numpy(x[start:start + 256])).numpy())
    prediction = np.concatenate(outputs).reshape(len(x), fitted["horizons"], 33, 2)
    return np.where(fitted["target_support"][None, ..., None], prediction, np.nan)


def _pack_fitted(path, fitted):
    arrays = {}
    def pack(value):
        if isinstance(value, np.ndarray):
            name = "array_" + str(len(arrays))
            arrays[name] = value
            return {"__ndarray__": name}
        if isinstance(value, dict):
            return {key: pack(v) for key, v in value.items()}
        if isinstance(value, (list, tuple)):
            return [pack(v) for v in value]
        return value
    tree = pack(fitted)
    path = Path(path)
    npz_path = path.with_suffix(".npz")
    if path.exists() or npz_path.exists():
        raise FileExistsError("Fitted predictor artifact already exists")
    atomic_npz(npz_path, **arrays)
    _json(path, {"arrays": npz_path.name, "arrays_sha256": _sha(npz_path), "tree": tree})


def _unpack_fitted(path):
    path = Path(path)
    stored = json.loads(path.read_text())
    npz_path = path.parent / stored["arrays"]
    if _sha(npz_path) != stored["arrays_sha256"]:
        raise ValueError("Fitted-predictor array hash mismatch")
    with np.load(npz_path, allow_pickle=False) as archive:
        def unpack(value):
            if isinstance(value, dict) and set(value) == {"__ndarray__"}:
                return archive[value["__ndarray__"]].copy()
            if isinstance(value, dict):
                return {key: unpack(v) for key, v in value.items()}
            if isinstance(value, list):
                return [unpack(v) for v in value]
            return value
        return unpack(stored["tree"])


def _features_from_model(cfg, task, model, dataset):
    import torch
    from .training import context_batch
    collected = defaultdict(list)
    model.eval().float()
    with torch.no_grad():
        for start in range(0, len(dataset.records), int(cfg.batch_size)):
            indices = np.arange(start, min(start + int(cfg.batch_size), len(dataset.records)))
            context = context_batch(dataset, indices, cfg.device, arm=task["arm"])
            for state in STATES:
                collected[state + "_context_ridge"].append(model.encode(context, state).float().cpu().numpy())
                collected[state + "_future_ridge"].append(model.predict_future(context, state).float().cpu().numpy())
            # This is deliberately separate from the context-only inference path.
            future = torch.from_numpy(dataset.arrays["future"][indices]).to(cfg.device)
            valid = torch.from_numpy(dataset.arrays["future_valid"][indices]).to(cfg.device)
            collected["observed_future_teacher_ridge"].append(
                model.encode_future_targets(future, valid, "teacher").float().cpu().numpy())
    # A horizon-specific decoder sees only its own predicted feature, not other
    # observed future intervals. Context features are repeated across horizons.
    return {key: np.concatenate(values) for key, values in collected.items()}


def _decoder_horizon_support(cfg, readouts):
    metadata = []
    for h, (horizon, readout) in enumerate(zip(cfg.horizons, readouts)):
        if readout["kind"] == "not_estimable":
            fitted_joints = 0
        else:
            model = readout["model"]
            fitted_coordinates = (np.isfinite(model["coefficient"][h]).all(0)
                                  & np.isfinite(model["intercept"][h]))
            fitted_joints = int(fitted_coordinates.reshape(33, 2).all(-1).sum())
        metadata.append({"horizon_seconds": float(horizon), "fitted_joint_count": fitted_joints,
                         "status": "not_estimable" if not fitted_joints else
                                   "estimable" if fitted_joints == 33 else "partial_joint_support",
                         "selection_horizon_seconds": readout["selection"]["selection_horizon_seconds"],
                         "reason": readout.get("reason")})
    return metadata


def _fit_feature_decoder(cfg, features, dataset, seed):
    if features.ndim == 2:
        readout = _fit_ridge(cfg, features, dataset, seed)
        return {"kind": "shared_context", "readout": readout,
                "horizon_support": _decoder_horizon_support(cfg, [readout] * len(cfg.horizons))}
    if features.ndim != 3 or features.shape[1] != len(cfg.horizons):
        raise ValueError("Feature horizon grid mismatch")
    # Each readout has all target heads for a uniform fitting interface; only its
    # matched horizon is used. This does not expose future observations to it.
    readouts = [
        _fit_ridge(cfg, features[:, h], dataset, seed, selection_horizon=cfg.horizons[h])
        for h in range(features.shape[1])]
    return {"kind": "horizon_specific", "readouts": readouts,
            "horizon_support": _decoder_horizon_support(cfg, readouts)}


def _predict_feature_decoder(features, fitted):
    if fitted["kind"] == "shared_context":
        return _ridge_predict(features, fitted["readout"])
    if (fitted["kind"] != "horizon_specific" or features.ndim != 3
            or features.shape[1] != len(fitted["readouts"])):
        raise ValueError("Frozen decoder feature horizon grid mismatch")
    return np.stack([np.full((len(features), 33, 2), np.nan) if readout["kind"] == "not_estimable"
                     else _ridge_predict(features[:, h], readout)[:, h]
                     for h, readout in enumerate(fitted["readouts"])], 1)


def _fit_baselines(cfg, dataset, seed):
    raw = _raw_features(dataset)
    y, valid = _targets(dataset)
    weights = _weights(dataset.records)[:, None, None]
    denominator = (weights * valid).sum(0)
    mean = (y * (weights * valid)[..., None]).sum(0) / np.maximum(denominator[..., None], 1e-15)
    mean[denominator == 0] = np.nan
    return {"raw_pose_ridge": _fit_ridge(cfg, raw, dataset, seed),
            "shuffled_past_ridge": _fit_ridge(cfg, _raw_features(dataset, shuffle_seed=seed), dataset, seed),
            "support_ridge": _fit_ridge(cfg, _raw_features(dataset, support_only=True), dataset, seed),
            "direct_mlp": _fit_mlp(cfg, raw, dataset, seed), "static_train_mean": mean}


def _predict_baselines(cfg, dataset, fitted, seed):
    predictions = kinematic_predictions(dataset, cfg.horizons)
    fallback_counts = {}
    for name, prediction in predictions.items():
        missing = ~np.isfinite(prediction).all(-1)
        # Missing PREFIX history is the only fallback selector. Future target
        # validity/coordinates never select an input or imputed forecast.
        predictions[name] = np.where(missing[..., None], fitted["static_train_mean"][None], prediction)
        fallback_counts[name] = missing
    raw = _raw_features(dataset)
    predictions.update({
        "raw_pose_ridge": _ridge_predict(raw, fitted["raw_pose_ridge"]),
        "shuffled_past_ridge": _ridge_predict(_raw_features(dataset, shuffle_seed=seed), fitted["shuffled_past_ridge"]),
        "support_ridge": _ridge_predict(_raw_features(dataset, support_only=True), fitted["support_ridge"]),
        "direct_mlp": _mlp_predict(raw, fitted["direct_mlp"]),
        "static_train_mean": np.broadcast_to(fitted["static_train_mean"], dataset.arrays["endpoint"].shape).copy()})
    return predictions, fallback_counts


def _support_hash(dataset):
    digest = hashlib.sha256()
    digest.update(json.dumps(dataset.records, sort_keys=True, allow_nan=False).encode())
    for key in ("endpoint", "endpoint_valid", "endpoint_times", "scale_valid"):
        value = np.ascontiguousarray(dataset.arrays[key])
        digest.update(key.encode())
        digest.update(value.dtype.str.encode())
        digest.update(value.tobytes())
    return digest.hexdigest()


def _evaluate(cfg, task, train, evaluation, checkpoint_path, output_dir, fitted_path):
    role = _role(evaluation)
    # This assertion precedes model loading, target reading or any optimizer call.
    if role in {"test", "calibration"} and fitted_path is None:
        raise ValueError("Test/calibration require frozen fitted_path; no refitting")
    seed, arm = int(task["seed"]), task["arm"]
    reuse_fitted = fitted_path is not None
    fitted = _unpack_fitted(fitted_path) if fitted_path is not None else None
    _partitions(train, evaluation, fitted)
    if any(r.get("mode") != cfg.mode for r in evaluation.records):
        raise ValueError("Evaluation mode and record mode disagree")
    checksum = _sha(checkpoint_path) if checkpoint_path is not None else None
    if fitted is not None:
        if (fitted["contract"] != _contract(cfg) or fitted["seed"] != seed or fitted["arm"] != arm
                or fitted["checkpoint_sha256"] != checksum):
            raise ValueError("Frozen readout is incompatible with config/seed/arm/checkpoint")
    directory = Path(output_dir).resolve()
    directory.mkdir(parents=True, exist_ok=True)
    if any(directory.iterdir()):
        raise FileExistsError(f"Evaluation output must be a new/empty directory: {directory}")
    model = None
    if checkpoint_path is not None:
        from .training import load_model, dataset_fingerprint
        model = load_model(cfg, checkpoint_path)
        metadata = model.checkpoint_metadata
        if any(metadata["task"].get(key) != task.get(key) for key in ("task_id", "arm", "seed", "fold")):
            raise ValueError("Checkpoint task identity differs from requested readout condition")
        if fitted is None and metadata["signature"]["dataset"] != dataset_fingerprint(train):
            raise ValueError("Readout fitting sources/arrays differ from SSL training dataset")
    if fitted is None:
        if any(r.get("mode") != cfg.mode for r in train.records):
            raise ValueError("Train mode and config mode disagree")
        fitted = {"contract": _contract(cfg), "seed": seed, "arm": arm,
                  "checkpoint_sha256": checksum, "train_role": "train",
                  "train_groups": sorted({str(r["group_id"]) for r in train.records}),
                  "train_videos": sorted({str(r["video_id"]) for r in train.records}),
                  "train_windows": sorted({str(r["window_id"]) for r in train.records}),
                  "train_target_support_sha256": _support_hash(train),
                  "baselines": _fit_baselines(cfg, train, seed), "decoders": {}}
        if model is not None:
            features = _features_from_model(cfg, task, model, train)
            fitted["decoders"] = {name: _fit_feature_decoder(cfg, values, train, seed)
                                  for name, values in features.items()}
        fitted_path = directory / "fitted.json"
        _pack_fitted(fitted_path, fitted)
    predictions, fallbacks = _predict_baselines(cfg, evaluation, fitted["baselines"], seed)
    if model is not None:
        features = _features_from_model(cfg, task, model, evaluation)
        if set(features) != set(fitted["decoders"]):
            raise ValueError("Frozen decoder feature states differ")
        predictions.update({name: _predict_feature_decoder(values, fitted["decoders"][name])
                            for name, values in features.items()})
    sources, summaries = [], []
    secondary_scores, secondary_arrays = {}, {}
    support = None
    for name, predicted in predictions.items():
        measured = primary_window_errors(predicted, evaluation.arrays["endpoint"],
                    evaluation.arrays["endpoint_valid"], evaluation.arrays["scale_valid"])
        support = measured["joint_support"]
        values = aggregate_window_errors(evaluation.records, measured, cfg.horizons, method=name, seed=seed)
        sources.extend(values["sources"])
        summaries.extend(values["summary"])
        secondary = secondary_motion_errors(predicted, evaluation.arrays["endpoint"],
                    evaluation.arrays["endpoint_valid"], evaluation.arrays["scale_valid"])
        for metric, secondary_measured in secondary.items():
            secondary_values = aggregate_window_errors(evaluation.records, secondary_measured,
                                    cfg.horizons, method=name, seed=seed)
            bucket = secondary_scores.setdefault(metric, {"sources": [], "summary": [],
                "secondary_only": True, "used_for_selection_or_gate": False,
                "support_rule": "primary observed >=3 bilateral pairs and valid prefix scale, plus both observed hips 23/24"})
            bucket["sources"].extend(secondary_values["sources"])
            bucket["summary"].extend(secondary_values["summary"])
            for key, value in secondary_measured.items():
                secondary_arrays[f"secondary__{metric}__{name}__{key}"] = value
    prediction_path = directory / "predictions.npz"
    atomic_npz(prediction_path, **{**predictions, "joint_support": support, **secondary_arrays,
                        **{name + "_prefix_mean_fallback": mask for name, mask in fallbacks.items()},
                        **{key: evaluation.arrays[key] for key in
                           ("endpoint", "endpoint_valid", "endpoint_times", "scale_valid")}})
    record_path, score_path = directory / "records.json", directory / "scores.json"
    decoder_support = {name: decoder.get("horizon_support", []) for name, decoder in fitted["decoders"].items()}
    _json(record_path, {"records": evaluation.records, "horizons": list(cfg.horizons),
                       "methods": list(predictions), "privileged_diagnostics": ["observed_future_teacher_ridge"],
                       "decoder_horizon_support": decoder_support,
                       "secondary_metrics": list(secondary_scores),
                       "prefix_mean_fallbacks": {name: int(mask.sum()) for name, mask in fallbacks.items()},
                       "support_sha256": _support_hash(evaluation), "role": role, "mode": cfg.mode})
    _json(score_path, {"sources": sources, "summary": summaries, "secondary": secondary_scores,
                      "unit": "Euclidean projected-image error / prefix-only projected body length",
                      "clinical_endpoint": False, "manual_annotation_efficiency": False})
    from .plots import plot_scores
    plot_path = plot_scores([r for r in sources if np.isclose(r["horizon_seconds"], PRIMARY_HORIZON)],
                            directory / "primary-source-errors.png")
    result = {"status": "evaluated", "role": role, "mode": cfg.mode, "seed": seed, "arm": arm,
              "task_id": task.get("task_id", f"{arm}-s{seed}"), "fold": int(task.get("fold", 0)),
              "checkpoint_path": str(Path(checkpoint_path).resolve()) if checkpoint_path else None,
              "checkpoint_sha256": checksum, "fitted_path": str(Path(fitted_path).resolve()),
              "fitted_sha256": _sha(fitted_path), "fitted_reused": reuse_fitted,
              "predictions_path": str(prediction_path), "predictions_sha256": _sha(prediction_path),
              "records_path": str(record_path), "scores_path": str(score_path),
              "records_sha256": _sha(record_path), "scores_sha256": _sha(score_path),
              "support_sha256": _support_hash(evaluation), "methods": list(predictions),
              "decoder_horizon_support": decoder_support,
              "summary": summaries,
              "secondary_summary": {metric: values["summary"] for metric, values in secondary_scores.items()},
              "plot_path": plot_path, "artifacts": {},
              "feature_spaces": {"context": "validity-weighted pooled final LayerNorm",
                                 "future": "fixed-horizon predicted feature; not target coordinates",
                                 "teacher_future": "EMA encoder plus online predictor",
                                 "observed_future_teacher": "privileged observed-future diagnostic; excluded from selection"}}
    result["artifact_path"] = str(directory / "evaluation.json")
    _json(result["artifact_path"], result)
    return result


def evaluate_baselines(cfg, train_dataset, eval_dataset, output_dir, *, seed=42, fitted_path=None):
    return _evaluate(cfg, {"arm": "baselines", "seed": seed, "fold": 0,
                          "task_id": f"baselines-s{seed}"}, train_dataset, eval_dataset,
                     None, output_dir, fitted_path)


def evaluate_condition(cfg, task, train_dataset, eval_dataset, checkpoint_path, output_dir, *, fitted_path=None):
    return _evaluate(cfg, task, train_dataset, eval_dataset, checkpoint_path, output_dir, fitted_path)


def _load_records(records):
    loaded = []
    for record in records:
        row = json.loads(Path(record).read_text()) if isinstance(record, (str, Path)) else record
        for name in ("scores", "records", "predictions", "fitted"):
            if name + "_sha256" in row and _sha(row[name + "_path"]) != row[name + "_sha256"]:
                raise ValueError(f"Changed evaluation {name} artifact")
        scores = json.loads(Path(row["scores_path"]).read_text())
        loaded.append((row, scores))
    if not loaded:
        raise ValueError("No evaluation records")
    if len({row["support_sha256"] for row, _ in loaded}) != 1:
        raise ValueError("Methods must use identical window IDs, endpoints and target-defined support")
    return loaded


def _indexed_scores(loaded, horizon):
    indexed, seen = {}, set()
    for record, scores in loaded:
        identity = (record["arm"], int(record["seed"]))
        if identity in seen:
            raise ValueError("Duplicate arm/seed evaluation")
        seen.add(identity)
        grouped = defaultdict(list)
        for row in scores["sources"]:
            if np.isclose(row["horizon_seconds"], horizon):
                grouped[row["method"]].append(row)
        for method, rows in grouped.items():
            key = ("baselines" if method in BASELINES else record["arm"], method, int(record["seed"]))
            if key in indexed and indexed[key] != rows:
                raise ValueError("Matched baseline differs between condition artifacts")
            indexed[key] = rows
    return indexed


def _collect(indexed, arm, method, seeds):
    rows = []
    for seed in seeds:
        key = (arm, method, int(seed))
        if key not in indexed:
            raise ValueError(f"Missing evaluation grid entry {key}")
        rows.extend(indexed[key])
    # Observability may exclude an entire video. All methods have the same
    # target support; missing prediction with eligible windows is never excluded.
    if any(r.get("missing_predictions", 0) for r in rows):
        return None
    rows = [r for r in rows if r["score"] is not None]
    return rows or None


def _point(rows):
    if not rows:
        return float("inf")
    seeds = sorted({r["seed"] for r in rows})
    videos = {r["video_id"] for r in rows}
    if len(rows) != len(seeds) * len(videos):
        raise ValueError("Seed/source score grid is incomplete")
    return float(np.mean([r["score"] for r in rows]))


def _contrast(cfg, candidate, reference):
    if candidate is None or reference is None:
        return {"status": "invalid_support"}
    try:
        result = paired_group_bootstrap(candidate, reference, repetitions=cfg.bootstrap_samples, seed=812)
    except ValueError as error:
        return {"status": "invalid_support", "reason": str(error)}
    gain, lower, upper = (result[k] for k in ("relative_improvement", "relative_improvement_lower_95", "relative_improvement_upper_95"))
    if gain is None or lower is None or upper is None:
        status = "invalid_support"
    elif upper <= 0:
        status = "failure"
    elif gain < cfg.min_relative_improvement:
        status = "subthreshold"
    elif lower > 0:
        status = "success"
    else:
        status = "inconclusive"
    return {"status": status, **result}


def _finish_comparison(cfg, loaded, indexed, selection, output_dir, *, final=False):
    seeds = selection["seeds"]
    arm, method, baseline = selection["selected_arm"], selection["selected_method"], selection["baseline_method"]
    candidate = _collect(indexed, arm, method, seeds)
    reference = _collect(indexed, "baselines", baseline, seeds)
    initial = "initialized_" + method.split("_", 1)[1]
    initialized = _collect(indexed, arm, initial, seeds)
    contrasts = {"baseline": _contrast(cfg, candidate, reference),
                 "initialized": _contrast(cfg, candidate, initialized)}
    status = contrasts["baseline"]["status"]
    # Beyond a static/random feature lift, expansion needs a positive initialized
    # contrast too. Threshold size is specified against the strongest baseline.
    initial_lower = contrasts["initialized"].get("relative_improvement_lower_95")
    ready = status == "success" and initial_lower is not None and initial_lower > 0
    if ready and arm == "future":
        for control in ("future_wrong_source", "future_wrong_time"):
            if all((control, method, s) in indexed for s in seeds):
                comparison = _contrast(cfg, candidate, _collect(indexed, control, method, seeds))
                contrasts[control] = comparison
                ready &= comparison.get("relative_improvement_lower_95") is not None and comparison["relative_improvement_lower_95"] > 0
            else:
                ready = False
                contrasts[control] = {"status": "missing_control"}
    directory = Path(output_dir).resolve()
    directory.mkdir(parents=True, exist_ok=True)
    result = {**selection, "status": "synthetic_only" if cfg.mode == "synthetic" else status,
              "statistical_status": status, "ready_for_expansion": bool(ready and cfg.mode == "real" and not final),
              "role": "test" if final else "development", "mode": cfg.mode,
              "comparisons": contrasts, "minimum_relative_improvement": cfg.min_relative_improvement,
              "primary_horizon_seconds": PRIMARY_HORIZON,
              "primary_estimand": "mean seed errors per video, then equal videos; ratio of error means; no coordinate ensemble",
              "selection_scope": "frozen development selection; no test refit/reselection" if final else "development only",
              "clinical_endpoint": False, "manual_annotation_efficiency": False,
              "checkpoint_paths": [r["checkpoint_path"] for r, _ in loaded if r["arm"] == arm],
              "fitted_paths": [r["fitted_path"] for r, _ in loaded if r["arm"] == arm],
              "locked_predictors": {str(r["seed"]): {key: r.get(key) for key in
                  ("task_id", "fold", "arm", "checkpoint_path", "checkpoint_sha256", "fitted_path", "fitted_sha256")}
                  for r, _ in loaded if r["arm"] == arm},
              "artifacts": [r.get("artifact_path") for r, _ in loaded]}
    source_path = directory / "paired-source-scores.json"
    _json(source_path, {"candidate": candidate, "baseline": reference, "initialized": initialized})
    result["source_scores_path"] = str(source_path)
    result["decision_path"] = str(directory / ("locked-test-summary.json" if final else "development-selection.json"))
    _json(result["decision_path"], result)
    return result


def compare_conditions(cfg, records, output_dir, *, expected_tasks=None):
    """Select one baseline/forecast method only on a complete development grid."""
    loaded = _load_records(records)
    if any(r["role"] != "development" for r, _ in loaded):
        raise ValueError("Baseline/model selection is development-only")
    if expected_tasks is None:
        raise ValueError("Explicit expected_tasks is required; never infer completeness from available files")
    expected = {(row["arm"], int(row["seed"])) for row in expected_tasks}
    actual = {(row["arm"], int(row["seed"])) for row, _ in loaded if row["arm"] != "baselines"}
    if not expected or expected != actual:
        raise ValueError(f"Incomplete/unexpected evaluation grid: missing={sorted(expected-actual)}, extra={sorted(actual-expected)}")
    seeds = sorted({seed for _, seed in expected})
    arms = sorted({arm for arm, _ in expected})
    if expected != {(arm, seed) for arm in arms for seed in seeds}:
        raise ValueError("Every requested arm must have the same frozen seed grid")
    indexed = _indexed_scores(loaded, PRIMARY_HORIZON)
    baseline = min(BASELINES, key=lambda name: (_point(_collect(indexed, "baselines", name, seeds)), name))
    candidates = [(arm, method) for arm in arms if arm in {"masked_index", "masked", "future"}
                  for method in ("online_context_ridge", "teacher_context_ridge", "online_future_ridge", "teacher_future_ridge")]
    if not candidates:
        raise ValueError("No legitimate forecasting candidates; shuffled/observed-future controls cannot win")
    selected = min(candidates, key=lambda value: (_point(_collect(indexed, *value, seeds)), value))
    selection = {"selected_arm": selected[0], "selected_method": selected[1],
                 "baseline_method": baseline, "seeds": seeds, "expected_tasks": list(expected_tasks)}
    return _finish_comparison(cfg, loaded, indexed, selection, output_dir)


def summarize_locked_conditions(cfg, records, selection, output_dir):
    """Report the already-frozen contrast without selecting anything on test."""
    loaded = _load_records(records)
    if any(r["role"] != "test" for r, _ in loaded):
        raise ValueError("Locked final summary requires exclusively test-role artifacts")
    if selection.get("role") != "development" or not selection.get("selected_arm"):
        raise ValueError("A development selection artifact is required")
    if selection["selected_method"].startswith(("observed_", "initialized_")):
        raise ValueError("Invalid frozen forecasting candidate")
    seeds = list(selection["seeds"])
    expected = {(selection["selected_arm"], int(seed)) for seed in seeds}
    actual = {(row["arm"], int(row["seed"])) for row, _ in loaded}
    if actual != expected:
        raise ValueError("Locked test grid differs from selected arm/seeds")
    if not all(row.get("fitted_reused") for row, _ in loaded):
        raise ValueError("Final evaluation must reuse frozen fitted predictors")
    locked = selection.get("locked_predictors", {})
    for row, _ in loaded:
        frozen = locked.get(str(row["seed"]))
        if frozen is None or any(row.get(key) != frozen.get(key) for key in
                ("arm", "task_id", "fold", "checkpoint_sha256", "fitted_sha256")):
            raise ValueError("Final checkpoint/readout differs from pre-test development lock")
    indexed = _indexed_scores(loaded, PRIMARY_HORIZON)
    return _finish_comparison(cfg, loaded, indexed,
        {key: selection[key] for key in ("selected_arm", "selected_method", "baseline_method", "seeds")},
        output_dir, final=True)
