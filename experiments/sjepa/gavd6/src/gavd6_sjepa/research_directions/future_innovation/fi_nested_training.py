"""Nested source-disjoint fits, checkpoint reload, and complete OOF predictions."""

from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch

from gavd6_sjepa.shared_infrastructure.artifact_io_operations import (
    atomic_save_joblib,
    atomic_save_torch,
    sha256_file,
)

from .fi_cohort import assign_source_folds
from .fi_contracts import (
    ARMS,
    equal_source_weights,
    load_model_contract,
    read_json,
    write_json,
    write_once_json,
)
from .fi_controls import controlled_history
from .fi_feature_cache import load_cache
from .fi_residual_models import (
    SkeletonResidualHead,
    fit_baseline,
    predict_head,
    train_head,
)
from .fi_validity_audits import require_audits


def isolated_split(cohort, train, heldout):
    train_sources = set(cohort.iloc[train].video_id)
    test_sources = set(cohort.iloc[heldout].video_id)
    if not len(train) or not len(heldout) or train_sources & test_sources:
        raise ValueError("Training and held-out source partitions overlap or are empty")


def partition_control(arm, indices, cohort, arrays, scope, audit):
    subset = cohort.iloc[indices]
    skeleton, donors = controlled_history(
        arm,
        arrays["skeleton"][indices],
        subset.window_id.to_numpy(),
        subset.video_id.to_numpy(),
        arrays["matching"][indices],
    )
    if donors is not None:
        recipients = subset.window_id.tolist()
        mapping = dict(zip(cohort.window_id, cohort.video_id))
        for recipient, donor in zip(recipients, donors["donor_window_ids"]):
            if donor not in recipients or mapping[recipient] == mapping[donor]:
                raise ValueError("Mismatch donor crossed its partition or source")
        audit.append({"scope": scope, "recipient_window_ids": recipients, **donors})
    return skeleton


def raw_source_weights(video_ids):
    return equal_source_weights(video_ids) * len(set(video_ids)) / len(video_ids)


def fit_outer_fold(root, outer_fold, device="cpu"):
    root = Path(root)
    require_audits(root)
    if (root / "reports/gate-decision.json").exists():
        raise ValueError("Final decision is sealed; fitting requires a new run ID")
    cohort, arrays = load_cache(root)
    config = load_model_contract(root)
    if outer_fold not in range(5):
        raise ValueError("Outer fold must be 0..4")
    directory = root / "models" / f"fold-{outer_fold}"
    directory.mkdir(parents=True, exist_ok=True)
    receipt_path = directory / "fold-complete.json"
    if receipt_path.exists():
        verify_fold(root, outer_fold)
        return
    # Incomplete folds can resume by deterministic refitting. Completed folds are immutable.
    train = np.flatnonzero(cohort.outer_fold.to_numpy() != outer_fold)
    test = np.flatnonzero(cohort.outer_fold.to_numpy() == outer_fold)
    isolated_split(cohort, train, test)
    inner_map = assign_source_folds(cohort.iloc[train].video_id, config.inner_folds)
    inner_labels = np.array(
        [inner_map[source] for source in cohort.iloc[train].video_id]
    )
    inner = [
        (train[inner_labels != fold], train[inner_labels == fold])
        for fold in range(config.inner_folds)
    ]
    for fit, validate in inner:
        isolated_split(cohort, fit, validate)
    split_audit = {
        "outer_fold": outer_fold,
        "outer_train": cohort.iloc[train].window_id.tolist(),
        "outer_test": cohort.iloc[test].window_id.tolist(),
        "inner": [],
        "preprocessing": [],
        "donors": [],
    }
    for index, (fit, validate) in enumerate(inner):
        split_audit["inner"].append(
            {
                "fold": index,
                "train": cohort.iloc[fit].window_id.tolist(),
                "validation": cohort.iloc[validate].window_id.tolist(),
            }
        )
    predictions, artifacts, selection_records = [], [], []
    for target_name in ("person", "background"):
        target = arrays[target_name]
        alphas = []
        for alpha in config.ridge_alphas:
            error, total = 0.0, 0.0
            for fit, validate in inner:
                baseline = fit_baseline(
                    arrays["baseline"][fit],
                    target[fit],
                    cohort.iloc[fit].window_id,
                    cohort.iloc[fit].video_id,
                    alpha,
                    config.target_variance_tolerance,
                )
                diff = baseline.predict(
                    arrays["baseline"][validate]
                ) - baseline.y_scaler.transform(target[validate])
                weights = raw_source_weights(cohort.iloc[validate].video_id.to_numpy())
                error += float(
                    np.sum(
                        weights * (diff[:, baseline.valid_features] ** 2).mean(axis=1)
                    )
                )
                total += float(weights.sum())
            alphas.append(error / total)
        alpha = config.ridge_alphas[int(np.argmin(alphas))]
        inner_baselines = []
        for fold, (fit, validate) in enumerate(inner):
            baseline = fit_baseline(
                arrays["baseline"][fit],
                target[fit],
                cohort.iloc[fit].window_id,
                cohort.iloc[fit].video_id,
                alpha,
                config.target_variance_tolerance,
            )
            inner_baselines.append(baseline)
            split_audit["preprocessing"].append(
                {
                    "scope": f"{target_name}/inner-{fold}",
                    "fitted_window_ids": list(baseline.x_scaler.training_window_ids),
                    "target_fitted_window_ids": list(
                        baseline.y_scaler.training_window_ids
                    ),
                    "evaluated_window_ids": cohort.iloc[validate].window_id.tolist(),
                }
            )
        baseline = fit_baseline(
            arrays["baseline"][train],
            target[train],
            cohort.iloc[train].window_id,
            cohort.iloc[train].video_id,
            alpha,
            config.target_variance_tolerance,
        )
        split_audit["preprocessing"].append(
            {
                "scope": f"{target_name}/outer",
                "fitted_window_ids": list(baseline.x_scaler.training_window_ids),
                "target_fitted_window_ids": list(baseline.y_scaler.training_window_ids),
                "evaluated_window_ids": cohort.iloc[test].window_id.tolist(),
            }
        )
        baseline_path = directory / f"{target_name}-baseline.joblib"
        atomic_save_joblib(baseline_path, baseline)
        baseline = joblib.load(baseline_path)
        artifacts.append(baseline_path)
        y_test = baseline.y_scaler.transform(target[test])
        base_test = baseline.predict(arrays["baseline"][test])
        base_train = baseline.predict(arrays["baseline"][train])
        x_train, x_test = (
            baseline.x_scaler.transform(arrays["baseline"][train]),
            baseline.x_scaler.transform(arrays["baseline"][test]),
        )
        target_arms = (
            ["background-target"]
            if target_name == "background"
            else [a for a in ARMS if a != "background-target"]
        )
        for arm in target_arms:
            inner_controls = []
            for fold, (fit, validate) in enumerate(inner):
                inner_controls.append(
                    (
                        partition_control(
                            arm,
                            fit,
                            cohort,
                            arrays,
                            f"inner-{fold}/train",
                            split_audit["donors"],
                        ),
                        partition_control(
                            arm,
                            validate,
                            cohort,
                            arrays,
                            f"inner-{fold}/validation",
                            split_audit["donors"],
                        ),
                    )
                )
            sk_train = partition_control(
                arm, train, cohort, arrays, "outer/train", split_audit["donors"]
            )
            sk_test = partition_control(
                arm, test, cohort, arrays, "outer/test", split_audit["donors"]
            )
            for seed in config.seeds:
                losses = np.zeros((len(config.weight_decays), len(config.updates)))
                total = np.zeros_like(losses)
                for decay_index, decay in enumerate(config.weight_decays):
                    for fold, (
                        (fit, validate),
                        inner_base,
                        (sk_fit, sk_validate),
                    ) in enumerate(zip(inner, inner_baselines, inner_controls)):
                        fitted_x = inner_base.x_scaler.transform(
                            arrays["baseline"][fit]
                        )
                        fitted_residual = inner_base.y_scaler.transform(
                            target[fit]
                        ) - inner_base.predict(arrays["baseline"][fit])
                        validation = {
                            "skeleton": sk_validate,
                            "x": inner_base.x_scaler.transform(
                                arrays["baseline"][validate]
                            ),
                            "residual": inner_base.y_scaler.transform(target[validate])
                            - inner_base.predict(arrays["baseline"][validate]),
                            "weights": raw_source_weights(
                                cohort.iloc[validate].video_id.to_numpy()
                            ),
                        }
                        _, history = train_head(
                            sk_fit,
                            fitted_x,
                            fitted_residual,
                            cohort.iloc[fit].video_id.to_numpy(),
                            inner_base.valid_features,
                            seed=seed,
                            weight_decay=decay,
                            updates=config.updates,
                            model_contract=config,
                            device=device,
                            validation=validation,
                        )
                        for update_index, record in enumerate(history):
                            losses[decay_index, update_index] += record[
                                "validation_squared_error"
                            ]
                            total[decay_index, update_index] += record[
                                "validation_weight"
                            ]
                            selection_records.append(
                                {
                                    "target": target_name,
                                    "arm": arm,
                                    "seed": seed,
                                    "inner_fold": fold,
                                    "alpha": alpha,
                                    "weight_decay": decay,
                                    **record,
                                }
                            )
                chosen = np.unravel_index(np.argmin(losses / total), losses.shape)
                decay, steps = (
                    config.weight_decays[chosen[0]],
                    config.updates[chosen[1]],
                )
                head, history = train_head(
                    sk_train,
                    x_train,
                    baseline.y_scaler.transform(target[train]) - base_train,
                    cohort.iloc[train].video_id.to_numpy(),
                    baseline.valid_features,
                    seed=seed,
                    weight_decay=decay,
                    updates=[steps],
                    model_contract=config,
                    device=device,
                )
                output_dir = directory / f"{arm}-seed{seed}"
                output_dir.mkdir(exist_ok=True)
                checkpoint = output_dir / "residual-head.pt"
                state = {
                    key: value.detach().cpu()
                    for key, value in head.state_dict().items()
                }
                atomic_save_torch(
                    checkpoint,
                    {
                        "state_dict": state,
                        "baseline_dim": x_train.shape[1],
                        "width": config.width,
                        "target_dim": 256,
                        "seed": seed,
                        "arm": arm,
                        "target": target_name,
                        "updates": int(steps),
                        "weight_decay": float(decay),
                        "baseline_sha256": sha256_file(baseline_path),
                        "cache_sha256": sha256_file(
                            root / "config/cache-contract.json"
                        ),
                        "training_window_ids": cohort.iloc[train].window_id.tolist(),
                        "parameter_count": sum(p.numel() for p in head.parameters()),
                    },
                )
                # Predict from the saved checkpoint, proving the checkpoint -> evaluation interface.
                saved = torch.load(checkpoint, map_location="cpu", weights_only=True)
                restored = SkeletonResidualHead(
                    132, saved["baseline_dim"], saved["width"], saved["target_dim"]
                ).to(device)
                restored.load_state_dict(saved["state_dict"], strict=True)
                correction = predict_head(restored, sk_test, x_test, device)
                full = base_test + correction
                if not np.isfinite(full).all():
                    raise ValueError("Non-finite outer prediction")
                artifacts.append(checkpoint)
                training_path = output_dir / "fit-summary.json"
                write_json(
                    training_path,
                    {
                        "ridge_alpha": alpha,
                        "ridge_inner_mse": list(map(float, alphas)),
                        "selected_weight_decay": float(decay),
                        "selected_updates": int(steps),
                        "inner_grid_mse": (losses / total).tolist(),
                        "training_history": history,
                        "parameter_count": saved["parameter_count"],
                    },
                )
                artifacts.append(training_path)
                for position, index in enumerate(test):
                    for feature in range(256):
                        predictions.append(
                            {
                                "window_id": cohort.iloc[index].window_id,
                                "video_id": cohort.iloc[index].video_id,
                                "outer_fold": outer_fold,
                                "arm": arm,
                                "seed": seed,
                                "target": target_name,
                                "target_feature": feature,
                                "y_true": float(y_test[position, feature]),
                                "y_pred_baseline": float(base_test[position, feature]),
                                "y_pred_full": float(full[position, feature]),
                                "y_reference": 0.0,
                                "target_mean": float(baseline.y_scaler.mean[feature]),
                                "target_scale": float(baseline.y_scaler.scale[feature]),
                                "target_training_variance": float(
                                    baseline.y_scaler.variance[feature]
                                ),
                                "valid_feature": bool(baseline.valid_features[feature]),
                            }
                        )
    prediction_path = root / "predictions" / f"fold-{outer_fold}.parquet"
    temporary = prediction_path.with_suffix(".tmp")
    pd.DataFrame(predictions).to_parquet(temporary, index=False)
    temporary.replace(prediction_path)
    split_path = directory / "split-audit.json"
    write_json(split_path, split_audit)
    selection_path = directory / "inner-selection.parquet"
    pd.DataFrame(selection_records).to_parquet(selection_path, index=False)
    artifacts.extend([prediction_path, split_path, selection_path])
    write_once_json(
        receipt_path,
        {
            "outer_fold": outer_fold,
            "cache_contract_sha256": sha256_file(root / "config/cache-contract.json"),
            "validity_sha256": sha256_file(root / "qc/validity-summary.json"),
            "artifacts": {str(p.relative_to(root)): sha256_file(p) for p in artifacts},
        },
    )
    verify_fold(root, outer_fold)


def verify_fold(root, fold):
    root = Path(root)
    cohort, _ = load_cache(root)
    receipt = read_json(root / "models" / f"fold-{fold}" / "fold-complete.json")
    if receipt["cache_contract_sha256"] != sha256_file(
        root / "config/cache-contract.json"
    ) or receipt["validity_sha256"] != sha256_file(root / "qc/validity-summary.json"):
        raise ValueError("Fit lineage mismatch")
    for path, digest in receipt["artifacts"].items():
        if sha256_file(root / path) != digest:
            raise ValueError(f"Fit artifact changed: {path}")
    audit = read_json(root / "models" / f"fold-{fold}" / "split-audit.json")
    source = dict(zip(cohort.window_id, cohort.video_id))
    expected_train = set(cohort.loc[cohort.outer_fold != fold, "window_id"])
    expected_test = set(cohort.loc[cohort.outer_fold == fold, "window_id"])
    if (
        set(audit["outer_train"]) != expected_train
        or set(audit["outer_test"]) != expected_test
    ):
        raise ValueError("Outer split audit differs from cohort")

    def isolated(a, b):
        if not a or not b or {source[w] for w in a} & {source[w] for w in b}:
            raise ValueError("Split/preprocessing source leakage")

    for row in audit["inner"]:
        if set(row["train"]) | set(row["validation"]) != expected_train:
            raise ValueError(
                "Inner split includes an outer test window or omits development data"
            )
        isolated(row["train"], row["validation"])
    for row in audit["preprocessing"]:
        if row["fitted_window_ids"] != row["target_fitted_window_ids"]:
            raise ValueError("Input and target preprocessing fit partitions differ")
        isolated(row["fitted_window_ids"], row["evaluated_window_ids"])
    for row in audit["donors"]:
        for recipient, donor in zip(
            row["recipient_window_ids"], row["donor_window_ids"]
        ):
            if (
                donor not in row["recipient_window_ids"]
                or source[recipient] == source[donor]
            ):
                raise ValueError("Mismatch donor crosses source or partition boundary")
    return pd.read_parquet(root / "predictions" / f"fold-{fold}.parquet")
