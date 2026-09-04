from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .artifacts import atomic_write_json, checkpoint_path, evaluation_path, sha256_file
from .config import ExperimentContext
from .data import PreparedCohort
from .evaluation import evaluation_result_digest, validate_evaluation_frame
from .governance import load_governance, submission_readiness
from .metrics import metric_bundle, source_weights, weighted_r2
from .training import implementation_digest
from .training import load_checkpoint
from .splitting import get_fold


def load_selected_evaluations(
    context: ExperimentContext,
    cohort: PreparedCohort,
    splits: dict[str, Any],
) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for variant in context.variants:
        for fold in context.folds:
            for seed in context.seeds:
                path = evaluation_path(context.artifact_root, variant, fold, seed)
                metadata_path = path.with_suffix(".json")
                if not path.exists() or not metadata_path.exists():
                    raise FileNotFoundError(f"Missing evaluation artifact: {path}")
                metadata = json.loads(metadata_path.read_text())
                checkpoint_file = checkpoint_path(
                    context.artifact_root, variant, fold, seed
                )
                checkpoint = load_checkpoint(
                    checkpoint_file,
                    context,
                    cohort,
                    splits,
                    fold,
                    seed,
                    variant,
                )
                expected = {
                    "schema": "neurips_laterality_evaluation/v3",
                    "protocol_digest": context.protocol_digest,
                    "context_digest": context.context_digest,
                    "cohort_digest": cohort.cohort_digest,
                    "split_digest": splits["split_digest"],
                    "fold": fold,
                    "seed": seed,
                    "variant": variant,
                    "implementation_digest": implementation_digest(),
                    "checkpoint_sha256": sha256_file(checkpoint_file),
                    "model_state_digest": checkpoint["model_state_digest"],
                }
                mismatches = [
                    key for key, value in expected.items() if metadata.get(key) != value
                ]
                if mismatches or metadata.get("csv_sha256") != sha256_file(path):
                    raise RuntimeError(f"Evaluation lineage/corruption failure for {path}")
                frame = pd.read_csv(path)
                validate_evaluation_frame(
                    frame,
                    context,
                    cohort,
                    get_fold(splits, fold),
                    fold,
                    seed,
                    variant,
                )
                if metadata.get("result_digest") != evaluation_result_digest(frame):
                    raise RuntimeError(f"Evaluation result digest failure for {path}")
                frames.append(frame)
    return pd.concat(frames, ignore_index=True)


def seed_average_predictions(
    results: pd.DataFrame, expected_seeds: tuple[int, ...] | None = None
) -> pd.DataFrame:
    identifiers = [
        "sequence_id",
        "video_id",
        "condition",
        "fold",
        "variant",
        "lane",
        "role",
    ]
    unique_key = identifiers + ["seed"]
    if results.duplicated(unique_key).any():
        raise RuntimeError("Duplicate sequence/lane/seed predictions cannot be averaged")
    for _, frame in results.groupby(identifiers, sort=False):
        if expected_seeds is not None and set(frame["seed"].astype(int)) != set(expected_seeds):
            raise RuntimeError("Prediction group does not contain every selected seed")
        for column in ("target", "neutral_threshold", "training_target_scale"):
            values = frame[column].to_numpy(dtype=np.float64)
            if np.nanmax(values) - np.nanmin(values) > 1e-12:
                raise RuntimeError(f"Seed-invariant field {column} differs across seeds")
    averaged = (
        results.groupby(identifiers, as_index=False)
        .agg(
            target=("target", "first"),
            prediction=("prediction", "mean"),
            mirror_prediction=("mirror_prediction", "mean"),
            neutral_threshold=("neutral_threshold", "first"),
            training_target_scale=("training_target_scale", "first"),
            seeds=("seed", "nunique"),
        )
    )
    return averaged


def metric_table(averaged: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for (variant, lane), frame in averaged.groupby(["variant", "lane"], sort=True):
        bundle = metric_bundle(
            frame["target"].to_numpy(),
            frame["prediction"].to_numpy(),
            frame["video_id"].astype(str).to_numpy(),
            neutral_threshold=frame["neutral_threshold"].to_numpy(),
        )
        normalized_residual = (
            frame["prediction"].to_numpy()
            + frame["mirror_prediction"].to_numpy()
        ) / (2.0 * frame["training_target_scale"].replace(0, np.nan).to_numpy())
        finite = np.isfinite(normalized_residual)
        antisymmetry_weights = source_weights(
            frame["video_id"].astype(str).to_numpy()
        )
        bundle["normalized_antisymmetry_error"] = float(
            np.sqrt(
                np.average(
                    np.square(normalized_residual[finite]),
                    weights=antisymmetry_weights[finite],
                )
            )
        )
        rows.append({"variant": variant, "lane": lane, **bundle})
    return pd.DataFrame(rows)


def optimization_seed_table(results: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for (variant, lane, seed), frame in results.groupby(
        ["variant", "lane", "seed"], sort=True
    ):
        bundle = metric_bundle(
            frame["target"].to_numpy(),
            frame["prediction"].to_numpy(),
            frame["video_id"].astype(str).to_numpy(),
            neutral_threshold=frame["neutral_threshold"].to_numpy(),
        )
        residual = (
            frame["prediction"].to_numpy()
            + frame["mirror_prediction"].to_numpy()
        ) / (
            2.0
            * frame["training_target_scale"].replace(0, np.nan).to_numpy()
        )
        finite = np.isfinite(residual)
        antisymmetry_weights = source_weights(
            frame["video_id"].astype(str).to_numpy()
        )
        bundle["normalized_antisymmetry_error"] = float(
            np.sqrt(
                np.average(
                    np.square(residual[finite]),
                    weights=antisymmetry_weights[finite],
                )
            )
        )
        rows.append({"variant": variant, "lane": lane, "seed": int(seed), **bundle})
    return pd.DataFrame(rows)


def _bootstrap_delta(
    averaged: pd.DataFrame,
    variant: str,
    lane_a: str,
    lane_b: str,
    repetitions: int,
    seed: int,
) -> dict[str, Any]:
    selected = averaged[
        (averaged["variant"] == variant) & averaged["lane"].isin([lane_a, lane_b])
    ]
    pivot = selected.pivot(
        index=["sequence_id", "video_id", "target"],
        columns="lane",
        values="prediction",
    ).reset_index()
    if lane_a not in pivot or lane_b not in pivot or pivot[[lane_a, lane_b]].isna().any().any():
        raise RuntimeError(f"Cannot form paired contrast {lane_a} versus {lane_b}")
    source_ids = np.asarray(sorted(pivot["video_id"].astype(str).unique()))
    # Reduce each source block to the sufficient statistics for source-balanced
    # R2/MAE. This is exactly equivalent to rebuilding duplicated clip tables for
    # every bootstrap draw, but makes thousands of repetitions essentially instant.
    statistics: list[dict[str, float]] = []
    for source in source_ids:
        block = pivot[pivot["video_id"].astype(str) == source]
        target_block = block["target"].to_numpy(dtype=np.float64)
        prediction_a = block[lane_a].to_numpy(dtype=np.float64)
        prediction_b = block[lane_b].to_numpy(dtype=np.float64)
        statistics.append(
            {
                "target_mean": float(target_block.mean()),
                "target_square_mean": float(np.square(target_block).mean()),
                "mse_a": float(np.square(target_block - prediction_a).mean()),
                "mse_b": float(np.square(target_block - prediction_b).mean()),
                "mae_a": float(np.abs(target_block - prediction_a).mean()),
                "mae_b": float(np.abs(target_block - prediction_b).mean()),
            }
        )
    statistic_frame = pd.DataFrame(statistics)
    rng = np.random.default_rng(seed)
    draws = rng.integers(0, len(source_ids), size=(repetitions, len(source_ids)))
    target_means = statistic_frame["target_mean"].to_numpy()[draws]
    target_square_means = statistic_frame["target_square_mean"].to_numpy()[draws]
    grand_means = target_means.mean(axis=1)
    denominator = np.sum(
        target_square_means
        - 2.0 * grand_means[:, None] * target_means
        + np.square(grand_means[:, None]),
        axis=1,
    )
    numerator_a = statistic_frame["mse_a"].to_numpy()[draws].sum(axis=1)
    numerator_b = statistic_frame["mse_b"].to_numpy()[draws].sum(axis=1)
    denominator_reference = np.maximum(target_square_means.sum(axis=1), 1.0)
    r2_deltas = np.divide(
        numerator_b - numerator_a,
        denominator,
        out=np.full(repetitions, np.nan),
        where=denominator > 1e-15 * denominator_reference,
    )
    mae_deltas = (
        statistic_frame["mae_a"].to_numpy()[draws]
        - statistic_frame["mae_b"].to_numpy()[draws]
    ).mean(axis=1)
    point_weights = source_weights(pivot["video_id"].astype(str).to_numpy())
    target = pivot["target"].to_numpy()
    point_delta = weighted_r2(target, pivot[lane_a].to_numpy(), point_weights) - weighted_r2(
        target, pivot[lane_b].to_numpy(), point_weights
    )
    return {
        "comparison_type": "within_variant_lane_difference",
        "variant": variant,
        "lane_a": lane_a,
        "lane_b": lane_b,
        "estimand": "source_balanced_r2_a_minus_b",
        "estimate": float(point_delta),
        "ci95_low": float(np.nanquantile(r2_deltas, 0.025)),
        "ci95_high": float(np.nanquantile(r2_deltas, 0.975)),
        "bootstrap_repetitions": repetitions,
        "finite_r2_repetitions": int(np.isfinite(r2_deltas).sum()),
        "uncertainty_scope": "source_resampling_conditional_on_fixed_cross_fitted_predictions",
        "mae_a_minus_b_estimate": float(
            np.average(np.abs(target - pivot[lane_a]), weights=point_weights)
            - np.average(np.abs(target - pivot[lane_b]), weights=point_weights)
        ),
        "mae_a_minus_b_ci95_low": float(np.quantile(mae_deltas, 0.025)),
        "mae_a_minus_b_ci95_high": float(np.quantile(mae_deltas, 0.975)),
    }


def _bootstrap_absolute_r2(
    averaged: pd.DataFrame,
    variant: str,
    lane: str,
    repetitions: int,
    seed: int,
) -> dict[str, Any]:
    frame = averaged[
        (averaged["variant"] == variant) & (averaged["lane"] == lane)
    ]
    statistics = []
    for _, block in frame.groupby("video_id"):
        target = block["target"].to_numpy(dtype=np.float64)
        prediction = block["prediction"].to_numpy(dtype=np.float64)
        statistics.append(
            (
                float(target.mean()),
                float(np.square(target).mean()),
                float(np.square(target - prediction).mean()),
            )
        )
    values = np.asarray(statistics, dtype=np.float64)
    if len(values) < 2:
        raise RuntimeError("Absolute source bootstrap needs at least two sources")
    draws = np.random.default_rng(seed).integers(
        0, len(values), size=(repetitions, len(values))
    )
    target_mean = values[:, 0][draws]
    target_square_mean = values[:, 1][draws]
    grand_mean = target_mean.mean(axis=1)
    denominator = np.sum(
        target_square_mean
        - 2.0 * grand_mean[:, None] * target_mean
        + np.square(grand_mean[:, None]),
        axis=1,
    )
    numerator = values[:, 2][draws].sum(axis=1)
    reference = np.maximum(target_square_mean.sum(axis=1), 1.0)
    estimates = np.divide(
        numerator,
        denominator,
        out=np.full(repetitions, np.nan),
        where=denominator > 1e-15 * reference,
    )
    estimates = 1.0 - estimates
    groups = frame["video_id"].astype(str).to_numpy()
    point = weighted_r2(
        frame["target"].to_numpy(),
        frame["prediction"].to_numpy(),
        source_weights(groups),
    )
    return {
        "comparison_type": "absolute_primary_metric",
        "variant": variant,
        "lane_a": lane,
        "lane_b": "",
        "estimand": "source_balanced_r2",
        "estimate": float(point),
        "ci95_low": float(np.nanquantile(estimates, 0.025)),
        "ci95_high": float(np.nanquantile(estimates, 0.975)),
        "bootstrap_repetitions": repetitions,
        "finite_r2_repetitions": int(np.isfinite(estimates).sum()),
        "uncertainty_scope": "source_resampling_conditional_on_fixed_cross_fitted_predictions",
    }


def bootstrap_table(context: ExperimentContext, averaged: pd.DataFrame) -> pd.DataFrame:
    repetitions = int(context.protocol["evaluation"]["bootstrap_repetitions"])
    base_seed = int(context.protocol["evaluation"]["bootstrap_seed"])
    contrasts = (
        ("learned_single_free", "random_single_free"),
        ("learned_two_pass_odd_zero", "random_two_pass_odd_zero"),
        ("learned_two_pass_odd_free", "random_two_pass_odd_free"),
        ("learned_two_pass_even_free", "random_two_pass_even_free"),
        ("learned_two_pass_odd_zero", "learned_two_pass_odd_free"),
        ("learned_two_pass_odd_free", "learned_single_free"),
        ("learned_two_pass_odd_free", "learned_two_pass_even_free"),
        ("learned_single_free", "nuisance_visibility_free"),
        ("learned_single_free", "nuisance_acquisition_free"),
        ("learned_single_free", "nuisance_annotation_free"),
        ("learned_single_free", "nuisance_all_free"),
        ("learned_single_plus_nuisance_free", "nuisance_all_free"),
        ("learned_single_plus_nuisance_free", "learned_single_free"),
        ("learned_framed_odd_zero", "learned_two_pass_odd_zero"),
    )
    rows: list[dict[str, Any]] = []
    for variant_index, variant in enumerate(context.variants):
        available = set(averaged.loc[averaged["variant"] == variant, "lane"])
        for contrast_index, (lane_a, lane_b) in enumerate(contrasts):
            if {lane_a, lane_b} <= available:
                rows.append(
                    _bootstrap_delta(
                        averaged,
                        variant,
                        lane_a,
                        lane_b,
                        repetitions,
                        base_seed + 101 * variant_index + contrast_index,
                    )
                )
    primary_variant = str(context.protocol["evaluation"]["primary_variant"])
    primary_lane = str(context.protocol["evaluation"]["primary_lane"])
    if primary_variant in set(context.variants):
        rows.append(
            _bootstrap_absolute_r2(
                averaged,
                primary_variant,
                primary_lane,
                repetitions,
                base_seed + 9001,
            )
        )
    if {"vanilla", "reflection_augmented"} <= set(context.variants):
        selected = averaged[
            (averaged["lane"] == primary_lane)
            & averaged["variant"].isin(["vanilla", "reflection_augmented"])
        ].copy()
        selected["lane"] = selected["variant"]
        selected["variant"] = "cross_variant"
        comparison = _bootstrap_delta(
            selected,
            "cross_variant",
            "reflection_augmented",
            "vanilla",
            repetitions,
            base_seed + 9011,
        )
        comparison["comparison_type"] = "training_variant_difference"
        comparison["variant"] = "reflection_augmented_minus_vanilla"
        comparison["evaluated_lane"] = primary_lane
        rows.append(comparison)
    return pd.DataFrame(rows)


def _checkpoint_prediction_frame(
    results: pd.DataFrame,
    variant: str,
    lane: str,
    prediction_name: str,
) -> pd.DataFrame:
    frame = results[
        (results["variant"] == variant) & (results["lane"] == lane)
    ][["sequence_id", "video_id", "target", "seed", "prediction"]].copy()
    if frame.empty:
        raise RuntimeError(f"Missing checkpoint predictions for {variant}/{lane}")
    if frame.duplicated(["sequence_id", "seed"]).any():
        raise RuntimeError(f"Duplicate checkpoint predictions for {variant}/{lane}")
    return frame.rename(columns={"prediction": prediction_name})


def _checkpoint_r2_bootstrap_row(
    results: pd.DataFrame,
    *,
    variant_a: str,
    lane_a: str,
    variant_b: str | None,
    lane_b: str | None,
    repetitions: int,
    seed: int,
    comparison_type: str,
) -> dict[str, Any]:
    """Bootstrap sources while preserving every registered checkpoint separately."""
    first = _checkpoint_prediction_frame(results, variant_a, lane_a, "prediction_a")
    if variant_b is None or lane_b is None:
        paired = first
    else:
        second = _checkpoint_prediction_frame(
            results, variant_b, lane_b, "prediction_b"
        )
        paired = first.merge(
            second,
            on=["sequence_id", "video_id", "target", "seed"],
            how="inner",
            validate="one_to_one",
        )
        if len(paired) != len(first) or len(paired) != len(second):
            raise RuntimeError(
                f"Checkpoint contrast lacks paired coverage: {variant_a}/{lane_a} "
                f"versus {variant_b}/{lane_b}"
            )

    seeds = np.asarray(sorted(paired["seed"].astype(int).unique()))
    sources = np.asarray(sorted(paired["video_id"].astype(str).unique()))
    if len(seeds) < 1 or len(sources) < 2:
        raise RuntimeError("Checkpoint bootstrap needs at least one seed and two sources")
    reference_rows = set(
        zip(
            paired.loc[paired["seed"] == seeds[0], "sequence_id"].astype(str),
            paired.loc[paired["seed"] == seeds[0], "video_id"].astype(str),
        )
    )
    for checkpoint_seed in seeds[1:]:
        observed_rows = set(
            zip(
                paired.loc[
                    paired["seed"] == checkpoint_seed, "sequence_id"
                ].astype(str),
                paired.loc[paired["seed"] == checkpoint_seed, "video_id"].astype(
                    str
                ),
            )
        )
        if observed_rows != reference_rows:
            raise RuntimeError("Every seed must predict the same sequence/source rows")

    target_mean = np.empty(len(sources), dtype=np.float64)
    target_square_mean = np.empty(len(sources), dtype=np.float64)
    mse_a = np.empty((len(seeds), len(sources)), dtype=np.float64)
    mse_b = (
        np.empty((len(seeds), len(sources)), dtype=np.float64)
        if "prediction_b" in paired
        else None
    )
    for source_index, source in enumerate(sources):
        source_reference = paired[
            (paired["seed"] == seeds[0])
            & (paired["video_id"].astype(str) == source)
        ]
        target_values = source_reference["target"].to_numpy(dtype=np.float64)
        target_mean[source_index] = float(target_values.mean())
        target_square_mean[source_index] = float(np.square(target_values).mean())
        for seed_index, checkpoint_seed in enumerate(seeds):
            block = paired[
                (paired["seed"] == checkpoint_seed)
                & (paired["video_id"].astype(str) == source)
            ]
            block_target = block["target"].to_numpy(dtype=np.float64)
            if len(block_target) != len(target_values) or not np.allclose(
                np.sort(block_target), np.sort(target_values), rtol=0.0, atol=1e-12
            ):
                raise RuntimeError("Target rows differ across optimization seeds")
            mse_a[seed_index, source_index] = float(
                np.square(
                    block_target
                    - block["prediction_a"].to_numpy(dtype=np.float64)
                ).mean()
            )
            if mse_b is not None:
                mse_b[seed_index, source_index] = float(
                    np.square(
                        block_target
                        - block["prediction_b"].to_numpy(dtype=np.float64)
                    ).mean()
                )

    draws = np.random.default_rng(seed).integers(
        0, len(sources), size=(repetitions, len(sources))
    )
    drawn_target_mean = target_mean[draws]
    drawn_target_square_mean = target_square_mean[draws]
    grand_mean = drawn_target_mean.mean(axis=1)
    denominator = np.sum(
        drawn_target_square_mean
        - 2.0 * grand_mean[:, None] * drawn_target_mean
        + np.square(grand_mean[:, None]),
        axis=1,
    )
    reference = np.maximum(drawn_target_square_mean.sum(axis=1), 1.0)
    valid_denominator = denominator > 1e-15 * reference

    def seed_marginal_r2(mse: np.ndarray) -> np.ndarray:
        numerator = mse[:, draws].sum(axis=2)
        per_seed = np.divide(
            numerator,
            denominator[None, :],
            out=np.full_like(numerator, np.nan),
            where=valid_denominator[None, :],
        )
        return np.nanmean(1.0 - per_seed, axis=0)

    bootstrap_a = seed_marginal_r2(mse_a)
    point_a = float(
        np.mean(
            [
                weighted_r2(
                    paired.loc[paired["seed"] == checkpoint_seed, "target"].to_numpy(),
                    paired.loc[
                        paired["seed"] == checkpoint_seed, "prediction_a"
                    ].to_numpy(),
                    source_weights(
                        paired.loc[
                            paired["seed"] == checkpoint_seed, "video_id"
                        ].astype(str).to_numpy()
                    ),
                )
                for checkpoint_seed in seeds
            ]
        )
    )
    if mse_b is None:
        estimates = bootstrap_a
        point = point_a
        estimand = "mean_registered_seed_source_balanced_r2"
    else:
        bootstrap_b = seed_marginal_r2(mse_b)
        estimates = bootstrap_a - bootstrap_b
        point_b = float(
            np.mean(
                [
                    weighted_r2(
                        paired.loc[
                            paired["seed"] == checkpoint_seed, "target"
                        ].to_numpy(),
                        paired.loc[
                            paired["seed"] == checkpoint_seed, "prediction_b"
                        ].to_numpy(),
                        source_weights(
                            paired.loc[
                                paired["seed"] == checkpoint_seed, "video_id"
                            ].astype(str).to_numpy()
                        ),
                    )
                    for checkpoint_seed in seeds
                ]
            )
        )
        point = point_a - point_b
        estimand = "mean_registered_seed_source_balanced_r2_a_minus_b"
    finite = estimates[np.isfinite(estimates)]
    if not len(finite):
        raise RuntimeError("Every checkpoint bootstrap R2 draw is degenerate")
    return {
        "comparison_type": comparison_type,
        "variant_a": variant_a,
        "lane_a": lane_a,
        "variant_b": variant_b or "",
        "lane_b": lane_b or "",
        "estimand": estimand,
        "estimate": point,
        "ci95_low": float(np.quantile(finite, 0.025)),
        "ci95_high": float(np.quantile(finite, 0.975)),
        "bootstrap_repetitions": repetitions,
        "finite_r2_repetitions": int(len(finite)),
        "registered_seed_count": int(len(seeds)),
        "uncertainty_scope": (
            "source_resampling_conditional_on_fixed_cross_fitted_checkpoints_"
            "with_metrics_computed_per_seed_before_averaging"
        ),
    }


def checkpoint_bootstrap_table(
    context: ExperimentContext, results: pd.DataFrame
) -> pd.DataFrame:
    repetitions = int(context.protocol["evaluation"]["bootstrap_repetitions"])
    base_seed = int(context.protocol["evaluation"]["bootstrap_seed"]) + 10000
    primary_variant = str(context.protocol["evaluation"]["primary_variant"])
    primary_lane = str(context.protocol["evaluation"]["primary_lane"])
    constructed_lane = str(
        context.protocol["evaluation"]["constructed_repair_lane"]
    )
    rows: list[dict[str, Any]] = []

    comparisons = [
        ("absolute_primary", primary_lane, None),
        ("primary_training_content", primary_lane, "random_single_free"),
        ("absolute_constructed", constructed_lane, None),
        (
            "constructed_training_content",
            constructed_lane,
            "random_two_pass_odd_zero",
        ),
        (
            "odd_feature_training_content",
            "learned_two_pass_odd_free",
            "random_two_pass_odd_free",
        ),
        (
            "even_feature_training_content",
            "learned_two_pass_even_free",
            "random_two_pass_even_free",
        ),
        (
            "zero_origin_constraint_effect",
            constructed_lane,
            "learned_two_pass_odd_free",
        ),
        (
            "two_pass_odd_feature_effect",
            "learned_two_pass_odd_free",
            primary_lane,
        ),
        (
            "odd_versus_even_parity",
            "learned_two_pass_odd_free",
            "learned_two_pass_even_free",
        ),
        ("primary_versus_visibility", primary_lane, "nuisance_visibility_free"),
        ("primary_versus_acquisition", primary_lane, "nuisance_acquisition_free"),
        ("primary_versus_annotation", primary_lane, "nuisance_annotation_free"),
        ("primary_versus_all_nuisance", primary_lane, "nuisance_all_free"),
        (
            "learned_increment_beyond_nuisance",
            "learned_single_plus_nuisance_free",
            "nuisance_all_free",
        ),
        (
            "nuisance_increment_beyond_primary",
            "learned_single_plus_nuisance_free",
            primary_lane,
        ),
    ]
    for index, (comparison_type, lane_a, lane_b) in enumerate(comparisons):
        rows.append(
            _checkpoint_r2_bootstrap_row(
                results,
                variant_a=primary_variant,
                lane_a=lane_a,
                variant_b=primary_variant if lane_b is not None else None,
                lane_b=lane_b,
                repetitions=repetitions,
                seed=base_seed + index,
                comparison_type=comparison_type,
            )
        )
    if {"vanilla", "reflection_augmented"} <= set(context.variants):
        rows.append(
            _checkpoint_r2_bootstrap_row(
                results,
                variant_a="reflection_augmented",
                lane_a=primary_lane,
                variant_b="vanilla",
                lane_b=primary_lane,
                repetitions=repetitions,
                seed=base_seed + 100,
                comparison_type="reflection_minus_vanilla_primary",
            )
        )
    return pd.DataFrame(rows)


def _native_symmetry_bootstrap_row(
    averaged: pd.DataFrame,
    *,
    variant_a: str,
    lane_a: str,
    variant_b: str | None,
    lane_b: str | None,
    repetitions: int,
    seed: int,
    comparison_type: str,
) -> dict[str, Any]:
    def residual_frame(variant: str, lane: str, label: str) -> pd.DataFrame:
        frame = averaged[
            (averaged["variant"] == variant) & (averaged["lane"] == lane)
        ].copy()
        residual = (
            frame["prediction"].to_numpy(dtype=np.float64)
            + frame["mirror_prediction"].to_numpy(dtype=np.float64)
        ) / (2.0 * frame["training_target_scale"].to_numpy(dtype=np.float64))
        return pd.DataFrame(
            {
                "sequence_id": frame["sequence_id"].astype(str),
                "video_id": frame["video_id"].astype(str),
                "seed": frame["seed"].astype(int),
                label: np.square(residual),
            }
        )

    first = residual_frame(variant_a, lane_a, "squared_a")
    if variant_b is None or lane_b is None:
        paired = first
    else:
        second = residual_frame(variant_b, lane_b, "squared_b")
        paired = first.merge(
            second,
            on=["sequence_id", "video_id", "seed"],
            how="inner",
            validate="one_to_one",
        )
        if len(paired) != len(first) or len(paired) != len(second):
            raise RuntimeError("Native-symmetry comparison lacks paired sequence coverage")
    source_stats = paired.groupby("video_id", as_index=False).mean(numeric_only=True)
    if len(source_stats) < 2:
        raise RuntimeError("Native-symmetry bootstrap needs at least two sources")
    draws = np.random.default_rng(seed).integers(
        0, len(source_stats), size=(repetitions, len(source_stats))
    )
    error_a = np.sqrt(source_stats["squared_a"].to_numpy()[draws].mean(axis=1))
    point_a = float(np.sqrt(source_stats["squared_a"].mean()))
    if "squared_b" in source_stats:
        error_b = np.sqrt(source_stats["squared_b"].to_numpy()[draws].mean(axis=1))
        estimates = error_a - error_b
        point = point_a - float(np.sqrt(source_stats["squared_b"].mean()))
    else:
        estimates = error_a
        point = point_a
    return {
        "comparison_type": comparison_type,
        "variant_a": variant_a,
        "lane_a": lane_a,
        "variant_b": variant_b or "",
        "lane_b": lane_b or "",
        "estimand": "source_balanced_normalized_native_antisymmetry_error_a_minus_b"
        if variant_b is not None
        else "source_balanced_normalized_native_antisymmetry_error",
        "estimate": point,
        "ci95_low": float(np.quantile(estimates, 0.025)),
        "ci95_high": float(np.quantile(estimates, 0.975)),
        "bootstrap_repetitions": repetitions,
        "uncertainty_scope": "source_resampling_conditional_on_fixed_cross_fitted_predictions",
    }


def native_symmetry_bootstrap_table(
    context: ExperimentContext, results: pd.DataFrame
) -> pd.DataFrame:
    repetitions = int(context.protocol["evaluation"]["bootstrap_repetitions"])
    base_seed = int(context.protocol["evaluation"]["bootstrap_seed"]) + 20000
    rows: list[dict[str, Any]] = []
    for index, variant in enumerate(context.variants):
        rows.append(
            _native_symmetry_bootstrap_row(
                results,
                variant_a=variant,
                lane_a="learned_single_free",
                variant_b=None,
                lane_b=None,
                repetitions=repetitions,
                seed=base_seed + 10 * index,
                comparison_type="absolute_native_learned_symmetry",
            )
        )
        rows.append(
            _native_symmetry_bootstrap_row(
                results,
                variant_a=variant,
                lane_a="learned_single_free",
                variant_b=variant,
                lane_b="random_single_free",
                repetitions=repetitions,
                seed=base_seed + 10 * index + 1,
                comparison_type="learned_minus_random_native_symmetry",
            )
        )
    if {"vanilla", "reflection_augmented"} <= set(context.variants):
        rows.append(
            _native_symmetry_bootstrap_row(
                results,
                variant_a="reflection_augmented",
                lane_a="learned_single_free",
                variant_b="vanilla",
                lane_b="learned_single_free",
                repetitions=repetitions,
                seed=base_seed + 100,
                comparison_type="reflection_minus_vanilla_native_symmetry",
            )
        )
    return pd.DataFrame(rows)


def native_symmetry_seed_table(results: pd.DataFrame) -> pd.DataFrame:
    """Compute output oddness per checkpoint; signed seed errors never average."""
    rows: list[dict[str, Any]] = []
    for (variant, lane, seed), frame in results.groupby(
        ["variant", "lane", "seed"], sort=True
    ):
        residual = (
            frame["prediction"].to_numpy(dtype=np.float64)
            + frame["mirror_prediction"].to_numpy(dtype=np.float64)
        ) / (2.0 * frame["training_target_scale"].to_numpy(dtype=np.float64))
        squared = pd.DataFrame(
            {
                "video_id": frame["video_id"].astype(str).to_numpy(),
                "squared": np.square(residual),
            }
        ).groupby("video_id", as_index=False)["squared"].mean()
        rows.append(
            {
                "variant": variant,
                "lane": lane,
                "seed": int(seed),
                "source_balanced_normalized_antisymmetry_error": float(
                    np.sqrt(squared["squared"].mean())
                ),
                "maximum_absolute_normalized_residual": float(
                    np.max(np.abs(residual))
                ),
                "source_count": int(len(squared)),
            }
        )
    return pd.DataFrame(rows)


def _representation_rows(
    results: pd.DataFrame,
    *,
    variant: str,
    state: str,
    value_name: str,
    primary_lane: str,
) -> pd.DataFrame:
    if state not in {"learned", "initial"}:
        raise ValueError(f"Unknown representation state: {state}")
    frame = results[
        (results["variant"] == variant) & (results["lane"] == primary_lane)
    ][
        [
            "sequence_id",
            "video_id",
            "seed",
            f"{state}_strict_equivariance_error",
        ]
    ].copy()
    if frame.empty or frame.duplicated(["sequence_id", "seed"]).any():
        raise RuntimeError(f"Invalid representation rows for {variant}/{state}")
    return frame.rename(
        columns={f"{state}_strict_equivariance_error": value_name}
    )


def _representation_bootstrap_row(
    results: pd.DataFrame,
    *,
    primary_lane: str,
    variant_a: str,
    state_a: str,
    variant_b: str | None,
    state_b: str | None,
    repetitions: int,
    seed: int,
    comparison_type: str,
) -> dict[str, Any]:
    first = _representation_rows(
        results,
        variant=variant_a,
        state=state_a,
        value_name="error_a",
        primary_lane=primary_lane,
    )
    if variant_b is None or state_b is None:
        paired = first
    else:
        second = _representation_rows(
            results,
            variant=variant_b,
            state=state_b,
            value_name="error_b",
            primary_lane=primary_lane,
        )
        paired = first.merge(
            second,
            on=["sequence_id", "video_id", "seed"],
            how="inner",
            validate="one_to_one",
        )
        if len(paired) != len(first) or len(paired) != len(second):
            raise RuntimeError("Representation comparison lacks paired checkpoint rows")
    source_stats = paired.groupby("video_id", as_index=False).mean(numeric_only=True)
    if len(source_stats) < 2:
        raise RuntimeError("Representation bootstrap needs at least two sources")
    draws = np.random.default_rng(seed).integers(
        0, len(source_stats), size=(repetitions, len(source_stats))
    )
    estimates_a = source_stats["error_a"].to_numpy()[draws].mean(axis=1)
    point_a = float(source_stats["error_a"].mean())
    if "error_b" in source_stats:
        estimates_b = source_stats["error_b"].to_numpy()[draws].mean(axis=1)
        estimates = estimates_a - estimates_b
        point = point_a - float(source_stats["error_b"].mean())
        estimand = "source_balanced_strict_representation_error_a_minus_b"
    else:
        estimates = estimates_a
        point = point_a
        estimand = "source_balanced_strict_representation_error"
    return {
        "comparison_type": comparison_type,
        "variant_a": variant_a,
        "state_a": state_a,
        "variant_b": variant_b or "",
        "state_b": state_b or "",
        "estimand": estimand,
        "estimate": point,
        "ci95_low": float(np.quantile(estimates, 0.025)),
        "ci95_high": float(np.quantile(estimates, 0.975)),
        "bootstrap_repetitions": repetitions,
        "registered_seed_count": int(paired["seed"].nunique()),
        "uncertainty_scope": (
            "source_resampling_conditional_on_fixed_cross_fitted_checkpoints_"
            "with_per_seed_representation_errors_retained"
        ),
    }


def representation_equivariance_bootstrap_table(
    context: ExperimentContext, results: pd.DataFrame
) -> pd.DataFrame:
    repetitions = int(context.protocol["evaluation"]["bootstrap_repetitions"])
    base_seed = int(context.protocol["evaluation"]["bootstrap_seed"]) + 30000
    primary_lane = str(context.protocol["evaluation"]["primary_lane"])
    rows: list[dict[str, Any]] = []
    for index, variant in enumerate(context.variants):
        rows.append(
            _representation_bootstrap_row(
                results,
                primary_lane=primary_lane,
                variant_a=variant,
                state_a="learned",
                variant_b=None,
                state_b=None,
                repetitions=repetitions,
                seed=base_seed + 10 * index,
                comparison_type="absolute_learned_strict_equivariance",
            )
        )
        rows.append(
            _representation_bootstrap_row(
                results,
                primary_lane=primary_lane,
                variant_a=variant,
                state_a="initial",
                variant_b=None,
                state_b=None,
                repetitions=repetitions,
                seed=base_seed + 10 * index + 1,
                comparison_type="absolute_initial_strict_equivariance",
            )
        )
        rows.append(
            _representation_bootstrap_row(
                results,
                primary_lane=primary_lane,
                variant_a=variant,
                state_a="learned",
                variant_b=variant,
                state_b="initial",
                repetitions=repetitions,
                seed=base_seed + 10 * index + 2,
                comparison_type="learned_minus_initial_strict_equivariance",
            )
        )
    if {"vanilla", "reflection_augmented"} <= set(context.variants):
        initial_augmented = _representation_rows(
            results,
            variant="reflection_augmented",
            state="initial",
            value_name="augmented_initial",
            primary_lane=primary_lane,
        )
        initial_vanilla = _representation_rows(
            results,
            variant="vanilla",
            state="initial",
            value_name="vanilla_initial",
            primary_lane=primary_lane,
        )
        initial_pair = initial_augmented.merge(
            initial_vanilla,
            on=["sequence_id", "video_id", "seed"],
            validate="one_to_one",
        )
        if len(initial_pair) != len(initial_augmented) or not np.allclose(
            initial_pair["augmented_initial"],
            initial_pair["vanilla_initial"],
            rtol=0.0,
            atol=1e-12,
        ):
            raise RuntimeError(
                "Training variants do not share the same paired initial representation"
            )
        rows.append(
            _representation_bootstrap_row(
                results,
                primary_lane=primary_lane,
                variant_a="reflection_augmented",
                state_a="learned",
                variant_b="vanilla",
                state_b="learned",
                repetitions=repetitions,
                seed=base_seed + 100,
                comparison_type="reflection_minus_vanilla_strict_equivariance",
            )
        )
    return pd.DataFrame(rows)


def representation_equivariance_seed_table(
    context: ExperimentContext, results: pd.DataFrame
) -> pd.DataFrame:
    primary_lane = str(context.protocol["evaluation"]["primary_lane"])
    selected = results[results["lane"] == primary_lane]
    rows: list[dict[str, Any]] = []
    for (variant, seed), frame in selected.groupby(["variant", "seed"], sort=True):
        for state in ("learned", "initial"):
            source_means = (
                pd.DataFrame(
                    {
                        "video_id": frame["video_id"].astype(str),
                        "error": frame[
                            f"{state}_strict_equivariance_error"
                        ].to_numpy(dtype=np.float64),
                    }
                )
                .groupby("video_id", as_index=False)["error"]
                .mean()
            )
            rows.append(
                {
                    "variant": variant,
                    "seed": int(seed),
                    "encoder_state": state,
                    "source_balanced_strict_equivariance_error": float(
                        source_means["error"].mean()
                    ),
                    "source_count": int(len(source_means)),
                }
            )
    return pd.DataFrame(rows)


def symmetry_gate_decisions(
    *,
    primary_r2_ci_low: float,
    minimum_primary_r2: float,
    native_output_error_ci_high: float,
    native_output_error_margin: float,
    primary_minus_initial_r2_ci_low: float,
    representation_error_ci_high: float,
    representation_error_margin: float,
    learned_minus_initial_representation_error_ci_high: float,
) -> dict[str, bool]:
    """Evaluate the predeclared conjunctive gates from scalar CI boundaries."""
    values = np.asarray(
        [
            primary_r2_ci_low,
            minimum_primary_r2,
            native_output_error_ci_high,
            native_output_error_margin,
            primary_minus_initial_r2_ci_low,
            representation_error_ci_high,
            representation_error_margin,
            learned_minus_initial_representation_error_ci_high,
        ],
        dtype=np.float64,
    )
    if not np.isfinite(values).all():
        raise ValueError("Symmetry gates require finite confidence boundaries")
    absolute_utility = bool(primary_r2_ci_low > minimum_primary_r2)
    output_margin = bool(native_output_error_ci_high < native_output_error_margin)
    training_content = bool(primary_minus_initial_r2_ci_low > 0.0)
    representation_margin = bool(
        representation_error_ci_high < representation_error_margin
    )
    representation_improvement = bool(
        learned_minus_initial_representation_error_ci_high < 0.0
    )
    native_probe = absolute_utility and output_margin and training_content
    strict_representation = representation_margin and representation_improvement
    return {
        "absolute_native_predictive_utility_ci": absolute_utility,
        "native_output_symmetry_margin_ci": output_margin,
        "native_training_content_beyond_initialization_ci": training_content,
        "native_probe_behavior_supported": native_probe,
        "strict_representation_error_margin_ci": representation_margin,
        "strict_representation_training_improvement_ci": representation_improvement,
        "strict_checkpoint_representation_equivariance_supported": strict_representation,
        "training_induced_symmetry_supported_conditional_on_pose_schema": bool(
            native_probe and strict_representation
        ),
    }


def _write_overview_figure(metrics: pd.DataFrame, output: Path) -> None:
    import matplotlib.pyplot as plt

    lanes = [
        "learned_single_free",
        "learned_two_pass_odd_zero",
        "random_single_free",
        "random_two_pass_odd_zero",
        "nuisance_all_free",
    ]
    selected = metrics[metrics["lane"].isin(lanes)].copy()
    if selected.empty:
        return
    pivot = selected.pivot(index="lane", columns="variant", values="source_balanced_r2")
    pivot = pivot.reindex([lane for lane in lanes if lane in pivot.index])
    axis = pivot.plot(kind="bar", figsize=(8, 4), color=["#3b6fb6", "#d0782a"])
    axis.axhline(0.0, color="black", linewidth=0.8)
    axis.set_ylabel("source-balanced cross-fitted $R^2$")
    axis.set_xlabel("")
    axis.set_title("Held-out source-video evaluation")
    axis.legend(title="training variant")
    axis.figure.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    axis.figure.savefig(output, format="svg")
    plt.close(axis.figure)


def aggregate_and_save(
    context: ExperimentContext,
    cohort: PreparedCohort,
    splits: dict[str, Any],
) -> dict[str, Any]:
    if cohort.attrition.get("profile") != context.profile:
        raise RuntimeError("Cannot report a cohort from a different run profile")
    if bool(cohort.attrition.get("synthetic_evidence")) != (not context.is_paper):
        raise RuntimeError("Cohort evidence marker disagrees with the active profile")
    results = load_selected_evaluations(context, cohort, splits)
    averaged = seed_average_predictions(results, context.seeds)
    expected_seed_count = len(context.seeds)
    if not (averaged["seeds"] == expected_seed_count).all():
        raise RuntimeError("At least one cross-fitted prediction is missing a selected seed")
    metrics = metric_table(averaged)
    metrics.insert(0, "analysis_set", "all_qc_eligible")
    metrics.insert(1, "estimand", "registered_seed_mean_prediction_ensemble")
    eligibility = cohort.table[["sequence_id", "authorized_coverage"]]
    sensitivity_rows = averaged.merge(
        eligibility, on="sequence_id", how="left", validate="many_to_one"
    )
    high_coverage = sensitivity_rows[
        sensitivity_rows["authorized_coverage"]
        >= float(
            context.protocol["evaluation"]["high_coverage_sensitivity_threshold"]
        )
    ]
    high_coverage_metrics = metric_table(high_coverage)
    high_coverage_metrics.insert(0, "analysis_set", "authorized_coverage_at_least_0.9")
    high_coverage_metrics.insert(
        1, "estimand", "registered_seed_mean_prediction_ensemble"
    )
    seed_metrics = optimization_seed_table(results)
    seed_metrics.insert(0, "estimand", "single_checkpoint")
    ensemble_bootstrap = bootstrap_table(context, averaged)
    checkpoint_bootstrap = checkpoint_bootstrap_table(context, results)
    native_symmetry_bootstrap = native_symmetry_bootstrap_table(context, results)
    native_symmetry_by_seed = native_symmetry_seed_table(results)
    representation_bootstrap = representation_equivariance_bootstrap_table(
        context, results
    )
    representation_by_seed = representation_equivariance_seed_table(
        context, results
    )
    report_root = context.artifact_root / "report"
    report_root.mkdir(parents=True, exist_ok=True)
    averaged.to_csv(report_root / "seed_averaged_predictions.csv", index=False)
    metrics.to_csv(report_root / "metrics.csv", index=False)
    high_coverage_metrics.to_csv(
        report_root / "high_coverage_sensitivity_metrics.csv", index=False
    )
    seed_metrics.to_csv(report_root / "optimization_seed_metrics.csv", index=False)
    ensemble_bootstrap.to_csv(
        report_root / "ensemble_prediction_source_bootstrap.csv", index=False
    )
    checkpoint_bootstrap.to_csv(
        report_root / "checkpoint_source_bootstrap.csv", index=False
    )
    native_symmetry_bootstrap.to_csv(
        report_root / "native_output_symmetry_source_bootstrap.csv", index=False
    )
    native_symmetry_by_seed.to_csv(
        report_root / "native_output_symmetry_by_seed.csv", index=False
    )
    representation_bootstrap.to_csv(
        report_root / "strict_representation_equivariance_source_bootstrap.csv",
        index=False,
    )
    representation_by_seed.to_csv(
        report_root / "strict_representation_equivariance_by_seed.csv", index=False
    )
    _write_overview_figure(metrics, report_root / "metric_overview.svg")

    expected_fold_count = int(context.protocol["splits"]["outer_folds"])
    paper_complete = (
        context.is_paper
        and set(context.folds) == set(range(expected_fold_count))
        and set(context.seeds) == set(context.protocol["training"]["paper"]["seeds"])
        and set(context.variants) == set(context.protocol["training"]["variants"])
    )
    odd_rows = native_symmetry_by_seed[
        native_symmetry_by_seed["lane"].isin(
            [
                "learned_two_pass_odd_zero",
                "random_two_pass_odd_zero",
                "learned_framed_odd_zero",
            ]
        )
    ]
    exactness_pass = bool(
        len(odd_rows)
        and (
            odd_rows["maximum_absolute_normalized_residual"]
            <= float(
                context.protocol["evaluation"]["decision_rules"][
                    "maximum_normalized_antisymmetry_error"
                ]
            )
        ).all()
    )
    constructed_advantage = checkpoint_bootstrap[
        checkpoint_bootstrap["comparison_type"]
        == "constructed_training_content"
    ]
    constructed_content_pass = bool(
        len(constructed_advantage) and (constructed_advantage["ci95_low"] > 0.0).all()
    )
    constructed_absolute = checkpoint_bootstrap[
        checkpoint_bootstrap["comparison_type"] == "absolute_constructed"
    ]
    constructed_absolute_pass = bool(
        len(constructed_absolute)
        and (
            constructed_absolute["ci95_low"]
            > float(
                context.protocol["evaluation"]["decision_rules"][
                    "minimum_constructed_source_balanced_r2"
                ]
            )
        ).all()
    )
    native_absolute = native_symmetry_bootstrap[
        (native_symmetry_bootstrap["comparison_type"] == "absolute_native_learned_symmetry")
        & (native_symmetry_bootstrap["variant_a"] == context.protocol["evaluation"]["primary_variant"])
    ]
    native_content = checkpoint_bootstrap[
        checkpoint_bootstrap["comparison_type"] == "primary_training_content"
    ]
    native_performance = checkpoint_bootstrap[
        checkpoint_bootstrap["comparison_type"] == "absolute_primary"
    ]
    native_margin_pass = bool(
        len(native_absolute)
        and (
            native_absolute["ci95_high"]
            < float(
                context.protocol["evaluation"]["decision_rules"][
                    "native_symmetry_normalized_error_margin"
                ]
            )
        ).all()
    )
    native_content_pass = bool(
        len(native_content) and (native_content["ci95_low"] > 0.0).all()
    )
    native_performance_pass = bool(
        len(native_performance)
        and (
            native_performance["ci95_low"]
            > float(
                context.protocol["evaluation"]["decision_rules"][
                    "minimum_primary_source_balanced_r2"
                ]
            )
        ).all()
    )
    primary_variant = str(context.protocol["evaluation"]["primary_variant"])
    representation_absolute = representation_bootstrap[
        (
            representation_bootstrap["comparison_type"]
            == "absolute_learned_strict_equivariance"
        )
        & (representation_bootstrap["variant_a"] == primary_variant)
    ]
    representation_training_effect = representation_bootstrap[
        (
            representation_bootstrap["comparison_type"]
            == "learned_minus_initial_strict_equivariance"
        )
        & (representation_bootstrap["variant_a"] == primary_variant)
    ]
    required_gate_rows = {
        "native_performance": native_performance,
        "native_output": native_absolute,
        "native_content": native_content,
        "representation_absolute": representation_absolute,
        "representation_training_effect": representation_training_effect,
    }
    malformed_gate_rows = {
        name: len(frame) for name, frame in required_gate_rows.items() if len(frame) != 1
    }
    if malformed_gate_rows:
        raise RuntimeError(f"Missing or duplicate registered gate rows: {malformed_gate_rows}")
    gate_decisions = symmetry_gate_decisions(
        primary_r2_ci_low=float(native_performance.iloc[0]["ci95_low"]),
        minimum_primary_r2=float(
            context.protocol["evaluation"]["decision_rules"][
                "minimum_primary_source_balanced_r2"
            ]
        ),
        native_output_error_ci_high=float(native_absolute.iloc[0]["ci95_high"]),
        native_output_error_margin=float(
            context.protocol["evaluation"]["decision_rules"][
                "native_symmetry_normalized_error_margin"
            ]
        ),
        primary_minus_initial_r2_ci_low=float(native_content.iloc[0]["ci95_low"]),
        representation_error_ci_high=float(
            representation_absolute.iloc[0]["ci95_high"]
        ),
        representation_error_margin=float(
            context.protocol["evaluation"]["representation_equivariance"][
                "maximum_error_margin"
            ]
        ),
        learned_minus_initial_representation_error_ci_high=float(
            representation_training_effect.iloc[0]["ci95_high"]
        ),
    )
    native_performance_pass = gate_decisions[
        "absolute_native_predictive_utility_ci"
    ]
    native_margin_pass = gate_decisions["native_output_symmetry_margin_ci"]
    native_content_pass = gate_decisions[
        "native_training_content_beyond_initialization_ci"
    ]
    native_probe_behavior_pass = gate_decisions["native_probe_behavior_supported"]
    representation_margin_pass = gate_decisions[
        "strict_representation_error_margin_ci"
    ]
    representation_training_effect_pass = gate_decisions[
        "strict_representation_training_improvement_ci"
    ]
    measured_nuisance_increment = checkpoint_bootstrap[
        checkpoint_bootstrap["comparison_type"]
        == "learned_increment_beyond_nuisance"
    ]
    measured_nuisance_increment_pass = bool(
        len(measured_nuisance_increment)
        and (measured_nuisance_increment["ci95_low"] > 0.0).all()
    )
    governance_payload = load_governance(context.suite_root / "governance" / "status.json")
    governance = submission_readiness(governance_payload)
    summary = {
        "schema": "neurips_laterality_report/v3",
        "profile": context.profile,
        "protocol_digest": context.protocol_digest,
        "context_digest": context.context_digest,
        "cohort_digest": cohort.cohort_digest,
        "selected_folds": list(context.folds),
        "selected_seeds": list(context.seeds),
        "selected_variants": list(context.variants),
        "synthetic_evidence": not context.is_paper,
        "high_coverage_sensitivity_sequences": int(
            high_coverage["sequence_id"].nunique()
        ),
        "paper_run_complete": paper_complete,
        "integrity_gates": {
            "output_oddness": exactness_pass,
            "all_outer_folds_and_registered_seeds": paper_complete,
        },
        "empirical_diagnostics": {
            "absolute_native_predictive_utility_ci": native_performance_pass,
            "native_output_symmetry_margin_ci": native_margin_pass,
            "native_training_content_beyond_initialization_ci": native_content_pass,
            "native_probe_behavior_supported": native_probe_behavior_pass,
            "strict_representation_error_margin_ci": representation_margin_pass,
            "strict_representation_training_improvement_ci": (
                representation_training_effect_pass
            ),
            "strict_checkpoint_representation_equivariance_supported": bool(
                representation_margin_pass and representation_training_effect_pass
            ),
            "training_induced_symmetry_supported_conditional_on_pose_schema": bool(
                native_probe_behavior_pass
                and representation_margin_pass
                and representation_training_effect_pass
            ),
            "constructed_output_exact_for_every_seed": exactness_pass,
            "constructed_absolute_predictive_utility_ci": constructed_absolute_pass,
            "constructed_training_content_beyond_initialization_ci": (
                constructed_content_pass
            ),
            "constructed_repair_contains_useful_learned_content": bool(
                exactness_pass
                and constructed_absolute_pass
                and constructed_content_pass
            ),
            "learned_increment_beyond_measured_nuisance_ci": (
                measured_nuisance_increment_pass
            ),
            "scope_warning": (
                "Strict token equivariance fixes the joint permutation and identity "
                "channel action; all learning claims remain conditional on BlazePose "
                "semantics, preprocessing, architecture, registered seeds, and measured controls."
            ),
            "interpretation": "A failed diagnostic is a result, never a reason to alter the locked protocol.",
        },
        "governance": governance,
        "submission_ready": bool(
            context.is_paper
            and paper_complete
            and exactness_pass
            and governance["ready"]
        ),
        "claim": context.protocol["claim_boundary"]["primary"],
        "unsupported_claims": context.protocol["claim_boundary"]["not_supported"],
    }
    atomic_write_json(report_root / "summary.json", summary)
    return {
        "summary": summary,
        "metrics": metrics,
        "high_coverage_metrics": high_coverage_metrics,
        "seed_metrics": seed_metrics,
        "bootstrap": checkpoint_bootstrap,
        "checkpoint_bootstrap": checkpoint_bootstrap,
        "ensemble_bootstrap": ensemble_bootstrap,
        "native_symmetry_bootstrap": native_symmetry_bootstrap,
        "native_symmetry_by_seed": native_symmetry_by_seed,
        "representation_bootstrap": representation_bootstrap,
        "representation_by_seed": representation_by_seed,
        "averaged": averaged,
    }
