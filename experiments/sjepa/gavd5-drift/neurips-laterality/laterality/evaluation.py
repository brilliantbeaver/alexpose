from __future__ import annotations

import json
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import Ridge

from .artifacts import atomic_write_json, checkpoint_path, evaluation_path, sha256_file
from .config import ExperimentContext, canonical_json_digest, model_config
from .data import PreparedCohort
from .geometry import FULL_MIRROR_PAIRS, anatomical_mirror
from .metrics import source_weights, weighted_r2
from .model import SJEPAGait, valid_patches
from .splitting import get_fold
from .symmetry import (
    strict_equivariance_errors,
    swap_token_joints,
    swap_token_validity,
)
from .training import implementation_digest, load_checkpoint, resolve_device


EXPECTED_LANES: tuple[str, ...] = (
    "learned_single_free",
    "learned_two_pass_odd_free",
    "learned_two_pass_odd_zero",
    "learned_two_pass_even_free",
    "random_single_free",
    "random_two_pass_odd_free",
    "random_two_pass_odd_zero",
    "random_two_pass_even_free",
    "learned_framed_odd_zero",
    "learned_global_pool_free",
    "nuisance_visibility_free",
    "nuisance_acquisition_free",
    "nuisance_annotation_free",
    "nuisance_all_free",
    "learned_single_plus_nuisance_free",
    "oracle_target_components",
)

REPRESENTATION_AUDIT_COLUMNS: tuple[str, ...] = (
    "learned_strict_equivariance_error",
    "learned_strict_equivariance_residual_energy",
    "learned_strict_equivariance_total_energy",
    "learned_strict_equivariance_common_tokens",
    "initial_strict_equivariance_error",
    "initial_strict_equivariance_residual_energy",
    "initial_strict_equivariance_total_energy",
    "initial_strict_equivariance_common_tokens",
)


@dataclass
class WeightedScaler:
    center: bool
    impute_: np.ndarray | None = None
    mean_: np.ndarray | None = None
    scale_: np.ndarray | None = None

    def fit(self, features: np.ndarray, weights: np.ndarray) -> "WeightedScaler":
        values = np.asarray(features, dtype=np.float64)
        weight = np.asarray(weights, dtype=np.float64)
        finite = np.isfinite(values)
        denominators = (finite * weight[:, None]).sum(axis=0)
        weighted_sum = np.where(finite, values, 0.0).T @ weight
        self.impute_ = np.divide(
            weighted_sum,
            denominators,
            out=np.zeros(values.shape[1], dtype=np.float64),
            where=denominators > 0,
        )
        clean = np.where(finite, values, self.impute_[None])
        self.mean_ = (
            np.average(clean, axis=0, weights=weight)
            if self.center
            else np.zeros(clean.shape[1], dtype=np.float64)
        )
        variance = np.average(np.square(clean - self.mean_), axis=0, weights=weight)
        self.scale_ = np.sqrt(np.maximum(variance, 0.0))
        self.scale_[~np.isfinite(self.scale_) | (self.scale_ < 1e-12)] = 1.0
        return self

    def transform(self, features: np.ndarray) -> np.ndarray:
        if self.impute_ is None or self.mean_ is None or self.scale_ is None:
            raise RuntimeError("Scaler is not fitted")
        values = np.asarray(features, dtype=np.float64)
        clean = np.where(np.isfinite(values), values, self.impute_[None])
        return (clean - self.mean_) / self.scale_


@dataclass(frozen=True)
class FeatureLane:
    name: str
    original: np.ndarray
    mirrored: np.ndarray
    constrained_odd: bool
    role: str


@dataclass(frozen=True)
class FeatureBundle:
    lanes: dict[str, FeatureLane]
    representation_audit: dict[str, np.ndarray]


def _encode_tokens(
    encoder,
    coordinates: np.ndarray,
    validity: np.ndarray,
    *,
    segment_length: int,
    batch_size: int = 32,
) -> tuple[np.ndarray, np.ndarray]:
    device = next(encoder.parameters()).device
    encoded: list[np.ndarray] = []
    patches: list[np.ndarray] = []
    encoder.eval()
    with torch.no_grad():
        for start in range(0, len(coordinates), batch_size):
            xyz = torch.as_tensor(
                coordinates[start : start + batch_size], dtype=torch.float32, device=device
            )
            valid = torch.as_tensor(
                validity[start : start + batch_size], dtype=torch.bool, device=device
            )
            patch_valid = valid_patches(valid, segment_length)
            tokens = encoder(xyz, patch_valid).reshape(
                len(xyz), encoder.segments, encoder.joints, encoder.embed_dim
            )
            encoded.append(tokens.cpu().numpy())
            patches.append(patch_valid.cpu().numpy())
    return np.concatenate(encoded), np.concatenate(patches)


def laterality_features(
    tokens: np.ndarray,
    patch_valid: np.ndarray,
    pairs: Sequence[Sequence[int]],
) -> np.ndarray:
    rows: list[np.ndarray] = []
    for sample in range(len(tokens)):
        pieces: list[np.ndarray] = []
        for left, right in pairs:
            # A bilateral contrast must compare the same temporal support. Separate
            # supports would let detector coverage masquerade as laterality.
            common_weight = (
                patch_valid[sample, :, left] & patch_valid[sample, :, right]
            ).astype(np.float64)
            left_mean = (
                np.average(tokens[sample, :, left], axis=0, weights=common_weight)
                if common_weight.sum()
                else np.zeros(tokens.shape[-1])
            )
            right_mean = (
                np.average(tokens[sample, :, right], axis=0, weights=common_weight)
                if common_weight.sum()
                else np.zeros(tokens.shape[-1])
            )
            pieces.extend((left_mean - right_mean, left_mean + right_mean))
        rows.append(np.concatenate(pieces))
    return np.stack(rows).astype(np.float64)


def pooled_features(tokens: np.ndarray, patch_valid: np.ndarray) -> np.ndarray:
    output: list[np.ndarray] = []
    for sample in range(len(tokens)):
        values = tokens[sample][patch_valid[sample]]
        if not len(values):
            raise ValueError("Sequence has no valid token for pooled control")
        output.append(np.concatenate((values.mean(axis=0), values.std(axis=0))))
    return np.stack(output).astype(np.float64)


def _mirror_missingness(features: np.ndarray) -> np.ndarray:
    mirrored = np.asarray(features, dtype=np.float64).copy()
    mirrored[:, 0::2] *= -1.0
    return mirrored


def _patch_missingness_features(
    patch_valid: np.ndarray, pairs: Sequence[Sequence[int]]
) -> np.ndarray:
    rows: list[list[float]] = []
    for sample in range(len(patch_valid)):
        fraction = patch_valid[sample].mean(axis=0)
        row: list[float] = []
        for left, right in pairs:
            common = float(
                (patch_valid[sample, :, left] & patch_valid[sample, :, right]).mean()
            )
            row.extend(
                (
                    float(fraction[left] - fraction[right]),
                    float(fraction[left] + fraction[right]),
                    common,
                )
            )
        rows.append(row)
    return np.asarray(rows, dtype=np.float64)


def _mirror_patch_missingness(features: np.ndarray) -> np.ndarray:
    mirrored = np.asarray(features, dtype=np.float64).copy()
    mirrored[:, 0::3] *= -1.0
    return mirrored


def build_feature_lanes(
    context: ExperimentContext,
    cohort: PreparedCohort,
    checkpoint: dict[str, Any],
) -> FeatureBundle:
    configuration = model_config(context)
    device = resolve_device()
    learned = SJEPAGait(**configuration).to(device)
    learned.load_state_dict(checkpoint["model_state"])
    learned.eval()
    floor = SJEPAGait(**configuration).to(device)
    floor.target_encoder.load_state_dict(checkpoint["initial_target_state"])
    floor.eval()

    mirrored_xyz, mirrored_valid = anatomical_mirror(
        cohort.model_xyz, cohort.model_valid
    )
    segment_length = int(configuration["segment_length"])
    learned_tokens, patch_valid = _encode_tokens(
        learned.target_encoder,
        cohort.model_xyz,
        cohort.model_valid,
        segment_length=segment_length,
    )
    learned_mirror_tokens, mirror_patch_valid = _encode_tokens(
        learned.target_encoder,
        mirrored_xyz,
        mirrored_valid,
        segment_length=segment_length,
    )
    floor_tokens, floor_patch = _encode_tokens(
        floor.target_encoder,
        cohort.model_xyz,
        cohort.model_valid,
        segment_length=segment_length,
    )
    floor_mirror_tokens, floor_mirror_patch = _encode_tokens(
        floor.target_encoder,
        mirrored_xyz,
        mirrored_valid,
        segment_length=segment_length,
    )
    if not np.array_equal(
        mirror_patch_valid, swap_token_validity(patch_valid, FULL_MIRROR_PAIRS)
    ):
        raise AssertionError("Learned-encoder mirror validity violates the group action")
    if not np.array_equal(
        floor_mirror_patch, swap_token_validity(floor_patch, FULL_MIRROR_PAIRS)
    ):
        raise AssertionError("Initial-encoder mirror validity violates the group action")

    representation_config = context.protocol["evaluation"][
        "representation_equivariance"
    ]
    representation_kwargs = {
        "minimum_common_tokens": int(
            representation_config["minimum_common_tokens"]
        ),
        "minimum_representation_energy": float(
            representation_config["minimum_representation_energy"]
        ),
    }
    learned_equivariance = strict_equivariance_errors(
        learned_tokens,
        learned_mirror_tokens,
        patch_valid,
        mirror_patch_valid,
        **representation_kwargs,
    )
    initial_equivariance = strict_equivariance_errors(
        floor_tokens,
        floor_mirror_tokens,
        floor_patch,
        floor_mirror_patch,
        **representation_kwargs,
    )
    representation_audit = {
        "learned_strict_equivariance_error": np.asarray(
            [item.value for item in learned_equivariance], dtype=np.float64
        ),
        "learned_strict_equivariance_residual_energy": np.asarray(
            [item.residual_energy for item in learned_equivariance], dtype=np.float64
        ),
        "learned_strict_equivariance_total_energy": np.asarray(
            [item.representation_energy for item in learned_equivariance],
            dtype=np.float64,
        ),
        "learned_strict_equivariance_common_tokens": np.asarray(
            [item.common_token_count for item in learned_equivariance], dtype=np.int64
        ),
        "initial_strict_equivariance_error": np.asarray(
            [item.value for item in initial_equivariance], dtype=np.float64
        ),
        "initial_strict_equivariance_residual_energy": np.asarray(
            [item.residual_energy for item in initial_equivariance], dtype=np.float64
        ),
        "initial_strict_equivariance_total_energy": np.asarray(
            [item.representation_energy for item in initial_equivariance],
            dtype=np.float64,
        ),
        "initial_strict_equivariance_common_tokens": np.asarray(
            [item.common_token_count for item in initial_equivariance], dtype=np.int64
        ),
    }
    pairs = context.protocol["target"]["pairs"]
    learned_free = laterality_features(learned_tokens, patch_valid, pairs)
    learned_mirror = laterality_features(
        learned_mirror_tokens, mirror_patch_valid, pairs
    )
    floor_free = laterality_features(floor_tokens, floor_patch, pairs)
    floor_mirror = laterality_features(floor_mirror_tokens, floor_mirror_patch, pairs)

    root_two = np.sqrt(2.0)
    learned_odd = (learned_free - learned_mirror) / root_two
    learned_even = (learned_free + learned_mirror) / root_two
    floor_odd = (floor_free - floor_mirror) / root_two
    floor_even = (floor_free + floor_mirror) / root_two

    # Reynolds averaging makes the exploratory wrapper exactly equivariant for
    # any encoder. Its behavior is imposed and is not emergence evidence.
    framed_tokens = 0.5 * (
        learned_tokens + swap_token_joints(learned_mirror_tokens)
    )
    if not np.array_equal(patch_valid, swap_token_validity(mirror_patch_valid)):
        raise AssertionError("Mirrored patch validity does not obey the group action")
    framed_mirror_tokens = swap_token_joints(framed_tokens)
    framed_mirror_patch = swap_token_validity(patch_valid)
    framed_free = laterality_features(framed_tokens, patch_valid, pairs)
    framed_mirror = laterality_features(
        framed_mirror_tokens, framed_mirror_patch, pairs
    )

    global_context = pooled_features(learned_tokens, patch_valid)
    global_context_mirror = pooled_features(
        learned_mirror_tokens, mirror_patch_valid
    )
    conditions = list(context.protocol["data"]["conditions"])
    condition_values = cohort.table["condition"].astype(str).to_numpy()
    annotation = np.stack(
        [(condition_values == condition).astype(np.float64) for condition in conditions],
        axis=1,
    )
    quality_numeric = cohort.table[
        ["raw_frame_count", "authorized_coverage", "authorized_patch_count"]
    ].to_numpy(dtype=np.float64)
    version_order = (
        sorted(
            context.protocol["data"]["extraction_provenance"][
                "extraction_version_counts"
            ]
        )
        if context.is_paper
        else ["synthetic_smoke_v1"]
    )
    extraction_values = cohort.table["extraction_version"].astype(str).to_numpy()
    version_features = np.stack(
        [(extraction_values == version).astype(np.float64) for version in version_order],
        axis=1,
    )
    quality = np.concatenate((quality_numeric, version_features), axis=1)
    model_missingness = _patch_missingness_features(patch_valid, pairs)
    combined_missingness = np.concatenate(
        (cohort.missingness, model_missingness), axis=1
    )
    combined_missingness_mirror = np.concatenate(
        (
            _mirror_missingness(cohort.missingness),
            _mirror_patch_missingness(model_missingness),
        ),
        axis=1,
    )
    nuisance_all = np.concatenate(
        (combined_missingness, quality, annotation), axis=1
    )
    nuisance_all_mirror = np.concatenate(
        (combined_missingness_mirror, quality, annotation), axis=1
    )
    lanes = {
        "learned_single_free": FeatureLane(
            "learned_single_free",
            learned_free,
            learned_mirror,
            False,
            "primary_single_pass_free_probe",
        ),
        "learned_two_pass_odd_free": FeatureLane(
            "learned_two_pass_odd_free",
            learned_odd,
            -learned_odd,
            False,
            "two_pass_odd_features_free_readout",
        ),
        "learned_two_pass_odd_zero": FeatureLane(
            "learned_two_pass_odd_zero",
            learned_odd,
            -learned_odd,
            True,
            "constructed_antisymmetric_repair_control",
        ),
        "learned_two_pass_even_free": FeatureLane(
            "learned_two_pass_even_free",
            learned_even,
            learned_even,
            False,
            "two_pass_even_falsification_control",
        ),
        "random_single_free": FeatureLane(
            "random_single_free",
            floor_free,
            floor_mirror,
            False,
            "paired_initialization_single_pass_floor",
        ),
        "random_two_pass_odd_free": FeatureLane(
            "random_two_pass_odd_free",
            floor_odd,
            -floor_odd,
            False,
            "paired_initialization_two_pass_odd_free_floor",
        ),
        "random_two_pass_odd_zero": FeatureLane(
            "random_two_pass_odd_zero",
            floor_odd,
            -floor_odd,
            True,
            "paired_initialization_constructed_odd_floor",
        ),
        "random_two_pass_even_free": FeatureLane(
            "random_two_pass_even_free",
            floor_even,
            floor_even,
            False,
            "paired_initialization_even_falsification_control",
        ),
        "learned_framed_odd_zero": FeatureLane(
            "learned_framed_odd_zero",
            (framed_free - framed_mirror) / root_two,
            (framed_mirror - framed_free) / root_two,
            True,
            "exploratory_analytic_reynolds_wrapper",
        ),
        "learned_global_pool_free": FeatureLane(
            "learned_global_pool_free",
            global_context,
            global_context_mirror,
            False,
            "descriptive_global_context_can_retain_side_information",
        ),
        "nuisance_visibility_free": FeatureLane(
            "nuisance_visibility_free",
            combined_missingness,
            combined_missingness_mirror,
            False,
            "measured_visibility_and_missingness_control",
        ),
        "nuisance_acquisition_free": FeatureLane(
            "nuisance_acquisition_free",
            quality,
            quality.copy(),
            False,
            "measured_pose_acquisition_control",
        ),
        "nuisance_annotation_free": FeatureLane(
            "nuisance_annotation_free",
            annotation,
            annotation.copy(),
            False,
            "dataset_annotation_control_not_diagnosis",
        ),
        "nuisance_all_free": FeatureLane(
            "nuisance_all_free",
            nuisance_all,
            nuisance_all_mirror,
            False,
            "combined_measured_nuisance_control",
        ),
        "learned_single_plus_nuisance_free": FeatureLane(
            "learned_single_plus_nuisance_free",
            np.concatenate((learned_free, nuisance_all), axis=1),
            np.concatenate((learned_mirror, nuisance_all_mirror), axis=1),
            False,
            "incremental_learned_features_plus_measured_nuisance",
        ),
        "oracle_target_components": FeatureLane(
            "oracle_target_components",
            cohort.pair_contrasts,
            -cohort.pair_contrasts,
            False,
            "target_component_self_consistency_oracle_not_baseline",
        ),
    }
    if set(lanes) != set(EXPECTED_LANES):
        raise AssertionError("Feature-lane registry drifted from the evaluation contract")
    return FeatureBundle(lanes=lanes, representation_audit=representation_audit)


def _fit_readout(
    train_features: np.ndarray,
    train_target: np.ndarray,
    train_groups: np.ndarray,
    *,
    alpha: float,
    constrained_odd: bool,
) -> tuple[WeightedScaler, Ridge]:
    if constrained_odd and not np.isfinite(train_features).all():
        raise ValueError("Constrained odd read-outs require finite features; imputation can break oddness")
    weights = source_weights(train_groups)
    scaler = WeightedScaler(center=not constrained_odd).fit(train_features, weights)
    model = Ridge(alpha=float(alpha), fit_intercept=not constrained_odd)
    model.fit(scaler.transform(train_features), train_target, sample_weight=weights)
    return scaler, model


def select_alpha(
    features: np.ndarray,
    target: np.ndarray,
    groups: np.ndarray,
    inner_folds: list[dict[str, Any]],
    alphas: Sequence[float],
    *,
    constrained_odd: bool,
) -> tuple[float, dict[str, float]]:
    scores: dict[str, float] = {}
    group_values = np.asarray(groups).astype(str)
    for alpha in alphas:
        fold_scores: list[float] = []
        for split in inner_folds:
            train_mask = np.isin(group_values, split["train_sources"])
            validation_mask = np.isin(group_values, split["validation_sources"])
            if train_mask.sum() == 0 or validation_mask.sum() == 0:
                raise AssertionError("Empty inner read-out partition")
            scaler, readout = _fit_readout(
                features[train_mask],
                target[train_mask],
                group_values[train_mask],
                alpha=float(alpha),
                constrained_odd=constrained_odd,
            )
            prediction = readout.predict(scaler.transform(features[validation_mask]))
            fold_scores.append(
                weighted_r2(
                    target[validation_mask],
                    prediction,
                    source_weights(group_values[validation_mask]),
                )
            )
        finite = np.asarray(fold_scores)[np.isfinite(fold_scores)]
        scores[str(alpha)] = float(finite.mean()) if len(finite) else float("-inf")
    best = max(map(float, alphas), key=lambda alpha: (scores[str(alpha)], -alpha))
    return float(best), scores


def evaluation_metadata_path(csv_path: Path) -> Path:
    return csv_path.with_suffix(".json")


def evaluation_result_digest(frame: pd.DataFrame) -> str:
    stable = frame.sort_values(["lane", "sequence_id"]).reset_index(drop=True)
    return canonical_json_digest(
        stable[
            [
                "lane",
                "sequence_id",
                "prediction",
                "mirror_prediction",
                *REPRESENTATION_AUDIT_COLUMNS,
            ]
        ].to_dict(orient="records")
    )


def validate_evaluation_frame(
    frame: pd.DataFrame,
    context: ExperimentContext,
    cohort: PreparedCohort,
    fold_payload: dict[str, Any],
    fold: int,
    seed: int,
    variant: str,
) -> None:
    required = {
        "sequence_id",
        "video_id",
        "condition",
        "target",
        "fold",
        "seed",
        "variant",
        "lane",
        "prediction",
        "mirror_prediction",
        "source_weight",
        *REPRESENTATION_AUDIT_COLUMNS,
    }
    if missing := required - set(frame.columns):
        raise RuntimeError(f"Evaluation CSV lacks required columns: {sorted(missing)}")
    expected_sequences = set(map(str, fold_payload["test_sequence_ids"]))
    if len(frame) != len(expected_sequences) * len(EXPECTED_LANES):
        raise RuntimeError("Evaluation CSV has incomplete or extra sequence/lane rows")
    if frame.duplicated(["sequence_id", "lane"]).any():
        raise RuntimeError("Evaluation CSV has duplicate sequence/lane rows")
    if set(frame["lane"].astype(str)) != set(EXPECTED_LANES):
        raise RuntimeError("Evaluation CSV lane coverage is invalid")
    for _, lane_frame in frame.groupby("lane"):
        if set(lane_frame["sequence_id"].astype(str)) != expected_sequences:
            raise RuntimeError("A lane does not cover every expected test sequence")
    if set(frame["video_id"].astype(str)) != set(map(str, fold_payload["test_sources"])):
        raise RuntimeError("Evaluation CSV source coverage differs from the outer test fold")
    if set(frame["fold"].astype(int)) != {int(fold)}:
        raise RuntimeError("Evaluation CSV fold column is inconsistent")
    if set(frame["seed"].astype(int)) != {int(seed)}:
        raise RuntimeError("Evaluation CSV seed column is inconsistent")
    if set(frame["variant"].astype(str)) != {str(variant)}:
        raise RuntimeError("Evaluation CSV variant column is inconsistent")

    truth = cohort.table.set_index("sequence_id")[["video_id", "condition", "target"]]
    aligned = truth.loc[frame["sequence_id"].astype(str)]
    if list(aligned["video_id"].astype(str)) != list(frame["video_id"].astype(str)):
        raise RuntimeError("Evaluation CSV sequence/source alignment is invalid")
    if list(aligned["condition"].astype(str)) != list(frame["condition"].astype(str)):
        raise RuntimeError("Evaluation CSV annotation alignment is invalid")
    if not np.allclose(
        aligned["target"].to_numpy(dtype=np.float64),
        frame["target"].to_numpy(dtype=np.float64),
        rtol=0.0,
        atol=1e-12,
    ):
        raise RuntimeError("Evaluation CSV target values differ from the locked cohort")
    for _, lane_frame in frame.groupby("lane"):
        totals = lane_frame.groupby("video_id")["source_weight"].sum().to_numpy()
        if not np.allclose(totals, 1.0, rtol=0.0, atol=1e-10):
            raise RuntimeError("Evaluation CSV does not give each source total weight one")
    for column in REPRESENTATION_AUDIT_COLUMNS:
        if (frame.groupby("sequence_id")[column].nunique(dropna=False) != 1).any():
            raise RuntimeError(
                f"Representation audit field {column} differs across read-out lanes"
            )
    minimum_common = int(
        context.protocol["evaluation"]["representation_equivariance"][
            "minimum_common_tokens"
        ]
    )
    for prefix in ("learned", "initial"):
        error = frame[f"{prefix}_strict_equivariance_error"].to_numpy(
            dtype=np.float64
        )
        residual = frame[
            f"{prefix}_strict_equivariance_residual_energy"
        ].to_numpy(dtype=np.float64)
        total = frame[f"{prefix}_strict_equivariance_total_energy"].to_numpy(
            dtype=np.float64
        )
        common = frame[f"{prefix}_strict_equivariance_common_tokens"].to_numpy(
            dtype=np.int64
        )
        if (
            not np.isfinite(error).all()
            or not np.isfinite(residual).all()
            or not np.isfinite(total).all()
            or np.any(error < -1e-12)
            or np.any(error > 2.0 + 1e-10)
            or np.any(residual < 0.0)
            or np.any(total <= 0.0)
            or np.any(common < minimum_common)
            or not np.allclose(error, residual / total, rtol=1e-10, atol=1e-12)
        ):
            raise RuntimeError(
                f"Invalid {prefix} strict representation-equivariance audit values"
            )


def evaluate_fold(
    context: ExperimentContext,
    cohort: PreparedCohort,
    splits: dict[str, Any],
    fold: int,
    seed: int,
    variant: str,
    *,
    reuse_valid: bool = True,
) -> pd.DataFrame:
    csv_path = evaluation_path(context.artifact_root, variant, fold, seed)
    metadata_path = evaluation_metadata_path(csv_path)
    checkpoint_file = checkpoint_path(context.artifact_root, variant, fold, seed)
    checkpoint = load_checkpoint(
        checkpoint_file,
        context,
        cohort,
        splits,
        fold,
        seed,
        variant,
    )
    checkpoint_sha256 = sha256_file(checkpoint_file)
    fold_payload = get_fold(splits, fold)
    if reuse_valid and csv_path.exists() and metadata_path.exists():
        metadata = json.loads(metadata_path.read_text())
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
            "checkpoint_sha256": checkpoint_sha256,
            "model_state_digest": checkpoint["model_state_digest"],
        }
        if all(metadata.get(key) == value for key, value in expected.items()):
            if metadata.get("csv_sha256") != sha256_file(csv_path):
                raise RuntimeError("Cached evaluation CSV is corrupt")
            cached = pd.read_csv(csv_path)
            validate_evaluation_frame(
                cached, context, cohort, fold_payload, fold, seed, variant
            )
            if metadata.get("result_digest") != evaluation_result_digest(cached):
                raise RuntimeError("Cached evaluation result digest mismatch")
            return cached
        raise RuntimeError("Cached evaluation lineage mismatch; choose a clean artifact root")
    feature_bundle = build_feature_lanes(context, cohort, checkpoint)
    lanes = feature_bundle.lanes
    groups = cohort.table["video_id"].astype(str).to_numpy()
    train_mask = np.isin(groups, fold_payload["train_sources"])
    test_mask = np.isin(groups, fold_payload["test_sources"])
    if (train_mask & test_mask).any() or not np.all(train_mask | test_mask):
        raise AssertionError("Outer evaluation partition is invalid")
    targets = cohort.targets
    source_target = (
        pd.DataFrame(
            {
                "video_id": groups[train_mask],
                "target": targets[train_mask],
            }
        )
        .groupby("video_id")["target"]
        .median()
        .to_numpy()
    )
    train_iqr = float(
        np.quantile(source_target, 0.75) - np.quantile(source_target, 0.25)
    )
    neutral_threshold = float(
        context.protocol["evaluation"]["neutral_band_iqr_fraction"] * train_iqr
    )
    train_scale = float(
        np.sqrt(
            np.average(
                np.square(
                    targets[train_mask]
                    - np.average(
                        targets[train_mask], weights=source_weights(groups[train_mask])
                    )
                ),
                weights=source_weights(groups[train_mask]),
            )
        )
    )
    rows: list[pd.DataFrame] = []
    alpha_diagnostics: dict[str, Any] = {}
    for lane_name, lane in lanes.items():
        if lane.constrained_odd and (
            not np.isfinite(lane.original).all()
            or not np.isfinite(lane.mirrored).all()
        ):
            raise ValueError(f"Constrained lane {lane_name} contains non-finite features")
        alpha, scores = select_alpha(
            lane.original[train_mask],
            targets[train_mask],
            groups[train_mask],
            fold_payload["inner_readout_folds"],
            context.protocol["evaluation"]["ridge_alphas"],
            constrained_odd=lane.constrained_odd,
        )
        scaler, readout = _fit_readout(
            lane.original[train_mask],
            targets[train_mask],
            groups[train_mask],
            alpha=alpha,
            constrained_odd=lane.constrained_odd,
        )
        prediction = readout.predict(scaler.transform(lane.original[test_mask]))
        mirror_prediction = readout.predict(
            scaler.transform(lane.mirrored[test_mask])
        )
        if lane.constrained_odd and not np.allclose(
            prediction + mirror_prediction, 0.0, rtol=0.0, atol=2e-7
        ):
            raise AssertionError(f"Constrained lane {lane_name} is not output-odd")
        selected_table = cohort.table.loc[test_mask].reset_index(drop=True)
        lane_frame = selected_table[
            ["sequence_id", "video_id", "condition", "target"]
        ].copy()
        lane_frame["fold"] = fold
        lane_frame["seed"] = seed
        lane_frame["variant"] = variant
        lane_frame["lane"] = lane_name
        lane_frame["role"] = lane.role
        lane_frame["prediction"] = prediction
        lane_frame["mirror_prediction"] = mirror_prediction
        lane_frame["alpha"] = alpha
        lane_frame["neutral_threshold"] = neutral_threshold
        lane_frame["training_target_scale"] = train_scale
        for column, values in feature_bundle.representation_audit.items():
            lane_frame[column] = values[test_mask]
        lane_frame["source_weight"] = source_weights(
            lane_frame["video_id"].astype(str).to_numpy()
        )
        rows.append(lane_frame)
        alpha_diagnostics[lane_name] = {"selected": alpha, "inner_scores": scores}
    output = pd.concat(rows, ignore_index=True)
    validate_evaluation_frame(
        output, context, cohort, fold_payload, fold, seed, variant
    )
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{csv_path.name}.", suffix=".tmp", dir=csv_path.parent
    )
    os.close(descriptor)
    try:
        output.to_csv(temporary_name, index=False)
        os.replace(temporary_name, csv_path)
    except Exception:
        Path(temporary_name).unlink(missing_ok=True)
        raise
    metadata = {
        "schema": "neurips_laterality_evaluation/v3",
        "protocol_digest": context.protocol_digest,
        "context_digest": context.context_digest,
        "cohort_digest": cohort.cohort_digest,
        "split_digest": splits["split_digest"],
        "checkpoint_lineage_digest": checkpoint["lineage_digest"],
        "checkpoint_sha256": checkpoint_sha256,
        "model_state_digest": checkpoint["model_state_digest"],
        "fold": fold,
        "seed": seed,
        "variant": variant,
        "implementation_digest": implementation_digest(),
        "train_sources": sorted(fold_payload["train_sources"]),
        "test_sources": sorted(fold_payload["test_sources"]),
        "alpha_diagnostics": alpha_diagnostics,
        "lanes": list(EXPECTED_LANES),
        "row_count": int(len(output)),
        "csv_sha256": sha256_file(csv_path),
        "result_digest": evaluation_result_digest(pd.read_csv(csv_path)),
    }
    atomic_write_json(metadata_path, metadata)
    return output


def evaluate_selected(
    context: ExperimentContext,
    cohort: PreparedCohort,
    splits: dict[str, Any],
) -> pd.DataFrame:
    outputs: list[pd.DataFrame] = []
    for variant in context.variants:
        for fold in context.folds:
            for seed in context.seeds:
                outputs.append(
                    evaluate_fold(context, cohort, splits, fold, seed, variant)
                )
    return pd.concat(outputs, ignore_index=True)
