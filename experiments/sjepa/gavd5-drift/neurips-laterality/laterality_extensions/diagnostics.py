"""Read-only diagnostics for planning the next laterality experiments.

The registered implementation is deliberately imported rather than edited.
These helpers do not train models, rewrite reports, or update the protocol.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Mapping

import numpy as np


def temporal_pooling_example() -> dict[str, np.ndarray]:
    """Return two constructed motions with equal means and opposite contrasts.

    These are one-dimensional toy trajectories, not human gait observations.
    Each trajectory has four equally spaced samples and two named sides.
    """
    first = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
    trajectories = np.stack([first, first[:, ::-1]])
    median_speed = np.median(np.abs(np.diff(trajectories, axis=1)), axis=1)
    denominator = median_speed.sum(axis=1)
    contrasts = (median_speed[:, 0] - median_speed[:, 1]) / denominator
    return {
        "trajectories": trajectories,
        "mean_position": trajectories.mean(axis=1),
        "median_speed": median_speed,
        "motion_contrast": contrasts,
    }


def reconstruct_processed_target(
    coordinates: np.ndarray,
    validity: np.ndarray,
    target_config: Mapping[str, Any],
) -> np.ndarray:
    """Apply the registered formula to the processed model inputs.

    This changes the measurement path: coordinates may have been interpolated,
    normalized, and resized before reaching this function. Uniform sample times
    describe relative clip position, not physical seconds. A constant time
    factor cancels in each speed contrast apart from the numerical epsilon.
    Missing results remain NaN; the function does not relax eligibility rules.
    """
    from laterality.geometry import paired_valid_target

    xyz = np.asarray(coordinates)
    valid = np.asarray(validity, dtype=bool)
    if xyz.ndim != 4 or xyz.shape[-2:] != (33, 3):
        raise ValueError("Coordinates must have shape [sequences, frames, 33, 3].")
    if valid.shape != xyz.shape[:-1]:
        raise ValueError("Validity must match the sequence, frame, and joint axes.")
    return np.array(
        [
            paired_valid_target(
                x,
                v,
                target_config["pairs"],
                int(target_config["minimum_common_transitions_per_pair"]),
                int(target_config["minimum_usable_pairs"]),
                float(target_config["epsilon"]),
            ).value
            for x, v in zip(xyz, valid)
        ],
        dtype=np.float64,
    )


def summarize_reconstruction(
    original: np.ndarray,
    reconstructed: np.ndarray,
    source_ids: np.ndarray,
) -> dict[str, int | float | str]:
    """Describe agreement on finite pairs with equal total weight per source.

    Direct R² here compares two calculations; it is not held-out model
    performance and is not an upper bound on a representation's information.
    The finite overlap is always reported because recomputation can fail for
    some inputs even when the original target was computable.
    """
    first = np.asarray(original, dtype=np.float64)
    second = np.asarray(reconstructed, dtype=np.float64)
    groups = np.asarray(source_ids).astype(str)
    if first.ndim != 1 or second.shape != first.shape or groups.shape != first.shape:
        raise ValueError("Targets and source identifiers must be matching vectors.")
    if np.any(np.char.strip(groups) == ""):
        raise ValueError("Every sequence needs a nonempty source identifier.")
    finite = np.isfinite(first) & np.isfinite(second)
    if not finite.any():
        raise ValueError("No finite paired calculations are available.")
    y, yhat, kept_groups = first[finite], second[finite], groups[finite]
    unique, inverse, counts = np.unique(kept_groups, return_inverse=True, return_counts=True)
    weights = 1.0 / counts[inverse]
    weights /= weights.sum()
    mean_y, mean_yhat = np.sum(weights * y), np.sum(weights * yhat)
    variance = float(np.sum(weights * (y - mean_y) ** 2))
    other_variance = float(np.sum(weights * (yhat - mean_yhat) ** 2))
    mse = float(np.sum(weights * (y - yhat) ** 2))
    correlation = (
        float(np.sum(weights * (y - mean_y) * (yhat - mean_yhat)))
        / np.sqrt(variance * other_variance)
        if variance > 0 and other_variance > 0
        else float("nan")
    )
    return {
        "analysis_kind": "exploratory input-reconstruction diagnostic",
        "input_sequences": int(len(first)),
        "input_sources": int(len(np.unique(groups))),
        "finite_sequences": int(finite.sum()),
        "finite_sources": int(len(unique)),
        "excluded_sequences": int((~finite).sum()),
        "direct_agreement_r2": 1.0 - mse / variance if variance > 0 else float("nan"),
        "weighted_correlation": float(correlation),
        "weighted_mae": float(np.sum(weights * np.abs(y - yhat))),
        "weighted_original_target_sd": float(np.sqrt(variance)),
        "weighted_sign_agreement": float(np.sum(weights * (np.sign(y) == np.sign(yhat)))),
        "maximum_absolute_difference": float(np.max(np.abs(y - yhat))),
    }


def read_registered_results(context: Any) -> dict[str, Any]:
    """Read selected stored report rows after checking their declared context.

    This checks report/cohort metadata and table structure. It does not rerun
    the bootstrap, validate every checkpoint, or independently reconstruct
    every prediction. Its output is retained evidence, not new evaluation.
    """
    report_root = Path(context.artifact_root) / "report"
    summary_path = report_root / "summary.json"
    if not summary_path.is_file():
        return {"status": "registered report unavailable", "rows": []}
    summary = json.loads(summary_path.read_text())
    if (
        summary.get("profile") != "paper"
        or summary.get("synthetic_evidence") is not False
        or summary.get("paper_run_complete") is not True
        or summary.get("protocol_digest") != context.protocol_digest
        or summary.get("context_digest") != context.context_digest
    ):
        raise ValueError("The report is incomplete, synthetic, or from another context.")
    metadata = json.loads((Path(context.artifact_root) / "cohort" / "metadata.json").read_text())
    if any(metadata.get(key) != summary.get(key) for key in ("cohort_digest", "protocol_digest", "context_digest")):
        raise ValueError("Stored report and cohort metadata disagree.")
    requested = [
        ("checkpoint_source_bootstrap.csv", "absolute_primary", "vanilla", "Prediction from pretrained features", "R²"),
        ("checkpoint_source_bootstrap.csv", "primary_training_content", "vanilla", "Pretrained minus untrained prediction", "change in R²"),
        ("checkpoint_source_bootstrap.csv", "reflection_minus_vanilla_primary", "reflection_augmented", "Effect of mirrored training clips on prediction", "change in R²"),
        ("strict_representation_equivariance_source_bootstrap.csv", "learned_minus_initial_strict_equivariance", "vanilla", "Effect of pretraining on strict token error", "change in error"),
        ("strict_representation_equivariance_source_bootstrap.csv", "reflection_minus_vanilla_strict_equivariance", "reflection_augmented", "Effect of mirrored training clips on strict token error", "change in error"),
    ]
    rows = []
    for filename, comparison, variant, label, unit in requested:
        with (report_root / filename).open(newline="") as handle:
            matches = [row for row in csv.DictReader(handle) if row["comparison_type"] == comparison and row["variant_a"] == variant]
        if len(matches) != 1:
            raise ValueError(f"Expected one registered row for {comparison} / {variant}.")
        selected = matches[0]
        numbers = [float(selected[key]) for key in ("estimate", "ci95_low", "ci95_high")]
        if not np.isfinite(numbers).all() or numbers[1] > numbers[2]:
            raise ValueError("A selected report interval is invalid.")
        rows.append({"question": label, "unit": unit, "estimate": numbers[0], "ci95_low": numbers[1], "ci95_high": numbers[2]})
    attrition = metadata["attrition"]
    return {
        "status": "retained real-data results; no new model evaluation",
        "sequences": int(attrition["accepted_sequences"]),
        "sources": int(attrition["accepted_sources"]),
        "scope": "held-out source videos within GAVD; conditional on the fitted pipeline",
        "rows": rows,
    }
