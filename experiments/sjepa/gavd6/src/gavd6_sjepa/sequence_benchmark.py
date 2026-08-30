"""Runnable sequence-level benchmark gate for Latent Laterality."""

from __future__ import annotations

import json
import math
import os
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from pathlib import PurePosixPath
from typing import Sequence

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

from .amass_core11_jepa import MIRROR_PAIRS
from .latent_laterality import (
    SequenceGaugeConfig,
    SequenceGaugeDraw,
    TwoStateDurationModel,
    apply_block_correction,
    apply_sequence_draw,
    block_odd_even_motion_targets,
    calibrate_duration_temperature,
    continuity_edge_logits,
    generate_sequence_draw,
    path_hamming_up_to_global_flip,
    relative_reduction,
    run_length_decode,
    semantic_permute,
    switch_f1,
)


@dataclass(frozen=True)
class BenchmarkExample:
    identity: str
    sequence_id: str
    split: str
    chart_label: int
    coordinates: np.ndarray
    valid: np.ndarray
    pelvis_world: np.ndarray
    draw: object
    path: np.ndarray
    corrupted: dict


def _atomic_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def _atomic_csv(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    frame.to_csv(temporary, index=False)
    os.replace(temporary, path)


def synthetic_travel_motion(
    identity_index: int,
    *,
    frames: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Side-asymmetric traveling motion with ambiguous crossing/turn boundaries."""

    rng = np.random.default_rng(seed + identity_index * 1009)
    time = np.arange(frames, dtype=np.float64)
    phase = 2.0 * np.pi * time / 16.0 + rng.uniform(-0.2, 0.2)
    angle = 0.45 * np.sin(2.0 * np.pi * time / 96.0 + rng.uniform(-0.3, 0.3))
    speed = 0.018 * (0.55 + 0.45 * np.square(np.sin(2.0 * np.pi * time / 64.0)))
    steps = np.stack([speed * np.cos(angle), speed * np.sin(angle), np.zeros_like(speed)], axis=1)
    pelvis_world = np.cumsum(steps, axis=0).astype(np.float32)
    coordinates = np.zeros((frames, 11, 3), dtype=np.float32)
    coordinates[:, 0] = pelvis_world
    asymmetry = rng.uniform(0.25, 0.55)
    for pair_index, (left, right) in enumerate(MIRROR_PAIRS):
        height = 0.18 * (5 - pair_index)
        left_amplitude = (0.025 + pair_index * 0.008) * (1.0 + asymmetry)
        right_amplitude = (0.025 + pair_index * 0.008) * (1.0 - asymmetry)
        pair_phase = phase + 0.16 * pair_index
        coordinates[:, left, 0] = pelvis_world[:, 0] + left_amplitude * np.sin(pair_phase)
        coordinates[:, right, 0] = pelvis_world[:, 0] - right_amplitude * np.sin(pair_phase)
        coordinates[:, left, 1] = height + 0.025 * np.cos(pair_phase)
        coordinates[:, right, 1] = height - 0.025 * np.cos(pair_phase)
        coordinates[:, left, 2] = 0.10 + 0.01 * pair_index
        coordinates[:, right, 2] = -0.10 - 0.01 * pair_index
    coordinates += rng.normal(0.0, 0.0008, coordinates.shape).astype(np.float32)
    valid = np.ones((frames, 11), dtype=bool)
    return coordinates, valid, pelvis_world


def make_synthetic_examples(
    *,
    identities: int,
    frames: int,
    seed: int,
    config: SequenceGaugeConfig,
) -> list[BenchmarkExample]:
    """Create globally chart-paired examples with identical observations."""

    examples: list[BenchmarkExample] = []
    for identity_index in range(identities):
        identity = f"synthetic-{identity_index:04d}"
        sequence_id = f"motion-{identity_index:04d}"
        coordinates, valid, pelvis = synthetic_travel_motion(
            identity_index, frames=frames, seed=seed
        )
        base_draw, path = generate_sequence_draw(
            coordinates,
            valid,
            sequence_id=sequence_id,
            identity=identity,
            split="synthetic",
            corruption_draw=0,
            seed=seed,
            config=config,
            pelvis_world=pelvis,
        )
        # (z, chart=0) and (Pz, chart=1) produce exactly the same unanchored
        # observation but opposite chart labels. Splits are by paired identity.
        for chart_label in (0, 1):
            source_coordinates = (
                coordinates if chart_label == 0 else semantic_permute(coordinates)
            )
            source_valid = valid if chart_label == 0 else semantic_permute(valid)
            draw = replace(base_draw, latent_chart_bit=chart_label)
            corrupted = apply_sequence_draw(
                source_coordinates,
                source_valid,
                draw,
                config=config,
            )
            examples.append(
                BenchmarkExample(
                    identity=identity,
                    sequence_id=f"{sequence_id}:chart-{chart_label}",
                    split="synthetic",
                    chart_label=chart_label,
                    coordinates=source_coordinates,
                    valid=source_valid,
                    pelvis_world=pelvis,
                    draw=draw,
                    path=path.copy(),
                    corrupted=corrupted,
                )
            )
    return examples


def make_manifest_examples(
    gauge_manifest: Path,
    tensor_root: Path,
    *,
    config: SequenceGaugeConfig,
) -> list[BenchmarkExample]:
    """Load only train/validation tensors named by a persistent draw manifest."""

    frame = pd.read_csv(gauge_manifest)
    required = {
        "tensor_relative_path",
        "identity",
        "split",
        "corruption_draw",
        "path_family",
        "gauge_path_rle",
        "switch_frames",
        "semantic_scope",
        "sensor_reflection_bit",
        "latent_chart_bit",
        "nuisance_boundary_frames",
        "occlusion_seed",
        "noise_seed",
        "generator_version",
    }
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"Gauge manifest is missing {sorted(missing)}")
    if "test" in set(frame.split):
        frame = frame.loc[frame.split != "test"].copy()
    root = tensor_root.resolve()
    examples = []
    for row in frame.to_dict(orient="records"):
        relative = PurePosixPath(str(row["tensor_relative_path"]))
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError(f"Unsafe tensor path: {relative}")
        path = root.joinpath(*relative.parts).resolve()
        path.relative_to(root)
        with np.load(path, allow_pickle=False) as archive:
            coordinates = np.asarray(archive["coordinates"], dtype=np.float32)
            valid = np.asarray(archive["valid"], dtype=bool)
            pelvis = (
                np.asarray(archive["pelvis_world_m"], dtype=np.float32)
                if "pelvis_world_m" in archive.files
                else coordinates[:, 0].copy()
            )
        draw = SequenceGaugeDraw(
            sequence_id=str(row.get("sequence_id", row["tensor_relative_path"])),
            identity=str(row["identity"]),
            split=str(row["split"]),
            corruption_draw=int(row["corruption_draw"]),
            path_family=str(row["path_family"]),
            gauge_path_rle=tuple(
                (int(bit), int(count)) for bit, count in json.loads(row["gauge_path_rle"])
            ),
            switch_frames=tuple(map(int, json.loads(row["switch_frames"]))),
            semantic_scope=str(row["semantic_scope"]),
            sensor_reflection_bit=int(row["sensor_reflection_bit"]),
            latent_chart_bit=int(row["latent_chart_bit"]),
            nuisance_boundary_frames=tuple(
                map(int, json.loads(row["nuisance_boundary_frames"]))
            ),
            occlusion_seed=int(row["occlusion_seed"]),
            noise_seed=int(row["noise_seed"]),
            generator_version=str(row["generator_version"]),
        )
        block_path = run_length_decode(draw.gauge_path_rle)
        corrupted = apply_sequence_draw(
            coordinates, valid, draw, config=config
        )
        examples.append(
            BenchmarkExample(
                identity=str(row["identity"]),
                sequence_id=f"{draw.sequence_id}:draw-{draw.corruption_draw}",
                split=str(row["split"]),
                chart_label=draw.latent_chart_bit,
                coordinates=coordinates,
                valid=valid,
                pelvis_world=pelvis,
                draw=draw,
                path=block_path,
                corrupted=corrupted,
            )
        )
    if not examples:
        raise ValueError("Gauge manifest produced no train/validation examples")
    return examples


def _identity_partition(examples: Sequence[BenchmarkExample]) -> tuple[list, list, list]:
    declared = {example.split for example in examples}
    if declared.issubset({"train", "validation"}) and "validation" in declared:
        train_identities = sorted(
            {example.identity for example in examples if example.split == "train"}
        )
        validation_identities = {
            example.identity for example in examples if example.split == "validation"
        }
        if len(train_identities) < 4 or len(validation_identities) < 2:
            raise ValueError("Real benchmark needs >=4 training and >=2 validation identities")
        cut = max(1, int(round(0.8 * len(train_identities))))
        cut = min(cut, len(train_identities) - 1)
        fit_ids = set(train_identities[:cut])
        calibration_ids = set(train_identities[cut:])
        return (
            [example for example in examples if example.identity in fit_ids],
            [example for example in examples if example.identity in calibration_ids],
            [example for example in examples if example.identity in validation_identities],
        )
    identities = sorted({example.identity for example in examples})
    if len(identities) < 12:
        raise ValueError("Benchmark needs at least twelve paired identities")
    first = int(round(0.50 * len(identities)))
    second = int(round(0.70 * len(identities)))
    groups = [set(identities[:first]), set(identities[first:second]), set(identities[second:])]
    return tuple(
        [example for example in examples if example.identity in selected]
        for selected in groups
    )


def _edge_arrays(examples: Sequence[BenchmarkExample], config: SequenceGaugeConfig):
    features, labels = [], []
    by_example = []
    for example in examples:
        raw = continuity_edge_logits(
            example.corrupted["coordinates"],
            example.corrupted["valid"],
            block_frames=config.block_frames,
            scale=1.0,
        )
        target = (example.path[1:] != example.path[:-1]).astype(np.int8)
        features.extend(raw.tolist())
        labels.extend(target.tolist())
        by_example.append(raw)
    return np.asarray(features)[:, None], np.asarray(labels), by_example


def _fit_continuity_head(
    examples: Sequence[BenchmarkExample], config: SequenceGaugeConfig
) -> LogisticRegression:
    features, labels, _ = _edge_arrays(examples, config)
    if len(np.unique(labels)) != 2:
        raise ValueError("Continuity fitting identities do not contain both edge classes")
    model = LogisticRegression(C=1.0, class_weight=None, max_iter=1000, random_state=0)
    model.fit(features, labels)
    prevalence = float(np.mean(labels))
    model.training_log_prior_ = math.log(
        np.clip(prevalence, 1e-6, 1.0 - 1e-6)
        / np.clip(1.0 - prevalence, 1e-6, 1.0 - 1e-6)
    )
    return model


def _continuity_logits(
    head: LogisticRegression,
    examples: Sequence[BenchmarkExample],
    config: SequenceGaugeConfig,
) -> list[np.ndarray]:
    output = []
    for example in examples:
        raw = continuity_edge_logits(
            example.corrupted["coordinates"],
            example.corrupted["valid"],
            block_frames=config.block_frames,
            scale=1.0,
        )
        output.append(
            (
                head.decision_function(raw[:, None])
                - float(head.training_log_prior_)
            ).astype(np.float64)
        )
    return output


def _mask_edge_features(example: BenchmarkExample, config: SequenceGaugeConfig) -> np.ndarray:
    valid = example.corrupted["valid"]
    blocks = len(valid) // config.block_frames
    rows = []
    for block in range(1, blocks):
        left = valid[block * config.block_frames - 1]
        right = valid[block * config.block_frames]
        swapped = semantic_permute(right)
        rows.append(
            [
                float(np.mean(left != right)),
                float(np.mean(left != swapped)),
                float(np.mean(left)),
                float(np.mean(right)),
            ]
        )
    return np.asarray(rows, dtype=np.float64)


def _fit_mask_head(
    examples: Sequence[BenchmarkExample], config: SequenceGaugeConfig
) -> LogisticRegression | None:
    features = np.concatenate([_mask_edge_features(example, config) for example in examples])
    labels = np.concatenate(
        [(example.path[1:] != example.path[:-1]).astype(np.int8) for example in examples]
    )
    if len(np.unique(labels)) != 2 or np.max(np.ptp(features, axis=0)) <= 1e-12:
        return None
    model = LogisticRegression(C=1.0, max_iter=1000, random_state=0).fit(features, labels)
    prevalence = float(np.mean(labels))
    model.training_log_prior_ = math.log(
        np.clip(prevalence, 1e-6, 1.0 - 1e-6)
        / np.clip(1.0 - prevalence, 1e-6, 1.0 - 1e-6)
    )
    return model


def _mask_logits(
    head: LogisticRegression | None,
    examples: Sequence[BenchmarkExample],
    config: SequenceGaugeConfig,
) -> list[np.ndarray]:
    output = []
    for example in examples:
        features = _mask_edge_features(example, config)
        if head is None:
            output.append(np.zeros(len(features), dtype=np.float64))
        else:
            output.append(
                (
                    head.decision_function(features) - float(head.training_log_prior_)
                ).astype(np.float64)
            )
    return output


def _absolute_features(example: BenchmarkExample) -> np.ndarray:
    coordinates = example.corrupted["coordinates"]
    valid = example.corrupted["valid"]
    masked = coordinates * valid[..., None]
    return np.concatenate(
        [masked.mean(axis=0).ravel(), masked.std(axis=0).ravel(), valid.mean(axis=0)]
    )


def _cluster_bootstrap_auc_upper(
    labels: np.ndarray,
    scores: np.ndarray,
    identities: np.ndarray,
    *,
    seed: int,
    draws: int = 500,
) -> tuple[float, float]:
    observed = float(roc_auc_score(labels, scores))
    unique = np.unique(identities)
    rng = np.random.default_rng(seed)
    estimates = []
    for _ in range(draws):
        sampled = rng.choice(unique, size=len(unique), replace=True)
        indices = np.concatenate([np.flatnonzero(identities == identity) for identity in sampled])
        if len(np.unique(labels[indices])) == 2:
            estimates.append(float(roc_auc_score(labels[indices], scores[indices])))
    return observed, float(np.quantile(estimates, 0.95))


def _orbit_nmse(prediction: np.ndarray, target: np.ndarray) -> float:
    scale = max(float(np.mean(np.square(target))), 1e-12)
    error = min(
        float(np.mean(np.square(prediction - target))),
        float(np.mean(np.square(-prediction - target))),
    )
    return error / scale


def run_sequence_benchmark(
    output_dir: Path,
    *,
    seed: int = 7,
    identities: int = 80,
    frames: int = 192,
    config: SequenceGaugeConfig | None = None,
    examples: Sequence[BenchmarkExample] | None = None,
    synthetic_smoke: bool = False,
) -> dict:
    output_dir = Path(output_dir)
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"Output directory is not empty: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    config = config or SequenceGaugeConfig(boundary_radius_frames=2)
    if examples is None:
        examples = make_synthetic_examples(
            identities=identities,
            frames=frames,
            seed=seed,
            config=config,
        )
        synthetic_smoke = True
    identities = len({example.identity for example in examples})
    fit_examples, calibration_examples, validation_examples = _identity_partition(examples)
    head = _fit_continuity_head(fit_examples, config)
    mask_head = _fit_mask_head(fit_examples, config)
    calibration_logits = _continuity_logits(head, calibration_examples, config)
    mask_calibration_logits = _mask_logits(mask_head, calibration_examples, config)
    duration_model = TwoStateDurationModel(max_duration=16)
    temperature = calibrate_duration_temperature(
        duration_model,
        calibration_logits,
        [example.path for example in calibration_examples],
    )
    mask_temperature = (
        calibrate_duration_temperature(
            duration_model,
            mask_calibration_logits,
            [example.path for example in calibration_examples],
        )
        if mask_head is not None
        else 1.0
    )
    validation_logits = _continuity_logits(head, validation_examples, config)
    validation_mask_logits = _mask_logits(mask_head, validation_examples, config)

    rows = []
    for example, logits, mask_logits in zip(
        validation_examples, validation_logits, validation_mask_logits
    ):
        posterior = duration_model.infer(logits, temperature=temperature)
        relative_posterior = duration_model.infer(
            logits, temperature=temperature, root_bit=0
        )
        prior_logits = np.zeros_like(logits)
        prior_nll = duration_model.equivalence_path_nll(prior_logits, example.path)
        mask_nll = duration_model.equivalence_path_nll(
            mask_logits, example.path, temperature=mask_temperature
        )
        continuity_nll = duration_model.equivalence_path_nll(
            logits, example.path, temperature=temperature
        )
        corrected, corrected_valid = apply_block_correction(
            example.corrupted["coordinates"],
            example.corrupted["valid"],
            posterior.map_path,
            block_frames=config.block_frames,
        )
        oracle, oracle_valid = apply_block_correction(
            example.corrupted["coordinates"],
            example.corrupted["valid"],
            example.path,
            block_frames=config.block_frames,
        )
        target_odd, target_even = block_odd_even_motion_targets(
            example.corrupted["nuisance_reference"],
            example.corrupted["nuisance_valid"],
            block_frames=config.block_frames,
        )
        observed_odd, observed_even = block_odd_even_motion_targets(
            example.corrupted["coordinates"],
            example.corrupted["valid"],
            block_frames=config.block_frames,
        )
        continuity_odd, continuity_even = block_odd_even_motion_targets(
            corrected, corrected_valid, block_frames=config.block_frames
        )
        oracle_odd, oracle_even = block_odd_even_motion_targets(
            oracle, oracle_valid, block_frames=config.block_frames
        )
        posterior_odd = (
            1.0 - 2.0 * relative_posterior.block_swap_probability
        ) * observed_odd
        edge_target = (example.path[1:] != example.path[:-1]).astype(float)
        rows.append(
            {
                "identity": example.identity,
                "sequence_id": example.sequence_id,
                "chart_label": example.chart_label,
                "path_family": example.draw.path_family,
                "swapped_event": bool(edge_target.any()),
                "prior_path_nll": prior_nll,
                "mask_only_path_nll": mask_nll,
                "continuity_path_nll": continuity_nll,
                "path_hamming": path_hamming_up_to_global_flip(
                    posterior.map_path, example.path
                ),
                "switch_f1": switch_f1(posterior.map_path, example.path),
                "predicted_switch_rate": float(
                    np.mean(posterior.map_path[1:] != posterior.map_path[:-1])
                ),
                "edge_brier": float(
                    np.mean(np.square(posterior.edge_switch_probability - edge_target))
                ),
                "uncorrected_odd_nmse": _orbit_nmse(observed_odd, target_odd),
                "continuity_odd_nmse": _orbit_nmse(continuity_odd, target_odd),
                "posterior_odd_nmse": _orbit_nmse(posterior_odd, target_odd),
                "oracle_odd_nmse": _orbit_nmse(oracle_odd, target_odd),
                "continuity_even_nmse": _orbit_nmse(continuity_even, target_even),
                "oracle_even_nmse": _orbit_nmse(oracle_even, target_even),
                "mean_edge_entropy": float(
                    np.mean(
                        -posterior.edge_switch_probability
                        * np.log(np.clip(posterior.edge_switch_probability, 1e-12, 1.0))
                        - (1.0 - posterior.edge_switch_probability)
                        * np.log(np.clip(1.0 - posterior.edge_switch_probability, 1e-12, 1.0))
                    )
                ),
            }
        )
    metrics = pd.DataFrame(rows)

    train_for_probe = fit_examples + calibration_examples
    train_x = np.stack([_absolute_features(example) for example in train_for_probe])
    train_y = np.asarray([example.chart_label for example in train_for_probe])
    probe = LogisticRegression(C=0.1, max_iter=1000, random_state=seed).fit(train_x, train_y)
    validation_x = np.stack([_absolute_features(example) for example in validation_examples])
    validation_y = np.asarray([example.chart_label for example in validation_examples])
    validation_score = probe.predict_proba(validation_x)[:, 1]
    absolute_auc, absolute_auc_upper = _cluster_bootstrap_auc_upper(
        validation_y,
        validation_score,
        np.asarray([example.identity for example in validation_examples]),
        seed=seed,
    )

    prior_nll = float(metrics.prior_path_nll.mean())
    mask_nll = float(metrics.mask_only_path_nll.mean())
    events = metrics.loc[metrics.swapped_event]
    if events.empty:
        raise RuntimeError("Validation draw has no local or repeated switch event")
    continuity_error = float(events.continuity_odd_nmse.mean())
    oracle_error = float(events.oracle_odd_nmse.mean())
    gates = {
        "mask_only_relative_path_nll_reduction": relative_reduction(prior_nll, mask_nll),
        "mask_only_gate_pass": relative_reduction(prior_nll, mask_nll) <= 0.01,
        "absolute_chart_auroc": absolute_auc,
        "absolute_chart_auroc_upper_95": absolute_auc_upper,
        "absolute_chart_gate_pass": absolute_auc_upper < 0.55,
        "oracle_vs_continuity_odd_error_reduction": relative_reduction(
            continuity_error, oracle_error
        ),
        "oracle_gate_pass": relative_reduction(continuity_error, oracle_error) >= 0.05,
        "continuity_event_odd_nmse": continuity_error,
        "oracle_event_odd_nmse": oracle_error,
        "continuity_path_nll": float(metrics.continuity_path_nll.mean()),
        "input_free_path_nll": prior_nll,
        "temperature": temperature,
        "mask_temperature": mask_temperature,
        "clean_false_switch_rate": float(
            metrics.loc[~metrics.swapped_event, "predicted_switch_rate"].mean()
        ),
    }
    gates["ready_for_sg_jepa"] = bool(
        gates["mask_only_gate_pass"]
        and gates["absolute_chart_gate_pass"]
        and gates["oracle_gate_pass"]
    )
    gate_rows = pd.DataFrame(
        [
            {
                "gate": "mask_only_leakage",
                "value": gates["mask_only_relative_path_nll_reduction"],
                "threshold": 0.01,
                "direction": "<=",
                "passed": gates["mask_only_gate_pass"],
            },
            {
                "gate": "absolute_chart_auroc_upper_95",
                "value": gates["absolute_chart_auroc_upper_95"],
                "threshold": 0.55,
                "direction": "<",
                "passed": gates["absolute_chart_gate_pass"],
            },
            {
                "gate": "oracle_vs_continuity_odd_error_reduction",
                "value": gates["oracle_vs_continuity_odd_error_reduction"],
                "threshold": 0.05,
                "direction": ">=",
                "passed": gates["oracle_gate_pass"],
            },
        ]
    )
    _atomic_csv(output_dir / "sequence_metrics.csv", metrics)
    _atomic_csv(output_dir / "benchmark_gates.csv", gate_rows)
    uncertainty = metrics.copy()
    uncertainty["uncertainty_stratum"] = pd.cut(
        uncertainty.mean_edge_entropy,
        bins=[-np.inf, 0.20, 0.50, np.inf],
        labels=["low", "medium", "high"],
    )
    uncertainty["posterior_minus_map_odd_nmse"] = (
        uncertainty.posterior_odd_nmse - uncertainty.continuity_odd_nmse
    )
    uncertainty_summary = (
        uncertainty.groupby("uncertainty_stratum", observed=True)
        .agg(
            sequences=("sequence_id", "size"),
            posterior_minus_map_odd_nmse=("posterior_minus_map_odd_nmse", "mean"),
            posterior_odd_nmse=("posterior_odd_nmse", "mean"),
            map_odd_nmse=("continuity_odd_nmse", "mean"),
        )
        .reset_index()
    )
    _atomic_csv(output_dir / "uncertainty_summary.csv", uncertainty_summary)
    _atomic_json(output_dir / "gate_decision.json", gates)
    _atomic_json(
        output_dir / "effective_config.json",
        {
            "seed": seed,
            "identities": identities,
            "frames": frames,
            "sequence_gauge": asdict(config),
            "identity_counts": {
                "fit": len({example.identity for example in fit_examples}),
                "calibration": len({example.identity for example in calibration_examples}),
                "validation": len({example.identity for example in validation_examples}),
            },
            "synthetic_smoke": synthetic_smoke,
            "test_split_evaluated": False,
        },
    )
    return gates


def run_synthetic_benchmark(
    output_dir: Path,
    *,
    seed: int = 7,
    identities: int = 80,
    frames: int = 192,
    config: SequenceGaugeConfig | None = None,
) -> dict:
    return run_sequence_benchmark(
        output_dir,
        seed=seed,
        identities=identities,
        frames=frames,
        config=config,
        synthetic_smoke=True,
    )
