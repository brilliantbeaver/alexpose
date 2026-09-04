"""AMASS-Gauge validation or sealed-test readout for the decisive training arms.

This evaluator deliberately uses the synthetic corruption's canonical source
only as a controlled kinematic target.  It never reports an anatomical sign:
the odd target is evaluated as a magnitude/orbit functional.  Continuity is
fit on development identities, the linear readout is fit on train identities,
and test identities are not used before an explicitly requested sealed test run.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import Ridge

from gavd6_sjepa.shared_infrastructure.artifact_io_operations import (
    atomic_write_dataframe_csv as _atomic_csv,
    atomic_write_json as _atomic_json,
)
from gavd6_sjepa.research_directions.reflection_equivariance.amass_core11_training_pipeline import TIME_PATCH_FRAMES
from gavd6_sjepa.research_directions.reflection_equivariance.jepa_model_architecture import STANDARD_VARIANTS, load_checkpoint, parity_channels
from .laterality_gauge_training_pipeline import GAUGE_ARMS
from .laterality_corruption_inference import (
    SequenceGaugeConfig,
    TwoStateDurationModel,
    apply_block_correction,
    block_odd_even_motion_targets,
    path_hamming_up_to_global_flip,
    relative_reduction,
    semantic_permute,
    switch_f1,
)
from .laterality_sequence_benchmarking import (
    BenchmarkExample,
    _continuity_logits,
    _fit_continuity_head,
    _identity_partition,
    load_manifest_sequence_config,
    make_manifest_examples,
)
from .laterality_study_protocol import require_benchmark_gate


def _patch_valid(valid: np.ndarray, block_frames: int) -> np.ndarray:
    complete = len(valid) // block_frames * block_frames
    return valid[:complete].reshape(-1, block_frames, valid.shape[1]).mean(1) >= 0.95


def _window_starts(frames: int, config: SequenceGaugeConfig) -> list[int]:
    if frames < config.window_frames:
        return []
    starts = list(range(0, frames - config.window_frames + 1, config.window_stride))
    final = frames - config.window_frames
    if starts[-1] != final:
        starts.append(final)
    return starts


@dataclass(frozen=True)
class GaugeWindows:
    coordinates: np.ndarray
    valid: np.ndarray
    targets: np.ndarray
    metadata: pd.DataFrame


def _window_targets(
    reference: np.ndarray, reference_valid: np.ndarray, config: SequenceGaugeConfig
) -> np.ndarray:
    odd, even = block_odd_even_motion_targets(
        reference, reference_valid, block_frames=config.block_frames
    )
    # The global chart is unidentifiable, so the controlled target preserves
    # odd information only up to sign.  It is never presented as anatomy.
    return np.asarray([np.mean(np.abs(odd)), np.mean(even)], dtype=np.float64)


def _collect_windows(
    examples: list[BenchmarkExample],
    logits: list[np.ndarray],
    duration: TwoStateDurationModel,
    temperature: float,
    *,
    config: SequenceGaugeConfig,
    correction: str,
) -> GaugeWindows:
    coordinates, valid, targets, rows = [], [], [], []
    for example, edge_logits in zip(examples, logits):
        observed = example.corrupted["coordinates"]
        observed_valid = example.corrupted["valid"]
        if correction == "continuity_map":
            posterior = duration.infer(edge_logits, temperature=temperature)
            observed, observed_valid = apply_block_correction(
                observed,
                observed_valid,
                posterior.map_path,
                block_frames=config.block_frames,
            )
        elif correction == "oracle":
            observed, observed_valid = apply_block_correction(
                observed,
                observed_valid,
                example.path,
                block_frames=config.block_frames,
            )
        elif correction != "none":
            raise ValueError(f"Unknown correction {correction!r}")
        for start in _window_starts(len(observed), config):
            stop = start + config.window_frames
            coordinates.append(observed[start:stop].astype(np.float32))
            valid.append(_patch_valid(observed_valid[start:stop], config.block_frames))
            targets.append(
                _window_targets(
                    example.corrupted["nuisance_reference"][start:stop],
                    example.corrupted["nuisance_valid"][start:stop],
                    config,
                )
            )
            rows.append(
                {
                    "identity": example.identity,
                    "sequence_id": example.sequence_id,
                    "start_frame": start,
                    "path_family": example.draw.path_family,
                    "swapped_event": bool(np.any(np.diff(example.path) != 0)),
                }
            )
    if not coordinates:
        raise ValueError("No complete evaluation windows were constructed")
    return GaugeWindows(
        np.stack(coordinates), np.stack(valid), np.stack(targets), pd.DataFrame(rows)
    )


def _raw_features(coordinates: np.ndarray, valid: np.ndarray) -> np.ndarray:
    frames = torch.from_numpy(coordinates)
    patch_valid = torch.from_numpy(valid)
    segments = patch_valid.shape[1]
    reshaped = frames.reshape(len(frames), segments, -1, frames.shape[2], 3)
    velocity = reshaped[:, :, 1:] - reshaped[:, :, :-1]
    features = torch.cat(
        [
            velocity.mean((1, 2)).flatten(1),
            velocity.std((1, 2), unbiased=False).flatten(1),
            patch_valid.float().mean(1),
        ],
        dim=1,
    )
    return features.numpy()


def _torch_semantic_permute(values: torch.Tensor, pairs) -> torch.Tensor:
    result = values.clone()
    original = values.clone()
    for left, right in pairs:
        result[..., left, :] = original[..., right, :]
        result[..., right, :] = original[..., left, :]
    return result


@torch.no_grad()
def _encoder_features(
    model, windows: GaugeWindows, *, batch_size: int, device: torch.device
) -> np.ndarray:
    if model.config.frames != windows.coordinates.shape[1]:
        raise ValueError("Checkpoint window length does not match the gauge manifest")
    if model.config.segment_length != TIME_PATCH_FRAMES:
        raise ValueError("Checkpoint tokenization differs from the Core11 gauge contract")
    output = []
    encoder = model.target_encoder.eval()
    for start in range(0, len(windows.coordinates), batch_size):
        stop = min(len(windows.coordinates), start + batch_size)
        coordinates = torch.from_numpy(windows.coordinates[start:stop]).to(device)
        valid = torch.from_numpy(windows.valid[start:stop]).to(device)
        if model.variant in STANDARD_VARIANTS:
            tokens = encoder(coordinates, valid_patch=valid)
            features = torch.cat(
                [tokens.mean(1), tokens.std(1, unbiased=False)], dim=1
            )
        else:
            orbit = torch.stack(
                [coordinates, _torch_semantic_permute(coordinates, model.config.mirror_pairs)],
                dim=1,
            )
            paired_valid = torch.stack(
                [
                    valid,
                    _torch_semantic_permute(
                        valid[..., None].float(), model.config.mirror_pairs
                    )[..., 0].bool(),
                ],
                dim=1,
            )
            even, odd = parity_channels(encoder(orbit, valid_patch=paired_valid))
            features = torch.cat([even, odd.abs()], dim=1)
        output.append(features.cpu().numpy())
    return np.concatenate(output)


def _balanced_weights(metadata: pd.DataFrame) -> np.ndarray:
    counts = metadata.groupby("identity").size()
    weights = metadata.identity.map(lambda item: 1.0 / counts[item]).to_numpy(float)
    return weights / weights.mean()


def _fit_and_score(
    label: str,
    train_features: np.ndarray,
    train: GaugeWindows,
    test_features: np.ndarray,
    test: GaugeWindows,
) -> tuple[pd.DataFrame, dict]:
    prediction = np.zeros_like(test.targets)
    for target_index in range(2):
        readout = Ridge(alpha=1.0)
        readout.fit(
            train_features,
            train.targets[:, target_index],
            sample_weight=_balanced_weights(train.metadata),
        )
        prediction[:, target_index] = readout.predict(test_features)
    scale = np.maximum(np.mean(np.abs(train.targets), axis=0), 1e-12)
    rows = test.metadata.copy()
    rows.insert(0, "representation", label)
    rows["odd_orbit_target"] = test.targets[:, 0]
    rows["odd_orbit_prediction"] = prediction[:, 0]
    rows["even_target"] = test.targets[:, 1]
    rows["even_prediction"] = prediction[:, 1]
    rows["odd_orbit_nmae"] = np.abs(prediction[:, 0] - test.targets[:, 0]) / scale[0]
    rows["even_nmae"] = np.abs(prediction[:, 1] - test.targets[:, 1]) / scale[1]
    macro = rows.groupby("identity")[["odd_orbit_nmae", "even_nmae"]].mean().mean()
    summary = {
        "representation": label,
        "evaluation_identities": int(rows.identity.nunique()),
        "evaluation_windows": len(rows),
        "identity_macro_odd_orbit_nmae": float(macro.odd_orbit_nmae),
        "identity_macro_even_nmae": float(macro.even_nmae),
        "feature_variance": float(np.var(test_features, axis=0).mean()),
        "signed_accuracy_reported": False,
    }
    return rows, summary


def _parse_run_dir(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("Use LABEL=/path/to/training-output")
    label, text = value.split("=", 1)
    if not label or not text:
        raise argparse.ArgumentTypeError("Run label and path must be nonempty")
    return label, Path(text)


def _load_runs(
    entries: list[tuple[str, Path]], *, manifest_sha256: str, expected_arms: set[str]
) -> list[tuple[str, str, Path]]:
    loaded = []
    arms = []
    for label, directory in entries:
        result_path = directory.resolve() / "run_result.json"
        payload = json.loads(result_path.read_text())
        arm = str(payload.get("arm", ""))
        if arm not in GAUGE_ARMS:
            raise ValueError(f"{result_path} is not a supported gauge training run")
        if payload.get("gauge_manifest_sha256") != manifest_sha256:
            raise ValueError(f"{result_path} was trained on a different gauge manifest")
        checkpoint = Path(payload["result"]["best_checkpoint"])
        if not checkpoint.is_file():
            raise FileNotFoundError(f"Missing best checkpoint {checkpoint}")
        loaded.append((label, arm, checkpoint))
        arms.append(arm)
    if set(arms) != expected_arms:
        names = ", ".join(sorted(expected_arms))
        raise ValueError(f"Provided runs must cover exactly: {names}")
    return loaded


def _test_path_metrics(
    examples: list[BenchmarkExample],
    logits: list[np.ndarray],
    duration: TwoStateDurationModel,
    temperature: float,
) -> pd.DataFrame:
    rows = []
    for example, edge_logits in zip(examples, logits):
        posterior = duration.infer(edge_logits, temperature=temperature)
        target = (example.path[1:] != example.path[:-1]).astype(float)
        rows.append(
            {
                "identity": example.identity,
                "sequence_id": example.sequence_id,
                "path_family": example.draw.path_family,
                "path_hamming": path_hamming_up_to_global_flip(posterior.map_path, example.path),
                "switch_f1": switch_f1(posterior.map_path, example.path),
                "path_nll": duration.equivalence_path_nll(
                    edge_logits, example.path, temperature=temperature
                ),
                "edge_brier": float(
                    np.mean(np.square(posterior.edge_switch_probability - target))
                ),
            }
        )
    return pd.DataFrame(rows)


def evaluate(args: argparse.Namespace) -> pd.DataFrame:
    output_dir = args.output_dir.resolve()
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"Output directory is not empty: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = args.gauge_manifest.resolve()
    import hashlib

    manifest_sha256 = hashlib.sha256(manifest_path.read_bytes()).hexdigest()
    require_benchmark_gate(args.gate_decision, gauge_manifest_sha256=manifest_sha256)
    config = load_manifest_sequence_config(manifest_path)
    development = make_manifest_examples(
        manifest_path, args.tensor_root.resolve(), config=config, chart_members=(0,)
    )
    fit_examples, calibration_examples, _ = _identity_partition(development)
    train_examples = [example for example in development if example.split == "train"]
    evaluation_examples = make_manifest_examples(
        manifest_path,
        args.tensor_root.resolve(),
        config=config,
        chart_members=(0,),
        splits=(args.split,),
    )
    head = _fit_continuity_head(fit_examples, config)
    duration = TwoStateDurationModel(max_duration=16)
    calibration_logits = _continuity_logits(head, calibration_examples, config)
    from .laterality_corruption_inference import calibrate_duration_temperature

    temperature = calibrate_duration_temperature(
        duration, calibration_logits, [example.path for example in calibration_examples]
    )
    train_logits = _continuity_logits(head, train_examples, config)
    evaluation_logits = _continuity_logits(head, evaluation_examples, config)
    train_uncorrected = _collect_windows(
        train_examples, train_logits, duration, temperature, config=config, correction="none"
    )
    evaluation_uncorrected = _collect_windows(
        evaluation_examples,
        evaluation_logits,
        duration,
        temperature,
        config=config,
        correction="none",
    )
    train_map = _collect_windows(
        train_examples,
        train_logits,
        duration,
        temperature,
        config=config,
        correction="continuity_map",
    )
    evaluation_map = _collect_windows(
        evaluation_examples,
        evaluation_logits,
        duration,
        temperature,
        config=config,
        correction="continuity_map",
    )
    train_oracle = _collect_windows(
        train_examples, train_logits, duration, temperature, config=config, correction="oracle"
    )
    evaluation_oracle = _collect_windows(
        evaluation_examples,
        evaluation_logits,
        duration,
        temperature,
        config=config,
        correction="oracle",
    )
    expected_arms = (
        set(GAUGE_ARMS)
        if args.mode == "decisive"
        else {"correction_first_sjepa", "sg_jepa"}
    )
    runs = _load_runs(
        [_parse_run_dir(item) for item in args.run_dir],
        manifest_sha256=manifest_sha256,
        expected_arms=expected_arms,
    )
    device = torch.device(args.device)
    summaries, predictions = [], []
    raw_rows, raw_summary = _fit_and_score(
        "raw_uncorrected",
        _raw_features(train_uncorrected.coordinates, train_uncorrected.valid),
        train_uncorrected,
        _raw_features(evaluation_uncorrected.coordinates, evaluation_uncorrected.valid),
        evaluation_uncorrected,
    )
    predictions.append(raw_rows)
    summaries.append(raw_summary)
    for label, training_arm, correction, train_windows, test_windows in (
        ("raw_continuity_map", "raw", "continuity_map", train_map, evaluation_map),
        ("raw_oracle", "oracle", "oracle", train_oracle, evaluation_oracle),
    ):
        rows, summary = _fit_and_score(
            label,
            _raw_features(train_windows.coordinates, train_windows.valid),
            train_windows,
            _raw_features(test_windows.coordinates, test_windows.valid),
            test_windows,
        )
        summary.update({"training_arm": training_arm, "input_correction": correction})
        predictions.append(rows)
        summaries.append(summary)
    for label, arm, checkpoint in runs:
        model, _, metadata = load_checkpoint(checkpoint)
        expected_variant = "standard_sjepa" if arm == "correction_first_sjepa" else "reflection_equivariant"
        if model.variant != expected_variant:
            raise ValueError(f"{checkpoint} has {model.variant!r}, expected {expected_variant!r}")
        correction = "continuity_map" if arm == "correction_first_sjepa" else "none"
        train_windows, evaluation_windows = (
            (train_map, evaluation_map)
            if correction == "continuity_map"
            else (train_uncorrected, evaluation_uncorrected)
        )
        model = model.to(device)
        rows, summary = _fit_and_score(
            label,
            _encoder_features(model, train_windows, batch_size=args.batch_size, device=device),
            train_windows,
            _encoder_features(model, evaluation_windows, batch_size=args.batch_size, device=device),
            evaluation_windows,
        )
        summary.update(
            {
                "training_arm": arm,
                "input_correction": correction,
                "checkpoint": str(checkpoint),
                "checkpoint_variant": metadata["variant"],
            }
        )
        predictions.append(rows)
        summaries.append(summary)
    path_metrics = _test_path_metrics(
        evaluation_examples, evaluation_logits, duration, temperature
    )
    summary_frame = pd.DataFrame(summaries)
    _atomic_csv(output_dir / "gauge_readout_summary.csv", summary_frame)
    _atomic_csv(
        output_dir / "gauge_readout_predictions.csv", pd.concat(predictions, ignore_index=True)
    )
    _atomic_csv(output_dir / "gauge_path_metrics.csv", path_metrics)
    _atomic_json(
        output_dir / "evaluation_contract.json",
        {
            "gauge_manifest_sha256": manifest_sha256,
            "sequence_gauge_config": config.__dict__,
            "development_splits_used_for_continuity": ["train", "validation"],
            "linear_readout_fit_split": "train",
            "evaluation_split": args.split,
            "aggregation_unit": "identity",
            "odd_output": "magnitude/orbit; no anatomical sign claim",
            "test_split_evaluated": args.split == "test",
            "temperature": temperature,
        },
    )
    return summary_frame


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gate-decision", type=Path, required=True)
    parser.add_argument("--gauge-manifest", type=Path, required=True)
    parser.add_argument("--tensor-root", type=Path, required=True)
    parser.add_argument(
        "--run-dir", action="append", default=[], help="LABEL=/path/to/training-output"
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--mode", choices=("decisive", "confirmation"), default="decisive")
    parser.add_argument("--split", choices=("validation", "test"), required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=64)
    return parser.parse_args(argv)


def main() -> int:
    print(evaluate(parse_args()).to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
