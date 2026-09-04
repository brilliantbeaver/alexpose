"""Common frozen readout for the three GAVD source-transfer routes."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import Ridge

from gavd6_sjepa.research_directions.reflection_equivariance.amass_core11_training_pipeline import (
    Core11WindowDataset,
    build_window_index,
    load_conversion_manifest,
)
from gavd6_sjepa.research_directions.reflection_equivariance.jepa_model_architecture import (
    STANDARD_VARIANTS,
    TrainConfig,
    build_model,
    lift_orbit,
    load_checkpoint,
    parity_channels,
    permute_bilateral_tokens,
)
from .laterality_corruption_inference import block_odd_even_motion_targets


def _raw_features(coordinates: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
    frames = coordinates.reshape(
        len(coordinates), valid.shape[1], -1, coordinates.shape[2], 3
    )
    velocity = frames[:, :, 1:] - frames[:, :, :-1]
    return torch.cat(
        [
            velocity.mean((1, 2)).flatten(1),
            velocity.std((1, 2), unbiased=False).flatten(1),
            valid.float().mean(1),
        ],
        dim=1,
    )


@torch.no_grad()
def _encoder_features(model, coordinates: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
    encoder = model.target_encoder.eval()
    if model.variant in STANDARD_VARIANTS:
        tokens = encoder(coordinates, valid_patch=valid)
        return torch.cat([tokens.mean(1), tokens.std(1, unbiased=False)], dim=1)
    orbit = lift_orbit(
        coordinates, model.config.mirror_pairs, model.config.mirror_channel
    )
    paired_valid = torch.stack(
        [valid, permute_bilateral_tokens(valid, model.config.mirror_pairs)], dim=1
    )
    even, odd = parity_channels(encoder(orbit, valid_patch=paired_valid))
    # Odd sign is unanchored on GAVD. Absolute odd coordinates preserve orbit
    # information without pretending the chart names anatomical side.
    return torch.cat([even, odd.abs()], dim=1)


def _window_targets(coordinates: torch.Tensor, valid_patch: torch.Tensor, segment: int = 4):
    valid_frame = valid_patch.repeat_interleave(segment, dim=1)
    odd, even = [], []
    for xyz, mask in zip(coordinates.numpy(), valid_frame.numpy()):
        block_odd, block_even = block_odd_even_motion_targets(
            xyz, mask, block_frames=segment
        )
        odd.append(float(np.mean(np.abs(block_odd))))
        even.append(float(np.mean(block_even)))
    return np.stack([odd, even], axis=1)


def _collect(
    dataset: Core11WindowDataset,
    model,
    *,
    batch_size: int,
    device: torch.device,
):
    features, raw, targets = [], [], []
    metadata = []
    for start in range(0, len(dataset), batch_size):
        items = [dataset[item] for item in range(start, min(len(dataset), start + batch_size))]
        coordinates = torch.stack([item["coordinates"] for item in items])
        valid = torch.stack([item["valid"] for item in items])
        raw.append(_raw_features(coordinates, valid).numpy())
        targets.append(_window_targets(coordinates, valid))
        features.append(
            _encoder_features(
                model,
                coordinates.to(device),
                valid.to(device),
            ).cpu().numpy()
        )
        for item in range(start, min(len(dataset), start + batch_size)):
            metadata.append(dataset.windows[item])
    return (
        np.concatenate(features),
        np.concatenate(raw),
        np.concatenate(targets),
        pd.DataFrame(metadata),
    )


def _balanced_weights(metadata: pd.DataFrame) -> np.ndarray:
    counts = metadata.groupby("identity").size()
    weights = metadata.identity.map(lambda item: 1.0 / counts[item]).to_numpy(float)
    return weights / weights.mean()


def _fit_and_score(
    train_features: np.ndarray,
    train_targets: np.ndarray,
    train_meta: pd.DataFrame,
    eval_features: np.ndarray,
    eval_targets: np.ndarray,
    eval_meta: pd.DataFrame,
) -> tuple[pd.DataFrame, dict]:
    prediction = np.zeros_like(eval_targets)
    for target in range(2):
        model = Ridge(alpha=1.0)
        model.fit(
            train_features,
            train_targets[:, target],
            sample_weight=_balanced_weights(train_meta),
        )
        prediction[:, target] = model.predict(eval_features)
    rows = eval_meta[["window_id", "identity", "start_frame"]].copy()
    rows["odd_magnitude_target"] = eval_targets[:, 0]
    rows["odd_magnitude_prediction"] = prediction[:, 0]
    rows["even_target"] = eval_targets[:, 1]
    rows["even_prediction"] = prediction[:, 1]
    scale = np.maximum(np.mean(np.abs(train_targets), axis=0), 1e-12)
    rows["odd_nmae"] = np.abs(prediction[:, 0] - eval_targets[:, 0]) / scale[0]
    rows["even_nmae"] = np.abs(prediction[:, 1] - eval_targets[:, 1]) / scale[1]
    macro = rows.groupby("identity")[["odd_nmae", "even_nmae"]].mean().mean()
    summary = {
        "source_videos": int(rows.identity.nunique()),
        "windows": len(rows),
        "video_macro_odd_orbit_nmae": float(macro.odd_nmae),
        "video_macro_even_nmae": float(macro.even_nmae),
        "feature_variance": float(np.var(eval_features, axis=0).mean()),
        "signed_accuracy_reported": False,
    }
    return rows, summary


def _parse_checkpoint(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("Checkpoint must be LABEL=/path/to/best.pt")
    label, path = value.split("=", 1)
    if not label or not path:
        raise argparse.ArgumentTypeError("Checkpoint label and path must be nonempty")
    return label, Path(path)


def evaluate(args: argparse.Namespace) -> pd.DataFrame:
    output_dir = args.output_dir.resolve()
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"Output directory is not empty: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest = load_conversion_manifest(args.gavd_manifest.resolve())
    if "video_id" in manifest and not (manifest.identity == manifest.video_id).all():
        raise ValueError("GAVD identity must equal source video for grouped evaluation")
    split_counts = manifest.groupby("identity").split.nunique()
    if (split_counts != 1).any():
        raise ValueError("A source video crosses train/validation/test splits")
    windows = build_window_index(manifest)
    train_dataset = Core11WindowDataset(
        manifest, windows, args.gavd_tensor_root, split="train"
    )
    eval_dataset = Core11WindowDataset(
        manifest, windows, args.gavd_tensor_root, split=args.split
    )
    device = torch.device(args.device)
    checkpoints = [_parse_checkpoint(value) for value in args.checkpoint]
    if not checkpoints:
        raise ValueError("At least one --checkpoint is required")
    summaries = []
    prediction_frames = []
    first_metadata = None
    for label, checkpoint in checkpoints:
        model, _, metadata = load_checkpoint(checkpoint)
        model = model.to(device)
        train_features, train_raw, train_targets, train_meta = _collect(
            train_dataset, model, batch_size=args.batch_size, device=device
        )
        eval_features, eval_raw, eval_targets, eval_meta = _collect(
            eval_dataset, model, batch_size=args.batch_size, device=device
        )
        predictions, summary = _fit_and_score(
            train_features,
            train_targets,
            train_meta,
            eval_features,
            eval_targets,
            eval_meta,
        )
        predictions.insert(0, "representation", label)
        prediction_frames.append(predictions)
        summaries.append({"representation": label, **summary})
        if first_metadata is None:
            first_metadata = metadata
            raw_predictions, raw_summary = _fit_and_score(
                train_raw,
                train_targets,
                train_meta,
                eval_raw,
                eval_targets,
                eval_meta,
            )
            raw_predictions.insert(0, "representation", "raw_coordinates")
            prediction_frames.append(raw_predictions)
            summaries.append({"representation": "raw_coordinates", **raw_summary})

    config = TrainConfig(**first_metadata["train_config"])
    random_model = build_model(config, first_metadata["variant"], args.random_seed).to(device)
    random_train, _, train_targets, train_meta = _collect(
        train_dataset, random_model, batch_size=args.batch_size, device=device
    )
    random_eval, _, eval_targets, eval_meta = _collect(
        eval_dataset, random_model, batch_size=args.batch_size, device=device
    )
    predictions, summary = _fit_and_score(
        random_train,
        train_targets,
        train_meta,
        random_eval,
        eval_targets,
        eval_meta,
    )
    predictions.insert(0, "representation", "random_encoder")
    prediction_frames.append(predictions)
    summaries.append({"representation": "random_encoder", **summary})
    summary_frame = pd.DataFrame(summaries)
    summary_frame.to_csv(output_dir / "source_transfer_summary.csv", index=False)
    pd.concat(prediction_frames, ignore_index=True).to_csv(
        output_dir / "source_transfer_predictions.csv", index=False
    )
    (output_dir / "evaluation_contract.json").write_text(
        json.dumps(
            {
                "split": args.split,
                "test_split_evaluated": args.split == "test",
                "aggregation_unit": "source_video",
                "odd_output": "magnitude/orbit; no anatomical sign claim",
                "condition_labels_in_jepa_or_readout": False,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )
    return summary_frame


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gavd-manifest", type=Path, required=True)
    parser.add_argument("--gavd-tensor-root", type=Path, required=True)
    parser.add_argument("--checkpoint", action="append", default=[])
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--split", choices=("validation", "test"), default="validation")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--random-seed", type=int, default=1103)
    return parser.parse_args(argv)


def main() -> int:
    print(evaluate(parse_args()).to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
