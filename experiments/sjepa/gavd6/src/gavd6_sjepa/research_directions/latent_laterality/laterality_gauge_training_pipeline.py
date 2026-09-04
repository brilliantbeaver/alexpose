"""Gate-protected correction-first and structured SG-JEPA training."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from gavd6_sjepa.research_directions.reflection_equivariance.amass_core11_training_pipeline import (
    TIME_PATCH_FRAMES,
    atomic_torch_save,
    checkpoint_payload,
    core11_train_config,
    fit_variant,
    make_train_loader,
)
from gavd6_sjepa.research_directions.reflection_equivariance.jepa_model_architecture import (
    INPUT_CONTRACT,
    PAIRED_MASK_CONTRACT,
    VICRegProjector,
    build_model,
    load_checkpoint,
    orbit_closed_target_masks,
    permute_bilateral_tokens,
    representation_metrics,
    sample_mask,
)
from .laterality_corruption_inference import (
    SequenceGaugeConfig,
    TwoStateDurationModel,
    apply_block_correction,
    semantic_permute_by_frame,
    sequence_gauge_config_json,
    structured_parity_prediction_loss,
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


GAUGE_ARMS = ("correction_first_sjepa", "sg_jepa", "uniform_posterior")


def _patch_valid(valid: np.ndarray, block_frames: int = TIME_PATCH_FRAMES) -> np.ndarray:
    complete = len(valid) // block_frames * block_frames
    return valid[:complete].reshape(-1, block_frames, valid.shape[1]).mean(1) >= 0.95


@dataclass(frozen=True)
class GaugeWindow:
    coordinates: np.ndarray
    valid: np.ndarray
    teacher_coordinates: np.ndarray
    teacher_valid: np.ndarray
    relative_probability: np.ndarray
    window_id: str


class GaugeWindowDataset(Dataset):
    def __init__(
        self,
        examples: list[BenchmarkExample],
        logits: list[np.ndarray],
        duration_model: TwoStateDurationModel,
        temperature: float,
        *,
        mode: str,
        config: SequenceGaugeConfig,
    ) -> None:
        if mode not in {"uncorrected", "map_corrected"}:
            raise ValueError(mode)
        self.windows: list[GaugeWindow] = []
        self.balance_source_groups = False
        self.sampling_groups: list[str] = []
        for example, edge_logits in zip(examples, logits):
            coordinates = example.corrupted["coordinates"]
            valid = example.corrupted["valid"]
            full_map = duration_model.infer(
                edge_logits, temperature=temperature, root_bit=0
            ).map_path
            if mode == "map_corrected":
                coordinates, valid = apply_block_correction(
                    coordinates,
                    valid,
                    full_map,
                    block_frames=config.block_frames,
                )
            frames = len(coordinates)
            starts = list(
                range(0, frames - config.window_frames + 1, config.window_stride)
            )
            final = frames - config.window_frames
            if final < 0:
                continue
            if starts[-1] != final:
                starts.append(final)
            observed_frame_path = example.corrupted["observed_frame_path"]
            for start in starts:
                stop = start + config.window_frames
                start_block = start // config.block_frames
                stop_block = start_block + config.window_frames // config.block_frames
                # Full-sequence evidence is retained while fixing only this
                # window's reference block to chart zero.
                representative = duration_model.infer(
                    edge_logits,
                    temperature=temperature,
                    anchor=(start_block, 0, 1e-9),
                )
                relative_probability = representative.block_swap_probability[
                    start_block:stop_block
                ]
                root_chart = int(observed_frame_path[start])
                teacher = example.corrupted["nuisance_reference"][start:stop].copy()
                teacher_valid = example.corrupted["nuisance_valid"][start:stop].copy()
                if root_chart:
                    chart = np.ones(len(teacher), dtype=np.int8)
                    teacher = semantic_permute_by_frame(teacher, chart)
                    teacher_valid = semantic_permute_by_frame(teacher_valid, chart)
                self.windows.append(
                    GaugeWindow(
                        coordinates=coordinates[start:stop].astype(np.float32),
                        valid=_patch_valid(valid[start:stop]),
                        teacher_coordinates=teacher.astype(np.float32),
                        teacher_valid=_patch_valid(teacher_valid),
                        relative_probability=relative_probability.astype(np.float32),
                        window_id=f"{example.sequence_id}:{start}",
                    )
                )
                self.sampling_groups.append(example.identity)
        if not self.windows:
            raise ValueError("No complete gauge windows were constructed")

    def __len__(self) -> int:
        return len(self.windows)

    def __getitem__(self, index: int) -> dict:
        item = self.windows[index]
        return {
            "coordinates": torch.from_numpy(item.coordinates.copy()),
            "valid": torch.from_numpy(item.valid.copy()),
            "teacher_coordinates": torch.from_numpy(item.teacher_coordinates.copy()),
            "teacher_valid": torch.from_numpy(item.teacher_valid.copy()),
            "relative_probability": torch.from_numpy(item.relative_probability.copy()),
            "window_id": item.window_id,
        }


def _torch_semantic_permute_coordinates(values: torch.Tensor, pairs) -> torch.Tensor:
    result = values.clone()
    original = values.clone()
    for left, right in pairs:
        result[..., left, :] = original[..., right, :]
        result[..., right, :] = original[..., left, :]
    return result


def _align_branch_one(values: torch.Tensor, pairs) -> torch.Tensor:
    result = values.clone()
    original = values.clone()
    for left, right in pairs:
        result[:, :, left] = original[:, :, right]
        result[:, :, right] = original[:, :, left]
    return result


def _semantic_orbit(coordinates: torch.Tensor, pairs) -> torch.Tensor:
    return torch.stack(
        [coordinates, _torch_semantic_permute_coordinates(coordinates, pairs)], dim=1
    )


def semantic_gauge_objective(
    model,
    coordinates: torch.Tensor,
    valid: torch.Tensor,
    teacher_coordinates: torch.Tensor,
    teacher_valid: torch.Tensor,
    relative_probability: torch.Tensor,
    target_mask: torch.Tensor,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Predict a random reference chart using a detached structured posterior."""

    pairs = model.config.mirror_pairs
    orbit = _semantic_orbit(coordinates, pairs)
    teacher_orbit = _semantic_orbit(teacher_coordinates, pairs)
    paired_valid = torch.stack(
        [valid, permute_bilateral_tokens(valid, pairs)], dim=1
    )
    paired_teacher_valid = torch.stack(
        [teacher_valid, permute_bilateral_tokens(teacher_valid, pairs)], dim=1
    )
    paired_mask = orbit_closed_target_masks(target_mask, pairs)
    if (paired_mask & ~paired_valid).any() or (paired_mask & ~paired_teacher_valid).any():
        raise ValueError("Gauge target mask includes an invalid student or teacher token")
    student = model.encoder(
        orbit, keep_mask=~paired_mask, valid_patch=paired_valid
    )
    predicted = model.predictor(student, paired_mask, return_full=True)
    with torch.no_grad():
        targets = model.target_encoder(
            teacher_orbit, valid_patch=paired_teacher_valid
        )
    segments = valid.shape[1]
    joints = valid.shape[2]
    predicted_first = predicted[0].reshape(len(coordinates), segments, joints, -1)
    predicted_second = _align_branch_one(
        predicted[1].reshape(len(coordinates), segments, joints, -1), pairs
    )
    target_first = targets[0].reshape(len(coordinates), segments, joints, -1)
    target_second = _align_branch_one(
        targets[1].reshape(len(coordinates), segments, joints, -1), pairs
    )
    predicted_even = 0.5 * (predicted_first + predicted_second)
    predicted_odd = 0.5 * (predicted_first - predicted_second)
    target_even = 0.5 * (target_first + target_second)
    target_odd = 0.5 * (target_first - target_second)
    selected_probability = relative_probability[:, :, None].expand(-1, -1, joints)[
        target_mask
    ].reshape(len(coordinates), -1)
    selected = target_mask[..., None].expand_as(predicted_even)
    shape = (len(coordinates), -1, predicted_even.shape[-1])
    loss = structured_parity_prediction_loss(
        predicted_even[selected].reshape(shape),
        predicted_odd[selected].reshape(shape),
        target_even[selected].reshape(shape),
        target_odd[selected].reshape(shape),
        selected_probability,
    )
    pooled_even = predicted_even.mean((1, 2))
    pooled_odd = predicted_odd.mean((1, 2))
    variance = torch.relu(1.0 - torch.sqrt(pooled_even.var(0, unbiased=False) + 1e-4)).mean()
    variance = variance + torch.relu(
        1.0 - torch.sqrt(pooled_odd.var(0, unbiased=False) + 1e-4)
    ).mean()
    total = loss + model.config.vicreg_weight * variance
    return total, {
        "structured_prediction_loss": loss,
        "variance_loss": variance,
        "even_features": pooled_even,
        "odd_features": pooled_odd,
    }


@torch.no_grad()
def _evaluate_sg(model, dataset, device, *, seed: int, uniform: bool) -> dict:
    model.eval()
    generator = torch.Generator(device="cpu").manual_seed(seed + 700_000)
    loader = torch.utils.data.DataLoader(dataset, batch_size=model.config.batch_size)
    totals = {"loss": 0.0, "weight": 0}
    features = {"even": [], "odd": []}
    for batch in loader:
        coordinates = batch["coordinates"].to(device)
        valid = batch["valid"].to(device)
        teacher = batch["teacher_coordinates"].to(device)
        teacher_valid = batch["teacher_valid"].to(device)
        eligible = valid & teacher_valid
        target_mask = sample_mask(
            eligible.cpu(),
            model.config.mask_fraction,
            generator,
            model.config.mask_joints,
        ).to(device)
        probability = batch["relative_probability"].to(device)
        if uniform:
            probability = torch.full_like(probability, 0.5)
        loss, terms = semantic_gauge_objective(
            model, coordinates, valid, teacher, teacher_valid, probability, target_mask
        )
        totals["loss"] += float(loss) * len(coordinates)
        totals["weight"] += len(coordinates)
        features["even"].append(terms["even_features"].cpu())
        features["odd"].append(terms["odd_features"].cpu())
    result = {"structured_loss": totals["loss"] / totals["weight"]}
    for name in ("even", "odd"):
        health = representation_metrics(torch.cat(features[name]))
        result.update({f"{name}_{key}": value for key, value in health.items()})
    return result


def fit_sg(
    model,
    train_dataset,
    validation_dataset,
    device,
    *,
    seed: int,
    output_dir: Path,
    epochs: int,
    uniform: bool,
) -> dict:
    model = model.to(device)
    projector = VICRegProjector(model.config.embed_dim).to(device)
    trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
    optimizer = torch.optim.AdamW(
        trainable,
        lr=model.config.learning_rate,
        betas=(0.9, 0.95),
        weight_decay=model.config.weight_decay,
    )
    updates_per_epoch = math.ceil(len(train_dataset) / model.config.batch_size)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(epochs * updates_per_epoch, 1)
    )
    mask_generator = torch.Generator(device="cpu").manual_seed(seed + 600_000)
    best = float("inf")
    history = []
    updates = 0
    best_path = output_dir / f"seed-{seed}_{'uniform_posterior' if uniform else 'sg_jepa'}_best.pt"
    for epoch in range(1, epochs + 1):
        loader = make_train_loader(
            train_dataset,
            model.config,
            updates_per_epoch,
            seed + epoch * 1009,
            num_workers=0,
        )
        model.train()
        running = 0.0
        weight = 0
        for batch in loader:
            coordinates = batch["coordinates"].to(device)
            valid = batch["valid"].to(device)
            teacher = batch["teacher_coordinates"].to(device)
            teacher_valid = batch["teacher_valid"].to(device)
            eligible = valid & teacher_valid
            target_mask = sample_mask(
                eligible.cpu(),
                model.config.mask_fraction,
                mask_generator,
                model.config.mask_joints,
            ).to(device)
            probability = batch["relative_probability"].to(device)
            if uniform:
                probability = torch.full_like(probability, 0.5)
            optimizer.zero_grad(set_to_none=True)
            loss, _ = semantic_gauge_objective(
                model,
                coordinates,
                valid,
                teacher,
                teacher_valid,
                probability,
                target_mask,
            )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable, 1.0)
            optimizer.step()
            scheduler.step()
            model.update_target(0.996)
            running += float(loss.detach()) * len(coordinates)
            weight += len(coordinates)
            updates += 1
        validation = _evaluate_sg(
            model, validation_dataset, device, seed=seed + epoch, uniform=uniform
        )
        row = {
            "epoch": epoch,
            "train_structured_loss": running / weight,
            **{f"validation_{key}": value for key, value in validation.items()},
        }
        history.append(row)
        if validation["structured_loss"] < best:
            best = validation["structured_loss"]
            payload = checkpoint_payload(
                model,
                projector,
                optimizer,
                variant=model.variant,
                seed=seed,
                config=model.config,
                updates=updates,
                epoch=epoch,
                validation_metrics=validation,
            )
            payload["metadata"].update(
                {
                    "study_arm": "uniform_posterior" if uniform else "sg_jepa",
                    "posterior_contract": "separately-calibrated-detached-v1",
                    "paired_mask_contract": PAIRED_MASK_CONTRACT,
                    "input_contract": INPUT_CONTRACT,
                }
            )
            atomic_torch_save(best_path, payload)
    pd.DataFrame(history).to_csv(output_dir / "history.csv", index=False)
    return {
        "best_checkpoint": str(best_path),
        "best_validation_structured_loss": best,
        "optimizer_updates": updates,
    }


def run(args: argparse.Namespace) -> dict:
    manifest_path = args.gauge_manifest.resolve()
    manifest_sha256 = hashlib.sha256(manifest_path.read_bytes()).hexdigest()
    gate = require_benchmark_gate(
        args.gate_decision, gauge_manifest_sha256=manifest_sha256
    )
    output_dir = args.output_dir.resolve()
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"Output directory is not empty: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    config = load_manifest_sequence_config(manifest_path)
    examples = make_manifest_examples(
        manifest_path,
        args.tensor_root.resolve(),
        config=config,
        # The second chart view is an audit/control, not a second independent
        # training exposure.  Orbit lifting supplies the complementary branch.
        chart_members=(0,),
    )
    fit_examples, calibration_examples, validation_examples = _identity_partition(examples)
    head = _fit_continuity_head(fit_examples, config)
    duration = TwoStateDurationModel(max_duration=16)
    from .laterality_corruption_inference import calibrate_duration_temperature

    calibration_logits = _continuity_logits(head, calibration_examples, config)
    temperature = calibrate_duration_temperature(
        duration, calibration_logits, [example.path for example in calibration_examples]
    )
    fit_logits = _continuity_logits(head, fit_examples, config)
    validation_logits = _continuity_logits(head, validation_examples, config)
    mode = "map_corrected" if args.arm == "correction_first_sjepa" else "uncorrected"
    train_dataset = GaugeWindowDataset(
        fit_examples, fit_logits, duration, temperature, mode=mode, config=config
    )
    validation_dataset = GaugeWindowDataset(
        validation_examples,
        validation_logits,
        duration,
        temperature,
        mode=mode,
        config=config,
    )
    train_config = core11_train_config(args.profile)
    device = torch.device(args.device)
    epochs = int(args.epochs or train_config.epochs)
    expected_variant = (
        "standard_sjepa"
        if args.arm == "correction_first_sjepa"
        else "reflection_equivariant"
    )
    if args.initial_checkpoint is not None:
        model, _, initial_metadata = load_checkpoint(args.initial_checkpoint.resolve())
        if model.variant != expected_variant:
            raise ValueError(
                f"Initial checkpoint variant {model.variant!r} does not match "
                f"the required {expected_variant!r}"
            )
    else:
        model = build_model(train_config, expected_variant, args.seed)
    if args.arm == "correction_first_sjepa":
        _, _, _, result, _ = fit_variant(
            model,
            train_dataset,
            validation_dataset,
            device,
            seed=args.seed,
            output_dir=output_dir,
            num_workers=0,
            max_epochs=epochs,
            patience=args.patience,
        )
    else:
        model = build_model(train_config, "reflection_equivariant", args.seed)
        result = fit_sg(
            model,
            train_dataset,
            validation_dataset,
            device,
            seed=args.seed,
            output_dir=output_dir,
            epochs=epochs,
            uniform=args.arm == "uniform_posterior",
        )
    payload = {
        "arm": args.arm,
        "seed": args.seed,
        "temperature": temperature,
        "gate_decision": gate,
        "gauge_manifest_sha256": manifest_sha256,
        "sequence_gauge_config": sequence_gauge_config_json(config),
        "effective_training_source_draws": len(examples),
        "test_split_evaluated": False,
        "initial_checkpoint": (
            str(args.initial_checkpoint.resolve())
            if args.initial_checkpoint is not None
            else None
        ),
        "result": result,
    }
    (output_dir / "run_result.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n"
    )
    return payload


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arm", choices=GAUGE_ARMS, required=True)
    parser.add_argument("--gate-decision", type=Path, required=True)
    parser.add_argument("--gauge-manifest", type=Path, required=True)
    parser.add_argument("--tensor-root", type=Path, required=True)
    parser.add_argument("--initial-checkpoint", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--profile", choices=("smoke", "full"), default="full")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--patience", type=int)
    return parser.parse_args(argv)


def main() -> int:
    print(json.dumps(run(parse_args()), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
