"""Resumable masked/future JEPA training, preserving complete sampling state.

No HAIC discovery, implicit data fallback, protected-test tuning, or distributed
training is performed here. Checkpoints are loadable with weights_only=True.
Exact restart requires the same config/data/device/runtime and deterministic
kernels. A total schedule horizon is fixed before the first optimizer update.
"""
from __future__ import annotations

from contextlib import nullcontext
import hashlib
import json
import math
from pathlib import Path
import random
import sys
import time
from types import SimpleNamespace

import numpy as np
import torch

from .masking import SourceBoutSampler, control_target_indices, patch_validity, sample_feature_mask
from .models import CONTEXT_KEYS, JEPA
from .objectives import feature_diagnostics, predictive_loss, vicreg_loss


FORMAT_VERSION = "temporal_gait_checkpoint_v1"
ARMS = {"masked", "masked_index", "future", "future_wrong_source", "future_wrong_time"}
MODEL_FIELDS = ("hidden_dim", "encoder_depth", "predictor_depth", "heads", "patch_size",
                "clock_channels", "horizons", "grid_hz")


def _config_dict(cfg):
    return cfg.to_dict() if hasattr(cfg, "to_dict") else vars(cfg).copy()


def _plain(value):
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(k): _plain(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [_plain(v) for v in value]
    return value


def _json_hash(value):
    return hashlib.sha256(json.dumps(_plain(value), sort_keys=True, allow_nan=False).encode()).hexdigest()


def _dataset_hash(dataset):
    digest = hashlib.sha256(json.dumps(_plain(dataset.records), sort_keys=True, allow_nan=False).encode())
    for name, array in sorted(dataset.arrays.items()):
        value = np.ascontiguousarray(array)
        if value.dtype.hasobject:
            raise ValueError("Object arrays are forbidden in training inputs")
        digest.update(name.encode())
        digest.update(str(value.dtype).encode())
        digest.update(str(value.shape).encode())
        digest.update(memoryview(value).cast("B"))
    return digest.hexdigest()


def dataset_fingerprint(dataset):
    """Public identity for checking readout and SSL training partitions match."""
    return _dataset_hash(dataset)


def _code_fingerprint():
    """Direct train calls carry code identity even without the workflow wrapper."""
    directory = Path(__file__).resolve().parent
    names = {"models.py", "masking.py", "objectives.py", "training.py", "config.py", "windows.py",
             "observations.py", "geometry.py", "manifests.py", "splits.py", "data.py",
             "preprocessing.py", "contracts.py"}
    files = {path.name: hashlib.sha256(path.read_bytes()).hexdigest()
             for path in sorted(directory.glob("*.py")) if path.name in names}
    return _json_hash(files)


def _signature(cfg, task, dataset):
    config = _plain(_config_dict(cfg))
    # Runtime output placement/early interruption do not change the schedule.
    for name in ("run_root", "resume_from", "stop_after_updates", "checkpoint_updates"):
        config.pop(name, None)
    return {"config": _json_hash(config), "dataset": _dataset_hash(dataset),
            "task": _json_hash({key: task[key] for key in ("task_id", "arm", "seed", "fold")}),
            "runtime": {"torch": str(torch.__version__), "numpy": np.__version__, "python": sys.version,
                        "device": str(cfg.device), "threads": torch.get_num_threads(),
                        "cuda_version": torch.version.cuda,
                        "matmul_precision": torch.get_float32_matmul_precision()},
            "code": _code_fingerprint(), "format": FORMAT_VERSION}


def context_batch(dataset, indices, device="cpu", *, arm="masked"):
    """Only this allow-list may cross the model's forecasting input boundary."""
    prefix = "index_" if arm == "masked_index" else ""
    result = {}
    for key in CONTEXT_KEYS:
        source = prefix + key
        if source not in dataset.arrays:
            raise ValueError(f"Prepared dataset is missing {source!r}; no silent input substitution")
        dtype = torch.bool if key == "context_valid" else torch.float32
        result[key] = torch.as_tensor(dataset.arrays[source][indices], dtype=dtype, device=device)
    return result


def _translation_view(context, magnitude):
    view = dict(context)
    shifts = (torch.rand(len(context["context"]), 1, 1, 2,
                         device=context["context"].device) * 2 - 1) * magnitude
    view["context"] = torch.where(context["context_valid"][..., None],
                                   context["context"] + shifts, 0)
    return view


def _rng_state(generators, device):
    return {"python": random.getstate(), "torch": torch.get_rng_state(),
            "cuda": torch.cuda.get_rng_state_all() if str(device).startswith("cuda") else [],
            "generators": {name: rng.bit_generator.state for name, rng in generators.items()}}


def _restore_rng(state, generators):
    random.setstate(state["python"])
    torch.set_rng_state(state["torch"].cpu())
    if state["cuda"]:
        if not torch.cuda.is_available():
            raise ValueError("CUDA checkpoint cannot exactly resume without CUDA")
        torch.cuda.set_rng_state_all([value.cpu() for value in state["cuda"]])
    for name, rng in generators.items():
        rng.bit_generator.state = state["generators"][name]


def _save_json(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(_plain(value), indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def _checkpoint(path, model, optimizer, cfg, task, signature, generators,
                update, attempts, history, draws):
    payload = {"format": FORMAT_VERSION, "config": _plain(_config_dict(cfg)),
               "task": _plain(task), "signature": signature, "update": update,
               "attempts": attempts, "model": model.state_dict(),
               "optimizer": optimizer.state_dict(), "rng": _rng_state(generators, cfg.device),
               "history": history, "draws": draws,
               "spaces": {"encode": "validity-weighted pooled final LayerNorm",
                          "centered_ce_v1": "centered teacher/raw feature-logit CE",
                          "feature_regression_v1": "per-vector LayerNorm SmoothL1",
                          "teacher_prediction": "EMA encoder plus online predictor"},
               "recipe_claim": "historical_schedule_adaptation; not exact historical replay"}
    temporary = path.with_suffix(".pt.tmp")
    torch.save(payload, temporary)
    temporary.replace(path)
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _load(path, device="cpu"):
    value = torch.load(path, map_location=device, weights_only=True)
    if value.get("format") != FORMAT_VERSION:
        raise ValueError("Unsupported or foreign checkpoint format")
    return value


def load_model(cfg, checkpoint_path):
    """Restore every encoder state without resetting or advancing caller RNG."""
    payload = _load(checkpoint_path)
    expected = _plain(_config_dict(cfg))
    for field in MODEL_FIELDS:
        if _plain(payload["config"][field]) != expected[field]:
            raise ValueError(f"Checkpoint/model config mismatch: {field}")
    with torch.random.fork_rng(devices=[]):
        model = JEPA(SimpleNamespace(**payload["config"]))
    model.load_state_dict(payload["model"], strict=True)
    model.to(cfg.device).eval()
    model.checkpoint_metadata = {"update": payload["update"], "signature": payload["signature"],
                                "arm": payload["task"]["arm"], "spaces": payload["spaces"],
                                "task": {key: payload["task"][key]
                                         for key in ("task_id", "seed", "fold", "arm")},
                                "objective": payload["config"].get("objective", "centered_ce_v1"),
                                "recipe_claim": payload["recipe_claim"]}
    return model


def train_condition(cfg, task, dataset, output_dir):
    """Fit an explicitly supplied training partition and emit restartable artifacts.

    Optional cfg.stop_after_updates interrupts after that many total optimizer
    updates without changing cfg.updates (the frozen schedule horizon). Resume
    only through an explicit cfg.resume_from checkpoint. Task rows may override
    neither optimizer settings nor data. Unsupported batches are logged/skipped;
    repeated absence of targets raises rather than creating zero-loss results.
    """
    invocation_start = time.perf_counter()
    arm = task["arm"]
    if arm not in ARMS:
        raise ValueError(f"Unknown training arm {arm!r}")
    if arm == "future_wrong_time" and len(cfg.horizons) < 2:
        raise ValueError("wrong_horizon_order control requires at least two fixed horizons")
    if arm == "masked_index" and (cfg.patch_size != 4 or bool(cfg.clock_channels)):
        raise ValueError("Primary masked_index attribution holds patch4 and clocks off")
    if not len(dataset) or any(row.get("role") != "train" for row in dataset.records):
        raise ValueError("Training accepts nonempty train-role windows only")
    if len({row["window_id"] for row in dataset.records}) != len(dataset):
        raise ValueError("Window IDs must be unique")
    if int(cfg.updates) < 1 or int(cfg.batch_size) < 1:
        raise ValueError("Positive optimizer budget and batch size required")
    objective = getattr(cfg, "objective", "centered_ce_v1")
    if objective not in {"centered_ce_v1", "feature_regression_v1"}:
        raise ValueError("Explicit supported objective required")
    precision = getattr(cfg, "precision", "fp32")
    if precision not in {"fp32", "float32", "bf16", "bfloat16"}:
        raise ValueError("Supported training precision: float32 or CUDA bfloat16")
    use_bfloat16 = precision in {"bf16", "bfloat16"}
    use_cuda = str(cfg.device).startswith("cuda")
    if use_bfloat16 and not str(cfg.device).startswith("cuda"):
        raise ValueError("bfloat16 training requires an explicitly selected CUDA device")
    if use_cuda:
        if not torch.cuda.is_available():
            raise RuntimeError("Explicit CUDA device unavailable; no automatic CPU fallback")
        torch.cuda.reset_peak_memory_stats(cfg.device)
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    resume = getattr(cfg, "resume_from", None)
    if any(output.glob("checkpoint-*.pt")) and not resume:
        raise FileExistsError("Existing checkpoints require explicit resume_from; refusing overwrite")
    signature = _signature(cfg, task, dataset)
    random.seed(int(task["seed"]))
    torch.manual_seed(int(task["seed"]))
    torch.use_deterministic_algorithms(True)
    generators = {"draw": np.random.default_rng(int(task["seed"]) + 11),
                  "mask": np.random.default_rng(int(task["seed"]) + 23),
                  "control": np.random.default_rng(int(task["seed"]) + 37)}
    sampler = SourceBoutSampler(dataset.records, generators["draw"])
    model = JEPA(cfg).to(cfg.device)
    parameters = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(parameters, lr=cfg.learning_rate, weight_decay=cfg.weight_decay)
    update, attempts, history, draws = 0, 0, [], []
    if resume:
        payload = _load(resume, cfg.device)
        if payload["signature"] != signature:
            raise ValueError("Incompatible resume: config, task, dataset, runtime or schedule changed")
        model.load_state_dict(payload["model"], strict=True)
        optimizer.load_state_dict(payload["optimizer"])
        update, attempts = payload["update"], payload["attempts"]
        history, draws = payload["history"], payload["draws"]
        _restore_rng(payload["rng"], generators)
    else:
        _checkpoint(output / "checkpoint-000000.pt", model, optimizer, cfg, task, signature,
                    generators, update, attempts, history, draws)
    starting_update, starting_attempts = update, attempts
    stop = min(int(cfg.updates), int(getattr(cfg, "stop_after_updates", None) or cfg.updates))
    if stop < update:
        raise ValueError("Requested stop precedes resumed optimizer update")
    schedule = getattr(cfg, "lr_schedule", "constant")
    if schedule not in {"constant", "cosine", "warmup_cosine"}:
        raise ValueError("Unknown learning-rate schedule")
    checkpoint_updates = set(int(x) for x in cfg.checkpoint_updates)
    checkpoint_updates.add(stop)
    failed_in_row = 0
    model.train()
    while update < stop:
        indices = sampler.draw(int(cfg.batch_size))
        context = context_batch(dataset, indices, cfg.device, arm=arm)
        attempts += 1
        target_indices = list(indices)
        autocast = (torch.autocast("cuda", dtype=torch.bfloat16)
                    if use_bfloat16 else nullcontext())
        with autocast:
            if arm.startswith("masked"):
                if "index_context_valid" not in dataset.arrays:
                    raise ValueError("Matched masking requires physical and index validity on identical windows")
                physical_valid = torch.as_tensor(dataset.arrays["context_valid"][indices],
                                                  device=cfg.device, dtype=torch.bool)
                index_valid = torch.as_tensor(dataset.arrays["index_context_valid"][indices],
                                               device=cfg.device, dtype=torch.bool)
                candidates = (patch_validity(physical_valid, cfg.patch_size) &
                              patch_validity(index_valid, cfg.patch_size))
                hidden = sample_feature_mask(candidates, cfg.mask_fraction, generators["mask"])
                predicted, target, valid = model.masked_prediction(context, hidden)
                name = "masked"
            else:
                target_indices = control_target_indices(dataset.records, indices, arm, generators["control"],
                    float(cfg.prefix_seconds) + max(cfg.horizons) + float(cfg.target_interval_seconds))
                future = torch.as_tensor(dataset.arrays["future"][target_indices], device=cfg.device,
                                         dtype=torch.float32)
                future_valid = torch.as_tensor(dataset.arrays["future_valid"][target_indices],
                                               device=cfg.device, dtype=torch.bool)
                predicted = model.predict_future(context)
                with torch.no_grad():
                    target = model.encode_future_targets(future, future_valid, "teacher")
                valid = future_valid.flatten(2).any(2)
                if arm == "future_wrong_time":
                    # Encode first, then permute complete targets AND support.
                    # This preserves their multiset even when teacher clocks
                    # are enabled; no future observation enters the predictor.
                    target = target.roll(1, dims=1)
                    valid = valid.roll(1, dims=1)
                valid = valid & context["context_valid"].flatten(1).any(1)[:, None]
                name = "future"
            predictive, example_loss, supported = predictive_loss(predicted, target, valid,
                objective=objective, center=getattr(model, "center_" + name),
                student_temperature=getattr(cfg, "student_temperature", .1),
                teacher_temperature=getattr(cfg, "teacher_temperature", .06))
            draw = {"attempt": attempts, "update_before": update,
                    "window_ids": [dataset.records[i]["window_id"] for i in indices],
                    "target_window_ids": [dataset.records[i]["window_id"] for i in target_indices],
                    "supported_examples": int(supported.sum()), "target_tokens": int(valid.sum()),
                    "valid_target_counts_per_example": valid.reshape(len(valid), -1).sum(1).tolist()}
            if arm.startswith("future"):
                observations = future_valid.flatten(2).sum(2)
                if arm == "future_wrong_time":
                    observations = observations.roll(1, dims=1)
                draw["valid_future_observations_per_example_horizon"] = observations.tolist()
            if arm == "future_wrong_time":
                draw["control"] = "wrong_horizon_order"
                draw["target_horizon_indices"] = np.roll(np.arange(len(cfg.horizons)), 1).tolist()
            if arm.startswith("masked"):
                draw["mask_sha256"] = hashlib.sha256(hidden.cpu().numpy().tobytes()).hexdigest()
                draw["matched_target_candidates"] = candidates.flatten(1).sum(1).tolist()
                whole_valid = patch_validity(context["context_valid"], cfg.patch_size)
                target_counts = hidden.flatten(1).sum(1)
                draw["context_usable_tokens"] = whole_valid.flatten(1).sum(1).tolist()
                candidate_fraction = (target_counts.float() / candidates.flatten(1).sum(1).clamp_min(1)).tolist()
                draw["realized_mask_fractions"] = candidate_fraction
                draw["realized_candidate_mask_fraction"] = candidate_fraction
                draw["realized_whole_body_mask_fraction"] = (
                    target_counts.float() / whole_valid.flatten(1).sum(1).clamp_min(1)).tolist()
                draw["realized_all_token_slots_mask_fraction"] = (
                    target_counts.float() / hidden[0].numel()).tolist()
                visible = whole_valid & ~hidden
                previous = torch.zeros_like(visible)
                following = torch.zeros_like(visible)
                previous[:, 1:] = visible[:, :-1]
                following[:, :-1] = visible[:, 1:]
                adjacent = (previous.float() + following.float()) * hidden
                fractions = ((adjacent > 0).flatten(1).sum(1).float() / target_counts.clamp_min(1)).tolist()
                averages = (adjacent.flatten(1).sum(1) / target_counts.clamp_min(1)).tolist()
                draw["visible_temporal_neighbor_fraction_per_example"] = [
                    value if count else None for value, count in zip(fractions, target_counts.tolist())]
                draw["visible_temporal_neighbors_per_target"] = [
                    value if count else None for value, count in zip(averages, target_counts.tolist())]
                draw["temporal_neighbor_definition"] = "immediate preceding/following patch, same joint, valid and unmasked"
            draws.append(draw)
            if not supported.any():
                failed_in_row += 1
                if failed_in_row >= 20:
                    _save_json(output / "failure.json", {"reason": "twenty_unsupported_batches",
                        "optimizer_updates": update, "attempts": attempts, "draws": draws})
                    raise ValueError("Twenty unsupported batches: no valid targets; no result was manufactured")
                continue
            failed_in_row = 0
            regularizer = predictive.new_zeros(())
            weight = float(getattr(cfg, "vicreg_weight", .05))
            if weight:
                magnitude = float(getattr(cfg, "view_translation", .02))
                first = model.encode(_translation_view(context, magnitude))[supported]
                second = model.encode(_translation_view(context, magnitude))[supported]
                regularizer = vicreg_loss(model.projector(first), model.projector(second))
            loss = predictive + weight * regularizer
        if not torch.isfinite(loss):
            raise FloatingPointError(f"Nonfinite loss before optimizer update {update + 1}")
        fraction = update / max(int(cfg.updates) - 1, 1)
        if schedule == "constant":
            lr = float(cfg.learning_rate)
        else:
            warmup = max(1, round(float(getattr(cfg, "warmup_fraction", .05)) * cfg.updates))
            lr = cfg.learning_rate * ((update + 1) / warmup if update < warmup else
                .5 * (1 + math.cos(math.pi * (update - warmup) / max(cfg.updates - warmup - 1, 1))))
        for group in optimizer.param_groups:
            group["lr"] = lr
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        if any(p.grad is not None for p in model.teacher.parameters()):
            raise AssertionError("EMA teacher received gradient")
        norm = torch.nn.utils.clip_grad_norm_(parameters, float(getattr(cfg, "gradient_clip", 1.)),
                                             error_if_nonfinite=True)
        optimizer.step()
        momentum = cfg.ema_end - (cfg.ema_end - cfg.ema_start) * .5 * (1 + math.cos(math.pi * fraction))
        model.update_teacher(momentum)
        model.update_center(target.detach().float(), valid, name,
                            float(getattr(cfg, "center_momentum", .9)))
        update += 1
        metrics = {"update": update, "attempt": attempts, "loss": float(loss.detach()),
                   "predictive_loss": float(predictive.detach()), "vicreg_loss": float(regularizer.detach()),
                   "lr": lr, "ema": float(momentum), "gradient_norm_before_clip": float(norm),
                   "supported_examples": int(supported.sum()),
                   "per_example_loss": [float(x) if torch.isfinite(x) else None for x in example_loss.detach()],
                   "predicted": feature_diagnostics(predicted, valid),
                   "teacher": feature_diagnostics(target, valid)}
        # This diagnostic forward is deterministic and does not advance any RNG.
        with torch.no_grad():
            metrics["context"] = feature_diagnostics(model.encode(context), supported)
        history.append(metrics)
        if update in checkpoint_updates:
            _checkpoint(output / f"checkpoint-{update:06d}.pt", model, optimizer, cfg, task, signature,
                        generators, update, attempts, history, draws)
    final_path = output / f"checkpoint-{update:06d}.pt"
    if not final_path.is_file():
        _checkpoint(final_path, model, optimizer, cfg, task, signature, generators,
                    update, attempts, history, draws)
    _save_json(output / "history.json", history)
    _save_json(output / "draws.json", draws)
    drawn_ids = {key for draw in draws for key in draw["window_ids"]}
    drawn_rows = [row for row in dataset.records if row["window_id"] in drawn_ids]
    target_ids = {key for draw in draws for key in draw["target_window_ids"]}
    target_rows = [row for row in dataset.records if row["window_id"] in target_ids]
    if use_cuda:
        torch.cuda.synchronize(cfg.device)
    elapsed = time.perf_counter() - invocation_start
    usage = {"wall_seconds_this_invocation": elapsed,
             "optimizer_updates_this_invocation": update - starting_update,
             "sampling_attempts_this_invocation": attempts - starting_attempts,
             "optimizer_updates_total": update,
             "cuda_peak_allocated_bytes": int(torch.cuda.max_memory_allocated(cfg.device)) if use_cuda else None,
             "gpu_name": torch.cuda.get_device_name(cfg.device) if use_cuda else None,
             "gpu_hours": elapsed / 3600 if use_cuda and cfg.mode == "real" else None,
             "gpu_hours_definition": "one_selected_CUDA_device_wall_hours_this_invocation; not Slurm-accounted GPU allocation",
             "scope": "nondeterministic_resource_receipt_excluded_from_loss_history"}
    record = {"task_id": task["task_id"], "arm": arm, "seed": task["seed"], "fold": task["fold"],
              "status": "trained" if update == cfg.updates else "interrupted",
              "mode": cfg.mode, "optimizer_updates": update, "attempts": attempts,
              "objective": objective, "recipe_claim": "historical_schedule_adaptation_not_exact_replay",
              "control": "wrong_horizon_order" if arm == "future_wrong_time" else arm,
              "checkpoint": str(final_path), "checkpoint_sha256": hashlib.sha256(final_path.read_bytes()).hexdigest(),
              "history": str(output / "history.json"), "draws": str(output / "draws.json"),
              "usage": usage,
              "unique_training_draws": {"windows": len(drawn_rows),
                  "videos": len({row["video_id"] for row in drawn_rows}),
                  "bouts": len({(row["video_id"], row["sequence_id"]) for row in drawn_rows}),
                  "groups": len({row["group_id"] for row in drawn_rows})},
              "unique_teacher_target_draws": {"windows": len(target_rows),
                  "videos": len({row["video_id"] for row in target_rows}),
                  "bouts": len({(row["video_id"], row["sequence_id"]) for row in target_rows}),
                  "groups": len({row["group_id"] for row in target_rows})},
              "signature": signature}
    _save_json(output / "training.json", record)
    return record
