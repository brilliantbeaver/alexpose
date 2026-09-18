"""Small restoration fits with explicit supervision, support, and resume identity.

Source/exposure and scientific cost gates belong to the workflow. This module
fits only explicitly provided training tensors and rejects non-train roles.
Measured wall time is reported; equal steps are never described as equal compute.
"""
from __future__ import annotations

from contextlib import nullcontext
from dataclasses import asdict, dataclass
import hashlib
import json
import math
from pathlib import Path
import time

import numpy as np
import torch

from ..temporal_gait.objectives import feature_diagnostics, per_example_mean, predictive_loss, vicreg_loss
from . import models as model_module
from .models import (ARMS, INPUT_KEYS, LATENT_ARMS, JOINTS, ModelConfig, RestorationModel,
                     frame_hidden, model_config, validate_inputs)

FORMAT = "synthetic-restoration-v2.1"


@dataclass(frozen=True)
class TrainConfig:
    updates: int = 200
    readout_updates: int = 100
    end_to_end_updates: int | None = None
    total_seconds_budget: float | None = None
    batch_size: int = 64
    learning_rate: float = 3e-4
    weight_decay: float = .01
    warmup_fraction: float = .05
    gradient_clip: float = 1.
    mask_fraction: float = .5
    ema_start: float = .99
    ema_end: float = .999
    student_temperature: float = .1
    teacher_temperature: float = .06
    center_momentum: float = .9
    vicreg_weight: float = .05
    translation_magnitude: float = .02
    smoothnet_acceleration_weight: float = .1
    seed: int = 17
    device: str = "cpu"
    precision: str = "float32"
    log_every: int = 20
    checkpoint_every: int = 100
    stop_after_updates: int | None = None
    resume_from: str | None = None

    def validate(self):
        if min(self.updates, self.readout_updates, self.batch_size, self.log_every, self.checkpoint_every) < 1:
            raise ValueError("Positive updates, readout updates, batch and logging intervals required")
        if self.end_to_end_updates is not None and self.end_to_end_updates < 1:
            raise ValueError("end_to_end_updates must be positive")
        if self.total_seconds_budget is not None and (not math.isfinite(self.total_seconds_budget) or self.total_seconds_budget <= 0):
            raise ValueError("A measured total_seconds_budget must be finite and positive")
        if self.stop_after_updates is not None and self.stop_after_updates < 0:
            raise ValueError("stop_after_updates cannot be negative")
        if min(self.learning_rate, self.gradient_clip, self.student_temperature, self.teacher_temperature) <= 0:
            raise ValueError("Positive learning rate, gradient cap and temperatures required")
        if self.weight_decay < 0 or not 0 <= self.warmup_fraction < 1:
            raise ValueError("Invalid weight decay or warmup fraction")
        if not 0 < self.mask_fraction < 1 or not 0 <= self.ema_start <= self.ema_end < 1:
            raise ValueError("Invalid mask or EMA schedule")
        if not 0 <= self.center_momentum < 1:
            raise ValueError("Invalid center momentum")
        if self.vicreg_weight < 0 or self.translation_magnitude < 0:
            raise ValueError("Regularizer weight and translation magnitude must be nonnegative")
        if not math.isfinite(self.smoothnet_acceleration_weight) or self.smoothnet_acceleration_weight < 0:
            raise ValueError("SmoothNet-style reference acceleration weight must be finite nonnegative")
        if self.precision not in {"float32", "bfloat16"}:
            raise ValueError("Use float32 or CUDA bfloat16")
        if self.precision == "bfloat16" and not self.device.startswith("cuda"):
            raise ValueError("bfloat16 requires explicitly allocated CUDA")
        return self


def _train_config(value):
    return (value if isinstance(value, TrainConfig) else TrainConfig(**value)).validate()


def _json(path, value):
    path = Path(path)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")
    tmp.replace(path)


def _hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def _array_hash(values):
    result = {}
    for key, value in sorted(values.items()):
        array = value.detach().cpu().contiguous().numpy() if isinstance(value, torch.Tensor) else np.asarray(value)
        if array.dtype.kind == "O":
            raise ValueError("Object arrays are not permitted")
        result[key] = {"shape": list(array.shape), "dtype": str(array.dtype),
                       "sha256": hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()}
    return result


def _input_tensors(inputs, device):
    if set(inputs) != INPUT_KEYS:
        raise ValueError("Input allow-list rejects reference/unknown fields")
    observed = inputs["observed"]
    is_bool = observed.dtype == torch.bool if isinstance(observed, torch.Tensor) else np.asarray(observed).dtype == np.bool_
    if not is_bool:
        raise ValueError("Observed flags must be explicit bool")
    result = {key: torch.as_tensor(value, device=device,
                dtype=torch.bool if key == "observed" else torch.float32) for key, value in inputs.items()}
    validate_inputs(result)
    return result


def _targets(targets, inputs, device):
    if not {"xy", "valid"} <= set(targets):
        raise ValueError("Training targets require separate xy and valid arrays")
    valid = targets["valid"]
    is_bool = valid.dtype == torch.bool if isinstance(valid, torch.Tensor) else np.asarray(valid).dtype == np.bool_
    if not is_bool:
        raise ValueError("Target validity must be explicit bool")
    result = {"xy": torch.as_tensor(targets["xy"], dtype=torch.float32, device=device),
              "valid": torch.as_tensor(valid, dtype=torch.bool, device=device)}
    if result["xy"].shape != inputs["xy"].shape or result["valid"].shape != inputs["observed"].shape:
        raise ValueError("Target shape differs from fixed input window grid")
    if (result["valid"] & ~torch.isfinite(result["xy"]).all(-1)).any():
        raise ValueError("Valid targets must have finite coordinates")
    return result


def _select(values, indices):
    return {key: value[indices] for key, value in values.items()}


def _signature(arm, inputs, targets, cfg, train_cfg, identity, donors, roles):
    from ..temporal_gait import objectives
    config = asdict(train_cfg)
    config.pop("resume_from")
    config.pop("stop_after_updates")
    files = [Path(__file__), Path(model_module.__file__), Path(objectives.__file__)]
    return {"format": FORMAT, "arm": arm, "model": asdict(cfg), "train": config,
        "caller_identity": identity, "inputs": _array_hash(inputs), "targets": _array_hash(targets),
        "donors": None if donors is None else donors.tolist(), "roles": roles,
        "code": {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in files},
        "runtime": {"torch": str(torch.__version__), "numpy": np.__version__,
                    "device": train_cfg.device, "cuda": torch.version.cuda,
                    "threads": torch.get_num_threads()}}


def _phases(arm, cfg):
    if arm in {"direct", "smoothnet", "static"}:
        return [("end_to_end", cfg.end_to_end_updates or cfg.updates + cfg.readout_updates)]
    if arm == "initialized":
        return [("readout", cfg.readout_updates)]
    return [("pretrain", cfg.updates), ("readout", cfg.readout_updates)]


def _configure_phase(model, phase, reset_readout):
    model.requires_grad_(False)
    if phase == "end_to_end":
        if hasattr(model, "practical"):
            model.practical.requires_grad_(True)
        else:
            model.encoder.requires_grad_(True)
            model.readout.requires_grad_(True)
    elif phase == "pretrain":
        model.encoder.requires_grad_(True)
        (model.predictor if model.arm in LATENT_ARMS else model.readout).requires_grad_(True)
        if model.arm in LATENT_ARMS:
            model.projector.requires_grad_(True)
    else:
        if reset_readout:
            model.readout.load_state_dict(model.initialized_readout.state_dict(), strict=True)
        model.readout.requires_grad_(True)
    model.train()
    if phase == "readout":
        model.encoder.eval()
    return [p for p in model.parameters() if p.requires_grad]


def _mask(cfg, observed, fraction, rng, device):
    shape = (len(observed), cfg.window_size // cfg.patch_size, JOINTS)
    # Artificial hiding samples observed tokens only. Missing-input restoration
    # queries are a separate set and do not consume this masking allowance.
    eligible = _patch_valid(observed, cfg).reshape(shape).cpu().numpy()
    result = np.zeros(shape, bool)
    for row, support in zip(result, eligible):
        available = int(support.sum())
        target = max(1, round(available * fraction)) if available else 0
        target = min(target, max(1, available - 1)) if available else 0
        # Random joint ordering and time-offset blocks prevent a fixed anatomical shortcut.
        joint_order = rng.permutation(JOINTS)
        start = int(rng.integers(shape[1]))
        slots = [(int((start + t) % shape[1]), int(j)) for t in range(shape[1]) for j in joint_order
                 if support[(start + t) % shape[1], j]]
        for t, j in slots[:target]:
            row[t, j] = True
    return torch.as_tensor(result, dtype=torch.bool, device=device)


def _patch_valid(valid, cfg):
    return valid.reshape(len(valid), cfg.window_size // cfg.patch_size, cfg.patch_size, JOINTS).any(2).flatten(1)


def coordinate_loss(predicted, target, valid):
    """Per-window reduction; no unsupported window becomes a zero-error success."""
    safe = torch.where(valid[..., None], target, 0)
    errors = (predicted.float() - safe.float()).square().mean(-1)
    return per_example_mean(errors, valid)


def smoothnet_loss(predicted, target, valid, timestamps, acceleration_weight=.1):
    """Body-12 adaptation: L1 positions plus reference acceleration in seconds.

    Acceleration is supervised against valid reference triples, never minimized
    toward zero. The declared development grid is weight 0 or .1. This is not an
    exact reproduction of the original model's schema, scale or training recipe.
    Unsupported triples are removed before differencing, preventing NaN gradients.
    """
    if predicted.shape != target.shape or valid.shape != predicted.shape[:-1]:
        raise ValueError("SmoothNet-style coordinate/target shape mismatch")
    if timestamps.shape != predicted.shape[:2] or predicted.shape[1] < 3:
        raise ValueError("Acceleration needs at least three aligned physical timestamps")
    dt = timestamps.float().diff(dim=1)
    if not torch.isfinite(dt).all() or (dt <= 0).any() or not torch.allclose(dt, dt[:, :1].expand_as(dt), atol=1e-6, rtol=1e-4):
        raise ValueError("SmoothNet-style acceleration requires a fixed physical-time grid")
    safe = torch.where(valid[..., None], target.float(), 0)
    positions = (predicted.float() - safe).abs().mean(-1)
    _, position_mean, supported = per_example_mean(positions, valid)
    triples = valid[:, :-2] & valid[:, 1:-1] & valid[:, 2:]
    def acceleration(x):
        velocity = x.diff(dim=1) / dt[..., None, None]
        return 2 * velocity.diff(dim=1) / (dt[:, :-1] + dt[:, 1:])[..., None, None]
    errors = (acceleration(predicted.float()) - acceleration(safe)).abs().mean(-1)
    _, acceleration_mean, acceleration_supported = per_example_mean(errors, triples)
    combined = torch.where(supported, position_mean, 0) + acceleration_weight * torch.where(
        acceleration_supported, acceleration_mean, 0)
    loss = combined[supported].mean() if supported.any() else predicted.sum() * 0
    example = torch.where(supported, combined, torch.full_like(combined, float("nan")))
    diagnostics = {"position_l1": float(position_mean[supported].detach().mean()) if supported.any() else None,
                   "reference_acceleration_l1": float(acceleration_mean[acceleration_supported].detach().mean()) if acceleration_supported.any() else None,
                   "acceleration_triple_count": int(triples.sum()), "acceleration_weight": acceleration_weight}
    return loss, example, supported, diagnostics


def _coordinate_patch_loss(predicted, target, target_valid, queries, cfg):
    """Same query tokens/support and window weighting as clean-target JEPA.

    Within each eligible token, coordinate MSE averages its valid target frames.
    Thus partially valid patches do not change relative token/window weights.
    """
    safe = torch.where(target_valid[..., None], target, 0)
    errors = (predicted.float() - safe.float()).square().mean(-1)
    shape = (len(predicted), cfg.window_size // cfg.patch_size, cfg.patch_size, JOINTS)
    flags = target_valid.reshape(shape)
    counts = flags.sum(2)
    errors = torch.where(flags, errors.reshape(shape), 0).sum(2) / counts.clamp_min(1)
    valid = queries.flatten(1) & (counts > 0).flatten(1)
    return (*per_example_mean(errors.flatten(1), valid), valid)


def _teacher_inputs(inputs, targets):
    # Privileged synthetic targets are used only inside the training teacher.
    return {"xy": targets["xy"], "observed": targets["valid"],
            "confidence": targets["valid"].float(), "timestamps": inputs["timestamps"]}


def _schedule(update, total, cfg):
    warmup = max(1, round(cfg.warmup_fraction * total)) if cfg.warmup_fraction else 0
    if update < warmup:
        return cfg.learning_rate * (update + 1) / warmup
    fraction = (update - warmup) / max(total - warmup, 1)
    return cfg.learning_rate * .5 * (1 + math.cos(math.pi * fraction))


def _synchronize(device):
    if device.startswith("cuda"):
        torch.cuda.synchronize(device)


def _save_checkpoint(path, model, optimizer, signature, state, generators):
    payload = {"format": FORMAT, "arm": model.arm, "model_config": model.configuration(),
        "signature": signature, "model": model.state_dict(), "optimizer": optimizer.state_dict(),
        "state": state, "rng": {"torch": torch.get_rng_state(),
            "cuda": torch.cuda.get_rng_state_all() if signature["train"]["device"].startswith("cuda") else [],
            "numpy": {name: rng.bit_generator.state for name, rng in generators.items()}}}
    path = Path(path)
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, tmp)
    tmp.replace(path)


def _restore_rng(payload, generators):
    torch.set_rng_state(payload["rng"]["torch"].cpu())
    if payload["rng"]["cuda"]:
        torch.cuda.set_rng_state_all([x.cpu() for x in payload["rng"]["cuda"]])
    for name, value in payload["rng"]["numpy"].items():
        generators[name].bit_generator.state = value


def fit_arm(arm, inputs, targets, model_cfg, train_cfg, output_dir, identity,
            donor_indices=None, roles=None):
    """Fit one explicit arm; return a model whose forward accepts only inputs.

    Direct/practical models use updates+readout_updates unless overridden by a
    measured equal-compute plan. Frozen arms reset and separately fit identical
    readouts. Shuffled donors must be supplied from allowed training strata; the
    caller verifies person/motion/nuisance matching, recorded in caller identity.
    Both input and target xy must use the SAME noisy-input-derived normalization;
    no target-derived origin/scale is computed here. A measured seconds budget is
    available for direct/practical competitors after timing the complete candidate.
    """
    if arm not in ARMS:
        raise ValueError(f"Unknown arm {arm!r}")
    cfg, tc = model_config(model_cfg), _train_config(train_cfg)
    if tc.total_seconds_budget is not None and arm not in {"direct", "smoothnet", "static"}:
        raise ValueError("Time-budget fitting applies to direct/practical competitors after candidate timing")
    if not identity:
        raise ValueError("Explicit source/protocol identity is required")
    _hash(identity)
    if tc.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("Requested CUDA unavailable; no CPU fallback")
    inputs = _input_tensors(inputs, tc.device)
    validate_inputs(inputs, cfg.window_size)
    targets = _targets(targets, inputs, tc.device)
    count = len(inputs["xy"])
    role_list = None if roles is None else list(roles)
    if role_list is not None and (len(role_list) != count or any(r != "train" for r in role_list)):
        raise ValueError("Fitting accepts only train-role examples")
    donors = None
    if arm == "shuffled_jepa":
        if donor_indices is None:
            raise ValueError("Shuffled pairing needs explicit source-stratified donor_indices")
        donors = np.asarray(donor_indices)
        if donors.shape != (count,) or donors.dtype.kind not in "iu" or np.any((donors < 0) | (donors >= count)):
            raise ValueError("Donor indices must identify training rows")
        if np.any(donors == np.arange(count)):
            raise ValueError("Shuffled pairing must exclude same-window donors")
    elif donor_indices is not None:
        raise ValueError("Donors apply only to shuffled_jepa")
    signature = _signature(arm, inputs, targets, cfg, tc, identity, donors, role_list)
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    if (list(output.glob("checkpoint-*.pt")) or (output / "model.pt").exists()) and not tc.resume_from:
        raise FileExistsError("Existing fit requires explicit resume or a new output directory")
    torch.manual_seed(tc.seed)
    if tc.device.startswith("cuda"):
        torch.cuda.manual_seed_all(tc.seed)
        torch.cuda.reset_peak_memory_stats(tc.device)
    generators = {"draw": np.random.default_rng(tc.seed + 11), "mask": np.random.default_rng(tc.seed + 23)}
    model = RestorationModel(arm, cfg).to(tc.device)
    phases = _phases(arm, tc)
    planned = sum(n for _, n in phases)
    stop = min(planned, planned if tc.stop_after_updates is None else tc.stop_after_updates)
    state = {"update": 0, "phase_index": 0, "phase_update": 0, "attempts": 0,
             "history": [], "draws": [], "elapsed_seconds": 0., "teacher_initialization_weight": 1.,
             "teacher_mean_age_updates": 0., "trainable_parameters_by_phase": {}}
    payload = None
    if tc.resume_from:
        payload = torch.load(tc.resume_from, map_location=tc.device, weights_only=True)
        if payload.get("format") != FORMAT or payload.get("signature") != signature:
            raise ValueError("Incompatible resume identity: data/config/code/runtime/protocol changed")
        model.load_state_dict(payload["model"], strict=True)
        state = payload["state"]
        _restore_rng(payload, generators)
        if stop < state["update"]:
            raise ValueError("Stop precedes resumed optimizer update")
    _synchronize(tc.device)
    started, prior_seconds = time.perf_counter(), float(state["elapsed_seconds"])
    unsupported = 0
    time_exhausted = False
    optimizer = None
    for phase_index, (phase, total) in enumerate(phases):
        if phase_index < state["phase_index"]:
            continue
        continuing = payload is not None and phase_index == state["phase_index"]
        if not continuing and phase_index != state["phase_index"]:
            state["phase_update"] = 0
        state["phase_index"] = phase_index
        parameters = _configure_phase(model, phase, reset_readout=phase == "readout" and not continuing)
        optimizer = torch.optim.AdamW(parameters, lr=tc.learning_rate, weight_decay=tc.weight_decay)
        state["trainable_parameters_by_phase"][phase] = sum(p.numel() for p in parameters)
        if continuing:
            optimizer.load_state_dict(payload["optimizer"])
        elif state["update"] == 0:
            _save_checkpoint(output / "checkpoint-000000.pt", model, optimizer, signature, state, generators)
        while state["phase_update"] < total and state["update"] < stop:
            _synchronize(tc.device)
            elapsed = prior_seconds + time.perf_counter() - started
            if tc.total_seconds_budget is not None and elapsed >= tc.total_seconds_budget:
                time_exhausted = True
                break
            indices = generators["draw"].integers(count, size=tc.batch_size)
            batch, truth = _select(inputs, indices), _select(targets, indices)
            state["attempts"] += 1
            hidden = None
            autocast = torch.autocast("cuda", dtype=torch.bfloat16) if tc.precision == "bfloat16" else nullcontext()
            with autocast:
                if phase == "pretrain":
                    hidden = _mask(cfg, batch["observed"], tc.mask_fraction, generators["mask"], tc.device)
                    missing_patch = (~batch["observed"]).reshape(len(indices),
                        cfg.window_size // cfg.patch_size, cfg.patch_size, JOINTS).any(2)
                    queries = hidden | missing_patch
                if phase == "pretrain" and arm in LATENT_ARMS:
                    online = model.encoder(batch, hidden)
                    predicted = model.predictor(online)
                    target_indices = indices if donors is None else donors[indices]
                    if arm == "ordinary_jepa":
                        teacher_inputs = batch
                        target_valid = batch["observed"]
                    else:
                        teacher_truth = _select(targets, target_indices)
                        teacher_inputs = _teacher_inputs(_select(inputs, target_indices), teacher_truth)
                        target_valid = teacher_truth["valid"]
                    with torch.no_grad():
                        target_tokens = model.teacher(teacher_inputs)
                    valid = queries.flatten(1) & _patch_valid(target_valid, cfg)
                    loss, per_example, supported = predictive_loss(predicted, target_tokens, valid,
                        objective="centered_ce_v1", center=model.center,
                        student_temperature=tc.student_temperature, teacher_temperature=tc.teacher_temperature)
                    predictive_value = loss.detach()
                    regularizer = loss.new_zeros(())
                    if tc.vicreg_weight and supported.sum() >= 2:
                        representations = []
                        for _ in range(2):
                            shift = (torch.rand(len(indices), 1, 1, 2, device=tc.device) * 2 - 1) * tc.translation_magnitude
                            translated = dict(batch, xy=torch.where(batch["observed"][..., None], batch["xy"] + shift, batch["xy"]))
                            representations.append(model.projector(model.encoder(translated).mean(1))[supported])
                        regularizer = vicreg_loss(*representations)
                        loss = loss + tc.vicreg_weight * regularizer
                else:
                    predicted = model(batch, hidden=hidden)
                    valid = truth["valid"]
                    if hidden is not None:
                        loss, per_example, supported, valid = _coordinate_patch_loss(
                            predicted, truth["xy"], valid, queries, cfg)
                    elif arm == "smoothnet":
                        loss, per_example, supported, practical_diagnostics = smoothnet_loss(
                            predicted, truth["xy"], valid, batch["timestamps"], tc.smoothnet_acceleration_weight)
                    else:
                        loss, per_example, supported = coordinate_loss(predicted, truth["xy"], valid)
            draw = {"attempt": state["attempts"], "phase": phase, "indices": indices.tolist(),
                    "target_indices": (indices if donors is None or phase != "pretrain" else donors[indices]).tolist(),
                    "target_support": valid.reshape(len(indices), -1).sum(1).tolist(),
                    "supported_examples": int(supported.sum())}
            if hidden is not None:
                eligible = _patch_valid(batch["observed"], cfg).reshape_as(hidden)
                draw.update(mask_sha256=hashlib.sha256(hidden.cpu().numpy().tobytes()).hexdigest(),
                    query_sha256=hashlib.sha256(queries.cpu().numpy().tobytes()).hexdigest(),
                    missing_input_query_tokens=int(missing_patch.sum()),
                    realized_all_slots_fraction=float(hidden.float().mean()),
                    hidden_observed_token_fraction=float((hidden & eligible).sum() / eligible.sum().clamp_min(1)),
                    input_observed_tokens=int(eligible.sum()))
            state["draws"].append(draw)
            if not supported.any():
                unsupported += 1
                if unsupported >= 20:
                    _json(output / "failure.json", {"status": "unsupported", "reason": "twenty_unsupported_batches",
                                                   "draws": state["draws"]})
                    raise ValueError("Twenty unsupported batches; no zero-error success")
                continue
            unsupported = 0
            if not torch.isfinite(loss):
                raise FloatingPointError("Nonfinite supported training loss")
            if tc.total_seconds_budget is None:
                lr = _schedule(state["phase_update"], total, tc)
            else:
                progress = min(elapsed / tc.total_seconds_budget, 1.)
                warmup = tc.warmup_fraction
                # Positive first-step warmup without inventing a throughput estimate.
                factor = max(progress / warmup, 1e-3) if warmup and progress < warmup else (
                    .5 * (1 + math.cos(math.pi * (progress - warmup) / max(1 - warmup, 1e-8))))
                lr = tc.learning_rate * factor
            for group in optimizer.param_groups:
                group["lr"] = lr
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            if hasattr(model, "teacher") and any(p.grad is not None for p in model.teacher.parameters()):
                raise AssertionError("Teacher received gradients")
            norm = torch.nn.utils.clip_grad_norm_(parameters, tc.gradient_clip, error_if_nonfinite=True)
            optimizer.step()
            entry = {"update": state["update"] + 1, "phase": phase, "phase_update": state["phase_update"] + 1,
                     "loss": float(loss.detach()), "learning_rate": lr, "gradient_norm": float(norm),
                     "supported_examples": int(supported.sum()), "target_count": int(valid.sum())}
            if arm == "smoothnet":
                entry["smoothnet_adaptation"] = practical_diagnostics
            if phase == "pretrain" and arm in LATENT_ARMS:
                fraction = state["phase_update"] / max(total - 1, 1)
                momentum = tc.ema_end - (tc.ema_end - tc.ema_start) * .5 * (1 + math.cos(math.pi * fraction))
                with torch.no_grad():
                    model.update_teacher(momentum)
                    counts = valid.sum(1)
                    means = (target_tokens.float() * valid[..., None]).sum(1) / counts[:, None].clamp_min(1)
                    model.center.mul_(tc.center_momentum).add_(means[supported].mean(0), alpha=1 - tc.center_momentum)
                    probabilities = ((target_tokens.float() - model.center) / tc.teacher_temperature).softmax(-1)
                    entropy = -(probabilities * probabilities.clamp_min(1e-12).log()).sum(-1)
                state["teacher_initialization_weight"] *= momentum
                state["teacher_mean_age_updates"] = momentum * (state["teacher_mean_age_updates"] + 1)
                entry.update(ema=momentum, teacher_entropy=float(entropy[valid].mean()),
                    predictive_loss=float(predictive_value), vicreg_loss=float(regularizer.detach()),
                    teacher_initialization_weight=state["teacher_initialization_weight"],
                    teacher_mean_age_updates=state["teacher_mean_age_updates"])
            state["update"] += 1
            state["phase_update"] += 1
            if state["update"] % tc.log_every == 0 or state["phase_update"] == total:
                with torch.no_grad():
                    if hasattr(model, "encoder"):
                        tokens = model.encoder(batch)
                        # Bounded SVD diagnostics; these do not certify useful gait information.
                        entry["online_features"] = feature_diagnostics(tokens.flatten(0, 1)[:2048])
                        entry["initialized_features"] = feature_diagnostics(model.initialized_encoder(batch).flatten(0, 1)[:2048])
                        zero_coordinates = dict(batch, xy=torch.zeros_like(batch["xy"]))
                        entry["coordinate_dependence_rms"] = float((tokens - model.encoder(zero_coordinates)).float().square().mean().sqrt())
                        if hasattr(model, "teacher"):
                            entry["teacher_features"] = feature_diagnostics(model.teacher(batch).flatten(0, 1)[:2048])
            state["history"].append(entry)
            if state["update"] % tc.checkpoint_every == 0 or state["phase_update"] == total or state["update"] == stop:
                _synchronize(tc.device)
                state["elapsed_seconds"] = prior_seconds + time.perf_counter() - started
                _save_checkpoint(output / f"checkpoint-{state['update']:06d}.pt", model, optimizer, signature, state, generators)
        if state["update"] >= stop or time_exhausted:
            break
        payload = None
    _synchronize(tc.device)
    state["elapsed_seconds"] = prior_seconds + time.perf_counter() - started
    if state["update"] == 0 and tc.total_seconds_budget is not None:
        _json(output / "failure.json", {"status": "insufficient_budget",
              "reason": "time_budget_exhausted_before_first_optimizer_update",
              "elapsed_seconds": state["elapsed_seconds"], "total_seconds_budget": tc.total_seconds_budget})
        raise ValueError("Time budget exhausted before any optimizer update; no trained result")
    complete = state["update"] == planned or time_exhausted
    if tc.total_seconds_budget is not None and state["elapsed_seconds"] >= tc.total_seconds_budget:
        time_exhausted = True
        complete = True
    _save_checkpoint(output / f"checkpoint-{state['update']:06d}.pt", model, optimizer, signature, state, generators)
    report = {"status": "complete" if complete else "interrupted", "evidence_origin": "caller_supplied_training_data",
        "arm": arm, "seed": tc.seed, "optimizer_updates": state["update"], "planned_updates": planned,
        "phases": [{"name": p, "updates": n} for p, n in phases], "elapsed_seconds": state["elapsed_seconds"],
        "gpu_hours": state["elapsed_seconds"] / 3600 if tc.device.startswith("cuda") else 0.,
        "cpu_seconds": state["elapsed_seconds"] if not tc.device.startswith("cuda") else 0.,
        "peak_device_bytes": int(torch.cuda.max_memory_allocated(tc.device)) if tc.device.startswith("cuda") else None,
        "cost_scope": "this trainer including diagnostic forwards/checkpoints; extraction/rendering are recorded separately",
        "compute_matching": "measured total-seconds budget" if tc.total_seconds_budget is not None else "matched updates; not equal compute",
        "total_seconds_budget": tc.total_seconds_budget,
        "time_budget_overrun_seconds": max(0., state["elapsed_seconds"] - tc.total_seconds_budget) if tc.total_seconds_budget is not None else None,
        "termination": "time_budget_reached" if time_exhausted else ("update_limit" if complete else "explicit_interruption"),
        "parameter_count_including_retained_states": sum(p.numel() for p in model.parameters()),
        "trainable_parameters_by_phase": state["trainable_parameters_by_phase"],
        "source_examples": count, "roles_verified": role_list is not None,
        "signature_sha256": _hash(signature), "history": state["history"], "draws": state["draws"],
        "objective": ("centered_ce_v1 then separate coordinate readout" if arm in LATENT_ARMS else
                      "position_l1_plus_reference_acceleration_l1" if arm == "smoothnet" else "coordinate_mse"),
        "teacher_initialization_weight": state["teacher_initialization_weight"] if arm in LATENT_ARMS else None,
        "teacher_mean_age_updates": state["teacher_mean_age_updates"] if arm in LATENT_ARMS else None}
    model.eval()
    model.training_report = report
    _json(output / "training.json", report)
    if complete:
        _save_checkpoint(output / "model.pt", model, optimizer, signature, state, generators)
    return model


def load_fitted(path, *, device="cpu", identity=None):
    """Load a retained fit for inference without advancing the caller's RNG."""
    payload = torch.load(path, map_location=device, weights_only=True)
    if payload.get("format") != FORMAT:
        raise ValueError("Foreign checkpoint format")
    if identity is not None and identity != payload["signature"]["caller_identity"]:
        raise ValueError("Checkpoint protocol/source identity mismatch")
    with torch.random.fork_rng(devices=[]):
        model = RestorationModel(payload["arm"], payload["model_config"])
    model.load_state_dict(payload["model"], strict=True)
    model.to(device).eval().requires_grad_(False)
    report_path = Path(path).parent / "training.json"
    model.training_report = json.loads(report_path.read_text()) if report_path.exists() else {}
    return model


@torch.inference_mode()
def predict(model, inputs, *, batch_size=64):
    """Return NumPy predictions in input order; no target argument is accepted."""
    if batch_size < 1:
        raise ValueError("Positive inference batch size required")
    device = next(model.parameters()).device
    values = _input_tensors(inputs, device)
    model.eval()
    return torch.cat([model(_select(values, slice(start, start + batch_size))).float().cpu()
                      for start in range(0, len(values["xy"]), batch_size)]).numpy()
