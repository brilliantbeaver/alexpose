"""Training-only, initialization-gradient calibration for the response extension."""
from __future__ import annotations

from dataclasses import asdict
from datetime import datetime
from contextlib import nullcontext
from pathlib import Path
import hashlib

import numpy as np
import torch

from ..synthetic_training_v2.models import ModelConfig, RestorationModel
from ..synthetic_training_v2.training import _coordinate_patch_loss, _patch_valid
from ..temporal_gait.objectives import predictive_loss, vicreg_loss
from .common import atomic_json, digest, read_json, sha256
from .masking import sample_mask
from .response_objectives import (VARIANTS, FORMULAS, SUPPORT_RULE, coordinate_response_loss,
                                  latent_response_loss, predictive_diagnostics)

FORMAT = "gait-fidelity-response-calibration-v1"
CALIBRATION_BATCHES = 32
CALIBRATION_SEED = 17
GRADIENT_RATIO = .1
SCHEDULE_KEYS = ("pretraining_updates", "readout_updates", "end_to_end_updates")


def response_code_identity():
    from . import training, measurements
    from ..synthetic_training_v2 import models, training as inherited_training
    from ..temporal_gait import objectives
    paths = [Path(__file__), Path(__file__).with_name("response_objectives.py"),
             Path(__file__).with_name("masking.py"), Path(training.__file__), Path(measurements.__file__)]
    result = {p.name: sha256(p) for p in paths}
    result.update({"inherited/" + Path(m.__file__).name: sha256(m.__file__)
                   for m in (models, inherited_training, objectives)})
    return result


def response_data_identity(bundle):
    from .training import _records, paired_indices, _array_signature
    from .data import IndexedArray
    records = _records(bundle)
    rows = np.unique(paired_indices(records))
    return digest(dict(inputs=_array_signature({k: IndexedArray(v, rows) for k, v in bundle.inputs.items()}),
                       targets=_array_signature({k: IndexedArray(v, rows) for k, v in bundle.targets.items()}),
                       records=[records[i] for i in rows]))


def calibration_protocol(config):
    training = {k: v for k, v in config.get("training", {}).items()
                if k not in (*SCHEDULE_KEYS, "resume_from")}
    return dict(model=asdict(ModelConfig(**config["model"]).validate()), training=training,
                measurement=config.get("measurement", {}), mask="graph_time", batches=CALIBRATION_BATCHES,
                seed=CALIBRATION_SEED, initial_gradient_ratio=GRADIENT_RATIO,
                gradient_parameters="all_trainable_parameters_unused_gradients_are_zero",
                reduction="sqrt(sum_batch_squared_l2_gradient/same_parameter_count/batches)",
                support=SUPPORT_RULE, formulas=FORMULAS)


def prepare_response_batch(bundle, selected, hidden, model_config, device="cpu"):
    """The trainer's independent context normalization and adjacent endpoint layout."""
    from .training import _take, normalize_batch, _tensors
    cfg = model_config if isinstance(model_config, ModelConfig) else ModelConfig(**model_config).validate()
    selected = np.asarray(selected, dtype=np.int64)
    if len(selected) % 2:
        raise ValueError("Response batches contain adjacent paired endpoints")
    raw = _take(bundle.inputs, selected)
    normalized, origin, scale, fallback = normalize_batch(raw, hidden, patch_size=cfg.patch_size)
    inputs = _tensors(normalized, device)
    valid = torch.as_tensor(bundle.targets["valid"][selected], dtype=torch.bool, device=device)
    target = torch.as_tensor((bundle.targets["xy"][selected] - origin[:, None, None]) /
                             scale[:, None, None, None], dtype=torch.float32, device=device)
    mask = torch.as_tensor(hidden, dtype=torch.bool, device=device)
    missing = (~inputs["observed"]).reshape(len(selected), cfg.window_size // cfg.patch_size, cfg.patch_size, 12).any(2)
    return dict(inputs=inputs, target=target, valid=valid, queries=mask | missing, hidden=mask,
                origin=origin, scale=scale, fallback=fallback, selected=selected)


def response_forward_losses(model, batch, training, variant):
    """One ordinary pretraining forward plus the specified auxiliary loss."""
    if variant not in VARIANTS or model.arm != VARIANTS[variant]:
        raise ValueError("Response variant does not match its encoder")
    inputs, target, valid, queries, hidden = (batch[k] for k in ("inputs", "target", "valid", "queries", "hidden"))
    extra = {}
    if model.arm == "paired_jepa":
        predicted_tokens = model.predictor(model.encoder(inputs, hidden))
        teacher_inputs = dict(xy=torch.where(valid[..., None], target, 0), observed=valid,
                              confidence=valid.float(), timestamps=inputs["timestamps"])
        with torch.no_grad():
            teacher_tokens = model.teacher(teacher_inputs)
        query_valid = queries.flatten(1) & _patch_valid(valid, model.cfg)
        temperatures = dict(student_temperature=float(training.get("student_temperature", .1)),
                            teacher_temperature=float(training.get("teacher_temperature", .06)))
        loss, _, supported = predictive_loss(predicted_tokens, teacher_tokens, query_valid,
                                             objective="centered_ce_v1", center=model.center, **temperatures)
        extra["predictive_loss"] = float(loss.detach())
        extra.update(predictive_diagnostics(predicted_tokens, teacher_tokens, query_valid,
                                           center=model.center, **temperatures))
        weight = float(training.get("vicreg_weight", .05))
        if weight and int(supported.sum()) >= 2:
            views = []
            for _ in range(2):
                shift = (torch.rand(len(target), 1, 1, 2, device=target.device) * 2 - 1) * float(training.get("translation_magnitude", .02))
                translated = dict(inputs, xy=torch.where(inputs["observed"][..., None], inputs["xy"] + shift, inputs["xy"]))
                views.append(model.projector(model.encoder(translated, hidden).mean(1))[supported])
            regularizer = vicreg_loss(*views)
            loss = loss + weight * regularizer
            extra["vicreg_loss"] = float(regularizer.detach())
        auxiliary, diagnostic = latent_response_loss(predicted_tokens, teacher_tokens, queries, valid, model.cfg,
                                                      variant=variant, center=model.center, **temperatures)
        result = dict(predicted_tokens=predicted_tokens, teacher_tokens=teacher_tokens)
    else:
        predicted = model(inputs, hidden)
        loss, _, supported, query_valid = _coordinate_patch_loss(predicted, target, valid, queries, model.cfg)
        extra["coordinate_or_practical_loss"] = float(loss.detach())
        auxiliary, diagnostic = coordinate_response_loss(predicted, target, queries, valid, batch["scale"], model.cfg)
        result = dict(predicted=predicted)
    extra["response_auxiliary"] = diagnostic
    return dict(base_loss=loss, auxiliary_loss=auxiliary, supported=supported,
                query_valid=query_valid, extra=extra, **result)


def gradient_statistics(loss, parameters, *, retain_graph=True):
    """Squared L2 norm and RMS over a fixed parameter set, including unused zeros."""
    parameters = list(parameters)
    gradients = torch.autograd.grad(loss, parameters, retain_graph=retain_graph, allow_unused=True)
    total = loss.new_zeros((), dtype=torch.float32)
    for gradient in gradients:
        if gradient is not None:
            total = total + gradient.detach().float().square().sum()
    value = float(total)
    if not np.isfinite(value):
        raise FloatingPointError("Nonfinite response component gradient")
    count = sum(p.numel() for p in parameters)
    return dict(squared_l2=value, l2=value ** .5, rms=(value / count) ** .5, parameters=count)


def _calibration_draws(bundle, config):
    from .training import (_records, paired_indices, hierarchical_pair_groups, draw_hierarchical_pairs,
                           repaired_pairs, paired_batch_cycles, draw_pair_batch, _measurement_settings)
    records = _records(bundle)
    pairs = paired_indices(records)
    tc = config.get("training", {})
    batch_size = int(tc.get("batch_size", 16))
    if batch_size < 2 or batch_size % 2:
        raise ValueError("Calibration needs an even endpoint batch size")
    sampling = tc.get("sampling", "matched_cycles")
    if sampling == "person_motion":
        groups = hierarchical_pair_groups(records, pairs)
    elif sampling == "matched_cycles":
        repaired, _ = repaired_pairs(bundle, pairs, CALIBRATION_SEED,
            tolerance=float(tc.get("repaired_tolerance", .1)), trials=int(tc.get("repaired_trials", 64)),
            **_measurement_settings(config))
        cycles = paired_batch_cycles(pairs, repaired)
    else:
        raise ValueError("Unknown calibration sampling rule")
    batch_rng, mask_rng = np.random.default_rng(CALIBRATION_SEED + 11), np.random.default_rng(CALIBRATION_SEED + 23)
    for _ in range(CALIBRATION_BATCHES):
        draw = (draw_hierarchical_pairs(groups, batch_size // 2, batch_rng) if sampling == "person_motion"
                else draw_pair_batch(cycles, batch_size // 2, batch_rng))
        selected = pairs[draw].reshape(-1)
        hidden = sample_mask(bundle.inputs["observed"][selected], "graph_time", rng=mask_rng,
                             fraction=float(tc.get("mask_fraction", .5)), patch_size=config["model"]["patch_size"])
        yield selected, hidden


def calibrate_response(bundle, config, output):
    """Calibrate on 32 seed-17 training batches at initialization; never optimize."""
    from .data import validate_bundle
    validate_bundle(bundle)
    if any(row["split"] == "confirmation" for row in bundle.records):
        raise PermissionError("Confirmation tensors cannot enter calibration")
    device = str(config.get("device", "cpu"))
    if device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("Requested calibration CUDA is unavailable")
    devices = list(range(torch.cuda.device_count())) if device.startswith("cuda") else []
    protocol = calibration_protocol(config)
    precision = config.get("training", {}).get("precision", "float32")
    if precision not in {"float32", "bfloat16"} or (precision == "bfloat16" and not device.startswith("cuda")):
        raise ValueError("Use float32 calibration, or bfloat16 on the allocated CUDA GPU")
    totals, reports, batch_receipts = {}, {}, []
    # fork_rng restores caller RNGs even if a calibration validation fails.
    with torch.random.fork_rng(devices=devices):
        models = {}
        for arm in ("paired_jepa", "coordinate"):
            torch.manual_seed(CALIBRATION_SEED)
            model = RestorationModel(arm, config["model"]).to(device).train().requires_grad_(False)
            model.encoder.requires_grad_(True)
            if arm == "paired_jepa":
                model.predictor.requires_grad_(True); model.projector.requires_grad_(True)
            else:
                model.readout.requires_grad_(True)
            models[arm] = model
        torch.manual_seed(CALIBRATION_SEED + 104729)
        for selected, hidden in _calibration_draws(bundle, config):
            batch = prepare_response_batch(bundle, selected, hidden, models["paired_jepa"].cfg, device)
            batch_receipts.append(dict(endpoint_indices=selected.tolist(), mask_sha256=hashlib.sha256(hidden.tobytes()).hexdigest()))
            autocast = torch.autocast("cuda", dtype=torch.bfloat16) if precision == "bfloat16" else nullcontext()
            with autocast:
                j = response_forward_losses(models["paired_jepa"], batch, config.get("training", {}), "jepa_delta_v1")
                endpoint, _ = latent_response_loss(j["predicted_tokens"], j["teacher_tokens"], batch["queries"], batch["valid"],
                    models["paired_jepa"].cfg, variant="jepa_endpoint_v1", center=models["paired_jepa"].center,
                    student_temperature=float(config.get("training", {}).get("student_temperature", .1)),
                    teacher_temperature=float(config.get("training", {}).get("teacher_temperature", .06)))
                c = response_forward_losses(models["coordinate"], batch, config.get("training", {}), "coordinate_delta_v1")
            for name, loss, arm in (("jepa_base", j["base_loss"], "paired_jepa"),
                                    ("jepa_delta_v1", j["auxiliary_loss"], "paired_jepa"),
                                    ("jepa_endpoint_v1", endpoint, "paired_jepa"),
                                    ("coordinate_base", c["base_loss"], "coordinate"),
                                    ("coordinate_delta_v1", c["auxiliary_loss"], "coordinate")):
                if not torch.isfinite(loss):
                    raise FloatingPointError("Nonfinite calibration loss")
                stats = gradient_statistics(loss, (p for p in models[arm].parameters() if p.requires_grad))
                totals[name] = totals.get(name, 0.) + stats["squared_l2"]
                reports[name] = dict(parameters=stats["parameters"])
    if any(not np.isfinite(v) or v <= 0 for v in totals.values()):
        raise FloatingPointError("Zero or nonfinite calibration gradient energy")
    lambda_j = GRADIENT_RATIO * (totals["jepa_base"] / max(totals["jepa_delta_v1"], totals["jepa_endpoint_v1"])) ** .5
    lambda_c = GRADIENT_RATIO * (totals["coordinate_base"] / totals["coordinate_delta_v1"]) ** .5
    for name, value in totals.items():
        reports[name].update(sum_squared_l2=value, rms=(value / (CALIBRATION_BATCHES * reports[name]["parameters"])) ** .5)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    path = output / "calibration.json"
    receipt = dict(format=FORMAT, status="complete", protocol=protocol, protocol_sha256=digest(protocol),
                   data_sha256=response_data_identity(bundle), code=response_code_identity(),
                   schedule={k: config.get("training", {}).get(k) for k in SCHEDULE_KEYS},
                   coefficients=dict(jepa_delta_v1=lambda_j, jepa_endpoint_v1=lambda_j, coordinate_delta_v1=lambda_c),
                   gradients=reports, batches=batch_receipts, batch_sha256=digest(batch_receipts),
                   calibration_receipt=str(path.resolve()), fixture=bool(config.get("fixture", False)))
    if path.exists() and read_json(path) != receipt:
        raise ValueError("An immutable calibration receipt already exists with different identity")
    atomic_json(path, receipt)
    return receipt


def verified_response_identity(config, variant, data_identity):
    """Validate the complete calibration contract; profile exceptions are explicit."""
    if variant not in VARIANTS:
        raise ValueError("Unknown response representation variant")
    settings = config.get("response", {})
    path = settings.get("calibration_receipt")
    if not path:
        raise ValueError("Response training requires its calibration receipt")
    receipt = read_json(path)
    if receipt.get("format") != FORMAT or receipt.get("status") != "complete":
        raise ValueError("Incomplete or unknown response calibration receipt")
    protocol = calibration_protocol(config)
    for name, expected in (("protocol", protocol), ("protocol_sha256", digest(protocol)),
                            ("data_sha256", data_identity), ("code", response_code_identity())):
        if receipt.get(name) != expected:
            raise ValueError(f"Response calibration mismatch: {name}")
    if receipt.get("batch_sha256") != digest(receipt.get("batches")) or len(receipt.get("batches", [])) != CALIBRATION_BATCHES:
        raise ValueError("Response calibration batch receipt is inconsistent")
    timing = settings.get("timing_probe", False)
    if type(timing) is not bool:
        raise ValueError("timing_probe must be boolean")
    for key in SCHEDULE_KEYS:
        actual, planned = config.get("training", {}).get(key), receipt["schedule"][key]
        allowed = timing and key in {"pretraining_updates", "readout_updates"} and type(actual) is int and type(planned) is int and 1 <= actual <= min(20, planned)
        if actual != planned and not allowed:
            raise ValueError(f"Response calibration schedule mismatch: {key}")
    coefficients = receipt.get("coefficients", {})
    if set(coefficients) != set(VARIANTS) or any(not np.isfinite(v) or v <= 0 for v in coefficients.values()):
        raise ValueError("Calibration coefficients must be positive and finite")
    if coefficients["jepa_delta_v1"] != coefficients["jepa_endpoint_v1"]:
        raise ValueError("The two JEPA variants must share their calibrated coefficient")
    deadline = settings.get("deadline_utc")
    grace = settings.get("checkpoint_grace_seconds", 120)
    if deadline is None and not config.get("fixture", False):
        raise ValueError("Source response training requires an explicit deadline_utc")
    if deadline is not None:
        try:
            parsed = datetime.fromisoformat(deadline.replace("Z", "+00:00"))
            if parsed.tzinfo is None:
                raise ValueError("timezone missing")
        except (TypeError, ValueError, AttributeError) as exc:
            raise ValueError("Response deadline must be an explicit timezone-aware ISO timestamp") from exc
    if type(grace) is not int or not 1 <= grace <= 600:
        raise ValueError("Checkpoint grace must be between 1 and 600 seconds")
    return dict(variant=variant, formula=FORMULAS[variant], support=SUPPORT_RULE,
                calibration_sha256=sha256(path), calibration_protocol_sha256=receipt["protocol_sha256"],
                coefficient=coefficients[variant], timing_probe=timing,
                deadline_utc=deadline, checkpoint_grace_seconds=grace)
