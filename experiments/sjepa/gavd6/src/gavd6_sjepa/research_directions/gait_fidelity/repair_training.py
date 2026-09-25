"""Immutable, matched frozen-readout fits for the scalar-versus-dense repair.

The source pretraining checkpoint is admitted by its bound file/signature hashes
and unchanged inherited implementation. New objective code has its own receipt;
no legacy checkpoint or global provenance rule is rewritten.
"""
from __future__ import annotations

from contextlib import nullcontext
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import time
from types import SimpleNamespace

import numpy as np
import torch

from ..synthetic_training_v2.models import ModelConfig, RestorationModel, CoordinateReadout
from ..synthetic_training_v2.training import coordinate_loss
from . import training as legacy
from .common import atomic_json, digest, read_json, sha256, utc_now
from .data import IndexedArray, select_rows, validate_bundle
from .repair_objectives import FORMULAS, repair_measurement_terms, angle_gradient_support
from .response_calibration import response_code_identity, response_data_identity

CALIBRATION_FORMAT = "gait-fidelity-repair-calibration-v1"
REPAIR_FORMAT = "gait-fidelity-readout-repair-v1"
VARIANTS = ("jepa_delta_v1", "jepa_endpoint_v1")
OBJECTIVES = ("scalar_low", "dense_change")
SCALAR_FRACTION = .1


def repair_code_identity():
    result = response_code_identity()
    for name in ("repair_training.py", "repair_objectives.py"):
        result[name] = sha256(Path(__file__).with_name(name))
    return result


def _deadline(config):
    settings = config.get("repair", {})
    text = settings.get("deadline_utc")
    grace = settings.get("checkpoint_grace_seconds", 120)
    if type(grace) is not int or not 1 <= grace <= 600:
        raise ValueError("Checkpoint grace must be an integer between 1 and 600 seconds")
    if text is None:
        if not config.get("fixture", False):
            raise ValueError("Source repair requires an explicit deadline_utc")
        return None, grace
    try:
        value = datetime.fromisoformat(text.replace("Z", "+00:00"))
        if value.tzinfo is None:
            raise ValueError("timezone missing")
    except (ValueError, TypeError, AttributeError) as exc:
        raise ValueError("Repair deadline must be a timezone-aware ISO timestamp") from exc
    return value, grace


def _deadline_reached(config):
    deadline, grace = _deadline(config)
    return (not config.get("fixture", False) and deadline is not None
            and (deadline-datetime.now(timezone.utc)).total_seconds() <= grace)


def _context(bundle, upstream, config, *, seed=None):
    """Read only training tensors; bind every inherited assumption to its source."""
    records = legacy._records(bundle)
    if any(row["split"] == "confirmation" for row in records):
        raise PermissionError("Confirmation tensors cannot enter repair training or calibration")
    people = {}
    for row in records:
        if people.setdefault(row["canonical_person_id"], row["split"]) != row["split"]:
            raise ValueError("A person crosses training and development splits")
    pairs = legacy.paired_indices(records)
    rows = np.unique(pairs)
    # Validation uses a lazy training-only view; held-out references remain unopened.
    validate_bundle(select_rows(bundle, rows))
    tc = {k: v for k, v in config.get("training", {}).items() if k != "resume_from"}
    cfg = ModelConfig(**config["model"]).validate()
    if cfg.window_size != bundle.inputs["xy"].shape[1]:
        raise ValueError("Repair model window differs from the prepared time grid")
    updates, batch_size = int(tc.get("readout_updates", 2000)), int(tc.get("batch_size", 16))
    lr = float(tc.get("learning_rate", 3e-4))
    weight = float(tc.get("change_weight", 1.))
    if updates < 1 or batch_size < 2 or batch_size % 2 or not np.isfinite(lr) or lr <= 0:
        raise ValueError("Repair needs positive updates/lr and an even endpoint batch size")
    if not np.isfinite(weight) or weight <= 0:
        raise ValueError("Repair comparison requires a positive inherited change_weight")
    for key in ("weight_decay", "warmup_fraction", "gradient_clip"):
        if key in tc and (not np.isfinite(tc[key]) or tc[key] < 0):
            raise ValueError(f"Invalid inherited {key}")
    if float(tc.get("gradient_clip", 1.)) <= 0:
        raise ValueError("gradient_clip must be positive")
    if tc.get("normalization", "context_only") != "context_only":
        raise ValueError("Repair retains context-only input normalization")
    if int(tc.get("checkpoint_every", 100)) < 1 or int(tc.get("log_every", 100)) < 1:
        raise ValueError("Checkpoint/log intervals must be positive")
    device = str(config.get("device", "cpu"))
    fixture = bool(config.get("fixture", False))
    if not fixture and not device.startswith("cuda"):
        raise RuntimeError("Source repair requires allocated CUDA; CPU is fixture-only")
    if device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("Requested CUDA is unavailable; no silent CPU fallback")
    precision = tc.get("precision", "float32")
    if precision not in {"float32", "bfloat16"} or (precision == "bfloat16" and not device.startswith("cuda")):
        raise ValueError("Use float32, or bfloat16 on allocated CUDA")
    _deadline(config)
    batches = config.get("repair", {}).get("calibration_batches", 32)
    if type(batches) is not int or not 1 <= batches <= 32:
        raise ValueError("Calibration must use between 1 and 32 fixed training batches")
    upstream = Path(upstream)
    payload = torch.load(upstream, map_location="cpu", weights_only=True)
    signature = payload.get("signature", {})
    variant, upstream_seed = signature.get("representation_variant"), signature.get("seed")
    if payload.get("format") != legacy.FORMAT or payload.get("status") != "complete" or signature.get("phase") != "pretrain":
        raise ValueError("Repair upstream must be complete Gait Fidelity pretraining")
    if variant not in VARIANTS or signature.get("encoder") != "paired_jepa" or signature.get("policy") != "graph_time":
        raise ValueError("Repair upstream must be the registered delta/endpoint JEPA representation")
    if type(upstream_seed) is not int or (seed is not None and int(seed) != upstream_seed):
        raise ValueError("Repair upstream seed mismatch")
    binding = config.get("repair", {}).get("upstream_bindings", {}).get(f"{variant}/seed-{upstream_seed}")
    if not isinstance(binding, dict):
        raise ValueError("Repair requires its immutable upstream_bindings entry")
    if Path(binding.get("checkpoint", "")).resolve() != upstream.resolve():
        raise ValueError("Repair upstream binding path mismatch")
    for key, actual in (("checkpoint_sha256", sha256(upstream)), ("signature_sha256", digest(signature))):
        if binding.get(key) != actual:
            raise ValueError(f"Repair upstream binding mismatch: {key}")
    if "signature" in binding and binding["signature"] != signature:
        raise ValueError("Repair upstream bound signature differs")
    data_identity = response_data_identity(bundle)
    expected = dict(model=asdict(cfg), training=tc, data_sha256=data_identity,
                    measurement=legacy._measurement_settings(config),
                    normalization="context_only_fixed_isotropic_input_observations",
                    evidence_status=bundle.evidence_status,
                    sampling_hz=float(bundle.provenance.get("hz", 25.)), code=response_code_identity())
    for key, value in expected.items():
        if signature.get(key) != value:
            raise ValueError(f"Repair upstream mismatch: {key}")
    if payload.get("updates") != int(tc.get("pretraining_updates", 2000)):
        raise ValueError("Repair upstream did not complete its declared pretraining updates")
    if signature.get("response", {}).get("variant") != variant:
        raise ValueError("Repair upstream response variant is inconsistent")
    sampling = tc.get("sampling", "matched_cycles")
    cycles, hierarchy, audit = [], None, None
    if sampling == "person_motion":
        hierarchy = legacy.hierarchical_pair_groups(records, pairs)
    elif sampling == "matched_cycles":
        repaired, audit = legacy.repaired_pairs(bundle, pairs, upstream_seed,
            tolerance=float(tc.get("repaired_tolerance", .1)), trials=int(tc.get("repaired_trials", 64)),
            **expected["measurement"])
        cycles = legacy.paired_batch_cycles(pairs, repaired)
    else:
        raise ValueError("Unknown inherited sampling rule")
    return dict(records=records, pairs=pairs, rows=rows, tc=tc, cfg=cfg, updates=updates,
        batch_size=batch_size, lr=lr, geometry_coefficient=weight, scalar_coefficient=SCALAR_FRACTION*weight,
        device=device, precision=precision, variant=variant, seed=upstream_seed, payload=payload,
        upstream=upstream, upstream_sha256=binding["checkpoint_sha256"],
        upstream_signature_sha256=binding["signature_sha256"], data_sha256=data_identity,
        measurement=expected["measurement"], sampling=sampling, cycles=cycles, hierarchy=hierarchy,
        pairing_audit=audit, calibration_batches=batches)


def _model(context):
    seed, device, cfg = context["seed"], context["device"], context["cfg"]
    torch.manual_seed(seed)
    if device.startswith("cuda"):
        torch.cuda.manual_seed_all(seed)
    model = RestorationModel("paired_jepa", cfg).to(device)
    model.load_state_dict(context["payload"]["model"], strict=True)
    model.requires_grad_(False)
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed+100003)
        model.readout = CoordinateReadout(cfg).to(device)
    model.readout.requires_grad_(True)
    model.train()
    model.encoder.eval()
    return model


def _draw(context, rng):
    count = context["batch_size"]//2
    ids = (legacy.draw_hierarchical_pairs(context["hierarchy"], count, rng)
           if context["sampling"] == "person_motion" else legacy.draw_pair_batch(context["cycles"], count, rng))
    return ids, context["pairs"][ids].reshape(-1)


def safe_coordinate_loss(predicted, target, valid):
    """Mask both operands before subtraction, including invalid NaN placeholders."""
    return coordinate_loss(torch.where(valid[..., None], predicted, 0),
                           torch.where(valid[..., None], target, 0), valid)


def _forward(bundle, model, context, selected):
    raw = legacy._take(bundle.inputs, selected)
    normalized, origin, scale, fallback = legacy.normalize_batch(raw, patch_size=context["cfg"].patch_size)
    device = context["device"]
    valid_np = bundle.targets["valid"][selected]
    truth = bundle.targets["xy"][selected]
    # Zero invalid coordinates before normalization arithmetic, not after NaNs propagate.
    safe_truth = np.where(valid_np[..., None], truth, origin[:, None, None])
    target = torch.as_tensor((safe_truth-origin[:, None, None])/scale[:, None, None, None], dtype=torch.float32, device=device)
    valid = torch.as_tensor(valid_np, dtype=torch.bool, device=device)
    autocast = torch.autocast("cuda", dtype=torch.bfloat16) if context["precision"] == "bfloat16" else nullcontext()
    with autocast:
        predicted = model(legacy._tensors(normalized, device))
        base, _, supported = safe_coordinate_loss(predicted, target, valid)
    predicted_px = predicted.float()*torch.as_tensor(scale[:, None, None, None], device=device) + torch.as_tensor(origin[:, None, None], device=device)
    target_px = torch.as_tensor(truth, dtype=torch.float32, device=device)
    t = context["cfg"].window_size
    terms = repair_measurement_terms(predicted_px.reshape(-1, 2, t, 12, 2),
        target_px.reshape(-1, 2, t, 12, 2), valid.reshape(-1, 2, t, 12), **context["measurement"])
    if not supported.all() or not terms["eligible"].any():
        raise ValueError("Repair batch lacks coordinate or paired reference support")
    if any(not bool(torch.isfinite(value)) for value in (base, terms["scalar"], terms["dense"], terms["geometry"])):
        raise FloatingPointError("Nonfinite repair objective")
    return base, terms, fallback


def _gradient(loss, parameters):
    gradients = torch.autograd.grad(loss, parameters, retain_graph=True, allow_unused=True)
    return torch.cat([(torch.zeros_like(parameter) if value is None else value).detach().float().reshape(-1)
                      for parameter, value in zip(parameters, gradients)])


def _stats(vector):
    squared = float(vector.double().square().sum())
    return dict(parameters=vector.numel(), squared_l2=squared, l2=math.sqrt(squared),
                rms=math.sqrt(squared/vector.numel()))


def _gradient_audit(base, terms, parameters):
    values = {name: _gradient(loss, parameters) for name, loss in
              (("coordinate", base), ("scalar", terms["scalar"]), ("dense", terms["dense"]), ("geometry", terms["geometry"]))}
    result = {name: _stats(value) for name, value in values.items()}
    for name in ("scalar", "dense", "geometry"):
        denominator = result["coordinate"]["l2"]*result[name]["l2"]
        result[name]["cosine_with_coordinate"] = (float((values["coordinate"].double()*values[name].double()).sum())/denominator
                                                      if denominator > 0 else None)
    return result


def _protocol(context, config):
    return dict(format=REPAIR_FORMAT, representation_variant=context["variant"], seed=context["seed"],
        upstream_sha256=context["upstream_sha256"], upstream_signature_sha256=context["upstream_signature_sha256"],
        data_sha256=context["data_sha256"], model=asdict(context["cfg"]), training=context["tc"],
        measurement=context["measurement"], code=repair_code_identity(),
        calibration_batches=context["calibration_batches"], batch_seed=context["seed"]+11,
        readout_initialization_seed=context["seed"]+100003, formulas=FORMULAS,
        scalar_coefficient=context["scalar_coefficient"], geometry_coefficient=context["geometry_coefficient"],
        matching="dense coefficient times initial dense gradient RMS equals low-scalar coefficient times initial scalar gradient RMS",
        gradient_parameters="all readout parameters; unused gradients counted as zero",
        reduction="sqrt(sum_batch_squared_l2_gradient/(batches*parameter_count))",
        sampling=context["sampling"], fixture=bool(config.get("fixture", False)))


def calibrate_repair(bundle, upstream: Path, config: dict, output: Path):
    """Fix one encoder/seed coefficient from <=32 training-only initial gradients."""
    context = _context(bundle, upstream, config)
    protocol = _protocol(context, config)
    output = Path(output)
    path = output/"calibration.json"
    if path.exists():
        return _verify_calibration(path, context, config)
    output.mkdir(parents=True, exist_ok=True)
    totals, batches = {key: 0. for key in ("coordinate", "scalar", "dense", "geometry")}, []
    devices = list(range(torch.cuda.device_count())) if context["device"].startswith("cuda") else []
    with torch.random.fork_rng(devices=devices):
        model = _model(context)
        parameters = [p for p in model.parameters() if p.requires_grad]
        initial = legacy._state_hash(model.readout)
        frozen = legacy._state_hash(model.encoder)
        rng = np.random.default_rng(context["seed"]+11)
        for _ in range(context["calibration_batches"]):
            if _deadline_reached(config):
                atomic_json(output/"interrupted.json", dict(status="interrupted", reason="deadline", completed_batches=len(batches)))
                raise legacy.ResponseDeadlineReached("Repair calibration reached deadline; no coefficient admitted")
            pair_ids, selected = _draw(context, rng)
            base, terms, fallback = _forward(bundle, model, context, selected)
            stats = _gradient_audit(base, terms, parameters)
            for name in totals:
                totals[name] += stats[name]["squared_l2"]
            batches.append(dict(endpoint_indices=selected.tolist(), pair_table_indices=pair_ids.tolist(),
                endpoint_multiset_sha256=hashlib.sha256(np.sort(selected).tobytes()).hexdigest(),
                gradients=stats, angle_gradient_support=angle_gradient_support(terms),
                measurement=terms["diagnostic"], normalization_fallbacks=fallback))
        if legacy._state_hash(model.readout) != initial or legacy._state_hash(model.encoder) != frozen:
            raise AssertionError("Calibration changed model weights")
    if any(not np.isfinite(totals[key]) or totals[key] <= 0 for key in ("scalar", "dense")):
        raise FloatingPointError("Zero or nonfinite auxiliary calibration gradient energy")
    coefficients = dict(scalar_low=context["scalar_coefficient"],
                        dense_change=context["scalar_coefficient"]*math.sqrt(totals["scalar"]/totals["dense"]),
                        geometry=context["geometry_coefficient"])
    count = sum(p.numel() for p in parameters)
    gradients = {name: dict(sum_squared_l2=value, parameters=count,
        rms=math.sqrt(value/(count*len(batches)))) for name, value in totals.items()}
    receipt = dict(format=CALIBRATION_FORMAT, status="complete", protocol=protocol,
        protocol_sha256=digest(protocol), coefficients=coefficients, gradients=gradients,
        batches=batches, batch_sha256=digest(batches), initial_readout_sha256=initial,
        frozen_encoder_sha256=frozen, calibration_receipt=str(path.resolve()),
        representation_variant=context["variant"], seed=context["seed"], training_only=True)
    receipt["identity_sha256"] = digest(receipt)
    atomic_json(path, receipt)
    return receipt


def _verify_calibration(path, context, config):
    receipt = read_json(path)
    if receipt.get("format") != CALIBRATION_FORMAT or receipt.get("status") != "complete":
        raise ValueError("Repair requires complete calibration")
    if receipt.get("identity_sha256") != digest({k: v for k, v in receipt.items() if k != "identity_sha256"}):
        raise ValueError("Repair calibration identity mismatch")
    protocol = _protocol(context, config)
    if receipt.get("protocol") != protocol or receipt.get("protocol_sha256") != digest(protocol):
        raise ValueError("Repair calibration protocol mismatch")
    batches = receipt.get("batches", [])
    if len(batches) != context["calibration_batches"] or receipt.get("batch_sha256") != digest(batches):
        raise ValueError("Repair calibration batch receipt mismatch")
    rng = np.random.default_rng(context["seed"]+11)
    totals = {name: 0. for name in ("coordinate", "scalar", "dense", "geometry")}
    for batch in batches:
        pair_ids, selected = _draw(context, rng)
        if batch.get("endpoint_indices") != selected.tolist() or batch.get("pair_table_indices") != pair_ids.tolist():
            raise ValueError("Repair calibration changed its fixed training-only batches")
        for name in totals:
            totals[name] += batch["gradients"][name]["squared_l2"]
    if any(not np.isfinite(totals[key]) or totals[key] <= 0 for key in ("scalar", "dense")):
        raise ValueError("Invalid calibration auxiliary gradient energy")
    expected = dict(scalar_low=context["scalar_coefficient"], geometry=context["geometry_coefficient"],
        dense_change=context["scalar_coefficient"]*math.sqrt(totals["scalar"]/totals["dense"]))
    if receipt.get("coefficients") != expected:
        raise ValueError("Repair calibration coefficients differ from the recorded gradient match")
    if receipt.get("training_only") is not True:
        raise ValueError("Repair calibration is not training-only")
    return receipt


@torch.inference_mode()
def predict_with_deadline(bundle, checkpoint, config, *, output):
    """Input-only streaming export with a cooperative deadline at every batch.

    Publish the final filename only when the full prediction array exists. An
    interrupted partial export is retained for inspection, never a fit receipt.
    """
    device = str(config.get("device", "cpu"))
    model, _ = legacy.load_model(checkpoint, device=device)
    if set(bundle.inputs) != legacy.INPUT_KEYS:
        raise ValueError("Repair inference input allow-list violation")
    batch_size = int(config.get("training", {}).get("batch_size", 32))
    if batch_size < 1:
        raise ValueError("Repair prediction batch size must be positive")
    output = Path(output)
    partial = output.with_name(output.name+".partial.npy")
    predictions = np.lib.format.open_memmap(partial, mode="w+", dtype=np.float32, shape=bundle.inputs["xy"].shape)
    try:
        for start in range(0, len(predictions), batch_size):
            if _deadline_reached(config):
                raise legacy.ResponseDeadlineReached("Repair prediction export reached deadline; partial array is not admitted")
            raw = legacy._take(bundle.inputs, slice(start, start+batch_size))
            normalized, origin, scale, _ = legacy.normalize_batch(raw)
            value = model(legacy._tensors(normalized, device)).float().cpu().numpy()
            predictions[start:start+len(value)] = value*scale[:, None, None, None]+origin[:, None, None]
        predictions.flush()
    finally:
        del predictions
    partial.replace(output)
    return output


def train_repair(bundle, upstream: Path, config: dict, output: Path, *, objective: str,
                 seed: int, calibration: Path):
    """Train one matched, frozen-encoder readout, retaining exact resume state."""
    if objective not in OBJECTIVES:
        raise ValueError("Unknown repair objective")
    context = _context(bundle, upstream, config, seed=seed)
    calibration = Path(calibration)
    calibrated = _verify_calibration(calibration, context, config)
    tc, device = context["tc"], context["device"]
    model = _model(context)
    initial_head = legacy._state_hash(model.readout)
    frozen_before = legacy._state_hash(model.encoder)
    if initial_head != calibrated["initial_readout_sha256"] or frozen_before != calibrated["frozen_encoder_sha256"]:
        raise ValueError("Repair calibration/readout initialization differs")
    parameters = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(parameters, lr=context["lr"], weight_decay=float(tc.get("weight_decay", .01)))
    rng = np.random.default_rng(seed+11)
    signature = dict(format=legacy.FORMAT, phase="readout", encoder="paired_jepa", policy="graph_time",
        objective=objective, seed=int(seed), representation_variant=context["variant"], model=asdict(context["cfg"]),
        training=tc, measurement=context["measurement"], data_sha256=context["data_sha256"],
        normalization="context_only_fixed_isotropic_input_observations", evidence_status=bundle.evidence_status,
        sampling_hz=float(bundle.provenance.get("hz", 25.)), device=device, code=repair_code_identity(),
        upstream_sha256=context["upstream_sha256"], upstream_signature_sha256=context["upstream_signature_sha256"],
        pairs_sha256=hashlib.sha256(context["pairs"].tobytes()).hexdigest(),
        batch_cycles_sha256=digest([cycle.tolist() for cycle in context["cycles"]]),
        repair=dict(format=REPAIR_FORMAT, protocol_sha256=calibrated["protocol_sha256"],
            calibration_sha256=sha256(calibration), calibration_identity_sha256=calibrated["identity_sha256"],
            angle_coefficient=calibrated["coefficients"][objective], geometry_coefficient=calibrated["coefficients"]["geometry"],
            initial_readout_sha256=initial_head, deadline_utc=config.get("repair", {}).get("deadline_utc"),
            checkpoint_grace_seconds=config.get("repair", {}).get("checkpoint_grace_seconds", 120),
            formulas=FORMULAS, upstream_representation_variant=context["variant"]))
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    resume = Path(config.get("training", {}).get("resume_from", output/"checkpoint.pt"))
    history, prior_seconds, start_step = [], 0., 0
    if resume.exists():
        retained = torch.load(resume, map_location=device, weights_only=True)
        if retained.get("format") != legacy.FORMAT or retained.get("signature") != signature:
            raise ValueError("Repair resume rejected: data, code, calibration or configuration changed")
        start_step = retained["updates"]
        if type(start_step) is not int or not 0 <= start_step <= context["updates"] or len(retained["history"]) != start_step:
            raise ValueError("Repair resume has inconsistent update count")
        model.load_state_dict(retained["model"], strict=True)
        optimizer.load_state_dict(retained["optimizer"])
        history, prior_seconds = retained["history"], retained["elapsed_seconds"]
        rng.bit_generator.state = retained["numpy_rng"]["batch"]
        torch.set_rng_state(retained["torch_rng"].cpu())
        if device.startswith("cuda"):
            torch.cuda.set_rng_state_all([value.cpu() for value in retained["cuda_rng"]])
        if legacy._state_hash(model.encoder) != frozen_before:
            raise ValueError("Repair resume modified the frozen encoder")
    started = time.perf_counter()

    def save(status):
        if device.startswith("cuda"):
            torch.cuda.synchronize(device)
        elapsed = prior_seconds+time.perf_counter()-started
        legacy._save(output/"checkpoint.pt", dict(format=legacy.FORMAT, status=status, signature=signature,
            model=model.state_dict(), optimizer=optimizer.state_dict(), updates=len(history), history=history,
            elapsed_seconds=elapsed, torch_rng=torch.get_rng_state(),
            cuda_rng=torch.cuda.get_rng_state_all() if device.startswith("cuda") else [],
            numpy_rng={"batch": rng.bit_generator.state}))
        return elapsed

    for step in range(start_step, context["updates"]):
        if _deadline_reached(config):
            save("interrupted")
            raise legacy.ResponseDeadlineReached("Repair deadline reached; resumable checkpoint saved")
        pair_ids, selected = _draw(context, rng)
        base, terms, fallback = _forward(bundle, model, context, selected)
        key = "scalar" if objective == "scalar_low" else "dense"
        coefficient = calibrated["coefficients"][objective]
        geometry_coefficient = calibrated["coefficients"]["geometry"]
        loss = base+coefficient*terms[key]+geometry_coefficient*terms["geometry"]
        if not bool(torch.isfinite(loss)):
            raise FloatingPointError("Nonfinite repair loss")
        warmup = max(1, round(float(tc.get("warmup_fraction", .05))*context["updates"]))
        progress = (step-warmup)/max(context["updates"]-warmup, 1)
        scheduled_lr = context["lr"]*((step+1)/warmup if step < warmup else .5*(1+math.cos(math.pi*progress)))
        for group in optimizer.param_groups:
            group["lr"] = scheduled_lr
        optimizer.zero_grad(set_to_none=True)
        audit = step == 0 or step+1 == context["updates"] or (step+1) % int(tc.get("log_every", 100)) == 0
        extra = dict(gradients=_gradient_audit(base, terms, parameters), angle_gradient_support=angle_gradient_support(terms)) if audit else {}
        loss.backward()
        if any(p.grad is not None for p in model.encoder.parameters()) or any(p.grad is not None for p in model.teacher.parameters()):
            raise AssertionError("A frozen encoder or privileged teacher received repair gradients")
        clip = float(tc.get("gradient_clip", 1.))
        norm = torch.nn.utils.clip_grad_norm_(parameters, clip, error_if_nonfinite=True)
        optimizer.step()
        history.append(dict(update=step+1, loss=float(loss.detach()), coordinate_loss=float(base.detach()),
            learning_rate=scheduled_lr, gradient_norm=float(norm), gradient_was_clipped=bool(float(norm) > clip),
            gradient_clip=clip, angle_coefficient=coefficient, geometry_coefficient=geometry_coefficient,
            endpoint_indices=selected.tolist(), pair_table_indices=pair_ids.tolist(),
            endpoint_multiset_sha256=hashlib.sha256(np.sort(selected).tobytes()).hexdigest(),
            normalization_fallbacks=fallback, measurement=terms["diagnostic"], **extra))
        if step == start_step or audit:
            event = dict(event="repair_training_progress", phase="readout", representation_variant=context["variant"],
                objective=objective, seed=int(seed), update=step+1, total_updates=context["updates"],
                elapsed_seconds=prior_seconds+time.perf_counter()-started,
                loss=float(loss.detach()), coordinate_loss=float(base.detach()),
                scalar_loss=terms["diagnostic"]["scalar_loss"], dense_loss=terms["diagnostic"]["dense_loss"],
                short_segment_penalty=terms["diagnostic"]["short_segment_penalty"],
                angle_coefficient=coefficient, geometry_coefficient=geometry_coefficient,
                gradient_norm=float(norm), gradient_clip=clip, gradient_was_clipped=bool(float(norm)>clip))
            line = json.dumps(event, allow_nan=False, sort_keys=True)
            # Each retry has its own attempt folder; a same-attempt resume appends
            # only newly executed updates. Timing never enters exact-fit history.
            with (output/"training.jsonl").open("a") as stream:
                stream.write(line+"\n")
                stream.flush()
            print(line, flush=True)
        if (step+1) % int(tc.get("checkpoint_every", 100)) == 0 or step+1 == context["updates"]:
            save("complete" if step+1 == context["updates"] else "interrupted")
    elapsed = save("complete")
    frozen_after = legacy._state_hash(model.encoder)
    if frozen_after != frozen_before:
        raise AssertionError("Frozen encoder changed during repair training")
    atomic_json(output/"history.json", history)
    np.save(output/"training-pairs.npy", context["pairs"], allow_pickle=False)
    if context["pairing_audit"] is not None:
        atomic_json(output/"pairing-audit.json", context["pairing_audit"])
    report = dict(status="complete", format=legacy.FORMAT, completed_utc=utc_now(), phase="readout",
        encoder="paired_jepa", representation_variant=context["variant"], objective=objective, seed=int(seed),
        recipe_id=f"R-repair-{context['variant']}-{objective}", checkpoint=str((output/"checkpoint.pt").resolve()),
        checkpoint_sha256=sha256(output/"checkpoint.pt"), signature_sha256=digest(signature),
        updates=context["updates"], training_rows=len(context["rows"]), training_pair_count=len(context["pairs"]),
        training_people=len({context["records"][i]["canonical_person_id"] for i in context["rows"]}),
        source_training_sha256=context["data_sha256"], upstream_sha256=context["upstream_sha256"],
        upstream_signature_sha256=context["upstream_signature_sha256"], repair=signature["repair"],
        endpoint_presentations=sum(len(row["endpoint_indices"]) for row in history),
        pair_presentations=sum(len(row["pair_table_indices"]) for row in history),
        initial_readout_sha256=initial_head, frozen_encoder_sha256=frozen_after, frozen_encoder_unchanged=True,
        sampling=context["sampling"], exposure_matching="same readout initialization, ordered endpoint draws, updates, optimizer and coordinate term",
        elapsed_seconds=elapsed, gpu_hours=elapsed/3600 if device.startswith("cuda") else 0.,
        allocation_accounting="phase compute only; scheduler allocation charged once by coordinator",
        device=device, precision=context["precision"], evidence_status=bundle.evidence_status,
        objective_detail=f"coordinate_MSE+{objective}+shared_short_segment_penalty",
        gradient_clipping=dict(updates=len(history), clipped_updates=sum(row["gradient_was_clipped"] for row in history)))
    development = np.flatnonzero([row["split"] == "development" for row in context["records"]])
    if len(development):
        observed = SimpleNamespace(inputs={k: IndexedArray(v, development) for k, v in bundle.inputs.items()})
        if config.get("fixture", False):
            path = output/"predictions.npz"
            predictions = legacy.predict(observed, output/"checkpoint.pt", config)
            np.savez_compressed(path, xy=predictions, indices=development)
        else:
            path = output/"predictions.npy"
            predict_with_deadline(observed, output/"checkpoint.pt", config, output=path)
            np.save(output/"prediction-indices.npy", development, allow_pickle=False)
            report.update(prediction_indices=str((output/"prediction-indices.npy").resolve()),
                          prediction_indices_sha256=sha256(output/"prediction-indices.npy"))
        report.update(predictions=str(path.resolve()), predictions_sha256=sha256(path), prediction_rows=len(development),
                      prediction_scope="development inputs only; confirmation never opened by trainer")
    atomic_json(output/"receipt.json", report)
    return report
