"""Phase-separated gait restoration with matched exposure and pixel measurements.

This reuses the tested body12 encoder, coordinate readout and JEPA objective
from synthetic-training-v2. Every final readout is independently initialized;
pretraining checkpoints are shared only under identical data/configuration.
"""
from __future__ import annotations

from collections import Counter, defaultdict
from contextlib import nullcontext
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import heapq
import json
import math
from pathlib import Path
import time

import numpy as np
import torch

from ..synthetic_training_v2.models import ModelConfig, RestorationModel, CoordinateReadout, LATENT_ARMS, INPUT_KEYS
from ..synthetic_training_v2.training import coordinate_loss, smoothnet_loss, _coordinate_patch_loss, _patch_valid
from ..temporal_gait.objectives import predictive_loss, vicreg_loss, feature_diagnostics
from .common import atomic_json, digest, sha256, utc_now
from .masking import sample_mask
from .measurements import knee_excursion, reference_support, measurement_loss

FORMAT = "gait-fidelity-training-v1"
ENCODERS = {"coordinate", "paired_jepa", "shuffled_jepa", "initialized", "direct", "static", "temporal_refiner"}
OBJECTIVES = {"base", "paired_change", "per_example_measurement", "repaired_change"}


class ResponseDeadlineReached(RuntimeError):
    """The response run saved an interrupted checkpoint before its deadline."""


def _records(bundle):
    return bundle.records.to_dict("records") if hasattr(bundle.records, "to_dict") else list(bundle.records)


def paired_indices(records):
    """Every movement endpoint is paired to its own baseline, metadata only.

    All loss regimes draw this same table. Baseline repetition across intervention
    levels is therefore explicit and identical, rather than extra exposure for
    the paired objective. Rows outside training are never candidates.
    """
    groups = defaultdict(list)
    for i, row in enumerate(records):
        if row.get("split") == "train":
            if row.get("held_intervention", False):
                raise ValueError("A held intervention cannot enter training")
            groups[row["pair_id"]].append(i)
    result = []
    for key in sorted(groups):
        rows = groups[key]
        baseline = [i for i in rows if records[i].get("movement_state") == "baseline"]
        if len(baseline) != 1:
            raise ValueError(f"Pair {key} needs exactly one training baseline")
        endpoints = [i for i in rows if i != baseline[0]]
        if not endpoints:
            raise ValueError(f"Pair {key} has no intervention/no-change endpoint")
        for i in sorted(endpoints, key=lambda j: (str(records[j].get("movement_state")), float(records[j].get("movement_magnitude", 0)))):
            for field in ("source_family_id", "physical_state", "camera_id", "naming", "observation", "extractor"):
                if records[i].get(field) != records[baseline[0]].get(field):
                    raise ValueError(f"Movement pair changes undeclared {field}")
            result.append([baseline[0], i])
    if not result:
        raise ValueError("No training movement pairs; preparation must precede training")
    return np.asarray(result, dtype=np.int64)


def _stratum(row, *, person=False):
    fields = ("physical_state", "camera_id", "naming", "observation", "extractor", "movement_state", "movement_magnitude")
    if person:
        fields = ("canonical_person_id",) + fields
    return tuple(str(row.get(k)) for k in fields)


def _derangement(families, rng):
    """A linear-memory cross-family bijection, preserving endpoint counts."""
    return _short_derangement(families, rng)


def _short_derangement(families, rng):
    """Cross-family cycles of two, plus one three-cycle for odd strata.

    Greedily pairing the two largest remaining family groups is feasible for a
    complete cross-family graph when no group exceeds half the stratum. For an
    odd stratum, removing one member from each of the three largest groups first
    leaves the same condition on the remaining even population. Random tie
    breaking and within-family order supply the reference-only matching trials.
    """
    groups = defaultdict(list)
    for i, family in enumerate(families):
        groups[str(family)].append(i)
    n = len(families)
    if n < 2 or max(map(len, groups.values())) > n // 2:
        raise ValueError("No endpoint-frequency-preserving cross-family re-pairing exists in a required stratum")
    for values in groups.values():
        rng.shuffle(values)
    permutation = np.full(n, -1, dtype=np.int64)

    heap = [(-len(values), float(rng.random()), key) for key, values in groups.items()]
    heapq.heapify(heap)

    def largest(count):
        return [heapq.heappop(heap)[2] for _ in range(min(count, len(heap)))]

    def replace(keys):
        for key in keys:
            if groups[key]:
                heapq.heappush(heap, (-len(groups[key]), float(rng.random()), key))

    if n % 2:
        keys = largest(3)
        if len(keys) != 3:
            raise ValueError("An odd re-pairing stratum needs three source families")
        cycle = [groups[key].pop() for key in keys]
        if rng.random() < .5:
            cycle.reverse()
        permutation[cycle] = np.roll(cycle, -1)
        replace(keys)
    while heap:
        keys = largest(2)
        if len(keys) != 2:
            raise AssertionError("Bounded-cycle pairing violated its feasibility condition")
        a, b = [groups[key].pop() for key in keys]
        permutation[a], permutation[b] = b, a
        replace(keys)
    return permutation


def paired_batch_cycles(pairs, repaired):
    """Small closed cycles keep each update's endpoint multiset identical."""
    if not np.array_equal(pairs[:, 0], repaired[:, 0]):
        raise ValueError("Re-pairing must retain the baseline endpoint roles")
    inverse = {int(value): i for i, value in enumerate(pairs[:, 1])}
    if len(inverse) != len(pairs):
        raise ValueError("Every intervention endpoint must identify one pair")
    try:
        permutation = np.asarray([inverse[int(value)] for value in repaired[:, 1]], dtype=np.int64)
    except KeyError as exc:
        raise ValueError("Re-pairing introduced a foreign endpoint") from exc
    if len(np.unique(permutation)) != len(permutation):
        raise ValueError("Re-pairing changed marginal endpoint frequency")
    seen, cycles = set(), []
    for start in range(len(permutation)):
        if start in seen:
            continue
        cycle, current = [], start
        while current not in seen:
            seen.add(current); cycle.append(current); current = int(permutation[current])
        if current != start or len(cycle) not in {2, 3}:
            raise ValueError("Shared batches require nonidentity two/three endpoint cycles")
        cycles.append(np.asarray(cycle, dtype=np.int64))
    return cycles


def draw_pair_batch(cycles, nominal_pairs, rng):
    """Pack whole cycles, exceeding the requested pair count by at most two."""
    chosen, size = [], 0
    while size < nominal_pairs:
        cycle = cycles[int(rng.integers(len(cycles)))]
        chosen.append(cycle)
        size += len(cycle)
    return np.concatenate(chosen)


def hierarchical_pair_groups(records, pairs):
    """Freeze equal-person, equal-motion, equal-window sampling from metadata."""
    groups = {}
    for index, (left, _) in enumerate(pairs):
        row = records[left]
        groups.setdefault(row['canonical_person_id'], {}).setdefault(row['motion_hash'], {}).setdefault(row['source_family_id'], []).append(index)
    return [[list(motion.values()) for _, motion in sorted(person.items())]
            for _, person in sorted(groups.items())]


def draw_hierarchical_pairs(groups, count, rng):
    result = []
    for _ in range(count):
        person = groups[int(rng.integers(len(groups)))]
        motion = person[int(rng.integers(len(person)))]
        family = motion[int(rng.integers(len(motion)))]
        result.append(family[int(rng.integers(len(family)))])
    return np.asarray(result, dtype=np.int64)


def shuffled_reference_indices(records, seed):
    """JEPA donors retain person, endpoint role and nuisance, change window."""
    groups = defaultdict(list)
    for i, row in enumerate(records):
        if row.get("split") == "train":
            groups[_stratum(row, person=True)].append(i)
    donors = np.arange(len(records))
    rng = np.random.default_rng(seed + 65537)
    for indices in groups.values():
        families = [records[i]["source_family_id"] for i in indices]
        permutation = _derangement(families, rng)
        donors[indices] = np.asarray(indices)[permutation]
    return donors


def repaired_pairs(bundle, pairs, seed, *, tolerance=.1, trials=64, min_segment_px=1., min_frames=16, min_fraction=.75):
    """Reference-only matching of re-paired labels, never stale differences.

    The right endpoints are permuted bijectively inside each role/movement/
    nuisance stratum and across source families. Choose the reference-only draw
    with minimum mean normalized Wasserstein mismatch; report every stratum and
    whether the worst mismatch meets the frozen tolerance. A failed tolerance
    remains a disclosed confound and does not discard the control's predictions.
    """
    if tolerance < 0 or trials < 1:
        raise ValueError("Re-pairing needs a nonnegative tolerance and positive trial count")
    records = _records(bundle)
    groups = defaultdict(list)
    for index, (_, right) in enumerate(pairs):
        groups[_stratum(records[right])].append(index)
    rng = np.random.default_rng(seed + 131071)
    target, valid = bundle.targets["xy"], bundle.targets["valid"]
    # Most admitted synthetic references have identical complete support. Cache
    # their individual endpoint values once; only pairs with different support
    # patterns need a new common-support quantile calculation in each trial.
    unique = np.unique(pairs)
    local_index = {int(row): i for i, row in enumerate(unique)}
    own_support = np.empty((len(unique), target.shape[1]), bool)
    own_value = np.empty(len(unique), float)
    for start in range(0, len(unique), 256):
        rows = unique[start:start + 256]
        support = reference_support(target[rows], valid[rows], min_segment_px=min_segment_px)
        own_support[start:start + len(rows)] = support
        own_value[start:start + len(rows)] = knee_excursion(target[rows], valid[rows], support=support,
            min_segment_px=min_segment_px, min_frames=min_frames, min_fraction=min_fraction)["asymmetry"]

    def changes(rows):
        local = np.asarray([[local_index[int(a)], local_index[int(b)]] for a, b in rows])
        same_support = (own_support[local[:, 0]] == own_support[local[:, 1]]).all(1)
        delta = own_value[local[:, 1]] - own_value[local[:, 0]]
        if same_support.all():
            return delta
        different = np.flatnonzero(~same_support)
        for start in range(0, len(different), 128):
            indices = different[start:start + 128]
            flat = rows[indices].reshape(-1)
            x, v = target[flat], valid[flat]
            support = reference_support(x, v, groups=np.arange(len(flat)).reshape(-1, 2), min_segment_px=min_segment_px)
            m = knee_excursion(x, v, support=support, min_segment_px=min_segment_px,
                              min_frames=min_frames, min_fraction=min_fraction)
            values = m["asymmetry"].reshape(-1, 2)
            delta[indices] = values[:, 1] - values[:, 0]
        return delta

    original = changes(pairs)
    if not np.isfinite(original).all():
        raise ValueError("Re-pairing distribution audit requires reference-eligible training pairs")
    best, best_score, best_metrics = None, math.inf, None
    for _ in range(trials):
        candidate = pairs.copy()
        for indices in groups.values():
            families = [records[pairs[i, 1]]["source_family_id"] for i in indices]
            order = _short_derangement(families, rng)
            candidate[indices, 1] = pairs[np.asarray(indices)[order], 1]
        delta = changes(candidate)
        if not np.isfinite(delta).all():
            continue
        metrics = [float(np.mean(np.abs(np.sort(delta[indices]) - np.sort(original[indices]))) / 180.) for indices in groups.values()]
        score = float(np.mean(metrics))
        if score < best_score:
            best, best_score, best_metrics = candidate, score, metrics
    if best is None:
        raise ValueError("No reference-eligible re-paired training table found")
    diagnostic = {"rule": "bijective_cross_source_family_within_endpoint_movement_nuisance_strata",
        "cycle_rule": "two_cycles_and_one_three_cycle_for_odd_strata",
        "selection": "minimum_mean_reference_only_Wasserstein_over_fixed_random_trials",
        "trials": trials, "tolerance": tolerance, "units": "Wasserstein_degrees_divided_by_180",
        "mean_mismatch": best_score, "worst_stratum_mismatch": max(best_metrics),
        "tolerance_passed": bool(max(best_metrics) <= tolerance), "pairs": len(pairs),
        "endpoint_frequency_preserved": bool(np.array_equal(np.sort(best[:, 1]), np.sort(pairs[:, 1]))),
        "mismatch_by_stratum": [{"stratum": list(key), "pairs": len(indices), "mismatch": mismatch}
            for (key, indices), mismatch in zip(groups.items(), best_metrics)],
        "claim_limit": "Pairing-specific attribution is unresolved if the declared distribution tolerance fails."}
    return best, diagnostic


def normalize_batch(inputs, hidden=None, *, patch_size=4):
    """Fixed isotropic normalization from retained observations only.

    For pretraining, hidden values are excluded before the origin/scale is
    computed. This strict context-only transform prevents aggregate leakage.
    Fewer than two context points or zero span use fixed (origin=0,scale=1)
    independent of hidden or reference values, with a retained failure count.
    """
    if set(inputs) != INPUT_KEYS:
        raise ValueError("Input allow-list rejects target/provenance fields")
    xy = np.asarray(inputs["xy"], dtype=np.float32)
    observed = np.asarray(inputs["observed"])
    if observed.dtype != np.bool_ or xy.ndim != 4 or xy.shape[-2:] != (12, 2) or observed.shape != xy.shape[:-1]:
        raise ValueError("Body12 coordinates and an explicit boolean observation mask are required")
    available = observed.copy()
    if hidden is not None:
        available &= ~np.repeat(np.asarray(hidden), patch_size, axis=1)
    origins, scales, fallback = [], [], 0
    for row, keep in zip(xy, available):
        points = row[keep]
        if len(points) >= 2:
            origin = np.median(points, axis=0)
            scale = float(np.linalg.norm(np.quantile(points, .95, axis=0) - np.quantile(points, .05, axis=0)))
        else:
            origin, scale = np.zeros(2), 0.
        if not np.isfinite(scale) or scale < 1e-6:
            origin, scale, fallback = np.zeros(2), 1., fallback + 1
        origins.append(origin)
        scales.append(scale)
    origin, scale = np.asarray(origins, np.float32), np.asarray(scales, np.float32)
    normalized = {k: np.asarray(v).copy() for k, v in inputs.items()}
    normalized["xy"] = (xy - origin[:, None, None]) / scale[:, None, None, None]
    normalized["confidence"] = np.where(np.isfinite(inputs["confidence"]), inputs["confidence"], 0).astype(np.float32)
    return normalized, origin, scale, fallback


def normalize_inputs(inputs):
    """The v2 normalization API, extended to any registered window length."""
    from ..synthetic_training_v2.data import Normalization
    normalized, origin, scale, _ = normalize_batch(inputs)
    return normalized, Normalization(origin, scale)


def _tensors(values, device):
    return {k: torch.as_tensor(v, dtype=torch.bool if k in {"observed", "valid"} else torch.float32, device=device)
            for k, v in values.items()}


def _take(values, rows):
    return {k: v[rows] for k, v in values.items()}


def _array_signature(values):
    result = {}
    for key, value in sorted(values.items()):
        h = hashlib.sha256()
        for start in range(0, len(value), 256):
            h.update(np.ascontiguousarray(value[start:start + 256]).tobytes())
        result[key] = dict(shape=list(value.shape), dtype=str(value.dtype), sha256=h.hexdigest())
    return result


def _state_hash(module):
    h = hashlib.sha256()
    for name, value in module.state_dict().items():
        h.update(name.encode()); h.update(value.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def _measurement_settings(config):
    m, t = config.get("measurement", {}), config.get("training", {})
    return dict(min_segment_px=float(m.get("min_segment_px", t.get("min_segment_px", 1.))),
                min_frames=int(m.get("min_frames", t.get("measurement_min_frames", 16))),
                min_fraction=float(m.get("min_coverage", t.get("measurement_min_fraction", .75))))


def _save(path, payload):
    tmp = Path(path).with_suffix(".tmp")
    torch.save(payload, tmp)
    tmp.replace(path)


def _phase_name(phase):
    return {"pretraining": "pretrain"}.get(phase, phase)


def train_phase(bundle, recipe: dict, phase: str, seed: int, config: dict, output: Path, upstream: Path | None = None):
    """Train exactly one declared phase and retain a resumable, hashed receipt."""
    phase = _phase_name(phase)
    encoder = recipe["encoder"]
    variant = recipe.get("representation_variant")
    policy = recipe.get("pretraining_mask", recipe.get("mask"))
    objective = recipe.get("readout_or_training_objective", "base")
    if phase not in {"pretrain", "readout", "end_to_end"} or encoder not in ENCODERS or objective not in OBJECTIVES:
        raise ValueError("Unknown gait-fidelity phase, encoder or objective")
    if variant is not None:
        from .response_objectives import VARIANTS
        if variant not in VARIANTS or encoder != VARIANTS[variant] or phase == "end_to_end" or policy != "graph_time":
            raise ValueError("Response representation, encoder, phase and graph_time policy must agree")
    if phase == "pretrain":
        # A shared representation has no downstream loss identity. It can feed
        # base/change/per-example readouts without a representative recipe name
        # falsely implying that its pretraining used those downstream labels.
        objective = "base"
    if (phase == "pretrain" and encoder not in {"coordinate", "paired_jepa", "shuffled_jepa"}) or (phase == "readout" and encoder in {"direct", "static", "temporal_refiner"}) or (phase == "end_to_end" and encoder not in {"direct", "static", "temporal_refiner"}):
        raise ValueError("Phase does not belong to the declared recipe")
    if phase == "pretrain" and not policy:
        raise ValueError("Pretraining requires an explicit query-mask policy")
    if upstream is not None and phase != "readout":
        raise ValueError("Only frozen readout loads an upstream phase")
    if phase == "readout" and encoder != "initialized" and upstream is None:
        raise ValueError("Frozen readout requires its completed matching pretraining checkpoint")
    if encoder == "initialized" and upstream is not None:
        raise ValueError("Initialized encoder control cannot consume pretraining")
    from .data import validate_bundle, IndexedArray
    if any(r['split'] == 'confirmation' for r in bundle.records):
        raise PermissionError('Confirmation tensors cannot enter a training process')
    validate_bundle(bundle)
    records = _records(bundle)
    pairs = paired_indices(records)
    rows = np.unique(pairs)
    tc = dict(config.get("training", {}))
    updates = int(tc.get({"pretrain": "pretraining_updates", "readout": "readout_updates", "end_to_end": "end_to_end_updates"}[phase], 4000 if phase == "end_to_end" else 2000))
    batch_size, lr = int(tc.get("batch_size", 16)), float(tc.get("learning_rate", 3e-4))
    if updates < 1 or batch_size < 2 or batch_size % 2 or not np.isfinite(lr) or lr <= 0:
        raise ValueError("Positive updates/lr and an even endpoint batch_size >=2 are required")
    for key in ("change_weight", "measurement_weight", "weight_decay", "vicreg_weight", "smoothnet_acceleration_weight"):
        if key in tc and (not np.isfinite(tc[key]) or tc[key] < 0):
            raise ValueError(f"{key} must be finite nonnegative")
    if tc.get("normalization", "context_only") != "context_only":
        raise ValueError("Registered implementation uses strict context-only pretraining normalization")
    device = str(config.get("device", "cpu"))
    if device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("Requested CUDA is unavailable; no silent CPU fallback")
    precision = tc.get("precision", "float32")
    if precision not in {"float32", "bfloat16"} or (precision == "bfloat16" and not device.startswith("cuda")):
        raise ValueError("Use float32, or bfloat16 on an allocated CUDA GPU")
    cfg = ModelConfig(**config["model"]).validate()
    if cfg.window_size != bundle.inputs["xy"].shape[1]:
        raise ValueError("Model query window differs from the prepared physical-time interval")
    measurement = _measurement_settings(config)
    # Every phase uses the same closed-cycle endpoint batches. Re-pairing changes
    # which endpoints form a supervised difference, never which inputs a model
    # sees at that update. Constructing the schedule from training references is
    # common to all arms and does not expose those references to the encoder.
    sampling = tc.get('sampling', 'matched_cycles')
    if sampling not in {'matched_cycles', 'person_motion'}:
        raise ValueError('Unknown training sampling rule')
    if sampling == 'person_motion':
        if objective == 'repaired_change':
            raise ValueError('repaired_change requires matched_cycles sampling to preserve marginal exposure')
        active_pairs, cycles, pairing_report = pairs, [], None
        hierarchy = hierarchical_pair_groups(records, pairs)
    else:
        re_paired, pairing_report = repaired_pairs(bundle, pairs, seed,
            tolerance=float(tc.get("repaired_tolerance", .1)), trials=int(tc.get("repaired_trials", 64)), **measurement)
        cycles = paired_batch_cycles(pairs, re_paired)
        active_pairs = re_paired if phase != "pretrain" and objective == "repaired_change" else pairs
        pairing_report["shared_endpoint_batches_all_phases"] = True
        pairing_report["used_for_measurement_pairing"] = phase != "pretrain" and objective == "repaired_change"
    donors = shuffled_reference_indices(records, seed) if phase == "pretrain" and encoder == "shuffled_jepa" else None
    code = {str(p.name): sha256(p) for p in [Path(__file__), Path(__file__).with_name("masking.py"), Path(__file__).with_name("measurements.py")]}
    from ..synthetic_training_v2 import models as old_models, training as old_training
    from ..temporal_gait import objectives as old_objectives
    code.update({f"inherited/{p.name}": sha256(p) for p in [Path(old_models.__file__), Path(old_training.__file__), Path(old_objectives.__file__)]})
    data_identity = digest(dict(inputs=_array_signature({k: IndexedArray(v, rows) for k, v in bundle.inputs.items()}),
        targets=_array_signature({k: IndexedArray(v, rows) for k, v in bundle.targets.items()}), records=[records[i] for i in rows]))
    signature = dict(format=FORMAT, phase=phase, encoder=encoder, policy=policy, objective=objective,
        seed=int(seed), model=asdict(cfg), training=tc, measurement=measurement, data_sha256=data_identity,
        evidence_status=bundle.evidence_status,
        sampling_hz=float(bundle.provenance.get('hz',25.)),
        code=code, normalization="context_only_fixed_isotropic_input_observations", device=device,
        pairs_sha256=hashlib.sha256(active_pairs.tobytes()).hexdigest(),
        batch_cycles_sha256=digest([cycle.tolist() for cycle in cycles]),
        upstream_sha256=sha256(upstream) if upstream is not None else None)
    signature["training"].pop("resume_from", None)
    response_identity = None
    if variant is not None:
        from .response_calibration import verified_response_identity
        response_identity = verified_response_identity(config, variant, data_identity)
        signature.update(representation_variant=variant, response=response_identity)
        code.update({name: sha256(Path(__file__).with_name(name))
                     for name in ("response_objectives.py", "response_calibration.py")})
    torch.manual_seed(seed)
    if device.startswith("cuda"):
        torch.cuda.manual_seed_all(seed)
        torch.cuda.reset_peak_memory_stats(device)
    arm = "smoothnet" if encoder == "temporal_refiner" else encoder
    model = RestorationModel(arm, cfg).to(device)
    if upstream is not None:
        pre = torch.load(upstream, map_location=device, weights_only=True)
        if pre.get("format") != FORMAT or pre.get("status") != "complete" or pre["signature"]["phase"] != "pretrain":
            raise ValueError("Upstream is not a complete Gait Fidelity pretraining checkpoint")
        for key in ("encoder", "policy", "seed", "model", "training", "data_sha256", "code", "normalization"):
            if pre["signature"].get(key) != signature[key]:
                raise ValueError(f"Upstream pretraining mismatch: {key}")
        for key in ("representation_variant", "response"):
            if pre["signature"].get(key) != signature.get(key):
                raise ValueError(f"Upstream pretraining mismatch: {key}")
        model.load_state_dict(pre["model"], strict=True)
    model.requires_grad_(False)
    if phase == "pretrain":
        model.encoder.requires_grad_(True)
        if arm in LATENT_ARMS:
            model.predictor.requires_grad_(True)
            model.projector.requires_grad_(True)
        else:
            model.readout.requires_grad_(True)
    elif phase == "readout":
        # A separate instance and optimizer for every final objective. Matching
        # initial readout weights across recipes makes the seed comparison fair.
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(seed + 100003)
            model.readout = CoordinateReadout(cfg).to(device)
        model.readout.requires_grad_(True)
    elif hasattr(model, "practical"):
        model.practical.requires_grad_(True)
    else:
        model.encoder.requires_grad_(True)
        model.readout.requires_grad_(True)
    model.train()
    if phase == "readout":
        model.encoder.eval()
    parameters = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(parameters, lr=lr, weight_decay=float(tc.get("weight_decay", .01)))
    generators = {"batch": np.random.default_rng(seed + 11), "mask": np.random.default_rng(seed + 23)}
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    resume = Path(config.get("training", {}).get("resume_from", output / "checkpoint.pt"))
    history, start_step, prior_seconds = [], 0, 0.
    frozen_before = _state_hash(model.encoder) if phase == "readout" else None
    if resume.is_file():
        retained = torch.load(resume, map_location=device, weights_only=True)
        if retained.get("signature") != signature:
            raise ValueError("Resume rejected: data, code, phase or configuration changed")
        model.load_state_dict(retained["model"], strict=True)
        optimizer.load_state_dict(retained["optimizer"])
        history, start_step, prior_seconds = retained["history"], retained["updates"], retained["elapsed_seconds"]
        for key, generator in generators.items():
            generator.bit_generator.state = retained["numpy_rng"][key]
        torch.set_rng_state(retained["torch_rng"].cpu())
        if device.startswith("cuda"):
            torch.cuda.set_rng_state_all([v.cpu() for v in retained["cuda_rng"]])
        if start_step > updates:
            raise ValueError("Resume exceeds the configured update budget")
    started = time.perf_counter()
    checkpoint_every = int(tc.get("checkpoint_every", 100))
    if checkpoint_every < 1:
        raise ValueError("checkpoint_every must be positive")

    def save(status):
        if device.startswith("cuda"):
            torch.cuda.synchronize(device)
        elapsed = prior_seconds + time.perf_counter() - started
        payload = dict(format=FORMAT, status=status, signature=signature, model=model.state_dict(),
            optimizer=optimizer.state_dict(), updates=len(history), history=history, elapsed_seconds=elapsed,
            torch_rng=torch.get_rng_state(), cuda_rng=torch.cuda.get_rng_state_all() if device.startswith("cuda") else [],
            numpy_rng={k: rng.bit_generator.state for k, rng in generators.items()})
        _save(output / "checkpoint.pt", payload)
        return elapsed

    for step in range(start_step, updates):
        if variant is not None and not config.get("fixture", False):
            deadline = response_identity.get("deadline_utc")
            if deadline is not None:
                remaining = (datetime.fromisoformat(deadline.replace("Z", "+00:00")) - datetime.now(timezone.utc)).total_seconds()
                if remaining <= response_identity["checkpoint_grace_seconds"]:
                    save("interrupted")
                    raise ResponseDeadlineReached(f"Response deadline reached; interrupted checkpoint saved at {output / 'checkpoint.pt'}")
        pair_draw = (draw_hierarchical_pairs(hierarchy, batch_size // 2, generators['batch']) if sampling == 'person_motion'
                     else draw_pair_batch(cycles, batch_size // 2, generators["batch"]))
        selected = active_pairs[pair_draw].reshape(-1)
        actual_batch = len(selected)
        expected_endpoints = np.sort(pairs[pair_draw].reshape(-1))
        if not np.array_equal(np.sort(selected), expected_endpoints):
            raise AssertionError("A control changed the per-update marginal endpoint exposure")
        raw = _take(bundle.inputs, selected)
        hidden, mask_receipt = None, None
        if phase == "pretrain":
            hidden, mask_receipt = sample_mask(raw["observed"], policy, rng=generators["mask"],
                fraction=float(tc.get("mask_fraction", .5)), patch_size=cfg.patch_size, return_receipt=True)
        normalized, origin, scale, fallback = normalize_batch(raw, hidden, patch_size=cfg.patch_size)
        inputs = _tensors(normalized, device)
        truth_np = bundle.targets["xy"][selected]
        valid = torch.as_tensor(bundle.targets["valid"][selected], dtype=torch.bool, device=device)
        target = torch.as_tensor((truth_np - origin[:, None, None]) / scale[:, None, None, None], dtype=torch.float32, device=device)
        hidden_tensor = None if hidden is None else torch.as_tensor(hidden, dtype=torch.bool, device=device)
        autocast = torch.autocast("cuda", dtype=torch.bfloat16) if precision == "bfloat16" else nullcontext()
        extra = {}
        with autocast:
            if phase == "pretrain":
                missing = (~inputs["observed"]).reshape(actual_batch, cfg.window_size // cfg.patch_size, cfg.patch_size, 12).any(2)
                queries = hidden_tensor | missing
            response_result = None
            if phase == "pretrain" and variant is not None:
                from .response_calibration import response_forward_losses
                response_result = response_forward_losses(model, dict(inputs=inputs, target=target, valid=valid,
                    queries=queries, hidden=hidden_tensor, scale=scale), tc, variant)
                loss = response_result["base_loss"] + response_identity["coefficient"] * response_result["auxiliary_loss"]
                supported, query_valid = response_result["supported"], response_result["query_valid"]
                extra.update(response_result["extra"])
                extra["response_coefficient"] = response_identity["coefficient"]
                if arm in LATENT_ARMS:
                    teacher_tokens = response_result["teacher_tokens"]
                else:
                    predicted = response_result["predicted"]
            elif phase == "pretrain" and arm in LATENT_ARMS:
                predicted_tokens = model.predictor(model.encoder(inputs, hidden_tensor))
                target_rows = selected if donors is None else donors[selected]
                donor_raw = _take(bundle.inputs, target_rows)
                # Each donor uses its own input/context-derived transform. Its
                # reference never determines normalization or student inputs.
                if donors is None:
                    donor_origin, donor_scale = origin, scale
                else:
                    _, donor_origin, donor_scale, _ = normalize_batch(donor_raw, hidden, patch_size=cfg.patch_size)
                teacher_valid = torch.as_tensor(bundle.targets["valid"][target_rows], dtype=torch.bool, device=device)
                teacher_xy = torch.as_tensor((bundle.targets["xy"][target_rows] - donor_origin[:, None, None]) / donor_scale[:, None, None, None], dtype=torch.float32, device=device)
                teacher_inputs = {"xy": torch.where(teacher_valid[..., None], teacher_xy, 0),
                    "observed": teacher_valid, "confidence": teacher_valid.float(),
                    "timestamps": torch.as_tensor(donor_raw["timestamps"], dtype=torch.float32, device=device)}
                with torch.no_grad():
                    teacher_tokens = model.teacher(teacher_inputs)
                query_valid = queries.flatten(1) & _patch_valid(teacher_valid, cfg)
                loss, _, supported = predictive_loss(predicted_tokens, teacher_tokens, query_valid,
                    objective="centered_ce_v1", center=model.center,
                    student_temperature=float(tc.get("student_temperature", .1)),
                    teacher_temperature=float(tc.get("teacher_temperature", .06)))
                extra["predictive_loss"] = float(loss.detach())
                weight = float(tc.get("vicreg_weight", .05))
                if weight and int(supported.sum()) >= 2:
                    views = []
                    for _ in range(2):
                        shift = (torch.rand(actual_batch, 1, 1, 2, device=device) * 2 - 1) * float(tc.get("translation_magnitude", .02))
                        translated = dict(inputs, xy=torch.where(inputs["observed"][..., None], inputs["xy"] + shift, inputs["xy"]))
                        views.append(model.projector(model.encoder(translated, hidden_tensor).mean(1))[supported])
                    regularizer = vicreg_loss(*views)
                    loss = loss + weight * regularizer
                    extra["vicreg_loss"] = float(regularizer.detach())
            else:
                predicted = model(inputs, hidden_tensor)
                if phase == "pretrain":
                    loss, _, supported, query_valid = _coordinate_patch_loss(predicted, target, valid, queries, cfg)
                elif arm == "smoothnet":
                    loss, _, supported, practical = smoothnet_loss(predicted, target, valid, inputs["timestamps"],
                        float(tc.get("smoothnet_acceleration_weight", .1)))
                    extra["temporal_refiner"] = practical
                else:
                    loss, _, supported = coordinate_loss(predicted, target, valid)
                extra["coordinate_or_practical_loss"] = float(loss.detach())
        if phase != "pretrain" and objective != "base":
            # Measure after the exact inverse isotropic transform in pixels;
            # reference-derived scales never enter the observation model.
            predicted_px = predicted.float() * torch.as_tensor(scale[:, None, None, None], device=device) + torch.as_tensor(origin[:, None, None], device=device)
            target_px = torch.as_tensor(truth_np, dtype=torch.float32, device=device)
            term, diagnostic = measurement_loss(predicted_px.reshape(-1, 2, cfg.window_size, 12, 2),
                target_px.reshape(-1, 2, cfg.window_size, 12, 2), valid.reshape(-1, 2, cfg.window_size, 12),
                objective=objective, **measurement)
            if diagnostic["supported_pairs"] == 0:
                raise ValueError("A measurement-supervised batch has no reference-eligible pairs; fix reference admission")
            coefficient = float(tc.get("measurement_weight" if objective == "per_example_measurement" else "change_weight", 1.))
            loss = loss + coefficient * term
            extra["measurement"] = diagnostic
            extra["measurement_coefficient"] = coefficient
        if not supported.any() or not torch.isfinite(loss):
            raise FloatingPointError("Unsupported or nonfinite training loss; no zero-error completion")
        warmup = max(1, round(float(tc.get("warmup_fraction", .05)) * updates))
        progress = (step - warmup) / max(updates - warmup, 1)
        scheduled_lr = lr * ((step + 1) / warmup if step < warmup else .5 * (1 + math.cos(math.pi * progress)))
        for group in optimizer.param_groups:
            group["lr"] = scheduled_lr
        optimizer.zero_grad(set_to_none=True)
        if response_result is not None and (step == 0 or step + 1 == updates or (step + 1) % int(tc.get("log_every", 100)) == 0):
            from .response_calibration import gradient_statistics
            extra["response_gradients"] = {
                "base": gradient_statistics(response_result["base_loss"], parameters),
                "auxiliary_unweighted": gradient_statistics(response_result["auxiliary_loss"], parameters),
                "auxiliary_coefficient": response_identity["coefficient"]}
        loss.backward()
        if hasattr(model, "teacher") and any(p.grad is not None for p in model.teacher.parameters()):
            raise AssertionError("Privileged teacher received gradients")
        if phase == "readout" and any(p.grad is not None for p in model.encoder.parameters()):
            raise AssertionError("Frozen encoder received readout gradients")
        norm = torch.nn.utils.clip_grad_norm_(parameters, float(tc.get("gradient_clip", 1.)), error_if_nonfinite=True)
        if variant is not None:
            extra.update(gradient_clip=float(tc.get("gradient_clip", 1.)),
                         gradient_was_clipped=bool(float(norm) > float(tc.get("gradient_clip", 1.))))
        optimizer.step()
        if phase == "pretrain" and arm in LATENT_ARMS:
            momentum = .999 - (.999 - .99) * .5 * (1 + math.cos(math.pi * step / max(1, updates - 1)))
            model.update_teacher(momentum)
            with torch.no_grad():
                counts = query_valid.sum(1)
                means = (teacher_tokens.float() * query_valid[..., None]).sum(1) / counts[:, None].clamp_min(1)
                model.center.mul_(.9).add_(means[supported].mean(0), alpha=.1)
            extra["ema"] = momentum
        entry = dict(update=step + 1, loss=float(loss.detach()), learning_rate=scheduled_lr,
            gradient_norm=float(norm), supported_examples=int(supported.sum()),
            endpoint_indices=selected.tolist(), pair_table_indices=pair_draw.tolist(),
            endpoint_multiset_sha256=hashlib.sha256(expected_endpoints.tobytes()).hexdigest(),
            target_indices=(selected if donors is None else donors[selected]).tolist(),
            normalization_fallbacks=fallback, **extra)
        if mask_receipt is not None:
            entry["mask"] = mask_receipt
        if (step + 1 == updates or (step + 1) % int(tc.get("log_every", 100)) == 0) and hasattr(model, "encoder"):
            with torch.no_grad():
                entry["feature_diagnostics"] = feature_diagnostics(model.encoder(inputs).flatten(0, 1)[:1024])
        history.append(entry)
        if (step + 1) % checkpoint_every == 0 or step + 1 == updates:
            save("complete" if step + 1 == updates else "interrupted")
    elapsed = save("complete")
    frozen_after = _state_hash(model.encoder) if phase == "readout" else None
    if frozen_before != frozen_after:
        raise AssertionError("Encoder changed during frozen-readout training")
    if phase == "readout" and any(p.grad is not None for p in model.encoder.parameters()):
        raise AssertionError("Frozen encoder gradients remain after fitting")
    atomic_json(output / "history.json", history)
    if pairing_report is not None:
        atomic_json(output / "pairing-audit.json", pairing_report)
    np.save(output / "training-pairs.npy", active_pairs, allow_pickle=False)
    atomic_json(output / "batch-pair-cycles.json", [cycle.tolist() for cycle in cycles])
    seen = {index for entry in history for index in entry['endpoint_indices']}
    coverage = {}
    for field in ('canonical_person_id', 'motion_hash', 'source_family_id'):
        total = {records[i][field] for i in rows}
        counts = Counter(records[i][field] for entry in history for i in entry['endpoint_indices'])
        coverage[field] = dict(available=len(total), observed=len(counts), fraction=len(counts)/len(total),
                              endpoint_presentations=dict(sorted(counts.items())),
                              unobserved=sorted(total-set(counts)))
    coverage['endpoint_rows'] = dict(available=len(rows), observed=len(seen), fraction=len(seen)/len(rows))
    atomic_json(output / 'sampling-coverage.json', coverage)
    report = dict(status="complete", format=FORMAT, completed_utc=utc_now(), phase=phase,
        encoder=encoder, recipe_id=recipe.get("recipe_id"), seed=int(seed), objective=objective,
        checkpoint=str((output / "checkpoint.pt").resolve()), checkpoint_sha256=sha256(output / "checkpoint.pt"),
        updates=updates, endpoint_presentations=sum(len(row["endpoint_indices"]) for row in history),
        pair_presentations=sum(len(row["pair_table_indices"]) for row in history),
        nominal_endpoint_batch_size=batch_size,
        realized_endpoint_batch_size_range=[min(len(row["endpoint_indices"]) for row in history),max(len(row["endpoint_indices"]) for row in history)],
        exposure_matching=("identical draws; uniform people, raw motions, source windows, then paired cells" if sampling == 'person_motion'
                           else "identical endpoint multisets at each shared update, via closed two/three pair cycles"),
        sampling=sampling,
        training_rows=len(rows), training_people=len({records[i]["canonical_person_id"] for i in rows}),
        training_pair_count=len(active_pairs), signature_sha256=digest(signature),
        sampling_coverage={key:{k:v for k,v in value.items() if k not in {'endpoint_presentations','unobserved'}} for key,value in coverage.items()},
        sampling_coverage_file=str((output/'sampling-coverage.json').resolve()),
        endpoint_presentations_per_available_row=sum(len(row['endpoint_indices']) for row in history)/len(rows),
        source_training_sha256=data_identity,
        elapsed_seconds=elapsed, gpu_hours=elapsed / 3600 if device.startswith("cuda") else 0.,
        allocation_accounting="phase compute only; scheduler allocation charged once by coordinator",
        frozen_encoder_sha256=frozen_after, frozen_encoder_unchanged=phase != "readout" or frozen_before == frozen_after,
        pairing_audit=pairing_report, evidence_status=getattr(bundle, "evidence_status", "unknown"),
        parameters_trainable=sum(p.numel() for p in parameters), device=device,
        gpu_name=torch.cuda.get_device_name(device) if device.startswith("cuda") else None,
        precision=precision, normalization="input-only; context-only before artificial masking",
        objective_detail="centered_ce_v1+VICReg" if phase == "pretrain" and arm in LATENT_ARMS else
                         "SmoothNet-style L1+reference acceleration adaptation" if arm == "smoothnet" else "coordinate_MSE" + ("+" + objective if objective != "base" else ""))
    if variant is not None:
        report.update(representation_variant=variant, response=response_identity)
        report["gradient_clipping"] = dict(updates=len(history), clipped_updates=sum(row["gradient_was_clipped"] for row in history))
        if phase == "pretrain":
            report["objective_detail"] += "+" + variant
            support_summary = {key: {} for key in ("canonical_person_id", "physical_state", "camera_id", "naming",
                                                   "observation", "extractor", "movement_state", "movement_magnitude")}
            for entry in history:
                for right, tokens in zip(entry["endpoint_indices"][1::2], entry["response_auxiliary"]["tokens_per_pair"]):
                    for field, values in support_summary.items():
                        row = values.setdefault(str(records[right].get(field)), dict(pairs=0, supported_pairs=0, supported_tokens=0))
                        row["pairs"] += 1
                        row["supported_pairs"] += int(tokens > 0)
                        row["supported_tokens"] += tokens
            atomic_json(output / "response-support.json", support_summary)
            report["response_support"] = support_summary
    if phase != "pretrain":
        from types import SimpleNamespace
        development = np.flatnonzero([row["split"] == "development" for row in records])
        if len(development):
            observed_bundle = SimpleNamespace(inputs={k: IndexedArray(v, development) for k, v in bundle.inputs.items()})
            if not config.get('fixture', False):
                path = output / 'predictions.npy'
                predict(observed_bundle, output / 'checkpoint.pt', config, output=path)
                np.save(output / 'prediction-indices.npy', development, allow_pickle=False)
                report['prediction_indices'] = str((output / 'prediction-indices.npy').resolve())
                report['prediction_indices_sha256'] = sha256(output / 'prediction-indices.npy')
            else:
                path = output / 'predictions.npz'
                predictions = predict(observed_bundle, output / "checkpoint.pt", config)
                np.savez_compressed(path, xy=predictions, indices=development)
            report["predictions"] = str(path.resolve())
            report["predictions_sha256"] = sha256(path)
            report["prediction_rows"] = len(development)
            report["prediction_scope"] = "development only; confirmation is never opened by the trainer"
    atomic_json(output / "receipt.json", report)
    return report


def load_model(checkpoint, *, device="cpu", require_final=True):
    payload = torch.load(checkpoint, map_location=device, weights_only=True)
    if payload.get("format") != FORMAT or payload.get("status") != "complete":
        raise ValueError("Inference requires a completed Gait Fidelity checkpoint")
    signature = payload["signature"]
    if require_final and signature["phase"] == "pretrain":
        raise ValueError("A pretraining checkpoint has no fitted deployment readout")
    with torch.random.fork_rng(devices=[]):
        arm = "smoothnet" if signature["encoder"] == "temporal_refiner" else signature["encoder"]
        model = RestorationModel(arm, signature["model"])
    model.load_state_dict(payload["model"], strict=True)
    return model.to(device).eval().requires_grad_(False), payload


@torch.inference_mode()
def predict(bundle, checkpoint: Path, config=None, *, output=None):
    """Return pixel coordinates, reading only observation arrays from the bundle."""
    config = config or {}
    device = str(config.get("device", "cpu"))
    model, payload = load_model(checkpoint, device=device)
    if set(bundle.inputs) != INPUT_KEYS:
        raise ValueError("Inference input allow-list violation")
    batch_size = int(config.get("training", {}).get("batch_size", 32))
    if batch_size < 1:
        raise ValueError("Prediction batch size must be positive")
    predictions = (np.lib.format.open_memmap(output, mode='w+', dtype=np.float32, shape=bundle.inputs['xy'].shape)
                   if output is not None else np.empty(bundle.inputs['xy'].shape, np.float32))
    for start in range(0, len(bundle.inputs["xy"]), batch_size):
        raw = _take(bundle.inputs, slice(start, start + batch_size))
        normalized, origin, scale, _ = normalize_batch(raw)
        value = model(_tensors(normalized, device)).float().cpu().numpy()
        predictions[start:start + len(value)] = value * scale[:, None, None, None] + origin[:, None, None]
    if isinstance(predictions, np.memmap):
        predictions.flush()
    return predictions
