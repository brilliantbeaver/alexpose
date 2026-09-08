"""Paired masking experiments with explicit budgets and a dense target path.

The original model and completed experiments are imported without modification.
Default settings are synthetic. A real run requires explicit confirmation.
"""
from __future__ import annotations

import copy
from dataclasses import asdict, replace
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import tempfile
import time
from typing import Mapping

import numpy as np
import pandas as pd
import torch
from torch.nn import functional as F

from laterality.config import SUITE_ROOT, canonical_json_digest
from laterality.geometry import FULL_MIRROR_PAIRS
from laterality.model import SJEPAGait, VICRegProjector, authorized_pool, geometric_view, vicreg_loss
from laterality.training import source_balanced_epoch_batches
from laterality_extensions.masked_learning import (
    GAIT_JOINTS, LearningDataset, LearningSettings, load_learning_dataset,
    masking_implementation_digest, resolve_learning_device,
)
from laterality_extensions.comparative_masks import MaskBudget, MaskPolicy, coverage_summary, eligible_landmarks, sample_mask

SCHEMA = "comparative_masking/v1"
STRUCTURED = {"whole_trajectory", "connected_region", "temporal_gap"}


def dense_prediction(model, view, target, valid_patch, target_mask):
    """Keep [clip, token, feature] identities until the loss selects targets."""
    if target_mask.dtype != torch.bool or target_mask.shape != valid_patch.shape:
        raise ValueError("Target mask must match boolean token validity")
    flat_mask = target_mask.flatten(1)
    flat_valid = valid_patch.flatten(1)
    if (target_mask & ~valid_patch).any() or (flat_mask.sum(1) < 1).any():
        raise ValueError("Each clip needs observed prediction targets")
    if ((flat_valid & ~flat_mask).sum(1) < 1).any():
        raise ValueError("Each clip needs visible context")
    context = model.view_encoder(view, valid_patch, hide_mask=target_mask)
    p = model.predictor
    tokens = p.encoder_to_predictor(context)
    tokens = torch.where(flat_mask[..., None], p.mask_token.expand_as(tokens), tokens)
    position = (p.time_pos[:, None] + p.joint_pos[None]).reshape(1, -1, tokens.shape[-1])
    predicted = p.output(p.norm(p.blocks(tokens + position, src_key_padding_mask=~flat_valid)))
    with torch.no_grad():
        targets = model.target_encoder(target, valid_patch)
    return predicted, targets


def per_clip_prediction_loss(predicted, targets, mask, center):
    """Average hidden targets within each clip, giving clips equal loss weight."""
    hidden = mask.flatten(1)
    if predicted.shape != targets.shape or predicted.shape[:2] != hidden.shape:
        raise ValueError("Dense features and mask dimensions disagree")
    probability = F.softmax((targets - center) / 0.06, dim=-1).detach()
    token_loss = -(probability * F.log_softmax(predicted / 0.10, dim=-1)).sum(-1)
    counts = hidden.sum(1)
    if (counts < 1).any():
        raise ValueError("Cannot average an empty target set")
    return ((token_loss * hidden).sum(1) / counts).mean()


def _digest_array(values):
    x = np.ascontiguousarray(values)
    return hashlib.sha256(str(x.dtype).encode() + str(x.shape).encode() + x.tobytes()).hexdigest()


def state_digest(model):
    h = hashlib.sha256()
    for name, value in sorted(model.state_dict().items()):
        h.update(name.encode()); h.update(value.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def implementation_digest():
    h = hashlib.sha256(masking_implementation_digest().encode())
    for name in ("comparative_masks.py", "comparative_training.py", "comparative_evaluation.py"):
        path = Path(__file__).with_name(name)
        if not path.is_file():
            raise FileNotFoundError(path)
        h.update(name.encode()); h.update(path.read_bytes())
    return h.hexdigest()


def _new_model(settings, frames, device):
    # Forking makes initialization independent of the caller's random draws.
    with torch.random.fork_rng():
        torch.manual_seed(settings.seed)
        model = SJEPAGait(frames=frames, joints=33, coordinate_dim=3,
            segment_length=settings.segment_length, embed_dim=settings.embed_dim,
            encoder_depth=settings.encoder_depth, predictor_depth=settings.predictor_depth,
            heads=settings.heads)
        projector = VICRegProjector(settings.embed_dim)
    return model.to(device), projector.to(device)


def source_schedule(dataset, settings):
    table = pd.DataFrame({"video_id": dataset.source_ids.astype(str)})
    rng = np.random.default_rng(np.random.SeedSequence([settings.seed, settings.fold, 101]))
    schedule = []
    while len(schedule) < settings.steps:
        for rows, _ in source_balanced_epoch_batches(table, list(dataset.train_sources),
                batch_size=settings.batch_size,
                updates_per_epoch=math.ceil(len(dataset.train_sources) / settings.batch_size), rng=rng):
            schedule.append(rows)
            if len(schedule) == settings.steps:
                break
    return np.stack(schedule)


def _policy_rng(settings, name, step, row):
    code = int.from_bytes(hashlib.sha256(name.encode()).digest()[:4], "little")
    return np.random.default_rng(np.random.SeedSequence([settings.seed, settings.fold, 102, code, step, row]))


def masks_for_batch(dataset, rows, settings, conditions, *, budgets=None, matched_to=None, step=0):
    """Construct paired masks before training; infeasibility never drops a clip."""
    budgets = dict(budgets or {})
    for name, budget in budgets.items():
        if name not in conditions:
            raise ValueError("A budget names an unknown condition")
        budget.validate(conditions[name])
    valid = dataset.valid[rows].reshape(len(rows), -1, settings.segment_length, 33).all(2)
    masks = {name: [] for name in conditions}
    coverage = {name: [] for name in conditions}
    if matched_to is None:
        if any(p.name in STRUCTURED for p in conditions.values()):
            raise ValueError("A structured comparison must name its matched_to condition")
        available = [int(valid[..., list(eligible_landmarks(p))].sum((1, 2)).min())
                     for p in conditions.values()]
        count = min(max(1, int(valid[..., GAIT_JOINTS].sum((1, 2)).min() * settings.mask_fraction)),
                    *available, int(valid.sum((1, 2)).min()) - 1)
        explicit = {b.hidden_count for b in budgets.values() if b.hidden_count is not None}
        if len(explicit) > 1:
            raise ValueError("Scattered comparisons must request one shared hidden count")
        if explicit:
            count = next(iter(explicit))
        if count < 1:
            raise ValueError("No shared positive target count is feasible")
    elif matched_to not in conditions or matched_to not in budgets:
        raise ValueError("matched_to requires a named condition and its declared budget")
    for batch_row, dataset_row in enumerate(rows):
        counts = count if matched_to is None else None
        order = list(conditions) if matched_to is None else [matched_to, *[n for n in conditions if n != matched_to]]
        for name in order:
            if name == matched_to:
                budget = budgets[name]
            else:
                if conditions[name].name in STRUCTURED:
                    raise ValueError("Compare one intact structure with scattered references per experiment")
                if name in budgets and budgets[name].hidden_count != counts:
                    raise ValueError("An explicit reference budget differs from the paired realized count")
                budget = MaskBudget(hidden_count=int(counts))
            result = sample_mask(dataset.xyz[dataset_row], valid[batch_row], conditions[name], budget,
                _policy_rng(settings, name, step, batch_row), segment_length=settings.segment_length,
                observation_valid=dataset.valid[dataset_row])
            if name == matched_to:
                counts = int(result.mask.sum())
            masks[name].append(result.mask)
            coverage[name].append(result.coverage)
    stacked = {name: np.stack(values) for name, values in masks.items()}
    counts = [m.sum((1, 2)) for m in stacked.values()]
    if any(not np.array_equal(c, counts[0]) for c in counts[1:]):
        raise AssertionError("Paired policies hid different counts in a clip")
    return stacked, coverage


def default_conditions():
    return {"gait_targets": MaskPolicy("gait"), "all_landmark_targets": MaskPolicy("uniform")}


def anatomical_conditions():
    return {**default_conditions(), **{f"random_set_{seed}": MaskPolicy("random_subset", subset_seed=seed)
        for seed in (31, 47, 59)}, "soft_gait_preference": MaskPolicy("soft_gait")}


def comparison_identity(dataset, settings, conditions, budgets, matched_to, checkpoint_steps):
    return {"schema": SCHEMA, "settings": asdict(settings), "frames": int(dataset.xyz.shape[1]),
        "conditions": {n: asdict(p) for n, p in conditions.items()},
        "budgets": {n: asdict(b) for n, b in (budgets or {}).items()}, "matched_to": matched_to,
        "checkpoint_steps": list(checkpoint_steps), "synthetic": dataset.synthetic,
        "train_sources": list(dataset.train_sources), "test_sources": list(dataset.test_sources),
        "sequence_ids": dataset.sequence_ids.astype(str).tolist(),
        "source_ids": dataset.source_ids.astype(str).tolist(),
        "data": {n: _digest_array(getattr(dataset, n)) for n in ("xyz", "valid", "targets")},
        "cohort_digest": dataset.cohort_digest, "split_digest": dataset.split_digest,
        "implementation": implementation_digest(),
        "runtime": {"python": platform.python_version(), "torch": str(torch.__version__), "numpy": np.__version__}}


def train_comparison(dataset: LearningDataset, settings: LearningSettings,
        conditions: Mapping[str, MaskPolicy] | None = None, *, budgets=None, matched_to=None,
        output_dir=None, checkpoint_steps=(), reuse=True, progress_callback=None):
    """Fit one paired comparison, optionally retaining completed compatible runs.

    Intermediate checkpoints are prespecified; interrupted runs are not resumed.
    All policies must be feasible on every scheduled training example.
    """
    settings.validate(); dataset.validate(settings.segment_length)
    if settings.fold != dataset.fold:
        raise ValueError("Dataset fold and settings fold differ")
    if settings.symmetry_weight:
        raise ValueError("Symmetry loss is a separate comparison")
    if not dataset.synthetic and not settings.confirm_real_run:
        raise PermissionError("Real training requires confirm_real_run=True")
    conditions = dict(conditions or default_conditions())
    if len(conditions) < 2 or any(not name or not name.replace("_", "").isalnum() for name in conditions):
        raise ValueError("Supply at least two conditions with simple distinct names")
    if any(n not in conditions for n in (budgets or {})):
        raise ValueError("A budget names an unknown condition")
    checkpoint_steps = tuple(sorted(set([*checkpoint_steps, settings.steps])))
    if any(not isinstance(s, int) or s < 1 or s > settings.steps for s in checkpoint_steps):
        raise ValueError("Checkpoint steps must be within the declared training exposure")
    device = resolve_learning_device(settings.device)
    settings = replace(settings, device=str(device))
    identity = comparison_identity(dataset, settings, conditions, budgets, matched_to, checkpoint_steps)
    key = canonical_json_digest(identity)
    destination = Path(output_dir) / key if output_dir is not None else None
    if destination is not None and destination.exists():
        if not reuse:
            raise FileExistsError(destination)
        return load_comparison(destination, identity)
    schedule = source_schedule(dataset, settings)
    if set(dataset.source_ids[schedule.ravel()].astype(str)) & set(dataset.test_sources):
        raise AssertionError("Test videos reached training")
    # Preflight every scheduled mask. This also records identical budgets before
    # any arm spends compute; mask draws cannot alter source or view schedules.
    prepared_masks, coverage = [], []
    for step, rows in enumerate(schedule):
        m, c = masks_for_batch(dataset, rows, settings, conditions, budgets=budgets, matched_to=matched_to, step=step)
        prepared_masks.append(m); coverage.append(c)
    reflection_rng = np.random.default_rng(np.random.SeedSequence([settings.seed, settings.fold, 103]))
    reflections = reflection_rng.random(schedule.shape) < settings.reflection_probability
    runs = {}
    from laterality.model import anatomical_reflect_tensor, valid_patches
    for name, policy in conditions.items():
        model, projector = _new_model(settings, dataset.xyz.shape[1], device)
        initial_model = copy.deepcopy(model).eval()
        initial_digest = state_digest(model)
        initial_projector_digest = state_digest(projector)
        trainable = [*model.view_encoder.parameters(), *model.predictor.parameters(), *projector.parameters()]
        optimizer = torch.optim.AdamW(trainable, lr=settings.learning_rate,
            weight_decay=settings.weight_decay, betas=(0.9, 0.95))
        history, count_history, actual_coverage, snapshots = [], [], [], {}
        started = time.monotonic()
        model.train(); projector.train()
        view_hash = hashlib.sha256()
        for step, rows in enumerate(schedule):
            valid = torch.as_tensor(dataset.valid[rows], dtype=torch.bool, device=device)
            xyz = torch.as_tensor(np.where(dataset.valid[rows, ..., None], dataset.xyz[rows], 0), dtype=torch.float32, device=device)
            mask = torch.as_tensor(prepared_masks[step][name], dtype=torch.bool, device=device)
            reflected = torch.as_tensor(reflections[step], dtype=torch.bool, device=device)
            if reflected.any():
                xyz, valid = anatomical_reflect_tensor(xyz, valid, reflected)
                changed = mask[reflected].clone()
                for left, right in FULL_MIRROR_PAIRS:
                    changed[:, :, [left, right]] = changed[:, :, [right, left]]
                mask = mask.clone(); mask[reflected] = changed
            patches = valid_patches(valid, settings.segment_length)
            step_coverage = []
            mirror = np.arange(33)
            for left, right in FULL_MIRROR_PAIRS:
                mirror[[left, right]] = mirror[[right, left]]
            for index, (hidden, observed) in enumerate(zip(mask.cpu().numpy(), patches.cpu().numpy())):
                details = {**coverage[step][name][index],
                    **coverage_summary(hidden, observed, segment_length=settings.segment_length),
                    "reflected": bool(reflections[step, index]), "coordinate_frame": "training view after anatomical reflection"}
                if reflections[step, index]:
                    for key in ("selected_landmarks", "eligible_landmarks"):
                        if key in details:
                            details[key] = sorted(map(int, mirror[details[key]]))
                step_coverage.append(details)
            actual_coverage.append(step_coverage)
            cuda_devices = [device.index or 0] if device.type == "cuda" else []
            with torch.random.fork_rng(devices=cuda_devices):
                torch.manual_seed(settings.seed + 100_003 * (settings.fold + 1) + step)
                both = geometric_view(torch.cat([xyz, xyz]), torch.cat([valid, valid]), max_degrees=8, translate=.03)
            view_hash.update(both.detach().cpu().numpy().tobytes())
            a, b = both.chunk(2)
            predicted, target = dense_prediction(model, a, xyz, patches, mask)
            prediction_loss = per_clip_prediction_loss(predicted, target, mask, model.target_center)
            both_valid = torch.cat([patches, patches])
            tokens = model.view_encoder(both, both_valid).reshape(2 * len(rows), -1, 33, settings.embed_dim)
            projected = projector(authorized_pool(tokens, both_valid, GAIT_JOINTS))
            regularizer = vicreg_loss(*projected.chunk(2))
            loss = prediction_loss + settings.vicreg_weight * regularizer
            if not torch.isfinite(loss):
                raise FloatingPointError("Non-finite training loss")
            optimizer.zero_grad(set_to_none=True); loss.backward()
            if any(p.grad is not None for p in model.target_encoder.parameters()):
                raise AssertionError("Teacher received gradients")
            torch.nn.utils.clip_grad_norm_(trainable, 1.0); optimizer.step()
            model.update_target(settings.ema_momentum)
            counts = mask.flatten(1).sum(1)
            with torch.no_grad():
                # Equal clip weights also govern the moving target center.
                mean = ((target * mask.flatten(1)[..., None]).sum(1) / counts[:, None]).mean(0)
                model.target_center.mul_(.9).add_(mean, alpha=.1)
            count_history.append(counts.cpu().tolist())
            history.append({"step": step + 1, "loss": float(loss.detach()),
                "masked_prediction_loss": float(prediction_loss.detach()),
                "variance_regularizer": float(regularizer.detach()),
                "hidden_count_min": int(counts.min()), "hidden_count_max": int(counts.max())})
            if step + 1 in checkpoint_steps:
                snapshots[step + 1] = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            if progress_callback is not None:
                progress_callback({"condition": name, "step": step + 1, "total_steps": settings.steps,
                    "condition_count": len(conditions), "loss": history[-1]["loss"]})
        runs[name] = {"model": model.eval(), "initial_model": initial_model, "projector": projector.eval(),
            "settings": asdict(settings), "policy": asdict(policy), "history": pd.DataFrame(history),
            "initial_state_digest": initial_digest, "initial_projector_digest": initial_projector_digest,
            "source_draw_digest": _digest_array(schedule),
            "view_digest": view_hash.hexdigest(), "hidden_token_counts": count_history,
            "coverage": actual_coverage, "checkpoints": snapshots,
            "elapsed_training_seconds": time.monotonic() - started}
    pairing = {"same_initialization": len({r["initial_state_digest"] for r in runs.values()}) == 1,
        "same_projector_initialization": len({r["initial_projector_digest"] for r in runs.values()}) == 1,
        "same_source_draws": len({r["source_draw_digest"] for r in runs.values()}) == 1,
        "same_geometric_views": len({r["view_digest"] for r in runs.values()}) == 1,
        "same_hidden_counts": len({json.dumps(r["hidden_token_counts"]) for r in runs.values()}) == 1}
    if not all(pairing.values()):
        raise AssertionError(f"Comparison controls failed: {pairing}")
    result = {"runs": runs, "pairing": pairing, "identity": identity,
        "source_schedule": schedule, "reflection_schedule": reflections,
        "synthetic": dataset.synthetic, "reused": False}
    if destination is not None:
        save_comparison(result, destination)
    return result


def save_comparison(result, destination):
    """Publish a complete result directory atomically; never overwrite a run."""
    destination = Path(destination)
    if destination.exists():
        raise FileExistsError(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=".incomplete_", dir=destination.parent))
    manifest = {"schema": SCHEMA, "complete": True, "identity": result["identity"],
        "pairing": result["pairing"], "runs": {}, "files": {}}
    for name, run in result["runs"].items():
        torch.save({"model": run["model"].state_dict(), "initial_model": run["initial_model"].state_dict(),
            "projector": run["projector"].state_dict(), "checkpoints": run["checkpoints"]}, staging / f"{name}.pt")
        run["history"].to_csv(staging / f"{name}_training.csv", index=False)
        manifest["runs"][name] = {k: run[k] for k in ("settings", "policy", "initial_state_digest", "initial_projector_digest",
            "source_draw_digest", "view_digest", "hidden_token_counts", "coverage", "elapsed_training_seconds")}
    np.save(staging / "source_schedule.npy", result["source_schedule"], allow_pickle=False)
    np.save(staging / "reflection_schedule.npy", result["reflection_schedule"], allow_pickle=False)
    for path in staging.iterdir():
        manifest["files"][path.name] = hashlib.sha256(path.read_bytes()).hexdigest()
    (staging / "manifest.json").write_text(json.dumps(manifest, indent=2, allow_nan=False) + "\n")
    # A competing completed directory is nonempty, so rename cannot replace it.
    if destination.exists():
        raise FileExistsError(destination)
    staging.rename(destination)


def load_comparison(destination, expected_identity):
    destination = Path(destination)
    manifest = json.loads((destination / "manifest.json").read_text())
    if manifest.get("schema") != SCHEMA or not manifest.get("complete"):
        raise ValueError("Saved comparison is incomplete or uses another format")
    if canonical_json_digest(manifest.get("identity")) != canonical_json_digest(expected_identity):
        raise ValueError("Saved comparison is incompatible with the requested experiment")
    names = set(expected_identity["conditions"])
    expected_files = {"source_schedule.npy", "reflection_schedule.npy", *[f"{n}.pt" for n in names], *[f"{n}_training.csv" for n in names]}
    if set(manifest.get("runs", {})) != names or set(manifest.get("files", {})) != expected_files:
        raise ValueError("Saved comparison is missing or duplicating declared artifacts")
    for name, digest in manifest["files"].items():
        path = destination / name
        if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != digest:
            raise ValueError(f"Saved comparison artifact failed verification: {name}")
    runs = {}
    settings = LearningSettings(**expected_identity["settings"])
    schedule = np.load(destination / "source_schedule.npy", allow_pickle=False)
    reflections = np.load(destination / "reflection_schedule.npy", allow_pickle=False)
    shape = (settings.steps, settings.batch_size)
    if (schedule.shape != shape or not np.issubdtype(schedule.dtype, np.integer)
            or schedule.min() < 0 or schedule.max() >= len(expected_identity["source_ids"])
            or reflections.shape != shape or reflections.dtype != bool):
        raise ValueError("Saved source or reflection schedule has an invalid shape or value")
    schedule_sources = np.asarray(expected_identity["source_ids"])[schedule.ravel()]
    if not set(schedule_sources) <= set(expected_identity["train_sources"]):
        raise ValueError("Saved source schedule contains a test video")
    for name, metadata in manifest["runs"].items():
        model, projector = _new_model(settings, expected_identity["frames"], settings.device)
        initial_model = copy.deepcopy(model)
        if (metadata["settings"] != expected_identity["settings"]
                or metadata["policy"] != expected_identity["conditions"][name]
                or metadata["source_draw_digest"] != _digest_array(schedule)
                or metadata["initial_state_digest"] != state_digest(initial_model)
                or metadata["initial_projector_digest"] != state_digest(projector)):
            raise ValueError("Saved run metadata disagree with the experiment or its controls")
        saved = torch.load(destination / f"{name}.pt", map_location=settings.device, weights_only=True)
        if set(saved["checkpoints"]) != set(expected_identity["checkpoint_steps"]):
            raise ValueError("Saved checkpoints do not match the declared update positions")
        for state in (saved["model"], saved["initial_model"], *saved["checkpoints"].values()):
            if set(state) != set(model.state_dict()) or any(
                    v.shape != model.state_dict()[k].shape or not torch.isfinite(v).all()
                    for k, v in state.items()):
                raise ValueError("Saved model or checkpoint is incomplete or nonfinite")
        if any(not torch.equal(v, saved["checkpoints"][settings.steps][k]) for k, v in saved["model"].items()):
            raise ValueError("Final model and final declared checkpoint disagree")
        model.load_state_dict(saved["model"]); initial_model.load_state_dict(saved["initial_model"])
        projector.load_state_dict(saved["projector"])
        history = pd.read_csv(destination / f"{name}_training.csv")
        if (history.step.tolist() != list(range(1, settings.steps + 1))
                or not np.isfinite(history.select_dtypes(include=np.number)).all().all()):
            raise ValueError("Training history does not cover the complete run")
        counts = np.asarray(metadata["hidden_token_counts"])
        if counts.shape != shape or not np.issubdtype(counts.dtype, np.integer) or counts.min() < 1:
            raise ValueError("Saved hidden counts have invalid coverage")
        if len(metadata["coverage"]) != settings.steps:
            raise ValueError("Saved mask coverage omits training updates")
        for step, coverage in enumerate(metadata["coverage"]):
            if len(coverage) != settings.batch_size:
                raise ValueError("Saved mask coverage omits clips")
            for row, c in enumerate(coverage):
                if (c["hidden_tokens"] != counts[step, row] or c["context_tokens"] < 1
                        or c["valid_tokens"] != c["hidden_tokens"] + c["context_tokens"]
                        or c["reflected"] != bool(reflections[step, row])
                        or sum(c[k] for k in ("left_hidden_tokens", "right_hidden_tokens", "midline_hidden_tokens")) != c["hidden_tokens"]):
                    raise ValueError("Saved mask coverage and target counts disagree")
        if state_digest(initial_model) != metadata["initial_state_digest"]:
            raise ValueError("Initial checkpoint does not match the retained controls")
        runs[name] = {**metadata, "model": model.eval(), "initial_model": initial_model.eval(),
            "projector": projector.eval(), "history": history, "checkpoints": saved["checkpoints"]}
    recomputed = {"same_initialization": len({r["initial_state_digest"] for r in runs.values()}) == 1,
        "same_projector_initialization": len({r["initial_projector_digest"] for r in runs.values()}) == 1,
        "same_source_draws": len({r["source_draw_digest"] for r in runs.values()}) == 1,
        "same_geometric_views": len({r["view_digest"] for r in runs.values()}) == 1,
        "same_hidden_counts": len({json.dumps(r["hidden_token_counts"]) for r in runs.values()}) == 1}
    if manifest["pairing"] != recomputed or not all(recomputed.values()):
        raise ValueError("Saved comparison controls did not pass")
    return {"runs": runs, "pairing": manifest["pairing"], "identity": expected_identity,
        "source_schedule": schedule, "reflection_schedule": reflections,
        "synthetic": expected_identity["synthetic"], "reused": True}


def saved_reference_recipe(root=SUITE_ROOT):
    """Read all 25 retained pairs and require one consistent reference recipe."""
    manifests = sorted((Path(root) / "artifacts/research_extensions/masking").glob("exploratory_real_*/manifest.json"))
    recipes, pairs = [], []
    for path in manifests:
        manifest = json.loads(path.read_text())
        if not manifest.get("complete") or manifest.get("synthetic"):
            continue
        if not all(manifest.get("pairing", {}).values()):
            raise ValueError("A retained reference comparison failed its controls")
        arm_recipes = []
        for run in manifest["runs"].values():
            s = dict(run["settings"])
            arm_recipes.append({k: v for k, v in s.items() if k not in {"seed", "fold", "mask_policy", "device", "confirm_real_run"}})
        if any(r != arm_recipes[0] for r in arm_recipes):
            raise ValueError("Reference arms differ in an unexpected setting")
        recipes.append(arm_recipes[0]); pairs.append((s["fold"], s["seed"]))
    if len(recipes) != 25 or set(pairs) != {(f, s) for f in range(5) for s in range(42, 47)}:
        raise ValueError("The expected complete reference grid was not found")
    if any(recipe != recipes[0] for recipe in recipes):
        raise ValueError("Saved reference recipes disagree")
    return {**recipes[0], "optimizer": "AdamW", "optimizer_betas": (0.9, .95),
        "learning_rate_schedule": "constant", "teacher_momentum_schedule": "constant",
        "target_center_beta": .9, "predictor_temperature": .10, "target_temperature": .06,
        "gradient_clip_norm": 1., "view_rotation_degrees": 8., "view_translation": .03,
        "input_landmarks": 33, "regularizer_landmarks": list(GAIT_JOINTS),
        "reference_pairs": len(pairs), "reference_trained_encoders": 2 * len(pairs)}


def real_settings(*, fold=0, seed=42, device="cpu", confirm_real_run=False):
    recipe = saved_reference_recipe()
    fields = set(LearningSettings.__dataclass_fields__)
    return LearningSettings(**{k: v for k, v in recipe.items() if k in fields},
        fold=fold, seed=seed, device=device, confirm_real_run=confirm_real_run)


def plan_real_comparison(*, conditions=None, folds=tuple(range(5)), seeds=tuple(range(42, 47)),
        device="cpu", output_dir=None, budgets=None, matched_to=None, validate_inputs=False):
    """Describe every requested fit. Optional input validation performs no training."""
    conditions = dict(conditions or default_conditions())
    folds, seeds = tuple(folds), tuple(seeds)
    if not folds or not seeds or len(set(folds)) != len(folds) or len(set(seeds)) != len(seeds):
        raise ValueError("Declare nonempty unique folds and seeds")
    if not set(folds) <= set(range(5)) or not set(seeds) <= set(range(42, 47)):
        raise ValueError("The declared real comparison uses folds 0–4 and seeds 42–46")
    settings = real_settings(device=device)
    rows = [{"fold": f, "seed": s, "condition": n, "updates": settings.steps,
             "batch_size": settings.batch_size} for f in folds for s in seeds for n in conditions]
    checks = []
    if validate_inputs:
        for fold in folds:
            data = load_learning_dataset(real=True, fold=fold)
            setting = replace(settings, fold=fold)
            # One pass over all training clips checks policy feasibility; full
            # scheduled masks are checked again before actual model training.
            for start in range(0, len(data.train_rows), setting.batch_size):
                masks_for_batch(data, data.train_rows[start:start + setting.batch_size], setting,
                    conditions, budgets=budgets, matched_to=matched_to, step=start)
            checks.append({"fold": fold, "train_clips": len(data.train_rows), "test_clips": len(data.test_rows),
                "train_sources": len(data.train_sources), "test_sources": len(data.test_sources)})
    return {"workload": pd.DataFrame(rows), "training_runs": len(rows),
        "optimizer_updates": len(rows) * settings.steps,
        "sample_presentations": len(rows) * settings.steps * settings.batch_size,
        "scope": "full declared comparison" if folds == tuple(range(5)) and seeds == tuple(range(42, 47)) else "pilot",
        "settings": asdict(settings), "recipe": saved_reference_recipe(),
        "conditions": {n: asdict(p) for n, p in conditions.items()},
        "budgets": {n: asdict(b) for n, b in (budgets or {}).items()}, "matched_to": matched_to,
        "folds": folds, "seeds": seeds, "input_checks": pd.DataFrame(checks),
        "output_dir": str(output_dir or SUITE_ROOT / "artifacts/comparative_masking"),
        "runtime_estimate": "Measure a training-source pilot; equal updates do not imply equal runtime."}


def run_real_comparison(plan, *, enabled=False):
    """Execute the displayed plan only after explicit enablement."""
    print(plan["workload"].to_string(index=False))
    print(f"{plan['scope']}: {plan['training_runs']} encoders; {plan['optimizer_updates']} updates")
    print(f"Output: {plan['output_dir']}")
    if not enabled:
        return {"status": "Training disabled", "plan": plan}
    conditions = {n: MaskPolicy(**p) for n, p in plan["conditions"].items()}
    budgets = {n: MaskBudget(**b) for n, b in plan["budgets"].items()}
    completed, predictions, expected_rows = [], [], []
    for fold in plan["folds"]:
        data = load_learning_dataset(real=True, fold=fold)
        expected_rows.append(pd.DataFrame({"sequence_id": data.sequence_ids[data.test_rows],
            "source_id": data.source_ids[data.test_rows], "fold": fold}))
        for seed in plan["seeds"]:
            settings = replace(LearningSettings(**plan["settings"]), fold=fold, seed=seed, confirm_real_run=True)
            result = train_comparison(data, settings, conditions, budgets=budgets,
                matched_to=plan["matched_to"], output_dir=plan["output_dir"])
            evaluation = evaluate_comparison(result, data, settings,
                output_dir=Path(plan["output_dir"]) / "evaluations")
            predictions.append(evaluation["predictions"])
            completed.append({"fold": fold, "seed": seed, "identity": result["identity"], "reused": result["reused"]})
            del result
    from laterality_extensions.comparative_evaluation import aggregate_predictions, paired_source_bootstrap
    table = pd.concat(predictions, ignore_index=True)
    expected = pd.concat(expected_rows, ignore_index=True)
    summaries = aggregate_predictions(table, expected, seeds=plan["seeds"], conditions=tuple(conditions),
        representations=tuple(table.representation.unique()), observations=tuple(table.observation.unique()))
    intervals = []
    first = next(iter(conditions))
    for other in list(conditions)[1:]:
        intervals.append(paired_source_bootstrap(table, first=other, reference=first,
            representation="pretrained_teacher"))
    tables = {"predictions": table, **summaries, "paired_intervals": pd.DataFrame(intervals)}
    summary_identity = {"runs": [canonical_json_digest(c["identity"]) for c in completed],
        "scope": plan["scope"], "aggregation": "pooled outer predictions per seed; source balanced",
        "implementation": implementation_digest()}
    _save_tables(tables, summary_identity, Path(plan["output_dir"]) / "summaries")
    return {"status": "Complete", "completed": completed, "plan": plan, **tables}


def _save_tables(tables, identity, parent):
    """Save an immutable evaluated result separately from training checkpoints."""
    parent = Path(parent)
    destination = parent / canonical_json_digest(identity)
    expected = {f"{name}.csv" for name in tables}
    if destination.exists():
        manifest = json.loads((destination / "manifest.json").read_text())
        if (not manifest.get("complete") or manifest.get("identity") != identity
                or set(manifest.get("files", {})) != expected):
            raise ValueError("Saved evaluation is incomplete or incompatible")
        for name, digest in manifest["files"].items():
            if hashlib.sha256((destination / name).read_bytes()).hexdigest() != digest:
                raise ValueError("Saved evaluation failed its content check")
        return destination
    parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=".incomplete_evaluation_", dir=parent))
    for name, table in tables.items():
        table.to_csv(temporary / f"{name}.csv", index=False)
    manifest = {"complete": True, "identity": identity,
        "files": {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in temporary.iterdir()}}
    (temporary / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    if destination.exists():
        raise FileExistsError(destination)
    temporary.rename(destination)
    return destination


def evaluate_comparison(result, dataset, settings, *, output_dir=None):
    """Apply the same fixed evaluation bank and training-only readout to all arms.

    The default missing-input scores concern already prepared coordinates.
    Raw missing-data preparation is a separate demonstrated helper in tutorial 13.
    """
    from laterality_extensions.comparative_evaluation import (
        evaluate_frozen_representations, make_evaluation_mask_bank,
        prepared_observation_sensitivity, predictor_diagnostics,
    )
    patches = dataset.valid.reshape(len(dataset.xyz), -1, settings.segment_length, 33).all(2)
    bank = make_evaluation_mask_bank(patches)
    observations = {f"prepared_{name}": prepared_observation_sensitivity(dataset, mask,
        segment_length=settings.segment_length) for name, mask in bank.items()}
    # The comparison definition is shared across folds and seeds; each training
    # cache identity separately identifies its particular fitted models.
    design = {k: result["identity"][k] for k in ("conditions", "budgets", "matched_to", "implementation")}
    comparison_id = canonical_json_digest(design)
    prediction_tables, diagnostics, validation, feature_checks = [], [], [], []
    for name, run in result["runs"].items():
        evaluation = evaluate_frozen_representations(run["model"], run["initial_model"], dataset,
            settings, condition=name, observation_datasets=observations, comparison_id=comparison_id)
        prediction_tables.append(evaluation["predictions"])
        for representation, readout in evaluation["readouts"].items():
            validation.append(readout.validation.assign(condition=name, representation=representation,
                selected_alpha=readout.selected_alpha, fold=dataset.fold, seed=settings.seed))
            feature_checks.append({"condition": name, "representation": representation,
                **readout.training_feature_diagnostics})
        for state, model in (("pretrained", run["model"]), ("initial", run["initial_model"])):
            diagnostics.append(predictor_diagnostics(model, dataset, bank, condition=name).assign(
                encoder_state=state, fold=dataset.fold, seed=settings.seed))
    tables = {"predictions": pd.concat(prediction_tables, ignore_index=True),
        "predictor_diagnostics": pd.concat(diagnostics, ignore_index=True),
        "readout_validation": pd.concat(validation, ignore_index=True),
        "training_feature_diagnostics": pd.DataFrame(feature_checks)}
    if output_dir is not None:
        identity = {"training": canonical_json_digest(result["identity"]),
            "evaluation": {"alphas": [.01, .1, 1., 10., 100.], "inner_folds": 3,
                "mask_bank_seed": 813, "mismatch_seed": 991, "missingness": "prepared-coordinate sensitivity"},
            "implementation": implementation_digest()}
        tables["output_path"] = _save_tables(tables, identity, output_dir)
    return tables
