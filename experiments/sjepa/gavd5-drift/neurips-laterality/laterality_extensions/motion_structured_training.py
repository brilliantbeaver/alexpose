"""Isolated JEPA runner for motion and structure comparisons, notebooks 15–18.

The model, source sampler, objective primitives and checked atomic artifact
format come from the existing suite. Old experiment implementations are untouched.
This runner always uses dense per-clip loss reduction, including ragged masks.
Completed jobs can be reused; interrupted jobs restart from their declared seed.
"""
from __future__ import annotations

import copy
from dataclasses import asdict, replace
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import pandas as pd
import torch

from laterality.config import SUITE_ROOT, canonical_json_digest
from laterality.model import authorized_pool, geometric_view, vicreg_loss
from .comparative_masks import GAIT_JOINTS
from .comparative_training import (
    _digest_array, _new_model, _save_tables, comparison_identity, dense_prediction,
    load_comparison, per_clip_prediction_loss, save_comparison,
    source_schedule, state_digest, training_implementation_digest,
)
from .masked_learning import LearningSettings, configure_learning_runtime, load_learning_dataset
from .motion_structured_masks import StudyArm, paired_study_masks, study_arms


def study_digest():
    digest = hashlib.sha256(training_implementation_digest().encode())
    for name in ("motion_structured_masks.py", "motion_structured_training.py"):
        digest.update(Path(__file__).with_name(name).read_bytes())
    return digest.hexdigest()


def train_mask_study(dataset, settings, *, experiment="motion", arms=None,
                     output_dir=None, checkpoint_steps=(), progress=None):
    """Train only declared training sources, with identical shared views per step."""
    settings.validate()
    dataset.validate(settings.segment_length)
    if dataset.fold != settings.fold:
        raise ValueError("Dataset and training fold disagree")
    if settings.symmetry_weight or settings.reflection_probability:
        raise ValueError("Reflection/symmetry training needs a separate experiment")
    if not dataset.synthetic and not settings.confirm_real_run:
        raise PermissionError("Real-data training must be explicitly enabled")
    runtime = configure_learning_runtime(settings.device)
    settings = replace(settings, device=runtime["device"])
    blocks = dataset.xyz.shape[1] // settings.segment_length
    arms = dict(arms or study_arms(experiment, blocks))
    # Validate names and specifications before either cache reuse or training.
    paired_study_masks(dataset, dataset.train_rows[:settings.batch_size], settings, arms)
    checkpoints = tuple(sorted(set((*checkpoint_steps, settings.steps))))
    if any(not isinstance(s, int) or isinstance(s, bool) or not 1 <= s <= settings.steps for s in checkpoints):
        raise ValueError("Checkpoint steps must lie within training exposure")
    identity = comparison_identity(dataset, settings, arms, {}, None, checkpoints)
    identity.update(study_schema="motion_structured/v1", study_implementation=study_digest(),
                    experiment=experiment, objective_reduction="mean targets per clip, then mean clips",
                    interruption_policy="restart incomplete jobs; reuse complete compatible jobs")
    destination = Path(output_dir) / canonical_json_digest(identity) if output_dir else None
    if destination is not None and destination.exists():
        return load_comparison(destination, identity)

    schedule = source_schedule(dataset, settings)
    mask_steps, coverage_steps = [], []
    # Fail on any infeasible scheduled mask before optimizing any arm.
    for step, rows in enumerate(schedule):
        masks, coverage = paired_study_masks(dataset, rows, settings, arms, step=step)
        mask_steps.append(masks)
        coverage_steps.append(coverage)
    model, projector = _new_model(settings, dataset.xyz.shape[1], settings.device)
    initial_digest, projector_digest = state_digest(model), state_digest(projector)
    runs, optimizers = {}, {}
    for name, arm in arms.items():
        net, proj = copy.deepcopy(model), copy.deepcopy(projector)
        trainable = [p for p in (*net.parameters(), *proj.parameters()) if p.requires_grad]
        optimizers[name] = (torch.optim.AdamW(trainable, lr=settings.learning_rate,
                            weight_decay=settings.weight_decay, betas=(0.9, 0.95)), trainable)
        runs[name] = {"model": net, "initial_model": copy.deepcopy(model).eval(),
                      "projector": proj, "settings": asdict(settings), "policy": asdict(arm),
                      "initial_state_digest": initial_digest, "initial_projector_digest": projector_digest,
                      "source_draw_digest": _digest_array(schedule), "history": [], "checkpoints": {},
                      "hidden_token_counts": [m[name].sum((1, 2)).tolist() for m in mask_steps],
                      "coverage": [c[name] for c in coverage_steps]}
    view_digest = hashlib.sha256()
    started = time.monotonic()
    for step, rows in enumerate(schedule):
        # Generate both views on CPU with an isolated stream, then share tensors.
        valid_cpu = torch.as_tensor(dataset.valid[rows], dtype=torch.bool)
        xyz_cpu = torch.as_tensor(np.where(dataset.valid[rows, ..., None], dataset.xyz[rows], 0), dtype=torch.float32)
        with torch.random.fork_rng(devices=[]):
            torch.random.default_generator.manual_seed(settings.seed + 100_003 * (settings.fold + 1) + step)
            both_cpu = geometric_view(torch.cat((xyz_cpu, xyz_cpu)), torch.cat((valid_cpu, valid_cpu)),
                                      max_degrees=8, translate=0.03)
        view_digest.update(both_cpu.numpy().tobytes())
        both = both_cpu.to(settings.device)
        xyz = xyz_cpu.to(settings.device)
        patches = valid_cpu.reshape(len(rows), blocks, settings.segment_length, 33).all(2).to(settings.device)
        both_valid = torch.cat((patches, patches))
        for name, run in runs.items():
            net, proj = run["model"], run["projector"]
            net.train(); proj.train(); net.target_encoder.eval()
            mask = torch.as_tensor(mask_steps[step][name], device=settings.device)
            predicted, target = dense_prediction(net, both[:len(rows)], xyz, patches, mask)
            predictive = per_clip_prediction_loss(predicted, target, mask, net.target_center)
            tokens = net.view_encoder(both, both_valid).reshape(2 * len(rows), blocks, 33, settings.embed_dim)
            regularizer = vicreg_loss(*proj(authorized_pool(tokens, both_valid, GAIT_JOINTS)).chunk(2))
            loss = predictive + settings.vicreg_weight * regularizer
            if not torch.isfinite(loss):
                raise FloatingPointError("Nonfinite JEPA loss")
            optimizer, trainable = optimizers[name]
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            if any(p.grad is not None for p in net.target_encoder.parameters()):
                raise AssertionError("Teacher received gradients")
            torch.nn.utils.clip_grad_norm_(trainable, 1.0)
            optimizer.step()
            net.update_target(settings.ema_momentum)
            with torch.no_grad():
                hidden = mask.flatten(1)
                mean = ((target * hidden[..., None]).sum(1) / hidden.sum(1)[:, None]).mean(0)
                net.target_center.mul_(0.9).add_(mean, alpha=0.1)
            counts = mask.sum((1, 2))
            run["history"].append({"step": step + 1, "loss": float(loss.detach()),
                "masked_prediction_loss": float(predictive.detach()), "variance_regularizer": float(regularizer.detach()),
                "hidden_count_min": int(counts.min()), "hidden_count_max": int(counts.max())})
            if step + 1 in checkpoints:
                run["checkpoints"][step + 1] = {k: v.detach().cpu().clone() for k, v in net.state_dict().items()}
        if progress is not None:
            progress({"step": step + 1, "total_steps": settings.steps, "encoders": len(runs)})
    elapsed = time.monotonic() - started
    resident_bytes = int(xyz.numel() * xyz.element_size() + both.numel() * both.element_size())
    for run in runs.values():
        run.update(history=pd.DataFrame(run["history"]), view_digest=view_digest.hexdigest(),
                   elapsed_training_seconds=elapsed, resident_input_bytes=resident_bytes,
                   execution_layout="shared CPU-generated views; dense per-clip target loss")
        run["model"].eval(); run["projector"].eval()
    pairing = {"same_initialization": len({r["initial_state_digest"] for r in runs.values()}) == 1,
               "same_projector_initialization": len({r["initial_projector_digest"] for r in runs.values()}) == 1,
               "same_source_draws": len({r["source_draw_digest"] for r in runs.values()}) == 1,
               "same_geometric_views": len({r["view_digest"] for r in runs.values()}) == 1,
               "same_hidden_counts": all(r["hidden_token_counts"] == next(iter(runs.values()))["hidden_token_counts"] for r in runs.values())}
    if not all(pairing.values()):
        raise AssertionError("Paired study controls failed")
    result = {"runs": runs, "pairing": pairing, "identity": identity, "source_schedule": schedule,
              "reflection_schedule": np.zeros(schedule.shape, dtype=bool), "synthetic": dataset.synthetic,
              "reused": False, "resumed_from_step": 0}
    if destination is not None:
        save_comparison(result, destination)
    return result


def plan_mask_study(*, experiments=("motion", "regions"), folds=tuple(range(5)),
                    seeds=tuple(range(42, 47)), device="auto", validate_inputs=False, output_dir=None):
    """Read-only real-data plan; validation never starts training or downloads data."""
    experiments, folds, seeds = tuple(experiments), tuple(folds), tuple(seeds)
    for values in (experiments, folds, seeds):
        if not values or len(set(values)) != len(values):
            raise ValueError("Experiments, folds and seeds must be nonempty and unique")
    if not set(folds) <= set(range(5)) or not set(seeds) <= set(range(42, 47)):
        raise ValueError("Real comparisons use folds 0–4 and seeds 42–46")
    recipe_path = SUITE_ROOT / "docs/figures/tutorial_masking_summary.json"
    reference = json.loads(recipe_path.read_text(encoding="utf-8"))
    settings = LearningSettings(**reference["shared_training_settings"], device=device)
    settings.validate()
    definitions = {name: study_arms(name, 16) for name in experiments}
    jobs = [{"experiment": e, "fold": f, "seed": s, "condition": c, "updates": settings.steps}
            for e in experiments for f in folds for s in seeds for c in definitions[e]]
    checks = []
    if validate_inputs:
        for fold in folds:
            data = load_learning_dataset(real=True, fold=fold)
            if data.xyz.shape[1] != 64 or settings.segment_length != 4:
                raise ValueError("Real structure declarations require 64 steps and four-step tokens")
            local = replace(settings, fold=fold)
            for experiment, arms in definitions.items():
                counts = []
                for start in range(0, len(data.train_rows), local.batch_size):
                    masks, _ = paired_study_masks(data, data.train_rows[start:start + local.batch_size], local, arms, step=start)
                    counts.extend(next(iter(masks.values())).sum((1, 2)).tolist())
                checks.append({"fold": fold, "experiment": experiment, "train_clips": len(data.train_rows),
                               "test_clips": len(data.test_rows), "train_sources": len(data.train_sources),
                               "test_sources": len(data.test_sources), "hidden_min": min(counts), "hidden_max": max(counts)})
    return {"settings": asdict(settings), "experiments": experiments, "folds": folds, "seeds": seeds,
            "arms": {e: {n: asdict(a) for n, a in arms.items()} for e, arms in definitions.items()},
            "workload": pd.DataFrame(jobs), "training_runs": len(jobs),
            "optimizer_updates": len(jobs) * settings.steps,
            "scope": "full development comparison" if set(folds) == set(range(5)) and set(seeds) == set(range(42, 47)) else "pilot",
            "input_checks": pd.DataFrame(checks), "study_implementation": study_digest(),
            "recipe_source": str(recipe_path),
            "recipe_digest": hashlib.sha256(recipe_path.read_bytes()).hexdigest(),
            "output_dir": str(output_dir or SUITE_ROOT / "artifacts/motion_structured")}


def run_mask_study(plan, *, enabled=False, log=print):
    """Execute exactly the reviewed plan; save trained and evaluated jobs separately."""
    from .motion_readout import evaluate_motion_readouts, aggregate_motion_study
    emit = log or (lambda message: None)
    emit(f"{plan['scope']}: {plan['training_runs']} encoders, {plan['optimizer_updates']:,} updates; {plan['output_dir']}")
    if not enabled:
        return {"status": "Training disabled", "plan": plan}
    if plan["study_implementation"] != study_digest():
        raise ValueError("Implementation changed; rebuild the study plan")
    base = LearningSettings(**plan["settings"])
    base.validate()
    for values in (plan["experiments"], plan["folds"], plan["seeds"]):
        if not values or len(values) != len(set(values)):
            raise ValueError("Plan declarations must be nonempty and unique")
    if (set(plan["arms"]) != set(plan["experiments"])
            or not set(plan["folds"]) <= set(range(5))
            or not set(plan["seeds"]) <= set(range(42, 47))):
        raise ValueError("Plan arms, folds or seeds are outside their declaration")
    expected_workload = pd.DataFrame([
        {"experiment": e, "fold": f, "seed": s, "condition": c, "updates": base.steps}
        for e in plan["experiments"] for f in plan["folds"] for s in plan["seeds"] for c in plan["arms"][e]])
    if (not plan["workload"].equals(expected_workload)
            or len(expected_workload) != plan["training_runs"]
            or plan["optimizer_updates"] != len(expected_workload) * base.steps):
        raise ValueError("Workload and aggregate counts disagree")
    evaluations, expected, outputs = [], [], []
    for fold in plan["folds"]:
        data = load_learning_dataset(real=True, fold=fold)
        expected.append(pd.DataFrame({"sequence_id": data.sequence_ids[data.test_rows],
                                     "source_id": data.source_ids[data.test_rows], "fold": fold}))
        for seed in plan["seeds"]:
            settings = replace(base, fold=fold, seed=seed, confirm_real_run=True)
            for experiment in plan["experiments"]:
                emit(f"Fold {fold}, seed {seed}: {experiment}")
                result = train_mask_study(data, settings, experiment=experiment,
                    arms={n: StudyArm(**a) for n, a in plan["arms"][experiment].items()},
                    output_dir=Path(plan["output_dir"]) / experiment,
                    progress=lambda update: emit(f"  update {update['step']}/{update['total_steps']}")
                    if update["step"] % 100 == 0 or update["step"] == update["total_steps"] else None)
                evaluation = evaluate_motion_readouts(result, data, settings, include_predictor=True)
                evaluation["predictions"]["experiment"] = experiment
                evaluation["selection"]["experiment"] = experiment
                _save_tables({k: evaluation[k] for k in ("predictions", "selection", "diagnostics", "predictor_diagnostics")},
                    evaluation["identity"], Path(plan["output_dir"]) / "evaluations")
                evaluations.append(evaluation["predictions"])
                outputs.append(evaluation["identity"])
    predictions = pd.concat(evaluations, ignore_index=True)
    summary = aggregate_motion_study(predictions, pd.concat(expected, ignore_index=True), plan)
    _save_tables({"predictions": predictions, **summary}, {"evaluations": outputs, "scope": plan["scope"]},
                 Path(plan["output_dir"]) / "summaries")
    return {"status": "Complete", "predictions": predictions, **summary, "plan": plan}
