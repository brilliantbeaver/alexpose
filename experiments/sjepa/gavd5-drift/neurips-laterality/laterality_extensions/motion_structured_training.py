"""Isolated JEPA runner for motion and structure comparisons, notebooks 15–18.

The model, source sampler, objective primitives and checked atomic artifact
format come from the existing suite. Old experiment implementations are untouched.
This runner always uses dense per-clip loss reduction, including ragged masks.
Completed jobs can be reused; interrupted jobs resume from a checked shared-arm
optimizer boundary when periodic resume checkpoints are enabled.
"""
from __future__ import annotations

import copy
from functools import wraps
from dataclasses import asdict, dataclass, replace
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
    _diagnostic_values, _digest_array, _encode_prevalidated,
    _load_training_resume, _new_model, _prediction_prevalidated,
    _resume_checksum_path, _resume_path, _save_tables, _synchronize_device,
    _write_training_resume, comparison_identity, load_comparison,
    save_comparison, source_schedule, state_digest, training_implementation_digest,
)
from .masked_learning import LearningSettings, configure_learning_runtime, load_learning_dataset
from .motion_structured_masks import StudyArm, paired_study_masks, study_arms
from .motion_runtime import (motion_numerical_context, motion_numerical_policy,
                             prediction_resident, update_teacher_foreach)


def study_digest():
    digest = hashlib.sha256(training_implementation_digest().encode())
    for name in ("motion_structured_masks.py", "motion_structured_training.py", "motion_runtime.py"):
        digest.update(Path(__file__).with_name(name).read_bytes())
    return digest.hexdigest()


def mask_study_identity(dataset, settings, experiment, arms, checkpoint_steps=(), *, precision="fp32"):
    """One compatibility contract for training and read-only checkpoint loading."""
    checkpoints = tuple(sorted(set((*checkpoint_steps, settings.steps))))
    identity = comparison_identity(dataset, settings, arms, {}, None, checkpoints)
    identity.update(study_schema="motion_structured/v1", study_implementation=study_digest(),
                    experiment=experiment, objective_reduction="mean targets per clip, then mean clips",
                    interruption_policy="resume exact shared-arm boundaries; reuse complete compatible jobs")
    identity["numerical_policy"] = motion_numerical_policy(settings.device, precision)
    return identity


@dataclass(frozen=True)
class _ResidentTrainTensors:
    """One immutable, training-only fold tensor bank on the selected device."""

    key: str
    xyz: torch.Tensor
    valid: torch.Tensor
    valid_patch: torch.Tensor
    global_to_local: np.ndarray
    resident_bytes: int


@dataclass(frozen=True)
class _PreparedMaskStudyInputs:
    """Schedules and masks prepared once, then indexed without host transfers."""

    schedule: np.ndarray
    schedule_rows: torch.Tensor
    tensors: _ResidentTrainTensors
    target_masks: dict[str, torch.Tensor]
    target_indices: dict[str, list]
    target_counts: dict[str, list[list[int]]]
    equal_counts: dict[str, list[bool]]
    coverage: dict[str, list[list[dict]]]
    view_digest: str
    resident_bytes: int
    tensor_cache_reused: bool


def _resident_train_tensors(dataset, settings, device, cache=None):
    """Keep only outer-training rows resident; outer-test tensors stay sealed."""
    train_rows = np.asarray(dataset.train_rows, dtype=np.int64)
    train_xyz = np.ascontiguousarray(dataset.xyz[train_rows])
    train_valid = np.ascontiguousarray(dataset.valid[train_rows])
    key = canonical_json_digest({
        "schema": "motion_structured_train_tensors/v1",
        "cohort": dataset.cohort_digest,
        "split": dataset.split_digest,
        "fold": int(dataset.fold),
        "segment_length": int(settings.segment_length),
        "device": str(device),
        "train_rows": _digest_array(train_rows),
        "xyz": _digest_array(train_xyz),
        "valid": _digest_array(train_valid),
    })
    if cache is not None and key in cache:
        return cache[key], True

    clean = np.where(train_valid[..., None], train_xyz, 0)
    patches = train_valid.reshape(
        len(train_rows), -1, settings.segment_length, 33
    ).all(axis=2)
    global_to_local = np.full(len(dataset.xyz), -1, dtype=np.int64)
    global_to_local[train_rows] = np.arange(len(train_rows), dtype=np.int64)
    xyz = torch.as_tensor(clean, dtype=torch.float32, device=device).contiguous()
    valid = torch.as_tensor(train_valid, dtype=torch.bool, device=device).contiguous()
    valid_patch = torch.as_tensor(
        patches, dtype=torch.bool, device=device
    ).contiguous()
    tensors = _ResidentTrainTensors(
        key=key,
        xyz=xyz,
        valid=valid,
        valid_patch=valid_patch,
        global_to_local=global_to_local,
        resident_bytes=sum(
            value.numel() * value.element_size()
            for value in (xyz, valid, valid_patch)
        ),
    )
    if cache is not None:
        cache[key] = tensors
    return tensors, False


def _prepare_mask_study_inputs(dataset, settings, arms, device, *, input_cache=None):
    """Validate the full schedule and make masks accelerator-resident once."""
    schedule = source_schedule(dataset, settings)
    if set(dataset.source_ids[schedule.ravel()].astype(str)) & set(dataset.test_sources):
        raise AssertionError("Outer-test videos reached the training schedule")

    mask_steps, coverage_steps = [], []
    score_cache = {} if input_cache is None else input_cache.setdefault("motion_scores", {})
    for step, rows in enumerate(schedule):
        masks, coverage = paired_study_masks(
            dataset, rows, settings, arms, step=step, score_cache=score_cache
        )
        mask_steps.append(masks)
        coverage_steps.append(coverage)

    tensors, cache_reused = _resident_train_tensors(
        dataset, settings, device, input_cache
    )
    local_schedule = tensors.global_to_local[schedule]
    if (local_schedule < 0).any():
        raise AssertionError("A scheduled row is absent from the training-only tensor bank")
    schedule_rows = torch.as_tensor(
        local_schedule, dtype=torch.long, device=device
    ).contiguous()
    target_masks, target_indices, target_counts, equal_counts, coverage = {}, {}, {}, {}, {}
    for name in arms:
        values = np.ascontiguousarray(
            np.stack([step_masks[name] for step_masks in mask_steps])
        )
        counts = values.reshape(settings.steps, settings.batch_size, -1).sum(2)
        target_masks[name] = torch.as_tensor(
            values, dtype=torch.bool, device=device
        ).contiguous()
        target_counts[name] = counts.astype(int).tolist()
        equal_counts[name] = [
            bool(np.all(step_counts == step_counts[0])) for step_counts in counts
        ]
        # Known output shapes avoid nonzero/boolean-indexing synchronization in
        # both predictor and teacher on every CUDA update.
        target_indices[name] = [
            torch.as_tensor(np.nonzero(value.reshape(settings.batch_size, -1))[1]
                .reshape(settings.batch_size, -1).copy(), dtype=torch.long, device=device)
            if equal else None
            for value, equal in zip(values, equal_counts[name])
        ]
        coverage[name] = [step_coverage[name] for step_coverage in coverage_steps]
    resident = tensors.resident_bytes + schedule_rows.numel() * schedule_rows.element_size()
    resident += sum(mask.numel() * mask.element_size() for mask in target_masks.values())
    resident += sum(index.numel() * index.element_size()
                    for indices in target_indices.values() for index in indices if index is not None)
    view_digest = canonical_json_digest({
        "schema": "motion_structured_geometric_views/v2",
        "source_schedule": _digest_array(schedule),
        "seeds": [
            settings.seed + 100_003 * (settings.fold + 1) + step
            for step in range(settings.steps)
        ],
        "device": str(device),
        "max_rotation_degrees": 8.0,
        "max_translation": 0.03,
    })
    return _PreparedMaskStudyInputs(
        schedule=schedule,
        schedule_rows=schedule_rows,
        tensors=tensors,
        target_masks=target_masks,
        target_indices=target_indices,
        target_counts=target_counts,
        equal_counts=equal_counts,
        coverage=coverage,
        view_digest=view_digest,
        resident_bytes=int(resident),
        tensor_cache_reused=cache_reused,
    )


def _numerical_scope(function):
    @wraps(function)
    def scoped(*args, **kwargs):
        with motion_numerical_context():
            return function(*args, **kwargs)
    return scoped


@_numerical_scope
def train_mask_study(dataset, settings, *, experiment="motion", arms=None,
                     output_dir=None, checkpoint_steps=(), progress=None,
                     resume_interval=0, input_cache=None, precision="fp32"):
    """Train paired arms with resident inputs and optional atomic job resume."""
    settings.validate()
    dataset.validate(settings.segment_length)
    if dataset.fold != settings.fold:
        raise ValueError("Dataset and training fold disagree")
    if settings.symmetry_weight or settings.reflection_probability:
        raise ValueError("Reflection/symmetry training needs a separate experiment")
    if not dataset.synthetic and not settings.confirm_real_run:
        raise PermissionError("Real-data training must be explicitly enabled")
    if not isinstance(resume_interval, int) or isinstance(resume_interval, bool) or resume_interval < 0:
        raise ValueError("resume_interval must be a nonnegative integer")
    runtime = configure_learning_runtime(settings.device)
    settings = replace(settings, device=runtime["device"])
    blocks = dataset.xyz.shape[1] // settings.segment_length
    arms = dict(arms or study_arms(experiment, blocks))
    # Validate names and specifications before either cache reuse or training.
    paired_study_masks(dataset, dataset.train_rows[:settings.batch_size], settings, arms)
    checkpoints = tuple(sorted(set((*checkpoint_steps, settings.steps))))
    if any(not isinstance(s, int) or isinstance(s, bool) or not 1 <= s <= settings.steps for s in checkpoints):
        raise ValueError("Checkpoint steps must lie within training exposure")
    identity = mask_study_identity(dataset, settings, experiment, arms, checkpoints, precision=precision)
    destination = Path(output_dir) / canonical_json_digest(identity) if output_dir else None
    if destination is not None and destination.exists():
        reused = load_comparison(destination, identity)
        stale_resume = _resume_path(destination)
        for path in (stale_resume, _resume_checksum_path(stale_resume)):
            path.unlink(missing_ok=True)
        return reused

    device = torch.device(settings.device)
    preparation_started = time.monotonic()
    prepared = _prepare_mask_study_inputs(
        dataset, settings, arms, device, input_cache=input_cache
    )
    preparation_seconds = time.monotonic() - preparation_started
    template_model, template_projector = _new_model(
        settings, dataset.xyz.shape[1], device
    )
    runtime_arms = {}
    for index, (name, policy) in enumerate(arms.items()):
        model = template_model if index == 0 else copy.deepcopy(template_model)
        projector = template_projector if index == 0 else copy.deepcopy(template_projector)
        initial_model = copy.deepcopy(model).eval().to("cpu")
        trainable = [
            *model.view_encoder.parameters(),
            *model.predictor.parameters(),
            *projector.parameters(),
        ]
        optimizer_options = {"fused": True} if device.type == "cuda" else {}
        optimizer = torch.optim.AdamW(
            trainable,
            lr=settings.learning_rate,
            weight_decay=settings.weight_decay,
            betas=(0.9, 0.95),
            **optimizer_options,
        )
        model.train()
        model.target_encoder.eval()
        projector.train()
        runtime_arms[name] = {
            "index": index,
            "policy": policy,
            "model": model,
            "projector": projector,
            "initial_model": initial_model,
            "initial_digest": state_digest(model),
            "initial_projector_digest": state_digest(projector),
            "trainable": trainable,
            "optimizer": optimizer,
            "diagnostics": [],
            "history_prefix": np.empty((0, 3), dtype=np.float64),
            "snapshots": {},
        }
    resume_path = (
        _resume_path(destination)
        if destination is not None and resume_interval
        else None
    )
    resumed_from_step, earlier_elapsed = (
        _load_training_resume(resume_path, identity, runtime_arms, device)
        if resume_path is not None
        else (0, 0.0)
    )
    if progress is not None and resumed_from_step:
        progress({
            "step": resumed_from_step,
            "total_steps": settings.steps,
            "encoders": len(runtime_arms),
            "resumed": True,
        })
    if device.type == "cuda":
        random_device_index = device.index if device.index is not None else torch.cuda.current_device()
        random_devices, random_device_type = [random_device_index], "cuda"
    elif device.type == "mps":
        random_device_index = 0
        random_devices, random_device_type = [0], "mps"
    else:
        random_device_index = None
        random_devices, random_device_type = [], None
    _synchronize_device(device)
    started = time.monotonic()
    for step in range(resumed_from_step, settings.steps):
        row_index = prepared.schedule_rows[step]
        xyz = prepared.tensors.xyz.index_select(0, row_index)
        valid = prepared.tensors.valid.index_select(0, row_index)
        patches = prepared.tensors.valid_patch.index_select(0, row_index)
        view_seed = settings.seed + 100_003 * (settings.fold + 1) + step
        with torch.random.fork_rng(devices=random_devices, device_type=random_device_type):
            if device.type == "cuda":
                with torch.cuda.device(random_device_index):
                    torch.cuda.manual_seed(view_seed)
            elif device.type == "mps":
                torch.mps.manual_seed(view_seed)
            else:
                torch.random.default_generator.manual_seed(view_seed)
            both = geometric_view(
                torch.cat((xyz, xyz), dim=0),
                torch.cat((valid, valid), dim=0),
                max_degrees=8,
                translate=0.03,
            )
        view_a, _ = both.chunk(2)
        both_valid = torch.cat((patches, patches))
        for name, arm in runtime_arms.items():
            net, proj = arm["model"], arm["projector"]
            mask = prepared.target_masks[name][step]
            predictive, target, counts = prediction_resident(
                net,
                view_a,
                xyz,
                patches,
                mask,
                prepared.target_indices[name][step],
                precision=precision,
            )
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=precision == "bf16"):
                tokens = _encode_prevalidated(net.view_encoder, both, both_valid).reshape(
                    2 * settings.batch_size, blocks, 33, settings.embed_dim
                )
                projected = proj(authorized_pool(tokens.float(), both_valid, GAIT_JOINTS))
            # Covariance and variance reductions stay FP32, including their matmuls.
            regularizer = vicreg_loss(*projected.float().chunk(2))
            loss = predictive + settings.vicreg_weight * regularizer
            if device.type == "cpu" and not bool(torch.isfinite(loss).all()):
                raise FloatingPointError(f"Non-finite JEPA loss for {name}")
            arm["optimizer"].zero_grad(set_to_none=True)
            loss.backward()
            if any(p.grad is not None for p in net.target_encoder.parameters()):
                raise AssertionError("Teacher received gradients")
            torch.nn.utils.clip_grad_norm_(arm["trainable"], 1.0)
            arm["optimizer"].step()
            if device.type == "cuda":
                update_teacher_foreach(net, settings.ema_momentum)
            else:
                net.update_target(settings.ema_momentum)
            with torch.no_grad():
                if prepared.equal_counts[name][step]:
                    mean = target.mean(dim=(0, 1))
                else:
                    hidden = mask.flatten(1)
                    mean = ((target * hidden[..., None]).sum(1) / counts[:, None]).mean(0)
                net.target_center.mul_(0.9).add_(mean, alpha=0.1)
            arm["diagnostics"].append(
                torch.stack((loss.detach(), predictive.detach(), regularizer.detach()))
            )
            if step + 1 in checkpoints:
                arm["snapshots"][step + 1] = {
                    key: value.detach().cpu().clone()
                    for key, value in net.state_dict().items()
                }
        if progress is not None:
            progress({"step": step + 1, "total_steps": settings.steps, "encoders": len(runtime_arms)})
        if resume_path is not None and (
            (step + 1) % resume_interval == 0 or step + 1 == settings.steps
        ):
            _synchronize_device(device)
            _write_training_resume(
                resume_path,
                identity,
                step + 1,
                runtime_arms,
                earlier_elapsed + time.monotonic() - started,
            )
    _synchronize_device(device)
    elapsed = earlier_elapsed + time.monotonic() - started
    runs = {}
    for name, arm in runtime_arms.items():
        values = np.concatenate((
            arm["history_prefix"],
            _diagnostic_values(arm, label=name),
        ))
        if values.shape != (settings.steps, 3) or not np.isfinite(values).all():
            raise FloatingPointError(f"Training history is incomplete or non-finite for {name}")
        counts = prepared.target_counts[name]
        history = pd.DataFrame([
            {
                "step": step + 1,
                "loss": float(values[step, 0]),
                "masked_prediction_loss": float(values[step, 1]),
                "variance_regularizer": float(values[step, 2]),
                "hidden_count_min": int(min(counts[step])),
                "hidden_count_max": int(max(counts[step])),
            }
            for step in range(settings.steps)
        ])
        runs[name] = {
            "model": arm["model"].eval(),
            "initial_model": arm["initial_model"],
            "projector": arm["projector"].eval(),
            "settings": asdict(settings),
            "policy": asdict(arm["policy"]),
            "initial_state_digest": arm["initial_digest"],
            "initial_projector_digest": arm["initial_projector_digest"],
            "source_draw_digest": _digest_array(prepared.schedule),
            "history": history,
            "checkpoints": arm["snapshots"],
            "hidden_token_counts": prepared.target_counts[name],
            "coverage": prepared.coverage[name],
            "view_digest": prepared.view_digest,
            "elapsed_training_seconds": elapsed,
            "preparation_seconds": preparation_seconds,
            "resident_input_bytes": prepared.resident_bytes,
            "execution_layout": (
                "training-only resident fold tensors and masks; device-side shared views; "
                "fixed-shape target gathers; fused AdamW and foreach EMA on CUDA; buffered diagnostics"
            ),
        }
    pairing = {"same_initialization": len({r["initial_state_digest"] for r in runs.values()}) == 1,
               "same_projector_initialization": len({r["initial_projector_digest"] for r in runs.values()}) == 1,
               "same_source_draws": len({r["source_draw_digest"] for r in runs.values()}) == 1,
               "same_geometric_views": len({r["view_digest"] for r in runs.values()}) == 1,
               "same_hidden_counts": all(r["hidden_token_counts"] == next(iter(runs.values()))["hidden_token_counts"] for r in runs.values())}
    if not all(pairing.values()):
        raise AssertionError("Paired study controls failed")
    result = {"runs": runs, "pairing": pairing, "identity": identity,
              "source_schedule": prepared.schedule,
              "reflection_schedule": np.zeros(prepared.schedule.shape, dtype=bool),
              "synthetic": dataset.synthetic, "reused": False,
              "resumed_from_step": resumed_from_step,
              "tensor_cache_reused": prepared.tensor_cache_reused}
    if destination is not None:
        save_comparison(result, destination)
        if resume_path is not None:
            for path in (resume_path, _resume_checksum_path(resume_path)):
                path.unlink(missing_ok=True)
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
