"""Real GAVD workflow shared by tutorials 15--18.

The registered preparation and source splitter remain the authorities. This
module supplies missing input artifacts, explicit coverage audits, and a common
training/evaluation grid that can be reopened in a fresh notebook kernel.
"""
from __future__ import annotations

from dataclasses import asdict, replace
import hashlib
import json
import os
from pathlib import Path
import shutil
import tempfile

import numpy as np
import pandas as pd

from laterality.artifacts import atomic_write_json, sha256_file
from laterality.config import SUITE_ROOT, canonical_json_digest, load_context
from laterality.data import (
    _extraction_provenance_versions, cohort_paths, load_cohort, prepare_cohort, save_cohort,
)
from laterality.splitting import build_source_splits, load_splits, save_splits, split_path
from .comparative_masks import motion_scores
from .comparative_training import _load_tables, _save_tables, load_comparison
from .masked_learning import (LearningSettings, configure_learning_runtime, load_learning_dataset,
                              learning_hardware_report)
from .motion_structured_masks import StudyArm, context_cue_audit, paired_study_masks, study_arms
from .motion_structured_training import mask_study_identity, plan_mask_study, study_digest, train_mask_study
from .motion_readout import aggregate_motion_study, evaluate_motion_readouts, motion_readout_identity
from .motion_readout import prepare_motion_evaluation_inputs
from .motion_runtime import motion_numerical_context

FOLDS = tuple(range(5))
SEEDS = tuple(range(42, 47))
EVALUATION_TABLES = ("predictions", "selection", "diagnostics", "predictor_diagnostics")


def _inventory_digest(names):
    return hashlib.sha256(("\n".join(sorted(names)) + "\n").encode()).hexdigest()


def locked_pose_view(context, *, log=print):
    """Recover only an EXACT protocol inventory from an expanded local cache.

    Extraction generation is only a candidate selector. The registered count
    AND inventory SHA must match before any files are copied. The normal loader
    subsequently checks annotations, per-archive metadata and provenance counts.
    No source files or protocol settings are changed.
    """
    config = context.protocol["data"]
    contract = config["inventory_contract"]
    paths = sorted(p for c in config["conditions"] for p in (context.pose_root / c).glob("*.npz"))
    names = [f"{p.parent.name}/{p.stem}" for p in paths]
    if len(paths) == contract["pose_archive_count"] and _inventory_digest(names) == contract["pose_inventory_sha256"]:
        return context.pose_root
    allowed = set(config["extraction_provenance"]["extraction_version_counts"])
    selected, excluded = [], []
    for path in paths:
        with np.load(path, allow_pickle=False) as archive:
            try:
                _, generation = _extraction_provenance_versions(archive, archive_path=path)
            except (ValueError, KeyError) as error:
                excluded.append({"path": str(path), "reason": str(error)})
                continue
        if generation in allowed:
            selected.append(path)
        else:
            excluded.append({"path": str(path), "reason": f"Unregistered extraction generation: {generation}"})
    selected_names = [f"{p.parent.name}/{p.stem}" for p in selected]
    if (len(selected) != contract["pose_archive_count"]
            or _inventory_digest(selected_names) != contract["pose_inventory_sha256"]):
        raise RuntimeError(
            f"Cannot recover the locked GAVD inventory from {context.pose_root}: "
            f"found {len(paths)} archives, {len(selected)} with registered provenance; "
            f"need {contract['pose_archive_count']} with SHA {contract['pose_inventory_sha256']}. "
            "Restore the registered pose cache described in notebooks 01--02. No synthetic fallback was used.")
    destination = context.artifact_root / "inputs" / f"poses_{contract['pose_inventory_sha256'][:12]}"
    files = {f"{p.parent.name}/{p.name}": sha256_file(p) for p in selected}
    if destination.exists():
        manifest = json.loads((destination / "inventory.json").read_text(encoding="utf-8"))
        if manifest["files"] != files:
            raise RuntimeError("Locked pose view no longer matches the source bytes; preserve and inspect it")
        for name, digest in files.items():
            if sha256_file(destination / name) != digest:
                raise RuntimeError(f"Corrupt locked pose copy: {name}")
        return destination
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=".poses_", dir=destination.parent))
    # On interruption leave the uniquely named staging directory for inspection.
    for path in selected:
        relative = f"{path.parent.name}/{path.name}"
        target = temporary / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, target)
        if sha256_file(target) != files[relative]:
            raise RuntimeError(f"Pose copy did not verify: {relative}")
    atomic_write_json(temporary / "inventory.json", {
        "schema": "motion_gavd_locked_pose_view/v1", "source_root": str(context.pose_root),
        "inventory_sha256": contract["pose_inventory_sha256"], "files": files,
        "excluded_from_registered_inventory": excluded,
    })
    os.replace(temporary, destination)
    if log:
        log(f"Verified copies of {len(selected)}/{len(paths)} archives: {destination}")
    return destination


def prepare_gavd_inputs(*, create_missing=True, log=print):
    """Reuse checked artifacts, or perform exactly the preparation in 01 and 02."""
    context = load_context(profile="paper")
    existing = [p.exists() for p in cohort_paths(context.artifact_root)]
    if any(existing) and not all(existing):
        raise RuntimeError(f"Partial cohort at {context.artifact_root}; inspect it before rebuilding. Nothing overwritten.")
    created = []
    if not all(existing):
        if not create_missing:
            raise FileNotFoundError(f"Prepare notebooks 01--02 first: {context.artifact_root}")
        if split_path(context.artifact_root).exists():
            raise RuntimeError("A split manifest exists without its cohort; inspect the existing artifacts first")
        pose_root = locked_pose_view(context, log=log)
        if log:
            log("Applying Notebook 01 QC, validity, normalization and coordinate-derived target rules...")
        cohort = prepare_cohort(replace(context, pose_root=pose_root))
        save_cohort(context, cohort)
        created.append("cohort")
    cohort = load_cohort(context)
    if not split_path(context.artifact_root).exists():
        if not create_missing:
            raise FileNotFoundError(f"Run Notebook 02: {split_path(context.artifact_root)}")
        specification = context.protocol["splits"]
        splits = build_source_splits(cohort.table, context.protocol["data"]["conditions"],
            int(specification["outer_folds"]), int(specification["inner_folds"]), int(specification["seed"]))
        save_splits(context, cohort, splits)
        created.append("source_splits")
    splits = load_splits(context, cohort)
    if log:
        log(f"REAL GAVD: {len(cohort.table)} clips, {cohort.table.video_id.nunique()} videos; "
            f"{', '.join(created) + ' created' if created else 'verified existing cohort and splits'}")
    return context, cohort, splits


def study_inputs(*, mode="gavd", folds=FOLDS, seeds=SEEDS, create_missing=True, log=print):
    """Declare all data roles once; seeds repeat the same video-disjoint folds."""
    if mode not in {"gavd", "synthetic"}:
        raise ValueError("DATA_MODE must be 'gavd' or explicit 'synthetic'")
    folds, seeds = tuple(folds), tuple(seeds)
    if (not folds or not seeds or len(set(folds)) != len(folds) or len(set(seeds)) != len(seeds)
            or not set(folds) <= set(FOLDS) or not set(seeds) <= set(SEEDS)):
        raise ValueError("Declare unique folds 0--4 and seeds 42--46")
    context = cohort = splits = None
    if mode == "gavd":
        context, cohort, splits = prepare_gavd_inputs(create_missing=create_missing, log=log)
    datasets = {fold: load_learning_dataset(real=mode == "gavd", fold=fold) for fold in folds}
    reference = datasets[folds[0]]
    census, roles, expected = [], [], []
    for fold, dataset in datasets.items():
        dataset.validate()
        if (dataset.synthetic != (mode == "synthetic") or dataset.fold != fold
                or dataset.cohort_digest != reference.cohort_digest or dataset.split_digest != reference.split_digest
                or not np.array_equal(dataset.sequence_ids, reference.sequence_ids)):
            raise ValueError("Grid inputs disagree on fold, cohort, split or real/synthetic identity")
        test_rows = dataset.test_rows
        expected.append(pd.DataFrame({"sequence_id": dataset.sequence_ids[test_rows],
                                     "source_id": dataset.source_ids[test_rows], "fold": fold}))
        for seed in seeds:
            census.append({"fold": fold, "seed": seed, "train_clips": len(dataset.train_rows),
                "test_clips": len(test_rows), "train_sources": len(dataset.train_sources),
                "test_sources": len(dataset.test_sources), "source_overlap": 0, "synthetic": dataset.synthetic})
            roles.append(pd.DataFrame({"sequence_id": dataset.sequence_ids, "source_id": dataset.source_ids,
                "fold": fold, "seed": seed, "role": np.where(np.isin(np.arange(len(dataset.xyz)), test_rows), "test", "train")}))
    expected = pd.concat(expected, ignore_index=True)
    if expected.sequence_id.duplicated().any() or expected.groupby("source_id").fold.nunique().gt(1).any():
        raise ValueError("A sequence/source appears in more than one outer test fold")
    if set(folds) == set(FOLDS) and set(expected.sequence_id) != set(reference.sequence_ids):
        raise ValueError("Five-fold test coverage does not cover the cohort exactly once")
    return {"mode": mode, "folds": folds, "seeds": seeds, "datasets": datasets,
            "census": pd.DataFrame(census), "memberships": pd.concat(roles, ignore_index=True),
            "expected": expected, "context": context, "cohort": cohort, "splits": splits}


def audit_training_masks(inputs, *, experiments=("motion",), batch_size=20, log=print):
    """Audit every training clip for each fold/seed; never use test targets.

    These are deterministic inspection batches, not the source-balanced training
    schedule. Descriptive means weight videos equally within every fold/seed.
    """
    records, examples = [], {}
    reference = next(iter(inputs["datasets"].values()))
    reference_valid = reference.valid.reshape(len(reference.xyz), -1, 4, 33).all(2)
    # Scores depend on the clip, not its fold, seed or masking policy.
    score_bank = [motion_scores(xyz, valid, observation_valid=observed)[0]
                  for xyz, valid, observed in zip(reference.xyz, reference_valid, reference.valid)]
    for fold, data in inputs["datasets"].items():
        blocks = data.xyz.shape[1] // 4
        valid = data.valid.reshape(len(data.xyz), blocks, 4, 33).all(2)
        source_counts = pd.Series(data.source_ids[data.train_rows]).value_counts()
        for seed in inputs["seeds"]:
            if log:
                log(f"Mask audit: fold {fold}, seed {seed}, {len(data.train_rows)} training clips; {', '.join(experiments)}")
            settings = LearningSettings(fold=fold, seed=seed, batch_size=batch_size)
            for experiment in experiments:
                arms = study_arms(experiment, blocks)
                for start in range(0, len(data.train_rows), batch_size):
                    rows = data.train_rows[start:start + batch_size]
                    masks, metadata = paired_study_masks(data, rows, settings, arms, step=start)
                    for local, row in enumerate(rows):
                        scores = score_bank[row]
                        for name, values in masks.items():
                            mask = values[local]
                            if (mask & ~valid[row]).any() or not mask.any() or not (valid[row] & ~mask).any():
                                raise AssertionError("A mask violated target/context validity")
                            left, right = mask[:, [23, 25, 27, 29, 31]], mask[:, [24, 26, 28, 30, 32]]
                            record = {"experiment": experiment, "condition": name, "fold": fold, "seed": seed,
                                "sequence_id": str(data.sequence_ids[row]), "source_id": str(data.source_ids[row]),
                                "role": "train", "weight": 1 / source_counts[data.source_ids[row]],
                                "hidden_tokens": int(mask.sum()), "valid_tokens": int(valid[row].sum()),
                                "hidden_fraction": float(mask.sum() / valid[row].sum()),
                                "target_motion": float(scores[mask].mean()),
                                "eligible_motion": float(scores[valid[row]].mean()),
                                "left_leg_targets": int(left.sum()), "right_leg_targets": int(right.sum()),
                                "both_legs_targeted": bool(left.any() and right.any()),
                                **context_cue_audit(mask, valid[row])}
                            records.append(record)
                            examples.setdefault((experiment, name), {"mask": mask, "valid": valid[row],
                                "xyz": data.xyz[row], "sequence_id": str(data.sequence_ids[row]),
                                "source_id": str(data.source_ids[row]), "fold": fold, "seed": seed,
                                "metadata": metadata[name][local]})
    table = pd.DataFrame(records)
    keys = ["experiment", "fold", "seed", "sequence_id"]
    if table.groupby(keys).hidden_tokens.nunique().gt(1).any():
        raise AssertionError("The comparison did not match realized hidden counts")
    metrics = ("hidden_tokens", "hidden_fraction", "target_motion", "eligible_motion", "both_legs_targeted",
               "temporal_bracket_fraction", "visible_neighbor_fraction")
    summary = []
    for key, group in table.groupby(["experiment", "condition", "fold", "seed"], sort=False):
        summary.append(dict(zip(("experiment", "condition", "fold", "seed"), key),
            clips=len(group), sources=group.source_id.nunique(), hidden_min=int(group.hidden_tokens.min()),
            hidden_max=int(group.hidden_tokens.max()),
            **{metric: float(np.average(group[metric], weights=group.weight)) for metric in metrics}))
    return {"per_clip": table, "summary": pd.DataFrame(summary), "examples": examples}


def gavd_plan(inputs, *, experiments=("motion", "regions"), device="auto", output_dir=None,
              precision="fp32", resume_interval=100):
    """Use the tracked real recipe; tiny models require explicit synthetic mode."""
    plan = plan_mask_study(experiments=experiments, folds=inputs["folds"], seeds=inputs["seeds"],
                           device=device, output_dir=output_dir)
    plan["data_mode"] = inputs["mode"]
    plan["cohort_digest"] = next(iter(inputs["datasets"].values())).cohort_digest
    plan["split_digest"] = next(iter(inputs["datasets"].values())).split_digest
    if precision not in {"fp32", "bf16"}:
        raise ValueError("Motion precision must be fp32 or bf16")
    if isinstance(resume_interval, bool) or not isinstance(resume_interval, int) or resume_interval < 0:
        raise ValueError("resume_interval must be a nonnegative integer")
    plan["execution"] = {"precision": precision, "resume_interval": resume_interval}
    if inputs["mode"] == "synthetic":
        plan["settings"] = asdict(LearningSettings(steps=1, device="cpu"))
        plan["execution"] = {"precision": "fp32", "resume_interval": 0}
        plan["scope"] = "synthetic software check; no GAVD evidence"
        if output_dir is None:
            plan["output_dir"] = str(SUITE_ROOT / "artifacts/motion_structured_synthetic")
        blocks = next(iter(inputs["datasets"].values())).xyz.shape[1] // 4
        plan["arms"] = {e: {n: asdict(a) for n, a in study_arms(e, blocks).items()} for e in experiments}
        plan["workload"]["updates"] = 1
        plan["optimizer_updates"] = plan["training_runs"]
    return plan


def _jobs(plan, inputs):
    if (plan["study_implementation"] != study_digest() or plan["data_mode"] != inputs["mode"]
            or tuple(plan["folds"]) != inputs["folds"] or tuple(plan["seeds"]) != inputs["seeds"]):
        raise ValueError("Rebuild the plan: implementation or input declarations changed")
    reference = next(iter(inputs["datasets"].values()))
    if plan["cohort_digest"] != reference.cohort_digest or plan["split_digest"] != reference.split_digest:
        raise ValueError("Plan cohort/split identity does not match current inputs")
    base = LearningSettings(**plan["settings"])
    base.validate()
    expected = pd.DataFrame([{"experiment": e, "fold": f, "seed": s, "condition": c, "updates": base.steps}
        for e in plan["experiments"] for f in plan["folds"] for s in plan["seeds"] for c in plan["arms"][e]])
    if (not expected.equals(plan["workload"]) or len(expected) != plan["training_runs"]
            or plan["optimizer_updates"] != len(expected) * base.steps):
        raise ValueError("Workload does not match the declared grid")
    runtime = configure_learning_runtime(base.device)
    for fold, data in inputs["datasets"].items():
        for seed in inputs["seeds"]:
            settings = replace(base, fold=fold, seed=seed, device=runtime["device"], confirm_real_run=True)
            for experiment in plan["experiments"]:
                arms = {n: StudyArm(**a) for n, a in plan["arms"][experiment].items()}
                identity = mask_study_identity(data, settings, experiment, arms,
                    precision=plan.get("execution", {}).get("precision", "fp32"))
                path = Path(plan["output_dir"]) / experiment / canonical_json_digest(identity)
                yield {"experiment": experiment, "fold": fold, "seed": seed, "data": data,
                       "settings": settings, "arms": arms, "identity": identity, "path": path}


def grid_status(plan, inputs):
    rows = []
    for job in _jobs(plan, inputs):
        manifest = job["path"] / "manifest.json"
        state = "missing"
        if job["path"].exists():
            payload = json.loads(manifest.read_text(encoding="utf-8"))
            if canonical_json_digest(payload["identity"]) != canonical_json_digest(job["identity"]) or payload.get("complete") is not True:
                raise RuntimeError(f"Incomplete or incompatible training artifact: {job['path']}")
            names = set(job["arms"])
            expected_files = {"source_schedule.npy", "reflection_schedule.npy"}
            expected_files.update(f"{name}{suffix}" for name in names for suffix in (".pt", "_training.csv"))
            if set(payload.get("files", {})) != expected_files or set(payload.get("runs", {})) != names:
                raise RuntimeError(f"Missing declared training files: {job['path']}")
            for name, digest in payload["files"].items():
                if sha256_file(job["path"] / name) != digest:
                    raise RuntimeError(f"Corrupt training artifact: {job['path'] / name}")
            state = "complete; content hashes verified"
        rows.append({k: job[k] for k in ("experiment", "fold", "seed")} | {
            "encoders": len(job["arms"]), "training_status": state, "training_directory": str(job["path"])})
    return pd.DataFrame(rows)


def _evaluation(job, plan, *, result=None, input_cache=None):
    identity = motion_readout_identity(job["identity"], include_predictor=True)
    parent = Path(plan["output_dir"]) / "evaluations"
    cached = _load_tables(identity, parent, EVALUATION_TABLES)
    if cached is None:
        if result is None:
            result = load_comparison(job["path"], job["identity"])
        if canonical_json_digest(result["identity"]) != canonical_json_digest(job["identity"]):
            raise ValueError("In-memory encoders do not match this evaluation job")
        prepared = None if input_cache is None else input_cache.get("evaluation")
        if prepared is None:
            prepared = prepare_motion_evaluation_inputs(job["data"],
                segment_length=job["settings"].segment_length,
                device=next(next(iter(result["runs"].values()))["model"].parameters()).device)
            if input_cache is not None:
                input_cache["evaluation"] = prepared
        with motion_numerical_context():
            evaluation = evaluate_motion_readouts(result, job["data"], job["settings"],
                include_predictor=True, prepared_inputs=prepared)
        tables = {name: evaluation[name].assign(experiment=job["experiment"], fold=job["fold"], seed=job["seed"])
                  for name in EVALUATION_TABLES}
        _save_tables(tables, identity, parent)
        cached = _load_tables(identity, parent, EVALUATION_TABLES)
    return cached[0], str(cached[1]), identity


def collect_gavd_grid(plan, inputs, *, log=print):
    """Notebook 18 entry point. Load/evaluate saved encoders; NEVER train one.

    An incomplete grid gets a status table and no pooled 'full grid' score.
    Current evaluation identities are recomputed, not trusted from saved paths.
    """
    status = grid_status(plan, inputs)
    if status.training_status.eq("missing").any():
        return {"status": "Incomplete: run the missing jobs in Notebook 17", "jobs": status,
                "missing_jobs": int(status.training_status.eq("missing").sum()), "plan": plan}
    tables = {name: [] for name in EVALUATION_TABLES}
    index, identities = [], []
    input_cache, active_fold = {}, None
    for job in _jobs(plan, inputs):
        if active_fold != job["fold"]:
            input_cache.clear()
            active_fold = job["fold"]
        if log:
            log(f"Readouts: {job['experiment']}, fold {job['fold']}, seed {job['seed']}")
        evaluated, directory, identity = _evaluation(job, plan, input_cache=input_cache)
        for name in EVALUATION_TABLES:
            tables[name].append(evaluated[name])
        identities.append(identity)
        index.append({k: job[k] for k in ("experiment", "fold", "seed")} | {
            "training_directory": str(job["path"]), "evaluation_directory": directory,
            "training_digest": canonical_json_digest(job["identity"]),
            "evaluation_digest": canonical_json_digest(identity)})
    combined = {name: pd.concat(values, ignore_index=True) for name, values in tables.items()}
    identity = {"schema": "motion_gavd_grid/v1", "evaluations": identities, "scope": plan["scope"],
                "workflow_implementation": sha256_file(__file__),
                "folds": list(plan["folds"]), "seeds": list(plan["seeds"]), "mode": inputs["mode"]}
    report_names = (*EVALUATION_TABLES, "per_seed", "summary", "paired_intervals", "jobs", "census", "memberships")
    cached = _load_tables(identity, Path(plan["output_dir"]) / "grids", report_names)
    if cached is not None:
        return {"status": "Complete", **cached[0], "directory": str(cached[1]), "plan": plan,
                "grid_cache_reused": True}
    scored = aggregate_motion_study(combined["predictions"], inputs["expected"], plan)
    report = {**combined, **scored, "jobs": pd.DataFrame(index), "census": inputs["census"],
              "memberships": inputs["memberships"]}
    destination = _save_tables(report, identity, Path(plan["output_dir"]) / "grids")
    return {"status": "Complete", **report, "directory": str(destination), "plan": plan}


def run_gavd_grid(plan, inputs, *, enabled=False, log=print):
    """Run every declared fold/seed/experiment; complete jobs are reusable."""
    if not enabled:
        return {"status": "Training disabled; GAVD inputs and full workload are ready",
                "jobs": grid_status(plan, inputs), "plan": plan}
    if inputs["mode"] == "gavd" and plan["settings"]["device"] == "auto":
        hardware = learning_hardware_report("auto")
        if hardware["nvidia_gpus"] and not hardware["cuda_available"]:
            raise RuntimeError("NVIDIA hardware is present but this kernel cannot use CUDA. "
                "Select the GAVD5 CUDA kernel, or explicitly set DEVICE='cpu' for an intentional CPU run.")
    train_cache, evaluation_cache, active_fold = {}, {}, None
    execution = plan.get("execution", {})
    for job in _jobs(plan, inputs):
        if active_fold != job["fold"]:
            train_cache.clear()
            evaluation_cache.clear()
            active_fold = job["fold"]
        if log:
            log(f"Train {job['experiment']}: fold {job['fold']}, seed {job['seed']} ({job['settings'].steps} updates)")
        def progress(update):
            if log and (update["step"] == 1 or update["step"] % 100 == 0 or update["step"] == update["total_steps"]):
                log(f"  paired update {update['step']}/{update['total_steps']}")
        trained = train_mask_study(job["data"], job["settings"], experiment=job["experiment"], arms=job["arms"],
                         output_dir=job["path"].parent, progress=progress, input_cache=train_cache,
                         resume_interval=execution.get("resume_interval", 100),
                         precision=execution.get("precision", "fp32"))
        if log:
            log("  Frozen readouts and predictor diagnostics (training-source ridge selection)...")
        _evaluation(job, plan, result=trained, input_cache=evaluation_cache)
        del trained
    return collect_gavd_grid(plan, inputs, log=log)


def readout_contrasts(per_seed):
    """Within-seed contrasts, after pooling all declared outer test folds."""
    rows = []
    for (experiment, condition, seed), group in per_seed.groupby(["experiment", "condition", "seed"]):
        indexed = group.set_index("representation")
        for label, first, reference in (
            ("trained over initial, mean", "pretrained_teacher__mean", "initial_online__mean"),
            ("trained over initial, motion", "pretrained_teacher__mean_motion", "initial_online__mean_motion"),
            ("motion over mean, trained", "pretrained_teacher__mean_motion", "pretrained_teacher__mean"),
            ("motion over mean, initial", "initial_online__mean_motion", "initial_online__mean"),
        ):
            rows.append({"experiment": experiment, "condition": condition, "seed": seed, "contrast": label,
                         "delta_r2": float(indexed.loc[first, "r2"] - indexed.loc[reference, "r2"]),
                         "delta_mae": float(indexed.loc[first, "mae"] - indexed.loc[reference, "mae"])})
    return pd.DataFrame(rows)
