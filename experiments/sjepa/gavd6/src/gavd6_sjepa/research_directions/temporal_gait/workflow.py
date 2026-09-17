"""Receipt-checked stage DAG. Test data remain unopened until an immutable lock."""
from pathlib import Path
import uuid
from .contracts import (atomic_json, read_json, write_once_json, freeze_run, verify_run,
                        receipt, verify_receipt, stage_lock, sha256_file, digest)
from .windows import WindowDataset, concatenate, prepare_bout


def plan_tasks(cfg, phase="pilot"):
    """Pure plan: no manifest, media, output or environment access."""
    if phase not in {"pilot", "develop", "confirm"}:
        raise ValueError("Unknown phase")
    selected = {"pilot": cfg.pilot_seeds, "develop": cfg.development_seeds, "confirm": cfg.seeds}[phase]
    rows = [dict(task_id=i, arm=arm, seed=seed, fold=0) for i, (arm, seed) in enumerate((a, s) for a in cfg.arms for s in cfg.seeds)]
    return [r for r in rows if r["seed"] in selected]


def inventory_and_freeze(cfg):
    identity = freeze_run(cfg)
    if cfg.mode == "synthetic":
        from .fixtures import fixture_bout
        videos, bouts = [], []
        for role, sources in (("train", range(4)), ("development", range(4, 7)), ("test", range(7, 10))):
            for source in sources:
                _, b = fixture_bout(source, 0, role)
                videos.append({"video_id": b["video_id"], "group_id": b["group_id"], "role": role, "reason": "explicit_contract_fixture"})
                for j in range(2):
                    _, bout = fixture_bout(source, j, role)
                    bouts.append({**bout, "eligible": True})
        report = {"mode": "synthetic", "videos": videos, "bouts": bouts, "test_status": "software_fixture_only", "media_access": "none"}
    else:
        from .manifests import inventory
        report = inventory(cfg)
        report["mode"] = cfg.mode
    path = cfg.root / "reports/inventory.json"
    write_once_json(path, report)
    grid = cfg.root / "manifests/task-grid.json"
    write_once_json(grid, {"schema_version": 1, "resolved_config": cfg.scientific_dict(), "tasks": plan_tasks(cfg, "confirm")})
    return receipt(cfg, "00", [path, grid], details={"identity": identity, "media_opened": False})


def _prepare_role(cfg, role):
    if role not in {"train", "development", "test"}:
        raise ValueError("Unsupported role; calibration uncertainty is not implemented")
    if role == "test":
        verify_test_open(cfg)
    # The generated inventory authorizes exact source roles and media paths.
    # Original manifest digests alone do not authenticate this derived roster.
    verify_receipt(cfg, "00")
    if cfg.mode == "synthetic":
        from .fixtures import dataset
        data, coverage = dataset(cfg, role)
        mutations = [{"status": "synthetic_fixture_tested_by_unit_tests"}]
    else:
        from .video import probe_pts, load_pose
        from .information_audit import assert_future_boundary
        inv = read_json(cfg.root / "reports/inventory.json")
        videos = {v["video_id"]: v for v in inv["videos"]}
        timing, shards, coverage, mutations = {}, [], [], []
        for bout in inv["bouts"]:
            if bout["role"] != role or not bout["eligible"]:
                continue
            video_id = bout["video_id"]
            if video_id not in timing:
                timing[video_id] = probe_pts(videos[video_id], cfg.ffprobe)
            pose = load_pose(inv["poses"][bout["sequence_id"]], bout, video_timing=timing[video_id], visibility_threshold=cfg.visibility_threshold)
            data_bout, report = prepare_bout(pose, cfg, bout)
            # Save per-bout shard BEFORE concatenation; no silent cohort/window cap.
            shard = cfg.root / "cache" / role / "bouts" / (digest(bout["sequence_id"])[:24] + ".npz")
            data_bout.save(shard)
            shards.append(data_bout)
            coverage.append(report)
            if len(data_bout):
                mutations.append({"sequence_id": bout["sequence_id"], **assert_future_boundary(pose, cfg, data_bout.records[0]["issue_time"])})
        data = concatenate(shards)
    target = cfg.root / "cache" / role / "windows.npz"
    data.save(target)
    reportpath = cfg.root / "reports" / f"coverage-{role}.json"
    atomic_json(reportpath, {"mode": cfg.mode, "role": role, "bouts": coverage,
                            "videos_used": len({r["video_id"] for r in data.records}),
                            "groups_used": len({r["group_id"] for r in data.records}), "unique_windows": len(data),
                            "available_seconds": sum(r["available_seconds"] for r in coverage),
                            "covered_context_seconds": sum(r["covered_context_seconds"] for r in coverage),
                            "mutations": mutations, "no_sequence_cap": True,
                            "memory_note": "This baseline assembles the role arrays in RAM. Provision RAM from measured window bytes; shards are retained."})
    return [target, target.with_suffix(".json"), reportpath]


def prepare_bouts(cfg, roles=("train", "development")):
    if tuple(roles) != ("train", "development"):
        raise ValueError("Public prepare never opens test/calibration")
    verify_receipt(cfg, "00")
    outputs = []
    for role in roles:
        outputs.extend(_prepare_role(cfg, role))
    return receipt(cfg, "01", outputs, roles=roles)


def load_role(cfg, role):
    if role not in {"train", "development", "test"}:
        raise ValueError("Unknown cached source partition")
    if role == "test":
        verify_test_open(cfg)
        verify_receipt(cfg, "07-data")
    else:
        # Bind every downstream reader to the original preparation, not only to
        # configuration or the later information-audit summary.
        verify_receipt(cfg, "01")
    data = WindowDataset.load(cfg.root / "cache" / role / "windows.npz")
    if cfg.cohort_scope == "historical_overlap":
        data = data.subset([i for i, r in enumerate(data.records) if r.get("historical_laterality_member")])
        if not len(data):
            raise ValueError("No audited historical-overlap windows in this role; do not substitute full cohort")
    return data


def audit_information(cfg):
    verify_receipt(cfg, "01")
    from .information_audit import audit_datasets
    from .evaluation import evaluate_baselines
    train, dev = load_role(cfg, "train"), load_role(cfg, "development")
    result = audit_datasets(train, dev, cfg)
    if result["measurement_gate"]:
        baseline = evaluate_baselines(cfg, train, dev, _fresh_output(cfg.root / "evaluation/E0"), seed=cfg.pilot_seeds[0])
        result["baseline"] = baseline
    path = cfg.root / "reports/information-audit.json"
    atomic_json(path, result)
    outputs = [path]
    if result.get("baseline"):
        outputs.extend(_artifact_paths(result["baseline"]))
    return receipt(cfg, "02", outputs, roles=("train", "development"), details=result)


def _artifact_paths(record):
    paths = []
    for key, value in record.items():
        if key.endswith("_path") and isinstance(value, str) and Path(value).is_file():
            paths.append(Path(value))
            # Model/readout metadata and numeric payload both needed for replay.
            for suffix in (".json", ".npz"):
                sibling = Path(value).with_suffix(suffix)
                if sibling.is_file() and sibling not in paths:
                    paths.append(sibling)
    return list(dict.fromkeys(paths))


def _fresh_output(path):
    """Preserve failed attempts; never search for substitute fits/checkpoints."""
    if path.exists() and any(path.iterdir()):
        return path.with_name(path.name + "-retry-" + uuid.uuid4().hex[:12])
    return path


def train_task(cfg, task_id, family, phase="pilot"):
    verify_receipt(cfg, "02")
    gate = read_json(cfg.root / "reports/information-audit.json")
    if not gate["measurement_gate"]:
        raise ValueError("Measurement gate stopped; do not train")
    tasks = {r["task_id"]: r for r in plan_tasks(cfg, phase)}
    if task_id not in tasks:
        raise ValueError("Task not in frozen phase grid")
    task = tasks[task_id]
    if (task["arm"].startswith("masked")) != (family == "masked"):
        raise ValueError("Task family mismatch")
    if phase != "pilot" and task["seed"] not in cfg.pilot_seeds:
        decision = read_json(cfg.root / "decisions/development.json")
        if decision.get("identity") != verify_run(cfg):
            raise ValueError("Stale development expansion decision")
        verify_receipt(cfg, "06-" + decision["phase"])
        if decision != read_json(cfg.root / "decisions" / (decision["phase"] + ".json")):
            raise ValueError("Expansion decision differs from receipted phase decision")
        if not decision.get("ready_for_expansion"):
            raise ValueError("Development stopped compute expansion")
    stage = f"task-{task_id:04d}"
    if (cfg.root / "receipts" / f"{stage}.json").is_file():
        prior = read_json(cfg.root / "receipts" / f"{stage}.json")
        if prior["status"] == "complete" or not cfg.resume_from:
            return verify_receipt(cfg, stage)
        if prior["identity"] != verify_run(cfg):
            raise ValueError("Incompatible interrupted task")
    from .training import train_condition
    output = cfg.root / "training" / f"task-{task_id:04d}"
    result = train_condition(cfg, task, load_role(cfg, "train"), output)
    atomic_json(output / "result.json", result)
    paths = [output / "result.json", *_artifact_paths(result)]
    paths.extend(p for p in output.iterdir() if p.is_file() and p.suffix in {".pt", ".json"})
    status = "complete" if result.get("status") == "trained" else "incomplete"
    return receipt(cfg, stage, list(dict.fromkeys(paths)), roles=("train",), status=status, details={"task": task, "result": result})


def evaluate_development(cfg, phase="pilot"):
    verify_receipt(cfg, "02")
    expected = plan_tasks(cfg, phase)
    missing = [t["task_id"] for t in expected if not (cfg.root / "receipts" / f"task-{t['task_id']:04d}.json").is_file()]
    if missing:
        incomplete = {"status": "incomplete", "missing_tasks": missing, "ready_for_expansion": False, "phase": phase, "mode": cfg.mode}
        atomic_json(cfg.root / "reports" / f"incomplete-{phase}.json", incomplete)
        return incomplete
    from .evaluation import evaluate_condition, compare_conditions
    train, dev = load_role(cfg, "train"), load_role(cfg, "development")
    records = []
    for task in expected:
        verified = verify_receipt(cfg, f"task-{task['task_id']:04d}")
        result = verified["details"]["result"]
        checkpoint = result.get("checkpoint_path", result.get("checkpoint"))
        if not checkpoint:
            raise ValueError("Training did not retain a checkpoint")
        record = evaluate_condition(cfg, task, train, dev, checkpoint, _fresh_output(cfg.root / "evaluation" / phase / f"task-{task['task_id']:04d}"))
        records.append(record)
    summary = compare_conditions(cfg, records, _fresh_output(cfg.root / "evaluation" / phase / "summary"), expected_tasks=expected)
    path = cfg.root / "evaluation" / phase / "conditions.json"
    atomic_json(path, records)
    decision = {**summary, "phase": phase, "mode": cfg.mode, "conditions_path": str(path),
                "identity": verify_run(cfg), "ready_for_expansion": bool(summary.get("ready_for_expansion", False) and cfg.mode == "real"),
                "scientific_status": "software_only" if cfg.mode == "synthetic" else "development_evaluated"}
    atomic_json(cfg.root / "decisions/development.json", decision)
    atomic_json(cfg.root / "decisions" / f"{phase}.json", decision)
    outputs = [path, cfg.root / "decisions" / f"{phase}.json"]
    outputs.extend(_artifact_paths(summary))
    for record in records:
        outputs.extend(_artifact_paths(record))
    return receipt(cfg, f"06-{phase}", list(dict.fromkeys(outputs)), roles=("train", "development"), details=decision)


def lock_and_calibrate(cfg, phase="confirm"):
    if cfg.mode == "real" and cfg.cohort_scope == "historical_overlap":
        raise ValueError("Historical-overlap study is development-only; no untouched test membership is possible")
    verify_receipt(cfg, f"06-{phase}")
    decision = read_json(cfg.root / "decisions" / f"{phase}.json")
    if cfg.mode == "real" and (phase != "confirm" or not decision.get("ready_for_expansion")):
        raise ValueError("Final lock requires successful frozen confirm-seed development decision; negative result remains development-only")
    conditions = read_json(decision["conditions_path"])
    bound = {}
    for record in conditions:
        for path in _artifact_paths(record):
            bound[str(path)] = sha256_file(path)
    lock = {"identity": verify_run(cfg), "mode": cfg.mode, "phase": phase, "selection": decision, "conditions": conditions,
            "artifacts": bound, "calibration": "not_required_no_uncertainty_model", "refit_policy": "none_all_decoders_fit_train_only",
            "primary": {"horizon": .5, "joints": list(range(25, 33)), "min_pairs": 3,
                        "seed_estimand": "mean_per_seed_source_error_then_equal_video; relative_gain=ratio_of_means",
                        "bootstrap": "same_complete_group_draws_across_methods_and_seeds",
                        "success": "point_relative_gain >= frozen_threshold AND gain_CI95_lower > 0",
                        "failure": "gain_CI95_upper <= 0", "otherwise": "inconclusive_or_subthreshold",
                        "threshold": cfg.min_relative_improvement, "multiplicity": "one_development_selected_JEPA_vs_one_selected_baseline; all_other_results_exploratory"}}
    write_once_json(cfg.root / "locks/analysis.json", lock)
    write_once_json(cfg.root / "locks/selection.json", decision)
    return receipt(cfg, "07-lock", [cfg.root / "locks/analysis.json", cfg.root / "locks/selection.json"], details={"calibration": "not_required", "test_opened": False})


def _verify_analysis_lock(cfg):
    # Authenticate the immutable lock itself, including for direct test readers;
    # a matching opening marker cannot authorize an edited selection document.
    verify_receipt(cfg, "07-lock")
    lock = read_json(cfg.root / "locks/analysis.json")
    if lock["identity"] != verify_run(cfg):
        raise ValueError("Frozen analysis identity mismatch")
    for path, expected in lock["artifacts"].items():
        if sha256_file(path) != expected:
            raise ValueError(f"Locked checkpoint/readout/predictions changed: {path}")
    return lock


def verify_test_open(cfg):
    _verify_analysis_lock(cfg)
    path = cfg.root / "test/opened.json"
    if not path.is_file():
        raise ValueError("Test-open receipt required before any test contents")
    opened = read_json(path)
    if (opened.get("analysis_sha256") != sha256_file(cfg.root / "locks/analysis.json")
            or opened.get("identity") != verify_run(cfg)
            or opened.get("status") != "opened_before_access" or opened.get("mode") != cfg.mode):
        raise ValueError("Stale or invalid test-open receipt")
    return opened


def evaluate_test(cfg):
    if cfg.mode == "real" and cfg.cohort_scope == "historical_overlap":
        raise ValueError("Historical-overlap run may not open untouched test data")
    verify_receipt(cfg, "07-lock")
    lock = _verify_analysis_lock(cfg)
    verify_receipt(cfg, "00")
    inv = read_json(cfg.root / "reports/inventory.json")
    if not any(v["role"] == "test" for v in inv["videos"]):
        raise ValueError("No untouched test roster; exploratory development only")
    opened = {"analysis_sha256": sha256_file(cfg.root / "locks/analysis.json"), "identity": verify_run(cfg),
              "mode": cfg.mode, "status": "opened_before_access"}
    write_once_json(cfg.root / "test/opened.json", opened)
    if (cfg.root / "receipts/07-data.json").is_file():
        data_receipt = verify_receipt(cfg, "07-data")
        outputs = [cfg.root / p for p in data_receipt["outputs"]]
    else:
        outputs = _prepare_role(cfg, "test")
        receipt(cfg, "07-data", outputs, roles=("test",))
    from .evaluation import evaluate_condition, summarize_locked_conditions
    data = load_role(cfg, "test")
    records = []
    for record in lock["conditions"]:
        if record["arm"] != lock["selection"]["selected_arm"]:
            continue
        task = {key: record[key] for key in ("task_id", "arm", "seed", "fold")}
        result = evaluate_condition(cfg, task, None, data, record["checkpoint_path"], _fresh_output(cfg.root / "test" / f"task-{task['task_id']:04d}"), fitted_path=record["fitted_path"])
        records.append(result)
        outputs.extend(_artifact_paths(result))
    atomic_json(cfg.root / "test/conditions.json", records)
    summary = summarize_locked_conditions(cfg, records, lock["selection"], _fresh_output(cfg.root / "test/summary"))
    outputs.extend(_artifact_paths(summary))
    outputs += [cfg.root / "test/opened.json", cfg.root / "test/conditions.json"]
    return receipt(cfg, "07-test", list(dict.fromkeys(outputs)), roles=("test",), details={"status": "evaluated_with_frozen_fits", "test_selection": False, "summary": summary})


def aggregate_and_audit_claims(cfg, phase="pilot"):
    verify_receipt(cfg, f"06-{phase}")
    decision = read_json(cfg.root / "decisions" / f"{phase}.json")
    tested = (cfg.root / "receipts/07-test.json").is_file()
    if tested:
        final = verify_receipt(cfg, "07-test")["details"]["summary"]
    report = {"mode": cfg.mode, "phase": phase, "development": decision,
              "status": "software_verified_only" if cfg.mode == "synthetic" else ("test_predictions_retained" if tested else "development_evaluated"),
              "real_runs_complete": cfg.mode == "real", "confirmatory_claim": False,
              "limitation": "No manuscript is edited automatically. Final evidence requires independent review of saved predictions/locked contrast.",
              "historical_result": "Retained trained-teacher deficit and stopped latent-laterality confirmation remain unchanged."}
    if tested:
        report["locked_test"] = final
    path = cfg.root / "reports/claim-audit.json"
    atomic_json(path, report)
    return receipt(cfg, "08", [path], details=report)


def run_stage(cfg, stage, task_id=None, role="development", phase="pilot"):
    cfg.validate()
    if role == "test" and stage not in {"test", "aggregate"}:
        raise ValueError("Only locked test stage may access test")
    lock_name = f"task-{task_id}" if stage in {"masked", "future"} else f"{stage}-{phase}"
    with stage_lock(cfg.root, lock_name):
        prior_stage = {"inventory": "00", "prepare": "01", "audit": "02", "evaluate": f"06-{phase}", "calibrate": "07-lock", "test": "07-test"}.get(stage)
        if prior_stage and (cfg.root / "receipts" / (prior_stage + ".json")).is_file():
            return verify_receipt(cfg, prior_stage)
        if stage == "inventory":
            return inventory_and_freeze(cfg)
        verify_run(cfg)
        if stage == "prepare":
            return prepare_bouts(cfg)
        if stage == "audit":
            return audit_information(cfg)
        if stage in {"masked", "future"}:
            return train_task(cfg, task_id, stage, phase)
        if stage in {"extensions", "cache-video"}:
            return {"status": "gated_not_implemented", "mode": cfg.mode, "reason": "E3 dense×deep and released-video transfer require a development-supported separate protocol; no substitute model or synthetic fallback."}
        if stage == "evaluate":
            return evaluate_development(cfg, phase)
        if stage == "calibrate":
            return lock_and_calibrate(cfg, phase)
        if stage == "test":
            return evaluate_test(cfg)
        if stage == "aggregate":
            return aggregate_and_audit_claims(cfg, phase)
        raise ValueError(f"Unknown stage: {stage}")
