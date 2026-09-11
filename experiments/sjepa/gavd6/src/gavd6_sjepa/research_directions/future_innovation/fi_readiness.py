"""Direct Experiment 0: input integrity and temporal boundary, without pixel selectivity."""

from pathlib import Path

import numpy as np
import pandas as pd

from gavd6_sjepa.shared_infrastructure.artifact_io_operations import atomic_write_dataframe_csv, sha256_file
from .fi_cohort import load_cohort
from .fi_contracts import DIRECT_PROTOCOL, check_run, load_model_contract, protocol_name, read_json, stable_key, write_once_json
from .fi_feature_cache import load_cache, load_window
from .fi_token_regions import pool_target
from .fi_validity_audits import audit_binding, future_pixel_leakage_test, target_variance_checks, _check_boolean_column


def readiness_checks(root, cohort, cache, stability, leakage):
    return {
        "data_contract_valid": True,
        "input_audit_complete": True,
        "teacher_stable": bool(stability.passed.all()),
        "causal_leakage_absent": bool(leakage.passed.all()),
        "target_variance_valid": all(target_variance_checks(
            cohort, cache, load_model_contract(root), targets=("person",)).values()),
    }


def run_readiness(root, adapter):
    root = Path(root)
    if protocol_name(check_run(root)) != DIRECT_PROTOCOL:
        raise ValueError("Readiness-only audit requires direct-v2")
    cohort = load_cohort(root, verify_artifacts=True)
    _, cache = load_cache(root)
    planned = pd.read_csv(root / "manifests/audit-windows.csv")
    projection = np.load(root / "config/projection-256.npy", allow_pickle=False)
    stability, leakage = [], []
    rows = cohort.set_index("window_id")
    for window in planned.window_id:
        video, _, boxes, _ = load_window(rows.loc[window])
        leak = future_pixel_leakage_test(adapter, video, np.random.default_rng(int(stable_key("leakage", window)[:16], 16)))
        leakage.append({"window_id": window, **leak})
        original = adapter.encode_full_target(video)
        repeat = adapter.encode_full_target(video)
        target = pool_target(original, boxes, allow_empty_background=True)[0] @ projection
        index = np.flatnonzero(cohort.window_id.to_numpy() == window)[0]
        target_error = float(np.max(np.abs(original - repeat)))
        cache_error = float(np.max(np.abs(target - cache["person"][index])))
        stability.append({"window_id": window, "context_max_abs": leak["repeat_max_abs"],
                          "target_max_abs": target_error, "cache_max_abs": cache_error,
                          "passed": bool(leak["stable"] and target_error <= 1e-6 and cache_error <= 1e-6)})
        for name, records in (("teacher-stability", stability), ("causal-leakage", leakage)):
            atomic_write_dataframe_csv(root / f"qc/{name}.csv", pd.DataFrame(records))
    checks = readiness_checks(root, cohort, cache, pd.DataFrame(stability), pd.DataFrame(leakage))
    write_once_json(root / "qc/readiness-summary.json", {
        "protocol": DIRECT_PROTOCOL, "binding": audit_binding(root), "passed": all(checks.values()),
        "checks": checks, "pixel_selectivity_evaluated": False,
        "artifacts": {name: sha256_file(root / name) for name in
                      ("qc/teacher-stability.csv", "qc/causal-leakage.csv")}})
    verify_readiness(root)
    return all(checks.values())


def verify_readiness(root):
    root = Path(root)
    if protocol_name(check_run(root)) != DIRECT_PROTOCOL:
        raise ValueError("Readiness record used by a different protocol")
    summary = read_json(root / "qc/readiness-summary.json")
    if not isinstance(summary, dict):
        raise ValueError("Readiness summary must be a JSON object")
    if summary.get("protocol") != DIRECT_PROTOCOL or summary.get("binding") != audit_binding(root):
        raise ValueError("Readiness lineage changed")
    files = {"qc/teacher-stability.csv", "qc/causal-leakage.csv"}
    if not isinstance(summary.get("artifacts"), dict) or set(summary["artifacts"]) != files:
        raise ValueError("Readiness record requires both integrity tables")
    for name, digest in summary["artifacts"].items():
        if sha256_file(root / name) != digest:
            raise ValueError(f"Readiness artifact changed: {name}")
    cohort = load_cohort(root, verify_artifacts=True)
    plan_path = "manifests/audit-windows.csv"
    if plan_path not in read_json(root / "config/cohort-contract.json")["artifacts"]:
        raise ValueError("Readiness plan is not frozen")
    planned = pd.read_csv(root / plan_path)
    count = read_json(root / "config/control-contract.json")["audit_count"]
    if len(planned) != count or not planned.window_id.is_unique or not set(planned.window_id) <= set(cohort.window_id):
        raise ValueError("Invalid readiness window plan")
    stable = pd.read_csv(root / "qc/teacher-stability.csv")
    leakage = pd.read_csv(root / "qc/causal-leakage.csv")
    for table in (stable, leakage):
        if len(table) != count or not table.window_id.is_unique or set(table.window_id) != set(planned.window_id):
            raise ValueError("Readiness omits or duplicates planned windows")
    stability_values = stable[["context_max_abs", "target_max_abs", "cache_max_abs"]].to_numpy()
    leakage_values = leakage[["repeat_max_abs", "max_abs_difference", "mean_abs_difference", "tolerance"]].to_numpy()
    if (not np.isfinite(stability_values).all() or not np.isfinite(leakage_values).all()
            or np.any(stability_values < 0) or np.any(leakage_values < 0)
            or np.any(leakage.mean_abs_difference > leakage.max_abs_difference)
            or not np.allclose(leakage.tolerance, np.maximum(1e-6, 2 * leakage.repeat_max_abs), rtol=0, atol=1e-15)):
        raise ValueError("Invalid readiness measurements")
    if not np.array_equal(stable.context_max_abs, leakage.set_index("window_id").loc[stable.window_id, "repeat_max_abs"]):
        raise ValueError("Readiness repeat measurements disagree")
    _check_boolean_column(stable, "passed", (stability_values <= 1e-6).all(axis=1))
    _check_boolean_column(leakage, "passed", leakage.max_abs_difference <= leakage.tolerance)
    _check_boolean_column(leakage, "stable", leakage.repeat_max_abs <= 1e-6)
    _, cache = load_cache(root)
    expected = readiness_checks(root, cohort, cache, stable, leakage)
    checks = summary.get("checks", {})
    if (not isinstance(checks, dict) or any(type(v) is not bool for v in checks.values())
            or checks != expected or type(summary.get("passed")) is not bool
            or summary["passed"] != all(expected.values()) or summary.get("pixel_selectivity_evaluated") is not False):
        raise ValueError("Readiness flags disagree with retained measurements")
    return summary
