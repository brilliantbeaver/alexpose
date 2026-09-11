"""Synthetic, explicitly non-evidentiary cache -> nested fit -> report smoke run."""

from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from gavd6_sjepa.shared_infrastructure.artifact_io_operations import (
    atomic_write_dataframe_csv,
    sha256_file,
)

from .fi_cohort import assign_source_folds
from .fi_contracts import (
    ModelContract,
    initialize_run,
    read_json,
    save_npz,
    write_json,
    write_once_json,
)
from .fi_feature_cache import cache_binding
from .fi_nested_training import fit_outer_fold
from .fi_reporting import build_report, score_gate
from .fi_token_regions import fixed_projection
from .fi_validity_audits import audit_binding


def synthetic_cache(root, *, protocol="legacy-v1", windows=50):
    if windows != 50:
        raise ValueError("Experiment 0 fixtures must retain the 50-clip gate")
    root = Path(root).resolve()
    root.mkdir(parents=True, exist_ok=True)
    fixture = root / "synthetic-input.txt"
    fixture.write_text("Synthetic smoke inputs; not a GAVD or V-JEPA result.\n")
    config = ModelContract(
        width=8,
        updates=(1, 2),
        ridge_alphas=(1.0, 10.0),
        weight_decays=(0.1,),
        bootstrap_repetitions=32,
    )
    initialize_run(
        root,
        protocol=protocol,
        sequence_manifest=fixture,
        video_manifest=fixture,
        annotations=[fixture],
        pose_model=fixture,
        vjepa_root=root,
        checkpoint=fixture,
        change_reason="synthetic implementation test",
        synthetic=True,
        model=config,
    )
    rng = np.random.default_rng(260905)
    sources = [f"synthetic-source-{index // 2:02d}" for index in range(windows)]
    folds = assign_source_folds(sources)
    cohort = pd.DataFrame(
        [
            {
                "window_id": f"synthetic-window-{index:02d}",
                "sequence_id": f"sequence-{index}",
                "video_id": source,
                "outer_fold": folds[source],
                "horizon_seconds": 8 / 30,
                "eligibility_reason": "synthetic fixture only",
            }
            for index, source in enumerate(sources)
        ]
    )
    atomic_write_dataframe_csv(root / "manifests/gate-windows.csv", cohort)
    run = read_json(root / "config/run-contract.json")
    run["manifest_sha256"] = sha256_file(root / "manifests/gate-windows.csv")
    write_json(root / "config/run-contract.json", run)
    audit_windows = pd.DataFrame(
        {
            "window_id": cohort.window_id.iloc[:(3 if protocol == "direct-v2" else 10)],
            "donor_window_id": cohort.window_id.iloc[10:10 + (3 if protocol == "direct-v2" else 10)].to_numpy(),
        }
    )
    atomic_write_dataframe_csv(root / "manifests/audit-windows.csv", audit_windows)
    atomic_write_dataframe_csv(
        root / "manifests/exclusions.csv",
        pd.DataFrame(columns=["sequence_id", "stage", "reason"]),
    )
    write_once_json(
        root / "config/cohort-contract.json",
        {
            "run_contract_sha256": sha256_file(root / "config/run-contract.json"),
            "manifest_sha256": sha256_file(root / "manifests/gate-windows.csv"),
            "artifacts": {
                "manifests/audit-windows.csv": sha256_file(
                    root / "manifests/audit-windows.csv"
                )
            },
        },
    )
    projection = fixed_projection(768)
    np.save(root / "config/projection-256.npy", projection)
    write_once_json(
        root / "config/projection-contract.json",
        {"sha256": sha256_file(root / "config/projection-256.npy")},
    )
    write_once_json(
        root / "config/nuisance-schema.json",
        {
            "columns": [f"synthetic_{i}" for i in range(16)],
            "context_embedding_columns": 0,
        },
    )
    x = rng.normal(size=(windows, 16)).astype(np.float32)
    skeleton = rng.normal(size=(windows, 32, 33, 4)).astype(np.float32)
    skeleton[..., 2] = rng.uniform(0.5, 1, (windows, 32, 33))
    skeleton[..., 3] = 1
    motion = skeleton[..., :2].mean(axis=(1, 2))
    person = (
        x @ rng.normal(size=(16, 256))
        + motion @ rng.normal(size=(2, 256)) * 4
        + rng.normal(size=(windows, 256)) * 0.1
    )
    background = x @ rng.normal(size=(16, 256)) + rng.normal(size=(windows, 256)) * 0.1
    binding = cache_binding(root)
    entries = []
    for index, row in cohort.iterrows():
        path = root / "teacher-cache" / f"{row.window_id}.npz"
        save_npz(
            path,
            baseline=x[index],
            skeleton=skeleton[index],
            person=person[index].astype(np.float32),
            background=background[index].astype(np.float32),
            matching=x[index, :8],
            window_id=np.array(row.window_id),
            binding=np.array(binding),
        )
        receipt = {
            "window_id": row.window_id,
            "path": str(path),
            "binding": binding,
            "sha256": sha256_file(path),
            "geometry_max_abs": 0.0,
        }
        write_once_json(path.with_suffix(".json"), receipt)
        entries.append(receipt)
    atomic_write_dataframe_csv(
        root / "manifests/cache-index.csv", pd.DataFrame(entries)
    )
    write_once_json(
        root / "config/cache-contract.json",
        {
            "binding": binding,
            "index_sha256": sha256_file(root / "manifests/cache-index.csv"),
            "schema_sha256": sha256_file(root / "config/nuisance-schema.json"),
        },
    )
    if protocol == "direct-v2":
        synthetic_readiness(root)
    else:
        synthetic_audit(root)
    return root


def synthetic_readiness(root):
    """Explicitly fabricated boundary measurements for the direct protocol smoke."""
    from .fi_feature_cache import load_cache
    from .fi_readiness import readiness_checks
    root = Path(root)
    if read_json(root / "config/run-contract.json")["synthetic"] is not True:
        raise ValueError("Synthetic readiness requires a synthetic run")
    planned = pd.read_csv(root / "manifests/audit-windows.csv")
    stable = pd.DataFrame({"window_id": planned.window_id, "context_max_abs": 0.0,
                           "target_max_abs": 0.0, "cache_max_abs": 0.0, "passed": True})
    leakage = pd.DataFrame({"window_id": planned.window_id, "repeat_max_abs": 0.0,
                            "max_abs_difference": 0.0, "mean_abs_difference": 0.0,
                            "tolerance": 1e-6, "passed": True, "stable": True})
    artifacts = {}
    for name, table in (("teacher-stability", stable), ("causal-leakage", leakage)):
        path = f"qc/{name}.csv"
        atomic_write_dataframe_csv(root / path, table)
        artifacts[path] = sha256_file(root / path)
    cohort, cache = load_cache(root)
    checks = readiness_checks(root, cohort, cache, stable, leakage)
    write_json(root / "qc/readiness-summary.json", {"protocol": "direct-v2",
               "binding": audit_binding(root), "checks": checks, "passed": all(checks.values()),
               "pixel_selectivity_evaluated": False, "artifacts": artifacts})


def synthetic_audit(root, *, ratio=3.0, direction_count=10):
    """Fabricated audit rows for software checks; never modify a real run."""
    root = Path(root)
    if read_json(root / "config/run-contract.json")["synthetic"] is not True:
        raise ValueError("Synthetic audits require an explicitly synthetic run")
    planned = pd.read_csv(root / "manifests/audit-windows.csv")
    sensitivity = planned.copy()
    sensitivity["person_change"] = [ratio] * direction_count + [0.5] * (10 - direction_count)
    sensitivity["background_change"] = 1.0
    sensitivity["person_larger"] = sensitivity.person_change > sensitivity.background_change
    stability = pd.DataFrame({"window_id": planned.window_id, "context_max_abs": 0.0,
                              "target_max_abs": 0.0, "cache_max_abs": 0.0, "passed": True})
    leakage = pd.DataFrame({"window_id": planned.window_id, "repeat_max_abs": 0.0,
                            "max_abs_difference": 0.0, "mean_abs_difference": 0.0,
                            "tolerance": 1e-6, "passed": True, "stable": True})
    artifacts = {}
    for name, table in (("target-sensitivity", sensitivity), ("teacher-stability", stability),
                        ("causal-leakage", leakage)):
        path = f"qc/{name}.csv"
        atomic_write_dataframe_csv(root / path, table)
        artifacts[path] = sha256_file(root / path)
    for window in planned.window_id:
        path = f"qc/pixel-edit-contact-sheets/{window}.jpg"
        (root / path).parent.mkdir(parents=True, exist_ok=True)
        panel = np.zeros((40, 180, 3), dtype=np.uint8)
        cv2.putText(panel, "SYNTHETIC FIXTURE", (4, 25), cv2.FONT_HERSHEY_SIMPLEX,
                    0.45, (255, 255, 255), 1)
        if not cv2.imwrite(str(root / path), panel):
            raise OSError("Cannot save synthetic contact sheet")
        artifacts[path] = sha256_file(root / path)
    measured_ratio = float(np.median(sensitivity.person_change))
    direction = float(sensitivity.person_larger.mean())
    thresholds = read_json(root / "config/thresholds.json")
    checks = {"teacher_stable": True, "causal_leakage_absent": True,
              "target_audit_complete": True, "data_contract_valid": True,
              "target_variance_valid": True,
              "target_sensitivity": measured_ratio >= thresholds["motion_to_background_change_min"],
              "person_edit_consistency": direction >= thresholds["person_edit_direction_fraction_min"]}
    write_json(root / "qc/validity-summary.json", {
        "binding": audit_binding(root), "passed": all(checks.values()), "checks": checks,
        "motion_to_background_change_ratio": measured_ratio,
        "person_edit_direction_fraction": direction, "artifacts": artifacts})


def run_smoke(root, *, protocol="legacy-v1"):
    root = synthetic_cache(root, protocol=protocol)
    for fold in range(5):
        fit_outer_fold(root, fold, "cpu")
    score_gate(root)
    result = build_report(root)
    if (
        result["allow_adapter_training"]
        or result["allow_full_experiment"]
        or not result["synthetic"]
    ):
        raise AssertionError(
            "Synthetic smoke must never authorize scientific advancement"
        )
    return result
