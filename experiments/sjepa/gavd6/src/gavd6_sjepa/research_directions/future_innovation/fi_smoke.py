"""Synthetic, explicitly non-evidentiary cache -> nested fit -> report smoke run."""

from pathlib import Path

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


def synthetic_cache(root):
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
    sources = [f"synthetic-source-{index // 2:02d}" for index in range(50)]
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
            "window_id": cohort.window_id.iloc[:10],
            "donor_window_id": cohort.window_id.iloc[10:20].to_numpy(),
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
    x = rng.normal(size=(50, 16)).astype(np.float32)
    skeleton = rng.normal(size=(50, 32, 33, 4)).astype(np.float32)
    skeleton[..., 2] = rng.uniform(0.5, 1, (50, 32, 33))
    skeleton[..., 3] = 1
    motion = skeleton[..., :2].mean(axis=(1, 2))
    person = (
        x @ rng.normal(size=(16, 256))
        + motion @ rng.normal(size=(2, 256)) * 4
        + rng.normal(size=(50, 256)) * 0.1
    )
    background = x @ rng.normal(size=(16, 256)) + rng.normal(size=(50, 256)) * 0.1
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
    write_once_json(
        root / "qc/validity-summary.json",
        {
            "binding": audit_binding(root),
            "passed": True,
            "checks": {
                name: True
                for name in (
                    "teacher_stable",
                    "causal_leakage_absent",
                    "target_audit_complete",
                    "target_sensitivity",
                    "person_edit_consistency",
                    "data_contract_valid",
                    "target_variance_valid",
                )
            },
            "motion_to_background_change_ratio": 3.0,
            "person_edit_direction_fraction": 1.0,
            "artifacts": {},
        },
    )
    return root


def run_smoke(root):
    root = synthetic_cache(root)
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
