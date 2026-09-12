"""Read saved direct-v2 fits without refitting, updating provenance, or scoring a gate.

Run only on trusted experiment artifacts (the ridge baseline is a joblib pickle).
Copied runs are supported: cache paths are resolved within the supplied run root.
No raw videos, teacher GPU, or original absolute HAIC paths are required.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import joblib
import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src"))

from gavd6_sjepa.research_directions.future_innovation.fi_contracts import (
    code_fingerprint, equal_source_weights, read_json,
)
from gavd6_sjepa.research_directions.future_innovation.fi_controls import controlled_history
from gavd6_sjepa.research_directions.future_innovation.fi_feature_cache import cache_binding
from gavd6_sjepa.research_directions.future_innovation.fi_residual_models import (
    SkeletonResidualHead, predict_head,
)
from gavd6_sjepa.shared_infrastructure.artifact_io_operations import sha256_file


def checked_file(root, relative, digest):
    path = (root / relative).resolve()
    if not path.is_relative_to(root) or sha256_file(path) != digest:
        raise ValueError(f"Artifact checksum/path mismatch: {relative}")
    return path


def row_basis(x):
    """Float64 SVD; rows span the actual training design, including dependencies."""
    values = torch.as_tensor(np.asarray(x), dtype=torch.float64)
    _, singular, vectors = torch.linalg.svd(values, full_matrices=False)
    tolerance = float(singular[0]) * max(values.shape) * np.finfo(np.float64).eps
    return vectors[singular > tolerance].numpy(), singular.numpy(), tolerance


def null_weights(weight, basis):
    weight = np.asarray(weight, dtype=np.float64)
    return weight - (weight @ basis.T) @ basis


def weighted_mse(values, weights, valid):
    return float(np.average(np.mean(values[:, valid] ** 2, axis=1), weights=weights))


def read_inputs(root):
    """Verify relevant receipts directly; production load_cache records provenance."""
    run = read_json(root / "config/run-contract.json")
    if run["protocol"] != "direct-v2":
        raise ValueError("This diagnostic covers direct-v2 person-target fits only")
    for name, digest in run["config_sha256"].items():
        checked_file(root, f"config/{name}", digest)
    cohort_contract = read_json(root / "config/cohort-contract.json")
    checked_file(root, "config/run-contract.json", cohort_contract["run_contract_sha256"])
    checked_file(root, "manifests/gate-windows.csv", cohort_contract["manifest_sha256"])
    cache_contract = read_json(root / "config/cache-contract.json")
    binding = cache_binding(root)
    if binding != cache_contract["binding"]:
        raise ValueError("Cache binding mismatch")
    checked_file(root, "manifests/cache-index.csv", cache_contract["index_sha256"])
    checked_file(root, "config/nuisance-schema.json", cache_contract["schema_sha256"])
    checked_file(root, "config/projection-256.npy",
                 read_json(root / "config/projection-contract.json")["sha256"])
    cohort = pd.read_csv(root / "manifests/gate-windows.csv")
    index = pd.read_csv(root / "manifests/cache-index.csv")
    if (not cohort.window_id.is_unique or not index.window_id.is_unique
            or set(index.window_id) != set(cohort.window_id)):
        raise ValueError("Duplicated or missing cohort/cache windows")
    index = index.set_index("window_id")
    arrays = {name: [] for name in ("baseline", "skeleton", "person", "matching")}
    for window in cohort.window_id:
        receipt = index.loc[window]
        path = checked_file(root, f"teacher-cache/{window}.npz", receipt.sha256)
        if Path(receipt.path).name != path.name or receipt.binding != binding:
            raise ValueError("Cache receipt identity mismatch")
        with np.load(path, allow_pickle=False) as saved:
            if str(saved["window_id"]) != window or str(saved["binding"]) != binding:
                raise ValueError("Cache payload identity mismatch")
            for name in arrays:
                value = saved[name]
                if not np.isfinite(value).all():
                    raise ValueError(f"Non-finite cached {name}")
                arrays[name].append(value)
    return run, cohort, {key: np.stack(value) for key, value in arrays.items()}


def diagnose_fold(root, fold, cohort, arrays):
    directory = root / "models" / f"fold-{fold}"
    receipt_path = directory / "fold-complete.json"
    receipt = read_json(receipt_path)
    checked_file(root, "config/cache-contract.json", receipt["cache_contract_sha256"])
    checked_file(root, "qc/readiness-summary.json", receipt["validity_sha256"])
    for relative, digest in receipt["artifacts"].items():
        checked_file(root, relative, digest)

    def artifact(relative):
        key = str(relative.relative_to(root))
        if key not in receipt["artifacts"]:
            raise ValueError(f"Required file absent from fit receipt: {key}")
        return relative

    train = np.flatnonzero(cohort.outer_fold.to_numpy() != fold)
    test = np.flatnonzero(cohort.outer_fold.to_numpy() == fold)
    if not len(train) or not len(test) or (set(cohort.iloc[train].video_id)
                                         & set(cohort.iloc[test].video_id)):
        raise ValueError("Empty or overlapping source partitions")
    split = read_json(artifact(directory / "split-audit.json"))
    if (split["outer_train"] != cohort.iloc[train].window_id.tolist()
            or split["outer_test"] != cohort.iloc[test].window_id.tolist()):
        raise ValueError("Saved split differs from cohort order")
    baseline_path = artifact(directory / "person-baseline.joblib")
    baseline = joblib.load(baseline_path)
    if list(baseline.x_scaler.training_window_ids) != split["outer_train"]:
        raise ValueError("Baseline scaler differs from training partition")
    x_train = baseline.x_scaler.transform(arrays["baseline"][train])
    x_test = baseline.x_scaler.transform(arrays["baseline"][test])
    y = baseline.y_scaler.transform(arrays["person"][test])
    base = baseline.predict(arrays["baseline"][test])
    residual = y - base
    training_residual = (baseline.y_scaler.transform(arrays["person"][train])
                         - baseline.predict(arrays["baseline"][train]))
    valid = baseline.valid_features
    weights = equal_source_weights(cohort.iloc[test].video_id)
    train_weights = equal_source_weights(cohort.iloc[train].video_id)
    basis, singular, tolerance = row_basis(x_train)
    predictions = pd.read_parquet(artifact(root / f"predictions/fold-{fold}.parquet"))
    models = []
    config = read_json(root / "config/model-contract.json")
    for arm in ("real-skeleton", "time-shuffle", "clip-mismatch", "no-skeleton"):
        part = cohort.iloc[test]
        sk, _ = controlled_history(arm, arrays["skeleton"][test], part.window_id,
                                   part.video_id, arrays["matching"][test])
        for seed in config["seeds"]:
            model_dir = directory / f"{arm}-seed{seed}"
            checkpoint = artifact(model_dir / "residual-head.pt")
            saved = torch.load(checkpoint, map_location="cpu", weights_only=True)
            checked_file(root, str(baseline_path.relative_to(root)), saved["baseline_sha256"])
            checked_file(root, "config/cache-contract.json", saved["cache_sha256"])
            if (saved["training_window_ids"] != split["outer_train"]
                    or saved["arm"] != arm or saved["seed"] != seed
                    or saved["target"] != "person"):
                raise ValueError("Checkpoint identity/training split mismatch")
            head = SkeletonResidualHead(132, saved["baseline_dim"],
                                        saved["width"], saved["target_dim"])
            head.load_state_dict(saved["state_dict"], strict=True)
            correction = predict_head(head, sk, x_test, "cpu").astype(np.float64)
            rows = predictions[(predictions.arm == arm) & (predictions.seed == seed)]
            if (len(rows) != len(test) * y.shape[1]
                    or rows.duplicated(["window_id", "target_feature"]).any()
                    or set(rows.target.astype(str)) != {"person"}):
                raise ValueError("Unexpected prediction rows")

            def matrix(column):
                return rows.pivot(index="window_id", columns="target_feature", values=column).loc[
                    part.window_id, np.arange(y.shape[1])].to_numpy()

            for observed, expected in ((matrix("y_true"), y),
                                       (matrix("y_pred_baseline"), base),
                                       (matrix("y_pred_full"), base + correction)):
                np.testing.assert_allclose(observed, expected, rtol=1e-4, atol=1e-5)
            w_x = head.output.weight.detach().numpy()[:, saved["width"]:]
            w_null = null_weights(w_x, basis)
            null_correction = x_test @ w_null.T
            x_correction = x_test @ w_x.T
            summary = read_json(artifact(model_dir / "fit-summary.json"))
            zero_inner = float(min(summary["ridge_inner_mse"]))
            chosen_inner = float(np.min(summary["inner_grid_mse"]))
            correction_mse = weighted_mse(correction, weights, valid)
            alignment_twice = float(2 * np.average(
                np.mean((residual * correction)[:, valid], axis=1), weights=weights))
            delta_mse = (weighted_mse(residual - correction, weights, valid)
                         - weighted_mse(residual, weights, valid))
            if not np.isclose(delta_mse, correction_mse - alignment_twice, atol=1e-9):
                raise ValueError("Error decomposition failed")
            models.append({
                "arm": arm, "seed": seed, "updates": saved["updates"],
                "weight_decay": saved["weight_decay"],
                "final_recorded_training_mse": summary["training_history"][-1]["training_mse"],
                "inner_zero_correction_mse": zero_inner,
                "inner_selected_head_mse": chosen_inner,
                "selected_head_worse_than_zero_inner": chosen_inner > zero_inner,
                "heldout_correction_mse": correction_mse,
                "heldout_twice_residual_alignment": alignment_twice,
                "heldout_added_mse": delta_mse,
                "heldout_null_correction_mse": weighted_mse(null_correction, weights, valid),
                "heldout_skeleton_plus_bias_mse": weighted_mse(correction - x_correction, weights, valid),
                "null_weight_fraction_squared": float(np.sum(w_null ** 2) / max(np.sum(w_x ** 2), 1e-30)),
                "null_removal_max_training_prediction_change": float(np.max(np.abs(x_train @ w_null.T))),
                "diagnostic_mse_after_null_removal": weighted_mse(
                    residual - correction + null_correction, weights, valid),
                "saved_prediction_max_abs_difference": float(np.max(np.abs(matrix("y_pred_full") - base - correction))),
            })
    zero_variance = baseline.x_scaler.variance <= 1e-16
    return {
        "fold": fold, "fold_receipt_sha256": sha256_file(receipt_path),
        "training_windows": len(train), "training_sources": len(set(cohort.iloc[train].video_id)),
        "baseline_features": x_train.shape[1], "training_design_rank": len(basis),
        "svd_tolerance": tolerance, "singular_values": singular.tolist(),
        "max_abs_scaled_train_x": float(np.max(np.abs(x_train))),
        "max_abs_scaled_test_x": float(np.max(np.abs(x_test))),
        "zero_variance_training_columns": int(zero_variance.sum()),
        "max_test_magnitude_in_zero_variance_columns": float(np.max(np.abs(x_test[:, zero_variance]))) if zero_variance.any() else 0.0,
        "baseline_training_residual_mse": weighted_mse(training_residual, train_weights, valid),
        "baseline_heldout_residual_mse": weighted_mse(residual, weights, valid),
        "models": models,
    }


def diagnose(root, folds=range(5)):
    root = Path(root).resolve()
    torch.set_num_threads(1)
    run, cohort, arrays = read_inputs(root)
    seal = root / "reports/final-report-contract.json"
    if seal.exists():
        for relative, digest in read_json(seal)["artifacts"].items():
            checked_file(root, relative, digest)
    return {
        "purpose": "Post hoc mechanism diagnosis; no new gate decision or fitted model",
        "run_root": str(root), "synthetic": run["synthetic"],
        "run_contract_sha256": sha256_file(root / "config/run-contract.json"),
        "initial_code_sha256": run["code_sha256"], "diagnostic_code_sha256": code_fingerprint(),
        "diagnostic_script_sha256": sha256_file(Path(__file__)),
        "diagnostic_runtime": {"python": sys.version, "torch": torch.__version__,
                               "numpy": np.__version__, "joblib": joblib.__version__},
        "report_seal_present_and_checked": seal.exists(),
        "metric": "Within-fold, source-weighted MSE in training-standardized units; not pooled gate R2",
        "folds": [diagnose_fold(root, fold, cohort, arrays) for fold in folds],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True,
                        help="New JSON file outside the experiment run")
    parser.add_argument("--fold", type=int, choices=range(5), action="append")
    args = parser.parse_args()
    output, root = args.output.resolve(), args.run_root.resolve()
    if output.is_relative_to(root) or output.exists():
        parser.error("Output must be a new file outside the experiment run")
    result = diagnose(root, sorted(set(args.fold)) if args.fold else range(5))
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x") as handle:
        json.dump(result, handle, indent=2, allow_nan=False)
        handle.write("\n")
    print(f"Read {len(result['folds'])} folds; wrote diagnosis to {output}")


if __name__ == "__main__":
    main()
