"""Forensic evaluation of cached direct-v2 results; never changes the input run.

Counterfactual scores are post hoc diagnostics, not a new gate or model selection.
Optional inner replays train temporary CPU heads in memory, using frozen choices.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from unittest.mock import patch

import joblib
import numpy as np
import pandas as pd
import torch

from diagnose_future_innovation_fits import (
    checked_file, read_inputs, row_basis, null_weights, weighted_mse,
)
from probe_future_innovation_residual_failure import ZeroOutputHead
from gavd6_sjepa.research_directions.future_innovation import fi_residual_models as residual_models
from gavd6_sjepa.research_directions.future_innovation.fi_contracts import (
    ModelContract, code_fingerprint, equal_source_weights, read_json,
    stable_key,
)
from gavd6_sjepa.research_directions.future_innovation.fi_controls import controlled_history
from gavd6_sjepa.research_directions.future_innovation.fi_metrics import (
    score_arrays, source_bootstrap_counts, source_error_sums, score_source_sums,
)
from gavd6_sjepa.research_directions.future_innovation.fi_reporting import assemble_oof
from gavd6_sjepa.research_directions.future_innovation.fi_residual_models import (
    SkeletonResidualHead, fit_baseline, predict_head, train_head,
)
from gavd6_sjepa.shared_infrastructure.artifact_io_operations import sha256_file

ARMS = ("real-skeleton", "time-shuffle", "clip-mismatch", "no-skeleton")


def unstable_columns(scaler, validation_x):
    """Identify unobserved training variation, separately from ordinary large z scores."""
    constant = scaler.variance <= 1e-16
    scaled = scaler.transform(validation_x)
    return np.flatnonzero(constant & (np.max(np.abs(scaled), axis=0) > 1e6))


def inner_diagnostics(root, fold, cohort, arrays, config, replay):
    folder = root / f"models/fold-{fold}"
    audit = read_json(folder / "split-audit.json")
    records = pd.read_parquet(folder / "inner-selection.parquet")
    schema = read_json(root / "config/nuisance-schema.json")
    names = [f"context_embedding_{i}" for i in range(schema["context_embedding_columns"])] + schema["columns"]
    index = {w: i for i, w in enumerate(cohort.window_id)}
    alpha = read_json(folder / "real-skeleton-seed7/fit-summary.json")["ridge_alpha"]
    results = []
    for split in audit["inner"]:
        fit = np.array([index[w] for w in split["train"]])
        validate = np.array([index[w] for w in split["validation"]])
        baseline = fit_baseline(arrays["baseline"][fit], arrays["person"][fit],
                                cohort.iloc[fit].window_id, cohort.iloc[fit].video_id,
                                alpha, config.target_variance_tolerance)
        xf = baseline.x_scaler.transform(arrays["baseline"][fit])
        xv = baseline.x_scaler.transform(arrays["baseline"][validate])
        rf = baseline.y_scaler.transform(arrays["person"][fit]) - baseline.predict(arrays["baseline"][fit])
        rv = baseline.y_scaler.transform(arrays["person"][validate]) - baseline.predict(arrays["baseline"][validate])
        weights = equal_source_weights(cohort.iloc[validate].video_id)
        bad = unstable_columns(baseline.x_scaler, arrays["baseline"][validate])
        entry = {"outer_fold": fold, "inner_fold": split["fold"],
                 "train_windows": len(fit), "validation_windows": len(validate),
                 "baseline_train_residual_mse": weighted_mse(rf, equal_source_weights(cohort.iloc[fit].video_id), baseline.valid_features),
                 "baseline_validation_residual_mse": weighted_mse(rv, weights, baseline.valid_features),
                 "max_scaled_validation_input": float(np.max(np.abs(xv))),
                 "unstable_columns": []}
        for column in bad:
            affected = np.abs(xv[:, column]) > 1e6
            entry["unstable_columns"].append({
                "index": int(column), "name": names[column],
                "training_variance": float(baseline.x_scaler.variance[column]),
                "scale": float(baseline.x_scaler.scale[column]),
                "max_abs_scaled_value": float(np.max(np.abs(xv[:, column]))),
                "training_values": np.unique(arrays["baseline"][fit, column]).tolist(),
                "validation_values": np.unique(arrays["baseline"][validate, column]).tolist(),
                "affected_windows": cohort.iloc[validate[affected]].window_id.tolist(),
                "affected_sources": sorted(set(cohort.iloc[validate[affected]].video_id)),
            })
        if replay and len(bad):
            # Match the production weighting, including its normalization for pooled selection.
            raw_weights = weights * cohort.iloc[validate].video_id.nunique() / len(validate)
            head, history = train_head(
                arrays["skeleton"][fit], xf, rf, cohort.iloc[fit].video_id.to_numpy(),
                baseline.valid_features, seed=7, weight_decay=0.1,
                updates=config.updates, model_contract=config,
                validation={"skeleton": arrays["skeleton"][validate], "x": xv,
                            "residual": rv, "weights": raw_weights})
            pred = predict_head(head, arrays["skeleton"][validate], xv, "cpu")
            neutral = xv.copy()
            neutral[:, bad] = 0
            neutral_prediction = predict_head(head, arrays["skeleton"][validate], neutral, "cpu")
            original = records[(records.arm == "real-skeleton") & (records.seed == 7)
                               & (records.inner_fold == split["fold"]) & (records.weight_decay == 0.1)]
            relative_differences = []
            for record in history:
                saved = original[original.updates == record["updates"]].iloc[0]
                relative_differences.append(abs(record["validation_squared_error"] - saved.validation_squared_error)
                                            / max(abs(saved.validation_squared_error), 1e-30))
            entry["replay"] = {
                "arm": "real-skeleton", "seed": 7, "updates": max(config.updates),
                "weight_decay": 0.1, "max_relative_difference_from_saved_validation_sse": max(relative_differences),
                "original_validation_mse": weighted_mse(rv - pred, weights, baseline.valid_features),
                "neutralized_constant_columns_validation_mse": weighted_mse(rv - neutral_prediction, weights, baseline.valid_features),
                "validation_history": history,
            }
        results.append(entry)
    return results


def evaluate(root, *, replay_inner=False, zero_init_outer=False):
    root = Path(root).resolve()
    torch.set_num_threads(1)
    run, cohort, cache = read_inputs(root)
    config = ModelContract(**read_json(root / "config/model-contract.json"))
    verified = 0
    for seal_name in ("final-report-contract.json", "scores-contract.json"):
        for relative, digest in read_json(root / "reports" / seal_name)["artifacts"].items():
            checked_file(root, relative, digest)
            verified += 1
    for fold in range(5):
        receipt = read_json(root / f"models/fold-{fold}/fold-complete.json")
        checked_file(root, "config/cache-contract.json", receipt["cache_contract_sha256"])
        for relative, digest in receipt["artifacts"].items():
            checked_file(root, relative, digest)
            verified += 1
    readiness = root / "qc/readiness-summary.json"
    readiness_checked = readiness.exists()
    if readiness_checked:
        for fold in range(5):
            checked_file(root, "qc/readiness-summary.json",
                         read_json(root / f"models/fold-{fold}/fold-complete.json")["validity_sha256"])
        for relative, digest in read_json(readiness)["artifacts"].items():
            checked_file(root, relative, digest)
            verified += 1
        expected_audit_binding = stable_key(*[sha256_file(root / p) for p in
                                             ("config/cache-contract.json", "config/cohort-contract.json")])
        if read_json(readiness)["binding"] != expected_audit_binding:
            raise ValueError("Readiness binding differs from cache/cohort")
        score_inputs = ["config/run-contract.json", "config/cache-contract.json", "qc/readiness-summary.json"]
        score_inputs += [f"models/fold-{fold}/fold-complete.json" for fold in range(5)]
        if read_json(root / "reports/scores-contract.json")["binding"] != stable_key(*[sha256_file(root / p) for p in score_inputs]):
            raise ValueError("Score binding differs from fitted artifacts")
    tables = pd.concat([pd.read_parquet(root / f"predictions/fold-{fold}.parquet") for fold in range(5)], ignore_index=True)
    saved_arrays = assemble_oof(cohort, cache, tables, config, arms=ARMS)
    weights = equal_source_weights(cohort.video_id)
    score_rows, per_fit, inner_rows = [], [], []
    counterfactuals = {key: {name: np.zeros_like(values[0]) for name in
                            ("remove_x_null", "remove_x_branch", "x_branch_only", "linear_interpolation")}
                       for key, values in saved_arrays.items()}
    if zero_init_outer:
        for seed in config.seeds:
            counterfactuals[("real-skeleton", seed)]["zero_initialized_refit"] = np.zeros((len(cohort), 256))
    max_prediction_difference = 0.0
    selected_rows = []
    for fold in range(5):
        folder = root / f"models/fold-{fold}"
        train = np.flatnonzero(cohort.outer_fold.to_numpy() != fold)
        test = np.flatnonzero(cohort.outer_fold.to_numpy() == fold)
        baseline_path = folder / "person-baseline.joblib"
        baseline = joblib.load(baseline_path)
        xtrain = baseline.x_scaler.transform(cache["baseline"][train])
        xtest = baseline.x_scaler.transform(cache["baseline"][test])
        residual = baseline.y_scaler.transform(cache["person"][train]) - baseline.predict(cache["baseline"][train])
        basis, _, _ = row_basis(xtrain)
        linear = np.linalg.lstsq(xtrain, residual, rcond=None)[0].T
        inner_table = pd.read_parquet(folder / "inner-selection.parquet")
        for arm in ARMS:
            part = cohort.iloc[test]
            sk, _ = controlled_history(arm, cache["skeleton"][test], part.window_id,
                                       part.video_id, cache["matching"][test])
            for seed in config.seeds:
                directory = folder / f"{arm}-seed{seed}"
                summary = read_json(directory / "fit-summary.json")
                saved = torch.load(directory / "residual-head.pt", map_location="cpu", weights_only=True)
                if (saved["arm"] != arm or saved["seed"] != seed or saved["target"] != "person"
                        or saved["training_window_ids"] != cohort.iloc[train].window_id.tolist()
                        or saved["baseline_sha256"] != sha256_file(baseline_path)):
                    raise ValueError("Checkpoint identity or baseline mismatch")
                head = SkeletonResidualHead(132, saved["baseline_dim"], saved["width"], saved["target_dim"])
                head.load_state_dict(saved["state_dict"], strict=True)
                if not all(torch.isfinite(v).all() for v in saved["state_dict"].values()):
                    raise ValueError("Non-finite checkpoint weights")
                true, base, full, valid = saved_arrays[(arm, seed)]
                correction = predict_head(head, sk, xtest, "cpu")
                difference = float(np.max(np.abs(full[test] - base[test] - correction)))
                np.testing.assert_allclose(full[test] - base[test], correction, rtol=1e-4, atol=1e-5)
                max_prediction_difference = max(max_prediction_difference, difference)
                wx = head.output.weight.detach().numpy()[:, saved["width"]:]
                wn = null_weights(wx, basis)
                null_prediction = xtest @ wn.T
                x_prediction = xtest @ wx.T
                variants = counterfactuals[(arm, seed)]
                variants["remove_x_null"][test] = full[test] - null_prediction
                variants["remove_x_branch"][test] = full[test] - x_prediction
                variants["x_branch_only"][test] = base[test] + x_prediction
                variants["linear_interpolation"][test] = base[test] + xtest @ linear.T
                if zero_init_outer and arm == "real-skeleton":
                    # Fixed saved outer settings, without new selection: diagnostic only.
                    with patch.object(residual_models, "SkeletonResidualHead", ZeroOutputHead):
                        refit, _ = train_head(
                            cache["skeleton"][train], xtrain, residual,
                            cohort.iloc[train].video_id.to_numpy(), baseline.valid_features,
                            seed=seed, weight_decay=saved["weight_decay"],
                            updates=[saved["updates"]], model_contract=config)
                    variants["zero_initialized_refit"][test] = base[test] + predict_head(refit, sk, xtest, "cpu")
                part_weights = equal_source_weights(cohort.iloc[test].video_id)
                per_fit.append({
                    "fold": fold, "arm": arm, "seed": seed,
                    "training_rank": len(basis), "null_dimensions": xtest.shape[1] - len(basis),
                    "null_weight_fraction_squared": float(np.sum(wn**2) / np.sum(wx**2)),
                    "null_removal_max_training_change": float(np.max(np.abs(xtrain @ wn.T))),
                    "heldout_correction_mse": weighted_mse(correction, part_weights, valid),
                    "heldout_null_component_mse": weighted_mse(null_prediction, part_weights, valid),
                    "heldout_skeleton_plus_bias_mse": weighted_mse(correction - x_prediction, part_weights, valid),
                    "baseline_training_residual_mse": weighted_mse(residual, equal_source_weights(cohort.iloc[train].video_id), valid),
                    "linear_residual_training_mse": weighted_mse(residual - xtrain @ linear.T, equal_source_weights(cohort.iloc[train].video_id), valid),
                    "recorded_head_training_mse": summary["training_history"][-1]["training_mse"],
                })
                selected = inner_table[(inner_table.arm == arm) & (inner_table.seed == seed)]
                grid = selected.groupby(["weight_decay", "updates"])[["validation_squared_error", "validation_weight"]].sum()
                mse = (grid.validation_squared_error / grid.validation_weight).unstack("updates").loc[list(config.weight_decays), list(config.updates)].to_numpy()
                np.testing.assert_allclose(mse, summary["inner_grid_mse"], atol=1e-10, rtol=1e-12)
                chosen = np.unravel_index(np.argmin(mse), mse.shape)
                assert summary["selected_updates"] == config.updates[chosen[1]] == saved["updates"]
                assert summary["selected_weight_decay"] == config.weight_decays[chosen[0]] == saved["weight_decay"]
                ridge = min(summary["ridge_inner_mse"])
                selected_rows.append({"fold": fold, "arm": arm, "seed": seed,
                                      "ridge_alpha": summary["ridge_alpha"], "ridge_inner_mse": ridge,
                                      "selected_inner_mse": float(mse.min()),
                                      "candidates_worse_than_ridge": int((mse > ridge).sum()),
                                      "updates": saved["updates"], "weight_decay": saved["weight_decay"]})
        inner_rows.extend(inner_diagnostics(root, fold, cohort, cache, config, replay_inner))
    saved_aggregate = pd.read_csv(root / "reports/aggregate-metrics.csv").set_index(["arm", "seed"])
    sufficient = {}
    for key, (true, base, full, valid) in saved_arrays.items():
        actual = score_arrays(true, base, full, weights, valid)[0]
        for metric in ("r2_baseline", "r2_full", "delta_r2", "f8"):
            np.testing.assert_allclose(actual[metric], saved_aggregate.loc[key, metric], atol=1e-12)
        score_rows.append({"arm": key[0], "seed": key[1], "variant": "saved_full", **actual})
        for name, prediction in counterfactuals[key].items():
            score_rows.append({"arm": key[0], "seed": key[1], "variant": name,
                               **score_arrays(true, base, prediction, weights, valid)[0]})
        sufficient[key] = source_error_sums(true, base, full, weights, cohort.video_id)
    bootstrap_rows = []
    for draw, counts in enumerate(source_bootstrap_counts(cohort.video_id, config.bootstrap_repetitions)):
        for arm in ARMS:
            seed_scores = [score_source_sums(sufficient[(arm, seed)], counts, saved_arrays[(arm, seed)][3])
                           for seed in config.seeds]
            bootstrap_rows.append({"draw": draw, "arm": arm, **{
                metric: float(np.mean([s[metric] for s in seed_scores]))
                for metric in ("r2_baseline", "r2_full", "delta_r2", "f8")}})
    actual_bootstrap = pd.DataFrame(bootstrap_rows).set_index(["draw", "arm"]).sort_index()
    original_bootstrap = pd.read_csv(root / "reports/source-bootstrap.csv").set_index(["draw", "arm"]).sort_index()
    np.testing.assert_allclose(actual_bootstrap, original_bootstrap[actual_bootstrap.columns], rtol=1e-10, atol=1e-12)
    return {"run_id": run["run_id"], "synthetic": run["synthetic"],
            "scope": "Saved prediction verification and post hoc mechanisms; no amended gate",
            "run_contract_sha256": sha256_file(root / "config/run-contract.json"),
            "source_sha256": code_fingerprint(), "script_sha256": sha256_file(Path(__file__)),
            "torch_version": torch.__version__, "receipt_artifact_checks": verified,
            "readiness_receipt_checked": readiness_checked,
            "saved_prediction_max_abs_difference": max_prediction_difference,
            "oof_rows_checked": len(tables), "bootstrap_rows_recomputed": len(actual_bootstrap),
            "scores": score_rows, "fits": per_fit, "selection": selected_rows, "inner_partitions": inner_rows}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--replay-inner", action="store_true")
    parser.add_argument("--zero-init-outer", action="store_true",
                        help="Replay real outer heads with zero output initialization and saved settings")
    args = parser.parse_args()
    output, root = args.output.resolve(), args.run_root.resolve()
    if output.exists() or output.is_relative_to(root):
        parser.error("Choose a new output file outside the input run")
    result = evaluate(root, replay_inner=args.replay_inner, zero_init_outer=args.zero_init_outer)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x") as handle:
        json.dump(result, handle, indent=2, allow_nan=False)
        handle.write("\n")
    print(f"Verified {len(result['fits'])} fits and {result['bootstrap_rows_recomputed']} bootstrap rows: {output}")


if __name__ == "__main__":
    main()
