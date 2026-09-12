"""Versioned, CPU-only student-accessibility panel on verified cached features.

This is an exploratory estimand change, not a revision of the direct-v3 gate.
All numerical fitting, controls and scoring reuse its calibrated implementation.
"""
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
import json
import platform
import time

import joblib
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

from ..future_innovation.fi_cache_reuse import verify_reused_readiness
from ..future_innovation.fi_cohort import assign_source_folds
from ..future_innovation.fi_contracts import (
    DIRECT_ARMS, code_fingerprint, equal_source_weights, read_json, save_npz,
    sha256_file, stage_lock, write_json, write_once_json,
)
from ..future_innovation.fi_feature_cache import load_cache
from ..future_innovation.fi_joint_models import JointModelContract, JointRidge, SupportedInput, fit_joint_baseline, skeleton_schema, temporal_features
from ..future_innovation.fi_joint_training import nested_partition, predict_selected, choose_candidate, summarize, candidate_specs
from ..future_innovation.fi_metrics import score_arrays, source_error_sums, source_bootstrap_counts, score_source_sums
from ..future_innovation.fi_nested_training import isolated_split, partition_control
from ...shared_infrastructure.artifact_io_operations import atomic_save_joblib, atomic_write_dataframe_csv

VERSION = "student-accessibility-v1"
PANELS = ("support", "posture")
BOOTSTRAP_SEED = 260905
TOLERANCE = {"prediction_atol": 1e-10, "prediction_rtol": 1e-10,
             "score_atol": 1e-12, "score_rtol": 1e-10}


def accessibility_features(cohort, skeleton, panel):
    """Fixed X: 331 support/timing columns, optionally 66 endpoint coordinates.

    Confidence and validity use all raw frames, including confidence where a
    joint is invalid. Frame-31 coordinates are NaN when invalid. Coordinates
    already use the parent prefix-derived torso scale; C is therefore a
    prefix-normalized endpoint, not an independently measured current image.
    """
    if panel not in PANELS:
        raise ValueError("Unknown accessibility panel")
    h = np.asarray(skeleton, dtype=np.float64)
    temporal_features(h)  # Reuse raw schema/finiteness validation.
    fps = np.asarray(cohort.decoded_fps, dtype=np.float64)
    if len(cohort) != len(h) or fps.shape != (len(h),) or not np.isfinite(fps).all() or np.any(fps <= 0):
        raise ValueError("Invalid cohort timing or history length")
    blocks, names, kinds = [], [], []
    for a in (0, 8, 16, 24):
        blocks.append(h[:, a:a+8, :, 2:4].mean(axis=1).reshape(len(h), -1))
        for joint in range(33):
            for channel in ("confidence", "validity"):
                names.append(f"support/bin{a:02d}_{a+8:02d}/joint{joint:02d}/{channel}")
                kinds.append("fraction")
    blocks.append(h[:, 31, :, 2:4].reshape(len(h), -1))
    for joint in range(33):
        for channel in ("confidence", "validity"):
            names.append(f"endpoint/joint{joint:02d}/{channel}")
            kinds.append("fraction")
    blocks.append(fps[:, None]); names.append("decoded_fps"); kinds.append("continuous")
    if panel == "posture":
        xy = np.where(h[:, 31, :, 3:4].astype(bool), h[:, 31, :, :2], np.nan)
        blocks.append(xy.reshape(len(h), -1))
        for joint in range(33):
            for channel in ("x", "y"):
                names.append(f"endpoint/joint{joint:02d}/{channel}")
                kinds.append("coordinate")
    x = np.concatenate(blocks, axis=1)
    return x, (tuple(names), tuple(kinds))


def specification():
    return {
        "version": VERSION, "status": "exploratory_development; no ADVANCE decision",
        "panels": list(PANELS), "baseline_dimensions": {"support": 331, "posture": 397},
        "features": "four ordered eight-frame raw confidence/validity means; frame31 confidence/validity; decoded_fps; posture adds frame31 valid xy",
        "coordinate_units": "parent cached xy, per-frame hip centering and median torso scale from all32prefix frames",
        "primary_contrast": "posture panel real-skeleton minus no-skeleton",
        "secondary_contrasts": "real versus shared baseline, time-shuffle and clip-mismatch; support panel exploratory",
        "interpretation": "finite-family accessibility of contextual teacher targets; no causal dynamics, clinical utility or distillation gain claim",
        "controls": "unchanged direct-v3 raw controls and context-matched partition-local donors; all arms use recipient X",
        "skeleton_schema": "ordered-bins-v1; 924 columns, unchanged",
        "target": "cached projected person[256], frames38-39 encoded with full64context; no new teacher inference",
        "model_contract": asdict(JointModelContract()), "outer_folds": 5,
        "seed_policy": "deterministic_once; seed0 is identity only",
        "bootstrap_seed": BOOTSTRAP_SEED, "bootstrap_repetitions": 2000,
        "bootstrap_scope": "paired whole-source multiplicities, conditional on saved models, no refitting or new sources",
        "metric": "source-balanced featurewise predictive R2 relative to outer-training target mean; intersection of outer training-variance masks",
        "selection": "same three inner source folds, six RGB/block-X penalties,36 joint candidates+exactbaseline per arm; baseline wins ties",
        "failures": "any required candidate failure makes measurement incomplete; baseline failure aborts",
        "positive_interpretation": "only a development lead if primary paired95%interval excludes0 and real exceeds shuffle and mismatch; no automatic scientific advance",
        "multiplicity": "one named primary contrast; all other comparisons descriptive; no choosing the best panel or target after results",
        "tolerance": TOLERANCE,
    }


def source_snapshot(root):
    root = Path(root)
    return {str(p.relative_to(root)): {"sha256": sha256_file(p), "size": p.stat().st_size,
            "mtime_ns": p.stat().st_mtime_ns} for p in sorted(root.rglob("*")) if p.is_file()}


def implementation_identity():
    package = Path(__file__).parent
    cli = Path(__file__).resolve().parents[4] / "scripts/research_directions/iclr_bridge/run_cached_panel.py"
    return {"future_innovation_sha256": code_fingerprint(),
            "bridge_files": {p.name: sha256_file(p) for p in (package / "__init__.py", Path(__file__), cli)}}


def _validate_roots(source, output, parent):
    for other in (source, parent):
        if output == other or output in other.parents or other in output.parents:
            raise ValueError("Panel outputs and inherited sources must be disjoint sibling trees")


def freeze_cached_panel(source_run, output_root, protocol_document=None):
    source, output = Path(source_run).resolve(), Path(output_root).resolve()
    lineage = read_json(source / "config/parent-lineage.json")
    parent = Path(lineage["parent_root"]).resolve()
    _validate_roots(source, output, parent)
    if output.exists():
        raise ValueError("Freeze requires a new output directory")
    verify_reused_readiness(source)
    cohort, arrays = load_cache(source)
    if len(cohort) != 50 or cohort.video_id.nunique() != 43 or sorted(cohort.outer_fold.unique()) != list(range(5)):
        raise ValueError("This frozen pilot requires the existing50clip/43source cohort")
    output.mkdir(parents=True)
    with stage_lock(output, "freeze"):
        write_once_json(output / "config/specification.json", specification())
        write_once_json(output / "config/source-snapshot.json", source_snapshot(source))
        write_once_json(output / "config/parent-snapshot.json", source_snapshot(parent))
        contract = {"version": VERSION, "created_utc": datetime.now(timezone.utc).isoformat(),
                    "source_run": str(source), "parent_root": str(parent),
                    "source_run_contract_sha256": sha256_file(source / "config/run-contract.json"),
                    "source_cache_contract_sha256": sha256_file(source / "config/cache-contract.json"),
                    "source_lineage_sha256": sha256_file(source / "config/parent-lineage.json"),
                    "specification_sha256": sha256_file(output / "config/specification.json"),
                    "source_snapshot_sha256": sha256_file(output / "config/source-snapshot.json"),
                    "parent_snapshot_sha256": sha256_file(output / "config/parent-snapshot.json"),
                    "implementation": implementation_identity(),
                    "runtime": {"python": platform.python_version(), "platform": platform.platform(),
                                "numpy": np.__version__, "pandas": pd.__version__},
                    "teacher_evidence": "reused read-only, no teacher or raw-video access"}
        if protocol_document is not None:
            path = Path(protocol_document)
            frozen = output / "config/frozen-protocol.md"
            frozen.write_bytes(path.read_bytes())
            contract["protocol_document_sha256"] = sha256_file(frozen)
        for panel in PANELS:
            _, schema = accessibility_features(cohort, arrays["skeleton"], panel)
            write_once_json(output / f"config/{panel}-schema.json", {"names": schema[0], "kinds": schema[1]})
        atomic_write_dataframe_csv(output / "manifests/cohort.csv", cohort)
        contract["cohort_sha256"] = sha256_file(output / "manifests/cohort.csv")
        contract["schema_sha256"] = {p: sha256_file(output / f"config/{p}-schema.json") for p in PANELS}
        write_once_json(output / "config/run-contract.json", contract)
    return contract


def _load_frozen(root):
    root = Path(root)
    contract = read_json(root / "config/run-contract.json")
    if contract["version"] != VERSION or contract["implementation"] != implementation_identity():
        raise ValueError("Frozen code/version mismatch; start a separate experiment")
    expected = json.loads(json.dumps(specification()))
    if read_json(root / "config/specification.json") != expected or sha256_file(root / "config/specification.json") != contract["specification_sha256"]:
        raise ValueError("Changed scientific specification")
    for name, key in (("source-snapshot", "source_snapshot_sha256"), ("parent-snapshot", "parent_snapshot_sha256")):
        if sha256_file(root / f"config/{name}.json") != contract[key]:
            raise ValueError("Changed lineage snapshot")
    if "protocol_document_sha256" in contract and sha256_file(root / "config/frozen-protocol.md") != contract["protocol_document_sha256"]:
        raise ValueError("Changed frozen protocol")
    source, parent = Path(contract["source_run"]), Path(contract["parent_root"])
    for directory, label in ((source, "source"), (parent, "parent")):
        if source_snapshot(directory) != read_json(root / f"config/{label}-snapshot.json"):
            raise ValueError(f"Inherited {label} artifacts changed")
    verify_reused_readiness(source)
    cohort, arrays = load_cache(source)
    if sha256_file(root / "manifests/cohort.csv") != contract["cohort_sha256"]:
        raise ValueError("Changed saved cohort")
    pd.testing.assert_frame_equal(pd.read_csv(root / "manifests/cohort.csv"), cohort, check_dtype=False)
    for panel in PANELS:
        _, schema = accessibility_features(cohort, arrays["skeleton"], panel)
        expected_schema = {"names": list(schema[0]), "kinds": list(schema[1])}
        if read_json(root / f"config/{panel}-schema.json") != expected_schema or sha256_file(root / f"config/{panel}-schema.json") != contract["schema_sha256"][panel]:
            raise ValueError("Changed feature schema")
    return contract, cohort, arrays


def _panel_arrays(cohort, arrays, panel):
    x, schema = accessibility_features(cohort, arrays["skeleton"], panel)
    return {**arrays, "baseline": x}, schema


def fit_panel_fold(root, panel, fold, cohort, arrays):
    """One atomic completion boundary; interrupted incomplete folds can refit."""
    directory = Path(root) / "models" / panel / f"fold-{fold}"
    receipt_path = directory / "complete.json"
    if receipt_path.exists():
        return _reload_fold(root, panel, fold, cohort, arrays)
    local, schema = _panel_arrays(cohort, arrays, panel)
    train = np.flatnonzero(cohort.outer_fold.to_numpy() != fold)
    test = np.flatnonzero(cohort.outer_fold.to_numpy() == fold)
    result = nested_partition(cohort, local, train, test, schema, JointModelContract())
    atomic_save_joblib(directory / "baseline.joblib", result["baseline"])
    for arm, model in result["models"].items():
        atomic_save_joblib(directory / f"{arm}.joblib", model)
    for name in ("selection", "baseline_selection", "split", "preprocessing", "diagnostics"):
        write_json(directory / f"{name}.json", result[name])
    files = [p for p in directory.iterdir() if p.name != "complete.json"]
    write_once_json(receipt_path, {"panel": panel, "outer_fold": fold, "version": VERSION,
                                 "complete": all(v["complete"] for v in result["selection"].values()),
                                 "artifacts": {p.name: sha256_file(p) for p in files}})
    return _reload_fold(root, panel, fold, cohort, arrays)


def _assert_close(a, b, score=False):
    np.testing.assert_allclose(a, b, rtol=TOLERANCE["score_rtol" if score else "prediction_rtol"],
                               atol=TOLERANCE["score_atol" if score else "prediction_atol"])


def _compare_json(actual, expected):
    if isinstance(expected, dict):
        if not isinstance(actual, dict) or set(actual) != set(expected):
            raise ValueError("Report schema mismatch")
        for key in expected:
            _compare_json(actual[key], expected[key])
    elif isinstance(expected, (float, np.floating)):
        _assert_close(actual, expected, score=True)
    elif actual != expected:
        raise ValueError("Report value mismatch")


def _reload_fold(root, panel, fold, cohort, arrays):
    directory = Path(root) / "models" / panel / f"fold-{fold}"
    receipt = read_json(directory / "complete.json")
    expected_files = {"baseline.joblib"} | {f"{a}.joblib" for a in DIRECT_ARMS} | {f"{n}.json" for n in ("selection", "baseline_selection", "split", "preprocessing", "diagnostics")}
    if receipt["panel"] != panel or receipt["outer_fold"] != fold or receipt["version"] != VERSION or set(receipt["artifacts"]) != expected_files:
        raise ValueError("Wrong fold artifact identities")
    for name, digest in receipt["artifacts"].items():
        if sha256_file(directory / name) != digest:
            raise ValueError("Altered fold artifact")
    local, schema = _panel_arrays(cohort, arrays, panel)
    train = np.flatnonzero(cohort.outer_fold.to_numpy() != fold)
    test = np.flatnonzero(cohort.outer_fold.to_numpy() == fold)
    isolated_split(cohort, train, test)
    split, selection = read_json(directory / "split.json"), read_json(directory / "selection.json")
    if split["outer_train"] != cohort.iloc[train].window_id.tolist() or split["outer_test"] != cohort.iloc[test].window_id.tolist():
        raise ValueError("Altered outer source split")
    if len(split["inner"]) != 3:
        raise ValueError("Missing inner source fold")
    mapping = assign_source_folds(cohort.iloc[train].video_id, 3)
    inner_partitions = []
    for f, inner in enumerate(split["inner"]):
        fit = train[[mapping[v] != f for v in cohort.iloc[train].video_id]]
        val = train[[mapping[v] == f for v in cohort.iloc[train].video_id]]
        if inner != {"fold": f, "train": cohort.iloc[fit].window_id.tolist(), "validation": cohort.iloc[val].window_id.tolist()}:
            raise ValueError("Altered inner source split")
        isolated_split(cohort, fit, val)
        inner_partitions.append((fit, val))
    source_by_window = dict(zip(cohort.window_id, cohort.video_id))
    for donor in split["donors"]:
        recipients = donor["recipient_window_ids"]
        for recipient, chosen in zip(recipients, donor["donor_window_ids"]):
            if chosen not in recipients or source_by_window[recipient] == source_by_window[chosen]:
                raise ValueError("Donor source/partition violation")
    base = joblib.load(directory / "baseline.joblib")
    config = JointModelContract()
    baseline_grid = read_json(directory / "baseline_selection.json")
    expected_baselines = {(f"rgb/x={a:g}", "rgb_reference", float(a), None) for a in config.ridge_alphas}
    if (len(baseline_grid) != 6 or {(c["candidate_id"], c["candidate_type"], c["lambda_x"], c["lambda_s"]) for c in baseline_grid} != expected_baselines):
        raise ValueError("Incomplete or altered shared-reference search")
    for candidate in baseline_grid:
        rebuilt = summarize({k: candidate[k] for k in ("candidate_id", "candidate_type", "lambda_x", "lambda_s")}, candidate["inner_folds"])
        if not rebuilt["valid"] or rebuilt["pooled_loss"] != candidate["pooled_loss"]:
            raise ValueError("Invalid baseline pooling")
    best = min(c["pooled_loss"] for c in baseline_grid)
    chosen_alpha = max(c["lambda_x"] for c in baseline_grid if c["pooled_loss"] <= best + config.tie_atol + config.tie_rtol * abs(best))
    reference = next(c for c in baseline_grid if c["lambda_x"] == chosen_alpha)
    if base.ridge.lambda_x != chosen_alpha or base.ridge.lambda_s is not None:
        raise ValueError("Saved baseline penalty differs from inner selection")
    fresh = fit_joint_baseline(local["baseline"][train], arrays["person"][train], cohort.iloc[train].window_id,
                               cohort.iloc[train].video_id, base.ridge.lambda_x, schema)
    if base.x_scaler.record() != fresh.x_scaler.record() or base.y_scaler.training_window_ids != fresh.y_scaler.training_window_ids:
        raise ValueError("Baseline fitted preprocessing does not match training sources")
    for name in ("mean", "scale", "variance"):
        _assert_close(getattr(base.y_scaler, name), getattr(fresh.y_scaler, name))
    np.testing.assert_array_equal(base.valid_features, fresh.valid_features)
    for name in ("x_weight", "s_weight", "intercept"):
        _assert_close(getattr(base.ridge, name), getattr(fresh.ridge, name))
    preprocessing = read_json(directory / "preprocessing.json")
    scopes = [(f"inner-{f}", fit, val) for f, (fit,val) in enumerate(inner_partitions)] + [("outer",train,test)]
    for scope, fit, held in scopes:
        refit = fit_joint_baseline(local["baseline"][fit], arrays["person"][fit], cohort.iloc[fit].window_id,
                                   cohort.iloc[fit].video_id, chosen_alpha, schema)
        yrecord = {k: v.tolist() if isinstance(v,np.ndarray) else v for k,v in asdict(refit.y_scaler).items()}
        expected = {"x": refit.x_scaler.record(), "y": yrecord, "heldout_x": refit.x_scaler.diagnostics(local["baseline"][held])}
        _compare_json(preprocessing[scope], json.loads(json.dumps(expected)))
    donor_reconstruction = []
    for arm in DIRECT_ARMS:
        for scope, fit, held in scopes:
            label = "test" if scope == "outer" else "validation"
            sf, sh = [temporal_features(partition_control(arm, ids, cohort, local, f"{arm}/{scope}/{part}", donor_reconstruction))
                      for ids,part in ((fit,"train"),(held,label))]
            transform = SupportedInput.fit(sf,equal_source_weights(cohort.iloc[fit].video_id),cohort.iloc[fit].window_id,
                                           cohort.iloc[fit].video_id,*skeleton_schema())
            _compare_json(preprocessing[f"{arm}/{scope}"],json.loads(json.dumps({"s":transform.record(),"heldout_s":transform.diagnostics(sh)})))
    if donor_reconstruction != split["donors"]:
        raise ValueError("Saved donors differ from deterministic partition-local matching")
    predictions, types, models = {}, {}, {}
    for arm in DIRECT_ARMS:
        candidates = selection[arm]["candidates"]
        if len(candidates) != 37 or len({c["candidate_id"] for c in candidates}) != 37:
            raise ValueError("Incomplete candidate search")
        expected_grid = {(c["candidate_id"],c["candidate_type"],c["lambda_x"],c["lambda_s"]) for c in candidate_specs(config)}
        expected_grid.add(("baseline_only","baseline_only",chosen_alpha,None))
        if {(c["candidate_id"],c["candidate_type"],c["lambda_x"],c["lambda_s"]) for c in candidates} != expected_grid:
            raise ValueError("Changed candidate penalty grid")
        baseline_candidate = next(c for c in candidates if c["candidate_type"] == "baseline_only")
        if baseline_candidate["inner_folds"] != reference["inner_folds"] or baseline_candidate["pooled_loss"] != reference["pooled_loss"]:
            raise ValueError("Arm baseline differs from shared selected reference")
        for candidate in candidates:
            rebuilt = summarize({k: candidate[k] for k in ("candidate_id", "candidate_type", "lambda_x", "lambda_s")}, candidate["inner_folds"])
            if rebuilt["valid"] != candidate["valid"] or rebuilt["pooled_loss"] != candidate["pooled_loss"]:
                raise ValueError("Candidate pooled loss or validity was altered")
        if selection[arm]["complete"] != all(c["valid"] for c in candidates):
            raise ValueError("Candidate failure concealed by completeness flag")
        chosen, reason = choose_candidate(selection[arm]["candidates"], JointModelContract())
        if selection[arm]["winner"] != chosen["candidate_id"] or selection[arm]["reason"] != reason:
            raise ValueError("Saved selection disagrees with candidate scores")
        model = joblib.load(directory / f"{arm}.joblib")
        if (model["candidate_id"] != chosen["candidate_id"] or model["checkpoint_type"] != chosen["candidate_type"]
                or model["training_window_ids"] != cohort.iloc[train].window_id.tolist()
                or model["lambda_x"] != chosen["lambda_x"] or model["lambda_s"] != chosen["lambda_s"]
                or model["arm"] != arm or model["seed"] != 0 or model["target"] != "person"):
            raise ValueError("Selected artifact identity mismatch")
        sf = temporal_features(partition_control(arm, train, cohort, local, "verify-training", []))
        scaler = SupportedInput.fit(sf, equal_source_weights(cohort.iloc[train].video_id), cohort.iloc[train].window_id,
                                    cohort.iloc[train].video_id, *skeleton_schema())
        if model["s_scaler"].record() != scaler.record():
            raise ValueError("Skeleton preprocessing includes wrong training evidence")
        if model["checkpoint_type"] == "joint_ridge":
            refit = JointRidge.fit(base.x_scaler.transform(local["baseline"][train]), scaler.transform(sf),
                base.y_scaler.transform(arrays["person"][train]), equal_source_weights(cohort.iloc[train].video_id),
                chosen["lambda_x"], chosen["lambda_s"])
            for name in ("x_weight", "s_weight", "intercept"):
                _assert_close(getattr(model["model"], name), getattr(refit, name))
            if model["model"].lambda_x != chosen["lambda_x"] or model["model"].lambda_s != chosen["lambda_s"]:
                raise ValueError("Wrong stored solver penalties")
        sk = partition_control(arm, test, cohort, local, "verify-test", [])
        predictions[arm] = predict_selected(model, base, local["baseline"][test], sk)
        types[arm] = model["checkpoint_type"]; models[arm] = model
    complete = all(s["complete"] for s in selection.values())
    if complete != receipt["complete"]:
        raise ValueError("Wrong completeness flag")
    return {"test": test, "baseline": base, "predictions": predictions, "types": types,
            "models": models, "complete": complete, "selection": selection}


def assemble_panel(root, panel, cohort, arrays):
    n, d = arrays["person"].shape
    bundle = {"y_true": np.zeros((n, d)), "baseline": np.zeros((n, d)),
              "target_mean": np.zeros((n, d)), "target_scale": np.zeros((n, d)),
              "window_id": cohort.window_id.to_numpy(dtype=str), "video_id": cohort.video_id.to_numpy(dtype=str),
              "outer_fold": cohort.outer_fold.to_numpy(), "raw_target": arrays["person"].astype(float)}
    bundle.update({arm: np.zeros((n, d)) for arm in DIRECT_ARMS})
    masks, counts, selected, complete = [], np.zeros(n, dtype=int), [], True
    local, _ = _panel_arrays(cohort, arrays, panel)
    for fold in range(5):
        result = _reload_fold(root, panel, fold, cohort, arrays)
        base, test = result["baseline"], result["test"]
        counts[test] += 1; masks.append(base.valid_features); complete &= result["complete"]
        bundle["y_true"][test] = base.y_scaler.transform(arrays["person"][test])
        bundle["baseline"][test] = base.predict(local["baseline"][test])
        bundle["target_mean"][test] = base.y_scaler.mean
        bundle["target_scale"][test] = base.y_scaler.scale
        for arm in DIRECT_ARMS:
            bundle[arm][test] = result["predictions"][arm]
            selected.append({"panel": panel, "fold": fold, "arm": arm,
                             "type": result["types"][arm], "candidate": result["models"][arm]["candidate_id"]})
    np.testing.assert_array_equal(counts, 1)
    bundle["valid_features"] = np.all(masks, axis=0)
    for arm in ("baseline", *DIRECT_ARMS):
        bundle[f"raw_{arm}"] = bundle[arm] * bundle["target_scale"] + bundle["target_mean"]
    return bundle, selected, complete


def evaluate_bundle(bundle):
    true, base, valid = bundle["y_true"], bundle["baseline"], bundle["valid_features"]
    weights = equal_source_weights(bundle["video_id"])
    scores = {arm: score_arrays(true, base, bundle[arm], weights, valid)[0] for arm in DIRECT_ARMS}
    sums = {arm: source_error_sums(true, base, bundle[arm], weights, bundle["video_id"]) for arm in DIRECT_ARMS}
    rows = []
    for draw, multiplicity in enumerate(source_bootstrap_counts(bundle["video_id"], 2000, BOOTSTRAP_SEED)):
        values = {arm: score_source_sums(total, multiplicity, valid) for arm, total in sums.items()}
        for arm in DIRECT_ARMS:
            rows.append({"draw": draw, "arm": arm, **values[arm],
                         "real_minus_arm": values["real-skeleton"]["r2_full"] - values[arm]["r2_full"]})
    bootstrap = pd.DataFrame(rows)
    contrasts = {}
    for arm in DIRECT_ARMS:
        draws = bootstrap.loc[bootstrap.arm == arm, "real_minus_arm"].to_numpy()
        contrasts[f"real_minus_{arm}"] = {"estimate": scores["real-skeleton"]["r2_full"] - scores[arm]["r2_full"],
            "ci025": float(np.quantile(draws, .025)), "ci975": float(np.quantile(draws, .975)),
            "positive_fraction": float(np.mean(draws > 0))}
    draws = bootstrap.loc[bootstrap.arm == "real-skeleton", "delta_r2"].to_numpy()
    contrasts["real_minus_baseline"] = {"estimate": scores["real-skeleton"]["delta_r2"],
        "ci025": float(np.quantile(draws, .025)), "ci975": float(np.quantile(draws, .975)),
        "positive_fraction": float(np.mean(draws > 0))}
    return scores, contrasts, bootstrap


def _prediction_frame(panel, bundle, root):
    n, d = bundle["y_true"].shape
    frames = []
    for arm in DIRECT_ARMS:
        paths = [f"models/{panel}/fold-{fold}/{arm}.joblib" for fold in bundle["outer_fold"]]
        hashes = [sha256_file(Path(root) / p) for p in paths]
        frames.append(pd.DataFrame({"panel": panel, "arm": arm, "seed": 0, "fit_identity": "deterministic",
            "window_id": np.repeat(bundle["window_id"], d), "video_id": np.repeat(bundle["video_id"], d),
            "outer_fold": np.repeat(bundle["outer_fold"], d), "target_feature": np.tile(np.arange(d), n),
            "valid_feature": np.tile(bundle["valid_features"], n), "y_true": bundle["y_true"].ravel(),
            "y_pred_baseline": bundle["baseline"].ravel(), "y_pred_full": bundle[arm].ravel(),
            "target_mean": bundle["target_mean"].ravel(), "target_scale": bundle["target_scale"].ravel(),
            "raw_target": bundle["raw_target"].ravel(), "raw_baseline": bundle["raw_baseline"].ravel(),
            "raw_prediction": bundle[f"raw_{arm}"].ravel(), "model_artifact": np.repeat(paths, d),
            "model_sha256": np.repeat(hashes, d)}))
    return pd.concat(frames, ignore_index=True)


def run_cached_panel(output_root):
    root = Path(output_root)
    with stage_lock(root, "accessibility-panel"), threadpool_limits(limits=1):
        if (root / "reports/completion.json").exists():
            return verify_cached_panel(root)
        contract, cohort, arrays = _load_frozen(root)
        started = time.monotonic(); summaries, selections, prediction_frames = {}, [], []
        complete = True
        for panel in PANELS:
            for fold in range(5):
                result = fit_panel_fold(root, panel, fold, cohort, arrays)
                print(f"{panel} fold{fold}: " + ", ".join(f"{a}={t}" for a,t in result["types"].items()), flush=True)
            bundle, selected, valid = assemble_panel(root, panel, cohort, arrays)
            scores, contrasts, bootstrap = evaluate_bundle(bundle)
            complete &= valid; selections += selected
            summaries[panel] = {"scores": scores, "contrasts": contrasts, "complete": valid,
                                "valid_feature_count": int(bundle["valid_features"].sum())}
            save_npz(root / f"predictions/{panel}-oof.npz", **bundle)
            atomic_write_dataframe_csv(root / f"reports/{panel}-bootstrap.csv", bootstrap)
            prediction_frames.append(_prediction_frame(panel, bundle, root))
        predictions = pd.concat(prediction_frames, ignore_index=True)
        predictions.to_parquet(root / "predictions/oof.parquet", index=False)
        atomic_write_dataframe_csv(root / "reports/selections.csv", pd.DataFrame(selections))
        main = summaries["posture"]["contrasts"]["real_minus_no-skeleton"]
        lead = (main["ci025"] > 0 and all(summaries["posture"]["contrasts"][f"real_minus_{a}"]["estimate"] > 0
                                                  for a in ("time-shuffle", "clip-mismatch")))
        report = {"version": VERSION, "measurement_complete": complete,
                  "status": "incomplete_measurement" if not complete else "development_lead" if lead else "no_supported_temporal_lead",
                  "scientific_advance": False, "panels": summaries, "wall_seconds": time.monotonic()-started,
                  "sources": int(cohort.video_id.nunique()), "clips": len(cohort), "interpretation": specification()["interpretation"]}
        write_json(root / "reports/panel-report.json", report)
        verification = verify_cached_panel(root, require_seal=False)
        write_json(root / "reports/numerical-verification.json", verification)
        artifacts = {str(p.relative_to(root)): sha256_file(p) for category in ("config", "manifests", "models", "predictions", "reports")
                     for p in sorted((root / category).rglob("*")) if p.is_file() and p.name != "completion.json"}
        write_once_json(root / "reports/completion.json", {"version": VERSION, "measurement_complete": complete, "artifacts": artifacts})
        return report


def verify_cached_panel(output_root, require_seal=True):
    """Read-only checkpoint reload, prediction and score reconstruction."""
    root = Path(output_root)
    _, cohort, arrays = _load_frozen(root)
    if require_seal:
        seal = read_json(root / "reports/completion.json")
        for name, digest in seal["artifacts"].items():
            if sha256_file(root / name) != digest:
                raise ValueError("Sealed artifact changed")
    report = read_json(root / "reports/panel-report.json")
    frames, selections, complete, rows = [], [], True, 0
    for panel in PANELS:
        bundle, selected, valid = assemble_panel(root, panel, cohort, arrays)
        selections += selected; complete &= valid
        with np.load(root / f"predictions/{panel}-oof.npz", allow_pickle=False) as saved:
            if set(saved.files) != set(bundle):
                raise ValueError("Wrong OOF arrays/schema")
            for name, value in bundle.items():
                if value.dtype.kind in "buUiS": np.testing.assert_array_equal(saved[name], value)
                else: _assert_close(saved[name], value)
        frames.append(_prediction_frame(panel, bundle, root))
        scores, contrasts, bootstrap = evaluate_bundle(bundle)
        expected = {"scores": scores, "contrasts": contrasts, "complete": valid,
                    "valid_feature_count": int(bundle["valid_features"].sum())}
        # All arithmetic is reconstructed; hashes cannot excuse a changed score.
        _compare_json(report["panels"][panel], expected)
        pd.testing.assert_frame_equal(pd.read_csv(root / f"reports/{panel}-bootstrap.csv"), bootstrap,
                                      check_dtype=False, rtol=1e-10, atol=1e-12)
        rows += len(bootstrap)
    expected_predictions = pd.concat(frames, ignore_index=True)
    actual = pd.read_parquet(root / "predictions/oof.parquet")
    keys = ["panel", "arm", "window_id", "target_feature"]
    if actual.duplicated(keys).any():
        raise ValueError("Duplicate prediction identity")
    pd.testing.assert_frame_equal(actual, expected_predictions, check_dtype=False, rtol=1e-10, atol=1e-10)
    pd.testing.assert_frame_equal(pd.read_csv(root / "reports/selections.csv"), pd.DataFrame(selections), check_dtype=False)
    main = report["panels"]["posture"]["contrasts"]["real_minus_no-skeleton"]
    lead = main["ci025"] > 0 and all(report["panels"]["posture"]["contrasts"][f"real_minus_{a}"]["estimate"] > 0 for a in ("time-shuffle", "clip-mismatch"))
    expected_status = "incomplete_measurement" if not complete else "development_lead" if lead else "no_supported_temporal_lead"
    if report["measurement_complete"] != complete or report["scientific_advance"] or report["status"] != expected_status:
        raise ValueError("Incorrect exploration status")
    return {"status": "passed", "version": VERSION, "measurement_complete": complete,
            "scientific_status": expected_status, "model_count": 40, "prediction_rows": len(actual),
            "bootstrap_rows": rows, "source_artifacts_unchanged": True, "parent_artifacts_unchanged": True,
            "read_only": True, "tolerance": TOLERANCE,
            "verification_scope": "reload and refit selected linear models and preprocessing from declared training sources; reconstruct OOF, all bootstrap draws and report; inner losses are checked for pooling and selection but not independently refitted"}
