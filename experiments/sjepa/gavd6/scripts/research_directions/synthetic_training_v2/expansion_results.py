#!/usr/bin/env python3
"""Verify expansion artifacts and add explicitly exploratory movement diagnostics.

This module does not submit jobs, rewrite the frozen experiment, or authorize a
scientific gate. Scheduler completion belongs to the enclosing suite controller.
"""
from __future__ import annotations

import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(Path(__file__).parent))

from check_results import assert_saved_metrics_match, read, require
from gavd6_sjepa.research_directions.synthetic_training_v2.config import RunConfig, STAGES
from gavd6_sjepa.research_directions.synthetic_training_v2.contracts import (
    TrackBundle, array_digest, atomic_json, code_identity, digest, sha256_file,
)

ENDPOINTS = ("visible", "all_valid_synthetic")
SCALE_POLICIES = ("frame_reference", "window_median_reference")
PAIRS = {"hip": (6, 7), "knee": (8, 9), "ankle": (10, 11)}
MEASURES = ("visible_nle", "lower_limb_nle", "missing_rate", "displacement_nle",
            "ankle_separation_mae", "amplitude_error", "amplitude_ratio", "event_timing_mae_s")


def verify_scope(config_path, repo=ROOT, *, allow_fixture_tests=False):
    """Read-only artifact verification; fixtures need an explicit test-only flag.

This checks exactly the saved scope, with no scheduler or human-review claim.
The existing receipt hashes and reconstructed metrics remain mandatory.
"""
    import numpy as np
    from gavd6_sjepa.research_directions.synthetic_training_v2.workflow import (
        _receipt, _record_groups, reconstruct_metrics,
    )

    cfg = RunConfig.load(config_path)
    fixture = cfg.mode == "fixture"
    require(cfg.mode == "source" or (fixture and allow_fixture_tests),
            "Expansion verification requires source mode; fixtures are software tests only.")
    repo = Path(repo).resolve()
    expected = dict(configuration=cfg.as_dict(), code=code_identity(repo),
                    protocol=sha256_file(repo / "docs/studies/synthetic-training-v2/protocol.md"))
    if not fixture:
        expected["source_manifest"] = sha256_file(Path(cfg.bundle) / "manifest.json")
    if cfg.decision_spec:
        from gavd6_sjepa.research_directions.synthetic_training_v2.decisions import load_decision_spec
        spec = load_decision_spec(cfg.decision_spec)
        expected.update(decision_spec=sha256_file(cfg.decision_spec),
                        calibration_artifact=spec["calibration_artifact_sha256"])
    if cfg.cost_ledger:
        expected["prior_cost_ledger"] = sha256_file(cfg.cost_ledger)
    identity = read(cfg.root / "identity.json")
    require(identity["signature"] == digest(expected), "Scope identity changed; retain the frozen code/configuration.")
    require(RunConfig.load(cfg.root / "effective-config.json").as_dict() == cfg.as_dict(),
            "Effective scope configuration differs from the requested configuration.")
    for stage in STAGES:
        _receipt(cfg, stage)
    bundle = TrackBundle.load(cfg.root / "data/bundle")
    require(bundle.evidence_status == "fixture-tested" if fixture else
            bundle.evidence_status in {"source-run", "automated-source-screen"},
            "Bundle evidence label is incompatible with scope mode.")
    bundle.validate(cfg.held_extractor)
    if not fixture:
        source = TrackBundle.load(cfg.bundle)
        require(source.records == bundle.records and source.provenance == bundle.provenance
                and array_digest(source.inputs) == array_digest(bundle.inputs)
                and array_digest(source.targets) == array_digest(bundle.targets),
                "Retained fitting data differ from the declared prepared bundle.")
    train, dev = bundle.subset("train"), bundle.subset("development")
    train_people = {r["canonical_person_id"] for r in train.records}
    dev_people = {r["canonical_person_id"] for r in dev.records}
    require(train_people and dev_people and not train_people & dev_people,
            "Training and development require nonempty, disjoint canonical people.")
    methods = {"unchanged", "filter0", "filter1", "filter2", *cfg.arms}
    expected_names = {f"{method}-{seed}" for method in methods for seed in cfg.seeds}
    for suffix in (".npz", ".json"):
        require({p.stem for p in (cfg.root / "predictions").glob(f"*{suffix}")} == expected_names,
                f"Prediction {suffix} roster differs from the declared methods/seeds.")
    for method in sorted(methods):
        for seed in cfg.seeds:
            path = cfg.root / "predictions" / f"{method}-{seed}.npz"
            meta = read(path.with_suffix(".json"))
            require(meta == dict(method=method, seed=seed, records=_record_groups(dev.records, seed),
                                 evidence_status=bundle.evidence_status),
                    f"Prediction metadata differ from the declared development panel: {path.name}")
            with np.load(path, allow_pickle=False) as saved:
                require(saved["prediction"].shape == dev.targets["xy"].shape,
                        f"Prediction shape differs: {path.name}")
                references = dict(timestamps=dev.inputs["timestamps"], input_xy=dev.inputs["xy"],
                    input_observed=dev.inputs["observed"], input_confidence=dev.inputs["confidence"],
                    target_xy=dev.targets["xy"], target_valid=dev.targets["valid"],
                    target_visible=dev.targets["visible"], eval_scale=dev.targets["eval_scale"])
                for key, value in references.items():
                    require(np.array_equal(saved[key], value, equal_nan=True),
                            f"Saved prediction reference differs: {path.name}: {key}")
    for arm in cfg.arms:
        for seed in cfg.seeds:
            folder = cfg.root / "fits" / f"{arm}-{seed}"
            fit = read(folder / "training.json")
            planned = cfg.readout_updates if arm == "initialized" else cfg.updates + cfg.readout_updates
            require(fit["status"] == "complete" and fit["arm"] == arm and fit["seed"] == seed
                    and fit["planned_updates"] == planned and (folder / "model.pt").is_file(),
                    f"Incomplete or mismatched fit: {arm}-{seed}")
            if cfg.resource_contrast == "matched_data_steps":
                require(fit["optimizer_updates"] == planned, f"Declared updates were not completed: {arm}-{seed}")
    metrics = reconstruct_metrics(cfg.root)
    require(len(metrics) == len(dev.records) * len(methods) * len(cfg.seeds),
            "Reconstructed metrics do not cover the declared comparison.")
    assert_saved_metrics_match(cfg.root / "evaluation/per-window.csv", metrics)
    for name in ("report.md", "data/achieved-size.json", "development-snapshot.json",
                 "evaluation/per-person-balanced-summary.csv", "evaluation/per-person.csv",
                 "evaluation/nuisance-strata.csv", "evaluation/training-seed-variability.csv",
                 "evaluation/contrasts.json", "evaluation/gates.json", "evaluation/accuracy-preservation.png"):
        path = cfg.root / name
        require(path.is_file() and path.stat().st_size > 0, f"Missing scope artifact: {name}")
    gates = read(cfg.root / "evaluation/gates.json")
    require(set(gates) == {"A", "B", "real_transfer", "personalization", "video"}, "Incomplete scientific gate record.")
    require(read(cfg.root / "development-snapshot.json")["confirmation_opened"] is False,
            "Expansion checks cannot certify confirmation access.")
    if bundle.evidence_status in {"fixture-tested", "automated-source-screen"}:
        require(gates["B"]["status"] == "insufficient_evidence", "Development screening cannot authorize Gate B.")
    return dict(status="SCOPE_ARTIFACTS_COMPLETE", results=str(cfg.root), source_identity=identity["signature"],
                evidence_status=bundle.evidence_status, training_people=len(train_people),
                development_people=len(dev_people), development_track_records=len(dev.records),
                seeds=list(cfg.seeds), methods=sorted(methods), per_window_metric_rows=len(metrics),
                scheduler="Not checked here; suite allocation accounting is separate.",
                scientific_scope="Synthetic development; no confirmation or real-transfer claim.")


def left_right_metrics(prediction, targets, records, *, method, evidence_status, endpoint="visible", missing_penalty=1.):
    """Per-window 2D error in the left-minus-right hip/knee/ankle vector.

A missing endpoint receives one reference-box diagonal of error; unsupported
reference frames never enter the denominator. This is a geometric diagnostic,
not a classifier of anatomical side or evidence of clinical asymmetry.
"""
    import numpy as np
    import pandas as pd

    require(endpoint in ENDPOINTS, "Unknown exploratory endpoint.")
    pred, truth = np.asarray(prediction, float), np.asarray(targets["xy"], float)
    require(pred.shape == truth.shape and pred.ndim == 4 and pred.shape[-2:] == (12, 2),
            "Left-right coordinates must have matching [N,T,12,2] shape.")
    require(len(records) == len(pred), "Left-right metadata count differs from predictions.")
    require(np.isfinite(missing_penalty) and missing_penalty > 0, "Missing penalty must be positive finite.")
    mask = np.asarray(targets["visible"] if endpoint == "visible" else targets["valid"])
    scale = np.asarray(targets["eval_scale"], float)
    require(mask.dtype == bool and mask.shape == pred.shape[:-1] and scale.shape == pred.shape[:2]
            and np.isfinite(scale).all() and (scale > 0).all(), "Invalid left-right support or reference scale.")
    require(not np.any(mask & ~np.isfinite(truth).all(-1)), "Supported reference coordinates must be finite.")
    if endpoint == "all_valid_synthetic":
        require(all(r.get("target_kind") == "synthetic_proxy" for r in records),
                "Hidden-joint diagnostics require explicit synthetic-proxy targets.")
    rows = []
    for pair, (left, right) in PAIRS.items():
        support = mask[:, :, left] & mask[:, :, right]
        finite = np.isfinite(pred[:, :, [left, right]]).all(axis=(-1, -2))
        residual = (pred[:, :, left] - pred[:, :, right]) - (truth[:, :, left] - truth[:, :, right])
        error = np.where(finite, np.linalg.norm(residual, axis=-1) / scale, missing_penalty)
        for index, record in enumerate(records):
            count = int(support[index].sum())
            rows.append({**record, "method": method, "evidence_status": evidence_status,
                         "endpoint": endpoint, "pair": pair, "analysis_status": "exploratory",
                         "left_right_nle": float(error[index, support[index]].mean()) if count else np.nan,
                         "reference_frames": count, "finite_prediction_frames": int((support[index] & finite[index]).sum()),
                         "missing_prediction_frames": int((support[index] & ~finite[index]).sum()),
                         "missing_penalty": float(missing_penalty)})
    return pd.DataFrame(rows)


def _balanced(frame, measures, extra=()):
    """Retain unsupported strata while balancing variants, windows, motions, people."""
    import pandas as pd
    from gavd6_sjepa.research_directions.synthetic_training_v2.evaluation import aggregate_metrics

    rows = []
    columns = ["endpoint", "scale_policy", *extra]
    for keys, group in frame.groupby(columns, sort=True, dropna=False):
        keys = keys if isinstance(keys, tuple) else (keys,)
        for condition, selected in [("all_conditions", group), *list(group.groupby("variant", sort=True))]:
            for measure in measures:
                result = aggregate_metrics(selected, measure)
                for key, value in zip(columns, keys):
                    result[key] = value
                result["condition"] = condition
                result["analysis_status"] = "exploratory"
                rows.append(result)
    return pd.concat(rows, ignore_index=True)


def _analysis_code():
    from postrun_checks import analysis_code
    return {**analysis_code(), str(Path(__file__).relative_to(ROOT)): sha256_file(__file__),
            "scripts/research_directions/synthetic_training_v2/check_results.py":
                sha256_file(Path(__file__).with_name("check_results.py"))}


def evaluation_targets(targets, records, endpoint, scale_policy):
    """Copy evaluation support/scale only; never mutate retained references."""
    import numpy as np

    require(endpoint in ENDPOINTS and scale_policy in SCALE_POLICIES, "Unknown diagnostic support or scale policy.")
    result = dict(targets)
    if endpoint == "all_valid_synthetic":
        require(all(r.get("target_kind") == "synthetic_proxy" for r in records),
                "Hidden-joint diagnostics require explicit synthetic-proxy targets.")
        result["visible"] = targets["valid"].copy()
    if scale_policy == "window_median_reference":
        scale = targets["eval_scale"]
        result["eval_scale"] = np.broadcast_to(np.median(scale, axis=1)[:, None], scale.shape).copy()
    return result


def run_analysis(config_path, output, *, allow_fixture_tests=False, max_windows=4):
    """Run the original three checks and separate exploratory support endpoints."""
    import numpy as np
    import pandas as pd
    from postrun_checks import analyze_source, source_files, _records, _timing_summary
    from diagnostics.timing import timing_diagnostics
    from gavd6_sjepa.research_directions.synthetic_training_v2.evaluation import evaluate_predictions

    verified = verify_scope(config_path, allow_fixture_tests=allow_fixture_tests)
    cfg, output = RunConfig.load(config_path), Path(output).expanduser().resolve()
    require({"initialized", "coordinate", "direct", "paired_jepa"} <= set(cfg.arms),
            "Standard post-run checks require initialized, coordinate, direct and paired_jepa.")
    if output.exists():
        raise FileExistsError(f"Preserve existing diagnostics; use a new output directory: {output}")
    require(not output.is_relative_to(cfg.root) and not cfg.root.is_relative_to(output),
            "Diagnostic output must be separate from source artifacts.")
    before, code = source_files(cfg.root), _analysis_code()
    standard = analyze_source(cfg.root, output, verified, checks="all", max_windows=max_windows)
    bundle = TrackBundle.load(cfg.root / "data/bundle")
    dev = bundle.subset("development")
    require(all(r.get("target_kind") == "synthetic_proxy" for r in dev.records),
            "Expanded support requires synthetic-proxy targets; real hidden joints remain unscored.")
    coordinates, bilateral, timings = [], [], []
    for seed in cfg.seeds:
        records = _records(dev.records, seed)
        predictions = {}
        for method in verified["methods"]:
            with np.load(cfg.root / "predictions" / f"{method}-{seed}.npz", allow_pickle=False) as saved:
                predictions[method] = saved["prediction"].copy()
        for method in ("joint_offset", "joint_affine"):
            with np.load(output / "predictions" / f"{method}.npz", allow_pickle=False) as saved:
                predictions[method] = saved["prediction"].copy()
        for endpoint in ENDPOINTS:
            for scale_policy in SCALE_POLICIES:
                targets = evaluation_targets(dev.targets, records, endpoint, scale_policy)
                for method, prediction in predictions.items():
                    frame = evaluate_predictions(prediction, targets, dev.inputs["timestamps"], records,
                                                  method=method, evidence_status=bundle.evidence_status)
                    frame["endpoint"], frame["analysis_status"] = endpoint, "exploratory"
                    frame["scale_policy"] = scale_policy
                    frame["original_visible_joint_frames"] = dev.targets["visible"].sum(axis=(1, 2))
                    frame["original_occluded_joint_frames"] = (dev.targets["valid"] & ~dev.targets["visible"]).sum(axis=(1, 2))
                    coordinates.append(frame)
                    relation = left_right_metrics(prediction, targets, records, method=method,
                        evidence_status=bundle.evidence_status, endpoint=endpoint)
                    relation["scale_policy"] = scale_policy
                    bilateral.append(relation)
                for method, prediction in {**predictions, "reference_oracle": dev.targets["xy"]}.items():
                    frame = timing_diagnostics(prediction, targets, dev.inputs["timestamps"], records,
                        method=method, evidence_status=bundle.evidence_status, tolerance_s=.12)
                    frame["endpoint"], frame["analysis_status"] = endpoint, "exploratory"
                    frame["scale_policy"] = scale_policy
                    timings.append(frame)
    coordinates, bilateral, timings = [pd.concat(rows, ignore_index=True) for rows in (coordinates, bilateral, timings)]
    coordinates.to_csv(output / "exploratory-per-window.csv", index=False)
    _balanced(coordinates, MEASURES).to_csv(output / "exploratory-balanced.csv", index=False)
    bilateral.to_csv(output / "exploratory-left-right.csv", index=False)
    _balanced(bilateral, ("left_right_nle",), ("pair",)).to_csv(output / "exploratory-left-right-balanced.csv", index=False)
    timings.to_csv(output / "exploratory-timing.csv", index=False)
    summaries = []
    for (endpoint, scale_policy), group in timings.groupby(["endpoint", "scale_policy"], sort=True):
        for condition, selected in [("all_conditions", group), *list(group.groupby("variant", sort=True))]:
            summary = _timing_summary(selected)
            summary["endpoint"], summary["condition"], summary["analysis_status"] = endpoint, condition, "exploratory"
            summary["scale_policy"] = scale_policy
            summaries.append(summary)
    pd.concat(summaries, ignore_index=True).to_csv(output / "exploratory-timing-summary.csv", index=False)
    (output / "exploratory-notes.md").write_text(
        "# Exploratory movement diagnostics\n\n"
        "The standard three post-run checks are retained in report.md. All added endpoints are exploratory.\n\n"
        "Visible support follows the original masks. all_valid_synthetic also scores occluded joints with valid "
        "projected synthetic references; it does not establish real hidden-joint accuracy. A copied evaluation mask "
        "changes only these diagnostic calculations, never the original targets, predictions or scientific gates.\n\n"
        "original_visible_joint_frames and original_occluded_joint_frames retain source support counts. Other "
        "visible/occluded count columns reflect the diagnostic mask, so all_valid_synthetic has no occluded "
        "scoring subset even where the original image was occluded.\n\n"
        "frame_reference retains the original per-frame reference-box diagonal. window_median_reference uses "
        "one median reference-box diagonal for the whole window. This exploratory sensitivity check separates "
        "normalization-induced amplitude and peak changes from coordinate changes. It is an evaluation-only "
        "reference scale, never a model input or fitting statistic. The original scale remains the primary endpoint.\n\n"
        "Left-right error is the length of (predicted left minus right) minus (reference left minus right), "
        "divided by the selected reference scale (framewise or fixed per window). Hip, knee and ankle pairs are separate. Missing predictions "
        "receive a penalty of one diagonal and remain in supported denominators. These 2D relationships are not "
        "clinical asymmetry or a laterality classifier.\n\n"
        "Balanced tables average conditions within windows, windows within motions, and motions within people. "
        "Unsupported motion scores stay missing. Timing summaries are descriptive event totals; read eligibility, "
        "missed/extra peaks and coverage with conditional timing error. The fixed tolerance is 0.12 seconds. "
        "The reference oracle is evaluated against itself under each support definition.\n\n"
        "Training-seed repeats reuse the same people and deterministic calibration predictions. They do not "
        "increase the number of independent people. No result here authorizes Gate B or real transfer.\n")
    require(source_files(cfg.root) == before and _analysis_code() == code,
            "Source artifacts or analysis code changed during expanded diagnostics.")
    result = dict(status="EXPANSION_ANALYSIS_COMPLETE", source_identity=verified["source_identity"],
                  config=str(Path(config_path).resolve()), config_sha256=sha256_file(config_path),
                  source=str(cfg.root), output=str(output), evidence_status=bundle.evidence_status,
                  standard_checks=standard, exploratory_endpoints=list(ENDPOINTS),
                  scale_policies=list(SCALE_POLICIES),
                  source_unchanged=True, analysis_code=code, scientific_gate_authorized=False,
                  files={str(p.relative_to(output)): sha256_file(p) for p in sorted(output.rglob("*")) if p.is_file()})
    atomic_json(output / "expansion-analysis.json", result)
    return result


def report_suite(config_paths, diagnostics_paths, output, *, allow_fixture_tests=False):
    """Combine seed summaries descriptively, keeping data/recipe panels separate."""
    import numpy as np
    import pandas as pd

    config_paths, diagnostics_paths = list(config_paths), list(diagnostics_paths)
    require(config_paths and len(config_paths) == len(diagnostics_paths), "One diagnostics directory is required per scope.")
    output = Path(output).expanduser().resolve()
    if output.exists():
        raise FileExistsError(f"Preserve existing suite reports: {output}")
    tables, lr_tables, timings, scopes = [], [], [], []
    for path, diagnostics in zip(config_paths, diagnostics_paths):
        cfg, diagnostics = RunConfig.load(path), Path(diagnostics).resolve()
        require(not output.is_relative_to(cfg.root) and not cfg.root.is_relative_to(output)
                and not output.is_relative_to(diagnostics) and not diagnostics.is_relative_to(output),
                "Suite report output must be separate from each source and diagnostic tree.")
        verified = verify_scope(path, allow_fixture_tests=allow_fixture_tests)
        analysis = read(diagnostics / "expansion-analysis.json")
        require(analysis["status"] == "EXPANSION_ANALYSIS_COMPLETE"
                and analysis["source_identity"] == verified["source_identity"]
                and analysis["config_sha256"] == sha256_file(path)
                and analysis["analysis_code"] == _analysis_code(), "Diagnostic provenance differs from the declared scope/code.")
        for name, expected in analysis["files"].items():
            require(sha256_file(diagnostics / name) == expected, f"Changed diagnostic artifact: {diagnostics / name}")
        recipe = dict(updates=cfg.updates, readout_updates=cfg.readout_updates, model=cfg.model,
                      arms=list(cfg.arms), resource_contrast=cfg.resource_contrast,
                      total_compute_seconds=cfg.total_compute_seconds, held_extractor=cfg.held_extractor,
                      batch_size=cfg.batch_size,
                      data_manifest=sha256_file(cfg.root / "data/bundle/manifest.json"))
        recipe_id = digest(recipe)[:16]
        scopes.append(dict(config=str(Path(path).resolve()), diagnostics=str(diagnostics),
                           recipe_id=recipe_id, recipe=recipe, verification=verified))
        for filename, collection in (("exploratory-balanced.csv", tables),
                                     ("exploratory-left-right-balanced.csv", lr_tables),
                                     ("exploratory-timing-summary.csv", timings)):
            frame = pd.read_csv(diagnostics / filename)
            frame["scope"], frame["recipe_id"] = cfg.run_id, recipe_id
            frame["updates"], frame["readout_updates"] = cfg.updates, cfg.readout_updates
            collection.append(frame)
    table, lr_table, timing = [pd.concat(rows, ignore_index=True) for rows in (tables, lr_tables, timings)]
    keys = ["recipe_id", "endpoint", "scale_policy", "condition", "method", "split", "extractor", "metric", "evidence_status"]
    require(not table.duplicated([*keys, "seed"]).any(), "Duplicate training seed within the same data/recipe panel.")
    summaries = []
    for labels, group in table.groupby(keys, sort=True, dropna=False):
        values = group.value.to_numpy(float)
        supported = bool(np.isfinite(values).all())
        require(group.people.nunique() == 1, "Independent person counts differ within a recipe panel.")
        summaries.append(dict(zip(keys, labels)) | dict(
            seed_count=len(group), seeds=" ".join(str(int(x)) for x in sorted(group.seed)),
            mean=float(values.mean()) if supported else np.nan,
            standard_deviation=float(values.std(ddof=1)) if supported and len(values) > 1 else np.nan,
            independent_people=int(group.people.iloc[0]), supported_seed_count=int(np.isfinite(values).sum()),
            uncertainty="Descriptive training-seed variation; no population interval."))
    summary = pd.DataFrame(summaries)
    output.mkdir(parents=True, exist_ok=False)
    table.to_csv(output / "per-seed-balanced.csv", index=False)
    lr_table.to_csv(output / "per-seed-left-right.csv", index=False)
    timing.to_csv(output / "per-seed-timing.csv", index=False)
    summary.to_csv(output / "training-seed-summary.csv", index=False)
    rows = ["# Expanded synthetic restoration: completed development scopes", "",
            "All source scopes and original metrics were verified from retained artifacts. Scheduler completion "
            "and allocation costs are recorded separately by the suite controller. These are synthetic development "
            "results, including explicitly exploratory support analyses; no scientific gate or real-transfer claim is authorized.", "",
            "Each data/recipe panel stays separate. Mean and sample standard deviation describe training-seed variation "
            "on the same people. Seeds, windows, rendering conditions and frames are not additional independent people. "
            "Calibration predictions repeat deterministically across seeds. Unsupported strata remain missing.", "",
            "| Panel | Updates + readout | Seeds | Training people | Development people | Evidence |",
            "|---|---:|---|---:|---:|---|"]
    panels = {}
    for scope in scopes:
        panels.setdefault(scope["recipe_id"], []).append(scope)
    for panel, group in panels.items():
        first = group[0]
        recipe, verified = first["recipe"], first["verification"]
        seeds = sorted(seed for item in group for seed in item["verification"]["seeds"])
        rows.append(f"| {panel} | {recipe['updates']} + {recipe['readout_updates']} | {seeds} | "
                    f"{verified['training_people']} | {verified['development_people']} | {verified['evidence_status']} |")
    rows += ["", "| Panel | Support | Method | Extractor | Coordinate mean | Seed SD | People |",
             "|---|---|---|---|---:|---:|---:|"]
    chosen = summary.loc[summary.condition.eq("all_conditions") & summary.metric.eq("visible_nle")
                         & summary.scale_policy.eq("frame_reference")]
    for row in chosen.itertuples():
        mean = f"{row.mean:.8f}" if np.isfinite(row.mean) else "unsupported"
        sd = f"{row.standard_deviation:.8f}" if np.isfinite(row.standard_deviation) else "unavailable"
        rows.append(f"| {row.recipe_id} | {row.endpoint} | {row.method} | {row.extractor} | {mean} | {sd} | {row.independent_people} |")
    rows += ["", "Full movement and condition scores are in per-seed-balanced.csv; hip/knee/ankle left-right "
             "vector errors are in per-seed-left-right.csv. Timing eligibility, missed/extra peaks, coverage and "
             "conditional error are in per-seed-timing.csv. Each diagnostics directory retains reference trajectories, "
             "calibration fits, per-joint residuals and training histories. The all_valid_synthetic endpoint includes "
             "occluded synthetic joints and does not substitute for independent real annotations. The table above "
             "retains the original frame_reference scale; CSVs also retain the exploratory window_median_reference "
             "sensitivity check. Inherited source data retain the original frontal camera and short observation windows.", ""]
    (output / "report.md").write_text("\n".join(rows))
    result = dict(status="EXPANSION_SUITE_REPORT_COMPLETE", output=str(output), scopes=scopes,
                  scientific_gate_authorized=False, analysis_status="exploratory_development",
                  files={str(p.relative_to(output)): sha256_file(p) for p in sorted(output.iterdir()) if p.is_file()})
    atomic_json(output / "suite-report.json", result)
    return result
