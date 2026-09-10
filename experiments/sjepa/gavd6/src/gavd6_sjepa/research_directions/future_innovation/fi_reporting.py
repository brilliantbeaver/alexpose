"""Recomputable OOF scoring, immutable evidence, and the experiment decision."""

from __future__ import annotations

from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd

from gavd6_sjepa.shared_infrastructure.artifact_io_operations import (
    atomic_write_dataframe_csv,
    sha256_file,
)

from .fi_contracts import (
    ARMS,
    GateThresholds,
    check_run,
    equal_source_weights,
    load_model_contract,
    read_json,
    stable_key,
    write_once_json,
)
from .fi_feature_cache import load_cache
from .fi_gate_decision import decide_gate
from .fi_metrics import score_arrays, source_bootstrap_indices
from .fi_nested_training import verify_fold
from .fi_residual_models import TrainingScaler
from .fi_validity_audits import require_audits


def scores_binding(root):
    root = Path(root)
    paths = [
        "config/run-contract.json",
        "config/cache-contract.json",
        "qc/validity-summary.json",
    ]
    paths += [f"models/fold-{fold}/fold-complete.json" for fold in range(5)]
    return stable_key(*[sha256_file(root / path) for path in paths])


def assemble_oof(cohort, cache, predictions, config):
    """Check identities, fitted statistics, target units, and every expected row."""
    keys = ["window_id", "arm", "seed", "target_feature"]
    if (
        predictions.duplicated(keys).any()
        or len(predictions) != 50 * len(ARMS) * len(config.seeds) * 256
    ):
        raise ValueError("Missing or duplicated OOF predictions")
    if (
        set(predictions.window_id) != set(cohort.window_id)
        or set(predictions.arm) != set(ARMS)
        or set(predictions.seed) != set(config.seeds)
        or set(predictions.target_feature) != set(range(256))
    ):
        raise ValueError(
            "Prediction windows, arms, seeds, or target dimensions differ from plan"
        )
    numeric = [
        "y_true",
        "y_pred_baseline",
        "y_pred_full",
        "y_reference",
        "target_mean",
        "target_scale",
        "target_training_variance",
    ]
    if (
        not np.isfinite(predictions[numeric].to_numpy()).all()
        or not (predictions.target_scale > 0).all()
    ):
        raise ValueError("Non-finite predictions or invalid target scales")
    metadata = cohort.set_index("window_id")
    for row in (
        predictions[["window_id", "video_id", "outer_fold"]]
        .drop_duplicates()
        .itertuples()
    ):
        if (
            row.video_id != metadata.loc[row.window_id, "video_id"]
            or row.outer_fold != metadata.loc[row.window_id, "outer_fold"]
        ):
            raise ValueError("Prediction source/fold does not match frozen cohort")
    result, shared = {}, {}
    for arm in ARMS:
        target_name = "background" if arm == "background-target" else "person"
        for seed in config.seeds:
            table = predictions[(predictions.arm == arm) & (predictions.seed == seed)]
            if len(table) != 50 * 256 or set(table.target) != {target_name}:
                raise ValueError("Incomplete arm/seed or wrong target")

            def matrix(name, table=table):
                return (
                    table.pivot(
                        index="window_id", columns="target_feature", values=name
                    )
                    .loc[cohort.window_id, range(256)]
                    .to_numpy()
                )

            values = {name: matrix(name) for name in numeric}
            flags = matrix("valid_feature")
            if flags.dtype != bool:
                raise ValueError("Feature-validity flags must be boolean")
            for fold in range(5):
                train = cohort.outer_fold.to_numpy() != fold
                test = ~train
                scaler = TrainingScaler.fit(
                    cache[target_name][train],
                    equal_source_weights(cohort.loc[train, "video_id"]),
                    cohort.loc[train, "window_id"],
                )
                for column, expected in (
                    ("target_mean", scaler.mean),
                    ("target_scale", scaler.scale),
                    ("target_training_variance", scaler.variance),
                ):
                    if not np.allclose(
                        values[column][test], expected[None], rtol=1e-10, atol=1e-12
                    ):
                        raise ValueError(
                            f"{column} was not fitted exclusively on outer-training sources"
                        )
                if not np.array_equal(
                    flags[test],
                    np.broadcast_to(
                        scaler.variance > config.target_variance_tolerance,
                        flags[test].shape,
                    ),
                ):
                    raise ValueError(
                        "Constant target dimensions marked valid or feature masks differ"
                    )
                expected_y = scaler.transform(cache[target_name][test])
                if not np.allclose(
                    values["y_true"][test], expected_y, rtol=1e-10, atol=1e-12
                ):
                    raise ValueError(
                        "Saved target does not match its cached target and training standardization"
                    )
            if np.any(values["y_reference"] != 0):
                raise ValueError(
                    "Reference must be each outer-training mean, zero in standardized units"
                )
            valid = flags.all(axis=0)
            if not valid.any():
                raise ValueError(
                    "No consistently valid projected features across all outer folds"
                )
            if target_name in shared:
                for name in (
                    "y_true",
                    "y_pred_baseline",
                    "target_mean",
                    "target_scale",
                ):
                    if not np.array_equal(values[name], shared[target_name][name]):
                        raise ValueError(
                            "Arms/seeds sharing a target did not share its baseline and target units"
                        )
            else:
                shared[target_name] = values
            result[(arm, seed)] = (
                values["y_true"],
                values["y_pred_baseline"],
                values["y_pred_full"],
                valid,
            )
    return result


def verify_scores(root):
    root = Path(root)
    check_run(root)
    contract = read_json(root / "reports/scores-contract.json")
    if contract["binding"] != scores_binding(root):
        raise ValueError("Scoring evidence lineage changed")
    for path, digest in contract["artifacts"].items():
        if sha256_file(root / path) != digest:
            raise ValueError(f"Scoring artifact changed: {path}")
    return contract


def score_gate(root):
    root = Path(root)
    require_audits(root)
    if (root / "reports/scores-contract.json").exists():
        verify_scores(root)
        return
    cohort, cache = load_cache(root)
    config = load_model_contract(root)
    tables = [verify_fold(root, fold) for fold in range(5)]
    predictions = pd.concat(tables, ignore_index=True)
    arrays = assemble_oof(cohort, cache, predictions, config)
    weights = equal_source_weights(cohort.video_id.to_numpy())
    aggregates, features = [], []
    for (arm, seed), (target, baseline, full, valid) in arrays.items():
        score, rbase, rfull, mask = score_arrays(target, baseline, full, weights, valid)
        aggregates.append(
            {
                "arm": arm,
                "seed": seed,
                "target": "background" if arm == "background-target" else "person",
                **score,
            }
        )
        for feature in range(256):
            features.append(
                {
                    "arm": arm,
                    "seed": seed,
                    "target_feature": feature,
                    "valid": bool(mask[feature]),
                    "r2_baseline": rbase[feature],
                    "r2_full": rfull[feature],
                    "delta_r2": rfull[feature] - rbase[feature],
                }
            )
    bootstrap = []
    for draw, indices in enumerate(
        source_bootstrap_indices(
            cohort.video_id.to_numpy(), config.bootstrap_repetitions
        )
    ):
        seed_scores = {}
        for (arm, seed), (target, base, full, valid) in arrays.items():
            # Carry each original weight with each sampled occurrence. Recomputing
            # source counts here would incorrectly cancel bootstrap multiplicity.
            score, _, _, _ = score_arrays(
                target[indices], base[indices], full[indices], weights[indices], valid
            )
            seed_scores[(arm, seed)] = score
        for arm in ARMS:
            record = {"draw": draw, "arm": arm}
            for metric in ("r2_baseline", "r2_full", "delta_r2", "f8"):
                values = [seed_scores[(arm, seed)][metric] for seed in config.seeds]
                record[metric] = (
                    float(np.mean(values))
                    if all(v is not None for v in values)
                    else None
                )
            bootstrap.append(record)
    aggregate_frame = pd.DataFrame(aggregates)
    bootstrap_frame = pd.DataFrame(bootstrap)
    summary, paired = {}, []
    real_draws = bootstrap_frame.loc[
        bootstrap_frame.arm == "real-skeleton", "delta_r2"
    ].to_numpy()
    for arm in ARMS:
        selected = bootstrap_frame[bootstrap_frame.arm == arm]
        summary[arm] = {}
        for metric in ("r2_baseline", "r2_full", "delta_r2", "f8"):
            values = selected[metric].to_numpy(dtype=float)
            finite = np.isfinite(values)
            summary[arm][metric] = {
                "ci025": float(np.quantile(values[finite], 0.025))
                if finite.any()
                else None,
                "ci975": float(np.quantile(values[finite], 0.975))
                if finite.any()
                else None,
                "defined_draws": int(finite.sum()),
            }
        if arm != "real-skeleton":
            diff = real_draws - selected.delta_r2.to_numpy()
            point = aggregate_frame.groupby("arm").delta_r2.mean()
            paired.append(
                {
                    "control": arm,
                    "real_minus_control": float(point["real-skeleton"] - point[arm]),
                    "ci025": float(np.quantile(diff, 0.025)),
                    "ci975": float(np.quantile(diff, 0.975)),
                    "real_beats_control_fraction": float(np.mean(diff > 0)),
                }
            )
    report_files = {
        "reports/featurewise-r2.csv": pd.DataFrame(features),
        "reports/aggregate-metrics.csv": aggregate_frame,
        "reports/source-bootstrap.csv": bootstrap_frame,
        "reports/paired-controls.csv": pd.DataFrame(paired),
    }
    for path, frame in report_files.items():
        atomic_write_dataframe_csv(root / path, frame)
    write_once_json(
        root / "reports/uncertainty.json",
        {
            "intervals": summary,
            "bootstrap_positive_fraction": float(np.mean(real_draws > 0)),
            "seed_aggregation": config.seed_aggregation,
            "bootstrap_repetitions": config.bootstrap_repetitions,
        },
    )
    saved_files = [*report_files, "reports/uncertainty.json"]
    predictions["y_true_raw"] = (
        predictions.y_true * predictions.target_scale + predictions.target_mean
    )
    predictions["y_pred_baseline_raw"] = (
        predictions.y_pred_baseline * predictions.target_scale + predictions.target_mean
    )
    predictions["y_pred_full_raw"] = (
        predictions.y_pred_full * predictions.target_scale + predictions.target_mean
    )
    for arm in ARMS:
        path = f"predictions/{arm}.parquet"
        predictions[predictions.arm == arm].to_parquet(root / path, index=False)
        saved_files.append(path)
    baseline = predictions[
        predictions.arm.isin(["real-skeleton", "background-target"])
    ].drop(columns=["y_pred_full", "y_pred_full_raw"])
    baseline.assign(arm="baseline").to_parquet(
        root / "predictions/baseline.parquet", index=False
    )
    saved_files.append("predictions/baseline.parquet")
    write_once_json(
        root / "reports/scores-contract.json",
        {
            "binding": scores_binding(root),
            "artifacts": {path: sha256_file(root / path) for path in saved_files},
            "controls_complete": True,
            "evaluation_contract_valid": True,
            "target_variance_valid": True,
        },
    )


def build_report(root):
    root = Path(root)
    run = check_run(root)
    if (root / "reports/final-report-contract.json").exists():
        sealed = read_json(root / "reports/final-report-contract.json")
        for path, digest in sealed["artifacts"].items():
            if sha256_file(root / path) != digest:
                raise ValueError("Sealed final report changed")
        return read_json(root / "reports/gate-decision.json")
    thresholds = GateThresholds(**read_json(root / "config/thresholds.json"))
    config = load_model_contract(root)
    score_frame, paired, uncertainty = None, None, None
    try:
        audit = require_audits(root)
        cohort, cache = load_cache(root)
        verify_scores(root)
        # Independently verify underlying predictions again, not just summary flags.
        assemble_oof(
            cohort,
            cache,
            pd.concat(
                [verify_fold(root, fold) for fold in range(5)], ignore_index=True
            ),
            config,
        )
        score_frame = pd.read_csv(root / "reports/aggregate-metrics.csv")
        paired = pd.read_csv(root / "reports/paired-controls.csv")
        uncertainty = read_json(root / "reports/uncertainty.json")
        point = score_frame.groupby("arm").delta_r2.mean()
        metrics = {
            "delta_r2_real": float(point["real-skeleton"]),
            "delta_r2_time_shuffle": float(point["time-shuffle"]),
            "delta_r2_clip_mismatch": float(point["clip-mismatch"]),
            "delta_r2_background_target": float(point["background-target"]),
            "delta_r2_no_skeleton": float(point["no-skeleton"]),
            "seed_real_gains": [
                float(
                    score_frame.loc[
                        (score_frame.arm == "real-skeleton")
                        & (score_frame.seed == seed),
                        "delta_r2",
                    ].iloc[0]
                )
                for seed in config.seeds
            ],
            "motion_to_background_change_ratio": audit[
                "motion_to_background_change_ratio"
            ],
            "person_edit_direction_fraction": audit["person_edit_direction_fraction"],
            "bootstrap_positive_fraction": uncertainty["bootstrap_positive_fraction"],
            "evaluation_contract_valid": True,
            "controls_complete": True,
            "target_variance_valid": True,
            **{
                key: audit["checks"][key]
                for key in (
                    "data_contract_valid",
                    "target_audit_complete",
                    "teacher_stable",
                    "causal_leakage_absent",
                )
            },
        }
        result = decide_gate(metrics, thresholds, len(config.seeds))
        reason = (
            "; ".join(name for name, passed in result["checks"].items() if not passed)
            or "All preregistered checks passed."
        )
        stage = "all five outer folds scored"
    except (ValueError, FileNotFoundError, KeyError) as error:
        result = {
            "decision": "STOP",
            "allow_full_experiment": False,
            "allow_adapter_training": False,
            "checks": {"complete_valid_evidence": False},
            "thresholds": asdict(thresholds),
            "metrics": None,
        }
        reason = str(error)
        stage = (
            "incomplete or invalid measurement; scientific hypothesis not established"
        )
    if run["synthetic"]:
        result.update(
            decision="STOP", allow_full_experiment=False, allow_adapter_training=False
        )
        result["checks"]["real_data_required"] = False
        reason = (
            "Synthetic implementation smoke test; contains no experimental evidence. "
            + reason
        )
    teacher = read_json(root / "config/teacher-contract.json")
    manifest = root / "manifests/gate-windows.csv"
    result.update(
        experiment=run["experiment"],
        run_id=run["run_id"],
        stage=stage,
        reason=reason,
        manifest_sha256=sha256_file(manifest) if manifest.exists() else None,
        vjepa_commit=teacher["commit"],
        checkpoint_sha256=teacher["checkpoint_sha256"],
        code_sha256=run["code_sha256"],
        synthetic=run["synthetic"],
    )
    write_once_json(root / "reports/gate-decision.json", result)
    lines = [
        f"# Experiment 0: {result['decision']}",
        "",
        reason,
        "",
        f"Run: `{root.name}`. Stage: {stage}.",
        "",
        f"Planned: 50 windows, at least 25 sources, five source-disjoint outer folds, seeds {list(config.seeds)}, all five arms.",
        "The 8-frame horizon locates a full-clip contextual target at frames 38–39; predictors only see frames 0–31.",
        "Annotated endpoints are allowed offline metadata. Source-disjoint evaluation does not establish participant-disjoint evaluation.",
        "",
    ]
    if manifest.exists():
        cohort = pd.read_csv(manifest)
        lines += [
            f"Cohort: {len(cohort)} windows from {cohort.video_id.nunique()} sources. Elapsed horizons range from {cohort.horizon_seconds.min():.4f} to {cohort.horizon_seconds.max():.4f} seconds.",
            "",
        ]
    lines += ["| Check | Passed |", "|---|---|"] + [
        f"| {name} | {passed} |" for name, passed in result["checks"].items()
    ]
    lines += ["", "Frozen thresholds:", ""] + [
        f"- {name}: {value}" for name, value in asdict(thresholds).items()
    ]
    if score_frame is not None:
        lines += [
            "",
            "Scores use predictive R² against each fold's training mean, uniform valid-feature averages, and equal total weight per source. Seeds are averaged as scores, not as predictions.",
            "",
            "| Arm | Target | Seed | Baseline R² | Full R² | ΔR² | F8 | Valid features |",
            "|---|---|---:|---:|---:|---:|---:|---:|",
        ]
        for row in score_frame.itertuples():
            f8 = f"{row.f8:.4f}" if pd.notna(row.f8) else "undefined"
            lines.append(
                f"| {row.arm} | {row.target} | {row.seed} | {row.r2_baseline:.4f} | {row.r2_full:.4f} | {row.delta_r2:.4f} | {f8} | {row.valid_feature_count} |"
            )
        lines += [
            "",
            "The background arm has its own background-target ridge fit. Background tokens still attend to person tokens.",
            "",
            "| Arm | Mean ΔR² | Source bootstrap 95% interval |",
            "|---|---:|---|",
        ]
        for arm in ARMS:
            interval = uncertainty["intervals"][arm]["delta_r2"]
            lines.append(
                f"| {arm} | {score_frame.loc[score_frame.arm == arm, 'delta_r2'].mean():.4f} | [{interval['ci025']:.4f}, {interval['ci975']:.4f}] |"
            )
        lines += [
            "",
            f"Positive real gain in {uncertainty['bootstrap_positive_fraction']:.1%} of paired source draws.",
            "",
            "| Control | Real minus control | 95% interval | Fraction real wins |",
            "|---|---:|---|---:|",
        ]
        for row in paired.itertuples():
            lines.append(
                f"| {row.control} | {row.real_minus_control:.4f} | [{row.ci025:.4f}, {row.ci975:.4f}] | {row.real_beats_control_fraction:.3f} |"
            )
        if score_frame.baseline_ceiling_warning.any():
            lines += [
                "",
                "Some baseline R² values are at least 0.95: little or no room remains for the required absolute gain. The baseline was not weakened.",
            ]
    audit_path = root / "qc/target-sensitivity.csv"
    if audit_path.exists():
        audit_rows = pd.read_csv(audit_path)
        lines += [
            "",
            "| Audit window | Person change | Background change | Person larger |",
            "|---|---:|---:|---|",
        ]
        for row in audit_rows.itertuples():
            lines.append(
                f"| {row.window_id} | {row.person_change:.4f} | {row.background_change:.4f} | {row.person_larger} |"
            )
    lines += [
        "",
        "Pixel edits use a different-source person region with reversed future timing and an inpainted static donor background, with matched inward feathering. Appearance changes and editing artifacts can contribute to sensitivity; this audit does not prove motion understanding.",
        "",
        (
            "Complete feature distributions, seed metrics, paired controls, bootstrap draws, fitted preprocessing, checkpoints, split audits, and raw/standardized predictions are saved beside this report."
            if score_frame is not None
            else "This run lacks a complete valid prediction set. Only completed stages and available failure evidence are saved; missing stages provide no scientific result."
        ),
        "",
        f"Full measurement experiment allowed: {result['allow_full_experiment']}. Adapter training allowed: false (requires the later frozen-S-JEPA result).",
        "A failed validity audit requires data/measurement repair in a new versioned run. A valid null result stops this gate; an inconclusive result calls for a larger preregistered confirmation.",
        "",
    ]
    report_path = root / "reports/gate-report.md"
    if report_path.exists():
        if report_path.read_text() != "\n".join(lines):
            raise ValueError("Final report already exists and differs")
    else:
        temporary = report_path.with_suffix(".tmp")
        temporary.write_text("\n".join(lines))
        temporary.replace(report_path)
    write_once_json(
        root / "reports/final-report-contract.json",
        {
            "artifacts": {
                path: sha256_file(root / path)
                for path in ("reports/gate-decision.json", "reports/gate-report.md")
            }
        },
    )
    return result
