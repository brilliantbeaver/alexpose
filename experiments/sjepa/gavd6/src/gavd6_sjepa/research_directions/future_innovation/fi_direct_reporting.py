"""Report the direct skeleton increment and a bounded next step toward JEPA training."""

from pathlib import Path
import shutil

import pandas as pd

from .fi_contracts import (DIRECT_PROTOCOL, GateThresholds, audit_summary_path, check_run, code_fingerprint,
                           experiment_arms, load_model_contract, measurement_complete, read_json,
                           sha256_file, verified_report_decision, write_json, write_once_json)
from .fi_feature_cache import load_cache
from .fi_gate_decision import decide_gate
from .fi_nested_training import verify_fold
from .fi_validity_audits import ValidityAuditRejected, require_audits


def build_direct_report(root):
    from .fi_reporting import assemble_oof, verify_scores
    root = Path(root)
    run = check_run(root)
    previous = verified_report_decision(root)
    seal = root / "reports/final-report-contract.json"
    if previous is not None and measurement_complete(previous) and seal.exists():
        if previous.get("protocol") != DIRECT_PROTOCOL or previous.get("run_id") != run["run_id"]:
            raise ValueError("Sealed result belongs to a different protocol/run")
        return previous
    thresholds = GateThresholds(**read_json(root / "config/thresholds.json"))
    model, arms = load_model_contract(root), experiment_arms(root)
    complete = False
    frame = paired = cohort = None
    try:
        readiness = require_audits(root)
        cohort, cache = load_cache(root)
        verify_scores(root)
        assemble_oof(cohort, cache, pd.concat([verify_fold(root, f) for f in range(5)], ignore_index=True), model, arms=arms)
        frame = pd.read_csv(root / "reports/aggregate-metrics.csv")
        paired = pd.read_csv(root / "reports/paired-controls.csv").set_index("control")
        uncertainty = read_json(root / "reports/uncertainty.json")
        points = frame.groupby("arm").delta_r2.mean()
        seeds = frame.pivot(index="seed", columns="arm", values="delta_r2").loc[list(model.seeds)]
        metrics = {"delta_r2_real": float(points["real-skeleton"]),
                   "delta_r2_time_shuffle": float(points["time-shuffle"]),
                   "delta_r2_clip_mismatch": float(points["clip-mismatch"]),
                   "delta_r2_no_skeleton": float(points["no-skeleton"]),
                   "seed_real_gains": seeds["real-skeleton"].tolist(),
                   "seed_skeleton_increments": (seeds["real-skeleton"] - seeds["no-skeleton"]).tolist(),
                   "bootstrap_positive_fraction": uncertainty["bootstrap_positive_fraction"],
                   "skeleton_increment_positive_fraction": float(paired.loc["no-skeleton", "real_beats_control_fraction"]),
                   "evaluation_contract_valid": True, "controls_complete": True, **readiness["checks"]}
        result = decide_gate(metrics, thresholds, len(model.seeds), protocol=DIRECT_PROTOCOL)
        result["skeleton_increment"] = {"estimate": float(paired.loc["no-skeleton", "real_minus_control"]),
                                         "ci025": float(paired.loc["no-skeleton", "ci025"]),
                                         "ci975": float(paired.loc["no-skeleton", "ci975"]),
                                         "per_seed": metrics["seed_skeleton_increments"]}
        complete = True
        reason = "; ".join(k for k, v in result["checks"].items() if not v) or "All direct-prediction criteria passed."
    except (ValueError, OSError, KeyError) as error:
        result = {"decision": "STOP", "allow_full_experiment": False, "allow_jepa_training_comparison": False,
                  "allow_adapter_training": False, "metrics": None, "checks": {"complete_valid_evidence": False},
                  "next_action": "repair_missing_or_invalid_evidence"}
        reason = str(error)
        if isinstance(error, ValidityAuditRejected):
            result.update(workflow_status="blocked_by_validity_audit", audit_summary_sha256=error.summary_sha256,
                          checks={**error.summary["checks"], "complete_valid_evidence": False})
    if run["synthetic"]:
        result.update(decision="STOP", allow_full_experiment=False, allow_jepa_training_comparison=False,
                      next_action="synthetic_implementation_check_only")
        reason = "Synthetic implementation test; no real prediction finding. " + reason
    manifest = root / "manifests/gate-windows.csv"
    readiness_path = audit_summary_path(root)
    teacher = read_json(root / "config/teacher-contract.json")
    result.update(checkpoint_sha256=teacher["checkpoint_sha256"],
                  manifest_sha256=sha256_file(manifest) if manifest.exists() else None,
                  readiness_sha256=sha256_file(readiness_path) if readiness_path.exists() else None,
                  protocol_sha256=sha256_file(root / "config/protocol-contract.json"),
                  run_contract_sha256=sha256_file(root / "config/run-contract.json"),
                  config_sha256=run["config_sha256"], initial_code_sha256=run["code_sha256"])
    result.update(protocol=DIRECT_PROTOCOL, experiment=run["experiment"], run_id=run["run_id"],
                  measurement_complete=complete, synthetic=run["synthetic"], reason=reason,
                  stage="all source-held-out comparisons scored" if complete else "incomplete predictive measurement",
                  cohort_windows=len(cohort) if complete else None,
                  cohort_sources=cohort.video_id.nunique() if complete else None,
                  report_code_sha256=code_fingerprint())
    lines = [f"# Experiment 0 direct prediction: {result['decision']}", "", reason, "",
             "Question: does skeleton coordinate/confidence history improve prediction beyond RGB, nuisance inputs, and a matched head retaining observation-validity masks?",
             "This protocol was specified after the legacy sensitivity rejection and before the new prediction results. It removes pixel-edit and background-selectivity gates.",
             "Inputs use frames 0–31. The teacher target at frames 38–39 is contextualized by the full clip; the score concerns those features, not decoded future movement.", ""]
    if complete:
        inc = result["skeleton_increment"]
        lines += [f"Cohort: {len(cohort)} windows from {cohort.video_id.nunique()} source videos; five outer folds and seeds {list(model.seeds)}.",
                  "The gate uses 50 clips selected from the full-GAVD candidate pool, with at most two clips per source. The full dataset is reserved for the subsequent real experiment. Source holdout does not establish participant independence.", "",
                  f"**Skeleton increment over the matched no-skeleton head: ΔR² = {inc['estimate']:.4f}, paired source-bootstrap 95% interval [{inc['ci025']:.4f}, {inc['ci975']:.4f}].**",
                  "An interval crossing zero leaves the increment uncertain. The decision uses the separately specified fraction of positive draws and seed checks.", "",
                  "| Arm | Seed | RGB+nuisance ridge R² | Full head R² | Gain over ridge |", "|---|---:|---:|---:|---:|"]
        for row in frame.itertuples():
            lines.append(f"| {row.arm} | {row.seed} | {row.r2_baseline:.4f} | {row.r2_full:.4f} | {row.delta_r2:.4f} |")
        lines += ["", "| Control | Real minus control ΔR² | Paired source-bootstrap 95% interval |", "|---|---:|---|"]
        for control, row in paired.iterrows():
            lines.append(f"| {control} | {row.real_minus_control:.4f} | [{row.ci025:.4f}, {row.ci975:.4f}] |")
        lines += ["", f"Per-seed skeleton increments, in seed order: {', '.join(f'{v:.4f}' for v in inc['per_seed'])}.",
                  "All preprocessing and model selection use training sources only. Scores balance sources and average featurewise R²; seeds are averaged as scores. Bootstrap intervals resample saved out-of-fold predictions and are conditional on those fits; model selection and training are not rerun within each draw. Repeated seeds and bootstrap draws do not add independent sources."]
    lines += ["", "| Criterion | Passed |", "|---|---|"]
    lines += [f"| {k} | {v} |" for k, v in result["checks"].items()]
    lines += ["", f"Next action: **{result['next_action']}**.",
              "ADVANCE recommends designing the full-GAVD experiment to compare trained JEPA representations against matched initial encoders and raw skeleton inputs. It does not launch training. This experiment alone establishes neither JEPA benefit nor clinical validity or isolated gait dynamics. Adapter distillation remains a separate decision.",
              "Unfiltered recording conditions can contribute to the contextual target. Background quality was not screened or certified. The matched no-skeleton control retains validity flags, and nuisance inputs include pose support; the measured increment concerns the additional coordinate/confidence history.", ""]
    lines += ["", "## Artifact identity", "",
              f"Checkpoint SHA-256: `{result['checkpoint_sha256']}`.",
              f"Gate manifest SHA-256: `{result['manifest_sha256']}`.",
              f"Readiness SHA-256: `{result['readiness_sha256']}`.",
              f"Protocol SHA-256: `{result['protocol_sha256']}`.",
              f"Run contract SHA-256: `{result['run_contract_sha256']}`.",
              f"Report code SHA-256: `{result['report_code_sha256']}`.",
              "The decision JSON retains all frozen configuration hashes and the initialization code hash.", ""]
    if previous is not None and not measurement_complete(previous):
        archive = root / "reports/attempts" / sha256_file(root / "reports/gate-decision.json")
        archive.mkdir(parents=True, exist_ok=True)
        for name in ("gate-decision.json", "gate-report.md", "final-report-contract.json"):
            source = root / "reports" / name
            if source.exists():
                shutil.copyfile(source, archive / name)
        seal.unlink(missing_ok=True)
    write_json(root / "reports/gate-decision.json", result)
    report = root / "reports/gate-report.md"
    temporary = report.with_suffix(".tmp")
    temporary.write_text("\n".join(lines))
    temporary.replace(report)
    if complete:
        write_once_json(seal, {"artifacts": {str(p.relative_to(root)): sha256_file(p)
                                            for p in (root / "reports/gate-decision.json", report)}})
    return result
