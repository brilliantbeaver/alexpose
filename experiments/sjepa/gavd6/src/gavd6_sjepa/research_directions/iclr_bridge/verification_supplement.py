"""Read-only post-fit verification hardening for the frozen accessibility pilot.

The original fitting implementation and sealed artifacts remain unchanged.
This supplement checks report metadata, the seal inventory, displayed selection
fields and fitted-state diagnostics that the original reconstruction omitted.
It adds no fitting choices or scientific comparisons.
"""
from pathlib import Path
import math

import numpy as np

from .cached_panel import (
    PANELS, VERSION, _compare_json, _load_frozen, _panel_arrays, _reload_fold,
    specification, verify_cached_panel,
)
from ..future_innovation.fi_contracts import DIRECT_ARMS, read_json, sha256_file
from ..future_innovation.fi_joint_models import JointModelContract, temporal_features
from ..future_innovation.fi_joint_training import choose_candidate, error_record, predict_selected
from ..future_innovation.fi_nested_training import partition_control

SUPPLEMENT_VERSION = "accessibility-verification-supplement-v1"


def expected_sealed_artifacts(contract):
    """The exact artifact schema produced by the frozen pilot writer."""
    names = {
        "config/specification.json", "config/source-snapshot.json",
        "config/parent-snapshot.json", "config/run-contract.json",
        "config/support-schema.json", "config/posture-schema.json",
        "manifests/cohort.csv", "predictions/support-oof.npz",
        "predictions/posture-oof.npz", "predictions/oof.parquet",
        "reports/support-bootstrap.csv", "reports/posture-bootstrap.csv",
        "reports/selections.csv", "reports/panel-report.json",
        "reports/numerical-verification.json",
    }
    if "protocol_document_sha256" in contract:
        names.add("config/frozen-protocol.md")
    for panel in PANELS:
        for fold in range(5):
            prefix = f"models/{panel}/fold-{fold}/"
            names |= {prefix + name for name in (
                "baseline.joblib", "complete.json", "selection.json",
                "baseline_selection.json", "split.json", "preprocessing.json",
                "diagnostics.json",
            )}
            names |= {prefix + f"{arm}.joblib" for arm in DIRECT_ARMS}
    return names


def _metadata_checks(root, contract, cohort, original):
    seal = read_json(root / "reports/completion.json")
    if set(seal) != {"version", "measurement_complete", "artifacts"}:
        raise ValueError("Completion seal schema changed")
    if seal["version"] != VERSION or type(seal["measurement_complete"]) is not bool:
        raise ValueError("Invalid completion seal version/completeness")
    if seal["measurement_complete"] != original["measurement_complete"]:
        raise ValueError("Completion seal contradicts reconstructed completeness")
    if set(seal["artifacts"]) != expected_sealed_artifacts(contract):
        raise ValueError("Completion seal omits or adds an artifact")
    report = read_json(root / "reports/panel-report.json")
    if set(report) != {"version", "measurement_complete", "status", "scientific_advance",
                       "panels", "wall_seconds", "sources", "clips", "interpretation"}:
        raise ValueError("Top-level report schema changed")
    if (report["version"] != VERSION or report["sources"] != cohort.video_id.nunique()
            or report["clips"] != len(cohort) or set(report["panels"]) != set(PANELS)
            or report["interpretation"] != specification()["interpretation"]
            or type(report["measurement_complete"]) is not bool
            or report["scientific_advance"] is not False):
        raise ValueError("Report metadata contradicts the frozen experiment")
    if not isinstance(report["wall_seconds"], (int, float)) or not math.isfinite(report["wall_seconds"]) or report["wall_seconds"] < 0:
        raise ValueError("Invalid reported wall time")


def verify_panel_supplement(output_root):
    """Compose original numerical reconstruction with the additional checks.

    No file is created, modified or rehashed in place. Callers may save the
    returned evidence in their own new output directory, outside the sealed run.
    Wall time is checked for validity only; timings cannot be independently
    recovered from a fitted model.
    """
    root = Path(output_root)
    original = verify_cached_panel(root)
    contract, cohort, arrays = _load_frozen(root)
    _metadata_checks(root, contract, cohort, original)
    candidate_count, diagnostic_count = 0, 0
    for panel in PANELS:
        local, _ = _panel_arrays(cohort, arrays, panel)
        for fold in range(5):
            directory = root / "models" / panel / f"fold-{fold}"
            result = _reload_fold(root, panel, fold, cohort, arrays)
            selection = read_json(directory / "selection.json")
            train = np.flatnonzero(cohort.outer_fold.to_numpy() != fold)
            base = result["baseline"]
            expected_diagnostics = []
            for arm in DIRECT_ARMS:
                candidates = selection[arm]["candidates"]
                winner, reason = choose_candidate(candidates, JointModelContract())
                reference = next(c for c in candidates if c["candidate_type"] == "baseline_only")
                for candidate in candidates:
                    selected = candidate["candidate_id"] == winner["candidate_id"]
                    improvement = reference["pooled_loss"] - candidate["pooled_loss"] if candidate["valid"] else None
                    expected_reason = reason if selected else (
                        "rejected: invalid numerical fit" if not candidate["valid"]
                        else "rejected by pooled inner loss/tie rule")
                    if type(candidate["selected"]) is not bool:
                        raise ValueError("Candidate selected flag is not boolean")
                    _compare_json({k: candidate[k] for k in ("improvement_over_baseline", "selected", "selection_reason")},
                                  {"improvement_over_baseline": improvement, "selected": selected,
                                   "selection_reason": expected_reason})
                    candidate_count += 1
                model = result["models"][arm]
                sf = temporal_features(partition_control(arm, train, cohort, local, "supplement/train", []))
                prediction = predict_selected(model, base, local["baseline"][train], sf, features=True)
                mse = error_record(base.y_scaler.transform(arrays["person"][train]), prediction,
                                   cohort.iloc[train].video_id, base.valid_features)["validation_mse"]
                is_joint = model["checkpoint_type"] == "joint_ridge"
                expected_diagnostics.append({
                    "arm": arm, "selected_type": model["checkpoint_type"],
                    "selected_candidate": model["candidate_id"],
                    "x_features": len(base.x_scaler.mask), "x_supported": int(base.x_scaler.mask.sum()),
                    "s_features": sf.shape[1], "s_supported": int(model["s_scaler"].mask.sum()),
                    "coefficient_count": (local["baseline"].shape[1] + (sf.shape[1] if is_joint else 0) + 1) * arrays["person"].shape[1],
                    "training_mse": mse, "selected_inner_mse": winner["pooled_loss"],
                    "loss_state": "closed_form; training outer fit, validation inner fits",
                    "complete": all(c["valid"] for c in candidates),
                })
                diagnostic_count += 1
            actual = read_json(directory / "diagnostics.json")
            if len(actual) != len(expected_diagnostics):
                raise ValueError("Wrong diagnostic record count")
            for saved, expected in zip(actual, expected_diagnostics):
                _compare_json(saved, expected)
    return {
        "status": "passed", "supplement_version": SUPPLEMENT_VERSION,
        "supplement_code_sha256": sha256_file(Path(__file__)),
        "original_verification": original,
        "checked_candidate_display_records": candidate_count,
        "checked_fitted_state_diagnostics": diagnostic_count,
        "checks": {"top_level_report_metadata": True, "exact_completion_artifact_schema": True,
                   "completion_consistency": True, "candidate_improvement_flags_reasons": True,
                   "outer_training_losses_and_dimensions": True,
                   "all_inner_outer_preprocessing": "reconstructed by original verifier",
                   "wall_time": "finite nonnegative value; elapsed timing is not independently reproducible"},
        "frozen_fitting_implementation_unchanged": True,
        "sealed_run_unchanged": True, "read_only": True,
    }
