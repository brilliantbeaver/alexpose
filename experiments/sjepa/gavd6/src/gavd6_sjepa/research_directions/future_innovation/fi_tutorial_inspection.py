"""Read-only, lightweight views of existing Future Innovation run artifacts.

These views never initialize a run, load a teacher, fit a model, or rebuild a
report. Artifact presence describes this local copy, not a live Slurm job.
"""

from pathlib import Path

import pandas as pd

from .fi_contracts import measurement_complete, read_json, verified_report_decision


STAGE_ARTIFACTS = (
    ("Setup", "config/run-contract.json"),
    ("Candidates", "config/candidates-contract.json"),
    ("Aligned cohort", "config/cohort-contract.json"),
    ("Teacher cache", "config/cache-contract.json"),
    ("Validity audits", "qc/validity-summary.json"),
    *((f"Outer fold {fold}", f"models/fold-{fold}/fold-complete.json") for fold in range(5)),
    ("Gate decision", "reports/gate-decision.json"),
    ("Sealed report", "reports/final-report-contract.json"),
)


def artifact_inventory(run_root):
    """List completion-record presence without certifying the underlying stage."""
    root = Path(run_root)
    return pd.DataFrame(
        [
            {"stage": stage, "record": name, "present_locally": (root / name).is_file()}
            for stage, name in [(stage, str(inspection_audit_path(root).relative_to(root))
                                  if name == "qc/validity-summary.json" else name)
                                 for stage, name in STAGE_ARTIFACTS]
        ]
    )


def inspection_audit_path(root):
    """Choose the saved protocol without invoking execution provenance writers."""
    root = Path(root)
    path = root / "config/run-contract.json"
    protocol = read_json(path).get("protocol", "legacy-v1") if path.is_file() else "direct-v2"
    return root / "qc" / ("readiness-summary.json" if protocol == "direct-v2" else "validity-summary.json")


def read_optional_table(run_root, relative_path):
    """Return None only for an absent file; malformed existing tables must fail."""
    path = Path(run_root) / relative_path
    return pd.read_csv(path) if path.is_file() else None


def inspect_report(run_root):
    """Describe saved evidence using the pipeline's existing report verification.

    A complete report needs the decision and narrative in its seal. Hash checks
    establish report integrity only; raw inputs and predictions are not rerun.
    Incomplete reports can be inspected without all remote inputs being mounted.
    """
    root = Path(run_root)
    result = {
        "state": "UNAVAILABLE",
        "explanation": "No gate decision is available in this local copy. Inspect the inventory or copy the run reports from HAIC.",
        "decision": None,
        "report_text": None,
        "seal_verified": False,
    }
    try:
        decision = verified_report_decision(root)
        if decision is None:
            return result
        if not isinstance(decision, dict) or decision.get("decision") not in {
            "ADVANCE", "STOP", "INCONCLUSIVE"
        }:
            raise ValueError("Unrecognized gate decision record")
        complete = measurement_complete(decision)
        if not isinstance(complete, bool):
            raise ValueError("Measurement completeness must be boolean")
        seal = root / "reports/final-report-contract.json"
        if seal.is_file():
            artifacts = read_json(seal)["artifacts"]
            required = {"reports/gate-decision.json", "reports/gate-report.md"}
            if not required.issubset(artifacts):
                raise ValueError("Report seal omits its decision or narrative")
            result["seal_verified"] = True
        result["decision"] = decision
        if not complete:
            result.update(
                state="INCOMPLETE",
                explanation="The saved STOP describes incomplete or invalid measurement. It is not a negative finding about skeleton information.",
            )
        elif decision.get("synthetic") is True:
            result.update(
                state="SYNTHETIC",
                explanation="Software demonstration only. It cannot support a scientific conclusion or authorize training.",
            )
        elif not result["seal_verified"] or decision.get("synthetic") is not False:
            result.update(
                state="UNVERIFIED",
                explanation="A complete real result needs an explicit real-data flag and a verified report seal. No advancement is inferred here.",
            )
        else:
            result.update(
                state=f"COMPLETE / {decision['decision']}",
                explanation="The saved real report is complete and its seal matches. Read its checks and limitations; the notebook has not rerun scientific validation.",
            )
        if result["seal_verified"]:
            result["report_text"] = (root / "reports/gate-report.md").read_text()
        return result
    except (OSError, ValueError, KeyError, TypeError, AttributeError) as error:
        result.update(
            state="INVALID",
            explanation=f"Cannot verify the saved report: {error}",
            decision=None,
            report_text=None,
            seal_verified=False,
        )
        return result
