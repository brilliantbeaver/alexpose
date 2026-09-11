"""Verify fresh tutorial kernels, optionally exercising execution on synthetic caches."""

import argparse
import json
from pathlib import Path
import sys
import tempfile
import nbformat

from nbclient.exceptions import CellExecutionError

from build_future_innovation_notebooks import NAMES, ROOT
from execute_future_innovation_notebook import execute_notebook


def verify_pipeline_smoke(output, protocol):
    """Test real CLI routing and nested fits; never substitute synthetic data in a real run."""
    sys.path.insert(0, str(ROOT / "src"))
    from gavd6_sjepa.research_directions.future_innovation.fi_smoke import synthetic_cache, synthetic_audit
    root = synthetic_cache(output / f"{protocol}-synthetic-run", protocol=protocol)
    def execute(number, **kwargs):
        return execute_notebook(number, mode="execute", run_root=root,
                                output_parent=root / "notebook_runs/manual", **kwargs)
    # No predictions yet: notebook 04 must retain a failed attempt and a diagnostic STOP.
    try:
        execute("04", timeout=180)
    except CellExecutionError:
        decision = json.loads((root / "reports/gate-decision.json").read_text())
        assert not decision["measurement_complete"]
        failed = root / "notebook_runs/manual" / NAMES["04"]
        assert nbformat.read(failed, as_version=4).metadata.fi_execution.status == "failed"
    else:
        raise AssertionError("An incomplete experiment must fail notebook execution.")
    # Inputs, cohort, features and audits are fabricated by the existing fixture.
    # Notebook 01 extraction needs real media/MediaPipe and is tested separately
    # through CLI-routing and data-flow tests. No teacher is loaded here.
    for number in ("00", "02", "03", "04"):
        execute(number, device="cpu", timeout=600)
    decision = json.loads((root / "reports/gate-decision.json").read_text())
    assert decision["synthetic"] and decision["measurement_complete"]
    assert decision["decision"] == "STOP"
    assert not decision["allow_full_experiment"] and not decision["allow_adapter_training"]
    assert len(list(root.glob("models/fold-*/fold-complete.json"))) == 5
    assert len(list(root.glob("models/**/residual-head.pt"))) == (60 if protocol == "direct-v2" else 75)
    if protocol == "direct-v2":
        assert decision["cohort_windows"] == 50 and decision["cohort_sources"] == 25
        assert not decision["allow_jepa_training_comparison"]
        assert "skeleton_increment" in decision and decision["readiness_sha256"]
        assert not (root / "qc/target-sensitivity.csv").exists()
        assert not list(root.glob("models/**/background-target"))
    # Resuming a specific array fold must preserve all scientific artifacts.
    watched = [p for name in ("models", "predictions", "reports") for p in (root / name).rglob("*") if p.is_file()]
    before = {p: (p.stat().st_mtime_ns, p.read_bytes()) for p in watched}
    execute("03", outer_fold=3, timeout=180)
    execute("04", timeout=180)
    assert before == {p: (p.stat().st_mtime_ns, p.read_bytes()) for p in watched}
    assert all(p.suffix == ".ipynb" for p in (root / "notebook_runs/manual").iterdir())
    # Reproduce the actual HAIC failure: cache exists but teacher audit fails.
    # Notebook 02 must show diagnostics; 03 must exist but never create fits.
    blocked_root = synthetic_cache(output / f"{protocol}-blocked-synthetic-run", protocol=protocol)
    failed_check = "target_sensitivity"
    if protocol == "legacy-v1":
        synthetic_audit(blocked_root, ratio=1.359831237852072, direction_count=8)
    else:
        import pandas as pd
        from gavd6_sjepa.research_directions.future_innovation.fi_contracts import read_json, write_json, sha256_file
        table = blocked_root / "qc/teacher-stability.csv"
        rows = pd.read_csv(table)
        rows.loc[0, ["target_max_abs", "passed"]] = [0.1, False]
        rows.to_csv(table, index=False)
        summary_path = blocked_root / "qc/readiness-summary.json"
        summary = read_json(summary_path)
        summary["artifacts"]["qc/teacher-stability.csv"] = sha256_file(table)
        summary["checks"]["teacher_stable"] = summary["passed"] = False
        write_json(summary_path, summary)
        failed_check = "teacher_stable"
    jobs = [("02", None), *(("03", fold) for fold in range(5)), ("04", None)]
    for number, fold in jobs:
        destination = execute_notebook(number, mode="execute", run_root=blocked_root,
                                       outer_fold=fold, device="cpu", timeout=180,
                                       output_parent=blocked_root / "notebook_runs/manual")
        notebook = nbformat.read(destination, as_version=4)
        record = notebook.metadata.fi_execution
        assert record.status == "blocked" and record.execution_completed
        assert record.scientific_outcome.measurement_complete is False
        assert record.scientific_outcome.synthetic is True
        assert record.scientific_outcome.failed_checks == [failed_check]
        assert not any(o.output_type == "error" for c in notebook.cells for o in c.get("outputs", []))
    blocked_decision = json.loads((blocked_root / "reports/gate-decision.json").read_text())
    assert blocked_decision["workflow_status"] == "blocked_by_validity_audit"
    assert blocked_decision["measurement_complete"] is False and blocked_decision["metrics"] is None
    assert not blocked_decision["allow_full_experiment"] and not blocked_decision["allow_adapter_training"]
    assert not list(blocked_root.glob("models/fold-*/*.json"))
    if protocol == "direct-v2":
        assert blocked_decision["protocol"] == protocol
        assert not blocked_decision["allow_jepa_training_comparison"]
    print(f"PASS synthetic execution, incomplete-report failure, verified audit rejection, and sealed resume: {root}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--only", choices=tuple(NAMES), nargs="+", default=list(NAMES))
    parser.add_argument("--mode", choices=("teach", "inspect"), default="teach")
    parser.add_argument("--run-root", type=Path)
    parser.add_argument("--output-parent", type=Path,
                        default=ROOT / "work/artifacts/notebook_runs/future_innovation")
    parser.add_argument("--pipeline-smoke", action="store_true",
                        help="Also test execution/failure/resume on a separate synthetic cache (no real teacher).")
    parser.add_argument("--pipeline-protocol", choices=("both", "direct-v2", "legacy-v1"), default="both")
    args = parser.parse_args()
    args.output_parent.mkdir(parents=True, exist_ok=True)
    output = Path(tempfile.mkdtemp(prefix="verification-", dir=args.output_parent))
    report = {"mode": args.mode, "attempts": [], "status": "running"}
    try:
        for number in args.only:
            attempt = execute_notebook(number, mode=args.mode, run_root=args.run_root,
                                       timeout=180, output_parent=output / "notebooks")
            report["attempts"].append(str(attempt))
        if args.pipeline_smoke:
            for protocol in (("direct-v2", "legacy-v1") if args.pipeline_protocol == "both" else (args.pipeline_protocol,)):
                verify_pipeline_smoke(output, protocol)
        report["status"] = "passed"
    except BaseException:
        report["status"] = "failed"
        raise
    finally:
        (output / "verification.json").write_text(json.dumps(report, indent=2) + "\n")
        print(f"Verification record: {output}", flush=True)


if __name__ == "__main__":
    main()
