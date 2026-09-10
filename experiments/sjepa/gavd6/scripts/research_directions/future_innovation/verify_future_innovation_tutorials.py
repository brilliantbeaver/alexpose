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


def verify_pipeline_smoke(output):
    """Test real CLI routing and nested fits; never substitute synthetic data in a real run."""
    sys.path.insert(0, str(ROOT / "src"))
    from gavd6_sjepa.research_directions.future_innovation.fi_smoke import synthetic_cache
    root = synthetic_cache(output / "synthetic-run")
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
    # Resuming a specific array fold must preserve all scientific artifacts.
    watched = [p for name in ("models", "predictions", "reports") for p in (root / name).rglob("*") if p.is_file()]
    before = {p: (p.stat().st_mtime_ns, p.read_bytes()) for p in watched}
    execute("03", outer_fold=3, timeout=180)
    execute("04", timeout=180)
    assert before == {p: (p.stat().st_mtime_ns, p.read_bytes()) for p in watched}
    assert all(p.suffix == ".ipynb" for p in (root / "notebook_runs/manual").iterdir())
    # Reproduce the actual HAIC failure: cache exists but teacher audit fails.
    # Notebook 02 must show diagnostics; 03 must exist but never create fits.
    blocked_root = synthetic_cache(output / "blocked-synthetic-run")
    summary_path = blocked_root / "qc/validity-summary.json"
    summary = json.loads(summary_path.read_text())
    summary["passed"] = False
    summary["checks"]["target_sensitivity"] = False
    summary_path.write_text(json.dumps(summary))
    for number, message in (("02", "Failed checks:"), ("03", "FITTING BLOCKED"), ("04", "INCOMPLETE")):
        try:
            execute_notebook(number, mode="execute", run_root=blocked_root, device="cpu", timeout=180,
                             output_parent=blocked_root / "notebook_runs/manual")
        except CellExecutionError:
            notebook = nbformat.read(blocked_root / "notebook_runs/manual" / NAMES[number], as_version=4)
            assert notebook.metadata.fi_execution.status == "failed"
            assert any(message in o.get("text", "") for c in notebook.cells for o in c.get("outputs", []))
        else:
            raise AssertionError(f"Notebook {number} must preserve the failed audit boundary")
    assert not list(blocked_root.glob("models/fold-*/*.json"))
    print(f"PASS synthetic execution, incomplete-report failure, and sealed resume: {root}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--only", choices=tuple(NAMES), nargs="+", default=list(NAMES))
    parser.add_argument("--mode", choices=("teach", "inspect"), default="teach")
    parser.add_argument("--run-root", type=Path)
    parser.add_argument("--output-parent", type=Path,
                        default=ROOT / "work/artifacts/notebook_runs/future_innovation")
    parser.add_argument("--pipeline-smoke", action="store_true",
                        help="Also test execution/failure/resume on a separate synthetic cache (no real teacher).")
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
            verify_pipeline_smoke(output)
        report["status"] = "passed"
    except BaseException:
        report["status"] = "failed"
        raise
    finally:
        (output / "verification.json").write_text(json.dumps(report, indent=2) + "\n")
        print(f"Verification record: {output}", flush=True)


if __name__ == "__main__":
    main()
