"""Notebook orchestration of the existing Experiment 0 CLI, without new models."""

from datetime import datetime, timezone
from dataclasses import dataclass
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys
from uuid import uuid4

from .fi_contracts import check_run, measurement_complete, read_json, sha256_file, stage_lock, audit_summary_path, protocol_name, DIRECT_PROTOCOL
from .fi_tutorial_inspection import inspect_report
from .fi_validity_audits import ValidityAuditRejected, require_audits


PROJECT = Path(__file__).resolve().parents[4]
STAGES = frozenset({"init-cached-run", "init-run", "build-cohort", "extract-poses", "cache-teacher",
                    "audit-teacher", "run-gate", "score-gate", "build-report"})
OUTCOME_MIME = "application/vnd.gavd6.notebook-outcome+json"


@dataclass(frozen=True)
class BlockedReport:
    audit: ValidityAuditRejected
    decision_sha256: str
    narrative_sha256: str


def publish_outcome(outcome):
    """Persist scientific status in notebook output, including interactive use."""
    from IPython.display import display
    display({OUTCOME_MIME: outcome, "text/plain": json.dumps(outcome, indent=2)}, raw=True)


def _recheck_rejection(error):
    try:
        require_audits(error.root)
    except ValidityAuditRejected as current:
        if current.summary_sha256 != error.summary_sha256:
            raise ValueError("Audit changed during notebook execution") from current
        return current
    raise ValueError("Rejected audit no longer matches the run")


def _publish_blocked(error):
    _recheck_rejection(error)
    print(f"TRAINING BLOCKED: {error}")
    publish_outcome({"status": "blocked", "execution_completed": True,
                     "scientific_status": "validity_rejected", "measurement_complete": False,
                     "synthetic": check_run(error.root)["synthetic"],
                     "failed_checks": error.failed_checks, "reason": str(error),
                     "audit_summary_sha256": error.summary_sha256})


def run_stage(command, run_root, *options):
    """Stream the production CLI; retain its command, output and failure status."""
    if command not in STAGES:
        raise ValueError(f"Unknown Future Innovation stage: {command}")
    root = Path(run_root).expanduser()
    root = (PROJECT / root).resolve()
    parent = Path(os.environ.get("FI_NOTEBOOK_LOG_DIR", root / "logs/notebooks/manual"))
    parent.mkdir(parents=True, exist_ok=True)
    label = os.environ.get("FI_NOTEBOOK_LABEL", "interactive")
    log_path = parent / f"{label}-{command}-{uuid4().hex[:12]}.log"
    argv = [sys.executable, "-u", "-m", "gavd6_sjepa.command_line_interface",
            "future-innovation", command, *map(str, options), "--run-root", str(root)]
    environment = {**os.environ, "PYTHONPATH": os.pathsep.join(filter(None, (
        str(PROJECT / "src"), os.environ.get("PYTHONPATH")))), "PYTHONUNBUFFERED": "1"}
    # Match fi-common.sh even when an interactive kernel was started without it.
    threads = os.environ.get("FI_TORCH_THREADS", "1")
    controls = {key: threads for key in ("FI_TORCH_THREADS", "OMP_NUM_THREADS",
                                        "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS")}
    controls.update(CUBLAS_WORKSPACE_CONFIG=":4096:8", PYTHONHASHSEED="260905", MPLBACKEND="Agg")
    environment.update(controls)
    receipt = {"argv": argv, "cwd": str(PROJECT), "started_utc": datetime.now(timezone.utc).isoformat(),
               "status": "running", "execution_controls": controls}
    log_path.write_text(json.dumps(receipt) + "\n")
    print(f"$ {shlex.join(argv)}\nStage log: {log_path}", flush=True)
    process = None
    try:
        with log_path.open("a", buffering=1) as log:
            process = subprocess.Popen(argv, cwd=PROJECT, env=environment, stdout=subprocess.PIPE,
                                       stderr=subprocess.STDOUT, text=True, bufsize=1)
            try:
                for line in process.stdout:
                    log.write(line)
                    print(line, end="", flush=True)
                returncode = process.wait()
            finally:
                if process.poll() is None:
                    process.terminate()
                    try:
                        process.wait(timeout=10)
                    except subprocess.TimeoutExpired:
                        process.kill()
                        process.wait()
                process.stdout.close()
        if returncode:
            raise subprocess.CalledProcessError(returncode, argv)
        receipt["status"] = "passed"
    except BaseException as error:
        receipt.update(status="failed", error=f"{type(error).__name__}: {error}")
        raise
    finally:
        receipt.update(finished_utc=datetime.now(timezone.utc).isoformat(),
                       returncode=process.returncode if process is not None else None)
        with log_path.open("a") as log:
            log.write("\n" + json.dumps(receipt) + "\n")


def attempt_stage(command, run_root, *options):
    """Let a notebook display diagnostics before it propagates stage failure."""
    if command in {"audit-teacher", "run-gate"} and (Path(run_root) / "config/run-contract.json").exists():
        try:
            if audit_summary_path(run_root).exists():
                require_audits(run_root)
        except ValidityAuditRejected as error:
            return error  # A verified rejection never launches a fitting job.
        except (OSError, ValueError, KeyError, TypeError):
            # Let the CLI retain its failure log and the notebook display QC.
            pass
    try:
        run_stage(command, run_root, *options)
    except subprocess.CalledProcessError as error:
        if command == "audit-teacher" and error.returncode == 2:
            # Exit 2 alone is not evidence. Revalidate the completed audit.
            try:
                require_audits(run_root)
            except ValidityAuditRejected as rejected:
                return rejected
            except (OSError, ValueError, KeyError, TypeError):
                pass
        return f"{command} failed (exit {error.returncode}); see the stage log and diagnostics below."
    return None


def require_stage_success(error):
    if isinstance(error, ValidityAuditRejected):
        _publish_blocked(error)
        return
    if error is not None:
        raise RuntimeError(error)


def initialize_from_environment(run_root):
    """Use the same input paths as job 01; a resume reads the frozen contract."""
    root = (PROJECT / Path(run_root).expanduser()).resolve()
    with stage_lock(root, "init"):
        if (root / "config/run-contract.json").is_file():
            check_run(root)
            print(f"Verified existing run: {root}")
            return
        if os.environ.get("FI_EXPERIMENT_PROTOCOL") == "direct-v3":
            from .fi_cache_reuse import initialize_cached_run
            parent = os.environ.get("FI_PARENT_ROOT")
            calibration = os.environ.get("FI_CALIBRATION")
            if not parent or not calibration:
                raise ValueError("direct-v3 initialization requires FI_PARENT_ROOT and FI_CALIBRATION")
            initialize_cached_run(root, parent, PROJECT / "docs/studies/future-innovation/direct-v3-repair-protocol.md", calibration)
            print(f"Initialized cached development run: {root}")
            return
        names = ("GAVD_FULL_ROOT", "VJEPA2_ROOT", "FI_TEACHER_CHECKPOINT", "FI_POSE_MODEL")
        missing = [key for key in names if not os.environ.get(key)]
        if missing:
            raise ValueError(f"New run requires {', '.join(missing)}; follow the HAIC setup guide.")
        paths = {key: (PROJECT / Path(os.environ[key]).expanduser()).resolve() for key in names}
        data = paths["GAVD_FULL_ROOT"]
        annotation = (PROJECT / Path(os.environ.get(
            "FI_ANNOTATION_ROOT", data / "annotations/GAVD/data")).expanduser()).resolve()
        partitions = [annotation / f"GAVD_Clinical_Annotations_{part}.csv" for part in range(1, 6)]
        for path in partitions:
            if not path.is_file():
                raise FileNotFoundError(f"Missing full-GAVD annotation partition: {path}")
        protocol_options = ["--protocol", os.environ.get("FI_EXPERIMENT_PROTOCOL", "direct-v2")]
        run_stage("init-run", root, *protocol_options,
                  "--sequence-manifest", data / "manifests/gavd_full_sequences.csv",
                  "--video-manifest", data / "manifests/gavd_full_videos.csv",
                  "--annotations", *partitions, "--youtube-dir", data / "youtube",
                  "--pose-model", paths["FI_POSE_MODEL"], "--vjepa-root", paths["VJEPA2_ROOT"],
                  "--checkpoint", paths["FI_TEACHER_CHECKPOINT"], "--change-reason",
                  os.environ.get("FI_CHANGE_REASON", "Experiment 0 initialization"))


def build_notebook_report(run_root):
    """Attempt scoring, then always ask the production report to explain failure."""
    root = Path(run_root)
    if (root / "config/run-contract.json").exists():
        try:
            if audit_summary_path(root).exists():
                require_audits(root)
        except ValidityAuditRejected as error:
            print(f"Skipping scoring: {error}")
            run_stage("build-report", root)
            _require_blocked_decision(root, error)
            return BlockedReport(error, sha256_file(root / "reports/gate-decision.json"),
                                 sha256_file(root / "reports/gate-report.md"))
        except (OSError, ValueError, KeyError, TypeError):
            # Preserve the score failure and diagnostic report for corrupt audits.
            pass
    try:
        run_stage("score-gate", run_root)
    except subprocess.CalledProcessError as error:
        print(f"Scoring failed (exit {error.returncode}); building the diagnostic STOP report.")
        # Build the diagnostic, but never hide a scoring error behind an older seal.
        run_stage("build-report", run_root)
        return False
    run_stage("build-report", run_root)
    return True


def _require_blocked_decision(root, error):
    _recheck_rejection(error)
    root = Path(root)
    decision = read_json(root / "reports/gate-decision.json")
    run = check_run(root)
    if (decision.get("decision") != "STOP" or decision.get("measurement_complete") is not False
            or decision.get("allow_full_experiment") is not False
            or decision.get("allow_adapter_training") is not False
            or (protocol_name(run) == DIRECT_PROTOCOL and
                (decision.get("allow_jepa_training_comparison") is not False
                 or decision.get("protocol") != DIRECT_PROTOCOL))
            or "metrics" not in decision or decision["metrics"] is not None
            or decision.get("audit_summary_sha256") != error.summary_sha256
            or decision.get("workflow_status") != "blocked_by_validity_audit"
            or decision.get("run_id") != run["run_id"]
            or decision.get("synthetic") is not run["synthetic"]
            or (root / "reports/final-report-contract.json").exists()):
        raise ValueError("Diagnostic STOP does not match the current rejected audit")
    return decision


def finish_notebook_report(run_root, *, scoring_succeeded):
    """Accept a verified audit STOP separately from a complete measurement."""
    if isinstance(scoring_succeeded, BlockedReport):
        attempt = scoring_succeeded
        root = Path(run_root)
        decision = _require_blocked_decision(root, attempt.audit)
        if (sha256_file(root / "reports/gate-decision.json") != attempt.decision_sha256
                or sha256_file(root / "reports/gate-report.md") != attempt.narrative_sha256):
            raise ValueError("Diagnostic report changed during notebook execution")
        _publish_blocked(attempt.audit)
        return decision
    decision = require_complete_report(run_root, scoring_succeeded=scoring_succeeded)
    print("Complete measurement:", decision["decision"], "— synthetic:", decision.get("synthetic"))
    publish_outcome({"status": "passed", "execution_completed": True,
                     "scientific_status": "synthetic" if decision.get("synthetic") else decision["decision"],
                     "measurement_complete": True, "synthetic": decision.get("synthetic")})
    return decision


def require_complete_report(run_root, *, scoring_succeeded):
    """A complete scientific STOP is success; missing or corrupt evidence is not."""
    evidence = inspect_report(run_root)
    decision = evidence["decision"]
    if (scoring_succeeded is not True or decision is None or not measurement_complete(decision)
            or not evidence["seal_verified"]
            or not (evidence["state"].startswith("COMPLETE / ") or evidence["state"] == "SYNTHETIC")):
        raise RuntimeError(f"Experiment incomplete or invalid: {evidence['state']}. "
                           f"See {Path(run_root) / 'reports'} and the saved stage logs.")
    return decision
