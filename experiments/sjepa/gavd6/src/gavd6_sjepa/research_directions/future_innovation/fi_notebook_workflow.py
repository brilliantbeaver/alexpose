"""Notebook orchestration of the existing Experiment 0 CLI, without new models."""

from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys
from uuid import uuid4

from .fi_contracts import check_run, measurement_complete, stage_lock
from .fi_tutorial_inspection import inspect_report


PROJECT = Path(__file__).resolve().parents[4]
STAGES = frozenset({"init-run", "build-cohort", "extract-poses", "cache-teacher",
                    "audit-teacher", "run-gate", "score-gate", "build-report"})


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
    try:
        run_stage(command, run_root, *options)
    except subprocess.CalledProcessError as error:
        return f"{command} failed (exit {error.returncode}); see the stage log and diagnostics below."
    return None


def require_stage_success(error):
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
        run_stage("init-run", root,
                  "--sequence-manifest", data / "manifests/gavd_full_sequences.csv",
                  "--video-manifest", data / "manifests/gavd_full_videos.csv",
                  "--annotations", *partitions, "--youtube-dir", data / "youtube",
                  "--pose-model", paths["FI_POSE_MODEL"], "--vjepa-root", paths["VJEPA2_ROOT"],
                  "--checkpoint", paths["FI_TEACHER_CHECKPOINT"], "--change-reason",
                  os.environ.get("FI_CHANGE_REASON", "Experiment 0 initialization"))


def build_notebook_report(run_root):
    """Attempt scoring, then always ask the production report to explain failure."""
    try:
        run_stage("score-gate", run_root)
    except subprocess.CalledProcessError as error:
        print(f"Scoring failed (exit {error.returncode}); building the diagnostic STOP report.")
        # Build the diagnostic, but never hide a scoring error behind an older seal.
        run_stage("build-report", run_root)
        return False
    run_stage("build-report", run_root)
    return True


def require_complete_report(run_root, *, scoring_succeeded):
    """A complete scientific STOP is success; missing or corrupt evidence is not."""
    evidence = inspect_report(run_root)
    decision = evidence["decision"]
    if (not scoring_succeeded or decision is None or not measurement_complete(decision)
            or not evidence["seal_verified"]
            or not (evidence["state"].startswith("COMPLETE / ") or evidence["state"] == "SYNTHETIC")):
        raise RuntimeError(f"Experiment incomplete or invalid: {evidence['state']}. "
                           f"See {Path(run_root) / 'reports'} and the saved stage logs.")
    return decision
