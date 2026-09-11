"""Execute notebooks into one notebook-only folder; keep logs outside that folder."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import platform
import re
import sys
import tempfile
from time import perf_counter

from jupyter_client import KernelManager
from jupyter_client.kernelspec import KernelSpecManager
from nbclient import NotebookClient
import nbformat

from build_future_innovation_notebooks import DESTINATION, NAMES, ROOT, render

sys.path.insert(0, str(ROOT / "src"))
from gavd6_sjepa.research_directions.future_innovation.fi_contracts import stage_lock
from gavd6_sjepa.research_directions.future_innovation.fi_notebook_workflow import OUTCOME_MIME


def atomic_text(path, content):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(content)
    temporary.replace(path)


def relocate_links(notebook, output):
    """Keep checkout documentation reachable when the executed copy moves."""
    for cell in notebook.cells:
        if cell.cell_type != "markdown":
            continue
        def replace(match):
            label, target = match.groups()
            if "://" in target or target.startswith("#"):
                return match.group(0)
            path, separator, fragment = target.partition("#")
            if path in NAMES.values():
                if path == NAMES["03"] and os.environ.get("FI_NOTEBOOK_ARRAY") == "1":
                    path = Path(path).stem + "_fold-0.ipynb"
                    label = "03 · Fold 0 (other folds are in this folder)"
            else:
                path = Path(os.path.relpath((DESTINATION / path).resolve(), output)).as_posix()
            return f"[{label}]({path}{separator}{fragment})"
        cell.source = re.sub(r"\[([^\]]+)\]\(([^)]+)\)", replace, cell.source)


def execute_notebook(number, *, mode="teach", run_root=None, outer_fold=None,
                     device="cuda", timeout=None, output_parent=None):
    """output_parent is the shared notebook folder, not a parent of per-notebook folders."""
    if number not in NAMES or mode not in {"teach", "inspect", "execute"}:
        raise ValueError("Choose notebook 00–04 and teach, inspect, or execute mode.")
    if mode == "execute" and run_root is None:
        raise ValueError("Execution requires an explicit --run-root.")
    if outer_fold is not None and (number != "03" or mode != "execute" or outer_fold not in range(5)):
        raise ValueError("--outer-fold is only valid for notebook 03 in execute mode (0–4).")
    if timeout is not None and timeout <= 0:
        raise ValueError("--timeout must be positive; omit for no per-cell time limit.")
    if device not in {"cpu", "cuda"}:
        raise ValueError("Teacher device must be cpu or cuda.")
    root = (ROOT / Path(run_root or "outputs/future-innovation/direct-v2").expanduser()).resolve()
    source = DESTINATION / NAMES[number]
    notebook = nbformat.read(source, as_version=4)
    if [(c.cell_type, c.source) for c in notebook.cells] != [
            (c.cell_type, c.source) for c in render(number).cells]:
        raise ValueError(f"Regenerate {source.name} before execution.")
    for cell in notebook.cells:
        if cell.cell_type == "code":
            cell.outputs = []
            cell.execution_count = None
            cell.metadata.pop("execution", None)
    default = (root / "notebook_runs/manual" if mode == "execute"
               else ROOT / "work/artifacts/notebook_runs/future_innovation" / mode)
    output = Path(output_parent or os.environ.get("FI_NOTEBOOK_OUTPUT_DIR") or default).expanduser().resolve()
    if output == DESTINATION.resolve():
        raise ValueError("Executed notebooks must be stored outside the canonical notebook directory.")
    label = source.stem + (f"_fold-{outer_fold}" if outer_fold is not None else "")
    destination = output / f"{label}.ipynb"
    relocate_links(notebook, output)
    logs_root = root if mode == "execute" else ROOT / "work/artifacts/notebook_logs"
    logs = logs_root / "logs/notebooks" / output.name
    # A lock protects both atomic saves and the whole kernel, including requeues.
    # Different folds use different notebook files and can run concurrently.
    lock_id = hashlib.sha256(str(destination).encode()).hexdigest()[:24]
    # Derive the lock location from the destination, so inspect/execute modes
    # and different run roots cannot acquire different locks for the same file.
    lock_root = output.parent.parent if output.parent.name == "notebook_runs" else output.parent
    with stage_lock(lock_root, f"notebook-{lock_id}"):
        output.mkdir(parents=True, exist_ok=True)
        logs.mkdir(parents=True, exist_ok=True)
        record = {"notebook": source.name, "mode": mode, "run_root": str(root), "outer_fold": outer_fold,
                  "teacher_device": device, "python": sys.version, "executable": sys.executable,
                  "platform": platform.platform(), "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
                  "started_utc": datetime.now(timezone.utc).isoformat(), "status": "running",
                  "stage_logs": str(logs), "slurm": {key: os.environ[key] for key in
                  ("SLURM_JOB_ID", "SLURM_ARRAY_JOB_ID", "SLURM_ARRAY_TASK_ID", "SLURM_RESTART_COUNT")
                  if key in os.environ}}
        # Hash orchestration sources once in metadata; do not duplicate source files.
        record["orchestration_sha256"] = {
            str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in
            (Path(__file__), Path(__file__).with_name("build_future_innovation_notebooks.py"),
             ROOT / "src/gavd6_sjepa/research_directions/future_innovation/fi_notebook_workflow.py")}

        def save():
            notebook.metadata["fi_execution"] = dict(record)
            atomic_text(destination, nbformat.writes(notebook))

        def cell_start(cell, cell_index, **kwargs):
            record["active_cell"] = cell_index
            save()
            if cell.cell_type == "code":
                print(f"{label}: cell {cell_index + 1}/{len(notebook.cells)}", flush=True)

        def cell_finished(cell, cell_index, **kwargs):
            notebook.cells[cell_index] = cell
            record["last_finished_cell"] = cell_index
            save()

        print(f"Executed notebook: {destination}", flush=True)
        save()
        started = perf_counter()
        try:
            # Kernel state and plot/font caches are temporary, never run deliverables.
            with tempfile.TemporaryDirectory(prefix="fi-notebook-kernel-") as temporary:
                kernels = Path(temporary)
                spec = kernels / "fi-notebook"
                spec.mkdir()
                (spec / "kernel.json").write_text(json.dumps({
                    "argv": [sys.executable, "-m", "ipykernel_launcher", "-f", "{connection_file}"],
                    "display_name": "Future Innovation", "language": "python"}))
                environment = {**os.environ, "GAVD6_ROOT": str(ROOT), "FI_TUTORIAL_MODE": mode,
                               "FI_RUN_ROOT": str(root), "FI_NOTEBOOK_LOG_DIR": str(logs),
                               "FI_NOTEBOOK_LABEL": label, "FI_NOTEBOOK_DEVICE": device,
                               "PYTHONUNBUFFERED": "1", "MPLCONFIGDIR": str(kernels / "matplotlib"),
                               "IPYTHONDIR": str(kernels / "ipython")}
                environment.pop("FI_NOTEBOOK_FOLD", None)
                environment.pop("FI_NOTEBOOK_ATTEMPT", None)
                if outer_fold is not None:
                    environment["FI_NOTEBOOK_FOLD"] = str(outer_fold)
                manager = KernelManager(kernel_name="fi-notebook",
                                        kernel_spec_manager=KernelSpecManager(kernel_dirs=[str(kernels)]))
                client = NotebookClient(notebook, km=manager, timeout=timeout, startup_timeout=120,
                                        resources={"metadata": {"path": str(DESTINATION)}},
                                        on_cell_start=cell_start, on_cell_executed=cell_finished,
                                        on_cell_error=cell_finished)
                try:
                    client.execute(env=environment)
                finally:
                    try:
                        if manager.has_kernel:
                            manager.shutdown_kernel(now=True)
                    finally:
                        manager.cleanup_resources()
            record["status"] = "passed"
            record["execution_completed"] = True
            outcomes = [o["data"][OUTCOME_MIME] for c in notebook.cells for o in c.get("outputs", [])
                        if OUTCOME_MIME in o.get("data", {})]
            if outcomes:
                outcome = outcomes[-1]
                if outcome.get("status") not in {"passed", "blocked"}:
                    raise ValueError("Unknown notebook completion outcome")
                record["status"] = outcome["status"]
                record["scientific_outcome"] = outcome
        except BaseException as error:
            # The cell already contains the full traceback; metadata needs a short summary.
            record.update(status="failed", execution_completed=False,
                          error=f"{type(error).__name__}: {str(error).splitlines()[-1:]}")
            raise
        finally:
            record.update(finished_utc=datetime.now(timezone.utc).isoformat(),
                          wall_seconds_including_kernel=round(perf_counter() - started, 3))
            save()
            print(f"{record['status'].upper()}: {destination}", flush=True)
    return destination


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--notebook", choices=tuple(NAMES), required=True)
    parser.add_argument("--mode", choices=("teach", "inspect", "execute"), default="teach")
    parser.add_argument("--run-root", type=Path)
    parser.add_argument("--outer-fold", type=int, choices=range(5))
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--timeout", type=int, help="Per-cell seconds; default unlimited (Slurm enforces job time).")
    parser.add_argument("--output-dir", "--output-parent", dest="output_parent", type=Path,
                        help="Shared folder containing only executed notebooks.")
    args = parser.parse_args()
    execute_notebook(args.notebook, mode=args.mode, run_root=args.run_root, outer_fold=args.outer_fold,
                     device=args.device, timeout=args.timeout, output_parent=args.output_parent)


if __name__ == "__main__":
    main()
