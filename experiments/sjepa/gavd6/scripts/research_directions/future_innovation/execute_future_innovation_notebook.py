"""Execute an unchanged tutorial in a fresh kernel and retain an isolated attempt."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import platform
import sys
import tempfile
from time import perf_counter

from jupyter_client import KernelManager
from jupyter_client.kernelspec import KernelSpecManager
from nbclient import NotebookClient
import nbformat

from build_future_innovation_notebooks import DESTINATION, NAMES, ROOT, render


def atomic_text(path, content):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(content)
    temporary.replace(path)


def execute_notebook(number, *, mode="teach", run_root=None, outer_fold=None,
                     device="cuda", timeout=None, output_parent=None):
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
    root = (ROOT / Path(run_root or "outputs/future-innovation/gate-v1").expanduser()).resolve()
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
    parent = Path(output_parent) if output_parent is not None else (
        root / "notebook_runs" if mode == "execute" else ROOT / "work/artifacts/notebook_runs/future_innovation")
    parent = parent.expanduser().resolve()
    parent.mkdir(parents=True, exist_ok=True)
    label = f"{mode}-{number}" + (f"-fold-{outer_fold}" if outer_fold is not None else "")
    output = Path(tempfile.mkdtemp(prefix=label + "-", dir=parent))
    destination = output / source.name
    environment = {**os.environ, "GAVD6_ROOT": str(ROOT), "FI_TUTORIAL_MODE": mode,
                   "FI_RUN_ROOT": str(root), "FI_NOTEBOOK_ATTEMPT": str(output),
                   "FI_NOTEBOOK_DEVICE": device, "PYTHONUNBUFFERED": "1",
                   "MPLCONFIGDIR": str(output / "matplotlib"), "IPYTHONDIR": str(output / "ipython")}
    environment.pop("FI_NOTEBOOK_FOLD", None)
    if outer_fold is not None:
        environment["FI_NOTEBOOK_FOLD"] = str(outer_fold)
    record = {"notebook": source.name, "mode": mode, "run_root": str(root), "outer_fold": outer_fold,
              "teacher_device": device, "python": sys.version, "executable": sys.executable,
              "platform": platform.platform(), "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
              "started_utc": datetime.now(timezone.utc).isoformat(), "status": "running",
              "slurm": {key: os.environ[key] for key in
                        ("SLURM_JOB_ID", "SLURM_ARRAY_JOB_ID", "SLURM_ARRAY_TASK_ID", "SLURM_RESTART_COUNT")
                        if key in os.environ}}
    # Keep the orchestration source as well as the notebook for later attribution.
    for script in (Path(__file__), Path(__file__).with_name("build_future_innovation_notebooks.py"),
                   ROOT / "src/gavd6_sjepa/research_directions/future_innovation/fi_notebook_workflow.py"):
        (output / script.name).write_bytes(script.read_bytes())

    def save():
        notebook.metadata["fi_execution"] = dict(record)
        atomic_text(destination, nbformat.writes(notebook))
        atomic_text(output / "execution.json", json.dumps(record, indent=2) + "\n")

    def cell_start(cell, cell_index, **kwargs):
        record["active_cell"] = cell_index
        save()
        if cell.cell_type == "code":
            print(f"{label}: cell {cell_index + 1}/{len(notebook.cells)}", flush=True)

    def cell_finished(cell, cell_index, **kwargs):
        notebook.cells[cell_index] = cell
        record["last_finished_cell"] = cell_index
        save()

    print(f"Notebook attempt: {output}", flush=True)
    save()
    started = perf_counter()
    try:
        with tempfile.TemporaryDirectory(prefix="fi-notebook-kernel-") as temporary:
            kernels = Path(temporary)
            spec = kernels / "fi-notebook"
            spec.mkdir()
            (spec / "kernel.json").write_text(json.dumps({
                "argv": [sys.executable, "-m", "ipykernel_launcher", "-f", "{connection_file}"],
                "display_name": "Future Innovation", "language": "python"}))
            manager = KernelManager(kernel_name="fi-notebook",
                                    kernel_spec_manager=KernelSpecManager(kernel_dirs=[str(kernels)]))
            client = NotebookClient(notebook, km=manager, timeout=timeout, startup_timeout=120,
                                    resources={"metadata": {"path": str(DESTINATION)}},
                                    on_cell_start=cell_start, on_cell_executed=cell_finished,
                                    on_cell_error=cell_finished)
            try:
                client.execute(env=environment)
            finally:
                if manager.has_kernel:
                    manager.shutdown_kernel(now=True)
                manager.cleanup_resources()
        record["status"] = "passed"
    except BaseException as error:
        record.update(status="failed", error=f"{type(error).__name__}: {error}")
        raise
    finally:
        record.update(finished_utc=datetime.now(timezone.utc).isoformat(),
                      wall_seconds_including_kernel=round(perf_counter() - started, 3))
        save()
        print(f"{record['status'].upper()}: {destination}", flush=True)
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--notebook", choices=tuple(NAMES), required=True)
    parser.add_argument("--mode", choices=("teach", "inspect", "execute"), default="teach")
    parser.add_argument("--run-root", type=Path)
    parser.add_argument("--outer-fold", type=int, choices=range(5))
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--timeout", type=int, help="Per-cell seconds; default unlimited (Slurm enforces job time).")
    parser.add_argument("--output-parent", type=Path)
    args = parser.parse_args()
    execute_notebook(args.notebook, mode=args.mode, run_root=args.run_root, outer_fold=args.outer_fold,
                     device=args.device, timeout=args.timeout, output_parent=args.output_parent)


if __name__ == "__main__":
    main()
