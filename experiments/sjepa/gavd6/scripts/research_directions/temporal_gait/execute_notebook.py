"""Execute a thin temporal-gait notebook, preserving partial outputs on failure."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import re
import sys
import tempfile

from jupyter_client import KernelManager
from jupyter_client.kernelspec import KernelSpecManager
from nbclient import NotebookClient
import nbformat

ROOT = Path(__file__).resolve().parents[3]
SOURCE = ROOT / "notebooks/temporal_gait"
NOTEBOOKS = {
    "00": "00_inventory_and_split.ipynb",
    "01": "01_full_bout_timing_and_windows.ipynb",
    "02": "02_information_and_baseline_audit.ipynb",
    "03": "03_time_faithful_masked_jepa.ipynb",
    "04": "04_causal_future_jepa.ipynb",
    "05": "05_optional_dense_and_video_transfer.ipynb",
    "06": "06_development_comparison.ipynb",
    "07": "07_locked_calibration_and_test.ipynb",
    "08": "08_aggregate_and_claim_audit.ipynb",
}
NOTEBOOK_STAGES = {"00": ("inventory",), "01": ("prepare",), "02": ("audit",),
                   "03": ("masked",), "04": ("future",),
                   "05": ("extensions", "cache-video"), "06": ("evaluate",),
                   "07": ("calibrate", "test"), "08": ("aggregate",)}


def declared_mode(config):
    """Label outputs from configuration metadata without opening any manifest."""
    path = config or os.environ.get("TG_CONFIG")
    try:
        payload = json.loads(Path(path).read_text()) if path else {}
        configured = payload.get("mode")
        override = os.environ.get("TG_MODE")
        if configured and override and configured != override:
            return "unknown"
        mode = override or configured or "real"
        return mode if mode in ("real", "synthetic") else "unknown"
    except (OSError, ValueError, AttributeError):
        return "unknown"


def relocate_links(notebook, output, *, staged_run_root=None):
    for cell in notebook.cells:
        if cell.cell_type != "markdown":
            continue
        def replace(match):
            label, target = match.groups()
            if "://" in target or target.startswith("#"):
                return match.group(0)
            path, separator, fragment = target.partition("#")
            if path in NOTEBOOKS.values() and staged_run_root is not None:
                number = next(n for n, name in NOTEBOOKS.items() if name == path)
                destination = staged_run_root / "notebook_runs" / f"run-{number}" / path
                path = Path(os.path.relpath(destination, output)).as_posix()
            else:
                path = Path(os.path.relpath((SOURCE / path).resolve(), output)).as_posix()
            return f"[{label}]({path}{separator}{fragment})"
        cell.source = re.sub(r"\[([^\]]+)\]\(([^)]+)\)", replace, cell.source)


def execute_notebook(number, *, run_root, config=None, output_dir=None,
                     stage=None, phase="pilot", task_id=None, execute=False,
                     timeout=None):
    if number not in NOTEBOOKS:
        raise ValueError("Choose notebook 00 through 08")
    stage = stage or NOTEBOOK_STAGES[number][0]
    if stage not in NOTEBOOK_STAGES[number]:
        raise ValueError(f"Notebook {number} cannot execute stage {stage}")
    if phase not in ("pilot", "develop", "confirm"):
        raise ValueError("Unknown experiment phase")
    if timeout is not None and timeout <= 0:
        raise ValueError("Timeout must be positive")
    run_root = Path(run_root).expanduser().resolve()
    override = output_dir or os.environ.get("TG_NOTEBOOK_OUTPUT_DIR")
    task_suffix = f"-task-{task_id}" if task_id is not None else ""
    output = Path(override or run_root / "notebook_runs" /
                  f"run-{number}-{phase}-{stage}{task_suffix}").expanduser().resolve()
    if output == SOURCE or SOURCE in output.parents:
        raise ValueError("Executed notebooks must be outside the source notebook directory")
    output.mkdir(parents=True, exist_ok=True)
    destination = output / NOTEBOOKS[number]
    if destination.exists():
        raise FileExistsError(f"Executed notebook exists: {destination}; choose a new output directory")
    notebook = nbformat.read(SOURCE / NOTEBOOKS[number], as_version=4)
    for cell in notebook.cells:
        if cell.cell_type == "code":
            cell.outputs, cell.execution_count = [], None
    # Source notebook links are reference links; tasks have different output folders.
    relocate_links(notebook, output)
    enabled = bool(execute or os.environ.get("TG_EXECUTE") == "1")
    mode = declared_mode(config)
    artifact = ("synthetic_software" if mode == "synthetic" else
                "real_execution" if mode == "real" else "execution_requested")
    record = {"notebook": number, "stage": stage, "phase": phase, "task_id": task_id,
              "run_root": str(run_root), "execution_enabled": enabled, "mode": mode,
              "artifact_status": artifact if enabled else "plan_only",
              "status": "running", "started_utc": datetime.now(timezone.utc).isoformat()}
    def save():
        notebook.metadata["temporal_gait_execution"] = record.copy()
        nbformat.write(notebook, destination)
    def started(cell, cell_index, **kwargs):
        if cell.cell_type == "code":
            print(f"Notebook {number}: cell {cell_index + 1}/{len(notebook.cells)}", flush=True)
    def finished(cell, cell_index, **kwargs):
        notebook.cells[cell_index] = cell
        save()
    print(f"Executed notebook: {destination}", flush=True)
    save()
    try:
        with tempfile.TemporaryDirectory(prefix="temporal-gait-kernel-") as temporary:
            kernel_root = Path(temporary)
            spec = kernel_root / "temporal-gait"
            spec.mkdir()
            (spec / "kernel.json").write_text(json.dumps({
                "argv": [sys.executable, "-m", "ipykernel_launcher", "-f", "{connection_file}"],
                "display_name": "Temporal gait", "language": "python"}))
            environment = {**os.environ, "GAVD6_ROOT": str(ROOT), "TG_RUN_ROOT": str(run_root),
                           "TG_STAGE": stage, "TG_PHASE": phase,
                           "TG_EXECUTE": "1" if enabled else "0",
                           "MPLCONFIGDIR": str(output / ".matplotlib"),
                           "IPYTHONDIR": str(kernel_root / "ipython"), "PYTHONUNBUFFERED": "1"}
            environment.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
            if config is not None:
                environment["TG_CONFIG"] = str(Path(config).expanduser().resolve())
            if task_id is not None:
                environment["TG_TASK_ID"] = str(task_id)
            else:
                environment.pop("TG_TASK_ID", None)
            manager = KernelManager(kernel_name="temporal-gait",
                kernel_spec_manager=KernelSpecManager(kernel_dirs=[str(kernel_root)]))
            client = NotebookClient(notebook, km=manager, timeout=timeout, startup_timeout=120,
                resources={"metadata": {"path": str(ROOT)}}, on_cell_start=started,
                on_cell_executed=finished, on_cell_error=finished)
            try:
                client.execute(env=environment)
            finally:
                try:
                    if manager.has_kernel:
                        manager.shutdown_kernel(now=True)
                finally:
                    manager.cleanup_resources()
        record["status"] = "completed"
    except BaseException as error:
        record.update(status="failed", error=f"{type(error).__name__}: {str(error)[-1200:]}")
        raise
    finally:
        record["finished_utc"] = datetime.now(timezone.utc).isoformat()
        save()
    return destination


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--notebook", choices=tuple(NOTEBOOKS), required=True)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--stage")
    parser.add_argument("--phase", choices=("pilot", "develop", "confirm"), default="pilot")
    parser.add_argument("--task-id", type=int)
    parser.add_argument("--execute", action="store_true", help="Enable the selected workflow stage")
    parser.add_argument("--timeout", type=int)
    args = parser.parse_args(argv)
    execute_notebook(args.notebook, run_root=args.run_root, config=args.config,
                     output_dir=args.output_dir, stage=args.stage, phase=args.phase,
                     task_id=args.task_id, execute=args.execute, timeout=args.timeout)


if __name__ == "__main__":
    main()
