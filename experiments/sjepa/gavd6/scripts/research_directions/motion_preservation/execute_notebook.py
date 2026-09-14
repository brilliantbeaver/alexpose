"""Execute a motion-preservation notebook and retain outputs, including failures."""

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
SOURCE = ROOT / "notebooks/motion_preservation"
NOTEBOOKS = {
    "00": "00_data_and_question.ipynb",
    "01": "01_make_controlled_pairs.ipynb",
    "02": "02_prior_flow_and_baselines.ipynb",
    "03": "03_train_and_calibrate.ipynb",
    "04": "04_preservation_and_repair.ipynb",
    "05": "05_gavd_visual_stress.ipynb",
}


def relocate_links(notebook, output: Path) -> None:
    """Keep local reading links usable in the executed copy on scratch."""
    for cell in notebook.cells:
        if cell.cell_type != "markdown":
            continue

        def replace(match):
            label, target = match.groups()
            if "://" in target or target.startswith("#"):
                return match.group(0)
            path, separator, fragment = target.partition("#")
            # Adjacent notebooks land in the same output folder for a batch.
            if path not in NOTEBOOKS.values():
                path = Path(os.path.relpath((SOURCE / path).resolve(), output)).as_posix()
            return f"[{label}]({path}{separator}{fragment})"

        cell.source = re.sub(r"\[([^\]]+)\]\(([^)]+)\)", replace, cell.source)


def execute_notebook(
    number: str,
    *,
    run_root: Path,
    mode: str | None = None,
    device: str | None = None,
    output_dir: Path | None = None,
    config: Path | None = None,
    timeout: int | None = None,
) -> Path:
    if number not in NOTEBOOKS:
        raise ValueError("Choose notebook 00 through 05.")
    if mode is not None and mode not in {"real", "demo"}:
        raise ValueError("Mode must be real or demo.")
    if timeout is not None and timeout <= 0:
        raise ValueError("Timeout must be positive; omit for no per-cell limit.")
    run_root = (ROOT / run_root.expanduser()).resolve()
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    output = Path(output_dir or os.environ.get("MP_NOTEBOOK_OUTPUT_DIR") or
                  run_root / "notebook_runs" / stamp).expanduser().resolve()
    if output == SOURCE or SOURCE in output.parents:
        raise ValueError("Save executed notebooks outside the source notebook folder.")
    output.mkdir(parents=True, exist_ok=True)
    destination = output / NOTEBOOKS[number]
    if destination.exists():
        raise FileExistsError(f"Executed notebook already exists: {destination}. Choose another output folder.")
    notebook = nbformat.read(SOURCE / NOTEBOOKS[number], as_version=4)
    for cell in notebook.cells:
        if cell.cell_type == "code":
            cell.outputs = []
            cell.execution_count = None
    relocate_links(notebook, output)
    record = {"run_root": str(run_root),
              "mode": mode or os.environ.get("MP_MODE", "config/default"),
              "device": device or os.environ.get("MP_DEVICE", "config/default"),
              "started_utc": stamp, "status": "running", "notebook": number}

    def save():
        notebook.metadata["motion_preservation_execution"] = record.copy()
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
        # Use the launcher's Python, avoiding a stale user-level Jupyter kernel.
        with tempfile.TemporaryDirectory(prefix="motion-preservation-kernel-") as temporary:
            kernel_root = Path(temporary)
            spec = kernel_root / "motion-preservation"
            spec.mkdir()
            (spec / "kernel.json").write_text(json.dumps({
                "argv": [sys.executable, "-m", "ipykernel_launcher", "-f", "{connection_file}"],
                "display_name": "Motion preservation", "language": "python",
            }))
            environment = {**os.environ, "GAVD6_ROOT": str(ROOT), "MP_RUN_ROOT": str(run_root),
                           "MPLCONFIGDIR": str(kernel_root / "matplotlib"),
                           "IPYTHONDIR": str(kernel_root / "ipython"), "PYTHONUNBUFFERED": "1"}
            if mode is not None:
                environment["MP_MODE"] = mode
            if device is not None:
                environment["MP_DEVICE"] = device
            if config is not None:
                environment["MP_CONFIG"] = str(config.expanduser().resolve())
            manager = KernelManager(
                kernel_name="motion-preservation",
                kernel_spec_manager=KernelSpecManager(kernel_dirs=[str(kernel_root)]),
            )
            client = NotebookClient(
                notebook, km=manager, timeout=timeout, startup_timeout=120,
                resources={"metadata": {"path": str(SOURCE)}},
                on_cell_start=started, on_cell_executed=finished, on_cell_error=finished,
            )
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
        print(f"{record['status'].upper()}: {destination}", flush=True)
    return destination


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--notebook", choices=tuple(NOTEBOOKS), required=True)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--mode", choices=("real", "demo"), help="Override config/environment mode.")
    parser.add_argument("--device", choices=("cpu", "cuda"), help="Override config/environment device.")
    parser.add_argument("--config", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--timeout", type=int, help="Per-cell seconds; Slurm otherwise sets the time limit.")
    args = parser.parse_args()
    execute_notebook(args.notebook, run_root=args.run_root, mode=args.mode, device=args.device,
                     config=args.config, output_dir=args.output_dir, timeout=args.timeout)


if __name__ == "__main__":
    main()
