"""Run a synthetic-training notebook with the selected Python and retain failures."""

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
SOURCE = ROOT / "notebooks/synthetic_training"
NOTEBOOKS = {
    "00": "00_question_and_assets.ipynb",
    "01": "01_prepare_source_data.ipynb",
    "02": "02_measure_source_trials.ipynb",
    "03": "03_fit_and_freeze_selectors.ipynb",
    "04": "04_prepare_real_evaluation.ipynb",
    "05": "05_choose_and_adapt.ipynb",
    "06": "06_measure_real_accuracy.ipynb",
    "07": "07_source_mechanism_report.ipynb",
    "08": "08_exchange_selected_lessons.ipynb",
}


def configured_students(role: str, student_id: str | None = None) -> list[str]:
    """Resolve an array roster without loading model weights or reading outcomes."""
    sys.path.insert(0, str(ROOT / "src"))
    from gavd6_sjepa.research_directions.synthetic_training.config import RunConfig

    cfg = RunConfig.from_env()
    students = cfg.source_students if role == "source" else cfg.deployment_students
    available = [str(student["student_id"]) for student in students]
    if student_id:
        if student_id not in available:
            raise ValueError(f"Student {student_id!r} is not in the configured {role} roster: {available}")
        return [student_id]
    if not available:
        raise ValueError(f"No students configured for {role}.")
    return available


def relocate_links(notebook, output: Path) -> None:
    """Point an executed notebook's reading links back to the source tutorials."""
    for cell in notebook.cells:
        if cell.cell_type != "markdown":
            continue

        def replace(match: re.Match) -> str:
            label, target = match.groups()
            if "://" in target or target.startswith("#"):
                return match.group(0)
            path, separator, fragment = target.partition("#")
            relative = Path(os.path.relpath((SOURCE / path).resolve(), output)).as_posix()
            return f"[{label}]({relative}{separator}{fragment})"

        cell.source = re.sub(r"\[([^\]]+)\]\(([^)]+)\)", replace, cell.source)


def execute_notebook(
    number: str,
    *,
    run_root: Path,
    config: Path | None = None,
    device: str | None = None,
    student_id: str | None = None,
    evaluation_split: str | None = None,
    output_dir: Path | None = None,
    timeout: int | None = None,
) -> Path:
    """Execute one stage; keep each attempt separate, including failed attempts."""
    if number not in NOTEBOOKS:
        raise ValueError("Choose notebook 00 through 08.")
    if timeout is not None and timeout <= 0:
        raise ValueError("Timeout must be positive, or omitted for no per-cell limit.")
    student_id = student_id or os.environ.get("ST_STUDENT_ID")
    if student_id and not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", student_id):
        raise ValueError("Student IDs must contain only letters, digits, underscores, hyphens or periods.")
    if number in {"02", "05", "08"} and not student_id:
        raise ValueError("Set ST_STUDENT_ID or pass --student-id for per-student notebook 02, 05 or 08.")
    evaluation_split = evaluation_split or os.environ.get("ST_EVALUATION_SPLIT", "early")
    if evaluation_split not in {"early", "confirmation"}:
        raise ValueError("Evaluation split must be early or confirmation.")
    run_root = (ROOT / run_root.expanduser()).resolve()
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    stage = run_root / "notebook_runs" / f"run-{number}"
    if number in {"02", "05", "08"}:
        stage /= student_id
    elif number == "06":
        stage /= evaluation_split
    output_override = output_dir or os.environ.get("ST_NOTEBOOK_OUTPUT_DIR")
    output = Path(output_override).expanduser().resolve() if output_override else stage / stamp
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
    record = {
        "run_root": str(run_root), "notebook": number, "student_id": student_id,
        "evaluation_split": evaluation_split if number == "06" else None,
        "started_utc": stamp, "status": "running", "python": sys.executable,
    }

    def save() -> None:
        notebook.metadata["synthetic_training_execution"] = record.copy()
        nbformat.write(notebook, destination)

    def started(cell, cell_index, **kwargs) -> None:
        if cell.cell_type == "code":
            print(f"Notebook {number}: cell {cell_index + 1}/{len(notebook.cells)}", flush=True)

    def finished(cell, cell_index, **kwargs) -> None:
        notebook.cells[cell_index] = cell
        save()

    print(f"Executed notebook: {destination}", flush=True)
    save()
    try:
        with tempfile.TemporaryDirectory(prefix="synthetic-training-kernel-") as temporary:
            kernel_root = Path(temporary)
            spec = kernel_root / "synthetic-training"
            spec.mkdir()
            (spec / "kernel.json").write_text(json.dumps({
                "argv": [sys.executable, "-m", "ipykernel_launcher", "-f", "{connection_file}"],
                "display_name": "Synthetic training", "language": "python",
            }))
            environment = {
                **os.environ, "GAVD6_ROOT": str(ROOT), "ST_RUN_ROOT": str(run_root),
                "MPLCONFIGDIR": str(kernel_root / "matplotlib"),
                "IPYTHONDIR": str(kernel_root / "ipython"), "PYTHONUNBUFFERED": "1",
            }
            if config is not None:
                environment["ST_CONFIG"] = str(config.expanduser().resolve())
            if device is not None:
                environment["ST_DEVICE"] = device
            if student_id is not None:
                environment["ST_STUDENT_ID"] = student_id
            environment["ST_EVALUATION_SPLIT"] = evaluation_split
            manager = KernelManager(
                kernel_name="synthetic-training",
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
    parser.add_argument("--notebook", choices=tuple(NOTEBOOKS))
    parser.add_argument("--run-root", type=Path)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--device", choices=("cpu", "cuda"))
    parser.add_argument("--student-id")
    parser.add_argument("--evaluation-split", choices=("early", "confirmation"))
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--timeout", type=int, help="Per-cell seconds; Slurm otherwise sets the time limit.")
    parser.add_argument("--list-students", choices=("source", "deployment"),
                        help="Print the configured roster for Slurm arrays, without running a notebook.")
    args = parser.parse_args()
    if args.list_students:
        if args.config:
            os.environ["ST_CONFIG"] = str(args.config.expanduser().resolve())
        if args.run_root:
            os.environ["ST_RUN_ROOT"] = str(args.run_root.expanduser().resolve())
        os.environ.setdefault("GAVD6_ROOT", str(ROOT))
        for student_id in configured_students(args.list_students, args.student_id or os.environ.get("ST_STUDENT_ID")):
            print(student_id)
        return
    if args.notebook is None or args.run_root is None:
        parser.error("--notebook and --run-root are required unless --list-students is used.")
    execute_notebook(
        args.notebook, run_root=args.run_root, config=args.config, device=args.device,
        student_id=args.student_id, evaluation_split=args.evaluation_split,
        output_dir=args.output_dir, timeout=args.timeout,
    )


if __name__ == "__main__":
    main()
