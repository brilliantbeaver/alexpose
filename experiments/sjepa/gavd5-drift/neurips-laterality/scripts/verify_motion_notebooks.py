"""Verify and optionally execute notebooks 15–18 with real-data flags disabled."""
from __future__ import annotations

import argparse
import base64
import hashlib
import json
import os
from pathlib import Path
import sys
import tempfile

import nbformat
from nbclient import NotebookClient
from jupyter_client import KernelManager

SUITE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).resolve().parent))
from build_research_notebooks import NOTEBOOKS, render


def protected_files():
    paths = [p for p in SUITE_ROOT.glob("*.ipynb") if p.name[:2] in {f"{n:02d}" for n in range(15)}]
    for folder in ("laterality", "config", "governance"):
        paths.extend(p for p in (SUITE_ROOT / folder).rglob("*") if p.is_file() and "__pycache__" not in p.parts)
    paths.extend(SUITE_ROOT / "tutorials" / f"research_{n:02d}.py" for n in range(7, 15))
    paths.extend(p for p in (SUITE_ROOT / "laterality_extensions").glob("*.py")
                 if not p.name.startswith(("motion_",)))
    paths.append(SUITE_ROOT / "PROTOCOL.md")
    return {str(p.relative_to(SUITE_ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--only", nargs="+", choices=("15", "16", "17", "18"), default=["15", "16", "17", "18"])
    args = parser.parse_args()
    before = protected_files()
    records, destination = [], None
    if args.execute:
        parent = SUITE_ROOT / "executed/motion_structured"
        parent.mkdir(parents=True, exist_ok=True)
        destination = Path(tempfile.mkdtemp(prefix="synthetic_", dir=parent))
    environment = dict(os.environ)
    for name in ("LATERALITY_MOTION_VALIDATE_REAL", "LATERALITY_MOTION_RUN_REAL",
                 "LATERALITY_RESEARCH_RUN_REAL", "LATERALITY_RESEARCH_VALIDATE_REAL"):
        environment[name] = "0"
    environment["LATERALITY_DEVICE"] = "cpu"
    environment.pop("LATERALITY_RETAINED_COMPARISON", None)
    environment["LATERALITY_MOTION_EXPERIMENTS"] = "motion,regions"
    environment["LATERALITY_RESEARCH_CPU_THREADS"] = "1"
    for number in args.only:
        path = SUITE_ROOT / NOTEBOOKS[number]
        if path.read_text(encoding="utf-8") != render(number):
            raise AssertionError(f"Source notebook differs from its builder: {path.name}")
        notebook = nbformat.read(path, as_version=4)
        nbformat.validate(notebook)
        code_cells = [c for c in notebook.cells if c.cell_type == "code"]
        if any(c.outputs or c.execution_count is not None for c in code_cells):
            raise AssertionError("Source notebooks must remain output-free")
        record = {"notebook": path.name, "source_valid": True, "executed": False}
        if args.execute:
            print(f"Executing synthetic notebook {number} with {sys.executable}", flush=True)
            manager = KernelManager(kernel_name="python3")
            manager.kernel_spec.argv = [sys.executable, "-m", "ipykernel_launcher", "-f", "{connection_file}"]
            try:
                NotebookClient(notebook, timeout=300, allow_errors=False, km=manager,
                    resources={"metadata": {"path": str(SUITE_ROOT)}}).execute(env=environment)
            finally:
                if manager.has_kernel:
                    manager.shutdown_kernel(now=True)
                manager.cleanup_resources()
            vector_count = 0
            for i, cell in enumerate(notebook.cells):
                if cell.cell_type != "code":
                    continue
                if cell.execution_count is None:
                    raise AssertionError("A code cell did not execute")
                for j, output in enumerate(cell.outputs):
                    data = output.get("data", {})
                    if "image/svg+xml" in data:
                        value = data["image/svg+xml"]
                        (destination / f"{number}_{i}_{j}.svg").write_text(
                            "".join(value) if isinstance(value, list) else value, encoding="utf-8")
                        vector_count += 1
                    if "image/png" in data:
                        (destination / f"{number}_{i}_{j}.png").write_bytes(base64.b64decode(data["image/png"]))
            if vector_count < 1:
                raise AssertionError("No vector figure rendered")
            nbformat.write(notebook, destination / path.name)
            record.update(executed=True, code_cells=len(code_cells), vector_outputs=vector_count)
        records.append(record)
        print(f"PASS {path.name}", flush=True)
    after = protected_files()
    if before != after:
        raise AssertionError("Protected research files changed")
    report = {"notebooks": records, "protected_files_unchanged": len(before),
              "real_training_executed": False, "real_validation_executed": False,
              "executed_directory": str(destination) if destination else None}
    if destination:
        (destination / "verification.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
