#!/usr/bin/env python3
"""Validate tutorials 11–14 and optionally execute their synthetic defaults."""
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

SUITE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).resolve().parent))
from build_research_notebooks import NOTEBOOKS, render


def protected_files():
    paths = [p for p in SUITE_ROOT.glob("*.ipynb") if p.name[:2] in {f"{n:02d}" for n in range(11)}]
    for folder in ("laterality", "config", "governance"):
        paths.extend(p for p in (SUITE_ROOT / folder).rglob("*") if p.is_file() and "__pycache__" not in p.parts)
    paths.extend(SUITE_ROOT / "tutorials" / f"research_{n:02d}.py" for n in range(7, 11))
    paths.extend(SUITE_ROOT / "laterality_extensions" / n for n in
                 ("masked_learning.py", "forecasting.py", "symmetry_learning.py", "diagnostics.py"))
    paths.append(SUITE_ROOT / "PROTOCOL.md")
    return {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--only", nargs="+", choices=("11", "12", "13", "14"), default=["11", "12", "13", "14"])
    args = parser.parse_args()
    before = protected_files()
    records = []
    destination = None
    if args.execute:
        parent = SUITE_ROOT / "executed/comparative_masking"
        parent.mkdir(parents=True, exist_ok=True)
        destination = Path(tempfile.mkdtemp(prefix="synthetic_", dir=parent))
        os.environ["PATH"] = str(Path(sys.executable).resolve().parent) + os.pathsep + os.environ.get("PATH", "")
    for number in args.only:
        path = SUITE_ROOT / NOTEBOOKS[number]
        if path.read_text() != render(number):
            raise AssertionError(f"Notebook differs from its editable source: {path.name}")
        notebook = nbformat.read(path, as_version=4)
        nbformat.validate(notebook)
        if any(c.execution_count is not None or c.outputs for c in notebook.cells if c.cell_type == "code"):
            raise AssertionError(f"Source notebook has execution outputs: {path.name}")
        record = {"notebook": path.name, "source_valid": True, "executed": False}
        if args.execute:
            print(f"Executing synthetic tutorial {number} in a fresh kernel", flush=True)
            NotebookClient(notebook, timeout=300, kernel_name="python3", allow_errors=False,
                resources={"metadata": {"path": str(SUITE_ROOT)}}).execute()
            code = [c for c in notebook.cells if c.cell_type == "code"]
            if any(c.execution_count is None for c in code):
                raise AssertionError(f"A code cell did not execute in {path.name}")
            svg_count = 0
            for index, cell in enumerate(code):
                for output_index, output in enumerate(cell.outputs):
                    data = output.get("data", {})
                    if "image/svg+xml" in data:
                        value = data["image/svg+xml"]
                        (destination / f"{number}_cell_{index}_{output_index}.svg").write_text(
                            "".join(value) if isinstance(value, list) else value)
                        svg_count += 1
                    if "image/png" in data:
                        (destination / f"{number}_cell_{index}_{output_index}.png").write_bytes(base64.b64decode(data["image/png"]))
            if svg_count < 1:
                raise AssertionError(f"No vector visual rendered in {path.name}")
            nbformat.write(notebook, destination / path.name)
            record.update(executed=True, code_cells=len(code), vector_outputs=svg_count, status="synthetic software check")
        records.append(record)
        print(f"PASS {path.name}", flush=True)
    if before != protected_files():
        raise AssertionError("A protected research file changed during verification")
    report = {"notebooks": records, "protected_files_unchanged": len(before),
              "real_training_executed": False}
    if destination is not None:
        (destination / "verification.json").write_text(json.dumps(report, indent=2) + "\n")
        print(f"Executed copies and figures: {destination}")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
