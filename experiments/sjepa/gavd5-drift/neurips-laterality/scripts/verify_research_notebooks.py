#!/usr/bin/env python3
"""Check exploratory tutorials, optionally executing their small default runs."""
from __future__ import annotations

import argparse
import base64
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile

import nbformat
from nbclient import NotebookClient

SUITE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).resolve().parent))
from build_research_notebooks import NOTEBOOKS as ALL_RESEARCH_NOTEBOOKS, render

# Preserve this verifier's original progress/execution contracts. The separate
# comparative verifier handles tutorials 11–14 and their synthetic defaults.
NOTEBOOKS = {number: ALL_RESEARCH_NOTEBOOKS[number] for number in ("07", "08", "09", "10")}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execute-smoke", action="store_true", help="execute defaults in separate fresh kernels; no real-data training")
    parser.add_argument("--save-executed", action="store_true", help="retain output-bearing copies under executed/research_extensions")
    args = parser.parse_args()
    if args.save_executed and not args.execute_smoke:
        parser.error("--save-executed requires --execute-smoke")
    for number, name in NOTEBOOKS.items():
        path = SUITE_ROOT / name
        if path.read_text() != render(number):
            raise AssertionError(f"Tutorial source and notebook differ: {name}")
        notebook = nbformat.read(path, as_version=4)
        nbformat.validate(notebook)
        code_source = "\n".join(
            cell.source for cell in notebook.cells if cell.cell_type == "code"
        )
        required_progress = (
            (
                "from notebook_progress import NotebookTaskProgress",
                "local_training_progress.start(",
                "progress_callback=report_training",
                "inspect.signature(",
                "importlib.reload(masked_learning)",
            )
            if number == "08"
            else (
                "from notebook_progress import NotebookTaskProgress",
                "tutorial_progress.start(",
                "tutorial_progress.complete(",
            )
        )
        for required in required_progress:
            if required not in code_source:
                raise AssertionError(f"Notebook progress contract is missing {required!r}: {name}")
        for cell in notebook.cells:
            if cell.cell_type == "code" and (cell.execution_count is not None or cell.outputs):
                raise AssertionError(f"Canonical notebook must be output-free: {name}")
        if "/Users/" in path.read_text():
            raise AssertionError(f"Notebook includes an author-specific path: {name}")
        print(f"Structure/source check: PASS {name}", flush=True)
    subprocess.run([sys.executable, "-m", "unittest", "discover", "-s", str(SUITE_ROOT / "tests"),
                    "-p", "test_research_*.py", "-v"], cwd=SUITE_ROOT.parent, check=True)
    if not args.execute_smoke:
        return
    # Disable real-data grids, even if they are enabled in the invoking shell.
    prior = os.environ.copy()
    reports = []
    destination = None
    try:
        for name in list(os.environ):
            if name.startswith("LATERALITY_RESEARCH_"):
                os.environ.pop(name)
        os.environ["MPLBACKEND"] = "Agg"
        os.environ["PATH"] = str(Path(sys.executable).resolve().parent) + os.pathsep + os.environ.get("PATH", "")
        if args.save_executed:
            parent = SUITE_ROOT / "executed" / "research_extensions"
            parent.mkdir(parents=True, exist_ok=True)
            destination = Path(tempfile.mkdtemp(prefix="synthetic_checks_", dir=parent))
        for name in NOTEBOOKS.values():
            notebook = nbformat.read(SUITE_ROOT / name, as_version=4)
            NotebookClient(notebook, timeout=180, kernel_name="python3", allow_errors=False,
                           resources={"metadata": {"path": str(SUITE_ROOT)}}).execute()
            code_cells = [cell for cell in notebook.cells if cell.cell_type == "code"]
            if any(cell.execution_count is None for cell in code_cells):
                raise AssertionError(f"A cell did not execute: {name}")
            images = sum("image/png" in output.get("data", {}) for cell in code_cells for output in cell.outputs)
            svg = sum("image/svg+xml" in output.get("data", {}) for cell in code_cells for output in cell.outputs)
            if images + svg < 1:
                raise AssertionError(f"Expected an inline figure: {name}")
            html_outputs = [
                output.get("data", {}).get("text/html", "")
                for cell in code_cells
                for output in cell.outputs
                if "text/html" in output.get("data", {})
            ]
            if name != NOTEBOOKS["08"] and not any(
                'role="progressbar"' in html and "100.0%" in html
                for html in html_outputs
            ):
                raise AssertionError(f"Expected a completed notebook progress display: {name}")
            reports.append({"notebook": name, "code_cells_executed": len(code_cells), "png_figures": images, "svg_figures": svg,
                            "status": "PASS", "real_training_run": False})
            if destination is not None:
                nbformat.write(notebook, destination / name)
                figure_index = 0
                for cell in code_cells:
                    for output in cell.outputs:
                        if "image/png" in output.get("data", {}):
                            figure_index += 1
                            png = output["data"]["image/png"]
                            (destination / f"{Path(name).stem}_figure_{figure_index}.png").write_bytes(base64.b64decode(png))
                        if "image/svg+xml" in output.get("data", {}):
                            svg_text = output["data"]["image/svg+xml"]
                            if isinstance(svg_text, list):
                                svg_text = "".join(svg_text)
                            (destination / f"{Path(name).stem}_cell_{cell.id}.svg").write_text(svg_text)
            print(f"Fresh kernel: PASS {name} ({len(code_cells)} code cells, {images} PNG, {svg} SVG)", flush=True)
    finally:
        os.environ.clear()
        os.environ.update(prior)
    if destination is not None:
        (destination / "verification.json").write_text(json.dumps(reports, indent=2) + "\n")
        print(f"Executed review copies: {destination}")


if __name__ == "__main__":
    main()
