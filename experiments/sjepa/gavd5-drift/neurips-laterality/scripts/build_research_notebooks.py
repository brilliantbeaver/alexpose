#!/usr/bin/env python3
"""Generate only the new 07–10 tutorials; never rewrite notebooks 00–06."""
from __future__ import annotations

import argparse
import hashlib
import importlib
from pathlib import Path
import sys

import nbformat

SUITE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SUITE_ROOT))
NOTEBOOKS = {
    "07": "07_research_questions_and_diagnostics.ipynb",
    "08": "08_matched_budget_masking.ipynb",
    "09": "09_symmetry_aware_jepa.ipynb",
    "10": "10_past_only_movement_prediction.ipynb",
}


def render(number: str) -> str:
    notebook = importlib.import_module(f"tutorials.research_{number}").build_notebook()
    notebook.metadata.update({
        "kernelspec": {"display_name": "Python 3 (ipykernel)", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.11"},
        "research_status": "Exploratory extension; synthetic examples are non-evidentiary.",
    })
    for index, cell in enumerate(notebook.cells):
        cell["id"] = hashlib.sha256(f"{number}:{index}:{cell.source}".encode()).hexdigest()[:12]
        if cell.cell_type == "code":
            cell.execution_count = None
            cell.outputs = []
            compile(cell.source, f"notebook_{number}_cell_{index}", "exec")
    nbformat.validate(notebook)
    return nbformat.writes(notebook) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="verify that notebooks match their tutorial sources")
    parser.add_argument("--only", choices=tuple(NOTEBOOKS), nargs="+", default=list(NOTEBOOKS))
    args = parser.parse_args()
    for number in args.only:
        path = SUITE_ROOT / NOTEBOOKS[number]
        rendered = render(number)
        if args.check:
            if not path.is_file() or path.read_text() != rendered:
                raise SystemExit(f"Regenerate {path.name} with this builder.")
        else:
            path.write_text(rendered)
        print(f"{'Checked' if args.check else 'Built'} {path.name}")


if __name__ == "__main__":
    main()
