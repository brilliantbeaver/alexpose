#!/usr/bin/env python3
"""Generate selected research tutorials; use --only to preserve earlier notebooks."""
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
    "11": "11_masking_patterns_and_coverage.ipynb",
    "12": "12_controlled_masking_pretraining.ipynb",
    "13": "13_masking_encoder_and_predictor_evaluation.ipynb",
    "14": "14_future_features_and_movement_prediction.ipynb",
    "15": "15_motion_weighted_masking.ipynb",
    "16": "16_structured_masking_and_context.ipynb",
    "17": "17_motion_and_structure_pretraining.ipynb",
    "18": "18_motion_information_and_readout.ipynb",
}

# Notebooks 17--18 operate on the real GAVD pretraining grid.  Their local
# setup script registers this kernel from .venv-cuda.  Keeping the preference
# in the generator matters: regenerating a notebook must not silently put the
# user back onto the CPU-only project ``python3`` kernel.
CUDA_NOTEBOOKS = frozenset({"17", "18"})
CUDA_KERNELSPEC = {
    "display_name": "GAVD5 CUDA (PyTorch 2.13)",
    "language": "python",
    "name": "gavd5-cuda",
}
DEFAULT_KERNELSPEC = {
    "display_name": "Python 3 (ipykernel)",
    "language": "python",
    "name": "python3",
}


def render(number: str) -> str:
    notebook = importlib.import_module(f"tutorials.research_{number}").build_notebook()
    notebook.metadata.update({
        "kernelspec": CUDA_KERNELSPEC if number in CUDA_NOTEBOOKS else DEFAULT_KERNELSPEC,
        "language_info": {"name": "python", "version": "3.11"},
        "research_status": ("GAVD development workflow; synthetic mode is an explicit software check."
                            if int(number) >= 15 else
                            "Exploratory extension; synthetic examples are non-evidentiary."),
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
    parser.add_argument(
        "--check-sources",
        action="store_true",
        help="compare cell types and sources while allowing retained execution outputs",
    )
    parser.add_argument("--only", choices=tuple(NOTEBOOKS), nargs="+", default=list(NOTEBOOKS))
    args = parser.parse_args()
    if args.check and args.check_sources:
        parser.error("choose either --check or --check-sources")
    for number in args.only:
        path = SUITE_ROOT / NOTEBOOKS[number]
        rendered = render(number)
        if args.check_sources:
            if not path.is_file():
                raise SystemExit(f"Missing {path.name}.")
            actual = nbformat.read(path, as_version=4)
            expected = nbformat.reads(rendered, as_version=4)
            actual_sources = [
                (cell.cell_type, cell.source) for cell in actual.cells
            ]
            expected_sources = [
                (cell.cell_type, cell.source) for cell in expected.cells
            ]
            if actual_sources != expected_sources:
                raise SystemExit(
                    f"Cell sources in {path.name} differ from its builder."
                )
        elif args.check:
            if not path.is_file() or path.read_text(encoding="utf-8") != rendered:
                raise SystemExit(f"Regenerate {path.name} with this builder.")
        else:
            path.write_text(rendered, encoding="utf-8")
        action = "Checked sources in" if args.check_sources else (
            "Checked" if args.check else "Built"
        )
        print(f"{action} {path.name}")


if __name__ == "__main__":
    main()
