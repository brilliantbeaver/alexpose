#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import nbformat
from nbclient import NotebookClient


SUITE_ROOT = Path(__file__).resolve().parents[1]
if str(SUITE_ROOT) not in sys.path:
    sys.path.insert(0, str(SUITE_ROOT))

from laterality.notebook_outputs import audit_inline_outputs  # noqa: E402


NOTEBOOKS = (
    "00_protocol_and_governance.ipynb",
    "01_cohort_and_target_audit.ipynb",
    "02_source_level_splits.ipynb",
    "03_fold_local_training.ipynb",
    "04_held_out_evaluation.ipynb",
    "05_aggregate_statistics.ipynb",
    "06_external_subject_gate.ipynb",
)


def verify_notebook_structure() -> None:
    for name in NOTEBOOKS:
        path = SUITE_ROOT / name
        if not path.is_file():
            raise AssertionError(f"Missing notebook: {path}")
        notebook = nbformat.read(path, as_version=4)
        nbformat.validate(notebook)
        serialized = path.read_text()
        if "/Users/" in serialized or "work.experiments" in serialized:
            raise AssertionError(f"Notebook is not portable/clean-room: {name}")
        for index, cell in enumerate(notebook.cells):
            if cell.cell_type == "code":
                if cell.get("execution_count") is not None:
                    raise AssertionError(f"Stored execution count in {name} cell {index}")
                if cell.get("outputs"):
                    raise AssertionError(f"Stored output in {name} cell {index}")
    print(f"notebook structure: PASS ({len(NOTEBOOKS)} clean notebooks)")


def run_tests() -> None:
    subprocess.run(
        [
            sys.executable,
            "-m",
            "unittest",
            "discover",
            "-s",
            str(SUITE_ROOT / "tests"),
            "-v",
        ],
        cwd=SUITE_ROOT.parent,
        check=True,
    )


def execute_smoke_notebooks() -> None:
    prior = dict(os.environ)
    try:
        with tempfile.TemporaryDirectory(prefix="neurips-laterality-smoke-") as temporary:
            os.environ["LATERALITY_PROFILE"] = "smoke"
            os.environ["LATERALITY_ARTIFACT_ROOT"] = temporary
            os.environ.pop("LATERALITY_EXTERNAL_MANIFEST", None)
            os.environ.pop("LATERALITY_EXTERNAL_GOVERNANCE", None)
            os.environ.pop("LATERALITY_EXTERNAL_POSE_ROOT", None)
            os.environ["LATERALITY_DISABLE_EXTERNAL_DOTENV"] = "1"
            os.environ["PATH"] = (
                str(Path(sys.executable).resolve().parent)
                + os.pathsep
                + os.environ.get("PATH", "")
            )
            for name in NOTEBOOKS:
                notebook = nbformat.read(SUITE_ROOT / name, as_version=4)
                client = NotebookClient(
                    notebook,
                    timeout=900,
                    kernel_name="python3",
                    allow_errors=False,
                    resources={"metadata": {"path": str(SUITE_ROOT)}},
                )
                client.execute(cwd=str(SUITE_ROOT))
                inline_audit = audit_inline_outputs(notebook, name=name)
                print(
                    f"fresh-kernel smoke: PASS {name} "
                    f"(figures={inline_audit['inline_png_figures']}, "
                    f"results={inline_audit['inline_result_payloads']})"
                )
            summary_path = Path(temporary) / "report" / "summary.json"
            summary = json.loads(summary_path.read_text())
            if not summary.get("synthetic_evidence") or summary.get("submission_ready"):
                raise AssertionError("Smoke report must be synthetic and submission-blocked")
    finally:
        os.environ.clear()
        os.environ.update(prior)


def main() -> None:
    parser = argparse.ArgumentParser(description="Verify the clean laterality suite")
    parser.add_argument(
        "--execute-smoke",
        action="store_true",
        help="also execute all notebooks in fresh kernels with isolated synthetic artifacts",
    )
    arguments = parser.parse_args()
    verify_notebook_structure()
    run_tests()
    if arguments.execute_smoke:
        execute_smoke_notebooks()
    print("neurips-laterality verification: PASS")


if __name__ == "__main__":
    main()
