#!/usr/bin/env python3
"""Execute canonical notebooks and save profile-labeled copies with inline output."""

from __future__ import annotations

import argparse
import os
import sys
import tempfile
from contextlib import nullcontext
from datetime import datetime, timezone
from pathlib import Path

import nbformat
from nbclient import NotebookClient


SUITE_ROOT = Path(__file__).resolve().parents[1]
if str(SUITE_ROOT) not in sys.path:
    sys.path.insert(0, str(SUITE_ROOT))

from laterality.config import load_context  # noqa: E402
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


def _output_directory(profile: str, digest: str, setting: str | None) -> Path:
    if setting:
        output = Path(setting).expanduser().resolve()
    else:
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        output = (
            SUITE_ROOT
            / "executed"
            / profile
            / f"protocol_{digest[:12]}"
            / timestamp
        )
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(
            f"Executed-notebook directory is not empty: {output}. "
            "Choose a new directory so prior evidence is not overwritten."
        )
    output.mkdir(parents=True, exist_ok=True)
    return output


def execute(profile: str, output_setting: str | None) -> Path:
    context = load_context(SUITE_ROOT / "config" / "protocol.json", profile=profile)
    output = _output_directory(profile, context.protocol_digest, output_setting)
    temporary_context = (
        tempfile.TemporaryDirectory(prefix="neurips-laterality-inline-smoke-")
        if profile == "smoke"
        else nullcontext(None)
    )
    prior = dict(os.environ)
    try:
        with temporary_context as temporary:
            os.environ["LATERALITY_PROFILE"] = profile
            if temporary is not None:
                os.environ["LATERALITY_ARTIFACT_ROOT"] = temporary
                os.environ.pop("LATERALITY_EXTERNAL_MANIFEST", None)
                os.environ.pop("LATERALITY_EXTERNAL_GOVERNANCE", None)
                os.environ.pop("LATERALITY_EXTERNAL_POSE_ROOT", None)
            os.environ["PATH"] = (
                str(Path(sys.executable).resolve().parent)
                + os.pathsep
                + os.environ.get("PATH", "")
            )
            for name in NOTEBOOKS:
                notebook = nbformat.read(SUITE_ROOT / name, as_version=4)
                notebook.metadata["laterality_execution"] = {
                    "profile": profile,
                    "synthetic_evidence": profile == "smoke",
                    "protocol_digest": context.protocol_digest,
                    "executed_at_utc": datetime.now(timezone.utc).isoformat(),
                    "status": "running",
                }
                destination = output / name
                try:
                    NotebookClient(
                        notebook,
                        timeout=900,
                        kernel_name="python3",
                        allow_errors=False,
                        resources={"metadata": {"path": str(SUITE_ROOT)}},
                    ).execute(cwd=str(SUITE_ROOT))
                    inline_audit = audit_inline_outputs(notebook, name=name)
                except Exception:
                    notebook.metadata["laterality_execution"]["status"] = "failed"
                    nbformat.write(notebook, destination, version=4)
                    raise
                notebook.metadata["laterality_execution"]["inline_output_audit"] = (
                    inline_audit
                )
                notebook.metadata["laterality_execution"]["status"] = "complete"
                nbformat.write(notebook, destination, version=4)
                print(
                    f"saved inline {profile} notebook: {destination} "
                    f"(figures={inline_audit['inline_png_figures']}, "
                    f"results={inline_audit['inline_result_payloads']})"
                )
    finally:
        os.environ.clear()
        os.environ.update(prior)
    return output


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Execute the clean notebooks and save separate copies containing inline "
            "tables and figures. Canonical source notebooks remain output-free."
        )
    )
    parser.add_argument("--profile", choices=("smoke", "paper"), default="smoke")
    parser.add_argument("--output-dir")
    parser.add_argument(
        "--confirm-paper-run",
        action="store_true",
        help="required for the 50-checkpoint paper computation",
    )
    arguments = parser.parse_args()
    if arguments.profile == "paper" and not arguments.confirm_paper_run:
        parser.error("paper execution requires --confirm-paper-run")
    output = execute(arguments.profile, arguments.output_dir)
    print(f"executed notebook set: {output}")


if __name__ == "__main__":
    main()
