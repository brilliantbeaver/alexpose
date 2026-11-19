"""Execute notebooks12 through17 with fresh kernels and one verified real run.

Executed copies and a complete error-aware report stay under the run directory.
Authored notebooks and historical notebook execution records are preserved.
"""
from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import sys
import tempfile
import time
import traceback
from urllib.parse import urlsplit, urlunsplit, unquote

import nbformat
from nbclient import NotebookClient
from jupyter_client import AsyncKernelManager
from jupyter_client.kernelspec import KernelSpecManager

BASE = Path(__file__).resolve().parents[1]
if str(BASE) not in sys.path:
    sys.path.insert(0, str(BASE))
from aaai_gait.iteration02_notebook_support import EvidenceRun, file_hash


def relocate_markdown_links(book, output: Path):
    """Keep executed copies navigable without changing any scientific code."""
    changed = 0
    def replace(match):
        nonlocal changed
        target = match.group(1)
        parsed = urlsplit(target)
        if parsed.scheme or parsed.netloc or not parsed.path or parsed.path.startswith("/"):
            return match.group(0)
        destination = (BASE / unquote(parsed.path)).resolve()
        if not destination.exists():
            return match.group(0)
        relative = os.path.relpath(destination, output)
        changed += 1
        return "](" + urlunsplit(("", "", relative, parsed.query, parsed.fragment)) + ")"
    for cell in book.cells:
        if cell.cell_type == "markdown":
            cell.source = re.sub(r"\]\(([^)]+)\)", replace, cell.source)
    return changed


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--only", choices=[str(value) for value in range(12, 18)])
    parser.add_argument("--historical", action="store_true", help="Replay navigation-updated notebooks00–11 with isolated output writes.")
    parser.add_argument("--allow-development", action="store_true")
    parser.add_argument("--cell-timeout", type=int, default=360)
    args = parser.parse_args()
    if args.historical and args.only:
        parser.error("--historical and --only cannot be combined")
    os.environ["AAAI_ITERATION02_RUN_DIR"] = str(args.run_dir.resolve())
    os.environ["AAAI_DEMO"] = "0"
    if args.allow_development:
        os.environ["AAAI_ITERATION02_ALLOW_DEVELOPMENT"] = "1"
    else:
        os.environ.pop("AAAI_ITERATION02_ALLOW_DEVELOPMENT", None)
    run = EvidenceRun(args.run_dir)
    output = run.path / ("historical_notebook_checks" if args.historical else "executed_notebooks")
    output.mkdir(exist_ok=True)
    report_path = output / (f"execution_report_{args.only}.json" if args.only else "execution_report.json")
    report = {"protocol_hash": run.protocol_hash, "run_directory": str(run.path),
              "mode": run.mode, "data_kind": "real_gavd", "python": sys.executable,
              "producer": "scripts/execute_iteration02_notebooks.py", "notebooks": [], "status": "running"}
    report_path.write_text(json.dumps(report, indent=2))
    paths = ([path for path in sorted(BASE.glob("[01][0-9]_*.ipynb")) if int(path.name[:2]) < 12]
             if args.historical else
             [path for path in sorted(BASE.glob("1[2-7]_*.ipynb")) if not args.only or path.name.startswith(args.only)])
    expected = 12 if args.historical else 1 if args.only else 6
    if len(paths) != expected:
        raise RuntimeError("Generate all six expected notebooks before execution.")
    old_artifacts = BASE / "artifacts/gavd"
    protected = {str(path.relative_to(old_artifacts)): file_hash(path)
                 for path in old_artifacts.rglob("*") if path.is_file()} if args.historical else {}
    replay = output / "replay_artifacts"
    if args.historical:
        shutil.copytree(old_artifacts, replay / "gavd", dirs_exist_ok=True)
        os.environ["AAAI_EPOCHS"] = "3"
        os.environ["AAAI_RUN_SPE_ABLATION"] = "0"
        report["historical_protection"] = {
            "original_artifacts": str(old_artifacts), "protected_files": len(protected),
            "replay_directory": str(replay), "inputs_and_scientific_code_unchanged": True,
            "explanation": "Only the runtime Settings.run_dir property routes writes to a copied cache. Settings fields and checkpoint identities are preserved.",
        }
        report_path.write_text(json.dumps(report, indent=2))
    with tempfile.TemporaryDirectory(prefix="aaai-iteration02-nb-", dir="/tmp") as temporary:
        spec_dir = Path(temporary) / "aaai-iteration02"
        spec_dir.mkdir()
        (spec_dir / "kernel.json").write_text(json.dumps({
            "argv": [sys.executable, "-m", "ipykernel_launcher", "-f", "{connection_file}"],
            "display_name": "AAAI iteration02 verified real run", "language": "python",
        }))
        manager = KernelSpecManager(kernel_dirs=[temporary])
        for path in paths:
            started = time.monotonic()
            book = nbformat.read(path, as_version=4)
            nbformat.validate(book)
            relocated_links = relocate_markdown_links(book, output)
            if args.historical:
                prelude = f'''# Validation-only output isolation. The authored scientific cells are unchanged.
import sys
from pathlib import Path
sys.path.insert(0, {str(BASE)!r})
import aaai_gait.settings as _historical_settings
_OriginalSettings = _historical_settings.Settings
class _ReplaySettings(_OriginalSettings):
    @property
    def run_dir(self):
        return Path({str(replay)!r}) / ("synthetic_demo" if self.demo else "gavd")
_historical_settings.Settings = _ReplaySettings
import aaai_gait.workflow as _historical_workflow
import aaai_gait.study_runner as _historical_study
_historical_workflow.Settings = _ReplaySettings
_historical_study.Settings = _ReplaySettings
print("Historical replay: existing real inputs; all capstone output writes are isolated.")
'''
                book.cells.insert(0, nbformat.v4.new_code_cell(prelude, metadata={"tags": ["validation-output-isolation"]}))
            client = NotebookClient(book, timeout=args.cell_timeout, kernel_name="aaai-iteration02",
                                    resources={"metadata": {"path": str(BASE)}})
            client.km = AsyncKernelManager(kernel_name="aaai-iteration02", kernel_spec_manager=manager,
                                          transport="ipc", ip=str(Path(temporary) / f"ipc-{path.name[:2]}"))
            record = {"notebook": path.name, "source_sha256": file_hash(path), "status": "running",
                      "executed_copy_markdown_links_relocated": relocated_links}
            try:
                client.execute()
                errors = [out for cell in book.cells if cell.cell_type == "code" for out in cell.get("outputs", []) if out.get("output_type") == "error"]
                if errors:
                    raise RuntimeError("An executed notebook contains an error output.")
                record["status"] = "passed"
                book.metadata.setdefault("iteration02", {})["status"] = ("historical real-data replay with isolated writes" if args.historical else "executed against verified real run")
                book.metadata["iteration02"]["protocol_hash"] = run.protocol_hash
            except Exception as error:
                record.update(status="failed", error_type=type(error).__name__, error=str(error), traceback=traceback.format_exc())
            finally:
                if client.km is not None and client.km.has_kernel:
                    asyncio.run(client.km.shutdown_kernel(now=True))
                executed_path = output / path.name
                nbformat.write(book, executed_path)
                record.update(seconds=round(time.monotonic() - started, 2), executed_path=str(executed_path),
                              executed_sha256=file_hash(executed_path), cells=len(book.cells),
                              executed_code_cells=sum(cell.cell_type == "code" and cell.execution_count is not None for cell in book.cells))
                report["notebooks"].append(record)
                report_path.write_text(json.dumps(report, indent=2))
                print(json.dumps({key: record[key] for key in ("notebook", "status", "seconds")}), flush=True)
            if record["status"] != "passed":
                report["status"] = "failed"
                report_path.write_text(json.dumps(report, indent=2))
                raise RuntimeError(f"Notebook failed; inspect {executed_path} and {report_path}.")
    report["status"] = "passed"
    report["all_six_independent"] = len(paths) == 6 and all(record["status"] == "passed" for record in report["notebooks"])
    if args.historical:
        current = {str(path.relative_to(old_artifacts)): file_hash(path) for path in old_artifacts.rglob("*") if path.is_file()}
        if current != protected:
            raise RuntimeError("Historical capstone artifacts changed during replay. Inspect the isolation audit.")
        report["historical_artifact_bytes_unchanged"] = True
        report["all_twelve_independent"] = all(record["status"] == "passed" for record in report["notebooks"])
    report_path.write_text(json.dumps(report, indent=2))
    print(f"Execution report: {report_path}")


if __name__ == "__main__":
    main()
