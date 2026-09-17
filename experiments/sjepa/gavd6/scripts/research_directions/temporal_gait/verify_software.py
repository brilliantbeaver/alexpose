"""Run the complete bounded CPU fixture in fresh notebook kernels; never real data.

The separate real-mode notebook is PLAN ONLY with intentionally nonexistent
manifest paths. This command does not certify scientific or HAIC readiness.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import sys
import time

from execute_notebook import NOTEBOOKS, SOURCE, execute_notebook

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src"))


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


@contextmanager
def fixture_environment(output):
    """Prevent inherited real-mode overrides; every kernel gets explicit config."""
    original = dict(os.environ)
    try:
        for key in list(os.environ):
            if key.startswith("TG_"):
                del os.environ[key]
        os.environ.update(GAVD6_ROOT=str(ROOT), MPLCONFIGDIR=str(output / ".matplotlib"),
                          CUBLAS_WORKSPACE_CONFIG=":4096:8", OMP_NUM_THREADS="1",
                          MKL_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1")
        yield
    finally:
        os.environ.clear()
        os.environ.update(original)


def verification_plan(cfg):
    from gavd6_sjepa.research_directions.temporal_gait.workflow import plan_tasks
    steps = [("00", "inventory", None), ("01", "prepare", None), ("02", "audit", None)]
    for task in plan_tasks(cfg, "pilot"):
        masked = task["arm"] in ("masked_index", "masked")
        steps.append(("03" if masked else "04", "masked" if masked else "future", task["task_id"]))
    return steps + [("05", "extensions", None), ("05", "cache-video", None),
                    ("06", "evaluate", None), ("07", "calibrate", None),
                    ("07", "test", None), ("08", "aggregate", None)]


def verify_software(run_root, *, timeout=1200):
    from gavd6_sjepa.research_directions.temporal_gait.config import MANIFEST_FIELDS, RunConfig
    from gavd6_sjepa.research_directions.temporal_gait.contracts import (
        atomic_json, read_json, verify_receipt, verify_run, write_once_json)

    requested = Path(run_root)
    if not requested.is_absolute() or requested.is_symlink():
        raise ValueError("Supply a new explicit absolute, non-symlink --run-root")
    output = requested.resolve()
    if output.exists() and (not output.is_dir() or any(output.iterdir())):
        raise FileExistsError("Software verification requires a new or empty output directory")
    if timeout <= 0:
        raise ValueError("Notebook cell timeout must be positive")
    # This is a fixed, small fixture, not an arbitrary configurable training entrypoint.
    payload = json.loads((ROOT / "slurm/temporal-gait/synthetic.example.json").read_text())
    payload["run_root"] = str(output)
    cfg = RunConfig(**payload).validate(check_input_paths=False)
    if cfg.mode != "synthetic" or cfg.device != "cpu" or cfg.updates > 2 or cfg.direct_updates > 2:
        raise ValueError("The shipped verifier fixture must remain bounded synthetic CPU work")
    if any(getattr(cfg, name) for name in MANIFEST_FIELDS):
        raise ValueError("The software fixture cannot contain private manifests")
    output.mkdir(parents=True, exist_ok=True)
    (output / ".matplotlib").mkdir()
    config_path = output / "fixture-config.json"
    write_once_json(config_path, cfg.to_dict())
    report = {"mode": "SYNTHETIC", "status": "running", "run_root": str(output),
              "real_data_accessed": False, "real_scientific_readiness": False,
              "started_utc": datetime.now(timezone.utc).isoformat(), "notebooks": [],
              "source_notebooks_sha256": {name: sha256(SOURCE / name) for name in NOTEBOOKS.values()},
              "note": "Software execution only; E3/RGB are gated, not trained comparators."}
    report_path = output / "verification.json"

    def run_one(number, stage, task_id, index, *, selected_config=config_path,
                selected_root=output, execute=True):
        folder = selected_root / "notebook_runs" / f"verify-{index:02d}-{stage}"
        record = {"notebook": number, "stage": stage, "task_id": task_id,
                  "phase": "pilot", "execution_enabled": execute,
                  "output_path": str(folder / NOTEBOOKS[number]), "status": "running"}
        report["notebooks"].append(record)
        started = time.perf_counter()
        try:
            destination = execute_notebook(number, run_root=selected_root,
                config=selected_config, output_dir=folder, stage=stage, phase="pilot",
                task_id=task_id, execute=execute, timeout=timeout)
            import nbformat
            notebook = nbformat.read(destination, as_version=4)
            if notebook.metadata.temporal_gait_execution.status != "completed":
                raise RuntimeError("Executed notebook lacks a completion record")
            if stage in ("extensions", "cache-video") and "gated_not_implemented" not in json.dumps(notebook):
                raise AssertionError("An unimplemented extension must report its gate")
            record.update(status="completed", executed_sha256=sha256(destination))
        except BaseException as error:
            record.update(status="failed", error=f"{type(error).__name__}: {error}")
            raise
        finally:
            record["elapsed_seconds"] = round(time.perf_counter() - started, 3)
            atomic_json(report_path, report)

    atomic_json(report_path, report)
    try:
        with fixture_environment(output):
            for index, (number, stage, task_id) in enumerate(verification_plan(cfg)):
                run_one(number, stage, task_id, index)
            identity = verify_run(cfg)
            for path in sorted((output / "receipts").glob("*.json")):
                verify_receipt(cfg, read_json(path)["stage"])
            decision = read_json(output / "decisions/development.json")
            claims = read_json(output / "reports/claim-audit.json")
            if decision["mode"] != "synthetic" or decision["ready_for_expansion"]:
                raise AssertionError("A fixture may not authorize real compute expansion")
            if claims["mode"] != "synthetic" or claims["real_runs_complete"] or claims["confirmatory_claim"]:
                raise AssertionError("Synthetic outputs must retain their claim limits")
            report["frozen_identity"] = identity
            report["core_status"] = claims["status"]
            # An additional real-mode plan must work without touching these nonexistent inputs.
            real_root = output / "real-plan-only"
            real_payload = cfg.to_dict()
            real_payload.update(mode="real", run_root=str(real_root))
            for name in MANIFEST_FIELDS:
                real_payload[name] = str(output / "intentionally-nonexistent-inputs" / f"{name}.json")
            real_cfg = RunConfig(**real_payload).validate(check_input_paths=False)
            real_config = output / "real-plan-config.json"
            write_once_json(real_config, real_cfg.to_dict())
            run_one("00", "inventory", None, len(report["notebooks"]),
                    selected_config=real_config, selected_root=real_root, execute=False)
            if (real_root / "config/identity.json").exists() or (real_root / "receipts").exists():
                raise AssertionError("A real plan-only notebook must not run inventory")
            report["real_plan_only_status"] = "completed_without_input_manifests"
        report["status"] = "software_verified_only"
    except BaseException as error:
        report.update(status="failed", error=f"{type(error).__name__}: {error}")
        raise
    finally:
        report["finished_utc"] = datetime.now(timezone.utc).isoformat()
        atomic_json(report_path, report)
        print(f"Software verification report: {report_path}", flush=True)
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, required=True,
                        help="New or empty absolute local output directory; no real-mode option exists")
    parser.add_argument("--timeout", type=int, default=1200, help="Positive per-cell timeout in seconds")
    args = parser.parse_args(argv)
    return verify_software(args.run_root, timeout=args.timeout)


if __name__ == "__main__":
    main()
