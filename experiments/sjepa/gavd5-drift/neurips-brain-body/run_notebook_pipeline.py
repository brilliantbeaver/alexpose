"""Execute the BrainBodyFM notebooks in dependency order with an audit report."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import time
import traceback


# Windows terminals commonly default to cp1252. Notebook tracebacks contain
# mathematical symbols used in leakage assertions; make reporting robust so a
# scientific error is never masked by a second encoding error.
for stream in (os.sys.stdout, os.sys.stderr):
    if hasattr(stream, "reconfigure"):
        stream.reconfigure(encoding="utf-8", errors="backslashreplace")

NUMBERED = [
    "00_sjepa_from_first_principles.ipynb",
    "01_gavd_manifest_and_youtube.ipynb",
    "02_extract_and_watch_skeletons.ipynb",
    "03_neurologic_keypoint_masking.ipynb",
    "04_pretrain_sjepa_on_normal.ipynb",
    "05_inspect_latent_motion.ipynb",
    "06_capstone_health_condition_classifiers.ipynb",
    "07_temporal_readout_diagnostic.ipynb",
    "08_normal_anchor_drift_and_consolidation.ipynb",
    "09_predictive_surprise_world_model.ipynb",
]
LATERALITY = [
    "nb_05a_signed_laterality_probe.ipynb",
    "nb_05b_reflection_reach_and_futures.ipynb",
    "nb_05c_reflection_equivariant_readout.ipynb",
    "nb_05d_reflection_equivariant_encoder.ipynb",
]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("smoke", "real"), default="smoke")
    parser.add_argument("--outer-fold", type=int, default=0)
    parser.add_argument("--model-seed", type=int, default=42)
    parser.add_argument("--objective", choices=("label_free", "group_ablation"), default="label_free")
    parser.add_argument("--timeout", type=int, default=1800)
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--stop", type=int, default=10)
    parser.add_argument("--include-laterality", action="store_true")
    parser.add_argument("--continue-on-error", action="store_true")
    args = parser.parse_args()

    notebook_dir = Path(__file__).resolve().parent
    experiment_dir = notebook_dir.parent
    selected = NUMBERED[args.start : args.stop]
    if args.include_laterality:
        selected += LATERALITY

    runtime_dir = experiment_dir / "work" / "cache" / "jupyter-runtime"
    runtime_dir.mkdir(parents=True, exist_ok=True)
    temp_dir = experiment_dir / "work" / "cache" / "tmp"
    temp_dir.mkdir(parents=True, exist_ok=True)
    objective_name = (
        "jepa_vicreg" if args.objective == "label_free" else "jepa_vicreg_group_ablation"
    )
    os.environ.update(
        {
            "GAVD_MODE": args.mode,
            "SJEPA_OUTER_FOLD": str(args.outer_fold),
            "SJEPA_MODEL_SEED": str(args.model_seed),
            # Notebooks 04-06 use this boolean to name their objective, while
            # notebooks 07-09 consume the canonical artifact stem. Keep both
            # views synchronized so a pipeline run cannot search for the
            # nonexistent ``*_label_free.pt`` bundle.
            "SJEPA_ENABLE_GROUP_LOSS": "1" if args.objective == "group_ablation" else "0",
            "SJEPA_OBJECTIVE": objective_name,
            "JUPYTER_RUNTIME_DIR": str(runtime_dir),
            # Managed Windows workspaces can reject jupyter_core's DACL call.
            "JUPYTER_ALLOW_INSECURE_WRITES": "true",
            "IPYTHONDIR": str(experiment_dir / "work" / "cache" / "ipython"),
            "MPLBACKEND": "Agg",
            "MPLCONFIGDIR": str(experiment_dir / "cache" / "matplotlib"),
            # Keep animation and kernel temporary files inside the writable tree.
            "TEMP": str(temp_dir),
            "TMP": str(temp_dir),
            "TMPDIR": str(temp_dir),
        }
    )
    # jupyter_core reads its insecure-write setting at import time on Windows.
    import nbformat
    from nbclient import NotebookClient

    if args.mode == "smoke":
        os.environ.update(
            {
                "GAVD_DOWNLOAD": "0",
                "GAVD_EXTRACT_POSES": "0",
                "SJEPA_RUN_PROFILE": "smoke",
            }
        )

    started = datetime.now(timezone.utc).replace(microsecond=0).isoformat()
    rows: list[dict[str, object]] = []
    exit_code = 0
    for name in selected:
        path = notebook_dir / name
        begin = time.perf_counter()
        print(f"\n=== {name} ===", flush=True)
        row: dict[str, object] = {"notebook": name}
        try:
            notebook = nbformat.read(path, as_version=4)
            _, notebook = nbformat.validator.normalize(notebook)
            client = NotebookClient(
                notebook,
                timeout=args.timeout,
                kernel_name="python3",
                resources={"metadata": {"path": str(experiment_dir)}},
            )
            client.execute()
            nbformat.write(notebook, path)
            row["status"] = "passed"
            print("PASS", flush=True)
        except Exception as exc:  # report the full dependency boundary
            exit_code = 1
            row.update(
                {
                    "status": "failed",
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                    "traceback": traceback.format_exc(),
                }
            )
            print(f"FAIL: {type(exc).__name__}: {exc}", flush=True)
            if not args.continue_on_error:
                rows.append(row)
                break
        finally:
            row["duration_seconds"] = round(time.perf_counter() - begin, 3)
        rows.append(row)

    report = {
        "started_at_utc": started,
        "finished_at_utc": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
        "mode": args.mode,
        "outer_fold": args.outer_fold,
        "model_seed": args.model_seed,
        "objective": objective_name,
        "python": os.sys.executable,
        "results": rows,
    }
    report_dir = experiment_dir / "work" / "artifacts" / args.mode
    report_dir.mkdir(parents=True, exist_ok=True)
    report_path = report_dir / "notebook_execution_report.json"
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"\nreport: {report_path}")
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
