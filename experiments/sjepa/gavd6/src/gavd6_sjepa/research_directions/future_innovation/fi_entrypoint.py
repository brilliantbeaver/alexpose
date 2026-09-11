"""Lazy command handlers for the Future Innovation Experiment 0 pipeline."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path


def parser_for(command):
    parser = argparse.ArgumentParser(prog=f"gavd6 future-innovation {command}")
    parser.add_argument(
        "--run-root",
        type=Path,
        default=os.environ.get("FI_RUN_ROOT"),
        help="Explicit versioned experiment directory (or FI_RUN_ROOT).",
    )
    return parser


def parsed(parser):
    args = parser.parse_args()
    if args.run_root is None:
        parser.error("--run-root is required (or set FI_RUN_ROOT)")
    args.run_root = args.run_root.expanduser().resolve()
    return args


def configure_threads():
    import torch

    torch.set_num_threads(int(os.environ.get("FI_TORCH_THREADS", "1")))


def init_main():
    parser = parser_for("init-run")
    parser.add_argument("--sequence-manifest", type=Path, required=True)
    parser.add_argument("--video-manifest", type=Path, required=True)
    parser.add_argument("--annotations", type=Path, nargs="+", required=True)
    parser.add_argument("--youtube-dir", type=Path, required=True)
    parser.add_argument("--video-root", action="append", type=Path,
                        default=[Path(p) for p in os.environ.get("FI_VIDEO_ROOTS", "").split(os.pathsep) if p],
                        help="Additional full-source storage directory; repeat or set FI_VIDEO_ROOTS (colon-separated on HAIC).")
    parser.add_argument("--pose-model", type=Path, required=True)
    parser.add_argument("--vjepa-root", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--protocol", choices=("direct-v2", "legacy-v1"), default="direct-v2")
    parser.add_argument(
        "--change-reason",
        default="Experiment 0 initialization",
        help="Optional description of this run.",
    )
    args = parsed(parser)
    from .fi_contracts import VJEPA_COMMIT, initialize_run

    commit = subprocess.check_output(
        ["git", "-C", str(args.vjepa_root), "rev-parse", "HEAD"], text=True
    ).strip()
    if commit != VJEPA_COMMIT:
        parser.error(f"Teacher must be pinned to {VJEPA_COMMIT}")
    import torch

    for directory in (args.youtube_dir, *args.video_root):
        if not directory.is_dir():
            parser.error(f"Full-source storage directory is unavailable: {directory}")
    root = initialize_run(
        args.run_root,
        sequence_manifest=args.sequence_manifest,
        video_manifest=args.video_manifest,
        annotations=args.annotations,
        pose_model=args.pose_model,
        vjepa_root=args.vjepa_root,
        checkpoint=args.checkpoint,
        change_reason=args.change_reason,
        youtube_dir=args.youtube_dir,
        video_roots=args.video_root,
        protocol=args.protocol,
    )
    environment = [
        f"Python: {sys.version}",
        f"Torch: {torch.__version__}",
        f"CUDA: {torch.version.cuda}",
        f"Executable: {sys.executable}",
    ]
    freeze = subprocess.run(
        ["uv", "pip", "freeze", "--python", sys.executable],
        capture_output=True,
        text=True,
        check=True,
    )
    environment += [freeze.stdout]
    (root / "config/environment.txt").write_text("\n".join(environment))
    from gavd6_sjepa.shared_infrastructure.artifact_io_operations import sha256_file

    from .fi_contracts import write_once_json

    write_once_json(
        root / "config/environment-contract.json",
        {"sha256": sha256_file(root / "config/environment.txt")},
    )
    print(f"Preregistered {root}")


def cohort_main():
    args = parsed(parser_for("build-cohort"))
    from .fi_cohort import build_candidates
    from .fi_contracts import check_run, stage_lock

    paths = check_run(args.run_root)["input_paths"]
    with stage_lock(args.run_root, "cohort"):
        build_candidates(args.run_root, **paths)


def poses_main():
    args = parsed(parser_for("extract-poses"))
    from .fi_cohort import extract_and_freeze
    from .fi_contracts import stage_lock

    with stage_lock(args.run_root, "poses"):
        extract_and_freeze(args.run_root)


def teacher_arguments(command):
    parser = parser_for(command)
    parser.add_argument("--device", choices=["cuda", "cpu"], default="cuda")
    return parsed(parser)


def record_teacher_runtime(root, stage):
    import torch

    from .fi_contracts import write_once_json

    # Slurm requeues retain the job ID but may change nodes/devices. Logging must
    # not reject that legitimate retry because an earlier attempt already exists.
    attempt = f"{os.environ.get('SLURM_JOB_ID', 'local')}-{os.getpid()}-{time.time_ns()}"
    write_once_json(
        root / "logs" / f"{stage}-runtime-{attempt}.json",
        {
            "python": sys.version,
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "gpu": torch.cuda.get_device_name() if torch.cuda.is_available() else None,
            "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        },
    )


def cache_main():
    args = teacher_arguments("cache-teacher")
    configure_threads()
    from .fi_contracts import stage_lock
    from .fi_feature_cache import cache_teacher, load_cache
    from .fi_vjepa_adapter import FrozenVJEPAAdapter

    with stage_lock(args.run_root, "cache"):
        if (args.run_root / "config/cache-contract.json").exists():
            cohort, _ = load_cache(args.run_root)
            print(f"Reusing verified teacher cache for {len(cohort)} windows")
            return
        record_teacher_runtime(args.run_root, "cache")
        cache_teacher(
            args.run_root, FrozenVJEPAAdapter.from_run(args.run_root, args.device)
        )


def audits_main():
    args = teacher_arguments("audit-teacher")
    configure_threads()
    from .fi_contracts import stage_lock, audit_summary_path, protocol_name, check_run, DIRECT_PROTOCOL
    from .fi_validity_audits import ValidityAuditRejected, require_audits, run_audits
    from .fi_vjepa_adapter import FrozenVJEPAAdapter

    with stage_lock(args.run_root, "audits"):
        if audit_summary_path(args.run_root).exists():
            try:
                require_audits(args.run_root)
            except ValidityAuditRejected as error:
                print(json.dumps({"validity_audits_passed": False, "reused": True,
                                  "failed_checks": error.failed_checks, "reason": str(error)}))
                return 2
            print(json.dumps({"validity_audits_passed": True, "reused": True}))
            return 0
        record_teacher_runtime(args.run_root, "audit")
        from .fi_readiness import run_readiness
        audit = run_readiness if protocol_name(check_run(args.run_root)) == DIRECT_PROTOCOL else run_audits
        passed = audit(
            args.run_root, FrozenVJEPAAdapter.from_run(args.run_root, args.device)
        )
    print(json.dumps({"validity_audits_passed": passed}))
    return 0 if passed else 2


def fit_main():
    parser = parser_for("run-gate")
    parser.add_argument(
        "--outer-fold",
        type=int,
        choices=range(5),
        help="One array task; omitted runs all five folds.",
    )
    parser.add_argument("--device", choices=["cuda", "cpu"], default="cpu")
    args = parsed(parser)
    configure_threads()
    from .fi_contracts import stage_lock
    from .fi_nested_training import fit_outer_fold

    for fold in range(5) if args.outer_fold is None else [args.outer_fold]:
        with stage_lock(args.run_root, f"fold-{fold}"):
            fit_outer_fold(args.run_root, fold, args.device)


def score_main():
    args = parsed(parser_for("score-gate"))
    from .fi_contracts import stage_lock
    from .fi_reporting import score_gate

    with stage_lock(args.run_root, "report"):
        score_gate(args.run_root)


def report_main():
    args = parsed(parser_for("build-report"))
    from .fi_contracts import stage_lock, write_json
    from .fi_reporting import build_report

    with stage_lock(args.run_root, "report"):
        try:
            result = build_report(args.run_root)
        except (ValueError, OSError, KeyError) as error:
            # Even invalid configuration or damaged report artifacts should leave
            # an actionable diagnostic. This is never a sealed scientific result.
            path = args.run_root / "reports/pipeline-error.json"
            write_json(
                path,
                {
                    "stage": "report",
                    "error": str(error),
                    "error_type": type(error).__name__,
                    "measurement_complete": False,
                    "allow_full_experiment": False,
                    "allow_adapter_training": False,
                },
            )
            print(
                f"Cannot build experiment report: {error}. See {path}", file=sys.stderr
            )
            return 1
        print(json.dumps(result, allow_nan=False, indent=2))


def smoke_main():
    parser = parser_for("smoke")
    parser.add_argument("--protocol", choices=("direct-v2", "legacy-v1"), default="direct-v2")
    args = parsed(parser)
    configure_threads()
    from .fi_smoke import run_smoke

    print(json.dumps(run_smoke(args.run_root, protocol=args.protocol), allow_nan=False, indent=2))
