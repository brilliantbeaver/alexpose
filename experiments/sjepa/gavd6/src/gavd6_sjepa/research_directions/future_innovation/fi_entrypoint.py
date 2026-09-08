"""Lazy command handlers for the Future Innovation Experiment 0 pipeline."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
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
    parser.add_argument("--pose-model", type=Path, required=True)
    parser.add_argument("--vjepa-root", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument(
        "--change-reason",
        required=True,
        help="Initial preregistration or reason for a new run version.",
    )
    args = parsed(parser)
    from .fi_contracts import VJEPA_COMMIT, initialize_run

    commit = subprocess.check_output(
        ["git", "-C", str(args.vjepa_root), "rev-parse", "HEAD"], text=True
    ).strip()
    if commit != VJEPA_COMMIT:
        parser.error(f"Teacher must be pinned to {VJEPA_COMMIT}")
    import torch

    if torch.__version__ != "2.6.0+cu124":
        parser.error(
            "HAIC scientific runs require the project's locked torch==2.6.0+cu124 runtime"
        )
    if not args.youtube_dir.joinpath("all").is_dir():
        parser.error("--youtube-dir must contain the full-source all/ cache")
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


def alignment_main():
    parser = parser_for("review-alignment")
    parser.add_argument("--reviewer", required=True)
    parser.add_argument(
        "--window-ids",
        nargs="+",
        required=True,
        help="Actually inspected windows covering all five folds.",
    )
    parser.add_argument("--note", required=True)
    args = parsed(parser)
    from .fi_cohort import review_alignment

    review_alignment(args.run_root, args.reviewer, args.window_ids, args.note)


def teacher_arguments(command):
    parser = parser_for(command)
    parser.add_argument("--device", choices=["cuda", "cpu"], default="cuda")
    return parsed(parser)


def record_teacher_runtime(root, stage):
    import torch

    from .fi_contracts import write_once_json

    attempt = os.environ.get("SLURM_JOB_ID", f"local-{os.getpid()}")
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
    from .fi_cohort import load_cohort, validate_alignment_review
    from .fi_contracts import stage_lock
    from .fi_feature_cache import cache_teacher
    from .fi_vjepa_adapter import FrozenVJEPAAdapter

    with stage_lock(args.run_root, "cache"):
        validate_alignment_review(args.run_root, load_cohort(args.run_root))
        record_teacher_runtime(args.run_root, "cache")
        cache_teacher(
            args.run_root, FrozenVJEPAAdapter.from_run(args.run_root, args.device)
        )


def audits_main():
    args = teacher_arguments("audit-teacher")
    configure_threads()
    from .fi_contracts import stage_lock
    from .fi_validity_audits import run_audits
    from .fi_vjepa_adapter import FrozenVJEPAAdapter

    with stage_lock(args.run_root, "audits"):
        record_teacher_runtime(args.run_root, "audit")
        passed = run_audits(
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


def capacity_main():
    parser = parser_for("assess-capacity")
    parser.add_argument("--reviewer", required=True)
    parser.add_argument("--finding", choices=["clear", "unresolved"], required=True)
    parser.add_argument(
        "--evidence",
        required=True,
        help="Interpret paired-controls.csv, its uncertainty, and per-seed results.",
    )
    args = parsed(parser)
    from .fi_reporting import assess_capacity

    assess_capacity(
        args.run_root,
        reviewer=args.reviewer,
        finding=args.finding,
        evidence=args.evidence,
    )


def report_main():
    args = parsed(parser_for("build-report"))
    from .fi_contracts import stage_lock
    from .fi_reporting import build_report

    with stage_lock(args.run_root, "report"):
        print(json.dumps(build_report(args.run_root), allow_nan=False, indent=2))


def smoke_main():
    args = parsed(parser_for("smoke"))
    configure_threads()
    from .fi_smoke import run_smoke

    print(json.dumps(run_smoke(args.run_root), allow_nan=False, indent=2))
