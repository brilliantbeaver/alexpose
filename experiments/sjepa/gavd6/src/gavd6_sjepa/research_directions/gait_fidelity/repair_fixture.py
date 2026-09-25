"""Execute the complete repair workflow on analytic CPU software fixtures.

This creates real tiny core and response fits before admitting their repair
dependency. The result validates workflow execution, never human-subject
evidence, effect magnitude, statistical power, GPU behavior, or acceptance.
"""
from __future__ import annotations

from pathlib import Path
import time

import torch

from .common import atomic_json, read_json, sha256, utc_now


def _core(work):
    from .config import initialize, load_config
    from .scheduler import run, verify_completed
    existed = (work/"config.json").exists()
    cfg = initialize(work, fixture=True, experiment_set="core")
    if not existed:
        # These are declared before the ordinary freeze and inherited by every
        # downstream arm; no fitted checkpoint is relabelled or fabricated.
        cfg["data"]["samples"] = 16
        cfg["model"].update(width=8, window_size=16)
        cfg["measurement"]["min_frames"] = 8
        cfg["training"].update(readout_updates=2)
        cfg["evaluation"]["bootstrap_draws"] = 20
        atomic_json(work/"config.json", cfg)
        cfg = load_config(work)
    state = read_json(work/"ledger.json") if (work/"ledger.json").exists() else {}
    if "evaluation" not in state.get("completed", {}):
        run(cfg, local=True, max_jobs=4)
    else:
        verify_completed(state["completed"]["evaluation"])
    summary = read_json(work/"evaluation/summary.json")
    if summary["status"] != "SOFTWARE_FIXTURE_COMPLETE":
        raise RuntimeError("Repair fixture core dependency is not a completed software fixture")
    return cfg, summary


def run_fixture(work):
    """Run actual three-seed dependencies, twelve repair fits and locked evaluation."""
    from .followup import initialize_followup, verify_followup
    from .scheduler import run as run_ordinary, verify_completed
    from .repair import initialize_repair, load_repair, fit_manifest, status, verify
    from .repair_cohort import lock_slim_confirmation, verify_slim_lock
    from .repair_execution import run

    work = Path(work).expanduser().resolve()
    if (work/"config.json").exists():
        raise ValueError("Fixture root must contain separate core, response and repair directories, not an existing study")
    work.mkdir(parents=True, exist_ok=True)
    marker = work/"fixture-complete.json"
    if marker.exists():
        retained = read_json(marker)
        cfg = load_repair(work/"repair", verify_dependencies=True)
        verify(cfg); verify_slim_lock(cfg)
        for path, expected in retained["summary_hashes"].items():
            if sha256(path) != expected:
                raise RuntimeError("Retained fixture summary changed")
        return retained
    started = time.monotonic()
    previous_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        _, core_summary = _core(work/"core")
        response_cfg = initialize_followup(work/"response", parent_work=work/"core", include_base_readouts=True)
        response_state = read_json(work/"response/ledger.json") if (work/"response/ledger.json").exists() else {}
        if "evaluation" not in response_state.get("completed", {}):
            run_ordinary(response_cfg, local=True, max_jobs=4)
        else:
            verify_completed(response_state["completed"]["evaluation"])
        response_verification = verify_followup(response_cfg)
        response_summary = read_json(work/"response/evaluation/response-summary.json")
        cfg = initialize_repair(work/"repair", response_work=work/"response", max_jobs=4)
        run(cfg, stage="development", local=True)
        run(cfg, stage="benchmark", local=True)
        lock_path = Path(cfg["repair"]["confirmation_lock"])
        if lock_path.exists():
            verify_slim_lock(cfg)
        else:
            lock_slim_confirmation(cfg, fit_manifest(cfg), exposure_ledger=None,
                reviewed_by="analytic_software_fixture_generator",
                evidence="Fresh programmatically generated identities, explicitly not evidence of unseen human participants",
                output=lock_path)
        run(cfg, stage="confirmation", local=True)
        verification = verify(cfg)
        state = status(cfg)
        summaries = {split: read_json(work/f"repair/{split}/evaluation/summary.json")
                     for split in ("development", "confirmation")}
        fits = fit_manifest(cfg)
        if (state["completed_readouts"] != 12 or state["completed_calibrations"] != 6 or len(fits) != 27
                or state["failed"] or state["active"] or not state["confirmation_complete"]):
            raise RuntimeError("Repair fixture did not finish its complete declared matrix")
        for split, summary in summaries.items():
            if (summary["fits"] != 27 or summary["methods"] != 9 or summary["seeds"] != [17, 29, 43]
                    or not summary["fixture"] or summary["independent_confirmation"]
                    or summary["clinical_validation"] or summary["evaluation_split"] != split):
                raise RuntimeError("Fixture evaluation lost its declared matrix or evidence boundary")
        if not summaries["confirmation"]["complete_planned_population"]:
            raise RuntimeError("Fixture confirmation lost a generated person or source window")
        files = [work/"core/evaluation/summary.json", work/"response/evaluation/response-summary.json",
                 work/"repair/development/evaluation/summary.json", work/"repair/confirmation/evaluation/summary.json"]
        receipt = dict(schema="gf-repair-software-fixture-v1", status="REPAIR_SOFTWARE_FIXTURE_COMPLETE",
            fixture=True, scientific_results=False, independent_confirmation=False, clinical_validation=False,
            evidence_boundary="Analytic software fixture only; no result about people, GPU throughput or statistical power",
            work=str(work), completed_utc=utc_now(), elapsed_seconds=time.monotonic()-started,
            core_fits=core_summary["trained_models"], response_fits=response_summary["trained_models"],
            new_readouts=12, calibrations=6, evaluation_methods=9, evaluation_fits=27, seeds=[17, 29, 43],
            development_people=summaries["development"]["evaluated_people"],
            generated_confirmation_people=summaries["confirmation"]["evaluated_people"],
            generated_confirmation_windows=summaries["confirmation"]["evaluated_windows"],
            response_verification=response_verification, verification=verification,
            summary_hashes={str(path): sha256(path) for path in files},
            reports={split: str(work/f"repair/{split}/evaluation/report.md") for split in summaries})
        atomic_json(marker, receipt)
        return receipt
    finally:
        torch.set_num_threads(previous_threads)
