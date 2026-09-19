#!/usr/bin/env python3
"""Read-only completion check for one managed HAIC source experiment."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src"))


def read(path):
    return json.loads(Path(path).read_text())


def require(condition, message):
    if not condition:
        raise ValueError(message)


def check_results(work):
    """Verify scheduler completion and retained evidence without fitting or writing."""
    import numpy as np
    import pandas as pd
    from haic import settled, state_for
    from gavd6_sjepa.research_directions.synthetic_training_v2.config import RunConfig, STAGES
    from gavd6_sjepa.research_directions.synthetic_training_v2.contracts import (
        TrackBundle, array_digest, code_identity, digest, sha256_file,
    )
    from gavd6_sjepa.research_directions.synthetic_training_v2.evaluation import KEYS
    from gavd6_sjepa.research_directions.synthetic_training_v2.workflow import (
        _receipt, _record_groups, reconstruct_metrics,
    )

    work = Path(work).expanduser().resolve()
    state = state_for(work)
    require(state.get("source_config"), "Source training has not been configured; finish preparation and submit source first.")
    config_path = Path(state["source_config"]).resolve()
    cfg = RunConfig.load(config_path)
    require(cfg.mode == "source", "The configured experiment is not a source run.")
    require(cfg.root.is_relative_to(work), "The source results directory is outside this managed run.")
    snapshot = settled(state)
    preparations = [job for job in state["jobs"] if job["phase"] == "prepare"]
    require(preparations, "No managed preparation job is recorded.")
    latest = {"prepare": preparations[-1]}
    for stage in STAGES:
        attempts = [job for job in state["jobs"] if job["phase"] == "source" and job["stage"] == stage]
        require(attempts, f"No managed job is recorded for source stage {stage}.")
        latest[stage] = attempts[-1]
        require(Path(attempts[-1]["scope"]).resolve() == config_path,
                f"Latest {stage} attempt belongs to a different source configuration.")
    for stage, job in latest.items():
        row = snapshot[job["job_id"]]
        require(row["state"] == "COMPLETED" and row["exit_code"] == "0:0",
                f"Latest {stage} job {job['job_id']} is {row['state']} ({row['exit_code']}); inspect its log.")

    prepared = Path(latest["prepare"]["output"])
    require(Path(cfg.bundle).resolve() == (prepared / "bundle").resolve(),
            "Source configuration does not use the latest prepared bundle.")
    require(read(prepared / "preparation-status.json")["status"] == "source_prepared",
            "Latest preparation did not produce a successful source bundle.")

    # Reconstruct the same immutable identity as workflow.initialize without
    # invoking its initialization/writing branch.
    expected_identity = dict(configuration=cfg.as_dict(), code=code_identity(ROOT),
                             protocol=sha256_file(ROOT / "docs/studies/synthetic-training-v2/protocol.md"),
                             source_manifest=sha256_file(Path(cfg.bundle) / "manifest.json"))
    if cfg.decision_spec:
        from gavd6_sjepa.research_directions.synthetic_training_v2.decisions import load_decision_spec
        spec = load_decision_spec(cfg.decision_spec)
        expected_identity["decision_spec"] = sha256_file(cfg.decision_spec)
        expected_identity["calibration_artifact"] = spec["calibration_artifact_sha256"]
    if cfg.cost_ledger:
        expected_identity["prior_cost_ledger"] = sha256_file(cfg.cost_ledger)
    require(read(cfg.root / "identity.json")["signature"] == digest(expected_identity),
            "Run identity changed: retain the original code, protocol, configuration and source bundle.")
    for stage in STAGES:
        _receipt(cfg, stage)

    source = TrackBundle.load(cfg.bundle)
    bundle = TrackBundle.load(cfg.root / "data/bundle")
    evidence_status = source.evidence_status
    require(evidence_status in {"source-run", "automated-source-screen"},
            "Prepared bundle is neither audited source evidence nor an explicit automated development screen.")
    for name, value in (("prepared", source), ("training", bundle)):
        require(value.evidence_status == evidence_status, f"{name} bundle evidence status differs from preparation.")
        value.validate(cfg.held_extractor)
    require(source.records == bundle.records and source.provenance == bundle.provenance
            and array_digest(source.inputs) == array_digest(bundle.inputs)
            and array_digest(source.targets) == array_digest(bundle.targets),
            "The retained training bundle differs from the prepared source bundle.")
    dev = bundle.subset("development")
    require(dev.records, "No development records are retained.")
    methods = {"unchanged", "filter0", "filter1", "filter2", *cfg.arms}
    expected_names = {f"{method}-{seed}" for method in methods for seed in cfg.seeds}
    for suffix in (".npz", ".json"):
        actual = {path.stem for path in (cfg.root / "predictions").glob(f"*{suffix}")}
        require(actual == expected_names, f"Prediction {suffix} files differ from the configured methods/seeds.")
    for method in methods:
        for seed in cfg.seeds:
            meta = read(cfg.root / "predictions" / f"{method}-{seed}.json")
            require(meta["method"] == method and meta["seed"] == seed and meta["evidence_status"] == evidence_status
                    and meta["records"] == _record_groups(dev.records, seed),
                    f"Prediction metadata differs from the development panel: {method}-{seed}.")
    for arm in cfg.arms:
        for seed in cfg.seeds:
            folder = cfg.root / "fits" / f"{arm}-{seed}"
            fit = read(folder / "training.json")
            planned = cfg.readout_updates if arm == "initialized" else cfg.updates + cfg.readout_updates
            require(fit["status"] == "complete" and fit["arm"] == arm and fit["seed"] == seed
                    and fit["planned_updates"] == planned and (folder / "model.pt").is_file(),
                    f"Incomplete or mismatched fit: {arm}-{seed}.")
            if cfg.resource_contrast == "matched_data_steps":
                require(fit["optimizer_updates"] == planned, f"Configured updates were not completed: {arm}-{seed}.")

    expected_rows = len(dev.records) * len(methods) * len(cfg.seeds)
    metrics = reconstruct_metrics(cfg.root)
    require(len(metrics) == expected_rows and set(metrics.method) == methods
            and set(metrics.seed) == set(cfg.seeds) and set(metrics.evidence_status) == {evidence_status},
            "Reconstructed prediction metrics do not cover the complete configured comparison.")
    saved = pd.read_csv(cfg.root / "evaluation/per-window.csv")
    require(set(saved.columns) == set(metrics.columns), "Saved per-window metric columns differ from reconstruction.")
    keys = ["method", *KEYS]
    pd.testing.assert_frame_equal(
        saved[metrics.columns].sort_values(keys).reset_index(drop=True),
        metrics.sort_values(keys).reset_index(drop=True),
        check_dtype=False, check_exact=False, rtol=1e-9, atol=1e-12,
    )
    required = ("report.md", "data/achieved-size.json", "development-snapshot.json",
                "evaluation/per-person-balanced-summary.csv", "evaluation/per-person.csv",
                "evaluation/nuisance-strata.csv", "evaluation/training-seed-variability.csv",
                "evaluation/contrasts.json", "evaluation/gates.json", "evaluation/accuracy-preservation.png")
    for name in required:
        path = cfg.root / name
        require(path.is_file() and path.stat().st_size > 0, f"Required result is missing or empty: {path}")
    gates = read(cfg.root / "evaluation/gates.json")
    require(set(gates) == {"A", "B", "real_transfer", "personalization", "video"},
            "Scientific gate records are incomplete.")
    snapshot_record = read(cfg.root / "development-snapshot.json")
    require(snapshot_record["confirmation_opened"] is False,
            "This source checker does not certify confirmation experiments.")
    finite = metrics["visible_nle"].to_numpy(float)
    if evidence_status == "automated-source-screen":
        require(gates["B"]["status"] == "insufficient_evidence",
                "An automated development screen must not authorize the scientific gate.")
    return dict(status="SOURCE_RESULTS_COMPLETE", evidence_status=evidence_status, results=str(cfg.root),
                development_people=len({row["canonical_person_id"] for row in dev.records}),
                development_track_records=len(dev.records), methods=sorted(methods), seeds=list(cfg.seeds),
                per_window_metric_rows=len(metrics), finite_visible_coordinate_rows=int(np.isfinite(finite).sum()),
                scheduler_and_receipts="All latest stages completed with 0:0; all retained receipt hashes verified",
                metric_reconstruction="Saved per-window values match retained predictions",
                gates=gates, scope="Synthetic development screening; scientific gate success and real transfer are separate.")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--work", type=Path, default=os.environ.get("STV2_WORK"),
                        help="Managed run directory; defaults to the sourced STV2_WORK")
    args = parser.parse_args(argv)
    if args.work is None:
        parser.error("Source the run's session.env, or supply --work.")
    try:
        result = check_results(args.work)
    except (OSError, ValueError, RuntimeError, KeyError, AssertionError, subprocess.SubprocessError) as error:
        print(f"STV2_RESULTS_NOT_COMPLETE: {error}", file=sys.stderr)
        return 1
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
