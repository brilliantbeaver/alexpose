"""Recalculate retained notebook/export evidence without requiring trained models.

The original checkpoint-backed workflow is preserved in
build_evidence_from_training_artifacts.py. Its earlier successful review is
historical; this command never upgrades retained-output verification to renewed
training or checkpoint verification.
"""
from __future__ import annotations

import csv
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess
import sys


DOCS = Path(__file__).resolve().parents[1]
ROOT = DOCS.parents[1]


def run_json(relative):
    command = [sys.executable, str(DOCS / relative)]
    result = subprocess.run(command, cwd=ROOT, text=True, capture_output=True, check=True)
    return json.loads(result.stdout)


def read_csv(path):
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def main():
    numbers = run_json("numerical_supplement/verify.py")
    assert numbers["numerical_checks"] == "PASS"
    core = run_json("review/verify_current_core_exports.py")
    recorded = read_csv(DOCS / "evidence/cohort_by_annotation.csv")
    for prefix, stage in (("raw", "raw_annotations"), ("public", "metadata_public"),
                          ("decoded", "decoded_locked_qc_rows"), ("qc", "pose_eligible")):
        assert sum(int(row[prefix + "_sequences"]) for row in recorded) == core["cohort"][stage]["sequences"]
        assert sum(int(row[prefix + "_sources"]) for row in recorded) == core["cohort"][stage]["sources"]
    roles = read_csv(DOCS / "evidence/fold0_role_counts.csv")
    for row in roles:
        current = core["pose_eligible_roles"][row["split_role"]]
        assert int(row["sequences"]) == current["sequences"]
        assert int(row["sources"]) == current["sources"]
    expected_counts = {"raw_kinematics": 10, "sjepa_latent": 6, "missingness_only": 6}
    for result in numbers["readouts"]:
        assert result["correct_sources"] == expected_counts[result["lane"]]
        assert core["readouts"][result["lane"]]["stroke_correct"] == 0

    paths = [
        DOCS / "numerical_supplement/test_source_predictions.csv",
        DOCS / "numerical_supplement/normal_validation_weighting.csv",
        DOCS / "numerical_supplement/provenance.json",
        DOCS / "review/verify_current_core_exports.py",
        ROOT / "work/artifacts/real/manifest.csv",
        ROOT / "work/artifacts/real/pose_cache_inventory.csv",
    ] + sorted((ROOT / "neurips-brain-body").glob("*.ipynb"))
    report = {
        "reviewed_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "scope": "Reanalysis of retained predictions, similarity summaries, notebook outputs, and cohort ledgers.",
        "numerical_verification": numbers,
        "notebook_and_cohort_verification": core,
        "current_checks": {
            "all_60_notebook_predictions_match_export": True,
            "classification_metrics_recomputed": True,
            "cohort_counts_match_recorded_tables": True,
            "role_counts_match_recorded_tables": True,
            "weighting_arithmetic_recomputed": True,
            "original_checkpoint_lineage_reverified": False,
            "pose_extraction_repeated": False,
            "training_or_model_inference_repeated": False,
        },
        "prior_artifact_review": {
            "record": "evidence/verification_manifest.json",
            "status": "Historical review; original checkpoint/cache checks were not rerun by this command.",
        },
        "inputs": [{"path": str(path.relative_to(ROOT)),
                    "sha256": hashlib.sha256(path.read_bytes()).hexdigest()} for path in paths],
        "limitations": core["verification_limits"],
    }
    output = DOCS / "evidence/current_verification.json"
    output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"current_checks": report["current_checks"],
                      "results": numbers["readouts"], "weighting": numbers["weighting"]}, indent=2))


if __name__ == "__main__":
    main()
