"""Refresh provenance from retained numerical records without changing CSV values.

The default works without the original checkpoint tree. It compares 60 source
predictions with notebook 06 and five weighting rows with the evidence table.
Use --require-full-check to additionally require the original artifact verifier.
Neither mode trains, extracts poses, or runs model inference.
"""
from __future__ import annotations

import argparse
import csv
from datetime import date
import hashlib
import json
import math
from pathlib import Path
import re
import subprocess
import sys

DOCS = Path(__file__).resolve().parents[1]
ROOT = DOCS.parents[1]
OUT = DOCS / "numerical_supplement"
FILES = ["test_source_predictions.csv", "normal_validation_weighting.csv"]


def read_rows(path):
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def require(condition, message):
    if not condition:
        raise ValueError(message)


def canonical_digest(path):
    return hashlib.sha256(path.read_bytes().replace(b"\r\n", b"\n")).hexdigest()


def check_retained_rows():
    predictions = read_rows(OUT / FILES[0])
    notebook_path = ROOT / "neurips-brain-body/06_capstone_health_condition_classifiers.ipynb"
    require(notebook_path.is_file(), "Notebook 06 is required for the saved-output crosscheck")
    nb = json.loads(notebook_path.read_text(encoding="utf-8"))
    output_parts = []
    for cell in nb["cells"]:
        for output in cell.get("outputs", []):
            if "text" in output:
                output_parts.append("".join(output["text"]))
            elif "text/plain" in output.get("data", {}):
                output_parts.append("".join(output["data"]["text/plain"]))
    pattern = re.compile(
        r"^\s*\d+\s+(sjepa_latent|missingness_only|raw_kinematics)\s+"
        r"([A-Za-z0-9_-]{11})\s+(\w+)\s+(\w+)\s*$", re.MULTILINE
    )
    saved = pattern.findall("\n".join(output_parts))
    require(len(saved) == 60, "Notebook 06 must expose all 60 saved prediction rows")
    aliases = {video: f"T{i + 1:02d}" for i, video in
               enumerate(sorted({row[1] for row in saved}))}
    expected = {(lane, aliases[video], actual, predicted)
                for lane, video, actual, predicted in saved}
    actual = {(row["lane"], row["source"], row["annotation"], row["prediction"])
              for row in predictions}
    require(len(predictions) == 60 and actual == expected,
            "Retained predictions disagree with notebook 06; records were not changed")
    weighting = read_rows(OUT / FILES[1])
    recorded = read_rows(DOCS / "evidence/validation_normal_source_weighting.csv")
    require(len(weighting) == len(recorded) == 5, "Expected five normal-validation sources")
    reference = {row["source"]: row for row in recorded}
    require(len({row["source"] for row in weighting}) == len(reference) == 5,
            "Weighting rows must have distinct source aliases")
    for row in weighting:
        require(row["source"] in reference, "Unknown weighting source alias")
        expected_row = reference[row["source"]]
        require(int(row["sequences"]) == int(expected_row["sequences"]),
                "Weighting clip count changed")
        require(math.isclose(float(row["mean_cosine"]), float(expected_row["mean_cosine"]),
                             rel_tol=0, abs_tol=1e-12), "Weighting source mean changed")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--require-full-check", action="store_true",
                        help="Also require original protocol/checkpoint verification; fails if absent")
    args = parser.parse_args()
    provenance_path = OUT / "provenance.json"
    provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
    require(all((OUT / filename).is_file() for filename in FILES),
            "Retained numerical CSVs are missing; no fallback values will be invented")
    if args.require_full_check:
        original_inputs = [
            ROOT / "work/artifacts/real/evaluation_protocol/eligible_sequence_manifest.csv",
            ROOT / "work/artifacts/real/evaluation_protocol/source_split_registry.csv",
            ROOT / "work/artifacts/real/checkpoints/sjepa_outer_fold_0_seed_42_jepa_vicreg.pt",
        ]
        missing = [str(path) for path in original_inputs if not path.is_file()]
        if missing:
            raise SystemExit(
                "Full checkpoint verification is unavailable; required inputs are absent:\n"
                + "\n".join(missing)
                + "\nThe default command verifies retained numerical records only."
            )
        subprocess.run([sys.executable, str(DOCS / "reproducibility/verify_core_artifacts.py")], check=True)
    check_retained_rows()
    historical = provenance.get("historical_raw_input_csv_sha256", provenance["input_csv_sha256"])
    provenance["historical_raw_input_csv_sha256"] = historical
    provenance["historical_input_hash_note"] = (
        "Original September 5 hashes used CRLF line endings. The retained LF files reproduce "
        "those historical hashes when converted to CRLF in memory; numerical values are unchanged."
    )
    provenance["input_csv_hash_convention"] = (
        "SHA-256 of UTF-8 file bytes with CRLF replaced by LF; no other normalization"
    )
    provenance["input_csv_sha256"] = {filename: canonical_digest(OUT / filename) for filename in FILES}
    provenance["numerical_verification"] = {
        "date": date.today().isoformat(),
        "mode": "full_artifacts_and_retained_outputs" if args.require_full_check else "retained_outputs",
        "notebook_prediction_rows_compared": 60,
        "weighting_rows_compared": 5,
        "original_fold_checkpoint_reverified": bool(args.require_full_check),
        "scope": "Saved-output consistency and numerical reconstruction; no retraining or inference.",
    }
    provenance["scope"] = (
        "Reproduces reported scores and weighting from retained numerical records. Model, objective, "
        "manifest, split, and checkpoint metadata describe the original recorded run; default refresh "
        "does not reverify its checkpoint. Source aliases remove direct video IDs without establishing "
        "person anonymity. No training, clinical validity, or patient independence is certified."
    )
    provenance_path.write_text(json.dumps(provenance, indent=2) + "\n", encoding="utf-8")
    subprocess.run([sys.executable, str(OUT / "verify.py")], check=True)
    print("Verified 60 saved notebook predictions and five weighting rows; CSV values were preserved.")


if __name__ == "__main__":
    main()
