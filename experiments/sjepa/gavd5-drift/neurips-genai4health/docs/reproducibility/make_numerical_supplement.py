"""Export aliased numerical inputs; no videos, trajectories, or private paths."""
from pathlib import Path
import csv
import hashlib
import json

DOCS = Path(__file__).resolve().parents[1]
ART = DOCS.parents[1] / "work/artifacts/real"
OUT = DOCS / "numerical_supplement"
OUT.mkdir(exist_ok=True)
prefix = "readout_outer_fold_0_seed_42_jepa_vicreg"
with (ART / f"{prefix}_source_predictions.csv").open(newline="", encoding="utf-8") as f:
    rows = list(csv.DictReader(f))
aliases = {v:f"T{i+1:02d}" for i,v in enumerate(sorted({r["video_id"] for r in rows}))}
with (OUT / "test_source_predictions.csv").open("w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=["lane", "source", "annotation", "prediction"])
    writer.writeheader()
    for r in rows:
        writer.writerow({"lane":r["lane"], "source":aliases[r["video_id"]],
                         "annotation":r["condition"], "prediction":r["prediction"]})
with (DOCS / "evidence/validation_normal_source_weighting.csv").open(newline="", encoding="utf-8") as f:
    weighting = list(csv.DictReader(f))
with (OUT / "normal_validation_weighting.csv").open("w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=["source", "sequences", "mean_cosine"])
    writer.writeheader()
    writer.writerows(weighting)
contract = json.loads((ART / f"{prefix}_contract.json").read_text(encoding="utf-8"))
sidecar = json.loads((ART / "checkpoints/sjepa_outer_fold_0_seed_42_jepa_vicreg.json").read_text(encoding="utf-8"))
provenance = {
    "audit_date":"2026-09-05", "fold":0, "seed":42,
    "manifest_sha256":contract["manifest_sha256"], "split_sha256":contract["split_sha256"],
    "checkpoint_sha256":contract["checkpoint_sha256"],
    "stage_checkpoints":[{"stage":s["stage"], "sha256":s["sha256"]} for s in sidecar["stage_checkpoints"]],
    "objective":sidecar["objective"], "model_config":sidecar["model_config"],
    "test_source_count":20, "normal_validation_source_count":5,
    "normal_validation_sequence_count":64,
    "expected_metrics":[{k:v for k,v in row.items() if k == "lane" or k.startswith("test_source_")} for row in contract["test_metrics"]],
    "input_csv_sha256":{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in OUT.glob("*.csv")},
    "scope":"Reproduces main reported metrics and weighting, not model training. Source aliases remove direct video IDs from these tables but do not establish person anonymity."
}
(OUT / "provenance.json").write_text(json.dumps(provenance, indent=2) + "\n", encoding="utf-8")
print("Exported 60 source predictions, five validation-source summaries, and hash/config record.")
