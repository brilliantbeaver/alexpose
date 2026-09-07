"""Read-only crosschecks of the current notebooks and portable numerical tables.

This is deliberately separate from verify_core_artifacts.py. The latter needs
the original fold checkpoint and registry, which may not accompany a checkout.
Here, saved notebook rows, annotation CSVs, the available QC ledger, and exported
predictions are checked without training, inference, downloads, or file writes.
"""
from __future__ import annotations

from collections import Counter
import csv
import hashlib
import io
import json
import math
from pathlib import Path
import re


ROOT = Path(__file__).resolve().parents[3]
DOCS = ROOT / "neurips-genai4health/docs"
ART = ROOT / "work/artifacts/real"
LABELS = ["normal", "parkinsons", "stroke", "myopathic", "cerebralpalsy"]


def rows(path):
    with path.open(newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def notebook(number):
    path, = sorted((ROOT / "neurips-brain-body").glob(f"{number:02d}_*.ipynb"))
    return json.loads(path.read_text(encoding="utf-8"))


def output_text(cell):
    parts = []
    for output in cell.get("outputs", []):
        if "text" in output:
            parts.append("".join(output["text"]))
        elif "text/plain" in output.get("data", {}):
            parts.append("".join(output["data"]["text/plain"]))
    return "\n".join(parts)


def scores(predictions):
    recalls, f1s = [], []
    for label in LABELS:
        tp = sum(a == p == label for a, p in predictions)
        fp = sum(a != label and p == label for a, p in predictions)
        fn = sum(a == label and p != label for a, p in predictions)
        recalls.append(tp / (tp + fn) if tp + fn else 0.0)
        f1s.append(2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.0)
    return {
        "correct_sources": sum(a == p for a, p in predictions),
        "sources": len(predictions),
        "accuracy": sum(a == p for a, p in predictions) / len(predictions),
        "balanced_accuracy": sum(recalls) / len(LABELS),
        "macro_f1": sum(f1s) / len(LABELS),
    }


def census(records, frame_key):
    return {
        "sequences": len(records),
        "sources": len({r["video_id"] for r in records}),
        "annotated_frame_rows": sum(int(float(r[frame_key])) for r in records),
    }


def main():
    nbs = {i: notebook(i) for i in range(7)}
    # Identical shared source is verified rather than presumed from notebook names.
    for index in (1, 4, 5):
        assert nbs[4]["cells"][index]["source"] == nbs[5]["cells"][index]["source"]
        assert nbs[4]["cells"][index]["source"] == nbs[6]["cells"][index]["source"]
    for number in (1, 2, 3):
        assert nbs[0]["cells"][1]["source"] == nbs[number]["cells"][1]["source"]
    assert nbs[0]["cells"][19]["source"] == nbs[3]["cells"][9]["source"]

    pattern = re.compile(
        r"^\s*\d+\s+(sjepa_latent|missingness_only|raw_kinematics)\s+"
        r"([A-Za-z0-9_-]{11})\s+(\w+)\s+(\w+)\s*$", re.MULTILINE
    )
    notebook_predictions = pattern.findall(output_text(nbs[6]["cells"][13]))
    assert len(notebook_predictions) == 60, "Saved notebook prediction rows changed"
    video_ids = sorted({r[1] for r in notebook_predictions})
    assert len(video_ids) == 20
    alias = {value: f"T{i + 1:02d}" for i, value in enumerate(video_ids)}
    notebook_aliased = {(lane, alias[video], actual, predicted)
                        for lane, video, actual, predicted in notebook_predictions}
    exported = rows(DOCS / "numerical_supplement/test_source_predictions.csv")
    assert len(exported) == 60
    assert notebook_aliased == {(r["lane"], r["source"], r["annotation"], r["prediction"])
                               for r in exported}, "Notebook/export prediction mismatch"
    evidence = {r["lane"]: r for r in rows(DOCS / "evidence/source_readout_metrics.csv")}
    readouts = {}
    for lane in sorted({r[0] for r in notebook_predictions}):
        values = [(actual, predicted) for name, _, actual, predicted in notebook_predictions
                  if name == lane]
        readouts[lane] = scores(values)
        for key in ("accuracy", "balanced_accuracy", "macro_f1"):
            assert math.isclose(readouts[lane][key], float(evidence[lane]["test_source_" + key]),
                                abs_tol=1e-12)
        readouts[lane]["class_source_counts"] = dict(Counter(a for a, _ in values))
        readouts[lane]["stroke_correct"] = sum(a == p == "stroke" for a, p in values)

    # This manifest is the saved public-metadata stage, not decoded eligibility.
    public = rows(ART / "manifest.csv")
    assert {r["eligibility_stage"] for r in public} == {"metadata_public"}
    stable_columns = ["condition", "video_id", "sequence_id", "first_frame",
                      "last_frame", "n_annotated_frames"]
    payload = io.StringIO(newline="")
    writer = csv.writer(payload, lineterminator="\n")
    writer.writerow(stable_columns)
    writer.writerows([[r[c] for c in stable_columns] for r in
                      sorted(public, key=lambda r: tuple(r[c] for c in stable_columns))])
    manifest_digest = hashlib.sha256(payload.getvalue().encode()).hexdigest()
    provenance = json.loads((DOCS / "numerical_supplement/provenance.json").read_text())
    assert manifest_digest == provenance["manifest_sha256"]

    qc = rows(ART / "pose_cache_inventory.csv")
    locked = [r for r in qc if r["in_locked_manifest"].lower() == "true"]
    eligible = [r for r in locked if r["pose_qc_eligible"].lower() == "true"]
    assert len({r["sequence_id"] for r in locked}) == len(locked)
    assert {r["manifest_sha256"] for r in locked} == {manifest_digest}
    assert {r["split_sha256"] for r in locked} == {provenance["split_sha256"]}
    assert {r["outer_fold"] for r in locked} <= {"0", "0.0"}
    for r in locked:
        expected = r["status"] == "ready" and float(r["neurologic_observed_fraction"]) >= 0.5
        assert expected == (r["pose_qc_eligible"].lower() == "true")
    source_roles = {}
    for r in locked:
        source_roles.setdefault(r["video_id"], set()).add(r["split_role"])
    assert all(len(values) == 1 for values in source_roles.values())
    assert set(video_ids) == {r["video_id"] for r in eligible if r["split_role"] == "test"}
    roles = {role: census([r for r in eligible if r["split_role"] == role], "frames")
             for role in ("train", "validation", "test")}

    raw = []
    for condition in LABELS:
        for path in sorted((ROOT / "data-gavd" / condition).glob("*.csv")):
            data = rows(path)
            assert data and {r["seq"] for r in data} == {path.stem}
            raw.append({"condition": condition, "sequence_id": path.stem,
                        "video_id": data[0]["id"],
                        "n_annotated_frames": len({r["frame_num"] for r in data})})
    cohort = {"raw_annotations": census(raw, "n_annotated_frames"),
              "metadata_public": census(public, "n_annotated_frames"),
              "decoded_locked_qc_rows": census(locked, "frames"),
              "pose_eligible": census(eligible, "frames")}
    assert [(value["sequences"], value["sources"]) for value in cohort.values()] == [
        (666, 103), (657, 100), (655, 98), (639, 97)]

    # Diagnose portability rather than silently replacing stored byte digests.
    digest_checks = {}
    historical_hashes = provenance.get("historical_raw_input_csv_sha256", provenance["input_csv_sha256"])
    for filename, expected in historical_hashes.items():
        data = (DOCS / "numerical_supplement" / filename).read_bytes()
        lf = data.replace(b"\r\n", b"\n")
        digest_checks[filename] = {
            "current_bytes_match_historical_digest": hashlib.sha256(data).hexdigest() == expected,
            "crlf_form_matches_historical_digest": hashlib.sha256(lf.replace(b"\n", b"\r\n")).hexdigest() == expected,
            "canonical_lf_matches_current_digest": hashlib.sha256(lf).hexdigest() == provenance["input_csv_sha256"][filename],
        }
    checkpoint = ART / "checkpoints/sjepa_outer_fold_0_seed_42_jepa_vicreg.pt"
    print(json.dumps({
        "status": "notebook rows and portable numerical values agree",
        "cohort": cohort, "pose_eligible_roles": roles, "readouts": readouts,
        "locked_legacy_geometry_rows": sum(r["resolution_safe_geometry"].lower() == "false" for r in locked),
        "original_fold_checkpoint_present": checkpoint.is_file(),
        "supplement_byte_digest_portability": digest_checks,
        "verification_limits": [
            "No training, pose extraction, model inference, or notebook execution was performed.",
            "The decoded count is reconstructed from the locked QC ledger, not fresh media decoding.",
            "Matching saved outputs and predictions does not reproduce training or certify clinical validity.",
            "Missing fold checkpoint/registry prevent renewed full checkpoint-lineage verification.",
        ],
    }, indent=2))


if __name__ == "__main__":
    main()
