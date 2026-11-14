#!/usr/bin/env python3
"""Recompute the supplied held-out metrics using only the Python standard library."""
import csv
import hashlib
import json
import math
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent
LABELS = ["normal", "ms", "pd"]
SYSTEMS = ["sjepa", "rf", "mean_pose", "visibility", "majority"]


def read_json(name):
    return json.loads((HERE / name).read_text())


def metric(rows, column, source_weighted):
    """Weight clips before pooling confusion counts and averaging class F1."""
    counts = Counter(row["source_id"] for row in rows)
    matrix = [[0.0 for _ in LABELS] for _ in LABELS]
    for row in rows:
        truth, prediction = LABELS.index(row["label"]), LABELS.index(row[column])
        weight = 1.0 / counts[row["source_id"]] if source_weighted else 1.0
        matrix[truth][prediction] += weight
    f1 = []
    for k in range(len(LABELS)):
        tp = matrix[k][k]
        denominator = sum(matrix[k]) + sum(row[k] for row in matrix)
        f1.append(2.0 * tp / denominator if denominator else 0.0)
    total = sum(sum(row) for row in matrix)
    return {
        "macro_f1": sum(f1) / len(f1),
        "accuracy": sum(matrix[k][k] for k in range(len(LABELS))) / total,
        "confusion": matrix,
    }


def compare(actual, expected, name):
    if isinstance(expected, dict):
        assert actual.keys() == expected.keys(), name
        for key in expected:
            compare(actual[key], expected[key], name + "." + key)
    elif isinstance(expected, list):
        assert len(actual) == len(expected), name
        for i, (value, target) in enumerate(zip(actual, expected)):
            compare(value, target, name + "[" + str(i) + "]")
    else:
        assert math.isclose(actual, expected, rel_tol=1e-12, abs_tol=1e-12), (
            name, actual, expected)


def summarize(rows, column):
    return {
        "clip_weighted": metric(rows, column, False),
        "source_weighted": metric(rows, column, True),
        "folds_source_weighted": [
            metric([row for row in rows if row["fold"] == fold], column, True)
            for fold in range(5)
        ],
    }


def main():
    manifest = read_json("manifest.json")
    for filename, expected_hash in manifest["sha256"].items():
        actual_hash = hashlib.sha256((HERE / filename).read_bytes()).hexdigest()
        assert actual_hash == expected_hash, "Changed packet file: " + filename

    with (HERE / "heldout_predictions.csv").open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    for row in rows:
        row["fold"] = int(row["fold"])
    expected = read_json("expected_metrics.json")
    splits = read_json("splits.json")["folds"]
    assert expected["label_order"] == LABELS
    assert len(rows) == 88
    by_clip = {row["clip_id"]: row for row in rows}
    assert len(by_clip) == 88, "Repeated held-out clip"
    assert set(by_clip) == {"C" + str(i).zfill(3) for i in range(1, 89)}
    sources = {row["source_id"] for row in rows}
    assert sources == {"S" + str(i).zfill(3) for i in range(1, 42)}
    source_labels = {}
    for row in rows:
        assert row["label"] in LABELS and row["fold"] in range(5)
        for name in SYSTEMS + ["rf_earlier"]:
            assert row["pred_" + name] in LABELS
        old_label = source_labels.setdefault(row["source_id"], row["label"])
        assert old_label == row["label"], "A source has inconsistent labels"
    assert dict(Counter(row["label"] for row in rows)) == expected["clip_counts"]
    assert dict(Counter(source_labels.values())) == expected["source_counts"]

    assert [fold["fold"] for fold in splits] == list(range(5))
    all_test_clips, all_test_sources = [], []
    for fold in splits:
        parts = [fold[name] for name in ["train", "validation", "test"]]
        for key, universe in [("sources", sources), ("clips", set(by_clip))]:
            groups = [set(part[key]) for part in parts]
            assert all(len(part[key]) == len(group) for part, group in zip(parts, groups))
            assert groups[0].isdisjoint(groups[1])
            assert groups[0].isdisjoint(groups[2])
            assert groups[1].isdisjoint(groups[2])
            assert set.union(*groups) == universe
        assert len(fold["train"]["sources"]) == 24
        for part in parts:
            assert {by_clip[clip]["source_id"] for clip in part["clips"]} == set(part["sources"])
            assert {clip for clip, row in by_clip.items() if row["source_id"] in part["sources"]} == set(part["clips"])
        test = fold["test"]
        assert {row["clip_id"] for row in rows if row["fold"] == fold["fold"]} == set(test["clips"])
        all_test_clips.extend(test["clips"])
        all_test_sources.extend(test["sources"])
    assert Counter(all_test_clips) == Counter({clip: 1 for clip in by_clip})
    assert Counter(all_test_sources) == Counter({source: 1 for source in sources})

    computed = {}
    for system in SYSTEMS:
        result = summarize(rows, "pred_" + system)
        compare(result, expected["primary"][system], "primary." + system)
        computed[system] = result
    earlier_rf = summarize(rows, "pred_rf_earlier")
    compare(earlier_rf, expected["earlier_rf"], "earlier_rf")
    assert sum(row["pred_rf"] != row["pred_rf_earlier"] for row in rows) == 12

    print("PASS: packet file hashes; 88 unique clips; 41 source groups; class counts.")
    print("PASS: source-disjoint splits; each clip predicted once; each source tested in one fold.")
    print("PASS: all pooled and per-fold macro-F1, accuracies, and confusion matrices.")
    print("\nPrimary results (macro-F1):")
    print("method          source-weighted     clip-weighted")
    for system in SYSTEMS:
        result = computed[system]
        print("%-15s %.12f      %.12f" % (
            system, result["source_weighted"]["macro_f1"], result["clip_weighted"]["macro_f1"]))
    print("earlier RF      %.12f      %.12f" % (
        earlier_rf["source_weighted"]["macro_f1"], earlier_rf["clip_weighted"]["macro_f1"]))
    difference = computed["sjepa"]["source_weighted"]["macro_f1"] - computed["rf"]["source_weighted"]["macro_f1"]
    print("\nPrimary S-JEPA minus RF, source-weighted macro-F1: %.12f" % difference)
    print("Historical-to-primary RF prediction changes: 12 of 88.")


if __name__ == "__main__":
    main()
