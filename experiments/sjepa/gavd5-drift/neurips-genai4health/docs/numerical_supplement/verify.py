"""Reproduce manuscript metrics using only Python's standard library.

Run python verify.py from this directory or provide its full path.
No network access, training data, third-party packages, or file writes are needed.
"""
from pathlib import Path
from collections import Counter
import csv
import hashlib
import json
import math

HERE = Path(__file__).resolve().parent
def read_csv(name):
    with (HERE / name).open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))

provenance = json.loads((HERE / "provenance.json").read_text(encoding="utf-8"))
for name, expected in provenance["input_csv_sha256"].items():
    assert hashlib.sha256((HERE/name).read_bytes()).hexdigest() == expected
rows = read_csv("test_source_predictions.csv")
labels = ["normal", "parkinsons", "stroke", "myopathic", "cerebralpalsy"]
results = []
source_sets = []
for lane in sorted({r["lane"] for r in rows}):
    data = [r for r in rows if r["lane"] == lane]
    assert len(data) == 20 and len({r["source"] for r in data}) == 20
    source_sets.append({(r["source"], r["annotation"]) for r in data})
    accuracy = sum(r["annotation"] == r["prediction"] for r in data)/len(data)
    recalls, f1s = [], []
    for label in labels:
        tp = sum(r["annotation"] == label and r["prediction"] == label for r in data)
        fp = sum(r["annotation"] != label and r["prediction"] == label for r in data)
        fn = sum(r["annotation"] == label and r["prediction"] != label for r in data)
        recalls.append(tp/(tp+fn) if tp+fn else 0)
        f1s.append(2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else 0)
    metrics = {"accuracy":accuracy, "balanced_accuracy":sum(recalls)/5, "macro_f1":sum(f1s)/5}
    expected = next(r for r in provenance["expected_metrics"] if r["lane"] == lane)
    for k,v in metrics.items(): assert math.isclose(v, expected["test_source_"+k], abs_tol=1e-12)
    results.append({"lane":lane, **metrics, "class_source_counts":dict(Counter(r["annotation"] for r in data))})
assert source_sets[0] == source_sets[1] == source_sets[2]
weights = read_csv("normal_validation_weighting.csv")
n = sum(int(r["sequences"]) for r in weights)
assert len(weights) == 5 and n == 64
clip = sum(int(r["sequences"])*float(r["mean_cosine"]) for r in weights)/n
source = sum(float(r["mean_cosine"]) for r in weights)/len(weights)
assert math.isclose(clip, .8890614510, abs_tol=1e-6)
assert math.isclose(source, .7010577321, abs_tol=1e-6)
gap = next(r["macro_f1"] for r in results if r["lane"]=="raw_kinematics") - next(r["macro_f1"] for r in results if r["lane"]=="sjepa_latent")
print(json.dumps({"all_checks":"PASS", "readouts":results,
                  "weighting":{"equal_clip":clip, "equal_source":source, "dominant_clip_share":60/64},
                  "raw_minus_latent_macro_f1":gap}, indent=2))
