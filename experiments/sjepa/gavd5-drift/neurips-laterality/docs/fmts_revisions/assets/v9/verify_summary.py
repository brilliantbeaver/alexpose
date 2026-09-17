"""Verify retained seed means and paired differences; never reconstruct source CIs.

Usage: python verify_summary.py [numerical_evidence.json]
Only Python's standard library is required.
"""
import json
import math
import statistics
import sys
from pathlib import Path


def verify(path):
    evidence = json.loads(Path(path).read_text())
    rows = evidence["readout_rows"]
    initial = next(r for r in rows if r["representation"] == "initial_online__mean_motion")
    results = []
    for row in rows:
        assert len(row["per_seed_r2"]) == len(evidence["cohort"]["seeds"])
        mean = statistics.mean(row["per_seed_r2"])
        assert math.isclose(mean, row["mean_r2"], abs_tol=1e-12)
        if row["representation"] != "pretrained_teacher__mean_motion":
            continue
        key = row["experiment"] + "/" + row["condition"]
        retained = evidence["trained_minus_initial"][key]
        differences = [a-b for a, b in zip(row["per_seed_r2"], initial["per_seed_r2"])]
        delta = statistics.mean(differences)
        assert all(d < 0 for d in differences)
        assert math.isclose(delta, retained["difference_r2"], abs_tol=1e-12)
        assert math.isclose(row["mean_mae"]-initial["mean_mae"], retained["difference_mae"], abs_tol=1e-12)
        results.append({"condition": key, "mean_r2": mean, "paired_difference_r2": delta})
    for interval in evidence["mask_intervals"]:
        arm, reference = interval["subtraction"].split(" minus ")
        selected = {r["condition"]: r for r in rows if r["experiment"] == interval["experiment"] and r["representation"] == "pretrained_teacher__mean_motion"}
        delta = statistics.mean(a-b for a,b in zip(selected[arm]["per_seed_r2"], selected[reference]["per_seed_r2"]))
        assert math.isclose(delta, float(interval["difference"]), abs_tol=1e-12)
    diagnostic = evidence["correspondence"]
    assert diagnostic["trained_rows"] == 5*5*5*3 == 375
    assert diagnostic["initial_rows"] == 5*5*3 == 75
    return {"aggregate_arithmetic": "passed", "paired_results": results,
            "source_bootstrap": "not rerun; raw source predictions unavailable",
            "diagnostic_rows": "denominators checked; individual model checks not rerun"}


if __name__ == "__main__":
    print(json.dumps(verify(sys.argv[1] if len(sys.argv)>1 else "numerical_evidence.json"), indent=2))
