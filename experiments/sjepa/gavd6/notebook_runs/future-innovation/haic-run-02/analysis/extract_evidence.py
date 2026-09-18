"""Extract saved notebook evidence without executing cells or loading run models.

Run from the repository root with:
    .venv/bin/python notebook_runs/haic-run-02/analysis/extract_evidence.py

Table strings retain notebook display precision. Raw MSE values are approximate:
they are recovered from the embedded SVG's linear axis and vector coordinates,
not from the absent reports/raw-errors.csv. Rendering the SVG is a separate step.
"""

import hashlib
import json
from pathlib import Path
import re
import statistics
import xml.etree.ElementTree as ET

from bs4 import BeautifulSoup


HERE = Path(__file__).resolve().parent
NOTEBOOK = HERE.parent / "23_source_learning_curves.ipynb"
notebook = json.loads(NOTEBOOK.read_text())
execution = notebook["metadata"]["fi_execution"]


def table(cell, output):
    html = "".join(notebook["cells"][cell]["outputs"][output]["data"]["text/html"])
    soup = BeautifulSoup(html, "html.parser")
    header = [col.get_text(" ", strip=True) for col in soup.select("thead tr")[0].find_all(["td", "th"])]
    rows = [
        [col.get_text(" ", strip=True) for col in row.find_all(["td", "th"])]
        for row in soup.select("tbody tr")
    ]
    return [dict(zip(["row", *header[1:]], row)) for row in rows]


tables = {
    "inventory": table(3, 1),
    "reservation": table(3, 2),
    "availability": table(3, 4),
    "unavailable_development": table(3, 6),
    "prepared": table(3, 8),
    "outer_folds": table(3, 9),
    "evidence_origins": table(3, 10),
    "plan": table(5, 0),
    "stages": table(7, 3),
    "arm_means": table(9, 2),
    "contrasts": table(9, 3),
    "per_subset_contrasts": table(9, 4),
}
plan = tables["plan"]
assert len(plan) == 60 and len({r["fit_id"] for r in plan}) == 50
assert all(r["status"] == "planned" for r in plan)
assert sum(int(r["clips"]) for r in tables["outer_folds"]) == 1403
assert sum(int(r["recordings"]) for r in tables["outer_folds"]) == 290
assert sum(int(r["annotated_sequences"]) for r in tables["unavailable_development"]) == 65
assert all(r["status"] == "passed" and r["returncode"] == "0" for r in tables["stages"])
assert execution["input_snapshot_before"] == execution["input_snapshot_after"]
assert not any(o["output_type"] == "error" for c in notebook["cells"] for o in c.get("outputs", []))

counts = {}
for size in ("40", "80", "160", "all"):
    rows = [r for r in plan if r["size"] == size]
    sources = [int(r["actual_sources"]) for r in rows]
    windows = [int(r["actual_windows"]) for r in rows]
    counts[size] = dict(
        source_range=[min(sources), max(sources)],
        window_range=[min(windows), max(windows)],
        mean_windows=statistics.mean(windows),
        unique_fits=len({r["fit_id"] for r in rows}),
    )
for fold in range(5):
    endpoint = [r for r in plan if r["size"] == "all" and int(r["outer_fold"]) == fold]
    assert len(endpoint) == 3 and len({r["fit_id"] for r in endpoint}) == 1

svg = "".join(notebook["cells"][9]["outputs"][6]["data"]["image/svg+xml"])
(HERE / "saved-learning-curve.svg").write_text(svg)
parser = ET.XMLParser(target=ET.TreeBuilder(insert_comments=True))
root = ET.fromstring(svg, parser=parser)
ns = {"s": "http://www.w3.org/2000/svg"}
axis = root.find('.//s:g[@id="axes_3"]', ns)
ticks = []
for tick in axis.findall('./s:g[@id="matplotlib.axis_6"]/s:g', ns):
    if not tick.get("id", "").startswith("ytick"):
        continue
    label = next(float(el.text.strip()) for el in tick.iter() if el.tag is ET.Comment)
    position = next(
        float(el.get("y")) for el in tick.iter()
        if isinstance(el.tag, str) and el.tag.endswith("use") and el.get("y")
    )
    ticks.append((position, label))
slope = (ticks[-1][1] - ticks[0][1]) / (ticks[-1][0] - ticks[0][0])
intercept = ticks[0][1] - slope * ticks[0][0]
assert max(abs(slope * pos + intercept - val) for pos, val in ticks) < 1e-9
raw_mse = {}
for line, arm in (("line2d_41", "real-skeleton"), ("line2d_42", "baseline")):
    path = axis.find(f'./s:g[@id="{line}"]/s:path', ns)
    values = [float(v) for v in re.findall(r"[-+]?\d*\.?\d+(?:[Ee][-+]?\d+)?", path.get("d"))]
    assert len(values) == 8
    raw_mse[arm] = dict(zip(("40", "80", "160", "all"),
                            [slope * y + intercept for y in values[1::2]]))

endpoint = {r["arm"]: r for r in tables["arm_means"] if r["size"] == "all"}
contrasts = {(r["size"], r["contrast"]): r for r in tables["contrasts"]}
gain = float(endpoint["real-skeleton"]["delta_r2"])
matched = float(contrasts["all", "matched_increment"]["estimate"])
shuffle = float(endpoint["time-shuffle"]["delta_r2"])
mismatch = float(endpoint["clip-mismatch"]["delta_r2"])
checks = dict(
    real_gain= gain >= .05,
    shuffle= gain >= 2 * max(shuffle, 0),
    mismatch= mismatch <= .01,
    matched_increment= matched > 0,
    real_bootstrap= float(contrasts["all", "real_gain"]["positive_fraction"]) >= .9,
    matched_bootstrap= float(contrasts["all", "matched_increment"]["positive_fraction"]) >= .9,
)
assert checks == dict(real_gain=False, shuffle=True, mismatch=True,
                      matched_increment=True, real_bootstrap=True, matched_bootstrap=True)
derived = dict(
    training_counts=counts,
    reconstructed_checks_from_displayed_values=checks,
    threshold_divided_by_observed_gain=.05 / gain,
    mismatch_minus_real_point_estimate=mismatch - gain,
    matched_increment_growth_point_estimate=matched - float(contrasts["40", "matched_increment"]["estimate"]),
    matched_increment_growth_positive_fraction_lower_bound=max(
        0.0,
        float(contrasts["all", "matched_increment"]["positive_fraction"])
        - float(contrasts["40", "matched_increment"]["positive_fraction"]),
    ),
    raw_mse_approximate_from_svg=raw_mse,
    raw_rgb_relative_error_reduction_40_to_all=1 - raw_mse["baseline"]["all"] / raw_mse["baseline"]["40"],
    raw_skeleton_relative_error_reduction_at_all=1 - raw_mse["real-skeleton"]["all"] / raw_mse["baseline"]["all"],
)
evidence = dict(
    notebook_sha256=hashlib.sha256(NOTEBOOK.read_bytes()).hexdigest(),
    scope="Saved displayed evidence; no model reconstruction or original-artifact integrity verification.",
    precision="Tables retain rounded notebook strings. SVG-derived raw MSE is approximate.",
    execution={k: v for k, v in execution.items() if not k.startswith("input_snapshot")},
    unchanged_snapshot_entry_count=len(execution["input_snapshot_before"]),
    saved_report_binding=execution["input_snapshot_before"]["reports/learning-curve.json"],
    tables=tables,
    derived=derived,
)
(HERE / "evidence.json").write_text(json.dumps(evidence, indent=2) + "\n")
print(json.dumps(derived, indent=2))
print("Saved evidence.json and saved-learning-curve.svg; notebook unchanged.")
