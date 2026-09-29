"""Compact protocol and declared-question figures; no new experiments or tests.

Repository: python scripts/build_summary_figures.py
Portable:   python scripts/build_summary_figures.py --from-audit
The portable mode uses the exact source selections retained in the provenance
JSON and does not need the original comparison files or private ledgers.
"""
from pathlib import Path
import argparse
import hashlib
import json

from paper_style import HERE, INK, GRAY, LIGHT, DIRECT, DELTA, plt, save


META = HERE / "evidence/summary-figure-provenance.json"
ROOT = next((p for p in HERE.parents if (p / "outputs/iclr").is_dir()), HERE / "inputs")
SPECS = [
    dict(id="core", path="outputs/iclr/walking-core/evaluation/comparisons.json",
         key=["response_error"], interval="crossed_person_seed_ci95",
         candidate="M-paired_jepa-graph_time-paired_change", comparator="P-direct-none-paired_change",
         question="1  Do reference features help?", comparison="Core / change vs direct / change",
         outcome="Response; crossed person/seed 95% CI", marker="o", color=GRAY),
    dict(id="difference", path="outputs/iclr/jepa-response/evaluation/comparisons.json",
         key=["response_error"], interval="crossed_person_seed_ci95",
         candidate="F-response-jepa_delta_v1-graph_time-paired_change",
         comparator="F-response-jepa_endpoint_v1-graph_time-paired_change",
         question="2  Does predicting a difference help?", comparison="Delta / change vs endpoint / change",
         outcome="Response; crossed person/seed 95% CI", marker="o", color=DELTA),
    dict(id="repair", path="outputs/iclr/readout-repair/development/evaluation/comparisons.json",
         key=["primary", "waveform_error"], interval="person_averaged_t_ci95",
         candidate="R-repair-jepa_delta_v1-dense_change", comparator="R-repair-jepa_delta_v1-scalar_low",
         question="3  Does denser supervision help?", comparison="Delta / dense vs delta / low scalar",
         outcome="ViTPose waveform; person-t 95% CI", marker="s", color=DELTA),
]


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def derive():
    sources, rows = {}, []
    for spec in SPECS:
        path = ROOT / spec["path"]
        data = json.loads(path.read_text())
        if spec["id"] == "repair":
            assert data["primary_extractor"] == "vitpose_base"
        for key in spec["key"]:
            data = data[key]
        assert data["candidate"] == spec["candidate"]
        assert data["comparator"] == spec["comparator"]
        assert data["people"] == 14 and data["seeds"] == 3
        low, high = data[spec["interval"]]
        rows.append(dict(id=spec["id"], source=spec["path"], selection=".".join(spec["key"]),
                         candidate=data["candidate"], comparator=data["comparator"],
                         metric=data["metric"], gain_deg=data["improvement"],
                         ci95_low_deg=low, ci95_high_deg=high, interval_key=spec["interval"],
                         people=data["people"], fitted_seeds=data["seeds"]))
        sources[spec["path"]] = sha(path)
    audit_path = HERE / "evidence/framing-claims.json"
    audit = json.loads(audit_path.read_text())
    wanted = {"F03", "F04", "F06", "F07", "F09", "F10"}
    claims = [{k: c[k] for k in ("id", "claim", "source_paths", "limits")}
              for c in audit["claims"] if c["id"] in wanted]
    assert len(claims) == len(wanted)
    count_claim = next(c["claim"] for c in claims if c["id"] == "F09")
    assert "112people" in count_claim.replace(" ", "")
    assert "14people" in count_claim.replace(" ", "")
    sources["evidence/framing-claims.json"] = sha(audit_path)
    config_path = ROOT / "outputs/iclr/walking-core/config.json"
    config = json.loads(config_path.read_text())
    assert config["model"]["window_size"] == 128
    assert config["model"]["patch_size"] == 4
    assert config["seeds"] == [17, 29, 43]
    sources["outputs/iclr/walking-core/config.json"] = sha(config_path)
    for claim in claims:
        for rel in claim["source_paths"]:
            path = ROOT / rel
            if path.is_file():
                sources[rel] = sha(path)
    return dict(source_sha256=sources, selected_comparisons=rows,
                protocol_claims=claims,
                protocol_values=dict(training_people=112, development_people=14,
                                     fitted_seeds=[17, 29, 43], samples=128, joints=12,
                                     coordinates=2, patch_frames=4, tokens=384))


def protocol(values):
    assert values["samples"] // values["patch_frames"] * values["joints"] == values["tokens"]
    fig = plt.figure(figsize=(5.5, 2.25))
    ax = fig.add_axes([.02, .025, .96, .95])
    ax.set(xlim=(0, 1), ylim=(0, 1)); ax.axis("off")
    def text(x, y, value, **kwargs):
        ax.text(x, y, value, va="center", **kwargs)
    def arrow(start, end):
        ax.annotate("", end, start, arrowprops=dict(arrowstyle="->", lw=.8, color=GRAY,
                                                   shrinkA=0, shrinkB=0))
    text(0, .955, "A  Separate people for fitting and evaluation", fontsize=9.5, fontweight="bold")
    text(0, .847, f'{values["training_people"]} training people', color=DIRECT, fontsize=9.3)
    text(.48, .847, f'{values["development_people"]} development people', fontsize=9.3)
    text(0, .745, "Training-only optimization + calibration", fontsize=8.5)
    text(.48, .745, "Reused across studies; 3 fitted seeds", fontsize=8.5)
    text(0, .645, "All windows and derived variants inherit their source person's split.", fontsize=8.5)
    ax.plot([0, 1], [.585, .585], color=LIGHT, lw=.65)
    text(0, .525, "B  Fixed endpoint shape and token slots", fontsize=9.5, fontweight="bold")
    text(.13, .408, r'$[B,128,12,2]$', ha="center", fontsize=10)
    text(.50, .408, "4-frame joint patches", ha="center", fontsize=8.8)
    text(.88, .408, r'$[B,128,12,2]$', ha="center", fontsize=10)
    arrow((.265, .408), (.345, .408)); arrow((.67, .408), (.75, .408))
    text(.13, .313, "Input endpoints", ha="center", fontsize=8.2)
    text(.50, .313, "384 tokens; missing slots kept", ha="center", fontsize=8.2)
    text(.88, .313, "Full-shape output", ha="center", fontsize=8.2)
    ax.plot([0, 1], [.252, .252], color=LIGHT, lw=.65)
    text(0, .195, "C  Reference support fixes eligibility; prediction failures remain counted.", fontsize=8.6)
    text(0, .080, "Average: conditions → windows → motions → people → seeds", fontsize=8.8)
    save(fig, "protocol")


def questions(rows):
    assert [r["id"] for r in rows] == [s["id"] for s in SPECS]
    fig = plt.figure(figsize=(5.5, 2.8))
    fig.text(.02, .965, "Same 14 development people and 3 fitted seeds", fontsize=9.2, va="top")
    ax = fig.add_axes([.615, .18, .365, .70])
    ax.set(xlim=(-1.4, 2.2), ylim=(0, 3), xticks=[-1, 0, 1, 2], yticks=[])
    ax.spines["left"].set_visible(False)
    ax.axvline(0, color=GRAY, lw=.7, zorder=1)
    ax.set_xlabel("Paired gain (°)", labelpad=1.5, fontsize=8.4)
    row_ys = [2.52, 1.51, .50]
    for spec, row, y in zip(SPECS, rows, row_ys):
        gain, low, high = (row[k] for k in ("gain_deg", "ci95_low_deg", "ci95_high_deg"))
        assert low < 0 < high and low < gain < high
        ax.errorbar(gain, y, xerr=[[gain-low], [high-gain]], fmt=spec["marker"],
                    ms=4.2, color=spec["color"], capsize=2.5, elinewidth=1, zorder=3)
        fy = .18 + y / 3 * .70
        fig.text(.02, fy+.035, spec["question"], fontsize=9.1, fontweight="bold", va="center")
        fig.text(.02, fy-.030, spec["comparison"], fontsize=8.5, va="center")
        fig.text(.02, fy-.092, spec["outcome"], fontsize=8.1, va="center", color=GRAY)
        value = f"{gain:.3f} [{low:.3f}, {high:.3f}]".replace("-", "−")
        ax.text(.5, y-.30, value, transform=ax.get_yaxis_transform(),
                ha="center", va="center", fontsize=8.3,
                bbox=dict(facecolor="white", edgecolor="none", pad=.3))
    fig.add_artist(plt.Line2D([.02, .98], [.395, .395], transform=fig.transFigure, color=LIGHT, lw=.7))
    fig.text(.5, .015, "Comparator error − candidate error; positive favors candidate", ha="center", fontsize=8)
    save(fig, "study-questions")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--from-audit", action="store_true", help="Use exact saved source selections in the portable provenance file")
    args = parser.parse_args()
    retained = json.loads(META.read_text()) if args.from_audit else derive()
    protocol(retained["protocol_values"])
    questions(retained["selected_comparisons"])
    metadata = {key: retained[key] for key in ("source_sha256", "selected_comparisons", "protocol_claims", "protocol_values")}
    metadata.update(scope="Explanatory protocol and three separately reported declared primary estimates; no new data, fitting, tests, pooled effect, or confirmation outcomes.",
                    gain_definition="Comparator error minus candidate error; positive favors the named candidate.",
                    interval_definition="First two rows: original crossed person/seed 95% intervals. Third row: original person-averaged Student-t 95% interval conditional on three fitted seeds; ViTPose waveform outcome, not response error.",
                    display="Three decimal places only for visible labels; all plotted positions and saved selections use the original unrounded values.",
                    protocol_limits="B counts endpoint rows, not people. Missing slots retained does not establish correct imputation. Training-only fitting/calibration does not assert development data were never read for bundle validation. No locked-cohort count or result is plotted.",
                    portability="Run with --from-audit using this JSON, the script, and paper_style.py; no original comparison files or machine-specific ledgers required.",
                    script_sha256=sha(Path(__file__)), shared_style_sha256=sha(Path(__file__).with_name("paper_style.py")))
    metadata["outputs_sha256"] = {"figures/"+p.name: sha(p) for name in ("protocol", "study-questions")
                                  for p in [HERE/"figures"/(name+suffix) for suffix in (".pdf", ".svg", ".png", "-gray.png")]}
    META.write_text(json.dumps(metadata, indent=2)+"\n")
    print(json.dumps({"figures": ["protocol", "study-questions"], "provenance": str(META.relative_to(HERE)),
                      "gains": [r["gain_deg"] for r in retained["selected_comparisons"]]}, indent=2))


if __name__ == "__main__":
    main()
