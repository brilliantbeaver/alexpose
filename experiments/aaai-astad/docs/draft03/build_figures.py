"""Build the six accessible draft03 figures from saved, immutable evidence.

Run with: experiments/aaai/.venv/bin/python experiments/aaai/docs/draft03/build_figures.py
No model fitting, scoring, cohort selection, or manuscript generation occurs here.
"""
from datetime import datetime, timezone
from pathlib import Path
import hashlib
import json
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import FancyBboxPatch
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw

B = Path(__file__).resolve().parent
BASE = B.parents[1]
FIG = B / "figures"
sys.path.insert(0, str(BASE))
from aaai_gait.data import JOINTS
from aaai_gait.media import EDGES

TEAL, BLUE, GREY, RED = "#167B7F", "#3E68A0", "#A3ADB7", "#AD4B35"
BRANCHES = [("loss_base", "Own-leg baseline (f)", GREY),
            ("loss_own", "Own-leg correction (f+h)", BLUE),
            ("loss_both", "Both-leg correction (f+g)", TEAL)]
POLICY_LABELS = ["Random\n(U)", "Whole joint\n(J)", "Anatomical\n(A)"]


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_json(path):
    return json.loads(Path(path).read_text())


def main():
    FIG.mkdir(exist_ok=True)
    evidence_path = B / "study_evidence.json"
    e = load_json(evidence_path)
    for name, expected in e["numerical_sources"].items():
        if sha(BASE / name) != expected:
            raise ValueError("Changed evidence source: " + name)
    run = BASE / "artifacts/iteration02" / e["protocol_hash"]
    external = BASE / "artifacts/iteration02" / e["external_protocol_hash"]
    normal = pd.DataFrame(e["normal_test_error"])
    completion_path = B / "advisory/completion_audit_v1/completion_summary.csv"
    completion = pd.read_csv(completion_path, dtype={"outer_fold": str})
    records = []
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 8.5,
                         "axes.titlesize": 9.5, "axes.labelsize": 8.5,
                         "axes.spines.top": False, "axes.spines.right": False,
                         "pdf.fonttype": 42, "svg.fonttype": "none"})

    def save(fig, name, caption, inputs, values=None, extra=None):
        files = {}
        for suffix in ("svg", "pdf", "png"):
            path = FIG / f"{name}.{suffix}"
            fig.savefig(path, dpi=180, facecolor="white")
            files[path.name] = sha(path)
        records.append(dict(name=name, caption=caption,
                            inputs={str(p.relative_to(BASE)): sha(p) for p in inputs},
                            outputs=files, plotted_values=values,
                            metadata=extra or {}, raster_photo_content=False))
        plt.close(fig)

    # One left-to-right reading order, with names defined before symbols.
    fig, ax = plt.subplots(figsize=(7.1, 3.65))
    fig.subplots_adjust(left=.015, right=.985, top=.98, bottom=.035)
    ax.set(xlim=(0, 10), ylim=(0, 5)); ax.axis("off")
    columns = [(0.05, 2.85), (3.55, 2.85), (7.05, 2.85)]
    for (x, w), title in zip(columns, ["1. Prepare the gait window", "2. Fill the same hidden gaps", "3. Compare with normal gait"]):
        ax.text(x+w/2, 4.72, title, ha="center", va="center", weight="bold", fontsize=9.3)

    def box(x, y, w, h, title, body, color="#EAF3F4", edge=TEAL):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=.025,rounding_size=.07",
                                   facecolor=color, edgecolor=edge, linewidth=.9))
        ax.text(x+w/2, y+h-.20, title, ha="center", va="top", fontsize=8.6, weight="bold")
        ax.text(x+w/2, y+.18, body, ha="center", va="bottom", fontsize=8.1, linespacing=1.35)

    box(.05, 3.05, 2.85, 1.35, "Observed walking", "32 samples at 15 Hz; 16 joints\nOne token is one joint\nover four frames")
    box(.05, 1.30, 2.85, 1.43, "Training masks", "Random (U)\nWhole joint (J)\nAnatomical (A)")
    ax.text(1.475, .75, "Small lower-leg groups include\nknee, ankle, heel and toe points\non one side over nearby frames.",
            ha="center", va="center", fontsize=8.1, color="#3D535C", linespacing=1.35)
    box(3.55, 3.17, 2.85, 1.22, "Own-leg baseline (f)", "Use visible movement\nfrom the target leg", "#F1F3F5", "#82929F")
    box(3.55, 1.78, 2.85, 1.22, "Own-leg correction (f+h)", "Add a correction using\nthe same leg", "#EAF0F8", BLUE)
    box(3.55, .39, 2.85, 1.22, "Both-leg correction (f+g)", "Add a correction using\nboth legs")
    box(7.05, 3.17, 2.85, 1.22, "Movement score (M)", "How unusual is movement\nmeasured from observed joints?", "#F1F3F5", "#82929F")
    box(7.05, 1.78, 2.85, 1.22, "Cross-leg score (R)", "How unusual is the benefit\nof adding the other leg?", "#EAF0F8", BLUE)
    box(7.05, .39, 2.85, 1.22, "Combined score (S)", "Fixed average:\nS = (M + R) / 2")
    ax.text(4.975, .15, "Compare f+h with f+g on the same gaps.", ha="center", va="center", fontsize=8.1)
    save(fig, "method", "The workflow uses one gait window and the same hidden targets for three predictors. The cross-leg benefit compares both-leg correction with own-leg correction. The movement score comes directly from observed-gait descriptors. The cross-leg score measures how unusual the prediction benefit is relative to normal reference data; the combined score is their fixed average. A token contains one joint across four frames. An anatomical lower-leg group contains knee, ankle, heel and toe points on one side over a short time span. The diagram motivates measurements, not a claim to recover neurological causes.",
         [evidence_path, run/"protocol.json"])

    fig, ax = plt.subplots(figsize=(3.5, 2.85))
    fig.subplots_adjust(left=.43, right=.98, top=.85, bottom=.32)
    matched = e["external"]["age_sex_sensitivity"]
    detection = [("Four condition groups", e["primary_difference"]),
                 ("All abnormal folders", e["broad_difference"]),
                 ("Laboratory", e["external"]["primary_difference"]),
                 ("Age/sex matched", dict(observed=matched["S_minus_M"], conditional_95_interval=matched["conditional_95_interval"]))]
    for y, (_, row) in enumerate(detection):
        point = row["observed"]; low, high = row["conditional_95_interval"]
        ax.errorbar(point, y, xerr=[[point-low], [high-point]], fmt="o", color=RED, capsize=3, markersize=4)
    ax.axvline(0, color="#6A7078", lw=1)
    ax.set_yticks(range(4), [label for label, _ in detection], fontsize=8)
    ax.set_ylim(3.5, -.5); ax.set_xlim(-.24, .02)
    ax.set_xticks([-.2, -.1, 0]); ax.grid(axis="x", alpha=.18)
    ax.set_xlabel("AUROC difference\nCombined minus Movement", labelpad=7)
    fig.text(.5, .95, "Does combining the scores help?", ha="center", va="top", fontsize=10, weight="bold")
    fig.text(.5, .055, "Negative values mean lower AUROC.", ha="center", fontsize=8)
    save(fig, "detection_difference", "The combined score performs worse than the movement score in all four comparisons. Dots show Combined minus Movement AUROC; bars show the unchanged conditional 95% intervals. The first two comparisons use GAVD, the third uses the laboratory participants, and the fourth uses the age/sex-matched laboratory subset. Intervals condition on fitted models and the declared evaluation design.",
         [evidence_path, run/"paired_intervals.json", external/"paired_effects.json", external/"age_sex_sensitivity.json"],
         values=[dict(label=label, difference=row["observed"], interval=row["conditional_95_interval"]) for label, row in detection])

    fig, axes = plt.subplots(1, 2, figsize=(7.1, 3.25))
    fig.subplots_adjust(left=.09, right=.98, top=.84, bottom=.29, wspace=.29)
    prediction_values = []
    for ax, task, title in zip(axes, ["U", "E_leg"], ["Scattered hiding", "Grouped hiding"]):
        table = normal[normal.bank.eq(task)].groupby("policy")[[col for col, _, _ in BRANCHES]].mean().reindex(["U", "J", "A"])
        x = np.arange(3)
        for k, (col, label, color) in enumerate(BRANCHES):
            ax.bar(x+(k-1)*.24, table[col], .23, label=label, color=color)
        ax.set_xticks(x, POLICY_LABELS); ax.set_xlabel("Training mask", labelpad=5)
        ax.set_title(title, weight="bold"); ax.grid(axis="y", alpha=.15); ax.set_axisbelow(True)
        prediction_values.extend(dict(evaluation=task, policy=p, **table.loc[p].to_dict()) for p in table.index)
    axes[0].set_ylabel("Mean squared prediction error")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", bbox_to_anchor=(.5, .012), ncol=3, frameon=False, fontsize=8)
    fig.text(.5, .97, "Normal GAVD completion under two hiding patterns", ha="center", va="top", fontsize=10, weight="bold")
    save(fig, "normal_prediction", "All 28 held-out normal GAVD recording sources contribute. Each bar averages errors over the same hierarchy of targets, windows, sequences and recording sources; eight lower-leg joints receive equal weight. Scattered hiding removes individual tokens across the sequence. Grouped hiding scores nearby knee, ankle, heel and toe points hidden together on one side. The two panels test different missing-observation patterns, so their absolute errors should not be treated as a controlled comparison of training masks. Units are squared torso-scaled image coordinates.",
         [evidence_path, run/"normal_error_table.csv"], values=prediction_values)

    fig, axes = plt.subplots(1, 2, figsize=(7.1, 3.45))
    fig.subplots_adjust(left=.10, right=.98, top=.78, bottom=.35, wspace=.43)
    timing = sorted([r for r in e["timing"]["summaries"] if r["scope"] == "outer_test_normal" and r["policy"] == "U" and r["variant"] == "timing"], key=lambda r: r["offset"])
    labels = ["4 frames\nearlier", "4 frames\nlater"]
    axes[0].bar(range(2), [r["paired_gain_over_own"] for r in timing], color=RED, width=.55)
    axes[0].set_ylabel("Own-leg error minus both-leg error")
    axes[0].set_title("Before shifting: does context help?", fontsize=9)
    for i, row in enumerate(timing):
        point = row["mean_effect"]; low, high = row["conditional_source_bootstrap_95_interval"]
        axes[1].errorbar(i, point, yerr=[[point-low], [high-point]], fmt="o", capsize=3, color=BLUE)
    axes[1].set_ylabel("Shifted error minus unshifted error")
    axes[1].set_title("After shifting: does error change?", fontsize=9)
    for ax in axes:
        ax.set_xticks([0, 1], labels); ax.set_xlim(-.6, 1.6); ax.axhline(0, color="#626A70", lw=.8)
        ax.grid(axis="y", alpha=.15); ax.set_axisbelow(True)
        ax.set_xlabel("Other-leg samples used", labelpad=5)
    axes[0].set_xlabel("Planned timing shift", labelpad=5)
    axes[1].ticklabel_format(axis="y", style="sci", scilimits=(-3, 3))
    fig.text(.5, .97, "Timing test: weak unshifted prediction limits interpretation", ha="center", va="top", fontsize=10, weight="bold")
    fig.text(.29, .075, "Above zero: both-leg correction helps.\nBelow zero: both-leg correction hurts.", ha="center", fontsize=8)
    fig.text(.77, .075, "Above zero: shifting raises error.\nBelow zero: shifting lowers error.", ha="center", fontsize=8)
    save(fig, "timing_limits", "The experiment substitutes permitted other-leg samples from four frames earlier or later, preserving the original target and body reference. Four frames correspond to about 0.27 seconds at 15 Hz. The left panel compares the two corrections on the common unshifted observations available for each shift: both-leg prediction is worse before either perturbation. The right panel shows the change after shifting, with conditional 95% intervals. All values are unchanged; 27 held-out normal recordings contribute at each offset. Because unshifted prediction already fails to help, these responses do not establish a coordination mechanism.",
         [evidence_path, run/"timing_summary.json"], values=timing)

    # Preserve the original selected examples, original coordinates and NaNs.
    run_manifest_path = run / "run_manifest.json"
    cohort = Path(load_json(run_manifest_path)["cohort_directory"])
    cohort_hashes = load_json(cohort / "artifact_hashes.json")
    for name in ("windows.npy", "window_index.csv"):
        if sha(cohort/name) != cohort_hashes[name]:
            raise ValueError("Changed example coordinate input: " + name)
    windows = np.load(cohort/"windows.npy", mmap_mode="r")
    metadata = pd.read_csv(cohort/"window_index.csv")
    review_path = run / "video_review/review_ledger.csv"
    review = pd.read_csv(review_path)
    cases = [review[(review.label == "normal") & (review.selection == "high")].iloc[0],
             review[(review.label == "parkinsons") & (review.selection == "low")].iloc[0]]
    if tuple(JOINTS) != (11, 12, 13, 14, 15, 16, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32):
        raise ValueError("Unexpected anatomical mapping")
    index = {joint: slot for slot, joint in enumerate(JOINTS)}
    if not all(a in index and b in index for a, b in EDGES):
        raise ValueError("A plotted edge uses an unselected joint")
    fig, axes = plt.subplots(1, 2, figsize=(3.5, 3.35))
    fig.subplots_adjust(left=.17, right=.98, top=.80, bottom=.29, wspace=.36)
    case_records = []
    colors, frames = ["#B7C6D0", "#5D929E", "#175B6D"], [0, 15, 31]
    for ax, case, title in zip(axes, cases, ["Normal folder\nHigher error", "Parkinsons folder\nLower error"]):
        matches = metadata[metadata.sequence_id.eq(case.sequence_id)]
        ix = (matches.start_seconds-case.preview_start_seconds).abs().idxmin()
        window = windows[ix]
        drawn = skipped = 0
        for color, frame in zip(colors, frames):
            for a, b in EDGES:
                pair = window[frame, [index[a], index[b]]]
                if not np.isfinite(pair).all():
                    skipped += 1
                    continue
                ax.plot(pair[:, 0], pair[:, 1], color=color, lw=1)
                drawn += 1
        ax.set_aspect("equal"); ax.invert_yaxis(); ax.set_title(title, fontsize=8.5)
        ax.grid(alpha=.12); ax.set_xlabel("Scaled image x", fontsize=8)
        ax.tick_params(labelsize=7)
        case_records.append(dict(source_id=case.source_id, sequence_id=case.sequence_id,
                                 selected_window_index=int(ix), start_seconds=float(metadata.loc[ix, "start_seconds"]),
                                 frame_indices=frames, finite_edges_drawn=drawn,
                                 missing_edges_skipped=skipped, source_prediction_mse=float(case.source_prediction_mse)))
    axes[0].set_ylabel("Scaled image y", fontsize=8)
    fig.text(.5, .97, "Prediction error is not a health ranking", ha="center", va="top", fontsize=9.5, weight="bold")
    handles = [Line2D([0], [0], color=c, lw=2, label=t) for c, t in zip(colors, ["Start", "Middle", "End"])]
    fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(.5, .12), ncol=3, frameon=False, fontsize=7.5)
    fig.text(.5, .035, "Measured 2D poses; missing joints stay missing.", ha="center", fontsize=7.8)
    save(fig, "failure_cases", "These are the unchanged high-error normal and low-error Parkinsons-folder examples selected by the saved review protocol. Curves show measured poses at samples 0, 15 and 31, with coordinates centered at the pelvis and scaled by torso size. Anatomical indices are mapped explicitly, and edges touching missing coordinates are omitted rather than filled. Viewpoint, amplitude and pose-estimation limits can affect error; these illustrations do not establish the cause of any individual score or verify a diagnosis.",
         [run_manifest_path, review_path, cohort/"artifact_hashes.json", cohort/"windows.npy", cohort/"window_index.csv",
          BASE/"aaai_gait/data.py", BASE/"aaai_gait/media.py"],
         extra=dict(examples=case_records, anatomical_joint_indices=list(JOINTS), anatomical_edges=[list(edge) for edge in EDGES], missingness="Finite original coordinates only; no imputation or joint substitution"))

    ext_path = external / "derivatives/interpolation_audit_v1/participant_comparison.csv"
    ext = pd.read_csv(ext_path)
    cols = ["loss_interpolation", "loss_base", "loss_own", "loss_both"]
    if not set(cols).issubset(ext) or len(ext) != 128:
        raise ValueError("Unexpected frozen interpolation comparison")
    fig, axes = plt.subplots(1, 2, figsize=(7.1, 3.3), gridspec_kw={"width_ratios": [1.15, 1]})
    fig.subplots_adjust(left=.085, right=.98, top=.82, bottom=.31, wspace=.61)
    fold_values = []
    for fold, frame in ext.groupby("outer_fold"):
        values = frame[cols].mean()
        axes[0].plot(range(4), values, marker="o", ms=3.5, lw=1.1, alpha=.80, label=f"Fold {fold+1}")
        fold_values.append(dict(outer_fold=int(fold), **values.to_dict()))
    axes[0].set_xticks(range(4), ["Linear\ninterpolation", "Own-leg\nbaseline\n(f)", "Own-leg\ncorrection\n(f+h)", "Both-leg\ncorrection\n(f+g)"], fontsize=7.5)
    axes[0].set_ylabel("Mean squared error (degrees²)")
    axes[0].set_title("Laboratory: normal completion", weight="bold", fontsize=9)
    axes[0].grid(axis="y", alpha=.15); axes[0].legend(fontsize=7, ncol=2, frameon=False, loc="upper right")
    mask_values = []
    for y, task in enumerate(["U", "E_leg"]):
        row = completion[(completion.dataset == "GAVD") & (completion.bank == task) & (completion.outer_fold == "all")].iloc[0]
        point, low, high = [100*row[name] for name in ("relative_mse_reduction", "relative_ci_low", "relative_ci_high")]
        axes[1].errorbar(point, y, xerr=[[point-low], [high-point]], fmt="o", capsize=3, color=TEAL)
        axes[1].text(point, y+.19, f"{point:.2f}%", ha="center", fontsize=8, color=TEAL)
        mask_values.append(dict(evaluation=task, relative_mse_reduction=float(row.relative_mse_reduction), interval=[float(row.relative_ci_low), float(row.relative_ci_high)]))
    axes[1].axvline(0, color="#7D858C", lw=.8)
    axes[1].set_yticks([0, 1], ["Scattered\nhiding", "Grouped\nhiding"], fontsize=8)
    axes[1].set_ylim(-.4, 1.6); axes[1].set_xlim(-2, 10.5)
    axes[1].set_xlabel("Error reduction with Anatomical\nversus Random training (%)")
    axes[1].set_title("GAVD: masking comparison", weight="bold", fontsize=9)
    axes[1].grid(axis="x", alpha=.15)
    fig.text(.5, .97, "Completion gains depend on the available context and the task", ha="center", va="top", fontsize=10, weight="bold")
    fig.text(.75, .06, "Both-leg correction is compared in both rows.\nPositive values indicate lower error.", ha="center", fontsize=8)
    save(fig, "positive_completion", "Left: the original mean errors for all four laboratory folds, each with 32 held-out healthy participants. Both-leg correction is compared with own-leg correction, giving a 5.2% mean reduction across participants; linear interpolation is the added descriptive benchmark. Right: the same both-leg predictor is trained with Anatomical or Random masking and evaluated on identical GAVD targets. Anatomical training reduces grouped-hiding error by 7.7%; the scattered-hiding difference is uncertain. Error bars are the unchanged conditional paired 95% intervals. These are post-outcome descriptive completion comparisons, not improved anomaly-detection results.",
         [evidence_path, completion_path, ext_path], values=dict(laboratory=fold_values, masking=mask_values))

    manifest = dict(created_at_utc=datetime.now(timezone.utc).isoformat(),
                    producer=str(Path(__file__).relative_to(BASE)), producer_sha256=sha(__file__),
                    status="rendered_from_saved_evidence_only", model_or_score_changes=False,
                    numerical_evidence_sha256=sha(evidence_path),
                    terminology={"training": {"U": "Random", "J": "Whole joint", "A": "Anatomical"},
                                 "evaluation": {"U": "Scattered hiding", "E_leg": "Grouped hiding"},
                                 "scores": {"M": "Movement score", "R": "Cross-leg score", "S": "Combined score"}},
                    figures=records)
    (FIG/"manifest.json").write_text(json.dumps(manifest, indent=2)+"\n")
    positive = next(row for row in records if row["name"] == "positive_completion")
    (FIG/"positive_completion_manifest.json").write_text(json.dumps(dict(producer_sha256=sha(__file__), **positive), indent=2)+"\n")
    (FIG/"caption_guidance.md").write_text("# Figure captions and reading guidance\n\n" + "\n\n".join(f"## {row['name']}\n\n{row['caption']}" for row in records) + "\n")
    contact = Image.new("RGB", (2080, 1950), "white")
    for i, record in enumerate(records):
        preview = Image.open(FIG/f"{record['name']}.png").convert("RGB")
        preview.thumbnail((1000, 600))
        tile = Image.new("RGB", (1040, 650), "#E3E6E8")
        tile.paste(preview, ((1040-preview.width)//2, 40))
        ImageDraw.Draw(tile).text((20, 14), record["name"], fill="black")
        contact.paste(tile, ((i % 2)*1040, (i // 2)*650))
    contact.save(FIG/"accessibility_contact_sheet.png")
    print(json.dumps(dict(figures=[row["name"] for row in records], formats=["svg", "pdf", "png"], producer_sha256=sha(__file__)), indent=2))


if __name__ == "__main__":
    main()
