"""Reproduce the September 20 research review without neural training.

Run from the experiment folder: .venv/bin/python scripts/review_research_evidence.py
Only review artifacts and figures are written; small RF fits verify saved results.
The bootstrap is a retrospective,
paired source-resampling sensitivity analysis conditional on fixed OOF predictions.
"""
from __future__ import annotations

import base64
from collections import Counter
import json
import os
from pathlib import Path
import platform
import sys

os.environ.setdefault("MPLCONFIGDIR", "/tmp/ms-research-matplotlib")
os.environ.setdefault("XDG_CACHE_HOME", "/tmp/ms-research-cache")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from sjepa.data import sliding_windows
from sjepa.full_experiment import score_records, summarize_oof
from sjepa.splits import LABELS, file_sha256, load_full_registry, partition_records, split_summary

RUN = ROOT / "artifacts/runs/full-v1/9496e61b050f/laptop-1d09c8e1eea5/capstone-20260921T011305448893Z"
FIG = ROOT / "images/research-review-2026-09-20"
REPORT = ROOT / "artifacts/reviews/2026-09-20-research-evidence.json"
COLORS = {"sjepa": "#176b87", "mean_pose": "#bb5b30", "rf": "#665493",
          "visibility": "#60816b", "majority": "#858b95"}
NAMES = {"sjepa": "S-JEPA", "mean_pose": "Mean pose", "rf": "RF (current extractor)",
         "visibility": "Visibility", "majority": "Majority"}


def save(fig, name):
    fig.savefig(FIG / f"{name}.svg", bbox_inches="tight")
    fig.savefig(FIG / f"{name}.png", dpi=160, bbox_inches="tight")
    plt.close(fig)


def macro_f1(cm):
    diagonal = np.diagonal(cm, axis1=-2, axis2=-1)
    denominator = cm.sum(-1) + cm.sum(-2)
    return np.divide(2 * diagonal, denominator, out=np.zeros_like(diagonal),
                     where=denominator != 0).mean(-1)


def paired_bootstrap(rows, repeats=20000, seed=20260920):
    sources = sorted({r["source_id"] for r in rows})
    matrices = np.zeros((len(sources), 2, 3, 3))
    source_labels = []
    for i, source in enumerate(sources):
        rr = [r for r in rows if r["source_id"] == source]
        assert len({r["true"] for r in rr}) == 1
        source_labels.append(rr[0]["true"])
        for r in rr:
            for j, system in enumerate(("sjepa", "mean_pose")):
                matrices[i, j, LABELS.index(r["true"]), LABELS.index(r[f"pred_{system}"])] += 1 / len(rr)
    # Every occurrence of a resampled source carries a total weight of one,
    # including repeated draws of the same source. Both models share all draws.
    rng = np.random.default_rng(seed)
    choices = []
    for label in LABELS:
        indices = np.flatnonzero(np.array(source_labels) == label)
        choices.append(rng.choice(indices, size=(repeats, len(indices)), replace=True))
    draws = np.concatenate(choices, axis=1)
    assert draws.shape == (repeats, len(sources))
    scores = macro_f1(matrices[draws].sum(axis=1))
    differences = scores[:, 0] - scores[:, 1]
    point = macro_f1(matrices.sum(axis=0))
    return dict(repeats=repeats, seed=seed, source_strata=Counter(source_labels),
                statistic="pooled source-weighted macro-F1(S-JEPA) minus macro-F1(mean pose)",
                point=float(point[0] - point[1]),
                percentile_95=np.quantile(differences, [0.025, 0.975]).tolist(),
                qualification="Retrospective descriptive sensitivity interval conditional on fixed OOF models, "
                "predictions, split and class source counts. Source clusters are resampled as independent; "
                "repeat participants and dependence induced by overlapping training sets are not resolved. "
                "No retraining, selection uncertainty, model-seed or split variation is included. "
                "Not a confirmatory population confidence interval or an equivalence test.")


def main():
    FIG.mkdir(parents=True, exist_ok=True)
    records, registry = load_full_registry(ROOT)
    rows = json.loads((RUN / "oof.json").read_text())
    results = json.loads((RUN / "results.json").read_text())
    metrics = summarize_oof(rows, records, registry)
    assert metrics == results["metrics"], "Retained metric mismatch"
    assert results["registry_sha256"] == registry["registry_sha256"]
    assert results["dataset_sha256"] == registry["dataset_sha256"]
    assert not results["smoke"]
    by_clip = {r.clip_name: r for r in records}
    dataset = []
    for label in LABELS:
        rr = [r for r in records if r.label == label]
        dataset.append(dict(label=label,
            raw_clips=sum(r["label"] == label for r in registry["inventory"]["raw_clips"]),
            cached_clips=len(rr), sources=len({r.source_id for r in rr}),
            frames=sum(r.n_frames for r in rr),
            min_frames=min(r.n_frames for r in rr), max_frames=max(r.n_frames for r in rr),
            padded_clips=sum(r.n_frames < 32 for r in rr),
            windows=sum(len(sliding_windows(r.load_norm(), 32, 16)) for r in rr)))
    folds = []
    for fold in registry["folds"]:
        subset = [r for r in rows if r["fold"] == fold["fold"]]
        folds.append(dict(fold=fold["fold"], **{
            system: score_records([by_clip[r["clip"]] for r in subset],
                                  [r[f"pred_{system}"] for r in subset], True).macro_f1
            for system in NAMES}))
    bootstrap = paired_bootstrap(rows)
    assert np.isclose(bootstrap["point"], metrics["sjepa"]["source_weighted"]["macro_f1"] -
                      metrics["mean_pose"]["source_weighted"]["macro_f1"], atol=1e-12)

    import cv2
    video_metadata = []
    for path in sorted((ROOT / "video-data-full").glob("*/*.mp4")):
        cap = cv2.VideoCapture(str(path))
        assert cap.isOpened(), path
        fps = cap.get(cv2.CAP_PROP_FPS)
        assert fps > 0
        video_metadata.append(dict(clip=path.stem, source_fps=fps,
            stride_sample_fps=fps / max(1, round(fps / 15)),
            width=int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), height=int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))))
        cap.release()

    embedding_file = RUN.parent / "fold-0/training_embeddings.npz"
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import silhouette_score
    with np.load(embedding_file, allow_pickle=False) as z:
        assert str(z["partition"]) == "train"
        assert str(z["registry_sha256"]) == registry["registry_sha256"]
        assert sorted(z["clips"].tolist()) == registry["folds"][0]["train_clips"]
        silhouettes = {k: float(silhouette_score(StandardScaler().fit_transform(z[k]), z["labels"]))
                       for k in ("ssl", "continued", "visibility")}

    # Independently check the defect in the currently installed RF extractor.
    from sjepa.classical import build_feature_matrix
    X, _, _, names = build_feature_matrix(records)
    ankle_duplicate = np.array_equal(X[:, names.index("left_ankle_range")],
                                     X[:, names.index("right_ankle_range")])
    assert ankle_duplicate
    rf_active = [int((X[[records.index(r) for r in partition_records(records, registry, k)[0]]].std(0) > 0).sum())
                 for k in range(5)]
    rf_predictions_verified = {}
    from sjepa.classical import train_rf_and_predict
    for k in range(5):
        train, _, test = partition_records(records, registry, k)
        tr, te = [records.index(r) for r in train], [records.index(r) for r in test]
        predicted = train_rf_and_predict(X[tr], [r.label for r in train], X[te], n_jobs=1)
        expected = {r["clip"]: r["pred_rf"] for r in rows if r["fold"] == k}
        assert predicted.tolist() == [expected[r.clip_name] for r in test]
        rf_predictions_verified[str(k)] = True

    notebook_outputs = {}
    for path in sorted(ROOT.glob("*.ipynb")):
        nb = json.loads(path.read_text())
        notebook_outputs[path.name] = dict(sha256=file_sha256(path),
            code_cells=sum(c["cell_type"] == "code" for c in nb["cells"]),
            executed_cells=sum(c.get("execution_count") is not None for c in nb["cells"]),
            errors=[o.get("ename") for c in nb["cells"] for o in c.get("outputs", []) if o.get("ename")])
        if path.name.startswith("03_"):
            for output in nb["cells"][16].get("outputs", []):
                encoded = output.get("data", {}).get("image/png")
                if encoded:
                    (FIG / "training-diagnostics-saved.png").write_bytes(base64.b64decode(encoded))

    import sklearn
    import torch
    from sjepa.augment import _flip_index, random_view
    from sjepa.masking_v2 import sample_target_mask
    synthetic = torch.zeros((1, 32, 33, 3))
    synthetic[:, :, 27, 0] = .25
    synthetic[:, :, 28, 0] = .60
    reflected = random_view(synthetic, max_rot_deg=0, max_translate=0,
                            scale_jitter=0, flip_prob=1)
    assert torch.allclose(reflected[:, :, 28, 0], -synthetic[:, :, 27, 0])
    mask = sample_target_mask(33, 8, np.random.default_rng(1)).reshape(8, 33)
    exposed_blocks = np.flatnonzero(mask[:, 27] & ~mask[:, 28]).tolist()
    assert exposed_blocks
    reflection_check = dict(original_left_ankle_x=.25,
        reflected_right_slot_x=float(reflected[0, 0, 28, 0]),
        sampler_seed=1, draw_index=0,
        left_target_right_context_time_blocks=exposed_blocks,
        interpretation="With the original slot mask, a reflection moves original target-joint coordinates "
        "to visible opposite-side context slots. This verifies a within-example information path, "
        "not its learned effect or any train/test source overlap.")
    mask_rng = np.random.default_rng(42)
    mask_bank = np.stack([sample_target_mask(33, 8, mask_rng).reshape(8, 33)
                          for _ in range(512)])
    permutation = _flip_index(33).numpy()
    assert np.array_equal(permutation[permutation], np.arange(33))
    exposed = mask_bank & ~mask_bank[..., permutation]
    # A counterfactual mask permutation blocks this particular coordinate path.
    # It does not implement all context/target identity handling in training.
    mapped_mask = mask_bank[..., permutation]
    mapped_exposed = mask_bank & ~mapped_mask[..., permutation]
    assert exposed.any() and not mapped_exposed.any()
    reflection_check["mask_bank_audit"] = dict(seed=42, draws=512,
        target_slots=int(mask_bank.sum()),
        exposed_target_slots_if_reflected=int(exposed.sum()),
        exposed_fraction_if_reflected=float(exposed.sum() / mask_bank.sum()),
        exposed_target_slots_with_permuted_mask=int(mapped_exposed.sum()),
        qualification="Standalone draws from the current sampler, all evaluated conditional on "
        "reflection. This is not a reconstruction of retained training masks or a measured "
        "effect on learning. The permuted-mask result checks a proposed correction to this "
        "coordinate path only; production training is unchanged.")
    code_paths = [ROOT / "scripts/review_research_evidence.py"] + [ROOT / "sjepa" / (name + ".py")
        for name in ("data", "splits", "full_experiment", "eval", "classical", "models", "tokenizer", "masking_v2", "train_v2", "losses", "augment")]
    report = dict(review_date="2026-09-20 America/Los_Angeles", run=str(RUN.relative_to(ROOT)),
        registry_sha256=registry["registry_sha256"], dataset_sha256=registry["dataset_sha256"],
        review_environment=dict(python=platform.python_version(), numpy=np.__version__,
            sklearn=sklearn.__version__, torch=torch.__version__, opencv=cv2.__version__),
        reviewed_code={str(p.relative_to(ROOT)): file_sha256(p) for p in code_paths},
        ambient_feature_extractor_sha256=file_sha256(ROOT.parents[1] / "ambient/classification/features.py"),
        inputs={str(p.relative_to(ROOT)): file_sha256(p) for p in
                (RUN / "results.json", RUN / "oof.json", RUN / "provenance.json", embedding_file)},
        notebooks=notebook_outputs, counts=dataset, splits=split_summary(records, registry),
        metrics_reproduced=True, metrics=metrics, fold_scores=folds, selections=results["selections"],
        bootstrap=bootstrap, training_silhouettes=silhouettes,
        rf_feature_count=len(names), rf_varying_features_by_fold=rf_active,
        rf_ankle_columns_duplicated=ankle_duplicate, rf_predictions_reproduced=rf_predictions_verified,
        reflection_mask_check=reflection_check,
        video_metadata=video_metadata,
        scope="Recomputed saved predictions, raw container metadata and fixed features; refit the small RF "
        "only to verify retained predictions. No pose re-extraction or neural training. "
        "Raw video bytes, participant identities and clinical labels are not independently authenticated.")
    REPORT.write_text(json.dumps(report, indent=2) + "\n")
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.titleweight": "semibold", "svg.fonttype": "none"})

    fig, axes = plt.subplots(1, 2, figsize=(11, 3.7), layout="constrained")
    positions = np.arange(3)
    for offset, key, label, color in [(-.24, "raw_clips", "Raw clips", "#b2c7d3"),
        (0, "cached_clips", "Usable clips", "#176b87"), (.24, "sources", "Source recordings", "#bb5b30")]:
        bars = axes[0].bar(positions + offset, [d[key] for d in dataset], .23, label=label, color=color)
        axes[0].bar_label(bars, padding=2)
    axes[0].set(xticks=positions, xticklabels=["Normal", "MS", "PD"], ylim=(0, 42), ylabel="Count",
                title="A. Clips and source recordings differ")
    axes[0].legend(frameon=False, fontsize=8)
    for i, label in enumerate(LABELS):
        counts = sorted(Counter(r.source_id for r in records if r.label == label).values())
        axes[1].scatter(counts, np.full(len(counts), i) + np.linspace(-.15,.15,len(counts)),
                        s=32, color=["#176b87", "#bb5b30", "#665493"][i], alpha=.7)
    axes[1].set(yticks=range(3), yticklabels=["Normal", "MS", "PD"], xticks=[1, 3, 5, 7, 9, 11, 13],
                xlabel="Usable clips per source (one dot per source)", title="B. Repeated clips are unevenly distributed")
    save(fig, "collection")

    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.1), layout="constrained")
    systems = list(NAMES)
    for i, system in enumerate(systems):
        for marker, weighting in [("o", "source_weighted"), ("s", "clip_weighted")]:
            score = metrics[system][weighting]["macro_f1"]
            axes[0].scatter(score, i + (-.11 if marker == "o" else .11), marker=marker,
                            color=COLORS[system], s=55, facecolors=COLORS[system] if marker == "o" else "none")
    axes[0].scatter([], [], color="#333333", label="Equal total source weight", marker="o")
    axes[0].scatter([], [], edgecolor="#333333", facecolor="none", label="Equal clip weight", marker="s")
    axes[0].set(yticks=range(5), yticklabels=[NAMES[s] for s in systems], xlim=(0, .6),
                xlabel="Pooled macro-F1 (higher is better)", title="A. Weighting changes the ranking")
    axes[0].invert_yaxis(); axes[0].legend(frameon=False, fontsize=8, loc="lower right")
    for system, marker in [("sjepa", "o"), ("mean_pose", "s")]:
        axes[1].plot(range(5), [f[system] for f in folds], marker=marker,
                      color=COLORS[system], label=NAMES[system])
    axes[1].set(xticks=range(5), ylim=(0, 1), xlabel="Outer test fold", ylabel="Source-weighted macro-F1",
                title="B. S-JEPA leads in one of five folds")
    axes[1].legend(frameon=False)
    save(fig, "model-comparison")

    fig, axes = plt.subplots(1, 2, figsize=(10.5, 3.8), layout="constrained")
    for axis, measure, title, ylabel in [(axes[0], "loss", "A. Final training loss is lower in every fold", "Latent cross-entropy"),
            (axes[1], "score", "B. Validation improves in two folds", "Source-weighted macro-F1")]:
        for stage, color, marker in [("ssl", "#176b87", "o"), ("continued", "#bb5b30", "s")]:
            values = [s["diagnostics"][stage]["final_loss"] if measure == "loss" else
                      s["validation_source_macro_f1"][stage] for s in results["selections"]]
            axis.plot(range(5), values, marker=marker, color=color,
                      label="800 updates" if stage == "ssl" else "800 + restarted 400")
        axis.set(xticks=range(5), xlabel="Fold", title=title, ylabel=ylabel)
        axis.legend(frameon=False, fontsize=8)
    save(fig, "continuation")

    fig, axes = plt.subplots(1, 2, figsize=(9, 3.7), layout="constrained")
    for ax, system in zip(axes, ("sjepa", "mean_pose")):
        cm = np.array(metrics[system]["clip_weighted"]["confusion"])
        ax.imshow(cm, cmap="Blues", vmin=0, vmax=24)
        for i in range(3):
            for j in range(3):
                ax.text(j, i, str(cm[i,j]), ha="center", va="center", color="white" if cm[i,j]>14 else "#152a38")
        ax.set(xticks=range(3), yticks=range(3), xticklabels=["Normal", "MS", "PD"],
               yticklabels=["Normal", "MS", "PD"], xlabel="Predicted label", ylabel="Dataset label",
               title=f"{NAMES[system]}: held-out clip counts")
    save(fig, "confusion")

    fig, axes = plt.subplots(1, 2, figsize=(11, 3.6), layout="constrained")
    phase = np.linspace(0, 1, 200)
    left = np.sin(2*np.pi*phase)
    for ax, right, title in [(axes[0], .65*np.sin(2*np.pi*(phase-.5)), "A. Left and right at the same time"),
                             (axes[1], .65*np.sin(2*np.pi*phase), "B. After aligning stride phase")]:
        ax.plot(phase, left, color="#176b87", label="Left (synthetic)")
        ax.plot(phase, right, color="#bb5b30", linestyle="--", label="Right (synthetic)")
        ax.set(title=title, xlabel="Fraction of one stride", ylabel="Illustrative joint signal", ylim=(-1.2,1.2))
        ax.axhline(0, color="#d8dfe3", lw=.8); ax.legend(frameon=False, fontsize=8)
    save(fig, "symmetry")

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.set(xlim=(0,12), ylim=(0,6)); ax.axis("off")
    def box(x, y, w, h, title, text, color="#eef4f7"):
        ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle="round,pad=0.08",fc=color,ec="#a5bbc5",lw=1))
        ax.text(x+.15,y+h-.23,title,weight="semibold",va="top",fontsize=11)
        ax.text(x+.15,y+h-.61,text,va="top",fontsize=9.5,linespacing=1.5)
    box(.1,4.05,3.3,1.6,"91 raw clips → 88 usable clips", "41 source IDs stay attached\nEach clip: T × 33 × 3\nChannels: x, y, visibility")
    box(4.05,4.05,3.65,1.6,"Split sources before windows", "Fold 0 example: 24 / 8 / 9 sources\n51 training / 19 validation / 18 test clips\nFive outer folds; one inner holdout")
    box(8.35,4.05,3.4,1.6,"Training windows and tokens", "32 frames per window; stride 16\nShort clips repeat their final frame\nBatch: 32 × 32 × 33 × 3\nTokens: 32 × 264 × 96")
    box(.1,.8,3.3,2.1,"Fit on training sources", "800 label-free updates\n400 more; training state restarts\nFrozen encoders; fit scaler + probe\nSource-uniform encoder sampling")
    box(4.05,.8,3.65,2.1,"Select on validation sources", "Compare validation F1 for two stages\nKeep original stage on a tie\nNo refit on train + validation\nTest scores do not select the stage")
    box(8.35,.8,3.4,2.1,"Evaluate held-out sources", "One 96-value vector per test clip\nOne prediction per clip and system\n88 predictions × five systems\nSource IDs retained for scoring")
    for start,end in [((3.5,4.85),(3.95,4.85)),((7.8,4.85),(8.25,4.85)),
                       ((3.5,1.85),(3.95,1.85)),((7.8,1.85),(8.25,1.85))]:
        ax.annotate("",xy=end,xytext=start,arrowprops={"arrowstyle":"->","color":"#536d7b","lw":1.5})
    ax.text(6,.12,"All windows and augmented copies inherit their source partition. Figures describe the retained laptop configuration.",ha="center",fontsize=10)
    save(fig, "split-and-shapes")
    print(json.dumps(dict(counts=dataset, metrics_reproduced=True, bootstrap=bootstrap,
                         silhouettes=silhouettes, rf_active=rf_active), indent=2))


if __name__ == "__main__":
    main()
